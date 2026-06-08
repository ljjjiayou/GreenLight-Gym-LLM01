"""Classify evaluation protocol diffs at hunk level.

This is a source-level audit only. It reads ``git diff`` for evaluation-
sensitive files and classifies each changed hunk into protocol delta categories.
It does not run replay, fill cache, call an online LLM, or authorize a new
protocol baseline.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVALUATION_SENSITIVE_PATHS = [
    "gl_gym/experiments/frozen_benchmark_protocol.py",
    "gl_gym/experiments/run_frozen_benchmark.py",
    "tests/test_frozen_benchmark.py",
    "tests/test_frozen_benchmark_protocol.py",
    "tests/test_planning_extensions.py",
]

SAFETY_BOUNDARY_TEST_ORACLE_PATHS = {
    "tests/test_planning_extensions.py",
}

FIXED_CATEGORIES = [
    "metadata_report_only",
    "stricter_safety_gate",
    "runner_control_surface",
    "cache_behavior",
    "summary_aggregation",
    "test_oracle",
    "formatting_or_doc_only",
    "unknown_requires_manual_review",
]

MARKERS: dict[str, tuple[str, ...]] = {
    "metadata_report_only": (
        "timesteps",
        "timestep_count",
        "max_timestep",
        "audit",
        "report",
        "summary field",
    ),
    "stricter_safety_gate": (
        "runtime_error_steps",
        "strict_cache_miss_runtime_error_steps",
        "reasons.append",
        "failures.append",
        "gate failure",
        "validate_strict_replay_result",
    ),
    "runner_control_surface": (
        "llm_rspc_v2_hot_dry_proposer",
        "agent_config_overrides",
        "parse_agent_config_overrides",
        "rspc_hot_dry_proposer_control",
        "tomato_safety_v2_enabled",
        "run_llm_trace",
        "AgentConfig",
        "dataclasses.fields",
        ", fields",
        "fields(",
        "controller",
        "controllers",
    ),
    "cache_behavior": (
        "cache_hit",
        "cache_miss",
        "cache miss",
        "strict_cache",
        "plan_cache",
        "missing key",
    ),
    "summary_aggregation": (
        "summary",
        "aggregate",
        "source_counts",
        "runtime_error_counts",
        "sum_metrics",
        "row count",
    ),
    "test_oracle": (
        "assert",
        "with self.assertRaises",
        "pytest.raises",
        "test_",
        "unittest",
    ),
}

HUNK_RE = re.compile(
    r"@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<context>.*)"
)


def _run_git(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _changed_line_text(lines: Sequence[str]) -> str:
    changed = []
    for line in lines:
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            changed.append(line[1:])
    return "\n".join(changed)


def _is_formatting_or_doc_only(changed_text: str) -> bool:
    nonblank = [line.strip() for line in changed_text.splitlines() if line.strip()]
    if not nonblank:
        return True
    doc_prefixes = ('"""', "'''", "#")
    return all(
        line.startswith(doc_prefixes)
        or line in {"(", ")", ",", "[", "]", "{", "}"}
        for line in nonblank
    )


def _contains_marker(text: str, markers: Sequence[str]) -> bool:
    lower = text.lower()
    return any(marker.lower() in lower for marker in markers)


def _category_reasons(path: str, changed_text: str, hunk_header: str) -> dict[str, str]:
    text = f"{hunk_header}\n{changed_text}"
    reasons: dict[str, str] = {}
    for category, markers in MARKERS.items():
        if _contains_marker(text, markers):
            reasons[category] = "matched marker in changed hunk"
    if path.startswith("tests/") and changed_text.strip():
        reasons["test_oracle"] = "changed hunk is in a test oracle file"
    if not reasons and _is_formatting_or_doc_only(changed_text):
        reasons["formatting_or_doc_only"] = "only comments, docstrings, punctuation, or blank lines changed"
    if not reasons:
        reasons["unknown_requires_manual_review"] = "no fixed category marker matched this hunk"
    return reasons


def _parse_diff_hunks(diff_text: str) -> list[dict[str, Any]]:
    current_path = ""
    current_hunk: dict[str, Any] | None = None
    hunks: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current_hunk
        if current_hunk is not None:
            current_hunk["changed_text"] = _changed_line_text(current_hunk["lines"])
            hunks.append(current_hunk)
            current_hunk = None

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            flush()
            parts = line.split()
            current_path = parts[3][2:].replace("\\", "/") if len(parts) >= 4 else ""
            continue
        if line.startswith("@@ "):
            flush()
            match = HUNK_RE.match(line)
            current_hunk = {
                "path": current_path,
                "hunk_header": line,
                "old_start": int(match.group("old_start")) if match else None,
                "old_count": int(match.group("old_count") or 1) if match else None,
                "new_start": int(match.group("new_start")) if match else None,
                "new_count": int(match.group("new_count") or 1) if match else None,
                "function_context": (match.group("context").strip() if match else ""),
                "lines": [],
            }
            continue
        if current_hunk is not None:
            current_hunk["lines"].append(line)
    flush()
    return [hunk for hunk in hunks if hunk.get("path") in EVALUATION_SENSITIVE_PATHS]


def _build_rows(diff_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, hunk in enumerate(_parse_diff_hunks(diff_text), start=1):
        path = str(hunk["path"])
        changed_text = str(hunk.get("changed_text", ""))
        reasons = _category_reasons(path, changed_text, str(hunk.get("hunk_header", "")))
        categories = [category for category in FIXED_CATEGORIES if category in reasons]
        changed_lines = [line for line in changed_text.splitlines() if line.strip()]
        rows.append(
            {
                "hunk_id": f"h{index:03d}",
                "path": path,
                "hunk_header": hunk.get("hunk_header", ""),
                "old_start": hunk.get("old_start"),
                "new_start": hunk.get("new_start"),
                "function_context": hunk.get("function_context", ""),
                "categories": categories,
                "primary_category": categories[0] if categories else "unknown_requires_manual_review",
                "category_reasons": reasons,
                "changed_line_count": len(changed_lines),
                "sample_changed_lines": changed_lines[:12],
                "affects_gate_semantics": "stricter_safety_gate" in categories,
                "affects_runner_control_surface": "runner_control_surface" in categories,
                "affects_test_oracle": "test_oracle" in categories,
                "affects_cache_behavior": "cache_behavior" in categories,
                "affects_summary_aggregation": "summary_aggregation" in categories,
                "safety_boundary_test_oracle": path in SAFETY_BOUNDARY_TEST_ORACLE_PATHS
                and "test_oracle" in categories,
                "requires_authorization": bool(
                    set(categories)
                    & {
                        "stricter_safety_gate",
                        "runner_control_surface",
                        "cache_behavior",
                        "summary_aggregation",
                        "test_oracle",
                        "unknown_requires_manual_review",
                    }
                ),
            }
        )
    return rows


def build_report(diff_text: str | None = None) -> dict[str, Any]:
    if diff_text is None:
        result = _run_git(["diff", "--", *EVALUATION_SENSITIVE_PATHS])
        diff_text = result.stdout if result.returncode == 0 else ""

    rows = _build_rows(diff_text)
    category_counts = {category: 0 for category in FIXED_CATEGORIES}
    for row in rows:
        for category in row["categories"]:
            category_counts[category] += 1

    unknown_count = category_counts["unknown_requires_manual_review"]
    gate_semantics_changed = category_counts["stricter_safety_gate"] > 0
    runner_surface_changed = category_counts["runner_control_surface"] > 0
    test_oracle_changed = category_counts["test_oracle"] > 0
    safety_boundary_test_oracle_count = sum(1 for row in rows if row.get("safety_boundary_test_oracle"))
    protocol_delta_explained = bool(rows) and unknown_count == 0
    blocked_categories = [
        category
        for category in ("runner_control_surface", "test_oracle", "unknown_requires_manual_review")
        if category_counts.get(category, 0) > 0
    ]
    if category_counts["cache_behavior"] > 0:
        blocked_categories.append("cache_behavior")
    if category_counts["summary_aggregation"] > 0:
        blocked_categories.append("summary_aggregation")

    return {
        "schema_version": "evaluation_protocol_delta_classification_v3",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / hunk-level protocol delta classification",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "controlled_replay_allowed": False,
        "metadata_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "protocol_isolation_pass": False,
        "protocol_delta_explained": protocol_delta_explained,
        "unknown_hunk_count": unknown_count,
        "gate_semantics_changed": gate_semantics_changed,
        "runner_control_surface_changed": runner_surface_changed,
        "test_oracle_changed": test_oracle_changed,
        "safety_boundary_test_oracle_changed": safety_boundary_test_oracle_count > 0,
        "safety_boundary_test_oracle_hunk_count": safety_boundary_test_oracle_count,
        "requires_protocol_baseline_authorization": bool(rows),
        "protocol_baseline_authorized": False,
        "recommended_authorization_path": "partial_accept_review_required" if rows else "no_protocol_delta_detected",
        "accepted_candidate_categories_for_review": ["metadata_report_only", "stricter_safety_gate"],
        "blocked_categories_before_promotion_evidence": sorted(set(blocked_categories)),
        "category_counts": category_counts,
        "rows": rows,
        "notes": [
            "This hunk-level classification does not prove old-vs-new replay equivalence.",
            "metadata_report_only and stricter_safety_gate deltas can remain protocol v2 candidates for review.",
            "runner_control_surface, test_oracle, cache_behavior, summary_aggregation, and unknown deltas remain blockers until explicitly reviewed or isolated.",
            "tests/test_planning_extensions.py is treated as safety-boundary test oracle scope, not replay protocol equivalence evidence.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Evaluation Protocol Delta Classification v3",
        "",
        "- Mode: audit-only / hunk-level source classification",
        "- Replay run: false",
        "- Cache fill run: false",
        "- Controlled replay allowed: false",
        "- Metadata replay allowed: false",
        "- Performance claim allowed: false",
        f"- Protocol isolation pass: {report.get('protocol_isolation_pass', False)}",
        f"- Protocol delta explained: {report.get('protocol_delta_explained', False)}",
        f"- Unknown hunk count: {report.get('unknown_hunk_count', 0)}",
        f"- Safety-boundary test oracle changed: {report.get('safety_boundary_test_oracle_changed', False)}",
        f"- Safety-boundary test oracle hunk count: {report.get('safety_boundary_test_oracle_hunk_count', 0)}",
        f"- Recommended authorization path: `{report.get('recommended_authorization_path', '')}`",
        "",
        "## Category Counts",
        "",
        "| category | hunks |",
        "| --- | ---: |",
    ]
    for category, count in dict(report.get("category_counts", {})).items():
        lines.append(f"| `{category}` | {count} |")
    lines.extend(
        [
            "",
            "## Hunk Rows",
            "",
            "| hunk | path | primary category | categories | safety-boundary test oracle | requires authorization |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.get("rows", []) or []:
        if isinstance(row, Mapping):
            lines.append(
                "| `{h}` | `{p}` | `{pc}` | {cats} | `{safety}` | `{auth}` |".format(
                    h=row.get("hunk_id", ""),
                    p=row.get("path", ""),
                    pc=row.get("primary_category", ""),
                    cats=", ".join(row.get("categories", [])),
                    safety=row.get("safety_boundary_test_oracle", False),
                    auth=row.get("requires_authorization", False),
                )
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report()
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json
    if not output_md.is_absolute():
        output_md = PROJECT_ROOT / output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"protocol_delta_explained={report['protocol_delta_explained']}")
    print(f"unknown_hunk_count={report['unknown_hunk_count']}")
    print(f"recommended_authorization_path={report['recommended_authorization_path']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
