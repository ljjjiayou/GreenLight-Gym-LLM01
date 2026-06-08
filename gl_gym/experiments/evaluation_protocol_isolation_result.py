"""Classify evaluation protocol diffs without running replay.

This is a source-level isolation result. It compares the current working-tree
evaluation protocol against the git baseline via ``git diff`` and classifies
the delta. It deliberately does not run replay, fill cache, or call an online
LLM.
"""

from __future__ import annotations

import argparse
import json
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
]

STRICT_GATE_MARKERS = (
    "runtime_error_steps",
    "strict_cache_miss_runtime_error_steps",
    "reasons.append",
    "failures.append",
)
METADATA_MARKERS = ("timesteps", "timestep_count", "max_timestep")
RUNNER_SURFACE_MARKERS = (
    "llm_rspc_v2_hot_dry_proposer",
    "agent_config_overrides",
    "rspc_hot_dry_proposer_control",
    "parse_agent_config_overrides",
)
TEST_ORACLE_MARKERS = ("assert", "with self.assertRaises", "test_")


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


def _changed_lines_from_diff(diff_text: str) -> dict[str, list[str]]:
    current_file = ""
    out: dict[str, list[str]] = {}
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[3][2:].replace("\\", "/")
                out.setdefault(current_file, [])
            continue
        if not current_file:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            out.setdefault(current_file, []).append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            out.setdefault(current_file, []).append(line[1:])
    return out


def _match_any(text: str, markers: Sequence[str]) -> bool:
    lower = text.lower()
    return any(marker.lower() in lower for marker in markers)


def _classify_lines(path: str, lines: Sequence[str]) -> dict[str, Any]:
    metadata_lines = [line for line in lines if _match_any(line, METADATA_MARKERS)]
    strict_gate_lines = [line for line in lines if _match_any(line, STRICT_GATE_MARKERS)]
    runner_lines = [line for line in lines if _match_any(line, RUNNER_SURFACE_MARKERS)]
    test_lines = [line for line in lines if path.startswith("tests/") and _match_any(line, TEST_ORACLE_MARKERS)]
    classified = set(metadata_lines + strict_gate_lines + runner_lines + test_lines)
    unclassified = [line for line in lines if line.strip() and line not in classified]
    categories: list[str] = []
    if metadata_lines:
        categories.append("metadata_report_only_delta")
    if strict_gate_lines:
        categories.append("stricter_safety_gate_delta")
    if runner_lines:
        categories.append("runner_control_surface_delta")
    if test_lines:
        categories.append("test_oracle_delta")
    if unclassified:
        categories.append("unclassified_protocol_delta")
    return {
        "path": path,
        "changed_line_count": len(lines),
        "categories": categories,
        "metadata_report_only_line_count": len(metadata_lines),
        "stricter_safety_gate_line_count": len(strict_gate_lines),
        "runner_control_surface_line_count": len(runner_lines),
        "test_oracle_line_count": len(test_lines),
        "unclassified_line_count": len(unclassified),
        "sample_lines": list(lines[:12]),
    }


def build_report(diff_text: str | None = None) -> dict[str, Any]:
    if diff_text is None:
        result = _run_git(["diff", "--", *EVALUATION_SENSITIVE_PATHS])
        diff_text = result.stdout if result.returncode == 0 else ""
    by_file = _changed_lines_from_diff(diff_text)
    rows = [_classify_lines(path, by_file.get(path, [])) for path in EVALUATION_SENSITIVE_PATHS if by_file.get(path)]
    all_categories = sorted({category for row in rows for category in row["categories"]})
    gate_semantics_changed = any("stricter_safety_gate_delta" in row["categories"] for row in rows)
    runner_surface_changed = any("runner_control_surface_delta" in row["categories"] for row in rows)
    test_oracle_changed = any("test_oracle_delta" in row["categories"] for row in rows)
    unclassified = any("unclassified_protocol_delta" in row["categories"] for row in rows)
    protocol_delta_explained = bool(rows) and not unclassified
    requires_authorization = bool(gate_semantics_changed or runner_surface_changed or test_oracle_changed or unclassified)
    return {
        "schema_version": "evaluation_protocol_isolation_result_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / source-level protocol isolation",
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
        "gate_semantics_changed": gate_semantics_changed,
        "runner_control_surface_changed": runner_surface_changed,
        "test_oracle_changed": test_oracle_changed,
        "requires_protocol_baseline_authorization": requires_authorization,
        "protocol_baseline_authorized": False,
        "decision": (
            "protocol_v2_candidate_requires_authorization"
            if requires_authorization
            else "no_evaluation_protocol_delta_detected"
        ),
        "delta_categories": all_categories,
        "rows": rows,
        "notes": [
            "This source-level isolation result does not prove replay equivalence.",
            "Stricter gate semantics can be a useful protocol v2 candidate, but they require explicit baseline authorization before promotion evidence can rely on them.",
            "Historical PASS summaries remain historical context only.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Evaluation Protocol Isolation Result",
        "",
        "- Mode: audit-only / source-level protocol isolation",
        "- Replay run: false",
        "- Cache fill run: false",
        "- Controlled replay allowed: false",
        "- Metadata replay allowed: false",
        "- Performance claim allowed: false",
        f"- Protocol isolation pass: {report.get('protocol_isolation_pass', False)}",
        f"- Protocol delta explained: {report.get('protocol_delta_explained', False)}",
        f"- Gate semantics changed: {report.get('gate_semantics_changed', False)}",
        f"- Requires protocol baseline authorization: {report.get('requires_protocol_baseline_authorization', False)}",
        f"- Decision: `{report.get('decision', '')}`",
        "",
        "## Delta Categories",
        "",
    ]
    for category in report.get("delta_categories", []) or []:
        lines.append(f"- `{category}`")
    lines.extend(["", "## Rows", "", "| path | categories | changed lines |", "| --- | --- | ---: |"])
    for row in report.get("rows", []) or []:
        if isinstance(row, Mapping):
            lines.append(
                f"| `{row.get('path', '')}` | {', '.join(row.get('categories', []))} | {row.get('changed_line_count', 0)} |"
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
    print(f"decision={report['decision']}")
    print(f"gate_semantics_changed={report['gate_semantics_changed']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
