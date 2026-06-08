"""Review safety-boundary test oracle hunks from protocol classification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _tags(text: str) -> list[str]:
    lower = text.lower()
    tags = []
    for marker, tag in (
        ("tomato_safety_v2", "tomato_safety_v2"),
        ("hot_dry", "hot_dry"),
        ("guardrail", "guardrail"),
        ("canopy", "canopy_boundary"),
        ("dew", "dew_boundary"),
        ("fallback", "fallback_or_rspc_candidate"),
        ("cache", "strict_replay_cache"),
        ("assert", "assertion_oracle"),
    ):
        if marker in lower and tag not in tags:
            tags.append(tag)
    return tags


def build_report(classification: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for row in classification.get("rows", []) or []:
        if not isinstance(row, Mapping) or not row.get("safety_boundary_test_oracle", False):
            continue
        sample_lines = list(row.get("sample_changed_lines", []) or [])
        text = "\n".join(str(line) for line in sample_lines)
        rows.append(
            {
                "hunk_id": row.get("hunk_id", ""),
                "path": row.get("path", ""),
                "hunk_header": row.get("hunk_header", ""),
                "coverage_tags": _tags(text),
                "sample_changed_lines": sample_lines,
                "adds_or_changes_hard_safety_assertion": "assert" in text.lower(),
                "changes_pass_fail_semantics": "assert" in text.lower() or "test_" in text.lower(),
                "could_hide_old_failure": "unknown_requires_manual_review",
                "could_introduce_false_positive_oracle": "unknown_requires_manual_review",
                "recommended_suite_treatment": "independent_safety_test_suite_candidate",
                "allowed_in_replay_protocol_equivalence": False,
            }
        )
    review_complete = len(rows) == int(classification.get("safety_boundary_test_oracle_hunk_count", 0) or 0)
    return {
        "schema_version": "safety_boundary_test_oracle_review_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / safety test oracle review",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "controlled_replay_allowed": False,
        "metadata_replay_allowed": False,
        "performance_claim_allowed": False,
        "protocol_baseline_authorized": False,
        "safety_boundary_test_oracle_review_complete": review_complete,
        "safety_boundary_test_oracle_hunk_count": len(rows),
        "independent_safety_test_suite_candidate": True,
        "allowed_in_replay_protocol_equivalence": False,
        "default_conclusion": "do_not_merge_into_replay_protocol_equivalence",
        "rows": rows,
        "next_action": "review_safety_boundary_oracle_before_protocol_baseline_authorization",
        "notes": [
            "This review is source-level and does not run tests or replay.",
            "Safety-boundary test oracle changes can be useful, but they are separate from old-vs-new replay protocol equivalence.",
            "Any unknown risk of hiding old failures or introducing false-positive oracle behavior requires manual scientific review.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Safety-Boundary Test Oracle Review",
        "",
        "- Mode: audit-only / source-level oracle review",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Review complete: {report.get('safety_boundary_test_oracle_review_complete', False)}",
        f"- Hunk count: {report.get('safety_boundary_test_oracle_hunk_count', 0)}",
        f"- Default conclusion: `{report.get('default_conclusion', '')}`",
        "",
        "| hunk | tags | suite treatment | replay equivalence |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.get("rows", []) or []:
        if isinstance(row, Mapping):
            lines.append(
                f"| `{row.get('hunk_id', '')}` | {', '.join(row.get('coverage_tags', []))} | `{row.get('recommended_suite_treatment', '')}` | `{row.get('allowed_in_replay_protocol_equivalence', False)}` |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.classification_json))
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
    print(f"safety_boundary_test_oracle_review_complete={report['safety_boundary_test_oracle_review_complete']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
