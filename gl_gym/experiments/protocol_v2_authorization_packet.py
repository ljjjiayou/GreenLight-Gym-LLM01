"""Build a protocol v2 authorization decision packet.

The packet is decision material only. It does not authorize a baseline, run
replay, fill cache, or modify controller behavior.
"""

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


def build_report(classification: Mapping[str, Any]) -> dict[str, Any]:
    category_counts = dict(classification.get("category_counts", {}) or {})
    blocked_categories = list(classification.get("blocked_categories_before_promotion_evidence", []) or [])
    for category in ("cache_behavior", "summary_aggregation", "test_oracle", "runner_control_surface", "unknown_requires_manual_review"):
        if category_counts.get(category, 0) > 0 and category not in blocked_categories:
            blocked_categories.append(category)
    unknown_hunk_count = int(classification.get("unknown_hunk_count", 0) or 0)
    has_protocol_delta = bool(classification.get("rows", []) or category_counts)
    partial_candidate_categories = ["metadata_report_only", "stricter_safety_gate"]
    safety_boundary_test_oracle_count = int(classification.get("safety_boundary_test_oracle_hunk_count", 0) or 0)

    options = [
        {
            "option_id": "reject_full_protocol_v2",
            "decision": "reject",
            "summary": "Reject the current protocol v2 candidate and restore or isolate evaluation protocol changes before promotion evidence.",
            "consequence": "Future replay can continue on the accepted protocol baseline after evaluation-sensitive diffs are removed or isolated.",
            "metadata_replay_allowed_after_option": False,
        },
        {
            "option_id": "accept_full_protocol_v2",
            "decision": "accept_full",
            "summary": "Accept all protocol v2 changes as a new baseline.",
            "consequence": "Requires explicit user/mainline authorization, protocol versioning, and clear statement that old PASS evidence is historical context only.",
            "metadata_replay_allowed_after_option": False,
        },
        {
            "option_id": "partial_accept_protocol_v2",
            "decision": "partial_accept",
            "summary": "Keep metadata/report-only and stricter safety gate deltas as protocol v2 candidates, while blocking runner control surface, cache behavior, summary aggregation, test oracle, and unknown deltas.",
            "consequence": "Recommended conservative path; still requires follow-up review, old-vs-new audit design, and user authorization before any replay evidence can rely on v2.",
            "metadata_replay_allowed_after_option": False,
        },
    ]

    return {
        "schema_version": "protocol_v2_authorization_packet_v2",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / protocol authorization decision packet",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "controlled_replay_allowed": False,
        "metadata_replay_allowed": False,
        "performance_claim_allowed": False,
        "protocol_baseline_authorized": False,
        "requires_user_authorization": has_protocol_delta,
        "recommended_decision": "partial_accept_review_required" if has_protocol_delta else "no_protocol_delta_detected",
        "accepted_candidate_categories_for_review": partial_candidate_categories,
        "blocked_categories_before_promotion_evidence": sorted(set(blocked_categories)),
        "unknown_hunk_count": unknown_hunk_count,
        "safety_boundary_test_oracle_hunk_count": safety_boundary_test_oracle_count,
        "safety_boundary_test_oracle_requires_separate_review": safety_boundary_test_oracle_count > 0,
        "protocol_delta_explained": bool(classification.get("protocol_delta_explained", False)),
        "classification_summary": {
            "category_counts": category_counts,
            "gate_semantics_changed": bool(classification.get("gate_semantics_changed", False)),
            "runner_control_surface_changed": bool(classification.get("runner_control_surface_changed", False)),
            "test_oracle_changed": bool(classification.get("test_oracle_changed", False)),
            "safety_boundary_test_oracle_changed": bool(classification.get("safety_boundary_test_oracle_changed", False)),
        },
        "options": options,
        "next_action": "protocol_v2_user_authorization_required" if has_protocol_delta else "cache_coverage_check_required",
        "notes": [
            "This packet does not authorize protocol v2.",
            "Historical frozen benchmark PASS evidence cannot be mixed with a changed protocol baseline without explicit versioning.",
            "The default recommendation is partial accept for review, not promotion evidence.",
            "Safety-boundary test oracle changes are not replay protocol equivalence and require separate review before promotion evidence.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Protocol v2 Authorization Packet",
        "",
        "- Mode: audit-only / decision packet",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Protocol baseline authorized: {report.get('protocol_baseline_authorized', False)}",
        f"- Recommended decision: `{report.get('recommended_decision', '')}`",
        f"- Safety-boundary test oracle hunks: {report.get('safety_boundary_test_oracle_hunk_count', 0)}",
        "",
        "## Candidate Categories For Review",
        "",
    ]
    for category in report.get("accepted_candidate_categories_for_review", []) or []:
        lines.append(f"- `{category}`")
    lines.extend(["", "## Blocked Categories", ""])
    for category in report.get("blocked_categories_before_promotion_evidence", []) or []:
        lines.append(f"- `{category}`")
    lines.extend(["", "## Options", "", "| option | decision | consequence |", "| --- | --- | --- |"])
    for option in report.get("options", []) or []:
        if isinstance(option, Mapping):
            lines.append(
                f"| `{option.get('option_id', '')}` | `{option.get('decision', '')}` | {option.get('consequence', '')} |"
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
    print(f"recommended_decision={report['recommended_decision']}")
    print(f"protocol_baseline_authorized={report['protocol_baseline_authorized']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
