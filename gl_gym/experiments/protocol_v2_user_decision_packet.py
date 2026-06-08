"""Build the user-facing protocol v2 decision packet.

This packet is decision material only. It does not authorize protocol v2,
run replay, fill cache, or modify controller behavior.
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


def build_report(
    *,
    classification: Mapping[str, Any],
    authorization: Mapping[str, Any],
    selected_decision: str = "",
) -> dict[str, Any]:
    category_counts = dict(classification.get("category_counts", {}) or {})
    blocked_categories = list(authorization.get("blocked_categories_before_promotion_evidence", []) or [])
    recommended_decision = "partial_accept_review_required"
    options = [
        {
            "option_id": "reject_full_protocol_v2",
            "label": "Reject full protocol v2",
            "recommended": False,
            "decision_status_after_selection": "requires_restore_or_isolation_of_protocol_v2_deltas",
            "effect_on_replay_comparability": "highest comparability with the previously accepted protocol after evaluation-sensitive diffs are restored or isolated",
            "effect_on_safety_gate": "new stricter gates are not adopted until separately redesigned and authorized",
            "effect_on_blocked_categories": "all protocol v2 categories stay outside promotion evidence",
            "metadata_replay_allowed_after_selection": False,
        },
        {
            "option_id": "partial_accept_protocol_v2",
            "label": "Partial accept protocol v2 for review",
            "recommended": True,
            "decision_status_after_selection": "review_candidate_not_baseline_authorization",
            "effect_on_replay_comparability": "old and new evidence remain separated until accepted categories are isolated and versioned",
            "effect_on_safety_gate": "metadata_report_only and explicitly authorized stricter_safety_gate deltas may remain candidates",
            "effect_on_blocked_categories": "cache_behavior, runner_control_surface, summary_aggregation, and test_oracle keep blocking promotion evidence",
            "metadata_replay_allowed_after_selection": False,
        },
        {
            "option_id": "accept_full_protocol_v2",
            "label": "Full accept protocol v2",
            "recommended": False,
            "decision_status_after_selection": "requires_explicit_protocol_version_v2_baseline_authorization",
            "effect_on_replay_comparability": "old benchmark PASS evidence becomes historical context and cannot be compared without versioning",
            "effect_on_safety_gate": "all changed gate and oracle semantics become protocol_v2 baseline only after user/mainline authorization",
            "effect_on_blocked_categories": "blocked categories are accepted as v2 semantics, but still require old-vs-new difference documentation",
            "metadata_replay_allowed_after_selection": False,
        },
    ]
    valid_selected_decisions = {str(option["option_id"]) for option in options}
    decision_recorded = bool(selected_decision)
    if decision_recorded and selected_decision not in valid_selected_decisions:
        raise ValueError(f"unknown selected_decision: {selected_decision}")
    selected_option = next(
        (option for option in options if option["option_id"] == selected_decision),
        None,
    )
    protocol_baseline_authorized = selected_decision == "accept_full_protocol_v2"
    if selected_decision == "partial_accept_protocol_v2":
        decision_status = "partial_accept_review_path_selected_not_baseline_authorization"
        next_action = "resolve_protocol_v2_blocked_categories"
    elif selected_decision == "reject_full_protocol_v2":
        decision_status = "reject_full_protocol_v2_selected_restore_or_isolate_required"
        next_action = "restore_or_isolate_protocol_v2_deltas"
    elif selected_decision == "accept_full_protocol_v2":
        decision_status = "full_protocol_v2_baseline_selected_requires_versioned_audit"
        next_action = "old_vs_new_protocol_audit_required"
    else:
        decision_status = "decision_material_only"
        next_action = "user_protocol_v2_decision_required"

    notes = [
        "The recommended path is partial accept for review, not protocol baseline authorization.",
        "Historical frozen benchmark PASS evidence remains historical context only.",
    ]
    if decision_recorded:
        notes.insert(
            0,
            "This record captures the delegated user decision path but does not run replay, fill cache, or change controller behavior.",
        )
    else:
        notes.insert(0, "This packet presents choices but does not make or record a user authorization decision.")

    return {
        "schema_version": (
            "protocol_v2_user_decision_record_v1"
            if decision_recorded
            else "protocol_v2_user_decision_packet_v1"
        ),
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / user decision packet",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "controlled_replay_allowed": False,
        "metadata_replay_allowed": False,
        "performance_claim_allowed": False,
        "protocol_baseline_authorized": protocol_baseline_authorized,
        "protocol_user_decision_complete": decision_recorded,
        "selected_decision": selected_decision or None,
        "selected_decision_status": decision_status,
        "selected_option": selected_option,
        "recommended_decision": recommended_decision,
        "accepted_candidate_categories_for_review": ["metadata_report_only", "stricter_safety_gate"],
        "blocked_categories_before_promotion_evidence": sorted(set(blocked_categories)),
        "category_counts": category_counts,
        "safety_boundary_test_oracle_hunk_count": int(
            classification.get("safety_boundary_test_oracle_hunk_count", 0) or 0
        ),
        "historical_benchmark_pass_context": {
            "historical_context_only": True,
            "not_current_working_tree_evidence": True,
            "not_protocol_v2_evidence": True,
            "not_metadata_replay_evidence": True,
        },
        "options": options,
        "next_action": next_action,
        "notes": notes,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Protocol v2 User Decision Packet",
        "",
        "- Mode: audit-only / user decision packet",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Protocol baseline authorized: {report.get('protocol_baseline_authorized', False)}",
        f"- Protocol user decision complete: {report.get('protocol_user_decision_complete', False)}",
        f"- Selected decision: `{report.get('selected_decision', None)}`",
        f"- Recommended decision: `{report.get('recommended_decision', '')}`",
        "",
        "## Options",
        "",
        "| option | recommended | replay comparability |",
        "| --- | --- | --- |",
    ]
    for option in report.get("options", []) or []:
        if isinstance(option, Mapping):
            lines.append(
                f"| `{option.get('option_id', '')}` | `{option.get('recommended', False)}` | {option.get('effect_on_replay_comparability', '')} |"
            )
    lines.extend(["", "## Historical PASS Boundary", ""])
    for key, value in dict(report.get("historical_benchmark_pass_context", {})).items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-json", required=True)
    parser.add_argument("--authorization-json", required=True)
    parser.add_argument(
        "--selected-decision",
        default="",
        choices=["", "reject_full_protocol_v2", "partial_accept_protocol_v2", "accept_full_protocol_v2"],
        help="Record a concrete user decision path. Empty keeps packet decision-only.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        classification=_load(args.classification_json),
        authorization=_load(args.authorization_json),
        selected_decision=args.selected_decision,
    )
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
    print(f"protocol_user_decision_complete={report['protocol_user_decision_complete']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
