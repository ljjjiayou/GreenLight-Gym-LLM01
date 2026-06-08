"""Plan how to handle protocol v2 categories that block promotion evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BLOCKED_CATEGORY_POLICIES: dict[str, dict[str, Any]] = {
    "cache_behavior": {
        "default_disposition": "isolate_until_cache_equivalence_or_explicit_authorization",
        "required_evidence": [
            "old-vs-new cache hit/miss equivalence",
            "strict miss semantics unchanged or explicitly versioned",
            "missing buffered action and parsed plan handling unchanged or explicitly versioned",
        ],
    },
    "runner_control_surface": {
        "default_disposition": "isolate_or_explicitly_authorize_runner_surface",
        "required_evidence": [
            "no default controller invocation change",
            "new runner arguments do not change baseline execution unless explicitly selected",
            "scenario list and controller label handling are versioned",
        ],
    },
    "summary_aggregation": {
        "default_disposition": "isolate_until_metric_equivalence",
        "required_evidence": [
            "summary row counts identical or versioned",
            "gate failure aggregation identical or versioned",
            "reward/profit/safety metric extraction identical or versioned",
        ],
    },
    "test_oracle": {
        "default_disposition": "independent_test_oracle_review_required",
        "required_evidence": [
            "test pass/fail changes are listed",
            "hard-safety assertions are justified against mainline",
            "oracle changes cannot hide historical failures",
        ],
    },
}


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def build_report(
    *,
    classification: Mapping[str, Any],
    user_decision: Mapping[str, Any],
    evidence_boundary_isolation: bool = False,
) -> dict[str, Any]:
    category_counts = dict(classification.get("category_counts", {}) or {})
    rows = []
    for category, policy in BLOCKED_CATEGORY_POLICIES.items():
        count = int(category_counts.get(category, 0) or 0)
        if category == "test_oracle":
            current_status = "independent_safety_or_test_oracle_review_required"
        elif category == "cache_behavior":
            current_status = "blocked_until_cache_equivalence_or_explicit_authorization"
        elif category == "runner_control_surface":
            current_status = "blocked_until_runner_surface_isolated_or_authorized"
        else:
            current_status = "blocked_until_old_vs_new_metric_equivalence"
        rows.append(
            {
                "category": category,
                "hunk_count": count,
                "promotion_blocker": count > 0,
                "allowed_for_promotion_evidence": False if count > 0 else True,
                "current_status": current_status if count > 0 else "no_current_hunks",
                "recommended_disposition": policy["default_disposition"],
                "v1_preflight_disposition": (
                    "excluded_from_current_metadata_replay_baseline"
                    if count > 0
                    else "no_current_hunks"
                ),
                "included_in_v1_metadata_replay_baseline": False,
                "included_in_protocol_v2_baseline": False,
                "allowed_next_actions": ["restore", "isolate", "explicitly_authorize"],
                "required_evidence": policy["required_evidence"],
            }
        )
    blocked_categories_resolved = all(not row["promotion_blocker"] for row in rows)
    protocol_user_decision_complete = bool(user_decision.get("protocol_user_decision_complete", False))
    selected_decision = user_decision.get("selected_decision")
    blocked_categories_resolved_for_v1_preflight = bool(
        evidence_boundary_isolation and protocol_user_decision_complete and selected_decision == "partial_accept_protocol_v2"
    )
    blocked_categories_resolved_for_protocol_v2 = False
    return {
        "schema_version": (
            "protocol_v2_blocked_categories_resolution_v3"
            if evidence_boundary_isolation
            else "protocol_v2_blocked_categories_plan_v2"
        ),
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / blocked category plan",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "controlled_replay_allowed": False,
        "metadata_replay_allowed": False,
        "performance_claim_allowed": False,
        "protocol_baseline_authorized": False,
        "protocol_user_decision_complete": protocol_user_decision_complete,
        "selected_protocol_decision": selected_decision,
        "blocked_categories_resolution_plan_ready": True,
        "blocked_categories_resolved": blocked_categories_resolved,
        "blocked_categories_resolved_for_v1_preflight": blocked_categories_resolved_for_v1_preflight,
        "blocked_categories_resolved_for_protocol_v2": blocked_categories_resolved_for_protocol_v2,
        "protocol_v2_baseline_authorization_status": "not_authorized_candidate_only",
        "v1_preflight_resolution_strategy": (
            "evidence_boundary_isolation"
            if evidence_boundary_isolation
            else "plan_only_no_resolution"
        ),
        "accepted_candidate_categories_for_review": ["metadata_report_only", "stricter_safety_gate"],
        "metadata_report_only_status": "report_only_candidate",
        "stricter_safety_gate_status": "protocol_v2_candidate_requires_future_authorization",
        "stricter_safety_gate_requires_explicit_authorization": int(
            category_counts.get("stricter_safety_gate", 0) or 0
        )
        > 0,
        "rows": rows,
        "next_action": "resolve_protocol_v2_blocked_categories",
        "notes": [
            "The delegated Option B decision selects a review path only; it does not authorize protocol v2 baseline.",
            "Evidence-boundary isolation can unblock v1 preflight planning while keeping protocol v2 baseline unauthorized.",
            "metadata_report_only may remain a low-risk review candidate.",
            "stricter_safety_gate can remain a candidate but still needs explicit authorization.",
            "Blocked categories remain outside promotion evidence until restored, isolated, or explicitly authorized.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Protocol v2 Blocked Categories Plan",
        "",
        "- Mode: audit-only / blocked category plan",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Blocked categories resolved: {report.get('blocked_categories_resolved', False)}",
        f"- Blocked categories resolved for v1 preflight: {report.get('blocked_categories_resolved_for_v1_preflight', False)}",
        f"- Blocked categories resolved for protocol v2: {report.get('blocked_categories_resolved_for_protocol_v2', False)}",
        f"- Protocol user decision complete: {report.get('protocol_user_decision_complete', False)}",
        f"- Selected protocol decision: `{report.get('selected_protocol_decision', None)}`",
        "",
        "| category | hunks | blocker | v1 preflight disposition | recommended disposition |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in report.get("rows", []) or []:
        if isinstance(row, Mapping):
            lines.append(
                f"| `{row.get('category', '')}` | {row.get('hunk_count', 0)} | `{row.get('promotion_blocker', False)}` | `{row.get('v1_preflight_disposition', '')}` | `{row.get('recommended_disposition', '')}` |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification-json", required=True)
    parser.add_argument("--user-decision-json", required=True)
    parser.add_argument(
        "--evidence-boundary-isolation",
        action="store_true",
        help="Record Option B v1 preflight isolation while keeping protocol v2 baseline unauthorized.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        classification=_load(args.classification_json),
        user_decision=_load(args.user_decision_json),
        evidence_boundary_isolation=args.evidence_boundary_isolation,
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
    print(f"blocked_categories_resolved={report['blocked_categories_resolved']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
