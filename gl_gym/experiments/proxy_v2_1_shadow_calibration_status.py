"""Summarize proxy-v2.1 shadow calibration readiness.

This report is read-only.  Proxy v2.1 is an audit-layer candidate derived from
existing proxy-v2 trace metadata, not a controller change.
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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _loads(paths: Sequence[str] | None) -> list[dict[str, Any]]:
    return [_load(path) for path in (paths or [])]


def _first(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return reports[0] if reports else {}


def build_report(
    *,
    shadow_validation_reports: Sequence[Mapping[str, Any]],
    candidate_repair_reports: Sequence[Mapping[str, Any]] = (),
    post_guardrail_reports: Sequence[Mapping[str, Any]] = (),
    action_diff_reports: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    shadow = _first(shadow_validation_reports)
    repair = _first(candidate_repair_reports)
    post = _first(post_guardrail_reports)
    action = _first(action_diff_reports)
    candidate = shadow.get("proxy_v2_1_shadow_candidate", {}) if isinstance(shadow.get("proxy_v2_1_shadow_candidate"), Mapping) else {}
    rejected_scales = candidate.get("rejected_scales", {}) if isinstance(candidate.get("rejected_scales"), Mapping) else {}
    v21_unwarned = int(_num(candidate.get("v2_unwarned_false_safe")))
    v21_affected = int(_num(candidate.get("pure_hot_dry_affected_steps")))
    baseline_affected = int(_num(candidate.get("baseline_proxy_v2_pure_hot_dry_affected_steps")))
    action_diff_zero = bool(
        int(_num(action.get("trace_count"))) > 0
        and int(_num(action.get("action_diff_steps"))) == 0
        and int(_num(action.get("missing_step_count"))) == 0
    )
    repair_conclusion = str(repair.get("candidate_pool_repair_conclusion", "") or "")
    generation_conclusion = str(repair.get("candidate_generation_shadow_conclusion", "") or "")
    harmful = int(_num(post.get("harmful_override_count")))
    harmful_v21 = int(_num(post.get("harmful_override_preventable_by_proxy_v2_1_count")))

    gates = {
        "metadata_only_action_diff_zero": action_diff_zero,
        "proxy_v2_1_false_safe_zero": bool(v21_unwarned == 0),
        "proxy_v2_1_reduces_pure_hot_dry_affected": bool(v21_affected < baseline_affected),
        "buffer_scale_0_5_rejected_due_to_false_safe": bool(
            str(rejected_scales.get("buffer_scale_0.5", {}).get("reason", "")) == "unsafe_due_to_false_safe"
        ),
        "candidate_gap_has_shadow_repair_evidence": (
            repair_conclusion in {"new_candidate_needed_shadow_evidence", "new_candidate_needed"}
            or generation_conclusion
            in {"new_candidate_needed_shadow_evidence_post_guardrail_safe", "post_guardrail_policy_blocks_candidate_repair"}
        ),
        "post_guardrail_not_fully_explained": bool(harmful > 0 and harmful_v21 < harmful),
    }
    if not gates["proxy_v2_1_false_safe_zero"]:
        next_action = "response_model_recalibration"
    elif generation_conclusion == "post_guardrail_policy_blocks_candidate_repair":
        next_action = "post_guardrail_root_cause_deepening"
    elif generation_conclusion == "new_candidate_needed_shadow_evidence_post_guardrail_safe":
        next_action = "candidate_generation_shadow_design"
    elif repair_conclusion == "new_candidate_needed_shadow_evidence":
        next_action = "candidate_generation_shadow_design"
    elif gates["post_guardrail_not_fully_explained"]:
        next_action = "post_guardrail_root_cause_deepening"
    elif bool(candidate.get("accepted_for_shadow_followup")):
        next_action = "proxy_v2_1_candidate_hold"
    else:
        next_action = "response_model_recalibration"

    return {
        "schema_version": "proxy_v2_1_shadow_calibration_status_v1",
        "mainline_alignment": {
            "affected_layers": ["Response Estimate", "Safety Boundary", "Candidate Pool", "Evaluation"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only / metadata-only",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "next_action": next_action,
        "gates": gates,
        "proxy_v2_1_shadow_candidate": candidate,
        "candidate_repair": {
            "candidate_pool_gap_steps": int(_num(repair.get("candidate_pool_gap_steps"))),
            "repair_candidate_available_steps": int(_num(repair.get("repair_candidate_available_steps"))),
            "candidate_pool_repair_conclusion": repair_conclusion,
            "candidate_generation_shadow_conclusion": generation_conclusion,
            "repair_quality_counts": dict(repair.get("repair_quality_counts", {})),
        },
        "post_guardrail": {
            "harmful_override_count": harmful,
            "harmful_override_preventable_by_proxy_v2_1_count": harmful_v21,
            "harmful_override_preventable_by_proxy_v2_1_rate": post.get(
                "harmful_override_preventable_by_proxy_v2_1_rate"
            ),
            "root_cause_summary": dict(post.get("root_cause_summary", {})),
            "fine_grained_root_cause_summary": dict(post.get("fine_grained_root_cause_summary", {})),
            "harmful_fine_grained_root_cause_summary": dict(post.get("harmful_fine_grained_root_cause_summary", {})),
            "harmful_root_cause_combinations": dict(post.get("harmful_root_cause_combinations", {})),
        },
        "action_diff": {
            "trace_count": int(_num(action.get("trace_count"))),
            "action_diff_steps": int(_num(action.get("action_diff_steps"))),
            "missing_step_count": int(_num(action.get("missing_step_count"))),
            "max_abs_delta": _num(action.get("max_abs_delta")),
        },
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    candidate = report.get("proxy_v2_1_shadow_candidate", {}) if isinstance(report.get("proxy_v2_1_shadow_candidate"), Mapping) else {}
    repair = report.get("candidate_repair", {}) if isinstance(report.get("candidate_repair"), Mapping) else {}
    post = report.get("post_guardrail", {}) if isinstance(report.get("post_guardrail"), Mapping) else {}
    action = report.get("action_diff", {}) if isinstance(report.get("action_diff"), Mapping) else {}
    lines = [
        "# Proxy v2.1 Shadow Calibration Status",
        "",
        "## Mainline Alignment",
        "",
        "- Mode: shadow-only / metadata-only",
        "- Default `llm_rspc_v2` changed: false",
        "- Reopens rejected preset: false",
        "- Controlled replay allowed: false",
        "",
        "## Decision",
        "",
        f"- Next action: `{report.get('next_action', '')}`",
        f"- Performance claim allowed: {report.get('performance_claim_allowed', False)}",
        "",
        "## Gates",
        "",
        "| gate | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("gates", {})).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Proxy v2.1 Candidate",
            "",
            f"- Buffer scale: {candidate.get('buffer_scale', '')}",
            f"- V2.1 unwarned false-safe: {candidate.get('v2_unwarned_false_safe', 0)}",
            f"- V2.1 pure-hot-dry affected steps: {candidate.get('pure_hot_dry_affected_steps', 0)}",
            f"- Baseline proxy v2 pure-hot-dry affected steps: {candidate.get('baseline_proxy_v2_pure_hot_dry_affected_steps', 0)}",
            f"- Accepted for shadow follow-up: {candidate.get('accepted_for_shadow_followup', False)}",
            f"- Rejected scales: {json.dumps(candidate.get('rejected_scales', {}), sort_keys=True)}",
            "",
            "## Candidate Repair",
            "",
            f"- Candidate-pool gap steps: {repair.get('candidate_pool_gap_steps', 0)}",
            f"- Repair-candidate available steps: {repair.get('repair_candidate_available_steps', 0)}",
            f"- Conclusion: `{repair.get('candidate_pool_repair_conclusion', '')}`",
            f"- Candidate generation shadow conclusion: `{repair.get('candidate_generation_shadow_conclusion', '')}`",
            f"- Repair quality counts: {json.dumps(repair.get('repair_quality_counts', {}), sort_keys=True)}",
            "",
            "## Post-Guardrail",
            "",
            f"- Harmful override count: {post.get('harmful_override_count', 0)}",
            f"- Preventable by proxy v2.1: {post.get('harmful_override_preventable_by_proxy_v2_1_count', 0)}",
            f"- Root causes: {json.dumps(post.get('root_cause_summary', {}), sort_keys=True)}",
            f"- Fine-grained root causes: {json.dumps(post.get('fine_grained_root_cause_summary', {}), sort_keys=True)}",
            f"- Harmful fine-grained root causes: {json.dumps(post.get('harmful_fine_grained_root_cause_summary', {}), sort_keys=True)}",
            "",
            "## Action Diff",
            "",
            f"- Trace count: {action.get('trace_count', 0)}",
            f"- Action diff steps: {action.get('action_diff_steps', 0)}",
            f"- Missing step count: {action.get('missing_step_count', 0)}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-validation-json", action="append", default=[])
    parser.add_argument("--candidate-repair-json", action="append", default=[])
    parser.add_argument("--post-guardrail-json", action="append", default=[])
    parser.add_argument("--action-diff-json", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        shadow_validation_reports=_loads(args.shadow_validation_json),
        candidate_repair_reports=_loads(args.candidate_repair_json),
        post_guardrail_reports=_loads(args.post_guardrail_json),
        action_diff_reports=_loads(args.action_diff_json),
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
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
