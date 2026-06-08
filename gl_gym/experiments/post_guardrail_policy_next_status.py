"""Consolidate next-step status after provenance and vent-floor shadow audits."""

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


def build_report(
    *,
    decision_packet: Mapping[str, Any],
    static_provenance: Mapping[str, Any],
    vent_policy: Mapping[str, Any],
    warning_audit: Mapping[str, Any],
    family_ablation: Mapping[str, Any],
    counterexample_shadow: Mapping[str, Any],
    action_diff: Mapping[str, Any],
) -> dict[str, Any]:
    cleanup_blocked = bool(decision_packet.get("blocking_paths"))
    provenance_needed = bool(static_provenance.get("runtime_provenance_instrumentation_required"))
    warning_blocker_needed = bool(warning_audit.get("requires_warning_to_action_blocker_shadow_design"))
    vent_policy_signal = int(_num(vent_policy.get("vent_floor_shadow_block_rows"))) > 0
    if cleanup_blocked:
        next_action = "mainline_diff_cleanup_required"
    elif provenance_needed:
        next_action = "post_guardrail_provenance_instrumentation_required"
    elif warning_blocker_needed:
        next_action = "warning_to_action_shadow_blocker_design"
    elif vent_policy_signal:
        next_action = "post_guardrail_vent_floor_shadow_policy_design"
    else:
        next_action = "candidate_space_redesign_shadow"
    action_diff_ok = int(_num(action_diff.get("action_diff_steps"))) == 0 and int(_num(action_diff.get("missing_step_count"))) == 0
    return {
        "schema_version": "post_guardrail_policy_next_status_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Candidate Pool", "Response Estimate"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only / audit-only",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "rejected_presets_remain_rejected": True,
        "next_action": next_action,
        "gates": {
            "mainline_diff_cleanup_complete": not cleanup_blocked,
            "metadata_only_action_diff_zero": action_diff_ok,
            "runtime_provenance_instrumentation_required": provenance_needed,
            "warning_to_action_shadow_blocker_needed": warning_blocker_needed,
            "vent_floor_shadow_policy_has_signal": vent_policy_signal,
            "counterexample_shadow_plan_available": bool(counterexample_shadow.get("rows")),
            "candidate_family_hard_safety_regression_present": bool(family_ablation.get("hard_safety_regression_present")),
        },
        "mainline_diff": {
            "blocking_paths": list(decision_packet.get("blocking_paths", [])),
        },
        "static_provenance": {
            "runtime_reason_missing_count": int(_num(static_provenance.get("runtime_reason_missing_count"))),
            "likely_source_family_counts": dict(static_provenance.get("likely_source_family_counts", {})),
        },
        "vent_floor_shadow_policy": {
            "label_counts": dict(vent_policy.get("label_counts", {})),
            "would_block_count": int(_num(vent_policy.get("would_block_count"))),
            "ready_for_controlled_replay": bool(vent_policy.get("ready_for_controlled_replay")),
        },
        "warning_not_actionable": {
            "count": int(_num(warning_audit.get("proxy_warning_not_actionable_count"))),
            "actual_hard_crossing_count": int(_num(warning_audit.get("actual_hard_crossing_count"))),
        },
        "counterexample_shadow": {
            "regime_counts": dict(counterexample_shadow.get("regime_counts", {})),
            "flag_counts": dict(counterexample_shadow.get("shadow_audit_flag_counts", {})),
            "replay_run": bool(counterexample_shadow.get("replay_run")),
        },
        "action_diff": {
            "action_diff_steps": int(_num(action_diff.get("action_diff_steps"))),
            "max_abs_delta": _num(action_diff.get("max_abs_delta")),
        },
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Policy Next Status",
        "",
        "- Mode: shadow-only / audit-only",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Gates",
        "",
        "| gate | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("gates", {})).items():
        lines.append(f"| {key} | {value} |")
    for title, key in (
        ("Static Provenance", "static_provenance"),
        ("Vent-Floor Shadow Policy", "vent_floor_shadow_policy"),
        ("Warning Not Actionable", "warning_not_actionable"),
        ("Counterexample Shadow", "counterexample_shadow"),
        ("Action Diff", "action_diff"),
    ):
        lines.extend(["", f"## {title}", "", "```json"])
        lines.append(json.dumps(report.get(key, {}), ensure_ascii=False, indent=2, sort_keys=True))
        lines.append("```")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-json", required=True)
    parser.add_argument("--static-provenance-json", required=True)
    parser.add_argument("--vent-policy-json", required=True)
    parser.add_argument("--warning-json", required=True)
    parser.add_argument("--family-json", required=True)
    parser.add_argument("--counterexample-shadow-json", required=True)
    parser.add_argument("--action-diff-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        decision_packet=_load(args.decision_json),
        static_provenance=_load(args.static_provenance_json),
        vent_policy=_load(args.vent_policy_json),
        warning_audit=_load(args.warning_json),
        family_ablation=_load(args.family_json),
        counterexample_shadow=_load(args.counterexample_shadow_json),
        action_diff=_load(args.action_diff_json),
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
