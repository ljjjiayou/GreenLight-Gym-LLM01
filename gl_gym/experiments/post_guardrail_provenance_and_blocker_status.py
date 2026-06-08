"""Consolidate provenance, blocker, and readiness gates for the next phase."""

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
    authorization: Mapping[str, Any],
    runtime_design: Mapping[str, Any],
    vent_v2: Mapping[str, Any],
    warning_design: Mapping[str, Any],
    joint_readiness: Mapping[str, Any],
    replay_readiness: Mapping[str, Any],
    action_diff: Mapping[str, Any],
) -> dict[str, Any]:
    mainline_blocked = bool(authorization.get("blocking_paths"))
    runtime_pending = bool(runtime_design.get("runtime_provenance_instrumentation_pending"))
    warning_needed = int(_num(warning_design.get("design_row_count"))) > 0
    joint_ready = bool(joint_readiness.get("ready_for_policy_judgment"))
    if mainline_blocked:
        next_action = "mainline_diff_authorization_required"
    elif runtime_pending:
        next_action = "runtime_provenance_instrumentation_pending"
    elif warning_needed:
        next_action = "warning_to_action_shadow_blocker_design"
    elif int(_num(vent_v2.get("design_row_count"))) > 0:
        next_action = "vent_floor_shadow_policy_v2_design"
    else:
        next_action = "response_model_recalibration"
    return {
        "schema_version": "post_guardrail_provenance_and_blocker_status_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Candidate Pool"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only / audit-only / design-only",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "rejected_presets_remain_rejected": True,
        "next_action": next_action,
        "gates": {
            "mainline_diff_authorization_complete": not mainline_blocked,
            "runtime_provenance_instrumentation_pending": runtime_pending,
            "warning_to_action_blocker_design_needed": warning_needed,
            "vent_floor_v2_design_available": int(_num(vent_v2.get("design_row_count"))) > 0,
            "joint_prediction_ready_for_policy_judgment": joint_ready,
            "counterexample_replay_input_ready": bool(replay_readiness.get("ready_for_future_shadow_replay_input")),
            "metadata_only_action_diff_zero": int(_num(action_diff.get("action_diff_steps"))) == 0,
        },
        "authorization": {
            "blocking_paths": list(authorization.get("blocking_paths", [])),
            "authorization_counts": dict(authorization.get("authorization_counts", {})),
        },
        "runtime_provenance": {
            "current_runtime_reason_missing_count": int(_num(runtime_design.get("current_runtime_reason_missing_count"))),
            "target_runtime_reason_missing_count": int(_num(runtime_design.get("target_runtime_reason_missing_count"))),
        },
        "vent_floor_v2": {
            "design_row_count": int(_num(vent_v2.get("design_row_count"))),
            "excluded_hard_warning_rows": int(_num(vent_v2.get("excluded_hard_warning_rows"))),
        },
        "warning_to_action": {
            "design_row_count": int(_num(warning_design.get("design_row_count"))),
            "actual_hard_crossing_rows_covered": int(_num(warning_design.get("actual_hard_crossing_rows_covered"))),
        },
        "joint_prediction": {
            "ready_for_policy_judgment": joint_ready,
            "missing_field_counts": dict(joint_readiness.get("missing_field_counts", {})),
        },
        "counterexample_replay_readiness": {
            "ready": bool(replay_readiness.get("ready_for_future_shadow_replay_input")),
            "recommended_replay_order": list(replay_readiness.get("recommended_replay_order", [])),
            "replay_run": bool(replay_readiness.get("replay_run")),
        },
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Provenance and Blocker Status",
        "",
        "- Mode: shadow-only / audit-only / design-only",
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
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-json", required=True)
    parser.add_argument("--runtime-design-json", required=True)
    parser.add_argument("--vent-v2-json", required=True)
    parser.add_argument("--warning-design-json", required=True)
    parser.add_argument("--joint-readiness-json", required=True)
    parser.add_argument("--replay-readiness-json", required=True)
    parser.add_argument("--action-diff-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        authorization=_load(args.authorization_json),
        runtime_design=_load(args.runtime_design_json),
        vent_v2=_load(args.vent_v2_json),
        warning_design=_load(args.warning_design_json),
        joint_readiness=_load(args.joint_readiness_json),
        replay_readiness=_load(args.replay_readiness_json),
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
