"""Consolidate shadow policy instrumentation readiness gates."""

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
    authorization_status: Mapping[str, Any],
    runtime_provenance_audit: Mapping[str, Any],
    joint_prediction_v2: Mapping[str, Any],
    action_diff: Mapping[str, Any],
    replay_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    authorization_complete = bool(authorization_status.get("authorization_complete", False))
    provenance_complete = (
        str(runtime_provenance_audit.get("audit_status", "")) == "runtime_provenance_complete"
        and int(_num(runtime_provenance_audit.get("runtime_reason_missing_count"))) == 0
        and int(_num(runtime_provenance_audit.get("unknown_post_guardrail_rewrite_count"))) == 0
    )
    joint_ready = bool(joint_prediction_v2.get("ready_for_policy_judgment", False))
    action_diff_zero = int(_num(action_diff.get("action_diff_steps"))) == 0
    if not authorization_complete:
        next_action = "mainline_diff_authorization_required"
    elif not provenance_complete:
        next_action = "runtime_provenance_metadata_replay_required"
    elif not joint_ready:
        next_action = "joint_prediction_completion_required"
    else:
        next_action = "counterexample_shadow_replay_plan"
    return {
        "schema_version": "post_guardrail_shadow_policy_instrumentation_status_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only / metadata-only instrumentation status",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "rejected_presets_remain_rejected": True,
        "next_action": next_action,
        "gates": {
            "mainline_diff_authorization_complete": authorization_complete,
            "runtime_provenance_complete": provenance_complete,
            "joint_prediction_ready_for_policy_judgment": joint_ready,
            "metadata_only_action_diff_zero": action_diff_zero,
            "counterexample_replay_input_ready": bool(replay_readiness.get("ready_for_future_shadow_replay_input")),
            "counterexample_replay_run": bool(replay_readiness.get("replay_run", False)),
        },
        "authorization": {
            "blocking_paths": list(authorization_status.get("blocking_paths", [])),
            "status_counts": dict(authorization_status.get("status_counts", {})),
        },
        "runtime_provenance": {
            "audit_status": runtime_provenance_audit.get("audit_status", ""),
            "record_count": int(_num(runtime_provenance_audit.get("record_count"))),
            "unknown_post_guardrail_rewrite_count": int(
                _num(runtime_provenance_audit.get("unknown_post_guardrail_rewrite_count"))
            ),
            "runtime_reason_missing_count": int(
                _num(runtime_provenance_audit.get("runtime_reason_missing_count"))
            ),
            "metadata_replay_required": bool(runtime_provenance_audit.get("metadata_replay_required", True)),
        },
        "joint_prediction_v2": {
            "ready_for_policy_judgment": joint_ready,
            "row_count": int(_num(joint_prediction_v2.get("row_count"))),
            "missing_field_counts": dict(joint_prediction_v2.get("missing_field_counts", {})),
        },
        "action_diff": {
            "action_diff_steps": int(_num(action_diff.get("action_diff_steps"))),
            "max_abs_delta": float(_num(action_diff.get("max_abs_delta"))),
        },
        "counterexample_replay_readiness": {
            "ready": bool(replay_readiness.get("ready_for_future_shadow_replay_input")),
            "replay_run": bool(replay_readiness.get("replay_run", False)),
            "recommended_replay_order": list(replay_readiness.get("recommended_replay_order", [])),
        },
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Shadow Policy Instrumentation Status",
        "",
        "- Mode: shadow-only / metadata-only instrumentation status",
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
    parser.add_argument("--authorization-status-json", required=True)
    parser.add_argument("--runtime-provenance-json", required=True)
    parser.add_argument("--joint-prediction-json", required=True)
    parser.add_argument("--action-diff-json", required=True)
    parser.add_argument("--replay-readiness-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        authorization_status=_load(args.authorization_status_json),
        runtime_provenance_audit=_load(args.runtime_provenance_json),
        joint_prediction_v2=_load(args.joint_prediction_json),
        action_diff=_load(args.action_diff_json),
        replay_readiness=_load(args.replay_readiness_json),
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
