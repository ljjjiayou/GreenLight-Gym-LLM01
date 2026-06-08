"""Build the Stage-B authorization request for canonical strict metadata replay.

This request records whether a user explicitly authorized Stage-B execution.
By default it is an authorization request only: no replay is executed, cache is
not filled, and no online LLM is called.
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


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _baseline_trace_status(execution_request: Mapping[str, Any]) -> dict[str, Any]:
    raw = execution_request.get("baseline_trace_precheck", {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def _stage_b_post_run_audits(execution_request: Mapping[str, Any], *, readiness_refresh_name: str) -> list[Any]:
    rows: list[Any] = []
    for row in execution_request.get("post_run_audits", []) or []:
        if not isinstance(row, Mapping):
            rows.append(row)
            continue
        item = dict(row)
        name = str(item.get("name", ""))
        if name.startswith("metadata_replay_readiness_checklist"):
            item["name"] = readiness_refresh_name
        elif name.startswith("strict_metadata_replay_summary"):
            item["name"] = "strict_metadata_replay_summary_20260524"
        rows.append(item)
    return rows


def build_report(
    *,
    two_stage_authorization_packet: Mapping[str, Any],
    execution_request: Mapping[str, Any],
    readiness: Mapping[str, Any],
    runtime_provenance_closure_status: Mapping[str, Any] | None = None,
    stage_b_user_authorized: bool = False,
    authorization_source: str = "not_authorized_in_this_turn",
) -> dict[str, Any]:
    stage_a = dict(two_stage_authorization_packet.get("stage_a", {}) or {})
    stage_b = dict(two_stage_authorization_packet.get("stage_b", {}) or {})
    baseline_trace = _baseline_trace_status(execution_request)
    overlay_runner_exists = bool(execution_request.get("overlay_runner_path_exists", False))
    closure_status = runtime_provenance_closure_status if isinstance(runtime_provenance_closure_status, Mapping) else {}
    runtime_provenance_closure_implementation_ready = bool(
        closure_status.get("runtime_provenance_closure_implementation_ready", False)
    ) if closure_status else True
    stage_b_rerun_authorization_required = bool(
        closure_status.get("stage_b_rerun_authorization_required", False)
    ) if closure_status else False
    stage_b_rerun_authorized = bool(
        closure_status.get("stage_b_rerun_authorized", False)
    ) if closure_status else False
    stage_b_request_ready = bool(
        two_stage_authorization_packet.get("two_stage_authorization_packet_ready", False)
        and stage_a.get("overlay_build_validation_complete", False)
        and stage_b.get("stage_b_ready_for_user_decision", False)
        and execution_request.get("canonical_strict_metadata_replay_execution_request_ready", False)
        and readiness.get("canonical_metadata_replay_authorization_request_ready", False)
        and baseline_trace.get("baseline_trace_qualified", False)
        and overlay_runner_exists
        and runtime_provenance_closure_implementation_ready
    )
    execution_allowed = bool(
        stage_b_request_ready
        and stage_b_user_authorized
        and ((not stage_b_rerun_authorization_required) or stage_b_rerun_authorized)
    )
    next_action = "execute_single_canonical_strict_metadata_replay"
    if not execution_allowed and stage_b_rerun_authorization_required:
        next_action = "await_explicit_stage_b_rerun_authorization_after_runtime_provenance_closure"
    elif not execution_allowed:
        next_action = "await_explicit_stage_b_user_authorization_for_canonical_strict_metadata_replay"
    readiness_refresh_name = (
        "metadata_replay_readiness_checklist_20260524_v17"
        if closure_status
        else "metadata_replay_readiness_checklist_20260524_v14"
    )
    return {
        "schema_version": "canonical_strict_metadata_replay_stage_b_authorization_request_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "stage-b-authorization-request / no replay",
            "reopens_rejected_preset": False,
            "canonical_case": execution_request.get("scenario", "y2020_d120_s44_n240"),
        },
        "stage_b_authorization_request_ready": stage_b_request_ready,
        "stage_b_user_authorized": bool(stage_b_user_authorized),
        "runtime_provenance_closure_status_provided": bool(closure_status),
        "runtime_provenance_closure_implementation_ready": runtime_provenance_closure_implementation_ready,
        "stage_b_rerun_authorization_required": stage_b_rerun_authorization_required,
        "stage_b_rerun_authorized": stage_b_rerun_authorized,
        "authorization_source": authorization_source,
        "metadata_replay_execution_allowed": execution_allowed,
        "metadata_replay_allowed": execution_allowed,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "default_llm_rspc_v2_changed": False,
        "rejected_presets_disabled": True,
        "scope": execution_request.get("scope", "canonical_failure_only"),
        "scenario": execution_request.get("scenario", "y2020_d120_s44_n240"),
        "controller": execution_request.get("controller", "llm_rspc_v2"),
        "max_steps": int(execution_request.get("max_steps", 240) or 240),
        "selected_cache_path": execution_request.get("selected_cache_path", ""),
        "actual_protocol_implementation_for_execution": execution_request.get(
            "actual_protocol_implementation_for_execution", ""
        ),
        "protocol_v1_overlay_root": execution_request.get("protocol_v1_overlay_root", ""),
        "overlay_runner_path": execution_request.get("overlay_runner_path", ""),
        "overlay_runner_path_exists": overlay_runner_exists,
        "planned_environment": execution_request.get("planned_environment", {}),
        "planned_command": execution_request.get("planned_command", []),
        "baseline_trace_precheck": baseline_trace,
        "execution_targets": execution_request.get("execution_targets", {}),
        "post_run_audits": _stage_b_post_run_audits(
            execution_request,
            readiness_refresh_name=readiness_refresh_name,
        ),
        "stage_b_stop_conditions": [
            "stage_b_user_authorized is false",
            "stage_b_rerun_authorization_required is true and stage_b_rerun_authorized is false",
            "scope expands beyond canonical_failure_only",
            "cache fill would be required",
            "online LLM would be called",
            "controller differs from llm_rspc_v2",
            "protocol implementation differs from protocol_v1_snapshot_ephemeral_overlay",
            "baseline action-diff trace is not qualified",
        ],
        "failure_taxonomy": [
            "action_changed",
            "metadata_missing",
            "runtime_provenance_missing",
            "joint_prediction_missing",
            "cache_mismatch",
            "protocol_implementation_mismatch",
            "overlay_path_or_hash_mismatch",
        ],
        "success_continuation": "if later authorized and passed, proceed only to expanded metadata coverage planning",
        "notes": [
            "This artifact is not a replay result.",
            "Stage-B authorization is false unless an explicit authorization source is recorded.",
            "Controlled replay and performance claims remain blocked regardless of this request.",
        ],
        "next_action": next_action,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Canonical Strict Metadata Replay Stage-B Authorization Request",
        "",
        "- Mode: stage-b-authorization-request / no replay",
        f"- Stage-B request ready: {report.get('stage_b_authorization_request_ready', False)}",
        f"- Stage-B user authorized: {report.get('stage_b_user_authorized', False)}",
        f"- Metadata replay execution allowed: {report.get('metadata_replay_execution_allowed', False)}",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Scope: `{report.get('scope', '')}`",
        f"- Scenario: `{report.get('scenario', '')}`",
        f"- Controller: `{report.get('controller', '')}`",
        f"- Protocol implementation: `{report.get('actual_protocol_implementation_for_execution', '')}`",
        "",
        "## Planned Command",
        "",
        "```text",
        " ".join(str(part) for part in report.get("planned_command", []) or []),
        "```",
        "",
        "## Stop Conditions",
        "",
    ]
    for item in report.get("stage_b_stop_conditions", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Post-Run Audits", ""])
    for row in report.get("post_run_audits", []) or []:
        if isinstance(row, Mapping):
            lines.append(f"- {row.get('name', '')}: `{row.get('status', '')}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-stage-authorization-packet-json", required=True)
    parser.add_argument("--execution-request-json", required=True)
    parser.add_argument("--readiness-json", required=True)
    parser.add_argument("--runtime-provenance-closure-status-json", default="")
    parser.add_argument("--stage-b-user-authorized", action="store_true")
    parser.add_argument("--authorization-source", default="not_authorized_in_this_turn")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        two_stage_authorization_packet=_load(args.two_stage_authorization_packet_json),
        execution_request=_load(args.execution_request_json),
        readiness=_load(args.readiness_json),
        runtime_provenance_closure_status=(
            _load(args.runtime_provenance_closure_status_json)
            if args.runtime_provenance_closure_status_json
            else None
        ),
        stage_b_user_authorized=args.stage_b_user_authorized,
        authorization_source=args.authorization_source,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"stage_b_authorization_request_ready={report['stage_b_authorization_request_ready']}")
    print(f"stage_b_user_authorized={report['stage_b_user_authorized']}")
    print(f"metadata_replay_execution_allowed={report['metadata_replay_execution_allowed']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
