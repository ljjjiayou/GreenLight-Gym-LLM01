"""Build the strict metadata replay execution authorization packet.

The packet is authorization material only. It does not execute metadata replay,
does not fill cache, does not call an online LLM, and does not change
controller behavior.
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


def _snapshot_modified_files(protocol_v1_snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in protocol_v1_snapshot.get("files", []) or []:
        if not isinstance(item, Mapping):
            continue
        status = str(item.get("working_tree_status", "") or "")
        if status:
            rows.append(
                {
                    "path": item.get("path", ""),
                    "working_tree_status": status,
                    "sha256": item.get("sha256", ""),
                    "object_available": bool(item.get("object_available", False)),
                }
            )
    return rows


def _initial_scenario(run_plan: Mapping[str, Any]) -> dict[str, Any]:
    rows = run_plan.get("allowed_initial_scenarios", []) or []
    for row in rows:
        if isinstance(row, Mapping):
            return dict(row)
    return {}


def _overlay_status(
    *,
    overlay_preflight: Mapping[str, Any] | None,
    v1_overlay_path: str,
) -> dict[str, Any]:
    if isinstance(overlay_preflight, Mapping):
        return {
            "path": overlay_preflight.get("protocol_v1_overlay_root") or None,
            "available": bool(overlay_preflight.get("protocol_v1_overlay_available", False)),
            "validated": bool(overlay_preflight.get("protocol_v1_overlay_validated", False)),
            "schema_version": overlay_preflight.get("schema_version", "not_provided"),
            "next_action": overlay_preflight.get("next_action", "not_provided"),
        }
    return {
        "path": v1_overlay_path or None,
        "available": bool(v1_overlay_path and _resolve(v1_overlay_path).exists()),
        "validated": False,
        "schema_version": "not_provided",
        "next_action": "protocol_v1_ephemeral_overlay_preflight_required",
    }


def build_report(
    *,
    strict_run_plan: Mapping[str, Any],
    protocol_v1_snapshot: Mapping[str, Any],
    readiness: Mapping[str, Any],
    overlay_preflight: Mapping[str, Any] | None = None,
    v1_overlay_path: str = "",
) -> dict[str, Any]:
    fixed_inputs = dict(strict_run_plan.get("fixed_inputs", {}) or {})
    scenario = _initial_scenario(strict_run_plan)
    modified_protocol_files = _snapshot_modified_files(protocol_v1_snapshot)
    working_tree_protocol_files_modified = bool(modified_protocol_files)
    overlay = _overlay_status(overlay_preflight=overlay_preflight, v1_overlay_path=v1_overlay_path)
    overlay_available = bool(overlay["available"])
    overlay_validated = bool(overlay["validated"])
    actual_protocol_implementation = (
        "protocol_v1_snapshot_ephemeral_overlay"
        if overlay_validated
        else "protocol_v1_snapshot_ephemeral_overlay_required"
    )
    execution_scope = "canonical_failure_only"
    post_run_audits = [
        "strict_metadata_replay_summary",
        "trace_action_diff_audit",
        "post_guardrail_runtime_provenance_audit",
        "post_guardrail_joint_prediction_readiness_refresh",
        "metadata_replay_readiness_checklist_v12",
    ]
    failure_taxonomy = [
        "action_changed",
        "metadata_missing",
        "runtime_provenance_missing",
        "joint_prediction_missing",
        "cache_mismatch",
        "protocol_implementation_mismatch",
    ]
    return {
        "schema_version": "strict_metadata_replay_execution_authorization_packet_v2",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "authorization-packet-only / no replay",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "authorization_packet_ready": bool(strict_run_plan.get("strict_metadata_replay_run_plan_ready", False)),
        "user_explicit_authorization_required": True,
        "metadata_replay_execution_allowed": False,
        "metadata_replay_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "default_llm_rspc_v2_changed": False,
        "rejected_presets_disabled": True,
        "execution_scope": execution_scope,
        "scenario": scenario.get("scenario_id", "y2020_d120_s44_n240"),
        "env_id": scenario.get("env_id", "TomatoEnv_y2020_d120_s44"),
        "controller": fixed_inputs.get("controller", "llm_rspc_v2"),
        "max_steps": int(fixed_inputs.get("max_steps", 240) or 240),
        "selected_cache_path": fixed_inputs.get("selected_cache_path", ""),
        "plan_cache_key_policy": fixed_inputs.get("plan_cache_key_policy", "scenario_timestep"),
        "actual_protocol_implementation_for_execution": actual_protocol_implementation,
        "protocol_v1_snapshot": {
            "snapshot_available": bool(protocol_v1_snapshot.get("snapshot_available", False)),
            "source_git_ref": protocol_v1_snapshot.get("source_git_ref", "HEAD"),
            "source_git_commit": protocol_v1_snapshot.get("source_git_commit", ""),
            "tracked_baseline_file_count": protocol_v1_snapshot.get("tracked_baseline_file_count", 0),
        },
        "current_working_tree_protocol_files_modified": working_tree_protocol_files_modified,
        "modified_protocol_files": modified_protocol_files,
        "protocol_v1_overlay_path": overlay["path"],
        "protocol_v1_overlay_available": overlay_available,
        "protocol_v1_overlay_validated": overlay_validated,
        "protocol_v1_overlay_preflight_schema_version": overlay["schema_version"],
        "protocol_v1_overlay_preflight_next_action": overlay["next_action"],
        "protocol_v1_overlay_validation_required": True,
        "protocol_implementation_execution_blocker": (
            "v1 snapshot ephemeral overlay must be built and validated before execution; do not fall back to current working tree protocol"
            if working_tree_protocol_files_modified and not overlay_validated
            else "v1 snapshot ephemeral overlay validated; execution still requires explicit user authorization"
        ),
        "strict_metadata_replay_run_plan_ready": bool(strict_run_plan.get("strict_metadata_replay_run_plan_ready", False)),
        "readiness_next_action": readiness.get("next_action", "not_provided"),
        "execution_targets": dict(strict_run_plan.get("targets", {}) or {}),
        "post_run_audits": post_run_audits,
        "stop_conditions": [
            "metadata replay execution is not explicitly authorized by the user",
            "protocol v1 ephemeral overlay cannot be built or validated",
            "cache coverage is not canonical_failure_only and pass=true",
            "online LLM would be called",
            "cache fill would be required",
            "selected controller differs from llm_rspc_v2",
            "rejected preset would be enabled",
        ],
        "failure_taxonomy": failure_taxonomy,
        "success_continuation": "if authorized and passed, proceed only to expanded metadata coverage planning; controlled replay remains false",
        "notes": [
            "This packet requests authorization material only and does not grant execution permission by itself.",
            "The current working tree has modified protocol/runner/test oracle files; current working tree protocol must not be used as promotion evidence.",
            "The initial replay scope is canonical_failure_only and cannot be expanded without separate cache coverage and authorization.",
        ],
        "next_action": "await_explicit_user_authorization_for_canonical_strict_metadata_replay",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strict Metadata Replay Execution Authorization Packet",
        "",
        "- Mode: authorization-packet-only / no replay",
        "- Metadata replay execution allowed: false",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- User explicit authorization required: {report.get('user_explicit_authorization_required', False)}",
        f"- Execution scope: `{report.get('execution_scope', '')}`",
        f"- Scenario: `{report.get('scenario', '')}`",
        f"- Protocol implementation: `{report.get('actual_protocol_implementation_for_execution', '')}`",
        f"- Protocol v1 overlay available: {report.get('protocol_v1_overlay_available', False)}",
        "",
        "## Protocol Implementation Boundary",
        "",
        f"- Current working-tree protocol files modified: `{report.get('current_working_tree_protocol_files_modified', False)}`",
        f"- Blocker: {report.get('protocol_implementation_execution_blocker', '')}",
        "",
        "## Post-Run Audits",
        "",
    ]
    for item in report.get("post_run_audits", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Stop Conditions", ""])
    for item in report.get("stop_conditions", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Failure Taxonomy", ""])
    for item in report.get("failure_taxonomy", []) or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-run-plan-json", required=True)
    parser.add_argument("--protocol-v1-snapshot-json", required=True)
    parser.add_argument("--readiness-json", required=True)
    parser.add_argument("--protocol-v1-overlay-preflight-json", default="")
    parser.add_argument("--v1-overlay-path", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        strict_run_plan=_load(args.strict_run_plan_json),
        protocol_v1_snapshot=_load(args.protocol_v1_snapshot_json),
        readiness=_load(args.readiness_json),
        overlay_preflight=(
            _load(args.protocol_v1_overlay_preflight_json)
            if args.protocol_v1_overlay_preflight_json
            else None
        ),
        v1_overlay_path=args.v1_overlay_path,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"authorization_packet_ready={report['authorization_packet_ready']}")
    print(f"metadata_replay_execution_allowed={report['metadata_replay_execution_allowed']}")
    print(f"protocol_v1_overlay_available={report['protocol_v1_overlay_available']}")
    print(f"protocol_v1_overlay_validated={report['protocol_v1_overlay_validated']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
