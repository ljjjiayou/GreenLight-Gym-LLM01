"""Build a two-stage authorization packet for canonical strict metadata replay.

Stage A records protocol v1 overlay build/validation evidence. Stage B remains
explicitly unauthorized by default. This script does not execute replay, fill
cache, call an online LLM, or grant execution permission.
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


def build_report(
    *,
    overlay_preflight: Mapping[str, Any],
    execution_request: Mapping[str, Any],
    readiness: Mapping[str, Any],
    overlay_preflight_path: str = "",
    execution_request_path: str = "",
    stage_b_user_authorized: bool = False,
) -> dict[str, Any]:
    stage_a_overlay_validated = bool(overlay_preflight.get("protocol_v1_overlay_validated", False))
    stage_a_overlay_available = bool(overlay_preflight.get("protocol_v1_overlay_available", False))
    stage_a_overlay_hash_match = int(overlay_preflight.get("hash_mismatch_count", 0) or 0) == 0
    stage_a_overlay_validation_complete = bool(
        stage_a_overlay_available
        and stage_a_overlay_validated
        and stage_a_overlay_hash_match
        and bool(overlay_preflight.get("compile_pass", False))
        and bool(overlay_preflight.get("import_pass", False))
    )
    protocol_impl = str(execution_request.get("actual_protocol_implementation_for_execution", ""))
    execution_request_ready = bool(
        execution_request.get("canonical_strict_metadata_replay_execution_request_ready", False)
    )
    stage_b_ready_for_user_decision = bool(
        stage_a_overlay_validation_complete
        and execution_request_ready
        and readiness.get("canonical_metadata_replay_authorization_request_ready", False)
        and protocol_impl == "protocol_v1_snapshot_ephemeral_overlay"
    )
    return {
        "schema_version": "strict_metadata_replay_two_stage_authorization_packet_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "authorization-closure-only / no replay",
            "reopens_rejected_preset": False,
            "canonical_case": execution_request.get("scenario", "y2020_d120_s44_n240"),
        },
        "two_stage_authorization_packet_ready": stage_b_ready_for_user_decision,
        "metadata_replay_execution_allowed": False,
        "metadata_replay_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "default_llm_rspc_v2_changed": False,
        "rejected_presets_disabled": True,
        "actual_protocol_implementation_for_execution": protocol_impl,
        "stage_a": {
            "name": "protocol_v1_overlay_build_validation",
            "overlay_build_validation_complete": stage_a_overlay_validation_complete,
            "overlay_available": stage_a_overlay_available,
            "overlay_validated": stage_a_overlay_validated,
            "overlay_hash_match": stage_a_overlay_hash_match,
            "hash_mismatch_count": int(overlay_preflight.get("hash_mismatch_count", 0) or 0),
            "object_missing_count": int(overlay_preflight.get("object_missing_count", 0) or 0),
            "compile_pass": bool(overlay_preflight.get("compile_pass", False)),
            "import_pass": bool(overlay_preflight.get("import_pass", False)),
            "overlay_root": overlay_preflight.get("protocol_v1_overlay_root"),
            "evidence_path": overlay_preflight_path,
        },
        "stage_b": {
            "name": "canonical_strict_metadata_replay_execution",
            "canonical_replay_user_authorized": bool(stage_b_user_authorized),
            "stage_b_ready_for_user_decision": stage_b_ready_for_user_decision,
            "execution_request_ready": execution_request_ready,
            "execution_request_path": execution_request_path,
            "scope": execution_request.get("scope", "canonical_failure_only"),
            "scenario": execution_request.get("scenario", "y2020_d120_s44_n240"),
            "controller": execution_request.get("controller", "llm_rspc_v2"),
            "max_steps": int(execution_request.get("max_steps", 240) or 240),
            "selected_cache_path": execution_request.get("selected_cache_path", ""),
            "baseline_trace_qualified": bool(
                dict(execution_request.get("baseline_trace_precheck", {})).get("baseline_trace_qualified", False)
            ),
            "metadata_replay_execution_allowed": False,
            "execution_permission_reason": "Stage B requires separate explicit user authorization after Stage A validation",
        },
        "stage_b_stop_conditions": [
            "canonical_replay_user_authorized is false",
            "protocol implementation is not protocol_v1_snapshot_ephemeral_overlay",
            "cache fill would be required",
            "online LLM would be called",
            "scope expands beyond canonical_failure_only",
            "controller differs from llm_rspc_v2",
            "rejected preset would be enabled",
        ],
        "stage_b_success_continuation": "if later authorized and passed, proceed only to expanded metadata coverage planning; controlled replay remains false",
        "notes": [
            "Stage A overlay validation is evidence for implementation feasibility only.",
            "Stage A complete does not authorize Stage B replay execution.",
            "This packet intentionally preserves metadata_replay_execution_allowed=false.",
        ],
        "next_action": "await_explicit_stage_b_user_authorization_for_canonical_strict_metadata_replay"
        if stage_b_ready_for_user_decision
        else "repair_two_stage_authorization_inputs",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    stage_a = dict(report.get("stage_a", {}))
    stage_b = dict(report.get("stage_b", {}))
    lines = [
        "# Strict Metadata Replay Two-Stage Authorization Packet",
        "",
        "- Mode: authorization-closure-only / no replay",
        f"- Two-stage packet ready: {report.get('two_stage_authorization_packet_ready', False)}",
        "- Metadata replay execution allowed: false",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Protocol implementation: `{report.get('actual_protocol_implementation_for_execution', '')}`",
        "",
        "## Stage A - Overlay Build / Validation",
        "",
        f"- Overlay build validation complete: {stage_a.get('overlay_build_validation_complete', False)}",
        f"- Overlay available: {stage_a.get('overlay_available', False)}",
        f"- Overlay validated: {stage_a.get('overlay_validated', False)}",
        f"- Overlay hash match: {stage_a.get('overlay_hash_match', False)}",
        f"- Evidence: `{stage_a.get('evidence_path', '')}`",
        "",
        "## Stage B - Canonical Replay Execution",
        "",
        f"- Stage B ready for user decision: {stage_b.get('stage_b_ready_for_user_decision', False)}",
        f"- Canonical replay user authorized: {stage_b.get('canonical_replay_user_authorized', False)}",
        f"- Scenario: `{stage_b.get('scenario', '')}`",
        f"- Controller: `{stage_b.get('controller', '')}`",
        f"- Scope: `{stage_b.get('scope', '')}`",
        "",
        "## Stage B Stop Conditions",
        "",
    ]
    for item in report.get("stage_b_stop_conditions", []) or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-preflight-json", required=True)
    parser.add_argument("--execution-request-json", required=True)
    parser.add_argument("--readiness-json", required=True)
    parser.add_argument("--stage-b-user-authorized", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        overlay_preflight=_load(args.overlay_preflight_json),
        execution_request=_load(args.execution_request_json),
        readiness=_load(args.readiness_json),
        overlay_preflight_path=args.overlay_preflight_json,
        execution_request_path=args.execution_request_json,
        stage_b_user_authorized=args.stage_b_user_authorized,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"two_stage_authorization_packet_ready={report['two_stage_authorization_packet_ready']}")
    print(f"metadata_replay_execution_allowed={report['metadata_replay_execution_allowed']}")
    print(f"stage_b_user_authorized={report['stage_b']['canonical_replay_user_authorized']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
