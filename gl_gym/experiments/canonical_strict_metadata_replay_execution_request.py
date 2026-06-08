"""Build the final canonical strict metadata replay execution request.

This artifact is a request package only. It does not execute replay, fill
cache, call an online LLM, or authorize execution by itself.
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


def _overlay_runner_path(overlay_root: str | Path | None) -> str:
    if not overlay_root:
        return ""
    return str(Path(overlay_root) / "gl_gym" / "experiments" / "run_frozen_benchmark.py")


def _planned_command(
    *,
    overlay_runner_path: str,
    output_dir: str,
    selected_cache_path: str,
    scenario_year: int,
    scenario_day: int,
    scenario_seed: int,
    controller: str,
    max_steps: int,
    key_policy: str,
) -> list[str]:
    return [
        sys.executable,
        overlay_runner_path,
        "--years",
        str(scenario_year),
        "--days",
        str(scenario_day),
        "--seeds",
        str(scenario_seed),
        "--controllers",
        controller,
        "--max-steps",
        str(max_steps),
        "--plan-cache-mode",
        "replay",
        "--plan-cache-path",
        selected_cache_path,
        "--plan-cache-strict",
        "--plan-cache-key-policy",
        key_policy,
        "--output-json",
        str(Path(output_dir) / "canonical_strict_metadata_replay_20260523.json"),
        "--output-trace-dir",
        str(Path(output_dir) / "traces"),
    ]


def build_report(
    *,
    authorization_packet: Mapping[str, Any],
    readiness: Mapping[str, Any],
    strict_run_plan: Mapping[str, Any],
    overlay_preflight: Mapping[str, Any],
    cache_coverage: Mapping[str, Any],
    baseline_trace_manifest: Mapping[str, Any],
    output_benchmark_dir: str = "gl_gym/result/benchmarks/canonical_strict_metadata_replay_20260523",
) -> dict[str, Any]:
    fixed_inputs = dict(strict_run_plan.get("fixed_inputs", {}) or {})
    scenario_rows = list(strict_run_plan.get("allowed_initial_scenarios", []) or [])
    scenario = dict(scenario_rows[0]) if scenario_rows and isinstance(scenario_rows[0], Mapping) else {}
    scenario_id = str(scenario.get("scenario_id", authorization_packet.get("scenario", "y2020_d120_s44_n240")))
    controller = str(fixed_inputs.get("controller", authorization_packet.get("controller", "llm_rspc_v2")))
    max_steps = int(fixed_inputs.get("max_steps", authorization_packet.get("max_steps", 240)) or 240)
    selected_cache_path = str(fixed_inputs.get("selected_cache_path", authorization_packet.get("selected_cache_path", "")))
    key_policy = str(fixed_inputs.get("plan_cache_key_policy", authorization_packet.get("plan_cache_key_policy", "scenario_timestep")))
    overlay_root = str(overlay_preflight.get("protocol_v1_overlay_root", authorization_packet.get("protocol_v1_overlay_path", "")) or "")
    overlay_runner = _overlay_runner_path(overlay_root)
    overlay_runner_exists = bool(overlay_runner and _resolve(overlay_runner).exists())
    protocol_impl = str(authorization_packet.get("actual_protocol_implementation_for_execution", ""))
    baseline_qualified = bool(baseline_trace_manifest.get("baseline_trace_qualified", False))
    action_diff_status = (
        "planned"
        if baseline_qualified
        else baseline_trace_manifest.get("action_diff_audit_status", "blocked_missing_baseline_trace")
    )
    request_ready = bool(
        readiness.get("canonical_metadata_replay_authorization_request_ready", False)
        and authorization_packet.get("authorization_packet_ready", False)
        and overlay_preflight.get("protocol_v1_overlay_validated", False)
        and cache_coverage.get("cache_coverage_pass", False)
        and strict_run_plan.get("strict_metadata_replay_run_plan_ready", False)
        and protocol_impl == "protocol_v1_snapshot_ephemeral_overlay"
        and overlay_runner_exists
    )
    output_dir = str(_resolve(output_benchmark_dir))
    command = _planned_command(
        overlay_runner_path=overlay_runner,
        output_dir=output_dir,
        selected_cache_path=selected_cache_path,
        scenario_year=int(scenario.get("year", 2020) or 2020),
        scenario_day=int(scenario.get("day", 120) or 120),
        scenario_seed=int(scenario.get("seed", 44) or 44),
        controller=controller,
        max_steps=max_steps,
        key_policy=key_policy,
    )
    post_run_audits = [
        {
            "name": "trace_action_diff_audit",
            "status": action_diff_status,
            "baseline_trace_dir": baseline_trace_manifest.get("selected_baseline_trace_dir"),
            "required_for_action_invariance_claim": True,
        },
        {
            "name": "post_guardrail_runtime_provenance_audit",
            "status": "planned",
            "required_for_metadata_completeness": True,
        },
        {
            "name": "post_guardrail_joint_prediction_readiness",
            "status": "planned",
            "required_for_metadata_completeness": True,
        },
        {
            "name": "strict_metadata_replay_summary_20260523",
            "status": "planned",
            "required_for_result_interpretation": True,
        },
        {
            "name": "metadata_replay_readiness_checklist_20260523_v12",
            "status": "planned",
            "required_for_gate_refresh": True,
        },
    ]
    return {
        "schema_version": "canonical_strict_metadata_replay_execution_request_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "execution-request-only / no replay",
            "reopens_rejected_preset": False,
            "canonical_case": scenario_id,
        },
        "canonical_strict_metadata_replay_execution_request_ready": request_ready,
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
        "scope": "canonical_failure_only",
        "scenario": scenario_id,
        "controller": controller,
        "max_steps": max_steps,
        "selected_cache_path": selected_cache_path,
        "plan_cache_mode": "replay",
        "plan_cache_strict": True,
        "plan_cache_key_policy": key_policy,
        "actual_protocol_implementation_for_execution": protocol_impl,
        "protocol_v1_overlay_root": overlay_root,
        "overlay_runner_path": overlay_runner,
        "overlay_runner_path_exists": overlay_runner_exists,
        "planned_environment": {
            "PYTHONPATH": f"{overlay_root};{PROJECT_ROOT}",
        },
        "planned_command": command,
        "output_benchmark_dir": output_dir,
        "baseline_trace_precheck": {
            "baseline_trace_manifest_ready": bool(baseline_trace_manifest.get("baseline_trace_manifest_ready", False)),
            "baseline_trace_qualified": baseline_qualified,
            "action_diff_audit_status": action_diff_status,
            "selected_baseline_trace_dir": baseline_trace_manifest.get("selected_baseline_trace_dir"),
            "selected_baseline_csv": baseline_trace_manifest.get("selected_baseline_csv"),
            "missing_baseline_trace_policy": "replay_request_can_exist_but_action_diff_audit_must_be_blocked",
        },
        "execution_targets": {
            "cache_hit_rate": 1.0,
            "runtime_error_steps": 0,
            "strict_cache_miss_runtime_error_steps": 0,
            "runtime_provenance_record_count": ">0",
            "runtime_reason_missing_count": 0,
            "unknown_post_guardrail_rewrite_count": 0,
            "joint_prediction_missing_fields": 0,
            "action_diff_steps": 0 if baseline_qualified else "blocked_missing_baseline_trace",
            "max_abs_delta": 0 if baseline_qualified else "blocked_missing_baseline_trace",
            "selected_control": "unchanged_if_baseline_trace_qualified",
            "post_guardrail_final_action": "unchanged_if_baseline_trace_qualified",
        },
        "post_run_audits": post_run_audits,
        "stop_conditions": [
            "user has not explicitly authorized canonical strict metadata replay execution",
            "protocol implementation is not protocol_v1_snapshot_ephemeral_overlay",
            "overlay runner path is missing",
            "cache coverage is not pass=true for canonical_failure_only",
            "online LLM would be called",
            "cache fill would be required",
            "controller differs from llm_rspc_v2",
            "scope expands beyond canonical_failure_only",
        ],
        "failure_taxonomy": [
            "action_changed",
            "metadata_missing",
            "runtime_provenance_missing",
            "joint_prediction_missing",
            "cache_mismatch",
            "protocol_implementation_mismatch",
        ],
        "success_continuation": "if authorized and passed, proceed only to expanded metadata coverage planning; controlled replay remains false",
        "notes": [
            "This request does not authorize or execute replay.",
            "Canonical cache coverage cannot be generalized to expanded families.",
            "Protocol v2 baseline remains unauthorized and blocked categories remain excluded from promotion evidence.",
        ],
        "next_action": "await_explicit_user_authorization_for_canonical_strict_metadata_replay",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Canonical Strict Metadata Replay Execution Request",
        "",
        "- Mode: execution-request-only / no replay",
        f"- Request ready: {report.get('canonical_strict_metadata_replay_execution_request_ready', False)}",
        f"- User explicit authorization required: {report.get('user_explicit_authorization_required', False)}",
        "- Metadata replay execution allowed: false",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Scope: `{report.get('scope', '')}`",
        f"- Scenario: `{report.get('scenario', '')}`",
        f"- Controller: `{report.get('controller', '')}`",
        f"- Protocol implementation: `{report.get('actual_protocol_implementation_for_execution', '')}`",
        f"- Overlay runner exists: {report.get('overlay_runner_path_exists', False)}",
        f"- Baseline action-diff status: `{dict(report.get('baseline_trace_precheck', {})).get('action_diff_audit_status', '')}`",
        "",
        "## Planned Command",
        "",
        "```text",
        " ".join(str(part) for part in report.get("planned_command", []) or []),
        "```",
        "",
        "## Post-Run Audits",
        "",
    ]
    for row in report.get("post_run_audits", []) or []:
        if isinstance(row, Mapping):
            lines.append(f"- {row.get('name', '')}: `{row.get('status', '')}`")
    lines.extend(["", "## Stop Conditions", ""])
    for item in report.get("stop_conditions", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Failure Taxonomy", ""])
    for item in report.get("failure_taxonomy", []) or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-packet-json", required=True)
    parser.add_argument("--readiness-json", required=True)
    parser.add_argument("--strict-run-plan-json", required=True)
    parser.add_argument("--overlay-preflight-json", required=True)
    parser.add_argument("--cache-coverage-json", required=True)
    parser.add_argument("--baseline-trace-manifest-json", required=True)
    parser.add_argument("--output-benchmark-dir", default="gl_gym/result/benchmarks/canonical_strict_metadata_replay_20260523")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        authorization_packet=_load(args.authorization_packet_json),
        readiness=_load(args.readiness_json),
        strict_run_plan=_load(args.strict_run_plan_json),
        overlay_preflight=_load(args.overlay_preflight_json),
        cache_coverage=_load(args.cache_coverage_json),
        baseline_trace_manifest=_load(args.baseline_trace_manifest_json),
        output_benchmark_dir=args.output_benchmark_dir,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"execution_request_ready={report['canonical_strict_metadata_replay_execution_request_ready']}")
    print(f"metadata_replay_execution_allowed={report['metadata_replay_execution_allowed']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
