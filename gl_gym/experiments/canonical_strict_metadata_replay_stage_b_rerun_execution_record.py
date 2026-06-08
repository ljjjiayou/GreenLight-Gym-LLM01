"""Record explicit Stage-B rerun authorization for canonical metadata replay."""

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


def _command_has_flags(command: Sequence[Any], flags: Sequence[str]) -> bool:
    text = " ".join(str(part) for part in command)
    return all(flag in text for flag in flags)


def _replace_option(command: list[Any], option: str, value: str) -> list[Any]:
    if not value:
        return command
    out = list(command)
    for idx, part in enumerate(out):
        if str(part) == option and idx + 1 < len(out):
            out[idx + 1] = value
            return out
    out.extend([option, value])
    return out


def build_report(
    *,
    rerun_request: Mapping[str, Any],
    authorization_source: str,
    rerun_output_json: str = "",
    rerun_output_trace_dir: str = "",
) -> dict[str, Any]:
    planned_command = list(rerun_request.get("planned_command", []) or [])
    planned_command = _replace_option(planned_command, "--output-json", rerun_output_json)
    planned_command = _replace_option(planned_command, "--output-trace-dir", rerun_output_trace_dir)
    precheck = {
        "rerun_request_ready": bool(rerun_request.get("stage_b_authorization_request_ready", False)),
        "runtime_provenance_closure_implementation_ready": bool(
            rerun_request.get("runtime_provenance_closure_implementation_ready", False)
        ),
        "scope_is_canonical_failure_only": rerun_request.get("scope") == "canonical_failure_only",
        "scenario_is_canonical_failure": rerun_request.get("scenario") == "y2020_d120_s44_n240",
        "controller_is_default_llm_rspc_v2": rerun_request.get("controller") == "llm_rspc_v2",
        "max_steps_is_240": int(rerun_request.get("max_steps", 0) or 0) == 240,
        "protocol_is_v1_overlay": (
            rerun_request.get("actual_protocol_implementation_for_execution")
            == "protocol_v1_snapshot_ephemeral_overlay"
        ),
        "overlay_runner_path_exists": bool(rerun_request.get("overlay_runner_path_exists", False)),
        "baseline_trace_qualified": bool(
            dict(rerun_request.get("baseline_trace_precheck", {}) or {}).get("baseline_trace_qualified", False)
        ),
        "planned_command_has_strict_replay_flags": _command_has_flags(
            planned_command,
            [
                "--plan-cache-mode replay",
                "--plan-cache-strict",
                "--plan-cache-key-policy scenario_timestep",
            ],
        ),
        "no_cache_fill": not bool(rerun_request.get("cache_fill_run", False)),
        "no_online_llm": not bool(rerun_request.get("online_llm_called", False)),
        "controlled_replay_blocked": not bool(rerun_request.get("controlled_replay_allowed", False)),
        "performance_claim_blocked": not bool(rerun_request.get("performance_claim_allowed", False)),
    }
    precheck_pass = all(precheck.values())
    return {
        "schema_version": "canonical_strict_metadata_replay_stage_b_rerun_execution_record_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "stage-b-rerun-execution-record / strict metadata replay only",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "stage_b_authorization_request_ready": bool(rerun_request.get("stage_b_authorization_request_ready", False)),
        "stage_b_user_authorized": True,
        "stage_b_rerun_authorization_required": True,
        "stage_b_rerun_authorized": bool(precheck_pass),
        "authorization_source": authorization_source,
        "precheck_pass": bool(precheck_pass),
        "precheck": precheck,
        "metadata_replay_execution_allowed": bool(precheck_pass),
        "metadata_replay_allowed": bool(precheck_pass),
        "controlled_replay_allowed": False,
        "counterexample_replay_allowed": False,
        "expanded_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "scope": rerun_request.get("scope", "canonical_failure_only"),
        "scenario": rerun_request.get("scenario", "y2020_d120_s44_n240"),
        "controller": rerun_request.get("controller", "llm_rspc_v2"),
        "max_steps": int(rerun_request.get("max_steps", 240) or 240),
        "selected_cache_path": rerun_request.get("selected_cache_path", ""),
        "actual_protocol_implementation_for_execution": rerun_request.get(
            "actual_protocol_implementation_for_execution", ""
        ),
        "protocol_v1_overlay_root": rerun_request.get("protocol_v1_overlay_root", ""),
        "overlay_runner_path": rerun_request.get("overlay_runner_path", ""),
        "planned_environment": rerun_request.get("planned_environment", {}),
        "planned_command": planned_command,
        "rerun_output_json": rerun_output_json,
        "rerun_output_trace_dir": rerun_output_trace_dir,
        "baseline_trace_precheck": rerun_request.get("baseline_trace_precheck", {}),
        "execution_targets": rerun_request.get("execution_targets", {}),
        "post_run_audits": rerun_request.get("post_run_audits", []),
        "failure_taxonomy": [
            "action_changed",
            "metadata_missing",
            "runtime_provenance_missing",
            "joint_prediction_missing",
            "cache_mismatch",
            "protocol_implementation_mismatch",
            "overlay_path_or_hash_mismatch",
            "runtime_error",
            "strict_cache_miss",
            "unknown_post_guardrail_rewrite",
        ],
        "next_action": (
            "execute_single_canonical_strict_metadata_replay_rerun"
            if precheck_pass
            else "stage_b_rerun_precheck_failure"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Canonical Strict Metadata Replay Stage-B Rerun Execution Record",
        "",
        f"- Stage-B rerun authorized: {report.get('stage_b_rerun_authorized', False)}",
        f"- Metadata replay execution allowed: {report.get('metadata_replay_execution_allowed', False)}",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Scenario: `{report.get('scenario', '')}`",
        f"- Controller: `{report.get('controller', '')}`",
        f"- Max steps: {report.get('max_steps', 0)}",
        f"- Protocol: `{report.get('actual_protocol_implementation_for_execution', '')}`",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Precheck",
        "",
        "| check | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("precheck", {})).items():
        lines.append(f"| {key} | `{value}` |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun-request-json", required=True)
    parser.add_argument("--authorization-source", required=True)
    parser.add_argument("--rerun-output-json", default="")
    parser.add_argument("--rerun-output-trace-dir", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        rerun_request=_load(args.rerun_request_json),
        authorization_source=args.authorization_source,
        rerun_output_json=args.rerun_output_json,
        rerun_output_trace_dir=args.rerun_output_trace_dir,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"stage_b_rerun_authorized={report['stage_b_rerun_authorized']}")
    print(f"metadata_replay_execution_allowed={report['metadata_replay_execution_allowed']}")
    print(f"precheck_pass={report['precheck_pass']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["precheck_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
