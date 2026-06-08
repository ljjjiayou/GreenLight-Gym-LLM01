"""v59 qwen3.7 structured-anchor opt-in shadow rollout artifacts.

This validates only the high-level structured planning-anchor contract. The
LLM is not allowed to emit executable actuator commands, and parsed anchors are
recorded as shadow provenance without changing final control.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gl_gym.experiments.profile_template_opt_in_shadow_rollout_v51 as v51  # noqa: E402
import gl_gym.experiments.profile_template_opt_in_shadow_rollout_v52 as v52  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (  # noqa: E402
    AUDIT_DIR,
    FAILURE_SCENARIOS,
)


MODEL_NAME = "qwen3.7-max"
BENCHMARK_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "qwen37_structured_anchor_opt_in_shadow_rollout_v59_20260601"
)
V59_TRACE_DIR = BENCHMARK_DIR / "traces"
V59_CACHE_PATH = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "plan_cache"
    / "qwen37_structured_anchor_opt_in_shadow_v59_20260601.json"
)
V58_READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v58.json"

ONLINE_PRECHECK_JSON = AUDIT_DIR / "online_llm_accessibility_precheck_result_v59_qwen37_20260601.json"
ONLINE_PRECHECK_MD = AUDIT_DIR / "online_llm_accessibility_precheck_result_v59_qwen37_20260601.md"
EXECUTION_RECORD_JSON = AUDIT_DIR / "qwen37_structured_anchor_opt_in_shadow_rollout_v59_execution_record_20260601.json"
EXECUTION_RECORD_MD = AUDIT_DIR / "qwen37_structured_anchor_opt_in_shadow_rollout_v59_execution_record_20260601.md"
RESULT_AUDIT_JSON = AUDIT_DIR / "qwen37_structured_anchor_opt_in_shadow_rollout_result_audit_20260601_v59.json"
RESULT_AUDIT_MD = AUDIT_DIR / "qwen37_structured_anchor_opt_in_shadow_rollout_result_audit_20260601_v59.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v59.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v59.md"

AGENT_OVERRIDES: dict[str, Any] = {
    "structured_anchor_parser_enabled": True,
    "structured_anchor_shadow_only": True,
    "max_tokens": 2048,
    "transition_gate_enabled": False,
    "profile_feasibility_gate_enabled": False,
    "profile_template_patch_enabled": False,
    "fallback_post_selection_veto_enabled": False,
    "recovery_anchor_enabled": False,
}
SCENARIO_GROUPS = list(v51.SCENARIO_GROUPS)


def _rel(path: str | Path) -> str:
    return v51._rel(path)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json_md(path_json: Path, path_md: Path, data: Mapping[str, Any], title: str) -> None:
    v51._write_json_md(path_json, path_md, data, title)


def build_online_precheck(*, probe_online: bool = True, model_name: str = MODEL_NAME) -> dict[str, Any]:
    result = v52.build_online_precheck(probe_online=probe_online, model_name=model_name)
    result["artifact"] = "online_llm_accessibility_precheck_result_v59_qwen37_20260601"
    result["model_name"] = model_name
    return result


def _v58_passed(v58_readiness_json: str | Path = V58_READINESS_JSON) -> bool:
    readiness = _load_json(v58_readiness_json)
    return bool(
        readiness.get("structured_anchor_parser_ready", False)
        and readiness.get("next_action") == "structured_anchor_opt_in_qwen37_shadow_rollout_plan"
    )


def _command_for_group(group: Mapping[str, str]) -> list[str]:
    output_json = BENCHMARK_DIR / f"qwen37_structured_anchor_opt_in_shadow_rollout_v59_{group['name']}.json"
    return [
        "python",
        "gl_gym\\experiments\\run_frozen_benchmark.py",
        "--years",
        str(group["years"]),
        "--days",
        str(group["days"]),
        "--seeds",
        str(group["seeds"]),
        "--controllers",
        "llm_rspc_v2",
        "--max-steps",
        "720",
        "--llm-model",
        MODEL_NAME,
        "--plan-cache-mode",
        "record",
        "--plan-cache-path",
        _rel(V59_CACHE_PATH),
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--agent-config-overrides",
        json.dumps(AGENT_OVERRIDES, sort_keys=True),
        "--output-json",
        _rel(output_json),
        "--output-trace-dir",
        _rel(V59_TRACE_DIR),
    ]


def build_execution_record(
    precheck: Mapping[str, Any] | None = None,
    *,
    v58_readiness_json: str | Path = V58_READINESS_JSON,
) -> dict[str, Any]:
    precheck = dict(precheck or build_online_precheck(probe_online=True))
    v58_pass = _v58_passed(v58_readiness_json)
    model_matches = str(precheck.get("model_name", "")) == MODEL_NAME
    executable = bool(v58_pass and model_matches and precheck.get("online_llm_accessible", False))
    blocked_reason = ""
    if not v58_pass:
        blocked_reason = "v58_structured_anchor_parser_not_ready"
    elif not model_matches:
        blocked_reason = "model_name_mismatch"
    elif not precheck.get("online_llm_accessible", False):
        blocked_reason = str(precheck.get("blocked_reason") or "online_llm_accessibility_blocked")
    commands = [
        {
            "group": group["name"],
            "years": group["years"],
            "days": group["days"],
            "seeds": group["seeds"],
            "argv": _command_for_group(group),
            "command": " ".join(_command_for_group(group)),
        }
        for group in SCENARIO_GROUPS
    ]
    return {
        "artifact": "qwen37_structured_anchor_opt_in_shadow_rollout_v59_execution_record_20260601",
        "authorization_source": "user_delegated_qwen37_structured_anchor_opt_in_shadow_rollout_authorization_20260601",
        "scope": "minimal_qwen37_structured_anchor_opt_in_shadow_rollout_only",
        "model_name": MODEL_NAME,
        "scenarios": list(FAILURE_SCENARIOS),
        "controller": "llm_rspc_v2",
        "controlled_controller_included": False,
        "max_steps": 720,
        "agent_config_overrides": dict(AGENT_OVERRIDES),
        "structured_anchor_parser_enabled": True,
        "structured_anchor_shadow_only": True,
        "transition_gate_enabled": False,
        "profile_feasibility_gate_enabled": False,
        "profile_template_patch_enabled": False,
        "fallback_post_selection_veto_enabled": False,
        "recovery_anchor_enabled": False,
        "plan_cache_mode": "record",
        "cache_path": _rel(V59_CACHE_PATH),
        "output_dir": _rel(BENCHMARK_DIR),
        "trace_dir": _rel(V59_TRACE_DIR),
        "online_llm_allowed": True,
        "online_llm_scope": "only these three v59 qwen3.7-max structured-anchor shadow rollout scenarios",
        "online_llm_credentials_present": bool(precheck.get("online_llm_credentials_present", False)),
        "online_llm_accessible": bool(precheck.get("online_llm_accessible", False)),
        "provider_error_detected": bool(precheck.get("provider_error_detected", False)),
        "v58_structured_anchor_parser_ready": bool(v58_pass),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "executable": bool(executable),
        "blocked_reason": blocked_reason,
        "commands": commands,
    }


def _plan_cache_entries(cache_path: str | Path = V59_CACHE_PATH) -> list[tuple[str, dict[str, Any]]]:
    data = _load_json(cache_path)
    entries = data.get("entries", data if isinstance(data, dict) else {})
    if not isinstance(entries, Mapping):
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for key, value in entries.items():
        if isinstance(value, Mapping):
            out.append((str(key), dict(value)))
    return out


def _read_trace_rows(trace_dir: str | Path, scenario: str) -> list[dict[str, Any]]:
    return v51._read_trace_rows(trace_dir, scenario)


def _model_consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted({str(row.get("model_name", "") or "") for row in rows})
    missing_count = sum(1 for row in rows if not str(row.get("model_name", "") or "").strip())
    provider_error_rows = [
        row
        for row in rows
        if any(
            token in str(row.get("runtime_error", "") or "").lower()
            for token in ("provider", "access denied", "permission", "api", "dashscope", "bailian")
        )
    ]
    return {
        "model_name_values": values,
        "model_name_missing_count": int(missing_count),
        "provider_error_row_count": int(len(provider_error_rows)),
        "model_consistency_pass": bool(rows and missing_count == 0 and values == [MODEL_NAME] and not provider_error_rows),
    }


def _structured_anchor_cache_audit(cache_path: str | Path = V59_CACHE_PATH) -> dict[str, Any]:
    entries = _plan_cache_entries(cache_path)
    attempts = 0
    valid = 0
    empty = 0
    final_leaks = 0
    invalid_clean = 0
    legacy_tool_actions = 0
    provider_errors = 0
    samples = []
    for key, entry in entries:
        anchor = entry.get("structured_anchor", {}) if isinstance(entry, Mapping) else {}
        if not isinstance(anchor, Mapping):
            continue
        if not bool(anchor.get("enabled", False)):
            continue
        attempts += 1
        is_valid = bool(anchor.get("valid", False))
        valid += int(is_valid)
        empty += int(bool(anchor.get("empty", False)))
        final_leaks += int(anchor.get("final_control_field_leak_count", 0) or 0)
        invalid_clean += int((not is_valid) and bool(anchor.get("clean_planning_evidence", False)))
        legacy_tool_actions += int(bool(anchor.get("legacy_tool_action_present", False)))
        provider_errors += int(
            any(
                token in str(entry.get("raw_response", "") or "").lower()
                for token in ("provider", "access denied", "permission", "dashscope", "bailian")
            )
        )
        if len(samples) < 8:
            samples.append(
                {
                    "key": key,
                    "valid": is_valid,
                    "empty": bool(anchor.get("empty", False)),
                    "errors": list(anchor.get("errors", []) or []),
                    "shadow_plan": dict(anchor.get("shadow_plan", {}) or {}),
                }
            )
    valid_rate = float(valid / attempts) if attempts else 0.0
    empty_rate = float(empty / attempts) if attempts else 0.0
    return {
        "cache_path": _rel(cache_path),
        "structured_anchor_attempt_count": int(attempts),
        "valid_structured_anchor_count": int(valid),
        "invalid_structured_anchor_count": int(max(attempts - valid, 0)),
        "valid_structured_anchor_rate": valid_rate,
        "empty_structured_anchor_count": int(empty),
        "empty_structured_anchor_rate": empty_rate,
        "final_control_field_leak_count": int(final_leaks),
        "invalid_anchor_clean_planning_evidence_count": int(invalid_clean),
        "legacy_tool_action_present_count": int(legacy_tool_actions),
        "provider_error_steps": int(provider_errors),
        "sample_entries": samples,
    }


def build_result_audit(
    *,
    trace_dir: str | Path = V59_TRACE_DIR,
    cache_path: str | Path = V59_CACHE_PATH,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    scenario_reports = []
    model_passes = []
    for scenario in scenarios:
        rows = _read_trace_rows(trace_dir, scenario)
        model = _model_consistency(rows)
        model_passes.append(bool(model["model_consistency_pass"]))
        scenario_reports.append(
            {
                "scenario_id": scenario,
                "trace_exists": bool(rows),
                "rows": int(len(rows)),
                "runtime_error_steps": int(sum(bool(row.get("runtime_error")) for row in rows)),
                "structured_anchor_attempt_rows": int(sum(bool(row.get("structured_anchor_attempted")) for row in rows)),
                "structured_anchor_valid_rows": int(sum(bool(row.get("structured_anchor_valid")) for row in rows)),
                "structured_anchor_final_control_field_leak_rows": int(
                    sum(int(row.get("structured_anchor_final_control_field_leak_count", 0) or 0) for row in rows)
                ),
                "model_consistency": model,
            }
        )
    cache_audit = _structured_anchor_cache_audit(cache_path)
    model_consistency_pass = bool(model_passes and all(model_passes))
    acceptance = {
        "model_consistency_pass": model_consistency_pass,
        "provider_error_steps_zero": bool(cache_audit["provider_error_steps"] == 0),
        "structured_anchor_attempt_count_positive": bool(cache_audit["structured_anchor_attempt_count"] > 0),
        "valid_structured_anchor_rate_ge_0_80": bool(cache_audit["valid_structured_anchor_rate"] >= 0.80),
        "final_control_field_leak_count_zero": bool(cache_audit["final_control_field_leak_count"] == 0),
        "empty_structured_anchor_rate_le_0_10": bool(cache_audit["empty_structured_anchor_rate"] <= 0.10),
        "invalid_anchor_not_clean_evidence": bool(cache_audit["invalid_anchor_clean_planning_evidence_count"] == 0),
    }
    acceptance["v59_acceptance_pass"] = bool(all(acceptance.values()))
    next_action = (
        "structured_anchor_to_profile_generator_shadow_bridge_plan"
        if acceptance["v59_acceptance_pass"]
        else "structured_anchor_prompt_schema_repair_plan"
    )
    return {
        "artifact": "qwen37_structured_anchor_opt_in_shadow_rollout_result_audit_20260601_v59",
        "current_stage": "v59_qwen37_structured_anchor_opt_in_shadow_rollout",
        "scope": "structured_anchor_output_contract_only",
        "model_name": MODEL_NAME,
        "trace_dir": _rel(trace_dir),
        "cache_path": _rel(cache_path),
        "scenarios": list(scenarios),
        "scenario_reports": scenario_reports,
        "cache_audit": cache_audit,
        "model_consistency_pass": model_consistency_pass,
        "provider_error_steps": int(cache_audit["provider_error_steps"]),
        "structured_anchor_attempt_count": int(cache_audit["structured_anchor_attempt_count"]),
        "valid_structured_anchor_rate": float(cache_audit["valid_structured_anchor_rate"]),
        "final_control_field_leak_count": int(cache_audit["final_control_field_leak_count"]),
        "empty_structured_anchor_rate": float(cache_audit["empty_structured_anchor_rate"]),
        "acceptance": acceptance,
        "structured_anchor_shadow_rollout_pass": bool(acceptance["v59_acceptance_pass"]),
        "next_action": next_action,
        "stop_taxonomy": [] if acceptance["v59_acceptance_pass"] else _stop_taxonomy({"acceptance": acceptance}),
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def _stop_taxonomy(result_audit: Mapping[str, Any], execution_record: Mapping[str, Any] | None = None) -> list[str]:
    record = execution_record or {}
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    if not bool(record.get("executable", True)):
        blocked = str(record.get("blocked_reason") or "")
        return [blocked] if blocked else ["execution_not_authorized"]
    taxonomy: list[str] = []
    if not acceptance.get("model_consistency_pass", False):
        taxonomy.append("model_consistency_failed")
    if not acceptance.get("provider_error_steps_zero", False):
        taxonomy.append("provider_error")
    if not acceptance.get("structured_anchor_attempt_count_positive", False):
        taxonomy.append("structured_anchor_not_attempted")
    if not acceptance.get("valid_structured_anchor_rate_ge_0_80", False):
        taxonomy.append("structured_anchor_valid_rate_low")
    if not acceptance.get("final_control_field_leak_count_zero", False):
        taxonomy.append("final_control_field_leak")
    if not acceptance.get("empty_structured_anchor_rate_le_0_10", False):
        taxonomy.append("empty_structured_anchor_rate_high")
    if not acceptance.get("invalid_anchor_not_clean_evidence", False):
        taxonomy.append("invalid_anchor_clean_evidence")
    return sorted(set(taxonomy or ["gate_insufficient"]))


def build_readiness(result_audit: Mapping[str, Any], execution_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    executable = bool((execution_record or {}).get("executable", False))
    passed = bool(acceptance.get("v59_acceptance_pass", False))
    if not executable:
        next_action = str((execution_record or {}).get("blocked_reason") or "online_llm_accessibility_blocked")
    elif passed:
        next_action = "structured_anchor_to_profile_generator_shadow_bridge_plan"
    else:
        next_action = "structured_anchor_prompt_schema_repair_plan"
    return {
        "artifact": "metadata_replay_readiness_checklist_20260601_v59",
        "current_stage": "v59_qwen37_structured_anchor_opt_in_shadow_rollout",
        "model_name": MODEL_NAME,
        "structured_anchor_shadow_rollout_done": bool(result_audit.get("structured_anchor_attempt_count", 0) > 0),
        "structured_anchor_shadow_rollout_pass": bool(passed),
        "execution_record_executable": bool(executable),
        "online_llm_accessible": bool((execution_record or {}).get("online_llm_accessible", False)),
        "model_consistency_pass": bool(acceptance.get("model_consistency_pass", False)),
        "provider_error_steps": int(result_audit.get("provider_error_steps", 0) or 0),
        "structured_anchor_attempt_count": int(result_audit.get("structured_anchor_attempt_count", 0) or 0),
        "valid_structured_anchor_rate": float(result_audit.get("valid_structured_anchor_rate", 0.0) or 0.0),
        "final_control_field_leak_count": int(result_audit.get("final_control_field_leak_count", 0) or 0),
        "empty_structured_anchor_rate": float(result_audit.get("empty_structured_anchor_rate", 0.0) or 0.0),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": next_action,
        "stop_taxonomy": [] if passed else _stop_taxonomy(result_audit, execution_record),
    }


def write_online_precheck(precheck: Mapping[str, Any]) -> None:
    _write_json_md(ONLINE_PRECHECK_JSON, ONLINE_PRECHECK_MD, precheck, "v59 Qwen3.7 Online LLM Accessibility Precheck")


def write_execution_record(record: Mapping[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    V59_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    _write_json_md(EXECUTION_RECORD_JSON, EXECUTION_RECORD_MD, record, "v59 Qwen3.7 Structured Anchor Opt-In Shadow Rollout Execution Record")


def write_result_artifacts(result_audit: Mapping[str, Any], readiness: Mapping[str, Any]) -> None:
    _write_json_md(RESULT_AUDIT_JSON, RESULT_AUDIT_MD, result_audit, "v59 Qwen3.7 Structured Anchor Opt-In Shadow Rollout Result Audit")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v59 Metadata Replay Readiness")


def run_execution_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = []
    if not bool(record.get("executable", False)):
        return [{"skipped": True, "blocked_reason": str(record.get("blocked_reason", ""))}]
    for item in record.get("commands", []) or []:
        argv = [str(part) for part in item.get("argv", [])]
        completed = subprocess.run(argv, cwd=PROJECT_ROOT, check=False)
        results.append({"group": item.get("group"), "returncode": int(completed.returncode), "command": item.get("command")})
        if completed.returncode != 0:
            break
    return results


def write_all(*, probe_online: bool = True, run_rollout: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    precheck = build_online_precheck(probe_online=probe_online)
    write_online_precheck(precheck)
    execution = build_execution_record(precheck)
    write_execution_record(execution)
    run_results = []
    if run_rollout:
        run_results = run_execution_record(execution)
    result = build_result_audit()
    if run_results:
        result["execution_run_results"] = run_results
    readiness = build_readiness(result, execution)
    write_result_artifacts(result, readiness)
    return precheck, execution, result, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-all", action="store_true")
    parser.add_argument("--run-rollout", action="store_true")
    parser.add_argument("--skip-online-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    precheck, execution, result, readiness = write_all(
        probe_online=not args.skip_online_probe,
        run_rollout=bool(args.run_rollout),
    )
    if not args.write_all:
        print(
            json.dumps(
                {
                    "model_name": MODEL_NAME,
                    "online_llm_accessible": precheck.get("online_llm_accessible"),
                    "executable": execution.get("executable"),
                    "v59_acceptance_pass": (result.get("acceptance", {}) or {}).get("v59_acceptance_pass"),
                    "valid_structured_anchor_rate": result.get("valid_structured_anchor_rate"),
                    "next_action": readiness.get("next_action"),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
