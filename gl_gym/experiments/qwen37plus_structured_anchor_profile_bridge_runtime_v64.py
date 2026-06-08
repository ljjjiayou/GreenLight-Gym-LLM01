"""v64 qwen3.7-plus structured-anchor runtime profile bridge shadow rollout.

This stage validates that valid structured anchors can be bridged at runtime
into IntentContract/profile-generator candidates. The bridge is shadow-only and
does not change the active plan, final controls, Tomato Safety, fallback, or
controller promotion state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gl_gym.experiments.profile_template_opt_in_shadow_rollout_v51 as v51  # noqa: E402
import gl_gym.experiments.qwen37plus_structured_anchor_prompt_retry_shadow_rollout_v63 as v63  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (  # noqa: E402
    AUDIT_DIR,
    FAILURE_SCENARIOS,
)


MODEL_NAME = "qwen3.7-plus"
ARTIFACT_DATE = "20260602"
BENCHMARK_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "qwen37plus_structured_anchor_profile_bridge_shadow_rollout_v64_20260602"
)
V64_TRACE_DIR = BENCHMARK_DIR / "traces"
V64_CACHE_PATH = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "plan_cache"
    / "qwen37plus_structured_anchor_profile_bridge_shadow_v64_20260602.json"
)
V63_READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260602_v63.json"

ONLINE_PRECHECK_JSON = AUDIT_DIR / "online_llm_accessibility_precheck_result_v64_qwen37plus_20260602.json"
ONLINE_PRECHECK_MD = AUDIT_DIR / "online_llm_accessibility_precheck_result_v64_qwen37plus_20260602.md"
EXECUTION_RECORD_JSON = AUDIT_DIR / "qwen37plus_structured_anchor_profile_bridge_runtime_shadow_v64_execution_record_20260602.json"
EXECUTION_RECORD_MD = AUDIT_DIR / "qwen37plus_structured_anchor_profile_bridge_runtime_shadow_v64_execution_record_20260602.md"
RESULT_AUDIT_JSON = AUDIT_DIR / "qwen37plus_structured_anchor_profile_bridge_runtime_shadow_result_audit_20260602_v64.json"
RESULT_AUDIT_MD = AUDIT_DIR / "qwen37plus_structured_anchor_profile_bridge_runtime_shadow_result_audit_20260602_v64.md"
CONSISTENCY_AUDIT_JSON = AUDIT_DIR / "qwen37plus_structured_anchor_profile_bridge_runtime_consistency_audit_20260602_v64.json"
CONSISTENCY_AUDIT_MD = AUDIT_DIR / "qwen37plus_structured_anchor_profile_bridge_runtime_consistency_audit_20260602_v64.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260602_v64.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260602_v64.md"

AGENT_OVERRIDES: dict[str, Any] = {
    "structured_anchor_parser_enabled": True,
    "structured_anchor_shadow_only": True,
    "structured_anchor_compact_json_prompt_enabled": True,
    "structured_anchor_retry_invalid_or_empty_enabled": True,
    "structured_anchor_retry_max_attempts": 1,
    "structured_anchor_list_max_items": 3,
    "structured_anchor_profile_bridge_enabled": True,
    "structured_anchor_profile_bridge_shadow_only": True,
    "structured_anchor_profile_bridge_record_provenance": True,
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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _final_cache_entries(cache_path: str | Path = V64_CACHE_PATH) -> list[tuple[str, dict[str, Any]]]:
    return [v63._final_event_entry("", items) for items in v63._group_events(cache_path).values()]


def _scenario_from_env(env_id: str, *, max_steps: int = 720) -> str:
    return v63._scenario_from_env(env_id, max_steps=max_steps)


def _bridge_from_entry(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = entry.get("parsed_plan", {}) if isinstance(entry.get("parsed_plan", {}), Mapping) else {}
    bridge = plan.get("structured_anchor_profile_bridge", {}) if isinstance(plan, Mapping) else {}
    if not isinstance(bridge, Mapping) or not bridge:
        bridge = entry.get("structured_anchor_profile_bridge", {}) if isinstance(entry.get("structured_anchor_profile_bridge", {}), Mapping) else {}
    return bridge if isinstance(bridge, Mapping) else {}


def build_online_precheck(*, probe_online: bool = True) -> dict[str, Any]:
    result = v63.build_online_precheck(probe_online=probe_online)
    result["artifact"] = "online_llm_accessibility_precheck_result_v64_qwen37plus_20260602"
    result["model_name"] = MODEL_NAME
    return result


def _v63_ready(path: str | Path = V63_READINESS_JSON) -> bool:
    readiness = _load_json(path)
    return bool(
        readiness.get("model_name") == MODEL_NAME
        and readiness.get("structured_anchor_prompt_retry_shadow_rollout_pass", False)
        and readiness.get("next_action") == "structured_anchor_to_profile_generator_runtime_shadow_bridge_plan"
    )


def _command_for_group(group: Mapping[str, str]) -> list[str]:
    output_json = BENCHMARK_DIR / f"qwen37plus_structured_anchor_profile_bridge_shadow_rollout_v64_{group['name']}.json"
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
        "--llm-max-iterations",
        "2",
        "--llm-max-tokens",
        "2048",
        "--plan-cache-mode",
        "record",
        "--plan-cache-path",
        _rel(V64_CACHE_PATH),
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--agent-config-overrides",
        json.dumps(AGENT_OVERRIDES, sort_keys=True),
        "--output-json",
        _rel(output_json),
        "--output-trace-dir",
        _rel(V64_TRACE_DIR),
    ]


def build_execution_record(
    precheck: Mapping[str, Any] | None = None,
    *,
    v63_readiness_json: str | Path = V63_READINESS_JSON,
) -> dict[str, Any]:
    precheck = dict(precheck or build_online_precheck(probe_online=True))
    v63_ready = _v63_ready(v63_readiness_json)
    model_matches = str(precheck.get("model_name", "")) == MODEL_NAME
    executable = bool(v63_ready and model_matches and precheck.get("online_llm_accessible", False))
    blocked_reason = ""
    if not v63_ready:
        blocked_reason = "v63_structured_anchor_prompt_retry_not_ready"
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
        "artifact": "qwen37plus_structured_anchor_profile_bridge_runtime_shadow_v64_execution_record_20260602",
        "authorization_source": "user_delegated_qwen37plus_structured_anchor_profile_bridge_runtime_shadow_authorization_20260602",
        "scope": "minimal_qwen37plus_structured_anchor_profile_bridge_runtime_shadow_only",
        "model_name": MODEL_NAME,
        "scenarios": list(FAILURE_SCENARIOS),
        "controller": "llm_rspc_v2",
        "controlled_controller_included": False,
        "max_steps": 720,
        "llm_max_iterations": 2,
        "agent_config_overrides": dict(AGENT_OVERRIDES),
        "cache_path": _rel(V64_CACHE_PATH),
        "output_dir": _rel(BENCHMARK_DIR),
        "trace_dir": _rel(V64_TRACE_DIR),
        "online_llm_allowed": True,
        "online_llm_scope": "only these three v64 qwen3.7-plus structured-anchor runtime bridge shadow scenarios",
        "online_llm_credentials_present": bool(precheck.get("online_llm_credentials_present", False)),
        "online_llm_accessible": bool(precheck.get("online_llm_accessible", False)),
        "v63_structured_anchor_prompt_retry_ready": bool(v63_ready),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "executable": bool(executable),
        "blocked_reason": blocked_reason,
        "commands": commands,
    }


def build_runtime_bridge_audit(cache_path: str | Path = V64_CACHE_PATH) -> dict[str, Any]:
    row_results: list[dict[str, Any]] = []
    for cache_key, entry in _final_cache_entries(cache_path):
        structured = entry.get("structured_anchor", {}) if isinstance(entry.get("structured_anchor", {}), Mapping) else {}
        bridge = _bridge_from_entry(entry)
        scenario = _scenario_from_env(str(entry.get("env_id") or ""))
        valid = bool(structured.get("valid", False))
        bridgeable = bool(bridge.get("bridgeable", False))
        profile_candidate_count = int(bridge.get("profile_candidate_count", 0) or 0)
        row_results.append(
            {
                "cache_key": cache_key,
                "scenario_id": scenario,
                "env_id": str(entry.get("env_id") or ""),
                "timestep": _as_int(entry.get("timestep"), 0),
                "valid_structured_anchor": valid,
                "runtime_bridge_attempted": bool(bridge.get("attempted", False) or bridge),
                "bridgeable": bridgeable,
                "failure_reason": str(bridge.get("failure_reason", "") or ""),
                "profile_candidate_count": profile_candidate_count,
                "selected_shadow_profile_name": str(bridge.get("selected_shadow_profile_name", "") or ""),
                "hard_safety_profile_violation": bool(bridge.get("hard_safety_profile_violation", False)),
                "final_control_change": bool(bridge.get("final_control_change", False)),
                "current_plan_modified": bool(bridge.get("current_plan_modified", False)),
                "low_level_action_generated": bool(bridge.get("low_level_action_generated", False)),
                "bridge": dict(bridge),
            }
        )

    valid_rows = [row for row in row_results if row["valid_structured_anchor"]]
    bridgeable_rows = [row for row in valid_rows if row["bridgeable"]]
    candidate_counts = [int(row["profile_candidate_count"]) for row in bridgeable_rows]
    failure_reasons = Counter(str(row.get("failure_reason") or "") for row in valid_rows if not row["bridgeable"])
    valid_count = len(valid_rows)
    runtime_bridge_attempt_count = sum(bool(row["runtime_bridge_attempted"]) for row in valid_rows)
    bridgeable_count = len(bridgeable_rows)
    hard_safety_count = sum(bool(row["hard_safety_profile_violation"]) for row in bridgeable_rows)
    acceptance = {
        "runtime_bridge_attempt_rate_ge_0_95": bool(valid_count and runtime_bridge_attempt_count / valid_count >= 0.95),
        "bridgeable_rate_ge_0_95": bool(valid_count and bridgeable_count / valid_count >= 0.95),
        "profile_candidate_generation_error_count_zero": bool(valid_count - bridgeable_count == 0),
        "profile_candidate_count_positive": bool(candidate_counts and min(candidate_counts) > 0),
        "hard_safety_profile_violation_count_zero": bool(hard_safety_count == 0),
        "final_control_change_count_zero": bool(sum(bool(row["final_control_change"]) for row in bridgeable_rows) == 0),
        "current_plan_modified_count_zero": bool(sum(bool(row["current_plan_modified"]) for row in bridgeable_rows) == 0),
        "low_level_action_generated_count_zero": bool(sum(bool(row["low_level_action_generated"]) for row in bridgeable_rows) == 0),
    }
    acceptance["runtime_bridge_contract_pass"] = bool(all(acceptance.values()))
    return {
        "artifact": "qwen37plus_structured_anchor_profile_bridge_runtime_consistency_audit_20260602_v64",
        "current_stage": "v64_qwen37plus_structured_anchor_profile_bridge_runtime_shadow",
        "scope": "runtime_structured_anchor_to_profile_generator_bridge_contract_only",
        "model_name": MODEL_NAME,
        "cache_path": _rel(cache_path),
        "valid_structured_anchor_count": int(valid_count),
        "runtime_bridge_attempt_count": int(runtime_bridge_attempt_count),
        "runtime_bridge_attempt_rate": float(runtime_bridge_attempt_count / valid_count) if valid_count else 0.0,
        "bridgeable_count": int(bridgeable_count),
        "bridgeable_rate": float(bridgeable_count / valid_count) if valid_count else 0.0,
        "profile_candidate_generation_error_count": int(valid_count - bridgeable_count),
        "profile_candidate_count_min": int(min(candidate_counts) if candidate_counts else 0),
        "profile_candidate_count_max": int(max(candidate_counts) if candidate_counts else 0),
        "hard_safety_profile_violation_count": int(hard_safety_count),
        "final_control_change_count": int(sum(bool(row["final_control_change"]) for row in bridgeable_rows)),
        "current_plan_modified_count": int(sum(bool(row["current_plan_modified"]) for row in bridgeable_rows)),
        "low_level_action_generated_count": int(sum(bool(row["low_level_action_generated"]) for row in bridgeable_rows)),
        "failure_reason_counts": dict(sorted((key, value) for key, value in failure_reasons.items() if key)),
        "sample_rows": row_results[:10],
        "acceptance": acceptance,
        "runtime_bridge_contract_pass": bool(acceptance["runtime_bridge_contract_pass"]),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


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


def build_result_audit(
    *,
    trace_dir: str | Path = V64_TRACE_DIR,
    cache_path: str | Path = V64_CACHE_PATH,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scenario_reports = []
    model_passes = []
    runtime_error_total = 0
    short_trace_count = 0
    for scenario in scenarios:
        rows = _read_trace_rows(trace_dir, scenario)
        model = _model_consistency(rows)
        model_passes.append(bool(model["model_consistency_pass"]))
        runtime_errors = int(sum(bool(row.get("runtime_error")) for row in rows))
        runtime_error_total += runtime_errors
        short_trace = bool(len(rows) < 720)
        short_trace_count += int(short_trace)
        scenario_reports.append(
            {
                "scenario_id": scenario,
                "trace_exists": bool(rows),
                "rows": int(len(rows)),
                "short_trace_before_horizon": short_trace,
                "runtime_error_steps": runtime_errors,
                "structured_anchor_attempt_rows": int(sum(_truthy(row.get("structured_anchor_attempted")) for row in rows)),
                "structured_anchor_valid_rows": int(sum(_truthy(row.get("structured_anchor_valid")) for row in rows)),
                "runtime_bridge_rows": int(sum(_truthy(row.get("structured_anchor_profile_bridge_bridgeable")) for row in rows)),
                "model_consistency": model,
            }
        )

    retry_audit = v63.build_retry_outcome_audit(cache_path)
    retry_audit.update(
        {
            "artifact": "qwen37plus_structured_anchor_retry_outcome_audit_20260602_v64",
            "current_stage": "v64_qwen37plus_structured_anchor_profile_bridge_runtime_shadow",
            "cache_path": _rel(cache_path),
        }
    )
    bridge_audit = build_runtime_bridge_audit(cache_path)
    bridge_acceptance = bridge_audit.get("acceptance", {}) if isinstance(bridge_audit.get("acceptance", {}), Mapping) else {}
    model_consistency_pass = bool(model_passes and all(model_passes))
    runtime_stability_pass = bool(runtime_error_total == 0 and short_trace_count == 0)
    acceptance = {
        "model_consistency_pass": model_consistency_pass,
        "provider_error_steps_zero": bool(retry_audit["provider_error_steps"] == 0),
        "valid_structured_anchor_rate_ge_0_95": bool(retry_audit["valid_structured_anchor_rate"] >= 0.95),
        "runtime_bridge_attempt_rate_ge_0_95": bool(bridge_acceptance.get("runtime_bridge_attempt_rate_ge_0_95", False)),
        "bridgeable_rate_ge_0_95": bool(bridge_acceptance.get("bridgeable_rate_ge_0_95", False)),
        "profile_candidate_generation_error_count_zero": bool(
            bridge_acceptance.get("profile_candidate_generation_error_count_zero", False)
        ),
        "profile_candidate_count_positive": bool(bridge_acceptance.get("profile_candidate_count_positive", False)),
        "hard_safety_profile_violation_count_zero": bool(bridge_acceptance.get("hard_safety_profile_violation_count_zero", False)),
        "final_control_change_count_zero": bool(bridge_acceptance.get("final_control_change_count_zero", False)),
        "current_plan_modified_count_zero": bool(bridge_acceptance.get("current_plan_modified_count_zero", False)),
        "low_level_action_generated_count_zero": bool(bridge_acceptance.get("low_level_action_generated_count_zero", False)),
        "runtime_stability_pass": runtime_stability_pass,
    }
    bridge_contract_keys = [key for key in acceptance if key != "runtime_stability_pass"]
    acceptance["runtime_bridge_contract_pass"] = bool(all(acceptance[key] for key in bridge_contract_keys))
    acceptance["v64_full_stability_pass"] = bool(acceptance["runtime_bridge_contract_pass"] and runtime_stability_pass)
    return (
        {
            "artifact": "qwen37plus_structured_anchor_profile_bridge_runtime_shadow_result_audit_20260602_v64",
            "current_stage": "v64_qwen37plus_structured_anchor_profile_bridge_runtime_shadow",
            "scope": "structured_anchor_runtime_profile_bridge_contract_only",
            "model_name": MODEL_NAME,
            "trace_dir": _rel(trace_dir),
            "cache_path": _rel(cache_path),
            "scenarios": list(scenarios),
            "scenario_reports": scenario_reports,
            "model_consistency_pass": model_consistency_pass,
            "provider_error_steps": int(retry_audit["provider_error_steps"]),
            "structured_anchor_attempt_count": int(retry_audit["structured_anchor_attempt_count"]),
            "valid_structured_anchor_rate": float(retry_audit["valid_structured_anchor_rate"]),
            "runtime_bridge_attempt_rate": float(bridge_audit["runtime_bridge_attempt_rate"]),
            "bridgeable_rate": float(bridge_audit["bridgeable_rate"]),
            "profile_candidate_generation_error_count": int(bridge_audit["profile_candidate_generation_error_count"]),
            "profile_candidate_count_min": int(bridge_audit["profile_candidate_count_min"]),
            "hard_safety_profile_violation_count": int(bridge_audit["hard_safety_profile_violation_count"]),
            "final_control_change_count": int(bridge_audit["final_control_change_count"]),
            "current_plan_modified_count": int(bridge_audit["current_plan_modified_count"]),
            "low_level_action_generated_count": int(bridge_audit["low_level_action_generated_count"]),
            "runtime_error_steps": int(runtime_error_total),
            "short_trace_scenario_count": int(short_trace_count),
            "runtime_bridge_contract_pass": bool(acceptance["runtime_bridge_contract_pass"]),
            "runtime_stability_pass": bool(runtime_stability_pass),
            "acceptance": acceptance,
            "next_action": (
                "structured_anchor_profile_bridge_h720_shadow_expansion_plan"
                if acceptance["v64_full_stability_pass"]
                else (
                    "runtime_failure_attribution_after_structured_bridge"
                    if acceptance["runtime_bridge_contract_pass"]
                    else "structured_anchor_profile_bridge_instrumentation_patch_plan"
                )
            ),
            "controlled_replay_allowed": False,
            "controlled_replay_execution_allowed": False,
            "metadata_replay_execution_allowed": False,
            "performance_claim_allowed": False,
            "promotion_evidence": False,
        },
        bridge_audit,
    )


def _stop_taxonomy(result_audit: Mapping[str, Any], execution_record: Mapping[str, Any] | None = None) -> list[str]:
    record = execution_record or {}
    if not bool(record.get("executable", True)):
        return [str(record.get("blocked_reason") or "execution_not_authorized")]
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    taxonomy: list[str] = []
    if not acceptance.get("model_consistency_pass", False):
        taxonomy.append("model_consistency_failed")
    if not acceptance.get("provider_error_steps_zero", False):
        taxonomy.append("provider_error")
    if not acceptance.get("valid_structured_anchor_rate_ge_0_95", False):
        taxonomy.append("structured_anchor_valid_rate_low")
    if not acceptance.get("runtime_bridge_attempt_rate_ge_0_95", False):
        taxonomy.append("runtime_bridge_attempt_rate_low")
    if not acceptance.get("bridgeable_rate_ge_0_95", False):
        taxonomy.append("bridgeable_rate_low")
    if not acceptance.get("profile_candidate_generation_error_count_zero", False):
        taxonomy.append("profile_candidate_generation_error")
    if not acceptance.get("hard_safety_profile_violation_count_zero", False):
        taxonomy.append("hard_safety_profile_violation")
    if not acceptance.get("final_control_change_count_zero", False):
        taxonomy.append("final_control_changed")
    if not acceptance.get("current_plan_modified_count_zero", False):
        taxonomy.append("current_plan_modified")
    if not acceptance.get("low_level_action_generated_count_zero", False):
        taxonomy.append("low_level_action_generated")
    if not acceptance.get("runtime_stability_pass", False):
        taxonomy.append("runtime_or_short_trace_persists")
    return sorted(set(taxonomy or ["gate_insufficient"]))


def build_readiness(result_audit: Mapping[str, Any], execution_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    bridge_pass = bool(result_audit.get("runtime_bridge_contract_pass", False))
    runtime_pass = bool(result_audit.get("runtime_stability_pass", False))
    executable = bool((execution_record or {}).get("executable", False))
    return {
        "artifact": "metadata_replay_readiness_checklist_20260602_v64",
        "current_stage": "v64_qwen37plus_structured_anchor_profile_bridge_runtime_shadow",
        "model_name": MODEL_NAME,
        "execution_record_executable": bool(executable),
        "runtime_bridge_contract_pass": bool(bridge_pass),
        "runtime_stability_pass": bool(runtime_pass),
        "model_consistency_pass": bool(result_audit.get("model_consistency_pass", False)),
        "provider_error_steps": int(result_audit.get("provider_error_steps", 0) or 0),
        "structured_anchor_attempt_count": int(result_audit.get("structured_anchor_attempt_count", 0) or 0),
        "valid_structured_anchor_rate": float(result_audit.get("valid_structured_anchor_rate", 0.0) or 0.0),
        "runtime_bridge_attempt_rate": float(result_audit.get("runtime_bridge_attempt_rate", 0.0) or 0.0),
        "bridgeable_rate": float(result_audit.get("bridgeable_rate", 0.0) or 0.0),
        "profile_candidate_generation_error_count": int(result_audit.get("profile_candidate_generation_error_count", 0) or 0),
        "profile_candidate_count_min": int(result_audit.get("profile_candidate_count_min", 0) or 0),
        "hard_safety_profile_violation_count": int(result_audit.get("hard_safety_profile_violation_count", 0) or 0),
        "runtime_error_steps": int(result_audit.get("runtime_error_steps", 0) or 0),
        "short_trace_scenario_count": int(result_audit.get("short_trace_scenario_count", 0) or 0),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": (
            "structured_anchor_profile_bridge_h720_shadow_expansion_plan"
            if bridge_pass and runtime_pass
            else (
                "runtime_failure_attribution_after_structured_bridge"
                if bridge_pass
                else "structured_anchor_profile_bridge_instrumentation_patch_plan"
            )
        ),
        "stop_taxonomy": [] if bridge_pass and runtime_pass else _stop_taxonomy(result_audit, execution_record),
    }


def write_online_precheck(precheck: Mapping[str, Any]) -> None:
    _write_json_md(ONLINE_PRECHECK_JSON, ONLINE_PRECHECK_MD, precheck, "v64 qwen3.7-plus Online LLM Accessibility Precheck")


def write_execution_record(record: Mapping[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    V64_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    _write_json_md(EXECUTION_RECORD_JSON, EXECUTION_RECORD_MD, record, "v64 qwen3.7-plus Structured Anchor Runtime Profile Bridge Execution Record")


def write_result_artifacts(result_audit: Mapping[str, Any], consistency_audit: Mapping[str, Any], readiness: Mapping[str, Any]) -> None:
    _write_json_md(RESULT_AUDIT_JSON, RESULT_AUDIT_MD, result_audit, "v64 qwen3.7-plus Runtime Profile Bridge Result Audit")
    _write_json_md(CONSISTENCY_AUDIT_JSON, CONSISTENCY_AUDIT_MD, consistency_audit, "v64 qwen3.7-plus Runtime Profile Bridge Consistency Audit")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v64 Metadata Replay Readiness")


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
    result, consistency = build_result_audit()
    if run_results:
        result["execution_run_results"] = run_results
    readiness = build_readiness(result, execution)
    write_result_artifacts(result, consistency, readiness)
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
                    "runtime_bridge_contract_pass": result.get("runtime_bridge_contract_pass"),
                    "runtime_stability_pass": result.get("runtime_stability_pass"),
                    "next_action": readiness.get("next_action"),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
