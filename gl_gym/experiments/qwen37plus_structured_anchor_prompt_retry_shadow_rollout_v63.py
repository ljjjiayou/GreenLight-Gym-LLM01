"""v63 qwen3.7-plus structured-anchor compact prompt/retry shadow rollout.

This stage validates the structured-anchor output contract only. It does not
run controlled replay, promote a controller, or let the LLM directly command
actuators.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gl_gym.experiments.profile_template_opt_in_shadow_rollout_v51 as v51  # noqa: E402
import gl_gym.experiments.qwen37plus_structured_anchor_failure_diagnosis_v62 as v62  # noqa: E402
import gl_gym.experiments.qwen37plus_structured_anchor_opt_in_shadow_rollout_v61 as v61  # noqa: E402
import gl_gym.experiments.qwen37plus_structured_anchor_profile_bridge_v60 as v60  # noqa: E402
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
    / "qwen37plus_structured_anchor_prompt_retry_shadow_rollout_v63_20260602"
)
V63_TRACE_DIR = BENCHMARK_DIR / "traces"
V63_CACHE_PATH = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "plan_cache"
    / "qwen37plus_structured_anchor_prompt_retry_shadow_v63_20260602.json"
)
V62_READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260602_v62.json"

ONLINE_PRECHECK_JSON = AUDIT_DIR / "online_llm_accessibility_precheck_result_v63_qwen37plus_20260602.json"
ONLINE_PRECHECK_MD = AUDIT_DIR / "online_llm_accessibility_precheck_result_v63_qwen37plus_20260602.md"
EXECUTION_RECORD_JSON = AUDIT_DIR / "qwen37plus_structured_anchor_prompt_retry_opt_in_shadow_rollout_v63_execution_record_20260602.json"
EXECUTION_RECORD_MD = AUDIT_DIR / "qwen37plus_structured_anchor_prompt_retry_opt_in_shadow_rollout_v63_execution_record_20260602.md"
RESULT_AUDIT_JSON = AUDIT_DIR / "qwen37plus_structured_anchor_prompt_retry_shadow_rollout_result_audit_20260602_v63.json"
RESULT_AUDIT_MD = AUDIT_DIR / "qwen37plus_structured_anchor_prompt_retry_shadow_rollout_result_audit_20260602_v63.md"
RETRY_AUDIT_JSON = AUDIT_DIR / "qwen37plus_structured_anchor_retry_outcome_audit_20260602_v63.json"
RETRY_AUDIT_MD = AUDIT_DIR / "qwen37plus_structured_anchor_retry_outcome_audit_20260602_v63.md"
BRIDGE_AUDIT_JSON = AUDIT_DIR / "qwen37plus_structured_anchor_profile_bridge_audit_20260602_v63.json"
BRIDGE_AUDIT_MD = AUDIT_DIR / "qwen37plus_structured_anchor_profile_bridge_audit_20260602_v63.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260602_v63.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260602_v63.md"

AGENT_OVERRIDES: dict[str, Any] = {
    "structured_anchor_parser_enabled": True,
    "structured_anchor_shadow_only": True,
    "structured_anchor_compact_json_prompt_enabled": True,
    "structured_anchor_retry_invalid_or_empty_enabled": True,
    "structured_anchor_retry_max_attempts": 1,
    "structured_anchor_list_max_items": 3,
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


def _cache_entries(cache_path: str | Path = V63_CACHE_PATH) -> list[tuple[str, dict[str, Any]]]:
    data = _load_json(cache_path)
    entries = data.get("entries", data if isinstance(data, Mapping) else {})
    if not isinstance(entries, Mapping):
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for key, value in entries.items():
        if isinstance(value, Mapping):
            item = dict(value)
            item.setdefault("key", str(key))
            out.append((str(key), item))
    return out


def _event_key(entry: Mapping[str, Any]) -> tuple[str, int]:
    return (str(entry.get("env_id") or ""), int(float(entry.get("timestep", 0) or 0)))


def _scenario_from_env(env_id: str, *, max_steps: int = 720) -> str:
    text = str(env_id or "")
    if text.startswith("TomatoEnv_"):
        text = text[len("TomatoEnv_") :]
    return text if "_n" in text else f"{text}_n{max_steps}"


def _attempt(entry: Mapping[str, Any]) -> int:
    structured = entry.get("structured_anchor", {})
    if isinstance(structured, Mapping):
        value = structured.get("attempt", entry.get("attempt", 0))
    else:
        value = entry.get("attempt", 0)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _structured(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    value = entry.get("structured_anchor", {})
    return value if isinstance(value, Mapping) else {}


def _valid(entry: Mapping[str, Any]) -> bool:
    return bool(_structured(entry).get("valid", False))


def _errors(entry: Mapping[str, Any]) -> list[str]:
    return [str(item) for item in _structured(entry).get("errors", []) or []]


def _group_events(cache_path: str | Path = V63_CACHE_PATH) -> dict[tuple[str, int], list[tuple[str, dict[str, Any]]]]:
    grouped: dict[tuple[str, int], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for key, entry in _cache_entries(cache_path):
        scenario = _scenario_from_env(str(entry.get("env_id") or ""))
        if scenario in set(FAILURE_SCENARIOS):
            grouped[_event_key(entry)].append((key, entry))
    for items in grouped.values():
        items.sort(key=lambda item: (_attempt(item[1]), item[0]))
    return dict(grouped)


def _final_event_entry(key: str, items: Sequence[tuple[str, dict[str, Any]]]) -> tuple[str, dict[str, Any]]:
    for item_key, entry in items:
        if _valid(entry):
            return item_key, entry
    return items[-1] if items else (key, {})


def _provider_error(raw: Any) -> bool:
    text = str(raw or "").lower()
    return any(token in text for token in ("provider", "access denied", "permission", "dashscope", "bailian", "api error"))


def build_retry_outcome_audit(cache_path: str | Path = V63_CACHE_PATH) -> dict[str, Any]:
    grouped = _group_events(cache_path)
    event_reports: list[dict[str, Any]] = []
    final_failure_counts: Counter[str] = Counter()
    attempt_failure_counts: Counter[str] = Counter()
    retry_success_count = 0
    events_with_retry_count = 0
    final_valid_count = 0
    final_empty_count = 0
    final_control_leaks = 0
    provider_error_steps = 0

    for (env_id, timestep), items in sorted(grouped.items(), key=lambda pair: (pair[0][0], pair[0][1])):
        final_key, final_entry = _final_event_entry("", items)
        final_structured = _structured(final_entry)
        final_valid = bool(final_structured.get("valid", False))
        final_valid_count += int(final_valid)
        final_empty_count += int((not final_valid) and bool(final_structured.get("empty", False)))
        final_control_leaks += int(final_structured.get("final_control_field_leak_count", 0) or 0)
        provider_error_steps += int(any(_provider_error(entry.get("raw_response")) for _, entry in items))

        attempt_reports = []
        for item_key, entry in items:
            structured = _structured(entry)
            primary = ""
            if bool(structured.get("attempted", False)) and not bool(structured.get("valid", False)):
                primary, _, _ = v62.classify_invalid_entry(entry)
                attempt_failure_counts[primary] += 1
            attempt_reports.append(
                {
                    "key": item_key,
                    "attempt": _attempt(entry),
                    "valid": bool(structured.get("valid", False)),
                    "empty": bool(structured.get("empty", False)),
                    "errors": _errors(entry),
                    "primary_failure_type": primary,
                    "clean_planning_evidence": bool(structured.get("clean_planning_evidence", False)),
                    "retry_provenance": dict(structured.get("retry_provenance", {}) or {}),
                }
            )

        retry_attempts = [report for report in attempt_reports if int(report.get("attempt", 0) or 0) > 1]
        events_with_retry_count += int(bool(retry_attempts))
        retry_success_count += int(final_valid and bool(retry_attempts))
        if not final_valid:
            primary = attempt_reports[-1].get("primary_failure_type") or "unknown_requires_instrumentation"
            final_failure_counts[str(primary)] += 1
        event_reports.append(
            {
                "env_id": env_id,
                "scenario_id": _scenario_from_env(env_id),
                "timestep": int(timestep),
                "attempt_count": int(len(items)),
                "retry_attempted": bool(retry_attempts),
                "retry_success": bool(final_valid and retry_attempts),
                "final_key": final_key,
                "final_attempt": int(_attempt(final_entry)),
                "final_valid": bool(final_valid),
                "final_clean_planning_evidence": bool(final_structured.get("clean_planning_evidence", False)),
                "attempts": attempt_reports,
            }
        )

    event_count = len(event_reports)
    final_valid_rate = float(final_valid_count / event_count) if event_count else 0.0
    final_empty_rate = float(final_empty_count / event_count) if event_count else 0.0
    return {
        "artifact": "qwen37plus_structured_anchor_retry_outcome_audit_20260602_v63",
        "current_stage": "v63_qwen37plus_structured_anchor_prompt_retry_shadow_rollout",
        "scope": "structured_anchor_retry_outcome_only",
        "model_name": MODEL_NAME,
        "cache_path": _rel(cache_path),
        "planning_event_count": int(event_count),
        "structured_anchor_attempt_count": int(sum(len(items) for items in grouped.values())),
        "final_valid_structured_anchor_count": int(final_valid_count),
        "valid_structured_anchor_rate": float(final_valid_rate),
        "empty_structured_anchor_rate": float(final_empty_rate),
        "events_with_retry_count": int(events_with_retry_count),
        "retry_success_count": int(retry_success_count),
        "final_failure_type_counts": dict(sorted(final_failure_counts.items())),
        "attempt_failure_type_counts": dict(sorted(attempt_failure_counts.items())),
        "invalid_json_truncated_count": int(final_failure_counts.get("invalid_json_truncated", 0)),
        "final_control_field_leak_count": int(final_control_leaks),
        "provider_error_steps": int(provider_error_steps),
        "invalid_or_salvaged_clean_evidence_count": 0,
        "event_reports": event_reports,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def _final_cache_entries(cache_path: str | Path = V63_CACHE_PATH) -> list[tuple[str, dict[str, Any]]]:
    return [_final_event_entry("", items) for items in _group_events(cache_path).values()]


def build_bridge_audit(
    *,
    cache_path: str | Path = V63_CACHE_PATH,
    trace_dir: str | Path = V63_TRACE_DIR,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    trace_rows = v60._trace_rows_by_scenario(trace_dir, scenarios)
    row_results = []
    for cache_key, entry in _final_cache_entries(cache_path):
        scenario = v60._env_id_to_scenario(str(entry.get("env_id") or ""))
        if scenario not in trace_rows:
            continue
        timestep = v60._as_int(entry.get("timestep"), 0)
        row = trace_rows.get(scenario, {}).get(timestep)
        result, _ = v60._bridge_one_entry(cache_key=cache_key, entry=entry, row=row)
        row_results.append(result)

    source_attempt_count = len(row_results)
    bridgeable = [row for row in row_results if bool(row.get("bridgeable", False))]
    bridgeable_count = len(bridgeable)
    bridge_input_coverage_rate = float(bridgeable_count / source_attempt_count) if source_attempt_count else 0.0
    candidate_counts = [int(row.get("profile_candidate_count", 0) or 0) for row in bridgeable]
    hard_safety_violations = sum(bool(row.get("hard_safety_profile_violation", False)) for row in bridgeable)
    failure_reasons = Counter(str(row.get("failure_reason") or "") for row in row_results if not bool(row.get("bridgeable", False)))
    acceptance = {
        "bridge_input_coverage_ge_0_90": bool(bridge_input_coverage_rate >= 0.90),
        "profile_candidate_generation_error_count_zero": bool((source_attempt_count - bridgeable_count) == 0),
        "profile_candidate_count_positive": bool(candidate_counts and min(candidate_counts) > 0),
        "hard_safety_profile_violation_count_zero": bool(hard_safety_violations == 0),
        "final_control_change_count_zero": True,
    }
    acceptance["v63_bridge_pass"] = bool(all(acceptance.values()))
    return {
        "artifact": "qwen37plus_structured_anchor_profile_bridge_audit_20260602_v63",
        "current_stage": "v63_qwen37plus_structured_anchor_prompt_retry_shadow_rollout",
        "scope": "final_event_structured_anchor_to_profile_generator_bridge_only",
        "model_name_for_next_online_stage": MODEL_NAME,
        "source_cache_path": _rel(cache_path),
        "source_trace_dir": _rel(trace_dir),
        "scenarios": list(scenarios),
        "source_attempt_count": int(source_attempt_count),
        "bridgeable_count": int(bridgeable_count),
        "bridge_input_coverage_rate": float(bridge_input_coverage_rate),
        "valid_structured_anchor_missing_count": int(failure_reasons.get("valid_structured_anchor_missing", 0)),
        "profile_candidate_generation_error_count": int(source_attempt_count - bridgeable_count),
        "profile_candidate_count_min": int(min(candidate_counts) if candidate_counts else 0),
        "profile_candidate_count_max": int(max(candidate_counts) if candidate_counts else 0),
        "hard_safety_profile_violation_count": int(hard_safety_violations),
        "final_control_change_count": 0,
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "sample_rows": row_results[:8],
        "acceptance": acceptance,
        "bridge_pass": bool(acceptance["v63_bridge_pass"]),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def build_online_precheck(*, probe_online: bool = True) -> dict[str, Any]:
    result = v61.build_online_precheck(probe_online=probe_online, model_name=MODEL_NAME)
    result["artifact"] = "online_llm_accessibility_precheck_result_v63_qwen37plus_20260602"
    result["model_name"] = MODEL_NAME
    return result


def _v62_ready(path: str | Path = V62_READINESS_JSON) -> bool:
    readiness = _load_json(path)
    return bool(
        readiness.get("model_name") == MODEL_NAME
        and readiness.get("repair_design_ready", False)
        and readiness.get("next_action") == "qwen37plus_structured_anchor_prompt_retry_opt_in_shadow_rollout_plan"
    )


def _command_for_group(group: Mapping[str, str]) -> list[str]:
    output_json = BENCHMARK_DIR / f"qwen37plus_structured_anchor_prompt_retry_shadow_rollout_v63_{group['name']}.json"
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
        _rel(V63_CACHE_PATH),
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--agent-config-overrides",
        json.dumps(AGENT_OVERRIDES, sort_keys=True),
        "--output-json",
        _rel(output_json),
        "--output-trace-dir",
        _rel(V63_TRACE_DIR),
    ]


def build_execution_record(
    precheck: Mapping[str, Any] | None = None,
    *,
    v62_readiness_json: str | Path = V62_READINESS_JSON,
) -> dict[str, Any]:
    precheck = dict(precheck or build_online_precheck(probe_online=True))
    v62_ready = _v62_ready(v62_readiness_json)
    executable = bool(v62_ready and precheck.get("online_llm_accessible", False) and precheck.get("model_name") == MODEL_NAME)
    blocked_reason = ""
    if not v62_ready:
        blocked_reason = "v62_prompt_retry_repair_design_not_ready"
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
        "artifact": "qwen37plus_structured_anchor_prompt_retry_opt_in_shadow_rollout_v63_execution_record_20260602",
        "authorization_source": "user_delegated_qwen37plus_structured_anchor_prompt_retry_shadow_rollout_authorization_20260602",
        "scope": "minimal_qwen37plus_structured_anchor_prompt_retry_shadow_rollout_only",
        "model_name": MODEL_NAME,
        "scenarios": list(FAILURE_SCENARIOS),
        "controller": "llm_rspc_v2",
        "controlled_controller_included": False,
        "max_steps": 720,
        "llm_max_iterations": 2,
        "agent_config_overrides": dict(AGENT_OVERRIDES),
        "cache_path": _rel(V63_CACHE_PATH),
        "output_dir": _rel(BENCHMARK_DIR),
        "trace_dir": _rel(V63_TRACE_DIR),
        "online_llm_allowed": True,
        "online_llm_credentials_present": bool(precheck.get("online_llm_credentials_present", False)),
        "online_llm_accessible": bool(precheck.get("online_llm_accessible", False)),
        "v62_prompt_retry_repair_design_ready": bool(v62_ready),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "executable": bool(executable),
        "blocked_reason": blocked_reason,
        "commands": commands,
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
    trace_dir: str | Path = V63_TRACE_DIR,
    cache_path: str | Path = V63_CACHE_PATH,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
                "model_consistency": model,
            }
        )

    retry_audit = build_retry_outcome_audit(cache_path)
    bridge_audit = build_bridge_audit(cache_path=cache_path, trace_dir=trace_dir, scenarios=scenarios)
    bridge_acceptance = bridge_audit.get("acceptance", {}) if isinstance(bridge_audit.get("acceptance"), Mapping) else {}
    model_consistency_pass = bool(model_passes and all(model_passes))
    acceptance = {
        "model_consistency_pass": model_consistency_pass,
        "provider_error_steps_zero": bool(retry_audit["provider_error_steps"] == 0),
        "structured_anchor_attempt_count_positive": bool(retry_audit["structured_anchor_attempt_count"] > 0),
        "valid_structured_anchor_rate_ge_0_90": bool(retry_audit["valid_structured_anchor_rate"] >= 0.90),
        "empty_structured_anchor_rate_le_0_03": bool(retry_audit["empty_structured_anchor_rate"] <= 0.03),
        "invalid_json_truncated_count_le_2": bool(retry_audit["invalid_json_truncated_count"] <= 2),
        "final_control_field_leak_count_zero": bool(retry_audit["final_control_field_leak_count"] == 0),
        "invalid_or_salvaged_not_clean_evidence": bool(retry_audit["invalid_or_salvaged_clean_evidence_count"] == 0),
        "bridge_input_coverage_ge_0_90": bool(bridge_acceptance.get("bridge_input_coverage_ge_0_90", False)),
        "hard_safety_profile_violation_count_zero": bool(
            bridge_acceptance.get("hard_safety_profile_violation_count_zero", False)
        ),
    }
    acceptance["v63_acceptance_pass"] = bool(all(acceptance.values()))
    return (
        {
            "artifact": "qwen37plus_structured_anchor_prompt_retry_shadow_rollout_result_audit_20260602_v63",
            "current_stage": "v63_qwen37plus_structured_anchor_prompt_retry_shadow_rollout",
            "scope": "structured_anchor_prompt_retry_contract_and_profile_bridge_only",
            "model_name": MODEL_NAME,
            "trace_dir": _rel(trace_dir),
            "cache_path": _rel(cache_path),
            "scenarios": list(scenarios),
            "scenario_reports": scenario_reports,
            "retry_audit_artifact": retry_audit["artifact"],
            "bridge_audit_artifact": bridge_audit["artifact"],
            "model_consistency_pass": model_consistency_pass,
            "provider_error_steps": int(retry_audit["provider_error_steps"]),
            "structured_anchor_attempt_count": int(retry_audit["structured_anchor_attempt_count"]),
            "valid_structured_anchor_rate": float(retry_audit["valid_structured_anchor_rate"]),
            "empty_structured_anchor_rate": float(retry_audit["empty_structured_anchor_rate"]),
            "invalid_json_truncated_count": int(retry_audit["invalid_json_truncated_count"]),
            "final_control_field_leak_count": int(retry_audit["final_control_field_leak_count"]),
            "bridge_input_coverage_rate": float(bridge_audit.get("bridge_input_coverage_rate", 0.0) or 0.0),
            "hard_safety_profile_violation_count": int(bridge_audit.get("hard_safety_profile_violation_count", 0) or 0),
            "acceptance": acceptance,
            "structured_anchor_prompt_retry_shadow_rollout_pass": bool(acceptance["v63_acceptance_pass"]),
            "next_action": (
                "structured_anchor_to_profile_generator_runtime_shadow_bridge_plan"
                if acceptance["v63_acceptance_pass"]
                else "structured_anchor_provider_instrumentation_patch_plan"
            ),
            "stop_taxonomy": [] if acceptance["v63_acceptance_pass"] else _stop_taxonomy({"acceptance": acceptance}),
            "controlled_replay_allowed": False,
            "controlled_replay_execution_allowed": False,
            "metadata_replay_execution_allowed": False,
            "performance_claim_allowed": False,
            "promotion_evidence": False,
        },
        retry_audit,
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
    if not acceptance.get("valid_structured_anchor_rate_ge_0_90", False):
        taxonomy.append("structured_anchor_valid_rate_low")
    if not acceptance.get("empty_structured_anchor_rate_le_0_03", False):
        taxonomy.append("empty_structured_anchor_rate_high")
    if not acceptance.get("invalid_json_truncated_count_le_2", False):
        taxonomy.append("invalid_json_truncated_persists")
    if not acceptance.get("final_control_field_leak_count_zero", False):
        taxonomy.append("final_control_field_leak")
    if not acceptance.get("bridge_input_coverage_ge_0_90", False):
        taxonomy.append("bridge_input_coverage_low")
    if not acceptance.get("hard_safety_profile_violation_count_zero", False):
        taxonomy.append("hard_safety_profile_violation")
    return sorted(set(taxonomy or ["gate_insufficient"]))


def build_readiness(result_audit: Mapping[str, Any], execution_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    passed = bool((result_audit.get("acceptance", {}) or {}).get("v63_acceptance_pass", False))
    executable = bool((execution_record or {}).get("executable", False))
    return {
        "artifact": "metadata_replay_readiness_checklist_20260602_v63",
        "current_stage": "v63_qwen37plus_structured_anchor_prompt_retry_shadow_rollout",
        "model_name": MODEL_NAME,
        "execution_record_executable": bool(executable),
        "structured_anchor_prompt_retry_shadow_rollout_done": bool(result_audit.get("structured_anchor_attempt_count", 0) > 0),
        "structured_anchor_prompt_retry_shadow_rollout_pass": bool(passed),
        "model_consistency_pass": bool(result_audit.get("model_consistency_pass", False)),
        "provider_error_steps": int(result_audit.get("provider_error_steps", 0) or 0),
        "structured_anchor_attempt_count": int(result_audit.get("structured_anchor_attempt_count", 0) or 0),
        "valid_structured_anchor_rate": float(result_audit.get("valid_structured_anchor_rate", 0.0) or 0.0),
        "empty_structured_anchor_rate": float(result_audit.get("empty_structured_anchor_rate", 0.0) or 0.0),
        "invalid_json_truncated_count": int(result_audit.get("invalid_json_truncated_count", 0) or 0),
        "bridge_input_coverage_rate": float(result_audit.get("bridge_input_coverage_rate", 0.0) or 0.0),
        "hard_safety_profile_violation_count": int(result_audit.get("hard_safety_profile_violation_count", 0) or 0),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": (
            "structured_anchor_to_profile_generator_runtime_shadow_bridge_plan"
            if passed
            else "structured_anchor_provider_instrumentation_patch_plan"
        ),
        "stop_taxonomy": [] if passed else _stop_taxonomy(result_audit, execution_record),
    }


def write_online_precheck(precheck: Mapping[str, Any]) -> None:
    _write_json_md(ONLINE_PRECHECK_JSON, ONLINE_PRECHECK_MD, precheck, "v63 qwen3.7-plus Online LLM Accessibility Precheck")


def write_execution_record(record: Mapping[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    V63_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    _write_json_md(EXECUTION_RECORD_JSON, EXECUTION_RECORD_MD, record, "v63 qwen3.7-plus Structured Anchor Prompt/Retry Shadow Rollout Execution Record")


def write_result_artifacts(
    result_audit: Mapping[str, Any],
    retry_audit: Mapping[str, Any],
    bridge_audit: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> None:
    _write_json_md(RESULT_AUDIT_JSON, RESULT_AUDIT_MD, result_audit, "v63 qwen3.7-plus Structured Anchor Prompt/Retry Shadow Rollout Result Audit")
    _write_json_md(RETRY_AUDIT_JSON, RETRY_AUDIT_MD, retry_audit, "v63 qwen3.7-plus Structured Anchor Retry Outcome Audit")
    _write_json_md(BRIDGE_AUDIT_JSON, BRIDGE_AUDIT_MD, bridge_audit, "v63 qwen3.7-plus Structured Anchor Profile Bridge Audit")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v63 Metadata Replay Readiness")


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
    result, retry_audit, bridge_audit = build_result_audit()
    if run_results:
        result["execution_run_results"] = run_results
    readiness = build_readiness(result, execution)
    write_result_artifacts(result, retry_audit, bridge_audit, readiness)
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
                    "v63_acceptance_pass": (result.get("acceptance", {}) or {}).get("v63_acceptance_pass"),
                    "valid_structured_anchor_rate": result.get("valid_structured_anchor_rate"),
                    "bridge_input_coverage_rate": result.get("bridge_input_coverage_rate"),
                    "next_action": readiness.get("next_action"),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
