"""v57 qwen3.7 tool-layer usage audit and structured-anchor design.

This is an offline attribution/design step. It reads the v56 traces, isolated
plan cache, result audit, and current tool registration. It does not run a
rollout, call an online LLM, change the default llm_rspc_v2 path, or authorize
controlled replay/promotion/performance claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


AUDIT_DIR = PROJECT_ROOT / "gl_gym" / "result" / "audits"
TOOLS_PY = PROJECT_ROOT / "gl_gym" / "agent" / "tools.py"
LLM_AGENT_PY = PROJECT_ROOT / "gl_gym" / "agent" / "llm_agent.py"
V56_TRACE_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "qwen37_recovery_anchor_opt_in_shadow_rollout_v56_20260601"
    / "traces"
)
V56_CACHE_PATH = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "plan_cache"
    / "qwen37_recovery_anchor_opt_in_shadow_v56_20260601.json"
)
V56_RESULT_AUDIT_JSON = (
    AUDIT_DIR / "qwen37_recovery_anchor_opt_in_shadow_rollout_result_audit_20260601_v56.json"
)
V56_READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v56.json"

TOOL_AUDIT_JSON = AUDIT_DIR / "qwen37_tool_layer_usage_audit_20260601_v57.json"
TOOL_AUDIT_MD = AUDIT_DIR / "qwen37_tool_layer_usage_audit_20260601_v57.md"
DEPRECATION_MATRIX_JSON = AUDIT_DIR / "llm_tool_layer_deprecation_matrix_20260601_v57.json"
DEPRECATION_MATRIX_MD = AUDIT_DIR / "llm_tool_layer_deprecation_matrix_20260601_v57.md"
STRUCTURED_ANCHOR_JSON = AUDIT_DIR / "qwen37_structured_anchor_contract_design_20260601_v57.json"
STRUCTURED_ANCHOR_MD = AUDIT_DIR / "qwen37_structured_anchor_contract_design_20260601_v57.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v57.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v57.md"

MODEL_NAME = "qwen3.7-max"
FAILURE_SCENARIOS = (
    "y2010_d180_s43_n720",
    "y2018_d181_s42_n720",
    "y2018_d181_s43_n720",
)
EXPECTED_TOOLS = (
    "get_status",
    "set_all_controls",
    "set_heating",
    "set_co2",
    "set_screen",
    "set_ventilation",
    "set_lamps",
    "set_blindscreen",
)
LOW_LEVEL_ACTION_TOOLS = (
    "set_heating",
    "set_co2",
    "set_screen",
    "set_ventilation",
    "set_lamps",
    "set_blindscreen",
)
STRUCTURED_ANCHOR_FIELDS = (
    "profile_intent",
    "target_temp",
    "target_co2",
    "target_rh",
    "risk_flags",
    "forbidden_intents",
    "planning_horizon_steps",
    "confidence",
)
FINAL_CONTROL_FIELDS = (
    "u_boil",
    "u_co2",
    "u_th_scr",
    "u_vent",
    "u_lamp",
    "u_bl_scr",
    "heating",
    "co2",
    "screen",
    "ventilation",
    "lighting",
    "shading",
    "blindscreen",
    "final_control",
)


def _rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


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
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    lines = [f"# {title}", ""]
    for key in (
        "artifact",
        "current_stage",
        "model_name",
        "dominant_tool_layer_failure_mode",
        "next_action",
        "controlled_replay_allowed",
        "performance_claim_allowed",
        "promotion_evidence",
    ):
        if key in data:
            lines.append(f"- `{key}`: `{data[key]}`")
    lines.extend(["", "```json", json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), "```", ""])
    path_md.write_text("\n".join(lines), encoding="utf-8")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _trace_path(trace_dir: str | Path, scenario: str) -> Path:
    p = Path(trace_dir)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p / f"{scenario}_llm_rspc_v2.csv"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def inventory_registered_tools(tools_py: str | Path = TOOLS_PY) -> dict[str, Any]:
    path = Path(tools_py)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    registered = sorted(set(re.findall(r'name\s*=\s*"([^"]+)"', text)))
    method_defs = sorted(set(re.findall(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", text)))
    missing_expected = [name for name in EXPECTED_TOOLS if name not in registered]
    return {
        "tools_py": _rel(path),
        "registered_tools": registered,
        "method_defs": method_defs,
        "expected_tools": list(EXPECTED_TOOLS),
        "missing_expected_tools": missing_expected,
        "get_status_registered": "get_status" in registered,
        "set_all_controls_registered": "set_all_controls" in registered,
        "low_level_action_tools_registered": [name for name in LOW_LEVEL_ACTION_TOOLS if name in registered],
        "tool_inventory_complete": not missing_expected,
    }


def _plan_cache_entries(cache_path: str | Path) -> list[tuple[str, dict[str, Any]]]:
    root = _load_json(cache_path)
    entries = root.get("entries", {})
    if isinstance(entries, dict):
        return [(str(key), value) for key, value in entries.items() if isinstance(value, dict)]
    if isinstance(entries, list):
        return [(str(index), value) for index, value in enumerate(entries) if isinstance(value, dict)]
    return []


def _setpoints_present(setpoints: Mapping[str, Any]) -> bool:
    return all(not _is_missing(setpoints.get(name)) for name in ("target_temp", "target_co2", "target_rh"))


def _setpoints_any_present(setpoints: Mapping[str, Any]) -> bool:
    return any(not _is_missing(setpoints.get(name)) for name in ("target_temp", "target_co2", "target_rh"))


def classify_plan_cache_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    buffered_action = entry.get("buffered_action", {})
    if not isinstance(buffered_action, Mapping):
        buffered_action = {}
    setpoints = entry.get("buffered_setpoints", {})
    if not isinstance(setpoints, Mapping):
        setpoints = {}
    raw_response = str(entry.get("raw_response", "") or "")
    parsed_plan = entry.get("parsed_plan", {})
    if not isinstance(parsed_plan, Mapping):
        parsed_plan = {}
    action_set = bool(buffered_action.get("action_set", False))
    all_setpoints = _setpoints_present(setpoints)
    any_setpoints = _setpoints_any_present(setpoints)
    if action_set and all_setpoints:
        classification = "set_all_controls_anchor_success"
    elif action_set and not all_setpoints:
        classification = "action_anchor_missing_required_setpoints"
    elif not action_set and any_setpoints:
        classification = "setpoint_only_anchor"
    else:
        classification = "empty_planning_anchor"

    fallback_info = entry.get("fallback_info", {})
    if not isinstance(fallback_info, Mapping):
        fallback_info = {}
    fallback_selected = str(fallback_info.get("selected_fallback_candidate", "") or "")
    parsed_anchor_source = str(parsed_plan.get("anchor_source", "") or "")
    anchor_source = str(entry.get("anchor_source", "") or parsed_anchor_source)
    setpoint_contract = entry.get("setpoint_contract") or parsed_plan.get("setpoint_contract") or {}
    if not isinstance(setpoint_contract, Mapping):
        setpoint_contract = {}
    filled = setpoint_contract.get("filled", [])
    corrected = setpoint_contract.get("corrected", [])

    return {
        "classification": classification,
        "clean_planning_evidence": bool(classification == "set_all_controls_anchor_success"),
        "action_set": action_set,
        "required_setpoints_present": all_setpoints,
        "any_setpoints_present": any_setpoints,
        "llm_action_found": bool(entry.get("llm_action_found", False)),
        "raw_response_present": bool(raw_response.strip()),
        "raw_response_mentions_get_status": "get_status" in raw_response,
        "raw_response_mentions_set_all_controls": "set_all_controls" in raw_response,
        "raw_response_mentions_low_level_tool": any(name in raw_response for name in LOW_LEVEL_ACTION_TOOLS),
        "anchor_source": anchor_source,
        "fallback_selected_candidate": fallback_selected,
        "setpoint_contract_filled_count": len(filled) if isinstance(filled, list) else 0,
        "setpoint_contract_corrected_count": len(corrected) if isinstance(corrected, list) else 0,
        "model_name": str(entry.get("model_name", "") or ""),
        "plan_cache_status": str((entry.get("plan_cache_event", {}) or {}).get("status", "") or ""),
    }


def audit_plan_cache(cache_path: str | Path = V56_CACHE_PATH) -> dict[str, Any]:
    entries = _plan_cache_entries(cache_path)
    classifications = Counter()
    anchor_sources = Counter()
    fallback_sources = Counter()
    status_counts = Counter()
    tool_mentions = Counter()
    model_names = Counter()
    setpoint_contract_filled_entries = 0
    setpoint_contract_corrected_entries = 0
    sample_empty: list[dict[str, Any]] = []
    for key, entry in entries:
        report = classify_plan_cache_entry(entry)
        classifications[report["classification"]] += 1
        if report["anchor_source"]:
            anchor_sources[report["anchor_source"]] += 1
        if report["fallback_selected_candidate"]:
            fallback_sources[report["fallback_selected_candidate"]] += 1
        if report["plan_cache_status"]:
            status_counts[report["plan_cache_status"]] += 1
        if report["model_name"]:
            model_names[report["model_name"]] += 1
        for name in (
            "raw_response_mentions_get_status",
            "raw_response_mentions_set_all_controls",
            "raw_response_mentions_low_level_tool",
        ):
            if report[name]:
                tool_mentions[name] += 1
        if report["setpoint_contract_filled_count"]:
            setpoint_contract_filled_entries += 1
        if report["setpoint_contract_corrected_count"]:
            setpoint_contract_corrected_entries += 1
        if report["classification"] == "empty_planning_anchor" and len(sample_empty) < 5:
            sample_empty.append(
                {
                    "key": key,
                    "anchor_source": report["anchor_source"],
                    "fallback_selected_candidate": report["fallback_selected_candidate"],
                    "plan_cache_status": report["plan_cache_status"],
                }
            )

    total = len(entries)
    successful = classifications.get("set_all_controls_anchor_success", 0)
    empty = classifications.get("empty_planning_anchor", 0)
    low_success_rate = (successful / total) if total else 0.0
    return {
        "cache_path": _rel(cache_path),
        "entry_count": total,
        "classification_counts": dict(sorted(classifications.items())),
        "set_all_controls_anchor_success_count": successful,
        "empty_planning_anchor_count": empty,
        "clean_planning_evidence_count": successful,
        "clean_planning_evidence_rate": low_success_rate,
        "empty_planning_anchor_is_not_clean_evidence": True,
        "anchor_source_counts": dict(anchor_sources.most_common()),
        "fallback_selected_candidate_counts": dict(fallback_sources.most_common()),
        "plan_cache_status_counts": dict(status_counts.most_common()),
        "model_name_counts": dict(model_names.most_common()),
        "raw_tool_mention_counts": dict(tool_mentions.most_common()),
        "setpoint_contract_filled_entry_count": int(setpoint_contract_filled_entries),
        "setpoint_contract_corrected_entry_count": int(setpoint_contract_corrected_entries),
        "sample_empty_planning_anchor_entries": sample_empty,
        "legacy_action_anchor_success_rate_low": bool(total > 0 and low_success_rate < 0.25),
    }


def audit_traces(
    trace_dir: str | Path = V56_TRACE_DIR,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    scenario_reports: list[dict[str, Any]] = []
    totals = Counter()
    model_names = Counter()
    for scenario in scenarios:
        rows = _read_csv_rows(_trace_path(trace_dir, scenario))
        source_counts = Counter(str(row.get("source", "") or "") for row in rows)
        anchor_counts = Counter(str(row.get("anchor_source", "") or "") for row in rows)
        plan_cache_status = Counter(str(row.get("plan_cache_status", "") or "") for row in rows)
        selected_fallback = Counter(str(row.get("selected_fallback_candidate", "") or "") for row in rows)
        for row in rows:
            if row.get("model_name"):
                model_names[str(row.get("model_name", ""))] += 1
        recovery_selected_steps = int(
            sum(
                str(row.get("anchor_source", "") or "") == "tomato_safety_projected_anchor"
                or str(row.get("selected_fallback_candidate", "") or "") == "tomato_safety_projected_anchor"
                for row in rows
            )
        )
        report = {
            "scenario_id": scenario,
            "trace_path": _rel(_trace_path(trace_dir, scenario)),
            "trace_exists": bool(rows),
            "row_count": len(rows),
            "runtime_error_steps": int(
                sum(
                    _truthy(row.get("runtime_error"))
                    or str(row.get("runtime_error_type", "") or "").strip() != ""
                    or str(row.get("source", "") or "") == "runtime_error"
                    for row in rows
                )
            ),
            "plan_cache_hit_steps": int(sum(_truthy(row.get("plan_cache_hit")) for row in rows)),
            "plan_cache_enabled_steps": int(sum(_truthy(row.get("plan_cache_enabled")) for row in rows)),
            "recovery_anchor_applied_steps": int(
                sum(_truthy(row.get("recovery_anchor_applied")) for row in rows) or recovery_selected_steps
            ),
            "recovery_anchor_selected_steps": recovery_selected_steps,
            "fallback_veto_applied_steps": int(
                sum(_truthy(row.get("profile_template_patch_fallback_veto_applied")) for row in rows)
            ),
            "profile_template_patch_applied_steps": int(
                sum(_truthy(row.get("profile_template_patch_applied")) for row in rows)
            ),
            "hard_safety_rewrite_preferred_steps": int(
                sum(_truthy(row.get("hard_safety_rewrite_preferred")) for row in rows)
            ),
            "source_counts": dict(source_counts.most_common()),
            "anchor_source_counts": dict(anchor_counts.most_common()),
            "plan_cache_status_counts": dict(plan_cache_status.most_common()),
            "selected_fallback_candidate_counts": dict(selected_fallback.most_common()),
        }
        scenario_reports.append(report)
        for key in (
            "row_count",
            "runtime_error_steps",
            "plan_cache_hit_steps",
            "plan_cache_enabled_steps",
            "recovery_anchor_applied_steps",
            "fallback_veto_applied_steps",
            "profile_template_patch_applied_steps",
            "hard_safety_rewrite_preferred_steps",
            "recovery_anchor_selected_steps",
        ):
            totals[key] += int(report[key])
    return {
        "trace_dir": _rel(trace_dir),
        "scenarios": list(scenarios),
        "scenario_reports": scenario_reports,
        "totals": dict(totals),
        "model_name_counts": dict(model_names.most_common()),
        "qwen37_model_rows": int(model_names.get(MODEL_NAME, 0)),
    }


def _count_text_mentions(path: str | Path, needles: Iterable[str]) -> dict[str, int]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    return {needle: text.count(needle) for needle in needles}


def build_tool_layer_usage_audit(
    *,
    cache_path: str | Path = V56_CACHE_PATH,
    trace_dir: str | Path = V56_TRACE_DIR,
    result_audit_json: str | Path = V56_RESULT_AUDIT_JSON,
    readiness_json: str | Path = V56_READINESS_JSON,
    tools_py: str | Path = TOOLS_PY,
    llm_agent_py: str | Path = LLM_AGENT_PY,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    tool_inventory = inventory_registered_tools(tools_py)
    plan_cache = audit_plan_cache(cache_path)
    traces = audit_traces(trace_dir, scenarios)
    result_audit = _load_json(result_audit_json)
    previous_readiness = _load_json(readiness_json)
    agent_mentions = _count_text_mentions(
        llm_agent_py,
        ("create_langchain_tools", "get_status", "set_all_controls", "buffered_action", "planning anchor empty"),
    )

    empty_count = int(plan_cache.get("empty_planning_anchor_count", 0))
    success_count = int(plan_cache.get("set_all_controls_anchor_success_count", 0))
    candidate_guardrail_pressure = "candidate_guardrail_pressure_persists" in set(
        previous_readiness.get("stop_taxonomy", []) or []
    )
    legacy_contract_unstable = bool(
        empty_count > 0
        and (
            success_count == 0
            or bool(plan_cache.get("legacy_action_anchor_success_rate_low", False))
        )
    )
    if legacy_contract_unstable:
        dominant = "legacy_tool_action_contract_unstable"
    elif candidate_guardrail_pressure:
        dominant = "profile_candidate_guardrail_conflict"
    else:
        dominant = "tool_layer_usage_attribution_completed_no_dominant_failure"

    return {
        "artifact": "qwen37_tool_layer_usage_audit_20260601_v57",
        "current_stage": "v57_llm_tool_layer_usage_attribution",
        "model_name": MODEL_NAME,
        "scope": "offline_tool_layer_usage_attribution_only",
        "inputs": {
            "cache_path": _rel(cache_path),
            "trace_dir": _rel(trace_dir),
            "result_audit_json": _rel(result_audit_json),
            "readiness_json": _rel(readiness_json),
            "tools_py": _rel(tools_py),
            "llm_agent_py": _rel(llm_agent_py),
        },
        "tool_inventory": tool_inventory,
        "llm_agent_tool_contract_mentions": agent_mentions,
        "plan_cache_audit": plan_cache,
        "trace_audit": traces,
        "v56_acceptance_pass": bool((result_audit.get("acceptance", {}) or {}).get("v56_acceptance_pass", False)),
        "v56_stop_taxonomy": list(previous_readiness.get("stop_taxonomy", []) or []),
        "legacy_tool_action_contract_unstable": legacy_contract_unstable,
        "profile_candidate_guardrail_conflict_persists": bool(candidate_guardrail_pressure),
        "dominant_tool_layer_failure_mode": dominant,
        "mainline_interpretation": {
            "get_status_still_used": bool(agent_mentions.get("get_status", 0) > 0),
            "set_all_controls_still_legacy_anchor_contract": bool(agent_mentions.get("set_all_controls", 0) > 0),
            "low_level_action_tools_are_registered": bool(tool_inventory["low_level_action_tools_registered"]),
            "empty_planning_anchor_is_not_qwen37_planning_evidence": True,
        },
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def build_tool_layer_deprecation_matrix(tool_inventory: Mapping[str, Any] | None = None) -> dict[str, Any]:
    inventory = dict(tool_inventory or inventory_registered_tools())
    registered = set(inventory.get("registered_tools", []))
    entries = [
        {
            "tool_name": "get_status",
            "registered": "get_status" in registered,
            "current_role": "read_only_state_and_diagnostic_context",
            "mainline_status": "retain_temporarily_read_only",
            "recommended_action": "keep_as_status_adapter_until_structured_anchor_parser_has_state_context",
            "deletion_authorized": False,
            "default_controller_change_authorized": False,
        },
        {
            "tool_name": "set_all_controls",
            "registered": "set_all_controls" in registered,
            "current_role": "legacy_action_and_setpoint_anchor_interface",
            "mainline_status": "replace_with_structured_planning_anchor_after_shadow_parser_pass",
            "recommended_action": "deprecate_as_action_generation_interface_only_after_opt_in_structured_anchor_validation",
            "deletion_authorized": False,
            "default_controller_change_authorized": False,
        },
    ]
    for name in LOW_LEVEL_ACTION_TOOLS:
        entries.append(
            {
                "tool_name": name,
                "registered": name in registered,
                "current_role": "legacy_low_level_actuator_tool",
                "mainline_status": "not_part_of_new_structured_anchor_mainline",
                "recommended_action": "leave_registered_for_legacy_compatibility_until_migration_gate_passes",
                "deletion_authorized": False,
                "default_controller_change_authorized": False,
            }
        )
    return {
        "artifact": "llm_tool_layer_deprecation_matrix_20260601_v57",
        "current_stage": "v57_tool_layer_deprecation_design",
        "scope": "classification_only_no_deletion",
        "tool_entries": entries,
        "deletion_authorized": False,
        "default_controller_change_authorized": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def build_structured_anchor_contract_design() -> dict[str, Any]:
    schema = {
        "profile_intent": {
            "type": "string",
            "required": True,
            "description": "High-level climate/profile intent, not a low-level actuator command.",
        },
        "target_temp": {"type": "number", "required": True, "unit": "degC"},
        "target_co2": {"type": "number", "required": True, "unit": "ppm"},
        "target_rh": {"type": "number", "required": True, "unit": "percent"},
        "risk_flags": {"type": "array[string]", "required": True},
        "forbidden_intents": {"type": "array[string]", "required": True},
        "planning_horizon_steps": {"type": "integer", "required": True, "minimum": 1},
        "confidence": {"type": "number", "required": True, "minimum": 0.0, "maximum": 1.0},
    }
    return {
        "artifact": "qwen37_structured_anchor_contract_design_20260601_v57",
        "current_stage": "v57_structured_anchor_contract_design",
        "model_name": MODEL_NAME,
        "contract_name": "structured_planning_anchor",
        "opt_in_only": True,
        "default_controller_change_authorized": False,
        "default_llm_rspc_v2_path_changed": False,
        "required_fields": list(STRUCTURED_ANCHOR_FIELDS),
        "schema": schema,
        "explicitly_excluded_fields": list(FINAL_CONTROL_FIELDS),
        "final_control_generation_allowed": False,
        "responsibility_split": {
            "llm": "emit profile intent, target setpoints, risk flags, forbidden intents, horizon, confidence",
            "profile_generator": "repair/shape targets into feasible trajectories",
            "candidate_scorer": "compose and score low-level actuator candidates",
            "tomato_safety": "enforce hard-safety projection and provenance",
            "fallback": "provide recovery anchors only when structured anchor is missing or unsafe",
        },
        "migration_gate": [
            "parse qwen3.7 structured anchor in shadow mode",
            "reject empty or incomplete anchors as non-clean planning evidence",
            "compare profile/candidate/guardrail pressure against v56",
            "only then consider replacing set_all_controls as the anchor interface",
        ],
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def build_readiness(
    tool_audit: Mapping[str, Any],
    deprecation_matrix: Mapping[str, Any],
    structured_design: Mapping[str, Any],
) -> dict[str, Any]:
    legacy_unstable = bool(tool_audit.get("legacy_tool_action_contract_unstable", False))
    conflict_persists = bool(tool_audit.get("profile_candidate_guardrail_conflict_persists", False))
    if legacy_unstable:
        next_action = "structured_anchor_opt_in_shadow_parser_plan"
    elif conflict_persists:
        next_action = "profile_candidate_guardrail_reconciliation_plan"
    else:
        next_action = "tool_layer_usage_monitoring_or_additional_instrumentation"
    return {
        "artifact": "metadata_replay_readiness_checklist_20260601_v57",
        "current_stage": "v57_llm_tool_layer_usage_attribution_and_structured_anchor_migration_design",
        "model_name": MODEL_NAME,
        "tool_layer_usage_audit_done": True,
        "tool_layer_deprecation_matrix_done": True,
        "structured_anchor_contract_design_done": True,
        "legacy_tool_action_contract_unstable": legacy_unstable,
        "profile_candidate_guardrail_conflict_persists": conflict_persists,
        "dominant_tool_layer_failure_mode": str(tool_audit.get("dominant_tool_layer_failure_mode", "")),
        "set_all_controls_deletion_authorized": bool(deprecation_matrix.get("deletion_authorized", True)),
        "structured_anchor_default_path_authorized": bool(
            structured_design.get("default_controller_change_authorized", True)
        ),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": next_action,
    }


def write_all() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    tool_audit = build_tool_layer_usage_audit()
    deprecation = build_tool_layer_deprecation_matrix(tool_audit["tool_inventory"])
    structured = build_structured_anchor_contract_design()
    readiness = build_readiness(tool_audit, deprecation, structured)
    _write_json_md(TOOL_AUDIT_JSON, TOOL_AUDIT_MD, tool_audit, "v57 Qwen3.7 Tool-Layer Usage Audit")
    _write_json_md(DEPRECATION_MATRIX_JSON, DEPRECATION_MATRIX_MD, deprecation, "v57 LLM Tool-Layer Deprecation Matrix")
    _write_json_md(
        STRUCTURED_ANCHOR_JSON,
        STRUCTURED_ANCHOR_MD,
        structured,
        "v57 Qwen3.7 Structured Anchor Contract Design",
    )
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v57 Metadata Replay Readiness")
    return tool_audit, deprecation, structured, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-all", action="store_true", help="Write all v57 JSON/MD artifacts.")
    return parser.parse_args()


def main() -> None:
    _parse_args()
    tool_audit, _, _, readiness = write_all()
    print(
        json.dumps(
            {
                "artifact": tool_audit["artifact"],
                "dominant_tool_layer_failure_mode": tool_audit["dominant_tool_layer_failure_mode"],
                "empty_planning_anchor_count": tool_audit["plan_cache_audit"]["empty_planning_anchor_count"],
                "set_all_controls_anchor_success_count": tool_audit["plan_cache_audit"][
                    "set_all_controls_anchor_success_count"
                ],
                "next_action": readiness["next_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
