"""Diagnose where PPO and LLM-RSPC differ in greenhouse control.

The goal is not to produce a final benchmark table. It records matched
per-step trajectories, groups them by meaningful greenhouse regimes, and
highlights action/cost/safety gaps that should drive the next controller change.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import yaml
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gl_gym.agent.expert_distillation import ACTION_NAMES
from gl_gym.agent.interface import GreenhouseAgentInterface
from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector, create_langchain_tools
from gl_gym.agent.profile_generator import score_profile_candidate_payloads
from gl_gym.agent.ppo_strategy_labeler import label_strategy
from gl_gym.common.utils import calculate_vpd_kpa
from gl_gym.cstcc.audit_writer import parse_selected_sequence_id
from gl_gym.environments.baseline import RuleBasedController as GreenhouseRuleController
from gl_gym.environments.tomato_env import TomatoEnv

load_dotenv()


METRIC_MAP = {
    "profit": "EPI",
    "revenue": "revenue",
    "heat_cost": "heat_cost",
    "co2_cost": "co2_cost",
    "elec_cost": "elec_cost",
    "fixed_cost": "fixed_costs",
    "temp_violation": "temp_violation",
    "co2_violation": "co2_violation",
    "rh_violation": "rh_violation",
    "lamp_violation": "lamp_violation",
}


RH_LOW_LIMIT = 50.0
RH_DRY_RISK = 55.0
RH_HIGH_LIMIT = 90.0
RH_DEW_RISK = 94.0
VPD_LOW_LIMIT = 0.25
VPD_HIGH_LIMIT = 1.20
DEW_MARGIN_RISK = 0.60
COLD_TEMP_LIMIT = 12.0


DEFAULT_RULE_PARAMS = {
    "lamps_on": 6.0,
    "lamps_off": 20.0,
    "lamps_day_start": 1,
    "lamps_day_stop": 365,
    "lamps_off_sun": 400.0,
    "lamp_rad_sum_limit": 1000.0,
    "temp_setpoint_day": 20.0,
    "temp_setpoint_night": 16.0,
    "heat_correction": 1.0,
    "heat_deadzone": 0.5,
    "co2_day": 850.0,
    "vent_heat_Pband": 4.0,
    "rh_max": 83.0,
    "mech_dehumid_Pband": 7.0,
    "vent_rh_Pband": 7.0,
    "t_vent_off": 10.0,
    "vent_cold_Pband": 5.0,
    "thScrSpDay": 25.0,
    "thScrSpNight": 10.0,
    "thScrPband": 5.0,
    "thScrDeadZone": 2.0,
    "thScrRh": 90.0,
    "thScrRhPband": 5.0,
    "lampExtraHeat": 2.0,
    "blScrExtraRh": 5.0,
    "rhMax": 90.0,
    "tHeatBand": 2.0,
    "co2Band": 180.0,
    "useBlScr": False,
}


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_env(config: Dict[str, Any], year: int, day: int, seed: int, uncertainty_scale: float) -> TomatoEnv:
    base = dict(config["GreenLightEnv"])
    base["training"] = False
    tomato = dict(config["TomatoEnv"])
    eval_options = dict(tomato["eval_options"])
    eval_options["eval_years"] = [int(year)]
    eval_options["eval_days"] = [int(day)]
    tomato["eval_options"] = eval_options
    env = TomatoEnv(
        reward_function=tomato["reward_function"],
        observation_modules=tomato["observation_modules"],
        constraints=tomato["constraints"],
        eval_options=tomato["eval_options"],
        reward_params=tomato["reward_params"],
        base_env_params=base,
        uncertainty_scale=float(uncertainty_scale),
    )
    env.reset(seed=seed)
    return env


def state_to_record(state: Any) -> Dict[str, float]:
    names = (
        "timestep",
        "day_of_year",
        "hour_of_day",
        "temp_air",
        "rh_air",
        "co2_air",
        "pipe_temp",
        "fruit_weight",
        "canopy_temp_24h",
        "temperature_sum",
        "glob_rad",
        "temp_out",
        "rh_out",
        "wind_speed",
        "dli",
        "dew_margin_air",
        "canopy_dew_margin",
        "forecast_rad_mean_1h",
        "forecast_rad_peak_2h",
        "forecast_temp_out_delta_1h",
        "forecast_rh_out_mean_1h",
        "forecast_wind_peak_1h",
        "forecast_humidity_risk",
    )
    record: Dict[str, float] = {}
    for name in names:
        try:
            record[name] = float(getattr(state, name))
        except Exception:
            record[name] = 0.0
    return record


def info_to_metrics(info: Dict[str, Any]) -> Dict[str, float]:
    return {name: float(info.get(source, 0.0)) for name, source in METRIC_MAP.items()}


def control_to_record(control: Iterable[float], prefix: str = "u") -> Dict[str, float]:
    values = np.asarray(control, dtype=np.float32).reshape(-1)
    return {f"{prefix}_{name}": float(values[i]) if i < len(values) else 0.0 for i, name in enumerate(ACTION_NAMES)}


def transition_gate_to_record(transition_gate: Any) -> Dict[str, Any]:
    if not isinstance(transition_gate, dict):
        transition_gate = {}
    before = transition_gate.get("before", {}) if isinstance(transition_gate.get("before", {}), dict) else {}
    after = transition_gate.get("after", {}) if isinstance(transition_gate.get("after", {}), dict) else {}
    record: Dict[str, Any] = {
        "transition_gate_enabled": bool(transition_gate.get("enabled", False)),
        "transition_gate_applied": bool(transition_gate.get("applied", False)),
        "transition_gate_bypassed": bool(transition_gate.get("bypassed", False)),
        "transition_gate_regime": str(transition_gate.get("regime", "") or ""),
        "transition_gate_reason": str(transition_gate.get("reason", "") or ""),
        "transition_gate_adjusted_fields": ",".join(str(x) for x in transition_gate.get("adjusted_fields", []) or []),
        "transition_gate_reversal_fields": ",".join(str(x) for x in transition_gate.get("reversal_fields", []) or []),
        "transition_gate_bypass_reasons": ",".join(str(x) for x in transition_gate.get("bypass_reasons", []) or []),
        "transition_gate_delta_limited_count": int(transition_gate.get("delta_limited_count", 0) or 0),
        "transition_gate_reversal_projected_count": int(transition_gate.get("reversal_projected_count", 0) or 0),
        "transition_gate_reversal_window_steps": int(transition_gate.get("reversal_window_steps", 0) or 0),
        "transition_gate_soft_limit_source": str(transition_gate.get("soft_limit_source", "") or ""),
        "transition_gate_soft_limit_error": str(transition_gate.get("soft_limit_error", "") or ""),
        "transition_gate_before": _compact_json(before),
        "transition_gate_after": _compact_json(after),
    }
    for name in ACTION_NAMES:
        record[f"transition_gate_before_{name}"] = float(before.get(name, 0.0) or 0.0)
        record[f"transition_gate_after_{name}"] = float(after.get(name, 0.0) or 0.0)
    return record


def profile_feasibility_gate_to_record(profile_gate: Any) -> Dict[str, Any]:
    if not isinstance(profile_gate, dict):
        profile_gate = {}
    original = profile_gate.get("original_targets", {}) if isinstance(profile_gate.get("original_targets", {}), dict) else {}
    repaired = profile_gate.get("repaired_targets", {}) if isinstance(profile_gate.get("repaired_targets", {}), dict) else {}
    mode = profile_gate.get("mode", {}) if isinstance(profile_gate.get("mode", {}), dict) else {}
    return {
        "profile_feasibility_gate_enabled": bool(profile_gate.get("enabled", False)),
        "profile_feasibility_gate_applied": bool(profile_gate.get("applied", False)),
        "profile_feasibility_gate_corrections": ",".join(str(x) for x in profile_gate.get("corrections", []) or []),
        "profile_feasibility_gate_source": str(profile_gate.get("source", "") or ""),
        "profile_feasibility_gate_mode": _compact_json(mode),
        "profile_feasibility_gate_vent_required_by_safety": bool(mode.get("vent_required_by_safety", False)),
        "profile_feasibility_gate_humidity_retention_allowed": bool(mode.get("humidity_retention_allowed", False)),
        "profile_feasibility_gate_co2_enrichment_allowed": bool(mode.get("co2_enrichment_allowed", False)),
        "profile_feasibility_gate_dry_side": bool(mode.get("dry_side", False)),
        "profile_feasibility_gate_original_target_temp": _diagnostic_float(original.get("target_temp")),
        "profile_feasibility_gate_original_target_co2": _diagnostic_float(original.get("target_co2")),
        "profile_feasibility_gate_original_target_rh": _diagnostic_float(original.get("target_rh")),
        "profile_feasibility_gate_repaired_target_temp": _diagnostic_float(repaired.get("target_temp")),
        "profile_feasibility_gate_repaired_target_co2": _diagnostic_float(repaired.get("target_co2")),
        "profile_feasibility_gate_repaired_target_rh": _diagnostic_float(repaired.get("target_rh")),
        "profile_feasibility_gate_hard_safety_veto_enabled": bool(profile_gate.get("hard_safety_veto_enabled", False)),
        "profile_feasibility_gate_hard_safety_veto_count": int(profile_gate.get("hard_safety_veto_count", 0) or 0),
    }


def profile_template_patch_to_record(profile_patch: Any) -> Dict[str, Any]:
    if not isinstance(profile_patch, dict):
        profile_patch = {}
    original = profile_patch.get("original_targets", {}) if isinstance(profile_patch.get("original_targets", {}), dict) else {}
    repaired = profile_patch.get("repaired_targets", {}) if isinstance(profile_patch.get("repaired_targets", {}), dict) else {}
    mode = profile_patch.get("mode", {}) if isinstance(profile_patch.get("mode", {}), dict) else {}
    recovery_action = (
        profile_patch.get("recovery_anchor_action", {})
        if isinstance(profile_patch.get("recovery_anchor_action", {}), dict)
        else {}
    )
    recovery_prediction = (
        profile_patch.get("recovery_anchor_prediction", {})
        if isinstance(profile_patch.get("recovery_anchor_prediction", {}), dict)
        else {}
    )
    return {
        "profile_template_patch_enabled": bool(profile_patch.get("enabled", False)),
        "profile_template_patch_applied": bool(profile_patch.get("applied", False)),
        "profile_template_patch_corrections": ",".join(str(x) for x in profile_patch.get("corrections", []) or []),
        "profile_template_patch_forbidden_combinations": ",".join(
            str(x) for x in profile_patch.get("forbidden_combinations", []) or []
        ),
        "profile_template_patch_source": str(profile_patch.get("source", "") or ""),
        "profile_template_patch_mode": _compact_json(mode),
        "profile_template_patch_vent_required_by_safety": bool(mode.get("vent_required_by_safety", False)),
        "profile_template_patch_humidity_retention_allowed": bool(mode.get("humidity_retention_allowed", False)),
        "profile_template_patch_co2_enrichment_allowed": bool(mode.get("co2_enrichment_allowed", False)),
        "profile_template_patch_dry_side": bool(mode.get("dry_side", False)),
        "profile_template_patch_original_target_temp": _diagnostic_float(original.get("target_temp")),
        "profile_template_patch_original_target_co2": _diagnostic_float(original.get("target_co2")),
        "profile_template_patch_original_target_rh": _diagnostic_float(original.get("target_rh")),
        "profile_template_patch_repaired_target_temp": _diagnostic_float(repaired.get("target_temp")),
        "profile_template_patch_repaired_target_co2": _diagnostic_float(repaired.get("target_co2")),
        "profile_template_patch_repaired_target_rh": _diagnostic_float(repaired.get("target_rh")),
        "profile_template_patch_fallback_veto_enabled": bool(profile_patch.get("fallback_veto_enabled", False)),
        "profile_template_patch_fallback_veto_applied": bool(profile_patch.get("fallback_veto_applied", False)),
        "profile_template_patch_fallback_veto_no_alternative": bool(
            profile_patch.get("fallback_veto_no_alternative", False)
        ),
        "profile_template_patch_fallback_veto_reason": str(profile_patch.get("fallback_veto_reason", "") or ""),
        "profile_template_patch_fallback_veto_selected_before": str(
            profile_patch.get("fallback_veto_selected_before", "") or ""
        ),
        "profile_template_patch_fallback_veto_selected_after": str(
            profile_patch.get("fallback_veto_selected_after", "") or ""
        ),
        "profile_template_patch_fallback_veto_safety_reasons": ",".join(
            str(x) for x in profile_patch.get("fallback_veto_safety_reasons", []) or []
        ),
        "profile_template_patch_fallback_veto_rewrite_fields": ",".join(
            str(x) for x in profile_patch.get("fallback_veto_rewrite_fields", []) or []
        ),
        "recovery_anchor_enabled": bool(profile_patch.get("recovery_anchor_enabled", False)),
        "recovery_anchor_applied": bool(profile_patch.get("recovery_anchor_applied", False)),
        "recovery_anchor_source": str(profile_patch.get("recovery_anchor_source", "") or ""),
        "recovery_anchor_reason": str(profile_patch.get("recovery_anchor_reason", "") or ""),
        "recovery_anchor_no_compatible_existing_candidate": bool(
            profile_patch.get("recovery_anchor_no_compatible_existing_candidate", False)
        ),
        "recovery_anchor_selected_before": str(profile_patch.get("recovery_anchor_selected_before", "") or ""),
        "recovery_anchor_selected_after": str(profile_patch.get("recovery_anchor_selected_after", "") or ""),
        "recovery_anchor_action": _compact_json(recovery_action),
        "recovery_anchor_prediction": _compact_json(recovery_prediction),
        "recovery_anchor_hard_safety_rewrite": bool(recovery_prediction.get("hard_safety_rewrite", False)),
        "recovery_anchor_tomato_safety_applied": bool(recovery_prediction.get("tomato_safety_applied", False)),
        "recovery_anchor_safety_reasons": ",".join(
            str(x) for x in profile_patch.get("recovery_anchor_safety_reasons", []) or []
        ),
        "recovery_anchor_rewrite_fields": ",".join(
            str(x) for x in profile_patch.get("recovery_anchor_rewrite_fields", []) or []
        ),
    }


def structured_anchor_to_record(structured_anchor: Any) -> Dict[str, Any]:
    if not isinstance(structured_anchor, dict):
        structured_anchor = {}
    shadow = structured_anchor.get("shadow_plan", {}) if isinstance(structured_anchor.get("shadow_plan", {}), dict) else {}
    return {
        "structured_anchor_parser_enabled": bool(structured_anchor.get("enabled", False)),
        "structured_anchor_shadow_only": bool(structured_anchor.get("shadow_only", True)),
        "structured_anchor_attempted": bool(structured_anchor.get("attempted", False)),
        "structured_anchor_valid": bool(structured_anchor.get("valid", False)),
        "structured_anchor_clean_planning_evidence": bool(structured_anchor.get("clean_planning_evidence", False)),
        "structured_anchor_empty": bool(structured_anchor.get("empty", False)),
        "structured_anchor_errors": ",".join(str(x) for x in structured_anchor.get("errors", []) or []),
        "structured_anchor_missing_fields": ",".join(str(x) for x in structured_anchor.get("missing_fields", []) or []),
        "structured_anchor_unexpected_fields": ",".join(str(x) for x in structured_anchor.get("unexpected_fields", []) or []),
        "structured_anchor_final_control_field_hits": ",".join(
            str(x) for x in structured_anchor.get("final_control_field_hits", []) or []
        ),
        "structured_anchor_final_control_field_leak_count": int(
            structured_anchor.get("final_control_field_leak_count", 0) or 0
        ),
        "structured_anchor_legacy_tool_action_present": bool(structured_anchor.get("legacy_tool_action_present", False)),
        "structured_anchor_profile_intent": str(shadow.get("profile_intent", "") or ""),
        "structured_anchor_target_temp": _diagnostic_float(shadow.get("target_temp")),
        "structured_anchor_target_co2": _diagnostic_float(shadow.get("target_co2")),
        "structured_anchor_target_rh": _diagnostic_float(shadow.get("target_rh")),
        "structured_anchor_risk_flags": ",".join(str(x) for x in shadow.get("risk_flags", []) or []),
        "structured_anchor_forbidden_intents": ",".join(str(x) for x in shadow.get("forbidden_intents", []) or []),
        "structured_anchor_planning_horizon_steps": int(shadow.get("planning_horizon_steps", 0) or 0),
        "structured_anchor_confidence": _diagnostic_float(shadow.get("confidence")),
    }


def structured_anchor_profile_bridge_to_record(bridge: Any) -> Dict[str, Any]:
    if not isinstance(bridge, dict) or not bridge:
        return {}
    diagnostics = bridge.get("profile_generator_diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    bridge_diagnostics = bridge.get("bridge_diagnostics", {})
    if not isinstance(bridge_diagnostics, dict):
        bridge_diagnostics = {}
    contract = bridge.get("target_contract", {})
    if not isinstance(contract, dict):
        contract = {}
    return {
        "structured_anchor_profile_bridge_enabled": bool(bridge.get("enabled", False)),
        "structured_anchor_profile_bridge_shadow_only": bool(bridge.get("shadow_only", True)),
        "structured_anchor_profile_bridge_attempted": bool(bridge.get("attempted", False)),
        "structured_anchor_profile_bridge_applied": bool(bridge.get("applied", False)),
        "structured_anchor_profile_bridge_bridgeable": bool(bridge.get("bridgeable", False)),
        "structured_anchor_profile_bridge_valid_structured_anchor": bool(bridge.get("valid_structured_anchor", False)),
        "structured_anchor_profile_bridge_failure_reason": str(bridge.get("failure_reason", "") or ""),
        "structured_anchor_profile_bridge_profile_candidate_count": int(bridge.get("profile_candidate_count", 0) or 0),
        "structured_anchor_profile_bridge_selected_shadow_profile": str(bridge.get("selected_shadow_profile_name", "") or ""),
        "structured_anchor_profile_bridge_score_safety_gate_reason": str(bridge.get("score_safety_gate_reason", "") or ""),
        "structured_anchor_profile_bridge_hard_safety_profile_violation": bool(bridge.get("hard_safety_profile_violation", False)),
        "structured_anchor_profile_bridge_final_control_change": bool(bridge.get("final_control_change", False)),
        "structured_anchor_profile_bridge_current_plan_modified": bool(bridge.get("current_plan_modified", False)),
        "structured_anchor_profile_bridge_low_level_action_generated": bool(bridge.get("low_level_action_generated", False)),
        "structured_anchor_profile_bridge_intent_regime": str(contract.get("regime", "") or ""),
        "structured_anchor_profile_bridge_planning_horizon_steps": int(
            bridge_diagnostics.get("planning_horizon_steps", diagnostics.get("horizon_steps", 0)) or 0
        ),
        "structured_anchor_profile_bridge_unknown_profile_intent_mapped": bool(
            bridge_diagnostics.get("unknown_profile_intent_mapped", False)
        ),
        "structured_anchor_profile_bridge_requested_shapes": ",".join(
            str(x) for x in diagnostics.get("requested_shapes", []) if str(x)
        )
        if isinstance(diagnostics.get("requested_shapes", []), list)
        else "",
        "structured_anchor_profile_bridge_unsupported_shapes": ",".join(
            str(x) for x in diagnostics.get("unsupported_shapes", []) if str(x)
        )
        if isinstance(diagnostics.get("unsupported_shapes", []), list)
        else "",
    }


def regime_labels(row: Dict[str, Any]) -> Dict[str, str]:
    hour = float(row.get("hour_of_day", 0.0))
    rh = float(row.get("rh_air", 70.0))
    temp = float(row.get("temp_air", 20.0))
    rad = float(row.get("glob_rad", 0.0))
    if 4.0 <= hour < 8.0:
        phase = "dawn"
    elif 8.0 <= hour < 18.0:
        phase = "day"
    elif 18.0 <= hour < 21.0:
        phase = "dusk"
    else:
        phase = "night"
    if rh >= 90.0:
        rh_band = "rh_ge_90"
    elif rh >= 85.0:
        rh_band = "rh_85_90"
    elif rh >= 80.0:
        rh_band = "rh_80_85"
    else:
        rh_band = "rh_lt_80"
    if temp < 15.0:
        temp_band = "temp_lt_15"
    elif temp < 18.0:
        temp_band = "temp_15_18"
    elif temp < 22.0:
        temp_band = "temp_18_22"
    else:
        temp_band = "temp_ge_22"
    if rad < 20.0:
        rad_band = "dark"
    elif rad < 120.0:
        rad_band = "low_rad"
    else:
        rad_band = "useful_rad"
    return {"phase": phase, "rh_band": rh_band, "temp_band": temp_band, "rad_band": rad_band}


def append_regime_labels(row: Dict[str, Any]) -> Dict[str, Any]:
    row.update(regime_labels(row))
    return row


def append_strategy_label(row: Dict[str, Any]) -> Dict[str, Any]:
    strategy = label_strategy(row, action_prefix="u")
    row.update(strategy.to_record(prefix="strategy"))
    return row


def _dry_vent_cap(temp_air: float) -> float:
    if temp_air >= 28.0:
        return 0.35
    if temp_air >= 24.0:
        return 0.18
    return 0.12


def append_climate_risk_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    """Add mechanism-level risk metrics without changing controller behavior."""
    temp_air = float(row.get("temp_air", 20.0))
    rh_air = float(row.get("rh_air", 70.0))
    vent = float(row.get("u_ventilation", 0.0))
    dew_margin = min(
        float(row.get("dew_margin_air", 3.0)),
        float(row.get("canopy_dew_margin", 3.0)),
    )
    vpd = float(calculate_vpd_kpa(temp_air, rh_air))
    dry_risk = rh_air < RH_DRY_RISK or vpd > VPD_HIGH_LIMIT
    dew_risk = rh_air >= RH_DEW_RISK or dew_margin < DEW_MARGIN_RISK
    extreme_dew_risk = rh_air >= RH_DEW_RISK or dew_margin < 0.40
    cold_vent_risk = temp_air < COLD_TEMP_LIMIT and vent > 0.12 and not extreme_dew_risk
    dry_vent_risk = dry_risk and vent > _dry_vent_cap(temp_air)

    row.update(
        {
            "vpd_air": vpd,
            "vpd_kpa": vpd,
            "rh_low_violation": max(RH_LOW_LIMIT - rh_air, 0.0),
            "rh_high_violation": max(rh_air - RH_HIGH_LIMIT, 0.0),
            "vpd_low_excess": max(VPD_LOW_LIMIT - vpd, 0.0),
            "vpd_high_excess": max(vpd - VPD_HIGH_LIMIT, 0.0),
            "dew_margin_min": dew_margin,
            "dry_risk": bool(dry_risk),
            "dew_risk": bool(dew_risk),
            "cold_vent_risk": bool(cold_vent_risk),
            "dry_vent_risk": bool(dry_vent_risk),
            "dry_vent_cap": float(_dry_vent_cap(temp_air)),
        }
    )
    return row


def _diagnostic_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _diagnostic_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "applied"}


def _load_action_record(value: Any) -> Dict[str, float]:
    if isinstance(value, dict):
        return {str(key): _diagnostic_float(val) for key, val in value.items()}
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except Exception:
            decoded = None
        if isinstance(decoded, dict):
            return {str(key): _diagnostic_float(val) for key, val in decoded.items()}
        if isinstance(decoded, list):
            return {
                str(name): _diagnostic_float(decoded[idx])
                for idx, name in enumerate(ACTION_NAMES)
                if idx < len(decoded)
            }
    if isinstance(value, (list, tuple, np.ndarray)):
        return {
            str(name): _diagnostic_float(value[idx])
            for idx, name in enumerate(ACTION_NAMES)
            if idx < len(value)
        }
    return {}


def _guardrail_before_after(row: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, float]]:
    before = _load_action_record(row.get("tomato_safety_v2_before", ""))
    after = _load_action_record(row.get("tomato_safety_v2_after", ""))
    fallback_pairs = {
        "heating": ("tomato_safety_v2_heat_before", "tomato_safety_v2_heat_after"),
        "ventilation": ("tomato_safety_v2_vent_before", "tomato_safety_v2_vent_after"),
        "screen": ("tomato_safety_v2_screen_before", "tomato_safety_v2_screen_after"),
        "shading": ("tomato_safety_v2_shade_before", "tomato_safety_v2_shade_after"),
    }
    for name, (before_key, after_key) in fallback_pairs.items():
        if before_key in row or after_key in row:
            before[name] = _diagnostic_float(row.get(before_key), before.get(name, 0.0))
            after[name] = _diagnostic_float(row.get(after_key), after.get(name, 0.0))
    return before, after


def derive_candidate_guardrail_compatibility(row: Dict[str, Any]) -> Dict[str, Any]:
    before, after = _guardrail_before_after(row)
    rewrite_fields = [
        name
        for name in ACTION_NAMES
        if abs(_diagnostic_float(after.get(name)) - _diagnostic_float(before.get(name))) > 1e-6
    ]
    delta_abs_sum = float(
        sum(abs(_diagnostic_float(after.get(name)) - _diagnostic_float(before.get(name))) for name in ACTION_NAMES)
    )
    reasons = str(row.get("tomato_safety_v2_reasons", "") or "")
    reason_text = reasons.lower()
    tomato_applied = _diagnostic_bool(row.get("tomato_safety_v2_applied"))
    hard_tokens = ("hard", "dew", "canopy", "extreme", "hot_temperature")
    if tomato_applied and any(token in reason_text for token in hard_tokens):
        compatibility_label = "hard_safety_rewrite"
    elif delta_abs_sum >= 0.50 or len(rewrite_fields) >= 2:
        compatibility_label = "major_rewrite"
    elif delta_abs_sum > 0.05 or tomato_applied:
        compatibility_label = "minor_rewrite"
    else:
        compatibility_label = "compatible"

    target_temp = _diagnostic_float(row.get("target_temp"))
    target_rh = _diagnostic_float(row.get("target_rh"))
    target_co2 = _diagnostic_float(row.get("target_co2"))
    temp_air = _diagnostic_float(row.get("temp_air"))
    vent = _diagnostic_float(row.get("u_ventilation"))
    heat = _diagnostic_float(row.get("u_heating"))
    conflicts: List[str] = []
    if target_rh >= 75.0 and vent >= 0.70:
        conflicts.append("target_rh_vs_high_vent")
    if target_co2 >= 800.0 and vent >= 0.30:
        conflicts.append("co2_target_vs_vent")
    if target_temp <= temp_air - 1.0 and heat >= 0.10:
        conflicts.append("cooling_target_vs_heat")
    if target_temp >= temp_air + 1.0 and vent >= 0.50:
        conflicts.append("heating_target_vs_vent")
    if _diagnostic_bool(row.get("dry_vent_risk")):
        conflicts.append("dry_risk_vs_high_vent")
    conflict_label = ",".join(conflicts) if conflicts else "none"

    filter_reason_fields = (
        "profile_scorer_safety_gate_reason",
        "profile_rspc_shadow_best_ineligible_reason",
        "profile_rspc_shadow_safety_gate_reason",
        "profile_rspc_shadow_raw_best_ineligible_reason",
        "profile_rspc_shadow_raw_best_safety_alignment",
        "rspc_action_post_shape_safety_gate_reason",
        "rspc_hot_dry_replay_safety_gate_reason",
        "rspc_hot_dry_proposer_control_safety_gate_reason",
        "rspc_tt_calibration_safety_gate_reason",
        "mc_sero_reject_reason",
    )
    filter_reasons = []
    for field in filter_reason_fields:
        value = str(row.get(field, "") or "").strip()
        if value and value.lower() not in {"none", "neutral_hold", "0", "false"}:
            filter_reasons.append(f"{field}={value}")

    if str(row.get("selected_fallback_candidate", "") or "").strip():
        selection_source = "fallback_candidate"
    elif int(_diagnostic_float(row.get("rspc_action_actual_candidate_count"), 0.0)) > 0 or int(
        _diagnostic_float(row.get("rspc_action_candidate_count"), 0.0)
    ) > 0:
        selection_source = "rollout_candidate"
    elif str(row.get("source", "") or "").strip().lower() == "anchor":
        selection_source = "anchor"
    elif str(row.get("buffered_control", "") or "").strip():
        selection_source = "buffered_control"
    else:
        selection_source = "unknown"

    return {
        "candidate_guardrail_compatibility_label": compatibility_label,
        "candidate_guardrail_rewrite_fields": ",".join(rewrite_fields),
        "candidate_guardrail_delta_abs_sum": delta_abs_sum,
        "profile_action_conflict_label": conflict_label,
        "candidate_selection_source": selection_source,
        "candidate_filter_reason_summary": "; ".join(filter_reasons),
    }


def append_candidate_guardrail_compatibility(row: Dict[str, Any]) -> Dict[str, Any]:
    row.update(derive_candidate_guardrail_compatibility(row))
    return row


def finalize_trace_row(row: Dict[str, Any]) -> Dict[str, Any]:
    append_regime_labels(row)
    append_climate_risk_metrics(row)
    append_strategy_label(row)
    append_candidate_guardrail_compatibility(row)
    return row


def _profile_value_at_step(profile: Any, key: str, created_timestep: int, step: int, default: float = 0.0) -> float:
    values = profile.get(key) if isinstance(profile, dict) else None
    if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
        idx = int(np.clip(int(step) - int(created_timestep), 0, len(values) - 1))
        try:
            return float(values[idx])
        except Exception:
            return float(default)
    try:
        return float(default)
    except Exception:
        return 0.0


def profile_generator_shadow_to_record(plan: Any, step: int, state: Any = None) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    intent = plan.get("intent_contract", {}) if isinstance(plan.get("intent_contract", {}), dict) else {}
    diagnostics = (
        plan.get("profile_generator_diagnostics", {})
        if isinstance(plan.get("profile_generator_diagnostics", {}), dict)
        else {}
    )
    candidates = plan.get("profile_candidates", []) if isinstance(plan.get("profile_candidates", []), list) else []
    selected_name = str(diagnostics.get("selected_shadow_profile_name") or "")
    scorer_diagnostics = dict(diagnostics)
    if state is not None and candidates:
        try:
            _scores, refreshed = score_profile_candidate_payloads(
                candidates,
                intent,
                state,
                current_plan=plan,
                horizon_steps=int(diagnostics.get("horizon_steps", 12) or 12),
            )
            scorer_diagnostics.update(refreshed)
        except Exception:
            pass
    scorer_selected_name = str(scorer_diagnostics.get("score_selected_shadow_profile_name") or "")
    candidate_names = [
        str(item.get("name"))
        for item in candidates
        if isinstance(item, dict) and str(item.get("name") or "")
    ]
    selected = next(
        (item for item in candidates if isinstance(item, dict) and str(item.get("name") or "") == selected_name),
        None,
    )
    scorer_selected = next(
        (item for item in candidates if isinstance(item, dict) and str(item.get("name") or "") == scorer_selected_name),
        None,
    )
    created = int(plan.get("created_timestep", int(step)) or int(step))
    actual_profile = plan.get("target_profile", {}) if isinstance(plan.get("target_profile", {}), dict) else {}
    actual_temp = float(plan.get("current_target_temp") or _profile_value_at_step(actual_profile, "target_temp", created, step))
    actual_co2 = float(plan.get("current_target_co2") or _profile_value_at_step(actual_profile, "target_co2", created, step))
    actual_rh = float(plan.get("current_target_rh") or _profile_value_at_step(actual_profile, "target_rh", created, step))
    selected_profile = selected.get("target_profile", {}) if isinstance(selected, dict) else {}
    selected_temp = _profile_value_at_step(selected_profile, "target_temp", created, step, default=actual_temp)
    selected_co2 = _profile_value_at_step(selected_profile, "target_co2", created, step, default=actual_co2)
    selected_rh = _profile_value_at_step(selected_profile, "target_rh", created, step, default=actual_rh)
    delta_temp = selected_temp - actual_temp
    delta_co2 = selected_co2 - actual_co2
    delta_rh = selected_rh - actual_rh
    scorer_profile = scorer_selected.get("target_profile", {}) if isinstance(scorer_selected, dict) else {}
    scorer_temp = _profile_value_at_step(scorer_profile, "target_temp", created, step, default=actual_temp)
    scorer_co2 = _profile_value_at_step(scorer_profile, "target_co2", created, step, default=actual_co2)
    scorer_rh = _profile_value_at_step(scorer_profile, "target_rh", created, step, default=actual_rh)
    scorer_delta_temp = scorer_temp - actual_temp
    scorer_delta_co2 = scorer_co2 - actual_co2
    scorer_delta_rh = scorer_rh - actual_rh
    score_by_candidate = scorer_diagnostics.get("score_by_candidate", {})
    if isinstance(score_by_candidate, dict):
        score_text = ",".join(f"{key}:{float(value):.6f}" for key, value in sorted(score_by_candidate.items()))
    else:
        score_text = ""
    score_breakdown = scorer_diagnostics.get("score_selected_breakdown", {})
    if not isinstance(score_breakdown, dict):
        score_breakdown = {}
    return {
        "intent_regime": intent.get("regime") if isinstance(intent, dict) else None,
        "intent_confidence": float(intent.get("confidence", 0.0)) if isinstance(intent, dict) else 0.0,
        "profile_generator_shadow_only": bool(diagnostics.get("shadow_only", False)),
        "profile_generator_candidate_count": int(diagnostics.get("candidate_count", len(candidates)) or 0),
        "profile_generator_selected_shadow_profile": selected_name,
        "profile_generator_selected_shadow_profile_reason": diagnostics.get("selected_shadow_profile_reason", ""),
        "profile_generator_requested_shapes": ",".join(
            str(x) for x in diagnostics.get("requested_shapes", []) if str(x)
        )
        if isinstance(diagnostics.get("requested_shapes", []), list)
        else "",
        "profile_generator_unsupported_shapes": ",".join(
            str(x) for x in diagnostics.get("unsupported_shapes", []) if str(x)
        )
        if isinstance(diagnostics.get("unsupported_shapes", []), list)
        else "",
        "profile_candidate_names": ",".join(candidate_names),
        "profile_selected_target_temp": float(selected_temp),
        "profile_selected_target_co2": float(selected_co2),
        "profile_selected_target_rh": float(selected_rh),
        "profile_selected_delta_target_temp": float(delta_temp),
        "profile_selected_delta_target_co2": float(delta_co2),
        "profile_selected_delta_target_rh": float(delta_rh),
        "profile_selected_abs_delta_sum": float(abs(delta_temp) + abs(delta_co2) / 100.0 + abs(delta_rh)),
        "profile_scorer_shadow_only": bool(scorer_diagnostics.get("score_shadow_only", False)),
        "profile_scorer_selected_shadow_profile": scorer_selected_name,
        "profile_scorer_selected_score": float(scorer_diagnostics.get("score_selected_shadow_profile_score", 0.0) or 0.0),
        "profile_scorer_margin_to_second": float(scorer_diagnostics.get("score_margin_to_second", 0.0) or 0.0),
        "profile_scorer_safety_gate_reason": str(scorer_diagnostics.get("score_safety_gate_reason", "none") or "none"),
        "profile_scorer_agrees_with_selector": bool(
            selected_name and scorer_selected_name and selected_name == scorer_selected_name
        ),
        "profile_scorer_ranked_candidates": ",".join(
            str(x) for x in scorer_diagnostics.get("score_ranked_candidates", []) if str(x)
        )
        if isinstance(scorer_diagnostics.get("score_ranked_candidates", []), list)
        else "",
        "profile_scorer_candidate_scores": score_text,
        "profile_scorer_target_temp": float(scorer_temp),
        "profile_scorer_target_co2": float(scorer_co2),
        "profile_scorer_target_rh": float(scorer_rh),
        "profile_scorer_delta_target_temp": float(scorer_delta_temp),
        "profile_scorer_delta_target_co2": float(scorer_delta_co2),
        "profile_scorer_delta_target_rh": float(scorer_delta_rh),
        "profile_scorer_abs_delta_sum": float(abs(scorer_delta_temp) + abs(scorer_delta_co2) / 100.0 + abs(scorer_delta_rh)),
        "profile_scorer_temperature_high_risk": float(score_breakdown.get("temperature_high_risk", 0.0) or 0.0),
        "profile_scorer_temperature_low_risk": float(score_breakdown.get("temperature_low_risk", 0.0) or 0.0),
        "profile_scorer_rh_low_risk": float(score_breakdown.get("rh_low_risk", 0.0) or 0.0),
        "profile_scorer_rh_high_risk": float(score_breakdown.get("rh_high_risk", 0.0) or 0.0),
        "profile_scorer_vpd_high_risk": float(score_breakdown.get("vpd_high_risk", 0.0) or 0.0),
        "profile_scorer_vpd_low_risk": float(score_breakdown.get("vpd_low_risk", 0.0) or 0.0),
        "profile_scorer_dew_risk": float(score_breakdown.get("dew_risk", 0.0) or 0.0),
        "profile_scorer_hot_dry_retention_gap": float(score_breakdown.get("hot_dry_retention_gap", 0.0) or 0.0),
        "profile_scorer_humid_relief_gap": float(score_breakdown.get("humid_relief_gap", 0.0) or 0.0),
        "profile_scorer_hot_danger_penalty": float(score_breakdown.get("hot_danger_penalty", 0.0) or 0.0),
        "profile_scorer_safety_gate_penalty": float(score_breakdown.get("safety_gate_penalty", 0.0) or 0.0),
    }


def profile_rspc_shadow_to_record(plan: Any) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    rollout = plan.get("rollout_selection", {})
    if not isinstance(rollout, dict):
        return {}
    shadow = rollout.get("profile_rspc_shadow", {})
    if not isinstance(shadow, dict):
        return {}
    top_candidates = shadow.get("top_candidates", [])
    if not isinstance(top_candidates, list):
        top_candidates = []
    top_names = [
        str(item.get("name"))
        for item in top_candidates
        if isinstance(item, dict) and str(item.get("name") or "")
    ]
    top_scores = []
    for item in top_candidates:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        try:
            score = float(item.get("selection_score", 0.0) or 0.0)
            raw_score = float(item.get("raw_score", 0.0) or 0.0)
            safety_score = float(item.get("safety_score", 0.0) or 0.0)
        except Exception:
            continue
        top_scores.append(f"{name}:{score:.6f}:{raw_score:.6f}:{safety_score:.6f}")
    raw_top_candidates = shadow.get("raw_top_candidates", [])
    if not isinstance(raw_top_candidates, list):
        raw_top_candidates = []
    raw_top_names = [
        str(item.get("name"))
        for item in raw_top_candidates
        if isinstance(item, dict) and str(item.get("name") or "")
    ]
    action_delta = shadow.get("best_action_delta_terms", {})
    if not isinstance(action_delta, dict):
        action_delta = {}
    score_delta = shadow.get("best_score_delta_terms", {})
    if not isinstance(score_delta, dict):
        score_delta = {}
    raw_action_delta = shadow.get("raw_best_action_delta_terms", {})
    if not isinstance(raw_action_delta, dict):
        raw_action_delta = {}
    raw_score_delta = shadow.get("raw_best_score_delta_terms", {})
    if not isinstance(raw_score_delta, dict):
        raw_score_delta = {}
    return {
        "profile_rspc_shadow_enabled": bool(shadow.get("enabled", False)),
        "profile_rspc_shadow_only": bool(shadow.get("shadow_only", False)),
        "profile_rspc_shadow_candidate_count": int(shadow.get("candidate_count", 0) or 0),
        "profile_rspc_shadow_eligible_candidate_count": int(shadow.get("eligible_candidate_count", 0) or 0),
        "profile_rspc_shadow_best_profile": str(shadow.get("best_profile", "") or ""),
        "profile_rspc_shadow_best_score": float(shadow.get("best_score", 0.0) or 0.0),
        "profile_rspc_shadow_best_raw_score": float(shadow.get("best_raw_score", 0.0) or 0.0),
        "profile_rspc_shadow_best_safety_score": float(shadow.get("best_safety_score", 0.0) or 0.0),
        "profile_rspc_shadow_best_eligible": bool(shadow.get("best_eligible", False)),
        "profile_rspc_shadow_best_ineligible_reason": str(shadow.get("best_ineligible_reason", "") or ""),
        "profile_rspc_shadow_baseline_score": float(shadow.get("baseline_score", 0.0) or 0.0),
        "profile_rspc_shadow_baseline_raw_score": float(shadow.get("baseline_raw_score", 0.0) or 0.0),
        "profile_rspc_shadow_margin": float(shadow.get("margin", 0.0) or 0.0),
        "profile_rspc_shadow_would_improve": bool(shadow.get("would_improve", False)),
        "profile_rspc_shadow_raw_best_profile": str(shadow.get("raw_best_profile", "") or ""),
        "profile_rspc_shadow_raw_best_score": float(shadow.get("raw_best_score", 0.0) or 0.0),
        "profile_rspc_shadow_raw_best_margin": float(shadow.get("raw_best_margin", 0.0) or 0.0),
        "profile_rspc_shadow_raw_best_would_improve": bool(shadow.get("raw_best_would_improve", False)),
        "profile_rspc_shadow_raw_best_eligible": bool(shadow.get("raw_best_eligible", False)),
        "profile_rspc_shadow_raw_best_ineligible_reason": str(shadow.get("raw_best_ineligible_reason", "") or ""),
        "profile_rspc_shadow_safety_gate_reason": str(shadow.get("safety_gate_reason", "none") or "none"),
        "profile_rspc_shadow_safety_alignment": str(shadow.get("safety_alignment", "neutral_hold") or "neutral_hold"),
        "profile_rspc_shadow_raw_best_safety_alignment": str(
            shadow.get("raw_best_safety_alignment", "neutral_hold") or "neutral_hold"
        ),
        "profile_rspc_shadow_scorer_selected_profile": str(shadow.get("scorer_selected_profile", "") or ""),
        "profile_rspc_shadow_best_agrees_with_scorer": bool(shadow.get("best_agrees_with_scorer", False)),
        "profile_rspc_shadow_top_candidates": ",".join(top_names),
        "profile_rspc_shadow_raw_top_candidates": ",".join(raw_top_names),
        "profile_rspc_shadow_top_scores": ",".join(top_scores),
        "profile_rspc_shadow_delta_heat": float(action_delta.get("heat", 0.0) or 0.0),
        "profile_rspc_shadow_delta_co2": float(action_delta.get("co2", 0.0) or 0.0),
        "profile_rspc_shadow_delta_screen": float(action_delta.get("screen", 0.0) or 0.0),
        "profile_rspc_shadow_delta_vent": float(action_delta.get("vent", 0.0) or 0.0),
        "profile_rspc_shadow_delta_lamp": float(action_delta.get("lamp", 0.0) or 0.0),
        "profile_rspc_shadow_delta_shade": float(action_delta.get("shade", 0.0) or 0.0),
        "profile_rspc_shadow_delta_score_temp": float(score_delta.get("temp", 0.0) or 0.0),
        "profile_rspc_shadow_delta_score_rh": float(score_delta.get("rh", 0.0) or 0.0),
        "profile_rspc_shadow_delta_score_dry": float(score_delta.get("dry", 0.0) or 0.0),
        "profile_rspc_shadow_delta_score_vpd": float(score_delta.get("vpd", 0.0) or 0.0),
        "profile_rspc_shadow_delta_score_dew": float(score_delta.get("dew", 0.0) or 0.0),
        "profile_rspc_shadow_delta_score_hot_dry": float(score_delta.get("hot_dry", 0.0) or 0.0),
        "profile_rspc_shadow_delta_score_energy": float(score_delta.get("energy", 0.0) or 0.0),
        "profile_rspc_shadow_raw_delta_vent": float(raw_action_delta.get("vent", 0.0) or 0.0),
        "profile_rspc_shadow_raw_delta_shade": float(raw_action_delta.get("shade", 0.0) or 0.0),
        "profile_rspc_shadow_raw_delta_score_temp": float(raw_score_delta.get("temp", 0.0) or 0.0),
        "profile_rspc_shadow_raw_delta_score_rh": float(raw_score_delta.get("rh", 0.0) or 0.0),
        "profile_rspc_shadow_raw_delta_score_dew": float(raw_score_delta.get("dew", 0.0) or 0.0),
    }


def profile_action_candidate_shadow_to_record(plan: Any) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    rollout = plan.get("rollout_selection", {})
    if not isinstance(rollout, dict):
        return {}
    shadow = rollout.get("profile_action_candidate_shadow", {})
    if not isinstance(shadow, dict):
        return {}
    candidates = shadow.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    best_score = shadow.get("best_score")
    try:
        best_score_value = None if best_score is None else float(best_score)
    except Exception:
        best_score_value = None
    return {
        "profile_action_candidate_shadow_enabled": bool(shadow.get("enabled", False)),
        "profile_action_candidate_shadow_only": bool(shadow.get("shadow_only", False)),
        "profile_action_candidate_shadow_candidate_count": int(shadow.get("candidate_count", 0) or 0),
        "profile_action_candidate_shadow_eligible_candidate_count": int(
            shadow.get("eligible_candidate_count", 0) or 0
        ),
        "profile_action_candidate_shadow_best_name": str(shadow.get("best_name", "") or ""),
        "profile_action_candidate_shadow_best_profile": str(shadow.get("best_profile", "") or ""),
        "profile_action_candidate_shadow_best_score": best_score_value,
        "profile_action_candidate_shadow_best_eligible": bool(shadow.get("best_eligible", False)),
        "profile_action_candidate_shadow_best_rejection_reason": str(
            shadow.get("best_rejection_reason", "") or ""
        ),
        "profile_action_candidate_shadow_final_action_changed": bool(
            shadow.get("final_action_changed", False)
        ),
        "profile_action_candidate_shadow_candidates_json": _compact_json(candidates),
    }


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return ""


def profile_action_envelope_shadow_to_record(plan: Any) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    rollout = plan.get("rollout_selection", {})
    if not isinstance(rollout, dict):
        return {}
    shadow = rollout.get("profile_action_envelope_shadow", {})
    if not isinstance(shadow, dict):
        return {}
    candidates = shadow.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    best_score = shadow.get("best_score")
    try:
        best_score_value = None if best_score is None else float(best_score)
    except Exception:
        best_score_value = None
    return {
        "profile_action_envelope_shadow_enabled": bool(shadow.get("enabled", False)),
        "profile_action_envelope_shadow_only": bool(shadow.get("shadow_only", False)),
        "profile_action_envelope_shadow_candidate_count": int(shadow.get("candidate_count", 0) or 0),
        "profile_action_envelope_shadow_eligible_candidate_count": int(
            shadow.get("eligible_candidate_count", 0) or 0
        ),
        "profile_action_envelope_shadow_best_name": str(shadow.get("best_name", "") or ""),
        "profile_action_envelope_shadow_best_profile": str(shadow.get("best_profile", "") or ""),
        "profile_action_envelope_shadow_best_score": best_score_value,
        "profile_action_envelope_shadow_best_eligible": bool(shadow.get("best_eligible", False)),
        "profile_action_envelope_shadow_best_rejection_reason": str(
            shadow.get("best_rejection_reason", "") or ""
        ),
        "profile_action_envelope_shadow_final_action_changed": bool(
            shadow.get("final_action_changed", False)
        ),
        "profile_action_envelope_shadow_candidates_json": _compact_json(candidates),
    }


def cstcc_shadow_to_record(plan: Any) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    shadow = plan.get("cstcc_shadow", {})
    if not isinstance(shadow, dict) or not shadow:
        rollout = plan.get("rollout_selection", {})
        if isinstance(rollout, dict):
            shadow = rollout.get("cstcc_shadow", {})
    if not isinstance(shadow, dict):
        return {}
    action_diff = shadow.get("shadow_action_difference_from_runtime") or {}
    if not isinstance(action_diff, dict):
        action_diff = {}
    violation_reasons = shadow.get("hard_constraint_violation_reason_distribution") or {}
    if not isinstance(violation_reasons, dict):
        violation_reasons = {}
    violation_fields = shadow.get("hard_constraint_violation_field_distribution") or {}
    if not isinstance(violation_fields, dict):
        violation_fields = {}
    candidate_violation_provenance = shadow.get("candidate_violation_provenance") or []
    if not isinstance(candidate_violation_provenance, list):
        candidate_violation_provenance = []
    selected_violation_reasons = shadow.get("selected_candidate_violation_reason_distribution") or {}
    if not isinstance(selected_violation_reasons, dict):
        selected_violation_reasons = {}
    selected_violation_fields = shadow.get("selected_candidate_violation_field_distribution") or {}
    if not isinstance(selected_violation_fields, dict):
        selected_violation_fields = {}
    rule_conservative_competitiveness = shadow.get("rule_conservative_competitiveness") or {}
    if not isinstance(rule_conservative_competitiveness, dict):
        rule_conservative_competitiveness = {}
    constraint_reference_action = shadow.get("constraint_reference_action") or {}
    if not isinstance(constraint_reference_action, dict):
        constraint_reference_action = {}
    hold_all_reference = shadow.get("hold_all_reference") or {}
    if not isinstance(hold_all_reference, dict):
        hold_all_reference = {}
    risk_flags = shadow.get("risk_flags") or {}
    if not isinstance(risk_flags, dict):
        risk_flags = {}
    risk_category_flags = shadow.get("risk_category_flags") or shadow.get("conservative_prior_risk_category_flags") or {}
    if not isinstance(risk_category_flags, dict):
        risk_category_flags = {}
    previous_shift_projection_mean_abs_delta = shadow.get("previous_shift_projection_mean_abs_delta_by_field") or {}
    if not isinstance(previous_shift_projection_mean_abs_delta, dict):
        previous_shift_projection_mean_abs_delta = {}
    previous_shift_internal_max_delta = shadow.get("previous_shift_internal_max_delta_after_projection_by_field") or {}
    if not isinstance(previous_shift_internal_max_delta, dict):
        previous_shift_internal_max_delta = {}
    previous_shift_violation_fields = shadow.get("previous_shift_all_violation_field_distribution") or {}
    if not isinstance(previous_shift_violation_fields, dict):
        previous_shift_violation_fields = {}
    prior_generation_diagnostics = shadow.get("prior_generation_diagnostics") or {}
    if not isinstance(prior_generation_diagnostics, dict):
        prior_generation_diagnostics = {}
    score_margin_to_selected = shadow.get("score_margin_to_selected_by_prior") or {}
    if not isinstance(score_margin_to_selected, dict):
        score_margin_to_selected = {}
    score_component_by_candidate = shadow.get("score_component_by_candidate") or {}
    if not isinstance(score_component_by_candidate, dict):
        score_component_by_candidate = {}
    rule_margin = score_margin_to_selected.get("rule_prior") or {}
    if not isinstance(rule_margin, dict):
        rule_margin = {}
    conservative_diag = prior_generation_diagnostics.get("conservative_prior") or {}
    if not isinstance(conservative_diag, dict):
        conservative_diag = {}
    conservative_margin = score_margin_to_selected.get("conservative_prior") or {}
    if not isinstance(conservative_margin, dict):
        conservative_margin = {}
    current_action = shadow.get("current_runtime_final_action") or {}
    selected_action = shadow.get("shadow_selected_first_action") or {}
    selected_sequence_id = str(shadow.get("shadow_selected_sequence_id", "") or "")
    selected_source_prior = str(shadow.get("shadow_selected_source_prior", "") or "")
    selected_template_name = str(shadow.get("shadow_selected_template_name", "") or "")
    if (not selected_source_prior) or (not selected_template_name):
        parsed_source_prior, parsed_template_name = parse_selected_sequence_id(selected_sequence_id)
        selected_source_prior = selected_source_prior or parsed_source_prior
        selected_template_name = selected_template_name or parsed_template_name
    return {
        "cstcc_shadow_enabled": bool(shadow.get("enabled", False)),
        "cstcc_shadow_only": bool(shadow.get("shadow_only", shadow.get("selected_action_is_shadow_only", True))),
        "cstcc_shadow_audit_success": bool(shadow.get("audit_success", False)),
        "cstcc_shadow_failure_type": str(shadow.get("failure_type", "") or ""),
        "cstcc_shadow_failure_message": str(shadow.get("failure_message", "") or ""),
        "cstcc_shadow_final_action_changed": bool(shadow.get("final_action_changed", False)),
        "cstcc_shadow_final_action_invariant_verified": bool(
            shadow.get("final_action_invariant_verified", False)
        ),
        "cstcc_shadow_online_llm_called": bool(shadow.get("online_llm_called", False)),
        "cstcc_shadow_predictive_rollout_executed": bool(shadow.get("predictive_rollout_executed", False)),
        "cstcc_shadow_real_tomato_safety_projection": bool(
            shadow.get("real_tomato_safety_projection", False)
        ),
        "cstcc_shadow_selected_action_is_shadow_only": bool(
            shadow.get("selected_action_is_shadow_only", True)
        ),
        "cstcc_shadow_prediction_level": int(shadow.get("prediction_level", 0) or 0),
        "cstcc_shadow_score_validity": str(shadow.get("score_validity", "") or ""),
        "cstcc_shadow_projection_level": str(shadow.get("projection_level", "") or ""),
        "cstcc_shadow_candidate_count": int(shadow.get("candidate_count", 0) or 0),
        "cstcc_shadow_feasible_candidate_count": int(shadow.get("feasible_candidate_count", 0) or 0),
        "cstcc_shadow_infeasible_candidate_ratio": float(
            shadow.get("infeasible_candidate_ratio", 0.0) or 0.0
        ),
        "cstcc_shadow_hard_constraint_violation_count": int(
            shadow.get("hard_constraint_violation_count", 0) or 0
        ),
        "cstcc_shadow_hard_constraint_violation_reason_distribution_json": _compact_json(
            violation_reasons
        ),
        "cstcc_shadow_hard_constraint_violation_field_distribution_json": _compact_json(
            violation_fields
        ),
        "cstcc_shadow_candidate_attribution_mode": str(
            shadow.get("candidate_attribution_mode", "selected_step_coincidence_attribution") or ""
        ),
        "cstcc_shadow_candidate_violation_provenance_json": _compact_json(candidate_violation_provenance),
        "cstcc_shadow_selected_candidate_violation_count": int(
            shadow.get("selected_candidate_violation_count", 0) or 0
        ),
        "cstcc_shadow_selected_candidate_violation_reason_distribution_json": _compact_json(
            selected_violation_reasons
        ),
        "cstcc_shadow_selected_candidate_violation_field_distribution_json": _compact_json(
            selected_violation_fields
        ),
        "cstcc_shadow_rule_conservative_competitiveness_json": _compact_json(
            rule_conservative_competitiveness
        ),
        "cstcc_shadow_prior_generation_diagnostics_json": _compact_json(prior_generation_diagnostics),
        "cstcc_shadow_score_margin_to_selected_by_prior_json": _compact_json(score_margin_to_selected),
        "cstcc_shadow_score_component_by_candidate_json": _compact_json(score_component_by_candidate),
        "cstcc_shadow_scorer_diagnostic_log_mode": str(
            shadow.get("scorer_diagnostic_log_mode", "") or ""
        ),
        "cstcc_shadow_rule_prior_selected_score_margin": (
            "" if rule_margin.get("total_margin") is None else float(rule_margin.get("total_margin", 0.0) or 0.0)
        ),
        "cstcc_shadow_rule_prior_margin_status": str(rule_margin.get("status", "") or ""),
        "cstcc_shadow_rule_prior_best_template_name": str(rule_margin.get("best_template_name", "") or ""),
        "cstcc_shadow_conservative_prior_confidence": (
            "" if conservative_diag.get("confidence") is None else float(conservative_diag.get("confidence", 0.0) or 0.0)
        ),
        "cstcc_shadow_conservative_prior_candidate_count": int(
            conservative_diag.get("candidate_count", 0) or 0
        ),
        "cstcc_shadow_conservative_prior_candidate_count_before_override": int(
            conservative_diag.get("candidate_count_before_override", 0) or 0
        ),
        "cstcc_shadow_conservative_prior_candidate_count_after_override": int(
            conservative_diag.get("candidate_count_after_override", 0) or 0
        ),
        "cstcc_shadow_conservative_prior_effective_candidate_count": int(
            conservative_diag.get("effective_candidate_count", conservative_diag.get("candidate_count", 0)) or 0
        ),
        "cstcc_shadow_conservative_prior_candidate_count_override_applied": bool(
            conservative_diag.get("candidate_count_override_applied", False)
        ),
        "cstcc_shadow_conservative_prior_candidate_count_override_reason": str(
            conservative_diag.get("candidate_count_override_reason", "") or ""
        ),
        "cstcc_shadow_conservative_prior_candidate_count_override_flags_json": _compact_json(
            conservative_diag.get("candidate_count_override_flags", [])
        ),
        "cstcc_shadow_conservative_prior_risk_category_flags_json": _compact_json(
            conservative_diag.get("risk_category_flags", risk_category_flags)
        ),
        "cstcc_shadow_conservative_prior_generated_template_count": int(
            conservative_diag.get("generated_template_count", 0) or 0
        ),
        "cstcc_shadow_conservative_prior_no_candidate_reason": str(
            conservative_diag.get("no_candidate_reason", "") or ""
        ),
        "cstcc_shadow_conservative_prior_selected_score_margin": (
            ""
            if conservative_margin.get("total_margin") is None
            else float(conservative_margin.get("total_margin", 0.0) or 0.0)
        ),
        "cstcc_shadow_conservative_prior_margin_status": str(conservative_margin.get("status", "") or ""),
        "cstcc_shadow_conservative_prior_best_template_name": str(
            conservative_margin.get("best_template_name", "") or ""
        ),
        "cstcc_shadow_constraint_reference_kind": str(shadow.get("constraint_reference_kind", "") or ""),
        "cstcc_shadow_constraint_reference_action_json": _compact_json(constraint_reference_action),
        "cstcc_shadow_hold_all_reference_kind": str(shadow.get("hold_all_reference_kind", "") or ""),
        "cstcc_shadow_hold_all_reference_json": _compact_json(hold_all_reference),
        "cstcc_shadow_risk_flags_json": _compact_json(risk_flags),
        "cstcc_shadow_risk_category_flags_json": _compact_json(risk_category_flags),
        "cstcc_shadow_previous_shift_all_candidate_count": int(
            shadow.get("previous_shift_all_candidate_count", 0) or 0
        ),
        "cstcc_shadow_previous_shift_all_max_delta_count": int(
            shadow.get("previous_shift_all_max_delta_count", 0) or 0
        ),
        "cstcc_shadow_previous_shift_all_selected_count": int(
            shadow.get("previous_shift_all_selected_count", 0) or 0
        ),
        "cstcc_shadow_previous_shift_projection_applied_count": int(
            shadow.get("previous_shift_projection_applied_count", 0) or 0
        ),
        "cstcc_shadow_previous_shift_first_delta_after_projection_max": float(
            shadow.get("previous_shift_first_delta_after_projection_max", 0.0) or 0.0
        ),
        "cstcc_shadow_previous_shift_projection_mean_abs_delta_by_field_json": _compact_json(
            previous_shift_projection_mean_abs_delta
        ),
        "cstcc_shadow_previous_shift_internal_max_delta_after_projection_by_field_json": _compact_json(
            previous_shift_internal_max_delta
        ),
        "cstcc_shadow_previous_shift_all_violation_field_distribution_json": _compact_json(
            previous_shift_violation_fields
        ),
        "cstcc_shadow_fallback_triggered": bool(shadow.get("fallback_triggered", False)),
        "cstcc_shadow_fallback_reason": str(shadow.get("fallback_reason", "") or ""),
        "cstcc_shadow_active_regime_before": str(shadow.get("active_regime_before", "") or ""),
        "cstcc_shadow_active_regime_after": str(shadow.get("active_regime_after", "") or ""),
        "cstcc_shadow_evidence_confidence": float(shadow.get("evidence_confidence", 0.0) or 0.0),
        "cstcc_shadow_effective_confidence": float(shadow.get("effective_confidence", 0.0) or 0.0),
        "cstcc_shadow_llm_influence_alpha": float(shadow.get("llm_influence_alpha", 0.0) or 0.0),
        "cstcc_shadow_audit_latency_ms": float(shadow.get("audit_latency_ms", 0.0) or 0.0),
        "cstcc_shadow_audit_json_size_kb": float(shadow.get("audit_json_size_kb", 0.0) or 0.0),
        "cstcc_shadow_selected_sequence_id": selected_sequence_id,
        "cstcc_shadow_selected_source_prior": selected_source_prior,
        "cstcc_shadow_selected_template_name": selected_template_name,
        "cstcc_shadow_action_diff_json": _compact_json(action_diff),
        "cstcc_shadow_current_runtime_final_action_json": _compact_json(current_action),
        "cstcc_shadow_selected_first_action_json": _compact_json(selected_action),
        "cstcc_shadow_json": _compact_json(shadow),
    }


RUNTIME_PROVENANCE_REQUIRED_FIELDS = (
    "hook_id",
    "rule_id",
    "rule_family",
    "rule_reason",
    "source_function",
    "pre_rule_action",
    "post_rule_action",
    "delta_action",
    "input_features",
)


def _post_guardrail_runtime_provenance_records(rollout: Any) -> List[Dict[str, Any]]:
    if not isinstance(rollout, dict):
        return []
    records = rollout.get("post_guardrail_runtime_provenance", [])
    if not isinstance(records, list):
        return []
    return [dict(item) for item in records if isinstance(item, dict)]


def _runtime_provenance_reason_missing(record: Dict[str, Any]) -> bool:
    if bool(record.get("reason_missing", False)):
        return True
    for field in RUNTIME_PROVENANCE_REQUIRED_FIELDS:
        value = record.get(field)
        if value is None or value == "" or value == {}:
            return True
    return False


def post_guardrail_runtime_provenance_to_record(rollout: Any) -> Dict[str, Any]:
    records = _post_guardrail_runtime_provenance_records(rollout)
    unknown_count = sum(
        1
        for record in records
        if str(record.get("rule_id", "") or "") == "unknown_post_guardrail_rewrite"
        or str(record.get("rule_family", "") or "").endswith("_unknown")
    )
    reason_missing_count = sum(1 for record in records if _runtime_provenance_reason_missing(record))
    return {
        "post_guardrail_runtime_provenance": records,
        "post_guardrail_runtime_provenance_count": int(len(records)),
        "post_guardrail_runtime_reason_missing_count": int(reason_missing_count),
        "unknown_post_guardrail_rewrite_count": int(unknown_count),
    }


def rspc_action_scoring_to_record(plan: Any) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    rollout = plan.get("rollout_selection", {})
    if not isinstance(rollout, dict):
        return {}
    audit = rollout.get("rspc_action_scoring", {})
    if not isinstance(audit, dict):
        return {}
    candidates = audit.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    selected_action = audit.get("selected_action", {})
    if not isinstance(selected_action, dict):
        selected_action = {}
    selected_terms = audit.get("selected_score_terms", {})
    if not isinstance(selected_terms, dict):
        selected_terms = {}
    post_shape_best_action = audit.get("post_shape_best_action", {})
    if not isinstance(post_shape_best_action, dict):
        post_shape_best_action = {}
    post_shape_selected_action = audit.get("post_shape_selected_action", {})
    if not isinstance(post_shape_selected_action, dict):
        post_shape_selected_action = {}
    post_shape_best_events = audit.get("post_shape_best_events", {})
    if not isinstance(post_shape_best_events, dict):
        post_shape_best_events = {}
    post_shape_action_delta = audit.get("post_shape_action_delta_terms", {})
    if not isinstance(post_shape_action_delta, dict):
        post_shape_action_delta = {}
    post_shape_score_delta = audit.get("post_shape_score_delta_terms", {})
    if not isinstance(post_shape_score_delta, dict):
        post_shape_score_delta = {}
    dry_override = rollout.get("dry_recovery_override", {})
    if not isinstance(dry_override, dict):
        dry_override = {}
    hot_dry_features = audit.get("hot_dry_features", {})
    if not isinstance(hot_dry_features, dict):
        hot_dry_features = {}
    tt_calibration = audit.get("target_tracking_calibration", {})
    if not isinstance(tt_calibration, dict):
        tt_calibration = {}
    tt_before_action = tt_calibration.get("before_action", {})
    if not isinstance(tt_before_action, dict):
        tt_before_action = {}
    tt_current_action = tt_calibration.get("current_action", {})
    if not isinstance(tt_current_action, dict):
        tt_current_action = {}
    tt_best_action = tt_calibration.get("best_action", {})
    if not isinstance(tt_best_action, dict):
        tt_best_action = {}
    tt_best_action_delta = tt_calibration.get("best_action_delta", {})
    if not isinstance(tt_best_action_delta, dict):
        tt_best_action_delta = {}
    tt_best_score_delta = tt_calibration.get("best_score_delta", {})
    if not isinstance(tt_best_score_delta, dict):
        tt_best_score_delta = {}
    tt_variants = tt_calibration.get("variants", [])
    if not isinstance(tt_variants, list):
        tt_variants = []
    controlled_replay = audit.get("hot_dry_action_proposer_controlled_replay", {})
    if not isinstance(controlled_replay, dict):
        controlled_replay = {}
    proposer_control = rollout.get("hot_dry_proposer_control", {})
    if not isinstance(proposer_control, dict):
        proposer_control = {}
    proposer_before_action = proposer_control.get("before_action", {})
    if not isinstance(proposer_before_action, dict):
        proposer_before_action = {}
    proposer_after_action = proposer_control.get("after_action", {})
    if not isinstance(proposer_after_action, dict):
        proposer_after_action = {}
    proposer_delta_action = proposer_control.get("delta_action", {})
    if not isinstance(proposer_delta_action, dict):
        proposer_delta_action = {}
    canopy_boundary = rollout.get("canopy_boundary_shadow", {})
    if not isinstance(canopy_boundary, dict):
        canopy_boundary = {}
    final_risk = rollout.get("post_guardrail_final_risk_shadow", {})
    if not isinstance(final_risk, dict):
        final_risk = {}
    replay_best_action = controlled_replay.get("best_action", {})
    if not isinstance(replay_best_action, dict):
        replay_best_action = {}
    replay_best_action_delta = controlled_replay.get("best_action_delta", {})
    if not isinstance(replay_best_action_delta, dict):
        replay_best_action_delta = {}
    replay_best_score_delta = controlled_replay.get("best_score_delta", {})
    if not isinstance(replay_best_score_delta, dict):
        replay_best_score_delta = {}
    replay_variants = controlled_replay.get("variants", [])
    if not isinstance(replay_variants, list):
        replay_variants = []
    selected_names = [str(item.get("name", "")) for item in candidates if isinstance(item, dict) and item.get("selected")]
    candidate_names = [str(item.get("name", "")) for item in candidates if isinstance(item, dict)]
    return {
        "rspc_action_audit_enabled": bool(audit.get("enabled", False)),
        "rspc_action_audit_shadow_only": bool(audit.get("shadow_only", False)),
        "rspc_action_actual_candidate_count": int(audit.get("actual_candidate_count", 0) or 0),
        "rspc_action_candidate_count": int(audit.get("candidate_count", len(candidates)) or 0),
        "rspc_action_candidate_names": ",".join(name for name in candidate_names if name),
        "rspc_action_selected_name": str(audit.get("selected_name", selected_names[0] if selected_names else "") or ""),
        "rspc_action_selected_score": float(audit.get("selected_score", 0.0) or 0.0),
        "rspc_action_hot_dry_candidates_enabled": bool(audit.get("hot_dry_candidates_enabled", False)),
        "rspc_action_hot_dry_candidate_count": int(audit.get("hot_dry_candidate_count", 0) or 0),
        "rspc_action_hot_dry_proposer_enabled": bool(audit.get("hot_dry_proposer_enabled", False)),
        "rspc_action_hot_dry_proposer_active": bool(audit.get("hot_dry_proposer_active", False)),
        "rspc_action_hot_dry_proposer_candidate_count": int(audit.get("hot_dry_proposer_candidate_count", 0) or 0),
        "rspc_action_hot_dry_proposer_gate_reason": str(audit.get("hot_dry_proposer_gate_reason", "none") or "none"),
        "rspc_hot_dry_replay_enabled": bool(controlled_replay.get("enabled", False)),
        "rspc_hot_dry_replay_shadow_only": bool(controlled_replay.get("shadow_only", False)),
        "rspc_hot_dry_replay_safe_hot_dry": bool(controlled_replay.get("safe_hot_dry", False)),
        "rspc_hot_dry_replay_safety_gate_reason": str(controlled_replay.get("safety_gate_reason", "none") or "none"),
        "rspc_hot_dry_replay_reason": str(controlled_replay.get("reason", "") or ""),
        "rspc_hot_dry_replay_proposer_count": int(controlled_replay.get("proposer_count", 0) or 0),
        "rspc_hot_dry_replay_variant_count": int(controlled_replay.get("variant_count", len(replay_variants)) or 0),
        "rspc_hot_dry_replay_would_apply": bool(controlled_replay.get("would_apply_shadow", False)),
        "rspc_hot_dry_replay_unsafe_preferred": bool(controlled_replay.get("unsafe_preferred", False)),
        "rspc_hot_dry_replay_unsafe_conflict": bool(controlled_replay.get("unsafe_conflict", False)),
        "rspc_hot_dry_replay_unsafe_filtered_count": int(controlled_replay.get("unsafe_filtered_count", 0) or 0),
        "rspc_hot_dry_replay_best_candidate_name": str(controlled_replay.get("best_candidate_name", "") or ""),
        "rspc_hot_dry_replay_best_variant": str(controlled_replay.get("best_variant", "") or ""),
        "rspc_hot_dry_replay_best_margin": float(controlled_replay.get("best_margin", 0.0) or 0.0),
        "rspc_hot_dry_replay_best_alignment": str(controlled_replay.get("best_alignment", "neutral_hold") or "neutral_hold"),
        "rspc_hot_dry_proposer_control_enabled": bool(proposer_control.get("enabled", False)),
        "rspc_hot_dry_proposer_control_applied": bool(proposer_control.get("applied", False)),
        "rspc_hot_dry_proposer_control_reason": str(proposer_control.get("reason", "") or ""),
        "rspc_hot_dry_proposer_control_candidate": str(proposer_control.get("candidate", "") or ""),
        "rspc_hot_dry_proposer_control_variant": str(proposer_control.get("variant", "") or ""),
        "rspc_hot_dry_proposer_control_margin": float(proposer_control.get("margin", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_min_margin": float(proposer_control.get("min_margin", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_strict_enabled": bool(proposer_control.get("strict_enabled", False)),
        "rspc_hot_dry_proposer_control_safe_hot_dry": bool(proposer_control.get("safe_hot_dry", False)),
        "rspc_hot_dry_proposer_control_safety_gate_reason": str(
            proposer_control.get("safety_gate_reason", "none") or "none"
        ),
        "rspc_hot_dry_proposer_control_temp_air": float(proposer_control.get("temp_air", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_rh_air": float(proposer_control.get("rh_air", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_vpd_air": float(proposer_control.get("vpd_air", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_canopy_dew_margin": float(
            proposer_control.get("canopy_dew_margin", 0.0) or 0.0
        ),
        "rspc_hot_dry_proposer_control_strict_candidate": str(
            proposer_control.get("strict_candidate", "") or ""
        ),
        "rspc_hot_dry_proposer_control_before_heat": float(proposer_before_action.get("heat", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_before_co2": float(proposer_before_action.get("co2", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_before_screen": float(proposer_before_action.get("screen", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_before_vent": float(proposer_before_action.get("vent", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_before_lamp": float(proposer_before_action.get("lamp", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_before_shade": float(proposer_before_action.get("shade", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_after_heat": float(proposer_after_action.get("heat", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_after_co2": float(proposer_after_action.get("co2", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_after_screen": float(proposer_after_action.get("screen", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_after_vent": float(proposer_after_action.get("vent", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_after_lamp": float(proposer_after_action.get("lamp", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_after_shade": float(proposer_after_action.get("shade", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_delta_heat": float(proposer_delta_action.get("heat", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_delta_co2": float(proposer_delta_action.get("co2", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_delta_screen": float(proposer_delta_action.get("screen", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_delta_vent": float(proposer_delta_action.get("vent", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_delta_lamp": float(proposer_delta_action.get("lamp", 0.0) or 0.0),
        "rspc_hot_dry_proposer_control_delta_shade": float(proposer_delta_action.get("shade", 0.0) or 0.0),
        "rspc_hot_dry_replay_best_heat": float(replay_best_action.get("heat", 0.0) or 0.0),
        "rspc_hot_dry_replay_best_co2": float(replay_best_action.get("co2", 0.0) or 0.0),
        "rspc_hot_dry_replay_best_screen": float(replay_best_action.get("screen", 0.0) or 0.0),
        "rspc_hot_dry_replay_best_vent": float(replay_best_action.get("vent", 0.0) or 0.0),
        "rspc_hot_dry_replay_best_lamp": float(replay_best_action.get("lamp", 0.0) or 0.0),
        "rspc_hot_dry_replay_best_shade": float(replay_best_action.get("shade", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_heat": float(replay_best_action_delta.get("heat", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_co2": float(replay_best_action_delta.get("co2", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_screen": float(replay_best_action_delta.get("screen", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_vent": float(replay_best_action_delta.get("vent", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_lamp": float(replay_best_action_delta.get("lamp", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_shade": float(replay_best_action_delta.get("shade", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_score_temp": float(replay_best_score_delta.get("temp", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_score_rh": float(replay_best_score_delta.get("rh", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_score_dry": float(replay_best_score_delta.get("dry", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_score_vpd": float(replay_best_score_delta.get("vpd", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_score_dew": float(replay_best_score_delta.get("dew", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_score_hot_dry": float(replay_best_score_delta.get("hot_dry", 0.0) or 0.0),
        "rspc_hot_dry_replay_delta_score_energy": float(replay_best_score_delta.get("energy", 0.0) or 0.0),
        "rspc_hot_dry_replay_variants_json": _compact_json(replay_variants),
        "rspc_action_hot_dry_active": bool(hot_dry_features.get("active", False)),
        "rspc_action_hot_dry_semantic_active": bool(hot_dry_features.get("semantic_active", False)),
        "rspc_action_hot_dry_shadow_active": bool(hot_dry_features.get("shadow_active", False)),
        "rspc_action_hot_dry_shadow_gate_reason": str(hot_dry_features.get("shadow_gate_reason", "none") or "none"),
        "rspc_action_hot_dry_dry_pressure": bool(hot_dry_features.get("dry_pressure", False)),
        "rspc_action_hot_dry_hot_pressure": bool(hot_dry_features.get("hot_pressure", False)),
        "rspc_action_hot_dry_strong_rad": bool(hot_dry_features.get("strong_rad", False)),
        "rspc_action_hot_dry_extreme_dew": bool(hot_dry_features.get("extreme_dew", False)),
        "rspc_action_selected_heat": float(selected_action.get("heat", 0.0) or 0.0),
        "rspc_action_selected_co2": float(selected_action.get("co2", 0.0) or 0.0),
        "rspc_action_selected_screen": float(selected_action.get("screen", 0.0) or 0.0),
        "rspc_action_selected_vent": float(selected_action.get("vent", 0.0) or 0.0),
        "rspc_action_selected_lamp": float(selected_action.get("lamp", 0.0) or 0.0),
        "rspc_action_selected_shade": float(selected_action.get("shade", 0.0) or 0.0),
        "rspc_action_selected_temp_next": float(selected_terms.get("temp_next", 0.0) or 0.0),
        "rspc_action_selected_rh_next": float(selected_terms.get("rh_next", 0.0) or 0.0),
        "rspc_action_selected_vpd_next": float(selected_terms.get("vpd_next", 0.0) or 0.0),
        "rspc_action_selected_predicted_temp_next": float(
            selected_terms.get("predicted_temp_next", selected_terms.get("temp_next", 0.0)) or 0.0
        ),
        "rspc_action_selected_predicted_rh_next": float(
            selected_terms.get("predicted_rh_next", selected_terms.get("rh_next", 0.0)) or 0.0
        ),
        "rspc_action_selected_predicted_vpd_next": float(
            selected_terms.get("predicted_vpd_next", selected_terms.get("vpd_next", 0.0)) or 0.0
        ),
        "rspc_action_selected_predicted_dew_margin_air_next": float(
            selected_terms.get("predicted_dew_margin_air_next", 0.0) or 0.0
        ),
        "rspc_action_selected_predicted_canopy_dew_margin_next": float(
            selected_terms.get("predicted_canopy_dew_margin_next", 0.0) or 0.0
        ),
        "rspc_action_selected_predicted_dew_lt0": bool(selected_terms.get("predicted_dew_lt0", False)),
        "rspc_action_selected_predicted_canopy_lt0": bool(selected_terms.get("predicted_canopy_lt0", False)),
        "rspc_action_selected_prediction_schema_version": str(
            selected_terms.get("prediction_schema_version", "") or ""
        ),
        "rspc_action_selected_prediction_horizon_steps": int(
            float(selected_terms.get("prediction_horizon_steps", 0.0) or 0.0)
        ),
        "rspc_action_selected_prediction_model": str(selected_terms.get("prediction_model", "") or ""),
        "rspc_action_selected_prediction_model_v2": str(selected_terms.get("prediction_model_v2", "") or ""),
        "rspc_action_selected_predicted_canopy_dew_margin_next_v2": float(
            selected_terms.get(
                "predicted_canopy_dew_margin_next_v2",
                selected_terms.get("predicted_canopy_dew_margin_next", 0.0),
            )
            or 0.0
        ),
        "rspc_action_selected_predicted_canopy_lt0_v2": bool(
            selected_terms.get("predicted_canopy_lt0_v2", False)
        ),
        "rspc_action_selected_predicted_canopy_warning_v2": bool(
            selected_terms.get("predicted_canopy_warning_v2", False)
        ),
        "rspc_action_selected_canopy_proxy_v2_risk_buffer": float(
            selected_terms.get("canopy_proxy_v2_risk_buffer", 0.0) or 0.0
        ),
        "rspc_action_selected_canopy_proxy_history_available": bool(
            selected_terms.get("canopy_proxy_history_available", False)
        ),
        "rspc_action_selected_canopy_proxy_v2_risk_terms_json": _compact_json(
            selected_terms.get("canopy_proxy_v2_risk_terms", {})
        ),
        "rspc_action_selected_temp_penalty": float(selected_terms.get("temp_penalty", 0.0) or 0.0),
        "rspc_action_selected_dry_penalty": float(selected_terms.get("dry_penalty", 0.0) or 0.0),
        "rspc_action_selected_vpd_penalty": float(selected_terms.get("vpd_penalty", 0.0) or 0.0),
        "rspc_action_selected_dew_penalty": float(selected_terms.get("dew_penalty", 0.0) or 0.0),
        "rspc_action_selected_hot_dry_penalty": float(selected_terms.get("hot_dry_penalty", 0.0) or 0.0),
        "rspc_action_post_shape_enabled": bool(audit.get("post_shape_enabled", False)),
        "rspc_action_post_shape_best_name": str(audit.get("post_shape_best_name", "") or ""),
        "rspc_action_post_shape_best_score": float(audit.get("post_shape_best_score", 0.0) or 0.0),
        "rspc_action_post_shape_best_eligible": bool(audit.get("post_shape_best_eligible", False)),
        "rspc_action_post_shape_best_is_proposer": bool(audit.get("post_shape_best_is_proposer", False)),
        "rspc_action_post_shape_raw_best_name": str(audit.get("post_shape_raw_best_name", "") or ""),
        "rspc_action_post_shape_raw_best_score": float(audit.get("post_shape_raw_best_score", 0.0) or 0.0),
        "rspc_action_post_shape_raw_best_is_proposer": bool(audit.get("post_shape_raw_best_is_proposer", False)),
        "rspc_action_post_shape_raw_best_alignment": str(
            audit.get("post_shape_raw_best_alignment", "neutral_hold") or "neutral_hold"
        ),
        "rspc_action_post_shape_raw_best_margin": float(audit.get("post_shape_raw_best_margin", 0.0) or 0.0),
        "rspc_action_post_shape_raw_would_switch": bool(audit.get("post_shape_raw_would_switch", False)),
        "rspc_action_post_shape_raw_unsafe_conflict": bool(audit.get("post_shape_raw_unsafe_conflict", False)),
        "rspc_action_post_shape_selected_score": float(audit.get("post_shape_selected_score", 0.0) or 0.0),
        "rspc_action_post_shape_margin": float(audit.get("post_shape_margin", 0.0) or 0.0),
        "rspc_action_post_shape_would_switch": bool(audit.get("post_shape_would_switch", False)),
        "rspc_action_post_shape_safety_gate_reason": str(audit.get("post_shape_safety_gate_reason", "none") or "none"),
        "rspc_action_post_shape_alignment": str(audit.get("post_shape_alignment", "neutral_hold") or "neutral_hold"),
        "rspc_action_post_shape_unsafe_conflict": bool(audit.get("post_shape_unsafe_conflict", False)),
        "rspc_action_post_shape_dry_benefit": bool(audit.get("post_shape_dry_benefit", False)),
        "rspc_action_post_shape_safe_relief": bool(audit.get("post_shape_safe_relief", False)),
        "rspc_action_post_shape_selected_heat": float(post_shape_selected_action.get("heat", 0.0) or 0.0),
        "rspc_action_post_shape_selected_co2": float(post_shape_selected_action.get("co2", 0.0) or 0.0),
        "rspc_action_post_shape_selected_screen": float(post_shape_selected_action.get("screen", 0.0) or 0.0),
        "rspc_action_post_shape_selected_vent": float(post_shape_selected_action.get("vent", 0.0) or 0.0),
        "rspc_action_post_shape_selected_lamp": float(post_shape_selected_action.get("lamp", 0.0) or 0.0),
        "rspc_action_post_shape_selected_shade": float(post_shape_selected_action.get("shade", 0.0) or 0.0),
        "rspc_action_post_shape_best_heat": float(post_shape_best_action.get("heat", 0.0) or 0.0),
        "rspc_action_post_shape_best_co2": float(post_shape_best_action.get("co2", 0.0) or 0.0),
        "rspc_action_post_shape_best_screen": float(post_shape_best_action.get("screen", 0.0) or 0.0),
        "rspc_action_post_shape_best_vent": float(post_shape_best_action.get("vent", 0.0) or 0.0),
        "rspc_action_post_shape_best_lamp": float(post_shape_best_action.get("lamp", 0.0) or 0.0),
        "rspc_action_post_shape_best_shade": float(post_shape_best_action.get("shade", 0.0) or 0.0),
        "rspc_action_post_shape_delta_heat": float(post_shape_action_delta.get("heat", 0.0) or 0.0),
        "rspc_action_post_shape_delta_co2": float(post_shape_action_delta.get("co2", 0.0) or 0.0),
        "rspc_action_post_shape_delta_screen": float(post_shape_action_delta.get("screen", 0.0) or 0.0),
        "rspc_action_post_shape_delta_vent": float(post_shape_action_delta.get("vent", 0.0) or 0.0),
        "rspc_action_post_shape_delta_lamp": float(post_shape_action_delta.get("lamp", 0.0) or 0.0),
        "rspc_action_post_shape_delta_shade": float(post_shape_action_delta.get("shade", 0.0) or 0.0),
        "rspc_action_post_shape_delta_score_temp": float(post_shape_score_delta.get("temp", 0.0) or 0.0),
        "rspc_action_post_shape_delta_score_rh": float(post_shape_score_delta.get("rh", 0.0) or 0.0),
        "rspc_action_post_shape_delta_score_dry": float(post_shape_score_delta.get("dry", 0.0) or 0.0),
        "rspc_action_post_shape_delta_score_vpd": float(post_shape_score_delta.get("vpd", 0.0) or 0.0),
        "rspc_action_post_shape_delta_score_dew": float(post_shape_score_delta.get("dew", 0.0) or 0.0),
        "rspc_action_post_shape_delta_score_hot_dry": float(post_shape_score_delta.get("hot_dry", 0.0) or 0.0),
        "rspc_action_post_shape_delta_score_energy": float(post_shape_score_delta.get("energy", 0.0) or 0.0),
        "rspc_action_post_shape_best_dry_recovery_applied": bool(
            post_shape_best_events.get("dry_recovery_applied", False)
        ),
        "rspc_action_post_shape_best_tomato_safety_applied": bool(
            post_shape_best_events.get("tomato_safety_applied", False)
        ),
        "rspc_action_post_shape_best_dry_recovery_vent_cap": float(
            post_shape_best_events.get("dry_recovery_vent_cap", 0.0) or 0.0
        ),
        "rspc_action_post_shape_best_tomato_safety_reasons": ",".join(
            str(item)
            for item in post_shape_best_events.get("tomato_safety_reasons", [])
            if str(item)
        )
        if isinstance(post_shape_best_events.get("tomato_safety_reasons", []), list)
        else str(post_shape_best_events.get("tomato_safety_reasons", "") or ""),
        "rspc_tt_calibration_enabled": bool(tt_calibration.get("enabled", False)),
        "rspc_tt_calibration_shadow_only": bool(tt_calibration.get("shadow_only", False)),
        "rspc_tt_calibration_triggered": bool(tt_calibration.get("triggered", False)),
        "rspc_tt_calibration_reason": str(tt_calibration.get("reason", "") or ""),
        "rspc_tt_calibration_safety_gate_reason": str(tt_calibration.get("safety_gate_reason", "none") or "none"),
        "rspc_tt_calibration_safe_hot_dry": bool(tt_calibration.get("safe_hot_dry", False)),
        "rspc_tt_calibration_variant_count": int(tt_calibration.get("variant_count", len(tt_variants)) or 0),
        "rspc_tt_calibration_best_variant_name": str(tt_calibration.get("best_variant_name", "") or ""),
        "rspc_tt_calibration_best_variant_score": float(tt_calibration.get("best_variant_score", 0.0) or 0.0),
        "rspc_tt_calibration_best_improvement_margin": float(
            tt_calibration.get("best_improvement_margin", 0.0) or 0.0
        ),
        "rspc_tt_calibration_best_alignment": str(tt_calibration.get("best_alignment", "neutral_hold") or "neutral_hold"),
        "rspc_tt_calibration_would_improve": bool(tt_calibration.get("would_improve", False)),
        "rspc_tt_calibration_unsafe_conflict": bool(tt_calibration.get("unsafe_conflict", False)),
        "rspc_tt_calibration_has_unsafe_variant": bool(tt_calibration.get("has_unsafe_variant", False)),
        "rspc_tt_calibration_unsafe_variant_count": int(tt_calibration.get("unsafe_variant_count", 0) or 0),
        "rspc_tt_calibration_current_score": float(tt_calibration.get("current_score", 0.0) or 0.0),
        "rspc_tt_calibration_before_heat": float(tt_before_action.get("heat", 0.0) or 0.0),
        "rspc_tt_calibration_before_co2": float(tt_before_action.get("co2", 0.0) or 0.0),
        "rspc_tt_calibration_before_screen": float(tt_before_action.get("screen", 0.0) or 0.0),
        "rspc_tt_calibration_before_vent": float(tt_before_action.get("vent", 0.0) or 0.0),
        "rspc_tt_calibration_before_lamp": float(tt_before_action.get("lamp", 0.0) or 0.0),
        "rspc_tt_calibration_before_shade": float(tt_before_action.get("shade", 0.0) or 0.0),
        "rspc_tt_calibration_current_heat": float(tt_current_action.get("heat", 0.0) or 0.0),
        "rspc_tt_calibration_current_co2": float(tt_current_action.get("co2", 0.0) or 0.0),
        "rspc_tt_calibration_current_screen": float(tt_current_action.get("screen", 0.0) or 0.0),
        "rspc_tt_calibration_current_vent": float(tt_current_action.get("vent", 0.0) or 0.0),
        "rspc_tt_calibration_current_lamp": float(tt_current_action.get("lamp", 0.0) or 0.0),
        "rspc_tt_calibration_current_shade": float(tt_current_action.get("shade", 0.0) or 0.0),
        "rspc_tt_calibration_best_heat": float(tt_best_action.get("heat", 0.0) or 0.0),
        "rspc_tt_calibration_best_co2": float(tt_best_action.get("co2", 0.0) or 0.0),
        "rspc_tt_calibration_best_screen": float(tt_best_action.get("screen", 0.0) or 0.0),
        "rspc_tt_calibration_best_vent": float(tt_best_action.get("vent", 0.0) or 0.0),
        "rspc_tt_calibration_best_lamp": float(tt_best_action.get("lamp", 0.0) or 0.0),
        "rspc_tt_calibration_best_shade": float(tt_best_action.get("shade", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_heat": float(tt_best_action_delta.get("heat", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_co2": float(tt_best_action_delta.get("co2", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_screen": float(tt_best_action_delta.get("screen", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_vent": float(tt_best_action_delta.get("vent", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_lamp": float(tt_best_action_delta.get("lamp", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_shade": float(tt_best_action_delta.get("shade", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_score_temp": float(tt_best_score_delta.get("temp", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_score_rh": float(tt_best_score_delta.get("rh", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_score_dry": float(tt_best_score_delta.get("dry", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_score_vpd": float(tt_best_score_delta.get("vpd", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_score_dew": float(tt_best_score_delta.get("dew", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_score_hot_dry": float(tt_best_score_delta.get("hot_dry", 0.0) or 0.0),
        "rspc_tt_calibration_best_delta_score_energy": float(tt_best_score_delta.get("energy", 0.0) or 0.0),
        "rspc_tt_calibration_variants_json": _compact_json(tt_variants),
        "rspc_action_dry_recovery_applied": bool(dry_override.get("applied", False)),
        "rspc_action_dry_recovery_hot_dry_vent_relief": bool(dry_override.get("hot_dry_vent_relief", False)),
        "rspc_action_dry_recovery_before": _compact_json(dry_override.get("before", [])),
        "rspc_action_dry_recovery_after": _compact_json(dry_override.get("after", [])),
        "canopy_boundary_shadow_enabled": bool(canopy_boundary.get("enabled", False)),
        "canopy_boundary_shadow_active": bool(canopy_boundary.get("canopy_dew_boundary_active", False)),
        "canopy_boundary_shadow_would_reject": bool(
            canopy_boundary.get("would_reject_for_canopy_dew_boundary", False)
        ),
        "canopy_boundary_shadow_reason": str(canopy_boundary.get("boundary_reject_reason", "") or ""),
        "canopy_boundary_shadow_predicted_canopy_dew_margin_next": float(
            canopy_boundary.get("predicted_canopy_dew_margin_next", 0.0) or 0.0
        ),
        "canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2": float(
            canopy_boundary.get(
                "predicted_canopy_dew_margin_next_v2",
                canopy_boundary.get("predicted_canopy_dew_margin_next", 0.0),
            )
            or 0.0
        ),
        "canopy_boundary_shadow_warning_v2": bool(canopy_boundary.get("canopy_boundary_warning_v2", False)),
        "canopy_boundary_shadow_would_reject_v2": bool(
            canopy_boundary.get("would_reject_for_canopy_dew_boundary_v2", False)
        ),
        "canopy_boundary_shadow_reason_v2": str(canopy_boundary.get("boundary_reject_reason_v2", "") or ""),
        "canopy_boundary_shadow_prediction_model_v2": str(canopy_boundary.get("prediction_model_v2", "") or ""),
        "canopy_boundary_shadow_v2_risk_buffer": float(
            canopy_boundary.get("canopy_proxy_v2_risk_buffer", 0.0) or 0.0
        ),
        "canopy_boundary_shadow_v2_risk_terms_json": _compact_json(
            canopy_boundary.get("canopy_proxy_v2_risk_terms", {})
        ),
        "canopy_boundary_shadow_predicted_dew_margin_air_next": float(
            canopy_boundary.get("predicted_dew_margin_air_next", 0.0) or 0.0
        ),
        "canopy_boundary_shadow_screen_delta": float(canopy_boundary.get("screen_delta", 0.0) or 0.0),
        "canopy_boundary_shadow_vent_delta": float(canopy_boundary.get("ventilation_delta", 0.0) or 0.0),
        "canopy_boundary_shadow_screen_increase_under_canopy_risk": bool(
            canopy_boundary.get("screen_increase_under_canopy_risk", False)
        ),
        "canopy_boundary_shadow_ventilation_decrease_under_canopy_risk": bool(
            canopy_boundary.get("ventilation_decrease_under_canopy_risk", False)
        ),
        "canopy_boundary_shadow_high_screen_under_canopy_risk": bool(
            canopy_boundary.get("high_screen_under_canopy_risk", False)
        ),
        "canopy_boundary_shadow_low_ventilation_under_canopy_risk": bool(
            canopy_boundary.get("low_ventilation_under_canopy_risk", False)
        ),
        "final_action_predicted_canopy_lt0": bool(final_risk.get("final_action_predicted_canopy_lt0", False)),
        "final_action_predicted_canopy_lt0_v2": bool(
            final_risk.get("final_action_predicted_canopy_lt0_v2", False)
        ),
        "final_action_predicted_canopy_warning_v2": bool(
            final_risk.get("final_action_predicted_canopy_warning_v2", False)
        ),
        "final_action_would_fail_canopy_boundary": bool(
            final_risk.get("final_action_would_fail_canopy_boundary", False)
        ),
        "final_action_would_fail_canopy_boundary_v2": bool(
            final_risk.get("final_action_would_fail_canopy_boundary_v2", False)
        ),
        "final_action_screen_vent_conflict": bool(final_risk.get("final_action_screen_vent_conflict", False)),
        "final_action_screen_vent_conflict_v2": bool(
            final_risk.get("final_action_screen_vent_conflict_v2", False)
        ),
        "final_action_high_screen_under_canopy_risk": bool(
            final_risk.get("final_action_high_screen_under_canopy_risk", False)
        ),
        "final_action_low_ventilation_under_canopy_risk": bool(
            final_risk.get("final_action_low_ventilation_under_canopy_risk", False)
        ),
        "final_action_risk_reason": str(final_risk.get("final_action_risk_reason", "") or ""),
        "final_action_risk_reason_v2": str(final_risk.get("final_action_risk_reason_v2", "") or ""),
        "final_action_predicted_canopy_dew_margin_next_v2": float(
            final_risk.get(
                "predicted_canopy_dew_margin_next_v2",
                final_risk.get("predicted_canopy_dew_margin_next", 0.0),
            )
            or 0.0
        ),
        "final_action_canopy_proxy_v2_risk_buffer": float(
            final_risk.get("canopy_proxy_v2_risk_buffer", 0.0) or 0.0
        ),
        "rspc_action_candidates_json": _compact_json(candidates),
    }


def fallback_selection_to_record(plan: Any) -> Dict[str, Any]:
    if not isinstance(plan, dict):
        return {}
    fallback = plan.get("fallback_selection", {})
    if not isinstance(fallback, dict):
        fallback = {}
    candidate_scores = fallback.get("fallback_candidate_scores", [])
    if not isinstance(candidate_scores, list):
        candidate_scores = []
    selected = str(
        fallback.get("selected_fallback_candidate")
        or fallback.get("source")
        or ""
    )
    breakdown = fallback.get("selected_fallback_score_breakdown", {})
    if not isinstance(breakdown, dict):
        breakdown = {}
    return {
        "selected_fallback_candidate": selected,
        "selected_fallback_score": float(fallback.get("score", 0.0) or 0.0),
        "selected_fallback_score_breakdown": _compact_json(breakdown),
        "fallback_candidate_count": int(len(candidate_scores)),
        "fallback_candidate_scores": _compact_json(candidate_scores),
        "fallback_selected_total_score": float(breakdown.get("total_score", breakdown.get("score", 0.0)) or 0.0),
        "fallback_selected_temp_penalty": float(breakdown.get("temp_penalty", 0.0) or 0.0),
        "fallback_selected_rh_penalty": float(breakdown.get("rh_penalty", 0.0) or 0.0),
        "fallback_selected_dry_penalty": float(breakdown.get("dry_penalty", 0.0) or 0.0),
        "fallback_selected_vpd_high_penalty": float(
            breakdown.get("vpd_high_penalty", breakdown.get("vpd_penalty", 0.0)) or 0.0
        ),
        "fallback_selected_vpd_low_penalty": float(breakdown.get("vpd_low_penalty", 0.0) or 0.0),
        "fallback_selected_dew_penalty": float(breakdown.get("dew_penalty", 0.0) or 0.0),
        "fallback_selected_canopy_dew_penalty": float(breakdown.get("canopy_dew_penalty", 0.0) or 0.0),
        "fallback_selected_energy_penalty": float(breakdown.get("energy_penalty", 0.0) or 0.0),
        "fallback_selected_smooth_penalty": float(breakdown.get("smooth_penalty", 0.0) or 0.0),
        "fallback_selected_heat_vent_conflict_penalty": float(
            breakdown.get("heat_vent_conflict_penalty", 0.0) or 0.0
        ),
        "fallback_selected_co2_leak_penalty": float(breakdown.get("co2_leak_penalty", 0.0) or 0.0),
        "fallback_selected_dry_vent_penalty": float(breakdown.get("dry_vent_penalty", 0.0) or 0.0),
        "fallback_selected_forecast_risk_penalty": float(
            breakdown.get("forecast_risk_penalty", 0.0) or 0.0
        ),
    }


def run_ppo_trace(
    model: PPO,
    vecnorm_path: Path,
    raw_env: TomatoEnv,
    year: int,
    day: int,
    seed: int,
    max_steps: int,
) -> List[Dict[str, Any]]:
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    obs = vec_env.reset()
    interface = GreenhouseAgentInterface(raw_env)
    rows: List[Dict[str, Any]] = []
    done = False
    step = 0
    while (not done) and step < max_steps:
        state = interface.get_state()
        action_cmd, _ = model.predict(obs, deterministic=True)
        action_cmd_1d = np.asarray(action_cmd, dtype=np.float32).reshape(-1)
        planned_control = raw_env.action_to_control(action_cmd_1d)
        obs, rewards, dones, infos = vec_env.step(action_cmd)
        applied_control = np.asarray(getattr(raw_env, "u", planned_control), dtype=np.float32)
        row: Dict[str, Any] = {
            "algo": "ppo",
            "year": int(year),
            "day": int(day),
            "seed": int(seed),
            "step": int(step),
            "reward": float(rewards[0]),
            "done": bool(dones[0]),
            "source": "ppo",
            **state_to_record(state),
            **control_to_record(applied_control),
            **control_to_record(action_cmd_1d, prefix="action_cmd"),
            **info_to_metrics(dict(infos[0])),
        }
        rows.append(finalize_trace_row(row))
        done = bool(dones[0])
        step += 1
    vec_env.close()
    return rows


def run_llm_trace(
    config: Dict[str, Any],
    api_key: str,
    year: int,
    day: int,
    seed: int,
    max_steps: int,
    llm_model: str,
    llm_interval: int,
    llm_max_iterations: int,
    llm_max_tokens: int,
    expert_rollout: bool,
    expert_policy_path: str,
    expert_max_distance: float,
    humidity_memory_enabled: bool,
    humidity_memory_path: str,
    humidity_memory_max_distance: float,
    humidity_memory_min_trust: float,
    humidity_memory_teacher_policy_id: str,
    humidity_memory_baseline_controller_id: str,
    humidity_memory_version: str,
    mc_sero_mode: str,
    tomato_safety_v2_enabled: bool,
    plan_cache_mode: str,
    plan_cache_path: str,
    plan_cache_strict: bool,
    plan_cache_key_policy: str,
    uncertainty_scale: float,
    rspc_hot_dry_proposer_control_enabled: bool = False,
    rspc_hot_dry_proposer_control_min_margin: float = AgentConfig.rspc_hot_dry_proposer_control_min_margin,
    rspc_hot_dry_proposer_control_strict_enabled: bool = False,
    agent_config_overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    raw_env = build_env(config, year, day, seed, uncertainty_scale)
    interface = GreenhouseAgentInterface(raw_env)
    if hasattr(create_langchain_tools, "instance"):
        delattr(create_langchain_tools, "instance")
    tools = create_langchain_tools(interface)
    agent_cfg = AgentConfig(
        model_name=llm_model,
        api_key=api_key,
        verbose=False,
        control_interval=int(llm_interval),
        max_iterations=int(llm_max_iterations),
        max_tokens=int(llm_max_tokens),
        expert_rollout_enabled=bool(expert_rollout),
        expert_policy_path=expert_policy_path or AgentConfig.expert_policy_path,
        expert_candidate_max_distance=float(expert_max_distance),
        humidity_memory_enabled=bool(humidity_memory_enabled),
        humidity_memory_path=humidity_memory_path or AgentConfig.humidity_memory_path,
        humidity_memory_max_distance=float(humidity_memory_max_distance),
        humidity_memory_min_trust=float(humidity_memory_min_trust),
        humidity_memory_teacher_policy_id=str(humidity_memory_teacher_policy_id or ""),
        humidity_memory_baseline_controller_id=str(humidity_memory_baseline_controller_id or ""),
        humidity_memory_version=str(humidity_memory_version or AgentConfig.humidity_memory_version),
        mc_sero_mode=str(mc_sero_mode or AgentConfig.mc_sero_mode),
        tomato_safety_v2_enabled=bool(tomato_safety_v2_enabled),
        plan_cache_mode=str(plan_cache_mode or "off"),
        plan_cache_path=str(plan_cache_path or AgentConfig.plan_cache_path),
        plan_cache_strict=bool(plan_cache_strict),
        plan_cache_key_policy=str(plan_cache_key_policy or AgentConfig.plan_cache_key_policy),
        rspc_hot_dry_proposer_control_enabled=bool(rspc_hot_dry_proposer_control_enabled),
        rspc_hot_dry_proposer_control_min_margin=float(rspc_hot_dry_proposer_control_min_margin),
        rspc_hot_dry_proposer_control_strict_enabled=bool(rspc_hot_dry_proposer_control_strict_enabled),
    )
    for key, value in dict(agent_config_overrides or {}).items():
        if not hasattr(agent_cfg, str(key)):
            raise ValueError(f"Unknown AgentConfig override: {key}")
        setattr(agent_cfg, str(key), value)
    agent = RuleBasedLLMDirector(
        agent_interface=interface,
        tools=tools,
        config=agent_cfg,
        env_id=f"TomatoEnv_y{int(year)}_d{int(day)}_s{int(seed)}",
        rule_params=dict(DEFAULT_RULE_PARAMS),
    )
    raw_env.reset(seed=seed)
    rows: List[Dict[str, Any]] = []
    done = False
    step = 0
    while (not done) and step < max_steps:
        state = interface.get_state()
        try:
            result = agent.step_with_rules()
        except Exception as exc:
            plan_cache_event = dict(getattr(agent, "last_plan_cache_event", {}) or {})
            rollout_selection = dict(getattr(agent, "last_rollout_selection", {}) or {})
            plan = agent.current_plan if isinstance(getattr(agent, "current_plan", None), dict) else {}
            if not plan_cache_event and isinstance(plan, dict):
                plan_cache_event = dict(plan.get("plan_cache_event", {}) or {})
            applied_control = getattr(raw_env, "u", np.zeros(6, dtype=np.float32))
            try:
                info = raw_env._get_info()
            except Exception:
                info = {}
            row = {
                "algo": "llm_director",
                "model_name": str(getattr(agent.config, "model_name", llm_model)),
                "year": int(year),
                "day": int(day),
                "seed": int(seed),
                "step": int(step),
                "reward": 0.0,
                "done": True,
                "source": "runtime_error",
                "replan_reason": "runtime_error",
                "runtime_error": str(exc),
                "runtime_error_type": (
                    "strict_plan_cache_miss"
                    if "plan cache miss" in str(exc).lower()
                    else "simulator_or_controller_error"
                ),
                "plan_cache_mode": plan_cache_event.get("mode") if isinstance(plan_cache_event, dict) else None,
                "plan_cache_enabled": bool(plan_cache_event.get("enabled", False)) if isinstance(plan_cache_event, dict) else False,
                "plan_cache_hit": bool(plan_cache_event.get("hit", False)) if isinstance(plan_cache_event, dict) else False,
                "plan_cache_status": plan_cache_event.get("status") if isinstance(plan_cache_event, dict) else None,
                "plan_cache_key": plan_cache_event.get("key") if isinstance(plan_cache_event, dict) else None,
                "plan_cache_attempt": int(plan_cache_event.get("attempt", 0) or 0) if isinstance(plan_cache_event, dict) else 0,
                "plan_cache_key_policy": plan_cache_event.get("key_policy") if isinstance(plan_cache_event, dict) else None,
                "anchor_source": plan.get("anchor_source") if isinstance(plan, dict) else None,
                "target_temp": plan.get("target_temp") if isinstance(plan, dict) else None,
                "target_co2": plan.get("target_co2") if isinstance(plan, dict) else None,
                "target_rh": plan.get("target_rh") if isinstance(plan, dict) else None,
                **profile_generator_shadow_to_record(plan, step, state),
                **profile_rspc_shadow_to_record({"rollout_selection": rollout_selection}),
                **profile_action_candidate_shadow_to_record({"rollout_selection": rollout_selection}),
                **profile_action_envelope_shadow_to_record({"rollout_selection": rollout_selection}),
                **cstcc_shadow_to_record({"rollout_selection": rollout_selection}),
                **rspc_action_scoring_to_record({"rollout_selection": rollout_selection}),
                **fallback_selection_to_record(plan),
                "rollout_source_before_error": rollout_selection.get("source"),
                **transition_gate_to_record(getattr(agent, "last_transition_gate", {})),
                **(
                    profile_feasibility_gate_to_record(getattr(agent, "last_profile_feasibility_gate", {}))
                    if bool(getattr(getattr(agent, "config", object()), "profile_feasibility_gate_enabled", False))
                    else {}
                ),
                **profile_template_patch_to_record(getattr(agent, "last_profile_template_patch", {})),
                **structured_anchor_to_record(getattr(agent, "last_structured_anchor", {})),
                **structured_anchor_profile_bridge_to_record(getattr(agent, "last_structured_anchor_profile_bridge", {})),
                **state_to_record(state),
                **control_to_record(applied_control),
                **info_to_metrics(info),
            }
            rows.append(finalize_trace_row(row))
            break
        info = raw_env._get_info()
        plan = result.get("plan", {}) if isinstance(result, dict) else {}
        plan_cache_event = plan.get("plan_cache_event", {}) if isinstance(plan, dict) else {}
        rollout = plan.get("rollout_selection", {}) if isinstance(plan, dict) else {}
        expert_info = rollout.get("expert_prediction", {}) if isinstance(rollout, dict) else {}
        humidity_memory_info = rollout.get("humidity_memory_prediction", {}) if isinstance(rollout, dict) else {}
        humidity_memory_shape = rollout.get("humidity_memory_post_guardrail_shape", {}) if isinstance(rollout, dict) else {}
        mc_sero_info = rollout.get("mc_sero_shadow", {}) if isinstance(rollout, dict) else {}
        tomato_v2_info = rollout.get("tomato_safety_v2", {}) if isinstance(rollout, dict) else {}
        tomato_v2_suppressed = tomato_v2_info.get("suppressed_replan", {}) if isinstance(tomato_v2_info, dict) else {}
        rollout_source = str(rollout.get("source", result.get("action", "unknown")))
        applied_control = result.get("applied_control", getattr(raw_env, "u", np.zeros(6)))
        row = {
            "algo": "llm_director",
            "model_name": str(getattr(agent.config, "model_name", llm_model)),
            "year": int(year),
            "day": int(day),
            "seed": int(seed),
            "step": int(step),
            "reward": float(result.get("reward", 0.0)),
            "done": bool(result.get("done", False)),
            "source": rollout_source,
            "replan_reason": result.get("replan_reason"),
            "anchor_source": plan.get("anchor_source") if isinstance(plan, dict) else None,
            "target_temp": plan.get("current_target_temp") if isinstance(plan, dict) else None,
            "target_co2": plan.get("current_target_co2") if isinstance(plan, dict) else None,
            "target_rh": plan.get("current_target_rh") if isinstance(plan, dict) else None,
            **profile_generator_shadow_to_record(plan, step, state),
            **profile_rspc_shadow_to_record(plan),
            **profile_action_candidate_shadow_to_record(plan),
            **profile_action_envelope_shadow_to_record(plan),
            **cstcc_shadow_to_record(result),
            **cstcc_shadow_to_record(plan),
            **rspc_action_scoring_to_record(plan),
            **fallback_selection_to_record(plan),
            "plan_cache_mode": plan_cache_event.get("mode") if isinstance(plan_cache_event, dict) else None,
            "plan_cache_enabled": bool(plan_cache_event.get("enabled", False)) if isinstance(plan_cache_event, dict) else False,
            "plan_cache_hit": bool(plan_cache_event.get("hit", False)) if isinstance(plan_cache_event, dict) else False,
            "plan_cache_status": plan_cache_event.get("status") if isinstance(plan_cache_event, dict) else None,
            "plan_cache_key": plan_cache_event.get("key") if isinstance(plan_cache_event, dict) else None,
            "plan_cache_attempt": int(plan_cache_event.get("attempt", 0) or 0) if isinstance(plan_cache_event, dict) else 0,
            "plan_cache_key_policy": plan_cache_event.get("key_policy") if isinstance(plan_cache_event, dict) else None,
            "rh_violation_debt": plan.get("rh_violation_debt", 0.0) if isinstance(plan, dict) else 0.0,
            "expert_available": bool(expert_info.get("available", False)) if isinstance(expert_info, dict) else False,
            "expert_accepted": bool(expert_info.get("accepted", False)) if isinstance(expert_info, dict) else False,
            "expert_feature_distance": float(expert_info.get("feature_distance", 0.0)) if isinstance(expert_info, dict) else 0.0,
            "humidity_memory_available": bool(humidity_memory_info.get("available", False)) if isinstance(humidity_memory_info, dict) else False,
            "humidity_memory_accepted": bool(humidity_memory_info.get("accepted", False)) if isinstance(humidity_memory_info, dict) else False,
            "humidity_memory_selected": rollout_source.startswith("humidity_memory"),
            "humidity_memory_case_id": humidity_memory_info.get("case_id") if isinstance(humidity_memory_info, dict) else None,
            "humidity_memory_distance": float(humidity_memory_info.get("distance", 0.0)) if isinstance(humidity_memory_info, dict) else 0.0,
            "humidity_memory_trust": float(humidity_memory_info.get("trust", 0.0)) if isinstance(humidity_memory_info, dict) else 0.0,
            "humidity_memory_support_count": int(humidity_memory_info.get("support_count", 0)) if isinstance(humidity_memory_info, dict) else 0,
            "humidity_memory_strategy_label": humidity_memory_info.get("strategy_label") if isinstance(humidity_memory_info, dict) else None,
            "humidity_memory_strategy_confidence": float(humidity_memory_info.get("strategy_confidence", 0.0)) if isinstance(humidity_memory_info, dict) else 0.0,
            "humidity_memory_reject_reason": humidity_memory_info.get("reject_reason") if isinstance(humidity_memory_info, dict) else None,
            "humidity_memory_horizon_penalty": float(humidity_memory_info.get("horizon_penalty", 0.0)) if isinstance(humidity_memory_info, dict) else 0.0,
            "humidity_memory_direct_saving": float(humidity_memory_info.get("direct_saving", 0.0)) if isinstance(humidity_memory_info, dict) else 0.0,
            "humidity_memory_recovery_heat_cost": float(humidity_memory_info.get("recovery_heat_cost", 0.0)) if isinstance(humidity_memory_info, dict) else 0.0,
            "humidity_memory_rh_debt_penalty": float(humidity_memory_info.get("rh_debt_penalty", 0.0)) if isinstance(humidity_memory_info, dict) else 0.0,
            "humidity_memory_best_non_hem_score": float(humidity_memory_info.get("best_non_hem_score", 0.0) or 0.0) if isinstance(humidity_memory_info, dict) else 0.0,
            "humidity_memory_horizon_adjusted_score": float(humidity_memory_info.get("horizon_adjusted_score", 0.0)) if isinstance(humidity_memory_info, dict) else 0.0,
            "humidity_memory_horizon_filter_rejected": bool(humidity_memory_info.get("horizon_filter_rejected", False)) if isinstance(humidity_memory_info, dict) else False,
            "humidity_memory_post_shape_applied": bool(humidity_memory_shape.get("applied", False)) if isinstance(humidity_memory_shape, dict) else False,
            "humidity_memory_post_shape_reason": humidity_memory_shape.get("reason") if isinstance(humidity_memory_shape, dict) else None,
            "mc_sero_enabled": bool(mc_sero_info.get("enabled", False)) if isinstance(mc_sero_info, dict) else False,
            "mc_sero_available": bool(mc_sero_info.get("available", False)) if isinstance(mc_sero_info, dict) else False,
            "mc_sero_would_select": bool(mc_sero_info.get("would_select", False)) if isinstance(mc_sero_info, dict) else False,
            "mc_sero_best_candidate": mc_sero_info.get("best_candidate") if isinstance(mc_sero_info, dict) else None,
            "mc_sero_best_control": mc_sero_info.get("best_control", []) if isinstance(mc_sero_info, dict) else [],
            "mc_sero_margin": float(mc_sero_info.get("margin", 0.0)) if isinstance(mc_sero_info, dict) else 0.0,
            "mc_sero_reject_reason": mc_sero_info.get("reject_reason") if isinstance(mc_sero_info, dict) else None,
            "mc_sero_baseline_source": mc_sero_info.get("baseline_source") if isinstance(mc_sero_info, dict) else None,
            "mc_sero_score_terms": mc_sero_info.get("score_terms", {}) if isinstance(mc_sero_info, dict) else {},
            "mc_sero_candidate_count": int(mc_sero_info.get("candidate_count", 0) or 0) if isinstance(mc_sero_info, dict) else 0,
            "mc_sero_horizon_steps": int(mc_sero_info.get("horizon_steps", 0) or 0) if isinstance(mc_sero_info, dict) else 0,
            "tomato_safety_v2_enabled": bool(tomato_v2_info.get("enabled", False)) if isinstance(tomato_v2_info, dict) else False,
            "tomato_safety_v2_applied": bool(tomato_v2_info.get("applied", False)) if isinstance(tomato_v2_info, dict) else False,
            "tomato_safety_v2_reasons": ",".join(str(x) for x in tomato_v2_info.get("reasons", [])) if isinstance(tomato_v2_info, dict) else "",
            "tomato_safety_v2_before": _compact_json(tomato_v2_info.get("before", [])) if isinstance(tomato_v2_info, dict) else "[]",
            "tomato_safety_v2_after": _compact_json(tomato_v2_info.get("after", [])) if isinstance(tomato_v2_info, dict) else "[]",
            "tomato_safety_v2_vent_before": float(tomato_v2_info.get("vent_before", 0.0)) if isinstance(tomato_v2_info, dict) else 0.0,
            "tomato_safety_v2_vent_after": float(tomato_v2_info.get("vent_after", 0.0)) if isinstance(tomato_v2_info, dict) else 0.0,
            "tomato_safety_v2_heat_before": float(tomato_v2_info.get("heat_before", 0.0)) if isinstance(tomato_v2_info, dict) else 0.0,
            "tomato_safety_v2_heat_after": float(tomato_v2_info.get("heat_after", 0.0)) if isinstance(tomato_v2_info, dict) else 0.0,
            "tomato_safety_v2_screen_before": float(tomato_v2_info.get("screen_before", 0.0)) if isinstance(tomato_v2_info, dict) else 0.0,
            "tomato_safety_v2_screen_after": float(tomato_v2_info.get("screen_after", 0.0)) if isinstance(tomato_v2_info, dict) else 0.0,
            "tomato_safety_v2_shade_before": float(tomato_v2_info.get("shade_before", 0.0)) if isinstance(tomato_v2_info, dict) else 0.0,
            "tomato_safety_v2_shade_after": float(tomato_v2_info.get("shade_after", 0.0)) if isinstance(tomato_v2_info, dict) else 0.0,
            "tomato_safety_v2_suppressed_replan": bool(tomato_v2_suppressed.get("applied", False)) if isinstance(tomato_v2_suppressed, dict) else False,
            "tomato_safety_v2_suppressed_replan_reason": str(tomato_v2_suppressed.get("reason", "")) if isinstance(tomato_v2_suppressed, dict) else "",
            "tomato_safety_v2_suppressed_replan_count": int(tomato_v2_info.get("suppressed_replan_step_count", 0) or 0) if isinstance(tomato_v2_info, dict) else 0,
            **post_guardrail_runtime_provenance_to_record(rollout),
            **transition_gate_to_record(result.get("transition_gate", {})),
            **(
                profile_feasibility_gate_to_record(result.get("profile_feasibility_gate", {}))
                if "profile_feasibility_gate" in result
                else {}
            ),
            **profile_template_patch_to_record(result.get("profile_template_patch", {})),
            **structured_anchor_to_record(result.get("structured_anchor", plan.get("structured_anchor", {}))),
            **structured_anchor_profile_bridge_to_record(
                result.get(
                    "structured_anchor_profile_bridge",
                    plan.get("structured_anchor_profile_bridge", {}),
                )
            ),
            **state_to_record(state),
            **control_to_record(applied_control),
            **control_to_record(result.get("anchor_control", np.zeros(6)), prefix="anchor"),
            **control_to_record(result.get("rule_control", np.zeros(6)), prefix="rule"),
            **info_to_metrics(info),
        }
        rows.append(finalize_trace_row(row))
        done = bool(result.get("done", False))
        step += 1
    return rows


def sum_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "steps": len(rows),
        "total_reward": float(sum(float(r.get("reward", 0.0)) for r in rows)),
    }
    for metric in METRIC_MAP:
        summary[f"total_{metric}"] = float(sum(float(r.get(metric, 0.0)) for r in rows))
    if rows:
        for action in ACTION_NAMES:
            summary[f"mean_u_{action}"] = float(mean(float(r.get(f"u_{action}", 0.0)) for r in rows))
        summary["mean_rh"] = float(mean(float(r.get("rh_air", 0.0)) for r in rows))
        summary["mean_temp"] = float(mean(float(r.get("temp_air", 0.0)) for r in rows))
        summary["mean_vpd"] = float(mean(float(r.get("vpd_air", 0.0)) for r in rows))
        summary["total_rh_low_violation"] = float(sum(float(r.get("rh_low_violation", 0.0)) for r in rows))
        summary["total_rh_high_violation"] = float(sum(float(r.get("rh_high_violation", 0.0)) for r in rows))
        summary["total_vpd_low_excess"] = float(sum(float(r.get("vpd_low_excess", 0.0)) for r in rows))
        summary["total_vpd_high_excess"] = float(sum(float(r.get("vpd_high_excess", 0.0)) for r in rows))
        summary["dry_risk_steps"] = int(sum(bool(r.get("dry_risk", False)) for r in rows))
        summary["dew_risk_steps"] = int(sum(bool(r.get("dew_risk", False)) for r in rows))
        summary["rh_ge_90_steps"] = int(sum(float(r.get("rh_air", 0.0)) >= 90.0 for r in rows))
        summary["rh_ge_94_steps"] = int(sum(float(r.get("rh_air", 0.0)) >= 94.0 for r in rows))
        summary["dew_margin_air_lt1_steps"] = int(sum(float(r.get("dew_margin_air", 99.0)) < 1.0 for r in rows))
        summary["dew_margin_air_lt0_steps"] = int(sum(float(r.get("dew_margin_air", 99.0)) < 0.0 for r in rows))
        summary["canopy_dew_margin_lt1_steps"] = int(sum(float(r.get("canopy_dew_margin", 99.0)) < 1.0 for r in rows))
        summary["canopy_dew_margin_lt0_steps"] = int(sum(float(r.get("canopy_dew_margin", 99.0)) < 0.0 for r in rows))
        summary["min_dew_margin_air"] = float(min(float(r.get("dew_margin_air", 99.0)) for r in rows))
        summary["min_canopy_dew_margin"] = float(min(float(r.get("canopy_dew_margin", 99.0)) for r in rows))
        summary["cold_vent_risk_steps"] = int(sum(bool(r.get("cold_vent_risk", False)) for r in rows))
        summary["dry_vent_risk_steps"] = int(sum(bool(r.get("dry_vent_risk", False)) for r in rows))
        source_counts: Dict[str, int] = {}
        for row in rows:
            source = str(row.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
        summary["source_counts"] = dict(sorted(source_counts.items()))
        runtime_error_rows = [r for r in rows if bool(r.get("runtime_error"))]
        summary["runtime_error_steps"] = int(len(runtime_error_rows))
        summary["strict_cache_miss_runtime_error_steps"] = int(
            sum(str(r.get("runtime_error_type", "")) == "strict_plan_cache_miss" for r in runtime_error_rows)
        )
        if any("post_guardrail_runtime_provenance_count" in r for r in rows):
            summary["post_guardrail_runtime_provenance_record_count"] = int(
                sum(int(r.get("post_guardrail_runtime_provenance_count", 0) or 0) for r in rows)
            )
            summary["post_guardrail_runtime_reason_missing_count"] = int(
                sum(int(r.get("post_guardrail_runtime_reason_missing_count", 0) or 0) for r in rows)
            )
            summary["unknown_post_guardrail_rewrite_count"] = int(
                sum(int(r.get("unknown_post_guardrail_rewrite_count", 0) or 0) for r in rows)
            )
        if any("cstcc_shadow_enabled" in r for r in rows):
            cstcc_rows = [r for r in rows if "cstcc_shadow_enabled" in r]
            cstcc_enabled_rows = [r for r in cstcc_rows if bool(r.get("cstcc_shadow_enabled", False))]
            cstcc_success_rows = [r for r in cstcc_enabled_rows if bool(r.get("cstcc_shadow_audit_success", False))]
            summary["cstcc_shadow_enabled_steps"] = int(len(cstcc_enabled_rows))
            summary["cstcc_shadow_audit_success_rate"] = float(
                len(cstcc_success_rows) / max(len(cstcc_enabled_rows), 1)
            )
            summary["cstcc_shadow_final_action_invariant_rate"] = float(
                sum(bool(r.get("cstcc_shadow_final_action_invariant_verified", False)) for r in cstcc_enabled_rows)
                / max(len(cstcc_enabled_rows), 1)
            )
            summary["cstcc_shadow_final_action_changed_steps"] = int(
                sum(bool(r.get("cstcc_shadow_final_action_changed", False)) for r in cstcc_enabled_rows)
            )
            summary["cstcc_shadow_online_llm_called_steps"] = int(
                sum(bool(r.get("cstcc_shadow_online_llm_called", False)) for r in cstcc_enabled_rows)
            )
            summary["cstcc_shadow_predictive_rollout_executed_steps"] = int(
                sum(bool(r.get("cstcc_shadow_predictive_rollout_executed", False)) for r in cstcc_enabled_rows)
            )
            summary["cstcc_shadow_real_tomato_safety_projection_steps"] = int(
                sum(bool(r.get("cstcc_shadow_real_tomato_safety_projection", False)) for r in cstcc_enabled_rows)
            )
            summary["cstcc_shadow_fallback_rate"] = float(
                sum(bool(r.get("cstcc_shadow_fallback_triggered", False)) for r in cstcc_success_rows)
                / max(len(cstcc_success_rows), 1)
            )
            summary["cstcc_shadow_mean_candidate_count"] = float(
                mean(int(r.get("cstcc_shadow_candidate_count", 0) or 0) for r in cstcc_success_rows)
            ) if cstcc_success_rows else 0.0
            summary["cstcc_shadow_mean_feasible_candidate_count"] = float(
                mean(int(r.get("cstcc_shadow_feasible_candidate_count", 0) or 0) for r in cstcc_success_rows)
            ) if cstcc_success_rows else 0.0
            summary["cstcc_shadow_mean_infeasible_candidate_ratio"] = float(
                mean(float(r.get("cstcc_shadow_infeasible_candidate_ratio", 0.0) or 0.0) for r in cstcc_success_rows)
            ) if cstcc_success_rows else 0.0
            summary["cstcc_shadow_hard_constraint_violation_count"] = int(
                sum(int(r.get("cstcc_shadow_hard_constraint_violation_count", 0) or 0) for r in cstcc_success_rows)
            )
            violation_counts: Dict[str, int] = {}
            violation_field_counts: Dict[str, int] = {}
            selected_violation_counts: Dict[str, int] = {}
            selected_violation_field_counts: Dict[str, int] = {}
            attribution_mode_counts: Dict[str, int] = {}
            selected_source_prior_counts: Dict[str, int] = {}
            selected_template_counts: Dict[str, int] = {}
            selected_candidate_violation_count = 0
            rule_conservative_status_counts: Dict[str, int] = {}
            conservative_no_candidate_reason_counts: Dict[str, int] = {}
            rule_prior_margin_values: List[float] = []
            conservative_prior_margin_values: List[float] = []
            conservative_risk_candidate_steps = 0
            conservative_risk_feasible_steps = 0
            conservative_risk_no_candidate_count = 0
            conservative_normal_candidate_steps = 0
            conservative_floor_applied_count = 0
            conservative_floor_reason_counts: Dict[str, int] = {}
            cstcc_action_fields = (
                "u_heating",
                "u_co2",
                "u_screen",
                "u_ventilation",
                "u_lighting",
                "u_shading",
            )
            previous_shift_candidate_count = 0
            previous_shift_max_delta_count = 0
            previous_shift_selected_count = 0
            previous_shift_projection_applied_count = 0
            previous_shift_first_delta_after_projection_max = 0.0
            previous_shift_projection_abs_sums: Dict[str, float] = {field: 0.0 for field in cstcc_action_fields}
            previous_shift_internal_max_delta: Dict[str, float] = {field: 0.0 for field in cstcc_action_fields}
            for row in cstcc_success_rows:
                attribution_mode = str(
                    row.get("cstcc_shadow_candidate_attribution_mode")
                    or "selected_step_coincidence_attribution"
                )
                attribution_mode_counts[attribution_mode] = attribution_mode_counts.get(attribution_mode, 0) + 1
                selected_candidate_violation_count += int(
                    row.get("cstcc_shadow_selected_candidate_violation_count", 0) or 0
                )
                previous_shift_candidate_count += int(
                    row.get("cstcc_shadow_previous_shift_all_candidate_count", 0) or 0
                )
                previous_shift_max_delta_count += int(
                    row.get("cstcc_shadow_previous_shift_all_max_delta_count", 0) or 0
                )
                previous_shift_selected_count += int(
                    row.get("cstcc_shadow_previous_shift_all_selected_count", 0) or 0
                )
                applied_count = int(row.get("cstcc_shadow_previous_shift_projection_applied_count", 0) or 0)
                previous_shift_projection_applied_count += applied_count
                previous_shift_first_delta_after_projection_max = max(
                    previous_shift_first_delta_after_projection_max,
                    abs(float(row.get("cstcc_shadow_previous_shift_first_delta_after_projection_max", 0.0) or 0.0)),
                )
                try:
                    previous_shift_projection_delta = json.loads(
                        str(row.get("cstcc_shadow_previous_shift_projection_mean_abs_delta_by_field_json") or "{}")
                    )
                except Exception:
                    previous_shift_projection_delta = {}
                if isinstance(previous_shift_projection_delta, dict):
                    for field in cstcc_action_fields:
                        previous_shift_projection_abs_sums[field] += (
                            abs(float(previous_shift_projection_delta.get(field, 0.0) or 0.0))
                            * max(applied_count, 1)
                        )
                try:
                    previous_shift_internal = json.loads(
                        str(row.get("cstcc_shadow_previous_shift_internal_max_delta_after_projection_by_field_json") or "{}")
                    )
                except Exception:
                    previous_shift_internal = {}
                if isinstance(previous_shift_internal, dict):
                    for field in cstcc_action_fields:
                        previous_shift_internal_max_delta[field] = max(
                            previous_shift_internal_max_delta[field],
                            abs(float(previous_shift_internal.get(field, 0.0) or 0.0)),
                        )
                try:
                    reasons = json.loads(
                        str(row.get("cstcc_shadow_hard_constraint_violation_reason_distribution_json") or "{}")
                    )
                except Exception:
                    reasons = {}
                if not isinstance(reasons, dict):
                    reasons = {}
                for reason, count in reasons.items():
                    violation_counts[str(reason)] = violation_counts.get(str(reason), 0) + int(count or 0)
                try:
                    fields = json.loads(
                        str(row.get("cstcc_shadow_hard_constraint_violation_field_distribution_json") or "{}")
                    )
                except Exception:
                    fields = {}
                if isinstance(fields, dict):
                    for field, count in fields.items():
                        violation_field_counts[str(field)] = violation_field_counts.get(str(field), 0) + int(count or 0)
                try:
                    selected_reasons = json.loads(
                        str(row.get("cstcc_shadow_selected_candidate_violation_reason_distribution_json") or "{}")
                    )
                except Exception:
                    selected_reasons = {}
                if isinstance(selected_reasons, dict):
                    for reason, count in selected_reasons.items():
                        selected_violation_counts[str(reason)] = selected_violation_counts.get(str(reason), 0) + int(count or 0)
                try:
                    selected_fields = json.loads(
                        str(row.get("cstcc_shadow_selected_candidate_violation_field_distribution_json") or "{}")
                    )
                except Exception:
                    selected_fields = {}
                if isinstance(selected_fields, dict):
                    for field, count in selected_fields.items():
                        selected_violation_field_counts[str(field)] = selected_violation_field_counts.get(str(field), 0) + int(count or 0)
                try:
                    competitiveness = json.loads(
                        str(row.get("cstcc_shadow_rule_conservative_competitiveness_json") or "{}")
                    )
                except Exception:
                    competitiveness = {}
                if isinstance(competitiveness, dict):
                    for source in ("rule_prior", "conservative_prior"):
                        status = str((competitiveness.get(source) or {}).get("status") or "missing")
                        key = f"{source}:{status}"
                        rule_conservative_status_counts[key] = rule_conservative_status_counts.get(key, 0) + 1
                conservative_reason = str(row.get("cstcc_shadow_conservative_prior_no_candidate_reason") or "")
                if conservative_reason:
                    conservative_no_candidate_reason_counts[conservative_reason] = (
                        conservative_no_candidate_reason_counts.get(conservative_reason, 0) + 1
                    )
                try:
                    risk_flags = json.loads(str(row.get("cstcc_shadow_risk_flags_json") or "{}"))
                except Exception:
                    risk_flags = {}
                risk_active = bool(isinstance(risk_flags, dict) and risk_flags.get("any_risk", False))
                conservative_generated = int(
                    row.get("cstcc_shadow_conservative_prior_generated_template_count", 0) or 0
                )
                conservative_status = str(row.get("cstcc_shadow_conservative_prior_margin_status") or "")
                if risk_active and conservative_generated > 0:
                    conservative_risk_candidate_steps += 1
                if risk_active and conservative_status in {"selected", "lower_score"}:
                    conservative_risk_feasible_steps += 1
                if risk_active and conservative_reason:
                    conservative_risk_no_candidate_count += 1
                if (not risk_active) and conservative_generated > 0:
                    conservative_normal_candidate_steps += 1
                if bool(row.get("cstcc_shadow_conservative_prior_candidate_count_override_applied", False)):
                    conservative_floor_applied_count += 1
                    reason = str(
                        row.get("cstcc_shadow_conservative_prior_candidate_count_override_reason")
                        or "unknown"
                    )
                    conservative_floor_reason_counts[reason] = conservative_floor_reason_counts.get(reason, 0) + 1
                conservative_margin = row.get("cstcc_shadow_conservative_prior_selected_score_margin")
                try:
                    if conservative_margin != "":
                        conservative_prior_margin_values.append(float(conservative_margin))
                except Exception:
                    pass
                rule_margin = row.get("cstcc_shadow_rule_prior_selected_score_margin")
                try:
                    if rule_margin != "":
                        rule_prior_margin_values.append(float(rule_margin))
                except Exception:
                    pass
                source_prior = str(row.get("cstcc_shadow_selected_source_prior") or "")
                template_name = str(row.get("cstcc_shadow_selected_template_name") or "")
                if (not source_prior) or (not template_name):
                    parsed_source_prior, parsed_template_name = parse_selected_sequence_id(
                        row.get("cstcc_shadow_selected_sequence_id")
                    )
                    source_prior = source_prior or parsed_source_prior
                    template_name = template_name or parsed_template_name
                if source_prior:
                    selected_source_prior_counts[source_prior] = selected_source_prior_counts.get(source_prior, 0) + 1
                if template_name:
                    selected_template_counts[template_name] = selected_template_counts.get(template_name, 0) + 1
            summary["cstcc_shadow_hard_constraint_violation_reason_distribution"] = dict(
                sorted(violation_counts.items())
            )
            summary["cstcc_shadow_hard_constraint_violation_field_distribution"] = dict(
                sorted(violation_field_counts.items())
            )
            summary["cstcc_shadow_candidate_attribution_mode_distribution"] = dict(
                sorted(attribution_mode_counts.items())
            )
            summary["cstcc_shadow_selected_candidate_violation_count"] = int(
                selected_candidate_violation_count
            )
            summary["cstcc_shadow_selected_candidate_violation_reason_distribution"] = dict(
                sorted(selected_violation_counts.items())
            )
            summary["cstcc_shadow_selected_candidate_violation_field_distribution"] = dict(
                sorted(selected_violation_field_counts.items())
            )
            summary["cstcc_shadow_rule_conservative_status_distribution"] = dict(
                sorted(rule_conservative_status_counts.items())
            )
            summary["cstcc_shadow_conservative_no_candidate_reason_distribution"] = dict(
                sorted(conservative_no_candidate_reason_counts.items())
            )
            summary["cstcc_shadow_conservative_risk_candidate_steps"] = int(conservative_risk_candidate_steps)
            summary["cstcc_shadow_conservative_risk_feasible_steps"] = int(conservative_risk_feasible_steps)
            summary["cstcc_shadow_conservative_risk_no_candidate_count"] = int(conservative_risk_no_candidate_count)
            summary["cstcc_shadow_conservative_normal_candidate_steps"] = int(conservative_normal_candidate_steps)
            summary["cstcc_shadow_conservative_floor_applied_count"] = int(conservative_floor_applied_count)
            summary["cstcc_shadow_conservative_floor_reason_distribution"] = dict(
                sorted(conservative_floor_reason_counts.items())
            )
            summary["cstcc_shadow_conservative_margin_coverage_count"] = int(len(conservative_prior_margin_values))
            summary["cstcc_shadow_conservative_selected_score_margin_mean"] = float(
                mean(conservative_prior_margin_values)
            ) if conservative_prior_margin_values else 0.0
            summary["cstcc_shadow_rule_prior_margin_coverage_steps"] = int(len(rule_prior_margin_values))
            summary["cstcc_shadow_rule_prior_selected_score_margin_mean"] = float(
                mean(rule_prior_margin_values)
            ) if rule_prior_margin_values else 0.0
            summary["cstcc_shadow_previous_shift_all_candidate_count"] = int(previous_shift_candidate_count)
            summary["cstcc_shadow_previous_shift_all_max_delta_count"] = int(previous_shift_max_delta_count)
            summary["cstcc_shadow_previous_shift_all_selected_count"] = int(previous_shift_selected_count)
            summary["cstcc_shadow_previous_shift_projection_applied_count"] = int(
                previous_shift_projection_applied_count
            )
            summary["cstcc_shadow_previous_shift_first_delta_after_projection_max"] = float(
                previous_shift_first_delta_after_projection_max
            )
            summary["cstcc_shadow_previous_shift_projection_mean_abs_delta_by_field"] = {
                field: float(previous_shift_projection_abs_sums[field] / max(previous_shift_projection_applied_count, 1))
                for field in cstcc_action_fields
            }
            summary["cstcc_shadow_previous_shift_internal_max_delta_after_projection_by_field"] = {
                field: float(previous_shift_internal_max_delta[field])
                for field in cstcc_action_fields
            }
            summary["cstcc_shadow_selected_source_prior_distribution"] = dict(
                sorted(selected_source_prior_counts.items())
            )
            summary["cstcc_shadow_selected_template_name_distribution"] = dict(
                sorted(selected_template_counts.items())
            )
            summary["cstcc_shadow_average_audit_latency_ms"] = float(
                mean(float(r.get("cstcc_shadow_audit_latency_ms", 0.0) or 0.0) for r in cstcc_enabled_rows)
            ) if cstcc_enabled_rows else 0.0
            summary["cstcc_shadow_average_audit_json_size_kb"] = float(
                mean(float(r.get("cstcc_shadow_audit_json_size_kb", 0.0) or 0.0) for r in cstcc_enabled_rows)
            ) if cstcc_enabled_rows else 0.0
            diff_values: Dict[str, List[float]] = {field: [] for field in cstcc_action_fields}
            for row in cstcc_success_rows:
                try:
                    diff = json.loads(str(row.get("cstcc_shadow_action_diff_json") or "{}"))
                except Exception:
                    diff = {}
                if not isinstance(diff, dict):
                    continue
                for field in cstcc_action_fields:
                    diff_values[field].append(abs(float(diff.get(field, 0.0) or 0.0)))
            summary["cstcc_shadow_mean_abs_action_diff_by_field"] = {
                field: float(mean(values)) if values else 0.0
                for field, values in diff_values.items()
            }
        if runtime_error_rows:
            runtime_error_counts: Dict[str, int] = {}
            for row in runtime_error_rows:
                error_type = str(row.get("runtime_error_type") or "runtime_error")
                runtime_error_counts[error_type] = runtime_error_counts.get(error_type, 0) + 1
            summary["runtime_error_counts"] = dict(sorted(runtime_error_counts.items()))
        if any("plan_cache_enabled" in r for r in rows):
            summary["plan_cache_enabled_steps"] = int(sum(bool(r.get("plan_cache_enabled", False)) for r in rows))
            summary["plan_cache_hit_steps"] = int(sum(bool(r.get("plan_cache_hit", False)) for r in rows))
        if any("profile_generator_candidate_count" in r for r in rows):
            profile_rows = [r for r in rows if int(r.get("profile_generator_candidate_count", 0) or 0) > 0]
            summary["profile_generator_shadow_steps"] = int(len(profile_rows))
            summary["mean_profile_generator_candidate_count"] = float(
                mean(int(r.get("profile_generator_candidate_count", 0) or 0) for r in rows)
            )
            summary["mean_profile_selected_abs_delta_sum"] = float(
                mean(float(r.get("profile_selected_abs_delta_sum", 0.0) or 0.0) for r in profile_rows)
            ) if profile_rows else 0.0
            intent_counts: Dict[str, int] = {}
            selected_counts: Dict[str, int] = {}
            unsupported_counts: Dict[str, int] = {}
            for row in profile_rows:
                regime = str(row.get("intent_regime") or "unknown")
                intent_counts[regime] = intent_counts.get(regime, 0) + 1
                selected = str(row.get("profile_generator_selected_shadow_profile") or "none")
                selected_counts[selected] = selected_counts.get(selected, 0) + 1
                unsupported = str(row.get("profile_generator_unsupported_shapes") or "")
                for item in [part.strip() for part in unsupported.split(",") if part.strip()]:
                    unsupported_counts[item] = unsupported_counts.get(item, 0) + 1
            summary["intent_regime_counts"] = dict(sorted(intent_counts.items()))
            summary["profile_generator_selected_counts"] = dict(sorted(selected_counts.items()))
            summary["profile_generator_unsupported_shape_counts"] = dict(sorted(unsupported_counts.items()))
            scorer_rows = [r for r in profile_rows if str(r.get("profile_scorer_selected_shadow_profile") or "")]
            if scorer_rows:
                summary["profile_scorer_shadow_steps"] = int(len(scorer_rows))
                summary["mean_profile_scorer_selected_score"] = float(
                    mean(float(r.get("profile_scorer_selected_score", 0.0) or 0.0) for r in scorer_rows)
                )
                summary["mean_profile_scorer_margin_to_second"] = float(
                    mean(float(r.get("profile_scorer_margin_to_second", 0.0) or 0.0) for r in scorer_rows)
                )
                summary["profile_scorer_selector_agreement_steps"] = int(
                    sum(bool(r.get("profile_scorer_agrees_with_selector", False)) for r in scorer_rows)
                )
                scorer_counts: Dict[str, int] = {}
                gate_counts: Dict[str, int] = {}
                for row in scorer_rows:
                    scorer = str(row.get("profile_scorer_selected_shadow_profile") or "none")
                    scorer_counts[scorer] = scorer_counts.get(scorer, 0) + 1
                    gate = str(row.get("profile_scorer_safety_gate_reason") or "none")
                    gate_counts[gate] = gate_counts.get(gate, 0) + 1
                summary["profile_scorer_selected_counts"] = dict(sorted(scorer_counts.items()))
                summary["profile_scorer_safety_gate_counts"] = dict(sorted(gate_counts.items()))
            rspc_rows = [r for r in profile_rows if bool(r.get("profile_rspc_shadow_enabled", False))]
            if rspc_rows:
                summary["profile_rspc_shadow_steps"] = int(len(rspc_rows))
                summary["mean_profile_rspc_shadow_margin"] = float(
                    mean(float(r.get("profile_rspc_shadow_margin", 0.0) or 0.0) for r in rspc_rows)
                )
                summary["profile_rspc_shadow_would_improve_steps"] = int(
                    sum(bool(r.get("profile_rspc_shadow_would_improve", False)) for r in rspc_rows)
                )
                rspc_best_counts: Dict[str, int] = {}
                rspc_gate_counts: Dict[str, int] = {}
                rspc_alignment_counts: Dict[str, int] = {}
                for row in rspc_rows:
                    best = str(row.get("profile_rspc_shadow_best_profile") or "none")
                    rspc_best_counts[best] = rspc_best_counts.get(best, 0) + 1
                    gate = str(row.get("profile_rspc_shadow_safety_gate_reason") or "none")
                    rspc_gate_counts[gate] = rspc_gate_counts.get(gate, 0) + 1
                    alignment = str(row.get("profile_rspc_shadow_safety_alignment") or "neutral_hold")
                    rspc_alignment_counts[alignment] = rspc_alignment_counts.get(alignment, 0) + 1
                summary["profile_rspc_shadow_best_counts"] = dict(sorted(rspc_best_counts.items()))
                summary["profile_rspc_shadow_safety_gate_counts"] = dict(sorted(rspc_gate_counts.items()))
                summary["profile_rspc_shadow_safety_alignment_counts"] = dict(sorted(rspc_alignment_counts.items()))
        if any("humidity_memory_available" in r for r in rows):
            summary["humidity_memory_available_steps"] = int(sum(bool(r.get("humidity_memory_available", False)) for r in rows))
            summary["humidity_memory_accepted_steps"] = int(sum(bool(r.get("humidity_memory_accepted", False)) for r in rows))
            summary["humidity_memory_selected_steps"] = int(sum(bool(r.get("humidity_memory_selected", False)) for r in rows))
            summary["humidity_memory_rejected_steps"] = int(
                sum(bool(r.get("humidity_memory_accepted", False)) and bool(r.get("humidity_memory_reject_reason")) for r in rows)
            )
            summary["mean_humidity_memory_horizon_penalty"] = float(
                mean(float(r.get("humidity_memory_horizon_penalty", 0.0)) for r in rows)
            )
        if any("mc_sero_enabled" in r for r in rows):
            summary["mc_sero_enabled_steps"] = int(sum(bool(r.get("mc_sero_enabled", False)) for r in rows))
            summary["mc_sero_available_steps"] = int(sum(bool(r.get("mc_sero_available", False)) for r in rows))
            summary["mc_sero_would_select_steps"] = int(sum(bool(r.get("mc_sero_would_select", False)) for r in rows))
            available_margins = [
                float(r.get("mc_sero_margin", 0.0))
                for r in rows
                if bool(r.get("mc_sero_available", False))
            ]
            summary["mean_mc_sero_margin"] = float(mean(available_margins)) if available_margins else 0.0
            candidate_counts: Dict[str, int] = {}
            reject_counts: Dict[str, int] = {}
            for row in rows:
                candidate = str(row.get("mc_sero_best_candidate") or "none")
                if candidate != "none":
                    candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1
                reason = str(row.get("mc_sero_reject_reason") or "")
                if reason:
                    reject_counts[reason] = reject_counts.get(reason, 0) + 1
            summary["mc_sero_best_candidate_counts"] = dict(sorted(candidate_counts.items()))
            summary["mc_sero_reject_reason_counts"] = dict(sorted(reject_counts.items()))
        if any("tomato_safety_v2_enabled" in r for r in rows):
            summary["tomato_safety_v2_enabled_steps"] = int(sum(bool(r.get("tomato_safety_v2_enabled", False)) for r in rows))
            summary["tomato_safety_v2_applied_steps"] = int(sum(bool(r.get("tomato_safety_v2_applied", False)) for r in rows))
            reason_counts: Dict[str, int] = {}
            for row in rows:
                reasons = str(row.get("tomato_safety_v2_reasons", "") or "")
                for reason in [part.strip() for part in reasons.split(",") if part.strip()]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
            summary["tomato_safety_v2_reason_counts"] = dict(sorted(reason_counts.items()))
            summary["tomato_safety_v2_suppressed_replan_steps"] = int(
                sum(bool(r.get("tomato_safety_v2_suppressed_replan", False)) for r in rows)
            )
        if any("transition_gate_enabled" in r for r in rows):
            summary["transition_gate_enabled_steps"] = int(sum(bool(r.get("transition_gate_enabled", False)) for r in rows))
            summary["transition_gate_applied_steps"] = int(sum(bool(r.get("transition_gate_applied", False)) for r in rows))
            summary["transition_gate_bypassed_steps"] = int(sum(bool(r.get("transition_gate_bypassed", False)) for r in rows))
            summary["transition_gate_delta_limited_count"] = int(
                sum(int(r.get("transition_gate_delta_limited_count", 0) or 0) for r in rows)
            )
            summary["transition_gate_reversal_projected_count"] = int(
                sum(int(r.get("transition_gate_reversal_projected_count", 0) or 0) for r in rows)
            )
        if any("profile_feasibility_gate_enabled" in r for r in rows):
            summary["profile_feasibility_gate_enabled_steps"] = int(
                sum(bool(r.get("profile_feasibility_gate_enabled", False)) for r in rows)
            )
            summary["profile_feasibility_gate_applied_steps"] = int(
                sum(bool(r.get("profile_feasibility_gate_applied", False)) for r in rows)
            )
            summary["profile_feasibility_gate_hard_safety_veto_count"] = int(
                sum(int(r.get("profile_feasibility_gate_hard_safety_veto_count", 0) or 0) for r in rows)
            )
    return summary


def summarize_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(row)
    return {name: sum_metrics(group_rows) for name, group_rows in sorted(grouped.items())}


def pairwise_step_diffs(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_algo: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for row in rows:
        by_algo.setdefault(str(row["algo"]), {})[int(row["step"])] = row
    ppo = by_algo.get("ppo", {})
    llm = by_algo.get("llm_director", {})
    common_steps = sorted(set(ppo) & set(llm))
    diffs: List[Dict[str, Any]] = []
    for step in common_steps:
        p = ppo[step]
        l = llm[step]
        row: Dict[str, Any] = {
            "step": step,
            "phase": p.get("phase"),
            "rh_band": p.get("rh_band"),
            "temp_band": p.get("temp_band"),
            "ppo_reward": float(p.get("reward", 0.0)),
            "llm_reward": float(l.get("reward", 0.0)),
            "reward_diff_llm_minus_ppo": float(l.get("reward", 0.0)) - float(p.get("reward", 0.0)),
            "ppo_profit": float(p.get("profit", 0.0)),
            "llm_profit": float(l.get("profit", 0.0)),
            "profit_diff_llm_minus_ppo": float(l.get("profit", 0.0)) - float(p.get("profit", 0.0)),
            "ppo_strategy_label": p.get("strategy_label", "unknown"),
            "ppo_strategy_confidence": float(p.get("strategy_confidence", 0.0)),
            "ppo_strategy_reason": p.get("strategy_reason", ""),
            "llm_strategy_label": l.get("strategy_label", "unknown"),
            "llm_strategy_confidence": float(l.get("strategy_confidence", 0.0)),
            "llm_strategy_reason": l.get("strategy_reason", ""),
        }
        for action in ACTION_NAMES:
            p_val = float(p.get(f"u_{action}", 0.0))
            l_val = float(l.get(f"u_{action}", 0.0))
            row[f"diff_{action}_llm_minus_ppo"] = l_val - p_val
            row[f"absdiff_{action}"] = abs(l_val - p_val)
        diffs.append(row)
    if not diffs:
        return {"common_steps": 0, "mean_abs_action_diff": {}, "largest_action_gap_steps": []}
    mean_abs = {
        action: float(mean(float(r[f"absdiff_{action}"]) for r in diffs))
        for action in ACTION_NAMES
    }
    largest = sorted(
        diffs,
        key=lambda r: sum(float(r[f"absdiff_{action}"]) for action in ACTION_NAMES),
        reverse=True,
    )[:10]
    return {
        "common_steps": len(common_steps),
        "mean_abs_action_diff": mean_abs,
        "sum_reward_diff_llm_minus_ppo": float(sum(r["reward_diff_llm_minus_ppo"] for r in diffs)),
        "sum_profit_diff_llm_minus_ppo": float(sum(r["profit_diff_llm_minus_ppo"] for r in diffs)),
        "largest_action_gap_steps": largest,
        "by_phase": summarize_pairwise_by(diffs, "phase"),
        "by_rh_band": summarize_pairwise_by(diffs, "rh_band"),
    }


def summarize_pairwise_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(row)
    out: Dict[str, Any] = {}
    for name, group_rows in sorted(grouped.items()):
        out[name] = {
            "steps": len(group_rows),
            "reward_diff_llm_minus_ppo": float(sum(r["reward_diff_llm_minus_ppo"] for r in group_rows)),
            "profit_diff_llm_minus_ppo": float(sum(r["profit_diff_llm_minus_ppo"] for r in group_rows)),
            "mean_abs_action_diff": {
                action: float(mean(float(r[f"absdiff_{action}"]) for r in group_rows))
                for action in ACTION_NAMES
            },
        }
    return out


def safety_patterns(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for algo in sorted({str(r["algo"]) for r in rows}):
        selected = [r for r in rows if r["algo"] == algo]
        heat_vent = [r for r in selected if float(r.get("u_heating", 0.0)) > 0.15 and float(r.get("u_ventilation", 0.0)) > 0.30]
        co2_leak = [r for r in selected if float(r.get("u_co2", 0.0)) > 0.05 and float(r.get("u_ventilation", 0.0)) > 0.20]
        lamp_risk = [
            r
            for r in selected
            if float(r.get("u_lighting", 0.0)) > 0.05
            and (float(r.get("rh_air", 0.0)) >= 85.0 or float(r.get("u_ventilation", 0.0)) >= 0.30)
        ]
        high_rh = [r for r in selected if float(r.get("rh_air", 0.0)) >= 90.0]
        low_rh = [r for r in selected if float(r.get("rh_air", 0.0)) < 50.0]
        dry_risk = [r for r in selected if bool(r.get("dry_risk", False))]
        dew_risk = [r for r in selected if bool(r.get("dew_risk", False))]
        cold_vent_risk = [r for r in selected if bool(r.get("cold_vent_risk", False))]
        dry_vent_risk = [r for r in selected if bool(r.get("dry_vent_risk", False))]
        high_vpd = [r for r in selected if float(r.get("vpd_air", 0.0)) > VPD_HIGH_LIMIT]
        out[algo] = {
            "heat_vent_conflict_steps": len(heat_vent),
            "co2_leak_steps": len(co2_leak),
            "lamp_risk_steps": len(lamp_risk),
            "state_rh_ge_90_steps": len(high_rh),
            "state_rh_lt_50_steps": len(low_rh),
            "state_vpd_gt_1p2_steps": len(high_vpd),
            "dry_risk_steps": len(dry_risk),
            "dew_risk_steps": len(dew_risk),
            "cold_vent_risk_steps": len(cold_vent_risk),
            "dry_vent_risk_steps": len(dry_vent_risk),
            "mean_vent_when_rh_ge_90": float(mean(float(r.get("u_ventilation", 0.0)) for r in high_rh)) if high_rh else 0.0,
            "mean_heat_when_rh_ge_90": float(mean(float(r.get("u_heating", 0.0)) for r in high_rh)) if high_rh else 0.0,
            "mean_vent_when_dry_risk": float(mean(float(r.get("u_ventilation", 0.0)) for r in dry_risk)) if dry_risk else 0.0,
        }
    return out


def strategy_label_audit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    audit: Dict[str, Any] = {}
    for algo in sorted({str(r["algo"]) for r in rows}):
        selected = [r for r in rows if r["algo"] == algo and "strategy_label" in r]
        label_counts: Dict[str, int] = {}
        best_counts: Dict[str, int] = {}
        by_rh: Dict[str, Dict[str, int]] = {}
        by_phase: Dict[str, Dict[str, int]] = {}
        confidence_values: List[float] = []
        for row in selected:
            label = str(row.get("strategy_label", "unknown"))
            best_label = str(row.get("strategy_best_label", "unknown"))
            label_counts[label] = label_counts.get(label, 0) + 1
            best_counts[best_label] = best_counts.get(best_label, 0) + 1
            confidence_values.append(float(row.get("strategy_confidence", 0.0)))
            rh_band = str(row.get("rh_band", "unknown"))
            phase = str(row.get("phase", "unknown"))
            by_rh.setdefault(rh_band, {})
            by_rh[rh_band][label] = by_rh[rh_band].get(label, 0) + 1
            by_phase.setdefault(phase, {})
            by_phase[phase][label] = by_phase[phase].get(label, 0) + 1
        audit[algo] = {
            "rows": len(selected),
            "confident_rows": sum(1 for r in selected if str(r.get("strategy_label")) not in {"unknown", "ambiguous"}),
            "ambiguous_rows": label_counts.get("ambiguous", 0),
            "unknown_rows": label_counts.get("unknown", 0),
            "mean_confidence": float(mean(confidence_values)) if confidence_values else 0.0,
            "label_counts": dict(sorted(label_counts.items())),
            "best_label_counts": dict(sorted(best_counts.items())),
            "by_rh_band": by_rh,
            "by_phase": by_phase,
        }
    return audit


def valuable_ppo_tendency_candidates(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find high-confidence PPO tendencies that look economically useful.

    This is only an audit aid. A sample is considered useful when PPO has better
    immediate profit and does not increase same-step temperature/RH violation
    beyond a small tolerance. Later work can replace this with horizon-aware
    attribution.
    """
    by_algo: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for row in rows:
        by_algo.setdefault(str(row["algo"]), {})[int(row["step"])] = row
    ppo = by_algo.get("ppo", {})
    llm = by_algo.get("llm_director", {})
    useful: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for step in sorted(set(ppo) & set(llm)):
        p = ppo[step]
        l = llm[step]
        label = str(p.get("strategy_label", "unknown"))
        confidence = float(p.get("strategy_confidence", 0.0))
        if label in {"unknown", "ambiguous"} or confidence < 0.55:
            continue
        profit_adv = float(p.get("profit", 0.0)) - float(l.get("profit", 0.0))
        rh_extra = float(p.get("rh_violation", 0.0)) - float(l.get("rh_violation", 0.0))
        temp_extra = float(p.get("temp_violation", 0.0)) - float(l.get("temp_violation", 0.0))
        record = {
            "step": int(step),
            "label": label,
            "confidence": confidence,
            "phase": p.get("phase"),
            "rh_band": p.get("rh_band"),
            "profit_advantage_ppo_minus_llm": profit_adv,
            "rh_violation_extra_ppo_minus_llm": rh_extra,
            "temp_violation_extra_ppo_minus_llm": temp_extra,
            "reason": p.get("strategy_reason", ""),
        }
        if profit_adv > 0.0 and rh_extra <= 0.05 and temp_extra <= 0.05:
            useful.append(record)
        else:
            rejected.append(record)
    label_counts: Dict[str, int] = {}
    for row in useful:
        label = str(row["label"])
        label_counts[label] = label_counts.get(label, 0) + 1
    return {
        "useful_count": len(useful),
        "rejected_count": len(rejected),
        "useful_by_label": dict(sorted(label_counts.items())),
        "examples": useful[:20],
        "rejected_examples": rejected[:20],
        "note": "Immediate same-step filter; use as an audit hint, not final causal attribution.",
    }


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {
                key: _compact_json(value) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            for row in rows
        )


def build_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# PPO vs LLM-RSPC Diagnostic Report",
        "",
        f"- Scenario: year={summary['scenario']['year']}, day={summary['scenario']['day']}, seed={summary['scenario']['seed']}",
        f"- Steps: {summary['scenario']['max_steps']}",
        f"- Expert rollout: {summary['scenario']['expert_rollout']}",
        f"- Plan cache: {summary['scenario'].get('plan_cache_mode', 'off')}",
        "",
        "## Aggregate",
        "| Algorithm | Reward | Profit | Revenue | Heat | CO2 | Elec | Temp V | RH V |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algo, item in summary["aggregate"].items():
        lines.append(
            f"| {algo} | {item['total_reward']:.3f} | {item['total_profit']:.5f} | "
            f"{item['total_revenue']:.5f} | {item['total_heat_cost']:.5f} | "
            f"{item['total_co2_cost']:.5f} | {item['total_elec_cost']:.5f} | "
            f"{item['total_temp_violation']:.3f} | {item['total_rh_violation']:.3f} |"
        )
    pair = summary.get("pairwise", {})
    lines.extend(
        [
            "",
            "## PPO - LLM Gap",
            f"- Common steps: {pair.get('common_steps', 0)}",
            f"- Sum reward diff (LLM minus PPO): {pair.get('sum_reward_diff_llm_minus_ppo', 0.0):.3f}",
            f"- Sum profit diff (LLM minus PPO): {pair.get('sum_profit_diff_llm_minus_ppo', 0.0):.5f}",
            "- Mean absolute action difference:",
        ]
    )
    for action, value in pair.get("mean_abs_action_diff", {}).items():
        lines.append(f"  - {action}: {value:.3f}")
    lines.extend(["", "## Safety Patterns"])
    for algo, item in summary.get("safety_patterns", {}).items():
        lines.append(
            f"- {algo}: heat+vent conflict={item['heat_vent_conflict_steps']}, "
            f"CO2 leak={item['co2_leak_steps']}, lamp risk={item['lamp_risk_steps']}, "
            f"RH>=90 state steps={item['state_rh_ge_90_steps']}"
        )
    llm_item = summary.get("aggregate", {}).get("llm_director", {})
    if "plan_cache_enabled_steps" in llm_item:
        lines.extend(
            [
                "",
                "## Plan Cache",
                f"- enabled rows={llm_item.get('plan_cache_enabled_steps', 0)}, "
                f"cache hits={llm_item.get('plan_cache_hit_steps', 0)}",
            ]
        )
    if "humidity_memory_available_steps" in llm_item:
        lines.extend(
            [
                "",
                "## Humidity Memory",
                f"- available={llm_item.get('humidity_memory_available_steps', 0)}, "
                f"accepted={llm_item.get('humidity_memory_accepted_steps', 0)}, "
                f"selected={llm_item.get('humidity_memory_selected_steps', 0)}, "
                f"rejected={llm_item.get('humidity_memory_rejected_steps', 0)}",
                f"- mean horizon penalty={llm_item.get('mean_humidity_memory_horizon_penalty', 0.0):.5f}",
            ]
        )
    if "mc_sero_enabled_steps" in llm_item:
        lines.extend(
            [
                "",
                "## MC-SERO Shadow",
                f"- enabled={llm_item.get('mc_sero_enabled_steps', 0)}, "
                f"available={llm_item.get('mc_sero_available_steps', 0)}, "
                f"would_select={llm_item.get('mc_sero_would_select_steps', 0)}",
                f"- mean margin={llm_item.get('mean_mc_sero_margin', 0.0):.5f}",
                f"- best candidates={llm_item.get('mc_sero_best_candidate_counts', {})}",
                f"- reject reasons={llm_item.get('mc_sero_reject_reason_counts', {})}",
            ]
        )
    lines.extend(["", "## Strategy Label Audit"])
    for algo, item in summary.get("strategy_label_audit", {}).items():
        lines.append(
            f"- {algo}: confident={item['confident_rows']}/{item['rows']}, "
            f"ambiguous={item['ambiguous_rows']}, unknown={item['unknown_rows']}, "
            f"mean confidence={item['mean_confidence']:.3f}"
        )
        lines.append(f"  - labels: {item['label_counts']}")
    useful = summary.get("valuable_ppo_tendencies", {})
    lines.extend(
        [
            "",
            "## Useful PPO Tendency Candidates",
            f"- Useful high-confidence samples: {useful.get('useful_count', 0)}",
            f"- Rejected high-confidence samples: {useful.get('rejected_count', 0)}",
            f"- Useful by label: {useful.get('useful_by_label', {})}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose PPO vs LLM-RSPC trajectories.")
    parser.add_argument("--config", type=str, default="gl_gym/configs/envs/TomatoEnv.yml")
    parser.add_argument("--model-path", type=str, default="train_data/AgriControl/ppo/deterministic/models/None/best_model.zip")
    parser.add_argument("--vecnorm-path", type=str, default="train_data/AgriControl/ppo/deterministic/envs/None/best_vecnormalize.pkl")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--day", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--uncertainty-scale", type=float, default=0.0)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--llm-model", type=str, default=AgentConfig.model_name)
    parser.add_argument("--llm-interval", type=int, default=12)
    parser.add_argument("--llm-max-iterations", type=int, default=1)
    parser.add_argument("--llm-max-tokens", type=int, default=260)
    parser.add_argument("--llm-plan-cache-mode", type=str, choices=["off", "record", "replay", "refresh"], default="off")
    parser.add_argument("--llm-plan-cache-path", type=str, default=AgentConfig.plan_cache_path)
    parser.add_argument("--llm-plan-cache-strict", action="store_true")
    parser.add_argument("--llm-plan-cache-key-policy", type=str, choices=["prompt", "scenario_timestep"], default=AgentConfig.plan_cache_key_policy)
    parser.add_argument("--llm-expert-rollout", action="store_true")
    parser.add_argument("--llm-expert-policy-path", type=str, default=AgentConfig.expert_policy_path)
    parser.add_argument("--llm-expert-max-distance", type=float, default=AgentConfig.expert_candidate_max_distance)
    parser.add_argument("--llm-humidity-memory", action="store_true")
    parser.add_argument("--llm-humidity-memory-path", type=str, default=AgentConfig.humidity_memory_path)
    parser.add_argument("--llm-humidity-memory-max-distance", type=float, default=AgentConfig.humidity_memory_max_distance)
    parser.add_argument("--llm-humidity-memory-min-trust", type=float, default=AgentConfig.humidity_memory_min_trust)
    parser.add_argument("--llm-humidity-memory-teacher-policy-id", type=str, default="")
    parser.add_argument("--llm-humidity-memory-baseline-controller-id", type=str, default="")
    parser.add_argument("--llm-humidity-memory-version", type=str, default=AgentConfig.humidity_memory_version)
    parser.add_argument("--llm-mc-sero-mode", type=str, choices=["off", "shadow"], default=AgentConfig.mc_sero_mode)
    parser.add_argument("--llm-tomato-safety-v2", action="store_true")
    parser.add_argument("--llm-cstcc-shadow", action="store_true")
    parser.add_argument("--llm-cstcc-shadow-log-root", type=str, default=AgentConfig.cstcc_shadow_audit_log_root)
    parser.add_argument("--llm-cstcc-shadow-sample-rate", type=float, default=AgentConfig.cstcc_shadow_sample_rate)
    parser.add_argument("--llm-cstcc-shadow-save-full-candidates", action="store_true")
    parser.add_argument("--llm-cstcc-shadow-save-raw-sequences", action="store_true")
    parser.add_argument("--llm-cstcc-shadow-save-projected-sequences", action="store_true")
    parser.add_argument("--llm-cstcc-shadow-fail-closed", action="store_true")
    parser.add_argument("--llm-cstcc-shadow-no-assert-invariant", action="store_true")
    parser.add_argument("--llm-cstcc-shadow-config", type=str, default="")
    parser.add_argument("--output-json", type=str, default="gl_gym/result/diagnostics/ppo_vs_llm_diag.json")
    parser.add_argument("--output-csv", type=str, default="gl_gym/result/diagnostics/ppo_vs_llm_trace.csv")
    parser.add_argument("--output-report", type=str, default="gl_gym/result/diagnostics/ppo_vs_llm_diag.md")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = Path(args.model_path)
    vecnorm_path = Path(args.vecnorm_path)
    model = PPO.load(model_path)

    rows: List[Dict[str, Any]] = []
    ppo_env = build_env(config, args.year, args.day, args.seed, args.uncertainty_scale)
    t0 = time.perf_counter()
    rows.extend(run_ppo_trace(model, vecnorm_path, ppo_env, args.year, args.day, args.seed, args.max_steps))
    ppo_elapsed = time.perf_counter() - t0

    llm_elapsed = 0.0
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key and args.llm_plan_cache_mode == "replay":
        api_key = "plan-cache-replay"
    if not args.skip_llm and api_key:
        t1 = time.perf_counter()
        rows.extend(
            run_llm_trace(
                config=config,
                api_key=api_key,
                year=args.year,
                day=args.day,
                seed=args.seed,
                max_steps=args.max_steps,
                llm_model=args.llm_model,
                llm_interval=args.llm_interval,
                llm_max_iterations=args.llm_max_iterations,
                llm_max_tokens=args.llm_max_tokens,
                expert_rollout=args.llm_expert_rollout,
                expert_policy_path=args.llm_expert_policy_path,
                expert_max_distance=args.llm_expert_max_distance,
                humidity_memory_enabled=args.llm_humidity_memory,
                humidity_memory_path=args.llm_humidity_memory_path,
                humidity_memory_max_distance=args.llm_humidity_memory_max_distance,
                humidity_memory_min_trust=args.llm_humidity_memory_min_trust,
                humidity_memory_teacher_policy_id=args.llm_humidity_memory_teacher_policy_id,
                humidity_memory_baseline_controller_id=args.llm_humidity_memory_baseline_controller_id,
                humidity_memory_version=args.llm_humidity_memory_version,
                mc_sero_mode=args.llm_mc_sero_mode,
                tomato_safety_v2_enabled=args.llm_tomato_safety_v2,
                plan_cache_mode=args.llm_plan_cache_mode,
                plan_cache_path=args.llm_plan_cache_path,
                plan_cache_strict=args.llm_plan_cache_strict,
                plan_cache_key_policy=args.llm_plan_cache_key_policy,
                uncertainty_scale=args.uncertainty_scale,
                agent_config_overrides={
                    "cstcc_shadow_enabled": bool(args.llm_cstcc_shadow),
                    "cstcc_shadow_audit_log_root": str(args.llm_cstcc_shadow_log_root),
                    "cstcc_shadow_sample_rate": float(args.llm_cstcc_shadow_sample_rate),
                    "cstcc_shadow_save_full_candidates": bool(args.llm_cstcc_shadow_save_full_candidates),
                    "cstcc_shadow_save_raw_sequences": bool(args.llm_cstcc_shadow_save_raw_sequences),
                    "cstcc_shadow_save_projected_sequences": bool(args.llm_cstcc_shadow_save_projected_sequences),
                    "cstcc_shadow_fail_closed": bool(args.llm_cstcc_shadow_fail_closed),
                    "cstcc_shadow_assert_final_action_invariant": not bool(args.llm_cstcc_shadow_no_assert_invariant),
                    "cstcc_shadow_config_path": str(args.llm_cstcc_shadow_config or "") or None,
                },
            )
        )
        llm_elapsed = time.perf_counter() - t1
    elif not args.skip_llm:
        print("BAILIAN_API_KEY is missing; PPO trace only.")

    aggregate = {}
    for algo in sorted({str(r["algo"]) for r in rows}):
        aggregate[algo] = sum_metrics([r for r in rows if r["algo"] == algo])

    summary = {
        "scenario": {
            "year": int(args.year),
            "day": int(args.day),
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "uncertainty_scale": float(args.uncertainty_scale),
            "expert_rollout": bool(args.llm_expert_rollout),
            "humidity_memory": bool(args.llm_humidity_memory),
            "humidity_memory_version": str(args.llm_humidity_memory_version),
            "mc_sero_mode": str(args.llm_mc_sero_mode),
            "tomato_safety_v2": bool(args.llm_tomato_safety_v2),
            "plan_cache_mode": str(args.llm_plan_cache_mode),
            "plan_cache_path": str(args.llm_plan_cache_path),
            "plan_cache_strict": bool(args.llm_plan_cache_strict),
            "plan_cache_key_policy": str(args.llm_plan_cache_key_policy),
            "cstcc_shadow_enabled": bool(args.llm_cstcc_shadow),
            "cstcc_shadow_log_root": str(args.llm_cstcc_shadow_log_root),
            "cstcc_shadow_sample_rate": float(args.llm_cstcc_shadow_sample_rate),
            "cstcc_shadow_fail_closed": bool(args.llm_cstcc_shadow_fail_closed),
            "cstcc_shadow_config_path": str(args.llm_cstcc_shadow_config or ""),
        },
        "timing": {"ppo_seconds": float(ppo_elapsed), "llm_seconds": float(llm_elapsed)},
        "aggregate": aggregate,
        "by_phase": {algo: summarize_by([r for r in rows if r["algo"] == algo], "phase") for algo in aggregate},
        "by_rh_band": {algo: summarize_by([r for r in rows if r["algo"] == algo], "rh_band") for algo in aggregate},
        "by_temp_band": {algo: summarize_by([r for r in rows if r["algo"] == algo], "temp_band") for algo in aggregate},
        "pairwise": pairwise_step_diffs(rows),
        "safety_patterns": safety_patterns(rows),
        "strategy_label_audit": strategy_label_audit(rows),
        "valuable_ppo_tendencies": valuable_ppo_tendency_candidates(rows),
    }

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    report_path = Path(args.output_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    write_rows_csv(csv_path, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(f"Saved diagnostic JSON to {json_path}")
    print(f"Saved trace CSV to {csv_path}")
    print(f"Saved report to {report_path}")
    print(build_report(summary))


if __name__ == "__main__":
    main()
