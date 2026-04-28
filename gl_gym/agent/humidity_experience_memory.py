"""Humidity-economic experience memory for LLM-RSPC.

This module keeps the first memory layer deliberately narrow. It only stores
validated low-cost free-air-exchange tendencies for moderate humidity risk.
PPO is treated as a candidate evidence source, not as ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from gl_gym.agent.expert_distillation import ACTION_NAMES
from gl_gym.agent.plan_intent import (
    action_record_for_labeling,
    infer_plan_intent,
    make_plan_from_targets,
    target_tracking_baseline_control,
)
from gl_gym.agent.ppo_strategy_labeler import label_strategy


CONTEXT_FEATURES: Tuple[str, ...] = (
    "temp_air",
    "rh_air",
    "temp_out",
    "rh_out",
    "glob_rad",
    "wind_speed",
    "vpd_kpa",
    "dew_margin_air",
    "canopy_dew_margin",
    "target_temp_delta",
    "target_rh_delta",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
)


CONTEXT_SCALES: Dict[str, float] = {
    "temp_air": 4.0,
    "rh_air": 10.0,
    "temp_out": 5.0,
    "rh_out": 15.0,
    "glob_rad": 250.0,
    "wind_speed": 4.0,
    "vpd_kpa": 0.4,
    "dew_margin_air": 2.0,
    "canopy_dew_margin": 2.0,
    "target_temp_delta": 3.0,
    "target_rh_delta": 8.0,
    "hour_sin": 1.0,
    "hour_cos": 1.0,
    "day_sin": 1.0,
    "day_cos": 1.0,
}


@dataclass
class HumidityExperienceConfig:
    min_rh: float = 78.0
    max_rh: float = 88.0
    min_rh_gap: float = 3.0
    max_rh_gap: float = 12.0
    min_vpd_kpa: float = 0.25
    max_vpd_kpa: float = 0.80
    min_air_dew_margin: float = 1.5
    min_canopy_dew_margin: float = 1.0
    min_abs_humidity_advantage: float = 0.5
    min_dew_point_advantage: float = 1.0
    min_vapor_pressure_advantage: float = 0.2
    hard_min_temp: float = 12.0
    min_temp_margin_to_target: float = -0.5
    min_temp_margin_to_hard: float = 1.0
    min_intent_confidence: float = 0.60
    min_strategy_confidence: float = 0.55
    min_ventilation: float = 0.50
    max_heating: float = 0.15
    max_co2: float = 0.05
    max_lighting: float = 0.05
    max_heat_vent_conflict: float = 0.12
    max_state_temp_gap: float = 2.0
    max_state_rh_gap: float = 4.0
    min_profit_advantage: float = 0.0
    min_cost_saving: float = 0.0
    max_rh_violation_extra: float = 0.05
    max_temp_violation_extra: float = 0.05
    allow_risk_based_intent_override: bool = True
    risk_override_max_rh: float = 86.5
    risk_override_max_rh_gap: float = 13.0
    risk_override_min_vpd_kpa: float = 0.28
    risk_override_min_air_dew_margin: float = 2.0
    retrieval_max_distance: float = 1.15
    min_trust: float = 0.50
    merge_max_distance: float = 0.45


@dataclass
class ExperienceAssessment:
    accepted: bool
    reasons: Tuple[str, ...]
    metrics: Dict[str, float]
    intent_label: str = "unknown"
    intent_confidence: float = 0.0
    strategy_label: str = "unknown"
    strategy_confidence: float = 0.0


@dataclass
class HumidityExperience:
    case_id: str
    status: str
    source: str
    scenario: Dict[str, Any]
    intent_label: str
    intent_confidence: float
    strategy_label: str
    strategy_confidence: float
    context: Dict[str, float]
    context_vector: List[float]
    target: Dict[str, float]
    base_action: List[float]
    expert_action: List[float]
    baseline_action: List[float]
    residual_action: List[float]
    evidence: Dict[str, float]
    trust: float
    support_count: int = 1
    supporting_cases: List[str] = field(default_factory=list)

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "HumidityExperience":
        return cls(
            case_id=str(record["case_id"]),
            status=str(record.get("status", "pending")),
            source=str(record.get("source", "unknown")),
            scenario=dict(record.get("scenario", {})),
            intent_label=str(record.get("intent_label", "unknown")),
            intent_confidence=float(record.get("intent_confidence", 0.0)),
            strategy_label=str(record.get("strategy_label", "unknown")),
            strategy_confidence=float(record.get("strategy_confidence", 0.0)),
            context={str(k): float(v) for k, v in dict(record.get("context", {})).items()},
            context_vector=[float(x) for x in record.get("context_vector", [])],
            target={str(k): float(v) for k, v in dict(record.get("target", {})).items()},
            base_action=[float(x) for x in record.get("base_action", [])],
            expert_action=[float(x) for x in record.get("expert_action", [])],
            baseline_action=[float(x) for x in record.get("baseline_action", [])],
            residual_action=[float(x) for x in record.get("residual_action", [])],
            evidence={str(k): float(v) for k, v in dict(record.get("evidence", {})).items()},
            trust=float(record.get("trust", 0.0)),
            support_count=int(record.get("support_count", 1)),
            supporting_cases=[str(x) for x in record.get("supporting_cases", [])],
        )


def _float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _nested_prefixed(record: Dict[str, Any], nested_key: str, flat_key: str, default: Any = None) -> Any:
    nested = record.get(nested_key)
    if isinstance(nested, dict) and flat_key in nested:
        return nested.get(flat_key, default)
    return record.get(flat_key, default)


def saturation_vapor_pressure_kpa(temp_c: float) -> float:
    return float(0.61078 * math.exp((17.27 * temp_c) / (temp_c + 237.3)))


def vapor_pressure_kpa(temp_c: float, rh_percent: float) -> float:
    return float(np.clip(rh_percent, 0.0, 100.0) / 100.0 * saturation_vapor_pressure_kpa(temp_c))


def absolute_humidity_g_m3(temp_c: float, rh_percent: float) -> float:
    # 216.7 * e[hPa] / T[K], with e[kPa] converted to hPa by *10.
    return float(2167.0 * vapor_pressure_kpa(temp_c, rh_percent) / (temp_c + 273.15))


def dew_point_c(temp_c: float, rh_percent: float) -> float:
    rh = float(np.clip(rh_percent, 1e-3, 100.0))
    a = 17.27
    b = 237.3
    gamma = math.log(rh / 100.0) + (a * temp_c) / (b + temp_c)
    return float((b * gamma) / (a - gamma))


def row_to_state(row: Dict[str, Any]) -> SimpleNamespace:
    values = {
        "timestep": _float(row, "timestep", _float(row, "step", 0.0)),
        "hour_of_day": _float(row, "hour_of_day", 12.0),
        "day_of_year": _float(row, "day_of_year", _float(row, "day", 180.0)),
        "temp_air": _float(row, "temp_air", 20.0),
        "rh_air": _float(row, "rh_air", 70.0),
        "co2_air": _float(row, "co2_air", 430.0),
        "glob_rad": _float(row, "glob_rad", 0.0),
        "temp_out": _float(row, "temp_out", _float(row, "temp_air", 20.0)),
        "rh_out": _float(row, "rh_out", 70.0),
        "wind_speed": _float(row, "wind_speed", 0.0),
        "pipe_temp": _float(row, "pipe_temp", 30.0),
        "fruit_weight": _float(row, "fruit_weight", 0.0),
        "canopy_temp_24h": _float(row, "canopy_temp_24h", _float(row, "temp_air", 20.0)),
        "temperature_sum": _float(row, "temperature_sum", 0.0),
        "dli": _float(row, "dli", 0.0),
        "dew_margin_air": _float(row, "dew_margin_air", 3.0),
        "canopy_dew_margin": _float(row, "canopy_dew_margin", 3.0),
        "forecast_humidity_risk": _float(row, "forecast_humidity_risk", 0.0),
        "forecast_rad_mean_1h": _float(row, "forecast_rad_mean_1h", _float(row, "glob_rad", 0.0)),
        "forecast_rad_peak_2h": _float(row, "forecast_rad_peak_2h", _float(row, "glob_rad", 0.0)),
        "forecast_temp_out_delta_1h": _float(row, "forecast_temp_out_delta_1h", 0.0),
        "forecast_rh_out_mean_1h": _float(row, "forecast_rh_out_mean_1h", _float(row, "rh_out", 70.0)),
        "forecast_wind_peak_1h": _float(row, "forecast_wind_peak_1h", _float(row, "wind_speed", 0.0)),
        "u_boil": _float(row, "u_heating", _float(row, "u_boil", 0.0)),
        "u_co2": _float(row, "u_co2", 0.0),
        "u_th_scr": _float(row, "u_screen", _float(row, "u_th_scr", 0.0)),
        "u_vent": _float(row, "u_ventilation", _float(row, "u_vent", 0.0)),
        "u_lamp": _float(row, "u_lighting", _float(row, "u_lamp", 0.0)),
        "u_bl_scr": _float(row, "u_shading", _float(row, "u_bl_scr", 0.0)),
    }
    return SimpleNamespace(**values)


def action_from_row(row: Dict[str, Any], prefix: str = "u") -> np.ndarray:
    if prefix == "u" and isinstance(row.get("action"), Sequence) and not isinstance(row.get("action"), str):
        values = np.asarray(row["action"], dtype=np.float32).reshape(-1)
    else:
        values = np.asarray([_float(row, f"{prefix}_{name}", 0.0) for name in ACTION_NAMES], dtype=np.float32)
    if values.shape[0] < len(ACTION_NAMES):
        values = np.pad(values, (0, len(ACTION_NAMES) - values.shape[0]))
    return np.clip(values[: len(ACTION_NAMES)], 0.0, 1.0).astype(np.float32)


def target_from_rows(primary: Dict[str, Any], secondary: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    secondary = secondary or {}
    nested_plan = primary.get("target_plan")
    if isinstance(nested_plan, dict):
        source = nested_plan
    else:
        source = primary
    return {
        "target_temp": _float(source, "target_temp", _float(secondary, "target_temp", 17.0)),
        "target_co2": _float(source, "target_co2", _float(secondary, "target_co2", 430.0)),
        "target_rh": _float(source, "target_rh", _float(secondary, "target_rh", 76.0)),
    }


def plan_from_rows(state_row: Dict[str, Any], target_row: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    target = target_from_rows(state_row, target_row)
    return make_plan_from_targets(
        row_to_state(state_row),
        target_temp=target["target_temp"],
        target_co2=target["target_co2"],
        target_rh=target["target_rh"],
        horizon=12,
        source="humidity_experience_memory",
    )


def infer_intent_from_rows(state_row: Dict[str, Any], target_row: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
    label = _nested_prefixed(state_row, "intent", "intent_label")
    confidence = _nested_prefixed(state_row, "intent", "intent_confidence")
    if label:
        return str(label), float(confidence or 0.0)
    state = row_to_state(state_row)
    intent = infer_plan_intent(state, plan_from_rows(state_row, target_row))
    return intent.label, float(intent.confidence)


def infer_strategy_from_row(row: Dict[str, Any], min_confidence: float = 0.55) -> Tuple[str, float]:
    label = _nested_prefixed(row, "strategy", "strategy_label")
    confidence = _nested_prefixed(row, "strategy", "strategy_confidence")
    if label:
        return str(label), float(confidence or 0.0)
    state = row_to_state(row)
    strategy = label_strategy(action_record_for_labeling(state, action_from_row(row)), min_confidence=min_confidence)
    return strategy.label, float(strategy.confidence)


def build_context(row: Dict[str, Any], target_row: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    target = target_from_rows(target_row or row, row)
    temp = _float(row, "temp_air", 20.0)
    rh = _float(row, "rh_air", 70.0)
    hour = _float(row, "hour_of_day", 12.0)
    day = _float(row, "day_of_year", _float(row, "day", 180.0))
    context = {
        "temp_air": temp,
        "rh_air": rh,
        "temp_out": _float(row, "temp_out", temp),
        "rh_out": _float(row, "rh_out", 70.0),
        "glob_rad": _float(row, "glob_rad", 0.0),
        "wind_speed": _float(row, "wind_speed", 0.0),
        "vpd_kpa": vapor_pressure_deficit_kpa(temp, rh),
        "dew_margin_air": _float(row, "dew_margin_air", temp - dew_point_c(temp, rh)),
        "canopy_dew_margin": _float(row, "canopy_dew_margin", temp - dew_point_c(temp, rh)),
        "target_temp_delta": target["target_temp"] - temp,
        "target_rh_delta": target["target_rh"] - rh,
        "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
        "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
        "day_sin": math.sin(2.0 * math.pi * day / 365.0),
        "day_cos": math.cos(2.0 * math.pi * day / 365.0),
    }
    return {key: float(context[key]) for key in CONTEXT_FEATURES}


def context_vector(context: Dict[str, float]) -> np.ndarray:
    return np.asarray([float(context[name]) / CONTEXT_SCALES[name] for name in CONTEXT_FEATURES], dtype=np.float32)


def vapor_pressure_deficit_kpa(temp_c: float, rh_percent: float) -> float:
    svp = saturation_vapor_pressure_kpa(temp_c)
    return float(max(0.0, svp - vapor_pressure_kpa(temp_c, rh_percent)))


def operating_cost(row: Dict[str, Any]) -> float:
    return _float(row, "heat_cost", 0.0) + _float(row, "co2_cost", 0.0) + _float(row, "elec_cost", 0.0)


def action_proxy_cost(action: Iterable[float]) -> float:
    values = np.asarray(list(action), dtype=np.float32).reshape(-1)
    if values.shape[0] < 6:
        values = np.pad(values, (0, 6 - values.shape[0]))
    return float(values[0] + 0.35 * values[1] + 1.25 * values[4])


def heat_vent_conflict(action: Iterable[float]) -> float:
    values = np.asarray(list(action), dtype=np.float32).reshape(-1)
    if values.shape[0] < 6:
        values = np.pad(values, (0, 6 - values.shape[0]))
    return float(values[0] * values[3])


def dehumidification_potential(row: Dict[str, Any]) -> Dict[str, float]:
    temp = _float(row, "temp_air", 20.0)
    rh = _float(row, "rh_air", 70.0)
    temp_out = _float(row, "temp_out", temp)
    rh_out = _float(row, "rh_out", rh)
    inside_ah = absolute_humidity_g_m3(temp, rh)
    outside_ah = absolute_humidity_g_m3(temp_out, rh_out)
    inside_dp = dew_point_c(temp, rh)
    outside_dp = dew_point_c(temp_out, rh_out)
    inside_vp = vapor_pressure_kpa(temp, rh)
    outside_vp = vapor_pressure_kpa(temp_out, rh_out)
    return {
        "inside_absolute_humidity": inside_ah,
        "outside_absolute_humidity": outside_ah,
        "absolute_humidity_advantage": inside_ah - outside_ah,
        "inside_dew_point": inside_dp,
        "outside_dew_point": outside_dp,
        "dew_point_advantage": inside_dp - outside_dp,
        "inside_vapor_pressure": inside_vp,
        "outside_vapor_pressure": outside_vp,
        "vapor_pressure_advantage": inside_vp - outside_vp,
    }


def screen_free_air_exchange_context(
    row: Dict[str, Any],
    target_row: Optional[Dict[str, Any]] = None,
    config: Optional[HumidityExperienceConfig] = None,
) -> Tuple[Tuple[str, ...], Dict[str, float]]:
    cfg = config or HumidityExperienceConfig()
    target = target_from_rows(target_row or row, row)
    temp = _float(row, "temp_air", 20.0)
    rh = _float(row, "rh_air", 70.0)
    rh_gap = rh - target["target_rh"]
    vpd = vapor_pressure_deficit_kpa(temp, rh)
    air_dew_margin = _float(row, "dew_margin_air", temp - dew_point_c(temp, rh))
    canopy_dew_margin = _float(row, "canopy_dew_margin", air_dew_margin)
    potential = dehumidification_potential(row)
    reasons: List[str] = []
    if not (cfg.min_rh <= rh <= cfg.max_rh):
        reasons.append("not_moderate_rh")
    if not (cfg.min_rh_gap <= rh_gap <= cfg.max_rh_gap):
        reasons.append("target_rh_gap_outside_band")
    if not (cfg.min_vpd_kpa <= vpd <= cfg.max_vpd_kpa):
        reasons.append("vpd_outside_safe_economic_band")
    if air_dew_margin < cfg.min_air_dew_margin:
        reasons.append("air_dew_margin_too_low")
    if canopy_dew_margin < cfg.min_canopy_dew_margin:
        reasons.append("canopy_dew_margin_too_low")
    outside_is_drier = (
        potential["absolute_humidity_advantage"] >= cfg.min_abs_humidity_advantage
        or potential["dew_point_advantage"] >= cfg.min_dew_point_advantage
        or potential["vapor_pressure_advantage"] >= cfg.min_vapor_pressure_advantage
    )
    if not outside_is_drier:
        reasons.append("outside_air_has_no_dehumidification_potential")
    if temp < target["target_temp"] + cfg.min_temp_margin_to_target:
        reasons.append("temperature_below_target_margin")
    if temp < cfg.hard_min_temp + cfg.min_temp_margin_to_hard:
        reasons.append("temperature_near_hard_minimum")
    metrics = {
        "rh_gap": rh_gap,
        "vpd_kpa": vpd,
        "air_dew_margin": air_dew_margin,
        "canopy_dew_margin": canopy_dew_margin,
        "target_temp": target["target_temp"],
        "target_rh": target["target_rh"],
        **potential,
    }
    return tuple(reasons), {key: float(value) for key, value in metrics.items()}


def risk_based_economic_intent_override(
    row: Dict[str, Any],
    metrics: Dict[str, float],
    context_reasons: Sequence[str],
    config: HumidityExperienceConfig,
) -> bool:
    """Detect moderate-risk states where the plan target is over-aggressive.

    This is intentionally narrow. It can only relax target-pressure and intent
    checks; it never bypasses dew, outside-air, temperature, action, or outcome
    safety checks.
    """

    if not config.allow_risk_based_intent_override:
        return False
    hard_blockers = {
        "not_moderate_rh",
        "vpd_outside_safe_economic_band",
        "air_dew_margin_too_low",
        "canopy_dew_margin_too_low",
        "outside_air_has_no_dehumidification_potential",
        "temperature_below_target_margin",
        "temperature_near_hard_minimum",
    }
    if any(reason in hard_blockers for reason in context_reasons):
        return False
    rh = _float(row, "rh_air", 70.0)
    return bool(
        rh <= config.risk_override_max_rh
        and metrics.get("rh_gap", 0.0) <= config.risk_override_max_rh_gap
        and metrics.get("vpd_kpa", 0.0) >= config.risk_override_min_vpd_kpa
        and metrics.get("air_dew_margin", 0.0) >= config.risk_override_min_air_dew_margin
    )


def assess_free_air_exchange_pair(
    ppo_row: Dict[str, Any],
    llm_row: Dict[str, Any],
    config: Optional[HumidityExperienceConfig] = None,
) -> ExperienceAssessment:
    cfg = config or HumidityExperienceConfig()
    reasons: List[str] = []
    context_reasons, metrics = screen_free_air_exchange_context(llm_row, target_row=llm_row, config=cfg)
    risk_override = risk_based_economic_intent_override(llm_row, metrics, context_reasons, cfg)
    if risk_override:
        context_reasons = tuple(reason for reason in context_reasons if reason != "target_rh_gap_outside_band")
        metrics["risk_based_intent_override"] = 1.0
    else:
        metrics["risk_based_intent_override"] = 0.0
    reasons.extend(context_reasons)

    ppo_temp = _float(ppo_row, "temp_air", _float(llm_row, "temp_air", 20.0))
    ppo_rh = _float(ppo_row, "rh_air", _float(llm_row, "rh_air", 70.0))
    llm_temp = _float(llm_row, "temp_air", ppo_temp)
    llm_rh = _float(llm_row, "rh_air", ppo_rh)
    if abs(ppo_temp - llm_temp) > cfg.max_state_temp_gap:
        reasons.append("ppo_llm_temperature_state_gap_too_large")
    if abs(ppo_rh - llm_rh) > cfg.max_state_rh_gap:
        reasons.append("ppo_llm_rh_state_gap_too_large")

    intent_label, intent_confidence = infer_intent_from_rows(llm_row, llm_row)
    if risk_override and intent_label != "economic_dehumidify":
        metrics["plan_intent_confidence_before_override"] = float(intent_confidence)
        metrics["plan_intent_overridden"] = 1.0
        intent_label = "economic_dehumidify"
        intent_confidence = max(float(intent_confidence), 0.65)
    else:
        metrics["plan_intent_overridden"] = 0.0
    if intent_label != "economic_dehumidify":
        reasons.append("intent_is_not_economic_dehumidify")
    if intent_confidence < cfg.min_intent_confidence:
        reasons.append("intent_confidence_too_low")

    strategy_label, strategy_confidence = infer_strategy_from_row(ppo_row, min_confidence=cfg.min_strategy_confidence)
    if strategy_label != "free_air_exchange":
        reasons.append("ppo_strategy_is_not_free_air_exchange")
    if strategy_confidence < cfg.min_strategy_confidence:
        reasons.append("ppo_strategy_confidence_too_low")

    ppo_action = action_from_row(ppo_row)
    heat, co2, _, vent, lamp, _ = [float(x) for x in ppo_action]
    if vent < cfg.min_ventilation:
        reasons.append("ppo_ventilation_too_low")
    if heat > cfg.max_heating:
        reasons.append("ppo_heating_too_high")
    if co2 > cfg.max_co2:
        reasons.append("ppo_co2_too_high")
    if lamp > cfg.max_lighting:
        reasons.append("ppo_lighting_too_high")
    if heat * vent > cfg.max_heat_vent_conflict:
        reasons.append("ppo_heat_vent_conflict_too_high")

    ppo_cost = operating_cost(ppo_row)
    llm_cost = operating_cost(llm_row)
    profit_adv = _float(ppo_row, "profit", 0.0) - _float(llm_row, "profit", 0.0)
    cost_saving = llm_cost - ppo_cost
    rh_extra = _float(ppo_row, "rh_violation", 0.0) - _float(llm_row, "rh_violation", 0.0)
    temp_extra = _float(ppo_row, "temp_violation", 0.0) - _float(llm_row, "temp_violation", 0.0)
    metrics.update(
        {
            "profit_advantage_ppo_minus_llm": profit_adv,
            "operating_cost_saving_llm_minus_ppo": cost_saving,
            "rh_violation_extra_ppo_minus_llm": rh_extra,
            "temp_violation_extra_ppo_minus_llm": temp_extra,
            "ppo_action_proxy_cost": action_proxy_cost(ppo_action),
            "ppo_heat_vent_conflict": heat_vent_conflict(ppo_action),
            "llm_action_proxy_cost": action_proxy_cost(action_from_row(llm_row)),
            "llm_heat_vent_conflict": heat_vent_conflict(action_from_row(llm_row)),
            "ppo_llm_temp_gap": ppo_temp - llm_temp,
            "ppo_llm_rh_gap": ppo_rh - llm_rh,
        }
    )
    if profit_adv < cfg.min_profit_advantage and cost_saving < cfg.min_cost_saving:
        reasons.append("no_economic_advantage")
    if rh_extra > cfg.max_rh_violation_extra:
        reasons.append("rh_violation_worse_than_llm")
    if temp_extra > cfg.max_temp_violation_extra:
        reasons.append("temp_violation_worse_than_llm")

    return ExperienceAssessment(
        accepted=not reasons,
        reasons=tuple(reasons),
        metrics={key: float(value) for key, value in metrics.items()},
        intent_label=intent_label,
        intent_confidence=float(intent_confidence),
        strategy_label=strategy_label,
        strategy_confidence=float(strategy_confidence),
    )


def trust_score(assessment: ExperienceAssessment) -> float:
    metrics = assessment.metrics
    if not assessment.accepted:
        return 0.0
    profit = max(0.0, metrics.get("profit_advantage_ppo_minus_llm", 0.0))
    cost = max(0.0, metrics.get("operating_cost_saving_llm_minus_ppo", 0.0))
    ah = max(0.0, metrics.get("absolute_humidity_advantage", 0.0))
    rh_extra = max(0.0, metrics.get("rh_violation_extra_ppo_minus_llm", 0.0))
    temp_extra = max(0.0, metrics.get("temp_violation_extra_ppo_minus_llm", 0.0))
    score = 0.35
    score += 0.20 * np.clip(assessment.intent_confidence, 0.0, 1.0)
    score += 0.20 * np.clip(assessment.strategy_confidence, 0.0, 1.0)
    score += 0.10 * np.clip(ah / 2.0, 0.0, 1.0)
    score += 0.10 * np.clip((profit + cost) / 0.0005, 0.0, 1.0)
    score -= 0.20 * np.clip((rh_extra + temp_extra) / 0.05, 0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def make_case_id(source: str, scenario: Dict[str, Any], residual: Sequence[float]) -> str:
    payload = json.dumps(
        {
            "source": source,
            "scenario": scenario,
            "residual": [float(x) for x in residual],
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def build_experience_from_pair(
    ppo_row: Dict[str, Any],
    llm_row: Dict[str, Any],
    source: str = "ppo_vs_llm",
    config: Optional[HumidityExperienceConfig] = None,
) -> Tuple[Optional[HumidityExperience], ExperienceAssessment]:
    assessment = assess_free_air_exchange_pair(ppo_row, llm_row, config=config)
    if not assessment.accepted:
        return None, assessment
    state = row_to_state(llm_row)
    plan = plan_from_rows(llm_row, llm_row)
    intent = infer_plan_intent(state, plan)
    base_action = target_tracking_baseline_control(state, plan, intent)
    ppo_action = action_from_row(ppo_row)
    llm_action = action_from_row(llm_row)
    residual = np.clip(ppo_action - base_action, -1.0, 1.0)
    scenario = {
        "year": int(_float(llm_row, "year", _float(ppo_row, "year", 0.0))),
        "day": int(_float(llm_row, "day", _float(ppo_row, "day", 0.0))),
        "seed": int(_float(llm_row, "seed", _float(ppo_row, "seed", 0.0))),
        "step": int(_float(llm_row, "step", _float(ppo_row, "step", 0.0))),
        "timestep": int(_float(llm_row, "timestep", _float(ppo_row, "timestep", 0.0))),
    }
    context = build_context(llm_row, llm_row)
    target = target_from_rows(llm_row, llm_row)
    case_id = make_case_id(source, scenario, residual)
    experience = HumidityExperience(
        case_id=case_id,
        status="pending",
        source=source,
        scenario=scenario,
        intent_label=assessment.intent_label,
        intent_confidence=assessment.intent_confidence,
        strategy_label=assessment.strategy_label,
        strategy_confidence=assessment.strategy_confidence,
        context=context,
        context_vector=[float(x) for x in context_vector(context)],
        target=target,
        base_action=[float(x) for x in base_action],
        expert_action=[float(x) for x in ppo_action],
        baseline_action=[float(x) for x in llm_action],
        residual_action=[float(x) for x in residual],
        evidence=assessment.metrics,
        trust=trust_score(assessment),
        support_count=1,
        supporting_cases=[case_id],
    )
    return experience, assessment


class HumidityExperienceMemory:
    def __init__(self, experiences: Optional[Iterable[HumidityExperience]] = None, config: Optional[HumidityExperienceConfig] = None):
        self.config = config or HumidityExperienceConfig()
        self.experiences: List[HumidityExperience] = []
        for experience in experiences or []:
            self.add(experience)

    def add(self, experience: HumidityExperience, merge: bool = True) -> None:
        if merge:
            match = self._find_merge_target(experience)
            if match is not None:
                self._merge_into(match, experience)
                return
        self.experiences.append(experience)

    def _find_merge_target(self, experience: HumidityExperience) -> Optional[HumidityExperience]:
        best: Optional[Tuple[float, HumidityExperience]] = None
        for existing in self.experiences:
            if existing.intent_label != experience.intent_label or existing.strategy_label != experience.strategy_label:
                continue
            distance = self.distance_between(existing.context_vector, experience.context_vector)
            if distance <= self.config.merge_max_distance and (best is None or distance < best[0]):
                best = (distance, existing)
        return None if best is None else best[1]

    @staticmethod
    def distance_between(left: Sequence[float], right: Sequence[float]) -> float:
        a = np.asarray(left, dtype=np.float32)
        b = np.asarray(right, dtype=np.float32)
        if a.shape != b.shape or a.size == 0:
            return float("inf")
        return float(np.sqrt(np.mean((a - b) ** 2)))

    @staticmethod
    def _weighted_average(left: Sequence[float], left_n: int, right: Sequence[float], right_n: int) -> List[float]:
        a = np.asarray(left, dtype=np.float32)
        b = np.asarray(right, dtype=np.float32)
        out = (a * float(left_n) + b * float(right_n)) / float(left_n + right_n)
        return [float(x) for x in out]

    def _merge_into(self, target: HumidityExperience, incoming: HumidityExperience) -> None:
        n0 = int(target.support_count)
        n1 = int(incoming.support_count)
        target.context_vector = self._weighted_average(target.context_vector, n0, incoming.context_vector, n1)
        target.base_action = self._weighted_average(target.base_action, n0, incoming.base_action, n1)
        target.expert_action = self._weighted_average(target.expert_action, n0, incoming.expert_action, n1)
        target.baseline_action = self._weighted_average(target.baseline_action, n0, incoming.baseline_action, n1)
        target.residual_action = self._weighted_average(target.residual_action, n0, incoming.residual_action, n1)
        target.trust = float(np.clip((target.trust * n0 + incoming.trust * n1) / (n0 + n1), 0.0, 1.0))
        target.support_count = n0 + n1
        target.supporting_cases = sorted(set(target.supporting_cases + incoming.supporting_cases))
        for key, value in incoming.evidence.items():
            if key in target.evidence:
                target.evidence[key] = float((target.evidence[key] * n0 + float(value) * n1) / (n0 + n1))
            else:
                target.evidence[key] = float(value)

    def retrieve(
        self,
        row: Dict[str, Any],
        target_row: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
        max_distance: Optional[float] = None,
        min_trust: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        cfg = self.config
        max_distance = cfg.retrieval_max_distance if max_distance is None else float(max_distance)
        min_trust = cfg.min_trust if min_trust is None else float(min_trust)
        context_reasons, context_metrics = screen_free_air_exchange_context(row, target_row=target_row or row, config=cfg)
        risk_override = risk_based_economic_intent_override(row, context_metrics, context_reasons, cfg)
        if risk_override:
            context_reasons = tuple(reason for reason in context_reasons if reason != "target_rh_gap_outside_band")
            context_metrics["risk_based_intent_override"] = 1.0
        else:
            context_metrics["risk_based_intent_override"] = 0.0
        intent_label, intent_confidence = infer_intent_from_rows(row, target_row or row)
        if risk_override and intent_label != "economic_dehumidify":
            intent_label = "economic_dehumidify"
            intent_confidence = max(float(intent_confidence), 0.65)
        if intent_label != "economic_dehumidify":
            context_reasons = tuple(list(context_reasons) + ["query_intent_is_not_economic_dehumidify"])
        if intent_confidence < cfg.min_intent_confidence:
            context_reasons = tuple(list(context_reasons) + ["query_intent_confidence_too_low"])
        if context_reasons:
            return []

        state = row_to_state(row)
        plan = plan_from_rows(row, target_row or row)
        intent = infer_plan_intent(state, plan)
        base_action = target_tracking_baseline_control(state, plan, intent)
        query_vec = context_vector(build_context(row, target_row or row))
        matches: List[Dict[str, Any]] = []
        for experience in self.experiences:
            if experience.status not in {"pending", "validated"}:
                continue
            if experience.intent_label != "economic_dehumidify" or experience.strategy_label != "free_air_exchange":
                continue
            if experience.trust < min_trust:
                continue
            distance = self.distance_between(query_vec, experience.context_vector)
            if distance > max_distance:
                continue
            residual = np.asarray(experience.residual_action, dtype=np.float32)
            candidate = np.clip(base_action + residual, 0.0, 1.0)
            matches.append(
                {
                    "case_id": experience.case_id,
                    "distance": float(distance),
                    "trust": float(experience.trust),
                    "support_count": int(experience.support_count),
                    "status": experience.status,
                    "candidate_action": [float(x) for x in candidate],
                    "base_action": [float(x) for x in base_action],
                    "residual_action": [float(x) for x in residual],
                    "context_metrics": context_metrics,
                    "score": float(experience.trust - 0.20 * distance),
                }
            )
        return sorted(matches, key=lambda item: item["score"], reverse=True)[: int(top_k)]

    def to_records(self) -> List[Dict[str, Any]]:
        return [experience.to_record() for experience in self.experiences]

    def save_jsonl(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for record in self.to_records():
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path, config: Optional[HumidityExperienceConfig] = None) -> "HumidityExperienceMemory":
        records: List[HumidityExperience] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(HumidityExperience.from_record(json.loads(line)))
        return cls(records, config=config)
