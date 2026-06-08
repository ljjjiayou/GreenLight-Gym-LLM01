"""Lightweight Intent Contract adapter for LLM-RSPC planning.

The contract is a high-level, auditable planning interface. In this MVP it is
derived from the existing setpoint plan so the current controller behavior stays
unchanged while downstream code can begin consuming structured intent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from gl_gym.common.utils import calculate_vpd_kpa


TARGET_LIMITS: Dict[str, Tuple[float, float]] = {
    "temp": (11.5, 24.0),
    "co2": (380.0, 850.0),
    "rh": (62.0, 88.0),
}

TARGET_WIDTHS: Dict[str, float] = {
    "temp": 0.75,
    "co2": 80.0,
    "rh": 3.0,
}

DEFAULT_CONSTRAINTS: Dict[str, float] = {
    "rh_hard_max": 90.0,
    "dew_margin_min": 1.0,
    "canopy_dew_margin_min": 1.0,
    "vpd_min": 0.35,
    "vpd_max": 1.60,
    "temp_min": 11.5,
    "temp_max": 32.0,
    "wind_vent_soft_cap": 0.85,
    "lamp_budget_soft_cap": 0.45,
    "forbid_co2_when_vent_gt": 0.25,
}

DEFAULT_PROFILE_SHAPE: Dict[str, str] = {
    "temp": "constant_hold",
    "co2": "daylight_gate",
    "rh": "constant_hold",
}

KNOWN_REGIMES = {
    "economy_hold",
    "daytime_growth",
    "co2_day_boost",
    "lighting_assist",
    "high_humidity_recovery",
    "hot_humid_relief",
    "cold_humid_recovery",
    "hot_dry_relief",
    "cold_dry_protect",
    "radiation_spike_relief",
    "wind_limited_safety",
    "shade_cooling",
    "night_heat_hold",
    "dawn_predehumidify",
    "cold_recovery",
}


@dataclass(frozen=True)
class IntentContract:
    regime: str
    target_range: Dict[str, Tuple[float, float]]
    priority: Tuple[str, ...]
    constraints: Dict[str, float]
    profile_shape: Dict[str, str]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": str(self.regime),
            "target_range": {
                key: [float(bounds[0]), float(bounds[1])]
                for key, bounds in sorted(self.target_range.items())
            },
            "priority": list(self.priority),
            "constraints": {key: float(value) for key, value in sorted(self.constraints.items())},
            "profile_shape": {key: str(value) for key, value in sorted(self.profile_shape.items())},
            "confidence": float(self.confidence),
        }


def _float_attr(obj: Any, name: str, default: float) -> float:
    try:
        value = getattr(obj, name, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _float_value(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _first_finite_attr(obj: Any, names: Sequence[str], default: float) -> float:
    for name in names:
        if not hasattr(obj, name):
            continue
        try:
            value = float(getattr(obj, name))
        except Exception:
            continue
        if np.isfinite(value):
            return float(value)
    return float(default)


def _current_profile_target(plan: Mapping[str, Any], key: str, state: Any) -> Optional[float]:
    profile = plan.get("target_profile", {}) if isinstance(plan, Mapping) else {}
    values = profile.get(key) if isinstance(profile, Mapping) else None
    if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
        created = int(_float_value(plan.get("created_timestep"), _float_attr(state, "timestep", 0.0)))
        idx = int(np.clip(int(_float_attr(state, "timestep", 0.0)) - created, 0, len(values) - 1))
        try:
            return float(values[idx])
        except Exception:
            pass
    try:
        value = plan.get(key)
        return None if value is None else float(value)
    except Exception:
        return None


def _fallback_setpoints(state: Any) -> Dict[str, float]:
    temp = _float_attr(state, "temp_air", 20.0)
    rh = _float_attr(state, "rh_air", 70.0)
    hour = _float_attr(state, "hour_of_day", 12.0)
    rad = _float_attr(state, "glob_rad", 0.0)
    is_day = 6.0 <= hour <= 18.0

    if temp < 12.5:
        target_temp = 15.0
    elif is_day and rad > 180.0:
        target_temp = 18.5
    elif is_day:
        target_temp = 17.0
    else:
        target_temp = 14.0

    target_co2 = 620.0 if is_day and rad > 140.0 else 430.0

    if rh > 88.0:
        target_rh = 72.0
    elif rh < 60.0:
        target_rh = 70.0
    else:
        target_rh = 76.0

    return {
        "temp": float(np.clip(target_temp, *TARGET_LIMITS["temp"])),
        "co2": float(np.clip(target_co2, *TARGET_LIMITS["co2"])),
        "rh": float(np.clip(target_rh, *TARGET_LIMITS["rh"])),
    }


def _setpoints_from_plan(state: Any, plan: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    fallback = _fallback_setpoints(state)
    if not isinstance(plan, Mapping):
        return fallback
    key_map = {"temp": "target_temp", "co2": "target_co2", "rh": "target_rh"}
    out: Dict[str, float] = {}
    for short_key, plan_key in key_map.items():
        target = _current_profile_target(plan, plan_key, state)
        low, high = TARGET_LIMITS[short_key]
        out[short_key] = float(np.clip(fallback[short_key] if target is None else target, low, high))
    return out


def _range_around(value: float, key: str) -> Tuple[float, float]:
    low, high = TARGET_LIMITS[key]
    width = TARGET_WIDTHS[key]
    return (
        float(np.clip(float(value) - width, low, high)),
        float(np.clip(float(value) + width, low, high)),
    )


def _state_context(state: Any, setpoints: Mapping[str, float]) -> Dict[str, float]:
    temp = _float_attr(state, "temp_air", 20.0)
    rh = _float_attr(state, "rh_air", 70.0)
    rad = _float_attr(state, "glob_rad", 0.0)
    forecast_rad_mean = _first_finite_attr(state, ("forecast_rad_mean_1h",), rad)
    forecast_rad_peak = _first_finite_attr(state, ("forecast_rad_peak_2h",), rad)
    rad_load = max(rad, forecast_rad_mean, forecast_rad_peak)
    hour = _float_attr(state, "hour_of_day", 12.0)
    dew_margin_air = _float_attr(state, "dew_margin_air", 3.0)
    canopy_margin = _float_attr(state, "canopy_dew_margin", 3.0)
    dew_margin = min(dew_margin_air, canopy_margin)
    wind_speed = _float_attr(state, "wind_speed", 1.0)
    temp_out = _float_attr(state, "temp_out", temp)
    rh_out = _float_attr(state, "rh_out", rh)
    temp_rise_1h = _first_finite_attr(
        state,
        ("temp_air_delta_1h", "temp_air_rise_1h", "temp_air_trend_c_per_hour"),
        0.0,
    )
    vpd = float(calculate_vpd_kpa(temp, rh))
    return {
        "temp": temp,
        "rh": rh,
        "rad": rad,
        "rad_load": rad_load,
        "forecast_rad_mean": forecast_rad_mean,
        "forecast_rad_peak": forecast_rad_peak,
        "hour": hour,
        "dew_margin": dew_margin,
        "dew_margin_air": dew_margin_air,
        "canopy_dew_margin": canopy_margin,
        "vpd": vpd,
        "wind_speed": wind_speed,
        "temp_out": temp_out,
        "rh_out": rh_out,
        "temp_rise_1h": temp_rise_1h,
        "target_temp_delta": float(setpoints["temp"] - temp),
        "target_co2_delta": float(setpoints["co2"] - _float_attr(state, "co2_air", 430.0)),
        "target_rh_delta": float(setpoints["rh"] - rh),
    }


def _infer_regime_and_shape(state: Any, setpoints: Mapping[str, float]) -> Tuple[str, Tuple[str, ...], Dict[str, str], float]:
    ctx = _state_context(state, setpoints)
    hour = ctx["hour"]
    is_dawn = 4.0 <= hour <= 8.0
    is_night = hour < 6.0 or hour > 18.0
    is_day = 6.0 <= hour <= 18.0
    high_wind = ctx["wind_speed"] >= 6.0
    dew_risk = ctx["dew_margin"] <= 1.2 or ctx["canopy_dew_margin"] <= 1.2
    cold = ctx["temp"] <= 16.5
    hot = ctx["temp"] >= 30.0
    high_rad = ctx["rad_load"] >= 600.0
    radiation_spike = ctx["rad_load"] >= 650.0 and (ctx["temp_rise_1h"] >= 0.5 or ctx["temp"] >= 29.0)
    hot_dry = (
        ctx["rad_load"] >= 500.0
        and (ctx["temp"] >= 28.0 or ctx["target_temp_delta"] <= -1.0)
        and (ctx["rh"] <= 62.0 or ctx["vpd"] >= 1.35 or ctx["target_rh_delta"] >= 6.0)
    )
    hot_humid = (ctx["temp"] >= 28.0 and (ctx["rh"] >= 85.0 or dew_risk)) or (hot and ctx["vpd"] <= 0.50)
    cold_humid = cold and (ctx["rh"] >= 86.0 or ctx["vpd"] <= 0.35 or dew_risk)
    cold_dry = cold and (ctx["rh"] <= 58.0 or ctx["vpd"] >= 0.85 or ctx["target_rh_delta"] >= 8.0)
    humid_risk = (
        ctx["rh"] >= 88.0
        or dew_risk
        or ctx["target_rh_delta"] <= -6.0
        or (ctx["rh"] >= 84.0 and ctx["vpd"] <= 0.45)
    )
    cooling_need = hot or ctx["target_temp_delta"] <= -2.0
    heating_need = ctx["temp"] <= 15.5 or ctx["target_temp_delta"] >= 2.0
    co2_growth_opportunity = is_day and ctx["rad_load"] >= 160.0 and ctx["target_co2_delta"] >= 80.0 and ctx["rh"] < 86.0
    lighting_opportunity = is_day and ctx["rad_load"] < 90.0 and ctx["rh"] < 82.0 and ctx["temp"] >= 15.0

    profile_shape = dict(DEFAULT_PROFILE_SHAPE)
    priority: Tuple[str, ...] = ("safety", "energy", "growth")
    confidence = 0.62
    regime = "economy_hold"

    if high_wind and (cooling_need or humid_risk or dew_risk):
        regime = "wind_limited_safety"
        profile_shape.update({"temp": "wind_limited_cooling", "rh": "wind_limited_dehumidify", "co2": "off_when_venting"})
        priority = ("safety", "structure", "temperature", "humidity", "energy")
        confidence = 0.86
    elif hot_humid:
        regime = "hot_humid_relief"
        profile_shape.update({"temp": "shade_cooling", "rh": "strict_dehumidify_then_relax", "co2": "off_when_venting"})
        priority = ("safety", "temperature", "humidity", "energy", "growth")
        confidence = 0.86
    elif cold_humid:
        regime = "cold_humid_recovery"
        profile_shape.update({"temp": "night_heat_hold", "rh": "strict_dehumidify_then_relax", "co2": "off_when_venting"})
        priority = ("safety", "temperature", "humidity", "energy", "growth")
        confidence = 0.84
    elif humid_risk and is_dawn:
        regime = "dawn_predehumidify"
        profile_shape.update({"temp": "dawn_ramp", "rh": "strict_dehumidify_then_relax"})
        priority = ("safety", "humidity", "energy", "growth")
        confidence = 0.82
    elif humid_risk:
        regime = "high_humidity_recovery"
        profile_shape.update({"rh": "strict_dehumidify_then_relax"})
        priority = ("safety", "humidity", "energy", "growth")
        confidence = 0.78
    elif hot_dry:
        regime = "hot_dry_relief"
        profile_shape.update({"temp": "shade_cooling", "rh": "hot_dry_protect"})
        priority = ("safety", "humidity", "temperature", "energy", "growth")
        confidence = 0.76
    elif cold_dry:
        regime = "cold_dry_protect"
        profile_shape.update({"temp": "night_heat_hold", "rh": "cold_dry_protect"})
        priority = ("safety", "temperature", "humidity", "energy", "growth")
        confidence = 0.76
    elif radiation_spike:
        regime = "radiation_spike_relief"
        profile_shape.update({"temp": "shade_cooling", "rh": "hot_dry_protect"})
        priority = ("safety", "temperature", "humidity", "energy", "growth")
        confidence = 0.74
    elif cooling_need and ctx["rad_load"] >= 250.0:
        regime = "shade_cooling"
        profile_shape.update({"temp": "shade_cooling"})
        priority = ("safety", "temperature", "energy", "growth")
        confidence = 0.72
    elif heating_need:
        regime = "night_heat_hold" if is_night else "cold_recovery"
        profile_shape.update({"temp": "night_heat_hold"})
        priority = ("safety", "temperature", "energy", "growth")
        confidence = 0.70
    elif co2_growth_opportunity:
        regime = "co2_day_boost"
        profile_shape.update({"co2": "co2_day_boost"})
        priority = ("safety", "growth", "energy")
        confidence = 0.70
    elif lighting_opportunity:
        regime = "lighting_assist"
        profile_shape.update({"temp": "low_light_hold", "co2": "low_light_gate"})
        priority = ("safety", "growth", "energy")
        confidence = 0.66
    elif is_day and ctx["rad_load"] >= 120.0:
        regime = "daytime_growth"
        profile_shape.update({"co2": "daylight_gate"})
        priority = ("safety", "growth", "energy")
        confidence = 0.65

    return regime, priority, profile_shape, confidence


def intent_contract_from_setpoint_plan(
    state: Any,
    plan: Optional[Mapping[str, Any]] = None,
) -> IntentContract:
    setpoints = _setpoints_from_plan(state, plan)
    regime, priority, profile_shape, confidence = _infer_regime_and_shape(state, setpoints)
    constraints = dict(DEFAULT_CONSTRAINTS)
    if regime == "hot_dry_relief":
        constraints["vpd_max"] = 1.45
        constraints["shade_min_when_rad_gt_600"] = 0.50
    elif regime in {"high_humidity_recovery", "dawn_predehumidify", "hot_humid_relief", "cold_humid_recovery"}:
        constraints["rh_hard_max"] = 88.0
        constraints["dew_margin_min"] = 1.2
        constraints["canopy_dew_margin_min"] = 1.2
        constraints["forbid_co2_when_vent_gt"] = 0.18
    elif regime == "cold_dry_protect":
        constraints["vpd_min"] = 0.25
        constraints["temp_min"] = 14.5
    elif regime == "radiation_spike_relief":
        constraints["temp_max"] = 31.0
        constraints["shade_min_when_rad_gt_600"] = 0.50
    elif regime == "wind_limited_safety":
        constraints["wind_vent_soft_cap"] = 0.55
        constraints["forbid_co2_when_vent_gt"] = 0.12

    return IntentContract(
        regime=regime,
        target_range={key: _range_around(value, key) for key, value in setpoints.items()},
        priority=priority,
        constraints=constraints,
        profile_shape=profile_shape,
        confidence=confidence,
    )


def _normalize_priority(value: Any, fallback: Sequence[str], corrected: list[str]) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        corrected.append("priority:fallback")
        return tuple(fallback)
    priority = tuple(str(item).strip() for item in value if str(item).strip())
    if not priority:
        corrected.append("priority:fallback")
        return tuple(fallback)
    return priority


def _normalize_target_range(
    value: Any,
    fallback: Mapping[str, Tuple[float, float]],
    corrected: list[str],
) -> Dict[str, Tuple[float, float]]:
    if not isinstance(value, Mapping):
        corrected.append("target_range:fallback")
        return dict(fallback)
    out: Dict[str, Tuple[float, float]] = {}
    for key in ("temp", "co2", "rh"):
        raw = value.get(key)
        low_limit, high_limit = TARGET_LIMITS[key]
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            out[key] = tuple(fallback[key])
            corrected.append(f"target_range.{key}:fallback")
            continue
        low = _float_value(raw[0], fallback[key][0])
        high = _float_value(raw[1], fallback[key][1])
        if high < low:
            low, high = high, low
            corrected.append(f"target_range.{key}:sorted")
        clipped = (float(np.clip(low, low_limit, high_limit)), float(np.clip(high, low_limit, high_limit)))
        if clipped != (low, high):
            corrected.append(f"target_range.{key}:clipped")
        out[key] = clipped
    return out


def coerce_intent_contract(
    data: Any,
    state: Any,
    plan: Optional[Mapping[str, Any]] = None,
) -> Tuple[IntentContract, Dict[str, Any]]:
    """Validate an optional intent payload, falling back to setpoint-derived intent."""

    fallback = intent_contract_from_setpoint_plan(state, plan)
    diagnostics: Dict[str, Any] = {
        "schema_version": "intent_contract_mvp_v1",
        "source": "provided",
        "fallback": False,
        "corrected": [],
    }

    if isinstance(data, IntentContract):
        return data, diagnostics
    if not isinstance(data, Mapping):
        diagnostics.update({"source": "setpoint_adapter", "fallback": True, "reason": "missing_or_invalid_payload"})
        return fallback, diagnostics

    corrected: list[str] = []
    regime = str(data.get("regime") or "").strip()
    if regime not in KNOWN_REGIMES:
        regime = fallback.regime
        corrected.append("regime:fallback")

    target_range = _normalize_target_range(data.get("target_range"), fallback.target_range, corrected)
    priority = _normalize_priority(data.get("priority"), fallback.priority, corrected)
    constraints = dict(fallback.constraints)
    if isinstance(data.get("constraints"), Mapping):
        for key, value in data["constraints"].items():
            try:
                constraints[str(key)] = float(value)
            except Exception:
                corrected.append(f"constraints.{key}:drop_non_numeric")
    else:
        corrected.append("constraints:fallback")

    profile_shape = dict(fallback.profile_shape)
    if isinstance(data.get("profile_shape"), Mapping):
        for key, value in data["profile_shape"].items():
            if str(key).strip() and str(value).strip():
                profile_shape[str(key)] = str(value)
    else:
        corrected.append("profile_shape:fallback")

    confidence = float(np.clip(_float_value(data.get("confidence"), fallback.confidence), 0.0, 1.0))
    contract = IntentContract(
        regime=regime,
        target_range=target_range,
        priority=priority,
        constraints=constraints,
        profile_shape=profile_shape,
        confidence=confidence,
    )
    diagnostics["corrected"] = sorted(set(corrected))
    diagnostics["fallback"] = bool(corrected)
    return contract, diagnostics
