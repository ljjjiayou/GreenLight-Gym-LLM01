"""Deterministic target-profile candidates for Intent-RSPC.

This module is shadow-only in v1: it turns an IntentContract into auditable
target-profile candidates, but it does not select actuator controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gl_gym.agent.intent_contract import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_PROFILE_SHAPE,
    IntentContract,
    TARGET_LIMITS,
    coerce_intent_contract,
    intent_contract_from_setpoint_plan,
)
from gl_gym.common.utils import calculate_vpd_kpa


PROFILE_KEYS = ("target_temp", "target_co2", "target_rh")
SHORT_KEYS = {"target_temp": "temp", "target_co2": "co2", "target_rh": "rh"}
SAFETY_GATE_REASONS = {"none", "temp_high_gate", "dew_gate", "canopy_gate", "rh_high_gate"}


@dataclass(frozen=True)
class ProfileCandidate:
    name: str
    regime: str
    target_profile: Dict[str, List[float]]
    priority: Tuple[str, ...]
    constraints: Dict[str, float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "regime": str(self.regime),
            "target_profile": {
                key: [float(value) for value in self.target_profile[key]]
                for key in PROFILE_KEYS
            },
            "priority": list(self.priority),
            "constraints": {key: float(value) for key, value in sorted(self.constraints.items())},
            "reason": str(self.reason),
        }


@dataclass(frozen=True)
class ProfileCandidateScore:
    name: str
    total_score: float
    breakdown: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "total_score": float(round(self.total_score, 6)),
            "breakdown": {key: float(round(value, 6)) for key, value in sorted(self.breakdown.items())},
        }


def _float_attr(obj: Any, name: str, default: float) -> float:
    try:
        value = getattr(obj, name, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _clip_series(values: Sequence[float], key: str, horizon: int) -> List[float]:
    low, high = TARGET_LIMITS[SHORT_KEYS[key]]
    data = [float(np.clip(float(value), low, high)) for value in values]
    if not data:
        data = [float((low + high) / 2.0)]
    if len(data) < horizon:
        data.extend([data[-1]] * (horizon - len(data)))
    elif len(data) > horizon:
        data = data[:horizon]
    return data


def _lin(start: float, end: float, horizon: int) -> List[float]:
    if horizon <= 1:
        return [float(end)]
    return [float(value) for value in np.linspace(float(start), float(end), int(horizon))]


def _bounds(intent: IntentContract, key: str) -> Tuple[float, float, float]:
    low_limit, high_limit = TARGET_LIMITS[key]
    raw = intent.target_range.get(key, (low_limit, high_limit))
    low = float(np.clip(float(raw[0]), low_limit, high_limit))
    high = float(np.clip(float(raw[1]), low_limit, high_limit))
    if high < low:
        low, high = high, low
    return low, high, float((low + high) / 2.0)


def _profile(
    intent: IntentContract,
    horizon: int,
    *,
    temp: Sequence[float],
    co2: Sequence[float],
    rh: Sequence[float],
) -> Dict[str, List[float]]:
    return {
        "target_temp": _clip_series(temp, "target_temp", horizon),
        "target_co2": _clip_series(co2, "target_co2", horizon),
        "target_rh": _clip_series(rh, "target_rh", horizon),
    }


def _constant_hold(intent: IntentContract, state: Any, horizon: int) -> Tuple[Dict[str, List[float]], str]:
    _t_low, _t_high, temp_mid = _bounds(intent, "temp")
    _c_low, _c_high, co2_mid = _bounds(intent, "co2")
    _r_low, _r_high, rh_mid = _bounds(intent, "rh")
    return _profile(intent, horizon, temp=[temp_mid] * horizon, co2=[co2_mid] * horizon, rh=[rh_mid] * horizon), (
        "hold setpoint-range midpoints"
    )


def _hot_dry_protect(intent: IntentContract, state: Any, horizon: int) -> Tuple[Dict[str, List[float]], str]:
    temp_low, _temp_high, temp_mid = _bounds(intent, "temp")
    co2_low, _co2_high, _co2_mid = _bounds(intent, "co2")
    _rh_low, rh_high, rh_mid = _bounds(intent, "rh")
    temp_target = min(temp_mid, temp_low + 0.35)
    rh_target = max(rh_mid, rh_high - 0.5)
    return _profile(
        intent,
        horizon,
        temp=[temp_target] * horizon,
        co2=[min(co2_low + 30.0, 430.0)] * horizon,
        rh=[rh_target] * horizon,
    ), "retain humidity under heat load while asking for cooler profile"


def _shade_cooling(intent: IntentContract, state: Any, horizon: int) -> Tuple[Dict[str, List[float]], str]:
    temp_low, _temp_high, temp_mid = _bounds(intent, "temp")
    co2_low, _co2_high, _co2_mid = _bounds(intent, "co2")
    _rh_low, _rh_high, rh_mid = _bounds(intent, "rh")
    start_temp = min(temp_mid, _float_attr(state, "temp_air", temp_mid))
    return _profile(
        intent,
        horizon,
        temp=_lin(start_temp, temp_low, horizon),
        co2=[min(co2_low + 30.0, 430.0)] * horizon,
        rh=[rh_mid] * horizon,
    ), "radiation-aware cooling target trajectory"


def _strict_dehumidify_then_relax(intent: IntentContract, state: Any, horizon: int) -> Tuple[Dict[str, List[float]], str]:
    _temp_low, _temp_high, temp_mid = _bounds(intent, "temp")
    co2_low, _co2_high, _co2_mid = _bounds(intent, "co2")
    rh_low, _rh_high, rh_mid = _bounds(intent, "rh")
    strict_steps = max(1, int(np.ceil(horizon / 3.0)))
    rh_values = [rh_low] * strict_steps + _lin(rh_low, rh_mid, max(1, horizon - strict_steps))
    return _profile(
        intent,
        horizon,
        temp=[temp_mid] * horizon,
        co2=[min(co2_low + 30.0, 430.0)] * horizon,
        rh=rh_values,
    ), "start with stricter RH target, then relax toward midpoint"


def _night_heat_hold(intent: IntentContract, state: Any, horizon: int) -> Tuple[Dict[str, List[float]], str]:
    _temp_low, temp_high, temp_mid = _bounds(intent, "temp")
    co2_low, _co2_high, _co2_mid = _bounds(intent, "co2")
    _rh_low, rh_high, rh_mid = _bounds(intent, "rh")
    temp_target = max(temp_mid, temp_high - 0.25)
    rh_target = min(rh_high, max(rh_mid, 76.0))
    return _profile(
        intent,
        horizon,
        temp=[temp_target] * horizon,
        co2=[min(co2_low + 30.0, 430.0)] * horizon,
        rh=[rh_target] * horizon,
    ), "hold warmth with conservative CO2 and moderate humidity"


def _dawn_predehumidify(intent: IntentContract, state: Any, horizon: int) -> Tuple[Dict[str, List[float]], str]:
    _temp_low, temp_high, temp_mid = _bounds(intent, "temp")
    co2_low, _co2_high, _co2_mid = _bounds(intent, "co2")
    rh_low, _rh_high, rh_mid = _bounds(intent, "rh")
    temp_values = _lin(temp_mid, max(temp_mid, temp_high - 0.25), horizon)
    rh_values = _lin(rh_low, min(rh_mid, rh_low + 2.0), horizon)
    return _profile(
        intent,
        horizon,
        temp=temp_values,
        co2=[min(co2_low + 30.0, 430.0)] * horizon,
        rh=rh_values,
    ), "pre-dawn humidity relief with gentle temperature ramp"


def _cold_humid_recovery(intent: IntentContract, state: Any, horizon: int) -> Tuple[Dict[str, List[float]], str]:
    _temp_low, temp_high, temp_mid = _bounds(intent, "temp")
    co2_low, _co2_high, _co2_mid = _bounds(intent, "co2")
    rh_low, _rh_high, rh_mid = _bounds(intent, "rh")
    temp_values = _lin(max(temp_mid, _float_attr(state, "temp_air", temp_mid)), temp_high, horizon)
    rh_values = _lin(rh_low, rh_mid, horizon)
    return _profile(
        intent,
        horizon,
        temp=temp_values,
        co2=[min(co2_low + 30.0, 430.0)] * horizon,
        rh=rh_values,
    ), "recover temperature while reducing high humidity risk"


def _cold_dry_protect(intent: IntentContract, state: Any, horizon: int) -> Tuple[Dict[str, List[float]], str]:
    _temp_low, temp_high, temp_mid = _bounds(intent, "temp")
    co2_low, _co2_high, _co2_mid = _bounds(intent, "co2")
    _rh_low, rh_high, rh_mid = _bounds(intent, "rh")
    temp_values = _lin(temp_mid, temp_high, horizon)
    rh_target = max(rh_mid, rh_high - 0.5)
    return _profile(
        intent,
        horizon,
        temp=temp_values,
        co2=[min(co2_low + 30.0, 430.0)] * horizon,
        rh=[rh_target] * horizon,
    ), "protect cold dry crop state without aggressive dehumidification"


def _co2_day_boost(intent: IntentContract, state: Any, horizon: int) -> Tuple[Dict[str, List[float]], str]:
    _temp_low, _temp_high, temp_mid = _bounds(intent, "temp")
    _co2_low, co2_high, co2_mid = _bounds(intent, "co2")
    _rh_low, _rh_high, rh_mid = _bounds(intent, "rh")
    return _profile(
        intent,
        horizon,
        temp=[temp_mid] * horizon,
        co2=_lin(co2_mid, co2_high, horizon),
        rh=[rh_mid] * horizon,
    ), "use daylight growth opportunity for gradual CO2 target increase"


def _lighting_assist(intent: IntentContract, state: Any, horizon: int) -> Tuple[Dict[str, List[float]], str]:
    _temp_low, temp_high, temp_mid = _bounds(intent, "temp")
    co2_low, _co2_high, co2_mid = _bounds(intent, "co2")
    _rh_low, _rh_high, rh_mid = _bounds(intent, "rh")
    temp_target = min(temp_high, max(temp_mid, 16.5))
    co2_target = min(co2_mid, co2_low + 80.0)
    return _profile(
        intent,
        horizon,
        temp=[temp_target] * horizon,
        co2=[co2_target] * horizon,
        rh=[rh_mid] * horizon,
    ), "low-light support without aggressive CO2 or humidity movement"


SHAPE_BUILDERS = {
    "constant_hold": _constant_hold,
    "hot_dry_protect": _hot_dry_protect,
    "shade_cooling": _shade_cooling,
    "strict_dehumidify_then_relax": _strict_dehumidify_then_relax,
    "night_heat_hold": _night_heat_hold,
    "dawn_predehumidify": _dawn_predehumidify,
    "cold_humid_recovery": _cold_humid_recovery,
    "cold_dry_protect": _cold_dry_protect,
    "co2_day_boost": _co2_day_boost,
    "lighting_assist": _lighting_assist,
}


REGIME_SHAPES = {
    "economy_hold": ("constant_hold",),
    "daytime_growth": ("constant_hold", "co2_day_boost"),
    "co2_day_boost": ("constant_hold", "co2_day_boost"),
    "lighting_assist": ("constant_hold", "lighting_assist"),
    "high_humidity_recovery": ("constant_hold", "strict_dehumidify_then_relax"),
    "hot_humid_relief": ("constant_hold", "shade_cooling", "strict_dehumidify_then_relax"),
    "cold_humid_recovery": ("constant_hold", "cold_humid_recovery", "strict_dehumidify_then_relax"),
    "hot_dry_relief": ("constant_hold", "hot_dry_protect", "shade_cooling"),
    "cold_dry_protect": ("constant_hold", "cold_dry_protect", "night_heat_hold"),
    "radiation_spike_relief": ("constant_hold", "shade_cooling", "hot_dry_protect"),
    "wind_limited_safety": ("constant_hold", "shade_cooling", "strict_dehumidify_then_relax"),
    "shade_cooling": ("constant_hold", "shade_cooling"),
    "night_heat_hold": ("constant_hold", "night_heat_hold"),
    "dawn_predehumidify": ("constant_hold", "dawn_predehumidify", "strict_dehumidify_then_relax"),
    "cold_recovery": ("constant_hold", "night_heat_hold"),
}


def _constant_fallback_contract(state: Any, current_plan: Optional[Mapping[str, Any]]) -> IntentContract:
    base = intent_contract_from_setpoint_plan(state, current_plan)
    return IntentContract(
        regime="economy_hold",
        target_range=base.target_range,
        priority=("safety", "energy", "growth"),
        constraints=dict(DEFAULT_CONSTRAINTS),
        profile_shape=dict(DEFAULT_PROFILE_SHAPE),
        confidence=0.50,
    )


def _coerce_for_profile_generation(
    intent: Any,
    state: Any,
    current_plan: Optional[Mapping[str, Any]],
) -> Tuple[IntentContract, Dict[str, Any]]:
    if isinstance(intent, IntentContract):
        return intent, {"source": "intent_contract", "fallback": False, "corrected": []}
    if not isinstance(intent, Mapping):
        return _constant_fallback_contract(state, current_plan), {
            "source": "constant_fallback",
            "fallback": True,
            "reason": "missing_or_invalid_intent",
            "corrected": [],
        }
    contract, diagnostics = coerce_intent_contract(intent, state, current_plan)
    corrected = list(diagnostics.get("corrected", []) or [])
    if "regime:fallback" in corrected:
        return _constant_fallback_contract(state, current_plan), {
            "source": "constant_fallback",
            "fallback": True,
            "reason": "invalid_regime",
            "corrected": corrected,
        }
    diagnostics["source"] = "provided"
    return contract, diagnostics


def _requested_shape_names(intent: IntentContract) -> Tuple[List[str], List[str]]:
    names: List[str] = ["constant_hold"]
    unsupported: List[str] = []
    for name in REGIME_SHAPES.get(intent.regime, ("constant_hold",)):
        if name not in names:
            names.append(name)
    for raw in intent.profile_shape.values():
        name = str(raw).strip()
        if not name:
            continue
        if name in SHAPE_BUILDERS:
            if name not in names:
                names.append(name)
        elif name not in unsupported:
            unsupported.append(name)
    return names, unsupported


def build_profile_candidates(
    intent: Any,
    state: Any,
    current_plan: Optional[Mapping[str, Any]] = None,
    horizon_steps: int = 12,
) -> List[ProfileCandidate]:
    contract, _diagnostics = _coerce_for_profile_generation(intent, state, current_plan)
    horizon = max(1, int(horizon_steps))
    names, _unsupported = _requested_shape_names(contract)
    candidates: List[ProfileCandidate] = []
    for name in names:
        builder = SHAPE_BUILDERS.get(name)
        if builder is None:
            continue
        target_profile, reason = builder(contract, state, horizon)
        candidates.append(
            ProfileCandidate(
                name=name,
                regime=contract.regime,
                target_profile=target_profile,
                priority=tuple(contract.priority),
                constraints=dict(contract.constraints),
                reason=reason,
            )
        )
    return candidates or [
        ProfileCandidate(
            name="constant_hold",
            regime=contract.regime,
            target_profile=_constant_hold(contract, state, horizon)[0],
            priority=tuple(contract.priority),
            constraints=dict(contract.constraints),
            reason="fallback constant profile",
        )
    ]


def selector_score(candidate: ProfileCandidate, intent: IntentContract) -> float:
    score = 0.0
    requested = {str(value) for value in intent.profile_shape.values()}
    if candidate.name in requested:
        score += 3.0
    if candidate.name in REGIME_SHAPES.get(intent.regime, ()):
        score += 1.5
    priorities = set(intent.priority)
    if "humidity" in priorities and candidate.name in {
        "hot_dry_protect",
        "strict_dehumidify_then_relax",
        "dawn_predehumidify",
        "cold_humid_recovery",
        "cold_dry_protect",
    }:
        score += 0.8
    if "temperature" in priorities and candidate.name in {
        "shade_cooling",
        "night_heat_hold",
        "cold_humid_recovery",
        "cold_dry_protect",
    }:
        score += 0.8
    if "growth" in priorities and candidate.name in {"co2_day_boost", "lighting_assist"}:
        score += 0.5
    if "safety" in priorities and candidate.name == "constant_hold":
        score += 0.1
    return float(score)


def _first_finite_attr(obj: Any, names: Sequence[str], default: float) -> float:
    for name in names:
        try:
            value = float(getattr(obj, name))
        except Exception:
            continue
        if np.isfinite(value):
            return float(value)
    return float(default)


def _score_context(state: Any, intent: IntentContract) -> Dict[str, Any]:
    temp = _float_attr(state, "temp_air", 20.0)
    rh = _float_attr(state, "rh_air", 70.0)
    rad = _float_attr(state, "glob_rad", 0.0)
    rad_load = max(
        rad,
        _first_finite_attr(state, ("forecast_rad_mean_1h",), rad),
        _first_finite_attr(state, ("forecast_rad_peak_2h",), rad),
    )
    canopy_margin = _float_attr(state, "canopy_dew_margin", 3.0)
    air_margin = _float_attr(state, "dew_margin_air", 3.0)
    temp_rise = _first_finite_attr(
        state,
        ("temp_air_delta_1h", "temp_air_rise_1h", "temp_air_trend_c_per_hour"),
        0.0,
    )
    temp_violation = _float_attr(state, "temp_violation", 0.0)
    vpd = float(calculate_vpd_kpa(temp, rh))
    dew_margin = min(air_margin, canopy_margin)
    dew_risk = rh >= 88.0 or dew_margin <= 1.2
    hot_dry = intent.regime in {"hot_dry_relief", "radiation_spike_relief"} or (
        rad_load >= 500.0 and (temp >= 28.0 or vpd >= 1.35) and (rh <= 62.0 or vpd >= 1.35)
    )
    hot_danger = temp >= 32.0 or (rad_load >= 650.0 and (temp_rise >= 0.5 or temp >= 30.5))
    humid_regime = intent.regime in {
        "high_humidity_recovery",
        "hot_humid_relief",
        "cold_humid_recovery",
        "dawn_predehumidify",
    }
    cold_dry = intent.regime == "cold_dry_protect" or (temp <= 16.5 and rh <= 58.0)
    ctx = {
        "temp": temp,
        "rh": rh,
        "co2": _float_attr(state, "co2_air", 430.0),
        "vpd": vpd,
        "rad_load": rad_load,
        "dew_margin": dew_margin,
        "canopy_dew_margin": canopy_margin,
        "air_dew_margin": air_margin,
        "temp_rise_1h": temp_rise,
        "temp_violation": temp_violation,
        "dew_risk": 1.0 if dew_risk else 0.0,
        "hot_dry": 1.0 if hot_dry else 0.0,
        "hot_danger": 1.0 if hot_danger else 0.0,
        "humid_regime": 1.0 if humid_regime else 0.0,
        "cold_dry": 1.0 if cold_dry else 0.0,
    }
    ctx["safety_gate_reason"] = _score_safety_gate_reason(ctx)
    return ctx


def _score_safety_gate_reason(ctx: Mapping[str, Any]) -> str:
    if float(ctx.get("canopy_dew_margin", 3.0)) < 1.0:
        return "canopy_gate"
    if float(ctx.get("air_dew_margin", 3.0)) < 1.0 or float(ctx.get("dew_margin", 3.0)) < 1.0:
        return "dew_gate"
    if float(ctx.get("rh", 70.0)) >= 90.0:
        return "rh_high_gate"
    temp_high = float(ctx.get("temp", 20.0)) >= 32.0 or float(ctx.get("temp_violation", 0.0)) > 0.0
    radiation_rise = float(ctx.get("rad_load", 0.0)) >= 650.0 and float(ctx.get("temp_rise_1h", 0.0)) >= 0.5
    if temp_high or radiation_rise:
        return "temp_high_gate"
    return "none"


def _safety_gate_penalty(candidate_name: str, gate_reason: str) -> float:
    name = str(candidate_name or "")
    gate = str(gate_reason or "none")
    if gate not in SAFETY_GATE_REASONS:
        gate = "none"
    if gate == "none":
        return 0.0
    if name == "hot_dry_protect":
        return 250.0
    if gate == "temp_high_gate":
        if name == "constant_hold":
            return 0.75
        if name in {"strict_dehumidify_then_relax", "dawn_predehumidify", "cold_humid_recovery"}:
            return 0.25
        return 0.0
    if gate in {"dew_gate", "canopy_gate", "rh_high_gate"}:
        if name in {"co2_day_boost", "lighting_assist"}:
            return 0.75
        if name == "shade_cooling":
            return 0.20
        return 0.0
    return 0.0


def _priority_multiplier(intent: IntentContract, topic: str) -> float:
    priorities = [str(item) for item in intent.priority]
    if topic == "safety":
        return 1.8 if "safety" in priorities else 1.0
    if topic == "humidity":
        return 1.4 if "humidity" in priorities else 1.0
    if topic == "temperature":
        return 1.3 if "temperature" in priorities else 1.0
    if topic == "growth":
        return 1.2 if "growth" in priorities else 0.8
    return 1.0


def _profile_smoothness(profile: Mapping[str, Sequence[float]]) -> float:
    penalty = 0.0
    for key, scale in (("target_temp", 1.0), ("target_co2", 120.0), ("target_rh", 4.0)):
        values = list(profile.get(key, []) or [])
        if len(values) <= 1:
            continue
        deltas = [abs(float(next_value) - float(value)) / scale for value, next_value in zip(values, values[1:])]
        penalty += float(np.mean([delta * delta for delta in deltas])) if deltas else 0.0
    return float(penalty)


def score_profile_candidate(
    candidate: ProfileCandidate,
    intent: IntentContract,
    state: Any,
    horizon_steps: Optional[int] = None,
) -> ProfileCandidateScore:
    """Score a profile candidate with a lightweight short-horizon climate proxy.

    Lower is better. This is intentionally shadow-only: the proxy is used for
    ranking diagnostics, not actuator selection.
    """

    constraints = {**DEFAULT_CONSTRAINTS, **dict(candidate.constraints or {}), **dict(intent.constraints or {})}
    horizon = max(1, int(horizon_steps or len(candidate.target_profile.get("target_temp", [])) or 1))
    ctx = _score_context(state, intent)
    temp = float(ctx["temp"])
    rh = float(ctx["rh"])
    co2 = float(ctx["co2"])
    base_margin = float(ctx["dew_margin"])
    temp_min = float(constraints.get("temp_min", TARGET_LIMITS["temp"][0]))
    temp_max = float(constraints.get("temp_max", 32.0))
    rh_hard_max = float(constraints.get("rh_hard_max", 90.0))
    dew_margin_min = float(constraints.get("dew_margin_min", 1.0))
    canopy_margin_min = float(constraints.get("canopy_dew_margin_min", dew_margin_min))
    vpd_min = float(constraints.get("vpd_min", 0.35))
    vpd_max = float(constraints.get("vpd_max", 1.60))

    temp_high_risk = 0.0
    temp_low_risk = 0.0
    rh_low_risk = 0.0
    rh_high_risk = 0.0
    vpd_high_risk = 0.0
    vpd_low_risk = 0.0
    dew_risk = 0.0
    hot_dry_retention_gap = 0.0
    humid_relief_gap = 0.0
    cold_dry_preserve_gap = 0.0
    co2_gap = 0.0
    hot_danger_penalty = 0.0
    safety_gate_reason = str(ctx.get("safety_gate_reason", "none"))
    gate_penalty = _safety_gate_penalty(candidate.name, safety_gate_reason)

    rad_push = max(0.0, float(ctx["rad_load"]) - 500.0) / 500.0
    for idx in range(horizon):
        target_temp = _profile_value(candidate.target_profile, "target_temp", idx, temp)
        target_co2 = _profile_value(candidate.target_profile, "target_co2", idx, co2)
        target_rh = _profile_value(candidate.target_profile, "target_rh", idx, rh)

        prev_temp = temp
        temp += 0.35 * (target_temp - temp) + 0.08 * rad_push
        rh += 0.35 * (target_rh - rh) - max(0.0, temp - prev_temp) * 0.9 + max(0.0, prev_temp - temp) * 0.25
        co2 += 0.30 * (target_co2 - co2)
        rh = float(np.clip(rh, 40.0, 96.0))
        vpd = float(calculate_vpd_kpa(temp, rh))
        margin_est = base_margin + 0.08 * (float(ctx["rh"]) - rh) + 0.04 * (temp - float(ctx["temp"]))

        temp_high_risk += max(0.0, temp - temp_max) ** 2
        temp_low_risk += max(0.0, temp_min - temp) ** 2
        rh_low_risk += max(0.0, TARGET_LIMITS["rh"][0] - rh) ** 2
        rh_high_risk += max(0.0, rh - rh_hard_max) ** 2
        vpd_high_risk += max(0.0, vpd - vpd_max) ** 2
        vpd_low_risk += max(0.0, vpd_min - vpd) ** 2
        dew_risk += max(0.0, max(dew_margin_min, canopy_margin_min) - margin_est) ** 2

        if ctx["hot_dry"] > 0.0 and ctx["dew_risk"] <= 0.0 and ctx["hot_danger"] <= 0.0:
            desired_rh = min(82.0, max(float(ctx["rh"]) + 2.5, _bounds(intent, "rh")[2]))
            hot_dry_retention_gap += max(0.0, desired_rh - rh) ** 2
        if ctx["humid_regime"] > 0.0 or ctx["dew_risk"] > 0.0:
            desired_rh = min(float(ctx["rh"]) - 2.0, 82.0)
            humid_relief_gap += max(0.0, rh - desired_rh) ** 2
        if ctx["cold_dry"] > 0.0:
            desired_rh = max(float(ctx["rh"]) + 1.0, _bounds(intent, "rh")[2])
            cold_dry_preserve_gap += max(0.0, desired_rh - rh) ** 2
        if ctx["hot_danger"] > 0.0:
            hot_danger_penalty += (
                max(0.0, temp - min(30.5, temp_max)) ** 2
                + max(0.0, target_temp - _bounds(intent, "temp")[0]) ** 2 * 0.004
                + max(0.0, target_rh - 75.5) ** 2 * 0.015
            )
        if intent.regime in {"daytime_growth", "co2_day_boost"} and ctx["dew_risk"] <= 0.0 and ctx["hot_danger"] <= 0.0:
            co2_gap += max(0.0, _bounds(intent, "co2")[2] - co2) / 120.0

    inv_horizon = 1.0 / float(horizon)
    breakdown = {
        "temperature_high_risk": temp_high_risk * inv_horizon * 7.0 * _priority_multiplier(intent, "safety"),
        "temperature_low_risk": temp_low_risk * inv_horizon * 3.0 * _priority_multiplier(intent, "temperature"),
        "rh_low_risk": rh_low_risk * inv_horizon * 0.12 * _priority_multiplier(intent, "humidity"),
        "rh_high_risk": rh_high_risk * inv_horizon * 0.22 * _priority_multiplier(intent, "safety"),
        "vpd_high_risk": vpd_high_risk * inv_horizon * 3.5 * _priority_multiplier(intent, "humidity"),
        "vpd_low_risk": vpd_low_risk * inv_horizon * 2.5 * _priority_multiplier(intent, "safety"),
        "dew_risk": dew_risk * inv_horizon * 8.0 * _priority_multiplier(intent, "safety"),
        "hot_dry_retention_gap": hot_dry_retention_gap * inv_horizon * 0.18 * _priority_multiplier(intent, "humidity"),
        "humid_relief_gap": humid_relief_gap * inv_horizon * 0.20 * _priority_multiplier(intent, "humidity"),
        "cold_dry_preserve_gap": cold_dry_preserve_gap * inv_horizon * 0.15 * _priority_multiplier(intent, "humidity"),
        "hot_danger_penalty": hot_danger_penalty * inv_horizon * 4.5 * _priority_multiplier(intent, "safety"),
        "co2_growth_gap": co2_gap * inv_horizon * 0.25 * _priority_multiplier(intent, "growth"),
        "smoothness_penalty": _profile_smoothness(candidate.target_profile) * 0.35,
        "safety_gate_penalty": gate_penalty,
    }
    total = float(sum(breakdown.values()))
    return ProfileCandidateScore(name=candidate.name, total_score=total, breakdown=breakdown)


def _profile_value(profile: Mapping[str, Sequence[float]], key: str, idx: int, default: float) -> float:
    values = profile.get(key) if isinstance(profile, Mapping) else None
    if isinstance(values, (list, tuple, np.ndarray)) and values:
        try:
            return float(values[min(max(int(idx), 0), len(values) - 1)])
        except Exception:
            return float(default)
    return float(default)


def score_profile_candidates(
    candidates: Sequence[ProfileCandidate],
    intent: IntentContract,
    state: Any,
    horizon_steps: Optional[int] = None,
) -> Tuple[List[ProfileCandidateScore], Dict[str, Any]]:
    ctx = _score_context(state, intent)
    gate_reason = str(ctx.get("safety_gate_reason", "none"))
    if gate_reason not in SAFETY_GATE_REASONS:
        gate_reason = "none"
    scores = [score_profile_candidate(candidate, intent, state, horizon_steps=horizon_steps) for candidate in candidates]
    scores = sorted(scores, key=lambda item: (item.total_score, item.name))
    selected = scores[0] if scores else None
    second = scores[1] if len(scores) > 1 else None
    diagnostics: Dict[str, Any] = {
        "score_schema_version": "profile_candidate_scorer_v1",
        "score_shadow_only": True,
        "scored_candidate_count": len(scores),
        "score_ranked_candidates": [score.name for score in scores],
        "score_by_candidate": {score.name: float(round(score.total_score, 6)) for score in scores},
        "score_safety_gate_reason": gate_reason,
        "score_safety_gate_active": gate_reason != "none",
    }
    if selected is not None:
        diagnostics["score_selected_shadow_profile_name"] = selected.name
        diagnostics["score_selected_shadow_profile_score"] = float(round(selected.total_score, 6))
        diagnostics["score_selected_breakdown"] = {
            key: float(round(value, 6)) for key, value in sorted(selected.breakdown.items())
        }
        diagnostics["score_margin_to_second"] = float(
            round((second.total_score - selected.total_score) if second is not None else 0.0, 6)
        )
    return scores, diagnostics


def score_profile_candidate_payloads(
    candidates: Sequence[Mapping[str, Any]],
    intent: Any,
    state: Any,
    current_plan: Optional[Mapping[str, Any]] = None,
    horizon_steps: Optional[int] = None,
) -> Tuple[List[ProfileCandidateScore], Dict[str, Any]]:
    contract, intent_diagnostics = _coerce_for_profile_generation(intent, state, current_plan)
    horizon = max(1, int(horizon_steps or 1))
    parsed: List[ProfileCandidate] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        raw_profile = item.get("target_profile", {})
        if not name or not isinstance(raw_profile, Mapping):
            continue
        profile = {
            key: _clip_series(raw_profile.get(key, []), key, horizon)
            for key in PROFILE_KEYS
        }
        raw_constraints = item.get("constraints", contract.constraints)
        constraints = dict(raw_constraints) if isinstance(raw_constraints, Mapping) else dict(contract.constraints)
        raw_priority = item.get("priority", contract.priority)
        priority = tuple(str(value) for value in raw_priority) if isinstance(raw_priority, (list, tuple)) else tuple(contract.priority)
        parsed.append(
            ProfileCandidate(
                name=name,
                regime=str(item.get("regime") or contract.regime),
                target_profile=profile,
                priority=priority,
                constraints=constraints,
                reason=str(item.get("reason") or "rescored profile candidate"),
            )
        )
    scores, diagnostics = score_profile_candidates(parsed, contract, state, horizon_steps=horizon)
    diagnostics["intent_diagnostics"] = dict(intent_diagnostics)
    return scores, diagnostics


def select_shadow_profile(
    candidates: Sequence[ProfileCandidate],
    intent: IntentContract,
) -> Tuple[Optional[ProfileCandidate], Dict[str, Any]]:
    if not candidates:
        return None, {"selected_shadow_profile_name": "", "selector_scores": {}}
    scores = {candidate.name: selector_score(candidate, intent) for candidate in candidates}
    selected = max(candidates, key=lambda candidate: (scores.get(candidate.name, 0.0), -list(candidates).index(candidate)))
    return selected, {
        "selected_shadow_profile_name": selected.name,
        "selector_scores": {key: round(value, 6) for key, value in sorted(scores.items())},
    }


def build_profile_generator_shadow_payload(
    intent: Any,
    state: Any,
    current_plan: Optional[Mapping[str, Any]] = None,
    horizon_steps: int = 12,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    contract, intent_diagnostics = _coerce_for_profile_generation(intent, state, current_plan)
    horizon = max(1, int(horizon_steps))
    names, unsupported = _requested_shape_names(contract)
    candidates = build_profile_candidates(contract, state, current_plan=current_plan, horizon_steps=horizon)
    selected, selector = select_shadow_profile(candidates, contract)
    score_rows, score_diagnostics = score_profile_candidates(candidates, contract, state, horizon_steps=horizon)
    score_by_name = {score.name: score.to_dict() for score in score_rows}
    scorer_selected_name = str(score_diagnostics.get("score_selected_shadow_profile_name") or "")
    selector_selected_name = str(selector.get("selected_shadow_profile_name") or "")
    diagnostics: Dict[str, Any] = {
        "schema_version": "profile_generator_v1",
        "shadow_only": True,
        "intent_regime": contract.regime,
        "intent_confidence": float(contract.confidence),
        "horizon_steps": int(horizon),
        "candidate_count": len(candidates),
        "requested_shapes": list(names),
        "unsupported_shapes": list(unsupported),
        "intent_diagnostics": dict(intent_diagnostics),
        **selector,
        **score_diagnostics,
        "score_selector_agreement": bool(
            scorer_selected_name and selector_selected_name and scorer_selected_name == selector_selected_name
        ),
    }
    if selected is not None:
        diagnostics["selected_shadow_profile_reason"] = selected.reason
    candidate_payloads: List[Dict[str, Any]] = []
    for candidate in candidates:
        payload = candidate.to_dict()
        score_payload = score_by_name.get(candidate.name)
        if score_payload is not None:
            payload["score"] = score_payload["total_score"]
            payload["score_breakdown"] = score_payload["breakdown"]
        candidate_payloads.append(payload)
    return candidate_payloads, diagnostics
