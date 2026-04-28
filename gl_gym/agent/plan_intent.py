"""Plan-conditioned intent utilities for PPO expert distillation.

The PPO policy does not expose a real intent variable. This module builds an
auditable proxy intent from state, target setpoints, and greenhouse risk
features. The proxy intent is used for filtering and conditioning distilled
expert data, not as a replacement for safety guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from gl_gym.common.utils import calculate_vpd_kpa


INTENT_LABELS = (
    "economy_hold",
    "economic_dehumidify",
    "safe_dehumidify",
    "heat_recovery",
    "heat_preservation",
    "cooling",
    "co2_enrichment",
    "lighting_assist",
    "humidity_preservation",
)


ACTION_KEYS = (
    "heating",
    "co2",
    "screen",
    "ventilation",
    "lighting",
    "shading",
)


INTENT_COMPATIBLE_STRATEGIES: Dict[str, Tuple[str, ...]] = {
    "economy_hold": ("economy_hold",),
    "economic_dehumidify": (
        "free_air_exchange",
        "vent_only_dehumidify",
        "screen_release_dehumidify",
    ),
    "safe_dehumidify": (
        "heat_vent_dehumidify",
        "vent_only_dehumidify",
        "screen_release_dehumidify",
        "free_air_exchange",
    ),
    "heat_recovery": ("heat_preservation",),
    "heat_preservation": ("heat_preservation", "economy_hold"),
    "cooling": ("cooling_ventilation", "shade_cooling"),
    "co2_enrichment": ("co2_enrichment",),
    "lighting_assist": ("lighting_assist",),
    "humidity_preservation": ("economy_hold", "heat_preservation"),
}


def _float_attr(obj: Any, name: str, default: float) -> float:
    try:
        value = getattr(obj, name, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _float_dict(data: Dict[str, Any], key: str, default: float) -> float:
    try:
        value = data.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))


def _inverse_ramp(value: float, low: float, high: float) -> float:
    return 1.0 - _ramp(value, low, high)


def _plan_target(plan: Optional[Dict[str, Any]], key: str, state: Any) -> Optional[float]:
    if not isinstance(plan, dict):
        return None
    profile = plan.get("target_profile", {})
    values = profile.get(key) if isinstance(profile, dict) else None
    if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
        created = int(plan.get("created_timestep", _float_attr(state, "timestep", 0.0)))
        idx = int(np.clip(int(_float_attr(state, "timestep", 0.0)) - created, 0, len(values) - 1))
        try:
            return float(values[idx])
        except Exception:
            pass
    value = plan.get(key)
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def humidity_risk_features(state: Any) -> Dict[str, float]:
    temp = _float_attr(state, "temp_air", 20.0)
    rh = _float_attr(state, "rh_air", 70.0)
    vpd = float(calculate_vpd_kpa(temp, rh))
    dew_margin = min(
        _float_attr(state, "dew_margin_air", 3.0),
        _float_attr(state, "canopy_dew_margin", 3.0),
    )
    forecast_humidity_risk = _float_attr(state, "forecast_humidity_risk", 0.0)
    risk = 0.0
    risk += 0.45 * _ramp(rh, 82.0, 92.0)
    risk += 0.25 * _inverse_ramp(vpd, 0.18, 0.45)
    risk += 0.20 * _inverse_ramp(dew_margin, 0.4, 1.4)
    risk += 0.10 * _ramp(forecast_humidity_risk, 0.2, 0.8)
    return {
        "humidity_risk": float(np.clip(risk, 0.0, 1.0)),
        "vpd_kpa": float(vpd),
        "dew_margin": float(dew_margin),
    }


def make_plan_from_targets(
    state: Any,
    target_temp: float,
    target_co2: float,
    target_rh: float,
    horizon: int = 12,
    source: str = "contract",
) -> Dict[str, Any]:
    horizon = max(1, int(horizon))
    timestep = int(_float_attr(state, "timestep", 0.0))
    return {
        "target_temp": float(target_temp),
        "target_co2": float(target_co2),
        "target_rh": float(target_rh),
        "target_profile": {
            "target_temp": [float(target_temp)] * horizon,
            "target_co2": [float(target_co2)] * horizon,
            "target_rh": [float(target_rh)] * horizon,
        },
        "created_timestep": timestep,
        "expires_timestep": timestep + horizon,
        "reason": source,
        "plan_interval": horizon,
    }


def default_setpoint_contract(
    state: Any,
    target_control: Optional[Iterable[float]] = None,
    horizon: int = 12,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build a state-only pseudo setpoint contract for PPO trajectory export."""

    control = np.asarray(list(target_control) if target_control is not None else np.zeros(6), dtype=np.float32)
    if control.shape[0] < 6:
        control = np.pad(control, (0, 6 - control.shape[0]))
    hour = _float_attr(state, "hour_of_day", 12.0)
    is_day = 6.0 <= hour <= 18.0
    total_rad = _float_attr(state, "glob_rad", 0.0) + float(control[4]) * 100.0
    vent_level = float(control[3])
    temp = _float_attr(state, "temp_air", 20.0)
    rh = _float_attr(state, "rh_air", 70.0)
    risk = humidity_risk_features(state)

    if temp < 12.5:
        target_temp = 15.0
    elif is_day and total_rad > 180.0:
        target_temp = 18.5
    elif is_day:
        target_temp = 17.0
    else:
        target_temp = 14.0
    target_temp = float(np.clip(target_temp, 11.5, 24.0))

    if is_day and total_rad > 140.0 and vent_level <= 0.25:
        target_co2 = 620.0
    else:
        target_co2 = 430.0
    target_co2 = float(np.clip(target_co2, 380.0, 850.0))

    if rh > 88.0:
        target_rh = 72.0
    elif rh < 60.0:
        target_rh = 70.0
    else:
        target_rh = 76.0
    target_rh = float(np.clip(target_rh, 62.0, 88.0))

    risk_cap = 88.0
    if (
        rh >= 90.0
        or (rh >= 82.0 and risk["vpd_kpa"] < 0.30)
        or (rh >= 78.0 and risk["vpd_kpa"] < 0.20)
        or risk["dew_margin"] < 0.8
    ):
        risk_cap = 72.0
    elif rh >= 88.0 or (rh >= 82.0 and risk["vpd_kpa"] < 0.40) or risk["dew_margin"] < 1.2:
        risk_cap = 76.0
    elif rh >= 86.0:
        risk_cap = 80.0
    if target_rh > risk_cap:
        target_rh = max(62.0, risk_cap)

    plan = make_plan_from_targets(
        state,
        target_temp=target_temp,
        target_co2=target_co2,
        target_rh=target_rh,
        horizon=horizon,
        source="pseudo_setpoint_contract",
    )
    diagnostics = {
        "source": "pseudo_setpoint_contract",
        "risk_cap": float(risk_cap),
        **risk,
    }
    return plan, diagnostics


@dataclass
class PlanIntent:
    label: str
    confidence: float
    target_temp: float
    target_co2: float
    target_rh: float
    target_temp_delta: float
    target_co2_delta: float
    target_rh_delta: float
    humidity_risk: float
    vpd_kpa: float
    dew_margin: float
    compatible_strategy_labels: Tuple[str, ...] = field(default_factory=tuple)
    reasons: Tuple[str, ...] = ()

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.55

    def to_record(self, prefix: str = "intent") -> Dict[str, Any]:
        return {
            f"{prefix}_label": self.label,
            f"{prefix}_confidence": float(self.confidence),
            f"{prefix}_target_temp": float(self.target_temp),
            f"{prefix}_target_co2": float(self.target_co2),
            f"{prefix}_target_rh": float(self.target_rh),
            f"{prefix}_target_temp_delta": float(self.target_temp_delta),
            f"{prefix}_target_co2_delta": float(self.target_co2_delta),
            f"{prefix}_target_rh_delta": float(self.target_rh_delta),
            f"{prefix}_humidity_risk": float(self.humidity_risk),
            f"{prefix}_vpd_kpa": float(self.vpd_kpa),
            f"{prefix}_dew_margin": float(self.dew_margin),
            f"{prefix}_compatible_strategy_labels_json": json.dumps(self.compatible_strategy_labels),
            f"{prefix}_reason": "; ".join(self.reasons),
        }


def infer_plan_intent(
    state: Any,
    plan: Optional[Dict[str, Any]] = None,
    target_control: Optional[Iterable[float]] = None,
    horizon: int = 12,
) -> PlanIntent:
    if plan is None:
        plan, _ = default_setpoint_contract(state, target_control=target_control, horizon=horizon)

    temp = _float_attr(state, "temp_air", 20.0)
    rh = _float_attr(state, "rh_air", 70.0)
    co2 = _float_attr(state, "co2_air", 430.0)
    rad = _float_attr(state, "glob_rad", 0.0)
    hour = _float_attr(state, "hour_of_day", 12.0)
    is_day = 6.0 <= hour <= 18.0
    risk = humidity_risk_features(state)

    target_temp = _plan_target(plan, "target_temp", state)
    target_co2 = _plan_target(plan, "target_co2", state)
    target_rh = _plan_target(plan, "target_rh", state)
    if target_temp is None or target_co2 is None or target_rh is None:
        fallback, _ = default_setpoint_contract(state, target_control=target_control, horizon=horizon)
        target_temp = target_temp if target_temp is not None else float(fallback["target_temp"])
        target_co2 = target_co2 if target_co2 is not None else float(fallback["target_co2"])
        target_rh = target_rh if target_rh is not None else float(fallback["target_rh"])

    target_temp = float(target_temp)
    target_co2 = float(target_co2)
    target_rh = float(target_rh)
    temp_delta = target_temp - temp
    co2_delta = target_co2 - co2
    rh_delta = target_rh - rh
    humidity_risk = float(risk["humidity_risk"])
    vpd = float(risk["vpd_kpa"])
    dew_margin = float(risk["dew_margin"])

    label = "economy_hold"
    confidence = 0.55
    reasons = []

    severe_humidity = rh_delta <= -10.0 or humidity_risk >= 0.72 or rh >= 90.0 or dew_margin < 0.8
    moderate_humidity = rh_delta <= -3.5 or humidity_risk >= 0.38 or rh >= 84.0
    if severe_humidity:
        label = "safe_dehumidify"
        confidence = max(_ramp(-rh_delta, 6.0, 16.0), _ramp(humidity_risk, 0.45, 0.85))
        reasons.append(f"RH target requires strong reduction: delta={rh_delta:.1f}, risk={humidity_risk:.2f}")
    elif moderate_humidity:
        if temp >= 15.0 and dew_margin >= 0.8:
            label = "economic_dehumidify"
            confidence = max(_ramp(-rh_delta, 2.0, 8.0), _ramp(humidity_risk, 0.25, 0.55))
            reasons.append(f"Moderate RH reduction with usable temperature: delta={rh_delta:.1f}, temp={temp:.1f}")
        else:
            label = "safe_dehumidify"
            confidence = max(_ramp(-rh_delta, 2.0, 8.0), _ramp(humidity_risk, 0.25, 0.65))
            reasons.append(f"Humidity reduction under cold/dew risk: delta={rh_delta:.1f}, temp={temp:.1f}")
    elif temp_delta >= 2.0:
        label = "heat_recovery"
        confidence = _ramp(temp_delta, 1.0, 4.0)
        reasons.append(f"Temperature target above current state: delta={temp_delta:.1f}")
    elif temp_delta >= 0.5 and (not is_day or temp < 15.0):
        label = "heat_preservation"
        confidence = _ramp(temp_delta, 0.2, 2.0)
        reasons.append(f"Small heating target during cold/night condition: delta={temp_delta:.1f}")
    elif temp_delta <= -2.0:
        label = "cooling"
        confidence = _ramp(-temp_delta, 1.0, 4.0)
        reasons.append(f"Temperature target below current state: delta={temp_delta:.1f}")
    elif is_day and co2_delta >= 80.0 and rad >= 80.0 and rh < 86.0:
        label = "co2_enrichment"
        confidence = min(_ramp(co2_delta, 40.0, 180.0), _ramp(rad, 60.0, 180.0))
        reasons.append(f"CO2 target above current state under useful radiation: delta={co2_delta:.0f}")
    elif is_day and rad < 80.0 and rh < 82.0 and temp_delta >= -0.5:
        label = "lighting_assist"
        confidence = min(_inverse_ramp(rad, 40.0, 100.0), _inverse_ramp(rh, 78.0, 86.0))
        reasons.append(f"Low radiation with safe humidity: rad={rad:.0f}, RH={rh:.1f}")
    elif rh_delta >= 8.0:
        label = "humidity_preservation"
        confidence = _ramp(rh_delta, 4.0, 14.0)
        reasons.append(f"RH target is above current state: delta={rh_delta:.1f}")
    else:
        label = "economy_hold"
        confidence = max(0.55, _inverse_ramp(abs(temp_delta) + abs(rh_delta) / 4.0, 1.0, 4.0))
        reasons.append(f"Targets are close to current state: dT={temp_delta:.1f}, dRH={rh_delta:.1f}")

    confidence = float(np.clip(confidence, 0.0, 1.0))
    return PlanIntent(
        label=label,
        confidence=confidence,
        target_temp=target_temp,
        target_co2=target_co2,
        target_rh=target_rh,
        target_temp_delta=float(temp_delta),
        target_co2_delta=float(co2_delta),
        target_rh_delta=float(rh_delta),
        humidity_risk=humidity_risk,
        vpd_kpa=vpd,
        dew_margin=dew_margin,
        compatible_strategy_labels=INTENT_COMPATIBLE_STRATEGIES.get(label, ()),
        reasons=tuple(reasons),
    )


def intent_to_dehumidify_mode(intent_label: str) -> str:
    if intent_label == "safe_dehumidify":
        return "strong"
    if intent_label == "economic_dehumidify":
        return "mild"
    return "normal"


def action_record_for_labeling(state: Any, control: Iterable[float], prefix: str = "u") -> Dict[str, float]:
    values = np.asarray(list(control), dtype=np.float32).reshape(-1)
    if values.shape[0] < len(ACTION_KEYS):
        values = np.pad(values, (0, len(ACTION_KEYS) - values.shape[0]))
    record: Dict[str, float] = {
        "temp_air": _float_attr(state, "temp_air", 20.0),
        "rh_air": _float_attr(state, "rh_air", 70.0),
        "co2_air": _float_attr(state, "co2_air", 430.0),
        "glob_rad": _float_attr(state, "glob_rad", 0.0),
        "hour_of_day": _float_attr(state, "hour_of_day", 12.0),
        "dew_margin_air": _float_attr(state, "dew_margin_air", 3.0),
        "canopy_dew_margin": _float_attr(state, "canopy_dew_margin", 3.0),
        "forecast_humidity_risk": _float_attr(state, "forecast_humidity_risk", 0.0),
    }
    record["vpd_kpa"] = float(calculate_vpd_kpa(record["temp_air"], record["rh_air"]))
    for key, value in zip(ACTION_KEYS, values):
        record[f"{prefix}_{key}"] = float(value)
    return record


def strategy_intent_alignment(
    strategy: Any,
    intent: PlanIntent,
    min_strategy_confidence: float = 0.55,
    min_intent_confidence: float = 0.50,
) -> Dict[str, Any]:
    strategy_label = str(getattr(strategy, "label", strategy))
    strategy_confidence = float(getattr(strategy, "confidence", 1.0 if strategy_label else 0.0))
    compatible = tuple(intent.compatible_strategy_labels)
    if strategy_label in {"unknown", "ambiguous", ""}:
        return {
            "aligned": False,
            "score": 0.0,
            "strategy_label": strategy_label,
            "intent_label": intent.label,
            "reason": "strategy label is not confident",
        }
    if strategy_confidence < min_strategy_confidence:
        return {
            "aligned": False,
            "score": 0.0,
            "strategy_label": strategy_label,
            "intent_label": intent.label,
            "reason": f"strategy confidence {strategy_confidence:.2f} below threshold",
        }
    if intent.confidence < min_intent_confidence:
        return {
            "aligned": False,
            "score": 0.0,
            "strategy_label": strategy_label,
            "intent_label": intent.label,
            "reason": f"intent confidence {intent.confidence:.2f} below threshold",
        }
    aligned = strategy_label in compatible
    score = min(float(strategy_confidence), float(intent.confidence)) if aligned else 0.0
    reason = "strategy matches plan intent" if aligned else "strategy does not match plan intent"
    return {
        "aligned": bool(aligned),
        "score": float(score),
        "strategy_label": strategy_label,
        "intent_label": intent.label,
        "compatible_strategy_labels": list(compatible),
        "reason": reason,
    }


def target_tracking_baseline_control(
    state: Any,
    plan: Optional[Dict[str, Any]] = None,
    intent: Optional[PlanIntent] = None,
) -> np.ndarray:
    """Small interpretable baseline used for residual expert training."""

    if intent is None:
        intent = infer_plan_intent(state, plan)
    temp = _float_attr(state, "temp_air", 20.0)
    rh = _float_attr(state, "rh_air", 70.0)
    rad = _float_attr(state, "glob_rad", 0.0)
    hour = _float_attr(state, "hour_of_day", 12.0)
    is_day = 6.0 <= hour <= 18.0
    control = np.zeros(6, dtype=np.float32)

    if intent.target_temp_delta > 0.2:
        control[0] = np.clip(0.06 * intent.target_temp_delta, 0.0, 0.42)
    elif intent.target_temp_delta < -0.5:
        control[3] = np.clip(0.08 * (-intent.target_temp_delta), 0.0, 0.55)
    if not is_day or temp < 14.5:
        control[2] = max(float(control[2]), 0.75)

    if intent.label == "economic_dehumidify":
        control[3] = max(float(control[3]), np.clip(0.24 + 0.05 * (-intent.target_rh_delta), 0.25, 0.72))
        control[2] = min(float(control[2]), 0.35)
        control[0] = min(float(control[0]), 0.08)
    elif intent.label == "safe_dehumidify":
        control[3] = max(float(control[3]), np.clip(0.35 + 0.035 * (-intent.target_rh_delta), 0.40, 0.85))
        control[2] = min(float(control[2]), 0.35)
        if temp < 16.0:
            control[0] = max(float(control[0]), np.clip(0.10 + 0.04 * (16.0 - temp), 0.10, 0.35))
    elif intent.label == "humidity_preservation":
        control[3] = min(float(control[3]), 0.08)
        if not is_day:
            control[2] = max(float(control[2]), 0.85)

    if intent.label == "co2_enrichment" and is_day and control[3] <= 0.25:
        control[1] = np.clip(intent.target_co2_delta / 350.0, 0.0, 0.55)
    if intent.label == "lighting_assist" and is_day and rh < 82.0:
        control[4] = np.clip((90.0 - rad) / 200.0, 0.0, 0.40)
    if intent.label == "cooling":
        control[3] = max(float(control[3]), np.clip(0.18 + 0.08 * (-intent.target_temp_delta), 0.20, 0.75))
        if rad > 350.0:
            control[5] = np.clip((rad - 300.0) / 500.0, 0.0, 0.55)

    if control[3] > 0.25:
        control[1] = 0.0
        control[4] = min(float(control[4]), 0.12)
    return np.clip(control, 0.0, 1.0).astype(np.float32)
