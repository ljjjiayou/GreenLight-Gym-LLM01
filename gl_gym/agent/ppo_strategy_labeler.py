"""Conservative strategy labels for continuous greenhouse control actions.

This module translates a continuous action vector into an auditable strategy
label. It is deliberately conservative: ambiguous actions keep their soft
scores but are not treated as confident labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from gl_gym.common.utils import calculate_vpd_kpa


STRATEGY_LABELS = (
    "economy_hold",
    "vent_only_dehumidify",
    "heat_vent_dehumidify",
    "screen_release_dehumidify",
    "heat_preservation",
    "free_air_exchange",
    "cooling_ventilation",
    "co2_enrichment",
    "lighting_assist",
    "shade_cooling",
)


ACTION_KEYS = (
    "heating",
    "co2",
    "screen",
    "ventilation",
    "lighting",
    "shading",
)


@dataclass
class StrategyLabel:
    label: str
    confidence: float
    best_label: str
    best_score: float
    second_label: str
    second_score: float
    scores: Dict[str, float] = field(default_factory=dict)
    reasons: Tuple[str, ...] = ()
    humidity_risk: float = 0.0
    vpd_kpa: float = 0.0
    dew_margin: float = 0.0

    @property
    def is_confident(self) -> bool:
        return self.label not in {"ambiguous", "unknown"}

    def to_record(self, prefix: str = "strategy") -> Dict[str, Any]:
        return {
            f"{prefix}_label": self.label,
            f"{prefix}_confidence": float(self.confidence),
            f"{prefix}_best_label": self.best_label,
            f"{prefix}_best_score": float(self.best_score),
            f"{prefix}_second_label": self.second_label,
            f"{prefix}_second_score": float(self.second_score),
            f"{prefix}_scores_json": json.dumps(self.scores, sort_keys=True),
            f"{prefix}_reason": "; ".join(self.reasons),
            f"{prefix}_humidity_risk": float(self.humidity_risk),
            f"{prefix}_vpd_kpa": float(self.vpd_kpa),
            f"{prefix}_dew_margin": float(self.dew_margin),
        }


def _ramp(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))


def _inverse_ramp(value: float, low: float, high: float) -> float:
    return 1.0 - _ramp(value, low, high)


def _row_value(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _action_values(row: Dict[str, Any], action_prefix: str = "u") -> Dict[str, float]:
    values = {}
    for key in ACTION_KEYS:
        values[key] = _row_value(row, f"{action_prefix}_{key}", 0.0)
    return values


def _humidity_risk_features(row: Dict[str, Any]) -> Tuple[float, float, float]:
    temp = _row_value(row, "temp_air", 20.0)
    rh = _row_value(row, "rh_air", 70.0)
    vpd = _row_value(row, "vpd_kpa", float(calculate_vpd_kpa(temp, rh)))
    dew_margin = min(
        _row_value(row, "dew_margin_air", 3.0),
        _row_value(row, "canopy_dew_margin", 3.0),
    )
    humidity_risk = 0.0
    humidity_risk += 0.45 * _ramp(rh, 82.0, 92.0)
    humidity_risk += 0.25 * _inverse_ramp(vpd, 0.18, 0.45)
    humidity_risk += 0.20 * _inverse_ramp(dew_margin, 0.4, 1.4)
    humidity_risk += 0.10 * _ramp(_row_value(row, "forecast_humidity_risk", 0.0), 0.2, 0.8)
    return float(np.clip(humidity_risk, 0.0, 1.0)), float(vpd), float(dew_margin)


def score_strategy_candidates(row: Dict[str, Any], action_prefix: str = "u") -> Tuple[Dict[str, float], Dict[str, float]]:
    action = _action_values(row, action_prefix=action_prefix)
    heat = action["heating"]
    co2 = action["co2"]
    screen = action["screen"]
    vent = action["ventilation"]
    lamp = action["lighting"]
    shade = action["shading"]

    temp = _row_value(row, "temp_air", 20.0)
    rh = _row_value(row, "rh_air", 70.0)
    rad = _row_value(row, "glob_rad", 0.0)
    hour = _row_value(row, "hour_of_day", 12.0)
    is_day = 6.0 <= hour <= 18.0
    is_dark = rad < 20.0
    humidity_risk, vpd, dew_margin = _humidity_risk_features(row)

    heat_high = _ramp(heat, 0.12, 0.28)
    heat_low = _inverse_ramp(heat, 0.06, 0.18)
    vent_high = _ramp(vent, 0.25, 0.62)
    vent_low = _inverse_ramp(vent, 0.10, 0.25)
    screen_closed = _ramp(screen, 0.55, 0.90)
    screen_open = _inverse_ramp(screen, 0.35, 0.70)
    temp_cold = _inverse_ramp(temp, 14.0, 17.0)
    temp_hot = _ramp(temp, 24.0, 29.0)
    useful_rad = _ramp(rad, 80.0, 180.0)
    low_rad = _inverse_ramp(rad, 60.0, 160.0)
    night_or_cold = max(1.0 if (not is_day or is_dark) else 0.0, temp_cold)

    scores = {
        "heat_vent_dehumidify": min(heat_high, vent_high) * humidity_risk * _inverse_ramp(temp, 25.0, 30.0),
        "vent_only_dehumidify": vent_high * heat_low * humidity_risk * _ramp(temp, 13.5, 17.0),
        "screen_release_dehumidify": screen_open * max(vent_high, 0.35) * humidity_risk * _inverse_ramp(heat, 0.10, 0.24),
        "heat_preservation": heat_high * vent_low * screen_closed * night_or_cold,
        "free_air_exchange": (
            vent_high
            * heat_low
            * _inverse_ramp(co2, 0.03, 0.12)
            * _inverse_ramp(lamp, 0.03, 0.12)
            * _inverse_ramp(temp, 23.0, 28.0)
            * max(_ramp(rh, 78.0, 86.0), _ramp(humidity_risk, 0.25, 0.55))
            * _inverse_ramp(humidity_risk, 0.58, 0.82)
        ),
        "cooling_ventilation": vent_high * heat_low * temp_hot * _inverse_ramp(humidity_risk, 0.45, 0.85),
        "co2_enrichment": _ramp(co2, 0.05, 0.22) * useful_rad * _inverse_ramp(vent, 0.12, 0.30),
        "lighting_assist": _ramp(lamp, 0.05, 0.25) * low_rad * (1.0 if is_day else 0.45) * _inverse_ramp(rh, 84.0, 90.0),
        "shade_cooling": _ramp(shade, 0.08, 0.45) * max(_ramp(rad, 250.0, 500.0), temp_hot),
        "economy_hold": (
            _inverse_ramp(heat, 0.04, 0.16)
            * _inverse_ramp(co2, 0.03, 0.12)
            * _inverse_ramp(lamp, 0.03, 0.12)
            * _inverse_ramp(vent, 0.28, 0.55)
            * _inverse_ramp(humidity_risk, 0.65, 0.95)
        ),
    }
    scores = {label: float(np.clip(value, 0.0, 1.0)) for label, value in scores.items()}
    features = {
        "humidity_risk": humidity_risk,
        "vpd_kpa": vpd,
        "dew_margin": dew_margin,
        "rh": rh,
        "temp": temp,
        "rad": rad,
        "hour": hour,
        **{f"u_{key}": value for key, value in action.items()},
    }
    return scores, features


def _reason_for_label(label: str, features: Dict[str, float]) -> Tuple[str, ...]:
    heat = features["u_heating"]
    vent = features["u_ventilation"]
    screen = features["u_screen"]
    co2 = features["u_co2"]
    lamp = features["u_lighting"]
    shade = features["u_shading"]
    rh = features["rh"]
    temp = features["temp"]
    risk = features["humidity_risk"]
    reasons = []
    if label == "heat_vent_dehumidify":
        reasons.append(f"heat={heat:.2f} and vent={vent:.2f} under humidity risk={risk:.2f}")
    elif label == "vent_only_dehumidify":
        reasons.append(f"vent={vent:.2f} with low heat={heat:.2f} under humidity risk={risk:.2f}")
    elif label == "screen_release_dehumidify":
        reasons.append(f"screen opened to {screen:.2f} under humidity risk={risk:.2f}")
    elif label == "heat_preservation":
        reasons.append(f"heat={heat:.2f}, closed screen={screen:.2f}, low vent={vent:.2f}, temp={temp:.1f}")
    elif label == "free_air_exchange":
        reasons.append(f"vent={vent:.2f} with low heat={heat:.2f} and no CO2/lamp under moderate RH")
    elif label == "cooling_ventilation":
        reasons.append(f"vent={vent:.2f} with low heat={heat:.2f} at temp={temp:.1f}")
    elif label == "co2_enrichment":
        reasons.append(f"co2={co2:.2f} with limited vent={vent:.2f}")
    elif label == "lighting_assist":
        reasons.append(f"lamp={lamp:.2f} with radiation={features['rad']:.0f}")
    elif label == "shade_cooling":
        reasons.append(f"shade={shade:.2f} at temp={temp:.1f}/radiation={features['rad']:.0f}")
    elif label == "economy_hold":
        reasons.append(f"low-cost hold: heat={heat:.2f}, vent={vent:.2f}, RH={rh:.1f}")
    return tuple(reasons)


def label_strategy(
    row: Dict[str, Any],
    action_prefix: str = "u",
    min_confidence: float = 0.55,
    min_margin: float = 0.12,
) -> StrategyLabel:
    scores, features = score_strategy_candidates(row, action_prefix=action_prefix)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_label, second_score = ranked[1] if len(ranked) > 1 else ("none", 0.0)
    margin = best_score - second_score
    if best_score < min_confidence:
        label = "unknown"
        confidence = best_score
        reasons = (f"best score {best_score:.2f} below threshold {min_confidence:.2f}",)
    elif margin < min_margin:
        label = "ambiguous"
        confidence = best_score
        reasons = (f"{best_label} score {best_score:.2f} too close to {second_label} {second_score:.2f}",)
    else:
        label = best_label
        confidence = best_score
        reasons = _reason_for_label(best_label, features)

    return StrategyLabel(
        label=label,
        confidence=float(confidence),
        best_label=best_label,
        best_score=float(best_score),
        second_label=second_label,
        second_score=float(second_score),
        scores=scores,
        reasons=reasons,
        humidity_risk=float(features["humidity_risk"]),
        vpd_kpa=float(features["vpd_kpa"]),
        dew_margin=float(features["dew_margin"]),
    )
