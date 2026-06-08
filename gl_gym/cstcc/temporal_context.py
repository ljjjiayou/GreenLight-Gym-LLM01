"""Temporal context feature extraction for C-STCC v80."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from .contracts import ACTION_FIELDS, TemporalContext, clamp, normalize_action


def _num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _series(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [_num(row.get(key)) for row in rows if row.get(key) not in (None, "")]


def _slope(values: Sequence[float], delta_t_hours: float) -> float:
    if len(values) < 2:
        return 0.0
    hours = max(delta_t_hours * (len(values) - 1), 1e-9)
    return (values[-1] - values[0]) / hours


def _discounted_debt(values: Sequence[float], *, threshold: float, direction: str, gamma: float, delta_t_hours: float) -> float:
    debt = 0.0
    recent_first = list(reversed(values))
    for index, value in enumerate(recent_first):
        if direction == "above":
            exceedance = max(0.0, value - threshold)
        else:
            exceedance = max(0.0, threshold - value)
        debt += (gamma**index) * exceedance * delta_t_hours
    return debt


def action_smoothness_summary(actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [normalize_action(action) for action in actions]
    action_count = len(normalized)
    transition_count = max(action_count - 1, 1)
    previous: dict[str, float] = {}
    signs: dict[str, int] = {}
    total_variation = {field: 0.0 for field in ACTION_FIELDS}
    max_jump = {field: 0.0 for field in ACTION_FIELDS}
    reversals: Counter[str] = Counter()
    for action in normalized:
        for field, value in action.items():
            if field in previous:
                delta = value - previous[field]
                total_variation[field] += abs(delta)
                max_jump[field] = max(max_jump[field], abs(delta))
                sign = 1 if delta > 1e-9 else -1 if delta < -1e-9 else 0
                if sign and signs.get(field, 0) and sign != signs[field]:
                    reversals[field] += 1
                if sign:
                    signs[field] = sign
            previous[field] = value
    return {
        "action_count": action_count,
        "action_history_length": action_count,
        "transition_count": transition_count,
        "total_variation_by_field": total_variation,
        "total_variation": sum(total_variation.values()),
        "total_variation_norm": clamp(sum(total_variation.values()) / transition_count),
        "max_jump_by_field": max_jump,
        "max_jump": max(max_jump.values()) if max_jump else 0.0,
        "max_jump_norm": clamp(max(max_jump.values()) if max_jump else 0.0),
        "reversal_count_by_field": dict(reversals),
        "reversal_count": sum(reversals.values()),
        "reversal_count_norm": clamp(sum(reversals.values()) / transition_count),
    }


def build_temporal_context(
    state_history: Sequence[Mapping[str, Any]],
    action_history: Sequence[Mapping[str, Any]],
    weather_history: Sequence[Mapping[str, Any]] | None = None,
    *,
    delta_t_hours: float = 1.0 / 12.0,
    gamma: float = 0.94,
    debt_refs: Mapping[str, float] | None = None,
) -> TemporalContext:
    """Build v80 level-0 temporal context features.

    Debts are accumulated in real time units, then normalized and saturated.
    """

    debt_refs = debt_refs or {
        "vpd_debt_kpa_h": 0.60,
        "heat_debt_c_h": 2.00,
        "dryness_debt_rh_h": 12.00,
    }
    temp = _series(state_history, "temp_air")
    rh = _series(state_history, "rh_air")
    vpd = _series(state_history, "vpd_air")
    co2 = _series(state_history, "co2_air")

    vpd_debt = _discounted_debt(vpd, threshold=2.0, direction="above", gamma=gamma, delta_t_hours=delta_t_hours)
    heat_debt = _discounted_debt(temp, threshold=28.0, direction="above", gamma=gamma, delta_t_hours=delta_t_hours)
    dryness_debt = _discounted_debt(rh, threshold=55.0, direction="below", gamma=gamma, delta_t_hours=delta_t_hours)

    missing_state_rate = 1.0 if not state_history else sum(
        1 for row in state_history for field in ("temp_air", "rh_air", "vpd_air") if row.get(field) in (None, "")
    ) / max(len(state_history) * 3, 1)
    weather_available = 1.0 if weather_history else 0.0
    data_quality = clamp(1.0 - missing_state_rate)
    solver_warning = 1.0 if any(str(row.get("runtime_warning", "") or row.get("runtime_error", "")).strip() for row in state_history) else 0.0

    return TemporalContext(
        trend_features={
            "temp_slope_c_per_h": _slope(temp, delta_t_hours),
            "rh_slope_pct_per_h": _slope(rh, delta_t_hours),
            "vpd_slope_kpa_per_h": _slope(vpd, delta_t_hours),
            "co2_slope_ppm_per_h": _slope(co2, delta_t_hours),
        },
        debt_features={
            "vpd_debt_kpa_h": vpd_debt,
            "heat_debt_c_h": heat_debt,
            "dryness_debt_rh_h": dryness_debt,
            "vpd_debt_norm": clamp(vpd_debt / max(debt_refs["vpd_debt_kpa_h"], 1e-9)),
            "heat_debt_norm": clamp(heat_debt / max(debt_refs["heat_debt_c_h"], 1e-9)),
            "dryness_debt_norm": clamp(dryness_debt / max(debt_refs["dryness_debt_rh_h"], 1e-9)),
        },
        action_smoothness_features=action_smoothness_summary(action_history),
        safety_pressure_features={
            "rewrite_pressure_norm": clamp(sum(_num(row.get("tomato_rewrite_magnitude")) for row in state_history)),
        },
        solver_risk_features={
            "solver_warning_score": solver_warning,
            "hot_dry_solver_sensitive_score": clamp(
                max(vpd[-1] - 2.2, 0.0) / 1.0 if vpd else 0.0
            ),
        },
        data_quality_features={
            "sensor_missing_rate": clamp(missing_state_rate),
            "weather_forecast_available": weather_available,
            "data_quality_score": data_quality,
            "recent_simulation_warning": solver_warning,
        },
        delta_t_hours=delta_t_hours,
        window_hours=delta_t_hours * max(len(state_history), 1),
    )
