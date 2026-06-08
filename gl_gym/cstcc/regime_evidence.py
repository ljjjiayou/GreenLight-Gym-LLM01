"""Regime-specific evidence scoring for C-STCC v80.1.

The LLM-reported regime is only a suggestion. This module checks whether the
current temporal context supports that specific regime before the suggestion can
meaningfully influence weights or state transitions.
"""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import REGIMES, clamp, normalize_action


def _positive_slope_score(value: Any, scale: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return clamp(number / max(scale, 1e-9))


def _negative_slope_score(value: Any, scale: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return clamp(-number / max(scale, 1e-9))


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def regime_evidence_score(
    regime: str,
    temporal_context: Mapping[str, Any],
    last_action: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return a [0, 1] evidence score and reason breakdown for a target regime."""

    action = normalize_action(last_action)
    trends = temporal_context.get("trend_features", {})
    debts = temporal_context.get("debt_features", {})
    solver = temporal_context.get("solver_risk_features", {})
    data = temporal_context.get("data_quality_features", {})
    data_quality = clamp(data.get("data_quality_score", 0.0))
    if regime not in REGIMES:
        return 0.0, {"invalid_regime": regime, "data_quality_score": data_quality}

    vpd_slope = _positive_slope_score(trends.get("vpd_slope_kpa_per_h", 0.0), 0.25)
    rh_decline = _negative_slope_score(trends.get("rh_slope_pct_per_h", 0.0), 5.0)
    temp_slope = _positive_slope_score(trends.get("temp_slope_c_per_h", 0.0), 1.0)
    co2_slope = _positive_slope_score(trends.get("co2_slope_ppm_per_h", 0.0), 100.0)
    vpd_debt = clamp(debts.get("vpd_debt_norm", 0.0))
    heat_debt = clamp(debts.get("heat_debt_norm", 0.0))
    dryness_debt = clamp(debts.get("dryness_debt_norm", 0.0))
    solver_warning = clamp(solver.get("solver_warning_score", 0.0))
    solver_sensitive = clamp(solver.get("hot_dry_solver_sensitive_score", 0.0))
    high_vent = clamp((action["u_ventilation"] - 0.65) / 0.35)
    high_co2 = clamp((action["u_co2"] - 0.45) / 0.55)
    high_light = clamp((action["u_lighting"] - 0.55) / 0.45)

    if regime == "NORMAL_BALANCED":
        risk = max(vpd_debt, heat_debt, dryness_debt, solver_warning, solver_sensitive)
        reasons = {"inverse_risk_score": 1.0 - risk, "data_quality_score": data_quality}
        return clamp(_mean([1.0 - risk, data_quality])), reasons
    if regime == "HEAT_ACCUMULATION":
        reasons = {"temp_slope_score": temp_slope, "heat_debt_norm": heat_debt, "data_quality_score": data_quality}
        return clamp(_mean([temp_slope, heat_debt, data_quality])), reasons
    if regime == "HIGH_VPD_DRY_STRESS":
        reasons = {
            "vpd_slope_score": vpd_slope,
            "rh_decline_score": rh_decline,
            "vpd_debt_norm": vpd_debt,
            "dryness_debt_norm": dryness_debt,
            "data_quality_score": data_quality,
        }
        return clamp(_mean([vpd_slope, rh_decline, max(vpd_debt, dryness_debt), data_quality])), reasons
    if regime == "HUMIDITY_EXCESS_DISEASE_RISK":
        # v80.1 has no explicit humidity-excess debt yet; use RH rising and
        # absence of dryness evidence as a conservative proxy.
        rh_rising = _positive_slope_score(trends.get("rh_slope_pct_per_h", 0.0), 5.0)
        low_dryness = 1.0 - dryness_debt
        reasons = {"rh_rising_score": rh_rising, "low_dryness_score": low_dryness, "data_quality_score": data_quality}
        return clamp(_mean([rh_rising, low_dryness, data_quality])), reasons
    if regime == "LOW_LIGHT_ENERGY_SAVING":
        # Level 0 often lacks PAR/radiation; avoid high evidence without data.
        reasons = {"high_lighting_score": high_light, "weather_forecast_available": clamp(data.get("weather_forecast_available", 0.0))}
        return clamp(_mean([high_light, clamp(data.get("weather_forecast_available", 0.0))]) * 0.6), reasons
    if regime == "CO2_VENTILATION_CONFLICT":
        conflict = min(high_co2, high_vent)
        reasons = {"high_co2_score": high_co2, "high_ventilation_score": high_vent, "co2_slope_score": co2_slope}
        return clamp(_mean([conflict, co2_slope, data_quality])), reasons
    if regime == "SOLVER_SENSITIVE_EMERGENCY":
        reasons = {"solver_warning_score": solver_warning, "solver_sensitive_score": solver_sensitive, "data_quality_score": data_quality}
        return clamp(max(solver_warning, solver_sensitive) * max(data_quality, 0.2)), reasons
    return 0.0, {"unhandled_regime": regime, "data_quality_score": data_quality}
