"""Finite semantic regime state machine for C-STCC v80."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import REGIMES, CalibratedSemanticState, SemanticSuggestion, clamp
from .weight_governor import calibrate_weights, effective_confidence, regime_stability_score


@dataclass(frozen=True, slots=True)
class RegimeSpec:
    name: str
    entry_condition: list[str]
    exit_condition: list[str]
    min_hold_steps: int
    allowed_transitions: tuple[str, ...]
    default_weight_profile: dict[str, float]
    tightened_constraints: tuple[str, ...] = ()
    preferred_priors: tuple[str, ...] = ()
    forbidden_conflicts: tuple[str, ...] = ()
    emergency_override: bool = False


BALANCED_WEIGHTS = {
    "temperature_tracking": 0.20,
    "humidity_recovery": 0.15,
    "vpd_risk": 0.25,
    "energy": 0.10,
    "action_smoothness": 0.15,
    "rewrite_pressure": 0.10,
    "solver_risk": 0.05,
}


REGIME_SPECS: dict[str, RegimeSpec] = {
    "NORMAL_BALANCED": RegimeSpec(
        name="NORMAL_BALANCED",
        entry_condition=["no dominant accumulated risk"],
        exit_condition=["risk debt or actuator conflict exceeds entry threshold"],
        min_hold_steps=6,
        allowed_transitions=tuple(regime for regime in REGIMES if regime != "NORMAL_BALANCED"),
        default_weight_profile=BALANCED_WEIGHTS,
        preferred_priors=("previous_plan_prior", "ppo_prior", "rule_prior"),
    ),
    "HEAT_ACCUMULATION": RegimeSpec(
        name="HEAT_ACCUMULATION",
        entry_condition=["temp_air above target", "heat_debt_norm rising"],
        exit_condition=["heat_debt_norm below hysteresis margin"],
        min_hold_steps=6,
        allowed_transitions=("NORMAL_BALANCED", "HIGH_VPD_DRY_STRESS", "SOLVER_SENSITIVE_EMERGENCY"),
        default_weight_profile={**BALANCED_WEIGHTS, "temperature_tracking": 0.30},
        tightened_constraints=("heating_smooth_recover", "ventilation_rate_limit"),
        preferred_priors=("rule_prior", "conservative_prior", "previous_plan_prior"),
    ),
    "HIGH_VPD_DRY_STRESS": RegimeSpec(
        name="HIGH_VPD_DRY_STRESS",
        entry_condition=["vpd_air above safe limit", "vpd_slope_short positive", "rh_slope_short negative"],
        exit_condition=["vpd_air below safe limit minus margin", "rh_slope_short nonnegative"],
        min_hold_steps=8,
        allowed_transitions=("NORMAL_BALANCED", "HEAT_ACCUMULATION", "SOLVER_SENSITIVE_EMERGENCY"),
        default_weight_profile={**BALANCED_WEIGHTS, "vpd_risk": 0.40, "humidity_recovery": 0.25},
        tightened_constraints=("ventilation_reversal_limit", "max_ventilation_delta"),
        preferred_priors=("conservative_prior", "previous_plan_prior", "rule_prior"),
        forbidden_conflicts=("strong_ventilation_when_dryness_debt_high",),
    ),
    "HUMIDITY_EXCESS_DISEASE_RISK": RegimeSpec(
        name="HUMIDITY_EXCESS_DISEASE_RISK",
        entry_condition=["rh_air above disease-risk threshold", "humidity_excess_debt_norm rising"],
        exit_condition=["rh_air below threshold minus hysteresis margin"],
        min_hold_steps=8,
        allowed_transitions=("NORMAL_BALANCED", "CO2_VENTILATION_CONFLICT", "SOLVER_SENSITIVE_EMERGENCY"),
        default_weight_profile={**BALANCED_WEIGHTS, "humidity_recovery": 0.30},
        tightened_constraints=("screen_dwell_hold", "ventilation_rate_limit"),
        preferred_priors=("rule_prior", "conservative_prior"),
    ),
    "LOW_LIGHT_ENERGY_SAVING": RegimeSpec(
        name="LOW_LIGHT_ENERGY_SAVING",
        entry_condition=["low radiation or low PAR", "energy pressure high"],
        exit_condition=["radiation recovers", "light deficit clears"],
        min_hold_steps=12,
        allowed_transitions=("NORMAL_BALANCED", "HEAT_ACCUMULATION"),
        default_weight_profile={**BALANCED_WEIGHTS, "energy": 0.20},
        tightened_constraints=("lamp_energy_saving_hold",),
        preferred_priors=("previous_plan_prior", "rule_prior"),
    ),
    "CO2_VENTILATION_CONFLICT": RegimeSpec(
        name="CO2_VENTILATION_CONFLICT",
        entry_condition=["co2 injection high", "ventilation high"],
        exit_condition=["co2 or ventilation conflict falls below hysteresis margin"],
        min_hold_steps=4,
        allowed_transitions=("NORMAL_BALANCED", "HIGH_VPD_DRY_STRESS", "HUMIDITY_EXCESS_DISEASE_RISK"),
        default_weight_profile={**BALANCED_WEIGHTS, "energy": 0.15},
        tightened_constraints=("co2_vent_conflict_reduce",),
        preferred_priors=("rule_prior", "conservative_prior"),
        forbidden_conflicts=("co2_injection_with_high_ventilation",),
    ),
    "SOLVER_SENSITIVE_EMERGENCY": RegimeSpec(
        name="SOLVER_SENSITIVE_EMERGENCY",
        entry_condition=["state approaches numerical instability or runtime warning appears"],
        exit_condition=["solver risk clears for multiple steps"],
        min_hold_steps=1,
        allowed_transitions=REGIMES,
        default_weight_profile={**BALANCED_WEIGHTS, "solver_risk": 0.30, "action_smoothness": 0.25},
        tightened_constraints=("conservative_stabilize", "max_action_delta"),
        preferred_priors=("conservative_prior", "rule_prior"),
        emergency_override=True,
    ),
}


def validate_regime(regime: str) -> bool:
    return regime in REGIME_SPECS


def evaluate_transition(
    active_regime: str,
    regime_age_steps: int,
    suggestion: SemanticSuggestion,
    *,
    evidence_confidence: float,
    data_quality_score: float,
    recent_switch_count: int = 0,
    default_weights: Mapping[str, float] | None = None,
    weight_bounds: Mapping[str, tuple[float, float] | list[float]] | None = None,
    alpha_max: float = 0.30,
    confidence_mode: str = "min",
) -> CalibratedSemanticState:
    """Evaluate the semantic transition and calibrated weight state."""

    if active_regime not in REGIME_SPECS:
        active_regime = "NORMAL_BALANCED"
    current_spec = REGIME_SPECS[active_regime]
    invalid_regime = None
    target_regime = suggestion.regime
    if not validate_regime(target_regime):
        invalid_regime = target_regime
        target_regime = active_regime
        transition_allowed = False
        transition_reason = "invalid_regime_kept_active"
    else:
        target_spec = REGIME_SPECS[target_regime]
        eff = effective_confidence(
            suggestion.llm_reported_confidence if suggestion.llm_reported_confidence is not None else suggestion.regime_confidence,
            evidence_confidence,
            mode=confidence_mode,
        )
        emergency = target_spec.emergency_override
        hold_satisfied = regime_age_steps >= current_spec.min_hold_steps
        transition_allowed = target_regime == active_regime or emergency or (
            target_regime in current_spec.allowed_transitions and hold_satisfied and eff >= 0.70
        )
        if target_regime == active_regime:
            transition_reason = "same_regime"
        elif emergency:
            transition_reason = "emergency_override"
        elif not hold_satisfied:
            transition_reason = "min_hold_not_satisfied"
            target_regime = active_regime
        elif eff < 0.70:
            transition_reason = "effective_confidence_below_threshold"
            target_regime = active_regime
        elif target_regime not in current_spec.allowed_transitions:
            transition_reason = "transition_not_allowed"
            target_regime = active_regime
        else:
            transition_reason = "allowed_by_state_machine"

    active_after = target_regime if transition_allowed else active_regime
    spec_after = REGIME_SPECS[active_after]
    min_hold = max(current_spec.min_hold_steps, 1)
    stability = regime_stability_score(regime_age_steps, min_hold, recent_switch_count=recent_switch_count)
    default_weights = default_weights or spec_after.default_weight_profile
    weight_bounds = weight_bounds or {key: (0.0, 1.0) for key in default_weights}
    calibrated, report = calibrate_weights(
        suggestion,
        default_weights,
        weight_bounds,
        evidence_confidence=evidence_confidence,
        data_quality_score=data_quality_score,
        regime_stability=stability,
        alpha_max=alpha_max,
        confidence_mode=confidence_mode,
    )
    return CalibratedSemanticState(
        active_regime=active_after,
        regime_age_steps=regime_age_steps + 1 if active_after == active_regime else 0,
        regime_confidence=clamp(suggestion.regime_confidence),
        effective_confidence=report["effective_confidence"],
        calibrated_weights=calibrated,
        tightened_constraints=list(spec_after.tightened_constraints),
        llm_influence_alpha=report["llm_influence_alpha"],
        transition_allowed=transition_allowed,
        transition_reason=transition_reason,
        invalid_regime=invalid_regime,
        data_quality_score=data_quality_score,
        regime_stability=stability,
    )
