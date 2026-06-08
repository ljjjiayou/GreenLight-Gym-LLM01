"""Confidence-aware LLM suggestion calibration for C-STCC v80."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import DEFAULT_WEIGHT_KEYS, SemanticSuggestion, clamp


def evidence_confidence_from_features(features: Mapping[str, Any]) -> float:
    """Estimate evidence confidence from normalized rule/data-quality features."""

    if not features:
        return 0.0
    values = [clamp(value) for value in features.values()]
    return sum(values) / max(len(values), 1)


def effective_confidence(
    llm_reported_confidence: float,
    evidence_confidence: float,
    *,
    mode: str = "min",
) -> float:
    llm = clamp(llm_reported_confidence)
    evidence = clamp(evidence_confidence)
    if mode == "weighted":
        return clamp(0.4 * llm + 0.6 * evidence)
    if mode != "min":
        raise ValueError(f"unsupported confidence mode: {mode}")
    return min(llm, evidence)


def regime_stability_score(active_regime_age: int, min_hold_steps: int, recent_switch_count: int = 0) -> float:
    hold_score = clamp(active_regime_age / max(min_hold_steps, 1))
    switch_penalty = 1.0 / (1.0 + max(0, recent_switch_count))
    return clamp(hold_score * switch_penalty)


def calibrate_weights(
    suggestion: SemanticSuggestion,
    default_weights: Mapping[str, float],
    weight_bounds: Mapping[str, tuple[float, float] | list[float]],
    *,
    evidence_confidence: float,
    data_quality_score: float = 1.0,
    regime_stability: float = 1.0,
    alpha_max: float = 0.30,
    confidence_mode: str = "min",
) -> tuple[dict[str, float], dict[str, Any]]:
    """Clip, blend, and smooth LLM weight suggestions into bounded weights."""

    eff = effective_confidence(
        suggestion.llm_reported_confidence if suggestion.llm_reported_confidence is not None else suggestion.regime_confidence,
        evidence_confidence,
        mode=confidence_mode,
    )
    alpha = clamp(alpha_max * eff * clamp(data_quality_score) * clamp(regime_stability), 0.0, alpha_max)
    calibrated: dict[str, float] = {}
    clipped_llm: dict[str, float] = {}
    for key in DEFAULT_WEIGHT_KEYS:
        lower, upper = weight_bounds.get(key, (0.0, 1.0))
        default_value = clamp(default_weights.get(key, 0.0), lower, upper)
        llm_value = suggestion.priority_weight_suggestions.get(key, default_value)
        clipped = clamp(llm_value, lower, upper)
        clipped_llm[key] = clipped
        calibrated[key] = clamp((1.0 - alpha) * default_value + alpha * clipped, lower, upper)

    report = {
        "confidence_mode": confidence_mode,
        "llm_reported_confidence": clamp(suggestion.llm_reported_confidence),
        "evidence_confidence": clamp(evidence_confidence),
        "effective_confidence": eff,
        "data_quality_score": clamp(data_quality_score),
        "regime_stability": clamp(regime_stability),
        "llm_influence_alpha": alpha,
        "clipped_llm_weights": clipped_llm,
    }
    return calibrated, report

