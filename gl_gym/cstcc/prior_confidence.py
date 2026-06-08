"""Interpretable prior confidence formulas for C-STCC v80."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .contracts import PriorCandidate, clamp, normalize_action


def candidate_count_from_confidence(confidence: float) -> int:
    conf = clamp(confidence)
    if conf <= 0.0:
        return 0
    if conf < 0.10:
        return 0
    if conf < 0.30:
        return 1
    if conf < 0.60:
        return 2
    if conf < 0.85:
        return 3
    return 4


def ppo_prior_confidence(
    *,
    base_conf: float = 0.80,
    ood_score: float = 0.0,
    rewrite_pressure: float = 0.0,
    reversal_risk: float = 0.0,
    min_conf: float = 0.05,
    max_conf: float = 1.0,
    solver_sensitive_emergency: bool = False,
) -> tuple[float, dict[str, float | str]]:
    if solver_sensitive_emergency:
        return 0.0, {"emergency_zeroed": "solver_sensitive_emergency"}
    raw = clamp(base_conf) * (1.0 - clamp(ood_score)) * math.exp(-clamp(rewrite_pressure)) * math.exp(-clamp(reversal_risk))
    conf = clamp(raw, min_conf, max_conf)
    return conf, {
        "base_conf": clamp(base_conf),
        "ood_score": clamp(ood_score),
        "rewrite_pressure": clamp(rewrite_pressure),
        "reversal_risk": clamp(reversal_risk),
        "raw_confidence": raw,
    }


def previous_plan_confidence(
    *,
    base_conf: float = 0.75,
    forecast_error_norm: float = 0.0,
    rewrite_pressure: float = 0.0,
    min_conf: float = 0.05,
    max_conf: float = 1.0,
    emergency_regime_shift: bool = False,
) -> tuple[float, dict[str, float | str]]:
    if emergency_regime_shift:
        return 0.0, {"emergency_zeroed": "emergency_regime_shift"}
    raw = clamp(base_conf) * math.exp(-clamp(forecast_error_norm)) * math.exp(-clamp(rewrite_pressure))
    conf = clamp(raw, min_conf, max_conf)
    return conf, {
        "base_conf": clamp(base_conf),
        "forecast_error_norm": clamp(forecast_error_norm),
        "rewrite_pressure": clamp(rewrite_pressure),
        "raw_confidence": raw,
    }


def conservative_prior_confidence(
    *,
    vpd_risk: float = 0.0,
    solver_risk: float = 0.0,
    crash_warning_score: float = 0.0,
    min_conf: float = 0.05,
    max_conf: float = 1.0,
) -> tuple[float, dict[str, float]]:
    raw = max(clamp(vpd_risk), clamp(solver_risk), clamp(crash_warning_score), min_conf)
    return clamp(raw, min_conf, max_conf), {
        "vpd_risk": clamp(vpd_risk),
        "solver_risk": clamp(solver_risk),
        "crash_warning_score": clamp(crash_warning_score),
        "raw_confidence": raw,
    }


def build_prior_candidate(source: str, base_action: Mapping[str, Any], confidence: float, reasons: Mapping[str, Any]) -> PriorCandidate:
    return PriorCandidate(
        source=source,
        base_action=normalize_action(base_action),
        confidence=confidence,
        confidence_reasons=dict(reasons),
        candidate_count=candidate_count_from_confidence(confidence),
    )

