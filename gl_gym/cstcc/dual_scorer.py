"""Level-0 sequence-only scoring and Tomato Safety dual-score records."""

from __future__ import annotations

from typing import Mapping

from .contracts import SafetyProjectionReport, ScoreBreakdown, clamp
from .constraints import soft_penalties


REASON_SEVERITY_WEIGHT = {
    "minor": 0.02,
    "medium": 0.05,
    "high": 0.10,
    "critical": 0.20,
}


def rewrite_penalty(report: SafetyProjectionReport, *, lambda_rewrite: float = 0.25) -> float:
    reason_penalty = 0.0
    for severity in report.rewrite_reason_severity.values():
        reason_penalty += REASON_SEVERITY_WEIGHT.get(severity, 0.05)
    return max(0.0, lambda_rewrite * report.rewrite_magnitude + reason_penalty)


def level0_sequence_score(sequence: list[dict[str, float]], penalties: Mapping[str, float] | None = None) -> tuple[float, dict[str, float]]:
    penalties = dict(penalties or soft_penalties(sequence))
    smoothness = 1.0 - clamp(penalties.get("action_total_variation_norm", 0.0))
    reversal = 1.0 - clamp(penalties.get("reversal_count_norm", 0.0))
    energy = 1.0 - clamp(penalties.get("energy_proxy_norm", 0.0))
    components = {
        "smoothness": smoothness,
        "low_reversal": reversal,
        "energy_proxy": energy,
    }
    raw_score = 0.45 * smoothness + 0.35 * reversal + 0.20 * energy
    return raw_score, components


def score_dual(
    raw_sequence: list[dict[str, float]],
    projected_sequence: list[dict[str, float]],
    projection_report: SafetyProjectionReport,
    *,
    uncertainty_penalty: float = 0.0,
    lambda_rewrite: float = 0.25,
) -> ScoreBreakdown:
    raw_score, raw_components = level0_sequence_score(raw_sequence)
    projected_score, projected_components = level0_sequence_score(projected_sequence)
    penalty = rewrite_penalty(projection_report, lambda_rewrite=lambda_rewrite)
    final_score = projected_score - penalty - max(0.0, uncertainty_penalty)
    return ScoreBreakdown(
        raw_score=raw_score,
        projected_score=projected_score,
        normalized_components={
            **{f"raw_{key}": value for key, value in raw_components.items()},
            **{f"projected_{key}": value for key, value in projected_components.items()},
        },
        final_score=final_score,
        uncertainty_penalty=max(0.0, uncertainty_penalty),
        rewrite_penalty=penalty,
    )

