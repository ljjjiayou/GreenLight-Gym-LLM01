"""Hard feasibility and soft penalty checks for C-STCC v80."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import ACTION_FIELDS, normalize_action
from .temporal_context import action_smoothness_summary


DEFAULT_MAX_DELTA = {
    "u_heating": 0.20,
    "u_co2": 0.25,
    "u_screen": 0.20,
    "u_ventilation": 0.20,
    "u_lighting": 0.30,
    "u_shading": 0.20,
}


def check_hard_constraints(
    sequence: Sequence[Mapping[str, Any]],
    *,
    previous_action: Mapping[str, Any] | None = None,
    max_delta: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    max_delta = dict(max_delta or DEFAULT_MAX_DELTA)
    violations: list[dict[str, Any]] = []
    previous = normalize_action(previous_action) if previous_action is not None else None
    for index, raw_action in enumerate(sequence):
        action = normalize_action(raw_action)
        for field in ACTION_FIELDS:
            raw_value = raw_action.get(field, 0.0)
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                violations.append({"step": index, "field": field, "reason": "non_numeric_action"})
                continue
            if value < 0.0 or value > 1.0:
                violations.append({"step": index, "field": field, "reason": "action_bounds"})
            if previous is not None and abs(action[field] - previous[field]) > max_delta.get(field, 1.0) + 1e-9:
                violations.append({"step": index, "field": field, "reason": "max_delta"})
        if action["u_co2"] > 0.50 and action["u_ventilation"] > 0.75:
            violations.append({"step": index, "field": "u_co2/u_ventilation", "reason": "co2_injection_with_high_ventilation"})
        if action["u_heating"] > 0.60 and action["u_ventilation"] > 0.75:
            violations.append({"step": index, "field": "u_heating/u_ventilation", "reason": "heating_with_strong_ventilation"})
        previous = action
    return violations


def soft_penalties(sequence: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    smooth = action_smoothness_summary(sequence)
    total_variation = float(smooth["total_variation"])
    reversal_count = float(smooth["reversal_count"])
    energy_proxy = sum(
        action.get("u_heating", 0.0) + action.get("u_lighting", 0.0) + 0.2 * action.get("u_co2", 0.0)
        for action in (normalize_action(item) for item in sequence)
    ) / max(len(sequence), 1)
    return {
        "action_total_variation_norm": min(1.0, total_variation / max(len(sequence), 1)),
        "reversal_count_norm": min(1.0, reversal_count / max(len(sequence), 1)),
        "energy_proxy_norm": min(1.0, energy_proxy),
    }

