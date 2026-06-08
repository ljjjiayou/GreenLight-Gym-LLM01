"""Finite smooth sequence templates for C-STCC v80."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import ACTION_FIELDS, clamp, normalize_action
from .constraints import DEFAULT_MAX_DELTA


TEMPLATE_NAMES = (
    "hold_all",
    "previous_shift_all",
    "ventilation_ramp_limited",
    "heating_smooth_recover",
    "co2_vent_conflict_reduce",
    "screen_dwell_hold",
    "ppo_follow_limited",
    "conservative_stabilize",
)


def _blend(current: float, target: float, max_delta: float) -> float:
    if target > current:
        return clamp(min(target, current + max_delta))
    return clamp(max(target, current - max_delta))


def _max_delta_by_field(max_delta_by_field: Mapping[str, float] | None = None) -> dict[str, float]:
    raw = dict(max_delta_by_field or DEFAULT_MAX_DELTA)
    return {
        field: max(0.0, float(raw.get(field, DEFAULT_MAX_DELTA.get(field, 1.0))))
        for field in ACTION_FIELDS
    }


def _action_delta(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, float]:
    left = normalize_action(lhs)
    right = normalize_action(rhs)
    return {field: left[field] - right[field] for field in ACTION_FIELDS}


def _rate_limit_project_action(
    target: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
    max_delta_by_field: Mapping[str, float],
) -> dict[str, float]:
    normalized_target = normalize_action(target)
    normalized_reference = normalize_action(reference)
    return {
        field: _blend(
            normalized_reference[field],
            normalized_target[field],
            float(max_delta_by_field.get(field, DEFAULT_MAX_DELTA.get(field, 1.0))),
        )
        for field in ACTION_FIELDS
    }


def _project_sequence_from_reference(
    raw_sequence: Sequence[Mapping[str, Any]],
    *,
    reference_action: Mapping[str, Any],
    max_delta_by_field: Mapping[str, float],
) -> list[dict[str, float]]:
    projected: list[dict[str, float]] = []
    previous = normalize_action(reference_action)
    for target in raw_sequence:
        action = _rate_limit_project_action(
            target,
            reference=previous,
            max_delta_by_field=max_delta_by_field,
        )
        projected.append(action)
        previous = action
    return projected


def _internal_max_delta(
    sequence: Sequence[Mapping[str, Any]],
    *,
    reference_action: Mapping[str, Any],
) -> dict[str, float]:
    previous = normalize_action(reference_action)
    result = {field: 0.0 for field in ACTION_FIELDS}
    for raw_action in sequence:
        action = normalize_action(raw_action)
        for field in ACTION_FIELDS:
            result[field] = max(result[field], abs(action[field] - previous[field]))
        previous = action
    return result


def _ramp_sequence(last_action: Mapping[str, Any], target_action: Mapping[str, Any], horizon: int, max_delta: float = 0.10) -> list[dict[str, float]]:
    current = normalize_action(last_action)
    target = normalize_action(target_action)
    sequence: list[dict[str, float]] = []
    for _ in range(max(1, horizon)):
        next_action = {
            field: _blend(current[field], target[field], max_delta)
            for field in ACTION_FIELDS
        }
        sequence.append(next_action)
        current = next_action
    return sequence


def generate_sequence_with_metadata(
    template_name: str,
    *,
    last_action: Mapping[str, Any],
    horizon: int = 12,
    prior_action: Mapping[str, Any] | None = None,
    previous_sequence: Sequence[Mapping[str, Any]] | None = None,
    max_delta_by_field: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Generate a finite smooth action sequence plus template-specific metadata."""

    if template_name not in TEMPLATE_NAMES:
        raise ValueError(f"unknown sequence template: {template_name}")
    horizon = max(1, int(horizon))
    last = normalize_action(last_action)
    prior = normalize_action(prior_action or last)
    max_delta = _max_delta_by_field(max_delta_by_field)

    if template_name == "hold_all":
        return [dict(last) for _ in range(horizon)], {}
    if template_name == "previous_shift_all":
        previous = [normalize_action(action) for action in (previous_sequence or [])]
        shifted = previous[1:horizon] if len(previous) > 1 else []
        fill = shifted[-1] if shifted else last
        while len(shifted) < horizon:
            shifted.append(dict(fill))
        raw_shifted = shifted[:horizon]
        projected = _project_sequence_from_reference(
            raw_shifted,
            reference_action=last,
            max_delta_by_field=max_delta,
        )
        original_first = raw_shifted[0] if raw_shifted else dict(last)
        projected_first = projected[0] if projected else dict(last)
        projection_delta = _action_delta(projected_first, original_first)
        total_projection_l1 = sum(
            abs(projected[index][field] - normalize_action(raw_shifted[index])[field])
            for index in range(min(len(projected), len(raw_shifted)))
            for field in ACTION_FIELDS
        )
        return projected, {
            "previous_shift_projection_applied": True,
            "previous_shift_projection_reference_kind": "current_runtime_final_action",
            "previous_shift_projection_reference_action": dict(last),
            "previous_shift_projection_max_delta_by_field": max_delta,
            "previous_shift_original_first_action": dict(original_first),
            "previous_shift_projected_first_action": dict(projected_first),
            "previous_shift_first_delta_before_projection_by_field": _action_delta(original_first, last),
            "previous_shift_first_delta_after_projection_by_field": _action_delta(projected_first, last),
            "previous_shift_projection_delta_by_field": projection_delta,
            "previous_shift_projection_step_count": len(projected),
            "previous_shift_projection_total_l1": total_projection_l1,
            "previous_shift_internal_max_delta_after_projection_by_field": _internal_max_delta(
                projected,
                reference_action=last,
            ),
        }
    if template_name == "ventilation_ramp_limited":
        target = dict(last)
        target["u_ventilation"] = prior["u_ventilation"]
        return _ramp_sequence(last, target, horizon, max_delta=0.08), {}
    if template_name == "heating_smooth_recover":
        target = dict(last)
        target["u_heating"] = prior["u_heating"]
        return _ramp_sequence(last, target, horizon, max_delta=0.06), {}
    if template_name == "co2_vent_conflict_reduce":
        target = dict(last)
        if last["u_ventilation"] > 0.70:
            target["u_co2"] = min(last["u_co2"], 0.20)
        else:
            target["u_co2"] = prior["u_co2"]
        return _ramp_sequence(last, target, horizon, max_delta=0.10), {}
    if template_name == "screen_dwell_hold":
        target = dict(last)
        target["u_screen"] = last["u_screen"]
        return _ramp_sequence(last, target, horizon, max_delta=0.05), {}
    if template_name == "ppo_follow_limited":
        return _ramp_sequence(last, prior, horizon, max_delta=0.08), {}
    if template_name == "conservative_stabilize":
        target = dict(last)
        target.update(
            {
                "u_heating": min(last["u_heating"], 0.30),
                "u_co2": min(last["u_co2"], 0.20),
                "u_ventilation": min(max(last["u_ventilation"], 0.25), 0.55),
                "u_lighting": min(last["u_lighting"], 0.40),
                "u_shading": min(max(last["u_shading"], 0.20), 0.70),
            }
        )
        return _ramp_sequence(last, target, horizon, max_delta=0.06), {}
    raise AssertionError("unreachable template dispatch")


def generate_sequence(
    template_name: str,
    *,
    last_action: Mapping[str, Any],
    horizon: int = 12,
    prior_action: Mapping[str, Any] | None = None,
    previous_sequence: Sequence[Mapping[str, Any]] | None = None,
    max_delta_by_field: Mapping[str, float] | None = None,
) -> list[dict[str, float]]:
    """Generate a finite smooth action sequence from a named template."""

    sequence, _metadata = generate_sequence_with_metadata(
        template_name,
        last_action=last_action,
        horizon=horizon,
        prior_action=prior_action,
        previous_sequence=previous_sequence,
        max_delta_by_field=max_delta_by_field,
    )
    return sequence
