"""v8174 offline risk-specific conservative template design audit.

This stage evaluates evaluator-only pseudo templates on existing C-STCC shadow
traces. The pseudo templates are not added to the runtime template pool and do
not change final actions, online LLM usage, predictive rollout, or Tomato
Safety projection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.cstcc.constraints import DEFAULT_MAX_DELTA, check_hard_constraints, soft_penalties
from gl_gym.cstcc.contracts import ACTION_FIELDS, clamp, normalize_action
from gl_gym.cstcc.dual_scorer import level0_sequence_score
from gl_gym.experiments.cstcc_v8173_combo_specific_energy_formula_iteration import (
    CANOPY_DEW_DEW_COMBO,
    DEFAULT_JSONL_ROOT,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TRACE_CSV,
    DEFAULT_V81731_JSON,
    DEFAULT_V81732_JSON,
    DEFAULT_V81732_POLICY_CURVE_CSV,
    DEFAULT_V81733_JSON,
    DEFAULT_V81734_JSON,
    DEFAULT_OUTPUT_PREFIX as DEFAULT_V81735_OUTPUT_PREFIX,
    ENERGY_WEIGHT,
    _bonus_vector,
    _v81734_golden_issues,
    formula_projected_energy_score,
)
from gl_gym.experiments.cstcc_v8173_conflict_resolver_policy_iteration import (
    _v81731_golden_issues,
    annotate_records_for_policy,
)
from gl_gym.experiments.cstcc_v8173_energy_proxy_formula_redesign import (
    _policy_local_best_vectors,
    enrich_records_with_candidate_deltas,
)
from gl_gym.experiments.cstcc_v8173_energy_proxy_reweight_design import (
    _v81732_golden_issues,
)
from gl_gym.experiments.cstcc_v8173_risk_type_specific_bonus_template_counterfactual import (
    _bool,
    _boundary_clean,
    _boundary_summary,
    _build_step_records,
    _compact_json,
    _load_csv,
    _load_json,
    _load_jsonl_rows,
    _missing_schema_fields,
    _num,
    _resolve,
    _source,
    _template,
    _write_csv,
    bonus_for_family,
)


DEFAULT_V81735_JSON = DEFAULT_V81735_OUTPUT_PREFIX.with_suffix(".json")
DEFAULT_OUTPUT_PREFIX = Path(
    "gl_gym/result/diagnostics/cstcc_v8174_risk_specific_template_design_20260608_y2020_d240_s42_n720"
)

PSEUDO_TEMPLATES = (
    "conservative_dew_stabilize",
    "conservative_canopy_dew_warmup",
    "conservative_dew_pulse_heat_then_vent",
    "conservative_dry_vent_guard",
)
DEW_TEMPLATES = {
    "conservative_dew_stabilize",
    "conservative_canopy_dew_warmup",
    "conservative_dew_pulse_heat_then_vent",
}
HUMIDITY_GATE_FIELDS = ("temp_air", "rh_air", "temp_out", "rh_out")
TARGET_RATE_LOW = 0.35
TARGET_RATE_HIGH = 0.75


def saturation_vapor_pressure_kpa(temp_c: float) -> float:
    return 0.61094 * math.exp((17.625 * temp_c) / max(temp_c + 243.04, 1e-6))


def actual_vapor_pressure_kpa(temp_c: float, rh_percent: float) -> float:
    return saturation_vapor_pressure_kpa(temp_c) * clamp(rh_percent / 100.0)


def absolute_humidity_g_m3(temp_c: float, rh_percent: float) -> float:
    vapor_hpa = actual_vapor_pressure_kpa(temp_c, rh_percent) * 10.0
    return 216.7 * vapor_hpa / max(temp_c + 273.15, 1e-6)


def dew_point_c(temp_c: float, rh_percent: float) -> float:
    rh = min(max(rh_percent, 1e-6), 100.0)
    alpha = math.log(rh / 100.0) + (17.625 * temp_c) / (243.04 + temp_c)
    return 243.04 * alpha / max(17.625 - alpha, 1e-6)


def humidity_gate(record: Mapping[str, Any]) -> dict[str, Any]:
    merged = record.get("merged") or {}
    missing = [field for field in HUMIDITY_GATE_FIELDS if field not in merged and field not in record]
    if missing:
        return {
            "humidity_gate_valid": False,
            "missing_humidity_gate_fields": missing,
            "outside_air_dryer": False,
            "ventilation_allowed_after_warmup": False,
        }
    temp_air = _num(merged.get("temp_air"), _num(record.get("temp_air")))
    rh_air = _num(merged.get("rh_air"), _num(record.get("rh_air")))
    temp_out = _num(merged.get("temp_out"), _num(record.get("temp_out")))
    rh_out = _num(merged.get("rh_out"), _num(record.get("rh_out")))
    indoor_ah = absolute_humidity_g_m3(temp_air, rh_air)
    outdoor_ah = absolute_humidity_g_m3(temp_out, rh_out)
    indoor_dew = dew_point_c(temp_air, rh_air)
    outdoor_dew = dew_point_c(temp_out, rh_out)
    ah_gap = indoor_ah - outdoor_ah
    dew_gap = indoor_dew - outdoor_dew
    outside_dryer = bool(ah_gap >= 0.40 or dew_gap >= 0.50)
    return {
        "humidity_gate_valid": True,
        "missing_humidity_gate_fields": [],
        "temp_air": temp_air,
        "rh_air": rh_air,
        "temp_out": temp_out,
        "rh_out": rh_out,
        "indoor_abs_humidity_g_m3": indoor_ah,
        "outdoor_abs_humidity_g_m3": outdoor_ah,
        "inside_minus_outside_abs_humidity_g_m3": ah_gap,
        "indoor_dew_point_c": indoor_dew,
        "outdoor_dew_point_c": outdoor_dew,
        "inside_minus_outside_dew_point_c": dew_gap,
        "outside_air_dryer": outside_dryer,
        "ventilation_allowed_after_warmup": outside_dryer,
    }


def _risk_flags(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("risk_flags") or {}


def _resolver(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("resolver") or {}


def _family(record: Mapping[str, Any]) -> str:
    return str(_resolver(record).get("resolved_risk_family_new") or "none")


def _combo_id(record: Mapping[str, Any]) -> str:
    return str(_resolver(record).get("risk_combo_id") or "none")


def _has_dew_or_canopy_risk(record: Mapping[str, Any]) -> bool:
    flags = _risk_flags(record)
    return bool(
        _bool(flags.get("dew_risk"))
        or _bool(flags.get("humidity_or_dew_risk"))
        or _bool(flags.get("canopy_dew_margin_lt1"))
        or _bool(flags.get("canopy_dew_margin_lt0"))
    )


def _has_dry_or_high_vpd_risk(record: Mapping[str, Any]) -> bool:
    flags = _risk_flags(record)
    return bool(_bool(flags.get("dry_risk")) or _bool(flags.get("high_vpd")) or _bool(flags.get("dry_or_high_vpd_risk")))


def _has_dry_vent_risk(record: Mapping[str, Any]) -> bool:
    return bool(_bool(_risk_flags(record).get("dry_vent_risk")))


def _blend(current: float, target: float, max_delta: float) -> float:
    if target > current:
        return clamp(min(target, current + max_delta))
    return clamp(max(target, current - max_delta))


def _step_toward(current: Mapping[str, Any], target: Mapping[str, Any], max_delta_by_field: Mapping[str, float]) -> dict[str, float]:
    current_action = normalize_action(current)
    target_action = normalize_action(target)
    return {
        field: _blend(current_action[field], target_action[field], float(max_delta_by_field.get(field, DEFAULT_MAX_DELTA[field])))
        for field in ACTION_FIELDS
    }


def _delta(lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> dict[str, float]:
    left = normalize_action(lhs)
    right = normalize_action(rhs)
    return {field: left[field] - right[field] for field in ACTION_FIELDS}


def _target_screen_slit(last: Mapping[str, float]) -> float:
    # In the existing action convention, lower u_screen means a more open slit.
    return clamp(min(last["u_screen"], max(0.0, last["u_screen"] - 0.06)))


def _pulse_window(hour: float) -> bool:
    return bool(4.0 <= hour <= 8.0 or 16.0 <= hour <= 21.0)


def _template_applicable(template_name: str, record: Mapping[str, Any]) -> tuple[bool, str]:
    if template_name in DEW_TEMPLATES:
        if not _has_dew_or_canopy_risk(record):
            return False, "dew_template_requires_dew_or_canopy_risk"
        if _has_dry_or_high_vpd_risk(record) and not _has_dew_or_canopy_risk(record):
            return False, "dew_template_blocked_for_dry_only_risk"
        return True, "dew_or_canopy_risk_active"
    if template_name == "conservative_dry_vent_guard":
        if _has_dry_vent_risk(record) or _has_dry_or_high_vpd_risk(record):
            return True, "dry_or_high_vpd_risk_active"
        return False, "dry_guard_requires_dry_or_high_vpd_risk"
    return False, "unknown_template"


def generate_pseudo_template_sequence(
    template_name: str,
    *,
    record: Mapping[str, Any],
    horizon: int = 12,
    max_delta_by_field: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    if template_name not in PSEUDO_TEMPLATES:
        raise ValueError(f"unknown v8174 pseudo template: {template_name}")
    last = normalize_action(record.get("current_runtime_final_action") or {})
    max_delta = dict(max_delta_by_field or DEFAULT_MAX_DELTA)
    gate = humidity_gate(record)
    hour = _num(record.get("hour_of_day"), _num((record.get("merged") or {}).get("hour_of_day")))
    outside_dryer = bool(gate.get("outside_air_dryer"))
    pulse_active = _pulse_window(hour)

    applicable, reason = _template_applicable(template_name, record)
    if not applicable:
        return [dict(last) for _ in range(max(1, horizon))], {
            "pseudo_template_name": template_name,
            "template_applicable": False,
            "template_inapplicable_reason": reason,
            "humidity_gate": gate,
            "phase_plan": ["inapplicable_hold"],
            "screen_intent": "none",
            "ventilation_policy": "none",
        }

    current = dict(last)
    sequence: list[dict[str, float]] = []
    phase_plan: list[str] = []
    for index in range(max(1, horizon)):
        target = dict(current)
        phase = "hold"
        if template_name == "conservative_dew_stabilize":
            phase = "dew_warm_stabilize"
            target["u_heating"] = min(max(last["u_heating"] + 0.10, 0.25), 0.55)
            target["u_screen"] = _target_screen_slit(last)
            target["u_ventilation"] = min(max(last["u_ventilation"] + (0.05 if outside_dryer else 0.0), 0.25), 0.55)
        elif template_name == "conservative_canopy_dew_warmup":
            if index < 4:
                phase = "canopy_warmup_screen_slit_before_vent"
                target["u_heating"] = min(max(last["u_heating"] + 0.14, 0.30), 0.58)
                target["u_screen"] = _target_screen_slit(last)
                target["u_ventilation"] = min(last["u_ventilation"], 0.38)
            else:
                phase = "canopy_post_warmup_vent_if_outside_dryer"
                target["u_heating"] = min(max(last["u_heating"] + 0.10, 0.28), 0.55)
                target["u_screen"] = _target_screen_slit(last)
                target["u_ventilation"] = min(max(last["u_ventilation"] + (0.08 if outside_dryer else 0.0), 0.25), 0.58)
        elif template_name == "conservative_dew_pulse_heat_then_vent":
            if not pulse_active:
                phase = "pulse_window_inactive_warmup_only"
                target["u_heating"] = min(max(last["u_heating"] + 0.08, 0.25), 0.50)
                target["u_screen"] = _target_screen_slit(last)
                target["u_ventilation"] = min(last["u_ventilation"], 0.40)
            elif index < 4:
                phase = "pulse_heat_screen_slit_before_vent"
                target["u_heating"] = min(max(last["u_heating"] + 0.16, 0.32), 0.60)
                target["u_screen"] = _target_screen_slit(last)
                target["u_ventilation"] = min(last["u_ventilation"], 0.38)
            else:
                phase = "pulse_vent_after_heat_if_outside_dryer"
                target["u_heating"] = min(max(last["u_heating"] + 0.10, 0.28), 0.55)
                target["u_screen"] = _target_screen_slit(last)
                target["u_ventilation"] = min(max(last["u_ventilation"] + (0.12 if outside_dryer else 0.0), 0.25), 0.62)
        elif template_name == "conservative_dry_vent_guard":
            phase = "dry_vent_guard_reduce_ventilation"
            target["u_heating"] = min(last["u_heating"], 0.30)
            target["u_ventilation"] = min(max(last["u_ventilation"] - 0.16, 0.15), 0.45)
            target["u_shading"] = min(max(last["u_shading"], 0.25), 0.75)
            target["u_co2"] = min(last["u_co2"], 0.20)
        action = _step_toward(current, target, max_delta)
        sequence.append(action)
        phase_plan.append(phase)
        current = action

    first_delta = _delta(sequence[0], last)
    vent_increase_steps = sum(1 for action in sequence if action["u_ventilation"] > last["u_ventilation"] + 1e-9)
    metadata = {
        "pseudo_template_name": template_name,
        "template_applicable": True,
        "template_inapplicable_reason": None,
        "humidity_gate": gate,
        "phase_plan": phase_plan,
        "phase_distribution": {phase: phase_plan.count(phase) for phase in sorted(set(phase_plan))},
        "screen_intent": "lower_u_screen_means_screen_slit_before_vent" if template_name in DEW_TEMPLATES else "unchanged",
        "ventilation_policy": (
            "vent_after_warmup_only_if_outside_air_dryer"
            if template_name in DEW_TEMPLATES
            else "reduce_or_cap_ventilation_under_dry_risk"
        ),
        "pulse_window_active": pulse_active,
        "outside_dryer_gate_applied": template_name in DEW_TEMPLATES,
        "ventilation_increase_steps": vent_increase_steps,
        "first_action_delta_from_reference": first_delta,
    }
    return sequence, metadata


def _candidate_from_sequence(
    *,
    record: Mapping[str, Any],
    template_name: str,
    sequence: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    last = normalize_action(record.get("current_runtime_final_action") or {})
    violations = check_hard_constraints(sequence, previous_action=last, max_delta=DEFAULT_MAX_DELTA)
    reason_distribution: dict[str, int] = {}
    field_distribution: dict[str, int] = {}
    for violation in violations:
        reason = str(violation.get("reason") or "unknown")
        field = str(violation.get("field") or "unknown")
        reason_distribution[reason] = reason_distribution.get(reason, 0) + 1
        field_distribution[field] = field_distribution.get(field, 0) + 1
    return {
        "candidate_id": f"conservative_prior:{template_name}:v8174",
        "source_prior": "conservative_prior",
        "template_name": template_name,
        "feasible": not violations and bool(metadata.get("template_applicable", True)),
        "violation_reason_distribution": dict(sorted(reason_distribution.items())),
        "violation_field_distribution": dict(sorted(field_distribution.items())),
        "first_action_delta_from_reference": dict(metadata.get("first_action_delta_from_reference") or _delta(sequence[0], last)),
        "template_metadata": dict(metadata),
        "raw_sequence": [normalize_action(action) for action in sequence],
    }


def _component_from_sequence(sequence: Sequence[Mapping[str, Any]], *, prior_bonus: float) -> dict[str, float]:
    score, components = level0_sequence_score([normalize_action(action) for action in sequence])
    return {
        "base_final_score": score,
        "final_score": score + prior_bonus,
        "prior_confidence_bonus": prior_bonus,
        "projected_energy_proxy": components["energy_proxy"],
        "raw_energy_proxy": components["energy_proxy"],
        "projected_smoothness": components["smoothness"],
        "raw_smoothness": components["smoothness"],
        "projected_low_reversal": components["low_reversal"],
        "raw_low_reversal": components["low_reversal"],
        "risk_static_hold_penalty": 0.0,
        "rewrite_penalty": 0.0,
        "uncertainty_penalty": 0.0,
    }


def _conservative_prior_bonus(record: Mapping[str, Any]) -> float:
    conservative_id = str(record.get("conservative_candidate_id") or "")
    components = record.get("components") or {}
    existing = components.get(conservative_id) or {}
    return _num(existing.get("prior_confidence_bonus"), 0.004)


def pseudo_candidates_for_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for template in PSEUDO_TEMPLATES:
        sequence, metadata = generate_pseudo_template_sequence(template, record=record)
        candidate = _candidate_from_sequence(record=record, template_name=template, sequence=sequence, metadata=metadata)
        component = _component_from_sequence(sequence, prior_bonus=_conservative_prior_bonus(record))
        candidates.append({**candidate, "score_component": component})
    return candidates


def _score_existing_candidate(
    *,
    formula: str,
    record: Mapping[str, Any],
    candidate_id: str,
    candidate: Mapping[str, Any],
    component: Mapping[str, Any],
    bonus_vector: Mapping[str, Any],
) -> float:
    family = _family(record)
    score = _num(component.get("final_score"), -1e18)
    if candidate.get("source_prior") == "conservative_prior" and _risk_flags(record).get("any_risk", False) and family != "none":
        score += bonus_for_family(bonus_vector, family)
    if formula == "formula_G_combo_specific_two_sided_semantics":
        formula_energy = formula_projected_energy_score(formula, record=record, candidate=candidate, component=component)
        score += ENERGY_WEIGHT * (formula_energy - _num(component.get("projected_energy_proxy")))
    return score


def winner_with_pseudo_templates(
    record: Mapping[str, Any],
    *,
    formula: str,
    bonus_vector: Mapping[str, Any],
) -> tuple[str, float, dict[str, Any]]:
    best_id = ""
    best_score = -1e18
    best_meta: dict[str, Any] = {}
    for candidate_id, component in (record.get("components") or {}).items():
        candidate = (record.get("candidates") or {}).get(candidate_id)
        if candidate is None or not candidate.get("feasible", False):
            continue
        score = _score_existing_candidate(
            formula=formula,
            record=record,
            candidate_id=candidate_id,
            candidate=candidate,
            component=component,
            bonus_vector=bonus_vector,
        )
        if score > best_score:
            best_id = candidate_id
            best_score = score
            best_meta = {"candidate_kind": "existing_runtime_shadow_candidate"}
    for candidate in pseudo_candidates_for_record(record):
        if not candidate.get("feasible", False):
            continue
        component = candidate.get("score_component") or {}
        score = _score_existing_candidate(
            formula=formula,
            record=record,
            candidate_id=str(candidate.get("candidate_id")),
            candidate=candidate,
            component=component,
            bonus_vector=bonus_vector,
        )
        if score > best_score:
            best_id = str(candidate.get("candidate_id"))
            best_score = score
            best_meta = {
                "candidate_kind": "v8174_evaluator_only_pseudo_template",
                "template_name": candidate.get("template_name"),
                "template_metadata": candidate.get("template_metadata", {}),
            }
    return best_id, best_score, best_meta


def _template_curve_rows(records: Sequence[Mapping[str, Any]], *, policy_best_vectors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, vector_row in policy_best_vectors.items():
        annotated = annotate_records_for_policy(records, policy)
        bonus_vector = _bonus_vector(vector_row)
        for formula in ("baseline_current_energy", "formula_G_combo_specific_two_sided_semantics"):
            risk_count = 0
            pseudo_selected = 0
            nonrisk_pseudo_selected = 0
            canopy_combo_count = 0
            canopy_combo_selected = 0
            dry_vent_count = 0
            dry_vent_selected = 0
            selected_templates: dict[str, int] = {}
            only_formula_selected_steps = 0
            for record in annotated:
                is_risk = bool(_risk_flags(record).get("any_risk", False))
                if is_risk:
                    risk_count += 1
                winner_id, _score, meta = winner_with_pseudo_templates(record, formula=formula, bonus_vector=bonus_vector)
                is_pseudo = meta.get("candidate_kind") == "v8174_evaluator_only_pseudo_template"
                template = str(meta.get("template_name") or _template(winner_id))
                if is_pseudo:
                    selected_templates[template] = selected_templates.get(template, 0) + 1
                    if is_risk:
                        pseudo_selected += 1
                    else:
                        nonrisk_pseudo_selected += 1
                if _combo_id(record) == CANOPY_DEW_DEW_COMBO:
                    canopy_combo_count += 1
                    if is_pseudo:
                        canopy_combo_selected += 1
                if _family(record) == "dry_vent":
                    dry_vent_count += 1
                    if is_pseudo:
                        dry_vent_selected += 1
                if formula == "formula_G_combo_specific_two_sided_semantics":
                    baseline_winner, _baseline_score, baseline_meta = winner_with_pseudo_templates(
                        record,
                        formula="baseline_current_energy",
                        bonus_vector=bonus_vector,
                    )
                    if is_pseudo and baseline_meta.get("candidate_kind") != "v8174_evaluator_only_pseudo_template":
                        only_formula_selected_steps += 1
            rows.append(
                {
                    "policy": policy,
                    "formula": formula,
                    "policy_local_bonus_vector_id": str(vector_row.get("vector_id") or ""),
                    "row_count": len(annotated),
                    "risk_row_count": risk_count,
                    "pseudo_template_risk_selected_count": pseudo_selected,
                    "pseudo_template_risk_selected_rate": pseudo_selected / max(risk_count, 1),
                    "nonrisk_pseudo_template_selected_count": nonrisk_pseudo_selected,
                    "canopy_dew_lt0_dew_combo_row_count": canopy_combo_count,
                    "canopy_dew_lt0_dew_combo_pseudo_selected_count": canopy_combo_selected,
                    "dry_vent_row_count": dry_vent_count,
                    "dry_vent_pseudo_selected_count": dry_vent_selected,
                    "selected_pseudo_template_distribution": selected_templates,
                    "only_formula_g_selected_steps": only_formula_selected_steps,
                }
            )
    return rows


def _template_feasibility_rows(records: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for template in PSEUDO_TEMPLATES:
        generated = 0
        applicable = 0
        feasible = 0
        max_delta_violations = 0
        over_ventilation_candidates = 0
        dry_risk_dew_misfires = 0
        for record in annotate_records_for_policy(records, policy):
            for candidate in pseudo_candidates_for_record(record):
                if candidate["template_name"] != template:
                    continue
                generated += 1
                metadata = candidate.get("template_metadata") or {}
                if metadata.get("template_applicable", False):
                    applicable += 1
                if candidate.get("feasible", False):
                    feasible += 1
                max_delta_violations += int((candidate.get("violation_reason_distribution") or {}).get("max_delta", 0) or 0)
                first_delta = candidate.get("first_action_delta_from_reference") or {}
                if template in DEW_TEMPLATES and _has_dry_or_high_vpd_risk(record) and not _has_dew_or_canopy_risk(record):
                    if metadata.get("template_applicable", False):
                        dry_risk_dew_misfires += 1
                if _num(first_delta.get("u_ventilation")) > 0.15:
                    over_ventilation_candidates += 1
        rows.append(
            {
                "policy": policy,
                "template_name": template,
                "generated_count": generated,
                "applicable_count": applicable,
                "feasible_count": feasible,
                "max_delta_violation_count": max_delta_violations,
                "over_ventilation_candidate_count": over_ventilation_candidates,
                "dry_risk_dew_template_misfire_count": dry_risk_dew_misfires,
            }
        )
    return rows


def _dew_phase_rows(records: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in annotate_records_for_policy(records, policy):
        if not _has_dew_or_canopy_risk(record):
            continue
        for candidate in pseudo_candidates_for_record(record):
            if candidate.get("template_name") not in DEW_TEMPLATES:
                continue
            metadata = candidate.get("template_metadata") or {}
            first_delta = candidate.get("first_action_delta_from_reference") or {}
            rows.append(
                {
                    "step": record.get("step"),
                    "hour_of_day": record.get("hour_of_day"),
                    "risk_combo_id": _combo_id(record),
                    "template_name": candidate.get("template_name"),
                    "template_applicable": metadata.get("template_applicable"),
                    "pulse_window_active": metadata.get("pulse_window_active"),
                    "outside_air_dryer": (metadata.get("humidity_gate") or {}).get("outside_air_dryer"),
                    "ventilation_increase_steps": metadata.get("ventilation_increase_steps"),
                    "first_delta_u_heating": first_delta.get("u_heating"),
                    "first_delta_u_screen": first_delta.get("u_screen"),
                    "first_delta_u_ventilation": first_delta.get("u_ventilation"),
                    "phase_distribution": metadata.get("phase_distribution", {}),
                }
            )
    return rows


def _outside_air_gate_rows(records: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in annotate_records_for_policy(records, policy):
        if not _risk_flags(record).get("any_risk", False):
            continue
        gate = humidity_gate(record)
        rows.append(
            {
                "step": record.get("step"),
                "hour_of_day": record.get("hour_of_day"),
                "risk_combo_id": _combo_id(record),
                "resolved_risk_family": _family(record),
                **gate,
            }
        )
    return rows


def _best_curve_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        canopy_selected = int(row.get("canopy_dew_lt0_dew_combo_pseudo_selected_count", 0) or 0)
        dry_total = int(row.get("dry_vent_row_count", 0) or 0)
        dry_selected = int(row.get("dry_vent_pseudo_selected_count", 0) or 0)
        nonrisk = int(row.get("nonrisk_pseudo_template_selected_count", 0) or 0)
        rate = _num(row.get("pseudo_template_risk_selected_rate"))
        rate_in_band = TARGET_RATE_LOW <= rate <= TARGET_RATE_HIGH
        baseline_formula = row.get("formula") == "baseline_current_energy"
        return (
            nonrisk == 0,
            canopy_selected > 0,
            dry_total == 0 or dry_selected < dry_total,
            rate_in_band,
            baseline_formula,
            canopy_selected,
            -dry_selected,
            -nonrisk,
            -abs(rate - 0.55),
        )

    return dict(max(rows, key=key)) if rows else {}


def _schema_humidity_issues(records: Sequence[Mapping[str, Any]]) -> list[str]:
    issues: list[str] = []
    for record in records:
        if not _risk_flags(record).get("any_risk", False):
            continue
        gate = humidity_gate(record)
        if not gate.get("humidity_gate_valid", False):
            issues.append(
                f"humidity_gate_fields_missing:step={record.get('step')}:{','.join(gate.get('missing_humidity_gate_fields', []))}"
            )
    return sorted(set(issues))


def _route_next_action(
    *,
    boundary: Mapping[str, Any],
    schema_issues: Sequence[str],
    best_curve: Mapping[str, Any],
    feasibility_rows: Sequence[Mapping[str, Any]],
    input_trace_count: int,
) -> tuple[str, list[str]]:
    if not _boundary_clean(boundary):
        return "stop_and_repair_cstcc_shadow_boundary", ["boundary_or_candidate_health_failure"]
    if schema_issues:
        return "v8174_trace_schema_humidity_ratio_repair_plan", list(schema_issues)
    if any(int(row.get("max_delta_violation_count", 0) or 0) > 0 for row in feasibility_rows):
        return "v8174_template_parameter_iteration_plan", ["pseudo_template_max_delta_violation"]
    if any(int(row.get("dry_risk_dew_template_misfire_count", 0) or 0) > 0 for row in feasibility_rows):
        return "v8174_template_parameter_iteration_plan", ["dew_template_misfires_on_dry_only_risk"]
    canopy_total = int(best_curve.get("canopy_dew_lt0_dew_combo_row_count", 0) or 0)
    canopy_selected = int(best_curve.get("canopy_dew_lt0_dew_combo_pseudo_selected_count", 0) or 0)
    dry_total = int(best_curve.get("dry_vent_row_count", 0) or 0)
    dry_selected = int(best_curve.get("dry_vent_pseudo_selected_count", 0) or 0)
    nonrisk = int(best_curve.get("nonrisk_pseudo_template_selected_count", 0) or 0)
    if nonrisk > 0:
        return "v8174_template_parameter_iteration_plan", ["nonrisk_pseudo_template_selected"]
    if canopy_total > 0 and canopy_selected <= 0:
        return "v8174_template_parameter_iteration_plan", ["canopy_dew_dew_template_still_not_competitive"]
    if dry_total > 0 and dry_selected >= dry_total:
        return "v8174_template_parameter_iteration_plan", ["dry_vent_pseudo_template_saturated"]
    if best_curve.get("formula") == "formula_G_combo_specific_two_sided_semantics":
        return "v8174_scorer_template_joint_calibration_plan", ["template_competes_only_with_risk_state_formula_g"]
    if input_trace_count <= 1:
        return "v8174_multi_scenario_existing_trace_scan_or_acquisition_plan", ["single_h720_trace_only"]
    return "v8174_1_opt_in_risk_template_shadow_trace_plan", ["multi_scenario_existing_trace_template_design_passed"]


def _v81735_golden_issues(v81735_json: Mapping[str, Any]) -> list[str]:
    if not v81735_json:
        return ["v81735_json_missing"]
    best = ((v81735_json.get("diagnostics") or {}).get("best_formula_row") or {})
    diagnostics = (v81735_json.get("diagnostics") or {}).get("curve_resolution_diagnostics") or {}
    issues: list[str] = []
    if v81735_json.get("next_action") != "v8174_risk_specific_template_design_plan":
        issues.append("golden_mismatch:v81735_next_action")
    if best.get("formula") != "baseline_current_energy":
        issues.append("golden_mismatch:v81735_best_formula")
    if int(best.get("dry_vent_conservative_selected_count", 0) or 0) != 21:
        issues.append("golden_mismatch:v81735_best_dry_vent_selected")
    if int(best.get("canopy_dew_lt0_dew_combo_conservative_selected_count", 0) or 0) != 0:
        issues.append("golden_mismatch:v81735_best_canopy_dew_dew_selected")
    if diagnostics.get("dry_vent_resolved_by_any_nonbaseline_formula") is not True:
        issues.append("golden_mismatch:v81735_dry_vent_resolution_diag")
    if diagnostics.get("canopy_dew_dew_recovered_by_any_nonbaseline_formula") is not False:
        issues.append("golden_mismatch:v81735_canopy_recovery_diag")
    return issues


def build_report(
    *,
    trace_csv_rows: list[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    jsonl_rows: list[Mapping[str, Any]],
    v81731_json: Mapping[str, Any],
    v81732_json: Mapping[str, Any],
    v81734_json: Mapping[str, Any],
    v81735_json: Mapping[str, Any],
    v81732_policy_curve_rows: list[Mapping[str, Any]],
    input_trace_count: int = 1,
    enforce_golden: bool = True,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    boundary = _boundary_summary(summary_json, jsonl_rows)
    schema_issues = _missing_schema_fields(jsonl_rows)
    records, reconstruction_issues = _build_step_records(trace_csv_rows=trace_csv_rows, jsonl_rows=jsonl_rows)
    records, delta_issues = enrich_records_with_candidate_deltas(records, jsonl_rows)
    policy_best_vectors = _policy_local_best_vectors(v81732_policy_curve_rows)
    policy = "canopy_dew_first_v1" if "canopy_dew_first_v1" in policy_best_vectors else next(iter(policy_best_vectors), "")
    schema_issues.extend(_schema_humidity_issues(records))
    if enforce_golden:
        schema_issues = sorted(
            set(
                schema_issues
                + reconstruction_issues
                + delta_issues
                + _v81731_golden_issues(v81731_json)
                + _v81732_golden_issues(v81732_json)
                + _v81734_golden_issues(v81734_json)
                + _v81735_golden_issues(v81735_json)
            )
        )
    else:
        schema_issues = sorted(set(schema_issues + reconstruction_issues + delta_issues))

    curve_rows = _template_curve_rows(records, policy_best_vectors=policy_best_vectors)
    feasibility_rows = _template_feasibility_rows(records, policy)
    dew_phase_rows = _dew_phase_rows(records, policy)
    gate_rows = _outside_air_gate_rows(records, policy)
    best_curve = _best_curve_row(curve_rows)
    next_action, readiness_reasons = _route_next_action(
        boundary=boundary,
        schema_issues=schema_issues,
        best_curve=best_curve,
        feasibility_rows=feasibility_rows,
        input_trace_count=input_trace_count,
    )
    multi_scenario = {
        "schema_version": "cstcc_v8174_multi_scenario_readiness_v1",
        "input_trace_count": input_trace_count,
        "multi_scenario_evidence_available": input_trace_count > 1,
        "readiness": "blocked_single_h720_trace_only" if input_trace_count <= 1 else "available",
        "required_next_evidence": [
            "multiple years/days/seeds with dew/humidity, dry/high-VPD, and mixed-risk windows",
            "final-action invariant and forbidden-boundary counts remain clean",
            "pseudo template does not select in nonrisk or dry-only windows",
        ],
    }
    report = {
        "schema_version": "cstcc_v8174_risk_specific_template_design_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_level": "existing-trace offline template design audit",
        "boundaries": {
            "online_llm_called": False,
            "predictive_rollout_executed": False,
            "real_tomato_safety_projection": False,
            "final_action_changed": False,
            "performance_claim_allowed": False,
            "promotion_evidence": False,
            "runtime_scoring_changed": False,
            "runtime_template_pool_changed": False,
            "default_controller_changed": False,
            "runtime_score_dual_changed": False,
        },
        "boundary_runtime_safety": boundary,
        "schema_issues": schema_issues,
        "counterfactual_definition": {
            "pseudo_templates_evaluator_only": True,
            "pseudo_templates_added_to_runtime_template_names": False,
            "runtime_candidate_pool_changed": False,
            "runtime_final_action_changed": False,
            "energy_formula_runtime_changed": False,
            "outside_humidity_gate_is_evaluator_only": True,
            "u_screen_semantics": "lower_u_screen_is_more_open_screen_slit",
        },
        "scientific_design_assumptions": {
            "condensation_control_signal": "dew point margin and VPD, with VPD below about 0.1 kPa treated as high condensation risk in the cited design rationale",
            "control_semantics": "dew/canopy risk needs warmup plus humidity-aware ventilation, while dry/high-VPD risk needs ventilation restraint",
            "references": [
                "https://edis.ifas.ufl.edu/publication/AE030",
                "https://www.uaf.edu/ces/publications/database/gardening/controlling-greenhouse-environment.php",
                "https://www.ishs.org/ishs-article/719_39",
            ],
            "claim_boundary": "references motivate pseudo-template design only; they are not project performance evidence",
        },
        "diagnostics": {
            "observed_gap": {
                "v81735_next_action": v81735_json.get("next_action"),
                "v81735_readiness_reasons": v81735_json.get("readiness_reasons", []),
                "hypothesis_revision": {
                    "old_hypothesis": "risk-state energy formulas can resolve canopy-dew + humidity conservative selection failure",
                    "failure_evidence_or_missing_evidence": "v8173.5 showed dry-vent can be suppressed by formulas, but canopy_dew_lt0+dew_or_humidity remained 0/6",
                    "new_hypothesis": "conservative templates need explicit dew/canopy-dew action semantics before scorer calibration or opt-in runtime tracing",
                    "next_experiment": "existing-trace evaluator-only risk-specific template design audit",
                    "default_controller_changed": False,
                    "online_llm_needed": False,
                    "hard_safety_impact": "none; audit only",
                    "success_condition": "dew/canopy pseudo templates improve canopy-dew combo without dry-risk misfire or nonrisk selection",
                    "stop_condition": "humidity schema missing, template infeasible, or canopy-dew combo remains uncompetitive",
                },
            },
            "selected_policy_for_tables": policy,
            "best_curve_row": best_curve,
            "pseudo_template_names": PSEUDO_TEMPLATES,
        },
        "acceptance": {
            "boundary_clean": _boundary_clean(boundary),
            "schema_clean": not schema_issues,
            "canopy_dew_lt0_dew_combo_improved": int(
                best_curve.get("canopy_dew_lt0_dew_combo_pseudo_selected_count", 0) or 0
            )
            > 0,
            "dry_vent_not_saturated": int(best_curve.get("dry_vent_pseudo_selected_count", 0) or 0)
            < int(best_curve.get("dry_vent_row_count", 0) or 0),
            "nonrisk_pseudo_template_selected_count_eq_0": int(
                best_curve.get("nonrisk_pseudo_template_selected_count", 0) or 0
            )
            == 0,
            "multi_scenario_evidence_available": input_trace_count > 1,
        },
        "next_action": next_action,
        "readiness_reasons": readiness_reasons,
        "notes": [
            "v8174 is offline template semantics design only.",
            "Pseudo templates are not added to TEMPLATE_NAMES and do not change runtime selection.",
            "A single H720 pass is not enough for generalization; multi-scenario evidence remains required.",
        ],
    }
    return report, {
        "template_candidate_curve": curve_rows,
        "dew_phase_table": dew_phase_rows,
        "outside_air_dryness_gate_table": gate_rows,
        "template_feasibility_table": feasibility_rows,
    }, multi_scenario


def build_markdown(report: Mapping[str, Any]) -> str:
    best = (report.get("diagnostics") or {}).get("best_curve_row") or {}
    boundary = report.get("boundary_runtime_safety", {})
    lines = [
        "# C-STCC v8174 Risk-Specific Conservative Template Design",
        "",
        "Existing-trace offline template design audit only. Runtime template pool and final actions are unchanged.",
        "",
        "## Boundary",
        "",
        f"- Audit success rate: {_num(boundary.get('audit_success_rate')):.3f}",
        f"- Final-action invariant rate: {_num(boundary.get('final_action_invariant_rate')):.3f}",
        f"- Online LLM / rollout / real projection steps: {boundary.get('online_llm_called_steps', 0)} / {boundary.get('predictive_rollout_executed_steps', 0)} / {boundary.get('real_tomato_safety_projection_steps', 0)}",
        "",
        "## Best Template Counterfactual",
        "",
        f"- Policy / formula: {best.get('policy', 'none')} / {best.get('formula', 'none')}",
        f"- Pseudo-template risk selected rate: {_num(best.get('pseudo_template_risk_selected_rate')):.3f}",
        f"- Canopy-dew/dew pseudo selected: {best.get('canopy_dew_lt0_dew_combo_pseudo_selected_count', 0)} / {best.get('canopy_dew_lt0_dew_combo_row_count', 0)}",
        f"- Dry-vent pseudo selected: {best.get('dry_vent_pseudo_selected_count', 0)} / {best.get('dry_vent_row_count', 0)}",
        f"- Nonrisk pseudo selected: {best.get('nonrisk_pseudo_template_selected_count', 0)}",
        "",
        "## Decision",
        "",
        f"- Next action: {report.get('next_action')}",
        f"- Readiness reasons: {', '.join(report.get('readiness_reasons', [])) or 'none'}",
    ]
    return "\n".join(lines) + "\n"


def build_multi_scenario_markdown(readiness: Mapping[str, Any]) -> str:
    lines = [
        "# C-STCC v8174 Multi-Scenario Readiness",
        "",
        f"- Input trace count: {readiness.get('input_trace_count')}",
        f"- Multi-scenario evidence available: {readiness.get('multi_scenario_evidence_available')}",
        f"- Readiness: {readiness.get('readiness')}",
        "",
        "## Required Next Evidence",
        "",
    ]
    lines.extend(f"- {item}" for item in readiness.get("required_next_evidence", []))
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    report: Mapping[str, Any],
    tables: Mapping[str, list[Mapping[str, Any]]],
    multi_scenario: Mapping[str, Any],
    output_prefix: str | Path,
) -> dict[str, str]:
    prefix = _resolve(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    suffix = "_20260608_y2020_d240_s42_n720"
    parent = prefix.parent
    paths = {
        "json": prefix.with_suffix(".json"),
        "md": prefix.with_suffix(".md"),
        "template_candidate_curve_csv": parent / f"cstcc_v8174_template_candidate_curve{suffix}.csv",
        "dew_phase_table_csv": parent / f"cstcc_v8174_dew_phase_table{suffix}.csv",
        "outside_air_dryness_gate_table_csv": parent / f"cstcc_v8174_outside_air_dryness_gate_table{suffix}.csv",
        "template_feasibility_table_csv": parent / f"cstcc_v8174_template_feasibility_table{suffix}.csv",
        "multi_scenario_readiness_json": parent / f"cstcc_v8174_multi_scenario_readiness{suffix}.json",
        "multi_scenario_readiness_md": parent / f"cstcc_v8174_multi_scenario_readiness{suffix}.md",
    }
    paths["json"].write_text(_compact_json(report) + "\n", encoding="utf-8")
    paths["md"].write_text(build_markdown(report), encoding="utf-8")
    paths["multi_scenario_readiness_json"].write_text(_compact_json(multi_scenario) + "\n", encoding="utf-8")
    paths["multi_scenario_readiness_md"].write_text(build_multi_scenario_markdown(multi_scenario), encoding="utf-8")
    _write_csv(paths["template_candidate_curve_csv"], tables.get("template_candidate_curve", []))
    _write_csv(paths["dew_phase_table_csv"], tables.get("dew_phase_table", []))
    _write_csv(paths["outside_air_dryness_gate_table_csv"], tables.get("outside_air_dryness_gate_table", []))
    _write_csv(paths["template_feasibility_table_csv"], tables.get("template_feasibility_table", []))
    return {key: str(value) for key, value in paths.items()}


def _load_inputs(args: argparse.Namespace) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], list[Mapping[str, Any]], int]:
    trace_paths = args.trace_csv or [str(DEFAULT_TRACE_CSV)]
    summary_paths = args.summary_json or [str(DEFAULT_SUMMARY_JSON)]
    jsonl_roots = args.jsonl_root or [str(DEFAULT_JSONL_ROOT)]
    trace_rows: list[Mapping[str, Any]] = []
    jsonl_rows: list[Mapping[str, Any]] = []
    summary = _load_json(summary_paths[0])
    for trace_path in trace_paths:
        trace_rows.extend(_load_csv(trace_path))
    for root in jsonl_roots:
        jsonl_rows.extend(_load_jsonl_rows(root))
    return trace_rows, summary, jsonl_rows, max(len(trace_paths), len(jsonl_roots), len(summary_paths))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v8174 C-STCC risk-specific template design audit.")
    parser.add_argument("--trace-csv", action="append", default=None)
    parser.add_argument("--summary-json", action="append", default=None)
    parser.add_argument("--jsonl-root", action="append", default=None)
    parser.add_argument("--v81731-json", type=str, default=str(DEFAULT_V81731_JSON))
    parser.add_argument("--v81732-json", type=str, default=str(DEFAULT_V81732_JSON))
    parser.add_argument("--v81734-json", type=str, default=str(DEFAULT_V81734_JSON))
    parser.add_argument("--v81735-json", type=str, default=str(DEFAULT_V81735_JSON))
    parser.add_argument("--v81732-policy-curve-csv", type=str, default=str(DEFAULT_V81732_POLICY_CURVE_CSV))
    parser.add_argument("--output-prefix", type=str, default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--no-golden-check", action="store_true")
    args = parser.parse_args()

    trace_rows, summary_json, jsonl_rows, input_trace_count = _load_inputs(args)
    report, tables, multi_scenario = build_report(
        trace_csv_rows=trace_rows,
        summary_json=summary_json,
        jsonl_rows=jsonl_rows,
        v81731_json=_load_json(args.v81731_json),
        v81732_json=_load_json(args.v81732_json),
        v81734_json=_load_json(args.v81734_json),
        v81735_json=_load_json(args.v81735_json),
        v81732_policy_curve_rows=_load_csv(args.v81732_policy_curve_csv),
        input_trace_count=input_trace_count,
        enforce_golden=not args.no_golden_check,
    )
    outputs = write_outputs(report=report, tables=tables, multi_scenario=multi_scenario, output_prefix=args.output_prefix)
    print(
        json.dumps(
            {"outputs": outputs, "next_action": report["next_action"], "readiness_reasons": report["readiness_reasons"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
