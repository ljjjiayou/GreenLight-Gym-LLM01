"""v8174.1 offline dry-risk template parameter iteration audit.

This stage evaluates evaluator-only dry-risk pseudo template parameters on
existing C-STCC shadow traces. It does not add templates to the runtime pool,
change final actions, call online LLMs, run predictive rollout, or claim
controller-quality improvement.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.cstcc.constraints import DEFAULT_MAX_DELTA, check_hard_constraints
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
    DEFAULT_V81734_JSON,
    DEFAULT_OUTPUT_PREFIX as DEFAULT_V81735_OUTPUT_PREFIX,
    ENERGY_WEIGHT,
    _bonus_vector,
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
    _template,
    _write_csv,
    bonus_for_family,
)
from gl_gym.experiments.cstcc_v8174_risk_specific_template_design import (
    DEFAULT_OUTPUT_PREFIX as DEFAULT_V8174_OUTPUT_PREFIX,
    DEFAULT_V81735_JSON,
    DEW_TEMPLATES,
    humidity_gate,
    _combo_id,
    _conservative_prior_bonus,
    _delta,
    _family,
    _has_dew_or_canopy_risk,
    _has_dry_or_high_vpd_risk,
    _has_dry_vent_risk,
    _risk_flags,
    _schema_humidity_issues,
    _score_existing_candidate,
    _step_toward,
    _v81734_golden_issues,
    _v81735_golden_issues,
    generate_pseudo_template_sequence as generate_v8174_pseudo_template_sequence,
)


DEFAULT_V81735_JSON = DEFAULT_V81735_OUTPUT_PREFIX.with_suffix(".json")
DEFAULT_V8174_JSON = DEFAULT_V8174_OUTPUT_PREFIX.with_suffix(".json")
DEFAULT_OUTPUT_PREFIX = Path(
    "gl_gym/result/diagnostics/cstcc_v8174_dry_risk_template_parameter_iteration_20260608_y2020_d240_s42_n720"
)

DRY_PSEUDO_TEMPLATES = (
    "conservative_dry_low_balance",
    "conservative_dry_moderate_relief",
    "conservative_dry_vent_guard_limited",
    "conservative_dry_external_humidification_needed",
)
PSEUDO_TEMPLATES = tuple(DEW_TEMPLATES) + DRY_PSEUDO_TEMPLATES
DRY_SCHEMA_FIELDS = ("temp_air", "rh_air", "vpd_air", "temp_out", "rh_out")
DRY_VENT_DRY_COMBO = "dry_vent+dry_or_high_vpd"
TARGET_RATE_LOW = 0.35
TARGET_RATE_HIGH = 0.75


def _dry_active(record: Mapping[str, Any]) -> bool:
    return bool(_has_dry_or_high_vpd_risk(record) or _has_dry_vent_risk(record))


def _severe_dew_or_canopy(record: Mapping[str, Any]) -> bool:
    flags = _risk_flags(record)
    return bool(
        _bool(flags.get("canopy_dew_margin_lt0"))
        or _num(record.get("canopy_dew_margin"), _num((record.get("merged") or {}).get("canopy_dew_margin"), 999.0))
        < 0.0
    )


def _dew_or_canopy_active(record: Mapping[str, Any]) -> bool:
    return bool(_has_dew_or_canopy_risk(record))


def annotate_dry_risk_age(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    dry_age = 0
    dry_vent_age = 0
    recent_dry: list[bool] = []
    for record in sorted((dict(row) for row in records), key=lambda item: int(_num(item.get("step")))):
        dry_now = _dry_active(record)
        dry_vent_now = _has_dry_vent_risk(record)
        dry_age = dry_age + 1 if dry_now else 0
        dry_vent_age = dry_vent_age + 1 if dry_vent_now else 0
        recent_dry.append(dry_now)
        if len(recent_dry) > 5:
            recent_dry.pop(0)
        record["dry_risk_age_steps"] = dry_age
        record["dry_vent_age_steps"] = dry_vent_age
        record["nearby_dry_risk_window_count"] = sum(1 for item in recent_dry if item)
        annotated.append(record)
    return annotated


def dry_risk_severity(record: Mapping[str, Any]) -> dict[str, Any]:
    flags = _risk_flags(record)
    vpd_air = _num(record.get("vpd_air"), _num((record.get("merged") or {}).get("vpd_air")))
    rh_air = _num(record.get("rh_air"), _num((record.get("merged") or {}).get("rh_air")))
    dry_age = int(_num(record.get("dry_risk_age_steps"), 1 if _dry_active(record) else 0))
    dry_vent_age = int(_num(record.get("dry_vent_age_steps"), 1 if _has_dry_vent_risk(record) else 0))
    dry_vent = _has_dry_vent_risk(record)
    dry_or_high_vpd = _has_dry_or_high_vpd_risk(record)
    if not (dry_vent or dry_or_high_vpd):
        return {
            "dry_severity": "none",
            "dry_severity_rank": 0,
            "dry_severity_reasons": [],
            "dry_risk_age_steps": dry_age,
            "dry_vent_age_steps": dry_vent_age,
            "vpd_air": vpd_air,
            "rh_air": rh_air,
            "strong_guard_allowed": False,
        }

    reasons: list[str] = []
    severe = False
    if vpd_air >= 1.50:
        severe = True
        reasons.append("vpd_ge_1_50")
    if dry_vent and rh_air <= 54.0:
        severe = True
        reasons.append("dry_vent_and_rh_le_54")
    if dry_vent and dry_vent_age >= 4:
        severe = True
        reasons.append("sustained_dry_vent_age_ge_4")
    if dry_age >= 8 and vpd_air >= 1.20:
        severe = True
        reasons.append("sustained_dry_risk_age_ge_8_with_vpd_ge_1_20")
    if severe:
        return {
            "dry_severity": "severe",
            "dry_severity_rank": 3,
            "dry_severity_reasons": reasons,
            "dry_risk_age_steps": dry_age,
            "dry_vent_age_steps": dry_vent_age,
            "vpd_air": vpd_air,
            "rh_air": rh_air,
            "strong_guard_allowed": True,
        }

    if dry_vent or vpd_air >= 1.15 or _bool(flags.get("high_vpd")) or rh_air < 65.0:
        if dry_vent:
            reasons.append("dry_vent_not_severe")
        if vpd_air >= 1.15:
            reasons.append("vpd_ge_1_15")
        if rh_air < 65.0:
            reasons.append("rh_lt_65")
        if _bool(flags.get("high_vpd")):
            reasons.append("high_vpd_flag")
        return {
            "dry_severity": "moderate",
            "dry_severity_rank": 2,
            "dry_severity_reasons": sorted(set(reasons)),
            "dry_risk_age_steps": dry_age,
            "dry_vent_age_steps": dry_vent_age,
            "vpd_air": vpd_air,
            "rh_air": rh_air,
            "strong_guard_allowed": False,
        }

    return {
        "dry_severity": "mild",
        "dry_severity_rank": 1,
        "dry_severity_reasons": ["dry_or_high_vpd_flag_below_moderate_threshold"],
        "dry_risk_age_steps": dry_age,
        "dry_vent_age_steps": dry_vent_age,
        "vpd_air": vpd_air,
        "rh_air": rh_air,
        "strong_guard_allowed": False,
    }


def conflict_resolver_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    dry = _dry_active(record)
    dew = _dew_or_canopy_active(record)
    severe_dew = _severe_dew_or_canopy(record)
    severity = dry_risk_severity(record)
    if dry and dew and severe_dew:
        conflict_type = "mixed_dew_canopy_precedence"
        resolved = "dew_canopy"
        reason = "severe_canopy_or_dew_margin_risk_has_priority_over_dry_template"
    elif dry and dew:
        conflict_type = "mixed_dry_dew"
        resolved = "dry" if severity["dry_severity_rank"] >= 2 else "dew_canopy"
        reason = "moderate_or_severe_dry_uses_dry_template_otherwise_dew_canopy_template"
    elif dry:
        conflict_type = "dry_only"
        resolved = "dry"
        reason = "dry_template_allowed_and_dew_template_blocked"
    elif dew:
        conflict_type = "dew_or_canopy_only"
        resolved = "dew_canopy"
        reason = "dew_canopy_template_allowed"
    else:
        conflict_type = "none"
        resolved = "none"
        reason = "no_risk"
    return {
        "conflict_type": conflict_type,
        "resolved_template_family": resolved,
        "conflict_priority_reason": reason,
        "dew_or_canopy_risk_active": dew,
        "severe_dew_or_canopy_risk": severe_dew,
        "dry_risk_active": dry,
        **severity,
    }


def _dry_template_applicable(template_name: str, record: Mapping[str, Any]) -> tuple[bool, str]:
    conflict = conflict_resolver_metadata(record)
    severity = conflict["dry_severity"]
    if template_name == "conservative_dry_external_humidification_needed":
        return (bool(conflict["dry_risk_active"]), "dry_risk_records_external_humidification_need")
    if not conflict["dry_risk_active"]:
        return False, "dry_template_requires_dry_or_high_vpd_risk"
    if conflict["resolved_template_family"] == "dew_canopy":
        return False, "blocked_by_severe_dew_canopy_conflict"
    if template_name == "conservative_dry_low_balance":
        return (severity == "mild", f"requires_mild_dry_observed_{severity}")
    if template_name == "conservative_dry_moderate_relief":
        return (severity == "moderate", f"requires_moderate_dry_observed_{severity}")
    if template_name == "conservative_dry_vent_guard_limited":
        return (severity == "severe", f"requires_severe_or_sustained_dry_vent_observed_{severity}")
    return False, "unknown_dry_template"


def _dry_target_action(template_name: str, last: Mapping[str, float]) -> dict[str, float]:
    target = dict(last)
    if template_name == "conservative_dry_low_balance":
        target["u_heating"] = clamp(max(0.0, last["u_heating"] - 0.04))
        target["u_ventilation"] = clamp(min(max(last["u_ventilation"], 0.18), 0.45))
        target["u_lighting"] = clamp(min(last["u_lighting"], 0.45))
        target["u_shading"] = clamp(max(last["u_shading"], 0.30))
    elif template_name == "conservative_dry_moderate_relief":
        target["u_heating"] = clamp(max(0.0, last["u_heating"] - 0.08))
        target["u_ventilation"] = clamp(min(max(last["u_ventilation"] - 0.10, 0.15), 0.35))
        target["u_lighting"] = clamp(min(last["u_lighting"], 0.35))
        target["u_shading"] = clamp(max(last["u_shading"], 0.40))
        target["u_co2"] = clamp(min(last["u_co2"], 0.20))
    elif template_name == "conservative_dry_vent_guard_limited":
        target["u_heating"] = clamp(max(0.0, last["u_heating"] - 0.10))
        target["u_ventilation"] = clamp(min(max(last["u_ventilation"] - 0.20, 0.10), 0.20))
        target["u_lighting"] = clamp(min(last["u_lighting"], 0.30))
        target["u_shading"] = clamp(max(last["u_shading"], 0.45))
        target["u_co2"] = clamp(min(last["u_co2"], 0.15))
    else:
        raise ValueError(f"unknown dry pseudo template: {template_name}")
    return normalize_action(target)


def generate_dry_pseudo_template_sequence(
    template_name: str,
    *,
    record: Mapping[str, Any],
    horizon: int = 12,
    max_delta_by_field: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    if template_name not in DRY_PSEUDO_TEMPLATES:
        raise ValueError(f"unknown v8174.1 dry pseudo template: {template_name}")
    last = normalize_action(record.get("current_runtime_final_action") or {})
    max_delta = dict(max_delta_by_field or DEFAULT_MAX_DELTA)
    conflict = conflict_resolver_metadata(record)
    applicable, reason = _dry_template_applicable(template_name, record)
    base_metadata: dict[str, Any] = {
        "pseudo_template_name": template_name,
        "template_applicable": applicable,
        "template_inapplicable_reason": None if applicable else reason,
        "template_source_stage": "v8174_1_dry_parameter_iteration",
        "conflict_resolver": conflict,
        "dry_severity_gate": dry_risk_severity(record),
        "humidity_gate": humidity_gate(record),
        "humidification_actuator_available": False,
        "generated_action_fields": list(ACTION_FIELDS),
        "external_humidification_needed": bool(conflict["dry_risk_active"]),
        "minimum_ventilation_floor": 0.10,
        "limited_guard_ventilation_target_band": {"min": 0.10, "max": 0.20},
    }
    if template_name == "conservative_dry_external_humidification_needed":
        sequence = [dict(last) for _ in range(max(1, horizon))]
        metadata = {
            **base_metadata,
            "candidate_score_eligible": False,
            "metadata_marker_only": True,
            "phase_plan": ["external_humidification_need_marker"] * max(1, horizon),
            "phase_distribution": {"external_humidification_need_marker": max(1, horizon)},
            "ventilation_policy": "no_actuator_change_marker_only",
            "first_action_delta_from_reference": _delta(sequence[0], last),
        }
        return sequence, metadata
    if not applicable:
        sequence = [dict(last) for _ in range(max(1, horizon))]
        metadata = {
            **base_metadata,
            "candidate_score_eligible": False,
            "metadata_marker_only": False,
            "phase_plan": ["inapplicable_hold"] * max(1, horizon),
            "phase_distribution": {"inapplicable_hold": max(1, horizon)},
            "ventilation_policy": "inapplicable",
            "first_action_delta_from_reference": _delta(sequence[0], last),
        }
        return sequence, metadata

    current = dict(last)
    target = _dry_target_action(template_name, last)
    sequence: list[dict[str, float]] = []
    phase = {
        "conservative_dry_low_balance": "mild_dry_balance_keep_moderate_vent",
        "conservative_dry_moderate_relief": "moderate_dry_reduce_heat_and_cap_vent",
        "conservative_dry_vent_guard_limited": "severe_dry_vent_limited_guard_keep_minimum_vent",
    }[template_name]
    for _index in range(max(1, horizon)):
        action = _step_toward(current, target, max_delta)
        sequence.append(action)
        current = action
    metadata = {
        **base_metadata,
        "candidate_score_eligible": True,
        "metadata_marker_only": False,
        "phase_plan": [phase] * max(1, horizon),
        "phase_distribution": {phase: max(1, horizon)},
        "ventilation_policy": (
            "keep_low_to_moderate_ventilation"
            if template_name == "conservative_dry_low_balance"
            else "cap_ventilation_but_preserve_minimum_fresh_air"
        ),
        "target_action": target,
        "final_sequence_action": sequence[-1],
        "first_action_delta_from_reference": _delta(sequence[0], last),
        "final_action_delta_from_reference": _delta(sequence[-1], last),
        "ventilation_floor_respected": min(action["u_ventilation"] for action in sequence) >= 0.10 - 1e-9,
        "ventilation_target_within_limited_guard_band": (
            0.10 - 1e-9 <= sequence[-1]["u_ventilation"] <= 0.20 + 1e-9
            if template_name == "conservative_dry_vent_guard_limited"
            else None
        ),
    }
    return sequence, metadata


def generate_pseudo_template_sequence(
    template_name: str,
    *,
    record: Mapping[str, Any],
    horizon: int = 12,
    max_delta_by_field: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    if template_name in DEW_TEMPLATES:
        sequence, metadata = generate_v8174_pseudo_template_sequence(
            template_name,
            record=record,
            horizon=horizon,
            max_delta_by_field=max_delta_by_field,
        )
        return sequence, {
            **metadata,
            "template_source_stage": "v8174_dew_canopy_preserved",
            "conflict_resolver": conflict_resolver_metadata(record),
            "candidate_score_eligible": bool(metadata.get("template_applicable", False)),
        }
    if template_name in DRY_PSEUDO_TEMPLATES:
        return generate_dry_pseudo_template_sequence(
            template_name,
            record=record,
            horizon=horizon,
            max_delta_by_field=max_delta_by_field,
        )
    raise ValueError(f"unknown v8174.1 pseudo template: {template_name}")


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
    hard_feasible = not violations and bool(metadata.get("template_applicable", True))
    return {
        "candidate_id": f"conservative_prior:{template_name}:v81741",
        "source_prior": "conservative_prior",
        "template_name": template_name,
        "feasible": hard_feasible,
        "score_eligible": hard_feasible and bool(metadata.get("candidate_score_eligible", True)),
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


def pseudo_candidates_for_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for template in PSEUDO_TEMPLATES:
        sequence, metadata = generate_pseudo_template_sequence(template, record=record)
        candidate = _candidate_from_sequence(record=record, template_name=template, sequence=sequence, metadata=metadata)
        component = _component_from_sequence(sequence, prior_bonus=_conservative_prior_bonus(record))
        candidates.append({**candidate, "score_component": component})
    return candidates


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
        if not candidate.get("feasible", False) or not candidate.get("score_eligible", True):
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
                "candidate_kind": "v8174_1_evaluator_only_pseudo_template",
                "template_name": candidate.get("template_name"),
                "template_metadata": candidate.get("template_metadata", {}),
            }
    return best_id, best_score, best_meta


def _is_pseudo(meta: Mapping[str, Any]) -> bool:
    return meta.get("candidate_kind") == "v8174_1_evaluator_only_pseudo_template"


def _template_curve_rows(records: Sequence[Mapping[str, Any]], *, policy_best_vectors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy, vector_row in policy_best_vectors.items():
        annotated = annotate_dry_risk_age(annotate_records_for_policy(records, policy))
        bonus_vector = _bonus_vector(vector_row)
        baseline_cache: dict[int, tuple[str, float, dict[str, Any]]] = {}
        for formula in ("baseline_current_energy", "formula_G_combo_specific_two_sided_semantics"):
            risk_count = 0
            pseudo_selected = 0
            nonrisk_pseudo_selected = 0
            canopy_combo_count = 0
            canopy_combo_selected = 0
            dry_vent_count = 0
            dry_vent_pseudo_selected = 0
            dry_vent_limited_guard_selected = 0
            mild_dry_limited_guard_selected = 0
            dry_formula_g_dependency_steps = 0
            selected_templates: dict[str, int] = {}
            selected_by_severity: dict[str, int] = {}
            severity_distribution: dict[str, int] = {}
            for record in annotated:
                is_risk = bool(_risk_flags(record).get("any_risk", False))
                if is_risk:
                    risk_count += 1
                severity = dry_risk_severity(record)["dry_severity"]
                if _dry_active(record):
                    severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
                winner_id, score, meta = winner_with_pseudo_templates(record, formula=formula, bonus_vector=bonus_vector)
                if formula == "baseline_current_energy":
                    baseline_cache[int(_num(record.get("step")))] = (winner_id, score, meta)
                is_pseudo = _is_pseudo(meta)
                template = str(meta.get("template_name") or _template(winner_id))
                if is_pseudo:
                    selected_templates[template] = selected_templates.get(template, 0) + 1
                    selected_by_severity[severity] = selected_by_severity.get(severity, 0) + 1
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
                        dry_vent_pseudo_selected += 1
                    if template == "conservative_dry_vent_guard_limited":
                        dry_vent_limited_guard_selected += 1
                        if severity == "mild":
                            mild_dry_limited_guard_selected += 1
                if formula == "formula_G_combo_specific_two_sided_semantics" and is_pseudo and template in DRY_PSEUDO_TEMPLATES:
                    baseline_meta = baseline_cache.get(int(_num(record.get("step"))), ("", 0.0, {}))[2]
                    if not _is_pseudo(baseline_meta) or str(baseline_meta.get("template_name")) not in DRY_PSEUDO_TEMPLATES:
                        dry_formula_g_dependency_steps += 1
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
                    "dry_vent_pseudo_selected_count": dry_vent_pseudo_selected,
                    "dry_vent_limited_guard_selected_count": dry_vent_limited_guard_selected,
                    "dry_vent_limited_guard_saturated": dry_vent_count > 0
                    and dry_vent_limited_guard_selected >= dry_vent_count,
                    "mild_dry_limited_guard_selected_count": mild_dry_limited_guard_selected,
                    "dry_formula_g_dependency_steps": dry_formula_g_dependency_steps,
                    "selected_pseudo_template_distribution": selected_templates,
                    "selected_pseudo_template_by_dry_severity": selected_by_severity,
                    "dry_severity_distribution": severity_distribution,
                }
            )
    return rows


def _best_curve_row(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        rate = _num(row.get("pseudo_template_risk_selected_rate"))
        rate_in_band = TARGET_RATE_LOW <= rate <= TARGET_RATE_HIGH
        canopy_selected = int(row.get("canopy_dew_lt0_dew_combo_pseudo_selected_count", 0) or 0)
        dry_total = int(row.get("dry_vent_row_count", 0) or 0)
        limited_guard = int(row.get("dry_vent_limited_guard_selected_count", 0) or 0)
        dry_formula_dependency = int(row.get("dry_formula_g_dependency_steps", 0) or 0)
        return (
            int(row.get("nonrisk_pseudo_template_selected_count", 0) or 0) == 0,
            canopy_selected > 0,
            dry_total == 0 or limited_guard < dry_total,
            int(row.get("mild_dry_limited_guard_selected_count", 0) or 0) == 0,
            dry_formula_dependency == 0,
            rate_in_band,
            canopy_selected,
            -limited_guard,
            -dry_formula_dependency,
            -abs(rate - 0.55),
        )

    return dict(max(rows, key=key)) if rows else {}


def _dry_severity_rows(records: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in annotate_dry_risk_age(annotate_records_for_policy(records, policy)):
        if not _dry_active(record):
            continue
        severity = dry_risk_severity(record)
        rows.append(
            {
                "step": record.get("step"),
                "hour_of_day": record.get("hour_of_day"),
                "resolved_risk_family": _family(record),
                "risk_combo_id": _combo_id(record),
                "dry_severity": severity["dry_severity"],
                "dry_severity_rank": severity["dry_severity_rank"],
                "dry_severity_reasons": severity["dry_severity_reasons"],
                "dry_risk_age_steps": severity["dry_risk_age_steps"],
                "dry_vent_age_steps": severity["dry_vent_age_steps"],
                "vpd_air": severity["vpd_air"],
                "rh_air": severity["rh_air"],
                "strong_guard_allowed": severity["strong_guard_allowed"],
            }
        )
    return rows


def _template_feasibility_rows(records: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    annotated = annotate_dry_risk_age(annotate_records_for_policy(records, policy))
    for template in PSEUDO_TEMPLATES:
        generated = applicable = feasible = score_eligible = 0
        max_delta_violations = 0
        ventilation_floor_violations = 0
        dry_risk_dew_misfires = 0
        external_marker_count = 0
        for record in annotated:
            for candidate in pseudo_candidates_for_record(record):
                if candidate["template_name"] != template:
                    continue
                generated += 1
                metadata = candidate.get("template_metadata") or {}
                if metadata.get("template_applicable", False):
                    applicable += 1
                if candidate.get("feasible", False):
                    feasible += 1
                if candidate.get("score_eligible", False):
                    score_eligible += 1
                if metadata.get("metadata_marker_only", False):
                    external_marker_count += 1
                max_delta_violations += int((candidate.get("violation_reason_distribution") or {}).get("max_delta", 0) or 0)
                if template in DEW_TEMPLATES and _dry_active(record) and not _dew_or_canopy_active(record):
                    if metadata.get("template_applicable", False):
                        dry_risk_dew_misfires += 1
                if (
                    template == "conservative_dry_vent_guard_limited"
                    and metadata.get("template_applicable", False)
                    and candidate.get("score_eligible", False)
                ):
                    raw_sequence = candidate.get("raw_sequence") or []
                    if raw_sequence and min(_num(action.get("u_ventilation")) for action in raw_sequence) < 0.10 - 1e-9:
                        ventilation_floor_violations += 1
        rows.append(
            {
                "policy": policy,
                "template_name": template,
                "generated_count": generated,
                "applicable_count": applicable,
                "feasible_count": feasible,
                "score_eligible_count": score_eligible,
                "external_marker_count": external_marker_count,
                "max_delta_violation_count": max_delta_violations,
                "ventilation_floor_violation_count": ventilation_floor_violations,
                "dry_risk_dew_template_misfire_count": dry_risk_dew_misfires,
            }
        )
    return rows


def _dry_vent_saturation_rows(records: Sequence[Mapping[str, Any]], *, policy_best_vectors: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _template_curve_rows(records, policy_best_vectors=policy_best_vectors):
        dry_total = int(row.get("dry_vent_row_count", 0) or 0)
        limited = int(row.get("dry_vent_limited_guard_selected_count", 0) or 0)
        rows.append(
            {
                "policy": row.get("policy"),
                "formula": row.get("formula"),
                "dry_vent_row_count": dry_total,
                "dry_vent_pseudo_selected_count": row.get("dry_vent_pseudo_selected_count"),
                "dry_vent_limited_guard_selected_count": limited,
                "dry_vent_limited_guard_selected_rate": limited / max(dry_total, 1),
                "dry_vent_limited_guard_saturated": dry_total > 0 and limited >= dry_total,
                "mild_dry_limited_guard_selected_count": row.get("mild_dry_limited_guard_selected_count"),
                "selected_pseudo_template_distribution": row.get("selected_pseudo_template_distribution"),
                "dry_severity_distribution": row.get("dry_severity_distribution"),
            }
        )
    return rows


def _conflict_resolver_rows(records: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in annotate_dry_risk_age(annotate_records_for_policy(records, policy)):
        if not _risk_flags(record).get("any_risk", False):
            continue
        conflict = conflict_resolver_metadata(record)
        rows.append(
            {
                "step": record.get("step"),
                "hour_of_day": record.get("hour_of_day"),
                "resolved_risk_family": _family(record),
                "risk_combo_id": _combo_id(record),
                "conflict_type": conflict["conflict_type"],
                "resolved_template_family": conflict["resolved_template_family"],
                "conflict_priority_reason": conflict["conflict_priority_reason"],
                "dew_or_canopy_risk_active": conflict["dew_or_canopy_risk_active"],
                "severe_dew_or_canopy_risk": conflict["severe_dew_or_canopy_risk"],
                "dry_risk_active": conflict["dry_risk_active"],
                "dry_severity": conflict["dry_severity"],
                "strong_guard_allowed": conflict["strong_guard_allowed"],
            }
        )
    return rows


def _dry_schema_issues(records: Sequence[Mapping[str, Any]]) -> list[str]:
    issues: list[str] = []
    for record in records:
        if not _risk_flags(record).get("any_risk", False):
            continue
        merged = record.get("merged") or {}
        missing = [field for field in DRY_SCHEMA_FIELDS if field not in merged and field not in record]
        if missing:
            issues.append(f"dry_risk_fields_missing:step={record.get('step')}:{','.join(missing)}")
    return sorted(set(issues))


def _v8174_golden_issues(v8174_json: Mapping[str, Any]) -> list[str]:
    if not v8174_json:
        return ["v8174_json_missing"]
    best = ((v8174_json.get("diagnostics") or {}).get("best_curve_row") or {})
    issues: list[str] = []
    if v8174_json.get("next_action") != "v8174_template_parameter_iteration_plan":
        issues.append("golden_mismatch:v8174_next_action")
    if int(best.get("canopy_dew_lt0_dew_combo_pseudo_selected_count", 0) or 0) != 6:
        issues.append("golden_mismatch:v8174_canopy_dew_dew_selected")
    if int(best.get("dry_vent_pseudo_selected_count", 0) or 0) != 21:
        issues.append("golden_mismatch:v8174_dry_vent_selected")
    if int(best.get("nonrisk_pseudo_template_selected_count", 0) or 0) != 0:
        issues.append("golden_mismatch:v8174_nonrisk_selected")
    return issues


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
        return "v8174_trace_schema_dry_risk_repair_plan", list(schema_issues)
    if any(int(row.get("max_delta_violation_count", 0) or 0) > 0 for row in feasibility_rows):
        return "v8174_template_parameter_iteration_plan", ["pseudo_template_max_delta_violation"]
    if any(int(row.get("ventilation_floor_violation_count", 0) or 0) > 0 for row in feasibility_rows):
        return "v8174_dry_severity_gate_iteration_plan", ["limited_guard_violates_minimum_ventilation_floor"]
    if any(int(row.get("dry_risk_dew_template_misfire_count", 0) or 0) > 0 for row in feasibility_rows):
        return "v8174_template_parameter_iteration_plan", ["dew_template_misfires_on_dry_only_risk"]
    canopy_total = int(best_curve.get("canopy_dew_lt0_dew_combo_row_count", 0) or 0)
    canopy_selected = int(best_curve.get("canopy_dew_lt0_dew_combo_pseudo_selected_count", 0) or 0)
    dry_total = int(best_curve.get("dry_vent_row_count", 0) or 0)
    limited_guard = int(best_curve.get("dry_vent_limited_guard_selected_count", 0) or 0)
    nonrisk = int(best_curve.get("nonrisk_pseudo_template_selected_count", 0) or 0)
    mild_limited = int(best_curve.get("mild_dry_limited_guard_selected_count", 0) or 0)
    dry_formula_dependency = int(best_curve.get("dry_formula_g_dependency_steps", 0) or 0)
    if nonrisk > 0:
        return "v8174_template_parameter_iteration_plan", ["nonrisk_pseudo_template_selected"]
    if canopy_total > 0 and canopy_selected <= 0:
        return "v8174_template_parameter_iteration_plan", ["canopy_dew_dew_template_still_not_competitive"]
    if dry_total > 0 and limited_guard >= dry_total:
        return "v8174_dry_severity_gate_iteration_plan", ["dry_vent_limited_guard_still_saturated"]
    if mild_limited > 0:
        return "v8174_dry_severity_gate_iteration_plan", ["mild_dry_triggers_limited_guard"]
    if dry_formula_dependency > 0:
        return "v8174_scorer_template_joint_calibration_plan", ["dry_template_competes_only_with_formula_g_or_extra_bonus"]
    if input_trace_count <= 1:
        return "v8174_multi_scenario_existing_trace_scan_or_acquisition_plan", ["single_h720_trace_only"]
    return "v8174_1_opt_in_risk_template_shadow_trace_plan", ["multi_scenario_existing_trace_template_parameter_passed"]


def build_report(
    *,
    trace_csv_rows: list[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    jsonl_rows: list[Mapping[str, Any]],
    v81731_json: Mapping[str, Any],
    v81732_json: Mapping[str, Any],
    v81734_json: Mapping[str, Any],
    v81735_json: Mapping[str, Any],
    v8174_json: Mapping[str, Any],
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
    schema_issues.extend(_dry_schema_issues(records))
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
                + _v8174_golden_issues(v8174_json)
            )
        )
    else:
        schema_issues = sorted(set(schema_issues + reconstruction_issues + delta_issues))

    curve_rows = _template_curve_rows(records, policy_best_vectors=policy_best_vectors)
    feasibility_rows = _template_feasibility_rows(records, policy)
    dry_severity_rows = _dry_severity_rows(records, policy)
    dry_vent_saturation_rows = _dry_vent_saturation_rows(records, policy_best_vectors=policy_best_vectors)
    conflict_rows = _conflict_resolver_rows(records, policy)
    best_curve = _best_curve_row(curve_rows)
    next_action, readiness_reasons = _route_next_action(
        boundary=boundary,
        schema_issues=schema_issues,
        best_curve=best_curve,
        feasibility_rows=feasibility_rows,
        input_trace_count=input_trace_count,
    )
    multi_scenario = {
        "schema_version": "cstcc_v8174_1_multi_scenario_readiness_v1",
        "input_trace_count": input_trace_count,
        "multi_scenario_evidence_available": input_trace_count > 1,
        "readiness": "blocked_single_h720_trace_only" if input_trace_count <= 1 else "available",
        "required_next_evidence": [
            "multiple years/days/seeds with dry/high-VPD, dry-vent, dew/canopy, and mixed-risk windows",
            "dry templates do not select in nonrisk windows and dew templates do not misfire in dry-only windows",
            "final-action invariant and forbidden-boundary counts remain clean",
        ],
    }
    v8174_best = ((v8174_json.get("diagnostics") or {}).get("best_curve_row") or {})
    report = {
        "schema_version": "cstcc_v8174_1_dry_risk_template_parameter_iteration_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_level": "existing-trace offline template parameter audit",
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
            "runtime_template_names_changed": False,
        },
        "boundary_runtime_safety": boundary,
        "schema_issues": schema_issues,
        "counterfactual_definition": {
            "pseudo_templates_evaluator_only": True,
            "dry_pseudo_templates_added_to_runtime_template_names": False,
            "runtime_candidate_pool_changed": False,
            "runtime_final_action_changed": False,
            "online_llm_called": False,
            "predictive_rollout_executed": False,
            "real_tomato_safety_projection": False,
            "humidification_actuator_available": False,
            "external_humidification_is_metadata_only": True,
            "action_fields": list(ACTION_FIELDS),
        },
        "scientific_design_assumptions": {
            "dry_risk_signal": "VPD and dry/high-VPD flags are used as dry-stress indicators; RH is retained as a secondary gate.",
            "actuator_limit": "current action space has no fogging, misting, or humidification actuator, so dry templates can only use heat, screen, vent, light, shade, co2 indirectly.",
            "references": [
                "https://www.canr.msu.edu/news/why_should_greenhouse_growers_pay_attention_to_vapor_pressure_deficit_and_n",
                "https://www.dpi.nsw.gov.au/agriculture/horticulture/greenhouse/structures-and-technology/evap-cooling",
                "https://ipm.cahnr.uconn.edu/reduce-greenhouse-humidity/",
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC4567830/",
            ],
            "claim_boundary": "references motivate evaluator-only template parameters; they are not project performance evidence",
        },
        "diagnostics": {
            "observed_gap": {
                "v8174_next_action": v8174_json.get("next_action"),
                "v8174_best_curve_row": v8174_best,
                "hypothesis_revision": {
                    "old_hypothesis": "one dry-vent conservative guard can represent all dry/high-VPD windows",
                    "failure_evidence_or_missing_evidence": "v8174 improved canopy_dew_lt0+dew_or_humidity to 6/6, but dry-vent remained saturated at 21/21",
                    "new_hypothesis": "dry-risk templates need severity-gated low/moderate/severe actions and explicit humidification capability metadata",
                    "next_experiment": "existing-trace dry-risk template parameter iteration audit",
                    "default_controller_changed": False,
                    "online_llm_needed": False,
                    "hard_safety_impact": "none; audit only",
                    "success_condition": "limited dry guard is not saturated, canopy improvement remains, and nonrisk selection remains zero",
                    "stop_condition": "dry schema missing, mild dry triggers limited guard, or template violates hard constraints",
                },
            },
            "selected_policy_for_tables": policy,
            "best_curve_row": best_curve,
            "dry_pseudo_template_names": DRY_PSEUDO_TEMPLATES,
            "all_pseudo_template_names": PSEUDO_TEMPLATES,
        },
        "acceptance": {
            "boundary_clean": _boundary_clean(boundary),
            "schema_clean": not schema_issues,
            "canopy_dew_lt0_dew_combo_preserved": int(
                best_curve.get("canopy_dew_lt0_dew_combo_pseudo_selected_count", 0) or 0
            )
            > 0,
            "dry_vent_limited_guard_not_saturated": int(
                best_curve.get("dry_vent_limited_guard_selected_count", 0) or 0
            )
            < int(best_curve.get("dry_vent_row_count", 0) or 0),
            "mild_dry_does_not_trigger_limited_guard": int(
                best_curve.get("mild_dry_limited_guard_selected_count", 0) or 0
            )
            == 0,
            "nonrisk_pseudo_template_selected_count_eq_0": int(
                best_curve.get("nonrisk_pseudo_template_selected_count", 0) or 0
            )
            == 0,
            "multi_scenario_evidence_available": input_trace_count > 1,
        },
        "next_action": next_action,
        "readiness_reasons": readiness_reasons,
        "notes": [
            "v8174.1 is offline template parameter design only.",
            "Dry external humidification is recorded as metadata because no humidifier actuator exists.",
            "A single H720 pass is not enough for generalization; multi-scenario evidence remains required.",
        ],
    }
    return report, {
        "dry_severity_table": dry_severity_rows,
        "dry_template_curve": curve_rows,
        "dry_vent_saturation_table": dry_vent_saturation_rows,
        "template_feasibility_table": feasibility_rows,
        "conflict_resolver_table": conflict_rows,
    }, multi_scenario


def build_markdown(report: Mapping[str, Any]) -> str:
    best = (report.get("diagnostics") or {}).get("best_curve_row") or {}
    boundary = report.get("boundary_runtime_safety", {})
    lines = [
        "# C-STCC v8174.1 Dry-Risk Template Parameter Iteration",
        "",
        "Existing-trace offline template parameter audit only. Runtime template pool and final actions are unchanged.",
        "",
        "## Boundary",
        "",
        f"- Audit success rate: {_num(boundary.get('audit_success_rate')):.3f}",
        f"- Final-action invariant rate: {_num(boundary.get('final_action_invariant_rate')):.3f}",
        f"- Online LLM / rollout / real projection steps: {boundary.get('online_llm_called_steps', 0)} / {boundary.get('predictive_rollout_executed_steps', 0)} / {boundary.get('real_tomato_safety_projection_steps', 0)}",
        "",
        "## Best Dry-Risk Counterfactual",
        "",
        f"- Policy / formula: {best.get('policy', 'none')} / {best.get('formula', 'none')}",
        f"- Pseudo-template risk selected rate: {_num(best.get('pseudo_template_risk_selected_rate')):.3f}",
        f"- Canopy-dew/dew pseudo selected: {best.get('canopy_dew_lt0_dew_combo_pseudo_selected_count', 0)} / {best.get('canopy_dew_lt0_dew_combo_row_count', 0)}",
        f"- Dry-vent pseudo selected: {best.get('dry_vent_pseudo_selected_count', 0)} / {best.get('dry_vent_row_count', 0)}",
        f"- Dry-vent limited guard selected: {best.get('dry_vent_limited_guard_selected_count', 0)} / {best.get('dry_vent_row_count', 0)}",
        f"- Mild dry limited-guard selections: {best.get('mild_dry_limited_guard_selected_count', 0)}",
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
        "# C-STCC v8174.1 Multi-Scenario Readiness",
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
        "dry_severity_table_csv": parent / f"cstcc_v8174_dry_severity_table{suffix}.csv",
        "dry_template_curve_csv": parent / f"cstcc_v8174_dry_template_curve{suffix}.csv",
        "dry_vent_saturation_table_csv": parent / f"cstcc_v8174_dry_vent_saturation_table{suffix}.csv",
        "template_feasibility_table_csv": parent / f"cstcc_v8174_dry_template_feasibility_table{suffix}.csv",
        "conflict_resolver_table_csv": parent / f"cstcc_v8174_dry_conflict_resolver_table{suffix}.csv",
        "multi_scenario_readiness_json": parent / f"cstcc_v8174_dry_multi_scenario_readiness{suffix}.json",
        "multi_scenario_readiness_md": parent / f"cstcc_v8174_dry_multi_scenario_readiness{suffix}.md",
    }
    paths["json"].write_text(_compact_json(report) + "\n", encoding="utf-8")
    paths["md"].write_text(build_markdown(report), encoding="utf-8")
    paths["multi_scenario_readiness_json"].write_text(_compact_json(multi_scenario) + "\n", encoding="utf-8")
    paths["multi_scenario_readiness_md"].write_text(build_multi_scenario_markdown(multi_scenario), encoding="utf-8")
    _write_csv(paths["dry_severity_table_csv"], tables.get("dry_severity_table", []))
    _write_csv(paths["dry_template_curve_csv"], tables.get("dry_template_curve", []))
    _write_csv(paths["dry_vent_saturation_table_csv"], tables.get("dry_vent_saturation_table", []))
    _write_csv(paths["template_feasibility_table_csv"], tables.get("template_feasibility_table", []))
    _write_csv(paths["conflict_resolver_table_csv"], tables.get("conflict_resolver_table", []))
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
    parser = argparse.ArgumentParser(description="Build v8174.1 C-STCC dry-risk template parameter audit.")
    parser.add_argument("--trace-csv", action="append", default=None)
    parser.add_argument("--summary-json", action="append", default=None)
    parser.add_argument("--jsonl-root", action="append", default=None)
    parser.add_argument("--v81731-json", type=str, default=str(DEFAULT_V81731_JSON))
    parser.add_argument("--v81732-json", type=str, default=str(DEFAULT_V81732_JSON))
    parser.add_argument("--v81734-json", type=str, default=str(DEFAULT_V81734_JSON))
    parser.add_argument("--v81735-json", type=str, default=str(DEFAULT_V81735_JSON))
    parser.add_argument("--v8174-json", type=str, default=str(DEFAULT_V8174_JSON))
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
        v8174_json=_load_json(args.v8174_json),
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
