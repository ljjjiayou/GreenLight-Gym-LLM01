"""v8173.5 offline combo-specific risk-state energy semantics audit.

This stage evaluates evaluator-only risk-state energy semantics on existing
H720 evidence. It does not change runtime scoring, template pools, final
actions, online LLM usage, predictive rollout, or Tomato Safety projection.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.cstcc.contracts import ACTION_FIELDS, clamp
from gl_gym.experiments.cstcc_v8173_conflict_resolver_policy_iteration import (
    DEFAULT_V81731_JSON,
    _spread,
    _target_gap,
    _target_label,
    _v81731_golden_issues,
    annotate_records_for_policy,
)
from gl_gym.experiments.cstcc_v8173_energy_proxy_formula_redesign import (
    DEFAULT_OUTPUT_PREFIX as DEFAULT_V81734_OUTPUT_PREFIX,
    DEFAULT_V81732_JSON,
    DEFAULT_V81732_POLICY_CURVE_CSV,
    DEFAULT_V81733_JSON,
    ENERGY_COMPONENTS,
    ENERGY_WEIGHT,
    POLICIES,
    TARGET_RATE_HIGH,
    TARGET_RATE_LOW,
    _bonus_vector,
    _candidate_delta,
    _dominant_after_formula,
    _policy_local_best_vectors,
    _v81733_golden_issues,
    enrich_records_with_candidate_deltas,
)
from gl_gym.experiments.cstcc_v8173_energy_proxy_reweight_design import (
    _v81732_golden_issues,
)
from gl_gym.experiments.cstcc_v8173_mixed_risk_conflict_resolver_design import (
    DEFAULT_V8173_JSON,
)
from gl_gym.experiments.cstcc_v8173_risk_type_specific_bonus_template_counterfactual import (
    DEFAULT_JSONL_ROOT,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TRACE_CSV,
    _bool,
    _boundary_clean,
    _boundary_summary,
    _build_step_records,
    _compact_json,
    _dominant_component,
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


DEFAULT_V81734_JSON = DEFAULT_V81734_OUTPUT_PREFIX.with_suffix(".json")
DEFAULT_OUTPUT_PREFIX = Path(
    "gl_gym/result/diagnostics/cstcc_v8173_combo_specific_energy_formula_iteration_20260607_y2020_d240_s42_n720"
)

FORMULAS = (
    "baseline_current_energy",
    "formula_E_risk_state_transition_credit",
    "formula_F_dry_vent_guard",
    "formula_G_combo_specific_two_sided_semantics",
)
REQUIRED_STATE_FIELDS = ("temp_air", "rh_air", "vpd_air", "canopy_dew_margin", "dew_margin_air")
CANOPY_DEW_DEW_COMBO = "canopy_dew_lt0+dew_or_humidity"
DRY_VENT_DRY_COMBO = "dry_vent+dry_or_high_vpd"


def _risk_flags(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("risk_flags") or {}


def _resolver(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("resolver") or {}


def _family(record: Mapping[str, Any]) -> str:
    return str(_resolver(record).get("resolved_risk_family_new") or "none")


def _combo_id(record: Mapping[str, Any]) -> str:
    return str(_resolver(record).get("risk_combo_id") or "none")


def _current_action(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("current_runtime_final_action") or {}


def _action_value(record: Mapping[str, Any], field: str) -> float:
    action = _current_action(record)
    merged = record.get("merged") or {}
    return _num(action.get(field), _num(merged.get(field)))


def _has_dew_or_canopy_risk(flags: Mapping[str, Any]) -> bool:
    return bool(
        _bool(flags.get("dew_risk"))
        or _bool(flags.get("humidity_or_dew_risk"))
        or _bool(flags.get("canopy_dew_margin_lt1"))
        or _bool(flags.get("canopy_dew_margin_lt0"))
    )


def _has_dry_or_high_vpd_risk(flags: Mapping[str, Any]) -> bool:
    return bool(_bool(flags.get("dry_risk")) or _bool(flags.get("high_vpd")) or _bool(flags.get("dry_or_high_vpd_risk")))


def _has_dry_vent_risk(flags: Mapping[str, Any]) -> bool:
    return bool(_bool(flags.get("dry_vent_risk")))


def risk_state_surrogate_deltas(record: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    """Estimate first-step risk-state direction from compact candidate deltas.

    Positive dew-margin deltas are better; negative RH deltas are better;
    positive VPD deltas are worse in dry/high-VPD windows. These are not
    simulator predictions. They are existing-trace counterfactual scoring
    proxies for whether an action direction is semantically risk-aligned.
    """

    flags = _risk_flags(record)
    delta = _candidate_delta(candidate)
    heat_up = max(0.0, _num(delta.get("u_heating")))
    heat_down = max(0.0, -_num(delta.get("u_heating")))
    vent_up = max(0.0, _num(delta.get("u_ventilation")))
    vent_down = max(0.0, -_num(delta.get("u_ventilation")))
    screen_close = max(0.0, _num(delta.get("u_screen")))
    screen_open = max(0.0, -_num(delta.get("u_screen")))
    shade_up = max(0.0, _num(delta.get("u_shading")))

    rh_air = _num(record.get("rh_air"))
    vpd_air = _num(record.get("vpd_air"))
    canopy_margin = _num(record.get("canopy_dew_margin"), 999.0)
    dew_margin = _num(record.get("dew_margin_air"), 999.0)
    current_vent = _action_value(record, "u_ventilation")

    humidity_factor = 1.25 if rh_air >= 90.0 or _bool(flags.get("humidity_or_dew_risk")) else 1.0
    canopy_factor = 1.35 if canopy_margin < 0.0 or _bool(flags.get("canopy_dew_margin_lt0")) else 1.15
    dry_factor = 1.25 if vpd_air >= 1.2 or _bool(flags.get("high_vpd")) else 1.0
    dew_margin_factor = 1.25 if dew_margin < 1.0 or _bool(flags.get("dew_risk")) else 1.0

    delta_canopy_dew_margin_hat = canopy_factor * (
        0.90 * heat_up + 0.55 * vent_up + 0.35 * screen_open - 0.55 * screen_close - 0.25 * vent_down
    )
    delta_dew_margin_air_hat = dew_margin_factor * (
        0.70 * heat_up + 0.65 * vent_up + 0.30 * screen_open - 0.40 * screen_close - 0.20 * vent_down
    )
    delta_rh_hat = humidity_factor * (-9.0 * vent_up - 6.0 * heat_up - 3.0 * screen_open + 4.0 * screen_close + 2.0 * vent_down)
    delta_vpd_hat = dry_factor * (4.5 * vent_up + 3.2 * heat_up - 3.5 * vent_down - 1.0 * screen_close - 0.50 * shade_up)

    dry_vent_relief_hat = 0.0
    dry_vent_worsening_hat = 0.0
    if _has_dry_vent_risk(flags):
        high_vent_pressure = max(0.0, current_vent - 0.70)
        dry_vent_relief_hat = min(1.0, 3.5 * vent_down + 1.2 * high_vent_pressure * min(1.0, 8.0 * vent_down))
        dry_vent_worsening_hat = min(
            1.0,
            3.5 * vent_up
            + 1.8 * heat_up
            + (0.65 if current_vent > 0.70 and vent_down < 0.02 else 0.0)
            + (0.25 if _has_dry_or_high_vpd_risk(flags) and heat_down <= 0.005 and vent_down < 0.02 else 0.0),
        )

    return {
        "delta_canopy_dew_margin_hat": delta_canopy_dew_margin_hat,
        "delta_dew_margin_air_hat": delta_dew_margin_air_hat,
        "delta_rh_hat": delta_rh_hat,
        "delta_vpd_hat": delta_vpd_hat,
        "dry_vent_relief_hat": dry_vent_relief_hat,
        "dry_vent_worsening_hat": dry_vent_worsening_hat,
        "current_ventilation": current_vent,
        "action_delta_u_heating": _num(delta.get("u_heating")),
        "action_delta_u_ventilation": _num(delta.get("u_ventilation")),
        "action_delta_u_screen": _num(delta.get("u_screen")),
    }


def _dew_recovery_credit(flags: Mapping[str, Any], surrogate: Mapping[str, float]) -> float:
    if not _has_dew_or_canopy_risk(flags):
        return 0.0
    margin_credit = 1.40 * max(0.0, _num(surrogate.get("delta_canopy_dew_margin_hat")))
    air_credit = 1.00 * max(0.0, _num(surrogate.get("delta_dew_margin_air_hat")))
    rh_credit = 0.020 * max(0.0, -_num(surrogate.get("delta_rh_hat")))
    vpd_penalty = 0.060 * max(0.0, _num(surrogate.get("delta_vpd_hat"))) if _has_dry_or_high_vpd_risk(flags) else 0.0
    no_relief_penalty = 0.05 if _bool(flags.get("canopy_dew_margin_lt0")) and margin_credit + air_credit <= 0.01 else 0.0
    return clamp(margin_credit + air_credit + rh_credit - vpd_penalty - no_relief_penalty)


def _dry_state_credit(flags: Mapping[str, Any], surrogate: Mapping[str, float]) -> float:
    if not _has_dry_or_high_vpd_risk(flags):
        return 0.0
    return clamp(0.080 * max(0.0, -_num(surrogate.get("delta_vpd_hat"))))


def _dry_state_penalty(flags: Mapping[str, Any], surrogate: Mapping[str, float]) -> float:
    if not _has_dry_or_high_vpd_risk(flags):
        return 0.0
    return clamp(0.090 * max(0.0, _num(surrogate.get("delta_vpd_hat"))))


def _dry_vent_guard_credit(flags: Mapping[str, Any], surrogate: Mapping[str, float]) -> float:
    if not _has_dry_vent_risk(flags):
        return 0.0
    return clamp(0.32 * max(0.0, _num(surrogate.get("dry_vent_relief_hat"))))


def _dry_vent_guard_penalty(flags: Mapping[str, Any], surrogate: Mapping[str, float]) -> float:
    if not _has_dry_vent_risk(flags):
        return 0.0
    return clamp(0.36 * max(0.0, _num(surrogate.get("dry_vent_worsening_hat"))))


def formula_projected_energy_score(
    formula: str,
    *,
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    component: Mapping[str, Any],
) -> float:
    original = _num(component.get("projected_energy_proxy"))
    flags = _risk_flags(record)
    surrogate = risk_state_surrogate_deltas(record, candidate)
    combo = _combo_id(record)

    if formula == "baseline_current_energy":
        return clamp(original)
    if formula == "formula_E_risk_state_transition_credit":
        return clamp(
            original
            + _dew_recovery_credit(flags, surrogate)
            + _dry_state_credit(flags, surrogate)
            - _dry_state_penalty(flags, surrogate)
        )
    if formula == "formula_F_dry_vent_guard":
        return clamp(original + _dry_vent_guard_credit(flags, surrogate) - _dry_vent_guard_penalty(flags, surrogate))
    if formula == "formula_G_combo_specific_two_sided_semantics":
        if combo == CANOPY_DEW_DEW_COMBO:
            return clamp(
                original
                + 1.20 * _dew_recovery_credit(flags, surrogate)
                - 0.50 * _dry_state_penalty(flags, surrogate)
            )
        if combo == DRY_VENT_DRY_COMBO:
            return clamp(
                original
                + 1.10 * _dry_vent_guard_credit(flags, surrogate)
                + 0.40 * _dry_state_credit(flags, surrogate)
                - 1.35 * _dry_vent_guard_penalty(flags, surrogate)
                - 0.65 * _dry_state_penalty(flags, surrogate)
            )
        return clamp(
            original
            + 0.55 * _dew_recovery_credit(flags, surrogate)
            + 0.55 * _dry_vent_guard_credit(flags, surrogate)
            + 0.35 * _dry_state_credit(flags, surrogate)
            - 0.55 * _dry_vent_guard_penalty(flags, surrogate)
            - 0.35 * _dry_state_penalty(flags, surrogate)
        )
    raise ValueError(f"unknown formula: {formula}")


def _is_conservative(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("feasible", False)
        and candidate.get("source_prior") == "conservative_prior"
        and candidate.get("template_name") == "conservative_stabilize"
    )


def _counterfactual_score(
    *,
    formula: str,
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    component: Mapping[str, Any],
    bonus_vector: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    family = _family(record)
    original_energy = _num(component.get("projected_energy_proxy"))
    formula_energy = formula_projected_energy_score(formula, record=record, candidate=candidate, component=component)
    formula_delta = formula_energy - original_energy
    assigned_bonus = 0.0
    if _is_conservative(candidate) and _risk_flags(record).get("any_risk", False) and family != "none":
        assigned_bonus = bonus_for_family(bonus_vector, family)
    surrogate = risk_state_surrogate_deltas(record, candidate)
    return (
        _num(component.get("final_score"), -1e18) + ENERGY_WEIGHT * formula_delta + assigned_bonus,
        {
            "original_projected_energy_proxy": original_energy,
            "formula_projected_energy_score": formula_energy,
            "formula_energy_delta": formula_delta,
            "assigned_bonus": assigned_bonus,
            **surrogate,
        },
    )


def _winner_with_formula(
    record: Mapping[str, Any],
    *,
    formula: str,
    bonus_vector: Mapping[str, Any],
) -> tuple[str, float, dict[str, float]]:
    best_id = ""
    best_score = -1e18
    best_meta: dict[str, float] = {}
    for candidate_id, component in (record.get("components") or {}).items():
        candidate = (record.get("candidates") or {}).get(candidate_id)
        if candidate is None or not candidate.get("feasible", False):
            continue
        score, meta = _counterfactual_score(
            formula=formula,
            record=record,
            candidate=candidate,
            component=component,
            bonus_vector=bonus_vector,
        )
        if score > best_score:
            best_id = candidate_id
            best_score = score
            best_meta = meta
    return best_id, best_score, best_meta


def _rate(item: Mapping[str, Any]) -> float:
    return int(item.get("conservative", 0) or 0) / max(int(item.get("count", 0) or 0), 1)


def _family_rows(policy: str, formula: str, vector_id: str, counts: Mapping[str, Mapping[str, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, item in sorted(counts.items()):
        count = int(item.get("count", 0) or 0)
        selected = int(item.get("conservative", 0) or 0)
        rate = selected / max(count, 1)
        rows.append(
            {
                "policy": policy,
                "formula": formula,
                "policy_local_bonus_vector_id": vector_id,
                "resolved_risk_family": family,
                "family_row_count": count,
                "family_conservative_selected_count": selected,
                "family_conservative_selected_rate": rate,
                "family_gap_to_target_rate": _target_gap(rate),
                "family_target_label": _target_label(rate),
            }
        )
    return rows


def _combo_rows(policy: str, formula: str, vector_id: str, counts: Mapping[str, Mapping[str, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for combo_id, item in sorted(counts.items()):
        count = int(item.get("count", 0) or 0)
        selected = int(item.get("conservative", 0) or 0)
        rate = selected / max(count, 1)
        rows.append(
            {
                "policy": policy,
                "formula": formula,
                "policy_local_bonus_vector_id": vector_id,
                "risk_combo_id": combo_id,
                "is_mixed_risk": "+" in combo_id,
                "combo_row_count": count,
                "combo_conservative_selected_count": selected,
                "combo_conservative_selected_rate": rate,
                "combo_gap_to_target_rate": _target_gap(rate),
                "combo_target_label": _target_label(rate),
            }
        )
    return rows


def _adjusted_component_gap_rows(records: Sequence[Mapping[str, Any]], formula: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for record in records:
        if not _risk_flags(record).get("any_risk", False):
            continue
        conservative_id = str(record.get("conservative_candidate_id") or "")
        zero_id = str(record.get("zero_bonus_winner_id") or "")
        components = record.get("components") or {}
        candidates = record.get("candidates") or {}
        conservative = candidates.get(conservative_id)
        zero = candidates.get(zero_id)
        if not conservative or not zero:
            continue
        conservative_components = components.get(conservative_id, {})
        zero_components = components.get(zero_id, {})
        adjusted = dict(record.get("zero_minus_conservative") or {})
        zero_energy = formula_projected_energy_score(formula, record=record, candidate=zero, component=zero_components)
        conservative_energy = formula_projected_energy_score(
            formula,
            record=record,
            candidate=conservative,
            component=conservative_components,
        )
        formula_gap = zero_energy - conservative_energy
        adjusted["raw_energy_proxy"] = formula_gap
        adjusted["projected_energy_proxy"] = formula_gap
        adjusted["final_score"] = _num(adjusted.get("final_score")) + ENERGY_WEIGHT * formula_gap
        grouped.setdefault(_family(record), []).append(adjusted)
    rows: list[dict[str, Any]] = []
    for family, items in sorted(grouped.items()):
        for component in ("base_final_score", "raw_smoothness", "raw_energy_proxy", "projected_energy_proxy", "final_score"):
            values = [_num(item.get(component)) for item in items]
            rows.append(
                {
                    "resolved_risk_family": family,
                    "component": component,
                    "row_count": len(items),
                    "mean": float(mean(values)) if values else 0.0,
                    "max": max(values) if values else 0.0,
                }
            )
    return rows


def _dominant_after_risk_state_formula(records: Sequence[Mapping[str, Any]], formula: str) -> str:
    return _dominant_component(
        [row for row in _adjusted_component_gap_rows(records, formula) if row.get("component") != "final_score"]
    )


def _curve_accepts(row: Mapping[str, Any]) -> bool:
    rate = _num(row.get("conservative_risk_selected_rate"))
    dry_total = int(row.get("dry_vent_row_count", 0) or 0)
    dry_selected = int(row.get("dry_vent_conservative_selected_count", 0) or 0)
    combo_total = int(row.get("canopy_dew_lt0_dew_combo_row_count", 0) or 0)
    combo_selected = int(row.get("canopy_dew_lt0_dew_combo_conservative_selected_count", 0) or 0)
    return bool(
        TARGET_RATE_LOW <= rate <= TARGET_RATE_HIGH
        and _num(row.get("risk_type_switch_spread"), 999.0) <= 0.30
        and _num(row.get("mixed_risk_combo_spread"), 999.0) <= 0.30
        and _num(row.get("single_risk_spread"), 999.0) <= 0.30
        and int(row.get("family_extreme_rate_count", 0) or 0) == 0
        and int(row.get("nonrisk_conservative_selected_count", 0) or 0) == 0
        and not _bool(row.get("rule_suppression_detected"))
        and not _bool(row.get("previous_rule_ppo_suppression_detected"))
        and row.get("dominant_component_after_formula") not in ENERGY_COMPONENTS
        and (dry_total == 0 or dry_selected < dry_total)
        and (combo_total == 0 or combo_selected > 0)
    )


def _better_curve(candidate: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    if not current:
        return True
    candidate_accepts = _curve_accepts(candidate)
    current_accepts = _curve_accepts(current)
    if candidate_accepts != current_accepts:
        return candidate_accepts
    candidate_rate = _num(candidate.get("conservative_risk_selected_rate"))
    current_rate = _num(current.get("conservative_risk_selected_rate"))
    candidate_rate_in_band = TARGET_RATE_LOW <= candidate_rate <= TARGET_RATE_HIGH
    current_rate_in_band = TARGET_RATE_LOW <= current_rate <= TARGET_RATE_HIGH
    if candidate_rate_in_band != current_rate_in_band:
        return candidate_rate_in_band
    return (
        _num(candidate.get("family_target_penalty")),
        _num(candidate.get("family_extreme_rate_count")),
        max(
            _num(candidate.get("risk_type_switch_spread")),
            _num(candidate.get("mixed_risk_combo_spread")),
            _num(candidate.get("single_risk_spread")),
        ),
        candidate.get("dominant_component_after_formula") in ENERGY_COMPONENTS,
        int(candidate.get("dry_vent_conservative_selected_count", 0) or 0)
        >= int(candidate.get("dry_vent_row_count", 0) or 0),
        int(candidate.get("canopy_dew_lt0_dew_combo_conservative_selected_count", 0) or 0) == 0,
        abs(candidate_rate - 0.55),
    ) < (
        _num(current.get("family_target_penalty")),
        _num(current.get("family_extreme_rate_count")),
        max(
            _num(current.get("risk_type_switch_spread")),
            _num(current.get("mixed_risk_combo_spread")),
            _num(current.get("single_risk_spread")),
        ),
        current.get("dominant_component_after_formula") in ENERGY_COMPONENTS,
        int(current.get("dry_vent_conservative_selected_count", 0) or 0) >= int(current.get("dry_vent_row_count", 0) or 0),
        int(current.get("canopy_dew_lt0_dew_combo_conservative_selected_count", 0) or 0) == 0,
        abs(current_rate - 0.55),
    )


def evaluate_formulas(
    records: Sequence[Mapping[str, Any]],
    *,
    policy_best_vectors: Mapping[str, Mapping[str, Any]],
    formulas: Sequence[str] = FORMULAS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    curve_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] = {}
    for policy, best_vector_row in policy_best_vectors.items():
        annotated = annotate_records_for_policy(records, policy)
        bonus_vector = _bonus_vector(best_vector_row)
        risk_records = [record for record in annotated if _risk_flags(record).get("any_risk", False)]
        selected_mismatch = sum(1 for record in annotated if record.get("runtime_selected_id") != record.get("zero_bonus_winner_id"))
        for formula in formulas:
            conservative_selected = 0
            conservative_risk_selected = 0
            nonrisk_conservative_selected = 0
            switch_runtime = 0
            switch_zero = 0
            switch_to_conservative = 0
            switch_from_previous = 0
            switch_from_rule = 0
            switch_from_ppo = 0
            family_counts: dict[str, dict[str, int]] = {}
            combo_counts: dict[str, dict[str, int]] = {}
            single_counts: dict[str, dict[str, int]] = {}
            mixed_counts: dict[str, dict[str, int]] = {}
            for record in annotated:
                family = _family(record)
                combo_id = _combo_id(record)
                is_mixed = "+" in combo_id
                is_risk = bool(_risk_flags(record).get("any_risk", False))
                winner_id, _winner_score, _winner_meta = _winner_with_formula(
                    record,
                    formula=formula,
                    bonus_vector=bonus_vector,
                )
                winner_source = _source(winner_id)
                winner_template = _template(winner_id)
                zero_id = str(record.get("zero_bonus_winner_id") or "")
                runtime_id = str(record.get("runtime_selected_id") or "")
                zero_source = _source(zero_id)
                is_conservative = winner_source == "conservative_prior" and winner_template == "conservative_stabilize"
                if is_conservative:
                    conservative_selected += 1
                    if is_risk:
                        conservative_risk_selected += 1
                    else:
                        nonrisk_conservative_selected += 1
                if winner_id != runtime_id:
                    switch_runtime += 1
                if winner_id != zero_id:
                    switch_zero += 1
                    if is_conservative and zero_source != "conservative_prior":
                        switch_to_conservative += 1
                        if zero_source == "previous_plan_prior":
                            switch_from_previous += 1
                        elif zero_source == "rule_prior":
                            switch_from_rule += 1
                        elif zero_source == "ppo_prior":
                            switch_from_ppo += 1
                if is_risk:
                    for bucket in (
                        family_counts.setdefault(family, {"count": 0, "conservative": 0}),
                        combo_counts.setdefault(combo_id, {"count": 0, "conservative": 0}),
                        (mixed_counts if is_mixed else single_counts).setdefault(family, {"count": 0, "conservative": 0}),
                    ):
                        bucket["count"] += 1
                        if is_conservative:
                            bucket["conservative"] += 1
            family_rates = [_rate(item) for family, item in family_counts.items() if family != "none" and item["count"] > 0]
            combo_rates = [_rate(item) for combo_id, item in combo_counts.items() if "+" in combo_id and item["count"] > 0]
            single_rates = [_rate(item) for family, item in single_counts.items() if family != "none" and item["count"] > 0]
            mixed_rates = [_rate(item) for family, item in mixed_counts.items() if family != "none" and item["count"] > 0]
            dry_vent_counts = family_counts.get("dry_vent") or {"count": 0, "conservative": 0}
            canopy_dew_combo_counts = combo_counts.get(CANOPY_DEW_DEW_COMBO) or {"count": 0, "conservative": 0}
            dominant = _dominant_after_risk_state_formula(annotated, formula)
            row = {
                "policy": policy,
                "formula": formula,
                "policy_local_bonus_vector_id": str(best_vector_row.get("vector_id") or ""),
                **_bonus_vector(best_vector_row),
                "row_count": len(annotated),
                "risk_row_count": len(risk_records),
                "conservative_selected_count": conservative_selected,
                "conservative_risk_selected_count": conservative_risk_selected,
                "conservative_risk_selected_rate": conservative_risk_selected / max(len(risk_records), 1),
                "nonrisk_conservative_selected_count": nonrisk_conservative_selected,
                "risk_type_switch_spread": _spread(family_rates),
                "mixed_risk_combo_spread": _spread(combo_rates),
                "single_risk_spread": _spread(single_rates),
                "mixed_risk_spread": _spread(mixed_rates),
                "family_target_penalty": sum(_target_gap(rate) for rate in family_rates),
                "family_extreme_rate_count": sum(1 for rate in family_rates if rate <= 0.0 or rate >= 1.0),
                "dry_vent_row_count": int(dry_vent_counts.get("count", 0) or 0),
                "dry_vent_conservative_selected_count": int(dry_vent_counts.get("conservative", 0) or 0),
                "canopy_dew_lt0_dew_combo_row_count": int(canopy_dew_combo_counts.get("count", 0) or 0),
                "canopy_dew_lt0_dew_combo_conservative_selected_count": int(
                    canopy_dew_combo_counts.get("conservative", 0) or 0
                ),
                "switch_relative_to_runtime_count": switch_runtime,
                "switch_relative_to_zero_bonus_count": switch_zero,
                "switch_to_conservative_count": switch_to_conservative,
                "switch_from_previous_to_conservative_count": switch_from_previous,
                "switch_from_rule_to_conservative_count": switch_from_rule,
                "switch_from_ppo_to_conservative_count": switch_from_ppo,
                "selected_id_vs_zero_bonus_winner_mismatch_count": selected_mismatch,
                "rule_suppression_detected": switch_from_rule > 0,
                "previous_rule_ppo_suppression_detected": conservative_risk_selected / max(len(risk_records), 1) > TARGET_RATE_HIGH,
                "dominant_component_after_formula": dominant,
            }
            curve_rows.append(row)
            family_rows.extend(_family_rows(policy, formula, str(best_vector_row.get("vector_id") or ""), family_counts))
            combo_rows.extend(_combo_rows(policy, formula, str(best_vector_row.get("vector_id") or ""), combo_counts))
            if _better_curve(row, best_row):
                best_row = dict(row)
    return curve_rows, family_rows, combo_rows, {
        "best_formula_row": best_row,
        "policy_count": len(policy_best_vectors),
        "formula_count": len(formulas),
    }


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _conservative_surrogates(records: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in annotate_records_for_policy(records, policy):
        if not _risk_flags(record).get("any_risk", False):
            continue
        conservative_id = str(record.get("conservative_candidate_id") or "")
        candidate = (record.get("candidates") or {}).get(conservative_id)
        if not candidate:
            continue
        rows.append(
            {
                "record": record,
                "candidate_id": conservative_id,
                **risk_state_surrogate_deltas(record, candidate),
            }
        )
    return rows


def combo_failure_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    best_formula: Mapping[str, Any],
    policy_best_vectors: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    policy = str(best_formula.get("policy") or "")
    formula = str(best_formula.get("formula") or "")
    bonus_vector = _bonus_vector(policy_best_vectors.get(policy, {}))
    grouped: dict[str, dict[str, Any]] = {}
    for item in _conservative_surrogates(records, policy):
        record = item["record"]
        combo = _combo_id(record)
        bucket = grouped.setdefault(
            combo,
            {
                "risk_combo_id": combo,
                "row_count": 0,
                "best_formula_conservative_selected_count": 0,
                "baseline_conservative_selected_count": 0,
                "delta_canopy_dew_margin_hat_values": [],
                "delta_dew_margin_air_hat_values": [],
                "delta_rh_hat_values": [],
                "delta_vpd_hat_values": [],
            },
        )
        bucket["row_count"] += 1
        baseline_winner, _baseline_score, _baseline_meta = _winner_with_formula(
            record,
            formula="baseline_current_energy",
            bonus_vector=bonus_vector,
        )
        winner, _score, _meta = _winner_with_formula(record, formula=formula, bonus_vector=bonus_vector)
        if _source(baseline_winner) == "conservative_prior":
            bucket["baseline_conservative_selected_count"] += 1
        if _source(winner) == "conservative_prior":
            bucket["best_formula_conservative_selected_count"] += 1
        for key in (
            "delta_canopy_dew_margin_hat",
            "delta_dew_margin_air_hat",
            "delta_rh_hat",
            "delta_vpd_hat",
        ):
            bucket[f"{key}_values"].append(_num(item.get(key)))
    rows: list[dict[str, Any]] = []
    for combo, bucket in sorted(grouped.items()):
        row_count = int(bucket["row_count"])
        rows.append(
            {
                "policy": policy,
                "formula": formula,
                "risk_combo_id": combo,
                "row_count": row_count,
                "baseline_conservative_selected_count": bucket["baseline_conservative_selected_count"],
                "best_formula_conservative_selected_count": bucket["best_formula_conservative_selected_count"],
                "baseline_conservative_selected_rate": bucket["baseline_conservative_selected_count"] / max(row_count, 1),
                "best_formula_conservative_selected_rate": bucket["best_formula_conservative_selected_count"] / max(row_count, 1),
                "mean_delta_canopy_dew_margin_hat": _mean_or_zero(bucket["delta_canopy_dew_margin_hat_values"]),
                "mean_delta_dew_margin_air_hat": _mean_or_zero(bucket["delta_dew_margin_air_hat_values"]),
                "mean_delta_rh_hat": _mean_or_zero(bucket["delta_rh_hat_values"]),
                "mean_delta_vpd_hat": _mean_or_zero(bucket["delta_vpd_hat_values"]),
            }
        )
    return rows


def dry_vent_guard_rows(records: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    items = [item for item in _conservative_surrogates(records, policy) if _family(item["record"]) == "dry_vent"]
    return [
        {
            "policy": policy,
            "risk_family": "dry_vent",
            "row_count": len(items),
            "mean_current_ventilation": _mean_or_zero([_num(item.get("current_ventilation")) for item in items]),
            "mean_action_delta_u_ventilation": _mean_or_zero([_num(item.get("action_delta_u_ventilation")) for item in items]),
            "mean_action_delta_u_heating": _mean_or_zero([_num(item.get("action_delta_u_heating")) for item in items]),
            "mean_delta_vpd_hat": _mean_or_zero([_num(item.get("delta_vpd_hat")) for item in items]),
            "mean_dry_vent_relief_hat": _mean_or_zero([_num(item.get("dry_vent_relief_hat")) for item in items]),
            "mean_dry_vent_worsening_hat": _mean_or_zero([_num(item.get("dry_vent_worsening_hat")) for item in items]),
        }
    ]


def canopy_dew_recovery_rows(records: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    items = [item for item in _conservative_surrogates(records, policy) if _combo_id(item["record"]) == CANOPY_DEW_DEW_COMBO]
    return [
        {
            "policy": policy,
            "risk_combo_id": CANOPY_DEW_DEW_COMBO,
            "row_count": len(items),
            "mean_action_delta_u_heating": _mean_or_zero([_num(item.get("action_delta_u_heating")) for item in items]),
            "mean_action_delta_u_ventilation": _mean_or_zero([_num(item.get("action_delta_u_ventilation")) for item in items]),
            "mean_action_delta_u_screen": _mean_or_zero([_num(item.get("action_delta_u_screen")) for item in items]),
            "mean_delta_canopy_dew_margin_hat": _mean_or_zero([_num(item.get("delta_canopy_dew_margin_hat")) for item in items]),
            "mean_delta_dew_margin_air_hat": _mean_or_zero([_num(item.get("delta_dew_margin_air_hat")) for item in items]),
            "mean_delta_rh_hat": _mean_or_zero([_num(item.get("delta_rh_hat")) for item in items]),
            "mean_delta_vpd_hat": _mean_or_zero([_num(item.get("delta_vpd_hat")) for item in items]),
        }
    ]


def switch_rows_for_formula(
    records: Sequence[Mapping[str, Any]],
    *,
    best_formula: Mapping[str, Any],
    policy_best_vectors: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    policy = str(best_formula.get("policy") or "")
    formula = str(best_formula.get("formula") or "")
    bonus_vector = _bonus_vector(policy_best_vectors.get(policy, {}))
    rows: list[dict[str, Any]] = []
    for record in annotate_records_for_policy(records, policy):
        winner_id, winner_score, meta = _winner_with_formula(record, formula=formula, bonus_vector=bonus_vector)
        zero_id = str(record.get("zero_bonus_winner_id") or "")
        runtime_id = str(record.get("runtime_selected_id") or "")
        is_risk = bool(_risk_flags(record).get("any_risk", False))
        if not is_risk and winner_id == zero_id and winner_id == runtime_id:
            continue
        rows.append(
            {
                "step": record.get("step"),
                "hour_of_day": record.get("hour_of_day"),
                "active_regime": record.get("active_regime"),
                "policy": policy,
                "formula": formula,
                "resolved_risk_family": _family(record),
                "risk_combo_id": _combo_id(record),
                "is_risk": is_risk,
                "runtime_selected_id": runtime_id,
                "zero_bonus_winner_id": zero_id,
                "counterfactual_winner_id": winner_id,
                "counterfactual_winner_score": winner_score,
                "switch_relative_to_runtime": winner_id != runtime_id,
                "switch_relative_to_zero_bonus": winner_id != zero_id,
                "switch_to_conservative": _source(winner_id) == "conservative_prior"
                and _source(zero_id) != "conservative_prior",
                **meta,
            }
        )
    return rows


def _v81734_golden_issues(v81734_json: Mapping[str, Any]) -> list[str]:
    if not v81734_json:
        return ["v81734_json_missing"]
    best = ((v81734_json.get("diagnostics") or {}).get("best_formula_row") or {})
    issues: list[str] = []
    if best.get("formula") != "baseline_current_energy":
        issues.append("golden_mismatch:v81734_best_formula")
    if abs(_num(best.get("conservative_risk_selected_rate")) - 0.5396825396825397) > 1e-9:
        issues.append("golden_mismatch:v81734_conservative_risk_selected_rate")
    if abs(_num(best.get("risk_type_switch_spread")) - 0.641025641025641) > 1e-9:
        issues.append("golden_mismatch:v81734_risk_type_switch_spread")
    if abs(_num(best.get("mixed_risk_combo_spread")) - 1.0) > 1e-9:
        issues.append("golden_mismatch:v81734_mixed_risk_combo_spread")
    if abs(_num(best.get("single_risk_spread")) - 0.375) > 1e-9:
        issues.append("golden_mismatch:v81734_single_risk_spread")
    if int(best.get("dry_vent_conservative_selected_count", 0) or 0) != 21:
        issues.append("golden_mismatch:v81734_dry_vent_selected")
    if int(best.get("canopy_dew_lt0_dew_combo_conservative_selected_count", 0) or 0) != 0:
        issues.append("golden_mismatch:v81734_canopy_dew_dew_selected")
    return issues


def _state_schema_issues(records: Sequence[Mapping[str, Any]]) -> list[str]:
    issues: list[str] = []
    for record in records:
        if not _risk_flags(record).get("any_risk", False):
            continue
        merged = record.get("merged") or {}
        missing = [field for field in REQUIRED_STATE_FIELDS if field not in merged]
        if missing:
            issues.append(f"risk_state_fields_missing:step={record.get('step')}:{','.join(missing)}")
    return sorted(set(issues))


def _route_next_action(
    *,
    boundary: Mapping[str, Any],
    schema_issues: Sequence[str],
    best_formula: Mapping[str, Any],
    curve_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[str, list[str]]:
    if not _boundary_clean(boundary):
        return "stop_and_repair_cstcc_shadow_boundary", ["boundary_or_candidate_health_failure"]
    if schema_issues:
        return "v8173_schema_or_golden_trace_repair_plan", list(schema_issues)
    if _curve_accepts(best_formula):
        return "v8173_6_opt_in_risk_state_formula_shadow_trace_plan", [
            "combo_specific_risk_state_formula_passes_all_gates"
        ]
    nonbaseline_rows = [row for row in (curve_rows or []) if row.get("formula") != "baseline_current_energy"]
    dry_rows = [row for row in nonbaseline_rows if int(row.get("dry_vent_row_count", 0) or 0) > 0]
    canopy_rows = [row for row in nonbaseline_rows if int(row.get("canopy_dew_lt0_dew_combo_row_count", 0) or 0) > 0]
    dry_guard_resolved_by_any_formula = any(
        int(row.get("dry_vent_conservative_selected_count", 0) or 0) < int(row.get("dry_vent_row_count", 0) or 0)
        for row in dry_rows
    )
    canopy_recovered_by_any_formula = any(
        int(row.get("canopy_dew_lt0_dew_combo_conservative_selected_count", 0) or 0) > 0 for row in canopy_rows
    )
    if dry_rows and not dry_guard_resolved_by_any_formula:
        return "v8173_dry_vent_guard_policy_or_template_design_plan", ["dry_vent_guard_still_saturated"]
    if canopy_rows and not canopy_recovered_by_any_formula:
        return "v8174_risk_specific_template_design_plan", ["canopy_dew_dew_combo_still_under_target"]
    dry_total = int(best_formula.get("dry_vent_row_count", 0) or 0)
    dry_selected = int(best_formula.get("dry_vent_conservative_selected_count", 0) or 0)
    combo_total = int(best_formula.get("canopy_dew_lt0_dew_combo_row_count", 0) or 0)
    combo_selected = int(best_formula.get("canopy_dew_lt0_dew_combo_conservative_selected_count", 0) or 0)
    if dry_total > 0 and dry_selected >= dry_total:
        return "v8173_dry_vent_guard_policy_or_template_design_plan", ["dry_vent_guard_still_saturated"]
    if combo_total > 0 and combo_selected <= 0:
        return "v8174_risk_specific_template_design_plan", ["canopy_dew_dew_combo_still_under_target"]
    if (
        best_formula.get("formula") != "baseline_current_energy"
        and best_formula.get("dominant_component_after_formula") in ENERGY_COMPONENTS
    ):
        return "v8173_score_weight_interface_design_plan", ["risk_state_formula_directional_but_energy_hook_too_weak"]
    return "v8173_combo_specific_energy_formula_iteration_plan", ["risk_state_formula_counterfactual_not_balanced"]


def build_report(
    *,
    trace_csv_rows: list[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    jsonl_rows: list[Mapping[str, Any]],
    v8173_json: Mapping[str, Any],
    v81731_json: Mapping[str, Any],
    v81732_json: Mapping[str, Any],
    v81733_json: Mapping[str, Any],
    v81734_json: Mapping[str, Any],
    v81732_policy_curve_rows: list[Mapping[str, Any]],
    formulas: Sequence[str] = FORMULAS,
    enforce_golden: bool = True,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    boundary = _boundary_summary(summary_json, jsonl_rows)
    schema_issues = _missing_schema_fields(jsonl_rows)
    records, reconstruction_issues = _build_step_records(trace_csv_rows=trace_csv_rows, jsonl_rows=jsonl_rows)
    records, delta_issues = enrich_records_with_candidate_deltas(records, jsonl_rows)
    policy_best_vectors = _policy_local_best_vectors(v81732_policy_curve_rows)
    missing_policies = [policy for policy in POLICIES if policy not in policy_best_vectors]
    schema_issues.extend(f"policy_local_best_vector_missing:{policy}" for policy in missing_policies)
    schema_issues.extend(_state_schema_issues(records))
    if enforce_golden:
        schema_issues = sorted(
            set(
                schema_issues
                + reconstruction_issues
                + delta_issues
                + _v81731_golden_issues(v81731_json)
                + _v81732_golden_issues(v81732_json)
                + _v81733_golden_issues(v81733_json)
                + _v81734_golden_issues(v81734_json)
            )
        )
    else:
        schema_issues = sorted(set(schema_issues + reconstruction_issues + delta_issues))

    curve_rows, family_rows, combo_rows, formula_diagnostics = evaluate_formulas(
        records,
        policy_best_vectors=policy_best_vectors,
        formulas=formulas,
    )
    best_formula = formula_diagnostics.get("best_formula_row") or {}
    switch_rows = switch_rows_for_formula(
        records,
        best_formula=best_formula,
        policy_best_vectors=policy_best_vectors,
    )
    combo_failure = combo_failure_rows(records, best_formula=best_formula, policy_best_vectors=policy_best_vectors)
    dry_guard = dry_vent_guard_rows(records, str(best_formula.get("policy") or ""))
    canopy_recovery = canopy_dew_recovery_rows(records, str(best_formula.get("policy") or ""))
    next_action, readiness_reasons = _route_next_action(
        boundary=boundary,
        schema_issues=schema_issues,
        best_formula=best_formula,
        curve_rows=curve_rows,
    )
    v81734_best = ((v81734_json.get("diagnostics") or {}).get("best_formula_row") or {})
    diagnostics = {
        "observed_gap": {
            "v81734_best_formula": v81734_best,
            "hypothesis_revision": {
                "old_hypothesis": "action-direction energy formulas can resolve mixed-risk conservative selection imbalance",
                "failure_evidence_or_missing_evidence": (
                    "v8173.4 best remained baseline_current_energy; dry_vent stayed saturated; "
                    "canopy_dew_lt0+dew_or_humidity stayed zero; raw_energy_proxy stayed dominant"
                ),
                "new_hypothesis": "combo-specific risk-state transition semantics are required before opt-in runtime tracing",
                "next_experiment": "existing-trace counterfactual risk-state formula audit",
                "default_controller_changed": False,
                "online_llm_needed": False,
                "hard_safety_impact": "none; audit only",
                "success_condition": "local combo extremes and family/mixed/single spread gates pass",
                "stop_condition": "schema/golden/boundary failure or persistent local extremes",
            },
        },
        "policy_local_best_vectors": policy_best_vectors,
        "best_formula_row": best_formula,
        "policy_count": len(policy_best_vectors),
        "formula_count": len(formulas),
        "surrogate_state_fields": REQUIRED_STATE_FIELDS,
        "curve_resolution_diagnostics": {
            "dry_vent_resolved_by_any_nonbaseline_formula": any(
                int(row.get("dry_vent_conservative_selected_count", 0) or 0)
                < int(row.get("dry_vent_row_count", 0) or 0)
                for row in curve_rows
                if row.get("formula") != "baseline_current_energy" and int(row.get("dry_vent_row_count", 0) or 0) > 0
            ),
            "canopy_dew_dew_recovered_by_any_nonbaseline_formula": any(
                int(row.get("canopy_dew_lt0_dew_combo_conservative_selected_count", 0) or 0) > 0
                for row in curve_rows
                if row.get("formula") != "baseline_current_energy"
                and int(row.get("canopy_dew_lt0_dew_combo_row_count", 0) or 0) > 0
            ),
        },
        "v8173_reference_next_action": v8173_json.get("next_action"),
        "v81731_reference_next_action": v81731_json.get("next_action"),
        "v81732_reference_next_action": v81732_json.get("next_action"),
        "v81733_reference_next_action": v81733_json.get("next_action"),
        "v81734_reference_next_action": v81734_json.get("next_action"),
    }
    report = {
        "schema_version": "cstcc_v8173_combo_specific_energy_formula_iteration_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_level": "existing-trace counterfactual scorer audit",
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
            "risk_state_semantics_are_evaluator_only": True,
            "runtime_level0_sequence_score_changed": False,
            "energy_weight_changed": False,
            "energy_weight_used": ENERGY_WEIGHT,
            "bonus_vector_changed": False,
            "prior_confidence_changed": False,
            "runtime_final_action_changed": False,
            "surrogate_deltas_are_not_simulator_predictions": True,
        },
        "diagnostics": diagnostics,
        "acceptance": {
            "boundary_clean": _boundary_clean(boundary),
            "nonrisk_conservative_selected_count_eq_0": int(best_formula.get("nonrisk_conservative_selected_count", 0) or 0)
            == 0,
            "conservative_risk_selected_rate_in_target_band": TARGET_RATE_LOW
            <= _num(best_formula.get("conservative_risk_selected_rate"))
            <= TARGET_RATE_HIGH,
            "risk_type_switch_spread_le_0_30": _num(best_formula.get("risk_type_switch_spread"), 999.0) <= 0.30,
            "mixed_risk_combo_spread_le_0_30": _num(best_formula.get("mixed_risk_combo_spread"), 999.0) <= 0.30,
            "single_risk_spread_le_0_30": _num(best_formula.get("single_risk_spread"), 999.0) <= 0.30,
            "family_extreme_rate_count_eq_0": int(best_formula.get("family_extreme_rate_count", 0) or 0) == 0,
            "rule_suppression_detected": bool(best_formula.get("rule_suppression_detected", False)),
            "previous_rule_ppo_suppression_detected": bool(best_formula.get("previous_rule_ppo_suppression_detected", False)),
            "dominant_component_after_formula_not_energy": best_formula.get("dominant_component_after_formula")
            not in ENERGY_COMPONENTS,
            "dry_vent_not_saturated": int(best_formula.get("dry_vent_conservative_selected_count", 0) or 0)
            < int(best_formula.get("dry_vent_row_count", 0) or 0),
            "canopy_dew_lt0_dew_combo_not_zero": int(
                best_formula.get("canopy_dew_lt0_dew_combo_conservative_selected_count", 0) or 0
            )
            > 0,
        },
        "next_action": next_action,
        "readiness_reasons": readiness_reasons,
        "notes": [
            "v8173.5 is offline counterfactual risk-state semantics design only.",
            "Surrogate deltas are audit approximations, not one-step simulator predictions.",
            "Passing v8173.5 would still not be reward, safety, promotion, or closed-loop evidence.",
        ],
    }
    return report, {
        "risk_state_formula_curve": curve_rows,
        "risk_state_formula_family_rate_table": family_rows,
        "risk_state_formula_combo_rate_table": combo_rows,
        "combo_failure_table": combo_failure,
        "dry_vent_guard_table": dry_guard,
        "canopy_dew_recovery_table": canopy_recovery,
        "risk_state_formula_switch_explainability": switch_rows,
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    boundary = report.get("boundary_runtime_safety", {})
    diagnostics = report.get("diagnostics", {})
    observed = (diagnostics.get("observed_gap") or {}).get("hypothesis_revision") or {}
    best = diagnostics.get("best_formula_row") or {}
    lines = [
        "# C-STCC v8173.5 Combo-Specific Risk-State Energy Semantics",
        "",
        "Existing-trace counterfactual audit only. No online LLM, no rollout, no runtime scorer/template change.",
        "",
        "## Hypothesis Revision",
        "",
        f"- Old hypothesis: {observed.get('old_hypothesis')}",
        f"- Failure evidence: {observed.get('failure_evidence_or_missing_evidence')}",
        f"- New hypothesis: {observed.get('new_hypothesis')}",
        "",
        "## Boundary",
        "",
        f"- Audit success rate: {_num(boundary.get('audit_success_rate')):.3f}",
        f"- Final-action invariant rate: {_num(boundary.get('final_action_invariant_rate')):.3f}",
        f"- Online LLM / rollout / real projection steps: {boundary.get('online_llm_called_steps', 0)} / {boundary.get('predictive_rollout_executed_steps', 0)} / {boundary.get('real_tomato_safety_projection_steps', 0)}",
        "",
        "## Best Risk-State Formula",
        "",
        f"- Policy / formula: {best.get('policy', 'none')} / {best.get('formula', 'none')}",
        f"- Conservative risk selected rate: {_num(best.get('conservative_risk_selected_rate')):.3f}",
        f"- Family / mixed combo / single spread: {_num(best.get('risk_type_switch_spread')):.3f} / {_num(best.get('mixed_risk_combo_spread')):.3f} / {_num(best.get('single_risk_spread')):.3f}",
        f"- Dry-vent selected: {best.get('dry_vent_conservative_selected_count', 0)} / {best.get('dry_vent_row_count', 0)}",
        f"- Canopy-dew/dew selected: {best.get('canopy_dew_lt0_dew_combo_conservative_selected_count', 0)} / {best.get('canopy_dew_lt0_dew_combo_row_count', 0)}",
        f"- Dominant component after formula: {best.get('dominant_component_after_formula', 'none')}",
        "",
        "## Decision",
        "",
        f"- Next action: {report.get('next_action')}",
        f"- Readiness reasons: {', '.join(report.get('readiness_reasons', [])) or 'none'}",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    report: Mapping[str, Any],
    tables: Mapping[str, list[Mapping[str, Any]]],
    output_prefix: str | Path,
) -> dict[str, str]:
    prefix = _resolve(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    suffix = "_20260607_y2020_d240_s42_n720"
    parent = prefix.parent
    paths = {
        "json": prefix.with_suffix(".json"),
        "md": prefix.with_suffix(".md"),
        "risk_state_formula_curve_csv": parent / f"cstcc_v8173_risk_state_formula_curve{suffix}.csv",
        "combo_failure_table_csv": parent / f"cstcc_v8173_combo_failure_table{suffix}.csv",
        "dry_vent_guard_table_csv": parent / f"cstcc_v8173_dry_vent_guard_table{suffix}.csv",
        "canopy_dew_recovery_table_csv": parent / f"cstcc_v8173_canopy_dew_recovery_table{suffix}.csv",
        "risk_state_formula_switch_explainability_csv": parent
        / f"cstcc_v8173_risk_state_formula_switch_explainability{suffix}.csv",
        "risk_state_formula_family_rate_table_csv": parent / f"cstcc_v8173_risk_state_formula_family_rate_table{suffix}.csv",
        "risk_state_formula_combo_rate_table_csv": parent / f"cstcc_v8173_risk_state_formula_combo_rate_table{suffix}.csv",
    }
    paths["json"].write_text(_compact_json(report) + "\n", encoding="utf-8")
    paths["md"].write_text(build_markdown(report), encoding="utf-8")
    _write_csv(paths["risk_state_formula_curve_csv"], tables.get("risk_state_formula_curve", []))
    _write_csv(paths["combo_failure_table_csv"], tables.get("combo_failure_table", []))
    _write_csv(paths["dry_vent_guard_table_csv"], tables.get("dry_vent_guard_table", []))
    _write_csv(paths["canopy_dew_recovery_table_csv"], tables.get("canopy_dew_recovery_table", []))
    _write_csv(
        paths["risk_state_formula_switch_explainability_csv"],
        tables.get("risk_state_formula_switch_explainability", []),
    )
    _write_csv(paths["risk_state_formula_family_rate_table_csv"], tables.get("risk_state_formula_family_rate_table", []))
    _write_csv(paths["risk_state_formula_combo_rate_table_csv"], tables.get("risk_state_formula_combo_rate_table", []))
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v8173.5 C-STCC combo-specific risk-state energy audit.")
    parser.add_argument("--trace-csv", type=str, default=str(DEFAULT_TRACE_CSV))
    parser.add_argument("--summary-json", type=str, default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--jsonl-root", type=str, default=str(DEFAULT_JSONL_ROOT))
    parser.add_argument("--v8173-json", type=str, default=str(DEFAULT_V8173_JSON))
    parser.add_argument("--v81731-json", type=str, default=str(DEFAULT_V81731_JSON))
    parser.add_argument("--v81732-json", type=str, default=str(DEFAULT_V81732_JSON))
    parser.add_argument("--v81733-json", type=str, default=str(DEFAULT_V81733_JSON))
    parser.add_argument("--v81734-json", type=str, default=str(DEFAULT_V81734_JSON))
    parser.add_argument("--v81732-policy-curve-csv", type=str, default=str(DEFAULT_V81732_POLICY_CURVE_CSV))
    parser.add_argument("--output-prefix", type=str, default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--no-golden-check", action="store_true")
    args = parser.parse_args()

    report, tables = build_report(
        trace_csv_rows=_load_csv(args.trace_csv),
        summary_json=_load_json(args.summary_json),
        jsonl_rows=_load_jsonl_rows(args.jsonl_root),
        v8173_json=_load_json(args.v8173_json),
        v81731_json=_load_json(args.v81731_json),
        v81732_json=_load_json(args.v81732_json),
        v81733_json=_load_json(args.v81733_json),
        v81734_json=_load_json(args.v81734_json),
        v81732_policy_curve_rows=_load_csv(args.v81732_policy_curve_csv),
        enforce_golden=not args.no_golden_check,
    )
    outputs = write_outputs(report=report, tables=tables, output_prefix=args.output_prefix)
    print(
        json.dumps(
            {"outputs": outputs, "next_action": report["next_action"], "readiness_reasons": report["readiness_reasons"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
