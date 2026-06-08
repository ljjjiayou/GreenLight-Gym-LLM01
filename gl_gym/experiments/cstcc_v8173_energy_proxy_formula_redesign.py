"""v8173.4 offline risk-semantic energy proxy formula redesign audit.

This stage evaluates formula candidates on existing H720 evidence only. The
formulas are evaluator-only counterfactuals: they do not change runtime
score_dual, template pools, final actions, online LLM usage, predictive rollout,
or Tomato Safety projection.
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
    DEFAULT_OUTPUT_PREFIX as DEFAULT_V81732_OUTPUT_PREFIX,
    DEFAULT_V81731_JSON,
    _is_better_curve_for_policy,
    _spread,
    _target_gap,
    _target_label,
    _v81731_golden_issues,
    annotate_records_for_policy,
)
from gl_gym.experiments.cstcc_v8173_energy_proxy_reweight_design import (
    DEFAULT_OUTPUT_PREFIX as DEFAULT_V81733_OUTPUT_PREFIX,
    DEFAULT_V81732_COMBO_CSV,
    DEFAULT_V81732_COMPONENT_CSV,
    DEFAULT_V81732_FAMILY_CSV,
    DEFAULT_V81732_POLICY_CURVE_CSV,
    DEFAULT_V81732_JSON,
    _v81732_golden_issues,
)
from gl_gym.experiments.cstcc_v8173_mixed_risk_conflict_resolver_design import (
    DEFAULT_V8173_JSON,
)
from gl_gym.experiments.cstcc_v8173_risk_type_specific_bonus_template_counterfactual import (
    COMPONENT_KEYS,
    DEFAULT_JSONL_ROOT,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TRACE_CSV,
    _bool,
    _boundary_clean,
    _boundary_summary,
    _build_step_records,
    _compact_json,
    _component_delta,
    _distribution,
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


DEFAULT_V81733_JSON = DEFAULT_V81733_OUTPUT_PREFIX.with_suffix(".json")
DEFAULT_OUTPUT_PREFIX = Path(
    "gl_gym/result/diagnostics/cstcc_v8173_energy_proxy_formula_redesign_20260607_y2020_d240_s42_n720"
)

POLICIES = (
    "canopy_dew_first_v1",
    "mixed_combo_specific_v1",
    "dew_before_canopy_lt1_v1",
    "severity_aware_non_additive_v2",
)
FORMULAS = (
    "baseline_current_energy",
    "formula_A_risk_mitigation_credit",
    "formula_B_risk_inconsistent_penalty",
    "formula_C_combo_specific_energy_semantics",
    "formula_D_net_risk_energy_proxy",
)
TARGET_RATE_LOW = 0.35
TARGET_RATE_HIGH = 0.75
ENERGY_WEIGHT = 0.20
ENERGY_COMPONENTS = {"raw_energy_proxy", "projected_energy_proxy"}


def _candidate_delta(candidate: Mapping[str, Any]) -> dict[str, float]:
    raw = candidate.get("first_action_delta_from_reference") or {}
    return {field: _num(raw.get(field), 0.0) for field in ACTION_FIELDS}


def _jsonl_candidate_delta_by_step(jsonl_rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, dict[str, float]]]:
    by_step: dict[int, dict[str, dict[str, float]]] = {}
    for index, row in enumerate(jsonl_rows):
        step = int(_num(row.get("step"), index))
        candidates = row.get("candidate_violation_provenance") or []
        if isinstance(candidates, str):
            candidates = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            candidate_id = str(candidate.get("candidate_id") or "")
            delta = candidate.get("first_action_delta_from_reference")
            if candidate_id and isinstance(delta, Mapping):
                by_step.setdefault(step, {})[candidate_id] = {
                    field: _num(delta.get(field), 0.0) for field in ACTION_FIELDS
                }
    return by_step


def enrich_records_with_candidate_deltas(
    records: Sequence[Mapping[str, Any]],
    jsonl_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    deltas = _jsonl_candidate_delta_by_step(jsonl_rows)
    issues: list[str] = []
    enriched_records: list[dict[str, Any]] = []
    for record in records:
        step = int(_num(record.get("step")))
        enriched = dict(record)
        candidates: dict[str, dict[str, Any]] = {
            str(candidate_id): dict(candidate)
            for candidate_id, candidate in (record.get("candidates") or {}).items()
        }
        for candidate_id, candidate in candidates.items():
            if candidate_id in deltas.get(step, {}):
                candidate["first_action_delta_from_reference"] = deltas[step][candidate_id]
            if (record.get("risk_flags") or {}).get("any_risk", False) and candidate.get("feasible", False):
                if "first_action_delta_from_reference" not in candidate:
                    issues.append(f"first_action_delta_missing:step={step}:candidate={candidate_id}")
        if not record.get("current_runtime_final_action"):
            issues.append(f"current_runtime_final_action_missing:step={step}")
        enriched["candidates"] = candidates
        enriched_records.append(enriched)
    return enriched_records, sorted(set(issues))


def _policy_local_best_vectors(policy_curve_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    best_by_policy: dict[str, dict[str, Any]] = {}
    for row in policy_curve_rows:
        policy = str(row.get("policy") or "")
        if policy not in POLICIES:
            continue
        candidate = dict(row)
        current = best_by_policy.get(policy, {})
        if _is_better_curve_for_policy(candidate, current, policy):
            best_by_policy[policy] = candidate
    return best_by_policy


def _bonus_vector(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "dry_or_high_vpd_bonus": _num(row.get("dry_or_high_vpd_bonus")),
        "dry_vent_bonus": _num(row.get("dry_vent_bonus")),
        "dew_or_humidity_bonus": _num(row.get("dew_or_humidity_bonus")),
        "canopy_dew_bonus": _num(row.get("canopy_dew_bonus")),
    }


def _risk_flags(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("risk_flags") or {}


def _resolver(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record.get("resolver") or {}


def _combo_id(record: Mapping[str, Any]) -> str:
    return str(_resolver(record).get("risk_combo_id") or "none")


def _family(record: Mapping[str, Any]) -> str:
    return str(_resolver(record).get("resolved_risk_family_new") or "none")


def _has_dew_or_canopy_risk(flags: Mapping[str, Any]) -> bool:
    return bool(
        _bool(flags.get("dew_risk"))
        or _bool(flags.get("humidity_or_dew_risk"))
        or _bool(flags.get("canopy_dew_margin_lt1"))
        or _bool(flags.get("canopy_dew_margin_lt0"))
    )


def _has_dry_or_high_vpd_risk(flags: Mapping[str, Any]) -> bool:
    return bool(_bool(flags.get("dry_risk")) or _bool(flags.get("high_vpd")) or _bool(flags.get("dry_or_high_vpd_risk")))


def _mitigation_credit(flags: Mapping[str, Any], delta: Mapping[str, float]) -> float:
    if not _has_dew_or_canopy_risk(flags):
        return 0.0
    heat_credit = min(0.06, 0.40 * max(0.0, _num(delta.get("u_heating"))))
    vent_credit = min(0.05, 0.25 * max(0.0, _num(delta.get("u_ventilation"))))
    screen_credit = min(0.04, 0.25 * max(0.0, -_num(delta.get("u_screen"))))
    severe_bonus = 0.02 if _bool(flags.get("canopy_dew_margin_lt0")) and (heat_credit or vent_credit or screen_credit) else 0.0
    return heat_credit + vent_credit + screen_credit + severe_bonus


def _inconsistent_penalty(flags: Mapping[str, Any], delta: Mapping[str, float]) -> float:
    penalty = 0.0
    if _has_dry_or_high_vpd_risk(flags):
        penalty += min(0.08, 0.35 * max(0.0, _num(delta.get("u_ventilation"))))
        penalty += min(0.05, 0.25 * max(0.0, _num(delta.get("u_heating"))))
    if _bool(flags.get("dry_vent_risk")):
        vent_delta = _num(delta.get("u_ventilation"))
        penalty += 0.06 if vent_delta >= -0.01 else max(0.0, 0.03 - 0.20 * abs(vent_delta))
    return penalty


def _actuator_conflict_penalty(flags: Mapping[str, Any], delta: Mapping[str, float]) -> float:
    if not (_has_dew_or_canopy_risk(flags) and _has_dry_or_high_vpd_risk(flags)):
        return 0.0
    heat_up = max(0.0, _num(delta.get("u_heating")))
    vent_up = max(0.0, _num(delta.get("u_ventilation")))
    if heat_up > 0.03 and vent_up > 0.03:
        return min(0.05, 0.20 * (heat_up + vent_up))
    return 0.0


def formula_projected_energy_score(
    formula: str,
    *,
    record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    component: Mapping[str, Any],
) -> float:
    original = _num(component.get("projected_energy_proxy"))
    flags = _risk_flags(record)
    delta = _candidate_delta(candidate)
    combo = _combo_id(record)
    if formula == "baseline_current_energy":
        return clamp(original)
    if formula == "formula_A_risk_mitigation_credit":
        return clamp(original + _mitigation_credit(flags, delta))
    if formula == "formula_B_risk_inconsistent_penalty":
        return clamp(original - _inconsistent_penalty(flags, delta))
    if formula == "formula_C_combo_specific_energy_semantics":
        score = original
        if combo == "canopy_dew_lt0+dew_or_humidity":
            score += 1.35 * _mitigation_credit(flags, delta)
        elif combo == "canopy_dew_lt1+dew_or_humidity":
            score += 0.75 * _mitigation_credit(flags, delta)
        elif combo == "dry_vent+dry_or_high_vpd":
            score -= 1.35 * _inconsistent_penalty(flags, delta)
        elif "dry_vent" in combo and "dry_or_high_vpd" in combo:
            score -= _inconsistent_penalty(flags, delta)
        else:
            score += 0.35 * _mitigation_credit(flags, delta) - 0.35 * _inconsistent_penalty(flags, delta)
        return clamp(score)
    if formula == "formula_D_net_risk_energy_proxy":
        return clamp(
            original
            + _mitigation_credit(flags, delta)
            - _inconsistent_penalty(flags, delta)
            - _actuator_conflict_penalty(flags, delta)
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
    candidate_id: str,
    candidate: Mapping[str, Any],
    component: Mapping[str, Any],
    bonus_vector: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    family = _family(record)
    original_energy = _num(component.get("projected_energy_proxy"))
    formula_energy = formula_projected_energy_score(formula, record=record, candidate=candidate, component=component)
    formula_delta = formula_energy - original_energy
    assigned_bonus = 0.0
    if _is_conservative(candidate) and (record.get("risk_flags") or {}).get("any_risk", False) and family != "none":
        assigned_bonus = bonus_for_family(bonus_vector, family)
    return (
        _num(component.get("final_score"), -1e18) + ENERGY_WEIGHT * formula_delta + assigned_bonus,
        {
            "original_projected_energy_proxy": original_energy,
            "formula_projected_energy_score": formula_energy,
            "formula_energy_delta": formula_delta,
            "assigned_bonus": assigned_bonus,
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
            candidate_id=candidate_id,
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


def _adjusted_component_gap_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    formula: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for record in records:
        if not (_risk_flags(record).get("any_risk", False)):
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
        zero_energy = formula_projected_energy_score(
            formula,
            record=record,
            candidate=zero,
            component=zero_components,
        )
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
        for component in COMPONENT_KEYS:
            values = [_num(item.get(component)) for item in items]
            rows.append(
                {
                    "resolved_risk_family": family,
                    "component": component,
                    "row_count": len(items),
                    "mean": float(mean(values)) if values else 0.0,
                    "p95": _quantile(values, 0.95),
                    "max": max(values) if values else 0.0,
                }
            )
    return rows


def _dominant_after_formula(records: Sequence[Mapping[str, Any]], formula: str) -> str:
    return _dominant_component(
        [row for row in _adjusted_component_gap_rows(records, formula=formula) if row.get("component") != "final_score"]
    )


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _curve_accepts(row: Mapping[str, Any]) -> bool:
    rate = _num(row.get("conservative_risk_selected_rate"))
    dry_count = int(row.get("dry_vent_conservative_selected_count", 0) or 0)
    dry_total = int(row.get("dry_vent_row_count", 0) or 0)
    combo_count = int(row.get("canopy_dew_lt0_dew_combo_conservative_selected_count", 0) or 0)
    combo_total = int(row.get("canopy_dew_lt0_dew_combo_row_count", 0) or 0)
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
        and (dry_total == 0 or dry_count < dry_total)
        and (combo_total == 0 or combo_count > 0)
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
            canopy_dew_combo_counts = combo_counts.get("canopy_dew_lt0+dew_or_humidity") or {"count": 0, "conservative": 0}
            dominant = _dominant_after_formula(annotated, formula)
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
    diagnostics = {
        "best_formula_row": best_row,
        "policy_count": len(policy_best_vectors),
        "formula_count": len(formulas),
    }
    return curve_rows, family_rows, combo_rows, diagnostics


def raw_projected_delta_rows(records: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    annotated = annotate_records_for_policy(records, policy)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in annotated:
        if _risk_flags(record).get("any_risk", False):
            groups.setdefault(_family(record), []).append(record)
            groups.setdefault(_combo_id(record), []).append(record)
    rows: list[dict[str, Any]] = []
    for group_id, items in sorted(groups.items()):
        raw_gaps: list[float] = []
        projected_gaps: list[float] = []
        for record in items:
            conservative_id = str(record.get("conservative_candidate_id") or "")
            zero_id = str(record.get("zero_bonus_winner_id") or "")
            zero_components = (record.get("components") or {}).get(zero_id, {})
            conservative_components = (record.get("components") or {}).get(conservative_id, {})
            raw_gaps.append(_num(zero_components.get("raw_energy_proxy")) - _num(conservative_components.get("raw_energy_proxy")))
            projected_gaps.append(
                _num(zero_components.get("projected_energy_proxy")) - _num(conservative_components.get("projected_energy_proxy"))
            )
        delta_values = [raw - projected for raw, projected in zip(raw_gaps, projected_gaps)]
        rows.append(
            {
                "policy": policy,
                "group_id": group_id,
                "row_count": len(items),
                "raw_energy_gap_mean": float(mean(raw_gaps)) if raw_gaps else 0.0,
                "projected_energy_gap_mean": float(mean(projected_gaps)) if projected_gaps else 0.0,
                "raw_minus_projected_gap_mean": float(mean(delta_values)) if delta_values else 0.0,
                "raw_minus_projected_gap_p95": _quantile(delta_values, 0.95),
                "raw_projected_identical": all(abs(value) < 1e-12 for value in delta_values),
            }
        )
    return rows


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


def _v81733_golden_issues(v81733_json: Mapping[str, Any]) -> list[str]:
    if not v81733_json:
        return ["v81733_json_missing"]
    best = ((v81733_json.get("diagnostics") or {}).get("best_reweight_vector") or {})
    issues: list[str] = []
    if best.get("factor_vector_id") != "canopy_lt0=1.0|canopy_lt1=1.0|dew=1.0|dry_vent=1.0|dry=1.0":
        issues.append("golden_mismatch:v81733_best_factor_vector")
    if abs(_num(best.get("conservative_risk_selected_rate")) - 0.5396825396825397) > 1e-9:
        issues.append("golden_mismatch:v81733_conservative_risk_selected_rate")
    if abs(_num(best.get("risk_type_switch_spread")) - 0.641025641025641) > 1e-9:
        issues.append("golden_mismatch:v81733_risk_type_switch_spread")
    if best.get("dominant_component_after_reweight") != "raw_energy_proxy":
        issues.append("golden_mismatch:v81733_dominant_component")
    return issues


def _route_next_action(
    *,
    boundary: Mapping[str, Any],
    schema_issues: Sequence[str],
    best_formula: Mapping[str, Any],
) -> tuple[str, list[str]]:
    if not _boundary_clean(boundary):
        return "stop_and_repair_cstcc_shadow_boundary", ["boundary_or_candidate_health_failure"]
    if schema_issues:
        return "v8173_schema_or_golden_trace_repair_plan", list(schema_issues)
    if _curve_accepts(best_formula):
        return "v8173_5_opt_in_formula_shadow_trace_plan", ["risk_semantic_energy_formula_passes_all_gates"]
    dry_total = int(best_formula.get("dry_vent_row_count", 0) or 0)
    dry_selected = int(best_formula.get("dry_vent_conservative_selected_count", 0) or 0)
    combo_total = int(best_formula.get("canopy_dew_lt0_dew_combo_row_count", 0) or 0)
    combo_selected = int(best_formula.get("canopy_dew_lt0_dew_combo_conservative_selected_count", 0) or 0)
    if combo_total > 0 and combo_selected > 0 and dry_total > 0 and dry_selected >= dry_total:
        return "v8173_dry_vent_guard_formula_iteration_plan", ["under_target_improved_but_dry_vent_still_saturated"]
    if _num(best_formula.get("mixed_risk_combo_spread"), 0.0) > 0.30:
        return "v8173_combo_specific_energy_formula_iteration_plan", ["specific_mixed_combo_extremes_remain"]
    dominant = str(best_formula.get("dominant_component_after_formula") or "")
    if dominant in {"base_final_score", "raw_smoothness", "projected_smoothness"}:
        return "v8174_risk_specific_template_design_plan", [f"energy_no_longer_dominant_but_{dominant}_dominates"]
    return "v8173_combo_specific_energy_formula_iteration_plan", ["formula_counterfactual_not_balanced"]


def build_report(
    *,
    trace_csv_rows: list[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    jsonl_rows: list[Mapping[str, Any]],
    v8173_json: Mapping[str, Any],
    v81731_json: Mapping[str, Any],
    v81732_json: Mapping[str, Any],
    v81733_json: Mapping[str, Any],
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
    if enforce_golden:
        schema_issues = sorted(
            set(
                schema_issues
                + reconstruction_issues
                + delta_issues
                + _v81731_golden_issues(v81731_json)
                + _v81732_golden_issues(v81732_json)
                + _v81733_golden_issues(v81733_json)
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
    raw_projected_rows: list[dict[str, Any]] = []
    for policy in policy_best_vectors:
        raw_projected_rows.extend(raw_projected_delta_rows(records, policy))
    raw_projected_identical = all(row.get("raw_projected_identical", False) for row in raw_projected_rows)
    switch_rows = switch_rows_for_formula(
        records,
        best_formula=best_formula,
        policy_best_vectors=policy_best_vectors,
    )
    next_action, readiness_reasons = _route_next_action(
        boundary=boundary,
        schema_issues=schema_issues,
        best_formula=best_formula,
    )
    diagnostics = {
        "observed_gap": {
            "v81733_noop_reweight_best": ((v81733_json.get("diagnostics") or {}).get("best_reweight_vector") or {}),
            "reweight_failure_interpretation": "risk-family energy-gap relief did not touch the core action-risk semantics",
            "dry_vent_over_target_previous": "dry_vent=21/21 over-target under v8173.2/v8173.3",
            "canopy_dew_lt0_dew_under_target_previous": "canopy_dew_lt0+dew_or_humidity=0/6 under-target under v8173.2/v8173.3",
        },
        "policy_local_best_vectors": policy_best_vectors,
        "policy_count": len(policy_best_vectors),
        "formula_count": len(formulas),
        "best_formula_row": best_formula,
        "raw_projected_energy_gap_identical": raw_projected_identical,
        "energy_redesign_layer": "energy_proxy_body" if raw_projected_identical else "projection_layer_or_energy_proxy_body",
        "v8173_reference_next_action": v8173_json.get("next_action"),
        "v81731_reference_next_action": v81731_json.get("next_action"),
        "v81732_reference_next_action": v81732_json.get("next_action"),
        "v81733_reference_next_action": v81733_json.get("next_action"),
    }
    report = {
        "schema_version": "cstcc_v8173_energy_proxy_formula_redesign_v1",
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
            "formula_redesign_is_evaluator_only": True,
            "runtime_level0_sequence_score_changed": False,
            "energy_weight_changed": False,
            "energy_weight_used": ENERGY_WEIGHT,
            "bonus_vector_changed": False,
            "prior_confidence_changed": False,
            "runtime_final_action_changed": False,
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
            "v8173.4 is offline counterfactual formula design only.",
            "A passing formula would still be existing-trace readiness, not reward, safety, promotion, or closed-loop evidence.",
            "Runtime energy proxy changes require a later opt-in shadow trace plan.",
        ],
    }
    return report, {
        "formula_curve": curve_rows,
        "formula_family_rate_table": family_rows,
        "formula_combo_rate_table": combo_rows,
        "formula_raw_projected_delta": raw_projected_rows,
        "formula_switch_explainability": switch_rows,
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    boundary = report.get("boundary_runtime_safety", {})
    diagnostics = report.get("diagnostics", {})
    observed = diagnostics.get("observed_gap") or {}
    best = diagnostics.get("best_formula_row") or {}
    lines = [
        "# C-STCC v8173.4 Risk-Semantic Energy Proxy Formula Redesign",
        "",
        "Existing-trace counterfactual audit only. No online LLM, no rollout, no runtime scorer/template change.",
        "",
        "## Observed Gap",
        "",
        f"- v8173.3 no-op best: {(observed.get('v81733_noop_reweight_best') or {}).get('factor_vector_id', 'none')}",
        f"- Interpretation: {observed.get('reweight_failure_interpretation')}",
        f"- Dry-vent gap: {observed.get('dry_vent_over_target_previous')}",
        f"- Canopy/dew gap: {observed.get('canopy_dew_lt0_dew_under_target_previous')}",
        "",
        "## Boundary",
        "",
        f"- Audit success rate: {_num(boundary.get('audit_success_rate')):.3f}",
        f"- Final-action invariant rate: {_num(boundary.get('final_action_invariant_rate')):.3f}",
        f"- Online LLM / rollout / real projection steps: {boundary.get('online_llm_called_steps', 0)} / {boundary.get('predictive_rollout_executed_steps', 0)} / {boundary.get('real_tomato_safety_projection_steps', 0)}",
        "",
        "## Best Formula",
        "",
        f"- Policy / formula: {best.get('policy', 'none')} / {best.get('formula', 'none')}",
        f"- Conservative risk selected rate: {_num(best.get('conservative_risk_selected_rate')):.3f}",
        f"- Family / mixed combo / single spread: {_num(best.get('risk_type_switch_spread')):.3f} / {_num(best.get('mixed_risk_combo_spread')):.3f} / {_num(best.get('single_risk_spread')):.3f}",
        f"- Dry-vent selected: {best.get('dry_vent_conservative_selected_count', 0)} / {best.get('dry_vent_row_count', 0)}",
        f"- Canopy-dew/dew selected: {best.get('canopy_dew_lt0_dew_combo_conservative_selected_count', 0)} / {best.get('canopy_dew_lt0_dew_combo_row_count', 0)}",
        f"- Dominant component after formula: {best.get('dominant_component_after_formula', 'none')}",
        f"- Raw/projected gap identical: {diagnostics.get('raw_projected_energy_gap_identical')}",
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
    stem = prefix.name
    parent = prefix.parent
    base = "energy_proxy_formula_redesign"
    paths = {
        "json": prefix.with_suffix(".json"),
        "md": prefix.with_suffix(".md"),
        "formula_curve_csv": (parent / stem.replace(base, "formula_curve")).with_suffix(".csv"),
        "formula_family_rate_table_csv": (parent / stem.replace(base, "formula_family_rate_table")).with_suffix(".csv"),
        "formula_combo_rate_table_csv": (parent / stem.replace(base, "formula_combo_rate_table")).with_suffix(".csv"),
        "formula_raw_projected_delta_csv": (parent / stem.replace(base, "formula_raw_projected_delta")).with_suffix(".csv"),
        "formula_switch_explainability_csv": (
            parent / stem.replace(base, "formula_switch_explainability")
        ).with_suffix(".csv"),
    }
    paths["json"].write_text(_compact_json(report) + "\n", encoding="utf-8")
    paths["md"].write_text(build_markdown(report), encoding="utf-8")
    _write_csv(paths["formula_curve_csv"], tables.get("formula_curve", []))
    _write_csv(paths["formula_family_rate_table_csv"], tables.get("formula_family_rate_table", []))
    _write_csv(paths["formula_combo_rate_table_csv"], tables.get("formula_combo_rate_table", []))
    _write_csv(paths["formula_raw_projected_delta_csv"], tables.get("formula_raw_projected_delta", []))
    _write_csv(paths["formula_switch_explainability_csv"], tables.get("formula_switch_explainability", []))
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v8173.4 C-STCC energy proxy formula redesign audit.")
    parser.add_argument("--trace-csv", type=str, default=str(DEFAULT_TRACE_CSV))
    parser.add_argument("--summary-json", type=str, default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--jsonl-root", type=str, default=str(DEFAULT_JSONL_ROOT))
    parser.add_argument("--v8173-json", type=str, default=str(DEFAULT_V8173_JSON))
    parser.add_argument("--v81731-json", type=str, default=str(DEFAULT_V81731_JSON))
    parser.add_argument("--v81732-json", type=str, default=str(DEFAULT_V81732_JSON))
    parser.add_argument("--v81733-json", type=str, default=str(DEFAULT_V81733_JSON))
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
