"""v8173.1 offline mixed-risk conflict resolver design for C-STCC.

This stage is an existing-trace counterfactual audit. The resolver is
evaluator-only: it does not change runtime scorer behavior, template pools,
online LLM use, predictive rollout, Tomato Safety projection, or final action.
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

from gl_gym.experiments.cstcc_v8173_risk_type_specific_bonus_template_counterfactual import (
    COMPONENT_KEYS,
    DEFAULT_JSONL_ROOT,
    DEFAULT_OUTPUT_PREFIX as DEFAULT_V8173_OUTPUT_PREFIX,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TRACE_CSV,
    RISK_FAMILIES,
    RISK_TYPE_KEYS,
    _bool,
    _boundary_clean,
    _boundary_summary,
    _build_step_records,
    _compact_json,
    _component_delta,
    _distribution,
    _dominant_component,
    _json_obj,
    _load_csv,
    _load_json,
    _load_jsonl_rows,
    _missing_schema_fields,
    _num,
    _parse_candidate_source,
    _resolve,
    _source,
    _template,
    _write_csv,
    bonus_for_family,
    bonus_vectors,
    vector_id,
)


DEFAULT_V8173_JSON = DEFAULT_V8173_OUTPUT_PREFIX.with_suffix(".json")
DEFAULT_OUTPUT_PREFIX = Path(
    "gl_gym/result/diagnostics/cstcc_v8173_mixed_risk_conflict_resolver_design_20260606_y2020_d240_s42_n720"
)

RESOLVER_POLICY = "severity_aware_non_additive_v1"
CONFLICT_TYPES = (
    "single_risk",
    "dry_vs_dew",
    "dry_vs_canopy_dew",
    "dew_vs_canopy_dew",
    "dry_vent_vs_dew",
    "multi_3plus",
    "none",
)


def risk_family_flags(flags: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "dry_vent": _bool(flags.get("dry_vent_risk")),
        "canopy_dew_lt0": _bool(flags.get("canopy_dew_margin_lt0")),
        "canopy_dew_lt1": _bool(flags.get("canopy_dew_margin_lt1")) and not _bool(flags.get("canopy_dew_margin_lt0")),
        "dew_or_humidity": _bool(flags.get("humidity_or_dew_risk")) or _bool(flags.get("dew_risk")),
        "dry_or_high_vpd": _bool(flags.get("dry_or_high_vpd_risk")) or _bool(flags.get("dry_risk")) or _bool(flags.get("high_vpd")),
    }


def active_risk_families(flags: Mapping[str, Any]) -> list[str]:
    families = risk_family_flags(flags)
    return [family for family in RISK_FAMILIES if family != "none" and families.get(family, False)]


def risk_combo_id(flags: Mapping[str, Any]) -> str:
    active = active_risk_families(flags)
    return "+".join(active) if active else "none"


def conflict_type(flags: Mapping[str, Any]) -> str:
    active = set(active_risk_families(flags))
    if not active:
        return "none"
    if len(active) == 1:
        return "single_risk"
    if active == {"dry_vent", "dry_or_high_vpd"}:
        return "single_risk"
    if len(active) >= 3:
        return "multi_3plus"
    if "dry_vent" in active and "dew_or_humidity" in active:
        return "dry_vent_vs_dew"
    if "dry_or_high_vpd" in active and "dew_or_humidity" in active:
        return "dry_vs_dew"
    if "dry_or_high_vpd" in active and ("canopy_dew_lt0" in active or "canopy_dew_lt1" in active):
        return "dry_vs_canopy_dew"
    if "dew_or_humidity" in active and ("canopy_dew_lt0" in active or "canopy_dew_lt1" in active):
        return "dew_vs_canopy_dew"
    return "multi_3plus"


def resolve_risk_family(record: Mapping[str, Any]) -> dict[str, Any]:
    flags = record.get("risk_flags") or {}
    merged = record.get("merged") or {}
    ventilation = _num(merged.get("u_ventilation"), _num((record.get("current_runtime_final_action") or {}).get("u_ventilation")))
    conflict = conflict_type(flags)
    active = active_risk_families(flags)
    if not active:
        family = "none"
        reason = "no_active_risk"
    elif _bool(flags.get("canopy_dew_margin_lt0")):
        family = "canopy_dew_lt0"
        reason = "severe_canopy_dew_lt0_priority"
    elif _bool(flags.get("dry_vent_risk")) and (_bool(flags.get("high_vpd")) or ventilation > 0.70):
        family = "dry_vent"
        reason = "dry_vent_high_vpd_or_strong_ventilation_priority"
    elif _bool(flags.get("canopy_dew_margin_lt1")):
        family = "canopy_dew_lt1"
        reason = "moderate_canopy_dew_lt1_priority"
    elif _bool(flags.get("humidity_or_dew_risk")) or _bool(flags.get("dew_risk")):
        family = "dew_or_humidity"
        if _bool(flags.get("high_vpd")) or _bool(flags.get("dry_or_high_vpd_risk")):
            reason = "humidity_high_vpd_mixed_conflict_prefers_dew_or_humidity"
        else:
            reason = "dew_or_humidity_priority"
    elif _bool(flags.get("dry_vent_risk")):
        family = "dry_vent"
        reason = "dry_vent_priority"
    elif _bool(flags.get("dry_or_high_vpd_risk")) or _bool(flags.get("dry_risk")) or _bool(flags.get("high_vpd")):
        family = "dry_or_high_vpd"
        reason = "dry_or_high_vpd_priority"
    else:
        family = "none"
        reason = "unassigned_risk_flags"
    return {
        "resolver_policy": RESOLVER_POLICY,
        "risk_combo_id": risk_combo_id(flags),
        "active_risk_families": active,
        "active_risk_flags": record.get("active_risk_flags") or [],
        "primary_risk_family_old": record.get("primary_risk_family") or "none",
        "resolved_risk_family_new": family,
        "conflict_type": conflict,
        "conflict_priority_reason": reason,
        "bonus_family": family,
        "template_family": _template_family(family),
    }


def _template_family(family: str) -> str:
    if family == "dry_vent":
        return "conservative_stabilize_dry_vent_guard"
    if family == "dry_or_high_vpd":
        return "conservative_stabilize_high_vpd"
    if family == "dew_or_humidity":
        return "conservative_stabilize_dew"
    if family in {"canopy_dew_lt0", "canopy_dew_lt1"}:
        return "conservative_stabilize_canopy_dew"
    return "conservative_stabilize"


def _winner_for_resolved_family(
    *,
    record: Mapping[str, Any],
    vector: Mapping[str, Any],
    resolved_family: str,
) -> tuple[str, float]:
    assigned_bonus = bonus_for_family(vector, resolved_family)
    best_id = ""
    best_score = -1e18
    components = record.get("components") or {}
    candidates = record.get("candidates") or {}
    for candidate_id, component in components.items():
        candidate = candidates.get(candidate_id)
        if candidate is None:
            source, template = _parse_candidate_source(candidate_id)
            candidate = {"source_prior": source, "template_name": template, "feasible": False}
        if not candidate.get("feasible", False):
            continue
        score = _num(component.get("final_score"), -1e18)
        if (
            candidate.get("source_prior") == "conservative_prior"
            and candidate.get("template_name") == "conservative_stabilize"
            and resolved_family != "none"
        ):
            score += assigned_bonus
        if score > best_score:
            best_id = candidate_id
            best_score = score
    return best_id, best_score


def annotate_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for record in records:
        enriched = dict(record)
        enriched["resolver"] = resolve_risk_family(enriched)
        annotated.append(enriched)
    return annotated


def mixed_risk_diagnostics(records: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    risk_records = [record for record in records if (record.get("risk_flags") or {}).get("any_risk", False)]
    mixed_records = [
        record
        for record in risk_records
        if len((record.get("resolver") or {}).get("active_risk_families") or []) >= 2
    ]
    single_records = [
        record
        for record in risk_records
        if len((record.get("resolver") or {}).get("active_risk_families") or []) == 1
    ]
    combo_groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in risk_records:
        combo_groups.setdefault(str((record.get("resolver") or {}).get("risk_combo_id") or "none"), []).append(record)
    combo_rows: list[dict[str, Any]] = []
    for combo_id, items in sorted(combo_groups.items()):
        old_families = [str((item.get("resolver") or {}).get("primary_risk_family_old") or "none") for item in items]
        resolved_families = [str((item.get("resolver") or {}).get("resolved_risk_family_new") or "none") for item in items]
        conflict_types = [str((item.get("resolver") or {}).get("conflict_type") or "none") for item in items]
        combo_rows.append(
            {
                "risk_combo_id": combo_id,
                "row_count": len(items),
                "is_mixed_risk": "+" in combo_id,
                "active_family_count": 0 if combo_id == "none" else len(combo_id.split("+")),
                "conflict_type_distribution": _distribution(conflict_types),
                "assigned_primary_family_by_combo": _distribution(old_families),
                "resolved_family_distribution": _distribution(resolved_families),
                "priority_reason_distribution": _distribution(
                    str((item.get("resolver") or {}).get("conflict_priority_reason") or "") for item in items
                ),
            }
        )
    diagnostics = {
        "risk_row_count": len(risk_records),
        "mixed_risk_row_count": len(mixed_records),
        "single_risk_row_count": len(single_records),
        "active_risk_flags_cardinality_distribution": _distribution(
            len(record.get("active_risk_flags") or []) for record in risk_records
        ),
        "active_risk_family_cardinality_distribution": _distribution(
            len((record.get("resolver") or {}).get("active_risk_families") or []) for record in risk_records
        ),
        "mixed_risk_combo_distribution": _distribution(
            (record.get("resolver") or {}).get("risk_combo_id") for record in mixed_records
        ),
        "conflict_type_distribution": _distribution(
            (record.get("resolver") or {}).get("conflict_type") for record in risk_records
        ),
        "resolved_family_distribution": _distribution(
            (record.get("resolver") or {}).get("resolved_risk_family_new") for record in risk_records
        ),
    }
    return diagnostics, combo_rows


def evaluate_resolver_vectors(
    records: Sequence[Mapping[str, Any]],
    vectors: Sequence[Mapping[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    vectors = list(vectors or bonus_vectors())
    risk_records = [record for record in records if (record.get("risk_flags") or {}).get("any_risk", False)]
    selected_mismatch = sum(1 for record in records if record.get("runtime_selected_id") != record.get("zero_bonus_winner_id"))
    curve_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    best_vector: dict[str, Any] = {}
    min_spread_vector: dict[str, Any] = {}

    for vector in vectors:
        vid = vector_id(vector)
        conservative_selected = 0
        conservative_risk_selected = 0
        nonrisk_conservative_selected = 0
        switch_runtime = 0
        switch_zero = 0
        switch_to_conservative = 0
        switch_from_previous = 0
        switch_from_rule = 0
        switch_from_ppo = 0
        switch_nonconservative_to_nonconservative = 0
        family_counts: dict[str, dict[str, int]] = {}
        combo_counts: dict[str, dict[str, int]] = {}

        for record in records:
            resolver = record.get("resolver") or {}
            family = str(resolver.get("resolved_risk_family_new") or "none")
            combo_id = str(resolver.get("risk_combo_id") or "none")
            is_risk = bool((record.get("risk_flags") or {}).get("any_risk", False))
            winner_id, winner_score = _winner_for_resolved_family(record=record, vector=vector, resolved_family=family)
            winner_source = _source(winner_id)
            winner_template = _template(winner_id)
            zero_id = str(record.get("zero_bonus_winner_id") or "")
            zero_source = _source(zero_id)
            runtime_id = str(record.get("runtime_selected_id") or "")
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
                elif winner_source != "conservative_prior" and zero_source != "conservative_prior":
                    switch_nonconservative_to_nonconservative += 1
                if len(switch_rows) < 5000:
                    zero_components = (record.get("components") or {}).get(zero_id, {})
                    conservative_components = (record.get("components") or {}).get(
                        record.get("conservative_candidate_id", ""),
                        {},
                    )
                    zero_minus_conservative = _component_delta(zero_components, conservative_components)
                    switch_rows.append(
                        {
                            "vector_id": vid,
                            **dict(vector),
                            "assigned_bonus": bonus_for_family(vector, family),
                            "step": record.get("step"),
                            "hour_of_day": record.get("hour_of_day"),
                            "risk_combo_id": combo_id,
                            "active_risk_flags": record.get("active_risk_flags"),
                            "primary_risk_family_old": resolver.get("primary_risk_family_old"),
                            "resolved_risk_family_new": family,
                            "conflict_type": resolver.get("conflict_type"),
                            "conflict_priority_reason": resolver.get("conflict_priority_reason"),
                            "runtime_selected_id": runtime_id,
                            "zero_bonus_winner_id": zero_id,
                            "counterfactual_winner_id": winner_id,
                            "counterfactual_winner_score": winner_score,
                            "switch_relative_to_runtime": winner_id != runtime_id,
                            "switch_relative_to_zero_bonus": winner_id != zero_id,
                            "switch_to_conservative": is_conservative and zero_source != "conservative_prior",
                            "energy_proxy_delta_on_switch": zero_minus_conservative.get("projected_energy_proxy", 0.0),
                            "smoothness_delta_on_switch": zero_minus_conservative.get("projected_smoothness", 0.0),
                            "prior_confidence_delta_on_switch": zero_minus_conservative.get("prior_confidence_bonus", 0.0),
                        }
                    )
            if is_risk:
                family_item = family_counts.setdefault(family, {"count": 0, "conservative": 0})
                combo_item = combo_counts.setdefault(combo_id, {"count": 0, "conservative": 0})
                family_item["count"] += 1
                combo_item["count"] += 1
                if is_conservative:
                    family_item["conservative"] += 1
                    combo_item["conservative"] += 1

        family_rates = [
            item["conservative"] / max(item["count"], 1)
            for family, item in family_counts.items()
            if family != "none" and item["count"] > 0
        ]
        mixed_combo_rates = [
            item["conservative"] / max(item["count"], 1)
            for combo_id, item in combo_counts.items()
            if "+" in combo_id and item["count"] > 0
        ]
        risk_spread = max(family_rates) - min(family_rates) if len(family_rates) >= 2 else 0.0
        combo_spread = max(mixed_combo_rates) - min(mixed_combo_rates) if len(mixed_combo_rates) >= 2 else 0.0
        curve = {
            "vector_id": vid,
            **dict(vector),
            "row_count": len(records),
            "risk_row_count": len(risk_records),
            "conservative_selected_count": conservative_selected,
            "conservative_risk_selected_count": conservative_risk_selected,
            "conservative_risk_selected_rate": conservative_risk_selected / max(len(risk_records), 1),
            "nonrisk_conservative_selected_count": nonrisk_conservative_selected,
            "risk_type_switch_spread": risk_spread,
            "mixed_risk_combo_spread": combo_spread,
            "switch_relative_to_runtime_count": switch_runtime,
            "switch_relative_to_zero_bonus_count": switch_zero,
            "switch_to_conservative_count": switch_to_conservative,
            "switch_from_previous_to_conservative_count": switch_from_previous,
            "switch_from_rule_to_conservative_count": switch_from_rule,
            "switch_from_ppo_to_conservative_count": switch_from_ppo,
            "switch_nonconservative_to_nonconservative_count": switch_nonconservative_to_nonconservative,
            "selected_id_vs_zero_bonus_winner_mismatch_count": selected_mismatch,
            "rule_suppression_detected": switch_from_rule > 0,
            "previous_rule_ppo_suppression_detected": conservative_risk_selected / max(len(risk_records), 1) > 0.85,
        }
        curve_rows.append(curve)
        if _is_better_curve(curve, best_vector):
            best_vector = dict(curve)
        if _is_lower_spread_curve(curve, min_spread_vector):
            min_spread_vector = dict(curve)

    diagnostics = {
        "bonus_vector_count": len(vectors),
        "best_balanced_vector": best_vector,
        "min_spread_vector": min_spread_vector,
        "min_spread_vector_conservative_rate": _num(min_spread_vector.get("conservative_risk_selected_rate")),
        "best_balanced_vector_selection_reason": _best_vector_reason(best_vector),
        "why_min_spread_vector_not_selected": _why_min_spread_not_selected(min_spread_vector, best_vector),
        "selected_id_vs_zero_bonus_winner_mismatch_count": selected_mismatch,
    }
    return curve_rows, switch_rows, diagnostics


def _is_better_curve(candidate: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    if not current:
        return True
    candidate_rate = _num(candidate.get("conservative_risk_selected_rate"))
    current_rate = _num(current.get("conservative_risk_selected_rate"))
    candidate_viable = _curve_viable(candidate)
    current_viable = _curve_viable(current)
    if candidate_viable != current_viable:
        return candidate_viable
    return (
        max(_num(candidate.get("risk_type_switch_spread")), _num(candidate.get("mixed_risk_combo_spread"))),
        _num(candidate.get("risk_type_switch_spread")),
        abs(candidate_rate - 0.55),
        -candidate_rate,
    ) < (
        max(_num(current.get("risk_type_switch_spread")), _num(current.get("mixed_risk_combo_spread"))),
        _num(current.get("risk_type_switch_spread")),
        abs(current_rate - 0.55),
        -current_rate,
    )


def _is_lower_spread_curve(candidate: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    if not current:
        return True
    return (
        _num(candidate.get("risk_type_switch_spread")),
        _num(candidate.get("mixed_risk_combo_spread")),
        abs(_num(candidate.get("conservative_risk_selected_rate")) - 0.55),
    ) < (
        _num(current.get("risk_type_switch_spread")),
        _num(current.get("mixed_risk_combo_spread")),
        abs(_num(current.get("conservative_risk_selected_rate")) - 0.55),
    )


def _curve_viable(row: Mapping[str, Any]) -> bool:
    rate = _num(row.get("conservative_risk_selected_rate"))
    return bool(
        0.25 <= rate <= 0.85
        and int(row.get("nonrisk_conservative_selected_count", 0) or 0) == 0
        and not _bool(row.get("rule_suppression_detected"))
        and not _bool(row.get("previous_rule_ppo_suppression_detected"))
    )


def _best_vector_reason(row: Mapping[str, Any]) -> str:
    if not row:
        return "no_candidate_vector"
    if _curve_viable(row):
        return "viable_rate_with_low_combined_family_and_combo_spread"
    return "lowest_available_tradeoff_after_viability_filters"


def _why_min_spread_not_selected(min_row: Mapping[str, Any], best_row: Mapping[str, Any]) -> str:
    if not min_row:
        return "no_min_spread_vector"
    if not best_row or min_row.get("vector_id") == best_row.get("vector_id"):
        return "min_spread_vector_is_best_balanced_vector"
    rate = _num(min_row.get("conservative_risk_selected_rate"))
    if not 0.25 <= rate <= 0.85:
        return "min_spread_vector_conservative_risk_selected_rate_outside_viable_range"
    if int(min_row.get("nonrisk_conservative_selected_count", 0) or 0) > 0:
        return "min_spread_vector_selects_conservative_on_nonrisk_rows"
    if _bool(min_row.get("rule_suppression_detected")):
        return "min_spread_vector_suppresses_rule_prior"
    if _bool(min_row.get("previous_rule_ppo_suppression_detected")):
        return "min_spread_vector_over_selects_conservative"
    if _num(min_row.get("mixed_risk_combo_spread")) > _num(best_row.get("mixed_risk_combo_spread")):
        return "min_spread_vector_has_higher_mixed_risk_combo_spread"
    return "best_balanced_vector_prefers_rate_near_target"


def min_spread_explainability_rows(diagnostics: Mapping[str, Any]) -> list[dict[str, Any]]:
    best = diagnostics.get("best_balanced_vector") or {}
    minimum = diagnostics.get("min_spread_vector") or {}
    rows = []
    for label, row in (("best_balanced_vector", best), ("min_spread_vector", minimum)):
        if not row:
            continue
        rows.append(
            {
                "vector_role": label,
                "vector_id": row.get("vector_id"),
                "conservative_risk_selected_rate": row.get("conservative_risk_selected_rate"),
                "risk_type_switch_spread": row.get("risk_type_switch_spread"),
                "mixed_risk_combo_spread": row.get("mixed_risk_combo_spread"),
                "nonrisk_conservative_selected_count": row.get("nonrisk_conservative_selected_count"),
                "rule_suppression_detected": row.get("rule_suppression_detected"),
                "previous_rule_ppo_suppression_detected": row.get("previous_rule_ppo_suppression_detected"),
                "selection_reason": diagnostics.get("best_balanced_vector_selection_reason") if label == "best_balanced_vector" else "",
                "why_not_selected": diagnostics.get("why_min_spread_vector_not_selected") if label == "min_spread_vector" else "",
            }
        )
    return rows


def _component_dominant_gap_from_v8173(v8173_json: Mapping[str, Any]) -> str:
    component_rows = ((v8173_json.get("diagnostics") or {}).get("zero_minus_conservative_component_quantiles") or [])
    return _dominant_component(component_rows)


def _template_variant_material(v8173_json: Mapping[str, Any]) -> bool:
    return bool(
        (((v8173_json.get("diagnostics") or {}).get("template_variant") or {}).get("template_variant_material_improvement"))
    )


def _route_next_action(
    *,
    boundary: Mapping[str, Any],
    schema_issues: Sequence[str],
    diagnostics: Mapping[str, Any],
    v8173_json: Mapping[str, Any],
) -> tuple[str, list[str]]:
    if not _boundary_clean(boundary):
        return "stop_and_repair_cstcc_shadow_boundary", ["boundary_or_candidate_health_failure"]
    if schema_issues:
        return "v8173_schema_or_golden_trace_repair_plan", list(schema_issues)
    best = diagnostics.get("best_balanced_vector") or {}
    if (
        _num(best.get("risk_type_switch_spread"), 999.0) <= 0.30
        and _num(best.get("mixed_risk_combo_spread"), 999.0) <= 0.30
        and int(best.get("nonrisk_conservative_selected_count", 0) or 0) == 0
        and not _bool(best.get("rule_suppression_detected"))
        and not _bool(best.get("previous_rule_ppo_suppression_detected"))
    ):
        return "v8173_2_opt_in_conflict_resolver_bonus_shadow_trace_plan", ["conflict_resolver_bonus_vector_balanced"]
    if _template_variant_material(v8173_json):
        return "v8174_risk_specific_template_design_plan", ["risk_specific_template_signal_material"]
    if _component_dominant_gap_from_v8173(v8173_json) in {"projected_energy_proxy", "raw_energy_proxy"} and _num(
        best.get("risk_type_switch_spread"), 0.0
    ) > 0.30:
        return "v8173_energy_proxy_reweight_design_plan", ["spread_still_high_and_energy_proxy_dominant"]
    if _num(best.get("mixed_risk_combo_spread"), 0.0) > 0.30 or _num(best.get("risk_type_switch_spread"), 0.0) > 0.30:
        return "v8173_2_conflict_resolver_policy_iteration", ["mixed_risk_combo_or_family_spread_still_high"]
    return "v8173_2_conflict_resolver_policy_iteration", ["conflict_resolver_needs_additional_diagnostics"]


def _golden_issues(v8173_json: Mapping[str, Any]) -> list[str]:
    if not v8173_json:
        return ["v8173_json_missing"]
    diagnostics = v8173_json.get("diagnostics") or {}
    best = diagnostics.get("best_balanced_vector") or {}
    template = (diagnostics.get("template_variant") or {}).get("best_template_variant") or {}
    issues: list[str] = []
    expected = {
        "row_count": 720,
        "risk_row_count": 126,
        "bonus_vector_count": 576,
        "vector_id": "dry=0.06|dry_vent=0.06|dew=0.10|canopy=0.12",
        "conservative_risk_selected_count": 106,
        "selected_id_vs_zero_bonus_winner_mismatch_count": 248,
        "template_name": "conservative_stabilize_dew",
        "next_action": "v8173_mixed_risk_conflict_resolver_design_plan",
    }
    for key in ("row_count", "risk_row_count", "bonus_vector_count"):
        if int(_num(diagnostics.get(key))) != expected[key]:
            issues.append(f"golden_mismatch:{key}")
    if best.get("vector_id") != expected["vector_id"]:
        issues.append("golden_mismatch:best_vector")
    if int(_num(best.get("conservative_risk_selected_count"))) != expected["conservative_risk_selected_count"]:
        issues.append("golden_mismatch:conservative_risk_selected_count")
    if abs(_num(best.get("risk_type_switch_spread")) - 0.45) > 1e-9:
        issues.append("golden_mismatch:risk_type_switch_spread")
    if int(_num(best.get("selected_id_vs_zero_bonus_winner_mismatch_count"))) != expected[
        "selected_id_vs_zero_bonus_winner_mismatch_count"
    ]:
        issues.append("golden_mismatch:selected_id_vs_zero_bonus_winner_mismatch_count")
    if template.get("template_name") != expected["template_name"]:
        issues.append("golden_mismatch:best_template_variant")
    if _template_variant_material(v8173_json):
        issues.append("golden_mismatch:template_variant_material_improvement")
    if v8173_json.get("next_action") != expected["next_action"]:
        issues.append("golden_mismatch:v8173_next_action")
    return issues


def build_report(
    *,
    trace_csv_rows: list[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    jsonl_rows: list[Mapping[str, Any]],
    v8173_json: Mapping[str, Any],
    vectors: Sequence[Mapping[str, float]] | None = None,
    enforce_golden: bool = True,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    boundary = _boundary_summary(summary_json, jsonl_rows)
    schema_issues = _missing_schema_fields(jsonl_rows)
    step_records, reconstruction_issues = _build_step_records(trace_csv_rows=trace_csv_rows, jsonl_rows=jsonl_rows)
    records = annotate_records(step_records)
    mixed_diagnostics, combo_rows = mixed_risk_diagnostics(records)
    curve_rows, switch_rows, resolver_diagnostics = evaluate_resolver_vectors(records, vectors=vectors)
    schema_issues = sorted(set(schema_issues + reconstruction_issues + (_golden_issues(v8173_json) if enforce_golden else [])))
    diagnostics = {
        **mixed_diagnostics,
        **resolver_diagnostics,
        "v8173_reference_next_action": v8173_json.get("next_action"),
        "v8173_reference_best_vector": ((v8173_json.get("diagnostics") or {}).get("best_balanced_vector") or {}).get("vector_id"),
        "v8173_reference_risk_type_switch_spread": ((v8173_json.get("diagnostics") or {}).get("best_balanced_vector") or {}).get(
            "risk_type_switch_spread"
        ),
        "v8173_reference_template_variant_material_improvement": _template_variant_material(v8173_json),
        "dominant_gap_from_v8173": _component_dominant_gap_from_v8173(v8173_json),
    }
    next_action, readiness_reasons = _route_next_action(
        boundary=boundary,
        schema_issues=schema_issues,
        diagnostics=diagnostics,
        v8173_json=v8173_json,
    )
    best = diagnostics.get("best_balanced_vector") or {}
    report = {
        "schema_version": "cstcc_v8173_mixed_risk_conflict_resolver_design_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_level": "existing-trace counterfactual scorer audit",
        "resolver_policy": RESOLVER_POLICY,
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
        },
        "boundary_runtime_safety": boundary,
        "schema_issues": schema_issues,
        "counterfactual_definition": {
            "resolver_is_evaluator_only": True,
            "global_bonus_used": False,
            "bonus_is_non_additive": True,
            "resolver_changes_runtime_primary_family": False,
            "resolver_changes_runtime_scorer": False,
            "runtime_final_action_changed": False,
        },
        "diagnostics": diagnostics,
        "acceptance": {
            "boundary_clean": _boundary_clean(boundary),
            "nonrisk_conservative_selected_count_eq_0": int(best.get("nonrisk_conservative_selected_count", 0) or 0) == 0,
            "risk_type_switch_spread_le_0_30": _num(best.get("risk_type_switch_spread"), 999.0) <= 0.30,
            "mixed_risk_combo_spread_le_0_30": _num(best.get("mixed_risk_combo_spread"), 999.0) <= 0.30,
            "selected_id_vs_zero_bonus_winner_mismatch_count_reported": "selected_id_vs_zero_bonus_winner_mismatch_count"
            in best,
            "rule_suppression_detected": bool(best.get("rule_suppression_detected", False)),
            "previous_rule_ppo_suppression_detected": bool(best.get("previous_rule_ppo_suppression_detected", False)),
        },
        "next_action": next_action,
        "readiness_reasons": readiness_reasons,
        "notes": [
            "v8173.1 is offline counterfactual resolver design only.",
            "The resolver does not change runtime scorer, template pool, final actions, or evidence level.",
            "It does not validate reward, profit, safety improvement, or closed-loop controller quality.",
        ],
    }
    return report, {
        "mixed_risk_combo_diagnostics": combo_rows,
        "conflict_resolver_curve": curve_rows,
        "min_spread_vector_explainability": min_spread_explainability_rows(diagnostics),
        "resolved_family_switch_explainability": switch_rows,
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    boundary = report.get("boundary_runtime_safety", {})
    diagnostics = report.get("diagnostics", {})
    best = diagnostics.get("best_balanced_vector", {})
    minimum = diagnostics.get("min_spread_vector", {})
    lines = [
        "# C-STCC v8173.1 Mixed-Risk Conflict Resolver Design",
        "",
        "Existing-trace counterfactual audit only. No online LLM, no rollout, no runtime scorer/template change.",
        "",
        "## Boundary",
        "",
        f"- Audit success rate: {_num(boundary.get('audit_success_rate')):.3f}",
        f"- Final-action invariant rate: {_num(boundary.get('final_action_invariant_rate')):.3f}",
        f"- Online LLM / rollout / real projection steps: {boundary.get('online_llm_called_steps', 0)} / {boundary.get('predictive_rollout_executed_steps', 0)} / {boundary.get('real_tomato_safety_projection_steps', 0)}",
        f"- Final action changed steps: {boundary.get('final_action_changed_steps', 0)}",
        "",
        "## Mixed-Risk Diagnostics",
        "",
        f"- Risk rows: {diagnostics.get('risk_row_count', 0)}",
        f"- Mixed / single risk rows: {diagnostics.get('mixed_risk_row_count', 0)} / {diagnostics.get('single_risk_row_count', 0)}",
        f"- Mixed combo distribution: {_compact_json(diagnostics.get('mixed_risk_combo_distribution', {}))}",
        f"- Conflict type distribution: {_compact_json(diagnostics.get('conflict_type_distribution', {}))}",
        "",
        "## Resolver Counterfactual",
        "",
        f"- Best vector: {best.get('vector_id', 'none')}",
        f"- Conservative risk selected rate: {_num(best.get('conservative_risk_selected_rate')):.3f}",
        f"- Risk-type / mixed-combo spread: {_num(best.get('risk_type_switch_spread')):.3f} / {_num(best.get('mixed_risk_combo_spread')):.3f}",
        f"- Min-spread vector: {minimum.get('vector_id', 'none')}",
        f"- Why min-spread not selected: {diagnostics.get('why_min_spread_vector_not_selected', 'none')}",
        f"- Selected id vs zero-bonus mismatch count: {diagnostics.get('selected_id_vs_zero_bonus_winner_mismatch_count', 0)}",
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
    base = "mixed_risk_conflict_resolver_design"
    paths = {
        "json": prefix.with_suffix(".json"),
        "md": prefix.with_suffix(".md"),
        "mixed_risk_combo_diagnostics_csv": (parent / stem.replace(base, "mixed_risk_combo_diagnostics")).with_suffix(".csv"),
        "conflict_resolver_curve_csv": (parent / stem.replace(base, "conflict_resolver_curve")).with_suffix(".csv"),
        "min_spread_vector_explainability_csv": (
            parent / stem.replace(base, "min_spread_vector_explainability")
        ).with_suffix(".csv"),
        "resolved_family_switch_explainability_csv": (
            parent / stem.replace(base, "resolved_family_switch_explainability")
        ).with_suffix(".csv"),
    }
    paths["json"].write_text(_compact_json(report) + "\n", encoding="utf-8")
    paths["md"].write_text(build_markdown(report), encoding="utf-8")
    _write_csv(paths["mixed_risk_combo_diagnostics_csv"], tables.get("mixed_risk_combo_diagnostics", []))
    _write_csv(paths["conflict_resolver_curve_csv"], tables.get("conflict_resolver_curve", []))
    _write_csv(paths["min_spread_vector_explainability_csv"], tables.get("min_spread_vector_explainability", []))
    _write_csv(paths["resolved_family_switch_explainability_csv"], tables.get("resolved_family_switch_explainability", []))
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v8173.1 mixed-risk conflict resolver design audit.")
    parser.add_argument("--trace-csv", type=str, default=str(DEFAULT_TRACE_CSV))
    parser.add_argument("--summary-json", type=str, default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--jsonl-root", type=str, default=str(DEFAULT_JSONL_ROOT))
    parser.add_argument("--v8173-json", type=str, default=str(DEFAULT_V8173_JSON))
    parser.add_argument("--output-prefix", type=str, default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--no-golden-check", action="store_true")
    args = parser.parse_args()

    report, tables = build_report(
        trace_csv_rows=_load_csv(args.trace_csv),
        summary_json=_load_json(args.summary_json),
        jsonl_rows=_load_jsonl_rows(args.jsonl_root),
        v8173_json=_load_json(args.v8173_json),
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
