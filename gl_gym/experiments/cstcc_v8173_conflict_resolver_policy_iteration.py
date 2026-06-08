"""v8173.2 offline C-STCC mixed-risk resolver policy iteration audit.

This stage compares evaluator-only conflict resolver policies on existing
v8171/v8173 H720 traces. It does not change runtime scorer behavior, template
pools, final actions, online LLM usage, predictive rollout, or Tomato Safety.
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

from gl_gym.experiments.cstcc_v8173_mixed_risk_conflict_resolver_design import (
    DEFAULT_OUTPUT_PREFIX as DEFAULT_V81731_OUTPUT_PREFIX,
    DEFAULT_V8173_JSON,
    RESOLVER_POLICY as BASELINE_POLICY,
    _curve_viable,
    _golden_issues,
    _template_variant_material,
    active_risk_families,
    conflict_type,
    resolve_risk_family,
    risk_combo_id,
)
from gl_gym.experiments.cstcc_v8173_risk_type_specific_bonus_template_counterfactual import (
    COMPONENT_KEYS,
    DEFAULT_JSONL_ROOT,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TRACE_CSV,
    RISK_FAMILIES,
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
    _parse_candidate_source,
    _resolve,
    _source,
    _template,
    _write_csv,
    bonus_for_family,
    bonus_vectors,
    vector_id,
)


DEFAULT_V81731_JSON = DEFAULT_V81731_OUTPUT_PREFIX.with_suffix(".json")
DEFAULT_OUTPUT_PREFIX = Path(
    "gl_gym/result/diagnostics/cstcc_v8173_conflict_resolver_policy_iteration_20260606_y2020_d240_s42_n720"
)

POLICIES = (
    "severity_aware_non_additive_v1",
    "severity_aware_non_additive_v2",
    "family_target_rate_balanced_v1",
    "canopy_dew_first_v1",
    "dew_before_canopy_lt1_v1",
    "dry_vent_strong_only_v1",
    "mixed_combo_specific_v1",
)
TARGET_RATE_LOW = 0.35
TARGET_RATE_HIGH = 0.80
BOUNDARY_FALSE_COUNTS = (
    "online_llm_called_steps",
    "predictive_rollout_executed_steps",
    "real_tomato_safety_projection_steps",
    "final_action_changed_steps",
)


def _ventilation(record: Mapping[str, Any]) -> float:
    merged = record.get("merged") or {}
    action = record.get("current_runtime_final_action") or {}
    return _num(merged.get("u_ventilation"), _num(action.get("u_ventilation")))


def resolve_policy_family(record: Mapping[str, Any], policy: str) -> dict[str, Any]:
    flags = record.get("risk_flags") or {}
    active = active_risk_families(flags)
    combo = risk_combo_id(flags)
    conflict = conflict_type(flags)
    ventilation = _ventilation(record)
    high_vpd = _bool(flags.get("high_vpd")) or _bool(flags.get("dry_or_high_vpd_risk"))

    if policy in {BASELINE_POLICY, "family_target_rate_balanced_v1"}:
        resolved = resolve_risk_family(record)
        resolved["resolver_policy"] = policy
        return resolved

    if not active:
        family, reason = "none", "no_active_risk"
    elif policy == "severity_aware_non_additive_v2":
        if _bool(flags.get("canopy_dew_margin_lt0")):
            family, reason = "canopy_dew_lt0", "v2_severe_canopy_dew_lt0_priority"
        elif _bool(flags.get("dry_vent_risk")) and (_bool(flags.get("high_vpd")) or ventilation > 0.70):
            family, reason = "dry_vent", "v2_dry_vent_only_when_high_vpd_or_strong_ventilation"
        elif _bool(flags.get("humidity_or_dew_risk")) or _bool(flags.get("dew_risk")):
            family, reason = "dew_or_humidity", "v2_dew_or_humidity_before_dry"
        elif _bool(flags.get("canopy_dew_margin_lt1")):
            family, reason = "canopy_dew_lt1", "v2_moderate_canopy_dew_after_dew"
        elif high_vpd:
            family, reason = "dry_or_high_vpd", "v2_dry_or_high_vpd_priority"
        else:
            family, reason = "none", "v2_unassigned"
    elif policy == "canopy_dew_first_v1":
        if _bool(flags.get("canopy_dew_margin_lt0")):
            family, reason = "canopy_dew_lt0", "canopy_dew_first_severe"
        elif _bool(flags.get("canopy_dew_margin_lt1")):
            family, reason = "canopy_dew_lt1", "canopy_dew_first_moderate"
        elif _bool(flags.get("dry_vent_risk")) and (_bool(flags.get("high_vpd")) or ventilation > 0.70):
            family, reason = "dry_vent", "canopy_dew_first_dry_vent_after_canopy"
        elif _bool(flags.get("humidity_or_dew_risk")) or _bool(flags.get("dew_risk")):
            family, reason = "dew_or_humidity", "canopy_dew_first_dew"
        elif high_vpd:
            family, reason = "dry_or_high_vpd", "canopy_dew_first_dry"
        else:
            family, reason = "none", "canopy_dew_first_unassigned"
    elif policy == "dew_before_canopy_lt1_v1":
        if _bool(flags.get("canopy_dew_margin_lt0")):
            family, reason = "canopy_dew_lt0", "dew_before_canopy_lt1_severe_canopy_first"
        elif _bool(flags.get("humidity_or_dew_risk")) or _bool(flags.get("dew_risk")):
            family, reason = "dew_or_humidity", "dew_before_canopy_lt1_dew_priority"
        elif _bool(flags.get("canopy_dew_margin_lt1")):
            family, reason = "canopy_dew_lt1", "dew_before_canopy_lt1_moderate_canopy"
        elif _bool(flags.get("dry_vent_risk")) and (_bool(flags.get("high_vpd")) or ventilation > 0.70):
            family, reason = "dry_vent", "dew_before_canopy_lt1_dry_vent"
        elif high_vpd:
            family, reason = "dry_or_high_vpd", "dew_before_canopy_lt1_dry"
        else:
            family, reason = "none", "dew_before_canopy_lt1_unassigned"
    elif policy == "dry_vent_strong_only_v1":
        dry_vent_strong = _bool(flags.get("dry_vent_risk")) and (_bool(flags.get("high_vpd")) or ventilation > 0.70)
        if _bool(flags.get("canopy_dew_margin_lt0")):
            family, reason = "canopy_dew_lt0", "dry_vent_strong_severe_canopy_first"
        elif dry_vent_strong:
            family, reason = "dry_vent", "dry_vent_strong_only_active"
        elif _bool(flags.get("canopy_dew_margin_lt1")):
            family, reason = "canopy_dew_lt1", "dry_vent_strong_canopy_lt1"
        elif _bool(flags.get("humidity_or_dew_risk")) or _bool(flags.get("dew_risk")):
            family, reason = "dew_or_humidity", "dry_vent_strong_dew"
        elif _bool(flags.get("dry_vent_risk")) or high_vpd:
            family, reason = "dry_or_high_vpd", "dry_vent_strong_fallback_to_dry_family"
        else:
            family, reason = "none", "dry_vent_strong_unassigned"
    elif policy == "mixed_combo_specific_v1":
        combo_map = {
            "canopy_dew_lt1+dew_or_humidity": ("dew_or_humidity", "combo_specific_lt1_dew_prefers_dew"),
            "dry_vent+dry_or_high_vpd": ("dry_vent", "combo_specific_dry_vent_guard"),
            "canopy_dew_lt0+dew_or_humidity+dry_or_high_vpd": (
                "canopy_dew_lt0",
                "combo_specific_severe_canopy_priority",
            ),
            "canopy_dew_lt0+dew_or_humidity": ("canopy_dew_lt0", "combo_specific_severe_canopy_dew"),
            "canopy_dew_lt1+dew_or_humidity+dry_or_high_vpd": (
                "dew_or_humidity",
                "combo_specific_lt1_dew_dry_prefers_dew",
            ),
            "dry_vent+canopy_dew_lt0+dew_or_humidity+dry_or_high_vpd": (
                "canopy_dew_lt0",
                "combo_specific_dry_vent_severe_canopy_priority",
            ),
            "dry_vent+canopy_dew_lt1+dew_or_humidity+dry_or_high_vpd": (
                "dry_vent",
                "combo_specific_dry_vent_lt1_guard",
            ),
        }
        if combo in combo_map:
            family, reason = combo_map[combo]
        else:
            fallback = resolve_risk_family(record)
            family, reason = fallback["resolved_risk_family_new"], "combo_specific_fallback_to_v1"
    else:
        raise ValueError(f"unknown resolver policy: {policy}")

    return {
        "resolver_policy": policy,
        "risk_combo_id": combo,
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


def annotate_records_for_policy(records: Sequence[Mapping[str, Any]], policy: str) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for record in records:
        enriched = dict(record)
        enriched["resolver"] = resolve_policy_family(enriched, policy)
        annotated.append(enriched)
    return annotated


def _winner_for_policy_record(record: Mapping[str, Any], vector: Mapping[str, Any]) -> tuple[str, float]:
    family = str((record.get("resolver") or {}).get("resolved_risk_family_new") or "none")
    assigned_bonus = bonus_for_family(vector, family)
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
            and family != "none"
        ):
            score += assigned_bonus
        if score > best_score:
            best_id = candidate_id
            best_score = score
    return best_id, best_score


def _target_gap(rate: float) -> float:
    if rate < TARGET_RATE_LOW:
        return TARGET_RATE_LOW - rate
    if rate > TARGET_RATE_HIGH:
        return rate - TARGET_RATE_HIGH
    return 0.0


def _target_label(rate: float) -> str:
    if rate < TARGET_RATE_LOW:
        return "under_target"
    if rate > TARGET_RATE_HIGH:
        return "over_target"
    return "within_target"


def evaluate_policy_vectors(
    records: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    vectors: Sequence[Mapping[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    vectors = list(vectors or bonus_vectors())
    annotated = annotate_records_for_policy(records, policy)
    risk_records = [record for record in annotated if (record.get("risk_flags") or {}).get("any_risk", False)]
    selected_mismatch = sum(1 for record in annotated if record.get("runtime_selected_id") != record.get("zero_bonus_winner_id"))
    curve_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []
    best_vector: dict[str, Any] = {}
    min_spread_vector: dict[str, Any] = {}

    for vector in vectors:
        vid = vector_id(vector)
        conservative_selected = 0
        conservative_risk_selected = 0
        nonrisk_conservative_selected = 0
        switch_zero = 0
        switch_runtime = 0
        switch_to_conservative = 0
        switch_from_previous = 0
        switch_from_rule = 0
        switch_from_ppo = 0
        family_counts: dict[str, dict[str, int]] = {}
        combo_counts: dict[str, dict[str, int]] = {}
        single_counts: dict[str, dict[str, int]] = {}
        mixed_counts: dict[str, dict[str, int]] = {}

        for record in annotated:
            resolver = record.get("resolver") or {}
            family = str(resolver.get("resolved_risk_family_new") or "none")
            combo_id = str(resolver.get("risk_combo_id") or "none")
            is_mixed = "+" in combo_id
            is_risk = bool((record.get("risk_flags") or {}).get("any_risk", False))
            winner_id, _winner_score = _winner_for_policy_record(record, vector)
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
        risk_type_spread = _spread(family_rates)
        mixed_combo_spread = _spread(combo_rates)
        single_spread = _spread(single_rates)
        mixed_spread = _spread(mixed_rates)
        family_target_penalty = sum(_target_gap(rate) for rate in family_rates)
        extreme_family_count = sum(1 for rate in family_rates if rate <= 0.0 or rate >= 1.0)
        row = {
            "policy": policy,
            "vector_id": vid,
            **dict(vector),
            "row_count": len(annotated),
            "risk_row_count": len(risk_records),
            "conservative_selected_count": conservative_selected,
            "conservative_risk_selected_count": conservative_risk_selected,
            "conservative_risk_selected_rate": conservative_risk_selected / max(len(risk_records), 1),
            "nonrisk_conservative_selected_count": nonrisk_conservative_selected,
            "risk_type_switch_spread": risk_type_spread,
            "mixed_risk_combo_spread": mixed_combo_spread,
            "single_risk_spread": single_spread,
            "mixed_risk_spread": mixed_spread,
            "family_target_penalty": family_target_penalty,
            "family_extreme_rate_count": extreme_family_count,
            "switch_relative_to_runtime_count": switch_runtime,
            "switch_relative_to_zero_bonus_count": switch_zero,
            "switch_to_conservative_count": switch_to_conservative,
            "switch_from_previous_to_conservative_count": switch_from_previous,
            "switch_from_rule_to_conservative_count": switch_from_rule,
            "switch_from_ppo_to_conservative_count": switch_from_ppo,
            "selected_id_vs_zero_bonus_winner_mismatch_count": selected_mismatch,
            "rule_suppression_detected": switch_from_rule > 0,
            "previous_rule_ppo_suppression_detected": conservative_risk_selected / max(len(risk_records), 1) > 0.85,
        }
        curve_rows.append(row)
        family_rows.extend(_family_rate_rows(policy, vid, vector, family_counts))
        combo_rows.extend(_combo_rate_rows(policy, vid, vector, combo_counts))
        if _is_better_curve_for_policy(row, best_vector, policy):
            best_vector = dict(row)
        if _is_min_spread_curve(row, min_spread_vector):
            min_spread_vector = dict(row)

    diagnostics = {
        "policy": policy,
        "best_vector": best_vector,
        "min_spread_vector": min_spread_vector,
        "why_min_spread_vector_not_selected": _why_min_spread_not_selected(min_spread_vector, best_vector),
        "best_vector_selection_reason": _best_vector_reason(best_vector, policy),
        "selected_id_vs_zero_bonus_winner_mismatch_count": selected_mismatch,
    }
    return curve_rows, family_rows, combo_rows, diagnostics


def _rate(item: Mapping[str, Any]) -> float:
    return int(item.get("conservative", 0) or 0) / max(int(item.get("count", 0) or 0), 1)


def _spread(values: Sequence[float]) -> float:
    return max(values) - min(values) if len(values) >= 2 else 0.0


def _family_rate_rows(
    policy: str,
    vid: str,
    vector: Mapping[str, Any],
    family_counts: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, item in sorted(family_counts.items()):
        count = int(item.get("count", 0) or 0)
        selected = int(item.get("conservative", 0) or 0)
        rate = selected / max(count, 1)
        rows.append(
            {
                "policy": policy,
                "vector_id": vid,
                **dict(vector),
                "resolved_risk_family": family,
                "family_row_count": count,
                "family_conservative_selected_count": selected,
                "family_conservative_selected_rate": rate,
                "family_gap_to_target_rate": _target_gap(rate),
                "family_target_label": _target_label(rate),
            }
        )
    return rows


def _combo_rate_rows(
    policy: str,
    vid: str,
    vector: Mapping[str, Any],
    combo_counts: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for combo_id, item in sorted(combo_counts.items()):
        count = int(item.get("count", 0) or 0)
        selected = int(item.get("conservative", 0) or 0)
        rate = selected / max(count, 1)
        rows.append(
            {
                "policy": policy,
                "vector_id": vid,
                **dict(vector),
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


def _curve_accepts(row: Mapping[str, Any]) -> bool:
    rate = _num(row.get("conservative_risk_selected_rate"))
    return bool(
        _num(row.get("risk_type_switch_spread"), 999.0) <= 0.30
        and _num(row.get("mixed_risk_combo_spread"), 999.0) <= 0.30
        and _num(row.get("single_risk_spread"), 999.0) <= 0.30
        and TARGET_RATE_LOW <= rate <= TARGET_RATE_HIGH
        and int(row.get("nonrisk_conservative_selected_count", 0) or 0) == 0
        and int(row.get("family_extreme_rate_count", 0) or 0) == 0
        and not _bool(row.get("rule_suppression_detected"))
        and not _bool(row.get("previous_rule_ppo_suppression_detected"))
    )


def _is_better_curve_for_policy(candidate: Mapping[str, Any], current: Mapping[str, Any], policy: str) -> bool:
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
    if policy == "family_target_rate_balanced_v1":
        return (
            _num(candidate.get("family_target_penalty")),
            _num(candidate.get("family_extreme_rate_count")),
            max(
                _num(candidate.get("risk_type_switch_spread")),
                _num(candidate.get("mixed_risk_combo_spread")),
                _num(candidate.get("single_risk_spread")),
            ),
            abs(candidate_rate - 0.55),
        ) < (
            _num(current.get("family_target_penalty")),
            _num(current.get("family_extreme_rate_count")),
            max(
                _num(current.get("risk_type_switch_spread")),
                _num(current.get("mixed_risk_combo_spread")),
                _num(current.get("single_risk_spread")),
            ),
            abs(current_rate - 0.55),
        )
    return (
        _num(candidate.get("family_target_penalty")),
        _num(candidate.get("family_extreme_rate_count")),
        max(
            _num(candidate.get("risk_type_switch_spread")),
            _num(candidate.get("mixed_risk_combo_spread")),
            _num(candidate.get("single_risk_spread")),
        ),
        abs(candidate_rate - 0.55),
        -candidate_rate,
    ) < (
        _num(current.get("family_target_penalty")),
        _num(current.get("family_extreme_rate_count")),
        max(
            _num(current.get("risk_type_switch_spread")),
            _num(current.get("mixed_risk_combo_spread")),
            _num(current.get("single_risk_spread")),
        ),
        abs(current_rate - 0.55),
        -current_rate,
    )


def _is_min_spread_curve(candidate: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    if not current:
        return True
    return (
        _num(candidate.get("risk_type_switch_spread")),
        _num(candidate.get("mixed_risk_combo_spread")),
        _num(candidate.get("single_risk_spread")),
        abs(_num(candidate.get("conservative_risk_selected_rate")) - 0.55),
    ) < (
        _num(current.get("risk_type_switch_spread")),
        _num(current.get("mixed_risk_combo_spread")),
        _num(current.get("single_risk_spread")),
        abs(_num(current.get("conservative_risk_selected_rate")) - 0.55),
    )


def _best_vector_reason(row: Mapping[str, Any], policy: str) -> str:
    if not row:
        return "no_candidate_vector"
    if _curve_accepts(row):
        return "passes_all_policy_iteration_gates"
    if policy == "family_target_rate_balanced_v1":
        return "minimized_family_target_penalty_before_spread_tradeoff"
    return "lowest_combined_spread_with_viable_rate_tradeoff"


def _why_min_spread_not_selected(min_row: Mapping[str, Any], best_row: Mapping[str, Any]) -> str:
    if not min_row:
        return "no_min_spread_vector"
    if not best_row or min_row.get("vector_id") == best_row.get("vector_id"):
        return "min_spread_vector_is_best_vector"
    rate = _num(min_row.get("conservative_risk_selected_rate"))
    if not TARGET_RATE_LOW <= rate <= TARGET_RATE_HIGH:
        return "min_spread_vector_conservative_rate_outside_target_band"
    if int(min_row.get("nonrisk_conservative_selected_count", 0) or 0) > 0:
        return "min_spread_vector_selects_conservative_on_nonrisk_rows"
    if int(min_row.get("family_extreme_rate_count", 0) or 0) > 0:
        return "min_spread_vector_has_family_extreme_rate"
    if _bool(min_row.get("rule_suppression_detected")):
        return "min_spread_vector_suppresses_rule_prior"
    if _bool(min_row.get("previous_rule_ppo_suppression_detected")):
        return "min_spread_vector_over_selects_conservative"
    return "best_vector_prefers_rate_and_family_target_balance"


def component_gap_by_family(records: Sequence[Mapping[str, Any]], policies: Sequence[str] = POLICIES) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in policies:
        annotated = annotate_records_for_policy(records, policy)
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for record in annotated:
            if not (record.get("risk_flags") or {}).get("any_risk", False):
                continue
            family = str((record.get("resolver") or {}).get("resolved_risk_family_new") or "none")
            grouped.setdefault(family, []).append(record)
        for family, items in sorted(grouped.items()):
            for component in COMPONENT_KEYS:
                values = [_num((item.get("zero_minus_conservative") or {}).get(component)) for item in items]
                rows.append(
                    {
                        "policy": policy,
                        "resolved_risk_family": family,
                        "component": component,
                        "row_count": len(items),
                        "mean": float(mean(values)) if values else 0.0,
                        "p50": _quantile(values, 0.50),
                        "p75": _quantile(values, 0.75),
                        "p90": _quantile(values, 0.90),
                        "p95": _quantile(values, 0.95),
                        "max": max(values) if values else 0.0,
                    }
                )
    return rows


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


def _best_policy_row(policy_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not policy_summaries:
        return {}
    return min(
        (dict(row) for row in policy_summaries),
        key=lambda row: (
            not _curve_accepts(row),
            not (TARGET_RATE_LOW <= _num(row.get("conservative_risk_selected_rate")) <= TARGET_RATE_HIGH),
            _num(row.get("family_target_penalty")),
            _num(row.get("family_extreme_rate_count")),
            max(
                _num(row.get("risk_type_switch_spread")),
                _num(row.get("mixed_risk_combo_spread")),
                _num(row.get("single_risk_spread")),
            ),
            abs(_num(row.get("conservative_risk_selected_rate")) - 0.55),
        ),
    )


def _dominant_component_for_policy(component_rows: Sequence[Mapping[str, Any]], policy: str) -> str:
    rows = [
        row
        for row in component_rows
        if row.get("policy") == policy and row.get("component") != "final_score"
    ]
    return _dominant_component(rows)


def _family_extreme_explained(best_family_rows: Sequence[Mapping[str, Any]]) -> bool:
    return not any(
        _num(row.get("family_conservative_selected_rate")) in {0.0, 1.0}
        and int(row.get("family_row_count", 0) or 0) >= 8
        for row in best_family_rows
    )


def _route_next_action(
    *,
    boundary: Mapping[str, Any],
    schema_issues: Sequence[str],
    best_policy: Mapping[str, Any],
    best_family_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    if not _boundary_clean(boundary):
        return "stop_and_repair_cstcc_shadow_boundary", ["boundary_or_candidate_health_failure"]
    if schema_issues:
        return "v8173_schema_or_golden_trace_repair_plan", list(schema_issues)
    if _curve_accepts(best_policy) and _family_extreme_explained(best_family_rows):
        return "v8173_3_opt_in_policy_resolver_bonus_shadow_trace_plan", ["policy_iteration_vector_passes_all_gates"]
    dominant = _dominant_component_for_policy(component_rows, str(best_policy.get("policy") or ""))
    if _num(best_policy.get("risk_type_switch_spread"), 0.0) > 0.30 and dominant in {
        "projected_energy_proxy",
        "raw_energy_proxy",
    }:
        return "v8173_energy_proxy_reweight_design_plan", ["family_spread_high_and_energy_proxy_dominant"]
    if dominant in {"base_final_score", "projected_smoothness", "raw_smoothness"}:
        return "v8174_risk_specific_template_design_plan", [f"family_spread_high_and_{dominant}_dominant"]
    if _num(best_policy.get("mixed_risk_combo_spread"), 0.0) > 0.30 and _num(
        best_policy.get("risk_type_switch_spread"), 0.0
    ) <= 0.45:
        return "v8173_3_combo_specific_policy_refinement_plan", ["mixed_combo_spread_remains_high"]
    return "v8173_3_combo_specific_policy_refinement_plan", ["family_rate_extremes_or_spread_need_policy_refinement"]


def _v81731_golden_issues(v81731_json: Mapping[str, Any]) -> list[str]:
    if not v81731_json:
        return ["v81731_json_missing"]
    diagnostics = v81731_json.get("diagnostics") or {}
    best = diagnostics.get("best_balanced_vector") or {}
    issues: list[str] = []
    expected = {
        "risk_row_count": 126,
        "mixed_risk_row_count": 78,
        "single_risk_row_count": 48,
        "vector_id": "dry=0.06|dry_vent=0.06|dew=0.10|canopy=0.12",
        "conservative_risk_selected_count": 106,
        "selected_id_vs_zero_bonus_winner_mismatch_count": 248,
    }
    for key in ("risk_row_count", "mixed_risk_row_count", "single_risk_row_count"):
        if int(_num(diagnostics.get(key))) != expected[key]:
            issues.append(f"golden_mismatch:v81731_{key}")
    if best.get("vector_id") != expected["vector_id"]:
        issues.append("golden_mismatch:v81731_best_vector")
    if int(_num(best.get("conservative_risk_selected_count"))) != expected["conservative_risk_selected_count"]:
        issues.append("golden_mismatch:v81731_conservative_risk_selected_count")
    if abs(_num(best.get("risk_type_switch_spread")) - 0.45) > 1e-9:
        issues.append("golden_mismatch:v81731_family_spread")
    if int(_num(best.get("selected_id_vs_zero_bonus_winner_mismatch_count"))) != expected[
        "selected_id_vs_zero_bonus_winner_mismatch_count"
    ]:
        issues.append("golden_mismatch:v81731_selected_id_vs_zero_bonus_winner_mismatch_count")
    return issues


def build_report(
    *,
    trace_csv_rows: list[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    jsonl_rows: list[Mapping[str, Any]],
    v8173_json: Mapping[str, Any],
    v81731_json: Mapping[str, Any],
    vectors: Sequence[Mapping[str, float]] | None = None,
    policies: Sequence[str] = POLICIES,
    enforce_golden: bool = True,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    boundary = _boundary_summary(summary_json, jsonl_rows)
    schema_issues = _missing_schema_fields(jsonl_rows)
    records, reconstruction_issues = _build_step_records(trace_csv_rows=trace_csv_rows, jsonl_rows=jsonl_rows)
    if enforce_golden:
        schema_issues = sorted(set(schema_issues + reconstruction_issues + _v81731_golden_issues(v81731_json)))
    else:
        schema_issues = sorted(set(schema_issues + reconstruction_issues))
    all_curve_rows: list[dict[str, Any]] = []
    all_family_rows: list[dict[str, Any]] = []
    all_combo_rows: list[dict[str, Any]] = []
    policy_summaries: list[dict[str, Any]] = []
    policy_details: dict[str, Any] = {}
    for policy in policies:
        curve_rows, family_rows, combo_rows, policy_diagnostics = evaluate_policy_vectors(
            records,
            policy=policy,
            vectors=vectors,
        )
        all_curve_rows.extend(curve_rows)
        all_family_rows.extend(family_rows)
        all_combo_rows.extend(combo_rows)
        best = dict(policy_diagnostics.get("best_vector") or {})
        best.update(
            {
                "policy": policy,
                "best_vector_selection_reason": policy_diagnostics.get("best_vector_selection_reason"),
                "why_min_spread_vector_not_selected": policy_diagnostics.get("why_min_spread_vector_not_selected"),
            }
        )
        policy_summaries.append(best)
        policy_details[policy] = policy_diagnostics
    component_rows = component_gap_by_family(records, policies=policies)
    best_policy = _best_policy_row(policy_summaries)
    best_family_rows = [
        row
        for row in all_family_rows
        if row.get("policy") == best_policy.get("policy") and row.get("vector_id") == best_policy.get("vector_id")
    ]
    next_action, readiness_reasons = _route_next_action(
        boundary=boundary,
        schema_issues=schema_issues,
        best_policy=best_policy,
        best_family_rows=best_family_rows,
        component_rows=component_rows,
    )
    diagnostics = {
        "policy_count": len(policies),
        "bonus_vector_count": len(vectors or bonus_vectors()),
        "policy_summaries": policy_summaries,
        "best_policy": best_policy,
        "best_policy_family_rate_table": best_family_rows,
        "best_policy_dominant_nonfinal_component": _dominant_component_for_policy(
            component_rows,
            str(best_policy.get("policy") or ""),
        ),
        "v8173_reference_next_action": v8173_json.get("next_action"),
        "v81731_reference_next_action": v81731_json.get("next_action"),
        "v81731_reference_best_vector": ((v81731_json.get("diagnostics") or {}).get("best_balanced_vector") or {}).get(
            "vector_id"
        ),
        "v81731_reference_family_spread": ((v81731_json.get("diagnostics") or {}).get("best_balanced_vector") or {}).get(
            "risk_type_switch_spread"
        ),
    }
    report = {
        "schema_version": "cstcc_v8173_conflict_resolver_policy_iteration_v1",
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
        },
        "boundary_runtime_safety": boundary,
        "schema_issues": schema_issues,
        "counterfactual_definition": {
            "policy_iteration_is_evaluator_only": True,
            "global_bonus_used": False,
            "bonus_grid_expanded": False,
            "bonus_is_non_additive": True,
            "resolver_changes_runtime_scorer": False,
            "resolver_changes_runtime_template_pool": False,
            "runtime_final_action_changed": False,
        },
        "diagnostics": diagnostics,
        "acceptance": {
            "boundary_clean": _boundary_clean(boundary),
            "nonrisk_conservative_selected_count_eq_0": int(best_policy.get("nonrisk_conservative_selected_count", 0) or 0)
            == 0,
            "risk_type_switch_spread_le_0_30": _num(best_policy.get("risk_type_switch_spread"), 999.0) <= 0.30,
            "mixed_risk_combo_spread_le_0_30": _num(best_policy.get("mixed_risk_combo_spread"), 999.0) <= 0.30,
            "single_risk_spread_le_0_30": _num(best_policy.get("single_risk_spread"), 999.0) <= 0.30,
            "conservative_risk_selected_rate_in_target_band": TARGET_RATE_LOW
            <= _num(best_policy.get("conservative_risk_selected_rate"))
            <= TARGET_RATE_HIGH,
            "family_extreme_rate_count_eq_0": int(best_policy.get("family_extreme_rate_count", 0) or 0) == 0,
            "selected_id_vs_zero_bonus_winner_mismatch_count_reported": "selected_id_vs_zero_bonus_winner_mismatch_count"
            in best_policy,
            "rule_suppression_detected": bool(best_policy.get("rule_suppression_detected", False)),
            "previous_rule_ppo_suppression_detected": bool(best_policy.get("previous_rule_ppo_suppression_detected", False)),
        },
        "next_action": next_action,
        "readiness_reasons": readiness_reasons,
        "notes": [
            "v8173.2 is offline counterfactual resolver policy iteration only.",
            "It does not change runtime scorer, template pool, final actions, or evidence level.",
            "It does not validate reward, profit, safety improvement, or closed-loop controller quality.",
        ],
    }
    return report, {
        "policy_curve": all_curve_rows,
        "policy_family_rate_table": all_family_rows,
        "policy_combo_rate_table": all_combo_rows,
        "policy_component_gap_by_family": component_rows,
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    boundary = report.get("boundary_runtime_safety", {})
    diagnostics = report.get("diagnostics", {})
    best = diagnostics.get("best_policy") or {}
    lines = [
        "# C-STCC v8173.2 Conflict Resolver Policy Iteration",
        "",
        "Existing-trace counterfactual audit only. No online LLM, no rollout, no runtime scorer/template change.",
        "",
        "## Boundary",
        "",
        f"- Audit success rate: {_num(boundary.get('audit_success_rate')):.3f}",
        f"- Final-action invariant rate: {_num(boundary.get('final_action_invariant_rate')):.3f}",
        f"- Online LLM / rollout / real projection steps: {boundary.get('online_llm_called_steps', 0)} / {boundary.get('predictive_rollout_executed_steps', 0)} / {boundary.get('real_tomato_safety_projection_steps', 0)}",
        "",
        "## Best Policy",
        "",
        f"- Policy: {best.get('policy', 'none')}",
        f"- Vector: {best.get('vector_id', 'none')}",
        f"- Conservative risk selected rate: {_num(best.get('conservative_risk_selected_rate')):.3f}",
        f"- Family / mixed combo / single spread: {_num(best.get('risk_type_switch_spread')):.3f} / {_num(best.get('mixed_risk_combo_spread')):.3f} / {_num(best.get('single_risk_spread')):.3f}",
        f"- Family target penalty: {_num(best.get('family_target_penalty')):.3f}",
        f"- Family extreme rate count: {best.get('family_extreme_rate_count', 0)}",
        f"- Dominant non-final component: {diagnostics.get('best_policy_dominant_nonfinal_component') or 'none'}",
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
    base = "conflict_resolver_policy_iteration"
    paths = {
        "json": prefix.with_suffix(".json"),
        "md": prefix.with_suffix(".md"),
        "policy_curve_csv": (parent / stem.replace(base, "policy_curve")).with_suffix(".csv"),
        "policy_family_rate_table_csv": (parent / stem.replace(base, "policy_family_rate_table")).with_suffix(".csv"),
        "policy_combo_rate_table_csv": (parent / stem.replace(base, "policy_combo_rate_table")).with_suffix(".csv"),
        "policy_component_gap_by_family_csv": (
            parent / stem.replace(base, "policy_component_gap_by_family")
        ).with_suffix(".csv"),
    }
    paths["json"].write_text(_compact_json(report) + "\n", encoding="utf-8")
    paths["md"].write_text(build_markdown(report), encoding="utf-8")
    _write_csv(paths["policy_curve_csv"], tables.get("policy_curve", []))
    _write_csv(paths["policy_family_rate_table_csv"], tables.get("policy_family_rate_table", []))
    _write_csv(paths["policy_combo_rate_table_csv"], tables.get("policy_combo_rate_table", []))
    _write_csv(paths["policy_component_gap_by_family_csv"], tables.get("policy_component_gap_by_family", []))
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v8173.2 C-STCC conflict resolver policy iteration audit.")
    parser.add_argument("--trace-csv", type=str, default=str(DEFAULT_TRACE_CSV))
    parser.add_argument("--summary-json", type=str, default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--jsonl-root", type=str, default=str(DEFAULT_JSONL_ROOT))
    parser.add_argument("--v8173-json", type=str, default=str(DEFAULT_V8173_JSON))
    parser.add_argument("--v81731-json", type=str, default=str(DEFAULT_V81731_JSON))
    parser.add_argument("--output-prefix", type=str, default=str(DEFAULT_OUTPUT_PREFIX))
    parser.add_argument("--no-golden-check", action="store_true")
    args = parser.parse_args()
    report, tables = build_report(
        trace_csv_rows=_load_csv(args.trace_csv),
        summary_json=_load_json(args.summary_json),
        jsonl_rows=_load_jsonl_rows(args.jsonl_root),
        v8173_json=_load_json(args.v8173_json),
        v81731_json=_load_json(args.v81731_json),
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
