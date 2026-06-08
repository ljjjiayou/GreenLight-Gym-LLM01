"""v8173.3 offline risk-conditioned energy proxy reweight audit.

This stage is an existing-trace counterfactual scorer audit. It reads the
v8171/v8173/v8173.1/v8173.2 H720 evidence chain and tests whether risk-family
conditioned energy-gap relief can reduce spread. It does not change runtime
scoring, template pools, final actions, online LLM usage, rollout, or Tomato
Safety projection.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.cstcc_v8173_conflict_resolver_policy_iteration import (
    DEFAULT_OUTPUT_PREFIX as DEFAULT_V81732_OUTPUT_PREFIX,
    DEFAULT_V81731_JSON,
    _spread,
    _target_gap,
    _target_label,
    _v81731_golden_issues,
    annotate_records_for_policy,
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
from gl_gym.experiments.cstcc_v8173_mixed_risk_conflict_resolver_design import (
    DEFAULT_V8173_JSON,
)


DEFAULT_V81732_JSON = DEFAULT_V81732_OUTPUT_PREFIX.with_suffix(".json")
DEFAULT_V81732_POLICY_CURVE_CSV = Path(
    "gl_gym/result/diagnostics/cstcc_v8173_policy_curve_20260606_y2020_d240_s42_n720.csv"
)
DEFAULT_V81732_FAMILY_CSV = Path(
    "gl_gym/result/diagnostics/cstcc_v8173_policy_family_rate_table_20260606_y2020_d240_s42_n720.csv"
)
DEFAULT_V81732_COMBO_CSV = Path(
    "gl_gym/result/diagnostics/cstcc_v8173_policy_combo_rate_table_20260606_y2020_d240_s42_n720.csv"
)
DEFAULT_V81732_COMPONENT_CSV = Path(
    "gl_gym/result/diagnostics/cstcc_v8173_policy_component_gap_by_family_20260606_y2020_d240_s42_n720.csv"
)
DEFAULT_OUTPUT_PREFIX = Path(
    "gl_gym/result/diagnostics/cstcc_v8173_energy_proxy_reweight_design_20260607_y2020_d240_s42_n720"
)

CANOPY_DEW_GRID = (1.0, 0.8, 0.6)
DEW_OR_HUMIDITY_GRID = (1.0, 0.8, 0.6)
DRY_GRID = (1.0, 0.9, 0.8)
TARGET_RATE_LOW = 0.35
TARGET_RATE_HIGH = 0.75
ENERGY_COMPONENTS = ("raw_energy_proxy", "projected_energy_proxy")


def reweight_factor_vectors() -> list[dict[str, float]]:
    return [
        {
            "canopy_dew_lt0_factor": canopy_lt0,
            "canopy_dew_lt1_factor": canopy_lt1,
            "dew_or_humidity_factor": dew,
            "dry_vent_factor": dry_vent,
            "dry_or_high_vpd_factor": dry,
        }
        for canopy_lt0, canopy_lt1, dew, dry_vent, dry in itertools.product(
            CANOPY_DEW_GRID,
            CANOPY_DEW_GRID,
            DEW_OR_HUMIDITY_GRID,
            DRY_GRID,
            DRY_GRID,
        )
    ]


def factor_vector_id(vector: Mapping[str, Any]) -> str:
    return (
        f"canopy_lt0={_num(vector.get('canopy_dew_lt0_factor')):.1f}|"
        f"canopy_lt1={_num(vector.get('canopy_dew_lt1_factor')):.1f}|"
        f"dew={_num(vector.get('dew_or_humidity_factor')):.1f}|"
        f"dry_vent={_num(vector.get('dry_vent_factor')):.1f}|"
        f"dry={_num(vector.get('dry_or_high_vpd_factor')):.1f}"
    )


def factor_for_family(vector: Mapping[str, Any], family: str, *, is_risk: bool) -> float:
    if not is_risk:
        return 1.0
    if family == "canopy_dew_lt0":
        return _num(vector.get("canopy_dew_lt0_factor"), 1.0)
    if family == "canopy_dew_lt1":
        return _num(vector.get("canopy_dew_lt1_factor"), 1.0)
    if family == "dew_or_humidity":
        return _num(vector.get("dew_or_humidity_factor"), 1.0)
    if family == "dry_vent":
        return _num(vector.get("dry_vent_factor"), 1.0)
    if family == "dry_or_high_vpd":
        return _num(vector.get("dry_or_high_vpd_factor"), 1.0)
    return 1.0


def _best_policy(v81732_json: Mapping[str, Any]) -> dict[str, Any]:
    return dict(((v81732_json.get("diagnostics") or {}).get("best_policy") or {}))


def _best_bonus_vector(best_policy: Mapping[str, Any]) -> dict[str, float]:
    return {
        "dry_or_high_vpd_bonus": _num(best_policy.get("dry_or_high_vpd_bonus")),
        "dry_vent_bonus": _num(best_policy.get("dry_vent_bonus")),
        "dew_or_humidity_bonus": _num(best_policy.get("dew_or_humidity_bonus")),
        "canopy_dew_bonus": _num(best_policy.get("canopy_dew_bonus")),
    }


def _filter_best_policy_rows(rows: Sequence[Mapping[str, Any]], best_policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    policy = str(best_policy.get("policy") or "")
    vid = str(best_policy.get("vector_id") or "")
    return [
        dict(row)
        for row in rows
        if str(row.get("policy") or "") == policy and str(row.get("vector_id") or "") == vid
    ]


def _family_label_map(best_family_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("resolved_risk_family") or "none"): str(row.get("family_target_label") or "unknown")
        for row in best_family_rows
    }


def _combo_label_map(best_combo_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("risk_combo_id") or "none"): str(row.get("combo_target_label") or "unknown")
        for row in best_combo_rows
    }


def _conservative_candidate(record: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    candidate_id = str(record.get("conservative_candidate_id") or "")
    return candidate_id, (record.get("components") or {}).get(candidate_id, {})


def _energy_gap(record: Mapping[str, Any]) -> dict[str, float]:
    zero_components = (record.get("components") or {}).get(str(record.get("zero_bonus_winner_id") or ""), {})
    _conservative_id, conservative_components = _conservative_candidate(record)
    raw_gap = max(0.0, _num(zero_components.get("raw_energy_proxy")) - _num(conservative_components.get("raw_energy_proxy")))
    projected_gap = max(
        0.0,
        _num(zero_components.get("projected_energy_proxy")) - _num(conservative_components.get("projected_energy_proxy")),
    )
    return {
        "raw_energy_gap": raw_gap,
        "projected_energy_gap": projected_gap,
        "combined_energy_gap": max(raw_gap, projected_gap),
        "final_score_gap": _num(zero_components.get("final_score")) - _num(conservative_components.get("final_score")),
    }


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


def _metric_summary(values: Sequence[float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(mean(values)) if values else 0.0,
        f"{prefix}_p50": _quantile(values, 0.50),
        f"{prefix}_p75": _quantile(values, 0.75),
        f"{prefix}_p90": _quantile(values, 0.90),
        f"{prefix}_p95": _quantile(values, 0.95),
        f"{prefix}_max": max(values) if values else 0.0,
    }


def aggregate_energy_gaps(
    records: Sequence[Mapping[str, Any]],
    *,
    group_name: str,
    group_key: Callable[[Mapping[str, Any]], str],
    label_key: Callable[[Mapping[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if not (record.get("risk_flags") or {}).get("any_risk", False):
            continue
        groups.setdefault(group_key(record), []).append(record)
    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        gaps = [_energy_gap(record) for record in items]
        row = {
            "group_name": group_name,
            "group_id": key,
            "row_count": len(items),
        }
        if label_key is not None:
            row["target_label_distribution"] = _distribution(label_key(record) for record in items)
        for metric in ("raw_energy_gap", "projected_energy_gap", "combined_energy_gap", "final_score_gap"):
            row.update(_metric_summary([gap[metric] for gap in gaps], metric))
        rows.append(row)
    return rows


def _is_conservative(candidate_id: str, candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("feasible", False)
        and candidate.get("source_prior") == "conservative_prior"
        and candidate.get("template_name") == "conservative_stabilize"
    )


def _energy_relief(record: Mapping[str, Any], factor: float) -> float:
    if factor >= 1.0:
        return 0.0
    return (1.0 - factor) * _energy_gap(record)["combined_energy_gap"]


def _winner_with_reweight(
    record: Mapping[str, Any],
    *,
    bonus_vector: Mapping[str, Any],
    factor_vector: Mapping[str, Any],
) -> tuple[str, float, dict[str, Any]]:
    resolver = record.get("resolver") or {}
    family = str(resolver.get("resolved_risk_family_new") or "none")
    is_risk = bool((record.get("risk_flags") or {}).get("any_risk", False))
    factor = factor_for_family(factor_vector, family, is_risk=is_risk)
    assigned_bonus = bonus_for_family(bonus_vector, family)
    best_id = ""
    best_score = -1e18
    best_meta: dict[str, Any] = {
        "assigned_factor": factor,
        "assigned_bonus": assigned_bonus,
        "energy_relief": 0.0,
        **_energy_gap(record),
    }
    for candidate_id, component in (record.get("components") or {}).items():
        candidate = (record.get("candidates") or {}).get(candidate_id)
        if candidate is None or not candidate.get("feasible", False):
            continue
        score = _num(component.get("final_score"), -1e18)
        meta = dict(best_meta)
        if _is_conservative(candidate_id, candidate) and is_risk and family != "none":
            relief = _energy_relief(record, factor)
            score += assigned_bonus + relief
            meta["energy_relief"] = relief
        elif _is_conservative(candidate_id, candidate) and family != "none":
            score += assigned_bonus
        if score > best_score:
            best_id = candidate_id
            best_score = score
            best_meta = meta
    return best_id, best_score, best_meta


def _target_counts_to_rows(
    *,
    policy: str,
    factor_id: str,
    factor_vector: Mapping[str, Any],
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
                "factor_vector_id": factor_id,
                **dict(factor_vector),
                "resolved_risk_family": family,
                "family_row_count": count,
                "family_conservative_selected_count": selected,
                "family_conservative_selected_rate": rate,
                "family_gap_to_target_rate": _target_gap(rate),
                "family_target_label": _target_label(rate),
            }
        )
    return rows


def _combo_counts_to_rows(
    *,
    policy: str,
    factor_id: str,
    factor_vector: Mapping[str, Any],
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
                "factor_vector_id": factor_id,
                **dict(factor_vector),
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


def _adjusted_component_rows(records: Sequence[Mapping[str, Any]], factor_vector: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for record in records:
        if not (record.get("risk_flags") or {}).get("any_risk", False):
            continue
        family = str((record.get("resolver") or {}).get("resolved_risk_family_new") or "none")
        factor = factor_for_family(factor_vector, family, is_risk=True)
        adjusted = dict(record.get("zero_minus_conservative") or {})
        for component in ENERGY_COMPONENTS:
            adjusted[component] = max(0.0, _num(adjusted.get(component)) * factor)
        adjusted["final_score"] = _num(adjusted.get("final_score")) - (1.0 - factor) * _energy_gap(record)[
            "combined_energy_gap"
        ]
        grouped.setdefault(family, []).append(adjusted)
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


def _dominant_after_reweight(records: Sequence[Mapping[str, Any]], factor_vector: Mapping[str, Any]) -> str:
    return _dominant_component(
        [row for row in _adjusted_component_rows(records, factor_vector) if row.get("component") != "final_score"]
    )


def _curve_accepts(row: Mapping[str, Any]) -> bool:
    rate = _num(row.get("conservative_risk_selected_rate"))
    return bool(
        TARGET_RATE_LOW <= rate <= TARGET_RATE_HIGH
        and _num(row.get("risk_type_switch_spread"), 999.0) <= 0.30
        and _num(row.get("mixed_risk_combo_spread"), 999.0) <= 0.30
        and _num(row.get("single_risk_spread"), 999.0) <= 0.30
        and int(row.get("family_extreme_rate_count", 0) or 0) == 0
        and int(row.get("nonrisk_conservative_selected_count", 0) or 0) == 0
        and not _bool(row.get("rule_suppression_detected"))
        and not _bool(row.get("previous_rule_ppo_suppression_detected"))
        and row.get("dominant_component_after_reweight") not in {"raw_energy_proxy", "projected_energy_proxy"}
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
        candidate.get("dominant_component_after_reweight") in {"raw_energy_proxy", "projected_energy_proxy"},
        abs(candidate_rate - 0.55),
    ) < (
        _num(current.get("family_target_penalty")),
        _num(current.get("family_extreme_rate_count")),
        max(
            _num(current.get("risk_type_switch_spread")),
            _num(current.get("mixed_risk_combo_spread")),
            _num(current.get("single_risk_spread")),
        ),
        current.get("dominant_component_after_reweight") in {"raw_energy_proxy", "projected_energy_proxy"},
        abs(current_rate - 0.55),
    )


def evaluate_reweight_factors(
    records: Sequence[Mapping[str, Any]],
    *,
    best_policy: Mapping[str, Any],
    factor_vectors: Sequence[Mapping[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    policy = str(best_policy.get("policy") or "")
    bonus_vector = _best_bonus_vector(best_policy)
    annotated = annotate_records_for_policy(records, policy)
    risk_records = [record for record in annotated if (record.get("risk_flags") or {}).get("any_risk", False)]
    selected_mismatch = sum(1 for record in annotated if record.get("runtime_selected_id") != record.get("zero_bonus_winner_id"))
    curve_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] = {}

    for factor_vector in factor_vectors or reweight_factor_vectors():
        factor_id = factor_vector_id(factor_vector)
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
        relief_values: list[float] = []

        for record in annotated:
            family = str((record.get("resolver") or {}).get("resolved_risk_family_new") or "none")
            combo_id = str((record.get("resolver") or {}).get("risk_combo_id") or "none")
            is_mixed = "+" in combo_id
            is_risk = bool((record.get("risk_flags") or {}).get("any_risk", False))
            winner_id, _winner_score, meta = _winner_with_reweight(
                record,
                bonus_vector=bonus_vector,
                factor_vector=factor_vector,
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
                relief_values.append(_num(meta.get("energy_relief")))
                for bucket in (
                    family_counts.setdefault(family, {"count": 0, "conservative": 0}),
                    combo_counts.setdefault(combo_id, {"count": 0, "conservative": 0}),
                    (mixed_counts if is_mixed else single_counts).setdefault(family, {"count": 0, "conservative": 0}),
                ):
                    bucket["count"] += 1
                    if is_conservative:
                        bucket["conservative"] += 1

        family_rates = [
            int(item.get("conservative", 0) or 0) / max(int(item.get("count", 0) or 0), 1)
            for family, item in family_counts.items()
            if family != "none" and int(item.get("count", 0) or 0) > 0
        ]
        combo_rates = [
            int(item.get("conservative", 0) or 0) / max(int(item.get("count", 0) or 0), 1)
            for combo_id, item in combo_counts.items()
            if "+" in combo_id and int(item.get("count", 0) or 0) > 0
        ]
        single_rates = [
            int(item.get("conservative", 0) or 0) / max(int(item.get("count", 0) or 0), 1)
            for family, item in single_counts.items()
            if family != "none" and int(item.get("count", 0) or 0) > 0
        ]
        mixed_rates = [
            int(item.get("conservative", 0) or 0) / max(int(item.get("count", 0) or 0), 1)
            for family, item in mixed_counts.items()
            if family != "none" and int(item.get("count", 0) or 0) > 0
        ]
        dominant_after = _dominant_after_reweight(annotated, factor_vector)
        row = {
            "policy": policy,
            "factor_vector_id": factor_id,
            **dict(factor_vector),
            "bonus_vector_id": str(best_policy.get("vector_id") or ""),
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
            "switch_relative_to_runtime_count": switch_runtime,
            "switch_relative_to_zero_bonus_count": switch_zero,
            "switch_to_conservative_count": switch_to_conservative,
            "switch_from_previous_to_conservative_count": switch_from_previous,
            "switch_from_rule_to_conservative_count": switch_from_rule,
            "switch_from_ppo_to_conservative_count": switch_from_ppo,
            "selected_id_vs_zero_bonus_winner_mismatch_count": selected_mismatch,
            "rule_suppression_detected": switch_from_rule > 0,
            "previous_rule_ppo_suppression_detected": conservative_risk_selected / max(len(risk_records), 1) > TARGET_RATE_HIGH,
            "dominant_component_after_reweight": dominant_after,
            "mean_energy_relief": float(mean(relief_values)) if relief_values else 0.0,
            "p95_energy_relief": _quantile(relief_values, 0.95),
        }
        curve_rows.append(row)
        family_rows.extend(
            _target_counts_to_rows(
                policy=policy,
                factor_id=factor_id,
                factor_vector=factor_vector,
                family_counts=family_counts,
            )
        )
        combo_rows.extend(
            _combo_counts_to_rows(
                policy=policy,
                factor_id=factor_id,
                factor_vector=factor_vector,
                combo_counts=combo_counts,
            )
        )
        if _better_curve(row, best_row):
            best_row = dict(row)

    diagnostics = {
        "best_reweight_vector": best_row,
        "factor_vector_count": len(factor_vectors or reweight_factor_vectors()),
    }
    return curve_rows, family_rows, combo_rows, diagnostics


def switch_rows_for_factor(
    records: Sequence[Mapping[str, Any]],
    *,
    best_policy: Mapping[str, Any],
    factor_vector: Mapping[str, Any],
) -> list[dict[str, Any]]:
    policy = str(best_policy.get("policy") or "")
    bonus_vector = _best_bonus_vector(best_policy)
    rows: list[dict[str, Any]] = []
    for record in annotate_records_for_policy(records, policy):
        family = str((record.get("resolver") or {}).get("resolved_risk_family_new") or "none")
        combo_id = str((record.get("resolver") or {}).get("risk_combo_id") or "none")
        is_risk = bool((record.get("risk_flags") or {}).get("any_risk", False))
        winner_id, winner_score, meta = _winner_with_reweight(
            record,
            bonus_vector=bonus_vector,
            factor_vector=factor_vector,
        )
        zero_id = str(record.get("zero_bonus_winner_id") or "")
        runtime_id = str(record.get("runtime_selected_id") or "")
        if not is_risk and winner_id == zero_id and winner_id == runtime_id:
            continue
        winner_components = (record.get("components") or {}).get(winner_id, {})
        zero_components = (record.get("components") or {}).get(zero_id, {})
        rows.append(
            {
                "step": record.get("step"),
                "hour_of_day": record.get("hour_of_day"),
                "active_regime": record.get("active_regime"),
                "resolved_risk_family": family,
                "risk_combo_id": combo_id,
                "is_risk": is_risk,
                "factor_vector_id": factor_vector_id(factor_vector),
                **dict(factor_vector),
                "assigned_bonus": meta.get("assigned_bonus"),
                "assigned_factor": meta.get("assigned_factor"),
                "energy_relief": meta.get("energy_relief"),
                "raw_energy_gap": meta.get("raw_energy_gap"),
                "projected_energy_gap": meta.get("projected_energy_gap"),
                "runtime_selected_id": runtime_id,
                "zero_bonus_winner_id": zero_id,
                "counterfactual_winner_id": winner_id,
                "counterfactual_winner_score": winner_score,
                "switch_relative_to_runtime": winner_id != runtime_id,
                "switch_relative_to_zero_bonus": winner_id != zero_id,
                "switch_to_conservative": _source(winner_id) == "conservative_prior"
                and _source(zero_id) != "conservative_prior",
                "zero_minus_counterfactual_final_score": _component_delta(zero_components, winner_components).get(
                    "final_score",
                    0.0,
                ),
            }
        )
    return rows


def _v81732_golden_issues(v81732_json: Mapping[str, Any]) -> list[str]:
    if not v81732_json:
        return ["v81732_json_missing"]
    diagnostics = v81732_json.get("diagnostics") or {}
    best = diagnostics.get("best_policy") or {}
    issues: list[str] = []
    expected = {
        "policy": "canopy_dew_first_v1",
        "vector_id": "dry=0.06|dry_vent=0.06|dew=0.08|canopy=0.10",
        "risk_row_count": 126,
        "conservative_risk_selected_count": 68,
        "risk_type_switch_spread": 0.641025641025641,
        "mixed_risk_combo_spread": 1.0,
        "single_risk_spread": 0.375,
        "dominant": "raw_energy_proxy",
    }
    if best.get("policy") != expected["policy"]:
        issues.append("golden_mismatch:v81732_best_policy")
    if best.get("vector_id") != expected["vector_id"]:
        issues.append("golden_mismatch:v81732_best_vector")
    if int(_num(best.get("risk_row_count"))) != expected["risk_row_count"]:
        issues.append("golden_mismatch:v81732_risk_row_count")
    if int(_num(best.get("conservative_risk_selected_count"))) != expected["conservative_risk_selected_count"]:
        issues.append("golden_mismatch:v81732_conservative_risk_selected_count")
    for key in ("risk_type_switch_spread", "mixed_risk_combo_spread", "single_risk_spread"):
        if abs(_num(best.get(key)) - expected[key]) > 1e-9:
            issues.append(f"golden_mismatch:v81732_{key}")
    if diagnostics.get("best_policy_dominant_nonfinal_component") != expected["dominant"]:
        issues.append("golden_mismatch:v81732_dominant_component")
    return issues


def _route_next_action(
    *,
    boundary: Mapping[str, Any],
    schema_issues: Sequence[str],
    best_reweight: Mapping[str, Any],
) -> tuple[str, list[str]]:
    if not _boundary_clean(boundary):
        return "stop_and_repair_cstcc_shadow_boundary", ["boundary_or_candidate_health_failure"]
    if schema_issues:
        return "v8173_schema_or_golden_trace_repair_plan", list(schema_issues)
    if _curve_accepts(best_reweight):
        return "v8173_4_opt_in_energy_reweight_shadow_trace_plan", ["risk_conditioned_energy_reweight_passes_gates"]
    dominant = str(best_reweight.get("dominant_component_after_reweight") or "")
    if dominant in {"raw_energy_proxy", "projected_energy_proxy"} and max(
        _num(best_reweight.get("risk_type_switch_spread")),
        _num(best_reweight.get("mixed_risk_combo_spread")),
        _num(best_reweight.get("single_risk_spread")),
    ) > 0.30:
        return "v8173_energy_proxy_formula_redesign_plan", ["spread_still_high_and_energy_proxy_still_dominant"]
    if (
        _num(best_reweight.get("mixed_risk_combo_spread"), 0.0) > 0.30
        or int(best_reweight.get("family_extreme_rate_count", 0) or 0) > 0
    ):
        return "v8173_combo_specific_energy_policy_iteration_plan", ["specific_combo_or_family_extreme_remains"]
    if dominant in {"base_final_score", "raw_smoothness", "projected_smoothness"}:
        return "v8174_risk_specific_template_design_plan", [f"energy_no_longer_dominant_but_{dominant}_dominates"]
    return "v8173_combo_specific_energy_policy_iteration_plan", ["counterfactual_reweight_not_balanced"]


def build_report(
    *,
    trace_csv_rows: list[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    jsonl_rows: list[Mapping[str, Any]],
    v8173_json: Mapping[str, Any],
    v81731_json: Mapping[str, Any],
    v81732_json: Mapping[str, Any],
    v81732_family_rows: list[Mapping[str, Any]],
    v81732_combo_rows: list[Mapping[str, Any]],
    v81732_component_rows: list[Mapping[str, Any]],
    factor_vectors: Sequence[Mapping[str, float]] | None = None,
    enforce_golden: bool = True,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    boundary = _boundary_summary(summary_json, jsonl_rows)
    schema_issues = _missing_schema_fields(jsonl_rows)
    records, reconstruction_issues = _build_step_records(trace_csv_rows=trace_csv_rows, jsonl_rows=jsonl_rows)
    best_policy = _best_policy(v81732_json)
    if not best_policy:
        schema_issues.append("v81732_best_policy_missing")
    if enforce_golden:
        schema_issues = sorted(
            set(
                schema_issues
                + reconstruction_issues
                + _v81731_golden_issues(v81731_json)
                + _v81732_golden_issues(v81732_json)
            )
        )
    else:
        schema_issues = sorted(set(schema_issues + reconstruction_issues))

    best_family_rows = _filter_best_policy_rows(v81732_family_rows, best_policy)
    best_combo_rows = _filter_best_policy_rows(v81732_combo_rows, best_policy)
    best_mixed_combo_rows = [row for row in best_combo_rows if _bool(row.get("is_mixed_risk"))]
    family_labels = _family_label_map(best_family_rows)
    combo_labels = _combo_label_map(best_combo_rows)
    annotated = annotate_records_for_policy(records, str(best_policy.get("policy") or ""))
    energy_gap_by_family = aggregate_energy_gaps(
        annotated,
        group_name="resolved_risk_family",
        group_key=lambda record: str((record.get("resolver") or {}).get("resolved_risk_family_new") or "none"),
        label_key=lambda record: family_labels.get(
            str((record.get("resolver") or {}).get("resolved_risk_family_new") or "none"),
            "unknown",
        ),
    )
    energy_gap_by_combo = aggregate_energy_gaps(
        annotated,
        group_name="risk_combo_id",
        group_key=lambda record: str((record.get("resolver") or {}).get("risk_combo_id") or "none"),
        label_key=lambda record: combo_labels.get(
            str((record.get("resolver") or {}).get("risk_combo_id") or "none"),
            "unknown",
        ),
    )
    energy_gap_by_single_vs_mixed = aggregate_energy_gaps(
        annotated,
        group_name="single_vs_mixed",
        group_key=lambda record: "mixed_risk" if "+" in str((record.get("resolver") or {}).get("risk_combo_id") or "") else "single_risk",
    )
    energy_gap_by_under_over_target = aggregate_energy_gaps(
        annotated,
        group_name="family_target_label",
        group_key=lambda record: family_labels.get(
            str((record.get("resolver") or {}).get("resolved_risk_family_new") or "none"),
            "unknown",
        ),
    )
    curve_rows, family_rows, combo_rows, reweight_diagnostics = evaluate_reweight_factors(
        records,
        best_policy=best_policy,
        factor_vectors=factor_vectors,
    )
    best_reweight = reweight_diagnostics.get("best_reweight_vector") or {}
    switch_rows = switch_rows_for_factor(
        records,
        best_policy=best_policy,
        factor_vector={
            key: _num(best_reweight.get(key), 1.0)
            for key in (
                "canopy_dew_lt0_factor",
                "canopy_dew_lt1_factor",
                "dew_or_humidity_factor",
                "dry_vent_factor",
                "dry_or_high_vpd_factor",
            )
        },
    )
    next_action, readiness_reasons = _route_next_action(
        boundary=boundary,
        schema_issues=schema_issues,
        best_reweight=best_reweight,
    )
    diagnostics = {
        "observed_gap": {
            "best_compromise_is_control_semantic_optimum": False,
            "v81732_best_policy": best_policy,
            "v81732_best_policy_family_rate_table": best_family_rows,
            "v81732_best_policy_mixed_combo_rate_table": best_mixed_combo_rows,
            "dry_vent_over_target": "dry_vent=21/21 selected under v8173.2 best compromise",
            "canopy_dew_lt0_dew_under_target": "canopy_dew_lt0+dew_or_humidity=0/6 selected under v8173.2 best compromise",
            "v81732_dominant_component": ((v81732_json.get("diagnostics") or {}).get("best_policy_dominant_nonfinal_component")),
        },
        "factor_vector_count": reweight_diagnostics.get("factor_vector_count", 0),
        "best_reweight_vector": best_reweight,
        "energy_gap_by_single_vs_mixed": energy_gap_by_single_vs_mixed,
        "energy_gap_by_under_over_target": energy_gap_by_under_over_target,
        "v8173_reference_next_action": v8173_json.get("next_action"),
        "v81731_reference_next_action": v81731_json.get("next_action"),
        "v81732_reference_next_action": v81732_json.get("next_action"),
        "v81732_component_rows_used": len(v81732_component_rows),
    }
    report = {
        "schema_version": "cstcc_v8173_energy_proxy_reweight_design_v1",
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
            "global_energy_weight_changed": False,
        },
        "boundary_runtime_safety": boundary,
        "schema_issues": schema_issues,
        "counterfactual_definition": {
            "energy_proxy_reweight_is_evaluator_only": True,
            "risk_conditioned_reweight_only": True,
            "global_energy_weight_multiplier_used": False,
            "bonus_vector_changed": False,
            "prior_confidence_changed": False,
            "runtime_score_dual_changed": False,
            "runtime_final_action_changed": False,
        },
        "diagnostics": diagnostics,
        "acceptance": {
            "boundary_clean": _boundary_clean(boundary),
            "nonrisk_conservative_selected_count_eq_0": int(best_reweight.get("nonrisk_conservative_selected_count", 0) or 0)
            == 0,
            "conservative_risk_selected_rate_in_target_band": TARGET_RATE_LOW
            <= _num(best_reweight.get("conservative_risk_selected_rate"))
            <= TARGET_RATE_HIGH,
            "risk_type_switch_spread_le_0_30": _num(best_reweight.get("risk_type_switch_spread"), 999.0) <= 0.30,
            "mixed_risk_combo_spread_le_0_30": _num(best_reweight.get("mixed_risk_combo_spread"), 999.0) <= 0.30,
            "single_risk_spread_le_0_30": _num(best_reweight.get("single_risk_spread"), 999.0) <= 0.30,
            "family_extreme_rate_count_eq_0": int(best_reweight.get("family_extreme_rate_count", 0) or 0) == 0,
            "rule_suppression_detected": bool(best_reweight.get("rule_suppression_detected", False)),
            "previous_rule_ppo_suppression_detected": bool(best_reweight.get("previous_rule_ppo_suppression_detected", False)),
            "dominant_component_after_reweight_not_energy": best_reweight.get("dominant_component_after_reweight")
            not in {"raw_energy_proxy", "projected_energy_proxy"},
        },
        "next_action": next_action,
        "readiness_reasons": readiness_reasons,
        "notes": [
            "v8173.3 is offline counterfactual design only.",
            "Energy proxy is not declared wrong; this audit tests whether risk-window energy gaps need conditioning.",
            "It does not validate reward, profit, safety improvement, or closed-loop controller quality.",
        ],
    }
    return report, {
        "energy_reweight_curve": curve_rows,
        "energy_gap_by_family": energy_gap_by_family,
        "energy_gap_by_combo": energy_gap_by_combo,
        "energy_reweight_switch_explainability": switch_rows,
        "energy_reweight_family_rate_table": family_rows,
        "energy_reweight_combo_rate_table": combo_rows,
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    boundary = report.get("boundary_runtime_safety", {})
    diagnostics = report.get("diagnostics", {})
    observed = diagnostics.get("observed_gap") or {}
    best_policy = observed.get("v81732_best_policy") or {}
    best = diagnostics.get("best_reweight_vector") or {}
    lines = [
        "# C-STCC v8173.3 Risk-Conditioned Energy Proxy Reweight Design",
        "",
        "Existing-trace counterfactual audit only. No online LLM, no rollout, no runtime scorer/template change.",
        "",
        "## Observed Gap",
        "",
        f"- v8173.2 best compromise: {best_policy.get('policy', 'none')} / {best_policy.get('vector_id', 'none')}",
        "- This is a target-band/spread compromise, not a claim of control-semantic optimality.",
        f"- Dominant v8173.2 component: {observed.get('v81732_dominant_component') or 'none'}",
        f"- dry_vent over-target: {observed.get('dry_vent_over_target')}",
        f"- canopy-dew/dew mixed under-target: {observed.get('canopy_dew_lt0_dew_under_target')}",
        "",
        "## Boundary",
        "",
        f"- Audit success rate: {_num(boundary.get('audit_success_rate')):.3f}",
        f"- Final-action invariant rate: {_num(boundary.get('final_action_invariant_rate')):.3f}",
        f"- Online LLM / rollout / real projection steps: {boundary.get('online_llm_called_steps', 0)} / {boundary.get('predictive_rollout_executed_steps', 0)} / {boundary.get('real_tomato_safety_projection_steps', 0)}",
        "",
        "## Best Reweight Vector",
        "",
        f"- Factor vector: {best.get('factor_vector_id', 'none')}",
        f"- Conservative risk selected rate: {_num(best.get('conservative_risk_selected_rate')):.3f}",
        f"- Family / mixed combo / single spread: {_num(best.get('risk_type_switch_spread')):.3f} / {_num(best.get('mixed_risk_combo_spread')):.3f} / {_num(best.get('single_risk_spread')):.3f}",
        f"- Dominant component after reweight: {best.get('dominant_component_after_reweight', 'none')}",
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
    base = "energy_proxy_reweight_design"
    paths = {
        "json": prefix.with_suffix(".json"),
        "md": prefix.with_suffix(".md"),
        "energy_reweight_curve_csv": (parent / stem.replace(base, "energy_reweight_curve")).with_suffix(".csv"),
        "energy_gap_by_family_csv": (parent / stem.replace(base, "energy_gap_by_family")).with_suffix(".csv"),
        "energy_gap_by_combo_csv": (parent / stem.replace(base, "energy_gap_by_combo")).with_suffix(".csv"),
        "energy_reweight_switch_explainability_csv": (
            parent / stem.replace(base, "energy_reweight_switch_explainability")
        ).with_suffix(".csv"),
    }
    paths["json"].write_text(_compact_json(report) + "\n", encoding="utf-8")
    paths["md"].write_text(build_markdown(report), encoding="utf-8")
    _write_csv(paths["energy_reweight_curve_csv"], tables.get("energy_reweight_curve", []))
    _write_csv(paths["energy_gap_by_family_csv"], tables.get("energy_gap_by_family", []))
    _write_csv(paths["energy_gap_by_combo_csv"], tables.get("energy_gap_by_combo", []))
    _write_csv(paths["energy_reweight_switch_explainability_csv"], tables.get("energy_reweight_switch_explainability", []))
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v8173.3 C-STCC energy proxy reweight design audit.")
    parser.add_argument("--trace-csv", type=str, default=str(DEFAULT_TRACE_CSV))
    parser.add_argument("--summary-json", type=str, default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--jsonl-root", type=str, default=str(DEFAULT_JSONL_ROOT))
    parser.add_argument("--v8173-json", type=str, default=str(DEFAULT_V8173_JSON))
    parser.add_argument("--v81731-json", type=str, default=str(DEFAULT_V81731_JSON))
    parser.add_argument("--v81732-json", type=str, default=str(DEFAULT_V81732_JSON))
    parser.add_argument("--v81732-family-csv", type=str, default=str(DEFAULT_V81732_FAMILY_CSV))
    parser.add_argument("--v81732-combo-csv", type=str, default=str(DEFAULT_V81732_COMBO_CSV))
    parser.add_argument("--v81732-component-csv", type=str, default=str(DEFAULT_V81732_COMPONENT_CSV))
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
        v81732_family_rows=_load_csv(args.v81732_family_csv),
        v81732_combo_rows=_load_csv(args.v81732_combo_csv),
        v81732_component_rows=_load_csv(args.v81732_component_csv),
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
