"""v8172 offline risk-aligned bonus calibration for C-STCC shadow traces.

This stage is an existing-trace counterfactual scorer audit. It does not run
runtime acquisition, call online LLM, execute predictive rollout, or change
final actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_TRACE_CSV = Path(
    "gl_gym/result/diagnostics/cstcc_v8171_conservative_allocation_trace_20260606_y2020_d240_s42_n720.csv"
)
DEFAULT_SUMMARY_JSON = Path(
    "gl_gym/result/diagnostics/cstcc_v8171_conservative_allocation_trace_20260606_y2020_d240_s42_n720.json"
)
DEFAULT_JSONL_ROOT = Path("logs/cstcc_shadow/v8171_conservative_allocation_20260606_y2020_d240_s42_n720")
DEFAULT_OUTPUT_PREFIX = Path(
    "gl_gym/result/diagnostics/cstcc_v8172_risk_aligned_bonus_calibration_20260606_y2020_d240_s42_n720"
)
BONUS_GRID = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12)
RISK_TYPE_KEYS = (
    "dew_risk",
    "humidity_or_dew_risk",
    "dry_risk",
    "high_vpd",
    "dry_or_high_vpd_risk",
    "dry_vent_risk",
    "canopy_dew_margin_lt1",
    "canopy_dew_margin_lt0",
)
BOUNDARY_FALSE_COUNTS = (
    "online_llm_called_steps",
    "predictive_rollout_executed_steps",
    "real_tomato_safety_projection_steps",
    "final_action_changed_steps",
)
COMPONENT_KEYS = (
    "base_final_score",
    "raw_smoothness",
    "raw_low_reversal",
    "raw_energy_proxy",
    "projected_smoothness",
    "projected_low_reversal",
    "projected_energy_proxy",
    "prior_confidence_bonus",
    "risk_static_hold_penalty",
    "rewrite_penalty",
    "uncertainty_penalty",
    "final_score",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except Exception:
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}
    return {}


def _load_csv(path: str | Path) -> list[dict[str, Any]]:
    input_path = _resolve(path)
    if not input_path.exists():
        return []
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json(path: str | Path) -> dict[str, Any]:
    input_path = _resolve(path)
    if not input_path.exists():
        return {}
    return json.loads(input_path.read_text(encoding="utf-8"))


def _load_jsonl_rows(root: str | Path) -> list[dict[str, Any]]:
    input_root = _resolve(root)
    rows: list[dict[str, Any]] = []
    if not input_root.exists():
        return rows
    for path in sorted(input_root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _compact_json(value) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            )


def _row_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    output: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        try:
            output[int(row.get("step", len(output)))] = row
        except Exception:
            continue
    return output


def _risk_flags(row: Mapping[str, Any]) -> dict[str, bool]:
    flags = _json_obj(row.get("risk_flags") or row.get("cstcc_shadow_risk_flags_json"))
    normalized = {str(key): _bool(value) for key, value in flags.items()} if flags else {}
    rh = _num(row.get("rh_air"), 70.0)
    vpd = _num(row.get("vpd_air"), _num(row.get("vpd_kpa")))
    canopy_dew = _num(row.get("canopy_dew_margin"), 999.0)
    dew_air = _num(row.get("dew_margin_air"), 999.0)
    normalized.setdefault("rh_ge_90", rh >= 90.0)
    normalized.setdefault("dew_risk", canopy_dew < 1.0 or dew_air < 1.0)
    normalized.setdefault("dry_risk", rh < 55.0)
    normalized.setdefault("high_vpd", vpd > 1.20)
    normalized["humidity_or_dew_risk"] = bool(
        normalized.get("humidity_or_dew_risk", False)
        or normalized.get("rh_ge_90", False)
        or normalized.get("dew_risk", False)
    )
    normalized["dry_or_high_vpd_risk"] = bool(
        normalized.get("dry_or_high_vpd_risk", False)
        or normalized.get("dry_risk", False)
        or normalized.get("high_vpd", False)
    )
    normalized["dry_vent_risk"] = bool(
        normalized.get("dry_vent_risk", False)
        or _bool(row.get("dry_vent_risk"))
        or _num(row.get("u_ventilation"), 0.0) > 0.70 and normalized.get("dry_risk", False)
    )
    normalized["canopy_dew_margin_lt1"] = bool(canopy_dew < 1.0)
    normalized["canopy_dew_margin_lt0"] = bool(canopy_dew < 0.0)
    normalized["any_risk"] = bool(
        normalized.get("any_risk", False)
        or any(value for key, value in normalized.items() if key != "any_risk")
    )
    return normalized


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


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


def _quantile_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean": float(mean(values)) if values else 0.0,
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "max": max(values) if values else 0.0,
    }


def _parse_candidate_source(candidate_id: str) -> tuple[str, str]:
    parts = str(candidate_id or "").split(":")
    return (parts[0] if parts else "", parts[1] if len(parts) > 1 else "")


def _candidate_info(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for candidate in row.get("candidate_violation_provenance") or []:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            continue
        source, template = _parse_candidate_source(candidate_id)
        output[candidate_id] = {
            "candidate_id": candidate_id,
            "source_prior": str(candidate.get("source_prior") or source),
            "template_name": str(candidate.get("template_name") or template),
            "feasible": bool(candidate.get("feasible", False)),
        }
    return output


def _score_components(row: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    raw = row.get("score_component_by_candidate") or {}
    if not isinstance(raw, Mapping):
        return {}
    output: dict[str, dict[str, float]] = {}
    for candidate_id, components in raw.items():
        if not isinstance(components, Mapping):
            continue
        output[str(candidate_id)] = {
            key: _num(components.get(key), 0.0)
            for key in COMPONENT_KEYS
            if key in components or key == "final_score"
        }
    return output


def _eligible_for_bonus(candidate: Mapping[str, Any], flags: Mapping[str, bool]) -> bool:
    return bool(
        flags.get("any_risk", False)
        and candidate.get("feasible", False)
        and candidate.get("source_prior") == "conservative_prior"
        and candidate.get("template_name") == "conservative_stabilize"
    )


def _winner_for_bonus(
    *,
    components: Mapping[str, Mapping[str, float]],
    candidates: Mapping[str, Mapping[str, Any]],
    flags: Mapping[str, bool],
    bonus: float,
) -> tuple[str, float]:
    best_id = ""
    best_score = -1e18
    for candidate_id, component in components.items():
        candidate = candidates.get(candidate_id)
        if candidate is None:
            source, template = _parse_candidate_source(candidate_id)
            candidate = {"source_prior": source, "template_name": template, "feasible": False}
        if not candidate.get("feasible", False):
            continue
        score = _num(component.get("final_score"), -1e18)
        if _eligible_for_bonus(candidate, flags):
            score += float(bonus)
        if score > best_score:
            best_id = candidate_id
            best_score = score
    return best_id, best_score


def _boundary_summary(summary_json: Mapping[str, Any], jsonl_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trace = summary_json.get("trace_summary", {}) if isinstance(summary_json.get("trace_summary", {}), Mapping) else {}
    answers = summary_json.get("answers", {}) if isinstance(summary_json.get("answers", {}), Mapping) else {}
    return {
        "jsonl_row_count": len(jsonl_rows),
        "audit_success_rate": _num(trace.get("cstcc_shadow_audit_success_rate"), _num(answers.get("audit_success_rate"))),
        "final_action_invariant_rate": _num(
            trace.get("cstcc_shadow_final_action_invariant_rate"),
            _num(answers.get("final_action_invariant_rate")),
        ),
        "final_action_changed_steps": int(_num(trace.get("cstcc_shadow_final_action_changed_steps"))),
        "online_llm_called_steps": int(_num(trace.get("cstcc_shadow_online_llm_called_steps"))),
        "predictive_rollout_executed_steps": int(_num(trace.get("cstcc_shadow_predictive_rollout_executed_steps"))),
        "real_tomato_safety_projection_steps": int(_num(trace.get("cstcc_shadow_real_tomato_safety_projection_steps"))),
        "fallback_rate": _num(trace.get("cstcc_shadow_fallback_rate"), _num(answers.get("fallback_rate"))),
        "mean_infeasible_candidate_ratio": _num(trace.get("cstcc_shadow_mean_infeasible_candidate_ratio")),
        "previous_shift_all_max_delta_count": int(_num(trace.get("cstcc_shadow_previous_shift_all_max_delta_count"))),
        "selected_candidate_max_delta_count": int(_num(trace.get("cstcc_shadow_selected_candidate_violation_count"))),
        "json_row_size_kb_max": _num(answers.get("json_size_kb_max")),
    }


def _missing_schema_fields(jsonl_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not jsonl_rows:
        return ["jsonl_rows_missing"]
    first_success = next((row for row in jsonl_rows if row.get("audit_success", False)), jsonl_rows[0])
    required = (
        "candidate_violation_provenance",
        "score_component_by_candidate",
        "score_margin_to_selected_by_prior",
        "risk_flags",
        "shadow_selected_sequence_id",
    )
    missing = [f"missing_field:{field}" for field in required if field not in first_success]
    if not _json_obj(first_success.get("score_component_by_candidate")):
        missing.append("missing_or_empty_field:score_component_by_candidate")
    return missing


def _component_gap_quantiles(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in COMPONENT_KEYS:
        values = [
            _num((record.get("selected_minus_competitor") or {}).get(key))
            for record in records
            if isinstance(record.get("selected_minus_competitor") or {}, Mapping)
        ]
        if not values:
            continue
        rows.append({"component": key, **_quantile_summary(values)})
    return rows


def evaluate_bonus_curve(
    *,
    trace_csv_rows: Sequence[Mapping[str, Any]],
    jsonl_rows: Sequence[Mapping[str, Any]],
    bonus_grid: Sequence[float] = BONUS_GRID,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    csv_by_step = _row_by_step(trace_csv_rows)
    success_rows = [row for row in jsonl_rows if row.get("audit_success", False)]
    risk_rows = 0
    conservative_risk_candidates = 0
    conservative_risk_feasible = 0
    conservative_margin_records: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    per_bonus: dict[float, dict[str, Any]] = {
        float(bonus): {
            "bonus": float(bonus),
            "row_count": len(success_rows),
            "risk_row_count": 0,
            "conservative_selected_count": 0,
            "conservative_risk_selected_count": 0,
            "nonrisk_conservative_selected_count": 0,
            "selected_switch_count": 0,
            "switch_from_previous_to_conservative_count": 0,
            "switch_from_rule_to_conservative_count": 0,
            "switch_from_ppo_to_conservative_count": 0,
            "switch_from_other_to_conservative_count": 0,
        }
        for bonus in bonus_grid
    }
    risk_type_counts: dict[tuple[float, str], dict[str, Any]] = {}

    for row in success_rows:
        step = int(_num(row.get("step"), len(switch_rows)))
        csv_row = csv_by_step.get(step, {})
        merged = {**dict(csv_row), **dict(row)}
        flags = _risk_flags(merged)
        candidates = _candidate_info(row)
        components = _score_components(row)
        selected_id = str(row.get("shadow_selected_sequence_id") or "")
        selected_source, selected_template = _parse_candidate_source(selected_id)
        baseline_id, baseline_score = _winner_for_bonus(
            components=components,
            candidates=candidates,
            flags=flags,
            bonus=0.0,
        )
        baseline_id = selected_id or baseline_id
        baseline_source, baseline_template = _parse_candidate_source(baseline_id)
        margins = _json_obj(row.get("score_margin_to_selected_by_prior"))
        conservative_margin = margins.get("conservative_prior") if isinstance(margins.get("conservative_prior"), Mapping) else {}
        conservative_total_margin = conservative_margin.get("total_margin")
        conservative_best_id = str(conservative_margin.get("best_candidate_id") or "")
        conservative_best = candidates.get(conservative_best_id, {})
        if flags.get("any_risk", False):
            risk_rows += 1
            if conservative_best_id:
                conservative_risk_candidates += 1
            if conservative_best_id and conservative_best.get("feasible", False):
                conservative_risk_feasible += 1
        if conservative_total_margin is not None:
            conservative_margin_records.append(
                {
                    "step": step,
                    "risk_flags": flags,
                    "total_margin": _num(conservative_total_margin),
                    "selected_candidate_id": conservative_margin.get("selected_candidate_id") or baseline_id,
                    "selected_source_prior": conservative_margin.get("selected_source_prior") or baseline_source,
                    "selected_template_name": conservative_margin.get("selected_template_name") or baseline_template,
                    "conservative_best_candidate_id": conservative_best_id,
                    "conservative_best_template_name": conservative_margin.get("best_template_name"),
                    "selected_minus_competitor": conservative_margin.get("selected_minus_competitor", {}),
                }
            )
        for bonus in bonus_grid:
            bonus = float(bonus)
            curve = per_bonus[bonus]
            if flags.get("any_risk", False):
                curve["risk_row_count"] += 1
            winner_id, winner_score = _winner_for_bonus(
                components=components,
                candidates=candidates,
                flags=flags,
                bonus=bonus,
            )
            winner_source, winner_template = _parse_candidate_source(winner_id)
            winner_is_conservative = winner_source == "conservative_prior" and winner_template == "conservative_stabilize"
            if winner_is_conservative:
                curve["conservative_selected_count"] += 1
                if flags.get("any_risk", False):
                    curve["conservative_risk_selected_count"] += 1
                else:
                    curve["nonrisk_conservative_selected_count"] += 1
            switched = bool(winner_id and winner_id != baseline_id)
            if switched:
                curve["selected_switch_count"] += 1
                if winner_is_conservative:
                    if baseline_source == "previous_plan_prior":
                        curve["switch_from_previous_to_conservative_count"] += 1
                    elif baseline_source == "rule_prior":
                        curve["switch_from_rule_to_conservative_count"] += 1
                    elif baseline_source == "ppo_prior":
                        curve["switch_from_ppo_to_conservative_count"] += 1
                    else:
                        curve["switch_from_other_to_conservative_count"] += 1
                    switch_rows.append(
                        {
                            "bonus": bonus,
                            "step": step,
                            "hour_of_day": _num(merged.get("hour_of_day")),
                            "active_regime": row.get("active_regime_after") or merged.get("cstcc_shadow_active_regime_after"),
                            "risk_flags": flags,
                            "baseline_candidate_id": baseline_id,
                            "baseline_source_prior": baseline_source,
                            "baseline_template_name": baseline_template,
                            "counterfactual_winner_id": winner_id,
                            "counterfactual_winner_score": winner_score,
                            "baseline_selected_score": baseline_score,
                            "conservative_margin": _num(conservative_total_margin),
                            "selected_minus_conservative": conservative_margin.get("selected_minus_competitor", {}),
                            "temp_air": _num(merged.get("temp_air")),
                            "rh_air": _num(merged.get("rh_air")),
                            "vpd_air": _num(merged.get("vpd_air")),
                            "canopy_dew_margin": _num(merged.get("canopy_dew_margin"), 999.0),
                            "dew_margin_air": _num(merged.get("dew_margin_air"), 999.0),
                        }
                    )
            for risk_type in RISK_TYPE_KEYS:
                key = (bonus, risk_type)
                item = risk_type_counts.setdefault(
                    key,
                    {
                        "bonus": bonus,
                        "risk_type": risk_type,
                        "risk_row_count": 0,
                        "conservative_selected_count": 0,
                        "selected_switch_count": 0,
                    },
                )
                if flags.get(risk_type, False):
                    item["risk_row_count"] += 1
                    if winner_is_conservative:
                        item["conservative_selected_count"] += 1
                    if switched:
                        item["selected_switch_count"] += 1

    curve_rows: list[dict[str, Any]] = []
    for bonus, curve in sorted(per_bonus.items()):
        risk_count = int(curve["risk_row_count"] or 0)
        row = dict(curve)
        row["conservative_risk_selected_rate"] = row["conservative_risk_selected_count"] / max(risk_count, 1)
        row["selected_switch_rate"] = row["selected_switch_count"] / max(int(row["row_count"]), 1)
        row["nonrisk_conservative_selected_count"] = int(row["nonrisk_conservative_selected_count"])
        curve_rows.append(row)

    risk_type_rows: list[dict[str, Any]] = []
    for (_bonus, _risk_type), item in sorted(risk_type_counts.items()):
        item = dict(item)
        item["conservative_selected_rate"] = item["conservative_selected_count"] / max(int(item["risk_row_count"]), 1)
        item["selected_switch_rate"] = item["selected_switch_count"] / max(int(item["risk_row_count"]), 1)
        risk_type_rows.append(item)

    component_rows = _component_gap_quantiles(conservative_margin_records)
    diagnostics = {
        "row_count": len(success_rows),
        "risk_row_count": risk_rows,
        "conservative_risk_candidate_steps": conservative_risk_candidates,
        "conservative_risk_feasible_steps": conservative_risk_feasible,
        "conservative_risk_margin_coverage": len(conservative_margin_records) / max(risk_rows, 1),
        "conservative_margin_quantiles": _quantile_summary([
            _num(record.get("total_margin")) for record in conservative_margin_records
        ]),
        "component_dominant_gap": _dominant_component(component_rows),
        "component_gap_quantiles": component_rows,
        "bonus_grid": list(float(bonus) for bonus in bonus_grid),
        "monotonic_or_near_monotonic": _near_monotonic([
            float(row["conservative_risk_selected_rate"]) for row in curve_rows
        ]),
        "nonrisk_conservative_selected_count_max": max(
            (int(row["nonrisk_conservative_selected_count"]) for row in curve_rows),
            default=0,
        ),
        "best_bonus_0_06_to_0_10": _best_bonus_in_range(curve_rows, 0.06, 0.10),
        "risk_type_switch_divergence": _risk_type_divergence(risk_type_rows),
    }
    return curve_rows, risk_type_rows, switch_rows, component_rows, diagnostics


def _dominant_component(component_rows: Sequence[Mapping[str, Any]]) -> str:
    if not component_rows:
        return ""
    return str(max(component_rows, key=lambda row: abs(_num(row.get("mean")))).get("component") or "")


def _near_monotonic(values: Sequence[float]) -> bool:
    previous = -1e-9
    violations = 0
    for value in values:
        if value + 1e-9 < previous:
            violations += 1
        previous = max(previous, value)
    return violations <= 1


def _best_bonus_in_range(curve_rows: Sequence[Mapping[str, Any]], low: float, high: float) -> dict[str, Any]:
    candidates = [
        dict(row)
        for row in curve_rows
        if low - 1e-9 <= float(row.get("bonus", 0.0)) <= high + 1e-9
    ]
    if not candidates:
        return {}
    return max(candidates, key=lambda row: (float(row.get("conservative_risk_selected_rate", 0.0)), -float(row.get("bonus", 0.0))))


def _risk_type_divergence(risk_type_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_bonus: dict[float, list[float]] = {}
    for row in risk_type_rows:
        if int(row.get("risk_row_count", 0) or 0) <= 0:
            continue
        bonus = float(row.get("bonus", 0.0) or 0.0)
        by_bonus.setdefault(bonus, []).append(float(row.get("conservative_selected_rate", 0.0) or 0.0))
    max_spread = 0.0
    max_bonus = 0.0
    for bonus, values in by_bonus.items():
        if len(values) < 2:
            continue
        spread = max(values) - min(values)
        if spread > max_spread:
            max_spread = spread
            max_bonus = bonus
    return {"max_spread": max_spread, "max_spread_bonus": max_bonus, "sharp": max_spread >= 0.60}


def _route_next_action(
    *,
    boundary: Mapping[str, Any],
    schema_issues: Sequence[str],
    diagnostics: Mapping[str, Any],
    curve_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    if (
        float(boundary.get("audit_success_rate", 0.0)) < 1.0
        or float(boundary.get("final_action_invariant_rate", 0.0)) < 1.0
        or any(int(boundary.get(field, 0) or 0) for field in BOUNDARY_FALSE_COUNTS)
        or float(boundary.get("fallback_rate", 0.0) or 0.0) > 0.0
        or float(boundary.get("mean_infeasible_candidate_ratio", 0.0) or 0.0) > 0.0
        or int(boundary.get("previous_shift_all_max_delta_count", 0) or 0) > 0
        or int(boundary.get("selected_candidate_max_delta_count", 0) or 0) > 0
    ):
        return "stop_and_repair_cstcc_shadow_boundary", ["boundary_or_candidate_health_failure"]
    if schema_issues:
        return "v8172_schema_repair_plan", list(schema_issues)
    if int(diagnostics.get("conservative_risk_candidate_steps", 0) or 0) < int(diagnostics.get("risk_row_count", 0) or 0):
        return "v8172_schema_repair_plan", ["conservative_risk_candidate_coverage_missing"]
    if int(diagnostics.get("nonrisk_conservative_selected_count_max", 0) or 0) > 0:
        return "v8172_energy_proxy_reweight_design_plan", ["nonrisk_conservative_selection_detected"]
    if bool((diagnostics.get("risk_type_switch_divergence") or {}).get("sharp", False)):
        return "v8173_risk_type_specific_bonus_or_template_plan", ["risk_type_switch_divergence_sharp"]
    best = diagnostics.get("best_bonus_0_06_to_0_10") or {}
    if float(best.get("conservative_risk_selected_rate", 0.0) or 0.0) > 0.0:
        return "v8172_1_opt_in_bonus_shadow_trace_plan", ["interpretable_bonus_curve_in_0_06_to_0_10"]
    max_bonus_row = max(curve_rows, key=lambda row: float(row.get("bonus", 0.0)), default={})
    if float(max_bonus_row.get("conservative_risk_selected_rate", 0.0) or 0.0) <= 0.0:
        return "v8172_energy_proxy_reweight_design_plan", ["bonus_grid_no_conservative_switch"]
    if str(diagnostics.get("component_dominant_gap") or "") in {"projected_energy_proxy", "raw_energy_proxy"}:
        return "v8172_energy_proxy_reweight_design_plan", ["energy_proxy_dominant_gap"]
    return "v8172_1_opt_in_bonus_shadow_trace_plan", ["bonus_curve_requires_followup_shadow_trace"]


def build_report(
    *,
    trace_csv_rows: list[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    jsonl_rows: list[Mapping[str, Any]],
    bonus_grid: Sequence[float] = BONUS_GRID,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    boundary = _boundary_summary(summary_json, jsonl_rows)
    schema_issues = _missing_schema_fields(jsonl_rows)
    curve_rows, risk_type_rows, switch_rows, component_rows, diagnostics = evaluate_bonus_curve(
        trace_csv_rows=trace_csv_rows,
        jsonl_rows=jsonl_rows,
        bonus_grid=bonus_grid,
    )
    next_action, readiness_reasons = _route_next_action(
        boundary=boundary,
        schema_issues=schema_issues,
        diagnostics=diagnostics,
        curve_rows=curve_rows,
    )
    report = {
        "schema_version": "cstcc_v8172_risk_aligned_bonus_calibration_v1",
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
            "default_controller_changed": False,
        },
        "boundary_runtime_safety": boundary,
        "schema_issues": schema_issues,
        "counterfactual_definition": {
            "bonus_applies_to": "feasible conservative_prior:conservative_stabilize in risk_flags.any_risk rows only",
            "bonus_grid": list(float(bonus) for bonus in bonus_grid),
            "runtime_final_action_changed": False,
            "baseline_score_dual_changed": False,
            "prior_confidence_changed": False,
            "rule_prior_changed": False,
        },
        "diagnostics": diagnostics,
        "acceptance": {
            "boundary_clean": next_action != "stop_and_repair_cstcc_shadow_boundary",
            "conservative_risk_candidate_coverage_full": diagnostics.get("conservative_risk_candidate_steps", 0)
            == diagnostics.get("risk_row_count", 0),
            "conservative_risk_margin_coverage_full": diagnostics.get("conservative_risk_margin_coverage", 0.0) >= 0.999,
            "monotonic_or_near_monotonic": diagnostics.get("monotonic_or_near_monotonic", False),
            "nonrisk_conservative_selected_count_eq_0": diagnostics.get("nonrisk_conservative_selected_count_max", 0) == 0,
        },
        "next_action": next_action,
        "readiness_reasons": readiness_reasons,
        "notes": [
            "v8172 is offline counterfactual scorer calibration only.",
            "It does not validate reward, profit, safety improvement, or closed-loop controller quality.",
        ],
    }
    return report, {
        "bonus_curve": curve_rows,
        "bonus_curve_by_risk_type": risk_type_rows,
        "switch_steps_topk": sorted(
            switch_rows,
            key=lambda row: (float(row.get("bonus", 0.0)), -float(row.get("conservative_margin", 0.0))),
        )[:500],
        "component_gap_quantiles": component_rows,
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    boundary = report.get("boundary_runtime_safety", {})
    diagnostics = report.get("diagnostics", {})
    quantiles = diagnostics.get("conservative_margin_quantiles", {})
    best = diagnostics.get("best_bonus_0_06_to_0_10", {})
    divergence = diagnostics.get("risk_type_switch_divergence", {})
    lines = [
        "# C-STCC v8172 Risk-Aligned Bonus Calibration",
        "",
        "Existing-trace counterfactual scorer audit only. No online LLM, no rollout, no final-action change.",
        "",
        "## Boundary",
        "",
        f"- Audit success rate: {float(boundary.get('audit_success_rate', 0.0)):.3f}",
        f"- Final-action invariant rate: {float(boundary.get('final_action_invariant_rate', 0.0)):.3f}",
        f"- Online LLM / rollout / real projection steps: {boundary.get('online_llm_called_steps', 0)} / {boundary.get('predictive_rollout_executed_steps', 0)} / {boundary.get('real_tomato_safety_projection_steps', 0)}",
        f"- Final action changed steps: {boundary.get('final_action_changed_steps', 0)}",
        "",
        "## Calibration",
        "",
        f"- Risk rows: {diagnostics.get('risk_row_count', 0)}",
        f"- Conservative risk candidate / feasible steps: {diagnostics.get('conservative_risk_candidate_steps', 0)} / {diagnostics.get('conservative_risk_feasible_steps', 0)}",
        f"- Conservative risk margin p50/p75/p90/p95/max: {float(quantiles.get('p50', 0.0)):.6f} / {float(quantiles.get('p75', 0.0)):.6f} / {float(quantiles.get('p90', 0.0)):.6f} / {float(quantiles.get('p95', 0.0)):.6f} / {float(quantiles.get('max', 0.0)):.6f}",
        f"- Dominant component gap: {diagnostics.get('component_dominant_gap') or 'none'}",
        f"- Best 0.06-0.10 bonus selected rate: {float(best.get('conservative_risk_selected_rate', 0.0)):.3f} at bonus {float(best.get('bonus', 0.0)):.2f}",
        f"- Nonrisk conservative selected count max: {diagnostics.get('nonrisk_conservative_selected_count_max', 0)}",
        f"- Risk-type switch spread: {float(divergence.get('max_spread', 0.0)):.3f} at bonus {float(divergence.get('max_spread_bonus', 0.0)):.2f}",
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
    paths = {
        "json": prefix.with_suffix(".json"),
        "md": prefix.with_suffix(".md"),
        "bonus_curve_csv": (parent / stem.replace("risk_aligned_bonus_calibration", "bonus_curve")).with_suffix(".csv"),
        "bonus_curve_by_risk_type_csv": (
            parent / stem.replace("risk_aligned_bonus_calibration", "bonus_curve_by_risk_type")
        ).with_suffix(".csv"),
        "switch_steps_topk_csv": (
            parent / stem.replace("risk_aligned_bonus_calibration", "switch_steps_topk")
        ).with_suffix(".csv"),
        "component_gap_quantiles_csv": (
            parent / stem.replace("risk_aligned_bonus_calibration", "component_gap_quantiles")
        ).with_suffix(".csv"),
    }
    paths["json"].write_text(_compact_json(report) + "\n", encoding="utf-8")
    paths["md"].write_text(build_markdown(report), encoding="utf-8")
    _write_csv(paths["bonus_curve_csv"], tables.get("bonus_curve", []))
    _write_csv(paths["bonus_curve_by_risk_type_csv"], tables.get("bonus_curve_by_risk_type", []))
    _write_csv(paths["switch_steps_topk_csv"], tables.get("switch_steps_topk", []))
    _write_csv(paths["component_gap_quantiles_csv"], tables.get("component_gap_quantiles", []))
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v8172 C-STCC risk-aligned bonus calibration.")
    parser.add_argument("--trace-csv", type=str, default=str(DEFAULT_TRACE_CSV))
    parser.add_argument("--summary-json", type=str, default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--jsonl-root", type=str, default=str(DEFAULT_JSONL_ROOT))
    parser.add_argument("--output-prefix", type=str, default=str(DEFAULT_OUTPUT_PREFIX))
    args = parser.parse_args()

    report, tables = build_report(
        trace_csv_rows=_load_csv(args.trace_csv),
        summary_json=_load_json(args.summary_json),
        jsonl_rows=_load_jsonl_rows(args.jsonl_root),
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
