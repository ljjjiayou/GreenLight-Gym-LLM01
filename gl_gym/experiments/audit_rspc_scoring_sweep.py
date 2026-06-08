"""Audit RSPC scoring-weight sweep results by preset.

This script is read-only. It compares opt-in scoring presets against the
``balanced`` reference under frozen replay and rejects presets that improve one
metric by quietly trading away runtime/cache/dew/canopy/temperature safety.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


METRIC_FIELDS = [
    "total_reward",
    "total_profit",
    "total_rh_low_violation",
    "total_rh_high_violation",
    "total_vpd_high_excess",
    "total_temp_violation",
    "dry_risk_steps",
    "dry_vent_risk_steps",
    "dew_risk_steps",
    "dew_margin_air_lt0_steps",
    "canopy_dew_margin_lt0_steps",
    "canopy_dew_margin_lt1_steps",
    "runtime_error_steps",
    "strict_cache_miss_runtime_error_steps",
    "plan_cache_enabled_steps",
    "plan_cache_hit_steps",
    "plan_cache_hit_rate",
]

DELTA_FIELDS = [
    "total_reward",
    "total_profit",
    "total_rh_low_violation",
    "total_vpd_high_excess",
    "total_temp_violation",
    "dew_margin_air_lt0_steps",
    "canopy_dew_margin_lt0_steps",
    "canopy_dew_margin_lt1_steps",
    "dry_vent_risk_steps",
]

RELATIVE_DELTA_FIELDS = [
    "total_reward",
    "total_rh_low_violation",
    "total_vpd_high_excess",
    "total_temp_violation",
]

EPS = 1e-9


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalise_sweep_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "preset": str(raw.get("preset", "") or ""),
        "scenario_id": str(raw.get("scenario_id", "") or ""),
        "controller": str(raw.get("controller", "") or ""),
        "source": str(raw.get("source", "") or ""),
    }
    for field in METRIC_FIELDS:
        row[field] = _num(raw.get(field, 0.0))
    enabled = row["plan_cache_enabled_steps"]
    hits = row["plan_cache_hit_steps"]
    if enabled > 0:
        row["plan_cache_hit_rate"] = hits / enabled
    return row


def _normalise_benchmark_summary(summary: Mapping[str, Any], preset: str, source: str) -> dict[str, Any]:
    aggregate = summary.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        aggregate = {}
    row = {
        "preset": str(preset),
        "scenario_id": str(summary.get("scenario_id", "") or ""),
        "controller": str(summary.get("controller", "") or ""),
        "source": str(source),
    }
    for field in METRIC_FIELDS:
        row[field] = _num(aggregate.get(field, 0.0))
    enabled = row["plan_cache_enabled_steps"]
    hits = row["plan_cache_hit_steps"]
    row["plan_cache_hit_rate"] = hits / enabled if enabled > 0 else _num(aggregate.get("plan_cache_hit_rate", 0.0))
    return row


def _rows_from_sweep_payload(payload: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []
    out = []
    for raw in rows:
        if isinstance(raw, Mapping):
            row = dict(raw)
            row.setdefault("source", source)
            out.append(_normalise_sweep_row(row))
    return out


def _rows_from_benchmark_payload(payload: Mapping[str, Any], preset: str, source: str) -> list[dict[str, Any]]:
    summaries = payload.get("summaries", []) if isinstance(payload, Mapping) else []
    if not isinstance(summaries, list):
        return []
    return [
        _normalise_benchmark_summary(summary, preset=preset, source=source)
        for summary in summaries
        if isinstance(summary, Mapping)
    ]


def load_rows(
    *,
    input_paths: Sequence[str | Path] = (),
    benchmark_json_paths: Sequence[str | Path] = (),
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for raw_path in input_paths:
        path = Path(raw_path)
        try:
            payload = _load_json(path)
        except Exception as exc:
            warnings.append(f"failed_to_read_input:{path}:{exc}")
            continue
        if isinstance(payload, Mapping):
            loaded = _rows_from_sweep_payload(payload, str(path))
            rows.extend(loaded)
            if not loaded:
                warnings.append(f"input_has_no_sweep_rows:{path}")
        else:
            warnings.append(f"input_not_json_object:{path}")
    for raw_path in benchmark_json_paths:
        path = Path(raw_path)
        try:
            payload = _load_json(path)
        except Exception as exc:
            warnings.append(f"failed_to_read_benchmark:{path}:{exc}")
            continue
        preset = path.stem
        loaded = _rows_from_benchmark_payload(payload, preset=preset, source=str(path))
        rows.extend(loaded)
        if not loaded:
            warnings.append(f"benchmark_has_no_summaries:{path}")
    return rows, warnings


def _limit_1pct(reference: float) -> float:
    if abs(reference) < EPS:
        return 1e-6
    return abs(reference) * 0.01 + EPS


def _relative_delta_pct(delta: float, reference: float) -> float:
    if abs(reference) < EPS:
        return 0.0 if abs(delta) < EPS else 100.0
    return float(delta / abs(reference) * 100.0)


def _row_gate_failures(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    preset = str(row.get("preset", ""))
    scenario = str(row.get("scenario_id", ""))
    controller = str(row.get("controller", ""))
    if _num(row.get("runtime_error_steps")) > 0:
        failures.append(
            {
                "preset": preset,
                "scenario_id": scenario,
                "controller": controller,
                "reason": "runtime_error_steps",
                "severity": "protocol",
                "value": _num(row.get("runtime_error_steps")),
            }
        )
    if _num(row.get("strict_cache_miss_runtime_error_steps")) > 0:
        failures.append(
            {
                "preset": preset,
                "scenario_id": scenario,
                "controller": controller,
                "reason": "strict_cache_miss_runtime_error_steps",
                "severity": "protocol",
                "value": _num(row.get("strict_cache_miss_runtime_error_steps")),
            }
        )
    enabled = _num(row.get("plan_cache_enabled_steps"))
    hits = _num(row.get("plan_cache_hit_steps"))
    if enabled > 0 and hits + EPS < enabled:
        failures.append(
            {
                "preset": preset,
                "scenario_id": scenario,
                "controller": controller,
                "reason": "plan_cache_hit_rate_below_1",
                "severity": "protocol",
                "value": hits / enabled,
            }
        )
    return failures


def _paired_failure(compare: Mapping[str, Any], reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    preset = str(compare.get("preset", ""))
    scenario = str(compare.get("scenario_id", ""))
    controller = str(compare.get("controller", ""))

    def add(reason: str, severity: str, delta: float) -> None:
        failures.append(
            {
                "preset": preset,
                "scenario_id": scenario,
                "controller": controller,
                "reason": reason,
                "severity": severity,
                "delta": float(delta),
            }
        )

    dew_delta = _num(compare.get("dew_margin_air_lt0_steps")) - _num(reference.get("dew_margin_air_lt0_steps"))
    canopy_delta = _num(compare.get("canopy_dew_margin_lt0_steps")) - _num(reference.get("canopy_dew_margin_lt0_steps"))
    temp_delta = _num(compare.get("total_temp_violation")) - _num(reference.get("total_temp_violation"))
    rh_low_delta = _num(compare.get("total_rh_low_violation")) - _num(reference.get("total_rh_low_violation"))
    vpd_delta = _num(compare.get("total_vpd_high_excess")) - _num(reference.get("total_vpd_high_excess"))

    if dew_delta > EPS:
        add("dew_margin_air_lt0_regression", "safety", dew_delta)
    if canopy_delta > EPS:
        add("canopy_dew_margin_lt0_regression", "safety", canopy_delta)
    if temp_delta > _limit_1pct(_num(reference.get("total_temp_violation"))):
        add("temp_violation_gt_1pct", "safety", temp_delta)

    rh_low_improved = (
        rh_low_delta < -_limit_1pct(_num(reference.get("total_rh_low_violation")))
        or rh_low_delta <= -1.0
    )
    safety_ok = dew_delta <= EPS and canopy_delta <= EPS and temp_delta <= _limit_1pct(
        _num(reference.get("total_temp_violation"))
    )
    if vpd_delta > _limit_1pct(_num(reference.get("total_vpd_high_excess"))) and not (
        rh_low_improved and safety_ok
    ):
        add("vpd_high_gt_1pct_without_rh_low_tradeoff", "performance", vpd_delta)
    if rh_low_delta > _limit_1pct(_num(reference.get("total_rh_low_violation"))):
        add("rh_low_gt_1pct", "performance", rh_low_delta)

    return failures


def _paired_soft_warnings(compare: Mapping[str, Any], reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    preset = str(compare.get("preset", ""))
    scenario = str(compare.get("scenario_id", ""))
    controller = str(compare.get("controller", ""))

    def add(reason: str, delta: float, severity: str = "warning") -> None:
        warnings.append(
            {
                "preset": preset,
                "scenario_id": scenario,
                "controller": controller,
                "reason": reason,
                "severity": severity,
                "delta": float(delta),
            }
        )

    reward_delta = _num(compare.get("total_reward")) - _num(reference.get("total_reward"))
    vpd_delta = _num(compare.get("total_vpd_high_excess")) - _num(reference.get("total_vpd_high_excess"))
    temp_delta = _num(compare.get("total_temp_violation")) - _num(reference.get("total_temp_violation"))
    dew_delta = _num(compare.get("dew_margin_air_lt0_steps")) - _num(reference.get("dew_margin_air_lt0_steps"))
    canopy_delta = _num(compare.get("canopy_dew_margin_lt0_steps")) - _num(reference.get("canopy_dew_margin_lt0_steps"))

    if vpd_delta > EPS:
        add("per_scenario_vpd_tradeoff", vpd_delta)
    if temp_delta > EPS:
        add("per_scenario_temp_tradeoff", temp_delta)
    if reward_delta > EPS and (vpd_delta > EPS or temp_delta > EPS or dew_delta > EPS or canopy_delta > EPS):
        add("reward_improved_with_safety_or_vpd_tradeoff", reward_delta)
    return warnings


def _build_pairs(rows: Sequence[Mapping[str, Any]], reference_preset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {(str(row["preset"]), str(row["scenario_id"]), str(row["controller"])): row for row in rows}
    reference_rows = {
        (str(row["scenario_id"]), str(row["controller"])): row
        for row in rows
        if str(row.get("preset", "")) == reference_preset
    }
    pairs: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in rows:
        preset = str(row.get("preset", ""))
        if preset == reference_preset:
            continue
        key = (str(row["scenario_id"]), str(row["controller"]))
        reference = reference_rows.get(key)
        if reference is None:
            missing.append(
                {
                    "preset": preset,
                    "scenario_id": key[0],
                    "controller": key[1],
                    "missing_reference_preset": reference_preset,
                }
            )
            continue
        delta = {
            "preset": preset,
            "scenario_id": key[0],
            "controller": key[1],
            "reference_preset": reference_preset,
        }
        for field in DELTA_FIELDS:
            diff = _num(row.get(field)) - _num(reference.get(field))
            delta[f"d_{field}"] = diff
            if field in RELATIVE_DELTA_FIELDS:
                delta[f"relative_d_{field}_pct"] = _relative_delta_pct(diff, _num(reference.get(field)))
        pairs.append(delta)
    return pairs, missing


def _aggregate_by_preset(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    soft_warnings: Sequence[Mapping[str, Any]],
    reference_preset: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("preset", ""))].append(row)
    pair_grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        pair_grouped[str(pair.get("preset", ""))].append(pair)
    failure_grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for failure in failures:
        failure_grouped[str(failure.get("preset", ""))].append(failure)
    warning_grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for warning in soft_warnings:
        warning_grouped[str(warning.get("preset", ""))].append(warning)

    reference_totals = {
        field: sum(_num(row.get(field)) for row in grouped.get(reference_preset, []))
        for field in RELATIVE_DELTA_FIELDS
    }

    out: dict[str, dict[str, Any]] = {}
    for preset, preset_rows in sorted(grouped.items()):
        enabled = sum(_num(row.get("plan_cache_enabled_steps")) for row in preset_rows)
        hits = sum(_num(row.get("plan_cache_hit_steps")) for row in preset_rows)
        summary: dict[str, Any] = {
            "scenario_count": len({str(row.get("scenario_id", "")) for row in preset_rows}),
            "row_count": len(preset_rows),
            "plan_cache_hit_rate": hits / enabled if enabled > 0 else 0.0,
            "failure_count": len(failure_grouped.get(preset, [])),
            "failure_reasons": sorted({str(item.get("reason", "")) for item in failure_grouped.get(preset, [])}),
            "per_scenario_warning_count": len(warning_grouped.get(preset, [])),
            "warning_reasons": sorted({str(item.get("reason", "")) for item in warning_grouped.get(preset, [])}),
        }
        for field in METRIC_FIELDS:
            if field == "plan_cache_hit_rate":
                continue
            summary[field] = sum(_num(row.get(field)) for row in preset_rows)
        for field in DELTA_FIELDS:
            values = [_num(pair.get(f"d_{field}")) for pair in pair_grouped.get(preset, [])]
            summary[f"sum_d_{field}"] = sum(values)
            if field in RELATIVE_DELTA_FIELDS:
                summary[f"relative_d_{field}_pct"] = _relative_delta_pct(
                    summary[f"sum_d_{field}"],
                    reference_totals.get(field, 0.0),
                )
        preset_pairs = list(pair_grouped.get(preset, []))
        positive = 0
        negative = 0
        zero = 0
        for pair in preset_pairs:
            net = (
                -_num(pair.get("d_total_rh_low_violation"))
                - _num(pair.get("d_total_vpd_high_excess"))
                - _num(pair.get("d_total_temp_violation"))
                + 0.02 * _num(pair.get("d_total_reward"))
            )
            hard_bad = _num(pair.get("d_dew_margin_air_lt0_steps")) > EPS or _num(pair.get("d_canopy_dew_margin_lt0_steps")) > EPS
            if hard_bad or net < -EPS:
                negative += 1
            elif net > EPS:
                positive += 1
            else:
                zero += 1
        pair_count = max(len(preset_pairs), 1)
        summary["positive_scenario_count"] = positive
        summary["negative_scenario_count"] = negative
        summary["zero_scenario_count"] = zero
        summary["positive_scenario_ratio"] = positive / pair_count if preset_pairs else 0.0
        summary["negative_scenario_ratio"] = negative / pair_count if preset_pairs else 0.0
        summary["zero_scenario_ratio"] = zero / pair_count if preset_pairs else 0.0
        summary["per_scenario_warning_ratio"] = len(warning_grouped.get(preset, [])) / pair_count if preset_pairs else 0.0
        if preset != reference_preset and summary.get("failure_count", 0) == 0:
            reward_delta = _num(summary.get("sum_d_total_reward"))
            rh_delta = _num(summary.get("sum_d_total_rh_low_violation"))
            vpd_delta = _num(summary.get("sum_d_total_vpd_high_excess"))
            if (
                reward_delta > EPS
                and abs(_num(summary.get("relative_d_total_reward_pct"))) < 0.1
                and abs(_num(summary.get("relative_d_total_rh_low_violation_pct"))) < 0.1
                and abs(_num(summary.get("relative_d_total_vpd_high_excess_pct"))) < 0.1
            ):
                summary["effect_size_too_small"] = True
            elif rh_delta < -EPS or vpd_delta < -EPS or reward_delta > EPS:
                summary["effect_size_too_small"] = False
            else:
                summary["effect_size_too_small"] = True
        out[preset] = summary
    return out


def _recommend_preset(
    aggregate_by_preset: Mapping[str, Mapping[str, Any]],
    reference_preset: str,
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rejected: dict[str, list[str]] = defaultdict(list)
    for failure in failures:
        rejected[str(failure.get("preset", ""))].append(str(failure.get("reason", "")))

    candidates: list[tuple[float, str, Mapping[str, Any]]] = []
    for preset, summary in aggregate_by_preset.items():
        if rejected.get(preset):
            continue
        if preset == reference_preset:
            continue
        dry_delta = _num(summary.get("sum_d_total_rh_low_violation"))
        vpd_delta = _num(summary.get("sum_d_total_vpd_high_excess"))
        reward_delta = _num(summary.get("sum_d_total_reward"))
        useful = dry_delta < -EPS or vpd_delta < -EPS or reward_delta > EPS
        if not useful:
            continue
        score = dry_delta + vpd_delta - 0.02 * reward_delta
        candidates.append((score, preset, summary))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if candidates:
        _score, preset, summary = candidates[0]
        row_count = int(_num(summary.get("row_count")))
        evidence_level = (
            "expanded_holdout_positive_signal"
            if row_count >= 9
            else "small_holdout_positive_signal"
        )
        next_action = (
            "run_hot_dry_stress_suite"
            if evidence_level == "expanded_holdout_positive_signal"
            else "expand_validation"
        )
        if bool(summary.get("effect_size_too_small")):
            next_action = "expand_validation_with_effect_size_caution"
        return {
            "best_safe_preset": preset,
            "recommended_next_validation_candidate": preset,
            "decision": "use_candidate_for_larger_hot_dry_split",
            "merge_recommendation": False,
            "evidence_level": evidence_level,
            "next_action": next_action,
            "reason": (
                f"{preset} has no hard audit failures and improves aggregate "
                f"RH-low/VPD/reward signal versus {reference_preset}. This is not a merge recommendation."
            ),
            "score": float(_score),
            "aggregate_delta": {
                field: _num(summary.get(f"sum_d_{field}"))
                for field in ("total_reward", "total_rh_low_violation", "total_vpd_high_excess", "total_temp_violation")
            },
            "relative_delta_pct": {
                field: _num(summary.get(f"relative_d_{field}_pct"))
                for field in RELATIVE_DELTA_FIELDS
            },
            "per_scenario_warning_count": int(_num(summary.get("per_scenario_warning_count"))),
            "positive_scenario_ratio": _num(summary.get("positive_scenario_ratio")),
            "negative_scenario_ratio": _num(summary.get("negative_scenario_ratio")),
            "zero_scenario_ratio": _num(summary.get("zero_scenario_ratio")),
            "per_scenario_warning_ratio": _num(summary.get("per_scenario_warning_ratio")),
            "effect_size_too_small": bool(summary.get("effect_size_too_small", False)),
            "performance_claim_allowed": not bool(summary.get("effect_size_too_small", False)),
        }
    return {
        "best_safe_preset": reference_preset,
        "recommended_next_validation_candidate": reference_preset,
        "decision": "keep_balanced_and_diagnose_mechanism",
        "merge_recommendation": False,
        "evidence_level": "rejected_due_to_gate" if failures else "insufficient",
        "next_action": "fix_protocol_or_safety_regression" if failures else "diagnose_mechanism",
        "reason": "No non-reference preset has a clean and useful paired signal.",
        "score": 0.0,
        "aggregate_delta": {},
        "relative_delta_pct": {},
        "per_scenario_warning_count": 0,
        "positive_scenario_ratio": 0.0,
        "negative_scenario_ratio": 0.0,
        "zero_scenario_ratio": 0.0,
        "per_scenario_warning_ratio": 0.0,
        "effect_size_too_small": False,
        "performance_claim_allowed": False,
    }


def audit_sweep(
    *,
    input_paths: Sequence[str | Path] = (),
    benchmark_json_paths: Sequence[str | Path] = (),
    reference_preset: str = "balanced",
) -> dict[str, Any]:
    rows, warnings = load_rows(input_paths=input_paths, benchmark_json_paths=benchmark_json_paths)
    row_failures: list[dict[str, Any]] = []
    for row in rows:
        row_failures.extend(_row_gate_failures(row))
    pairs, missing_pairs = _build_pairs(rows, reference_preset)
    paired_failures: list[dict[str, Any]] = []
    soft_warnings: list[dict[str, Any]] = []
    reference_map = {
        (str(row["scenario_id"]), str(row["controller"])): row
        for row in rows
        if str(row.get("preset", "")) == reference_preset
    }
    for row in rows:
        if str(row.get("preset", "")) == reference_preset:
            continue
        reference = reference_map.get((str(row.get("scenario_id", "")), str(row.get("controller", ""))))
        if reference is not None:
            paired_failures.extend(_paired_failure(row, reference))
            soft_warnings.extend(_paired_soft_warnings(row, reference))
    for missing in missing_pairs:
        warnings.append(
            "missing_reference_pair:"
            f"{missing['preset']}:{missing['scenario_id']}:{missing['controller']}"
        )

    failures = row_failures + paired_failures
    aggregate_by_preset = _aggregate_by_preset(rows, pairs, failures, soft_warnings, reference_preset)
    recommendation = _recommend_preset(aggregate_by_preset, reference_preset, failures)
    return {
        "schema_version": "rspc_scoring_sweep_audit_v1",
        "reference_preset": reference_preset,
        "aggregate": {
            "decision": "fail" if failures else "pass",
            "row_count": len(rows),
            "preset_count": len({str(row.get("preset", "")) for row in rows}),
            "paired_delta_count": len(pairs),
            "failure_count": len(failures),
            "warning_count": len(warnings) + len(soft_warnings),
            "soft_warning_count": len(soft_warnings),
        },
        "rows": rows,
        "paired_deltas": pairs,
        "aggregate_by_preset": aggregate_by_preset,
        "gate_failures": failures,
        "rejected_presets": {
            preset: sorted({str(item.get("reason", "")) for item in failures if str(item.get("preset", "")) == preset})
            for preset in sorted({str(item.get("preset", "")) for item in failures})
            if preset
        },
        "per_scenario_failure_reasons": failures,
        "soft_warnings": soft_warnings,
        "recommendation": recommendation,
        "warnings": warnings,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}"
    return str(value)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_None._\n"
    return (
        "| " + " | ".join(headers) + " |\n"
        + "| " + " | ".join("---" for _ in headers) + " |\n"
        + "\n".join("| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in rows)
        + "\n"
    )


def build_markdown_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit["aggregate"]
    recommendation = audit["recommendation"]
    lines = [
        "# RSPC Scoring Sweep Preset Audit",
        "",
        f"- Decision: **{str(aggregate['decision']).upper()}**",
        f"- Reference preset: `{audit['reference_preset']}`",
        f"- Rows: {aggregate['row_count']}",
        f"- Paired deltas: {aggregate['paired_delta_count']}",
        f"- Gate failures: {aggregate['failure_count']}",
        f"- Recommended next validation candidate: `{recommendation.get('recommended_next_validation_candidate', recommendation['best_safe_preset'])}`",
        f"- Recommendation: {recommendation['decision']}",
        f"- Merge recommendation: **{str(recommendation.get('merge_recommendation', False)).upper()}**",
        f"- Performance claim allowed: **{str(recommendation.get('performance_claim_allowed', False)).upper()}**",
        f"- Evidence level: `{recommendation.get('evidence_level', '')}`",
        f"- Next action: `{recommendation.get('next_action', '')}`",
        f"- Reason: {recommendation['reason']}",
        "",
        "## Aggregate By Preset",
        "",
    ]
    aggregate_rows = []
    for preset, summary in audit["aggregate_by_preset"].items():
        aggregate_rows.append(
            [
                preset,
                summary.get("scenario_count", 0),
                summary.get("plan_cache_hit_rate", 0.0),
                summary.get("failure_count", 0),
                summary.get("sum_d_total_reward", 0.0),
                summary.get("sum_d_total_rh_low_violation", 0.0),
                summary.get("sum_d_total_vpd_high_excess", 0.0),
                summary.get("sum_d_total_temp_violation", 0.0),
                summary.get("relative_d_total_reward_pct", 0.0),
                summary.get("relative_d_total_rh_low_violation_pct", 0.0),
                summary.get("relative_d_total_vpd_high_excess_pct", 0.0),
                summary.get("per_scenario_warning_count", 0),
                summary.get("positive_scenario_ratio", 0.0),
                summary.get("negative_scenario_ratio", 0.0),
                summary.get("zero_scenario_ratio", 0.0),
                summary.get("effect_size_too_small", ""),
            ]
        )
    lines.append(
        _table(
            [
                "preset",
                "scenarios",
                "cache_hit_rate",
                "failures",
                "d_reward",
                "d_RHlow",
                "d_VPDhi",
                "d_temp",
                "rel_reward_%",
                "rel_RHlow_%",
                "rel_VPDhi_%",
                "warnings",
                "pos_ratio",
                "neg_ratio",
                "zero_ratio",
                "small_effect",
            ],
            aggregate_rows,
        )
    )
    if recommendation.get("effect_size_too_small"):
        lines.extend(
            [
                "",
                "> Do not claim performance improvement; use this preset only as a validation candidate.",
                "",
            ]
        )
    lines.extend(["", "## Gate Failures", ""])
    lines.append(
        _table(
            ["preset", "scenario", "controller", "reason", "severity", "delta/value"],
            [
                [
                    item.get("preset", ""),
                    item.get("scenario_id", ""),
                    item.get("controller", ""),
                    item.get("reason", ""),
                    item.get("severity", ""),
                    item.get("delta", item.get("value", "")),
                ]
                for item in audit["gate_failures"]
            ],
        )
    )
    lines.extend(["", "## Paired Deltas", ""])
    lines.append(
        _table(
            ["preset", "scenario", "controller", "d_reward", "d_RHlow", "d_VPDhi", "d_temp", "d_dew_lt0", "d_canopy_lt0"],
            [
                [
                    item.get("preset", ""),
                    item.get("scenario_id", ""),
                    item.get("controller", ""),
                    item.get("d_total_reward", 0.0),
                    item.get("d_total_rh_low_violation", 0.0),
                    item.get("d_total_vpd_high_excess", 0.0),
                    item.get("d_total_temp_violation", 0.0),
                    item.get("d_dew_margin_air_lt0_steps", 0.0),
                    item.get("d_canopy_dew_margin_lt0_steps", 0.0),
                ]
                for item in audit["paired_deltas"]
            ],
        )
    )
    if audit["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit["warnings"])
    if audit.get("soft_warnings"):
        lines.extend(["", "## Soft Warnings", ""])
        lines.append(
            _table(
                ["preset", "scenario", "controller", "reason", "delta"],
                [
                    [
                        item.get("preset", ""),
                        item.get("scenario_id", ""),
                        item.get("controller", ""),
                        item.get("reason", ""),
                        item.get("delta", ""),
                    ]
                    for item in audit.get("soft_warnings", [])
                ],
            )
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="*", default=[], help="Sweep output JSON file(s).")
    parser.add_argument("--benchmark-json", nargs="*", default=[], help="Benchmark JSON file(s); preset inferred from stem.")
    parser.add_argument("--reference-preset", default="balanced")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = audit_sweep(
        input_paths=args.input,
        benchmark_json_paths=args.benchmark_json,
        reference_preset=args.reference_preset,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(audit), encoding="utf-8")
    print(
        f"decision={audit['aggregate']['decision']} "
        f"best_safe_preset={audit['recommendation']['best_safe_preset']} "
        f"failures={audit['aggregate']['failure_count']}"
    )
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 1 if audit["aggregate"]["decision"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
