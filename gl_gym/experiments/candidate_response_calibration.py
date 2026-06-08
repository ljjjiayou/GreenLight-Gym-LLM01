"""Calibrate candidate response estimates against observed trace response.

The script is read-only.  It reports candidate-level prediction errors when
the trace contains prediction fields; otherwise it explicitly marks missing
prediction metadata and still reports observed 1/3/6-step responses for the
applied action, especially screen-increase plus ventilation-decrease patterns.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METRICS = {
    "temp": (("predicted_temp_next", "temp_next"), "temp_air"),
    "rh": (("predicted_rh_next", "rh_next"), "rh_air"),
    "vpd": (("predicted_vpd_next", "vpd_next"), "vpd_air"),
    "dew_margin": (("predicted_dew_margin_air_next", "dew_margin_next"), "dew_margin_air"),
    "canopy_dew_margin": (
        ("predicted_canopy_dew_margin_next", "canopy_dew_margin_next"),
        "canopy_dew_margin",
    ),
}
HORIZONS = (1, 3, 6)
NEAR_BOUNDARY_THRESHOLDS = (0.25, 0.5, 1.0)
BUFFER_SENSITIVITY_SCALES = (0.5, 0.75, 1.0, 1.25)
CANONICAL_PREDICTION_FIELDS = (
    "prediction_schema_version",
    "prediction_model",
    "prediction_horizon_steps",
    "predicted_temp_next",
    "predicted_rh_next",
    "predicted_vpd_next",
    "predicted_dew_margin_air_next",
    "predicted_canopy_dew_margin_next",
    "predicted_dew_lt0",
    "predicted_canopy_lt0",
)
LEGACY_PREDICTION_FIELDS = ("temp_next", "rh_next", "vpd_next", "dew_margin_next", "canopy_dew_margin_next")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_jsonish(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return default
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            continue
    return default


def _read_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_trace(root: str | Path, scenario_id: str) -> Path:
    path = Path(root)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file():
        return path
    matches = sorted(path.rglob(f"{scenario_id}*.csv"))
    if not matches:
        raise FileNotFoundError(f"trace not found for {scenario_id} under {path}")
    return matches[0]


def _iter_trace_paths(root: str | Path, scenario_id: str) -> list[Path]:
    path = Path(root)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file():
        return [path]
    matches = sorted(path.rglob(f"{scenario_id}*.csv"))
    if not matches:
        raise FileNotFoundError(f"trace not found for {scenario_id} under {path}")
    return matches


def _rows_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(round(_num(row.get("step", row.get("timestep", 0))))): row for row in rows}


def _parse_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _parse_jsonish(row.get("rspc_action_candidates_json"), default=[])
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _selected_candidate(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    selected_name = str(row.get("rspc_action_selected_name", "") or row.get("selected_fallback_candidate", "") or "").strip()
    for candidate in candidates:
        if _truthy(candidate.get("selected")) or str(candidate.get("name", "") or "").strip() == selected_name:
            return candidate
    return candidates[0] if candidates else {}


def _score_terms(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    terms = candidate.get("post_shape_score_terms") or candidate.get("score_terms") or {}
    return terms if isinstance(terms, Mapping) else {}


def _first_present(terms: Mapping[str, Any], keys: Sequence[str]) -> tuple[str | None, Any]:
    for key in keys:
        if key in terms:
            return key, terms.get(key)
    return None, None


def _prediction_schema_status(terms: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in CANONICAL_PREDICTION_FIELDS if key not in terms]
    legacy_present = [key for key in LEGACY_PREDICTION_FIELDS if key in terms]
    if not missing:
        family = "canonical"
    elif legacy_present:
        family = "legacy"
    else:
        family = "missing"
    return {
        "prediction_field_family": family,
        "missing_canonical_fields": missing,
        "legacy_prediction_fields_present": legacy_present,
    }


def _post_guardrail_outcome(row: Mapping[str, Any], next_row: Mapping[str, Any] | None) -> str:
    try:
        from gl_gym.experiments.post_guardrail_override_audit import classify_override

        result = classify_override(row, next_row)
        return str(result.get("outcome_label", "") or "")
    except Exception:
        return ""


def _final_action(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "screen": _num(row.get("u_screen")),
        "vent": _num(row.get("u_ventilation")),
        "shade": _num(row.get("u_shading")),
        "heat": _num(row.get("u_heating")),
        "co2": _num(row.get("u_co2")),
        "lamp": _num(row.get("u_lighting")),
    }


def _screen_up_vent_down(row: Mapping[str, Any], prev_row: Mapping[str, Any] | None) -> bool:
    if prev_row is None:
        return False
    action = _final_action(row)
    prev = _final_action(prev_row)
    return action["screen"] >= prev["screen"] + 0.05 and action["vent"] <= prev["vent"] - 0.05


def _is_pure_hot_dry(row: Mapping[str, Any]) -> bool:
    return bool(
        (
            _num(row.get("rh_low_violation")) > 0.0
            or _num(row.get("vpd_high_excess")) > 0.0
            or _truthy(row.get("rspc_action_hot_dry_active"))
            or _truthy(row.get("rspc_action_hot_dry_semantic_active"))
        )
        and _num(row.get("dew_margin_air"), 3.0) >= 1.0
        and _num(row.get("canopy_dew_margin"), 3.0) >= 1.0
        and _num(row.get("temp_air"), 20.0) < 32.0
    )


def _observed_response(rows_by_step: Mapping[int, Mapping[str, Any]], step: int) -> dict[str, dict[str, float]]:
    current = rows_by_step.get(step, {})
    out: dict[str, dict[str, float]] = {}
    for horizon in HORIZONS:
        future = rows_by_step.get(step + horizon)
        if future is None:
            continue
        out[str(horizon)] = {
            "d_temp": _num(future.get("temp_air")) - _num(current.get("temp_air")),
            "d_rh": _num(future.get("rh_air")) - _num(current.get("rh_air")),
            "d_vpd": _num(future.get("vpd_air")) - _num(current.get("vpd_air")),
            "d_dew_margin": _num(future.get("dew_margin_air")) - _num(current.get("dew_margin_air")),
            "d_canopy_dew_margin": _num(future.get("canopy_dew_margin")) - _num(current.get("canopy_dew_margin")),
            "future_canopy_dew_margin": _num(future.get("canopy_dew_margin")),
        }
    return out


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(float(v) for v in values)

    def pick(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = int(round(q * (len(ordered) - 1)))
        idx = max(0, min(len(ordered) - 1, idx))
        return ordered[idx]

    return {
        "count": len(ordered),
        "mean": _mean(ordered),
        "min": ordered[0],
        "p50": pick(0.50),
        "p90": pick(0.90),
        "max": ordered[-1],
    }


def _rank_correlation(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 0.0 or den_y <= 0.0:
        return None
    return float(num / (den_x * den_y))


def calibrate_trace(
    *,
    trace_path: str | Path,
    preset: str,
    scenario_id: str,
    start_step: int,
    end_step: int,
) -> dict[str, Any]:
    rows_by_step = _rows_by_step(_read_trace(trace_path))
    prediction_errors: dict[str, list[float]] = defaultdict(list)
    prediction_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    missing_prediction_counts: Counter[str] = Counter()
    false_safe = 0
    false_risk = 0
    predicted_canopy_count = 0
    false_safe_details: list[dict[str, Any]] = []
    schema_counts: Counter[str] = Counter()
    near_boundary_biases: list[float] = []
    near_boundary: dict[str, dict[str, Any]] = {
        f"lt_{threshold:g}": {
            "threshold": threshold,
            "count": 0,
            "v1_errors": [],
            "v2_errors": [],
            "risk_buffers": [],
            "false_safe_count": 0,
            "v2_warning_count": 0,
            "v2_unwarned_false_safe_count": 0,
        }
        for threshold in NEAR_BOUNDARY_THRESHOLDS
    }
    canopy_v1_errors: list[float] = []
    canopy_v2_errors: list[float] = []
    risk_buffers: list[float] = []
    false_safe_confidence_margins: list[float] = []
    buffer_sensitivity: dict[str, Counter[str]] = {f"buffer_scale_{scale:g}": Counter() for scale in BUFFER_SENSITIVITY_SCALES}
    records: list[dict[str, Any]] = []
    pattern_records: list[dict[str, Any]] = []

    for step in range(int(start_step), int(end_step) + 1):
        row = rows_by_step.get(step)
        next_row = rows_by_step.get(step + 1)
        if row is None or next_row is None:
            continue
        candidates = _parse_candidates(row)
        selected = _selected_candidate(row, candidates)
        terms = _score_terms(selected)
        schema_status = _prediction_schema_status(terms)
        schema_counts[str(schema_status.get("prediction_field_family", "missing"))] += 1
        metric_record: dict[str, Any] = {}
        for name, (pred_keys, actual_key) in METRICS.items():
            pred_key, pred_value = _first_present(terms, pred_keys)
            if pred_key is None:
                missing_prediction_counts[name] += 1
                metric_record[f"predicted_{name}_next"] = None
                metric_record[f"actual_{name}_next"] = _num(next_row.get(actual_key))
                continue
            predicted = _num(pred_value)
            actual = _num(next_row.get(actual_key))
            err = predicted - actual
            prediction_errors[name].append(err)
            prediction_pairs[name].append((predicted, actual))
            if name == "canopy_dew_margin" and _num(row.get("canopy_dew_margin"), 99.0) < 1.5:
                near_boundary_biases.append(err)
            metric_record[f"predicted_{name}_next"] = predicted
            metric_record[f"actual_{name}_next"] = actual
            metric_record[f"{name}_error"] = err
        canopy_pred_key, canopy_pred_value = _first_present(
            terms,
            ("predicted_canopy_dew_margin_next", "canopy_dew_margin_next"),
        )
        if canopy_pred_key is not None:
            predicted_canopy_count += 1
            predicted = _num(canopy_pred_value)
            actual = _num(next_row.get("canopy_dew_margin"))
            predicted_v2 = _num(terms.get("predicted_canopy_dew_margin_next_v2"), predicted)
            risk_buffer = _num(terms.get("canopy_proxy_v2_risk_buffer"), 0.0)
            warning_v2 = _truthy(terms.get("predicted_canopy_warning_v2"))
            current_canopy = _num(row.get("canopy_dew_margin"))
            v1_error = predicted - actual
            v2_error = predicted_v2 - actual
            canopy_v1_errors.append(v1_error)
            canopy_v2_errors.append(v2_error)
            risk_buffers.append(risk_buffer)
            pure_hot_dry = _is_pure_hot_dry(row)
            for scale in BUFFER_SENSITIVITY_SCALES:
                scale_key = f"buffer_scale_{scale:g}"
                adjusted_pred = predicted - scale * risk_buffer
                adjusted_warning = adjusted_pred < 0.25
                near_boundary_hard = current_canopy < 1.0 and actual < 0.0
                if predicted >= 0.0 and actual < 0.0:
                    buffer_sensitivity[scale_key]["false_safe_v1"] += 1
                    if adjusted_warning:
                        buffer_sensitivity[scale_key]["false_safe_v1_warned"] += 1
                    else:
                        buffer_sensitivity[scale_key]["false_safe_v1_unwarned"] += 1
                if adjusted_warning and pure_hot_dry and actual >= 0.0:
                    buffer_sensitivity[scale_key]["pure_hot_dry_affected_steps"] += 1
                if near_boundary_hard:
                    buffer_sensitivity[scale_key]["near_boundary_hard_events"] += 1
                    if adjusted_warning:
                        buffer_sensitivity[scale_key]["near_boundary_hard_events_warned"] += 1
            for bucket in near_boundary.values():
                if current_canopy < float(bucket["threshold"]):
                    bucket["count"] += 1
                    bucket["v1_errors"].append(v1_error)
                    bucket["v2_errors"].append(v2_error)
                    bucket["risk_buffers"].append(risk_buffer)
                    if warning_v2:
                        bucket["v2_warning_count"] += 1
                    if predicted >= 0.0 and actual < 0.0:
                        bucket["false_safe_count"] += 1
                        if not warning_v2:
                            bucket["v2_unwarned_false_safe_count"] += 1
            if predicted >= 0.0 and actual < 0.0:
                false_safe += 1
                false_safe_confidence_margins.append(predicted)
                action = _final_action(row)
                prev_action = _final_action(rows_by_step.get(step - 1, {}))
                false_safe_details.append(
                    {
                        "scenario_id": scenario_id,
                        "preset": preset,
                        "step": step,
                        "candidate_name": str(selected.get("name", "") or ""),
                        "prediction_model": str(terms.get("prediction_model", "") or ""),
                        "prediction_model_v2": str(terms.get("prediction_model_v2", "") or ""),
                        "prediction_field_family": str(schema_status.get("prediction_field_family", "")),
                        "predicted_canopy_dew_margin_next": predicted,
                        "predicted_canopy_dew_margin_next_v2": predicted_v2,
                        "predicted_canopy_warning_v2": warning_v2,
                        "canopy_proxy_v2_risk_buffer": risk_buffer,
                        "actual_canopy_dew_margin_next": actual,
                        "prediction_error": predicted - actual,
                        "current_canopy_dew_margin": _num(row.get("canopy_dew_margin")),
                        "current_dew_margin_air": _num(row.get("dew_margin_air")),
                        "screen_delta": action["screen"] - prev_action["screen"],
                        "ventilation_delta": action["vent"] - prev_action["vent"],
                        "final_action_screen": action["screen"],
                        "final_action_ventilation": action["vent"],
                        "post_guardrail_outcome": _post_guardrail_outcome(row, next_row),
                        "tomato_safety_v2_applied": _truthy(row.get("tomato_safety_v2_applied")),
                        "tomato_safety_v2_reasons": str(row.get("tomato_safety_v2_reasons", "") or ""),
                    }
                )
            if predicted < 0.0 and actual >= 0.0:
                false_risk += 1
        prev_row = rows_by_step.get(step - 1)
        observed = _observed_response(rows_by_step, step)
        is_pattern = _screen_up_vent_down(row, prev_row)
        record = {
            "preset": preset,
            "scenario_id": scenario_id,
            "step": step,
            "selected_candidate": str(selected.get("name", "") or ""),
            "screen_up_vent_down": is_pattern,
            "action": _final_action(row),
            "observed_response": observed,
            "prediction": metric_record,
            "prediction_schema": schema_status,
            "prediction_model": str(terms.get("prediction_model", "") or ""),
            "prediction_model_v2": str(terms.get("prediction_model_v2", "") or ""),
            "predicted_canopy_dew_margin_next_v2": _num(
                terms.get("predicted_canopy_dew_margin_next_v2"),
                metric_record.get("predicted_canopy_dew_margin_next") or 0.0,
            ),
            "predicted_canopy_warning_v2": _truthy(terms.get("predicted_canopy_warning_v2")),
            "canopy_proxy_v2_risk_buffer": _num(terms.get("canopy_proxy_v2_risk_buffer"), 0.0),
            "missing_prediction_metrics": [name for name in METRICS if f"predicted_{name}_next" in metric_record and metric_record[f"predicted_{name}_next"] is None],
        }
        records.append(record)
        if is_pattern:
            pattern_records.append(record)

    metric_summary: dict[str, Any] = {}
    for name in METRICS:
        errors = prediction_errors.get(name, [])
        metric_summary[name] = {
            "prediction_count": len(errors),
            "missing_prediction_count": int(missing_prediction_counts.get(name, 0)),
            "mae": _mean([abs(err) for err in errors]),
            "bias": _mean(errors),
            "rank_correlation": _rank_correlation(prediction_pairs.get(name, [])),
        }
    near_boundary_summary: dict[str, Any] = {}
    for label, bucket in near_boundary.items():
        count = int(bucket["count"])
        near_boundary_summary[label] = {
            "threshold": float(bucket["threshold"]),
            "count": count,
            "v1_mae": _mean([abs(err) for err in bucket["v1_errors"]]),
            "v1_bias": _mean(bucket["v1_errors"]),
            "v2_mae": _mean([abs(err) for err in bucket["v2_errors"]]),
            "v2_bias": _mean(bucket["v2_errors"]),
            "risk_buffer_mean": _mean(bucket["risk_buffers"]),
            "false_safe_count": int(bucket["false_safe_count"]),
            "v2_warning_count": int(bucket["v2_warning_count"]),
            "v2_unwarned_false_safe_count": int(bucket["v2_unwarned_false_safe_count"]),
            "v2_warning_rate": float(bucket["v2_warning_count"] / count) if count else None,
        }
    v2_warned_false_safe_count = int(
        sum(1 for item in false_safe_details if bool(item.get("predicted_canopy_warning_v2", False)))
    )
    buffer_sensitivity_summary: dict[str, Any] = {}
    for scale_key, counts in buffer_sensitivity.items():
        hard_events = int(counts.get("near_boundary_hard_events", 0))
        buffer_sensitivity_summary[scale_key] = {
            "v2_unwarned_false_safe": int(counts.get("false_safe_v1_unwarned", 0)),
            "v2_warned_false_safe": int(counts.get("false_safe_v1_warned", 0)),
            "pure_hot_dry_affected_steps": int(counts.get("pure_hot_dry_affected_steps", 0)),
            "near_boundary_recall": (
                float(counts.get("near_boundary_hard_events_warned", 0) / hard_events) if hard_events else None
            ),
        }
    return {
        "preset": preset,
        "scenario_id": scenario_id,
        "trace_path": str(trace_path),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "record_count": len(records),
        "prediction_schema_counts": dict(sorted(schema_counts.items())),
        "missing_prediction_metadata": bool(len(records) == 0 or int(schema_counts.get("canonical", 0)) < len(records)),
        "metric_summary": metric_summary,
        "false_safe_canopy_count": false_safe,
        "false_risk_canopy_count": false_risk,
        "predicted_canopy_count": predicted_canopy_count,
        "false_safe_rate": float(false_safe / predicted_canopy_count) if predicted_canopy_count else None,
        "false_safe_details": false_safe_details,
        "false_safe_steps": [int(item["step"]) for item in false_safe_details],
        "v2_warning_false_safe_count": v2_warned_false_safe_count,
        "v2_unwarned_false_safe_count": int(false_safe - v2_warned_false_safe_count),
        "min_predicted_margin_before_false_safe": min(
            (float(item["predicted_canopy_dew_margin_next"]) for item in false_safe_details),
            default=None,
        ),
        "false_safe_confidence_margin_distribution": _distribution(false_safe_confidence_margins),
        "mean_prediction_bias_near_boundary": _mean(near_boundary_biases),
        "near_boundary_summary": near_boundary_summary,
        "proxy_error_distribution": {
            "canopy_v1_error": _distribution(canopy_v1_errors),
            "canopy_v2_error": _distribution(canopy_v2_errors),
        },
        "risk_buffer_adequacy": {
            "risk_buffer_distribution": _distribution(risk_buffers),
            "v2_unwarned_false_safe_count": int(false_safe - v2_warned_false_safe_count),
            "v2_warned_false_safe_count": v2_warned_false_safe_count,
            "false_safe_count": int(false_safe),
        },
        "proxy_v2_buffer_sensitivity": buffer_sensitivity_summary,
        "screen_up_vent_down_count": len(pattern_records),
        "screen_up_vent_down_records": pattern_records,
        "records": records,
    }


def build_report(
    *,
    trace_dirs: Sequence[str | Path],
    scenario_id: str,
    start_step: int,
    end_step: int,
) -> dict[str, Any]:
    traces = []
    seen: set[Path] = set()
    for raw in trace_dirs:
        for trace_path in _iter_trace_paths(raw, scenario_id):
            resolved = trace_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            preset = trace_path.parent.name or "trace"
            if preset.lower() == "traces":
                preset = trace_path.stem
            traces.append(
                calibrate_trace(
                    trace_path=trace_path,
                    preset=preset,
                    scenario_id=scenario_id,
                    start_step=start_step,
                    end_step=end_step,
                )
            )
    return {
        "schema_version": "candidate_response_calibration_v1",
        "scenario_id": scenario_id,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "trace_count": len(traces),
        "traces": traces,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate Response Calibration",
        "",
        f"- Scenario: `{report.get('scenario_id', '')}`",
        f"- Steps: {report.get('start_step')}..{report.get('end_step')}",
        "",
        "| preset | missing_prediction_metadata | schema_counts | false_safe_rate | v2_warned_false_safe | v2_unwarned_false_safe | screen_up_vent_down | canopy_pred_count |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trace in report.get("traces", []) or []:
        false_safe_rate = trace.get("false_safe_rate")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(trace.get("preset", "")),
                    str(trace.get("missing_prediction_metadata", "")),
                    json.dumps(trace.get("prediction_schema_counts", {}), sort_keys=True),
                    "-" if false_safe_rate is None else f"{_num(false_safe_rate):.3f}",
                    str(trace.get("v2_warning_false_safe_count", 0)),
                    str(trace.get("v2_unwarned_false_safe_count", 0)),
                    str(trace.get("screen_up_vent_down_count", 0)),
                    str(trace.get("predicted_canopy_count", 0)),
                ]
            )
                + " |"
            )
    detail_rows = [
        item
        for trace in report.get("traces", []) or []
        for item in (trace.get("false_safe_details", []) or [])
        if isinstance(item, Mapping)
    ]
    if detail_rows:
        lines.extend(["", "## False-Safe Details", ""])
        lines.append(
            "| preset | step | candidate | v1_pred | v2_pred | v2_warning | actual_next | screen_delta | vent_delta | post_guardrail | tomato |"
        )
        lines.append("| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |")
        for item in detail_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("preset", "")),
                        str(item.get("step", "")),
                        str(item.get("candidate_name", "")),
                        f"{_num(item.get('predicted_canopy_dew_margin_next')):.3f}",
                        f"{_num(item.get('predicted_canopy_dew_margin_next_v2')):.3f}",
                        str(item.get("predicted_canopy_warning_v2", False)),
                        f"{_num(item.get('actual_canopy_dew_margin_next')):.3f}",
                        f"{_num(item.get('screen_delta')):.3f}",
                        f"{_num(item.get('ventilation_delta')):.3f}",
                        str(item.get("post_guardrail_outcome", "")),
                        str(item.get("tomato_safety_v2_reasons", "")) or "-",
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Metric Summary", ""])
    lines.append("| preset | metric | pred_count | missing | mae | bias | rank_corr |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for trace in report.get("traces", []) or []:
        for metric, summary in dict(trace.get("metric_summary", {})).items():
            corr = summary.get("rank_correlation")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(trace.get("preset", "")),
                        str(metric),
                        str(summary.get("prediction_count", 0)),
                        str(summary.get("missing_prediction_count", 0)),
                        f"{_num(summary.get('mae')):.3f}",
                        f"{_num(summary.get('bias')):.3f}",
                        "-" if corr is None else f"{_num(corr):.3f}",
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Near-Boundary Summary", ""])
    lines.append("| preset | bucket | count | v1_mae | v2_mae | false_safe | v2_unwarned_false_safe | v2_warning_rate |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for trace in report.get("traces", []) or []:
        for bucket, summary in dict(trace.get("near_boundary_summary", {})).items():
            warning_rate = summary.get("v2_warning_rate")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(trace.get("preset", "")),
                        str(bucket),
                        str(summary.get("count", 0)),
                        f"{_num(summary.get('v1_mae')):.3f}",
                        f"{_num(summary.get('v2_mae')):.3f}",
                        str(summary.get("false_safe_count", 0)),
                        str(summary.get("v2_unwarned_false_safe_count", 0)),
                        "-" if warning_rate is None else f"{_num(warning_rate):.3f}",
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Proxy v2 Buffer Sensitivity", ""])
    lines.append("| preset | scale | v2_unwarned_false_safe | v2_warned_false_safe | pure_hot_dry_affected | near_boundary_recall |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for trace in report.get("traces", []) or []:
        for scale, summary in dict(trace.get("proxy_v2_buffer_sensitivity", {})).items():
            recall = summary.get("near_boundary_recall") if isinstance(summary, Mapping) else None
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(trace.get("preset", "")),
                        str(scale),
                        str(summary.get("v2_unwarned_false_safe", 0) if isinstance(summary, Mapping) else 0),
                        str(summary.get("v2_warned_false_safe", 0) if isinstance(summary, Mapping) else 0),
                        str(summary.get("pure_hot_dry_affected_steps", 0) if isinstance(summary, Mapping) else 0),
                        "-" if recall is None else f"{_num(recall):.3f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", action="append", required=True)
    parser.add_argument("--scenario-id", "--focus-scenario", dest="scenario_id", default="y2020_d120_s44_n240_llm_rspc_v2")
    parser.add_argument("--scenario-list-json", default="")
    parser.add_argument("--start-step", type=int, default=220)
    parser.add_argument("--end-step", type=int, default=239)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        trace_dirs=args.trace_dir,
        scenario_id=args.scenario_id,
        start_step=args.start_step,
        end_step=args.end_step,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json
    if not output_md.is_absolute():
        output_md = PROJECT_ROOT / output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"trace_count={report['trace_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
