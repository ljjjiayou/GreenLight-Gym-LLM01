"""Cross-scenario proxy-v2 shadow validation from trace CSV files.

The audit is read-only and uses final/shadow metadata already emitted in the
trace.  It is meant to answer whether proxy_v2 has useful lead-time warning
behavior beyond the single canonical failure case.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROXY_V2_1_CANDIDATE_SCALE = 0.75
PROXY_V2_REJECTED_SCALES = (0.5,)


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


def _read_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _scenario_from_path(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_llm_rspc_v2")] if stem.endswith("_llm_rspc_v2") else stem


def _iter_trace_paths(trace_roots: Sequence[str | Path], scenario_filter: set[str] | None = None) -> list[Path]:
    paths: list[Path] = []
    for raw in trace_roots:
        root = Path(raw)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        matches = [root] if root.is_file() else sorted(root.rglob("*.csv"))
        for match in matches:
            scenario = _scenario_from_path(match)
            if scenario_filter and scenario not in scenario_filter and match.stem not in scenario_filter:
                continue
            paths.append(match)
    if not paths:
        raise FileNotFoundError("no trace CSV files matched proxy-v2 validation inputs")
    return paths


def _load_scenario_filter(path: str | Path | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    payload = json.loads(p.read_text(encoding="utf-8"))
    items = payload.get("executable_scenarios") or payload.get("selected_scenarios") or payload.get("scenarios") or []
    out: set[str] = set()
    for item in items:
        if isinstance(item, Mapping):
            scenario = str(item.get("scenario_id", "") or "")
            if scenario:
                out.add(scenario)
    return out or None


def _is_pure_hot_dry(row: Mapping[str, Any]) -> bool:
    return bool(
        (
            _truthy(row.get("rspc_action_hot_dry_active"))
            or _truthy(row.get("rspc_action_hot_dry_semantic_active"))
            or _num(row.get("rh_low_violation")) > 0.0
            or _num(row.get("vpd_high_excess")) > 0.0
        )
        and _num(row.get("dew_margin_air"), 3.0) >= 1.0
        and _num(row.get("canopy_dew_margin"), 3.0) >= 1.0
        and _num(row.get("temp_air"), 20.0) < 32.0
    )


def _rows_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(round(_num(row.get("step", row.get("timestep", 0))))): row for row in rows}


def _action(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "screen": _num(row.get("u_screen")),
        "vent": _num(row.get("u_ventilation")),
        "heat": _num(row.get("u_heating")),
        "shade": _num(row.get("u_shading")),
        "lamp": _num(row.get("u_lighting")),
        "co2": _num(row.get("u_co2")),
    }


def _action_pattern(row: Mapping[str, Any], prev_row: Mapping[str, Any] | None) -> str:
    action = _action(row)
    prev = _action(prev_row or {})
    screen_up = action["screen"] > prev["screen"] + 0.02
    vent_down = action["vent"] < prev["vent"] - 0.02
    high_screen = action["screen"] >= 0.55
    low_vent = action["vent"] <= 0.25
    if screen_up and vent_down:
        return "screen_up_vent_down"
    if high_screen and low_vent:
        return "high_screen_low_vent"
    if vent_down:
        return "vent_down"
    if screen_up:
        return "screen_up"
    return "other"


def _warning_reason(row: Mapping[str, Any], v2_pred: float, *, prefix: str = "") -> str:
    reason = str(row.get("final_action_risk_reason_v2", "") or row.get("boundary_reject_reason", "") or "")
    if reason:
        return f"{prefix}{reason}" if prefix else reason
    if v2_pred < 0.0:
        return f"{prefix}predicted_canopy_lt0" if prefix else "predicted_canopy_lt0_v2"
    if v2_pred < 0.25:
        return f"{prefix}predicted_canopy_warning_margin" if prefix else "predicted_canopy_warning_margin_v2"
    return f"{prefix}warning_flag_without_reason" if prefix else "warning_flag_without_reason"


def _warning_classification(
    row: Mapping[str, Any],
    *,
    actual_next: float,
    v2_pred: float,
    pure_hot_dry: bool,
    action_pattern: str,
) -> str:
    if row is None:
        return "metadata_insufficient"
    if actual_next < 0.0:
        return "reasonable_near_boundary_warning"
    if not pure_hot_dry:
        return "reasonable_near_boundary_warning" if _num(row.get("canopy_dew_margin"), 99.0) < 1.0 else "metadata_insufficient"
    current_canopy = _num(row.get("canopy_dew_margin"), 99.0)
    current_dew = _num(row.get("dew_margin_air"), 99.0)
    if current_canopy < 1.5 or current_dew < 1.5 or v2_pred < 0.0:
        return "reasonable_near_boundary_warning"
    if action_pattern in {"screen_up_vent_down", "high_screen_low_vent", "vent_down"}:
        return "dry_relief_interference_risk"
    return "over_conservative_warning"


def _sensitivity(rows: Mapping[int, Mapping[str, Any]], scales: Sequence[float] = (0.5, 0.75, 1.0, 1.25)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    steps = sorted(rows)
    for scale in scales:
        key = f"buffer_scale_{scale:g}"
        counts: Counter[str] = Counter()
        class_counts: Counter[str] = Counter()
        action_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        for step in steps:
            row = rows.get(step, {})
            next_row = rows.get(step + 1)
            if next_row is None:
                continue
            actual_next = _num(next_row.get("canopy_dew_margin"), 99.0)
            v1_pred = _num(row.get("canopy_boundary_shadow_predicted_canopy_dew_margin_next"), 99.0)
            v2_pred = _num(
                row.get("final_action_predicted_canopy_dew_margin_next_v2"),
                _num(row.get("canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2"), v1_pred),
            )
            implied_buffer = max(0.0, v1_pred - v2_pred)
            adjusted_pred = v1_pred - scale * implied_buffer
            warning = adjusted_pred < 0.25
            pure_hot_dry = _is_pure_hot_dry(row)
            action_pattern = _action_pattern(row, rows.get(step - 1))
            warning_class = _warning_classification(
                row,
                actual_next=actual_next,
                v2_pred=adjusted_pred,
                pure_hot_dry=pure_hot_dry,
                action_pattern=action_pattern,
            )
            near_boundary = _num(row.get("canopy_dew_margin"), 99.0) < 1.0
            if actual_next < 0.0 and v1_pred >= 0.0:
                counts["false_safe_v1"] += 1
                if warning:
                    counts["false_safe_v1_warned"] += 1
                else:
                    counts["false_safe_v1_unwarned"] += 1
            if warning and pure_hot_dry and actual_next >= 0.0:
                counts["pure_hot_dry_affected"] += 1
            if warning:
                counts["warning"] += 1
                class_counts[warning_class] += 1
                action_counts[action_pattern] += 1
                reason_counts[_warning_reason(row, adjusted_pred, prefix=f"scale_{scale:g}_")] += 1
            if near_boundary and actual_next < 0.0:
                counts["near_boundary_hard_events"] += 1
                if warning:
                    counts["near_boundary_hard_events_warned"] += 1
        hard = counts.get("near_boundary_hard_events", 0)
        out[key] = {
            "buffer_scale": float(scale),
            "v2_unwarned_false_safe": int(counts.get("false_safe_v1_unwarned", 0)),
            "v2_warned_false_safe": int(counts.get("false_safe_v1_warned", 0)),
            "pure_hot_dry_affected_steps": int(counts.get("pure_hot_dry_affected", 0)),
            "warning_count": int(counts.get("warning", 0)),
            "near_boundary_recall": float(counts.get("near_boundary_hard_events_warned", 0) / hard) if hard else None,
            "warning_classification_counts": dict(sorted(class_counts.items())),
            "warning_action_pattern_counts": dict(sorted(action_counts.items())),
            "warning_reason_counts": dict(sorted(reason_counts.items())),
        }
    return out


def audit_trace(path: str | Path) -> dict[str, Any]:
    trace_path = Path(path)
    rows = _rows_by_step(_read_trace(trace_path))
    steps = sorted(rows)
    preset = trace_path.parent.name or "trace"
    scenario = _scenario_from_path(trace_path)
    counts: Counter[str] = Counter()
    first_warning_step: int | None = None
    first_actual_canopy_lt0_step: int | None = None
    warning_classes: Counter[str] = Counter()
    action_patterns: Counter[str] = Counter()
    warning_reasons: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for step in steps:
        row = rows.get(step, {})
        next_row = rows.get(step + 1)
        if next_row is None:
            continue
        actual_next = _num(next_row.get("canopy_dew_margin"), 99.0)
        v1_pred = _num(row.get("canopy_boundary_shadow_predicted_canopy_dew_margin_next"), 99.0)
        v2_pred = _num(
            row.get("final_action_predicted_canopy_dew_margin_next_v2"),
            _num(row.get("canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2"), v1_pred),
        )
        warning_v2 = _truthy(row.get("final_action_predicted_canopy_warning_v2")) or _truthy(
            row.get("canopy_boundary_shadow_warning_v2")
        )
        pure_hot_dry = _is_pure_hot_dry(row)
        prev_row = rows.get(step - 1)
        action_pattern = _action_pattern(row, prev_row)
        warning_reason = _warning_reason(row, v2_pred)
        warning_class = (
            _warning_classification(
                row,
                actual_next=actual_next,
                v2_pred=v2_pred,
                pure_hot_dry=pure_hot_dry,
                action_pattern=action_pattern,
            )
            if warning_v2
            else ""
        )
        if warning_v2:
            counts["warning_v2"] += 1
            warning_classes[warning_class] += 1
            action_patterns[action_pattern] += 1
            warning_reasons[warning_reason] += 1
            if first_warning_step is None:
                first_warning_step = step
        if actual_next < 0.0:
            counts["actual_canopy_lt0_next"] += 1
            if first_actual_canopy_lt0_step is None:
                first_actual_canopy_lt0_step = step + 1
            if v1_pred >= 0.0:
                counts["false_safe_v1"] += 1
                if warning_v2:
                    counts["false_safe_v1_warned_by_v2"] += 1
                else:
                    counts["false_safe_v1_unwarned_by_v2"] += 1
        if warning_v2 and actual_next >= 0.0:
            counts["warning_without_hard_event_next"] += 1
            if pure_hot_dry:
                counts["pure_hot_dry_warning_without_hard_event_next"] += 1
        if pure_hot_dry:
            counts["pure_hot_dry_steps"] += 1
        if warning_v2 and pure_hot_dry:
            counts["affected_pure_hot_dry_steps"] += 1
        records.append(
            {
                "step": step,
                "actual_canopy_dew_margin_next": actual_next,
                "v1_predicted_canopy_dew_margin_next": v1_pred,
                "v2_predicted_canopy_dew_margin_next": v2_pred,
                "warning_v2": warning_v2,
                "pure_hot_dry": pure_hot_dry,
                "warning_classification": warning_class,
                "warning_reason": warning_reason if warning_v2 else "",
                "action_pattern": action_pattern,
                "canopy_dew_margin": _num(row.get("canopy_dew_margin"), 99.0),
                "dew_margin_air": _num(row.get("dew_margin_air"), 99.0),
                "rh_air": _num(row.get("rh_air"), 0.0),
                "vpd_air": _num(row.get("vpd_air"), 0.0),
                "u_screen": _num(row.get("u_screen"), 0.0),
                "u_ventilation": _num(row.get("u_ventilation"), 0.0),
            }
        )
    lead_time_steps = (
        first_actual_canopy_lt0_step - first_warning_step
        if first_actual_canopy_lt0_step is not None and first_warning_step is not None
        else None
    )
    return {
        "scenario_id": scenario,
        "preset": preset,
        "trace_path": str(trace_path),
        "counts": dict(sorted(counts.items())),
        "first_warning_step": first_warning_step,
        "first_actual_canopy_lt0_step": first_actual_canopy_lt0_step,
        "lead_time_steps": lead_time_steps,
        "prevention_window_available": bool(
            first_actual_canopy_lt0_step is not None
            and first_warning_step is not None
            and first_warning_step <= first_actual_canopy_lt0_step - 1
        ),
        "warning_classification_counts": dict(sorted(warning_classes.items())),
        "warning_action_pattern_counts": dict(sorted(action_patterns.items())),
        "warning_reason_counts": dict(sorted(warning_reasons.items())),
        "buffer_sensitivity": _sensitivity(rows),
        "records": records[:50],
    }


def build_report(*, trace_roots: Sequence[str | Path], scenario_list_json: str | Path | None = None) -> dict[str, Any]:
    scenario_filter = _load_scenario_filter(scenario_list_json)
    traces = [audit_trace(path) for path in _iter_trace_paths(trace_roots, scenario_filter)]
    aggregate: Counter[str] = Counter()
    by_scenario: dict[str, Counter[str]] = defaultdict(Counter)
    warning_classes: Counter[str] = Counter()
    action_patterns: Counter[str] = Counter()
    warning_reasons: Counter[str] = Counter()
    sensitivity: dict[str, Counter[str]] = defaultdict(Counter)
    sensitivity_classes: dict[str, Counter[str]] = defaultdict(Counter)
    sensitivity_actions: dict[str, Counter[str]] = defaultdict(Counter)
    sensitivity_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    sensitivity_scenarios: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for trace in traces:
        counts = Counter({str(k): int(v) for k, v in dict(trace.get("counts", {})).items()})
        aggregate.update(counts)
        by_scenario[str(trace.get("scenario_id", ""))].update(counts)
        warning_classes.update({str(k): int(v) for k, v in dict(trace.get("warning_classification_counts", {})).items()})
        action_patterns.update({str(k): int(v) for k, v in dict(trace.get("warning_action_pattern_counts", {})).items()})
        warning_reasons.update({str(k): int(v) for k, v in dict(trace.get("warning_reason_counts", {})).items()})
        for scale, item in dict(trace.get("buffer_sensitivity", {})).items():
            if isinstance(item, Mapping):
                sensitivity[str(scale)]["v2_unwarned_false_safe"] += int(_num(item.get("v2_unwarned_false_safe")))
                sensitivity[str(scale)]["v2_warned_false_safe"] += int(_num(item.get("v2_warned_false_safe")))
                sensitivity[str(scale)]["pure_hot_dry_affected_steps"] += int(_num(item.get("pure_hot_dry_affected_steps")))
                sensitivity[str(scale)]["warning_count"] += int(_num(item.get("warning_count")))
                sensitivity_scenarios[str(scale)][str(trace.get("scenario_id", ""))]["pure_hot_dry_affected_steps"] += int(
                    _num(item.get("pure_hot_dry_affected_steps"))
                )
                sensitivity_scenarios[str(scale)][str(trace.get("scenario_id", ""))]["warning_count"] += int(
                    _num(item.get("warning_count"))
                )
                sensitivity_classes[str(scale)].update(
                    {str(k): int(v) for k, v in dict(item.get("warning_classification_counts", {})).items()}
                )
                sensitivity_actions[str(scale)].update(
                    {str(k): int(v) for k, v in dict(item.get("warning_action_pattern_counts", {})).items()}
                )
                sensitivity_reasons[str(scale)].update(
                    {str(k): int(v) for k, v in dict(item.get("warning_reason_counts", {})).items()}
                )
                if item.get("near_boundary_recall") is not None:
                    sensitivity[str(scale)]["near_boundary_recall_reports"] += 1
                    sensitivity[str(scale)]["near_boundary_recall_sum"] += int(round(1000000 * _num(item.get("near_boundary_recall"))))
    top_pure_hot_dry_scenarios = sorted(
        (
            {
                "scenario_id": scenario,
                "affected_pure_hot_dry_steps": int(counts.get("affected_pure_hot_dry_steps", 0)),
                "warning_v2": int(counts.get("warning_v2", 0)),
            }
            for scenario, counts in by_scenario.items()
            if int(counts.get("affected_pure_hot_dry_steps", 0)) > 0
        ),
        key=lambda item: (-int(item["affected_pure_hot_dry_steps"]), str(item["scenario_id"])),
    )
    sensitivity_summary: dict[str, Any] = {}
    for scale, counts in sorted(sensitivity.items()):
        recall_reports = int(counts.get("near_boundary_recall_reports", 0))
        top_scenarios = sorted(
            (
                {
                    "scenario_id": scenario,
                    "affected_pure_hot_dry_steps": int(values.get("pure_hot_dry_affected_steps", 0)),
                    "warning_count": int(values.get("warning_count", 0)),
                }
                for scenario, values in sensitivity_scenarios.get(scale, {}).items()
                if int(values.get("pure_hot_dry_affected_steps", 0)) > 0
            ),
            key=lambda item: (-int(item["affected_pure_hot_dry_steps"]), str(item["scenario_id"])),
        )
        sensitivity_summary[scale] = {
            "v2_unwarned_false_safe": int(counts.get("v2_unwarned_false_safe", 0)),
            "v2_warned_false_safe": int(counts.get("v2_warned_false_safe", 0)),
            "pure_hot_dry_affected_steps": int(counts.get("pure_hot_dry_affected_steps", 0)),
            "warning_count": int(counts.get("warning_count", 0)),
            "near_boundary_recall_mean": (
                float(counts.get("near_boundary_recall_sum", 0) / 1000000.0 / recall_reports)
                if recall_reports
                else None
            ),
            "warning_classification_counts": dict(sorted(sensitivity_classes.get(scale, Counter()).items())),
            "warning_action_pattern_counts": dict(sorted(sensitivity_actions.get(scale, Counter()).items())),
            "warning_reason_counts": dict(sorted(sensitivity_reasons.get(scale, Counter()).items())),
            "top_pure_hot_dry_false_positive_scenarios": top_scenarios[:10],
        }
    candidate_scale_key = f"buffer_scale_{PROXY_V2_1_CANDIDATE_SCALE:g}"
    candidate_summary = dict(sensitivity_summary.get(candidate_scale_key, {}))
    current_affected = int(aggregate.get("affected_pure_hot_dry_steps", 0))
    candidate_affected = int(_num(candidate_summary.get("pure_hot_dry_affected_steps")))
    candidate_unwarned = int(_num(candidate_summary.get("v2_unwarned_false_safe")))
    rejected_scales: dict[str, Any] = {}
    for scale in PROXY_V2_REJECTED_SCALES:
        scale_key = f"buffer_scale_{scale:g}"
        item = sensitivity_summary.get(scale_key, {})
        rejected_scales[scale_key] = {
            "reason": "unsafe_due_to_false_safe" if int(_num(item.get("v2_unwarned_false_safe"))) > 0 else "not_rejected",
            "v2_unwarned_false_safe": int(_num(item.get("v2_unwarned_false_safe"))),
            "pure_hot_dry_affected_steps": int(_num(item.get("pure_hot_dry_affected_steps"))),
        }
    proxy_v2_1 = {
        "candidate_name": "proxy_v2_1_shadow_candidate",
        "buffer_scale": PROXY_V2_1_CANDIDATE_SCALE,
        "scale_key": candidate_scale_key,
        "v2_unwarned_false_safe": candidate_unwarned,
        "pure_hot_dry_affected_steps": candidate_affected,
        "baseline_proxy_v2_pure_hot_dry_affected_steps": current_affected,
        "reduces_pure_hot_dry_affected_steps": bool(candidate_affected < current_affected),
        "accepted_for_shadow_followup": bool(candidate_unwarned == 0 and candidate_affected < current_affected),
        "rejected_scales": rejected_scales,
        "top_pure_hot_dry_false_positive_scenarios": candidate_summary.get(
            "top_pure_hot_dry_false_positive_scenarios", []
        ),
    }
    return {
        "schema_version": "proxy_v2_shadow_validation_audit_v1",
        "trace_count": len(traces),
        "scenario_count": len({str(t.get("scenario_id", "")) for t in traces}),
        "aggregate_counts": dict(sorted(aggregate.items())),
        "false_safe_v1_unwarned_by_v2": int(aggregate.get("false_safe_v1_unwarned_by_v2", 0)),
        "affected_pure_hot_dry_steps": int(aggregate.get("affected_pure_hot_dry_steps", 0)),
        "pure_hot_dry_warning_without_hard_event_next": int(
            aggregate.get("pure_hot_dry_warning_without_hard_event_next", 0)
        ),
        "scenario_counts": {key: dict(sorted(value.items())) for key, value in sorted(by_scenario.items())},
        "warning_classification_counts": dict(sorted(warning_classes.items())),
        "warning_action_pattern_counts": dict(sorted(action_patterns.items())),
        "warning_reason_counts": dict(sorted(warning_reasons.items())),
        "top_pure_hot_dry_false_positive_scenarios": top_pure_hot_dry_scenarios[:10],
        "buffer_sensitivity_summary": sensitivity_summary,
        "proxy_v2_1_shadow_candidate": proxy_v2_1,
        "traces": traces,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Proxy v2 Shadow Validation Audit",
        "",
        f"- Trace count: {report.get('trace_count', 0)}",
        f"- Scenario count: {report.get('scenario_count', 0)}",
        f"- V1 false-safe not warned by v2: {report.get('false_safe_v1_unwarned_by_v2', 0)}",
        f"- Affected pure-hot-dry steps: {report.get('affected_pure_hot_dry_steps', 0)}",
        f"- Pure-hot-dry warning without next hard event: {report.get('pure_hot_dry_warning_without_hard_event_next', 0)}",
        "",
        "## Proxy v2.1 Shadow Candidate",
        "",
    ]
    candidate = report.get("proxy_v2_1_shadow_candidate", {}) if isinstance(report.get("proxy_v2_1_shadow_candidate"), Mapping) else {}
    lines.extend(
        [
            f"- Candidate: `{candidate.get('candidate_name', '')}`",
            f"- Buffer scale: {candidate.get('buffer_scale', '')}",
            f"- V2.1 unwarned false-safe: {candidate.get('v2_unwarned_false_safe', 0)}",
            f"- V2.1 pure-hot-dry affected steps: {candidate.get('pure_hot_dry_affected_steps', 0)}",
            f"- Baseline proxy v2 pure-hot-dry affected steps: {candidate.get('baseline_proxy_v2_pure_hot_dry_affected_steps', 0)}",
            f"- Accepted for shadow follow-up: {candidate.get('accepted_for_shadow_followup', False)}",
            f"- Rejected scales: {json.dumps(candidate.get('rejected_scales', {}), sort_keys=True)}",
            "",
        ]
    )
    lines.extend(
        [
        "## Warning Breakdown",
        "",
        "| category | counts |",
        "| --- | --- |",
        f"| warning_classification | {json.dumps(report.get('warning_classification_counts', {}), sort_keys=True)} |",
        f"| action_pattern | {json.dumps(report.get('warning_action_pattern_counts', {}), sort_keys=True)} |",
        f"| warning_reason | {json.dumps(report.get('warning_reason_counts', {}), sort_keys=True)} |",
        "",
        "## Top Pure-Hot-Dry Affected Scenarios",
        "",
        "| scenario | affected_pure_hot_dry_steps | warning_v2 |",
        "| --- | ---: | ---: |",
        ]
    )
    for item in report.get("top_pure_hot_dry_false_positive_scenarios", []) or []:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"| {item.get('scenario_id', '')} | {item.get('affected_pure_hot_dry_steps', 0)} | {item.get('warning_v2', 0)} |"
        )
    lines.extend(
        [
        "",
        "## Buffer Sensitivity",
        "",
        "| scale | v2_unwarned_false_safe | v2_warned_false_safe | pure_hot_dry_affected | near_boundary_recall_mean |",
        "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scale, item in dict(report.get("buffer_sensitivity_summary", {})).items():
        recall = item.get("near_boundary_recall_mean") if isinstance(item, Mapping) else None
        lines.append(
            "| "
            + " | ".join(
                [
                    str(scale),
                    str(item.get("v2_unwarned_false_safe", 0) if isinstance(item, Mapping) else 0),
                    str(item.get("v2_warned_false_safe", 0) if isinstance(item, Mapping) else 0),
                    str(item.get("pure_hot_dry_affected_steps", 0) if isinstance(item, Mapping) else 0),
                    "-" if recall is None else f"{_num(recall):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Trace Summary",
            "",
        "| scenario | preset | v1_false_safe | v2_unwarned | warning_v2 | pure_hot_dry_affected | first_warning | first_hard | lead_time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trace in report.get("traces", []) or []:
        counts = trace.get("counts", {}) if isinstance(trace.get("counts"), Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(trace.get("scenario_id", "")),
                    str(trace.get("preset", "")),
                    str(counts.get("false_safe_v1", 0)),
                    str(counts.get("false_safe_v1_unwarned_by_v2", 0)),
                    str(counts.get("warning_v2", 0)),
                    str(counts.get("affected_pure_hot_dry_steps", 0)),
                    str(trace.get("first_warning_step", "")),
                    str(trace.get("first_actual_canopy_lt0_step", "")),
                    str(trace.get("lead_time_steps", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", action="append", required=True)
    parser.add_argument("--scenario-list-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(trace_roots=args.trace_dir, scenario_list_json=args.scenario_list_json or None)
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
    print(f"false_safe_v1_unwarned_by_v2={report['false_safe_v1_unwarned_by_v2']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
