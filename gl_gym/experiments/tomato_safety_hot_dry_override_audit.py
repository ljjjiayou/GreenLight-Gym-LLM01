"""Narrow audit for Tomato Safety v2 hot-dry overrides under no safety gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gl_gym.experiments.post_score_override_root_cause_audit import (  # noqa: E402
    ACTION_KEYS,
    _action_delta,
    _mean,
    _tomato_action,
    classify_row as classify_root_cause_row,
)
from gl_gym.experiments.profile_scorer_decision_audit import (  # noqa: E402
    discover_traces,
    read_trace,
    trace_identity,
)


PATTERNS = (
    "true_dry_worsening_release",
    "vent_only_dry_push",
    "canopy_reserve_relief",
    "balanced_hot_dry_relief",
    "humidity_buffered_relief",
    "minor_or_unclear",
)


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _parse_reasons(row: Mapping[str, Any]) -> List[str]:
    raw = row.get("tomato_safety_v2_reasons", "")
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def classify_override_pattern(row: Mapping[str, Any]) -> Dict[str, Any]:
    root = classify_root_cause_row(row)
    if root.get("classification") != "tomato_safety_overrides_safe_hot_dry":
        return {
            "target": False,
            "pattern": "not_target",
            "root_classification": str(root.get("classification", "")),
        }

    before = _tomato_action(row, "before")
    after = _tomato_action(row, "after")
    delta = _action_delta(after, before)
    reasons = _parse_reasons(row)
    vent_delta = float(delta.get("vent", 0.0) or 0.0)
    screen_delta = float(delta.get("screen", 0.0) or 0.0)
    shade_delta = float(delta.get("shade", 0.0) or 0.0)

    if screen_delta < -0.10 or shade_delta < -0.20:
        pattern = "true_dry_worsening_release"
    elif vent_delta > 0.10 and screen_delta < 0.05 and shade_delta < 0.05:
        pattern = "vent_only_dry_push"
    elif (
        "canopy_dew_hot_dry_conflict_buffer" in reasons
        or "cooldown_canopy_reserve_buffer" in reasons
        or "severe_dry_canopy_reserve" in reasons
    ):
        pattern = "canopy_reserve_relief"
    elif vent_delta > 0.10 and (screen_delta >= 0.10 or shade_delta >= 0.10):
        pattern = "balanced_hot_dry_relief"
    elif screen_delta >= 0.10 or shade_delta >= 0.10:
        pattern = "humidity_buffered_relief"
    else:
        pattern = "minor_or_unclear"

    calibration_candidate = pattern in {"true_dry_worsening_release", "vent_only_dry_push"}
    likely_false_positive = pattern in {
        "canopy_reserve_relief",
        "balanced_hot_dry_relief",
        "humidity_buffered_relief",
    }
    return {
        "target": True,
        "pattern": pattern,
        "root_classification": str(root.get("classification", "")),
        "calibration_candidate": bool(calibration_candidate),
        "likely_false_positive": bool(likely_false_positive),
        "reasons": reasons,
        "delta": delta,
        "before": before,
        "after": after,
        "safety_gate_reason": str(root.get("safety_gate_reason", "none") or "none"),
    }


def _mean_delta(items: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in ACTION_KEYS:
        values = [
            float(item.get("delta", {}).get(key, 0.0) or 0.0)
            for item in items
            if isinstance(item.get("delta", {}), Mapping)
        ]
        out[key] = round(float(mean(values)) if values else 0.0, 6)
    return out


def _recommendation(summary: Mapping[str, Any]) -> Dict[str, str]:
    target = int(summary.get("target_steps", 0) or 0)
    if target <= 0:
        return {"decision": "needs_target_rows", "reason": "No Tomato Safety hot-dry override target rows found."}
    calibration = int(summary.get("calibration_candidate_steps", 0) or 0)
    false_positive = int(summary.get("likely_false_positive_steps", 0) or 0)
    if calibration > 0 and calibration >= false_positive:
        return {
            "decision": "calibrate_tomato_hot_dry_override",
            "reason": "Dry-worsening Tomato Safety overrides dominate target rows.",
        }
    if false_positive > calibration:
        return {
            "decision": "refine_root_cause_dry_worsening_metric",
            "reason": "Most target rows are buffered relief actions rather than pure dry-worsening overrides.",
        }
    return {"decision": "inspect_minor_overrides", "reason": "Target rows are present but not clearly actionable."}


def audit_trace(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    classified = [classify_override_pattern(row) for row in rows]
    targets = [item for item in classified if bool(item.get("target", False))]
    summary: Dict[str, Any] = {
        **trace_identity(path),
        "rows": int(len(rows)),
        "target_steps": int(len(targets)),
        "calibration_candidate_steps": int(sum(bool(item.get("calibration_candidate", False)) for item in targets)),
        "likely_false_positive_steps": int(sum(bool(item.get("likely_false_positive", False)) for item in targets)),
        "pattern_counts": _counts(item.get("pattern") for item in targets),
        "reason_counts": _counts(reason for item in targets for reason in item.get("reasons", [])),
        "safety_gate_counts": _counts(item.get("safety_gate_reason") for item in targets),
        "mean_tomato_delta": _mean_delta(targets),
        "warnings": [],
    }
    if not targets:
        summary["warnings"].append("no_tomato_safety_hot_dry_override_rows")
    summary["recommendation"] = _recommendation(summary)
    return summary


def audit_traces(inputs: Sequence[str | Path]) -> Dict[str, Any]:
    traces = [audit_trace(path) for path in discover_traces(inputs)]
    aggregate: Dict[str, Any] = {
        "trace_count": int(len(traces)),
        "rows": int(sum(int(trace.get("rows", 0) or 0) for trace in traces)),
        "target_steps": int(sum(int(trace.get("target_steps", 0) or 0) for trace in traces)),
        "calibration_candidate_steps": int(
            sum(int(trace.get("calibration_candidate_steps", 0) or 0) for trace in traces)
        ),
        "likely_false_positive_steps": int(
            sum(int(trace.get("likely_false_positive_steps", 0) or 0) for trace in traces)
        ),
        "pattern_counts": {},
        "reason_counts": {},
        "safety_gate_counts": {},
        "warnings": sorted({warning for trace in traces for warning in trace.get("warnings", [])}),
    }
    for key in ("pattern_counts", "reason_counts", "safety_gate_counts"):
        counts: Dict[str, int] = {}
        for trace in traces:
            for name, value in trace.get(key, {}).items():
                counts[str(name)] = counts.get(str(name), 0) + int(value)
        aggregate[key] = dict(sorted(counts.items()))
    aggregate["mean_tomato_delta"] = {
        key: round(
            _mean(float(trace.get("mean_tomato_delta", {}).get(key, 0.0) or 0.0) for trace in traces),
            6,
        )
        for key in ACTION_KEYS
    }
    aggregate["recommendation"] = _recommendation(aggregate)
    return {"aggregate": aggregate, "traces": traces}


def _fmt_counts(counts: Mapping[str, Any], limit: int = 8) -> str:
    if not counts:
        return "-"
    items = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items)


def _fmt_delta(delta: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}:{float(delta.get(key, 0.0) or 0.0):+.3f}" for key in ACTION_KEYS)


def build_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {}) if isinstance(audit.get("aggregate", {}), Mapping) else {}
    rec = aggregate.get("recommendation", {}) if isinstance(aggregate.get("recommendation", {}), Mapping) else {}
    lines = [
        "# Tomato Safety Hot-Dry Override Narrow Audit v1",
        "",
        "## Aggregate",
        "",
        f"- decision: `{rec.get('decision', 'unknown')}`",
        f"- reason: {rec.get('reason', '')}",
        f"- traces: {aggregate.get('trace_count', 0)}",
        f"- target steps: {aggregate.get('target_steps', 0)}",
        f"- calibration candidate steps: {aggregate.get('calibration_candidate_steps', 0)}",
        f"- likely false-positive steps: {aggregate.get('likely_false_positive_steps', 0)}",
        f"- pattern counts: {_fmt_counts(aggregate.get('pattern_counts', {}))}",
        f"- reason counts: {_fmt_counts(aggregate.get('reason_counts', {}))}",
        f"- safety gate counts: {_fmt_counts(aggregate.get('safety_gate_counts', {}))}",
        f"- mean Tomato Safety delta: {_fmt_delta(aggregate.get('mean_tomato_delta', {}))}",
        "",
        "## Traces",
        "",
        "| trace | controller | decision | target | calibrate | false-positive | patterns | mean delta |",
        "|---|---:|---|---:|---:|---:|---|---|",
    ]
    for trace in audit.get("traces", []):
        rec = trace.get("recommendation", {}) if isinstance(trace.get("recommendation", {}), Mapping) else {}
        lines.append(
            "| {trace_id} | {controller} | `{decision}` | {target} | {calib} | {fp} | {patterns} | {delta} |".format(
                trace_id=trace.get("trace_id", ""),
                controller=trace.get("controller", ""),
                decision=rec.get("decision", "unknown"),
                target=trace.get("target_steps", 0),
                calib=trace.get("calibration_candidate_steps", 0),
                fp=trace.get("likely_false_positive_steps", 0),
                patterns=_fmt_counts(trace.get("pattern_counts", {}), limit=4),
                delta=_fmt_delta(trace.get("mean_tomato_delta", {})),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Tomato Safety v2 hot-dry override target rows.")
    parser.add_argument("--input-trace", action="append", required=True, help="Trace CSV/JSONL file or directory.")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-report", type=str, default="")
    args = parser.parse_args()

    audit = audit_traces(args.input_trace)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(audit)
    if args.output_report:
        path = Path(args.output_report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
