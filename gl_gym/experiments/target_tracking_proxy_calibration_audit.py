"""Audit shadow calibration for hot-dry target-tracking proxy behavior."""

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

from gl_gym.experiments.profile_scorer_decision_audit import (  # noqa: E402
    discover_traces,
    read_trace,
    trace_identity,
)


CLASSIFICATIONS = (
    "calibrated_improves_safe_hot_dry",
    "calibrated_no_gain",
    "calibrated_unsafe_conflict",
    "not_triggered_safety_gate",
    "not_triggered_no_dry_push",
    "missing_metadata",
)


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _truthy(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _parse_variants(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    value = row.get("rspc_tt_calibration_variants_json", "[]")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def classify_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    if "rspc_tt_calibration_enabled" not in row:
        return {
            "classification": "missing_metadata",
            "triggered": False,
            "safe_hot_dry": False,
            "unsafe_conflict": False,
            "has_unsafe_variant": False,
            "would_improve": False,
        }
    if not _truthy(row, "rspc_tt_calibration_enabled"):
        return {
            "classification": "missing_metadata",
            "triggered": False,
            "safe_hot_dry": False,
            "unsafe_conflict": False,
            "has_unsafe_variant": False,
            "would_improve": False,
        }

    gate = str(row.get("rspc_tt_calibration_safety_gate_reason", "none") or "none")
    reason = str(row.get("rspc_tt_calibration_reason", "") or "")
    triggered = _truthy(row, "rspc_tt_calibration_triggered")
    safe_hot_dry = _truthy(row, "rspc_tt_calibration_safe_hot_dry")
    unsafe_conflict = _truthy(row, "rspc_tt_calibration_unsafe_conflict")
    variants = _parse_variants(row)
    unsafe_variant_count = int(_num(row, "rspc_tt_calibration_unsafe_variant_count", 0.0))
    if unsafe_variant_count <= 0:
        unsafe_variant_count = int(sum(bool(item.get("unsafe_conflict", False)) for item in variants))
    has_unsafe_variant = _truthy(row, "rspc_tt_calibration_has_unsafe_variant") or unsafe_variant_count > 0
    would_improve = _truthy(row, "rspc_tt_calibration_would_improve")
    margin = _num(row, "rspc_tt_calibration_best_improvement_margin")

    if not triggered:
        if gate != "none" or reason == "safety_gate_active":
            classification = "not_triggered_safety_gate"
        else:
            classification = "not_triggered_no_dry_push"
    elif unsafe_conflict:
        classification = "calibrated_unsafe_conflict"
    elif safe_hot_dry and would_improve and margin > 0.05:
        classification = "calibrated_improves_safe_hot_dry"
    else:
        classification = "calibrated_no_gain"

    return {
        "classification": classification,
        "triggered": bool(triggered),
        "reason": reason,
        "safety_gate_reason": gate,
        "safe_hot_dry": bool(safe_hot_dry),
        "best_variant_name": str(row.get("rspc_tt_calibration_best_variant_name", "") or ""),
        "best_alignment": str(row.get("rspc_tt_calibration_best_alignment", "neutral_hold") or "neutral_hold"),
        "best_improvement_margin": float(margin),
        "unsafe_conflict": bool(unsafe_conflict),
        "has_unsafe_variant": bool(has_unsafe_variant),
        "unsafe_variant_count": int(unsafe_variant_count),
        "would_improve": bool(would_improve),
        "variant_count": int(_num(row, "rspc_tt_calibration_variant_count", len(variants))),
    }


def _recommendation(summary: Mapping[str, Any]) -> Dict[str, str]:
    if int(summary.get("metadata_steps", 0) or 0) <= 0:
        return {"decision": "needs_trace_metadata", "reason": "No target-tracking calibration metadata found."}
    if int(summary.get("unsafe_conflict_steps", 0) or 0) > 0:
        return {
            "decision": "unsafe_to_connect",
            "reason": "The selected calibration variant creates an unsafe conflict.",
        }
    triggered = int(summary.get("triggered_steps", 0) or 0)
    if triggered <= 0:
        return {"decision": "no_calibration_targets", "reason": "No vent-only target-tracking dry-push rows triggered."}
    improves = int(summary.get("improvement_steps", 0) or 0)
    if improves > 0:
        return {
            "decision": "calibration_signal_detected",
            "reason": "Safe hot-dry target-tracking calibration has positive shadow-score signal.",
        }
    return {"decision": "calibration_no_gain", "reason": "Calibration targets exist, but variants do not improve score."}


def audit_trace(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    classified = [classify_row(row) for row in rows]
    metadata_rows = [item for item in classified if item["classification"] != "missing_metadata"]
    triggered_rows = [item for item in metadata_rows if bool(item.get("triggered", False))]
    improvement_rows = [item for item in triggered_rows if item["classification"] == "calibrated_improves_safe_hot_dry"]
    summary: Dict[str, Any] = {
        **trace_identity(path),
        "rows": int(len(rows)),
        "metadata_steps": int(len(metadata_rows)),
        "triggered_steps": int(len(triggered_rows)),
        "safe_hot_dry_triggered_steps": int(sum(bool(item.get("safe_hot_dry", False)) for item in triggered_rows)),
        "improvement_steps": int(len(improvement_rows)),
        "unsafe_conflict_steps": int(sum(bool(item.get("unsafe_conflict", False)) for item in triggered_rows)),
        "unsafe_variant_steps": int(sum(bool(item.get("has_unsafe_variant", False)) for item in triggered_rows)),
        "unsafe_variant_total": int(sum(int(item.get("unsafe_variant_count", 0) or 0) for item in triggered_rows)),
        "classification_counts": _counts(item.get("classification") for item in classified),
        "reason_counts": _counts(item.get("reason") for item in metadata_rows),
        "safety_gate_counts": _counts(item.get("safety_gate_reason") for item in metadata_rows),
        "best_variant_counts": _counts(item.get("best_variant_name") for item in triggered_rows),
        "alignment_counts": _counts(item.get("best_alignment") for item in triggered_rows),
        "mean_best_improvement_margin": round(
            float(mean(float(item.get("best_improvement_margin", 0.0) or 0.0) for item in triggered_rows))
            if triggered_rows
            else 0.0,
            6,
        ),
        "warnings": [],
    }
    if not metadata_rows:
        summary["warnings"].append("missing_target_tracking_calibration_metadata")
    summary["recommendation"] = _recommendation(summary)
    return summary


def audit_traces(inputs: Sequence[str | Path]) -> Dict[str, Any]:
    traces = [audit_trace(path) for path in discover_traces(inputs)]
    aggregate: Dict[str, Any] = {
        "trace_count": int(len(traces)),
        "rows": int(sum(int(trace.get("rows", 0) or 0) for trace in traces)),
        "metadata_steps": int(sum(int(trace.get("metadata_steps", 0) or 0) for trace in traces)),
        "triggered_steps": int(sum(int(trace.get("triggered_steps", 0) or 0) for trace in traces)),
        "safe_hot_dry_triggered_steps": int(
            sum(int(trace.get("safe_hot_dry_triggered_steps", 0) or 0) for trace in traces)
        ),
        "improvement_steps": int(sum(int(trace.get("improvement_steps", 0) or 0) for trace in traces)),
        "unsafe_conflict_steps": int(sum(int(trace.get("unsafe_conflict_steps", 0) or 0) for trace in traces)),
        "unsafe_variant_steps": int(sum(int(trace.get("unsafe_variant_steps", 0) or 0) for trace in traces)),
        "unsafe_variant_total": int(sum(int(trace.get("unsafe_variant_total", 0) or 0) for trace in traces)),
        "classification_counts": {},
        "reason_counts": {},
        "safety_gate_counts": {},
        "best_variant_counts": {},
        "alignment_counts": {},
        "warnings": sorted({warning for trace in traces for warning in trace.get("warnings", [])}),
    }
    for key in (
        "classification_counts",
        "reason_counts",
        "safety_gate_counts",
        "best_variant_counts",
        "alignment_counts",
    ):
        counts: Dict[str, int] = {}
        for trace in traces:
            for name, value in trace.get(key, {}).items():
                counts[str(name)] = counts.get(str(name), 0) + int(value)
        aggregate[key] = dict(sorted(counts.items()))
    aggregate["mean_best_improvement_margin"] = round(
        float(mean(float(trace.get("mean_best_improvement_margin", 0.0) or 0.0) for trace in traces)) if traces else 0.0,
        6,
    )
    aggregate["recommendation"] = _recommendation(aggregate)
    return {"aggregate": aggregate, "traces": traces}


def _fmt_counts(counts: Mapping[str, Any], limit: int = 8) -> str:
    if not counts:
        return "-"
    items = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items)


def build_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {}) if isinstance(audit.get("aggregate", {}), Mapping) else {}
    rec = aggregate.get("recommendation", {}) if isinstance(aggregate.get("recommendation", {}), Mapping) else {}
    lines = [
        "# Target Tracking Proxy Calibration Shadow Audit v1",
        "",
        "## Aggregate",
        "",
        f"- decision: `{rec.get('decision', 'unknown')}`",
        f"- reason: {rec.get('reason', '')}",
        f"- traces: {aggregate.get('trace_count', 0)}",
        f"- metadata steps: {aggregate.get('metadata_steps', 0)}",
        f"- triggered steps: {aggregate.get('triggered_steps', 0)}",
        f"- safe hot-dry triggered steps: {aggregate.get('safe_hot_dry_triggered_steps', 0)}",
        f"- improvement steps: {aggregate.get('improvement_steps', 0)}",
        f"- unsafe conflict steps: {aggregate.get('unsafe_conflict_steps', 0)}",
        f"- unsafe variant steps: {aggregate.get('unsafe_variant_steps', 0)}",
        f"- unsafe variant total: {aggregate.get('unsafe_variant_total', 0)}",
        f"- classification counts: {_fmt_counts(aggregate.get('classification_counts', {}))}",
        f"- reason counts: {_fmt_counts(aggregate.get('reason_counts', {}))}",
        f"- safety gate counts: {_fmt_counts(aggregate.get('safety_gate_counts', {}))}",
        f"- best variant counts: {_fmt_counts(aggregate.get('best_variant_counts', {}))}",
        f"- alignment counts: {_fmt_counts(aggregate.get('alignment_counts', {}))}",
        f"- mean best improvement margin: {aggregate.get('mean_best_improvement_margin', 0.0)}",
        "",
        "## Traces",
        "",
        "| trace | controller | decision | metadata | triggered | improve | unsafe best | unsafe variants | best variants | alignments |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for trace in audit.get("traces", []):
        rec = trace.get("recommendation", {}) if isinstance(trace.get("recommendation", {}), Mapping) else {}
        lines.append(
            "| {trace_id} | {controller} | `{decision}` | {metadata} | {triggered} | {improve} | {unsafe} | {unsafe_variants} | {variants} | {alignments} |".format(
                trace_id=trace.get("trace_id", ""),
                controller=trace.get("controller", ""),
                decision=rec.get("decision", "unknown"),
                metadata=trace.get("metadata_steps", 0),
                triggered=trace.get("triggered_steps", 0),
                improve=trace.get("improvement_steps", 0),
                unsafe=trace.get("unsafe_conflict_steps", 0),
                unsafe_variants=trace.get("unsafe_variant_steps", 0),
                variants=_fmt_counts(trace.get("best_variant_counts", {}), limit=4),
                alignments=_fmt_counts(trace.get("alignment_counts", {}), limit=4),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit target-tracking proxy calibration shadow metadata.")
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
