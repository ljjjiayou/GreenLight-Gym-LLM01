"""Audit runtime controlled shadow replay fields for hot-dry action proposers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gl_gym.experiments.profile_scorer_decision_audit import (  # noqa: E402
    discover_traces,
    read_trace,
    trace_identity,
)


def _truthy(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def classify_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    if "rspc_hot_dry_replay_enabled" not in row:
        return {"classification": "missing_metadata"}
    if not _truthy(row, "rspc_hot_dry_replay_enabled"):
        return {"classification": "disabled"}
    control_applied = _truthy(row, "rspc_hot_dry_proposer_control_applied")
    control_strict = _truthy(row, "rspc_hot_dry_proposer_control_strict_enabled")
    control_reason = str(row.get("rspc_hot_dry_proposer_control_reason", "") or "")
    control_safe_hot_dry = _truthy(row, "rspc_hot_dry_proposer_control_safe_hot_dry")
    control_gate = str(row.get("rspc_hot_dry_proposer_control_safety_gate_reason", "none") or "none")
    control_unsafe = bool(control_applied and (not control_safe_hot_dry or control_gate != "none"))
    strict_filter_reasons = {
        "candidate_not_allowed",
        "margin_below_strict_min",
        "not_severe_dry",
        "temp_headroom_low",
        "canopy_reserve_low",
    }
    if control_unsafe:
        classification = "controlled_access_unsafe_apply"
    elif control_applied and control_strict:
        classification = "strict_eligible_applied"
    elif control_applied:
        classification = "controlled_access_applied"
    elif control_strict and control_reason in strict_filter_reasons:
        classification = control_reason
    elif _truthy(row, "rspc_hot_dry_replay_unsafe_preferred") or _truthy(
        row, "rspc_hot_dry_replay_unsafe_conflict"
    ):
        classification = "unsafe_conflict"
    elif _truthy(row, "rspc_hot_dry_replay_would_apply"):
        classification = "controlled_replay_candidate"
    else:
        reason = str(row.get("rspc_hot_dry_replay_reason", "") or "").strip()
        classification = reason or "neutral_hold"
    return {
        "classification": classification,
        "reason": str(row.get("rspc_hot_dry_replay_reason", "") or ""),
        "safe_hot_dry": _truthy(row, "rspc_hot_dry_replay_safe_hot_dry"),
        "would_apply": _truthy(row, "rspc_hot_dry_replay_would_apply"),
        "unsafe_preferred": _truthy(row, "rspc_hot_dry_replay_unsafe_preferred"),
        "unsafe_conflict": _truthy(row, "rspc_hot_dry_replay_unsafe_conflict"),
        "unsafe_filtered_count": int(_num(row, "rspc_hot_dry_replay_unsafe_filtered_count", 0.0)),
        "control_enabled": _truthy(row, "rspc_hot_dry_proposer_control_enabled"),
        "control_strict": control_strict,
        "control_applied": control_applied,
        "control_unsafe": control_unsafe,
        "control_reason": control_reason,
        "best_candidate": str(row.get("rspc_hot_dry_replay_best_candidate_name", "") or ""),
        "best_variant": str(row.get("rspc_hot_dry_replay_best_variant", "") or ""),
        "best_margin": _num(row, "rspc_hot_dry_replay_best_margin", 0.0),
    }


def _recommendation(summary: Mapping[str, Any]) -> Dict[str, str]:
    if int(summary.get("metadata_steps", 0) or 0) <= 0:
        return {"decision": "needs_trace_metadata", "reason": "No runtime controlled replay fields found."}
    if int(summary.get("unsafe_preferred_steps", 0) or 0) > 0:
        return {
            "decision": "unsafe_to_connect",
            "reason": "Runtime controlled replay still has unsafe preferred rows.",
        }
    if int(summary.get("unsafe_applied_steps", 0) or 0) > 0:
        return {
            "decision": "unsafe_to_connect",
            "reason": "Runtime controlled access applied outside safe hot-dry eligibility.",
        }
    if int(summary.get("strict_applied_steps", 0) or 0) > 0:
        return {
            "decision": "strict_eligible_applied",
            "reason": "Strict runtime controlled access applied only on severe safe hot-dry rows.",
        }
    if int(summary.get("applied_steps", 0) or 0) > 0:
        return {
            "decision": "controlled_access_applied",
            "reason": "Runtime controlled access applied only on eligible safe hot-dry rows.",
        }
    if int(summary.get("would_apply_steps", 0) or 0) > 0:
        return {
            "decision": "controlled_replay_signal_detected",
            "reason": "Runtime controlled replay finds gated safe hot-dry proposer actions.",
        }
    return {"decision": "no_controlled_replay_signal", "reason": "No safe controlled replay candidate rows found."}


def audit_trace(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    classified = [classify_row(row) for row in rows]
    metadata = [item for item in classified if item["classification"] != "missing_metadata"]
    would_apply = [item for item in metadata if bool(item.get("would_apply", False))]
    summary: Dict[str, Any] = {
        **trace_identity(path),
        "rows": int(len(rows)),
        "metadata_steps": int(len(metadata)),
        "would_apply_steps": int(len(would_apply)),
        "applied_steps": int(sum(bool(item.get("control_applied", False)) for item in metadata)),
        "strict_applied_steps": int(
            sum(bool(item.get("control_applied", False)) and bool(item.get("control_strict", False)) for item in metadata)
        ),
        "strict_filtered_steps": int(
            sum(
                item.get("classification")
                in {
                    "candidate_not_allowed",
                    "margin_below_strict_min",
                    "not_severe_dry",
                    "temp_headroom_low",
                    "canopy_reserve_low",
                }
                for item in metadata
            )
        ),
        "unsafe_preferred_steps": int(sum(bool(item.get("unsafe_preferred", False)) for item in metadata)),
        "unsafe_conflict_steps": int(sum(bool(item.get("unsafe_conflict", False)) for item in metadata)),
        "unsafe_applied_steps": int(sum(bool(item.get("control_unsafe", False)) for item in metadata)),
        "unsafe_filtered_rows": int(sum(int(item.get("unsafe_filtered_count", 0) or 0) > 0 for item in metadata)),
        "unsafe_filtered_total": int(sum(int(item.get("unsafe_filtered_count", 0) or 0) for item in metadata)),
        "classification_counts": _counts(item.get("classification") for item in metadata),
        "reason_counts": _counts(item.get("reason") for item in metadata),
        "best_candidate_counts": _counts(item.get("best_candidate") for item in would_apply),
        "best_variant_counts": _counts(item.get("best_variant") for item in would_apply),
        "max_margin": round(max((float(item.get("best_margin", 0.0) or 0.0) for item in would_apply), default=0.0), 6),
        "warnings": [],
    }
    if not metadata:
        summary["warnings"].append("missing_runtime_controlled_replay_metadata")
    summary["recommendation"] = _recommendation(summary)
    return summary


def audit_traces(inputs: Sequence[str | Path]) -> Dict[str, Any]:
    traces = [audit_trace(path) for path in discover_traces(inputs)]
    aggregate: Dict[str, Any] = {
        "trace_count": int(len(traces)),
        "rows": int(sum(int(trace.get("rows", 0) or 0) for trace in traces)),
        "metadata_steps": int(sum(int(trace.get("metadata_steps", 0) or 0) for trace in traces)),
        "would_apply_steps": int(sum(int(trace.get("would_apply_steps", 0) or 0) for trace in traces)),
        "applied_steps": int(sum(int(trace.get("applied_steps", 0) or 0) for trace in traces)),
        "strict_applied_steps": int(sum(int(trace.get("strict_applied_steps", 0) or 0) for trace in traces)),
        "strict_filtered_steps": int(sum(int(trace.get("strict_filtered_steps", 0) or 0) for trace in traces)),
        "unsafe_preferred_steps": int(sum(int(trace.get("unsafe_preferred_steps", 0) or 0) for trace in traces)),
        "unsafe_conflict_steps": int(sum(int(trace.get("unsafe_conflict_steps", 0) or 0) for trace in traces)),
        "unsafe_applied_steps": int(sum(int(trace.get("unsafe_applied_steps", 0) or 0) for trace in traces)),
        "unsafe_filtered_rows": int(sum(int(trace.get("unsafe_filtered_rows", 0) or 0) for trace in traces)),
        "unsafe_filtered_total": int(sum(int(trace.get("unsafe_filtered_total", 0) or 0) for trace in traces)),
        "classification_counts": {},
        "reason_counts": {},
        "best_candidate_counts": {},
        "best_variant_counts": {},
        "warnings": sorted({warning for trace in traces for warning in trace.get("warnings", [])}),
    }
    for key in ("classification_counts", "reason_counts", "best_candidate_counts", "best_variant_counts"):
        counts: Dict[str, int] = {}
        for trace in traces:
            for name, value in trace.get(key, {}).items():
                counts[str(name)] = counts.get(str(name), 0) + int(value)
        aggregate[key] = dict(sorted(counts.items()))
    aggregate["recommendation"] = _recommendation(aggregate)
    return {"schema_version": "hot_dry_controlled_replay_trace_audit_v1", "aggregate": aggregate, "traces": traces}


def _fmt_counts(counts: Mapping[str, Any]) -> str:
    if not counts:
        return "-"
    items = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
    return ", ".join(f"{key}:{value}" for key, value in items)


def build_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {}) if isinstance(audit.get("aggregate", {}), Mapping) else {}
    rec = aggregate.get("recommendation", {}) if isinstance(aggregate.get("recommendation", {}), Mapping) else {}
    lines = [
        "# Hot-Dry Controlled Replay Trace Audit v1",
        "",
        "## Aggregate",
        "",
        f"- decision: `{rec.get('decision', 'unknown')}`",
        f"- reason: {rec.get('reason', '')}",
        f"- traces: {aggregate.get('trace_count', 0)}",
        f"- metadata steps: {aggregate.get('metadata_steps', 0)}",
        f"- would-apply steps: {aggregate.get('would_apply_steps', 0)}",
        f"- runtime applied steps: {aggregate.get('applied_steps', 0)}",
        f"- strict applied steps: {aggregate.get('strict_applied_steps', 0)}",
        f"- strict filtered steps: {aggregate.get('strict_filtered_steps', 0)}",
        f"- unsafe preferred steps: {aggregate.get('unsafe_preferred_steps', 0)}",
        f"- unsafe applied steps: {aggregate.get('unsafe_applied_steps', 0)}",
        f"- unsafe filtered rows: {aggregate.get('unsafe_filtered_rows', 0)}",
        f"- classification counts: {_fmt_counts(aggregate.get('classification_counts', {}))}",
        f"- best candidates: {_fmt_counts(aggregate.get('best_candidate_counts', {}))}",
        f"- best variants: {_fmt_counts(aggregate.get('best_variant_counts', {}))}",
        "",
        "## Traces",
        "",
        "| trace | controller | decision | metadata | would apply | applied | unsafe applied | filtered | best candidates |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for trace in audit.get("traces", []):
        if not isinstance(trace, Mapping):
            continue
        trace_rec = trace.get("recommendation", {}) if isinstance(trace.get("recommendation", {}), Mapping) else {}
        lines.append(
            "| {trace} | {controller} | `{decision}` | {metadata} | {apply} | {applied} | {unsafe} | {filtered} | {best} |".format(
                trace=trace.get("trace_id", ""),
                controller=trace.get("controller", ""),
                decision=trace_rec.get("decision", "unknown"),
                metadata=trace.get("metadata_steps", 0),
                apply=trace.get("would_apply_steps", 0),
                applied=trace.get("applied_steps", 0),
                unsafe=trace.get("unsafe_applied_steps", 0),
                filtered=trace.get("unsafe_filtered_rows", 0),
                best=_fmt_counts(trace.get("best_candidate_counts", {})),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-trace", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args(argv)

    audit = audit_traces(args.input_trace)
    output_json = Path(args.output_json)
    output_report = Path(args.output_report)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    output_report.write_text(build_report(audit), encoding="utf-8")
    print(f"Saved controlled replay trace JSON to {output_json}")
    print(f"Saved controlled replay trace report to {output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
