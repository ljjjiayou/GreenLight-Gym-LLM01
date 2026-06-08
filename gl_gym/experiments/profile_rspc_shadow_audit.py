"""Audit profile-conditioned RSPC shadow margins from benchmark traces."""

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
    risk_flags,
    trace_identity,
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


def _mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    return float(mean(data)) if data else 0.0


def _is_shadow_row(row: Mapping[str, Any]) -> bool:
    return _truthy(row, "profile_rspc_shadow_enabled")


def _gate_reason(row: Mapping[str, Any]) -> str:
    return str(row.get("profile_rspc_shadow_safety_gate_reason") or "none").strip() or "none"


def _best_profile(row: Mapping[str, Any]) -> str:
    return str(row.get("profile_rspc_shadow_best_profile") or "").strip()


def _raw_best_profile(row: Mapping[str, Any]) -> str:
    return str(row.get("profile_rspc_shadow_raw_best_profile") or "").strip()


def _would_improve(row: Mapping[str, Any]) -> bool:
    return _truthy(row, "profile_rspc_shadow_would_improve") and _num(row, "profile_rspc_shadow_margin", 0.0) > 0.0


def _raw_would_improve(row: Mapping[str, Any]) -> bool:
    return (
        _truthy(row, "profile_rspc_shadow_raw_best_would_improve")
        and _num(row, "profile_rspc_shadow_raw_best_margin", 0.0) > 0.0
    )


def _alignment(row: Mapping[str, Any]) -> str:
    return str(row.get("profile_rspc_shadow_safety_alignment") or "neutral_hold").strip() or "neutral_hold"


def _safe_hot_dry(row: Mapping[str, Any]) -> bool:
    flags = risk_flags(row)
    return bool(flags["dry_performance"] and not flags["safety"])


def _summarize_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "steps": int(len(rows)),
        "mean_margin": round(_mean(_num(row, "profile_rspc_shadow_margin", 0.0) for row in rows), 6),
        "positive_margin_steps": int(sum(_would_improve(row) for row in rows)),
        "best_profile_counts": _counts(_best_profile(row) for row in rows),
        "gate_counts": _counts(_gate_reason(row) for row in rows),
        "alignment_counts": _counts(_alignment(row) for row in rows),
        "unsafe_conflict_positive_margin_steps": int(
            sum(_would_improve(row) and _alignment(row) == "unsafe_conflict" for row in rows)
        ),
    }


def _matrix_by(rows: Sequence[Mapping[str, Any]], key_fn) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(key_fn(row) or "unknown"), []).append(row)
    return {key: _summarize_group(group) for key, group in sorted(groups.items())}


def _recommendation(summary: Mapping[str, Any]) -> Dict[str, Any]:
    shadow_steps = int(summary.get("shadow_steps", 0) or 0)
    if shadow_steps <= 0:
        return {"decision": "needs_trace_metadata", "reason": "No profile RSPC shadow metadata found."}

    hot_dry_gate_positive = int(summary.get("hot_dry_positive_under_safety_gate_steps", 0) or 0)
    unsafe_conflict_positive = int(summary.get("unsafe_conflict_positive_margin_steps", 0) or 0)
    unsafe_profile_positive = int(summary.get("unsafe_profile_positive_under_safety_gate_steps", 0) or 0)
    if hot_dry_gate_positive > 0 or unsafe_profile_positive > 0:
        return {
            "decision": "unsafe_to_connect",
            "reason": "forbidden profile has positive shadow margin under safety gates.",
        }
    if unsafe_conflict_positive > 0:
        return {
            "decision": "unsafe_to_connect",
            "reason": "profile shadow has positive margin with safety-conflict alignment.",
        }

    safe_hot_dry_steps = int(summary.get("safe_hot_dry_steps", 0) or 0)
    safe_hot_dry_positive = int(summary.get("safe_hot_dry_positive_margin_steps", 0) or 0)
    coverage = float(safe_hot_dry_positive / safe_hot_dry_steps) if safe_hot_dry_steps else 0.0
    if coverage >= 0.30:
        return {
            "decision": "profile_signal_detected",
            "reason": "Positive margin concentrates in safe hot-dry rows.",
            "safe_hot_dry_positive_coverage": round(coverage, 6),
        }
    return {
        "decision": "weak_profile_signal",
        "reason": "Profile-conditioned shadow rarely improves the RSPC score in safe hot-dry rows.",
        "safe_hot_dry_positive_coverage": round(coverage, 6),
    }


def audit_trace(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    shadow_rows = [row for row in rows if _is_shadow_row(row)]
    safe_hot_dry = [row for row in shadow_rows if _safe_hot_dry(row)]
    safety_rows = [row for row in shadow_rows if risk_flags(row)["safety"]]
    safety_gate_rows = [row for row in shadow_rows if _gate_reason(row) != "none"]
    forbidden_profiles = {"hot_dry_protect", "co2_day_boost", "lighting_assist"}
    hot_dry_positive_under_gate = [
        row
        for row in safety_gate_rows
        if _best_profile(row) == "hot_dry_protect" and _would_improve(row)
    ]
    unsafe_profile_positive_under_gate = [
        row
        for row in safety_gate_rows
        if _best_profile(row) in forbidden_profiles and _would_improve(row)
    ]
    raw_forbidden_positive_under_gate = [
        row
        for row in safety_gate_rows
        if _raw_best_profile(row) in forbidden_profiles and _raw_would_improve(row)
    ]
    summary: Dict[str, Any] = {
        **trace_identity(path),
        "rows": int(len(rows)),
        "shadow_steps": int(len(shadow_rows)),
        "candidate_count_mean": round(
            _mean(_num(row, "profile_rspc_shadow_candidate_count", 0.0) for row in shadow_rows),
            6,
        ),
        "mean_margin": round(_mean(_num(row, "profile_rspc_shadow_margin", 0.0) for row in shadow_rows), 6),
        "positive_margin_steps": int(sum(_would_improve(row) for row in shadow_rows)),
        "safe_hot_dry_steps": int(len(safe_hot_dry)),
        "safe_hot_dry_positive_margin_steps": int(sum(_would_improve(row) for row in safe_hot_dry)),
        "safety_steps": int(len(safety_rows)),
        "positive_margin_safety_steps": int(sum(_would_improve(row) for row in safety_rows)),
        "safety_aligned_positive_margin_steps": int(
            sum(_would_improve(row) and _alignment(row) == "safe_relief" for row in safety_rows)
        ),
        "unsafe_conflict_positive_margin_steps": int(
            sum(_would_improve(row) and _alignment(row) == "unsafe_conflict" for row in shadow_rows)
        ),
        "dry_benefit_positive_margin_steps": int(
            sum(_would_improve(row) and _alignment(row) == "dry_benefit" for row in shadow_rows)
        ),
        "safety_gate_steps": int(len(safety_gate_rows)),
        "hot_dry_positive_under_safety_gate_steps": int(len(hot_dry_positive_under_gate)),
        "unsafe_profile_positive_under_safety_gate_steps": int(len(unsafe_profile_positive_under_gate)),
        "raw_forbidden_positive_under_safety_gate_steps": int(len(raw_forbidden_positive_under_gate)),
        "best_profile_counts": _counts(_best_profile(row) for row in shadow_rows),
        "raw_best_profile_counts": _counts(_raw_best_profile(row) for row in shadow_rows),
        "gate_counts": _counts(_gate_reason(row) for row in shadow_rows),
        "alignment_counts": _counts(_alignment(row) for row in shadow_rows),
        "by_gate": _matrix_by(shadow_rows, _gate_reason),
        "by_regime": _matrix_by(shadow_rows, lambda row: row.get("intent_regime") or "unknown"),
        "warnings": [],
    }
    if not shadow_rows:
        summary["warnings"].append("missing_profile_rspc_shadow_metadata")
    summary["recommendation"] = _recommendation(summary)
    return summary


def audit_traces(inputs: Sequence[str | Path]) -> Dict[str, Any]:
    traces = [audit_trace(path) for path in discover_traces(inputs)]
    aggregate_rows = int(sum(int(trace.get("rows", 0) or 0) for trace in traces))
    shadow_steps = int(sum(int(trace.get("shadow_steps", 0) or 0) for trace in traces))
    safe_hot_dry_steps = int(sum(int(trace.get("safe_hot_dry_steps", 0) or 0) for trace in traces))
    safe_hot_dry_positive = int(sum(int(trace.get("safe_hot_dry_positive_margin_steps", 0) or 0) for trace in traces))
    safety_steps = int(sum(int(trace.get("safety_steps", 0) or 0) for trace in traces))
    safety_positive = int(sum(int(trace.get("positive_margin_safety_steps", 0) or 0) for trace in traces))
    safety_aligned_positive = int(
        sum(int(trace.get("safety_aligned_positive_margin_steps", 0) or 0) for trace in traces)
    )
    unsafe_conflict_positive = int(
        sum(int(trace.get("unsafe_conflict_positive_margin_steps", 0) or 0) for trace in traces)
    )
    dry_benefit_positive = int(sum(int(trace.get("dry_benefit_positive_margin_steps", 0) or 0) for trace in traces))
    hot_dry_gate_positive = int(
        sum(int(trace.get("hot_dry_positive_under_safety_gate_steps", 0) or 0) for trace in traces)
    )
    unsafe_profile_gate_positive = int(
        sum(int(trace.get("unsafe_profile_positive_under_safety_gate_steps", 0) or 0) for trace in traces)
    )
    raw_forbidden_gate_positive = int(
        sum(int(trace.get("raw_forbidden_positive_under_safety_gate_steps", 0) or 0) for trace in traces)
    )
    aggregate = {
        "trace_count": int(len(traces)),
        "rows": aggregate_rows,
        "shadow_steps": shadow_steps,
        "safe_hot_dry_steps": safe_hot_dry_steps,
        "safe_hot_dry_positive_margin_steps": safe_hot_dry_positive,
        "safety_steps": safety_steps,
        "positive_margin_safety_steps": safety_positive,
        "safety_aligned_positive_margin_steps": safety_aligned_positive,
        "unsafe_conflict_positive_margin_steps": unsafe_conflict_positive,
        "dry_benefit_positive_margin_steps": dry_benefit_positive,
        "hot_dry_positive_under_safety_gate_steps": hot_dry_gate_positive,
        "unsafe_profile_positive_under_safety_gate_steps": unsafe_profile_gate_positive,
        "raw_forbidden_positive_under_safety_gate_steps": raw_forbidden_gate_positive,
        "mean_margin": round(
            _mean(float(trace.get("mean_margin", 0.0) or 0.0) for trace in traces if int(trace.get("shadow_steps", 0) or 0) > 0),
            6,
        ),
        "best_profile_counts": {},
        "raw_best_profile_counts": {},
        "gate_counts": {},
        "alignment_counts": {},
    }
    for trace in traces:
        for key, value in trace.get("best_profile_counts", {}).items():
            aggregate["best_profile_counts"][key] = aggregate["best_profile_counts"].get(key, 0) + int(value)
        for key, value in trace.get("raw_best_profile_counts", {}).items():
            aggregate["raw_best_profile_counts"][key] = aggregate["raw_best_profile_counts"].get(key, 0) + int(value)
        for key, value in trace.get("gate_counts", {}).items():
            aggregate["gate_counts"][key] = aggregate["gate_counts"].get(key, 0) + int(value)
        for key, value in trace.get("alignment_counts", {}).items():
            aggregate["alignment_counts"][key] = aggregate["alignment_counts"].get(key, 0) + int(value)
    aggregate["best_profile_counts"] = dict(sorted(aggregate["best_profile_counts"].items()))
    aggregate["raw_best_profile_counts"] = dict(sorted(aggregate["raw_best_profile_counts"].items()))
    aggregate["gate_counts"] = dict(sorted(aggregate["gate_counts"].items()))
    aggregate["alignment_counts"] = dict(sorted(aggregate["alignment_counts"].items()))
    aggregate["recommendation"] = _recommendation(aggregate)
    return {
        "schema_version": "profile_rspc_shadow_audit_v1",
        "aggregate": aggregate,
        "traces": traces,
    }


def build_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {}) if isinstance(audit.get("aggregate", {}), Mapping) else {}
    recommendation = aggregate.get("recommendation", {}) if isinstance(aggregate.get("recommendation", {}), Mapping) else {}
    lines = [
        "# Profile RSPC Shadow Audit v1",
        "",
        "## Aggregate",
        "",
        f"- decision: `{recommendation.get('decision', 'unknown')}`",
        f"- reason: {recommendation.get('reason', '')}",
        f"- traces: {aggregate.get('trace_count', 0)}",
        f"- shadow steps: {aggregate.get('shadow_steps', 0)}",
        f"- safe hot-dry positive margin: {aggregate.get('safe_hot_dry_positive_margin_steps', 0)} / {aggregate.get('safe_hot_dry_steps', 0)}",
        f"- safety positive margin steps: {aggregate.get('positive_margin_safety_steps', 0)} / {aggregate.get('safety_steps', 0)}",
        f"- safety-aligned positive margin steps: {aggregate.get('safety_aligned_positive_margin_steps', 0)}",
        f"- unsafe-conflict positive margin steps: {aggregate.get('unsafe_conflict_positive_margin_steps', 0)}",
        f"- hot-dry positive under safety gate: {aggregate.get('hot_dry_positive_under_safety_gate_steps', 0)}",
        f"- unsafe profile positive under safety gate: {aggregate.get('unsafe_profile_positive_under_safety_gate_steps', 0)}",
        f"- raw forbidden positive under safety gate: {aggregate.get('raw_forbidden_positive_under_safety_gate_steps', 0)}",
        f"- best profile counts: {aggregate.get('best_profile_counts', {})}",
        f"- raw best profile counts: {aggregate.get('raw_best_profile_counts', {})}",
        f"- gate counts: {aggregate.get('gate_counts', {})}",
        f"- alignment counts: {aggregate.get('alignment_counts', {})}",
        "",
        "## Traces",
        "",
        "| trace | controller | decision | shadow | safe hot-dry +margin | safety aligned +margin | unsafe conflict +margin | best profiles |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for trace in audit.get("traces", []):
        if not isinstance(trace, Mapping):
            continue
        rec = trace.get("recommendation", {}) if isinstance(trace.get("recommendation", {}), Mapping) else {}
        lines.append(
            "| {trace} | {controller} | `{decision}` | {shadow} | {safe_pos}/{safe_total} | {safe_align} | {unsafe_conflict} | {best} |".format(
                trace=trace.get("trace_id", ""),
                controller=trace.get("controller", ""),
                decision=rec.get("decision", "unknown"),
                shadow=trace.get("shadow_steps", 0),
                safe_pos=trace.get("safe_hot_dry_positive_margin_steps", 0),
                safe_total=trace.get("safe_hot_dry_steps", 0),
                safe_align=trace.get("safety_aligned_positive_margin_steps", 0),
                unsafe_conflict=trace.get("unsafe_conflict_positive_margin_steps", 0),
                best=trace.get("best_profile_counts", {}),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-trace", nargs="+", required=True, help="Trace CSV/JSONL files or directories.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args(argv)

    audit = audit_traces(args.input_trace)
    output_json = Path(args.output_json)
    output_report = Path(args.output_report)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    output_report.write_text(build_report(audit), encoding="utf-8")
    print(f"Saved profile RSPC shadow audit JSON to {output_json}")
    print(f"Saved profile RSPC shadow audit report to {output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
