"""Build replay windows from profile-conditioned RSPC shadow traces."""

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


FORBIDDEN_SAFETY_PROFILES = {"hot_dry_protect", "co2_day_boost", "lighting_assist"}
ALLOWED_REPLAY_ALIGNMENTS = {"dry_benefit", "safe_relief"}


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


def _step(row: Mapping[str, Any]) -> int:
    return int(round(_num(row, "step", 0.0)))


def _alignment(row: Mapping[str, Any]) -> str:
    return str(row.get("profile_rspc_shadow_safety_alignment") or "neutral_hold").strip() or "neutral_hold"


def _gate(row: Mapping[str, Any]) -> str:
    return str(row.get("profile_rspc_shadow_safety_gate_reason") or "none").strip() or "none"


def _best_profile(row: Mapping[str, Any]) -> str:
    return str(row.get("profile_rspc_shadow_best_profile") or "").strip()


def _raw_best_profile(row: Mapping[str, Any]) -> str:
    return str(row.get("profile_rspc_shadow_raw_best_profile") or "").strip()


def _is_shadow_row(row: Mapping[str, Any]) -> bool:
    return _truthy(row, "profile_rspc_shadow_enabled")


def _safe_hot_dry(row: Mapping[str, Any]) -> bool:
    flags = risk_flags(row)
    return bool(flags["dry_performance"] and not flags["safety"])


def _is_unsafe_signal(row: Mapping[str, Any]) -> bool:
    if _alignment(row) == "unsafe_conflict":
        return True
    gate = _gate(row)
    best = _best_profile(row)
    return bool(gate != "none" and best in FORBIDDEN_SAFETY_PROFILES)


def _is_replay_signal(row: Mapping[str, Any], *, margin_threshold: float = 0.0) -> bool:
    if not _is_shadow_row(row):
        return False
    if _is_unsafe_signal(row):
        return False
    if not _truthy(row, "profile_rspc_shadow_best_eligible"):
        return False
    if not _truthy(row, "profile_rspc_shadow_would_improve"):
        return False
    if _num(row, "profile_rspc_shadow_margin", 0.0) <= float(margin_threshold):
        return False
    return _alignment(row) in ALLOWED_REPLAY_ALIGNMENTS


def _raw_forbidden_positive(row: Mapping[str, Any]) -> bool:
    return (
        _gate(row) != "none"
        and _raw_best_profile(row) in FORBIDDEN_SAFETY_PROFILES
        and _truthy(row, "profile_rspc_shadow_raw_best_would_improve")
        and _num(row, "profile_rspc_shadow_raw_best_margin", 0.0) > 0.0
    )


def _risk_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        for key, active in risk_flags(row).items():
            if active:
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _summarize_window(rows: Sequence[Mapping[str, Any]], signal_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    profile = _best_profile(signal_rows[0]) if signal_rows else ""
    return {
        "profile": profile,
        "start_step": _step(rows[0]),
        "end_step": _step(rows[-1]),
        "duration_steps": int(len(rows)),
        "signal_steps": int(len(signal_rows)),
        "gap_steps": int(len(rows) - len(signal_rows)),
        "mean_margin": round(_mean(_num(row, "profile_rspc_shadow_margin", 0.0) for row in signal_rows), 6),
        "cumulative_margin": round(sum(_num(row, "profile_rspc_shadow_margin", 0.0) for row in signal_rows), 6),
        "mean_delta_vent": round(_mean(_num(row, "profile_rspc_shadow_delta_vent", 0.0) for row in signal_rows), 6),
        "mean_delta_shade": round(_mean(_num(row, "profile_rspc_shadow_delta_shade", 0.0) for row in signal_rows), 6),
        "mean_delta_screen": round(_mean(_num(row, "profile_rspc_shadow_delta_screen", 0.0) for row in signal_rows), 6),
        "alignment_counts": _counts(_alignment(row) for row in signal_rows),
        "gate_counts": _counts(_gate(row) for row in signal_rows),
        "regime_counts": _counts(row.get("intent_regime") for row in signal_rows),
        "risk_counts": _risk_counts(rows),
        "safe_hot_dry_signal_steps": int(sum(_safe_hot_dry(row) for row in signal_rows)),
    }


def build_replay_windows(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_signal_steps: int = 2,
    max_gap_steps: int = 1,
    margin_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    min_signal_steps = max(1, int(min_signal_steps))
    max_gap_steps = max(0, int(max_gap_steps))
    windows: List[Dict[str, Any]] = []
    active_rows: List[Mapping[str, Any]] = []
    active_signal_rows: List[Mapping[str, Any]] = []
    active_profile = ""
    gap_count = 0

    def flush() -> None:
        nonlocal active_rows, active_signal_rows, active_profile, gap_count
        if len(active_signal_rows) >= min_signal_steps and active_rows:
            windows.append(_summarize_window(active_rows, active_signal_rows))
        active_rows = []
        active_signal_rows = []
        active_profile = ""
        gap_count = 0

    for row in rows:
        signal = _is_replay_signal(row, margin_threshold=margin_threshold)
        profile = _best_profile(row) if signal else active_profile
        if signal:
            if active_rows and active_profile and profile != active_profile:
                flush()
            active_profile = profile
            active_rows.append(row)
            active_signal_rows.append(row)
            gap_count = 0
            continue
        if active_rows and gap_count < max_gap_steps:
            active_rows.append(row)
            gap_count += 1
            continue
        flush()
    flush()
    return windows


def _recommendation(summary: Mapping[str, Any], *, coverage_threshold: float) -> Dict[str, Any]:
    if int(summary.get("shadow_steps", 0) or 0) <= 0:
        return {"decision": "needs_trace_metadata", "reason": "No profile RSPC shadow metadata found."}
    if int(summary.get("unsafe_signal_steps", 0) or 0) > 0:
        return {"decision": "unsafe_to_replay", "reason": "Replay signal includes safety-conflict rows."}
    windows = int(summary.get("replay_window_count", 0) or 0)
    coverage = float(summary.get("safe_hot_dry_signal_coverage", 0.0) or 0.0)
    if windows > 0 and coverage >= float(coverage_threshold):
        return {
            "decision": "ready_for_profile_shadow_replay",
            "reason": "Replay windows persist in safe hot-dry rows without eligible safety conflicts.",
            "safe_hot_dry_signal_coverage": round(coverage, 6),
        }
    if windows > 0:
        return {
            "decision": "weak_replay_signal",
            "reason": "Replay windows exist but safe hot-dry coverage is below threshold.",
            "safe_hot_dry_signal_coverage": round(coverage, 6),
        }
    return {
        "decision": "no_replay_signal",
        "reason": "No replay windows survived hysteresis.",
        "safe_hot_dry_signal_coverage": round(coverage, 6),
    }


def audit_trace(
    path: str | Path,
    *,
    min_signal_steps: int = 2,
    max_gap_steps: int = 1,
    margin_threshold: float = 0.0,
    coverage_threshold: float = 0.20,
) -> Dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    shadow_rows = [row for row in rows if _is_shadow_row(row)]
    signal_rows = [row for row in shadow_rows if _is_replay_signal(row, margin_threshold=margin_threshold)]
    windows = build_replay_windows(
        shadow_rows,
        min_signal_steps=min_signal_steps,
        max_gap_steps=max_gap_steps,
        margin_threshold=margin_threshold,
    )
    safe_hot_dry_steps = int(sum(_safe_hot_dry(row) for row in shadow_rows))
    safe_hot_dry_signal_steps = int(sum(_safe_hot_dry(row) for row in signal_rows))
    unsafe_signal_steps = int(sum(_is_unsafe_signal(row) and _truthy(row, "profile_rspc_shadow_would_improve") for row in shadow_rows))
    raw_forbidden_steps = int(sum(_raw_forbidden_positive(row) for row in shadow_rows))
    summary: Dict[str, Any] = {
        **trace_identity(path),
        "rows": int(len(rows)),
        "shadow_steps": int(len(shadow_rows)),
        "signal_steps": int(len(signal_rows)),
        "unsafe_signal_steps": unsafe_signal_steps,
        "raw_forbidden_positive_under_safety_gate_steps": raw_forbidden_steps,
        "safe_hot_dry_steps": safe_hot_dry_steps,
        "safe_hot_dry_signal_steps": safe_hot_dry_signal_steps,
        "safe_hot_dry_signal_coverage": round(
            float(safe_hot_dry_signal_steps / safe_hot_dry_steps) if safe_hot_dry_steps else 0.0,
            6,
        ),
        "replay_window_count": int(len(windows)),
        "replay_signal_profile_counts": _counts(_best_profile(row) for row in signal_rows),
        "replay_signal_alignment_counts": _counts(_alignment(row) for row in signal_rows),
        "replay_signal_gate_counts": _counts(_gate(row) for row in signal_rows),
        "replay_windows": windows,
        "warnings": [],
    }
    if not shadow_rows:
        summary["warnings"].append("missing_profile_rspc_shadow_metadata")
    summary["recommendation"] = _recommendation(summary, coverage_threshold=coverage_threshold)
    return summary


def audit_traces(
    inputs: Sequence[str | Path],
    *,
    min_signal_steps: int = 2,
    max_gap_steps: int = 1,
    margin_threshold: float = 0.0,
    coverage_threshold: float = 0.20,
) -> Dict[str, Any]:
    traces = [
        audit_trace(
            path,
            min_signal_steps=min_signal_steps,
            max_gap_steps=max_gap_steps,
            margin_threshold=margin_threshold,
            coverage_threshold=coverage_threshold,
        )
        for path in discover_traces(inputs)
    ]
    shadow_steps = int(sum(int(trace.get("shadow_steps", 0) or 0) for trace in traces))
    signal_steps = int(sum(int(trace.get("signal_steps", 0) or 0) for trace in traces))
    unsafe_signal_steps = int(sum(int(trace.get("unsafe_signal_steps", 0) or 0) for trace in traces))
    raw_forbidden = int(sum(int(trace.get("raw_forbidden_positive_under_safety_gate_steps", 0) or 0) for trace in traces))
    safe_hot_dry_steps = int(sum(int(trace.get("safe_hot_dry_steps", 0) or 0) for trace in traces))
    safe_hot_dry_signal_steps = int(sum(int(trace.get("safe_hot_dry_signal_steps", 0) or 0) for trace in traces))
    windows = [window for trace in traces for window in trace.get("replay_windows", []) if isinstance(window, Mapping)]
    aggregate: Dict[str, Any] = {
        "trace_count": int(len(traces)),
        "shadow_steps": shadow_steps,
        "signal_steps": signal_steps,
        "unsafe_signal_steps": unsafe_signal_steps,
        "raw_forbidden_positive_under_safety_gate_steps": raw_forbidden,
        "safe_hot_dry_steps": safe_hot_dry_steps,
        "safe_hot_dry_signal_steps": safe_hot_dry_signal_steps,
        "safe_hot_dry_signal_coverage": round(
            float(safe_hot_dry_signal_steps / safe_hot_dry_steps) if safe_hot_dry_steps else 0.0,
            6,
        ),
        "replay_window_count": int(len(windows)),
        "replay_signal_profile_counts": {},
        "replay_signal_alignment_counts": {},
        "replay_signal_gate_counts": {},
        "top_replay_windows": sorted(
            windows,
            key=lambda item: (-float(item.get("cumulative_margin", 0.0)), -int(item.get("signal_steps", 0))),
        )[:12],
    }
    for trace in traces:
        for key, value in trace.get("replay_signal_profile_counts", {}).items():
            aggregate["replay_signal_profile_counts"][key] = (
                aggregate["replay_signal_profile_counts"].get(key, 0) + int(value)
            )
        for key, value in trace.get("replay_signal_alignment_counts", {}).items():
            aggregate["replay_signal_alignment_counts"][key] = (
                aggregate["replay_signal_alignment_counts"].get(key, 0) + int(value)
            )
        for key, value in trace.get("replay_signal_gate_counts", {}).items():
            aggregate["replay_signal_gate_counts"][key] = (
                aggregate["replay_signal_gate_counts"].get(key, 0) + int(value)
            )
    aggregate["replay_signal_profile_counts"] = dict(sorted(aggregate["replay_signal_profile_counts"].items()))
    aggregate["replay_signal_alignment_counts"] = dict(sorted(aggregate["replay_signal_alignment_counts"].items()))
    aggregate["replay_signal_gate_counts"] = dict(sorted(aggregate["replay_signal_gate_counts"].items()))
    aggregate["recommendation"] = _recommendation(aggregate, coverage_threshold=coverage_threshold)
    return {
        "schema_version": "profile_to_rspc_shadow_replay_v2",
        "config": {
            "min_signal_steps": int(min_signal_steps),
            "max_gap_steps": int(max_gap_steps),
            "margin_threshold": float(margin_threshold),
            "safe_hot_dry_coverage_threshold": float(coverage_threshold),
        },
        "aggregate": aggregate,
        "traces": traces,
    }


def build_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {}) if isinstance(audit.get("aggregate", {}), Mapping) else {}
    recommendation = aggregate.get("recommendation", {}) if isinstance(aggregate.get("recommendation", {}), Mapping) else {}
    lines = [
        "# Profile-to-RSPC Shadow Replay v2",
        "",
        "## Aggregate",
        "",
        f"- decision: `{recommendation.get('decision', 'unknown')}`",
        f"- reason: {recommendation.get('reason', '')}",
        f"- traces: {aggregate.get('trace_count', 0)}",
        f"- shadow steps: {aggregate.get('shadow_steps', 0)}",
        f"- replay signal steps: {aggregate.get('signal_steps', 0)}",
        f"- replay windows: {aggregate.get('replay_window_count', 0)}",
        f"- safe hot-dry signal coverage: {aggregate.get('safe_hot_dry_signal_steps', 0)} / {aggregate.get('safe_hot_dry_steps', 0)} = {aggregate.get('safe_hot_dry_signal_coverage', 0.0)}",
        f"- unsafe signal steps: {aggregate.get('unsafe_signal_steps', 0)}",
        f"- raw forbidden positive under safety gate: {aggregate.get('raw_forbidden_positive_under_safety_gate_steps', 0)}",
        f"- signal profile counts: {aggregate.get('replay_signal_profile_counts', {})}",
        f"- signal alignment counts: {aggregate.get('replay_signal_alignment_counts', {})}",
        "",
        "## Traces",
        "",
        "| trace | controller | decision | signal | windows | safe hot-dry signal | unsafe | profiles |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for trace in audit.get("traces", []):
        if not isinstance(trace, Mapping):
            continue
        rec = trace.get("recommendation", {}) if isinstance(trace.get("recommendation", {}), Mapping) else {}
        lines.append(
            "| {trace} | {controller} | `{decision}` | {signal} | {windows} | {safe_signal}/{safe_total} | {unsafe} | {profiles} |".format(
                trace=trace.get("trace_id", ""),
                controller=trace.get("controller", ""),
                decision=rec.get("decision", "unknown"),
                signal=trace.get("signal_steps", 0),
                windows=trace.get("replay_window_count", 0),
                safe_signal=trace.get("safe_hot_dry_signal_steps", 0),
                safe_total=trace.get("safe_hot_dry_steps", 0),
                unsafe=trace.get("unsafe_signal_steps", 0),
                profiles=trace.get("replay_signal_profile_counts", {}),
            )
        )
    lines.extend(["", "## Top Replay Windows", "", "| profile | start | end | signal | margin | alignments | regimes |", "|---|---:|---:|---:|---:|---|---|"])
    for window in aggregate.get("top_replay_windows", []):
        if not isinstance(window, Mapping):
            continue
        lines.append(
            "| {profile} | {start} | {end} | {signal} | {margin} | {alignments} | {regimes} |".format(
                profile=window.get("profile", ""),
                start=window.get("start_step", 0),
                end=window.get("end_step", 0),
                signal=window.get("signal_steps", 0),
                margin=window.get("cumulative_margin", 0.0),
                alignments=window.get("alignment_counts", {}),
                regimes=window.get("regime_counts", {}),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-trace", nargs="+", required=True, help="Trace CSV/JSONL files or directories.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--min-signal-steps", type=int, default=2)
    parser.add_argument("--max-gap-steps", type=int, default=1)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--safe-hot-dry-coverage-threshold", type=float, default=0.20)
    args = parser.parse_args(argv)

    audit = audit_traces(
        args.input_trace,
        min_signal_steps=args.min_signal_steps,
        max_gap_steps=args.max_gap_steps,
        margin_threshold=args.margin_threshold,
        coverage_threshold=args.safe_hot_dry_coverage_threshold,
    )
    output_json = Path(args.output_json)
    output_report = Path(args.output_report)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    output_report.write_text(build_report(audit), encoding="utf-8")
    print(f"Saved profile-to-RSPC shadow replay JSON to {output_json}")
    print(f"Saved profile-to-RSPC shadow replay report to {output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
