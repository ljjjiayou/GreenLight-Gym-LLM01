"""Build replay windows for controlled shadow hot-dry action proposers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gl_gym.experiments.hot_dry_action_proposer_controlled_shadow_audit import (  # noqa: E402
    classify_row,
)
from gl_gym.experiments.profile_scorer_decision_audit import (  # noqa: E402
    discover_traces,
    read_trace,
    risk_flags,
    trace_identity,
)
from gl_gym.experiments.rspc_action_scoring_audit import (  # noqa: E402
    ACTION_KEYS,
    _action,
    _candidate_name,
    _parse_candidates,
    _selected_candidate,
)


SIGNAL_CLASSIFICATION = "controlled_shadow_safe_proposer_gain"


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


def _mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    return float(mean(data)) if data else 0.0


def _step(row: Mapping[str, Any]) -> int:
    return int(round(_num(row, "step", 0.0)))


def _is_metadata(item: Mapping[str, Any]) -> bool:
    return str(item.get("classification", "")) != "missing_metadata"


def _is_signal(item: Mapping[str, Any], *, margin_threshold: float) -> bool:
    return (
        str(item.get("classification", "")) == SIGNAL_CLASSIFICATION
        and bool(item.get("safe_hot_dry", False))
        and float(item.get("best_gain_margin", 0.0) or 0.0) > float(margin_threshold)
    )


def _risk_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        for key, active in risk_flags(row).items():
            if active:
                counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _action_delta(row: Mapping[str, Any], item: Mapping[str, Any]) -> Dict[str, float]:
    candidates = _parse_candidates(row)
    if not candidates:
        return {key: 0.0 for key in ACTION_KEYS}
    selected = _selected_candidate(row, candidates)
    selected_action = _action(selected)
    best_name = str(item.get("best_gain_name", "") or "").strip()
    best = next((candidate for candidate in candidates if _candidate_name(candidate) == best_name), {})
    best_action = _action(best)
    return {
        key: round(float(best_action.get(key, 0.0) or 0.0) - float(selected_action.get(key, 0.0) or 0.0), 6)
        for key in ACTION_KEYS
    }


def _summarize_window(
    pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
    signal_pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> Dict[str, Any]:
    rows = [row for row, _item in pairs]
    signal_rows = [row for row, _item in signal_pairs]
    signal_items = [item for _row, item in signal_pairs]
    deltas = [_action_delta(row, item) for row, item in signal_pairs]
    return {
        "start_step": _step(rows[0]),
        "end_step": _step(rows[-1]),
        "duration_steps": int(len(rows)),
        "signal_steps": int(len(signal_pairs)),
        "gap_steps": int(len(rows) - len(signal_pairs)),
        "mean_margin": round(_mean(float(item.get("best_gain_margin", 0.0) or 0.0) for item in signal_items), 6),
        "cumulative_margin": round(sum(float(item.get("best_gain_margin", 0.0) or 0.0) for item in signal_items), 6),
        "best_gain_counts": _counts(item.get("best_gain_name") for item in signal_items),
        "best_gain_variant_counts": _counts(item.get("best_gain_variant") for item in signal_items),
        "selected_counts": _counts(item.get("selected_name") for item in signal_items),
        "regime_counts": _counts(row.get("intent_regime") for row in signal_rows),
        "risk_counts": _risk_counts(rows),
        "mean_delta_heat": round(_mean(delta["heat"] for delta in deltas), 6),
        "mean_delta_co2": round(_mean(delta["co2"] for delta in deltas), 6),
        "mean_delta_screen": round(_mean(delta["screen"] for delta in deltas), 6),
        "mean_delta_vent": round(_mean(delta["vent"] for delta in deltas), 6),
        "mean_delta_lamp": round(_mean(delta["lamp"] for delta in deltas), 6),
        "mean_delta_shade": round(_mean(delta["shade"] for delta in deltas), 6),
    }


def build_replay_windows(
    rows: Sequence[Mapping[str, Any]],
    classified: Sequence[Mapping[str, Any]],
    *,
    min_signal_steps: int = 2,
    max_gap_steps: int = 1,
    margin_threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    min_signal_steps = max(1, int(min_signal_steps))
    max_gap_steps = max(0, int(max_gap_steps))
    windows: List[Dict[str, Any]] = []
    active_pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    active_signal_pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    active_name = ""
    gap_count = 0

    def flush() -> None:
        nonlocal active_pairs, active_signal_pairs, active_name, gap_count
        if len(active_signal_pairs) >= min_signal_steps and active_pairs:
            windows.append(_summarize_window(active_pairs, active_signal_pairs))
        active_pairs = []
        active_signal_pairs = []
        active_name = ""
        gap_count = 0

    for row, item in zip(rows, classified):
        signal = _is_signal(item, margin_threshold=margin_threshold)
        name = str(item.get("best_gain_name", "") or "").strip() if signal else active_name
        if signal:
            if active_pairs and active_name and name != active_name:
                flush()
            active_name = name
            active_pairs.append((row, item))
            active_signal_pairs.append((row, item))
            gap_count = 0
            continue
        if active_pairs and gap_count < max_gap_steps:
            active_pairs.append((row, item))
            gap_count += 1
            continue
        flush()
    flush()
    return windows


def _recommendation(summary: Mapping[str, Any], *, coverage_threshold: float) -> Dict[str, Any]:
    if int(summary.get("metadata_steps", 0) or 0) <= 0:
        return {"decision": "needs_trace_metadata", "reason": "No RSPC action scoring metadata found."}
    if int(summary.get("unsafe_preferred_steps", 0) or 0) > 0:
        return {
            "decision": "unsafe_to_replay",
            "reason": "At least one controlled scoring variant prefers an unsafe proposer.",
        }
    windows = int(summary.get("replay_window_count", 0) or 0)
    coverage = float(summary.get("safe_hot_dry_signal_coverage", 0.0) or 0.0)
    if windows > 0 and coverage >= float(coverage_threshold):
        return {
            "decision": "ready_for_controlled_shadow_replay",
            "reason": "Safe proposer gains persist in replay windows without unsafe preference.",
            "safe_hot_dry_signal_coverage": round(coverage, 6),
        }
    if windows > 0:
        return {
            "decision": "weak_controlled_replay_signal",
            "reason": "Replay windows exist but safe hot-dry signal coverage is below threshold.",
            "safe_hot_dry_signal_coverage": round(coverage, 6),
        }
    return {
        "decision": "no_controlled_replay_signal",
        "reason": "No safe proposer replay windows survived hysteresis.",
        "safe_hot_dry_signal_coverage": round(coverage, 6),
    }


def audit_trace(
    path: str | Path,
    *,
    min_signal_steps: int = 2,
    max_gap_steps: int = 1,
    margin_threshold: float = 0.05,
    coverage_threshold: float = 0.10,
) -> Dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    identity = trace_identity(path)
    classified = [classify_row(row) for row in rows]
    metadata_items = [item for item in classified if _is_metadata(item)]
    safe_items = [item for item in metadata_items if bool(item.get("safe_hot_dry", False))]
    signal_items = [item for item in safe_items if _is_signal(item, margin_threshold=margin_threshold)]
    unsafe_steps = int(sum(str(item.get("classification", "")) == "controlled_shadow_unsafe_proposer" for item in safe_items))
    windows = build_replay_windows(
        rows,
        classified,
        min_signal_steps=min_signal_steps,
        max_gap_steps=max_gap_steps,
        margin_threshold=margin_threshold,
    )
    for window in windows:
        window.update(
            {
                "trace_id": identity.get("trace_id", path.stem),
                "scenario_id": identity.get("scenario_id", path.stem),
                "controller": identity.get("controller", ""),
            }
        )
    margins = [float(item.get("best_gain_margin", 0.0) or 0.0) for item in signal_items]
    summary: Dict[str, Any] = {
        **identity,
        "rows": int(len(rows)),
        "metadata_steps": int(len(metadata_items)),
        "safe_hot_dry_steps": int(len(safe_items)),
        "signal_steps": int(len(signal_items)),
        "unsafe_preferred_steps": unsafe_steps,
        "safe_hot_dry_signal_coverage": round(
            float(len(signal_items) / len(safe_items)) if safe_items else 0.0,
            6,
        ),
        "replay_window_count": int(len(windows)),
        "classification_counts": _counts(item.get("classification") for item in metadata_items),
        "best_gain_counts": _counts(item.get("best_gain_name") for item in signal_items),
        "best_gain_variant_counts": _counts(item.get("best_gain_variant") for item in signal_items),
        "mean_signal_margin": round(float(mean(margins)) if margins else 0.0, 6),
        "replay_windows": windows,
        "warnings": [],
    }
    if not metadata_items:
        summary["warnings"].append("missing_rspc_action_scoring_metadata")
    summary["recommendation"] = _recommendation(summary, coverage_threshold=coverage_threshold)
    return summary


def audit_traces(
    inputs: Sequence[str | Path],
    *,
    min_signal_steps: int = 2,
    max_gap_steps: int = 1,
    margin_threshold: float = 0.05,
    coverage_threshold: float = 0.10,
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
    windows = [window for trace in traces for window in trace.get("replay_windows", []) if isinstance(window, Mapping)]
    aggregate: Dict[str, Any] = {
        "trace_count": int(len(traces)),
        "rows": int(sum(int(trace.get("rows", 0) or 0) for trace in traces)),
        "metadata_steps": int(sum(int(trace.get("metadata_steps", 0) or 0) for trace in traces)),
        "safe_hot_dry_steps": int(sum(int(trace.get("safe_hot_dry_steps", 0) or 0) for trace in traces)),
        "signal_steps": int(sum(int(trace.get("signal_steps", 0) or 0) for trace in traces)),
        "unsafe_preferred_steps": int(sum(int(trace.get("unsafe_preferred_steps", 0) or 0) for trace in traces)),
        "safe_hot_dry_signal_coverage": 0.0,
        "replay_window_count": int(len(windows)),
        "classification_counts": {},
        "best_gain_counts": {},
        "best_gain_variant_counts": {},
        "top_replay_windows": sorted(
            windows,
            key=lambda item: (-float(item.get("cumulative_margin", 0.0)), -int(item.get("signal_steps", 0))),
        )[:12],
        "warnings": sorted({warning for trace in traces for warning in trace.get("warnings", [])}),
    }
    if aggregate["safe_hot_dry_steps"]:
        aggregate["safe_hot_dry_signal_coverage"] = round(
            float(aggregate["signal_steps"] / aggregate["safe_hot_dry_steps"]),
            6,
        )
    for key in ("classification_counts", "best_gain_counts", "best_gain_variant_counts"):
        counts: Dict[str, int] = {}
        for trace in traces:
            for name, value in trace.get(key, {}).items():
                counts[str(name)] = counts.get(str(name), 0) + int(value)
        aggregate[key] = dict(sorted(counts.items()))
    margins = [
        float(trace.get("mean_signal_margin", 0.0) or 0.0)
        for trace in traces
        if float(trace.get("mean_signal_margin", 0.0) or 0.0) > 0.0
    ]
    aggregate["mean_signal_margin"] = round(float(mean(margins)) if margins else 0.0, 6)
    aggregate["recommendation"] = _recommendation(aggregate, coverage_threshold=coverage_threshold)
    return {
        "schema_version": "hot_dry_action_proposer_shadow_replay_v1",
        "config": {
            "min_signal_steps": int(min_signal_steps),
            "max_gap_steps": int(max_gap_steps),
            "margin_threshold": float(margin_threshold),
            "safe_hot_dry_coverage_threshold": float(coverage_threshold),
        },
        "aggregate": aggregate,
        "traces": traces,
    }


def _fmt_counts(counts: Mapping[str, Any], limit: int = 8) -> str:
    if not counts:
        return "-"
    items = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items)


def build_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {}) if isinstance(audit.get("aggregate", {}), Mapping) else {}
    rec = aggregate.get("recommendation", {}) if isinstance(aggregate.get("recommendation", {}), Mapping) else {}
    lines = [
        "# Hot-Dry Action Proposer Shadow Replay v1",
        "",
        "## Aggregate",
        "",
        f"- decision: `{rec.get('decision', 'unknown')}`",
        f"- reason: {rec.get('reason', '')}",
        f"- traces: {aggregate.get('trace_count', 0)}",
        f"- metadata steps: {aggregate.get('metadata_steps', 0)}",
        f"- safe hot-dry steps: {aggregate.get('safe_hot_dry_steps', 0)}",
        f"- signal steps: {aggregate.get('signal_steps', 0)}",
        f"- replay windows: {aggregate.get('replay_window_count', 0)}",
        f"- safe hot-dry signal coverage: {aggregate.get('signal_steps', 0)} / {aggregate.get('safe_hot_dry_steps', 0)} = {aggregate.get('safe_hot_dry_signal_coverage', 0.0)}",
        f"- unsafe preferred steps: {aggregate.get('unsafe_preferred_steps', 0)}",
        f"- mean signal margin: {aggregate.get('mean_signal_margin', 0.0)}",
        f"- classification counts: {_fmt_counts(aggregate.get('classification_counts', {}))}",
        f"- best gain counts: {_fmt_counts(aggregate.get('best_gain_counts', {}))}",
        f"- best gain variant counts: {_fmt_counts(aggregate.get('best_gain_variant_counts', {}))}",
        "",
        "## Traces",
        "",
        "| trace | controller | decision | safe hot-dry | signal | windows | unsafe | best gains |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for trace in audit.get("traces", []):
        if not isinstance(trace, Mapping):
            continue
        trace_rec = trace.get("recommendation", {}) if isinstance(trace.get("recommendation", {}), Mapping) else {}
        lines.append(
            "| {trace} | {controller} | `{decision}` | {safe} | {signal} | {windows} | {unsafe} | {gains} |".format(
                trace=trace.get("trace_id", ""),
                controller=trace.get("controller", ""),
                decision=trace_rec.get("decision", "unknown"),
                safe=trace.get("safe_hot_dry_steps", 0),
                signal=trace.get("signal_steps", 0),
                windows=trace.get("replay_window_count", 0),
                unsafe=trace.get("unsafe_preferred_steps", 0),
                gains=_fmt_counts(trace.get("best_gain_counts", {}), limit=4),
            )
        )
    lines.extend(
        [
            "",
            "## Top Replay Windows",
            "",
            "| trace | controller | start | end | signal | margin | best gains | variants | d_screen | d_vent | d_shade | regimes |",
            "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---|",
        ]
    )
    for window in aggregate.get("top_replay_windows", []):
        if not isinstance(window, Mapping):
            continue
        lines.append(
            "| {trace} | {controller} | {start} | {end} | {signal} | {margin} | {gains} | {variants} | {screen} | {vent} | {shade} | {regimes} |".format(
                trace=window.get("trace_id", ""),
                controller=window.get("controller", ""),
                start=window.get("start_step", 0),
                end=window.get("end_step", 0),
                signal=window.get("signal_steps", 0),
                margin=window.get("cumulative_margin", 0.0),
                gains=_fmt_counts(window.get("best_gain_counts", {}), limit=3),
                variants=_fmt_counts(window.get("best_gain_variant_counts", {}), limit=3),
                screen=window.get("mean_delta_screen", 0.0),
                vent=window.get("mean_delta_vent", 0.0),
                shade=window.get("mean_delta_shade", 0.0),
                regimes=_fmt_counts(window.get("regime_counts", {}), limit=3),
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
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--safe-hot-dry-coverage-threshold", type=float, default=0.10)
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
    print(f"Saved hot-dry action proposer replay JSON to {output_json}")
    print(f"Saved hot-dry action proposer replay report to {output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
