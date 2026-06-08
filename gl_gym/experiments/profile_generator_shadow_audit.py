"""Audit shadow-only deterministic profile candidates from benchmark traces."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


TRACE_NAME_RE = re.compile(
    r"^y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)_n(?P<max_steps>\d+)_(?P<controller>.+)$"
)


def _coerce(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


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


def _split_csv(value: Any) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def read_trace(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", newline="", encoding="utf-8") as f:
        return [{key: _coerce(value) for key, value in row.items()} for row in csv.DictReader(f)]


def discover_traces(inputs: Sequence[str | Path]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix.lower() in {".csv", ".jsonl"}:
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.csv") if TRACE_NAME_RE.match(p.stem)))
            files.extend(sorted(p for p in path.rglob("*.jsonl") if TRACE_NAME_RE.match(p.stem)))
    by_stem: Dict[str, Path] = {}
    priority = {".csv": 0, ".jsonl": 1}
    for file in sorted(set(files)):
        current = by_stem.get(file.stem)
        if current is None or priority.get(file.suffix.lower(), 9) < priority.get(current.suffix.lower(), 9):
            by_stem[file.stem] = file
    return sorted(by_stem.values())


def _trace_identity(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    match = TRACE_NAME_RE.match(path.stem)
    if not match:
        return {"trace_id": path.stem, "controller": ""}
    data = match.groupdict()
    return {
        "trace_id": path.stem,
        "scenario_id": f"y{data['year']}_d{data['day']}_s{data['seed']}_n{data['max_steps']}",
        "year": int(data["year"]),
        "day": int(data["day"]),
        "seed": int(data["seed"]),
        "max_steps": int(data["max_steps"]),
        "controller": data["controller"],
    }


def _segments(rows: Sequence[Mapping[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    active: List[Mapping[str, Any]] = []
    for row in rows:
        if _num(row, "profile_selected_abs_delta_sum", 0.0) >= threshold:
            active.append(row)
            continue
        if active:
            windows.append(_summarize_window(active))
            active = []
    if active:
        windows.append(_summarize_window(active))
    return windows


def _scorer_segments(rows: Sequence[Mapping[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    active: List[Mapping[str, Any]] = []
    for row in rows:
        if _num(row, "profile_scorer_abs_delta_sum", 0.0) >= threshold:
            active.append(row)
            continue
        if active:
            windows.append(_summarize_scorer_window(active))
            active = []
    if active:
        windows.append(_summarize_scorer_window(active))
    return windows


def _summarize_window(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "start_step": int(_num(rows[0], "step", 0.0)),
        "end_step": int(_num(rows[-1], "step", 0.0)),
        "duration_steps": int(len(rows)),
        "mean_abs_delta_sum": round(mean(_num(row, "profile_selected_abs_delta_sum", 0.0) for row in rows), 6),
        "max_abs_delta_sum": round(max(_num(row, "profile_selected_abs_delta_sum", 0.0) for row in rows), 6),
        "intent_regime_counts": _counts(row.get("intent_regime") for row in rows),
        "selected_profile_counts": _counts(row.get("profile_generator_selected_shadow_profile") for row in rows),
        "mean_delta_target_temp": round(mean(_num(row, "profile_selected_delta_target_temp", 0.0) for row in rows), 6),
        "mean_delta_target_co2": round(mean(_num(row, "profile_selected_delta_target_co2", 0.0) for row in rows), 6),
        "mean_delta_target_rh": round(mean(_num(row, "profile_selected_delta_target_rh", 0.0) for row in rows), 6),
    }


def _summarize_scorer_window(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "start_step": int(_num(rows[0], "step", 0.0)),
        "end_step": int(_num(rows[-1], "step", 0.0)),
        "duration_steps": int(len(rows)),
        "mean_abs_delta_sum": round(mean(_num(row, "profile_scorer_abs_delta_sum", 0.0) for row in rows), 6),
        "max_abs_delta_sum": round(max(_num(row, "profile_scorer_abs_delta_sum", 0.0) for row in rows), 6),
        "mean_score": round(mean(_num(row, "profile_scorer_selected_score", 0.0) for row in rows), 6),
        "mean_margin": round(mean(_num(row, "profile_scorer_margin_to_second", 0.0) for row in rows), 6),
        "intent_regime_counts": _counts(row.get("intent_regime") for row in rows),
        "scorer_selected_counts": _counts(row.get("profile_scorer_selected_shadow_profile") for row in rows),
        "mean_delta_target_temp": round(mean(_num(row, "profile_scorer_delta_target_temp", 0.0) for row in rows), 6),
        "mean_delta_target_co2": round(mean(_num(row, "profile_scorer_delta_target_co2", 0.0) for row in rows), 6),
        "mean_delta_target_rh": round(mean(_num(row, "profile_scorer_delta_target_rh", 0.0) for row in rows), 6),
        "mean_dew_risk": round(mean(_num(row, "profile_scorer_dew_risk", 0.0) for row in rows), 6),
        "mean_hot_dry_retention_gap": round(
            mean(_num(row, "profile_scorer_hot_dry_retention_gap", 0.0) for row in rows),
            6,
        ),
        "mean_humid_relief_gap": round(mean(_num(row, "profile_scorer_humid_relief_gap", 0.0) for row in rows), 6),
    }


def audit_trace(path: str | Path, *, delta_threshold: float = 2.0) -> Dict[str, Any]:
    rows = sorted(read_trace(path), key=lambda row: _num(row, "step", 0.0))
    identity = _trace_identity(path)
    profile_rows = [row for row in rows if int(_num(row, "profile_generator_candidate_count", 0.0)) > 0]
    warnings: List[str] = []
    if rows and not profile_rows:
        warnings.append("missing_profile_generator_metadata")
    if not rows:
        warnings.append("empty_trace")

    scorer_rows = [row for row in profile_rows if str(row.get("profile_scorer_selected_shadow_profile") or "")]
    selected_deltas = [_num(row, "profile_selected_abs_delta_sum", 0.0) for row in profile_rows]
    scorer_deltas = [_num(row, "profile_scorer_abs_delta_sum", 0.0) for row in scorer_rows]
    return {
        **identity,
        "path": str(Path(path)),
        "steps": int(len(rows)),
        "profile_steps": int(len(profile_rows)),
        "profile_coverage": round(len(profile_rows) / max(len(rows), 1), 6),
        "mean_abs_delta_sum": round(mean(selected_deltas), 6) if selected_deltas else 0.0,
        "max_abs_delta_sum": round(max(selected_deltas), 6) if selected_deltas else 0.0,
        "shadow_only_steps": int(sum(_truthy(row, "profile_generator_shadow_only") for row in profile_rows)),
        "intent_regime_counts": _counts(row.get("intent_regime") for row in profile_rows),
        "selected_profile_counts": _counts(row.get("profile_generator_selected_shadow_profile") for row in profile_rows),
        "scorer_steps": int(len(scorer_rows)),
        "scorer_selected_counts": _counts(row.get("profile_scorer_selected_shadow_profile") for row in scorer_rows),
        "scorer_selector_agreement_steps": int(
            sum(_truthy(row, "profile_scorer_agrees_with_selector") for row in scorer_rows)
        ),
        "mean_scorer_selected_score": round(
            mean(_num(row, "profile_scorer_selected_score", 0.0) for row in scorer_rows),
            6,
        )
        if scorer_rows
        else 0.0,
        "mean_scorer_margin_to_second": round(
            mean(_num(row, "profile_scorer_margin_to_second", 0.0) for row in scorer_rows),
            6,
        )
        if scorer_rows
        else 0.0,
        "mean_scorer_abs_delta_sum": round(mean(scorer_deltas), 6) if scorer_deltas else 0.0,
        "candidate_name_counts": _counts(
            name
            for row in profile_rows
            for name in _split_csv(row.get("profile_candidate_names"))
        ),
        "unsupported_shape_counts": _counts(
            name
            for row in profile_rows
            for name in _split_csv(row.get("profile_generator_unsupported_shapes"))
        ),
        "large_delta_windows": _segments(profile_rows, delta_threshold),
        "scorer_large_delta_windows": _scorer_segments(scorer_rows, delta_threshold),
        "warnings": warnings,
    }


def audit_traces(inputs: Sequence[str | Path], *, delta_threshold: float = 2.0) -> Dict[str, Any]:
    files = discover_traces(inputs)
    traces = [audit_trace(path, delta_threshold=delta_threshold) for path in files]
    warnings = []
    if not files:
        warnings.append("no_trace_files_discovered")
    return {
        "schema_version": "profile_generator_shadow_audit_v1",
        "trace_count": len(traces),
        "profile_trace_count": int(sum(int(trace.get("profile_steps", 0)) > 0 for trace in traces)),
        "scorer_trace_count": int(sum(int(trace.get("scorer_steps", 0)) > 0 for trace in traces)),
        "delta_threshold": float(delta_threshold),
        "warnings": warnings,
        "traces": traces,
    }


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _fmt_counts(counts: Mapping[str, Any], limit: int = 4) -> str:
    items = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items) if items else "-"


def build_report(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Profile Generator Shadow Audit v1",
        "",
        f"- Trace files: {audit.get('trace_count', 0)}",
        f"- Traces with profile metadata: {audit.get('profile_trace_count', 0)}",
        f"- Traces with profile scorer metadata: {audit.get('scorer_trace_count', 0)}",
        f"- Large-delta threshold: {float(audit.get('delta_threshold', 0.0)):.3f}",
        "",
        "## Trace Summary",
        "",
        "| trace | controller | steps | profile steps | scorer steps | coverage | mean delta | scorer mean delta | scorer agreement | regimes | selector selected | scorer selected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for trace in audit.get("traces", []) or []:
        scorer_steps = int(trace.get("scorer_steps", 0))
        agreement = int(trace.get("scorer_selector_agreement_steps", 0))
        lines.append(
            "| {trace} | {controller} | {steps} | {profile_steps} | {scorer_steps} | {coverage:.2f} | {mean:.3f} | {scorer_mean:.3f} | {agreement:.2f} | {regimes} | {selected} | {scorer_selected} |".format(
                trace=trace.get("trace_id", ""),
                controller=trace.get("controller", ""),
                steps=int(trace.get("steps", 0)),
                profile_steps=int(trace.get("profile_steps", 0)),
                scorer_steps=scorer_steps,
                coverage=float(trace.get("profile_coverage", 0.0)),
                mean=float(trace.get("mean_abs_delta_sum", 0.0)),
                scorer_mean=float(trace.get("mean_scorer_abs_delta_sum", 0.0)),
                agreement=(agreement / scorer_steps) if scorer_steps else 0.0,
                regimes=_fmt_counts(trace.get("intent_regime_counts", {}) or {}),
                selected=_fmt_counts(trace.get("selected_profile_counts", {}) or {}),
                scorer_selected=_fmt_counts(trace.get("scorer_selected_counts", {}) or {}),
            )
        )

    lines.extend(
        [
            "",
            "## Large Delta Windows",
            "",
            "| trace | steps | duration | mean delta | max delta | regimes | selected | dT | dCO2 | dRH |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for trace in audit.get("traces", []) or []:
        for window in trace.get("large_delta_windows", []) or []:
            lines.append(
                "| {trace} | {start}-{end} | {duration} | {mean:.3f} | {maxd:.3f} | {regimes} | {selected} | {dt:+.3f} | {dc:+.3f} | {drh:+.3f} |".format(
                    trace=trace.get("trace_id", ""),
                    start=window.get("start_step"),
                    end=window.get("end_step"),
                    duration=int(window.get("duration_steps", 0)),
                    mean=float(window.get("mean_abs_delta_sum", 0.0)),
                    maxd=float(window.get("max_abs_delta_sum", 0.0)),
                    regimes=_fmt_counts(window.get("intent_regime_counts", {}) or {}),
                    selected=_fmt_counts(window.get("selected_profile_counts", {}) or {}),
                    dt=float(window.get("mean_delta_target_temp", 0.0)),
                    dc=float(window.get("mean_delta_target_co2", 0.0)),
                    drh=float(window.get("mean_delta_target_rh", 0.0)),
                )
            )

    lines.extend(
        [
            "",
            "## Scorer Large Delta Windows",
            "",
            "| trace | steps | duration | mean delta | score | margin | regimes | scorer selected | dT | dCO2 | dRH | dew risk | hot-dry gap | humid gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trace in audit.get("traces", []) or []:
        for window in trace.get("scorer_large_delta_windows", []) or []:
            lines.append(
                "| {trace} | {start}-{end} | {duration} | {mean:.3f} | {score:.3f} | {margin:.3f} | {regimes} | {selected} | {dt:+.3f} | {dc:+.3f} | {drh:+.3f} | {dew:.3f} | {hotdry:.3f} | {humid:.3f} |".format(
                    trace=trace.get("trace_id", ""),
                    start=window.get("start_step"),
                    end=window.get("end_step"),
                    duration=int(window.get("duration_steps", 0)),
                    mean=float(window.get("mean_abs_delta_sum", 0.0)),
                    score=float(window.get("mean_score", 0.0)),
                    margin=float(window.get("mean_margin", 0.0)),
                    regimes=_fmt_counts(window.get("intent_regime_counts", {}) or {}),
                    selected=_fmt_counts(window.get("scorer_selected_counts", {}) or {}),
                    dt=float(window.get("mean_delta_target_temp", 0.0)),
                    dc=float(window.get("mean_delta_target_co2", 0.0)),
                    drh=float(window.get("mean_delta_target_rh", 0.0)),
                    dew=float(window.get("mean_dew_risk", 0.0)),
                    hotdry=float(window.get("mean_hot_dry_retention_gap", 0.0)),
                    humid=float(window.get("mean_humid_relief_gap", 0.0)),
                )
            )

    trace_warnings = [
        f"{trace.get('trace_id')}: {', '.join(trace.get('warnings', []))}"
        for trace in audit.get("traces", []) or []
        if trace.get("warnings")
    ]
    all_warnings = list(audit.get("warnings", []) or []) + trace_warnings
    if all_warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in all_warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit deterministic profile-generator shadow metadata in traces.")
    parser.add_argument("--input-trace", nargs="+", required=True, help="Trace CSV/JSONL files or directories.")
    parser.add_argument("--delta-threshold", type=float, default=2.0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    audit = audit_traces(args.input_trace, delta_threshold=float(args.delta_threshold))
    write_json(args.output_json, audit)
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(audit), encoding="utf-8")
    print(f"Saved profile-generator shadow audit JSON to {args.output_json}")
    print(f"Saved profile-generator shadow audit report to {args.output_report}")


if __name__ == "__main__":
    main()
