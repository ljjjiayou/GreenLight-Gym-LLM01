"""Extract explanatory windows between two RSPC scoring preset trace sets.

This script compares a reference trace directory, usually ``balanced``, with a
candidate preset trace directory, usually ``hot_dry_relief``.  It is read-only
and is intended to explain *why* a preset changed RH/VPD/temp metrics rather
than to select actions.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

ACTION_FIELDS = ["u_heating", "u_co2", "u_screen", "u_ventilation", "u_lighting", "u_shading"]
STATE_FIELDS = [
    "temp_air",
    "rh_air",
    "vpd_air",
    "vpd_kpa",
    "dew_margin_air",
    "canopy_dew_margin",
    "glob_rad",
    "wind_speed",
]
LOCAL_METRIC_FIELDS = [
    "rh_low_violation",
    "vpd_high_excess",
    "temp_violation",
    "dew_margin_air_lt0_steps",
    "canopy_dew_margin_lt0_steps",
]
SELECTED_CANDIDATE_FIELDS = [
    "selected_fallback_candidate",
    "selected_candidate",
    "rspc_selected_candidate",
    "strategy_best_label",
]
BREAKDOWN_FIELDS = [
    "selected_fallback_score_breakdown",
    "fallback_selected_score_breakdown",
    "rspc_selected_score_breakdown",
]

EPS = 1e-9


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _jsonish(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return text


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _trace_map(trace_dir: str | Path) -> dict[str, Path]:
    root = Path(trace_dir)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return {path.stem: path for path in sorted(root.rglob("*.csv"))}


def _first_present(row: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        if field in row and str(row.get(field, "")).strip() != "":
            return row.get(field)
    return ""


def _action(row: Mapping[str, Any]) -> dict[str, float]:
    return {field: _num(row.get(field)) for field in ACTION_FIELDS if field in row}


def _diff_dict(compare: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    keys = sorted(set(compare) | set(baseline))
    return {key: _num(compare.get(key)) - _num(baseline.get(key)) for key in keys}


def _state_summary(row: Mapping[str, Any]) -> dict[str, float]:
    return {field: _num(row.get(field)) for field in STATE_FIELDS if field in row}


def _metric_delta(compare: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    delta = {
        field: _num(compare.get(field)) - _num(baseline.get(field))
        for field in LOCAL_METRIC_FIELDS
        if field in compare or field in baseline
    }
    if "dew_margin_air_lt0_steps" not in delta and ("dew_margin_air" in compare or "dew_margin_air" in baseline):
        delta["dew_margin_air_lt0_steps"] = (
            1.0 if _num(compare.get("dew_margin_air"), 99.0) < 0.0 else 0.0
        ) - (1.0 if _num(baseline.get("dew_margin_air"), 99.0) < 0.0 else 0.0)
    if "canopy_dew_margin_lt0_steps" not in delta and ("canopy_dew_margin" in compare or "canopy_dew_margin" in baseline):
        delta["canopy_dew_margin_lt0_steps"] = (
            1.0 if _num(compare.get("canopy_dew_margin"), 99.0) < 0.0 else 0.0
        ) - (1.0 if _num(baseline.get("canopy_dew_margin"), 99.0) < 0.0 else 0.0)
    return delta


def _actions_differ(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    return any(abs(_num(a.get(key)) - _num(b.get(key))) > 1e-6 for key in set(a) | set(b))


def _label_window(
    *,
    metric_delta: Mapping[str, float],
    candidate_changed: bool,
    action_changed: bool,
    tomato_changed: bool,
) -> str:
    if _num(metric_delta.get("temp_violation")) > EPS:
        return "temp_tradeoff"
    if _num(metric_delta.get("vpd_high_excess")) > EPS:
        return "vpd_tradeoff"
    if _num(metric_delta.get("rh_low_violation")) < -EPS or _num(metric_delta.get("vpd_high_excess")) < -EPS:
        return "useful_hot_dry_relief"
    if tomato_changed and action_changed:
        return "guardrail_overrode_scorer"
    if candidate_changed or action_changed:
        return "no_effect"
    return "no_material_change"


def _row_window(
    *,
    scenario_id: str,
    baseline: Mapping[str, Any],
    compare: Mapping[str, Any],
) -> dict[str, Any] | None:
    baseline_action = _action(baseline)
    compare_action = _action(compare)
    action_delta = _diff_dict(compare_action, baseline_action)
    action_changed = _actions_differ(compare_action, baseline_action)

    baseline_candidate = str(_first_present(baseline, SELECTED_CANDIDATE_FIELDS))
    compare_candidate = str(_first_present(compare, SELECTED_CANDIDATE_FIELDS))
    candidate_changed = baseline_candidate != compare_candidate

    baseline_breakdown = _jsonish(_first_present(baseline, BREAKDOWN_FIELDS))
    compare_breakdown = _jsonish(_first_present(compare, BREAKDOWN_FIELDS))
    breakdown_changed = baseline_breakdown != compare_breakdown

    tomato_baseline = str(baseline.get("tomato_safety_v2_reasons", ""))
    tomato_compare = str(compare.get("tomato_safety_v2_reasons", ""))
    tomato_changed = (
        str(baseline.get("tomato_safety_v2_applied", "")) != str(compare.get("tomato_safety_v2_applied", ""))
        or tomato_baseline != tomato_compare
    )
    metrics = _metric_delta(compare, baseline)
    local_metric_changed = any(abs(value) > EPS for value in metrics.values())
    if not (action_changed or candidate_changed or breakdown_changed or tomato_changed or local_metric_changed):
        return None

    label = _label_window(
        metric_delta=metrics,
        candidate_changed=candidate_changed,
        action_changed=action_changed,
        tomato_changed=tomato_changed,
    )
    return {
        "scenario_id": scenario_id,
        "step": int(round(_num(compare.get("step", baseline.get("step", 0))))),
        "timestep": int(round(_num(compare.get("timestep", baseline.get("timestep", compare.get("step", 0)))))),
        "state": _state_summary(compare),
        "baseline_action": baseline_action,
        "compare_action": compare_action,
        "action_delta": action_delta,
        "baseline_selected_candidate": baseline_candidate,
        "compare_selected_candidate": compare_candidate,
        "candidate_changed": candidate_changed,
        "baseline_score_breakdown": baseline_breakdown,
        "compare_score_breakdown": compare_breakdown,
        "breakdown_changed": breakdown_changed,
        "baseline_tomato_safety_v2_applied": baseline.get("tomato_safety_v2_applied", ""),
        "compare_tomato_safety_v2_applied": compare.get("tomato_safety_v2_applied", ""),
        "baseline_tomato_safety_v2_reasons": tomato_baseline,
        "compare_tomato_safety_v2_reasons": tomato_compare,
        "tomato_safety_changed": tomato_changed,
        "local_metric_delta": metrics,
        "interpretation_label": label,
    }


def extract_delta_windows(
    *,
    baseline_trace_dir: str | Path,
    compare_trace_dir: str | Path,
    max_windows_per_trace: int = 1000,
) -> dict[str, Any]:
    baseline_map = _trace_map(baseline_trace_dir)
    compare_map = _trace_map(compare_trace_dir)
    common = sorted(set(baseline_map) & set(compare_map))
    missing_baseline = sorted(set(compare_map) - set(baseline_map))
    missing_compare = sorted(set(baseline_map) - set(compare_map))

    windows: list[dict[str, Any]] = []
    scenario_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for stem in common:
        baseline_rows = {int(round(_num(row.get("step", row.get("timestep", 0))))): row for row in _read_csv(baseline_map[stem])}
        compare_rows = {int(round(_num(row.get("step", row.get("timestep", 0))))): row for row in _read_csv(compare_map[stem])}
        trace_windows: list[dict[str, Any]] = []
        for step in sorted(set(baseline_rows) & set(compare_rows)):
            window = _row_window(scenario_id=stem, baseline=baseline_rows[step], compare=compare_rows[step])
            if window is not None:
                trace_windows.append(window)
        trace_windows = trace_windows[:max_windows_per_trace]
        for window in trace_windows:
            windows.append(window)
            scenario_counts[str(window["scenario_id"])] += 1
            label_counts[str(window["interpretation_label"])] += 1

    return {
        "schema_version": "rspc_scoring_delta_windows_v1",
        "baseline_trace_dir": str(baseline_trace_dir),
        "compare_trace_dir": str(compare_trace_dir),
        "trace_count": len(common),
        "missing_baseline_traces": missing_baseline,
        "missing_compare_traces": missing_compare,
        "window_count": len(windows),
        "label_counts": dict(sorted(label_counts.items())),
        "scenario_window_counts": dict(sorted(scenario_counts.items())),
        "windows": windows,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# RSPC Scoring Delta Windows",
        "",
        f"- Baseline trace dir: `{report.get('baseline_trace_dir', '')}`",
        f"- Compare trace dir: `{report.get('compare_trace_dir', '')}`",
        f"- Trace count: {report.get('trace_count', 0)}",
        f"- Window count: {report.get('window_count', 0)}",
        "",
        "## Label Counts",
        "",
        "| label | count |",
        "| --- | ---: |",
    ]
    for label, count in dict(report.get("label_counts", {})).items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Windows", ""])
    lines.append(
        "| scenario | step | label | d_RHlow | d_VPDhi | d_temp | baseline_candidate | compare_candidate | action_delta |"
    )
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |")
    for window in list(report.get("windows", []))[:120]:
        metrics = window.get("local_metric_delta", {}) if isinstance(window, Mapping) else {}
        action_delta = window.get("action_delta", {}) if isinstance(window, Mapping) else {}
        compact_action = json.dumps(action_delta, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(window.get("scenario_id", "")),
                    str(window.get("step", "")),
                    str(window.get("interpretation_label", "")),
                    f"{_num(metrics.get('rh_low_violation')):.3f}",
                    f"{_num(metrics.get('vpd_high_excess')):.3f}",
                    f"{_num(metrics.get('temp_violation')):.3f}",
                    str(window.get("baseline_selected_candidate", "")),
                    str(window.get("compare_selected_candidate", "")),
                    f"`{compact_action}`",
                ]
            )
            + " |"
        )
    if report.get("missing_baseline_traces") or report.get("missing_compare_traces"):
        lines.extend(["", "## Missing Traces", ""])
        for stem in report.get("missing_baseline_traces", []):
            lines.append(f"- missing baseline: {stem}")
        for stem in report.get("missing_compare_traces", []):
            lines.append(f"- missing compare: {stem}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trace-dir", required=True)
    parser.add_argument("--compare-trace-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--max-windows-per-trace", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = extract_delta_windows(
        baseline_trace_dir=args.baseline_trace_dir,
        compare_trace_dir=args.compare_trace_dir,
        max_windows_per_trace=args.max_windows_per_trace,
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
    print(f"windows={report['window_count']} traces={report['trace_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
