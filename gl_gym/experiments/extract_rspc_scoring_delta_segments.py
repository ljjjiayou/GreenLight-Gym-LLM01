"""Group RSPC scoring delta windows into mechanism-level segments."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.extract_rspc_scoring_delta_windows import (
    ACTION_FIELDS,
    _num,
    build_markdown_report as build_window_markdown_report,
    extract_delta_windows,
)


def _action_magnitude(window: Mapping[str, Any], *, min_action_delta: float) -> dict[str, float]:
    raw = window.get("action_delta", {}) if isinstance(window.get("action_delta"), Mapping) else {}
    return {field: _num(raw.get(field)) for field in ACTION_FIELDS if abs(_num(raw.get(field))) >= min_action_delta}


def _segment_interpretation(segment: Mapping[str, Any]) -> str:
    if _num(segment.get("cumulative_d_canopy_lt0")) > 1e-9 or _num(segment.get("cumulative_d_dew_lt0")) > 1e-9:
        return "hard_safety_regression"
    if _num(segment.get("guardrail_overrode_count")) >= max(1, int(segment.get("length", 0)) // 2):
        return "guardrail_dominated"
    if _num(segment.get("cumulative_d_temp")) > 1e-9:
        return "temp_tradeoff_segment"
    if _num(segment.get("cumulative_d_VPDhi")) > 1e-9 and _num(segment.get("cumulative_d_RHlow")) >= -1e-9:
        return "vpd_tradeoff_segment"
    if _num(segment.get("cumulative_d_RHlow")) < -1e-9 or _num(segment.get("cumulative_d_VPDhi")) < -1e-9:
        return "useful_humidity_retention"
    if _num(segment.get("candidate_changed_count")) == 0 and (
        abs(_num(segment.get("cumulative_d_RHlow"))) > 1e-9
        or abs(_num(segment.get("cumulative_d_VPDhi"))) > 1e-9
        or abs(_num(segment.get("cumulative_d_temp"))) > 1e-9
    ):
        return "delayed_effect"
    return "no_material_effect"


def _build_segment(scenario_id: str, windows: Sequence[Mapping[str, Any]], index: int, *, min_action_delta: float) -> dict[str, Any]:
    label_counts = Counter(str(w.get("interpretation_label", "")) for w in windows)
    action_sums: Counter[str] = Counter()
    abs_action_sums: Counter[str] = Counter()
    candidate_changed_count = 0
    guardrail_count = 0
    cumulative = Counter()
    for window in windows:
        metrics = window.get("local_metric_delta", {}) if isinstance(window.get("local_metric_delta"), Mapping) else {}
        cumulative["rh_low_violation"] += _num(metrics.get("rh_low_violation"))
        cumulative["vpd_high_excess"] += _num(metrics.get("vpd_high_excess"))
        cumulative["temp_violation"] += _num(metrics.get("temp_violation"))
        cumulative["dew_margin_air_lt0_steps"] += _num(metrics.get("dew_margin_air_lt0_steps"))
        cumulative["canopy_dew_margin_lt0_steps"] += _num(metrics.get("canopy_dew_margin_lt0_steps"))
        if bool(window.get("candidate_changed")):
            candidate_changed_count += 1
        if str(window.get("interpretation_label")) == "guardrail_overrode_scorer" or bool(window.get("tomato_safety_changed")):
            guardrail_count += 1
        for field, value in _action_magnitude(window, min_action_delta=min_action_delta).items():
            action_sums[field] += value
            abs_action_sums[field] += abs(value)
    dominant_actuator = ""
    if abs_action_sums:
        dominant_actuator = sorted(abs_action_sums.items(), key=lambda item: (-item[1], item[0]))[0][0]
    segment = {
        "segment_id": f"{scenario_id}__seg{index:03d}",
        "scenario_id": scenario_id,
        "start_step": int(windows[0].get("step", 0)),
        "end_step": int(windows[-1].get("step", 0)),
        "length": len(windows),
        "dominant_label": label_counts.most_common(1)[0][0] if label_counts else "",
        "label_counts": dict(sorted(label_counts.items())),
        "cumulative_d_RHlow": float(cumulative["rh_low_violation"]),
        "cumulative_d_VPDhi": float(cumulative["vpd_high_excess"]),
        "cumulative_d_temp": float(cumulative["temp_violation"]),
        "cumulative_d_dew_lt0": float(cumulative["dew_margin_air_lt0_steps"]),
        "cumulative_d_canopy_lt0": float(cumulative["canopy_dew_margin_lt0_steps"]),
        "dominant_actuator_delta": dominant_actuator,
        "action_delta_sum": dict(sorted(action_sums.items())),
        "candidate_changed_count": candidate_changed_count,
        "guardrail_overrode_count": guardrail_count,
    }
    segment["interpretation"] = _segment_interpretation(segment)
    segment["net_benefit_score"] = (
        -_num(segment["cumulative_d_RHlow"])
        - _num(segment["cumulative_d_VPDhi"])
        - _num(segment["cumulative_d_temp"])
    )
    return segment


def build_segments_from_windows(
    window_report: Mapping[str, Any],
    *,
    min_action_delta: float = 1e-4,
    max_gap: int = 2,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for window in window_report.get("windows", []) or []:
        if isinstance(window, Mapping):
            grouped[str(window.get("scenario_id", ""))].append(window)
    segments: list[dict[str, Any]] = []
    for scenario_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda w: int(w.get("step", 0)))
        current: list[Mapping[str, Any]] = []
        segment_index = 1
        for window in ordered:
            if not current:
                current = [window]
                continue
            gap = int(window.get("step", 0)) - int(current[-1].get("step", 0))
            if gap <= max_gap:
                current.append(window)
            else:
                segments.append(_build_segment(scenario_id, current, segment_index, min_action_delta=min_action_delta))
                segment_index += 1
                current = [window]
        if current:
            segments.append(_build_segment(scenario_id, current, segment_index, min_action_delta=min_action_delta))

    by_scenario: dict[str, dict[str, Any]] = {}
    for scenario_id in sorted(grouped):
        scenario_segments = [s for s in segments if s["scenario_id"] == scenario_id]
        positive = sorted(scenario_segments, key=lambda s: (-_num(s.get("net_benefit_score")), s["segment_id"]))[:5]
        negative = sorted(scenario_segments, key=lambda s: (_num(s.get("net_benefit_score")), s["segment_id"]))[:5]
        by_scenario[scenario_id] = {
            "segment_count": len(scenario_segments),
            "top_positive_segments": [s["segment_id"] for s in positive],
            "top_negative_segments": [s["segment_id"] for s in negative],
        }
    interpretation_counts = Counter(str(segment.get("interpretation", "")) for segment in segments)
    return {
        "schema_version": "rspc_scoring_delta_segments_v1",
        "window_report": {
            "trace_count": window_report.get("trace_count", 0),
            "window_count": window_report.get("window_count", 0),
            "baseline_trace_dir": window_report.get("baseline_trace_dir", ""),
            "compare_trace_dir": window_report.get("compare_trace_dir", ""),
        },
        "segment_count": len(segments),
        "interpretation_counts": dict(sorted(interpretation_counts.items())),
        "segments_by_scenario": by_scenario,
        "segments": segments,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# RSPC Scoring Delta Segments",
        "",
        f"- Source windows: {report.get('window_report', {}).get('window_count', 0)}",
        f"- Segments: {report.get('segment_count', 0)}",
        "",
        "## Interpretation Counts",
        "",
        "| interpretation | count |",
        "| --- | ---: |",
    ]
    for label, count in dict(report.get("interpretation_counts", {})).items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Segments", ""])
    lines.append("| segment | scenario | steps | interp | d_RHlow | d_VPDhi | d_temp | actuator | cand_changed | guardrail |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |")
    for segment in list(report.get("segments", []))[:160]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(segment.get("segment_id", "")),
                    str(segment.get("scenario_id", "")),
                    f"{segment.get('start_step')}..{segment.get('end_step')}",
                    str(segment.get("interpretation", "")),
                    f"{_num(segment.get('cumulative_d_RHlow')):.3f}",
                    f"{_num(segment.get('cumulative_d_VPDhi')):.3f}",
                    f"{_num(segment.get('cumulative_d_temp')):.3f}",
                    str(segment.get("dominant_actuator_delta", "")),
                    f"{_num(segment.get('candidate_changed_count')):.0f}",
                    f"{_num(segment.get('guardrail_overrode_count')):.0f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trace-dir", required=True)
    parser.add_argument("--compare-trace-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--window-output-json", default="")
    parser.add_argument("--window-output-md", default="")
    parser.add_argument("--min-action-delta", type=float, default=1e-4)
    parser.add_argument("--max-gap", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    windows = extract_delta_windows(
        baseline_trace_dir=args.baseline_trace_dir,
        compare_trace_dir=args.compare_trace_dir,
    )
    if args.window_output_json:
        path = Path(args.window_output_json)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(windows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if args.window_output_md:
        path = Path(args.window_output_md)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_window_markdown_report(windows), encoding="utf-8")
    report = build_segments_from_windows(windows, min_action_delta=args.min_action_delta, max_gap=args.max_gap)
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
    print(f"segments={report['segment_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
