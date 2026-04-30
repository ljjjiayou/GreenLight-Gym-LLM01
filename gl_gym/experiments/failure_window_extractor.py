"""Extract mechanism-level failure windows from benchmark step traces."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


ACTION_KEYS = (
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
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


def _num(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _truthy(row: Dict[str, Any], key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
        reader = csv.DictReader(f)
        return [{key: _coerce(value) for key, value in row.items()} for row in reader]


def _window_flags(rows: Sequence[Dict[str, Any]], kind: str) -> List[bool]:
    flags: List[bool] = []
    for row in rows:
        rh = _num(row, "rh_air", 70.0)
        vpd = _num(row, "vpd_air", _num(row, "vpd_kpa", 0.0))
        temp = _num(row, "temp_air", 20.0)
        vent = _num(row, "u_ventilation", 0.0)
        target_rh = row.get("target_rh")
        dew_margin = _num(row, "dew_margin_min", min(_num(row, "dew_margin_air", 3.0), _num(row, "canopy_dew_margin", 3.0)))
        extreme_dew = rh >= 94.0 or dew_margin < 0.4
        if kind == "dry_stress_window":
            flags.append(rh < 50.0 or vpd > 1.20 or _num(row, "rh_low_violation") > 0.0 or _num(row, "vpd_high_excess") > 0.0)
        elif kind == "cold_recovery_window":
            flags.append(_num(row, "temp_violation") > 0.0 or temp < 12.0)
        elif kind == "target_action_mismatch_window":
            mismatch = target_rh is not None and _num(row, "target_rh") - rh > 10.0 and vent > 0.12
            flags.append(bool(mismatch))
        elif kind == "cold_vent_risk_window":
            flags.append(_truthy(row, "cold_vent_risk") or (temp < 15.5 and vent > 0.25 and not extreme_dew))
        elif kind == "shadow_opportunity_window":
            candidate = str(row.get("mc_sero_best_candidate") or "")
            source = str(row.get("source") or "")
            flags.append(
                _truthy(row, "mc_sero_would_select")
                and _num(row, "mc_sero_margin") >= 0.10
                and candidate not in {"", "none", source}
            )
        else:
            raise ValueError(f"Unknown window kind: {kind}")
    return flags


def _segments(flags: Sequence[bool], *, merge_gap: int = 1) -> List[tuple[int, int]]:
    raw: List[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            raw.append((start, index - 1))
            start = None
    if start is not None:
        raw.append((start, len(flags) - 1))
    if not raw:
        return []
    merged = [raw[0]]
    for start, end in raw[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= merge_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _row_brief(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step": int(_num(row, "step", 0)),
        "hour": round(_num(row, "hour_of_day", 0.0), 3),
        "temp": round(_num(row, "temp_air", 0.0), 3),
        "rh": round(_num(row, "rh_air", 0.0), 3),
        "vpd": round(_num(row, "vpd_air", _num(row, "vpd_kpa", 0.0)), 3),
        "vent": round(_num(row, "u_ventilation", 0.0), 3),
        "heat": round(_num(row, "u_heating", 0.0), 3),
        "source": row.get("source"),
        "target_rh": row.get("target_rh"),
        "mc_sero_best_candidate": row.get("mc_sero_best_candidate"),
    }


def _summarize_window(
    rows: Sequence[Dict[str, Any]],
    *,
    kind: str,
    trace_id: str,
    start_index: int,
    end_index: int,
    context_steps: int,
) -> Dict[str, Any]:
    selected = list(rows[start_index : end_index + 1])
    pre = list(rows[max(0, start_index - context_steps) : start_index])
    post = list(rows[end_index + 1 : min(len(rows), end_index + 1 + context_steps)])
    start_row = selected[0]
    end_row = selected[-1]
    action_means = {
        key: float(mean(_num(row, key) for row in selected))
        for key in ACTION_KEYS
    }
    source_counts = _counts(row.get("source") for row in selected)
    candidate_counts = _counts(row.get("mc_sero_best_candidate") for row in selected if row.get("mc_sero_best_candidate"))
    reason_counts = _counts(
        reason
        for row in selected
        for reason in str(row.get("tomato_safety_v2_reasons") or "").split(",")
        if reason.strip()
    )
    return {
        "trace_id": trace_id,
        "kind": kind,
        "start_step": int(_num(start_row, "step", start_index)),
        "end_step": int(_num(end_row, "step", end_index)),
        "duration_steps": int(len(selected)),
        "start_hour": float(_num(start_row, "hour_of_day", 0.0)),
        "end_hour": float(_num(end_row, "hour_of_day", 0.0)),
        "min_rh": float(min(_num(row, "rh_air", 100.0) for row in selected)),
        "max_rh": float(max(_num(row, "rh_air", 0.0) for row in selected)),
        "min_temp": float(min(_num(row, "temp_air", 100.0) for row in selected)),
        "max_vpd": float(max(_num(row, "vpd_air", _num(row, "vpd_kpa", 0.0)) for row in selected)),
        "rh_low_area": float(sum(_num(row, "rh_low_violation") for row in selected)),
        "rh_high_area": float(sum(_num(row, "rh_high_violation") for row in selected)),
        "vpd_high_area": float(sum(_num(row, "vpd_high_excess") for row in selected)),
        "temp_violation_area": float(sum(_num(row, "temp_violation") for row in selected)),
        "mean_actions": action_means,
        "source_counts": source_counts,
        "mc_sero_best_candidate_counts": candidate_counts,
        "tomato_safety_v2_reason_counts": reason_counts,
        "tomato_safety_v2_suppressed_replan_steps": int(
            sum(_truthy(row, "tomato_safety_v2_suppressed_replan") for row in selected)
        ),
        "pre_context": [_row_brief(row) for row in pre],
        "post_context": [_row_brief(row) for row in post],
    }


def extract_failure_windows(
    rows: Sequence[Dict[str, Any]],
    *,
    trace_id: str = "trace",
    context_steps: int = 6,
    merge_gap: int = 1,
) -> List[Dict[str, Any]]:
    kinds = [
        "dry_stress_window",
        "cold_recovery_window",
        "target_action_mismatch_window",
        "cold_vent_risk_window",
        "shadow_opportunity_window",
    ]
    windows: List[Dict[str, Any]] = []
    sorted_rows = sorted(rows, key=lambda row: _num(row, "step", 0.0))
    for kind in kinds:
        flags = _window_flags(sorted_rows, kind)
        for start_index, end_index in _segments(flags, merge_gap=merge_gap):
            windows.append(
                _summarize_window(
                    sorted_rows,
                    kind=kind,
                    trace_id=trace_id,
                    start_index=start_index,
                    end_index=end_index,
                    context_steps=context_steps,
                )
            )
    return sorted(
        windows,
        key=lambda item: (
            str(item["trace_id"]),
            int(item["start_step"]),
            str(item["kind"]),
        ),
    )


def build_report(windows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Failure Window Report",
        "",
        f"- Windows: {len(windows)}",
        "",
        "| trace | kind | steps | duration | min RH | max VPD | temp viol | mean vent | sources |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for window in windows:
        mean_vent = float(window.get("mean_actions", {}).get("u_ventilation", 0.0))
        lines.append(
            "| {trace} | {kind} | {start}-{end} | {duration} | {min_rh:.2f} | {max_vpd:.2f} | {temp:.3f} | {vent:.3f} | {sources} |".format(
                trace=window.get("trace_id"),
                kind=window.get("kind"),
                start=window.get("start_step"),
                end=window.get("end_step"),
                duration=window.get("duration_steps"),
                min_rh=float(window.get("min_rh", 0.0)),
                max_vpd=float(window.get("max_vpd", 0.0)),
                temp=float(window.get("temp_violation_area", 0.0)),
                vent=mean_vent,
                sources=json.dumps(window.get("source_counts", {}), ensure_ascii=False),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract failure windows from benchmark traces.")
    parser.add_argument("--input-trace", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--context-steps", type=int, default=6)
    parser.add_argument("--merge-gap", type=int, default=1)
    args = parser.parse_args()

    all_windows: List[Dict[str, Any]] = []
    for trace in args.input_trace:
        path = Path(trace)
        all_windows.extend(
            extract_failure_windows(
                read_trace(path),
                trace_id=path.stem,
                context_steps=int(args.context_steps),
                merge_gap=int(args.merge_gap),
            )
        )

    output = {"window_count": len(all_windows), "windows": all_windows}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(build_report(all_windows), encoding="utf-8")
    print(f"Saved failure windows JSON to {output_json}")
    print(f"Saved failure windows report to {output_report}")


if __name__ == "__main__":
    main()
