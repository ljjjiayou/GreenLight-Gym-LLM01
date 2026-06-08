"""Audit the canonical selected-suite canopy dew failure case.

The script is read-only.  It compares a baseline trace against a scoring
preset trace and extracts the first extra canopy-dew hard-safety violation plus
the surrounding actuator, candidate, and guardrail evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ACTION_FIELDS = ("u_heating", "u_co2", "u_screen", "u_ventilation", "u_lighting", "u_shading")
VECTOR_ACTION_FIELDS = ("u_heating", "u_co2", "u_screen", "u_ventilation", "u_lighting", "u_shading")
STATE_FIELDS = ("temp_air", "rh_air", "vpd_air", "dew_margin_air", "canopy_dew_margin", "glob_rad", "wind_speed")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_json(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _read_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_trace(root: str | Path, scenario_id: str) -> Path:
    path = Path(root)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file():
        return path
    matches = sorted(path.rglob(f"{scenario_id}*.csv"))
    if not matches:
        raise FileNotFoundError(f"trace not found for {scenario_id} under {path}")
    return matches[0]


def _rows_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(round(_num(row.get("step", row.get("timestep", 0))))): row for row in rows}


def _action(row: Mapping[str, Any]) -> dict[str, float]:
    return {field: _num(row.get(field)) for field in ACTION_FIELDS if field in row}


def _action_delta(compare: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    left = _action(compare)
    right = _action(baseline)
    return {field: _num(left.get(field)) - _num(right.get(field)) for field in sorted(set(left) | set(right))}


def _selected_candidate(row: Mapping[str, Any]) -> str:
    for key in ("rspc_action_selected_name", "selected_fallback_candidate", "strategy_best_label"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def _vector_action(row: Mapping[str, Any], key: str) -> dict[str, float]:
    values = _parse_json(row.get(key), default=[])
    if not isinstance(values, list):
        return {}
    return {field: _num(values[idx]) for idx, field in enumerate(VECTOR_ACTION_FIELDS) if idx < len(values)}


def _row_summary(step: int, baseline: Mapping[str, Any], compare: Mapping[str, Any]) -> dict[str, Any]:
    state = {
        f"baseline_{field}": _num(baseline.get(field))
        for field in STATE_FIELDS
        if field in baseline
    }
    state.update({f"compare_{field}": _num(compare.get(field)) for field in STATE_FIELDS if field in compare})
    return {
        "step": step,
        "baseline_canopy_lt0": _num(baseline.get("canopy_dew_margin"), 99.0) < 0.0,
        "compare_canopy_lt0": _num(compare.get("canopy_dew_margin"), 99.0) < 0.0,
        "extra_canopy_lt0": (
            _num(compare.get("canopy_dew_margin"), 99.0) < 0.0
            and _num(baseline.get("canopy_dew_margin"), 99.0) >= 0.0
        ),
        "state": state,
        "baseline_action": _action(baseline),
        "compare_action": _action(compare),
        "action_delta": _action_delta(compare, baseline),
        "baseline_selected_candidate": _selected_candidate(baseline),
        "compare_selected_candidate": _selected_candidate(compare),
        "baseline_tomato_safety_v2_reasons": str(baseline.get("tomato_safety_v2_reasons", "") or ""),
        "compare_tomato_safety_v2_reasons": str(compare.get("tomato_safety_v2_reasons", "") or ""),
        "baseline_tomato_safety_v2_applied": _truthy(baseline.get("tomato_safety_v2_applied")),
        "compare_tomato_safety_v2_applied": _truthy(compare.get("tomato_safety_v2_applied")),
        "compare_tomato_safety_before": _vector_action(compare, "tomato_safety_v2_before"),
        "compare_tomato_safety_after": _vector_action(compare, "tomato_safety_v2_after"),
        "compare_dry_recovery_before": _vector_action(compare, "rspc_action_dry_recovery_before"),
        "compare_dry_recovery_after": _vector_action(compare, "rspc_action_dry_recovery_after"),
    }


def audit_failure_case(
    *,
    baseline_trace: str | Path,
    compare_trace: str | Path,
    scenario_id: str,
    start_step: int = 220,
    end_step: int = 239,
    case_id: str = "hot_dry_relief_y2020_d120_s44_canopy_lt0_regression",
) -> dict[str, Any]:
    baseline_rows = _rows_by_step(_read_trace(baseline_trace))
    compare_rows = _rows_by_step(_read_trace(compare_trace))
    rows: list[dict[str, Any]] = []
    first_extra_step: int | None = None
    for step in range(int(start_step), int(end_step) + 1):
        if step not in baseline_rows or step not in compare_rows:
            continue
        summary = _row_summary(step, baseline_rows[step], compare_rows[step])
        rows.append(summary)
        if summary["extra_canopy_lt0"] and first_extra_step is None:
            first_extra_step = step
    screen_delta_sum = sum(_num(row.get("action_delta", {}).get("u_screen")) for row in rows)
    vent_delta_sum = sum(_num(row.get("action_delta", {}).get("u_ventilation")) for row in rows)
    report = {
        "schema_version": "hot_dry_failure_case_audit_v1",
        "case_id": case_id,
        "scenario_id": scenario_id,
        "baseline_trace": str(baseline_trace),
        "compare_trace": str(compare_trace),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "hard_safety_regression": first_extra_step is not None,
        "first_extra_canopy_lt0_step": first_extra_step,
        "baseline_canopy_lt0_steps": sum(1 for row in rows if row["baseline_canopy_lt0"]),
        "compare_canopy_lt0_steps": sum(1 for row in rows if row["compare_canopy_lt0"]),
        "extra_canopy_lt0_steps": sum(1 for row in rows if row["extra_canopy_lt0"]),
        "screen_delta_sum": float(screen_delta_sum),
        "ventilation_delta_sum": float(vent_delta_sum),
        "tomato_safety_reason_counts": {},
        "rows": rows,
    }
    counts: dict[str, int] = {}
    for row in rows:
        for reason in str(row.get("compare_tomato_safety_v2_reasons", "") or "").split(","):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    report["tomato_safety_reason_counts"] = dict(sorted(counts.items()))
    return report


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Hot-Dry Failure Case Audit",
        "",
        f"- Case: `{report.get('case_id', '')}`",
        f"- Scenario: `{report.get('scenario_id', '')}`",
        f"- Steps: {report.get('start_step')}..{report.get('end_step')}",
        f"- Hard safety regression: **{str(report.get('hard_safety_regression')).upper()}**",
        f"- First extra canopy <0 step: {report.get('first_extra_canopy_lt0_step')}",
        f"- Extra canopy <0 steps: {report.get('extra_canopy_lt0_steps', 0)}",
        f"- Sum d_screen: { _num(report.get('screen_delta_sum')):.4f}",
        f"- Sum d_ventilation: { _num(report.get('ventilation_delta_sum')):.4f}",
        "",
        "## Window Rows",
        "",
        "| step | base_canopy | cmp_canopy | extra_lt0 | d_screen | d_vent | base_candidate | cmp_candidate | cmp_tomato_reasons |",
        "| ---: | ---: | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in report.get("rows", []) or []:
        state = row.get("state", {}) if isinstance(row.get("state"), Mapping) else {}
        delta = row.get("action_delta", {}) if isinstance(row.get("action_delta"), Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("step", "")),
                    f"{_num(state.get('baseline_canopy_dew_margin')):.3f}",
                    f"{_num(state.get('compare_canopy_dew_margin')):.3f}",
                    str(row.get("extra_canopy_lt0", "")),
                    f"{_num(delta.get('u_screen')):.4f}",
                    f"{_num(delta.get('u_ventilation')):.4f}",
                    str(row.get("baseline_selected_candidate", "")),
                    str(row.get("compare_selected_candidate", "")),
                    str(row.get("compare_tomato_safety_v2_reasons", "")) or "-",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trace-dir", required=True)
    parser.add_argument("--compare-trace-dir", required=True)
    parser.add_argument("--scenario-id", default="y2020_d120_s44_n240_llm_rspc_v2")
    parser.add_argument("--start-step", type=int, default=220)
    parser.add_argument("--end-step", type=int, default=239)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    baseline = _resolve_trace(args.baseline_trace_dir, args.scenario_id)
    compare = _resolve_trace(args.compare_trace_dir, args.scenario_id)
    report = audit_failure_case(
        baseline_trace=baseline,
        compare_trace=compare,
        scenario_id=args.scenario_id,
        start_step=args.start_step,
        end_step=args.end_step,
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
    print(f"hard_safety_regression={report['hard_safety_regression']} first_extra={report['first_extra_canopy_lt0_step']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
