"""Audit shadow canopy-dew boundary metadata from frozen replay traces.

The audit is read-only.  It checks whether the metadata-only boundary would
flag the canonical y2020_d120_s44 canopy regression window and how often it
would also flag clean pure-hot-dry rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def _iter_trace_paths(root: str | Path, scenario_id: str) -> list[Path]:
    path = Path(root)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file():
        return [path]
    matches = sorted(path.rglob(f"{scenario_id}*.csv"))
    if not matches:
        raise FileNotFoundError(f"trace not found for {scenario_id} under {path}")
    return matches


def _rows_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(round(_num(row.get("step", row.get("timestep", 0))))): row for row in rows}


def _is_pure_hot_dry_proxy(row: Mapping[str, Any]) -> bool:
    return bool(
        (
            _truthy(row.get("rspc_action_hot_dry_active"))
            or _truthy(row.get("rspc_action_hot_dry_semantic_active"))
            or _num(row.get("rh_low_violation")) > 0.0
            or _num(row.get("vpd_high_excess")) > 0.0
        )
        and _num(row.get("dew_margin_air"), 3.0) >= 1.0
        and _num(row.get("canopy_dew_margin"), 3.0) >= 1.0
        and _num(row.get("temp_air"), 20.0) < 32.0
    )


def _step_record(row: Mapping[str, Any], step: int) -> dict[str, Any]:
    would_reject = _truthy(row.get("canopy_boundary_shadow_would_reject"))
    warning_v2 = _truthy(row.get("canopy_boundary_shadow_warning_v2")) or _truthy(
        row.get("final_action_predicted_canopy_warning_v2")
    )
    would_reject_v2 = _truthy(row.get("canopy_boundary_shadow_would_reject_v2")) or _truthy(
        row.get("final_action_would_fail_canopy_boundary_v2")
    )
    active = _truthy(row.get("canopy_boundary_shadow_active"))
    boundary_warning = bool(would_reject or warning_v2 or would_reject_v2)
    return {
        "step": int(step),
        "boundary_active": active,
        "would_reject": would_reject,
        "would_reject_v2": would_reject_v2,
        "warning_v2": warning_v2,
        "boundary_warning": boundary_warning,
        "reason": str(row.get("canopy_boundary_shadow_reason", "") or ""),
        "reason_v2": str(row.get("canopy_boundary_shadow_reason_v2", "") or row.get("final_action_risk_reason_v2", "") or ""),
        "predicted_canopy_dew_margin_next": _num(
            row.get("canopy_boundary_shadow_predicted_canopy_dew_margin_next"),
            99.0,
        ),
        "predicted_canopy_dew_margin_next_v2": _num(
            row.get("canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2"),
            _num(row.get("final_action_predicted_canopy_dew_margin_next_v2"), 99.0),
        ),
        "screen_delta": _num(row.get("canopy_boundary_shadow_screen_delta")),
        "ventilation_delta": _num(row.get("canopy_boundary_shadow_vent_delta")),
        "screen_increase_under_canopy_risk": _truthy(
            row.get("canopy_boundary_shadow_screen_increase_under_canopy_risk")
        ),
        "ventilation_decrease_under_canopy_risk": _truthy(
            row.get("canopy_boundary_shadow_ventilation_decrease_under_canopy_risk")
        ),
        "high_screen_under_canopy_risk": _truthy(row.get("canopy_boundary_shadow_high_screen_under_canopy_risk")),
        "low_ventilation_under_canopy_risk": _truthy(
            row.get("canopy_boundary_shadow_low_ventilation_under_canopy_risk")
        ),
        "pure_hot_dry_proxy": _is_pure_hot_dry_proxy(row),
        "state": {
            "temp_air": _num(row.get("temp_air")),
            "rh_air": _num(row.get("rh_air")),
            "vpd_air": _num(row.get("vpd_air")),
            "dew_margin_air": _num(row.get("dew_margin_air")),
            "canopy_dew_margin": _num(row.get("canopy_dew_margin")),
        },
    }


def audit_trace(
    *,
    trace_path: str | Path,
    preset: str,
    scenario_id: str,
    start_step: int,
    end_step: int,
    failure_step: int,
) -> dict[str, Any]:
    rows_by_step = _rows_by_step(_read_trace(trace_path))
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    would_block_failure = False
    first_boundary_warning_step: int | None = None
    affected_pure_hot_dry = 0
    false_positive_steps: list[int] = []
    false_positive_pure_hot_dry_steps: list[int] = []
    for step in range(int(start_step), int(end_step) + 1):
        row = rows_by_step.get(step)
        if row is None:
            continue
        record = _step_record(row, step)
        records.append(record)
        if record["boundary_active"]:
            counts["boundary_active"] += 1
        if record["would_reject"]:
            counts["would_reject"] += 1
        if record["would_reject_v2"]:
            counts["would_reject_v2"] += 1
        if record["warning_v2"]:
            counts["warning_v2"] += 1
        if record["boundary_warning"]:
            counts["boundary_warning"] += 1
            if first_boundary_warning_step is None:
                first_boundary_warning_step = int(step)
            if step <= int(failure_step):
                would_block_failure = True
            if record["pure_hot_dry_proxy"]:
                affected_pure_hot_dry += 1
                false_positive_steps.append(step)
                false_positive_pure_hot_dry_steps.append(step)
        if record["screen_increase_under_canopy_risk"]:
            counts["screen_increase_under_canopy_risk"] += 1
        if record["ventilation_decrease_under_canopy_risk"]:
            counts["ventilation_decrease_under_canopy_risk"] += 1
        if record["high_screen_under_canopy_risk"]:
            counts["high_screen_under_canopy_risk"] += 1
        if record["low_ventilation_under_canopy_risk"]:
            counts["low_ventilation_under_canopy_risk"] += 1
    lead_time_steps = (
        int(failure_step) - int(first_boundary_warning_step)
        if first_boundary_warning_step is not None
        else None
    )
    catches_hard_event = bool(any(r["step"] == int(failure_step) and r["boundary_warning"] for r in records))
    catches_pre_failure_warning = bool(
        first_boundary_warning_step is not None and first_boundary_warning_step <= int(failure_step) - 1
    )
    prevention_window_available = catches_pre_failure_warning
    hard_event_only = bool(catches_hard_event and not catches_pre_failure_warning)
    return {
        "preset": preset,
        "scenario_id": scenario_id,
        "trace_path": str(trace_path),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "failure_step": int(failure_step),
        "first_boundary_warning_step": first_boundary_warning_step,
        "lead_time_steps": lead_time_steps,
        "catches_hard_event": catches_hard_event,
        "catches_pre_failure_warning": catches_pre_failure_warning,
        "hard_event_only": hard_event_only,
        "prevention_window_available": prevention_window_available,
        "controlled_repair_allowed": bool(prevention_window_available),
        "would_block_failure_case": bool(would_block_failure),
        "affected_pure_hot_dry_steps": int(affected_pure_hot_dry),
        "false_positive_steps": false_positive_steps,
        "false_positive_pure_hot_dry_steps": false_positive_pure_hot_dry_steps,
        "counts": dict(sorted(counts.items())),
        "records": records,
    }


def build_report(
    *,
    trace_dirs: Sequence[str | Path],
    scenario_id: str,
    start_step: int,
    end_step: int,
    failure_step: int,
) -> dict[str, Any]:
    traces = []
    aggregate: Counter[str] = Counter()
    seen: set[Path] = set()
    for raw in trace_dirs:
        trace_paths = _iter_trace_paths(raw, scenario_id)
        for trace_path in trace_paths:
            resolved = trace_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            preset = trace_path.parent.name or "trace"
            if preset.lower() == "traces":
                preset = trace_path.stem
            trace = audit_trace(
                trace_path=trace_path,
                preset=preset,
                scenario_id=scenario_id,
                start_step=start_step,
                end_step=end_step,
                failure_step=failure_step,
            )
            traces.append(trace)
            aggregate.update(trace.get("counts", {}))
    return {
        "schema_version": "canopy_boundary_shadow_audit_v1",
        "scenario_id": scenario_id,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "failure_step": int(failure_step),
        "trace_count": len(traces),
        "aggregate_counts": dict(sorted(aggregate.items())),
        "would_block_any_failure_case": any(bool(t.get("would_block_failure_case")) for t in traces),
        "prevention_window_available": any(bool(t.get("prevention_window_available")) for t in traces),
        "controlled_repair_allowed": any(bool(t.get("controlled_repair_allowed")) for t in traces),
        "affected_pure_hot_dry_steps": int(sum(int(t.get("affected_pure_hot_dry_steps", 0)) for t in traces)),
        "traces": traces,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Canopy Boundary Shadow Audit",
        "",
        f"- Scenario: `{report.get('scenario_id', '')}`",
        f"- Steps: {report.get('start_step')}..{report.get('end_step')}",
        f"- Failure step: {report.get('failure_step')}",
        f"- Would block any failure case: **{str(report.get('would_block_any_failure_case')).upper()}**",
        f"- Prevention window available: **{str(report.get('prevention_window_available')).upper()}**",
        f"- Controlled repair allowed: **{str(report.get('controlled_repair_allowed')).upper()}**",
        f"- Affected pure-hot-dry proxy steps: {report.get('affected_pure_hot_dry_steps', 0)}",
        "",
        "| preset | would_block | first_warning | lead_time | pre_failure | hard_event_only | active | reject_v1 | reject_v2 | warning_v2 | pure_hot_dry_affected |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trace in report.get("traces", []) or []:
        counts = trace.get("counts", {}) if isinstance(trace.get("counts"), Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(trace.get("preset", "")),
                    str(trace.get("would_block_failure_case", False)),
                    str(trace.get("first_boundary_warning_step", "")),
                    str(trace.get("lead_time_steps", "")),
                    str(trace.get("catches_pre_failure_warning", False)),
                    str(trace.get("hard_event_only", False)),
                    str(counts.get("boundary_active", 0)),
                    str(counts.get("would_reject", 0)),
                    str(counts.get("would_reject_v2", 0)),
                    str(counts.get("warning_v2", 0)),
                    str(trace.get("affected_pure_hot_dry_steps", 0)),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", action="append", required=True)
    parser.add_argument("--scenario-id", default="y2020_d120_s44_n240_llm_rspc_v2")
    parser.add_argument("--start-step", type=int, default=220)
    parser.add_argument("--end-step", type=int, default=239)
    parser.add_argument("--failure-step", type=int, default=234)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        trace_dirs=args.trace_dir,
        scenario_id=args.scenario_id,
        start_step=args.start_step,
        end_step=args.end_step,
        failure_step=args.failure_step,
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
    print(f"would_block_any_failure_case={report['would_block_any_failure_case']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
