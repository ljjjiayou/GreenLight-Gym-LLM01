"""Compare 6D greenhouse action sequences between two trace roots."""

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

ACTION_FIELDS = (
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _read_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _trace_paths(root: str | Path, scenario_id: str) -> dict[str, Path]:
    path = Path(root)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file():
        return {path.parent.name or path.stem: path}
    matches = sorted(path.rglob(f"{scenario_id}*.csv"))
    out: dict[str, Path] = {}
    for match in matches:
        key = match.parent.name or match.stem
        if key.lower() == "traces":
            key = match.stem
        out[key] = match
    if not out:
        raise FileNotFoundError(f"trace not found for {scenario_id} under {path}")
    return out


def _rows_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(round(_num(row.get("step", row.get("timestep", 0))))): row for row in rows}


def _action(row: Mapping[str, Any]) -> dict[str, float]:
    return {field: _num(row.get(field)) for field in ACTION_FIELDS}


def compare_trace_pair(
    *,
    baseline_path: str | Path,
    compare_path: str | Path,
    label: str,
    tolerance: float,
) -> dict[str, Any]:
    base_rows = _rows_by_step(_read_trace(baseline_path))
    cmp_rows = _rows_by_step(_read_trace(compare_path))
    steps = sorted(set(base_rows) & set(cmp_rows))
    diff_rows: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    for step in steps:
        base_action = _action(base_rows[step])
        cmp_action = _action(cmp_rows[step])
        deltas = {field: cmp_action[field] - base_action[field] for field in ACTION_FIELDS}
        step_max = max(abs(value) for value in deltas.values()) if deltas else 0.0
        max_abs_delta = max(max_abs_delta, step_max)
        if step_max > tolerance:
            diff_rows.append(
                {
                    "step": step,
                    "max_abs_delta": step_max,
                    "deltas": deltas,
                    "baseline_action": base_action,
                    "compare_action": cmp_action,
                }
            )
    missing_steps = sorted(set(base_rows) ^ set(cmp_rows))
    return {
        "label": label,
        "baseline_path": str(baseline_path),
        "compare_path": str(compare_path),
        "compared_steps": len(steps),
        "missing_step_count": len(missing_steps),
        "missing_steps": missing_steps[:25],
        "action_diff_steps": len(diff_rows),
        "max_abs_delta": max_abs_delta,
        "diff_rows": diff_rows[:50],
    }


def build_report(
    *,
    baseline_trace_dir: str | Path,
    compare_trace_dir: str | Path,
    scenario_id: str,
    tolerance: float,
) -> dict[str, Any]:
    baseline_paths = _trace_paths(baseline_trace_dir, scenario_id)
    compare_paths = _trace_paths(compare_trace_dir, scenario_id)
    labels = sorted(set(baseline_paths) & set(compare_paths))
    if not labels and len(baseline_paths) == 1 and len(compare_paths) == 1:
        # Canonical Stage-B compares one qualified historical trace with one
        # replay trace; their parent directory labels often differ.
        labels = ["single_trace_pair"]
        baseline_paths = {"single_trace_pair": next(iter(baseline_paths.values()))}
        compare_paths = {"single_trace_pair": next(iter(compare_paths.values()))}
    traces = [
        compare_trace_pair(
            baseline_path=baseline_paths[label],
            compare_path=compare_paths[label],
            label=label,
            tolerance=tolerance,
        )
        for label in labels
    ]
    missing_labels = sorted(set(baseline_paths) ^ set(compare_paths))
    return {
        "schema_version": "trace_action_diff_audit_v1",
        "scenario_id": scenario_id,
        "tolerance": float(tolerance),
        "trace_count": len(traces),
        "missing_trace_labels": missing_labels,
        "action_diff_steps": int(sum(int(trace.get("action_diff_steps", 0)) for trace in traces)),
        "missing_step_count": int(sum(int(trace.get("missing_step_count", 0)) for trace in traces)),
        "max_abs_delta": max((_num(trace.get("max_abs_delta")) for trace in traces), default=0.0),
        "traces": traces,
    }


def _load_scenario_ids(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    payload = json.loads(p.read_text(encoding="utf-8"))
    items = payload.get("executable_scenarios") or payload.get("selected_scenarios") or payload.get("scenarios") or []
    out: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            scenario = str(item.get("scenario_id", "") or "")
            if scenario:
                out.append(scenario)
    return out


def build_suite_report(
    *,
    baseline_trace_dir: str | Path,
    compare_trace_dir: str | Path,
    scenario_ids: Sequence[str],
    tolerance: float,
) -> dict[str, Any]:
    reports = [
        build_report(
            baseline_trace_dir=baseline_trace_dir,
            compare_trace_dir=compare_trace_dir,
            scenario_id=scenario,
            tolerance=tolerance,
        )
        for scenario in scenario_ids
    ]
    return {
        "schema_version": "trace_action_diff_audit_v1",
        "scenario_count": len(reports),
        "trace_count": int(sum(int(report.get("trace_count", 0)) for report in reports)),
        "action_diff_steps": int(sum(int(report.get("action_diff_steps", 0)) for report in reports)),
        "missing_step_count": int(sum(int(report.get("missing_step_count", 0)) for report in reports)),
        "max_abs_delta": max((_num(report.get("max_abs_delta")) for report in reports), default=0.0),
        "scenario_reports": reports,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    if "scenario_reports" in report:
        lines = [
            "# Trace Action Diff Audit",
            "",
            f"- Scenario count: {report.get('scenario_count', 0)}",
            f"- Trace count: {report.get('trace_count', 0)}",
            f"- Action diff steps: {report.get('action_diff_steps', 0)}",
            f"- Missing step count: {report.get('missing_step_count', 0)}",
            f"- Max abs delta: {_num(report.get('max_abs_delta')):.8f}",
            "",
            "| scenario | traces | diff_steps | missing_steps | max_abs_delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for item in report.get("scenario_reports", []) or []:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item.get("scenario_id", "")),
                        str(item.get("trace_count", 0)),
                        str(item.get("action_diff_steps", 0)),
                        str(item.get("missing_step_count", 0)),
                        f"{_num(item.get('max_abs_delta')):.8f}",
                    ]
                )
                + " |"
            )
        return "\n".join(lines) + "\n"
    lines = [
        "# Trace Action Diff Audit",
        "",
        f"- Scenario: `{report.get('scenario_id', '')}`",
        f"- Trace count: {report.get('trace_count', 0)}",
        f"- Action diff steps: {report.get('action_diff_steps', 0)}",
        f"- Missing step count: {report.get('missing_step_count', 0)}",
        f"- Max abs delta: {_num(report.get('max_abs_delta')):.8f}",
        "",
        "| trace | compared_steps | diff_steps | missing_steps | max_abs_delta |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for trace in report.get("traces", []) or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(trace.get("label", "")),
                    str(trace.get("compared_steps", 0)),
                    str(trace.get("action_diff_steps", 0)),
                    str(trace.get("missing_step_count", 0)),
                    f"{_num(trace.get('max_abs_delta')):.8f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trace-dir", required=True)
    parser.add_argument("--compare-trace-dir", required=True)
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--scenario-list-json", default="")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.scenario_list_json:
        report = build_suite_report(
            baseline_trace_dir=args.baseline_trace_dir,
            compare_trace_dir=args.compare_trace_dir,
            scenario_ids=_load_scenario_ids(args.scenario_list_json),
            tolerance=args.tolerance,
        )
    else:
        if not args.scenario_id:
            raise SystemExit("--scenario-id or --scenario-list-json is required")
        report = build_report(
            baseline_trace_dir=args.baseline_trace_dir,
            compare_trace_dir=args.compare_trace_dir,
            scenario_id=args.scenario_id,
            tolerance=args.tolerance,
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
    print(f"action_diff_steps={report['action_diff_steps']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
