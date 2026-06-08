"""Suite-level shadow audit for hypothetical canopy-safe dry-relief candidates.

This audit does not add candidates to the controller and does not change action
selection.  It extends the single canonical repair audit to selected
pure-hot-dry false-positive scenarios so candidate gaps and possible blocker
warnings can be interpreted together.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.candidate_pool_audit import classify_step
from gl_gym.experiments.candidate_pool_repair_shadow import (
    _action,
    _num,
    _proxy_response,
    build_hypothetical_candidates,
)
from gl_gym.experiments.proxy_v2_shadow_validation_audit import _is_pure_hot_dry

PROXY_V2_1_BUFFER_SCALE = 0.75

DEFAULT_WINDOWS = (
    ("canonical_failure", "y2020_d120_s44_n240", 220, 239),
    ("pure_hot_dry_false_positive", "y2015_d240_s42_n240", 0, 239),
    ("pure_hot_dry_false_positive", "y2015_d240_s43_n240", 0, 239),
    ("pure_hot_dry_false_positive", "y2015_d240_s44_n240", 0, 239),
    ("pure_hot_dry_false_positive", "y2020_d240_s44_n240", 0, 239),
)


def _read_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rows_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(round(_num(row.get("step", row.get("timestep", 0))))): row for row in rows}


def _resolve_root(path: str | Path) -> Path:
    root = Path(path)
    return root if root.is_absolute() else PROJECT_ROOT / root


def _iter_trace_paths(trace_dirs: Sequence[str | Path], scenario_id: str) -> list[Path]:
    paths: list[Path] = []
    for raw in trace_dirs:
        root = _resolve_root(raw)
        matches = [root] if root.is_file() else sorted(root.rglob(f"{scenario_id}*.csv"))
        paths.extend(matches)
    if not paths:
        raise FileNotFoundError(f"trace not found for {scenario_id} under {trace_dirs}")
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out


def _preset_from_path(path: Path) -> str:
    preset = path.parent.name or "trace"
    return path.stem if preset.lower() == "traces" else preset


def _v21_prediction(row: Mapping[str, Any]) -> tuple[float, bool]:
    v1_pred = _num(row.get("canopy_boundary_shadow_predicted_canopy_dew_margin_next"), 99.0)
    v2_pred = _num(
        row.get("final_action_predicted_canopy_dew_margin_next_v2"),
        _num(row.get("canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2"), v1_pred),
    )
    buffer = max(0.0, v1_pred - v2_pred)
    pred = v1_pred - PROXY_V2_1_BUFFER_SCALE * buffer
    return pred, bool(pred < 0.25)


def _current_action_useful(row: Mapping[str, Any], prev_row: Mapping[str, Any] | None) -> bool:
    prediction = _proxy_response(row, _action(row), prev_row)
    return bool(
        _num(prediction.get("predicted_RHlow_delta")) <= 0.0
        and _num(prediction.get("predicted_VPDhi_delta")) <= 0.0
    )


def _classify_repair_record(row: Mapping[str, Any], prev_row: Mapping[str, Any] | None, classification: Mapping[str, Any]) -> dict[str, Any]:
    candidates = build_hypothetical_candidates(row, prev_row, classification.get("post_guardrail_delta", {}))
    feasible = [item for item in candidates if bool(item.get("feasible"))]
    useful = [item for item in feasible if bool(item.get("dry_relief_useful"))]
    useful_post_safe = [item for item in useful if bool(item.get("post_guardrail_remains_safe"))]
    dry_tradeoff = [item for item in feasible if bool(item.get("dry_tradeoff"))]
    conservative = [item for item in feasible if bool(item.get("conservative_ineffective_action"))]
    if useful_post_safe:
        label = "repair_post_guardrail_safe"
    elif useful:
        label = "repair_becomes_unsafe_after_post_guardrail"
    elif dry_tradeoff or conservative:
        label = "repair_dry_relief_lost"
    else:
        label = "no_safe_repair_found"
    best = useful_post_safe[0] if useful_post_safe else (useful[0] if useful else (feasible[0] if feasible else {}))
    return {
        "repair_class": label,
        "best_candidate": best.get("name", "") if isinstance(best, Mapping) else "",
        "best_candidate_family": best.get("candidate_family", "") if isinstance(best, Mapping) else "",
        "feasible_count": len(feasible),
        "useful_count": len(useful),
        "useful_post_guardrail_safe_count": len(useful_post_safe),
        "dry_tradeoff_count": len(dry_tradeoff),
        "conservative_ineffective_count": len(conservative),
        "hypothetical_candidates": candidates,
    }


def _audit_trace_window(
    *,
    trace_path: Path,
    scenario_id: str,
    category: str,
    start_step: int,
    end_step: int,
) -> dict[str, Any]:
    rows = _rows_by_step(_read_trace(trace_path))
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for step in range(int(start_step), int(end_step) + 1):
        row = rows.get(step)
        if row is None:
            continue
        next_row = rows.get(step + 1)
        prev_row = rows.get(step - 1)
        classification = classify_step(row, next_row)
        source_label = str(classification.get("label", "") or "")
        v21_pred, warning_v21 = _v21_prediction(row)
        actual_next_canopy = _num(next_row.get("canopy_dew_margin"), 99.0) if next_row else 99.0
        pure_hot_dry = _is_pure_hot_dry(row)
        current_useful = _current_action_useful(row, prev_row)
        record: dict[str, Any] | None = None
        if source_label == "no_safe_candidate_available":
            repair = _classify_repair_record(row, prev_row, classification)
            counts["candidate_pool_gap"] += 1
            counts[repair["repair_class"]] += 1
            record = {
                "record_type": "candidate_pool_gap",
                **repair,
            }
        elif warning_v21 and pure_hot_dry and actual_next_canopy >= 0.0:
            label = "repair_false_positive_blocker" if current_useful else "acceptable_shadow_warning"
            counts[label] += 1
            record = {
                "record_type": "pure_hot_dry_warning",
                "repair_class": label,
                "current_action_useful_dry_relief": current_useful,
            }
        if record is not None:
            record.update(
                {
                    "scenario_id": scenario_id,
                    "preset": _preset_from_path(trace_path),
                    "category": category,
                    "step": step,
                    "source_label": source_label,
                    "proxy_v2_1_predicted_canopy_dew_margin_next": v21_pred,
                    "proxy_v2_1_warning": warning_v21,
                    "actual_next_canopy_dew_margin": actual_next_canopy,
                    "pure_hot_dry": pure_hot_dry,
                    "state": {
                        "temp_air": _num(row.get("temp_air")),
                        "rh_air": _num(row.get("rh_air")),
                        "vpd_air": _num(row.get("vpd_air")),
                        "dew_margin_air": _num(row.get("dew_margin_air")),
                        "canopy_dew_margin": _num(row.get("canopy_dew_margin")),
                        "u_screen": _num(row.get("u_screen")),
                        "u_ventilation": _num(row.get("u_ventilation")),
                    },
                }
            )
            records.append(record)
    return {
        "trace_path": str(trace_path),
        "scenario_id": scenario_id,
        "preset": _preset_from_path(trace_path),
        "category": category,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "counts": dict(sorted(counts.items())),
        "records": records,
    }


def _parse_window(text: str) -> tuple[str, str, int, int]:
    parts = text.split(":")
    if len(parts) != 4:
        raise ValueError("--window must be category:scenario:start:end")
    category, scenario, start, end = parts
    return category, scenario, int(start), int(end)


def build_report(
    *,
    trace_dirs: Sequence[str | Path],
    windows: Sequence[tuple[str, str, int, int]] = DEFAULT_WINDOWS,
) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    aggregate: Counter[str] = Counter()
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    by_scenario: dict[str, Counter[str]] = defaultdict(Counter)
    for category, scenario_id, start_step, end_step in windows:
        for trace_path in _iter_trace_paths(trace_dirs, scenario_id):
            trace = _audit_trace_window(
                trace_path=trace_path,
                scenario_id=scenario_id,
                category=category,
                start_step=start_step,
                end_step=end_step,
            )
            traces.append(trace)
            counts = Counter({str(k): int(v) for k, v in dict(trace.get("counts", {})).items()})
            aggregate.update(counts)
            by_category[category].update(counts)
            by_scenario[scenario_id].update(counts)
    candidate_gap = int(aggregate.get("candidate_pool_gap", 0))
    safe_repair = int(aggregate.get("repair_post_guardrail_safe", 0))
    blocker = int(aggregate.get("repair_false_positive_blocker", 0))
    no_safe = int(aggregate.get("no_safe_repair_found", 0))
    return {
        "schema_version": "candidate_generation_shadow_suite_v1",
        "mainline_alignment": {
            "affected_layers": ["Candidate Pool", "Safety Boundary", "Evaluation"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only / metadata-only",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "trace_count": len(traces),
        "window_count": len(windows),
        "aggregate_counts": dict(sorted(aggregate.items())),
        "category_counts": {key: dict(sorted(value.items())) for key, value in sorted(by_category.items())},
        "scenario_counts": {key: dict(sorted(value.items())) for key, value in sorted(by_scenario.items())},
        "candidate_pool_gap_steps": candidate_gap,
        "post_guardrail_safe_useful_repair_count": safe_repair,
        "pure_hot_dry_false_positive_blocker_count": blocker,
        "pure_hot_dry_false_positive_blocker_rate": (
            float(blocker / (blocker + int(aggregate.get("acceptable_shadow_warning", 0))))
            if blocker or int(aggregate.get("acceptable_shadow_warning", 0))
            else 0.0
        ),
        "no_safe_repair_found_count": no_safe,
        "candidate_generation_shadow_conclusion": (
            "candidate_generation_shadow_expand"
            if candidate_gap > 0 and safe_repair >= candidate_gap and blocker == 0
            else "candidate_repair_has_pure_hot_dry_blocker_risk"
            if blocker > 0
            else "no_safe_repair_found"
            if no_safe > 0
            else "insufficient_candidate_gap_evidence"
        ),
        "traces": traces,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate Generation Shadow Suite",
        "",
        "## Mainline Alignment",
        "",
        "- Mode: shadow-only / metadata-only",
        "- Default `llm_rspc_v2` changed: false",
        "- Reopens rejected preset: false",
        "- Controlled replay allowed: false",
        "",
        "## Summary",
        "",
        f"- Trace count: {report.get('trace_count', 0)}",
        f"- Candidate-pool gap steps: {report.get('candidate_pool_gap_steps', 0)}",
        f"- Post-guardrail safe useful repair count: {report.get('post_guardrail_safe_useful_repair_count', 0)}",
        f"- Pure-hot-dry false-positive blocker count: {report.get('pure_hot_dry_false_positive_blocker_count', 0)}",
        f"- Pure-hot-dry false-positive blocker rate: {_num(report.get('pure_hot_dry_false_positive_blocker_rate')):.3f}",
        f"- Conclusion: `{report.get('candidate_generation_shadow_conclusion', '')}`",
        "",
        "## Aggregate Counts",
        "",
        "| label | count |",
        "| --- | ---: |",
    ]
    for label, count in dict(report.get("aggregate_counts", {})).items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Scenario Counts", "", "| scenario | counts |", "| --- | --- |"])
    for scenario, counts in dict(report.get("scenario_counts", {})).items():
        lines.append(f"| {scenario} | {json.dumps(counts, sort_keys=True)} |")
    lines.extend(
        [
            "",
            "## Records",
            "",
            "| category | scenario | preset | step | type | class | best_candidate | v2_1_warning | actual_next_canopy |",
            "| --- | --- | --- | ---: | --- | --- | --- | --- | ---: |",
        ]
    )
    for trace in report.get("traces", []) or []:
        if not isinstance(trace, Mapping):
            continue
        for row in trace.get("records", []) or []:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("category", "")),
                        str(row.get("scenario_id", "")),
                        str(row.get("preset", "")),
                        str(row.get("step", "")),
                        str(row.get("record_type", "")),
                        str(row.get("repair_class", "")),
                        str(row.get("best_candidate", "")),
                        str(row.get("proxy_v2_1_warning", False)),
                        f"{_num(row.get('actual_next_canopy_dew_margin')):.3f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", action="append", required=True)
    parser.add_argument("--window", action="append", default=[], help="category:scenario:start:end")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    windows = [_parse_window(item) for item in args.window] if args.window else list(DEFAULT_WINDOWS)
    report = build_report(trace_dirs=args.trace_dir, windows=windows)
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
    print(f"candidate_pool_gap_steps={report['candidate_pool_gap_steps']}")
    print(f"conclusion={report['candidate_generation_shadow_conclusion']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
