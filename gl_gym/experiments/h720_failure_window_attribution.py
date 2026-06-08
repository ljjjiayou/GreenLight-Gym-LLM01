"""Build v40 H720 failure-window attribution and stability-guard design.

The audit is read-only. It consumes the v39 H720 traces and compares the fixed
CVODES failure scenarios against the paired PPO baseline window. It does not run
rollouts, call online LLMs, authorize controlled replay, or mutate the default
controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FAILURE_SCENARIOS = (
    "y2010_d180_s43_n720",
    "y2018_d181_s42_n720",
    "y2018_d181_s43_n720",
)

ACTION_FIELDS = ("u_heating", "u_ventilation", "u_co2", "u_lighting", "u_screen", "u_shading")
TARGET_FIELDS = ("target_temp", "target_rh", "target_co2")
BOUNDARY_FIELDS = ("temp_air", "rh_air", "vpd_air", "canopy_dew_margin", "dew_margin_air")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


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
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "applied"}


def _parts_from_scenario_id(value: str) -> dict[str, int]:
    match = re.search(r"y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)(?:_n(?P<steps>\d+))?", str(value))
    if not match:
        return {"year": 0, "day": 0, "seed": 0, "max_steps": 720}
    return {
        "year": int(match.group("year")),
        "day": int(match.group("day")),
        "seed": int(match.group("seed")),
        "max_steps": int(match.group("steps") or 720),
    }


def _trace_path_for(trace_dir: Path, scenario_id: str, controller: str) -> Path:
    expected = trace_dir / f"{scenario_id}_{controller}.csv"
    if expected.exists():
        return expected
    parts = _parts_from_scenario_id(scenario_id)
    prefix = f"y{parts['year']}_d{parts['day']}_s{parts['seed']}_"
    matches = sorted(trace_dir.glob(f"{prefix}*_{controller}.csv"))
    return matches[0] if matches else expected


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _window(rows: Sequence[Mapping[str, Any]], *, start_step: int, end_step: int) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if start_step <= int(_num(row.get("step"), -1)) <= end_step
    ]


def _runtime_failure_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    for row in rows:
        error_text = " ".join(str(row.get(key, "") or "") for key in ("runtime_error", "runtime_error_type", "replan_reason"))
        if "CV_CONV_FAILURE" in error_text or "CVODES" in error_text or "CvodesInterface" in error_text:
            return int(_num(row.get("step"), -1))
    if rows:
        return int(_num(rows[-1].get("step"), len(rows) - 1))
    return None


def _action_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    jumps = 0
    oscillations = 0
    screen_vent_conflicts = 0
    shade_screen_conflicts = 0
    action_reason_counts: Counter[str] = Counter()
    previous: dict[str, float] = {}
    signs: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for row in rows:
        step = int(_num(row.get("step"), 0))
        if _num(row.get("u_ventilation")) >= 0.85 and _num(row.get("u_screen")) >= 0.30:
            screen_vent_conflicts += 1
        if _num(row.get("u_shading")) >= 0.80 and _num(row.get("u_screen")) >= 0.30:
            shade_screen_conflicts += 1
        for field in ACTION_FIELDS:
            value = _num(row.get(field))
            if field in previous:
                delta = value - previous[field]
                if abs(delta) > 0.20:
                    jumps += 1
                    action_reason_counts[f"jump:{field}"] += 1
                    if len(examples) < 15:
                        examples.append({"step": step, "issue": "large_action_delta", "field": field, "delta": delta})
                sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
                if sign and signs.get(field, 0) and sign != signs[field]:
                    oscillations += 1
                    action_reason_counts[f"oscillation:{field}"] += 1
                    if len(examples) < 15:
                        examples.append({"step": step, "issue": "action_oscillation", "field": field, "delta": delta})
                if sign:
                    signs[field] = sign
            previous[field] = value
    return {
        "large_action_delta_count": jumps,
        "action_oscillation_count": oscillations,
        "screen_vent_conflict_count": screen_vent_conflicts,
        "shade_screen_conflict_count": shade_screen_conflicts,
        "reason_counts": dict(sorted(action_reason_counts.items())),
        "examples": examples,
    }


def _guardrail_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewrite_count = 0
    provenance_count = 0
    missing_reason = 0
    unknown_rewrite = 0
    reason_counts: Counter[str] = Counter()
    for row in rows:
        applied = _truthy(row.get("tomato_safety_v2_applied"))
        changed = any(
            abs(_num(row.get(f"tomato_safety_v2_{name}_after")) - _num(row.get(f"tomato_safety_v2_{name}_before"))) > 1e-6
            for name in ("heat", "vent", "screen", "shade")
        )
        if applied or changed:
            rewrite_count += 1
            reasons = str(row.get("tomato_safety_v2_reasons", "") or "")
            for reason in [part.strip() for part in reasons.split(",") if part.strip()]:
                reason_counts[reason] += 1
        provenance_count += int(_num(row.get("post_guardrail_runtime_provenance_count")))
        missing_reason += int(_num(row.get("post_guardrail_runtime_reason_missing_count")))
        unknown_rewrite += int(_num(row.get("unknown_post_guardrail_rewrite_count")))
    return {
        "guardrail_rewrite_count": rewrite_count,
        "runtime_provenance_record_count": provenance_count,
        "runtime_reason_missing_count": missing_reason,
        "unknown_post_guardrail_rewrite_count": unknown_rewrite,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _profile_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reversals = 0
    jumps = 0
    dew_conflicts = 0
    vpd_conflicts = 0
    previous: dict[str, float] = {}
    signs: dict[str, int] = {}
    for row in rows:
        for field in TARGET_FIELDS:
            value = _num(row.get(field))
            if field in previous:
                delta = value - previous[field]
                if abs(delta) > {"target_temp": 2.0, "target_rh": 8.0, "target_co2": 200.0}[field]:
                    jumps += 1
                sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
                if sign and signs.get(field, 0) and sign != signs[field]:
                    reversals += 1
                if sign:
                    signs[field] = sign
            previous[field] = value
        if _truthy(row.get("dew_risk")) and (_num(row.get("target_rh")) >= 80.0 or _num(row.get("profile_selected_delta_target_rh")) > 0.0):
            dew_conflicts += 1
        if _num(row.get("vpd_high_excess")) > 0 and _num(row.get("profile_selected_delta_target_rh")) < 0:
            vpd_conflicts += 1
    return {
        "profile_target_reversal_count": reversals,
        "profile_target_jump_count": jumps,
        "profile_dew_conflict_count": dew_conflicts,
        "profile_vpd_conflict_count": vpd_conflicts,
    }


def _candidate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    missing = 0
    filtered = 0
    anchor_selected = 0
    fallback = 0
    reason_counts: Counter[str] = Counter()
    for row in rows:
        candidate_count = int(_num(row.get("rspc_action_candidate_count"), _num(row.get("rspc_action_actual_candidate_count"))))
        if candidate_count <= 0:
            missing += 1
        best_eligible = str(row.get("rspc_action_post_shape_best_eligible", "") or "").strip().lower()
        raw_reason = str(row.get("rspc_action_post_shape_safety_gate_reason", "") or "")
        reason = "" if raw_reason.strip().lower() in {"", "none", "null"} else raw_reason
        if best_eligible == "false" or reason:
            filtered += 1
            if reason:
                reason_counts[reason] += 1
        if str(row.get("rspc_action_selected_name", "") or "") == "anchor":
            anchor_selected += 1
        if str(row.get("selected_fallback_candidate", "") or ""):
            fallback += 1
    return {
        "candidate_missing_count": missing,
        "candidate_filtered_count": filtered,
        "anchor_selected_count": anchor_selected,
        "fallback_selected_count": fallback,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _boundary_metrics(rows: Sequence[Mapping[str, Any]], ppo_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def stats(window: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | None]]:
        result: dict[str, dict[str, float | None]] = {}
        for field in BOUNDARY_FIELDS:
            values = [_num(row.get(field)) for row in window if row.get(field) not in (None, "")]
            result[field] = {
                "start": values[0] if values else None,
                "end": values[-1] if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "delta": (values[-1] - values[0]) if len(values) >= 2 else 0.0,
            }
        return result
    hard_warning_count = sum(
        1
        for row in rows
        if _truthy(row.get("final_action_would_fail_canopy_boundary_v2")) or _truthy(row.get("final_action_predicted_canopy_lt0_v2"))
    )
    return {
        "llm_stats": stats(rows),
        "ppo_stats": stats(ppo_rows),
        "hard_safety_warning_count": hard_warning_count,
        "dry_risk_count": sum(1 for row in rows if _truthy(row.get("dry_risk"))),
        "dew_risk_count": sum(1 for row in rows if _truthy(row.get("dew_risk"))),
    }


def _dominant_area(report: Mapping[str, Any]) -> tuple[str, bool]:
    scores = {
        "action_instability": int(report["action_instability"]["large_action_delta_count"])
        + int(report["action_instability"]["action_oscillation_count"])
        + int(report["action_instability"]["screen_vent_conflict_count"]),
        "guardrail_rewrite_pressure": int(report["guardrail_rewrite_pressure"]["guardrail_rewrite_count"]),
        "profile_target_pressure": int(report["profile_target_pressure"]["profile_target_reversal_count"])
        + int(report["profile_target_pressure"]["profile_target_jump_count"])
        + int(report["profile_target_pressure"]["profile_dew_conflict_count"])
        + int(report["profile_target_pressure"]["profile_vpd_conflict_count"]),
        "candidate_pool_pressure": int(report["candidate_pool_pressure"]["candidate_missing_count"])
        + int(report["candidate_pool_pressure"]["candidate_filtered_count"])
        + int(report["candidate_pool_pressure"]["anchor_selected_count"]),
        "simulator_boundary_pressure": int(report["simulator_boundary_pressure"]["hard_safety_warning_count"])
        + int(report["simulator_boundary_pressure"]["dry_risk_count"]),
    }
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    dominant = ordered[0][0] if ordered else "unknown"
    clear = len(ordered) >= 2 and ordered[0][1] > 0 and ordered[0][1] >= ordered[1][1] * 1.10
    return dominant, clear


def _scenario_report(
    *,
    scenario_id: str,
    trace_dir: Path,
    window_steps: int,
) -> dict[str, Any]:
    llm_rows = _read_rows(_trace_path_for(trace_dir, scenario_id, "llm_rspc_v2"))
    ppo_rows = _read_rows(_trace_path_for(trace_dir, scenario_id, "ppo"))
    failure_step = _runtime_failure_step(llm_rows)
    if failure_step is None:
        start_step = 0
        end_step = -1
    else:
        end_step = failure_step
        start_step = max(0, failure_step - window_steps + 1)
    llm_window = _window(llm_rows, start_step=start_step, end_step=end_step)
    ppo_window = _window(ppo_rows, start_step=start_step, end_step=end_step)
    scenario = {
        "scenario_id": scenario_id,
        "failure_step": failure_step,
        "window_steps_requested": window_steps,
        "window_start_step": start_step,
        "window_end_step": end_step,
        "llm_window_row_count": len(llm_window),
        "ppo_window_row_count": len(ppo_window),
        "action_instability": _action_metrics(llm_window),
        "guardrail_rewrite_pressure": _guardrail_metrics(llm_window),
        "profile_target_pressure": _profile_metrics(llm_window),
        "candidate_pool_pressure": _candidate_metrics(llm_window),
        "simulator_boundary_pressure": _boundary_metrics(llm_window, ppo_window),
    }
    dominant, clear = _dominant_area(scenario)
    scenario["dominant_attribution"] = dominant
    scenario["dominant_attribution_clear"] = clear
    return scenario


def _aggregate(scenarios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    for scenario in scenarios:
        totals["large_action_delta_count"] += int(scenario["action_instability"]["large_action_delta_count"])
        totals["action_oscillation_count"] += int(scenario["action_instability"]["action_oscillation_count"])
        totals["screen_vent_conflict_count"] += int(scenario["action_instability"]["screen_vent_conflict_count"])
        totals["guardrail_rewrite_count"] += int(scenario["guardrail_rewrite_pressure"]["guardrail_rewrite_count"])
        totals["runtime_provenance_record_count"] += int(scenario["guardrail_rewrite_pressure"]["runtime_provenance_record_count"])
        totals["runtime_reason_missing_count"] += int(scenario["guardrail_rewrite_pressure"]["runtime_reason_missing_count"])
        totals["unknown_post_guardrail_rewrite_count"] += int(scenario["guardrail_rewrite_pressure"]["unknown_post_guardrail_rewrite_count"])
        totals["profile_target_reversal_count"] += int(scenario["profile_target_pressure"]["profile_target_reversal_count"])
        totals["profile_target_jump_count"] += int(scenario["profile_target_pressure"]["profile_target_jump_count"])
        totals["profile_dew_conflict_count"] += int(scenario["profile_target_pressure"]["profile_dew_conflict_count"])
        totals["profile_vpd_conflict_count"] += int(scenario["profile_target_pressure"]["profile_vpd_conflict_count"])
        totals["candidate_missing_count"] += int(scenario["candidate_pool_pressure"]["candidate_missing_count"])
        totals["candidate_filtered_count"] += int(scenario["candidate_pool_pressure"]["candidate_filtered_count"])
        totals["anchor_selected_count"] += int(scenario["candidate_pool_pressure"]["anchor_selected_count"])
        totals["fallback_selected_count"] += int(scenario["candidate_pool_pressure"]["fallback_selected_count"])
        totals["hard_safety_warning_count"] += int(scenario["simulator_boundary_pressure"]["hard_safety_warning_count"])
        totals["dry_risk_count"] += int(scenario["simulator_boundary_pressure"]["dry_risk_count"])
        totals["dew_risk_count"] += int(scenario["simulator_boundary_pressure"]["dew_risk_count"])
    return dict(sorted(totals.items()))


def build_report(
    *,
    trace_dir: str | Path,
    v39_readiness: Mapping[str, Any] | None = None,
    failure_scenarios: Sequence[str] = DEFAULT_FAILURE_SCENARIOS,
    window_steps: int = 60,
) -> dict[str, Any]:
    trace_root = _resolve(trace_dir)
    scenario_reports = [
        _scenario_report(scenario_id=scenario_id, trace_dir=trace_root, window_steps=window_steps)
        for scenario_id in failure_scenarios
    ]
    metrics = _aggregate(scenario_reports)
    dominant_counts = Counter(str(item.get("dominant_attribution", "")) for item in scenario_reports)
    main_attribution = dominant_counts.most_common(1)[0][0] if dominant_counts else "unknown"
    attribution_clear = bool(
        scenario_reports
        and all(bool(item.get("dominant_attribution_clear", False)) for item in scenario_reports)
        and len(dominant_counts) == 1
    )
    if main_attribution in {"action_instability", "guardrail_rewrite_pressure"}:
        recommended_next = "shadow_rate_guard_minimal_patch_plan"
    elif attribution_clear:
        recommended_next = f"{main_attribution}_targeted_design"
    else:
        recommended_next = "additional_runtime_instrumentation_before_fix"
    failures: list[str] = []
    if any(int(item.get("llm_window_row_count", 0) or 0) < min(window_steps, int((item.get("failure_step") or 0)) + 1) for item in scenario_reports):
        failures.append("short_failure_window")
    if metrics.get("runtime_reason_missing_count", 0) or metrics.get("unknown_post_guardrail_rewrite_count", 0):
        failures.append("runtime_provenance_incomplete")
    if metrics.get("hard_safety_warning_count", 0):
        failures.append("hard_safety_warning_requires_review")
    return {
        "schema_version": "h720_failure_window_attribution_20260530_v40",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "h720 failure-window attribution",
        },
        "source_readiness_schema": (v39_readiness or {}).get("schema_version", ""),
        "failure_scenarios": list(failure_scenarios),
        "window_steps": window_steps,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "scenario_reports": scenario_reports,
        "metrics": metrics,
        "dominant_attribution_counts": dict(sorted(dominant_counts.items())),
        "main_attribution": main_attribution,
        "attribution_clear": attribution_clear,
        "failure_taxonomy": sorted(set(failures)),
        "next_action": recommended_next,
    }


def build_guard_design(report: Mapping[str, Any]) -> dict[str, Any]:
    guard_recommended = str(report.get("next_action", "")) == "shadow_rate_guard_minimal_patch_plan"
    return {
        "schema_version": "rate_guard_action_smoothing_design_20260530_v40",
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "default_llm_rspc_v2_changed": False,
        "shadow_only_design": True,
        "rate_guard_recommended": guard_recommended,
        "design_principles": [
            "limit short-cycle action reversals before changing candidate generation",
            "cap single-step action deltas for vent/screen/shade/heat",
            "do not weaken dew/canopy hard-safety boundaries",
            "run first as shadow audit before any default controller integration",
        ],
        "initial_thresholds": {
            "max_single_step_delta": 0.20,
            "max_direction_reversal_window_steps": 6,
            "target_actuators": ["u_heating", "u_ventilation", "u_screen", "u_shading"],
        },
        "evidence": {
            "main_attribution": report.get("main_attribution"),
            "attribution_clear": report.get("attribution_clear"),
            "metrics": report.get("metrics", {}),
        },
        "metrics": report.get("metrics", {}),
        "next_action": "shadow_rate_guard_minimal_patch_plan" if guard_recommended else report.get("next_action"),
    }


def build_readiness(report: Mapping[str, Any], guard_design: Mapping[str, Any]) -> dict[str, Any]:
    next_action = str(guard_design.get("next_action") or report.get("next_action"))
    if not report.get("attribution_clear", False) and next_action != "shadow_rate_guard_minimal_patch_plan":
        next_action = "additional_runtime_instrumentation_before_fix"
    return {
        "schema_version": "metadata_replay_readiness_checklist_20260530_v40",
        "mainline_alignment": report.get("mainline_alignment", {}),
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "default_llm_rspc_v2_changed": False,
        "failure_window_attribution_complete": True,
        "rate_guard_design_ready": bool(guard_design.get("rate_guard_recommended", False)),
        "metrics": report.get("metrics", {}),
        "main_attribution": report.get("main_attribution"),
        "attribution_clear": report.get("attribution_clear"),
        "failure_taxonomy": report.get("failure_taxonomy", []),
        "next_action": next_action,
    }


def _markdown_table(metrics: Mapping[str, Any]) -> list[str]:
    lines = ["| metric | value |", "| --- | ---: |"]
    for key, value in metrics.items():
        lines.append(f"| {key} | `{value}` |")
    return lines


def build_markdown_report(report: Mapping[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Metrics",
        "",
        *_markdown_table(report.get("metrics", {}) or {}),
    ]
    if report.get("main_attribution"):
        lines.extend(["", "## Attribution", ""])
        lines.append(f"- Main attribution: `{report.get('main_attribution')}`")
        lines.append(f"- Attribution clear: `{report.get('attribution_clear')}`")
    if report.get("scenario_reports"):
        lines.extend(["", "## Scenario Windows", "", "| scenario | failure step | dominant attribution | clear |", "| --- | ---: | --- | --- |"])
        for item in report.get("scenario_reports", []) or []:
            lines.append(
                f"| `{item.get('scenario_id')}` | `{item.get('failure_step')}` | "
                f"`{item.get('dominant_attribution')}` | `{item.get('dominant_attribution_clear')}` |"
            )
    failures = report.get("failure_taxonomy", []) or []
    if failures:
        lines.extend(["", "## Failure Taxonomy", ""])
        lines.extend(f"- `{item}`" for item in failures)
    return "\n".join(lines) + "\n"


def _write_pair(path_json: str | Path, path_md: str | Path, report: Mapping[str, Any], *, title: str) -> None:
    json_path = _resolve(path_json)
    md_path = _resolve(path_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(build_markdown_report(report, title=title), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--v39-readiness-json", required=True)
    parser.add_argument("--window-steps", type=int, default=60)
    parser.add_argument("--failure-scenario", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--guard-design-json", required=True)
    parser.add_argument("--guard-design-md", required=True)
    parser.add_argument("--readiness-json", required=True)
    parser.add_argument("--readiness-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    failure_scenarios = args.failure_scenario or list(DEFAULT_FAILURE_SCENARIOS)
    report = build_report(
        trace_dir=args.trace_dir,
        v39_readiness=_load(args.v39_readiness_json),
        failure_scenarios=failure_scenarios,
        window_steps=args.window_steps,
    )
    guard_design = build_guard_design(report)
    readiness = build_readiness(report, guard_design)
    _write_pair(args.output_json, args.output_md, report, title="H720 Failure Window Attribution v40")
    _write_pair(args.guard_design_json, args.guard_design_md, guard_design, title="Rate Guard Action Smoothing Design v40")
    _write_pair(args.readiness_json, args.readiness_md, readiness, title="Metadata Replay Readiness Checklist v40")
    print(f"main_attribution={report['main_attribution']}")
    print(f"attribution_clear={report['attribution_clear']}")
    print(f"next_action={readiness['next_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
