"""Build v42 shadow transition-gate audit for H720 failure windows.

The audit is read-only. It computes a shadow action sequence for the fixed
CVODES failure windows without touching llm_agent.py, candidate generation,
safety hard boundaries, or controlled replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ACTION_FIELDS = ("u_heating", "u_ventilation", "u_co2", "u_lighting", "u_screen", "u_shading")
TARGET_FIELDS = ("u_heating", "u_ventilation", "u_screen", "u_shading")
PASSIVE_FIELDS = ("u_co2", "u_lighting")
DEFAULT_FAILURE_SCENARIOS = (
    "y2010_d180_s43_n720",
    "y2018_d181_s42_n720",
    "y2018_d181_s43_n720",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load_json(path: str | Path) -> dict[str, Any]:
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _trace_path(trace_dir: str | Path, scenario_id: str, controller: str) -> Path:
    root = _resolve(trace_dir)
    expected = root / f"{scenario_id}_{controller}.csv"
    if expected.exists():
        return expected
    match = re.match(r"y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)_", scenario_id)
    if not match:
        return expected
    prefix = f"y{match.group('year')}_d{match.group('day')}_s{match.group('seed')}_"
    matches = sorted(root.glob(f"{prefix}*_{controller}.csv"))
    return matches[0] if matches else expected


def _runtime_failure_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    for row in rows:
        error_text = " ".join(
            str(row.get(key, "") or "") for key in ("runtime_error", "runtime_error_type", "replan_reason")
        )
        if "CV_CONV_FAILURE" in error_text or "CVODES" in error_text or "CvodesInterface" in error_text:
            return int(_num(row.get("step"), -1))
    if rows:
        return int(_num(rows[-1].get("step"), len(rows) - 1))
    return None


def _window_bounds(rows: Sequence[Mapping[str, Any]], window_steps: int) -> tuple[int, int]:
    failure_step = _runtime_failure_step(rows)
    if failure_step is None:
        return 0, -1
    return max(0, failure_step - window_steps + 1), failure_step


def _window(rows: Sequence[Mapping[str, Any]], start_step: int, end_step: int) -> list[Mapping[str, Any]]:
    return [row for row in rows if start_step <= int(_num(row.get("step"), -1)) <= end_step]


def _action(row: Mapping[str, Any]) -> dict[str, float]:
    return {field: _num(row.get(field)) for field in ACTION_FIELDS}


def _regime(row: Mapping[str, Any]) -> str:
    if _truthy(row.get("dew_risk")) or _num(row.get("canopy_dew_margin"), 99.0) < 3.0:
        return "dew_or_canopy_risk"
    if _truthy(row.get("dry_risk")) or (_num(row.get("vpd_air")) >= 2.25 and _num(row.get("rh_air"), 100.0) <= 45.0):
        return "hot_dry_pressure"
    if _num(row.get("rh_air")) >= 85.0:
        return "high_humidity"
    if _num(row.get("temp_air")) >= 28.0:
        return "warm_or_hot"
    return "neutral_or_mild"


def _hard_safety_bypass(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if _truthy(row.get("tomato_safety_v2_applied")):
        reasons.append("tomato_safety_v2_applied")
    if _truthy(row.get("dew_risk")):
        reasons.append("dew_risk")
    if _num(row.get("canopy_dew_margin"), 99.0) < 3.0:
        reasons.append("canopy_dew_margin_lt3")
    if _truthy(row.get("final_action_would_fail_canopy_boundary_v2")):
        reasons.append("final_action_would_fail_canopy_boundary_v2")
    if _truthy(row.get("final_action_predicted_canopy_lt0_v2")):
        reasons.append("final_action_predicted_canopy_lt0_v2")
    return bool(reasons), reasons


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_soft_limits(envelope: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    limits: dict[str, dict[str, float]] = {}
    for item in envelope.get("transition_stats_by_regime", []) or []:
        if str(item.get("label")) != "stable_positive":
            continue
        regime = str(item.get("regime", "neutral_or_mild"))
        limits[regime] = {}
        stats = item.get("action_delta_stats", {}) or {}
        for field in TARGET_FIELDS:
            p95 = _num((stats.get(field, {}) or {}).get("p95"), 0.20)
            limits[regime][field] = max(0.02, min(p95 if p95 > 0 else 0.20, 0.20))
    return limits


def _limit_for(limits: Mapping[str, Mapping[str, float]], regime: str, field: str) -> float:
    if regime in limits and field in limits[regime]:
        return float(limits[regime][field])
    if "neutral_or_mild" in limits and field in limits["neutral_or_mild"]:
        return float(limits["neutral_or_mild"][field])
    return 0.20


def _metrics(rows: Sequence[Mapping[str, Any]], *, action_key: str = "action") -> dict[str, Any]:
    previous: dict[str, float] = {}
    signs: dict[str, int] = {}
    large_delta = 0
    oscillations = 0
    screen_vent = 0
    max_abs_delta = 0.0
    for row in rows:
        action = row[action_key] if action_key in row else _action(row)
        if _num(action.get("u_ventilation")) >= 0.85 and _num(action.get("u_screen")) >= 0.30:
            screen_vent += 1
        for field in ACTION_FIELDS:
            value = _num(action.get(field))
            if field in previous:
                delta = value - previous[field]
                max_abs_delta = max(max_abs_delta, abs(delta))
                if abs(delta) > 0.20:
                    large_delta += 1
                sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
                if sign and signs.get(field, 0) and sign != signs[field]:
                    oscillations += 1
                if sign:
                    signs[field] = sign
            previous[field] = value
    return {
        "transition_count": max(0, len(rows) - 1),
        "large_action_delta_count": large_delta,
        "action_oscillation_count": oscillations,
        "screen_vent_conflict_count": screen_vent,
        "max_abs_delta": max_abs_delta,
    }


def _seed_previous_action(rows: Sequence[Mapping[str, Any]], start_step: int) -> dict[str, float] | None:
    previous_candidates = [row for row in rows if int(_num(row.get("step"), -1)) < start_step]
    if previous_candidates:
        return _action(previous_candidates[-1])
    return None


def _apply_shadow_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    initial_previous_action: Mapping[str, float] | None,
    soft_limits: Mapping[str, Mapping[str, float]],
    reversal_window_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    previous_shadow = dict(initial_previous_action or _action(rows[0])) if rows else {}
    previous_signs: dict[str, int] = {}
    shadow_records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for row in rows:
        step = int(_num(row.get("step"), 0))
        original = _action(row)
        bypass, bypass_reasons = _hard_safety_bypass(row)
        regime = _regime(row)
        shadow = dict(original)
        adjusted_fields: list[str] = []
        reversal_fields: list[str] = []
        if bypass:
            counters["hard_safety_bypass_count"] += 1
            shadow = dict(original)
        else:
            for field in TARGET_FIELDS:
                desired_delta = original[field] - previous_shadow[field]
                sign = 1 if desired_delta > 1e-6 else -1 if desired_delta < -1e-6 else 0
                limit = _limit_for(soft_limits, regime, field)
                reversal = bool(sign and previous_signs.get(field, 0) and sign != previous_signs[field])
                if reversal:
                    clipped_delta = 0.0
                else:
                    clipped_delta = max(-limit, min(limit, desired_delta))
                if abs(clipped_delta - desired_delta) > 1e-9:
                    adjusted_fields.append(field)
                    counters["delta_limited_count"] += 1
                if reversal:
                    reversal_fields.append(field)
                    counters["reversal_projected_count"] += 1
                shadow[field] = _clamp01(previous_shadow[field] + clipped_delta)
        for field in PASSIVE_FIELDS:
            shadow[field] = original[field]
        for field in TARGET_FIELDS:
            delta = shadow[field] - previous_shadow[field]
            sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
            if sign:
                previous_signs[field] = sign
        shadow_records.append(
            {
                "step": step,
                "regime": regime,
                "original_action": original,
                "shadow_action": shadow,
                "hard_safety_bypass": bypass,
                "bypass_reasons": bypass_reasons,
                "adjusted_fields": adjusted_fields,
                "reversal_fields": reversal_fields,
                "reversal_window_steps": reversal_window_steps,
            }
        )
        previous_shadow = dict(shadow)
    return shadow_records, dict(sorted(counters.items()))


def _scenario_report(
    *,
    scenario_id: str,
    trace_dir: str | Path,
    soft_limits: Mapping[str, Mapping[str, float]],
    window_steps: int,
    reversal_window_steps: int,
) -> dict[str, Any]:
    llm_rows = _read_rows(_trace_path(trace_dir, scenario_id, "llm_rspc_v2"))
    ppo_rows = _read_rows(_trace_path(trace_dir, scenario_id, "ppo"))
    start_step, end_step = _window_bounds(llm_rows, window_steps)
    llm_window = _window(llm_rows, start_step, end_step)
    ppo_window = _window(ppo_rows, start_step, end_step)
    initial_previous = _seed_previous_action(llm_rows, start_step)
    shadow_records, counters = _apply_shadow_gate(
        llm_window,
        initial_previous_action=initial_previous,
        soft_limits=soft_limits,
        reversal_window_steps=reversal_window_steps,
    )
    original_metric_rows = [{"action": _action(row)} for row in llm_window]
    ppo_metric_rows = [{"action": _action(row)} for row in ppo_window]
    original_metrics = _metrics(original_metric_rows)
    shadow_metrics = _metrics(shadow_records, action_key="shadow_action")
    ppo_metrics = _metrics(ppo_metric_rows)
    return {
        "scenario_id": scenario_id,
        "window_start_step": start_step,
        "window_end_step": end_step,
        "llm_window_row_count": len(llm_window),
        "ppo_window_row_count": len(ppo_window),
        "original_metrics": original_metrics,
        "shadow_metrics": shadow_metrics,
        "ppo_metrics": ppo_metrics,
        "shadow_gate_counters": counters,
        "delta_reduction": {
            "large_action_delta_count": original_metrics["large_action_delta_count"] - shadow_metrics["large_action_delta_count"],
            "action_oscillation_count": original_metrics["action_oscillation_count"] - shadow_metrics["action_oscillation_count"],
            "screen_vent_conflict_count": original_metrics["screen_vent_conflict_count"] - shadow_metrics["screen_vent_conflict_count"],
        },
        "shadow_records": shadow_records,
    }


def _aggregate(scenarios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    for scenario in scenarios:
        for prefix, key in (("original", "original_metrics"), ("shadow", "shadow_metrics"), ("ppo", "ppo_metrics")):
            metrics = scenario.get(key, {}) or {}
            for metric_name in ("large_action_delta_count", "action_oscillation_count", "screen_vent_conflict_count"):
                totals[f"{metric_name}_{prefix}"] += int(metrics.get(metric_name, 0))
        counters = scenario.get("shadow_gate_counters", {}) or {}
        for key, value in counters.items():
            totals[key] += int(value)
    totals["large_action_delta_reduction"] = totals["large_action_delta_count_original"] - totals["large_action_delta_count_shadow"]
    totals["action_oscillation_reduction"] = totals["action_oscillation_count_original"] - totals["action_oscillation_count_shadow"]
    totals["screen_vent_conflict_reduction"] = totals["screen_vent_conflict_count_original"] - totals["screen_vent_conflict_count_shadow"]
    return dict(sorted(totals.items()))


def build_audit(
    *,
    trace_dir: str | Path,
    envelope: Mapping[str, Any],
    failure_scenarios: Sequence[str] = DEFAULT_FAILURE_SCENARIOS,
    window_steps: int = 60,
    reversal_window_steps: int = 6,
) -> dict[str, Any]:
    soft_limits = _extract_soft_limits(envelope)
    scenario_reports = [
        _scenario_report(
            scenario_id=scenario_id,
            trace_dir=trace_dir,
            soft_limits=soft_limits,
            window_steps=window_steps,
            reversal_window_steps=reversal_window_steps,
        )
        for scenario_id in failure_scenarios
    ]
    metrics = _aggregate(scenario_reports)
    effective = (
        metrics.get("large_action_delta_reduction", 0) > 0
        and metrics.get("action_oscillation_reduction", 0) > 0
        and metrics.get("screen_vent_conflict_count_shadow", 0) <= metrics.get("screen_vent_conflict_count_original", 0)
    )
    next_action = "minimal_shadow_rollout_with_transition_gate_design" if effective else "profile_candidate_guardrail_compatibility_audit"
    return {
        "schema_version": "shadow_transition_gate_patch_audit_20260530_v42",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow transition gate patch audit",
        },
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "shadow_only": True,
        "failure_scenarios": list(failure_scenarios),
        "window_steps": window_steps,
        "soft_limits": soft_limits,
        "scenario_reports": scenario_reports,
        "metrics": metrics,
        "transition_gate_shadow_effective": effective,
        "next_action": next_action,
    }


def build_comparison(audit: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for scenario in audit.get("scenario_reports", []) or []:
        rows.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "original_metrics": scenario.get("original_metrics", {}),
                "shadow_metrics": scenario.get("shadow_metrics", {}),
                "ppo_metrics": scenario.get("ppo_metrics", {}),
                "delta_reduction": scenario.get("delta_reduction", {}),
                "shadow_gate_counters": scenario.get("shadow_gate_counters", {}),
            }
        )
    return {
        "schema_version": "shadow_transition_gate_failure_window_comparison_20260530_v42",
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "summary_metrics": audit.get("metrics", {}),
        "scenario_comparisons": rows,
        "next_action": audit.get("next_action"),
    }


def build_readiness(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "metadata_replay_readiness_checklist_20260530_v42",
        "mainline_alignment": audit.get("mainline_alignment", {}),
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "default_llm_rspc_v2_changed": False,
        "shadow_transition_gate_audit_complete": True,
        "transition_gate_shadow_effective": bool(audit.get("transition_gate_shadow_effective", False)),
        "metrics": audit.get("metrics", {}),
        "next_action": audit.get("next_action"),
    }


def _markdown_table(metrics: Mapping[str, Any]) -> list[str]:
    lines = ["| metric | value |", "| --- | ---: |"]
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            rendered = "`complex`"
        else:
            rendered = f"`{value}`"
        lines.append(f"| {key} | {rendered} |")
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
        *_markdown_table(report.get("metrics", {}) or report.get("summary_metrics", {}) or {}),
    ]
    if report.get("scenario_reports"):
        lines.extend(
            [
                "",
                "## Scenario Comparison",
                "",
                "| scenario | large delta original | large delta shadow | oscillation original | oscillation shadow | bypass |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in report.get("scenario_reports", []) or []:
            original = item.get("original_metrics", {}) or {}
            shadow = item.get("shadow_metrics", {}) or {}
            counters = item.get("shadow_gate_counters", {}) or {}
            lines.append(
                f"| `{item.get('scenario_id')}` | `{original.get('large_action_delta_count', 0)}` | "
                f"`{shadow.get('large_action_delta_count', 0)}` | `{original.get('action_oscillation_count', 0)}` | "
                f"`{shadow.get('action_oscillation_count', 0)}` | `{counters.get('hard_safety_bypass_count', 0)}` |"
            )
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
    parser.add_argument("--v40-attribution-json", required=True)
    parser.add_argument("--v41-stable-dataset-json", required=True)
    parser.add_argument("--v41-unstable-dataset-json", required=True)
    parser.add_argument("--v41-envelope-json", required=True)
    parser.add_argument("--failure-scenario", action="append", default=[])
    parser.add_argument("--window-steps", type=int, default=60)
    parser.add_argument("--reversal-window-steps", type=int, default=6)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--audit-md", required=True)
    parser.add_argument("--comparison-json", required=True)
    parser.add_argument("--comparison-md", required=True)
    parser.add_argument("--readiness-json", required=True)
    parser.add_argument("--readiness-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Load all declared sources for provenance and to fail fast on missing artifacts.
    _load_json(args.v40_attribution_json)
    _load_json(args.v41_stable_dataset_json)
    _load_json(args.v41_unstable_dataset_json)
    envelope = _load_json(args.v41_envelope_json)
    failure_scenarios = args.failure_scenario or list(DEFAULT_FAILURE_SCENARIOS)
    audit = build_audit(
        trace_dir=args.trace_dir,
        envelope=envelope,
        failure_scenarios=failure_scenarios,
        window_steps=args.window_steps,
        reversal_window_steps=args.reversal_window_steps,
    )
    comparison = build_comparison(audit)
    readiness = build_readiness(audit)
    _write_pair(args.audit_json, args.audit_md, audit, title="Shadow Transition Gate Patch Audit v42")
    _write_pair(args.comparison_json, args.comparison_md, comparison, title="Shadow Transition Gate Failure Window Comparison v42")
    _write_pair(args.readiness_json, args.readiness_md, readiness, title="Metadata Replay Readiness Checklist v42")
    print(f"transition_gate_shadow_effective={audit['transition_gate_shadow_effective']}")
    print(f"next_action={readiness['next_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
