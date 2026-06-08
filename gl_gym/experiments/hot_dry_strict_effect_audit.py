"""Audit paired holdout effects for the strict hot-dry proposer controller."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gl_gym.experiments.profile_scorer_decision_audit import (  # noqa: E402
    discover_traces,
    read_trace,
    trace_identity,
)


BASELINE_CONTROLLER = "llm_rspc_v2"
STRICT_CONTROLLER = "llm_rspc_v2_hot_dry_proposer_strict"

DELTA_METRICS = {
    "total_reward": "d_reward",
    "total_rh_low_violation": "d_RHlow",
    "total_vpd_high_excess": "d_VPDhi",
    "total_temp_violation": "d_temp",
    "total_rh_high_violation": "d_RHhigh",
    "canopy_dew_margin_lt0_steps": "d_canopy_lt0",
}

STRICT_FILTER_REASONS = {
    "candidate_not_allowed",
    "margin_below_strict_min",
    "not_severe_dry",
    "temp_headroom_low",
    "canopy_reserve_low",
}


def _truthy(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _discover_benchmark_json(inputs: Sequence[str | Path]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.glob("*.json")))
    return sorted(set(files))


def _load_summaries(inputs: Sequence[str | Path]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for path in _discover_benchmark_json(inputs):
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data.get("summaries", []):
            if isinstance(item, dict):
                summaries.append(item)
    return summaries


def _trace_stats(inputs: Sequence[str | Path]) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], List[Dict[str, Any]]]:
    stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    applied_rows: List[Dict[str, Any]] = []
    for path in discover_traces(inputs):
        identity = trace_identity(path)
        scenario_id = str(identity.get("scenario_id", "") or "")
        controller = str(identity.get("controller", "") or "")
        rows = read_trace(path)
        applied = []
        unsafe = 0
        strict_filtered = 0
        for row in rows:
            reason = str(row.get("rspc_hot_dry_proposer_control_reason", "") or "")
            is_applied = _truthy(row, "rspc_hot_dry_proposer_control_applied")
            is_safe = _truthy(row, "rspc_hot_dry_proposer_control_safe_hot_dry")
            gate = str(row.get("rspc_hot_dry_proposer_control_safety_gate_reason", "none") or "none")
            if reason in STRICT_FILTER_REASONS:
                strict_filtered += 1
            if is_applied:
                unsafe += int((not is_safe) or gate != "none")
                record = {
                    "trace_id": identity.get("trace_id", ""),
                    "scenario_id": scenario_id,
                    "controller": controller,
                    "step": int(_num(row, "step", 0.0)),
                    "temp_air": _num(row, "temp_air", 0.0),
                    "rh_air": _num(row, "rh_air", 0.0),
                    "vpd_air": _num(row, "vpd_air", 0.0),
                    "canopy_dew_margin": _num(row, "canopy_dew_margin", 0.0),
                    "candidate": str(row.get("rspc_hot_dry_proposer_control_candidate", "") or ""),
                    "margin": _num(row, "rspc_hot_dry_proposer_control_margin", 0.0),
                    "delta_screen": _num(row, "rspc_hot_dry_proposer_control_delta_screen", 0.0),
                    "delta_vent": _num(row, "rspc_hot_dry_proposer_control_delta_vent", 0.0),
                    "delta_shade": _num(row, "rspc_hot_dry_proposer_control_delta_shade", 0.0),
                    "reason": reason,
                }
                applied.append(record)
                applied_rows.append(record)
        stats[(scenario_id, controller)] = {
            "trace_id": identity.get("trace_id", ""),
            "rows": len(rows),
            "strict_applied_steps": len(applied),
            "unsafe_applied_steps": unsafe,
            "strict_filtered_steps": strict_filtered,
        }
    return stats, applied_rows


def _scenario_day(scenario_id: str) -> int:
    for part in str(scenario_id).split("_"):
        if part.startswith("d"):
            try:
                return int(part[1:])
            except ValueError:
                return -1
    return -1


def _round(value: float) -> float:
    return round(float(value), 6)


def audit_effects(
    benchmark_inputs: Sequence[str | Path],
    trace_inputs: Sequence[str | Path],
    *,
    max_vpd_d120_regression: float = 0.05,
) -> Dict[str, Any]:
    summaries = _load_summaries(benchmark_inputs)
    by_pair: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for item in summaries:
        scenario_id = str(item.get("scenario_id", "") or "")
        controller = str(item.get("controller", "") or "")
        if scenario_id and controller:
            by_pair[(scenario_id, controller)] = item

    trace_stats, applied_rows = _trace_stats(trace_inputs)
    scenario_ids = sorted({scenario for scenario, _controller in by_pair})
    scenarios: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for scenario_id in scenario_ids:
        baseline = by_pair.get((scenario_id, BASELINE_CONTROLLER))
        strict = by_pair.get((scenario_id, STRICT_CONTROLLER))
        if not baseline or not strict:
            warnings.append(f"missing_pair:{scenario_id}")
            continue
        base_agg = baseline.get("aggregate", {}) if isinstance(baseline.get("aggregate", {}), Mapping) else {}
        strict_agg = strict.get("aggregate", {}) if isinstance(strict.get("aggregate", {}), Mapping) else {}
        deltas = {
            output: _round(_num(strict_agg, source, 0.0) - _num(base_agg, source, 0.0))
            for source, output in DELTA_METRICS.items()
        }
        tstats = trace_stats.get((scenario_id, STRICT_CONTROLLER), {})
        scenario = {
            "scenario_id": scenario_id,
            "day": _scenario_day(scenario_id),
            "runtime_error_steps": int(_num(strict_agg, "runtime_error_steps", 0.0)),
            "cache_hit_steps": int(_num(strict_agg, "plan_cache_hit_steps", 0.0)),
            "strict_applied_steps": int(tstats.get("strict_applied_steps", 0) or 0),
            "strict_filtered_steps": int(tstats.get("strict_filtered_steps", 0) or 0),
            "unsafe_applied_steps": int(tstats.get("unsafe_applied_steps", 0) or 0),
            **deltas,
        }
        scenarios.append(scenario)

    unsafe = [s for s in scenarios if int(s.get("unsafe_applied_steps", 0) or 0) > 0]
    runtime_bad = [
        s
        for s in scenarios
        if int(s.get("runtime_error_steps", 0) or 0) > 0 or int(s.get("cache_hit_steps", 0) or 0) != 240
    ]
    d180_d240_regress = [
        s
        for s in scenarios
        if int(s.get("day", -1)) in {180, 240}
        and (
            float(s.get("d_RHlow", 0.0) or 0.0) > 1e-6
            or float(s.get("d_VPDhi", 0.0) or 0.0) > 1e-6
            or float(s.get("d_temp", 0.0) or 0.0) > 1e-6
            or float(s.get("d_canopy_lt0", 0.0) or 0.0) > 0.0
        )
    ]
    d120 = [s for s in scenarios if int(s.get("day", -1)) == 120]
    d120_positive = [
        s
        for s in d120
        if (
            float(s.get("d_reward", 0.0) or 0.0) > 1e-6
            or float(s.get("d_RHlow", 0.0) or 0.0) < -1e-6
        )
        and float(s.get("d_VPDhi", 0.0) or 0.0) <= max_vpd_d120_regression
    ]
    strict_applied_total = sum(int(s.get("strict_applied_steps", 0) or 0) for s in scenarios)
    if warnings:
        decision = "needs_data"
        reason = "Missing paired controller summaries."
    elif unsafe or runtime_bad:
        decision = "holdout_failed_runtime_or_safety"
        reason = "Runtime/cache or unsafe-applied failures were found."
    elif d180_d240_regress:
        decision = "holdout_failed_regression"
        reason = "A d180/d240 holdout scenario regressed on RH/VPD/temp/canopy metrics."
    elif strict_applied_total <= 0:
        decision = "strict_not_triggered"
        reason = "Strict proposer did not trigger on holdout scenarios."
    elif d120_positive:
        decision = "holdout_positive_signal"
        reason = "At least one d120 holdout improved reward or RH-low without meaningful VPD regression."
    else:
        decision = "no_stable_benefit"
        reason = "Strict proposer was safe, but holdout benefit was not positive."

    aggregate = {
        "scenario_count": len(scenarios),
        "strict_applied_steps": int(strict_applied_total),
        "unsafe_applied_steps": int(sum(int(s.get("unsafe_applied_steps", 0) or 0) for s in scenarios)),
        "runtime_error_steps": int(sum(int(s.get("runtime_error_steps", 0) or 0) for s in scenarios)),
        "d120_positive_count": int(len(d120_positive)),
        "d180_d240_regression_count": int(len(d180_d240_regress)),
        "warnings": warnings,
        "recommendation": {"decision": decision, "reason": reason},
    }
    return {
        "schema_version": "hot_dry_strict_effect_audit_v1",
        "baseline_controller": BASELINE_CONTROLLER,
        "strict_controller": STRICT_CONTROLLER,
        "aggregate": aggregate,
        "scenarios": scenarios,
        "strict_applied_rows": applied_rows,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {}) if isinstance(audit.get("aggregate", {}), Mapping) else {}
    rec = aggregate.get("recommendation", {}) if isinstance(aggregate.get("recommendation", {}), Mapping) else {}
    lines = [
        "# Hot-Dry Strict Proposer Holdout Effect Audit v1",
        "",
        "## Aggregate",
        "",
        f"- decision: `{rec.get('decision', 'unknown')}`",
        f"- reason: {rec.get('reason', '')}",
        f"- scenarios: {aggregate.get('scenario_count', 0)}",
        f"- strict applied steps: {aggregate.get('strict_applied_steps', 0)}",
        f"- unsafe applied steps: {aggregate.get('unsafe_applied_steps', 0)}",
        f"- runtime error steps: {aggregate.get('runtime_error_steps', 0)}",
        f"- d120 positive count: {aggregate.get('d120_positive_count', 0)}",
        f"- d180/d240 regression count: {aggregate.get('d180_d240_regression_count', 0)}",
        "",
        "## Scenario Paired Delta",
        "",
        "| scenario | applied | unsafe | cache | runtime | d_reward | d_RHlow | d_VPDhi | d_temp | d_RHhigh | d_canopy<0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit.get("scenarios", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {scenario} | {applied} | {unsafe} | {cache} | {runtime} | {reward} | {rhlow} | {vpd} | {temp} | {rhhigh} | {canopy} |".format(
                scenario=row.get("scenario_id", ""),
                applied=row.get("strict_applied_steps", 0),
                unsafe=row.get("unsafe_applied_steps", 0),
                cache=row.get("cache_hit_steps", 0),
                runtime=row.get("runtime_error_steps", 0),
                reward=_fmt(row.get("d_reward", 0.0)),
                rhlow=_fmt(row.get("d_RHlow", 0.0)),
                vpd=_fmt(row.get("d_VPDhi", 0.0)),
                temp=_fmt(row.get("d_temp", 0.0)),
                rhhigh=_fmt(row.get("d_RHhigh", 0.0)),
                canopy=_fmt(row.get("d_canopy_lt0", 0.0)),
            )
        )
    lines.extend(["", "## Strict Applied Rows", ""])
    rows = [row for row in audit.get("strict_applied_rows", []) if isinstance(row, Mapping)]
    if not rows:
        lines.append("- none")
    else:
        lines.extend(
            [
                "| scenario | step | temp | RH | VPD | canopy margin | candidate | margin | d_screen | d_vent | d_shade |",
                "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| {scenario} | {step} | {temp} | {rh} | {vpd} | {canopy} | {candidate} | {margin} | {screen} | {vent} | {shade} |".format(
                    scenario=row.get("scenario_id", ""),
                    step=row.get("step", 0),
                    temp=_fmt(row.get("temp_air", 0.0)),
                    rh=_fmt(row.get("rh_air", 0.0)),
                    vpd=_fmt(row.get("vpd_air", 0.0)),
                    canopy=_fmt(row.get("canopy_dew_margin", 0.0)),
                    candidate=row.get("candidate", ""),
                    margin=_fmt(row.get("margin", 0.0)),
                    screen=_fmt(row.get("delta_screen", 0.0)),
                    vent=_fmt(row.get("delta_vent", 0.0)),
                    shade=_fmt(row.get("delta_shade", 0.0)),
                )
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-json", nargs="+", required=True)
    parser.add_argument("--trace-dir", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--max-vpd-d120-regression", type=float, default=0.05)
    args = parser.parse_args(argv)

    audit = audit_effects(
        args.benchmark_json,
        args.trace_dir,
        max_vpd_d120_regression=float(args.max_vpd_d120_regression),
    )
    output_json = Path(args.output_json)
    output_report = Path(args.output_report)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    output_report.write_text(build_report(audit), encoding="utf-8")
    print(f"Saved strict effect JSON to {output_json}")
    print(f"Saved strict effect report to {output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
