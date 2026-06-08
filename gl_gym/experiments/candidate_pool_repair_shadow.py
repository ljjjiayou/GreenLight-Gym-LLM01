"""Shadow-only hypothetical repair candidates for canopy-risk pool gaps.

This script does not add candidates to the controller.  It only asks whether a
simple canopy-safe action shape appears feasible in rows where the current
candidate pool audit reports no safe dry-relief candidate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.candidate_pool_audit import classify_step

ACTION_FIELDS = ("heat", "co2", "screen", "vent", "lamp", "shade")
TRACE_ACTION_FIELDS = {
    "heat": "u_heating",
    "co2": "u_co2",
    "screen": "u_screen",
    "vent": "u_ventilation",
    "lamp": "u_lighting",
    "shade": "u_shading",
}


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


def _action(row: Mapping[str, Any]) -> dict[str, float]:
    return {name: _num(row.get(field)) for name, field in TRACE_ACTION_FIELDS.items()}


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _apply_action_delta(action: Mapping[str, float], delta: Mapping[str, Any] | None) -> dict[str, float]:
    if not delta:
        return {field: float(action.get(field, 0.0)) for field in ACTION_FIELDS}
    return {field: _clip01(_num(action.get(field)) + _num(delta.get(field))) for field in ACTION_FIELDS}


def _candidate_family(name: str) -> str:
    if "hold_screen" in name or "no_screen" in name:
        return "canopy_safe_hold_screen"
    if "min_vent" in name:
        return "canopy_safe_min_vent"
    if "balanced" in name:
        return "canopy_safe_balanced_repair"
    return "canopy_safe_other"


def _vpd_kpa(temp_c: float, rh_pct: float) -> float:
    saturation = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    return float(saturation * (1.0 - max(0.0, min(100.0, rh_pct)) / 100.0))


def _proxy_response(
    row: Mapping[str, Any],
    action: Mapping[str, float],
    prev_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    temp = _num(row.get("temp_air"), 20.0)
    rh = _num(row.get("rh_air"), 70.0)
    dew_margin = _num(row.get("dew_margin_air"), 3.0)
    canopy_margin = _num(row.get("canopy_dew_margin"), 3.0)
    rad = _num(row.get("glob_rad"), 0.0)
    heat = _num(action.get("heat"))
    screen = _num(action.get("screen"))
    vent = _num(action.get("vent"))
    lamp = _num(action.get("lamp"))
    shade = _num(action.get("shade"))
    current = _action(row)
    screen_delta = screen - _num(current.get("screen"))
    vent_delta = vent - _num(current.get("vent"))
    prev_canopy = _num(prev_row.get("canopy_dew_margin"), canopy_margin) if prev_row else canopy_margin
    canopy_trend = canopy_margin - prev_canopy

    temp_next = temp + 0.55 * heat + 0.12 * lamp + 0.04 * screen - 0.16 * vent - 0.12 * shade * min(rad / 500.0, 1.0)
    rh_next = max(35.0, min(100.0, rh - 3.8 * vent - 1.1 * heat + 0.30 * screen - 0.15 * lamp))
    vpd_next = _vpd_kpa(temp_next, rh_next)
    rh_rise = max(rh_next - rh, 0.0)
    rh_drop = max(rh - rh_next, 0.0)
    temp_delta = temp_next - temp
    dew_next = dew_margin + 0.06 * temp_delta + 0.012 * rh_drop + 0.20 * max(vent_delta, 0.0) - 0.025 * rh_rise - 0.14 * max(screen_delta, 0.0) - 0.10 * max(-vent_delta, 0.0)
    canopy_next = canopy_margin + 0.08 * temp_delta + 0.010 * rh_drop + 0.24 * max(vent_delta, 0.0) - 0.030 * rh_rise - 0.32 * max(screen_delta, 0.0) - 0.20 * max(-vent_delta, 0.0)
    negative_trend = max(-canopy_trend, 0.0)
    risk_terms = {
        "base": 0.20,
        "screen_delta": 0.35 * max(screen_delta, 0.0),
        "ventilation_delta": 0.25 * max(-vent_delta, 0.0),
        "high_screen": 0.30 if screen >= 0.75 else (0.15 if screen >= 0.55 else 0.0),
        "low_ventilation": 0.25 if vent <= 0.25 else (0.10 if vent <= 0.35 else 0.0),
        "rh_high": 0.15 * max(rh - 85.0, 0.0) / 10.0,
        "near_boundary": 0.25 if canopy_margin < 1.5 else (0.10 if canopy_next < 1.5 else 0.0),
        "negative_canopy_trend": min(0.60, 0.30 * negative_trend),
        "low_air_dew_margin": 0.15 if dew_margin < 1.5 else 0.0,
    }
    risk_buffer = float(sum(float(value) for value in risk_terms.values()))
    canopy_next_v2 = float(canopy_next - risk_buffer)
    warning_v2 = bool(
        canopy_next_v2 < 0.25
        or (
            (canopy_margin < 1.5 or canopy_next < 1.0)
            and screen >= 0.55
            and vent <= 0.25
            and (vent_delta < -0.02 or negative_trend > 0.35)
        )
    )
    rh_low_delta = max(55.0 - rh_next, 0.0) - max(55.0 - rh, 0.0)
    vpd_hi_delta = max(vpd_next - 1.20, 0.0) - max(_num(row.get("vpd_air"), 0.0) - 1.20, 0.0)
    return {
        "predicted_temp_next": float(temp_next),
        "predicted_rh_next": float(rh_next),
        "predicted_vpd_next": float(vpd_next),
        "predicted_dew_margin_air_next": float(dew_next),
        "predicted_canopy_dew_margin_next": float(canopy_next),
        "predicted_canopy_dew_margin_next_v1": float(canopy_next),
        "predicted_canopy_dew_margin_next_v2": float(canopy_next_v2),
        "predicted_canopy_warning_v2": warning_v2,
        "predicted_canopy_lt0_v2": bool(canopy_next_v2 < 0.0),
        "canopy_proxy_v2_risk_buffer": risk_buffer,
        "canopy_proxy_v2_risk_terms": risk_terms,
        "predicted_dew_lt0": bool(dew_next < 0.0),
        "predicted_canopy_lt0": bool(canopy_next < 0.0),
        "predicted_RHlow_delta": float(rh_low_delta),
        "predicted_VPDhi_delta": float(vpd_hi_delta),
        "predicted_temp_delta": float(temp_delta),
        "energy_proxy": float(0.70 * heat + 0.22 * _num(action.get("co2")) + 1.05 * lamp + 0.05 * vent),
    }


def build_hypothetical_candidates(
    row: Mapping[str, Any],
    prev_row: Mapping[str, Any] | None = None,
    post_guardrail_delta: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    base = _action(row)
    vent = _num(base.get("vent"))
    screen = _num(base.get("screen"))
    shade = _num(base.get("shade"))
    candidates: list[tuple[str, dict[str, float]]] = []

    no_screen = dict(base)
    no_screen["screen"] = min(screen, max(0.0, screen - 0.10))
    candidates.append(("canopy_safe_no_screen_gain", no_screen))

    hold_screen = dict(base)
    hold_screen["screen"] = min(screen, 0.55)
    candidates.append(("canopy_safe_hold_screen", hold_screen))

    min_vent = dict(base)
    min_vent["vent"] = max(vent, 0.42)
    candidates.append(("canopy_safe_min_vent", min_vent))

    hold = dict(base)
    hold["screen"] = min(screen, 0.55)
    hold["vent"] = max(vent, 0.35)
    candidates.append(("canopy_safe_hold_screen_vent", hold))

    balanced = dict(base)
    balanced["screen"] = min(screen, 0.60)
    balanced["vent"] = max(vent, 0.32)
    balanced["shade"] = max(shade, 0.35 if _num(row.get("glob_rad")) > 250.0 else shade)
    balanced["lamp"] = 0.0
    balanced["co2"] = 0.0
    candidates.append(("canopy_safe_dry_relief_balanced", balanced))
    candidates.append(("canopy_safe_balanced_repair", balanced))

    out = []
    for name, action in candidates:
        prediction = _proxy_response(row, action, prev_row)
        post_guardrail_shadow_action = _apply_action_delta(action, post_guardrail_delta)
        post_guardrail_prediction = _proxy_response(row, post_guardrail_shadow_action, prev_row)
        post_guardrail_reintroduces_canopy_risk = bool(
            post_guardrail_prediction["predicted_canopy_lt0_v2"]
            or post_guardrail_prediction["predicted_canopy_warning_v2"]
            or post_guardrail_prediction["predicted_dew_lt0"]
        )
        post_guardrail_remains_safe = bool(
            not post_guardrail_reintroduces_canopy_risk
            and post_guardrail_prediction["predicted_temp_next"] < 32.0
            and post_guardrail_prediction["predicted_rh_next"] >= _num(row.get("rh_air"), 70.0) - 3.0
            and post_guardrail_prediction["predicted_vpd_next"] <= _num(row.get("vpd_air"), 0.0) + 0.15
        )
        rhlow_delta = _num(prediction.get("predicted_RHlow_delta"))
        vpdhi_delta = _num(prediction.get("predicted_VPDhi_delta"))
        dry_relief_useful = bool(rhlow_delta <= 0.0 and vpdhi_delta <= 0.0)
        dry_tradeoff = bool(rhlow_delta > 0.05 or vpdhi_delta > 0.05)
        conservative_ineffective = bool(abs(rhlow_delta) < 0.01 and abs(vpdhi_delta) < 0.01)
        feasibility_reasons: list[str] = []
        if prediction["predicted_canopy_lt0_v2"] or prediction["predicted_canopy_warning_v2"]:
            feasibility_reasons.append("canopy_risk_v2")
        if prediction["predicted_dew_lt0"]:
            feasibility_reasons.append("dew_risk")
        if prediction["predicted_temp_next"] >= 32.0:
            feasibility_reasons.append("temp_risk")
        if prediction["predicted_rh_next"] < _num(row.get("rh_air"), 70.0) - 3.0:
            feasibility_reasons.append("rh_tradeoff")
        if prediction["predicted_vpd_next"] > _num(row.get("vpd_air"), 0.0) + 0.15:
            feasibility_reasons.append("vpd_tradeoff")
        feasible = bool(
            not prediction["predicted_canopy_lt0_v2"]
            and not prediction["predicted_canopy_warning_v2"]
            and not prediction["predicted_dew_lt0"]
            and prediction["predicted_temp_next"] < 32.0
            and prediction["predicted_rh_next"] >= _num(row.get("rh_air"), 70.0) - 3.0
            and prediction["predicted_vpd_next"] <= _num(row.get("vpd_air"), 0.0) + 0.15
        )
        out.append(
            {
                "name": name,
                "candidate_family": _candidate_family(name),
                "action": {field: float(action.get(field, 0.0)) for field in ACTION_FIELDS},
                "prediction": prediction,
                "post_guardrail_shadow_action": post_guardrail_shadow_action,
                "post_guardrail_shadow_prediction": post_guardrail_prediction,
                "post_guardrail_remains_safe": post_guardrail_remains_safe,
                "post_guardrail_reintroduces_canopy_risk": post_guardrail_reintroduces_canopy_risk,
                "feasible": feasible,
                "dry_relief_useful": dry_relief_useful,
                "dry_tradeoff": dry_tradeoff,
                "conservative_ineffective_action": conservative_ineffective,
                "tradeoff_quality": (
                    "useful_dry_relief"
                    if dry_relief_useful
                    else ("dry_tradeoff" if dry_tradeoff else ("conservative_ineffective" if conservative_ineffective else "neutral"))
                ),
                "feasibility_reason": "ok" if feasible else ",".join(feasibility_reasons or ["unknown"]),
            }
        )
    return out


def _trace_repair_conclusion(
    gap_steps: int,
    repair_available_steps: int,
    useful_repair_steps: int = 0,
) -> str:
    if gap_steps <= 0:
        return "guardrail_filter_sufficient"
    if useful_repair_steps >= gap_steps:
        return "new_candidate_needed_shadow_evidence"
    if repair_available_steps >= gap_steps:
        return "new_candidate_needed"
    if repair_available_steps <= 0:
        return "no_safe_repair_found"
    return "response_model_unreliable"


def audit_trace(
    *,
    trace_path: str | Path,
    preset: str,
    scenario_id: str,
    start_step: int,
    end_step: int,
) -> dict[str, Any]:
    rows_by_step = _rows_by_step(_read_trace(trace_path))
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for step in range(int(start_step), int(end_step) + 1):
        row = rows_by_step.get(step)
        if row is None:
            continue
        classification = classify_step(row, rows_by_step.get(step + 1))
        label = str(classification.get("label", ""))
        if label != "no_safe_candidate_available":
            continue
        post_guardrail_delta = classification.get("post_guardrail_delta", {})
        candidates = build_hypothetical_candidates(row, rows_by_step.get(step - 1), post_guardrail_delta)
        feasible = [item for item in candidates if bool(item.get("feasible", False))]
        post_guardrail_safe_feasible = [
            item for item in feasible if bool(item.get("post_guardrail_remains_safe", False))
        ]
        best_canopy_safe = min(feasible, key=lambda item: _num(item.get("prediction", {}).get("energy_proxy"))) if feasible else {}
        best_tradeoff = (
            min(
                feasible,
                key=lambda item: (
                    _num(item.get("prediction", {}).get("predicted_VPDhi_delta"))
                    + 0.2 * _num(item.get("prediction", {}).get("predicted_RHlow_delta"))
                    + 0.05 * _num(item.get("prediction", {}).get("energy_proxy"))
                ),
            )
            if feasible
            else {}
        )
        rejected = [item.get("name", "") for item in candidates if not bool(item.get("feasible", False))]
        repair_available = bool(feasible)
        useful_feasible = [item for item in feasible if bool(item.get("dry_relief_useful"))]
        useful_post_guardrail_safe = [
            item for item in useful_feasible if bool(item.get("post_guardrail_remains_safe", False))
        ]
        dry_tradeoff_feasible = [item for item in feasible if bool(item.get("dry_tradeoff"))]
        conservative_ineffective_feasible = [item for item in feasible if bool(item.get("conservative_ineffective_action"))]
        candidate_pool_gap = True
        counts["no_safe_candidate_available"] += 1
        counts["candidate_pool_gap"] += 1
        if repair_available:
            counts["repair_candidate_available"] += 1
            if useful_feasible:
                counts["feasible_useful_dry_relief"] += 1
                if useful_post_guardrail_safe:
                    counts["post_guardrail_safe_useful_repair"] += 1
                else:
                    counts["post_guardrail_reintroduces_risk_for_useful_repair"] += 1
            elif dry_tradeoff_feasible:
                counts["feasible_but_dry_tradeoff"] += 1
            elif conservative_ineffective_feasible:
                counts["feasible_but_conservative_ineffective"] += 1
            else:
                counts["feasible_neutral_repair"] += 1
        else:
            counts["no_hypothetical_safe_candidate"] += 1
        records.append(
            {
                "preset": preset,
                "scenario_id": scenario_id,
                "step": step,
                "source_label": label,
                "candidate_pool_gap": candidate_pool_gap,
                "candidate_pool_gap_confirmed": bool(not repair_available),
                "repair_candidate_available": repair_available,
                "post_guardrail_safe_repair_available": bool(post_guardrail_safe_feasible),
                "best_hypothetical_candidate": best_canopy_safe.get("name", "") if best_canopy_safe else "",
                "best_canopy_safe_candidate": best_canopy_safe.get("name", "") if best_canopy_safe else "",
                "best_tradeoff_candidate": best_tradeoff.get("name", "") if best_tradeoff else "",
                "best_post_guardrail_safe_candidate": (
                    post_guardrail_safe_feasible[0].get("name", "") if post_guardrail_safe_feasible else ""
                ),
                "rejected_hypothetical_candidates": rejected,
                "feasible_useful_dry_relief_count": len(useful_feasible),
                "feasible_dry_tradeoff_count": len(dry_tradeoff_feasible),
                "feasible_conservative_ineffective_count": len(conservative_ineffective_feasible),
                "best_candidate_tradeoff_quality": str(best_tradeoff.get("tradeoff_quality", "") if best_tradeoff else ""),
                "repair_should_be_new_candidate": repair_available,
                "repair_should_be_guardrail_filter_only": bool(not repair_available),
                "post_guardrail_policy_blocks_candidate_repair": bool(repair_available and not post_guardrail_safe_feasible),
                "repair_conclusion": (
                    "new_candidate_needed_shadow_evidence_post_guardrail_safe"
                    if useful_post_guardrail_safe
                    else "post_guardrail_policy_blocks_candidate_repair"
                    if repair_available and not post_guardrail_safe_feasible
                    else "new_candidate_needed_shadow_evidence"
                    if useful_feasible
                    else ("new_candidate_needed" if repair_available else "no_safe_repair_found")
                ),
                "hypothetical_candidates": candidates,
                "state": {
                    "temp_air": _num(row.get("temp_air")),
                    "rh_air": _num(row.get("rh_air")),
                    "vpd_air": _num(row.get("vpd_air")),
                    "dew_margin_air": _num(row.get("dew_margin_air")),
                    "canopy_dew_margin": _num(row.get("canopy_dew_margin")),
                },
            }
        )
    gap_steps = int(counts.get("candidate_pool_gap", 0))
    repair_steps = int(counts.get("repair_candidate_available", 0))
    useful_steps = int(counts.get("feasible_useful_dry_relief", 0))
    post_guardrail_safe_useful_steps = int(counts.get("post_guardrail_safe_useful_repair", 0))
    return {
        "preset": preset,
        "scenario_id": scenario_id,
        "trace_path": str(trace_path),
        "counts": dict(sorted(counts.items())),
        "repair_conclusion": _trace_repair_conclusion(gap_steps, repair_steps, useful_steps),
        "candidate_generation_shadow_conclusion": (
            "new_candidate_needed_shadow_evidence_post_guardrail_safe"
            if gap_steps > 0 and post_guardrail_safe_useful_steps >= gap_steps
            else "post_guardrail_policy_blocks_candidate_repair"
            if useful_steps > 0 and post_guardrail_safe_useful_steps < useful_steps
            else _trace_repair_conclusion(gap_steps, repair_steps, useful_steps)
        ),
        "records": records,
    }


def build_report(
    *,
    trace_dirs: Sequence[str | Path],
    scenario_id: str,
    start_step: int,
    end_step: int,
) -> dict[str, Any]:
    traces = []
    aggregate: Counter[str] = Counter()
    seen: set[Path] = set()
    for raw in trace_dirs:
        for trace_path in _iter_trace_paths(raw, scenario_id):
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
            )
            traces.append(trace)
            aggregate.update(trace.get("counts", {}))
    gap_steps = int(aggregate.get("candidate_pool_gap", 0))
    repair_steps = int(aggregate.get("repair_candidate_available", 0))
    useful_steps = int(aggregate.get("feasible_useful_dry_relief", 0))
    post_guardrail_safe_useful_steps = int(aggregate.get("post_guardrail_safe_useful_repair", 0))
    return {
        "schema_version": "candidate_pool_repair_shadow_v1",
        "scenario_id": scenario_id,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "trace_count": len(traces),
        "aggregate_counts": dict(sorted(aggregate.items())),
        "candidate_pool_gap_steps": gap_steps,
        "repair_candidate_available_steps": repair_steps,
        "candidate_pool_gap_confirmed": bool(gap_steps > 0),
        "candidate_pool_repair_conclusion": _trace_repair_conclusion(gap_steps, repair_steps, useful_steps),
        "candidate_generation_shadow_conclusion": (
            "new_candidate_needed_shadow_evidence_post_guardrail_safe"
            if gap_steps > 0 and post_guardrail_safe_useful_steps >= gap_steps
            else "post_guardrail_policy_blocks_candidate_repair"
            if useful_steps > 0 and post_guardrail_safe_useful_steps < useful_steps
            else _trace_repair_conclusion(gap_steps, repair_steps, useful_steps)
        ),
        "repair_quality_counts": {
            "feasible_useful_dry_relief": int(aggregate.get("feasible_useful_dry_relief", 0)),
            "post_guardrail_safe_useful_repair": int(aggregate.get("post_guardrail_safe_useful_repair", 0)),
            "post_guardrail_reintroduces_risk_for_useful_repair": int(
                aggregate.get("post_guardrail_reintroduces_risk_for_useful_repair", 0)
            ),
            "feasible_but_dry_tradeoff": int(aggregate.get("feasible_but_dry_tradeoff", 0)),
            "feasible_but_conservative_ineffective": int(aggregate.get("feasible_but_conservative_ineffective", 0)),
            "feasible_neutral_repair": int(aggregate.get("feasible_neutral_repair", 0)),
        },
        "traces": traces,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate Pool Repair Shadow",
        "",
        f"- Scenario: `{report.get('scenario_id', '')}`",
        f"- Steps: {report.get('start_step')}..{report.get('end_step')}",
        f"- Candidate-pool gap steps: {report.get('candidate_pool_gap_steps', 0)}",
        f"- Repair-candidate available steps: {report.get('repair_candidate_available_steps', 0)}",
        f"- Candidate-pool gap confirmed: {report.get('candidate_pool_gap_confirmed', False)}",
        f"- Repair conclusion: `{report.get('candidate_pool_repair_conclusion', '')}`",
        f"- Candidate generation shadow conclusion: `{report.get('candidate_generation_shadow_conclusion', '')}`",
        f"- Repair quality counts: {json.dumps(report.get('repair_quality_counts', {}), sort_keys=True)}",
        "",
        "| preset | no_safe_rows | repair_available | useful | dry_tradeoff | conservative_noop | no_hypothetical | conclusion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for trace in report.get("traces", []) or []:
        counts = trace.get("counts", {}) if isinstance(trace.get("counts"), Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(trace.get("preset", "")),
                    str(counts.get("no_safe_candidate_available", 0)),
                    str(counts.get("repair_candidate_available", 0)),
                    str(counts.get("feasible_useful_dry_relief", 0)),
                    str(counts.get("feasible_but_dry_tradeoff", 0)),
                    str(counts.get("feasible_but_conservative_ineffective", 0)),
                    str(counts.get("no_hypothetical_safe_candidate", 0)),
                    str(trace.get("repair_conclusion", "")),
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
    print(f"candidate_pool_gap_steps={report['candidate_pool_gap_steps']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
