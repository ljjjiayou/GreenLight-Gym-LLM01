"""Build v44 CVODES failure causal-attribution artifacts.

This audit is offline-only. It reads existing PPO, original llm_rspc_v2, and
v43 transition-gated traces. It does not run rollout, call online LLMs, mutate
the default controller, authorize controlled replay, or make performance
claims.
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

FAILURE_SCENARIOS = (
    "y2010_d180_s43_n720",
    "y2018_d181_s42_n720",
    "y2018_d181_s43_n720",
)
ACTION_FIELDS = ("u_heating", "u_co2", "u_screen", "u_ventilation", "u_lighting", "u_shading")
RATE_GUARD_TARGET_FIELDS = ("u_heating", "u_screen", "u_ventilation", "u_shading")
PROFILE_FIELDS = ("target_temp", "target_rh", "target_co2")
BOUNDARY_FIELDS = ("temp_air", "rh_air", "vpd_air", "canopy_dew_margin", "dew_margin_air")

AUDIT_DIR = PROJECT_ROOT / "gl_gym" / "result" / "audits"
ORIGINAL_TRACE_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "stable_long_horizon_evaluation_pool_v38_20260529"
    / "traces_h720"
)
V43_TRACE_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "transition_gate_shadow_rollout_v43_20260530"
    / "traces"
)
V40_ATTRIBUTION_JSON = AUDIT_DIR / "h720_failure_window_attribution_20260530_v40.json"
V43_RESULT_JSON = AUDIT_DIR / "transition_gate_shadow_rollout_result_audit_20260530_v43.json"

CAUSAL_JSON = AUDIT_DIR / "cvodes_failure_causal_attribution_20260531_v44.json"
CAUSAL_MD = AUDIT_DIR / "cvodes_failure_causal_attribution_20260531_v44.md"
COMPAT_JSON = AUDIT_DIR / "profile_guardrail_compatibility_audit_20260531_v44.json"
COMPAT_MD = AUDIT_DIR / "profile_guardrail_compatibility_audit_20260531_v44.md"
COUNTERFACTUAL_JSON = AUDIT_DIR / "action_channel_counterfactual_design_20260531_v44.json"
COUNTERFACTUAL_MD = AUDIT_DIR / "action_channel_counterfactual_design_20260531_v44.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260531_v44.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260531_v44.md"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _rel(path: str | Path) -> str:
    p = _resolve(path)
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


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


def _trace_path(trace_dir: str | Path, scenario_id: str, controller: str) -> Path:
    root = _resolve(trace_dir)
    expected = root / f"{scenario_id}_{controller}.csv"
    if expected.exists():
        return expected
    parts = _parts_from_scenario_id(scenario_id)
    prefix = f"y{parts['year']}_d{parts['day']}_s{parts['seed']}_"
    matches = sorted(root.glob(f"{prefix}*_{controller}.csv"))
    return matches[0] if matches else expected


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    p = _resolve(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _runtime_failure_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    for row in rows:
        text = " ".join(str(row.get(key, "") or "") for key in ("runtime_error", "runtime_error_type", "replan_reason"))
        if text.strip() and (
            "CV_CONV_FAILURE" in text
            or "CVODES" in text
            or "CvodesInterface" in text
            or "runtime_error" in text
            or str(row.get("runtime_error", "") or "").strip()
            or str(row.get("runtime_error_type", "") or "").strip()
        ):
            return int(_num(row.get("step"), -1))
    return None


def _window(rows: Sequence[Mapping[str, Any]], *, center_step: int | None, window_steps: int) -> list[Mapping[str, Any]]:
    if not rows:
        return []
    if center_step is None:
        center_step = int(_num(rows[-1].get("step"), len(rows) - 1))
    start = max(0, int(center_step) - int(window_steps) + 1)
    return [row for row in rows if start <= int(_num(row.get("step"), -1)) <= int(center_step)]


def _window_meta(rows: Sequence[Mapping[str, Any]], *, center_step: int | None, window_steps: int) -> dict[str, Any]:
    window = _window(rows, center_step=center_step, window_steps=window_steps)
    if not rows:
        return {
            "row_count": 0,
            "window_start_step": None,
            "window_end_step": None,
            "short_window": True,
        }
    return {
        "row_count": len(window),
        "window_start_step": int(_num(window[0].get("step"), 0)) if window else None,
        "window_end_step": int(_num(window[-1].get("step"), 0)) if window else None,
        "short_window": len(window) < window_steps,
    }


def _action_channel_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    previous: dict[str, float] = {}
    signs: dict[str, int] = {}
    jumps: Counter[str] = Counter()
    reversals: Counter[str] = Counter()
    large_delta_examples: list[dict[str, Any]] = []
    screen_vent_conflicts = 0
    shade_screen_conflicts = 0
    heat_vent_conflicts = 0
    for row in rows:
        step = int(_num(row.get("step"), 0))
        if _num(row.get("u_ventilation")) >= 0.85 and _num(row.get("u_screen")) >= 0.30:
            screen_vent_conflicts += 1
        if _num(row.get("u_shading")) >= 0.80 and _num(row.get("u_screen")) >= 0.30:
            shade_screen_conflicts += 1
        if _num(row.get("u_heating")) >= 0.10 and _num(row.get("u_ventilation")) >= 0.40:
            heat_vent_conflicts += 1
        for field in ACTION_FIELDS:
            value = _num(row.get(field))
            if field in previous:
                delta = value - previous[field]
                if abs(delta) > 0.20:
                    jumps[field] += 1
                    if len(large_delta_examples) < 15:
                        large_delta_examples.append({"step": step, "field": field, "delta": delta})
                sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
                if sign and signs.get(field, 0) and sign != signs[field]:
                    reversals[field] += 1
                if sign:
                    signs[field] = sign
            previous[field] = value
    total_jumps = sum(jumps.values())
    total_reversals = sum(reversals.values())
    channel_score = total_jumps + total_reversals + screen_vent_conflicts + shade_screen_conflicts + heat_vent_conflicts
    return {
        "large_delta_count_by_channel": dict(sorted(jumps.items())),
        "reversal_count_by_channel": dict(sorted(reversals.items())),
        "large_action_delta_count": int(total_jumps),
        "action_reversal_count": int(total_reversals),
        "screen_vent_conflict_count": int(screen_vent_conflicts),
        "shade_screen_conflict_count": int(shade_screen_conflicts),
        "heat_vent_conflict_count": int(heat_vent_conflicts),
        "score": int(channel_score),
        "examples": large_delta_examples,
    }


def _boundary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stats: dict[str, dict[str, float | None]] = {}
    for field in BOUNDARY_FIELDS:
        values = [_num(row.get(field)) for row in rows if row.get(field) not in (None, "")]
        stats[field] = {
            "start": values[0] if values else None,
            "end": values[-1] if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "delta": values[-1] - values[0] if len(values) >= 2 else 0.0,
        }
    counts = {
        "temp_ge_32_count": sum(_num(row.get("temp_air")) >= 32.0 for row in rows),
        "temp_ge_35_count": sum(_num(row.get("temp_air")) >= 35.0 for row in rows),
        "vpd_ge_2_5_count": sum(_num(row.get("vpd_air")) >= 2.5 for row in rows),
        "vpd_ge_3_0_count": sum(_num(row.get("vpd_air")) >= 3.0 for row in rows),
        "rh_ge_90_count": sum(_num(row.get("rh_air")) >= 90.0 for row in rows),
        "canopy_margin_lt1_count": sum(_num(row.get("canopy_dew_margin"), 99.0) < 1.0 for row in rows),
        "canopy_margin_lt0_count": sum(_num(row.get("canopy_dew_margin"), 99.0) < 0.0 for row in rows),
        "dry_risk_count": sum(_truthy(row.get("dry_risk")) for row in rows),
        "dew_risk_count": sum(_truthy(row.get("dew_risk")) for row in rows),
        "hard_safety_warning_count": sum(
            _truthy(row.get("final_action_would_fail_canopy_boundary_v2"))
            or _truthy(row.get("final_action_predicted_canopy_lt0_v2"))
            for row in rows
        ),
    }
    score = (
        counts["temp_ge_32_count"]
        + 2 * counts["temp_ge_35_count"]
        + counts["vpd_ge_2_5_count"]
        + 2 * counts["vpd_ge_3_0_count"]
        + counts["canopy_margin_lt1_count"]
        + 2 * counts["canopy_margin_lt0_count"]
        + counts["dry_risk_count"]
        + counts["hard_safety_warning_count"]
    )
    return {"stats": stats, "counts": counts, "score": int(score)}


def _profile_guardrail_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    profile_jumps = 0
    profile_reversals = 0
    profile_dew_conflicts = 0
    profile_vpd_conflicts = 0
    target_action_mismatch = 0
    previous: dict[str, float] = {}
    signs: dict[str, int] = {}
    guardrail_rewrite = 0
    transition_bypass = 0
    candidate_filtered = 0
    anchor_selected = 0
    reason_counts: Counter[str] = Counter()
    for row in rows:
        for field in PROFILE_FIELDS:
            value = _num(row.get(field))
            if field in previous:
                delta = value - previous[field]
                if abs(delta) > {"target_temp": 2.0, "target_rh": 8.0, "target_co2": 200.0}[field]:
                    profile_jumps += 1
                sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
                if sign and signs.get(field, 0) and sign != signs[field]:
                    profile_reversals += 1
                if sign:
                    signs[field] = sign
            previous[field] = value
        if _truthy(row.get("dew_risk")) and (_num(row.get("target_rh")) >= 80.0 or _num(row.get("profile_selected_delta_target_rh")) > 0):
            profile_dew_conflicts += 1
        if _num(row.get("vpd_high_excess")) > 0 and _num(row.get("profile_selected_delta_target_rh")) < 0:
            profile_vpd_conflicts += 1
        if _num(row.get("target_rh")) >= 75.0 and _num(row.get("u_ventilation")) >= 0.70:
            target_action_mismatch += 1
        applied = _truthy(row.get("tomato_safety_v2_applied"))
        changed = any(
            abs(_num(row.get(f"tomato_safety_v2_{name}_after")) - _num(row.get(f"tomato_safety_v2_{name}_before"))) > 1e-6
            for name in ("heat", "vent", "screen", "shade")
        )
        if applied or changed:
            guardrail_rewrite += 1
            for reason in [part.strip() for part in str(row.get("tomato_safety_v2_reasons", "") or "").split(",") if part.strip()]:
                reason_counts[reason] += 1
        if _truthy(row.get("transition_gate_bypassed")):
            transition_bypass += 1
        best_eligible = str(row.get("rspc_action_post_shape_best_eligible", "") or "").lower()
        raw_reason = str(row.get("rspc_action_post_shape_safety_gate_reason", "") or "")
        if best_eligible == "false" or raw_reason.strip().lower() not in {"", "none", "null"}:
            candidate_filtered += 1
        if str(row.get("rspc_action_selected_name", "") or "") == "anchor" or str(row.get("source", "") or "").startswith("anchor"):
            anchor_selected += 1
    score = (
        profile_jumps
        + profile_reversals
        + profile_dew_conflicts
        + profile_vpd_conflicts
        + target_action_mismatch
        + guardrail_rewrite
        + transition_bypass
        + candidate_filtered
        + anchor_selected
    )
    return {
        "profile_target_jump_count": int(profile_jumps),
        "profile_target_reversal_count": int(profile_reversals),
        "profile_dew_conflict_count": int(profile_dew_conflicts),
        "profile_vpd_conflict_count": int(profile_vpd_conflicts),
        "target_action_mismatch_count": int(target_action_mismatch),
        "guardrail_rewrite_count": int(guardrail_rewrite),
        "transition_gate_bypass_count": int(transition_bypass),
        "candidate_filtered_count": int(candidate_filtered),
        "anchor_selected_count": int(anchor_selected),
        "guardrail_reason_counts": dict(sorted(reason_counts.items())),
        "score": int(score),
    }


def _sequence_history_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    max_run = 0
    current_run = 0
    high_pressure_steps = 0
    previous_conflict = False
    for row in rows:
        conflict = (
            _num(row.get("u_ventilation")) >= 0.85 and _num(row.get("u_screen")) >= 0.30
        ) or _truthy(row.get("transition_gate_applied")) or _truthy(row.get("transition_gate_bypassed"))
        if conflict:
            high_pressure_steps += 1
            current_run = current_run + 1 if previous_conflict else 1
        else:
            current_run = 0
        max_run = max(max_run, current_run)
        previous_conflict = conflict
    score = high_pressure_steps + max_run
    return {
        "high_pressure_steps": int(high_pressure_steps),
        "max_consecutive_high_pressure_steps": int(max_run),
        "score": int(score),
    }


def _score_scenario(
    *,
    original_window: Sequence[Mapping[str, Any]],
    ppo_window: Sequence[Mapping[str, Any]],
    gated_window: Sequence[Mapping[str, Any]],
    pending: bool,
) -> dict[str, int]:
    target_window = gated_window if gated_window else original_window
    action = _action_channel_metrics(target_window)
    boundary = _boundary_metrics(target_window)
    ppo_boundary = _boundary_metrics(ppo_window)
    profile_guardrail = _profile_guardrail_metrics(target_window)
    sequence = _sequence_history_metrics(target_window)
    simulator_boundary_score = min(int(boundary["score"]), int(ppo_boundary["score"])) if ppo_window else 0
    if pending:
        simulator_boundary_score = 0
    return {
        "action_channel_induced": int(action["score"]),
        "state_trajectory_induced": int(boundary["score"]),
        "profile_guardrail_conflict": int(profile_guardrail["score"]),
        "simulator_boundary_sensitive": int(simulator_boundary_score),
        "sequence_history_sensitive": int(sequence["score"]),
    }


def _dominant(scores: Mapping[str, int]) -> tuple[str, bool]:
    ordered = sorted(scores.items(), key=lambda item: (-int(item[1]), item[0]))
    if not ordered or int(ordered[0][1]) <= 0:
        return "unknown", False
    clear = len(ordered) == 1 or int(ordered[0][1]) >= max(1, int(ordered[1][1])) * 1.10
    return ordered[0][0], clear


def _scenario_report(
    *,
    scenario_id: str,
    original_trace_dir: Path,
    v43_trace_dir: Path,
    window_steps: int,
) -> dict[str, Any]:
    original_rows = _read_rows(_trace_path(original_trace_dir, scenario_id, "llm_rspc_v2"))
    ppo_rows = _read_rows(_trace_path(original_trace_dir, scenario_id, "ppo"))
    gated_rows = _read_rows(_trace_path(v43_trace_dir, scenario_id, "llm_rspc_v2"))
    original_failure_step = _runtime_failure_step(original_rows)
    gated_failure_step = _runtime_failure_step(gated_rows)
    pending = not bool(gated_rows)
    original_window = _window(original_rows, center_step=original_failure_step, window_steps=window_steps)
    gated_window = _window(gated_rows, center_step=gated_failure_step, window_steps=window_steps)
    ppo_alignment_step = gated_failure_step if gated_failure_step is not None else original_failure_step
    ppo_window = _window(ppo_rows, center_step=ppo_alignment_step, window_steps=window_steps)
    original_aligned_to_gated = _window(original_rows, center_step=gated_failure_step, window_steps=window_steps) if gated_failure_step is not None else []
    target_window = gated_window if gated_window else original_window
    scores = _score_scenario(
        original_window=original_window,
        ppo_window=ppo_window,
        gated_window=gated_window,
        pending=pending,
    )
    dominant, clear = _dominant(scores)
    if pending:
        dominant = "pending_trace_not_evaluated"
        clear = False
    return {
        "scenario_id": scenario_id,
        "status": "pending_trace_not_evaluated" if pending else "evaluated",
        "original_trace_path": _rel(_trace_path(original_trace_dir, scenario_id, "llm_rspc_v2")),
        "ppo_trace_path": _rel(_trace_path(original_trace_dir, scenario_id, "ppo")),
        "v43_trace_path": _rel(_trace_path(v43_trace_dir, scenario_id, "llm_rspc_v2")),
        "original_trace_exists": bool(original_rows),
        "ppo_trace_exists": bool(ppo_rows),
        "v43_trace_exists": bool(gated_rows),
        "original_failure_step": original_failure_step,
        "v43_failure_step": gated_failure_step,
        "v43_failed_earlier_than_original": bool(
            gated_failure_step is not None and original_failure_step is not None and gated_failure_step < original_failure_step
        ),
        "windows": {
            "original_pre_failure": _window_meta(original_rows, center_step=original_failure_step, window_steps=window_steps),
            "v43_pre_failure": _window_meta(gated_rows, center_step=gated_failure_step, window_steps=window_steps),
            "ppo_aligned": _window_meta(ppo_rows, center_step=ppo_alignment_step, window_steps=window_steps),
            "original_aligned_to_v43_failure": _window_meta(
                original_rows,
                center_step=gated_failure_step,
                window_steps=window_steps,
            )
            if gated_failure_step is not None
            else {"row_count": 0, "window_start_step": None, "window_end_step": None, "short_window": True},
        },
        "original_pre_failure_metrics": {
            "action_channel": _action_channel_metrics(original_window),
            "boundary": _boundary_metrics(original_window),
            "profile_guardrail": _profile_guardrail_metrics(original_window),
            "sequence_history": _sequence_history_metrics(original_window),
        },
        "v43_pre_failure_metrics": {
            "action_channel": _action_channel_metrics(gated_window),
            "boundary": _boundary_metrics(gated_window),
            "profile_guardrail": _profile_guardrail_metrics(gated_window),
            "sequence_history": _sequence_history_metrics(gated_window),
        },
        "ppo_aligned_metrics": {
            "action_channel": _action_channel_metrics(ppo_window),
            "boundary": _boundary_metrics(ppo_window),
        },
        "original_aligned_to_v43_metrics": {
            "action_channel": _action_channel_metrics(original_aligned_to_gated),
            "boundary": _boundary_metrics(original_aligned_to_gated),
            "profile_guardrail": _profile_guardrail_metrics(original_aligned_to_gated),
            "sequence_history": _sequence_history_metrics(original_aligned_to_gated),
        },
        "causal_scores": scores,
        "dominant_attribution": dominant,
        "dominant_attribution_clear": bool(clear),
        "evaluated_row_count": len(target_window),
    }


def _aggregate_scores(scenarios: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for scenario in scenarios:
        if scenario.get("status") != "evaluated":
            continue
        for key, value in (scenario.get("causal_scores", {}) or {}).items():
            totals[str(key)] += int(value)
    return dict(sorted(totals.items()))


def _next_action(dominant: str, clear: bool) -> str:
    if not clear:
        return "additional_runtime_profile_candidate_instrumentation"
    mapping = {
        "action_channel_induced": "channel_specific_rate_guard_design",
        "state_trajectory_induced": "profile_trajectory_fix_plan",
        "profile_guardrail_conflict": "candidate_guardrail_compatibility_scoring_plan",
        "simulator_boundary_sensitive": "numerical_stress_test_design",
        "sequence_history_sensitive": "minimal_sequence_counterfactual_design",
    }
    return mapping.get(dominant, "additional_runtime_profile_candidate_instrumentation")


def build_causal_attribution(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v43_trace_dir: str | Path = V43_TRACE_DIR,
    v40_attribution: Mapping[str, Any] | None = None,
    v43_result: Mapping[str, Any] | None = None,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    window_steps: int = 120,
) -> dict[str, Any]:
    original_root = _resolve(original_trace_dir)
    v43_root = _resolve(v43_trace_dir)
    scenario_reports = [
        _scenario_report(
            scenario_id=scenario_id,
            original_trace_dir=original_root,
            v43_trace_dir=v43_root,
            window_steps=window_steps,
        )
        for scenario_id in failure_scenarios
    ]
    aggregate = _aggregate_scores(scenario_reports)
    dominant, clear = _dominant(aggregate)
    pending = [item["scenario_id"] for item in scenario_reports if item.get("status") != "evaluated"]
    if pending and not any(item.get("status") == "evaluated" for item in scenario_reports):
        dominant = "pending_trace_not_evaluated"
        clear = False
    return {
        "schema_version": "cvodes_failure_causal_attribution_20260531_v44",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "offline cvodes causal attribution",
        },
        "failure_scenarios": list(failure_scenarios),
        "window_steps": int(window_steps),
        "source_v40_schema": (v40_attribution or {}).get("schema_version", ""),
        "source_v43_schema": (v43_result or {}).get("schema_version", ""),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "online_llm_called": False,
        "new_rollout_run": False,
        "scenario_reports": scenario_reports,
        "pending_trace_scenarios": pending,
        "aggregate_causal_scores": aggregate,
        "dominant_attribution": dominant,
        "dominant_attribution_clear": bool(clear),
        "failure_taxonomy": sorted(
            set(
                [dominant]
                + (["pending_trace_not_evaluated"] if pending else [])
                + (["v43_transition_gate_failed_earlier"] if any(s.get("v43_failed_earlier_than_original") for s in scenario_reports) else [])
            )
        ),
        "next_action": _next_action(dominant, clear),
    }


def build_profile_guardrail_compatibility(causal: Mapping[str, Any]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for scenario in causal.get("scenario_reports", []) or []:
        v43_metrics = ((scenario.get("v43_pre_failure_metrics", {}) or {}).get("profile_guardrail", {}) or {})
        original_metrics = ((scenario.get("original_pre_failure_metrics", {}) or {}).get("profile_guardrail", {}) or {})
        status = scenario.get("status")
        conflict_score = int(v43_metrics.get("score", 0) if status == "evaluated" else original_metrics.get("score", 0))
        reports.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "status": status,
                "profile_guardrail_conflict_score": conflict_score,
                "v43_profile_guardrail_metrics": v43_metrics,
                "original_profile_guardrail_metrics": original_metrics,
                "compatibility_decision": "pending_trace_not_evaluated"
                if status != "evaluated"
                else ("incompatible_requires_scoring_plan" if conflict_score > 0 else "no_profile_guardrail_conflict_detected"),
            }
        )
    incompatible = any(item["compatibility_decision"] == "incompatible_requires_scoring_plan" for item in reports)
    return {
        "schema_version": "profile_guardrail_compatibility_audit_20260531_v44",
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "scenario_reports": reports,
        "profile_guardrail_compatibility_issue_detected": bool(incompatible),
        "next_action": "candidate_guardrail_compatibility_scoring_plan"
        if incompatible
        else "additional_runtime_profile_candidate_instrumentation",
    }


def build_action_counterfactual_design(causal: Mapping[str, Any]) -> dict[str, Any]:
    designs: list[dict[str, Any]] = []
    for scenario in causal.get("scenario_reports", []) or []:
        if scenario.get("status") != "evaluated":
            designs.append(
                {
                    "scenario_id": scenario.get("scenario_id"),
                    "status": "pending_trace_not_evaluated",
                    "counterfactuals": [],
                }
            )
            continue
        action_metrics = ((scenario.get("v43_pre_failure_metrics", {}) or {}).get("action_channel", {}) or {})
        jump_counts = Counter(action_metrics.get("large_delta_count_by_channel", {}) or {})
        reversal_counts = Counter(action_metrics.get("reversal_count_by_channel", {}) or {})
        combined = Counter()
        for key, value in jump_counts.items():
            combined[key] += int(value)
        for key, value in reversal_counts.items():
            combined[key] += int(value)
        top_channels = [name.replace("u_", "") for name, _ in combined.most_common(3)]
        if not top_channels:
            top_channels = ["ventilation", "screen", "shading"]
        designs.append(
            {
                "scenario_id": scenario.get("scenario_id"),
                "status": "design_only_not_executed",
                "failure_step": scenario.get("v43_failure_step"),
                "target_channels": top_channels,
                "counterfactuals": [
                    "llm_state_with_ppo_matched_action",
                    "llm_state_with_single_channel_replaced_by_ppo",
                    "llm_state_with_profile_preserved_but_guardrail_compatible_candidate",
                    "short_sequence_minimal_trigger_search",
                ],
                "execution_authorized": False,
            }
        )
    return {
        "schema_version": "action_channel_counterfactual_design_20260531_v44",
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "counterfactual_execution_authorized": False,
        "scenario_designs": designs,
        "next_action": "await_explicit_counterfactual_execution_authorization_after_v44_review",
    }


def build_readiness(causal: Mapping[str, Any], compatibility: Mapping[str, Any], counterfactual: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "metadata_replay_readiness_checklist_20260531_v44",
        "stage": "v44_cvodes_failure_causal_attribution",
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "default_llm_rspc_v2_changed": False,
        "online_llm_called": False,
        "new_rollout_run": False,
        "causal_attribution_complete": True,
        "dominant_attribution": causal.get("dominant_attribution"),
        "dominant_attribution_clear": bool(causal.get("dominant_attribution_clear", False)),
        "pending_trace_scenarios": causal.get("pending_trace_scenarios", []),
        "profile_guardrail_compatibility_issue_detected": bool(
            compatibility.get("profile_guardrail_compatibility_issue_detected", False)
        ),
        "counterfactual_execution_authorized": bool(counterfactual.get("counterfactual_execution_authorized", False)),
        "failure_taxonomy": causal.get("failure_taxonomy", []),
        "next_action": causal.get("next_action", "additional_runtime_profile_candidate_instrumentation"),
    }


def _write_json_md(path_json: Path, path_md: Path, payload: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_md.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    lines = [
        f"# {title}",
        "",
        f"- Controlled replay allowed: `{payload.get('controlled_replay_allowed', False)}`",
        f"- Performance claim allowed: `{payload.get('performance_claim_allowed', False)}`",
        f"- Promotion evidence: `{payload.get('promotion_evidence', False)}`",
        f"- Next action: `{payload.get('next_action', '')}`",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        "```",
        "",
    ]
    path_md.write_text("\n".join(lines), encoding="utf-8")


def write_all(
    *,
    causal_path: Path = CAUSAL_JSON,
    causal_md_path: Path = CAUSAL_MD,
    compatibility_path: Path = COMPAT_JSON,
    compatibility_md_path: Path = COMPAT_MD,
    counterfactual_path: Path = COUNTERFACTUAL_JSON,
    counterfactual_md_path: Path = COUNTERFACTUAL_MD,
    readiness_path: Path = READINESS_JSON,
    readiness_md_path: Path = READINESS_MD,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v43_trace_dir: str | Path = V43_TRACE_DIR,
    v40_json: str | Path = V40_ATTRIBUTION_JSON,
    v43_json: str | Path = V43_RESULT_JSON,
    window_steps: int = 120,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    causal = build_causal_attribution(
        original_trace_dir=original_trace_dir,
        v43_trace_dir=v43_trace_dir,
        v40_attribution=_load_json(v40_json),
        v43_result=_load_json(v43_json),
        window_steps=window_steps,
        failure_scenarios=failure_scenarios,
    )
    compatibility = build_profile_guardrail_compatibility(causal)
    counterfactual = build_action_counterfactual_design(causal)
    readiness = build_readiness(causal, compatibility, counterfactual)
    _write_json_md(causal_path, causal_md_path, causal, "v44 CVODES Failure Causal Attribution")
    _write_json_md(compatibility_path, compatibility_md_path, compatibility, "v44 Profile Guardrail Compatibility Audit")
    _write_json_md(counterfactual_path, counterfactual_md_path, counterfactual, "v44 Action Channel Counterfactual Design")
    _write_json_md(readiness_path, readiness_md_path, readiness, "v44 Metadata Replay Readiness Checklist")
    return {
        "causal": causal,
        "compatibility": compatibility,
        "counterfactual": counterfactual,
        "readiness": readiness,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-trace-dir", default=_rel(ORIGINAL_TRACE_DIR))
    parser.add_argument("--v43-trace-dir", default=_rel(V43_TRACE_DIR))
    parser.add_argument("--v40-json", default=_rel(V40_ATTRIBUTION_JSON))
    parser.add_argument("--v43-json", default=_rel(V43_RESULT_JSON))
    parser.add_argument("--window-steps", type=int, default=120)
    parser.add_argument("--failure-scenario", action="append", default=[])
    parser.add_argument("--write-all", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scenarios = args.failure_scenario or list(FAILURE_SCENARIOS)
    result = write_all(
        original_trace_dir=args.original_trace_dir,
        v43_trace_dir=args.v43_trace_dir,
        v40_json=args.v40_json,
        v43_json=args.v43_json,
        window_steps=int(args.window_steps),
        failure_scenarios=scenarios,
    )
    if not args.write_all:
        print(json.dumps(result["readiness"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
