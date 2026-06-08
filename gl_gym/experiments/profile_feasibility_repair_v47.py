"""Build v47 profile-feasibility repair and hard-safety veto artifacts.

This audit is offline-only. It repairs profile targets on an in-memory row copy
and re-ranks existing candidate JSON with a hard-safety veto. It does not run
rollout, call online LLMs, mutate the default controller, authorize controlled
replay, or make performance claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (
    AUDIT_DIR,
    FAILURE_SCENARIOS,
    ORIGINAL_TRACE_DIR,
    V43_TRACE_DIR,
    WEIGHTS,
    _aggregate_reports,
    _candidate_issue,
    _candidate_penalty_terms,
    _candidate_score,
    _discover_stable_scenarios,
    _load_json,
    _num,
    _parse_candidates,
    _rel,
    _row_dry_risk,
    _row_original_issue,
    _runtime_failure_step,
    _trace_path,
    _truthy,
    _read_rows,
    _window,
)
from gl_gym.experiments.diagnose_ppo_vs_llm import derive_candidate_guardrail_compatibility


V45_AUDIT_JSON = AUDIT_DIR / "profile_candidate_guardrail_compatibility_audit_20260531_v45.json"
V46_COMPARISON_JSON = AUDIT_DIR / "candidate_guardrail_shadow_scoring_comparison_20260531_v46.json"

AUDIT_JSON = AUDIT_DIR / "profile_trajectory_feasibility_audit_20260531_v47.json"
AUDIT_MD = AUDIT_DIR / "profile_trajectory_feasibility_audit_20260531_v47.md"
COMPARISON_JSON = AUDIT_DIR / "profile_feasibility_shadow_scoring_comparison_20260531_v47.json"
COMPARISON_MD = AUDIT_DIR / "profile_feasibility_shadow_scoring_comparison_20260531_v47.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260531_v47.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260531_v47.md"


DEFAULTS = {
    "fallback_co2_min_rad": 120.0,
    "fallback_co2_max_vent": 0.20,
    "dry_rh_on": 55.0,
    "dry_vpd_on": 1.20,
    "dry_target_rh_floor": 70.0,
    "dry_temp_target_cap": 20.0,
    "rh_preemptive_threshold": 86.0,
    "rh_control_limit": 90.0,
    "rh_target_preemptive_cap": 80.0,
    "rh_target_high_cap": 76.0,
    "rh_target_extreme_cap": 72.0,
}


def _reason_text(row: Mapping[str, Any]) -> str:
    fields = (
        "tomato_safety_v2_reasons",
        "profile_scorer_safety_gate_reason",
        "profile_rspc_shadow_safety_gate_reason",
        "rspc_action_post_shape_safety_gate_reason",
    )
    return " ".join(str(row.get(field, "") or "") for field in fields).lower()


def _max_vent(row: Mapping[str, Any]) -> float:
    return max(
        _num(row.get("u_ventilation")),
        _num(row.get("tomato_safety_v2_vent_after")),
        _num(row.get("tomato_safety_v2_vent_before")),
        _num(row.get("rspc_action_selected_vent")),
    )


def profile_feasibility_mode(row: Mapping[str, Any]) -> dict[str, Any]:
    """Classify whether profile intents are feasible under current safety state."""

    text = _reason_text(row)
    rh = _num(row.get("rh_air"), 70.0)
    temp = _num(row.get("temp_air"), 20.0)
    rad = _num(row.get("glob_rad"), 0.0)
    vpd = _num(row.get("vpd_air"), _num(row.get("vpd"), 0.0))
    dew_margin = min(_num(row.get("dew_margin_air"), 3.0), _num(row.get("canopy_dew_margin"), 3.0))
    canopy_margin = _num(row.get("canopy_dew_margin"), 3.0)
    tomato_vent_raise = (
        _truthy(row.get("tomato_safety_v2_applied"))
        and _num(row.get("tomato_safety_v2_vent_after")) > _num(row.get("tomato_safety_v2_vent_before")) + 0.05
    )
    high_temp_pressure = (
        temp >= 32.0
        or _num(row.get("temp_violation"), 0.0) > 0.0
        or "temp_high_gate" in text
        or "hot_temperature" in text
        or "temperature_high" in text
    )
    dew_canopy_pressure = (
        dew_margin < 1.0
        or canopy_margin < 1.0
        or _truthy(row.get("dew_risk"))
        or "dew" in text
        or "canopy" in text
    )
    rh_hard_pressure = rh >= DEFAULTS["rh_control_limit"] or "rh_high_gate" in text
    safety_vent_reason = tomato_vent_raise and any(
        token in text for token in ("hard", "dew", "canopy", "extreme", "hot_temperature", "rh_high")
    )
    vent_required = bool(high_temp_pressure or dew_canopy_pressure or rh_hard_pressure or safety_vent_reason)
    dry_side = bool(
        _row_dry_risk(row)
        or rh < DEFAULTS["dry_rh_on"]
        or vpd > DEFAULTS["dry_vpd_on"]
    )
    vent = _max_vent(row)
    co2_allowed = bool(
        not vent_required
        and vent <= DEFAULTS["fallback_co2_max_vent"]
        and rad >= DEFAULTS["fallback_co2_min_rad"]
    )
    humidity_allowed = bool(not vent_required and not dew_canopy_pressure and not rh_hard_pressure)
    return {
        "vent_required_by_safety": vent_required,
        "humidity_retention_allowed": humidity_allowed,
        "co2_enrichment_allowed": co2_allowed,
        "high_temp_pressure": bool(high_temp_pressure),
        "dew_canopy_pressure": bool(dew_canopy_pressure),
        "rh_hard_pressure": bool(rh_hard_pressure),
        "safety_vent_reason": bool(safety_vent_reason),
        "dry_side": bool(dry_side),
        "max_vent": float(vent),
        "rh_air": float(rh),
        "temp_air": float(temp),
        "vpd_air": float(vpd),
        "dew_margin": float(dew_margin),
        "glob_rad": float(rad),
    }


def repair_profile_targets(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an in-memory row copy with repaired target fields."""

    repaired = dict(row)
    mode = profile_feasibility_mode(row)
    target_temp = _num(row.get("target_temp"), _num(row.get("temp_air"), 20.0))
    target_rh = _num(row.get("target_rh"), 76.0)
    target_co2 = _num(row.get("target_co2"), 430.0)
    corrections: list[str] = []

    if mode["dew_canopy_pressure"] or mode["rh_hard_pressure"]:
        cap = DEFAULTS["rh_target_extreme_cap"]
    elif mode["vent_required_by_safety"]:
        cap = DEFAULTS["rh_target_high_cap"]
    elif mode["rh_air"] >= DEFAULTS["rh_preemptive_threshold"]:
        cap = DEFAULTS["rh_target_preemptive_cap"]
    else:
        cap = 88.0
    if target_rh > cap:
        target_rh = cap
        corrections.append("target_rh:safety_cap")

    if mode["dry_side"] and not mode["vent_required_by_safety"]:
        if target_rh < DEFAULTS["dry_target_rh_floor"]:
            target_rh = DEFAULTS["dry_target_rh_floor"]
            corrections.append("target_rh:dry_recovery_floor")
        if target_temp > DEFAULTS["dry_temp_target_cap"]:
            target_temp = DEFAULTS["dry_temp_target_cap"]
            corrections.append("target_temp:dry_recovery_cap")

    if not mode["co2_enrichment_allowed"] and target_co2 > 430.0:
        target_co2 = 430.0
        corrections.append("target_co2:vent_or_radiation_cap")

    if mode["vent_required_by_safety"] and target_rh >= 75.0:
        target_rh = min(target_rh, DEFAULTS["rh_target_high_cap"] - 1.0)
        corrections.append("target_rh:high_vent_compatibility_cap")

    repaired.update(
        {
            "target_temp": float(target_temp),
            "target_rh": float(target_rh),
            "target_co2": float(target_co2),
            "profile_repair_original_target_temp": _num(row.get("target_temp")),
            "profile_repair_original_target_rh": _num(row.get("target_rh")),
            "profile_repair_original_target_co2": _num(row.get("target_co2")),
            "profile_repair_corrections": ",".join(sorted(set(corrections))) if corrections else "none",
            "profile_feasibility_mode": json.dumps(mode, sort_keys=True),
        }
    )
    return repaired, {
        "mode": mode,
        "corrections": sorted(set(corrections)),
        "original": {
            "target_temp": _num(row.get("target_temp")),
            "target_rh": _num(row.get("target_rh")),
            "target_co2": _num(row.get("target_co2")),
        },
        "repaired": {
            "target_temp": float(target_temp),
            "target_rh": float(target_rh),
            "target_co2": float(target_co2),
        },
    }


def _adjust_terms_for_feasibility(terms: dict[str, Any], mode: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(terms)
    conflicts = list(out.get("profile_action_conflicts", []) or [])
    if mode.get("vent_required_by_safety") and out.get("dry_risk_high_vent_conflict"):
        out["dry_risk_high_vent_conflict"] = 0
        conflicts = [item for item in conflicts if item != "dry_risk_vs_high_vent"]
    out["profile_action_conflicts"] = conflicts
    out["profile_action_conflict_count"] = len(conflicts)
    return out


def score_candidate_with_feasibility(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    repaired, repair = repair_profile_targets(row)
    terms = _adjust_terms_for_feasibility(_candidate_penalty_terms(repaired, candidate), repair["mode"])
    original_score = _candidate_score(candidate)
    adjustment = (
        WEIGHTS["profile_action_conflict"] * terms["profile_action_conflict_count"]
        + WEIGHTS["dry_risk_high_vent_conflict"] * terms["dry_risk_high_vent_conflict"]
        + WEIGHTS["actuator_inconsistency"] * terms["actuator_inconsistency_count"]
        + WEIGHTS["expected_rewrite_field"] * terms["expected_rewrite_field_count"]
        + WEIGHTS["candidate_filter_reason"] * terms["candidate_filter_reason_count"]
        + WEIGHTS["anchor_or_previous_compatible_bonus"] * terms["anchor_or_previous_compatible_bonus"]
    )
    original_issue = _row_original_issue(row)
    hard_veto = bool(terms.get("hard_safety_rewrite_predicted") and not original_issue.get("hard_safety_rewrite"))
    return {
        "name": str(candidate.get("name") or "candidate"),
        "original_score": float(original_score),
        "adjustment": float(adjustment),
        "adjusted_score": float(original_score + adjustment),
        "selected_original": bool(candidate.get("selected", False)),
        "hard_safety_vetoed": hard_veto,
        "profile_repair": repair,
        **terms,
    }


def shadow_score_row_with_repair(row: Mapping[str, Any]) -> dict[str, Any]:
    candidates = _parse_candidates(row)
    repaired, repair = repair_profile_targets(row)
    if not candidates:
        return {
            "status": "candidate_metadata_missing",
            "step": int(_num(row.get("step"), -1)),
            "candidate_count": 0,
            "original_selected_name": "",
            "shadow_selected_name": "",
            "shadow_changed": False,
            "hard_safety_rewrite_preferred_new": False,
            "profile_repair": repair,
            "repaired_row_preview": {
                "target_temp": repaired.get("target_temp"),
                "target_rh": repaired.get("target_rh"),
                "target_co2": repaired.get("target_co2"),
            },
            "original_selected": None,
            "shadow_selected": None,
        }
    scored = [score_candidate_with_feasibility(row, candidate) for candidate in candidates]
    original_selected = next((item for item in scored if item["selected_original"]), None)
    if original_selected is None:
        original_selected = min(scored, key=lambda item: item["original_score"])
    original_hard = bool(original_selected.get("hard_safety_rewrite_predicted", False))
    for item in scored:
        if item.get("hard_safety_rewrite_predicted") and not original_hard:
            item["hard_safety_vetoed"] = True
    eligible = [item for item in scored if not item.get("hard_safety_vetoed")]
    shadow_selected = min(eligible, key=lambda item: item["adjusted_score"]) if eligible else original_selected
    hard_new = bool(
        shadow_selected["hard_safety_rewrite_predicted"]
        and not bool(original_selected.get("hard_safety_rewrite_predicted", False))
    )
    return {
        "status": "scored",
        "step": int(_num(row.get("step"), -1)),
        "candidate_count": len(scored),
        "original_selected_name": original_selected["name"],
        "shadow_selected_name": shadow_selected["name"],
        "shadow_changed": original_selected["name"] != shadow_selected["name"],
        "hard_safety_rewrite_preferred_new": hard_new,
        "profile_repair": repair,
        "original_selected": original_selected,
        "shadow_selected": shadow_selected,
        "top_candidates": sorted(scored, key=lambda item: (bool(item.get("hard_safety_vetoed")), item["adjusted_score"]))[:5],
    }


def summarize_repaired_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored_rows = [shadow_score_row_with_repair(row) for row in rows]
    valid = [item for item in scored_rows if item["status"] == "scored"]
    missing = [item for item in scored_rows if item["status"] == "candidate_metadata_missing"]
    original_counts: Counter[str] = Counter()
    repaired_counts: Counter[str] = Counter()
    correction_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    hard_new = 0
    changed = 0
    for row, scored in zip(rows, scored_rows):
        original_issue = _row_original_issue(row)
        repaired_issue = _candidate_issue(scored.get("shadow_selected") if scored["status"] == "scored" else None)
        for key, value in original_issue.items():
            if value:
                original_counts[key] += 1
        for key, value in repaired_issue.items():
            if value:
                repaired_counts[key] += 1
        repair = scored.get("profile_repair", {}) or {}
        mode = repair.get("mode", {}) or {}
        for key in ("vent_required_by_safety", "humidity_retention_allowed", "co2_enrichment_allowed", "dry_side"):
            if mode.get(key):
                mode_counts[key] += 1
        for correction in repair.get("corrections", []) or []:
            correction_counts[correction] += 1
        if scored["status"] == "scored":
            changed += int(bool(scored.get("shadow_changed")))
            hard_new += int(bool(scored.get("hard_safety_rewrite_preferred_new")))
    return {
        "row_count": len(rows),
        "scored_row_count": len(valid),
        "candidate_metadata_missing_steps": len(missing),
        "shadow_changed_steps": int(changed),
        "shadow_change_rate": float(changed / len(valid)) if valid else 0.0,
        "hard_safety_rewrite_preferred_new_steps": int(hard_new),
        "profile_repair_correction_counts": dict(sorted(correction_counts.items())),
        "profile_feasibility_mode_counts": dict(sorted(mode_counts.items())),
        "original": {
            "candidate_guardrail_issue_steps": int(original_counts["candidate_guardrail_issue"]),
            "major_or_hard_rewrite_predicted_steps": int(original_counts["major_or_hard_rewrite"]),
            "profile_action_conflict_steps": int(original_counts["profile_action_conflict"]),
            "target_rh_vs_high_vent_steps": int(original_counts["target_rh_vs_high_vent"]),
            "dry_risk_vs_high_vent_steps": int(original_counts["dry_risk_vs_high_vent"]),
            "hard_safety_rewrite_preferred_steps": int(original_counts["hard_safety_rewrite"]),
        },
        "repaired": {
            "candidate_guardrail_issue_steps": int(repaired_counts["candidate_guardrail_issue"]),
            "major_or_hard_rewrite_predicted_steps": int(repaired_counts["major_or_hard_rewrite"]),
            "profile_action_conflict_steps": int(repaired_counts["profile_action_conflict"]),
            "target_rh_vs_high_vent_steps": int(repaired_counts["target_rh_vs_high_vent"]),
            "dry_risk_vs_high_vent_steps": int(repaired_counts["dry_risk_vs_high_vent"]),
            "hard_safety_rewrite_preferred_steps": int(repaired_counts["hard_safety_rewrite"]),
        },
        "top_shadow_changes": [
            {
                "step": item["step"],
                "original_selected_name": item["original_selected_name"],
                "shadow_selected_name": item["shadow_selected_name"],
                "profile_repair": item.get("profile_repair"),
                "original_selected": item.get("original_selected"),
                "shadow_selected": item.get("shadow_selected"),
            }
            for item in valid
            if item["shadow_changed"]
        ][:25],
    }


def _scenario_report(
    scenario_id: str,
    *,
    trace_dir: str | Path,
    window_steps: int,
    mode: str,
) -> dict[str, Any]:
    trace = _trace_path(trace_dir, scenario_id)
    rows = _read_rows(trace)
    failure_step = _runtime_failure_step(rows)
    window = _window(rows, center_step=failure_step, window_steps=window_steps)
    summary = summarize_repaired_rows(window)
    return {
        "scenario_id": scenario_id,
        "mode": mode,
        "trace": _rel(trace),
        "trace_exists": bool(rows),
        "failure_step": failure_step,
        "window_steps": window_steps,
        "window_row_count": len(window),
        **summary,
    }


def _aggregate_repaired_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = _aggregate_reports(reports)
    correction_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    repaired = Counter()
    for report in reports:
        correction_counts.update(report.get("profile_repair_correction_counts", {}) or {})
        mode_counts.update(report.get("profile_feasibility_mode_counts", {}) or {})
        for key, value in (report.get("repaired", {}) or {}).items():
            repaired[key] += int(value or 0)
    base["profile_repair_correction_counts"] = dict(sorted(correction_counts.items()))
    base["profile_feasibility_mode_counts"] = dict(sorted(mode_counts.items()))
    base["repaired"] = dict(sorted(repaired.items()))
    base.pop("shadow", None)
    return base


def _accepted(main: Mapping[str, Any], stable: Mapping[str, Any]) -> bool:
    original = main.get("original", {}) or {}
    repaired = main.get("repaired", {}) or {}
    return bool(
        repaired.get("profile_action_conflict_steps", 0) < original.get("profile_action_conflict_steps", 0)
        and repaired.get("candidate_guardrail_issue_steps", 0) < original.get("candidate_guardrail_issue_steps", 0)
        and repaired.get("major_or_hard_rewrite_predicted_steps", 0) <= original.get("major_or_hard_rewrite_predicted_steps", 0)
        and stable.get("shadow_change_rate", 1.0) <= 0.35
        and stable.get("hard_safety_rewrite_preferred_new_steps", 0) == 0
    )


def build_profile_audit(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    window_steps: int = 120,
) -> dict[str, Any]:
    stable_scenarios = _discover_stable_scenarios(original_trace_dir, failure_scenarios)
    reports = [
        _scenario_report(scenario, trace_dir=original_trace_dir, window_steps=window_steps, mode="main_failure_pre_window")
        for scenario in failure_scenarios
    ] + [
        _scenario_report(scenario, trace_dir=original_trace_dir, window_steps=window_steps, mode="stable_control_pre_window")
        for scenario in stable_scenarios
    ]
    return {
        "schema_version": "profile_trajectory_feasibility_audit_20260531_v47",
        "stage": "v47_profile_feasibility_repair",
        "online_llm_called": False,
        "new_rollout_run": False,
        "default_llm_rspc_v2_changed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "boundary_defaults": dict(DEFAULTS),
        "failure_scenarios": list(failure_scenarios),
        "stable_control_scenarios": stable_scenarios,
        "scenario_reports": reports,
        "aggregate": _aggregate_repaired_reports(reports),
    }


def build_comparison(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v43_trace_dir: str | Path = V43_TRACE_DIR,
    v45_audit_json: str | Path = V45_AUDIT_JSON,
    v46_comparison_json: str | Path = V46_COMPARISON_JSON,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    window_steps: int = 120,
) -> dict[str, Any]:
    v45 = _load_json(v45_audit_json)
    v46 = _load_json(v46_comparison_json)
    stable_scenarios = _discover_stable_scenarios(original_trace_dir, failure_scenarios)
    main_reports = [
        _scenario_report(scenario, trace_dir=original_trace_dir, window_steps=window_steps, mode="main_failure_pre_window")
        for scenario in failure_scenarios
    ]
    stable_reports = [
        _scenario_report(scenario, trace_dir=original_trace_dir, window_steps=window_steps, mode="stable_control_pre_window")
        for scenario in stable_scenarios
    ]
    pending = []
    v43_reports = []
    for scenario in failure_scenarios:
        trace = _trace_path(v43_trace_dir, scenario)
        if trace.exists():
            v43_reports.append(_scenario_report(scenario, trace_dir=v43_trace_dir, window_steps=window_steps, mode="v43_existing_trace"))
        else:
            pending.append(scenario)
    main = _aggregate_repaired_reports(main_reports)
    stable = _aggregate_repaired_reports(stable_reports)
    v43 = _aggregate_repaired_reports(v43_reports)
    accepted = _accepted(main, stable)
    return {
        "schema_version": "profile_feasibility_shadow_scoring_comparison_20260531_v47",
        "mainline_alignment": {
            "impact_layer": "Evaluation / Safety Boundary / Metadata Trace",
            "default_llm_rspc_v2_changed": False,
            "mode": "offline_profile_feasibility_shadow_scoring",
        },
        "source_v45_schema": v45.get("schema_version", ""),
        "source_v45_top_issue_type": v45.get("top_issue_type", ""),
        "source_v46_schema": v46.get("schema_version", ""),
        "source_v46_acceptance_pass": bool((v46.get("acceptance", {}) or {}).get("shadow_scoring_acceptance_pass", False)),
        "online_llm_called": False,
        "new_rollout_run": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "failure_scenarios": list(failure_scenarios),
        "stable_control_scenarios": stable_scenarios,
        "pending_v43_trace_scenarios": pending,
        "main_failure_reports": main_reports,
        "stable_control_reports": stable_reports,
        "v43_existing_trace_reports": v43_reports,
        "main_failure_aggregate": main,
        "stable_control_aggregate": stable,
        "v43_existing_trace_aggregate": v43,
        "acceptance": {
            "profile_feasibility_acceptance_pass": accepted,
            "profile_action_conflict_steps_repaired_lt_original": main.get("repaired", {}).get("profile_action_conflict_steps", 0)
            < main.get("original", {}).get("profile_action_conflict_steps", 0),
            "candidate_guardrail_issue_steps_repaired_lt_original": main.get("repaired", {}).get("candidate_guardrail_issue_steps", 0)
            < main.get("original", {}).get("candidate_guardrail_issue_steps", 0),
            "major_or_hard_rewrite_predicted_steps_repaired_lte_original": main.get("repaired", {}).get("major_or_hard_rewrite_predicted_steps", 0)
            <= main.get("original", {}).get("major_or_hard_rewrite_predicted_steps", 0),
            "stable_shadow_change_rate_lte_0_35": stable.get("shadow_change_rate", 1.0) <= 0.35,
            "stable_hard_safety_rewrite_preferred_new_zero": stable.get("hard_safety_rewrite_preferred_new_steps", 0) == 0,
        },
        "next_action": "minimal_profile_feasibility_gate_shadow_rollout_plan"
        if accepted
        else "profile_metadata_instrumentation_or_template_fix_plan",
    }


def build_readiness(comparison: Mapping[str, Any]) -> dict[str, Any]:
    stable = comparison.get("stable_control_aggregate", {}) or {}
    return {
        "schema_version": "metadata_replay_readiness_checklist_20260531_v47",
        "stage": "v47_profile_feasibility_repair",
        "profile_feasibility_shadow_complete": True,
        "profile_feasibility_acceptance_pass": bool(
            (comparison.get("acceptance", {}) or {}).get("profile_feasibility_acceptance_pass", False)
        ),
        "default_llm_rspc_v2_changed": False,
        "online_llm_called": False,
        "new_rollout_run": False,
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "stable_shadow_change_rate": float(stable.get("shadow_change_rate", 0.0) or 0.0),
        "stable_hard_safety_rewrite_preferred_new_steps": int(stable.get("hard_safety_rewrite_preferred_new_steps", 0) or 0),
        "pending_v43_trace_scenarios": list(comparison.get("pending_v43_trace_scenarios", [])),
        "patch_execution_authorized": False,
        "next_action": comparison.get("next_action", "profile_metadata_instrumentation_or_template_fix_plan"),
    }


def _write_json_md(path_json: Path, path_md: Path, data: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    lines = [f"# {title}", ""]
    for key in (
        "schema_version",
        "stage",
        "source_v45_top_issue_type",
        "next_action",
    ):
        if key in data:
            lines.append(f"- `{key}`: `{data[key]}`")
    if "acceptance" in data:
        lines.append(f"- `profile_feasibility_acceptance_pass`: `{data['acceptance'].get('profile_feasibility_acceptance_pass')}`")
    for key in (
        "online_llm_called",
        "new_rollout_run",
        "controlled_replay_allowed",
        "controlled_replay_execution_allowed",
        "metadata_replay_execution_allowed",
        "performance_claim_allowed",
        "promotion_evidence",
    ):
        if key in data:
            lines.append(f"- `{key}`: `{str(data[key]).lower()}`")
    for key in ("aggregate", "main_failure_aggregate", "stable_control_aggregate", "v43_existing_trace_aggregate", "acceptance"):
        if key in data:
            lines.extend(["", f"## {key}", "```json", json.dumps(data[key], indent=2, sort_keys=True), "```"])
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v43_trace_dir: str | Path = V43_TRACE_DIR,
    v45_audit_json: str | Path = V45_AUDIT_JSON,
    v46_comparison_json: str | Path = V46_COMPARISON_JSON,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    window_steps: int = 120,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit = build_profile_audit(
        original_trace_dir=original_trace_dir,
        failure_scenarios=failure_scenarios,
        window_steps=window_steps,
    )
    comparison = build_comparison(
        original_trace_dir=original_trace_dir,
        v43_trace_dir=v43_trace_dir,
        v45_audit_json=v45_audit_json,
        v46_comparison_json=v46_comparison_json,
        failure_scenarios=failure_scenarios,
        window_steps=window_steps,
    )
    readiness = build_readiness(comparison)
    _write_json_md(AUDIT_JSON, AUDIT_MD, audit, "v47 Profile Trajectory Feasibility Audit")
    _write_json_md(COMPARISON_JSON, COMPARISON_MD, comparison, "v47 Profile Feasibility Shadow Scoring Comparison")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v47 Metadata Replay Readiness")
    return audit, comparison, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-trace-dir", type=str, default=str(ORIGINAL_TRACE_DIR))
    parser.add_argument("--v43-trace-dir", type=str, default=str(V43_TRACE_DIR))
    parser.add_argument("--v45-audit-json", type=str, default=str(V45_AUDIT_JSON))
    parser.add_argument("--v46-comparison-json", type=str, default=str(V46_COMPARISON_JSON))
    parser.add_argument("--failure-scenario", action="append", default=[])
    parser.add_argument("--window-steps", type=int, default=120)
    parser.add_argument("--write-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios = args.failure_scenario or list(FAILURE_SCENARIOS)
    audit, comparison, readiness = write_all(
        original_trace_dir=args.original_trace_dir,
        v43_trace_dir=args.v43_trace_dir,
        v45_audit_json=args.v45_audit_json,
        v46_comparison_json=args.v46_comparison_json,
        failure_scenarios=scenarios,
        window_steps=int(args.window_steps),
    )
    if not args.write_all:
        print(json.dumps({"audit": audit, "comparison": comparison, "readiness": readiness}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
