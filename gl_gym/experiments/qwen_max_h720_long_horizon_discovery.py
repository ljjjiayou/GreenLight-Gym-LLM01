"""Build v39 Qwen-Max H720 long-horizon discovery audits.

This script is intentionally audit-only. It consumes the authorized v38 H720
execution record and trace directory, then emits v39 diagnostic artifacts for
profile trajectory quality, candidate/action-pool gaps, control stability, and
next-step opportunity selection. It never authorizes controlled replay or
promotion evidence.
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

ACTION_FIELDS = ("u_heating", "u_ventilation", "u_co2", "u_lighting", "u_screen", "u_shading")
TARGET_FIELDS = ("target_temp", "target_rh", "target_co2")
CONTROLLED_CONTROLLERS = {"llm_rspc_v2_hot_dry_proposer_strict"}


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
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "applied"}


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


def _action_changed(row: Mapping[str, Any], prefix: str) -> bool:
    for name in ("heat", "vent", "screen", "shade"):
        before = _num(row.get(f"{prefix}_{name}_before"))
        after = _num(row.get(f"{prefix}_{name}_after"))
        if abs(after - before) > 1e-6:
            return True
    return False


def _append_example(examples: list[dict[str, Any]], *, step: int, issue: str, detail: str = "") -> None:
    if len(examples) < 20:
        examples.append({"step": step, "issue": issue, "detail": detail})


def _iter_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _trace_completeness(path: Path, horizon_steps: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "trace_path": str(path),
            "trace_exists": False,
            "row_count": 0,
            "last_step": None,
            "runtime_error_count": 0,
            "cvodes_failure_count": 0,
            "short_trace_before_horizon": True,
        }
    row_count = 0
    last_step: int | None = None
    runtime_error_count = 0
    cvodes_failure_count = 0
    for row in _iter_rows(path):
        row_count += 1
        step = int(_num(row.get("step"), row_count - 1))
        last_step = step
        error_text = " ".join(
            str(row.get(key, "") or "")
            for key in ("runtime_error", "runtime_error_type", "replan_reason")
        )
        if row.get("runtime_error") or row.get("runtime_error_type") or row.get("replan_reason") == "runtime_error":
            runtime_error_count += 1
            if "CVODES" in error_text or "CV_CONV_FAILURE" in error_text or "CvodesInterface" in error_text:
                cvodes_failure_count += 1
    return {
        "trace_path": str(path),
        "trace_exists": True,
        "row_count": row_count,
        "last_step": last_step,
        "runtime_error_count": runtime_error_count,
        "cvodes_failure_count": cvodes_failure_count,
        "short_trace_before_horizon": last_step is None or last_step < horizon_steps - 1,
    }


def _audit_llm_trace(path: Path, horizon_steps: int) -> dict[str, Any]:
    completeness = _trace_completeness(path, horizon_steps)
    metrics: Counter[str] = Counter()
    candidate_reasons: Counter[str] = Counter()
    profile_examples: list[dict[str, Any]] = []
    candidate_examples: list[dict[str, Any]] = []
    stability_examples: list[dict[str, Any]] = []
    missing_fields: set[str] = set()

    previous_targets: dict[str, float] = {}
    previous_target_signs: dict[str, int] = {}
    previous_actions: dict[str, float] = {}
    previous_action_signs: dict[str, int] = {}

    if not path.exists():
        return {
            "completeness": completeness,
            "metrics": dict(metrics),
            "candidate_reason_counts": {},
            "missing_fields": sorted(TARGET_FIELDS + ACTION_FIELDS),
            "profile_examples": profile_examples,
            "candidate_examples": candidate_examples,
            "stability_examples": stability_examples,
        }

    fieldnames: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = set(TARGET_FIELDS + ACTION_FIELDS)
        missing_fields.update(name for name in required if name not in fieldnames)
        for row_index, row in enumerate(reader):
            step = int(_num(row.get("step"), row_index))

            target_deltas: dict[str, float] = {}
            for target in TARGET_FIELDS:
                value = _num(row.get(target))
                if target in previous_targets:
                    delta = value - previous_targets[target]
                    target_deltas[target] = delta
                    sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
                    if sign and previous_target_signs.get(target, 0) and sign != previous_target_signs[target]:
                        metrics["profile_target_reversal_count"] += 1
                        _append_example(profile_examples, step=step, issue="profile_target_reversal", detail=target)
                    if sign:
                        previous_target_signs[target] = sign
                previous_targets[target] = value

            temp_delta = abs(_num(row.get("profile_selected_delta_target_temp"), target_deltas.get("target_temp", 0.0)))
            rh_delta = abs(_num(row.get("profile_selected_delta_target_rh"), target_deltas.get("target_rh", 0.0)))
            co2_delta = abs(_num(row.get("profile_selected_delta_target_co2"), target_deltas.get("target_co2", 0.0)))
            if temp_delta > 2.0:
                metrics["profile_temp_jump_count"] += 1
                _append_example(profile_examples, step=step, issue="profile_temp_jump", detail=f"{temp_delta:.3g}")
            if rh_delta > 8.0:
                metrics["profile_rh_jump_count"] += 1
                _append_example(profile_examples, step=step, issue="profile_rh_jump", detail=f"{rh_delta:.3g}")
            if co2_delta > 200.0:
                metrics["profile_co2_jump_count"] += 1
                _append_example(profile_examples, step=step, issue="profile_co2_jump", detail=f"{co2_delta:.3g}")

            target_co2 = _num(row.get("target_co2"))
            vent = _num(row.get("u_ventilation"))
            if target_co2 >= 800.0 and vent >= 0.30:
                metrics["profile_co2_vent_conflict_count"] += 1
                _append_example(profile_examples, step=step, issue="profile_co2_vent_conflict")
            if _truthy(row.get("dew_risk")) and (_num(row.get("target_rh")) >= 80.0 or _num(row.get("profile_selected_delta_target_rh")) > 0.0):
                metrics["profile_dew_conflict_count"] += 1
                _append_example(profile_examples, step=step, issue="profile_dew_conflict")
            if _num(row.get("vpd_high_excess")) > 0.0 and _num(row.get("profile_selected_delta_target_rh")) < 0.0:
                metrics["profile_vpd_conflict_count"] += 1
                _append_example(profile_examples, step=step, issue="profile_vpd_conflict")

            candidate_count = int(_num(row.get("rspc_action_candidate_count"), _num(row.get("rspc_action_actual_candidate_count"))))
            actual_candidate_count = int(_num(row.get("rspc_action_actual_candidate_count"), candidate_count))
            if candidate_count <= 0 or actual_candidate_count <= 0:
                metrics["candidate_missing_count"] += 1
                _append_example(candidate_examples, step=step, issue="candidate_missing")
            best_eligible = str(row.get("rspc_action_post_shape_best_eligible", "") or "").strip().lower()
            safety_reason = str(row.get("rspc_action_post_shape_safety_gate_reason", "") or "")
            if best_eligible == "false" or safety_reason:
                metrics["candidate_filtered_count"] += 1
                if safety_reason:
                    candidate_reasons[safety_reason] += 1
                _append_example(candidate_examples, step=step, issue="candidate_filtered", detail=safety_reason)
            if _truthy(row.get("rspc_action_post_shape_unsafe_conflict")) or _truthy(row.get("rspc_hot_dry_replay_unsafe_conflict")):
                metrics["candidate_unsafe_conflict_count"] += 1
                _append_example(candidate_examples, step=step, issue="candidate_unsafe_conflict")

            fallback_selected = str(row.get("selected_fallback_candidate", "") or "")
            if fallback_selected:
                metrics["fallback_selected_count"] += 1
                _append_example(candidate_examples, step=step, issue="fallback_selected", detail=fallback_selected)

            guardrail_rewrite = _truthy(row.get("tomato_safety_v2_applied")) or _action_changed(row, "tomato_safety_v2")
            if guardrail_rewrite:
                metrics["guardrail_rewrite_count"] += 1
                reason = str(row.get("tomato_safety_v2_reasons", "") or row.get("final_action_risk_reason_v2", "") or "")
                if reason:
                    candidate_reasons[f"guardrail:{reason}"] += 1
                _append_example(stability_examples, step=step, issue="guardrail_rewrite", detail=reason)

            if _truthy(row.get("final_action_would_fail_canopy_boundary_v2")) or _truthy(row.get("final_action_predicted_canopy_lt0_v2")):
                metrics["hard_safety_warning_count"] += 1
                _append_example(stability_examples, step=step, issue="hard_safety_warning")

            metrics["runtime_provenance_record_count"] += int(_num(row.get("post_guardrail_runtime_provenance_count")))
            metrics["runtime_reason_missing_count"] += int(_num(row.get("post_guardrail_runtime_reason_missing_count")))
            metrics["unknown_post_guardrail_rewrite_count"] += int(_num(row.get("unknown_post_guardrail_rewrite_count")))

            for action in ACTION_FIELDS:
                value = _num(row.get(action))
                if action in previous_actions:
                    delta = value - previous_actions[action]
                    if abs(delta) > 0.20:
                        metrics["large_action_delta_count"] += 1
                        _append_example(stability_examples, step=step, issue="large_action_delta", detail=action)
                    sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
                    if sign and previous_action_signs.get(action, 0) and sign != previous_action_signs[action]:
                        metrics["action_oscillation_count"] += 1
                        _append_example(stability_examples, step=step, issue="action_oscillation", detail=action)
                    if sign:
                        previous_action_signs[action] = sign
                previous_actions[action] = value

    metrics["trace_row_count"] = int(completeness.get("row_count", 0) or 0)
    metrics["trace_missing_field_count"] = len(missing_fields)
    return {
        "completeness": completeness,
        "metrics": dict(sorted(metrics.items())),
        "candidate_reason_counts": dict(sorted(candidate_reasons.items())),
        "missing_fields": sorted(missing_fields),
        "profile_examples": profile_examples,
        "candidate_examples": candidate_examples,
        "stability_examples": stability_examples,
        "field_count": len(fieldnames),
    }


def _scope_ok(execution_record: Mapping[str, Any]) -> bool:
    controllers = set(str(item) for item in execution_record.get("controllers", []) or [])
    if controllers & CONTROLLED_CONTROLLERS:
        return False
    for record in execution_record.get("planned_command_records", []) or []:
        if str(record.get("controller", "")) in CONTROLLED_CONTROLLERS:
            return False
        command = " ".join(str(part) for part in record.get("command", []) or [])
        if any(name in command for name in CONTROLLED_CONTROLLERS):
            return False
    return True


def _sum_metric(scenarios: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(int(((item.get("llm_audit", {}) or {}).get("metrics", {}) or {}).get(key, 0) or 0) for item in scenarios)


def _rank_opportunities(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenarios = list(report.get("scenario_reports", []) or [])
    totals = dict(report.get("metrics", {}) or {})
    opportunities = [
        {
            "area": "runtime_attribution",
            "priority_score": int(totals.get("trace_incomplete_count", 0) or 0) * 100
            + int(totals.get("default_runtime_error_count", 0) or 0) * 50
            + int(totals.get("cvodes_failure_count", 0) or 0) * 25,
            "next_action": "runtime_attribution_and_stability_guard_design",
        },
        {
            "area": "profile_trajectory",
            "priority_score": _sum_metric(scenarios, "profile_temp_jump_count")
            + _sum_metric(scenarios, "profile_rh_jump_count")
            + _sum_metric(scenarios, "profile_co2_jump_count")
            + _sum_metric(scenarios, "profile_target_reversal_count")
            + _sum_metric(scenarios, "profile_co2_vent_conflict_count")
            + _sum_metric(scenarios, "profile_dew_conflict_count")
            + _sum_metric(scenarios, "profile_vpd_conflict_count"),
            "next_action": "v40_profile_trajectory_generator_minimal_design",
        },
        {
            "area": "candidate_action_pool",
            "priority_score": _sum_metric(scenarios, "candidate_missing_count")
            + _sum_metric(scenarios, "candidate_filtered_count")
            + _sum_metric(scenarios, "candidate_unsafe_conflict_count")
            + _sum_metric(scenarios, "fallback_selected_count"),
            "next_action": "v40_candidate_archetype_pool_gap_repair_design",
        },
        {
            "area": "rate_guard_action_smoothing",
            "priority_score": _sum_metric(scenarios, "large_action_delta_count")
            + _sum_metric(scenarios, "action_oscillation_count")
            + _sum_metric(scenarios, "guardrail_rewrite_count"),
            "next_action": "v40_rate_guard_and_action_smoothing_design",
        },
    ]
    ranked = sorted(opportunities, key=lambda item: (-int(item["priority_score"]), str(item["area"])))
    if ranked and int(ranked[0]["priority_score"]) <= 0:
        return [
            {
                "area": "h1440_shrink_validation",
                "priority_score": 0,
                "next_action": "build_h1440_shrink_validation_plan",
            }
        ]
    return ranked


def build_report(*, execution_record: Mapping[str, Any], trace_dir: str | Path) -> dict[str, Any]:
    horizon = int(execution_record.get("horizon_steps", 720) or 720)
    trace_root = _resolve(trace_dir)
    scenario_reports: list[dict[str, Any]] = []
    for raw in execution_record.get("eligible_scenarios", []) or []:
        if not isinstance(raw, Mapping):
            continue
        sid = str(raw.get("scenario_id", ""))
        ppo_path = _trace_path_for(trace_root, sid, "ppo")
        llm_path = _trace_path_for(trace_root, sid, "llm_rspc_v2")
        scenario_reports.append(
            {
                "scenario_id": sid,
                "year": int(raw.get("year", 0)),
                "day": int(raw.get("day", 0)),
                "seed": int(raw.get("seed", 0)),
                "regime": str(raw.get("regime", "")),
                "source_tier": str(raw.get("source_tier", "")),
                "horizon_steps": horizon,
                "baseline_report": _trace_completeness(ppo_path, horizon),
                "llm_audit": _audit_llm_trace(llm_path, horizon),
            }
        )

    controlled_scope_ok = _scope_ok(execution_record)
    trace_incomplete_count = sum(
        1 for item in scenario_reports if bool((item["llm_audit"]["completeness"] or {}).get("short_trace_before_horizon", True))
    )
    missing_metadata_count = sum(
        int((item["llm_audit"].get("metrics", {}) or {}).get("trace_missing_field_count", 0) or 0) for item in scenario_reports
    )
    hard_safety_warning_count = _sum_metric(scenario_reports, "hard_safety_warning_count")
    failure_taxonomy: list[str] = []
    if not controlled_scope_ok:
        failure_taxonomy.append("controlled_controller_scope_violation")
    if trace_incomplete_count:
        failure_taxonomy.append("trace_incomplete_or_short")
    if missing_metadata_count:
        failure_taxonomy.append("metadata_missing")
    if hard_safety_warning_count:
        failure_taxonomy.append("hard_safety_warning_requires_review")
    if _sum_metric(scenario_reports, "runtime_reason_missing_count") or _sum_metric(scenario_reports, "unknown_post_guardrail_rewrite_count"):
        failure_taxonomy.append("runtime_provenance_incomplete")

    metrics = {
        "scenario_count": len(scenario_reports),
        "horizon_steps": horizon,
        "trace_incomplete_count": trace_incomplete_count,
        "baseline_runtime_error_count": sum(int((item["baseline_report"] or {}).get("runtime_error_count", 0) or 0) for item in scenario_reports),
        "default_runtime_error_count": sum(int(((item["llm_audit"] or {}).get("completeness", {}) or {}).get("runtime_error_count", 0) or 0) for item in scenario_reports),
        "cvodes_failure_count": sum(
            int((item["baseline_report"] or {}).get("cvodes_failure_count", 0) or 0)
            + int(((item["llm_audit"] or {}).get("completeness", {}) or {}).get("cvodes_failure_count", 0) or 0)
            for item in scenario_reports
        ),
        "profile_issue_count": _sum_metric(scenario_reports, "profile_temp_jump_count")
        + _sum_metric(scenario_reports, "profile_rh_jump_count")
        + _sum_metric(scenario_reports, "profile_co2_jump_count")
        + _sum_metric(scenario_reports, "profile_target_reversal_count")
        + _sum_metric(scenario_reports, "profile_co2_vent_conflict_count")
        + _sum_metric(scenario_reports, "profile_dew_conflict_count")
        + _sum_metric(scenario_reports, "profile_vpd_conflict_count"),
        "candidate_gap_count": _sum_metric(scenario_reports, "candidate_missing_count")
        + _sum_metric(scenario_reports, "candidate_filtered_count")
        + _sum_metric(scenario_reports, "candidate_unsafe_conflict_count")
        + _sum_metric(scenario_reports, "fallback_selected_count"),
        "stability_issue_count": _sum_metric(scenario_reports, "large_action_delta_count")
        + _sum_metric(scenario_reports, "action_oscillation_count")
        + _sum_metric(scenario_reports, "guardrail_rewrite_count"),
        "runtime_provenance_record_count": _sum_metric(scenario_reports, "runtime_provenance_record_count"),
        "runtime_reason_missing_count": _sum_metric(scenario_reports, "runtime_reason_missing_count"),
        "unknown_post_guardrail_rewrite_count": _sum_metric(scenario_reports, "unknown_post_guardrail_rewrite_count"),
        "hard_safety_warning_count": hard_safety_warning_count,
        "metadata_missing_field_count": missing_metadata_count,
    }
    report: dict[str, Any] = {
        "schema_version": "qwen_max_h720_long_horizon_discovery_20260529_v39",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "qwen-max h720 long-horizon discovery",
        },
        "model_policy": {
            "active_model_family": "qwen-max-latest",
            "qwen_3_7_migration_allowed": False,
        },
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "controlled_scope_ok": controlled_scope_ok,
        "scenario_reports": scenario_reports,
        "metrics": metrics,
        "failure_taxonomy": sorted(set(failure_taxonomy)),
    }
    opportunities = _rank_opportunities(report)
    report["opportunities"] = opportunities
    if trace_incomplete_count or metrics["default_runtime_error_count"] or metrics["cvodes_failure_count"]:
        report["next_action"] = "runtime_attribution_and_stability_guard_design"
    elif hard_safety_warning_count:
        report["next_action"] = "safety_boundary_diagnosis_before_v40"
    else:
        report["next_action"] = opportunities[0]["next_action"] if opportunities else "v39_result_review"
    return report


def _scenario_metric_rows(report: Mapping[str, Any], keys: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for item in report.get("scenario_reports", []) or []:
        metrics = ((item.get("llm_audit", {}) or {}).get("metrics", {}) or {})
        rows.append({"scenario_id": item.get("scenario_id", ""), **{key: int(metrics.get(key, 0) or 0) for key in keys}})
    return rows


def build_artifacts(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    profile_keys = [
        "profile_temp_jump_count",
        "profile_rh_jump_count",
        "profile_co2_jump_count",
        "profile_target_reversal_count",
        "profile_co2_vent_conflict_count",
        "profile_dew_conflict_count",
        "profile_vpd_conflict_count",
    ]
    candidate_keys = [
        "candidate_missing_count",
        "candidate_filtered_count",
        "candidate_unsafe_conflict_count",
        "fallback_selected_count",
        "guardrail_rewrite_count",
    ]
    stability_keys = [
        "large_action_delta_count",
        "action_oscillation_count",
        "guardrail_rewrite_count",
        "hard_safety_warning_count",
        "runtime_provenance_record_count",
        "runtime_reason_missing_count",
        "unknown_post_guardrail_rewrite_count",
    ]
    common = {
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }
    reason_counts: Counter[str] = Counter()
    for item in report.get("scenario_reports", []) or []:
        reason_counts.update(((item.get("llm_audit", {}) or {}).get("candidate_reason_counts", {}) or {}))
    return {
        "qwen_max_h720_long_horizon_result_audit_20260529_v39": {
            **common,
            "schema_version": "qwen_max_h720_long_horizon_result_audit_20260529_v39",
            "metrics": report.get("metrics", {}),
            "failure_taxonomy": report.get("failure_taxonomy", []),
            "scenario_reports": [
                {
                    "scenario_id": item.get("scenario_id"),
                    "baseline_report": item.get("baseline_report"),
                    "llm_completeness": (item.get("llm_audit", {}) or {}).get("completeness", {}),
                }
                for item in report.get("scenario_reports", []) or []
            ],
            "next_action": report.get("next_action"),
        },
        "qwen_max_h720_profile_trajectory_quality_audit_20260529_v39": {
            **common,
            "schema_version": "qwen_max_h720_profile_trajectory_quality_audit_20260529_v39",
            "metrics": {key: int(report.get("metrics", {}).get("profile_issue_count", 0) or 0) if key == "profile_issue_count" else _sum_metric(report.get("scenario_reports", []) or [], key) for key in ["profile_issue_count", *profile_keys]},
            "scenario_rows": _scenario_metric_rows(report, profile_keys),
            "examples": [
                {"scenario_id": item.get("scenario_id"), "examples": (item.get("llm_audit", {}) or {}).get("profile_examples", [])}
                for item in report.get("scenario_reports", []) or []
            ],
        },
        "qwen_max_h720_candidate_action_pool_gap_catalog_20260529_v39": {
            **common,
            "schema_version": "qwen_max_h720_candidate_action_pool_gap_catalog_20260529_v39",
            "metrics": {key: int(report.get("metrics", {}).get("candidate_gap_count", 0) or 0) if key == "candidate_gap_count" else _sum_metric(report.get("scenario_reports", []) or [], key) for key in ["candidate_gap_count", *candidate_keys]},
            "scenario_rows": _scenario_metric_rows(report, candidate_keys),
            "reason_counts": dict(sorted(reason_counts.items())),
            "examples": [
                {"scenario_id": item.get("scenario_id"), "examples": (item.get("llm_audit", {}) or {}).get("candidate_examples", [])}
                for item in report.get("scenario_reports", []) or []
            ],
        },
        "qwen_max_h720_control_stability_audit_20260529_v39": {
            **common,
            "schema_version": "qwen_max_h720_control_stability_audit_20260529_v39",
            "metrics": {key: int(report.get("metrics", {}).get("stability_issue_count", 0) or 0) if key == "stability_issue_count" else _sum_metric(report.get("scenario_reports", []) or [], key) for key in ["stability_issue_count", *stability_keys]},
            "scenario_rows": _scenario_metric_rows(report, stability_keys),
            "examples": [
                {"scenario_id": item.get("scenario_id"), "examples": (item.get("llm_audit", {}) or {}).get("stability_examples", [])}
                for item in report.get("scenario_reports", []) or []
            ],
        },
        "qwen_max_h720_opportunity_catalog_20260529_v39": {
            **common,
            "schema_version": "qwen_max_h720_opportunity_catalog_20260529_v39",
            "opportunities": report.get("opportunities", []),
            "selected_next_action": report.get("next_action"),
            "notes": [
                "Shadow discovery evidence only; not performance or promotion evidence.",
                "Cache is used as isolated record/replay support, not as the main discovery gate.",
            ],
        },
        "metadata_replay_readiness_checklist_20260529_v39": {
            **common,
            "schema_version": "metadata_replay_readiness_checklist_20260529_v39",
            "mainline_alignment": report.get("mainline_alignment", {}),
            "model_policy": report.get("model_policy", {}),
            "metrics": report.get("metrics", {}),
            "failure_taxonomy": report.get("failure_taxonomy", []),
            "next_action": report.get("next_action"),
        },
    }


def _markdown_for_artifact(name: str, artifact: Mapping[str, Any]) -> str:
    lines = [
        f"# {name}",
        "",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
    ]
    if artifact.get("next_action") or artifact.get("selected_next_action"):
        lines.append(f"- Next action: `{artifact.get('next_action') or artifact.get('selected_next_action')}`")
    lines.extend(["", "## Metrics", "", "| metric | value |", "| --- | ---: |"])
    for key, value in dict(artifact.get("metrics", {})).items():
        lines.append(f"| {key} | `{value}` |")
    if artifact.get("opportunities"):
        lines.extend(["", "## Opportunities", "", "| area | score | next action |", "| --- | ---: | --- |"])
        for item in artifact.get("opportunities", []) or []:
            lines.append(f"| `{item.get('area')}` | `{item.get('priority_score')}` | `{item.get('next_action')}` |")
    if artifact.get("failure_taxonomy"):
        lines.extend(["", "## Failure Taxonomy", ""])
        lines.extend(f"- `{item}`" for item in artifact.get("failure_taxonomy", []) or [])
    return "\n".join(lines) + "\n"


def write_artifacts(report: Mapping[str, Any], output_dir: str | Path) -> list[Path]:
    output_root = _resolve(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, artifact in build_artifacts(report).items():
        json_path = output_root / f"{name}.json"
        md_path = output_root / f"{name}.md"
        json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        md_path.write_text(_markdown_for_artifact(name, artifact), encoding="utf-8")
        written.extend([json_path, md_path])
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-record-json", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", default="gl_gym/result/audits")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(execution_record=_load(args.execution_record_json), trace_dir=args.trace_dir)
    written = write_artifacts(report, args.output_dir)
    print(f"scenario_count={report['metrics']['scenario_count']}")
    print(f"profile_issue_count={report['metrics']['profile_issue_count']}")
    print(f"candidate_gap_count={report['metrics']['candidate_gap_count']}")
    print(f"stability_issue_count={report['metrics']['stability_issue_count']}")
    print(f"next_action={report['next_action']}")
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
