"""Build v65 qwen3.7-plus structured-bridge runtime failure attribution artifacts.

This stage is offline-only. It reads the v64 structured-anchor runtime bridge
shadow traces and audits, separates bridge-contract evidence from runtime
stability, and attributes the remaining short traces without running rollout,
calling an online LLM, changing llm_rspc_v2, or authorizing controlled replay.
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

import gl_gym.experiments.cvodes_failure_causal_attribution_v44 as v44  # noqa: E402
import gl_gym.experiments.qwen37plus_structured_anchor_profile_bridge_runtime_v64 as v64  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import AUDIT_DIR  # noqa: E402


MODEL_NAME = "qwen3.7-plus"
ARTIFACT_DATE = "20260603"
FAILURE_SCENARIOS = tuple(v64.FAILURE_SCENARIOS)
WINDOW_STEPS = 120

V64_TRACE_DIR = v64.V64_TRACE_DIR
V39_TRACE_DIR = v44.ORIGINAL_TRACE_DIR
V64_RESULT_JSON = v64.RESULT_AUDIT_JSON
V64_CONSISTENCY_JSON = v64.CONSISTENCY_AUDIT_JSON
V44_CAUSAL_JSON = v44.CAUSAL_JSON
V40_ATTRIBUTION_JSON = v44.V40_ATTRIBUTION_JSON

ATTRIBUTION_JSON = AUDIT_DIR / "qwen37plus_structured_bridge_runtime_failure_attribution_20260603_v65.json"
ATTRIBUTION_MD = AUDIT_DIR / "qwen37plus_structured_bridge_runtime_failure_attribution_20260603_v65.md"
DELTA_JSON = AUDIT_DIR / "qwen37plus_structured_bridge_profile_candidate_guardrail_delta_audit_20260603_v65.json"
DELTA_MD = AUDIT_DIR / "qwen37plus_structured_bridge_profile_candidate_guardrail_delta_audit_20260603_v65.md"
SIMULATOR_JSON = AUDIT_DIR / "qwen37plus_structured_bridge_simulator_boundary_audit_20260603_v65.json"
SIMULATOR_MD = AUDIT_DIR / "qwen37plus_structured_bridge_simulator_boundary_audit_20260603_v65.md"
REPAIR_JSON = AUDIT_DIR / "qwen37plus_structured_bridge_runtime_repair_design_20260603_v65.json"
REPAIR_MD = AUDIT_DIR / "qwen37plus_structured_bridge_runtime_repair_design_20260603_v65.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260603_v65.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260603_v65.md"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


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
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    p = _resolve(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _trace_path(trace_dir: str | Path, scenario_id: str, controller: str = "llm_rspc_v2") -> Path:
    root = _resolve(trace_dir)
    expected = root / f"{scenario_id}_{controller}.csv"
    if expected.exists():
        return expected
    match = re.search(r"y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)", scenario_id)
    if not match:
        return expected
    prefix = f"y{match.group('year')}_d{match.group('day')}_s{match.group('seed')}_"
    matches = sorted(root.glob(f"{prefix}*_{controller}.csv"))
    return matches[0] if matches else expected


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
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "applied"}


def _error_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(row.get(key, "") or "")
        for key in ("runtime_error", "runtime_error_type", "error", "replan_reason")
    ).strip()


def _has_runtime_failure_signal(row: Mapping[str, Any]) -> bool:
    if str(row.get("runtime_error", "") or "").strip():
        return True
    if str(row.get("runtime_error_type", "") or "").strip():
        return True
    if str(row.get("error", "") or "").strip():
        return True
    return "runtime" in str(row.get("replan_reason", "") or "").lower()


def classify_runtime_error(text: str) -> str:
    lowered = str(text or "").lower()
    if not lowered:
        return ""
    if "invalid argument" in lowered or "[errno 22]" in lowered:
        return "simulator_io_invalid_argument"
    if "cvodes" in lowered or "cvodesinterface" in lowered or "cv_conv_failure" in lowered or "casadi" in lowered:
        return "cvodes_or_casadi_failure"
    return "runtime_error_other"


def _runtime_failure_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for row in rows:
        if _has_runtime_failure_signal(row):
            return row
    return None


def _step(row: Mapping[str, Any] | None, default: int = -1) -> int:
    if row is None:
        return int(default)
    return int(_num(row.get("step"), default))


def _window(rows: Sequence[Mapping[str, Any]], *, center_step: int | None, window_steps: int) -> list[Mapping[str, Any]]:
    if not rows:
        return []
    if center_step is None:
        center_step = _step(rows[-1], len(rows) - 1)
    start = max(0, int(center_step) - int(window_steps) + 1)
    return [row for row in rows if start <= _step(row) <= int(center_step)]


def _window_meta(rows: Sequence[Mapping[str, Any]], window: Sequence[Mapping[str, Any]], *, window_steps: int) -> dict[str, Any]:
    return {
        "row_count": int(len(window)),
        "window_steps_requested": int(window_steps),
        "window_start_step": _step(window[0], 0) if window else None,
        "window_end_step": _step(window[-1], 0) if window else None,
        "short_window": bool(len(window) < window_steps),
        "trace_row_count": int(len(rows)),
    }


def _bridge_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempted = sum(_truthy(row.get("structured_anchor_profile_bridge_attempted")) for row in rows)
    bridgeable = sum(_truthy(row.get("structured_anchor_profile_bridge_bridgeable")) for row in rows)
    applied = sum(_truthy(row.get("structured_anchor_profile_bridge_applied")) for row in rows)
    final_change = sum(_truthy(row.get("structured_anchor_profile_bridge_final_control_change")) for row in rows)
    plan_modified = sum(_truthy(row.get("structured_anchor_profile_bridge_current_plan_modified")) for row in rows)
    low_level = sum(_truthy(row.get("structured_anchor_profile_bridge_low_level_action_generated")) for row in rows)
    hard_safety = sum(_truthy(row.get("structured_anchor_profile_bridge_hard_safety_profile_violation")) for row in rows)
    valid_anchor = sum(_truthy(row.get("structured_anchor_profile_bridge_valid_structured_anchor")) for row in rows)
    candidate_counts = [
        int(_num(row.get("structured_anchor_profile_bridge_profile_candidate_count"), 0))
        for row in rows
        if row.get("structured_anchor_profile_bridge_profile_candidate_count") not in (None, "")
    ]
    return {
        "runtime_bridge_attempt_count": int(attempted),
        "bridgeable_count": int(bridgeable),
        "bridge_applied_count": int(applied),
        "valid_structured_anchor_count": int(valid_anchor),
        "profile_candidate_count_min": int(min(candidate_counts) if candidate_counts else 0),
        "profile_candidate_count_max": int(max(candidate_counts) if candidate_counts else 0),
        "hard_safety_profile_violation_count": int(hard_safety),
        "final_control_change_count": int(final_change),
        "current_plan_modified_count": int(plan_modified),
        "low_level_action_generated_count": int(low_level),
        "control_path_unchanged_failure": bool(attempted > 0 and final_change == 0 and plan_modified == 0 and low_level == 0),
    }


def _action_similarity(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_step = {_step(row): row for row in right}
    compared = 0
    near_equal = 0
    abs_delta_sum = 0.0
    examples: list[dict[str, Any]] = []
    for row in left:
        other = by_step.get(_step(row))
        if other is None:
            continue
        for field in v44.ACTION_FIELDS:
            compared += 1
            delta = abs(_num(row.get(field)) - _num(other.get(field)))
            abs_delta_sum += delta
            if delta <= 1e-6:
                near_equal += 1
            elif len(examples) < 10:
                examples.append({"step": _step(row), "field": field, "abs_delta": delta})
    return {
        "compared_action_values": int(compared),
        "near_equal_action_values": int(near_equal),
        "near_equal_rate": float(near_equal / compared) if compared else 0.0,
        "mean_abs_delta": float(abs_delta_sum / compared) if compared else 0.0,
        "delta_examples": examples,
    }


def _profile_candidate_guardrail_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = v44._profile_guardrail_metrics(rows)
    candidate_issue = 0
    hard_or_major_rewrite = 0
    profile_action_conflict = 0
    guardrail_reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for row in rows:
        label = str(row.get("candidate_guardrail_compatibility_label", "") or "").lower()
        conflict = str(row.get("profile_action_conflict_label", "") or "").lower()
        source = str(row.get("candidate_selection_source", "") or row.get("source", "") or "")
        if source:
            source_counts[source] += 1
        if label and label not in {"compatible", "ok", "none", "null"}:
            candidate_issue += 1
        if conflict and conflict not in {"compatible", "ok", "none", "null"}:
            profile_action_conflict += 1
        rewrite_fields = str(row.get("candidate_guardrail_rewrite_fields", "") or "")
        if rewrite_fields:
            hard_or_major_rewrite += 1
        for reason in str(row.get("final_action_risk_reason_v2", "") or row.get("tomato_safety_v2_reasons", "") or "").split(","):
            reason = reason.strip()
            if reason:
                guardrail_reason_counts[reason] += 1
    base.update(
        {
            "candidate_guardrail_issue_steps": int(candidate_issue),
            "profile_action_conflict_steps": int(profile_action_conflict),
            "major_or_hard_rewrite_predicted_steps": int(hard_or_major_rewrite),
            "selection_source_counts": dict(sorted(source_counts.items())),
            "final_action_or_guardrail_reason_counts": dict(sorted(guardrail_reason_counts.items())),
        }
    )
    base["score"] = int(base.get("score", 0)) + candidate_issue + profile_action_conflict + hard_or_major_rewrite
    return base


def _state_boundary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    base = v44._boundary_metrics(rows)
    counts = dict(base.get("counts", {}) or {})
    counts.update(
        {
            "final_action_canopy_warning_count": sum(_truthy(row.get("final_action_predicted_canopy_warning_v2")) for row in rows),
            "final_action_canopy_fail_count": sum(_truthy(row.get("final_action_would_fail_canopy_boundary_v2")) for row in rows),
            "canopy_boundary_shadow_warning_count": sum(_truthy(row.get("canopy_boundary_shadow_warning_v2")) for row in rows),
            "rh_low_violation_count": sum(_num(row.get("rh_low_violation"), 0) > 0 for row in rows),
            "rh_high_violation_count": sum(_num(row.get("rh_high_violation"), 0) > 0 for row in rows),
            "temp_violation_count": sum(_num(row.get("temp_violation"), 0) > 0 for row in rows),
        }
    )
    base["counts"] = counts
    base["score"] = int(base.get("score", 0)) + sum(
        int(counts[key])
        for key in (
            "final_action_canopy_warning_count",
            "final_action_canopy_fail_count",
            "canopy_boundary_shadow_warning_count",
            "rh_low_violation_count",
            "rh_high_violation_count",
            "temp_violation_count",
        )
    )
    return base


def _scenario_report(
    *,
    scenario_id: str,
    v64_trace_dir: str | Path,
    v39_trace_dir: str | Path,
    window_steps: int,
) -> dict[str, Any]:
    v64_trace = _trace_path(v64_trace_dir, scenario_id)
    v39_trace = _trace_path(v39_trace_dir, scenario_id)
    rows = _read_rows(v64_trace)
    original_rows = _read_rows(v39_trace)
    failure_row = _runtime_failure_row(rows)
    original_failure_row = _runtime_failure_row(original_rows)
    failure_step = _step(failure_row, len(rows) - 1 if rows else -1)
    original_failure_step = _step(original_failure_row, len(original_rows) - 1 if original_rows else -1)
    error_text = _error_text(failure_row or {})
    error_kind = classify_runtime_error(error_text)
    original_error_kind = classify_runtime_error(_error_text(original_failure_row or {}))
    failure_window = _window(rows, center_step=failure_step if failure_row else None, window_steps=window_steps)
    original_window = _window(original_rows, center_step=failure_step if rows else original_failure_step, window_steps=window_steps)
    bridge = _bridge_metrics(failure_window)
    action = v44._action_channel_metrics(failure_window)
    sequence = v44._sequence_history_metrics(failure_window)
    profile_guardrail = _profile_candidate_guardrail_metrics(failure_window)
    boundary = _state_boundary_metrics(failure_window)
    original_profile_guardrail = _profile_candidate_guardrail_metrics(original_window)
    original_action = v44._action_channel_metrics(original_window)
    original_boundary = _state_boundary_metrics(original_window)
    similarity = _action_similarity(failure_window, original_window)
    causal_scores = {
        "control_path_unchanged_failure": int(1 if bridge["control_path_unchanged_failure"] else 0),
        "profile_candidate_guardrail_pressure": int(profile_guardrail["score"]),
        "state_boundary_pressure": int(boundary["score"]),
        "action_sequence_pressure": int(action["score"] + sequence["score"]),
        "simulator_io_boundary": int(1 if error_kind else 0),
    }
    ranked = sorted(causal_scores.items(), key=lambda item: (-int(item[1]), item[0]))
    dominant = ranked[0][0] if ranked and ranked[0][1] > 0 else "unknown_requires_instrumentation"
    return {
        "scenario_id": scenario_id,
        "trace_path": _rel(v64_trace),
        "v39_trace_path": _rel(v39_trace),
        "trace_exists": bool(rows),
        "v39_trace_exists": bool(original_rows),
        "row_count": int(len(rows)),
        "short_trace_before_horizon": bool(len(rows) < 720),
        "failure_step": int(failure_step),
        "runtime_error_kind": error_kind,
        "runtime_error_text": error_text[:500],
        "v39_failure_step": int(original_failure_step),
        "v39_runtime_error_kind": original_error_kind,
        "failure_window": _window_meta(rows, failure_window, window_steps=window_steps),
        "v39_aligned_window": _window_meta(original_rows, original_window, window_steps=window_steps),
        "bridge_metrics": bridge,
        "control_path_unchanged_failure": bool(bridge["control_path_unchanged_failure"]),
        "profile_candidate_guardrail_metrics": profile_guardrail,
        "action_sequence_metrics": {
            "action_channel": action,
            "sequence_history": sequence,
        },
        "state_boundary_metrics": boundary,
        "v39_aligned_metrics": {
            "profile_candidate_guardrail": original_profile_guardrail,
            "action_channel": original_action,
            "state_boundary": original_boundary,
        },
        "v39_final_action_similarity": similarity,
        "causal_scores": causal_scores,
        "dominant_attribution": dominant,
    }


def _aggregate(scenario_reports: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for report in scenario_reports:
        for key, value in (report.get("causal_scores", {}) or {}).items():
            totals[str(key)] += int(value)
    return dict(sorted(totals.items()))


def _dominant(aggregate: Mapping[str, int]) -> str:
    items = sorted(aggregate.items(), key=lambda item: (-int(item[1]), item[0]))
    return items[0][0] if items and int(items[0][1]) > 0 else "unknown_requires_instrumentation"


def _next_action(
    dominant: str,
    *,
    control_path_unchanged_count: int,
    invalid_argument_count: int,
    scenario_count: int,
) -> str:
    if control_path_unchanged_count == scenario_count and dominant in {
        "control_path_unchanged_failure",
        "profile_candidate_guardrail_pressure",
    }:
        return "profile_candidate_guardrail_reconciliation_patch_plan"
    if dominant == "action_sequence_pressure":
        return "transition_gate_v2_with_profile_compatibility_plan"
    if dominant == "state_boundary_pressure":
        return "profile_generator_runtime_candidate_selection_plan"
    if invalid_argument_count:
        return "simulator_runtime_instrumentation_patch_plan"
    if dominant == "simulator_io_boundary":
        return "simulator_runtime_instrumentation_patch_plan"
    if dominant == "profile_candidate_guardrail_pressure":
        return "profile_candidate_guardrail_reconciliation_patch_plan"
    return "simulator_runtime_instrumentation_patch_plan"


def build_runtime_failure_attribution(
    *,
    v64_trace_dir: str | Path = V64_TRACE_DIR,
    v39_trace_dir: str | Path = V39_TRACE_DIR,
    v64_result_json: str | Path = V64_RESULT_JSON,
    v64_consistency_json: str | Path = V64_CONSISTENCY_JSON,
    v44_causal_json: str | Path = V44_CAUSAL_JSON,
    v40_attribution_json: str | Path = V40_ATTRIBUTION_JSON,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    window_steps: int = WINDOW_STEPS,
) -> dict[str, Any]:
    reports = [
        _scenario_report(
            scenario_id=scenario,
            v64_trace_dir=v64_trace_dir,
            v39_trace_dir=v39_trace_dir,
            window_steps=window_steps,
        )
        for scenario in failure_scenarios
    ]
    aggregate = _aggregate(reports)
    dominant = _dominant(aggregate)
    runtime_kind_counts = Counter(str(report.get("runtime_error_kind") or "missing_runtime_error") for report in reports)
    control_path_unchanged_count = sum(bool(report.get("control_path_unchanged_failure")) for report in reports)
    invalid_argument_count = int(runtime_kind_counts.get("simulator_io_invalid_argument", 0))
    next_action = _next_action(
        dominant,
        control_path_unchanged_count=control_path_unchanged_count,
        invalid_argument_count=invalid_argument_count,
        scenario_count=len(reports),
    )
    v64_result = _load_json(v64_result_json)
    v64_consistency = _load_json(v64_consistency_json)
    return {
        "artifact": "qwen37plus_structured_bridge_runtime_failure_attribution_20260603_v65",
        "schema_version": "qwen37plus_structured_bridge_runtime_failure_attribution_20260603_v65",
        "current_stage": "v65_qwen37plus_structured_bridge_runtime_failure_attribution",
        "model_name": MODEL_NAME,
        "scope": "offline_runtime_failure_attribution_after_structured_bridge",
        "failure_scenarios": list(failure_scenarios),
        "window_steps": int(window_steps),
        "source_v64_result_artifact": v64_result.get("artifact", ""),
        "source_v64_runtime_bridge_contract_pass": bool(v64_result.get("runtime_bridge_contract_pass", False)),
        "source_v64_runtime_stability_pass": bool(v64_result.get("runtime_stability_pass", False)),
        "source_v64_consistency_artifact": v64_consistency.get("artifact", ""),
        "source_v64_final_control_change_count": int(v64_consistency.get("final_control_change_count", 0) or 0),
        "source_v64_current_plan_modified_count": int(v64_consistency.get("current_plan_modified_count", 0) or 0),
        "source_v64_low_level_action_generated_count": int(v64_consistency.get("low_level_action_generated_count", 0) or 0),
        "source_v44_artifact": _load_json(v44_causal_json).get("schema_version", ""),
        "source_v40_artifact": _load_json(v40_attribution_json).get("schema_version", ""),
        "online_llm_called": False,
        "new_rollout_run": False,
        "default_llm_rspc_v2_changed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "scenario_reports": reports,
        "runtime_error_kind_counts": dict(sorted(runtime_kind_counts.items())),
        "control_path_unchanged_failure_count": int(control_path_unchanged_count),
        "aggregate_causal_scores": aggregate,
        "dominant_attribution": dominant,
        "failure_taxonomy": sorted(
            set(
                [dominant]
                + [key for key, value in runtime_kind_counts.items() if value]
                + (["control_path_unchanged_after_bridge"] if control_path_unchanged_count else [])
            )
        ),
        "next_action": next_action,
    }


def build_profile_candidate_guardrail_delta_audit(attribution: Mapping[str, Any]) -> dict[str, Any]:
    reports = []
    issue_detected = False
    for report in attribution.get("scenario_reports", []) or []:
        current = report.get("profile_candidate_guardrail_metrics", {}) or {}
        original = ((report.get("v39_aligned_metrics", {}) or {}).get("profile_candidate_guardrail", {}) or {})
        delta_score = int(current.get("score", 0)) - int(original.get("score", 0))
        issue_detected = issue_detected or int(current.get("score", 0)) > 0
        reports.append(
            {
                "scenario_id": report.get("scenario_id"),
                "failure_step": report.get("failure_step"),
                "current_score": int(current.get("score", 0)),
                "v39_aligned_score": int(original.get("score", 0)),
                "delta_score_vs_v39_aligned": int(delta_score),
                "candidate_guardrail_issue_steps": int(current.get("candidate_guardrail_issue_steps", 0)),
                "profile_action_conflict_steps": int(current.get("profile_action_conflict_steps", 0)),
                "guardrail_rewrite_count": int(current.get("guardrail_rewrite_count", 0)),
                "major_or_hard_rewrite_predicted_steps": int(current.get("major_or_hard_rewrite_predicted_steps", 0)),
                "selection_source_counts": current.get("selection_source_counts", {}),
                "bridge_final_control_change_count": int((report.get("bridge_metrics", {}) or {}).get("final_control_change_count", 0)),
            }
        )
    return {
        "artifact": "qwen37plus_structured_bridge_profile_candidate_guardrail_delta_audit_20260603_v65",
        "current_stage": "v65_qwen37plus_structured_bridge_runtime_failure_attribution",
        "scenario_reports": reports,
        "profile_candidate_guardrail_pressure_detected": bool(issue_detected),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": "profile_candidate_guardrail_reconciliation_patch_plan"
        if issue_detected
        else "simulator_runtime_instrumentation_patch_plan",
    }


def build_simulator_boundary_audit(attribution: Mapping[str, Any]) -> dict[str, Any]:
    reports = []
    kind_counts: Counter[str] = Counter()
    for report in attribution.get("scenario_reports", []) or []:
        kind = str(report.get("runtime_error_kind") or "missing_runtime_error")
        kind_counts[kind] += 1
        boundary = report.get("state_boundary_metrics", {}) or {}
        reports.append(
            {
                "scenario_id": report.get("scenario_id"),
                "failure_step": report.get("failure_step"),
                "runtime_error_kind": kind,
                "runtime_error_text": report.get("runtime_error_text", ""),
                "short_trace_before_horizon": bool(report.get("short_trace_before_horizon", False)),
                "window": report.get("failure_window", {}),
                "boundary_score": int(boundary.get("score", 0)),
                "boundary_counts": boundary.get("counts", {}),
                "boundary_stats": boundary.get("stats", {}),
            }
        )
    return {
        "artifact": "qwen37plus_structured_bridge_simulator_boundary_audit_20260603_v65",
        "current_stage": "v65_qwen37plus_structured_bridge_runtime_failure_attribution",
        "runtime_error_kind_counts": dict(sorted(kind_counts.items())),
        "scenario_reports": reports,
        "cvodes_or_casadi_failure_count": int(kind_counts.get("cvodes_or_casadi_failure", 0)),
        "simulator_io_invalid_argument_count": int(kind_counts.get("simulator_io_invalid_argument", 0)),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": "simulator_runtime_instrumentation_patch_plan"
        if kind_counts.get("simulator_io_invalid_argument", 0)
        else attribution.get("next_action", "simulator_runtime_instrumentation_patch_plan"),
    }


def build_runtime_repair_design(
    attribution: Mapping[str, Any],
    delta_audit: Mapping[str, Any],
    simulator_audit: Mapping[str, Any],
) -> dict[str, Any]:
    next_action = str(attribution.get("next_action") or "simulator_runtime_instrumentation_patch_plan")
    design_map = {
        "profile_candidate_guardrail_reconciliation_patch_plan": [
            "Do not expand H720 yet.",
            "Design a shadow-only candidate/profile reconciliation patch that scores profile candidates against guardrail and Tomato Safety compatibility before final candidate preference.",
            "Keep structured bridge shadow-only until runtime instability is attributed away from the unchanged final-control path.",
        ],
        "transition_gate_v2_with_profile_compatibility_plan": [
            "Design transition smoothing only after profile compatibility is included, so the gate does not smooth toward an incompatible profile/action pair.",
            "Keep hard-safety bypass provenance separate from transition pressure.",
        ],
        "profile_generator_runtime_candidate_selection_plan": [
            "Move from bridge-only provenance to shadow selection among profile_generator candidates.",
            "Do not apply selected profile to current_plan until a shadow audit shows no hard-safety profile violation.",
        ],
        "simulator_runtime_instrumentation_patch_plan": [
            "Add simulator I/O and CVODES diagnostics around the final pre-failure step before further patch rollout.",
            "Record action vector, state vector, finite-value checks, solver error code, and candidate/source provenance at the failure boundary.",
        ],
    }
    return {
        "artifact": "qwen37plus_structured_bridge_runtime_repair_design_20260603_v65",
        "current_stage": "v65_qwen37plus_structured_bridge_runtime_failure_attribution",
        "dominant_attribution": attribution.get("dominant_attribution"),
        "failure_taxonomy": attribution.get("failure_taxonomy", []),
        "profile_candidate_guardrail_pressure_detected": bool(
            delta_audit.get("profile_candidate_guardrail_pressure_detected", False)
        ),
        "simulator_io_invalid_argument_count": int(simulator_audit.get("simulator_io_invalid_argument_count", 0) or 0),
        "cvodes_or_casadi_failure_count": int(simulator_audit.get("cvodes_or_casadi_failure_count", 0) or 0),
        "recommended_next_action": next_action,
        "design_recommendations": design_map.get(next_action, design_map["simulator_runtime_instrumentation_patch_plan"]),
        "rollout_execution_authorized": False,
        "online_llm_called": False,
        "new_rollout_run": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": next_action,
    }


def build_readiness(
    attribution: Mapping[str, Any],
    delta_audit: Mapping[str, Any],
    simulator_audit: Mapping[str, Any],
    repair_design: Mapping[str, Any],
) -> dict[str, Any]:
    bridge_pass = bool(attribution.get("source_v64_runtime_bridge_contract_pass", False))
    runtime_stability_pass = False
    return {
        "artifact": "metadata_replay_readiness_checklist_20260603_v65",
        "current_stage": "v65_qwen37plus_structured_bridge_runtime_failure_attribution",
        "model_name": MODEL_NAME,
        "runtime_bridge_contract_pass": bool(bridge_pass),
        "runtime_stability_pass": bool(runtime_stability_pass),
        "runtime_failure_attribution_complete": True,
        "dominant_attribution": attribution.get("dominant_attribution"),
        "failure_taxonomy": attribution.get("failure_taxonomy", []),
        "runtime_error_kind_counts": attribution.get("runtime_error_kind_counts", {}),
        "profile_candidate_guardrail_pressure_detected": bool(
            delta_audit.get("profile_candidate_guardrail_pressure_detected", False)
        ),
        "simulator_io_invalid_argument_count": int(simulator_audit.get("simulator_io_invalid_argument_count", 0) or 0),
        "cvodes_or_casadi_failure_count": int(simulator_audit.get("cvodes_or_casadi_failure_count", 0) or 0),
        "rollout_execution_authorized": False,
        "online_llm_called": False,
        "new_rollout_run": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": repair_design.get("next_action", "simulator_runtime_instrumentation_patch_plan"),
    }


def _write_json_md(path_json: Path, path_md: Path, payload: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
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
    v64_trace_dir: str | Path = V64_TRACE_DIR,
    v39_trace_dir: str | Path = V39_TRACE_DIR,
    window_steps: int = WINDOW_STEPS,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    attribution = build_runtime_failure_attribution(
        v64_trace_dir=v64_trace_dir,
        v39_trace_dir=v39_trace_dir,
        window_steps=window_steps,
        failure_scenarios=failure_scenarios,
    )
    delta = build_profile_candidate_guardrail_delta_audit(attribution)
    simulator = build_simulator_boundary_audit(attribution)
    repair = build_runtime_repair_design(attribution, delta, simulator)
    readiness = build_readiness(attribution, delta, simulator, repair)
    _write_json_md(ATTRIBUTION_JSON, ATTRIBUTION_MD, attribution, "v65 qwen3.7-plus Structured Bridge Runtime Failure Attribution")
    _write_json_md(DELTA_JSON, DELTA_MD, delta, "v65 qwen3.7-plus Profile/Candidate/Guardrail Delta Audit")
    _write_json_md(SIMULATOR_JSON, SIMULATOR_MD, simulator, "v65 qwen3.7-plus Simulator Boundary Audit")
    _write_json_md(REPAIR_JSON, REPAIR_MD, repair, "v65 qwen3.7-plus Runtime Repair Design")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v65 Metadata Replay Readiness")
    return {
        "attribution": attribution,
        "delta": delta,
        "simulator": simulator,
        "repair": repair,
        "readiness": readiness,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v64-trace-dir", default=_rel(V64_TRACE_DIR))
    parser.add_argument("--v39-trace-dir", default=_rel(V39_TRACE_DIR))
    parser.add_argument("--window-steps", type=int, default=WINDOW_STEPS)
    parser.add_argument("--failure-scenario", action="append", default=[])
    parser.add_argument("--write-all", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = write_all(
        v64_trace_dir=args.v64_trace_dir,
        v39_trace_dir=args.v39_trace_dir,
        window_steps=int(args.window_steps),
        failure_scenarios=args.failure_scenario or list(FAILURE_SCENARIOS),
    )
    if not args.write_all:
        print(json.dumps(result["readiness"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
