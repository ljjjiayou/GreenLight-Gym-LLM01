"""Build metadata replay readiness from closure and blocking audit reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _sum_rows(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(_num(row.get(field)) for row in rows if isinstance(row, Mapping))


def _next_action(
    *,
    expert_resolved: bool,
    protocol_delta_no_unknown: bool,
    protocol_user_decision_complete: bool,
    blocked_categories_resolved: bool,
    blocked_categories_resolved_for_v1_preflight: bool,
    safety_oracle_review_complete: bool,
    protocol_v1_preflight_ready: bool,
    old_vs_new_design_ready: bool,
    old_vs_new_audit_ready: bool,
    default_path_plan_ready: bool,
    cache_coverage_ready: bool,
    strict_metadata_replay_planning_allowed: bool,
    strict_metadata_replay_run_plan_ready: bool,
    protocol_v1_overlay_validated: bool,
    execution_authorization_packet_ready: bool,
    baseline_trace_manifest_ready: bool,
    canonical_execution_request_ready: bool,
    two_stage_authorization_packet_ready: bool,
    stage_b_authorization_request_ready: bool,
    stage_b_user_authorized: bool,
) -> str:
    if not expert_resolved:
        return "expert_distillation_restore_or_authorization_required"
    if not protocol_delta_no_unknown:
        return "protocol_delta_classification_required"
    if not protocol_user_decision_complete:
        return "protocol_v2_user_decision_required"
    if not blocked_categories_resolved_for_v1_preflight:
        return "protocol_v2_blocked_categories_resolution_required"
    if not safety_oracle_review_complete:
        return "safety_boundary_test_oracle_review_required"
    if not old_vs_new_design_ready:
        return "old_vs_new_protocol_audit_design_required"
    if not old_vs_new_audit_ready:
        return "old_vs_new_protocol_audit_readiness_required"
    if not protocol_v1_preflight_ready:
        return "protocol_v1_snapshot_manifest_required"
    if not default_path_plan_ready:
        return "default_path_action_invariance_plan_required"
    if not cache_coverage_ready:
        return "cache_coverage_check_required"
    if strict_metadata_replay_planning_allowed and not strict_metadata_replay_run_plan_ready:
        return "strict_metadata_replay_plan_can_be_drafted"
    if strict_metadata_replay_planning_allowed and strict_metadata_replay_run_plan_ready and not protocol_v1_overlay_validated:
        return "protocol_v1_ephemeral_overlay_preflight_required"
    if (
        strict_metadata_replay_planning_allowed
        and strict_metadata_replay_run_plan_ready
        and protocol_v1_overlay_validated
        and not execution_authorization_packet_ready
    ):
        return "strict_metadata_replay_execution_authorization_packet_required"
    if (
        strict_metadata_replay_planning_allowed
        and strict_metadata_replay_run_plan_ready
        and protocol_v1_overlay_validated
        and execution_authorization_packet_ready
        and not baseline_trace_manifest_ready
    ):
        return "canonical_action_diff_baseline_trace_manifest_required"
    if (
        strict_metadata_replay_planning_allowed
        and strict_metadata_replay_run_plan_ready
        and protocol_v1_overlay_validated
        and execution_authorization_packet_ready
        and baseline_trace_manifest_ready
        and not canonical_execution_request_ready
    ):
        return "canonical_strict_metadata_replay_execution_request_required"
    if (
        strict_metadata_replay_planning_allowed
        and strict_metadata_replay_run_plan_ready
        and protocol_v1_overlay_validated
        and execution_authorization_packet_ready
        and baseline_trace_manifest_ready
        and canonical_execution_request_ready
        and not two_stage_authorization_packet_ready
    ):
        return "strict_metadata_replay_two_stage_authorization_packet_required"
    if (
        strict_metadata_replay_planning_allowed
        and strict_metadata_replay_run_plan_ready
        and protocol_v1_overlay_validated
        and execution_authorization_packet_ready
        and baseline_trace_manifest_ready
        and canonical_execution_request_ready
        and two_stage_authorization_packet_ready
        and not stage_b_authorization_request_ready
    ):
        return "canonical_stage_b_authorization_request_required"
    if (
        strict_metadata_replay_planning_allowed
        and strict_metadata_replay_run_plan_ready
        and protocol_v1_overlay_validated
        and execution_authorization_packet_ready
        and baseline_trace_manifest_ready
        and canonical_execution_request_ready
        and two_stage_authorization_packet_ready
        and stage_b_authorization_request_ready
        and not stage_b_user_authorized
    ):
        return "await_explicit_stage_b_user_authorization_for_canonical_strict_metadata_replay"
    if (
        strict_metadata_replay_planning_allowed
        and strict_metadata_replay_run_plan_ready
        and protocol_v1_overlay_validated
        and execution_authorization_packet_ready
        and baseline_trace_manifest_ready
        and canonical_execution_request_ready
        and two_stage_authorization_packet_ready
        and stage_b_authorization_request_ready
        and stage_b_user_authorized
    ):
        return "execute_single_canonical_strict_metadata_replay"
    if strict_metadata_replay_planning_allowed and strict_metadata_replay_run_plan_ready:
        return "canonical_strict_metadata_replay_execution_authorization_required"
    return "strict_metadata_replay_plan_can_be_drafted"


def build_report(
    *,
    closure: Mapping[str, Any],
    protocol: Mapping[str, Any],
    default_path: Mapping[str, Any],
    expert_status: Mapping[str, Any] | None = None,
    default_path_plan: Mapping[str, Any] | None = None,
    cache_plan: Mapping[str, Any] | None = None,
    delta_classification: Mapping[str, Any] | None = None,
    protocol_authorization: Mapping[str, Any] | None = None,
    protocol_user_decision: Mapping[str, Any] | None = None,
    blocked_categories_plan: Mapping[str, Any] | None = None,
    safety_oracle_review: Mapping[str, Any] | None = None,
    old_vs_new_design: Mapping[str, Any] | None = None,
    old_vs_new_readiness: Mapping[str, Any] | None = None,
    cache_scenario_plan: Mapping[str, Any] | None = None,
    cache_coverage: Mapping[str, Any] | None = None,
    expanded_cache_coverage_plan: Mapping[str, Any] | None = None,
    strict_metadata_replay_run_plan: Mapping[str, Any] | None = None,
    protocol_v1_overlay_preflight: Mapping[str, Any] | None = None,
    execution_authorization_packet: Mapping[str, Any] | None = None,
    baseline_trace_manifest: Mapping[str, Any] | None = None,
    canonical_execution_request: Mapping[str, Any] | None = None,
    two_stage_authorization_packet: Mapping[str, Any] | None = None,
    stage_b_authorization_request: Mapping[str, Any] | None = None,
    strict_metadata_replay_summary: Mapping[str, Any] | None = None,
    trace_action_diff_audit: Mapping[str, Any] | None = None,
    runtime_provenance_audit: Mapping[str, Any] | None = None,
    joint_prediction_readiness: Mapping[str, Any] | None = None,
    runtime_provenance_closure_status: Mapping[str, Any] | None = None,
    expanded_execution_record: Mapping[str, Any] | None = None,
    controlled_replay_admission_review: Mapping[str, Any] | None = None,
    controlled_canary_execution_record: Mapping[str, Any] | None = None,
    controlled_canary_summary_audit: Mapping[str, Any] | None = None,
    controlled_canary_trace_audit: Mapping[str, Any] | None = None,
    controlled_canary_strict_effect_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    authorization_complete = bool(closure.get("mainline_diff_authorization_complete", False))
    protocol_pass = bool(protocol.get("protocol_isolation_pass", False))
    protocol_requires_authorization = bool(protocol.get("requires_protocol_baseline_authorization", False))
    protocol_authorized = bool(protocol.get("protocol_baseline_authorized", False))
    if isinstance(protocol_authorization, Mapping):
        protocol_authorized = bool(protocol_authorization.get("protocol_baseline_authorized", False))
        protocol_requires_authorization = bool(protocol_authorization.get("requires_user_authorization", True))
    delta_unknown_count = (
        int(delta_classification.get("unknown_hunk_count", 0) or 0)
        if isinstance(delta_classification, Mapping)
        else None
    )
    protocol_delta_no_unknown = (
        delta_unknown_count == 0
        if delta_unknown_count is not None
        else bool(protocol.get("protocol_delta_explained", False))
    )
    protocol_delta_explained = (
        bool(delta_classification.get("protocol_delta_explained", False))
        if isinstance(delta_classification, Mapping)
        else bool(protocol.get("protocol_delta_explained", False))
    )
    protocol_ready = bool(protocol_pass and protocol_delta_no_unknown and (not protocol_requires_authorization or protocol_authorized))
    default_path_pass = bool(default_path.get("default_path_evidence_pass", False))
    expert_resolved = (
        not bool(expert_status.get("hard_blocker", False))
        if isinstance(expert_status, Mapping)
        else not any("expert_distillation.py" in str(path) for path in closure.get("blocking_paths", []) or [])
    )
    default_path_plan_ready = bool(default_path_plan.get("plan_ready", False)) if isinstance(default_path_plan, Mapping) else default_path_pass
    old_vs_new_design_ready = bool(old_vs_new_design.get("design_ready", False)) if isinstance(old_vs_new_design, Mapping) else False
    old_vs_new_audit_ready = (
        bool(old_vs_new_readiness.get("old_vs_new_protocol_audit_ready", False))
        if isinstance(old_vs_new_readiness, Mapping)
        else False
    )
    protocol_user_decision_complete = (
        bool(protocol_user_decision.get("protocol_user_decision_complete", False))
        if isinstance(protocol_user_decision, Mapping)
        else False
    )
    blocked_categories_resolved = (
        bool(blocked_categories_plan.get("blocked_categories_resolved", False))
        if isinstance(blocked_categories_plan, Mapping)
        else False
    )
    blocked_categories_resolved_for_v1_preflight = (
        bool(blocked_categories_plan.get("blocked_categories_resolved_for_v1_preflight", blocked_categories_resolved))
        if isinstance(blocked_categories_plan, Mapping)
        else blocked_categories_resolved
    )
    blocked_categories_resolved_for_protocol_v2 = (
        bool(blocked_categories_plan.get("blocked_categories_resolved_for_protocol_v2", blocked_categories_resolved))
        if isinstance(blocked_categories_plan, Mapping)
        else blocked_categories_resolved
    )
    safety_oracle_review_complete = (
        bool(safety_oracle_review.get("safety_boundary_test_oracle_review_complete", False))
        if isinstance(safety_oracle_review, Mapping)
        else False
    )
    generic_cache_plan_ready = isinstance(cache_plan, Mapping) and bool(cache_plan.get("schema_version"))
    cache_scenario_plan_ready = (
        bool(cache_scenario_plan.get("cache_scenario_plan_ready", False))
        if isinstance(cache_scenario_plan, Mapping)
        else False
    )
    cache_plan_ready = bool(generic_cache_plan_ready and (cache_scenario_plan_ready or not isinstance(cache_scenario_plan, Mapping)))
    cache_coverage_pass = bool(cache_coverage.get("cache_coverage_pass", False)) if isinstance(cache_coverage, Mapping) else False
    cache_coverage_ready = bool(cache_plan_ready and cache_coverage_pass)
    expanded_cache_coverage_plan_ready = (
        bool(expanded_cache_coverage_plan.get("expanded_cache_coverage_plan_ready", False))
        if isinstance(expanded_cache_coverage_plan, Mapping)
        else False
    )
    expanded_cache_coverage_ready = (
        bool(expanded_cache_coverage_plan.get("expanded_cache_coverage_ready", False))
        if isinstance(expanded_cache_coverage_plan, Mapping)
        else False
    )
    strict_metadata_replay_run_plan_ready = (
        bool(strict_metadata_replay_run_plan.get("strict_metadata_replay_run_plan_ready", False))
        if isinstance(strict_metadata_replay_run_plan, Mapping)
        else False
    )
    protocol_v1_overlay_available = (
        bool(protocol_v1_overlay_preflight.get("protocol_v1_overlay_available", False))
        if isinstance(protocol_v1_overlay_preflight, Mapping)
        else False
    )
    protocol_v1_overlay_validated = (
        bool(protocol_v1_overlay_preflight.get("protocol_v1_overlay_validated", False))
        if isinstance(protocol_v1_overlay_preflight, Mapping)
        else False
    )
    execution_authorization_packet_ready = (
        bool(execution_authorization_packet.get("authorization_packet_ready", False))
        if isinstance(execution_authorization_packet, Mapping)
        else False
    )
    baseline_trace_manifest_ready = (
        bool(baseline_trace_manifest.get("baseline_trace_manifest_ready", False))
        if isinstance(baseline_trace_manifest, Mapping)
        else False
    )
    baseline_trace_qualified = (
        bool(baseline_trace_manifest.get("baseline_trace_qualified", False))
        if isinstance(baseline_trace_manifest, Mapping)
        else False
    )
    action_diff_audit_status = (
        baseline_trace_manifest.get("action_diff_audit_status", "not_provided")
        if isinstance(baseline_trace_manifest, Mapping)
        else "not_provided"
    )
    canonical_execution_request_ready = (
        bool(canonical_execution_request.get("canonical_strict_metadata_replay_execution_request_ready", False))
        if isinstance(canonical_execution_request, Mapping)
        else False
    )
    two_stage_authorization_packet_ready = (
        bool(two_stage_authorization_packet.get("two_stage_authorization_packet_ready", False))
        if isinstance(two_stage_authorization_packet, Mapping)
        else False
    )
    stage_a_overlay_validated = (
        bool(two_stage_authorization_packet.get("stage_a", {}).get("overlay_validated", False))
        if isinstance(two_stage_authorization_packet, Mapping)
        else protocol_v1_overlay_validated
    )
    stage_a_overlay_build_validation_complete = (
        bool(two_stage_authorization_packet.get("stage_a", {}).get("overlay_build_validation_complete", False))
        if isinstance(two_stage_authorization_packet, Mapping)
        else protocol_v1_overlay_validated
    )
    stage_a_overlay_hash_match = (
        bool(two_stage_authorization_packet.get("stage_a", {}).get("overlay_hash_match", False))
        if isinstance(two_stage_authorization_packet, Mapping)
        else False
    )
    stage_b_user_authorized = (
        bool(two_stage_authorization_packet.get("stage_b", {}).get("canonical_replay_user_authorized", False))
        if isinstance(two_stage_authorization_packet, Mapping)
        else False
    )
    stage_b_authorization_request_ready = (
        bool(stage_b_authorization_request.get("stage_b_authorization_request_ready", False))
        if isinstance(stage_b_authorization_request, Mapping)
        else False
    )
    if isinstance(stage_b_authorization_request, Mapping):
        stage_b_user_authorized = bool(stage_b_authorization_request.get("stage_b_user_authorized", stage_b_user_authorized))
    protocol_v1_snapshot_available = (
        bool(old_vs_new_readiness.get("old_protocol_snapshot_available", False))
        if isinstance(old_vs_new_readiness, Mapping)
        else False
    )
    protocol_v1_snapshot_manifest_available = (
        bool(old_vs_new_readiness.get("protocol_v1_snapshot_manifest_available", False))
        if isinstance(old_vs_new_readiness, Mapping)
        else False
    )
    protocol_v1_preflight_ready = bool(
        protocol_delta_no_unknown
        and protocol_user_decision_complete
        and blocked_categories_resolved_for_v1_preflight
        and safety_oracle_review_complete
        and old_vs_new_design_ready
        and old_vs_new_audit_ready
        and protocol_v1_snapshot_available
        and protocol_v1_snapshot_manifest_available
    )
    strict_metadata_replay_planning_allowed = bool(
        expert_resolved
        and protocol_v1_preflight_ready
        and protocol_user_decision_complete
        and blocked_categories_resolved_for_v1_preflight
        and safety_oracle_review_complete
        and old_vs_new_design_ready
        and old_vs_new_audit_ready
        and default_path_plan_ready
        and cache_coverage_ready
    )
    canonical_metadata_replay_authorization_request_ready = bool(
        strict_metadata_replay_planning_allowed
        and strict_metadata_replay_run_plan_ready
        and protocol_v1_overlay_validated
        and execution_authorization_packet_ready
        and baseline_trace_manifest_ready
        and canonical_execution_request_ready
        and two_stage_authorization_packet_ready
        and stage_b_authorization_request_ready
    )
    metadata_replay_execution_allowed = bool(
        canonical_metadata_replay_authorization_request_ready
        and stage_b_user_authorized
        and isinstance(stage_b_authorization_request, Mapping)
        and bool(stage_b_authorization_request.get("metadata_replay_execution_allowed", False))
    )
    controlled_canary_input_hint = any(
        isinstance(item, Mapping)
        for item in (
            controlled_canary_execution_record,
            controlled_canary_summary_audit,
            controlled_canary_trace_audit,
            controlled_canary_strict_effect_audit,
        )
    )
    metadata_replay_core_post_run_inputs_provided = any(
        isinstance(item, Mapping)
        for item in (
            strict_metadata_replay_summary,
            trace_action_diff_audit,
        )
    )
    post_run_inputs_provided = bool(
        metadata_replay_core_post_run_inputs_provided
        or (
            not controlled_canary_input_hint
            and any(
                isinstance(item, Mapping)
                for item in (
                    runtime_provenance_audit,
                    joint_prediction_readiness,
                )
            )
        )
    )
    summary_rows = (
        list(strict_metadata_replay_summary.get("rows", []) or [])
        if isinstance(strict_metadata_replay_summary, Mapping)
        else []
    )
    cache_enabled_steps = _sum_rows(summary_rows, "plan_cache_enabled_steps")
    cache_hit_steps = _sum_rows(summary_rows, "plan_cache_hit_steps")
    cache_hit_rate = (cache_hit_steps / cache_enabled_steps) if cache_enabled_steps else 0.0
    runtime_error_steps = _sum_rows(summary_rows, "runtime_error_steps")
    strict_cache_miss_runtime_error_steps = _sum_rows(summary_rows, "strict_cache_miss_runtime_error_steps")
    summary_gate_pass = (
        bool(strict_metadata_replay_summary.get("aggregate", {}).get("decision") == "pass")
        if isinstance(strict_metadata_replay_summary, Mapping)
        else False
    )
    action_diff_trace_count = (
        int(trace_action_diff_audit.get("trace_count", 0) or 0)
        if isinstance(trace_action_diff_audit, Mapping)
        else 0
    )
    action_diff_steps = (
        int(trace_action_diff_audit.get("action_diff_steps", 0) or 0)
        if isinstance(trace_action_diff_audit, Mapping)
        else 0
    )
    action_missing_steps = (
        int(trace_action_diff_audit.get("missing_step_count", 0) or 0)
        if isinstance(trace_action_diff_audit, Mapping)
        else 0
    )
    action_max_abs_delta = (
        _num(trace_action_diff_audit.get("max_abs_delta"))
        if isinstance(trace_action_diff_audit, Mapping)
        else 0.0
    )
    runtime_record_count = (
        int(runtime_provenance_audit.get("record_count", 0) or 0)
        if isinstance(runtime_provenance_audit, Mapping)
        else 0
    )
    runtime_reason_missing_count = (
        int(runtime_provenance_audit.get("runtime_reason_missing_count", 0) or 0)
        if isinstance(runtime_provenance_audit, Mapping)
        else 0
    )
    unknown_post_guardrail_rewrite_count = (
        int(runtime_provenance_audit.get("unknown_post_guardrail_rewrite_count", 0) or 0)
        if isinstance(runtime_provenance_audit, Mapping)
        else 0
    )
    joint_missing_fields = (
        sum(int(value or 0) for value in dict(joint_prediction_readiness.get("missing_field_counts", {})).values())
        if isinstance(joint_prediction_readiness, Mapping)
        else 0
    )
    joint_row_count = (
        int(joint_prediction_readiness.get("row_count", 0) or 0)
        if isinstance(joint_prediction_readiness, Mapping)
        else 0
    )
    joint_ready = (
        bool(joint_prediction_readiness.get("ready_for_shadow_audit", False))
        if isinstance(joint_prediction_readiness, Mapping)
        else False
    )
    runtime_provenance_closure_status_provided = isinstance(runtime_provenance_closure_status, Mapping)
    runtime_provenance_closure_implementation_ready = (
        bool(runtime_provenance_closure_status.get("runtime_provenance_closure_implementation_ready", False))
        if runtime_provenance_closure_status_provided
        else False
    )
    runtime_provenance_trace_export_implemented = (
        bool(runtime_provenance_closure_status.get("runtime_provenance_trace_export_implemented", False))
        if runtime_provenance_closure_status_provided
        else False
    )
    runtime_provenance_audit_trace_jsonl_supported = (
        bool(runtime_provenance_closure_status.get("audit_supports_trace_jsonl", False))
        if runtime_provenance_closure_status_provided
        else False
    )
    runtime_provenance_audit_trace_csv_supported = (
        bool(runtime_provenance_closure_status.get("audit_supports_trace_csv", False))
        if runtime_provenance_closure_status_provided
        else False
    )
    stage_b_rerun_authorization_required = (
        bool(runtime_provenance_closure_status.get("stage_b_rerun_authorization_required", False))
        if runtime_provenance_closure_status_provided
        else False
    )
    stage_b_rerun_authorized = (
        bool(runtime_provenance_closure_status.get("stage_b_rerun_authorized", False))
        if runtime_provenance_closure_status_provided
        else False
    )
    if isinstance(stage_b_authorization_request, Mapping):
        stage_b_rerun_authorized = bool(
            stage_b_authorization_request.get("stage_b_rerun_authorized", stage_b_rerun_authorized)
        )
    expanded_execution_record_provided = isinstance(expanded_execution_record, Mapping)
    first_wave_expanded_authorized = (
        bool(expanded_execution_record.get("first_wave_expanded_metadata_replay_authorized", False))
        if expanded_execution_record_provided
        else False
    )
    second_wave_expanded_authorized = (
        bool(expanded_execution_record.get("second_wave_expanded_metadata_replay_authorized", False))
        if expanded_execution_record_provided
        else False
    )
    expanded_scope = (
        str(expanded_execution_record.get("scope", "not_provided"))
        if expanded_execution_record_provided
        else "not_provided"
    )
    controlled_admission_provided = isinstance(controlled_replay_admission_review, Mapping)
    minimal_controlled_canary_allowed = (
        bool(controlled_replay_admission_review.get("minimal_controlled_canary_allowed", False))
        if controlled_admission_provided
        else False
    )
    controlled_admission_next_action = (
        str(controlled_replay_admission_review.get("next_action", "controlled_replay_admission_review_required"))
        if controlled_admission_provided
        else "controlled_replay_admission_review_required"
    )
    controlled_canary_execution_record_provided = isinstance(controlled_canary_execution_record, Mapping)
    controlled_canary_summary_provided = isinstance(controlled_canary_summary_audit, Mapping)
    controlled_canary_trace_provided = isinstance(controlled_canary_trace_audit, Mapping)
    controlled_canary_effect_provided = isinstance(controlled_canary_strict_effect_audit, Mapping)
    minimal_controlled_canary_executed = bool(
        controlled_canary_execution_record_provided
        and bool(controlled_canary_execution_record.get("minimal_controlled_canary_authorized", False))
        and controlled_canary_summary_provided
        and controlled_canary_trace_provided
        and controlled_canary_effect_provided
        and isinstance(runtime_provenance_audit, Mapping)
        and isinstance(joint_prediction_readiness, Mapping)
    )
    controlled_summary_aggregate = (
        controlled_canary_summary_audit.get("aggregate", {})
        if controlled_canary_summary_provided
        else {}
    )
    controlled_trace_aggregate = (
        controlled_canary_trace_audit.get("aggregate", {})
        if controlled_canary_trace_provided
        else {}
    )
    controlled_effect_aggregate = (
        controlled_canary_strict_effect_audit.get("aggregate", {})
        if controlled_canary_effect_provided
        else {}
    )
    controlled_summary_pass = bool(
        controlled_summary_aggregate.get("decision") == "pass"
        and int(controlled_summary_aggregate.get("failure_count", 0) or 0) == 0
    )
    controlled_cache_hit_steps = _num(
        sum(
            _num(row.get("plan_cache_hit_steps"))
            for row in controlled_canary_summary_audit.get("rows", [])
            if isinstance(row, Mapping)
        )
        if controlled_canary_summary_provided
        else 0
    )
    controlled_cache_enabled_steps = _num(
        sum(
            _num(row.get("plan_cache_enabled_steps"))
            for row in controlled_canary_summary_audit.get("rows", [])
            if isinstance(row, Mapping)
        )
        if controlled_canary_summary_provided
        else 0
    )
    controlled_cache_hit_rate = (
        controlled_cache_hit_steps / controlled_cache_enabled_steps
        if controlled_cache_enabled_steps
        else 0.0
    )
    controlled_runtime_error_steps = _num(
        sum(
            _num(row.get("runtime_error_steps"))
            for row in controlled_canary_summary_audit.get("rows", [])
            if isinstance(row, Mapping)
        )
        if controlled_canary_summary_provided
        else 0
    )
    controlled_strict_cache_miss_steps = _num(
        sum(
            _num(row.get("strict_cache_miss_runtime_error_steps"))
            for row in controlled_canary_summary_audit.get("rows", [])
            if isinstance(row, Mapping)
        )
        if controlled_canary_summary_provided
        else 0
    )
    controlled_unsafe_preferred_steps = int(_num(controlled_trace_aggregate.get("unsafe_preferred_steps")))
    controlled_unsafe_conflict_steps = int(_num(controlled_trace_aggregate.get("unsafe_conflict_steps")))
    controlled_unsafe_applied_steps = int(_num(controlled_trace_aggregate.get("unsafe_applied_steps")))
    controlled_applied_steps = int(_num(controlled_trace_aggregate.get("applied_steps")))
    controlled_strict_applied_steps = int(_num(controlled_trace_aggregate.get("strict_applied_steps")))
    controlled_action_changes_scope_ok = controlled_applied_steps == controlled_strict_applied_steps
    controlled_runtime_provenance_complete = bool(
        runtime_record_count > 0
        and runtime_reason_missing_count == 0
        and unknown_post_guardrail_rewrite_count == 0
    )
    controlled_joint_prediction_complete = bool(joint_missing_fields == 0 and joint_row_count > 0 and joint_ready)
    controlled_hard_safety_non_regression = bool(
        controlled_summary_pass
        and all(
            _num(row.get("d_canopy_dew_margin_lt0_steps")) <= 0.0
            and _num(row.get("d_dew_margin_air_lt0_steps")) <= 0.0
            for row in controlled_canary_summary_audit.get("paired_deltas", [])
            if isinstance(row, Mapping)
        )
    ) if controlled_canary_summary_provided else False
    controlled_effect_safety_ok = bool(
        controlled_canary_effect_provided
        and int(_num(controlled_effect_aggregate.get("runtime_error_steps"))) == 0
        and int(_num(controlled_effect_aggregate.get("unsafe_applied_steps"))) == 0
        and not controlled_effect_aggregate.get("warnings", [])
    )
    controlled_canary_failures: list[str] = []
    if controlled_canary_execution_record_provided or controlled_canary_summary_provided or controlled_canary_trace_provided:
        if not minimal_controlled_canary_executed:
            controlled_canary_failures.append("controlled_canary_audit_inputs_missing")
        if not controlled_summary_pass:
            controlled_canary_failures.append("controlled_summary_gate_failed")
        if controlled_cache_hit_rate < 1.0:
            controlled_canary_failures.append("cache_mismatch")
        if controlled_runtime_error_steps > 0:
            controlled_canary_failures.append("runtime_error")
        if controlled_strict_cache_miss_steps > 0:
            controlled_canary_failures.append("strict_cache_miss")
        if controlled_unsafe_preferred_steps > 0:
            controlled_canary_failures.append("controlled_unsafe_preferred")
        if controlled_unsafe_conflict_steps > 0:
            controlled_canary_failures.append("controlled_unsafe_conflict")
        if controlled_unsafe_applied_steps > 0:
            controlled_canary_failures.append("controlled_unsafe_applied")
        if not controlled_action_changes_scope_ok:
            controlled_canary_failures.append("controlled_action_change_outside_strict_eligible_applied")
        if not controlled_runtime_provenance_complete:
            controlled_canary_failures.append("runtime_provenance_missing")
        if not controlled_joint_prediction_complete:
            controlled_canary_failures.append("joint_prediction_missing")
        if not controlled_hard_safety_non_regression:
            controlled_canary_failures.append("hard_safety_regression")
        if not controlled_effect_safety_ok:
            controlled_canary_failures.append("strict_effect_audit_failed")
    controlled_canary_pass = bool(minimal_controlled_canary_executed and not controlled_canary_failures)
    post_run_failures: list[str] = []
    if post_run_inputs_provided:
        if not summary_gate_pass:
            post_run_failures.append("metadata_missing")
        if cache_hit_rate < 1.0:
            post_run_failures.append("cache_mismatch")
        if runtime_error_steps > 0:
            post_run_failures.append("runtime_error")
        if strict_cache_miss_runtime_error_steps > 0:
            post_run_failures.append("strict_cache_miss")
        if action_diff_trace_count <= 0:
            post_run_failures.append("blocked_missing_baseline_trace")
        if action_diff_steps > 0 or action_missing_steps > 0 or action_max_abs_delta > 0:
            post_run_failures.append("action_changed")
        if runtime_record_count <= 0 or runtime_reason_missing_count > 0:
            post_run_failures.append("runtime_provenance_missing")
        if unknown_post_guardrail_rewrite_count > 0:
            post_run_failures.append("unknown_post_guardrail_rewrite")
        if joint_missing_fields > 0 or joint_row_count <= 0 or not joint_ready:
            post_run_failures.append("joint_prediction_missing")
        if not protocol_v1_overlay_validated:
            post_run_failures.append("protocol_implementation_mismatch")
        if not stage_a_overlay_hash_match:
            post_run_failures.append("overlay_hash_mismatch")
    post_run_pass = bool(post_run_inputs_provided and not post_run_failures)
    effective_execution_allowed = (
        False
        if post_run_inputs_provided or stage_b_rerun_authorization_required
        else metadata_replay_execution_allowed
    )
    next_action_value = _next_action(
        expert_resolved=expert_resolved,
        protocol_delta_no_unknown=protocol_delta_no_unknown,
        protocol_user_decision_complete=protocol_user_decision_complete,
        blocked_categories_resolved=blocked_categories_resolved,
        blocked_categories_resolved_for_v1_preflight=blocked_categories_resolved_for_v1_preflight,
        safety_oracle_review_complete=safety_oracle_review_complete,
        protocol_v1_preflight_ready=protocol_v1_preflight_ready,
        old_vs_new_design_ready=old_vs_new_design_ready,
        old_vs_new_audit_ready=old_vs_new_audit_ready,
        default_path_plan_ready=default_path_plan_ready,
        cache_coverage_ready=cache_coverage_ready,
        strict_metadata_replay_planning_allowed=strict_metadata_replay_planning_allowed,
        strict_metadata_replay_run_plan_ready=strict_metadata_replay_run_plan_ready,
        protocol_v1_overlay_validated=protocol_v1_overlay_validated,
        execution_authorization_packet_ready=execution_authorization_packet_ready,
        baseline_trace_manifest_ready=baseline_trace_manifest_ready,
        canonical_execution_request_ready=canonical_execution_request_ready,
        two_stage_authorization_packet_ready=two_stage_authorization_packet_ready,
        stage_b_authorization_request_ready=stage_b_authorization_request_ready,
        stage_b_user_authorized=stage_b_user_authorized,
    )
    if post_run_inputs_provided:
        if expanded_execution_record_provided:
            expanded_success_next_action = (
                "expanded_metadata_coverage_consolidation_or_controlled_replay_admission_review"
                if second_wave_expanded_authorized
                else "expanded_metadata_coverage_planning_continue_or_scenario_cache_discovery"
            )
            next_action_value = (
                expanded_success_next_action
                if post_run_pass
                else "expanded_metadata_failure_diagnosis_required"
            )
        else:
            next_action_value = (
                "expanded_metadata_coverage_planning_only"
                if post_run_pass
                else "stage_b_failure_diagnosis_required"
            )
    elif runtime_provenance_closure_status_provided:
        next_action_value = (
            "await_explicit_stage_b_rerun_authorization_after_runtime_provenance_closure"
            if runtime_provenance_closure_implementation_ready
            else "runtime_provenance_closure_implementation_required"
        )
    if controlled_admission_provided:
        next_action_value = (
            "execute_minimal_controlled_canary"
            if minimal_controlled_canary_allowed
            else controlled_admission_next_action
        )
    if minimal_controlled_canary_executed:
        next_action_value = (
            "minimal_controlled_canary_result_review_or_expansion_admission"
            if controlled_canary_pass
            else "minimal_controlled_canary_failure_diagnosis_required"
        )
    schema_version = "metadata_replay_readiness_checklist_v14"
    if minimal_controlled_canary_executed:
        schema_version = "metadata_replay_readiness_checklist_v21"
    elif controlled_admission_provided:
        schema_version = "metadata_replay_readiness_checklist_v20"
    elif post_run_inputs_provided:
        schema_version = (
            "metadata_replay_readiness_checklist_v19"
            if expanded_execution_record_provided and second_wave_expanded_authorized
            else "metadata_replay_readiness_checklist_v18"
            if expanded_execution_record_provided and first_wave_expanded_authorized
            else "metadata_replay_readiness_checklist_v17"
            if runtime_provenance_closure_status_provided and stage_b_rerun_authorized
            else "metadata_replay_readiness_checklist_v15"
        )
    elif runtime_provenance_closure_status_provided:
        schema_version = "metadata_replay_readiness_checklist_v16"
    return {
        "schema_version": schema_version,
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / readiness checklist",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "metadata_replay_allowed": effective_execution_allowed,
        "metadata_replay_allowed_reason": "this stage only permits drafting a future strict metadata replay plan after readiness gates close",
        "strict_metadata_replay_planning_allowed": strict_metadata_replay_planning_allowed,
        "strict_metadata_replay_run_plan_ready": strict_metadata_replay_run_plan_ready,
        "metadata_replay_execution_allowed": effective_execution_allowed,
        "canonical_metadata_replay_authorization_request_ready": canonical_metadata_replay_authorization_request_ready,
        "metadata_replay_execution_allowed_reason": "strict metadata replay execution still requires a separate explicit authorization and action-invariance run plan",
        "controlled_replay_allowed": False,
        "minimal_controlled_canary_allowed": (
            False if minimal_controlled_canary_executed else minimal_controlled_canary_allowed
        ),
        "minimal_controlled_canary_admission_allowed": minimal_controlled_canary_allowed,
        "minimal_controlled_canary_executed": minimal_controlled_canary_executed,
        "minimal_controlled_canary_pass": controlled_canary_pass,
        "performance_claim_allowed": False,
        "stage_b_replay_executed": post_run_inputs_provided,
        "first_wave_expanded_metadata_replay_executed": bool(
            post_run_inputs_provided and expanded_execution_record_provided and first_wave_expanded_authorized
        ),
        "second_wave_expanded_metadata_replay_executed": bool(
            post_run_inputs_provided and expanded_execution_record_provided and second_wave_expanded_authorized
        ),
        "canonical_strict_metadata_replay_pass": post_run_pass,
        "failure_taxonomy": sorted(set(post_run_failures + controlled_canary_failures)),
        "post_run_metrics": {
            "cache_hit_rate": cache_hit_rate,
            "runtime_error_steps": runtime_error_steps,
            "strict_cache_miss_runtime_error_steps": strict_cache_miss_runtime_error_steps,
            "action_diff_trace_count": action_diff_trace_count,
            "action_diff_steps": action_diff_steps,
            "action_missing_step_count": action_missing_steps,
            "max_abs_delta": action_max_abs_delta,
            "runtime_provenance_record_count": runtime_record_count,
            "runtime_reason_missing_count": runtime_reason_missing_count,
            "unknown_post_guardrail_rewrite_count": unknown_post_guardrail_rewrite_count,
            "joint_prediction_missing_fields": joint_missing_fields,
            "joint_prediction_row_count": joint_row_count,
            "controlled_canary_cache_hit_rate": controlled_cache_hit_rate,
            "controlled_canary_runtime_error_steps": controlled_runtime_error_steps,
            "controlled_canary_strict_cache_miss_runtime_error_steps": controlled_strict_cache_miss_steps,
            "controlled_canary_unsafe_preferred_steps": controlled_unsafe_preferred_steps,
            "controlled_canary_unsafe_conflict_steps": controlled_unsafe_conflict_steps,
            "controlled_canary_unsafe_applied_steps": controlled_unsafe_applied_steps,
            "controlled_canary_applied_steps": controlled_applied_steps,
            "controlled_canary_strict_applied_steps": controlled_strict_applied_steps,
            "controlled_canary_action_changes_scope_ok": controlled_action_changes_scope_ok,
            "controlled_canary_hard_safety_non_regression": controlled_hard_safety_non_regression,
        },
        "checks": {
            "expert_distillation_resolved": expert_resolved,
            "mainline_diff_authorization_complete": authorization_complete,
            "protocol_isolation_pass": protocol_pass,
            "protocol_delta_explained": protocol_delta_explained,
            "protocol_delta_no_unknown": protocol_delta_no_unknown,
            "protocol_unknown_hunk_count": delta_unknown_count if delta_unknown_count is not None else "not_provided",
            "protocol_baseline_authorized": protocol_authorized,
            "protocol_ready_for_metadata_replay_planning": protocol_ready,
            "protocol_v1_preflight_ready": protocol_v1_preflight_ready,
            "protocol_authorization_decision": (
                protocol_authorization.get("recommended_decision", "not_provided")
                if isinstance(protocol_authorization, Mapping)
                else "not_provided"
            ),
            "protocol_user_decision_complete": protocol_user_decision_complete,
            "blocked_categories_resolved": blocked_categories_resolved,
            "blocked_categories_resolved_for_v1_preflight": blocked_categories_resolved_for_v1_preflight,
            "blocked_categories_resolved_for_protocol_v2": blocked_categories_resolved_for_protocol_v2,
            "safety_boundary_test_oracle_review_complete": safety_oracle_review_complete,
            "safety_boundary_test_oracle_changed": (
                bool(delta_classification.get("safety_boundary_test_oracle_changed", False))
                if isinstance(delta_classification, Mapping)
                else False
            ),
            "safety_boundary_test_oracle_hunk_count": (
                int(delta_classification.get("safety_boundary_test_oracle_hunk_count", 0) or 0)
                if isinstance(delta_classification, Mapping)
                else 0
            ),
            "old_vs_new_protocol_audit_design_ready": old_vs_new_design_ready,
            "old_vs_new_protocol_audit_ready": old_vs_new_audit_ready,
            "old_protocol_snapshot_available": protocol_v1_snapshot_available,
            "protocol_v1_snapshot_manifest_available": protocol_v1_snapshot_manifest_available,
            "default_path_evidence_pass": default_path_pass,
            "default_path_action_invariance_plan_ready": default_path_plan_ready,
            "cache_coverage_pass": cache_coverage_pass if isinstance(cache_coverage, Mapping) else "not_run",
            "cache_coverage_plan_ready": cache_plan_ready,
            "generic_cache_coverage_plan_ready": generic_cache_plan_ready,
            "cache_scenario_plan_ready": cache_scenario_plan_ready,
            "cache_scenario_plan_deduplicated": (
                bool(cache_scenario_plan.get("scenario_list_deduplicated", False))
                if isinstance(cache_scenario_plan, Mapping)
                else False
            ),
            "cache_coverage_claim_scope": (
                cache_scenario_plan.get("coverage_claim_scope", "not_provided")
                if isinstance(cache_scenario_plan, Mapping)
                else "not_provided"
            ),
            "cache_filename_seed_mismatch_explained": (
                bool(cache_scenario_plan.get("cache_filename_seed_mismatch_explained", False))
                if isinstance(cache_scenario_plan, Mapping)
                else False
            ),
            "expanded_cache_coverage_plan_ready": expanded_cache_coverage_plan_ready,
            "expanded_cache_coverage_ready": expanded_cache_coverage_ready,
            "strict_metadata_replay_run_plan_ready": strict_metadata_replay_run_plan_ready,
            "protocol_v1_overlay_available": protocol_v1_overlay_available,
            "protocol_v1_overlay_validated": protocol_v1_overlay_validated,
            "protocol_v1_overlay_root": (
                protocol_v1_overlay_preflight.get("protocol_v1_overlay_root", "not_provided")
                if isinstance(protocol_v1_overlay_preflight, Mapping)
                else "not_provided"
            ),
            "execution_authorization_packet_ready": execution_authorization_packet_ready,
            "baseline_trace_manifest_ready": baseline_trace_manifest_ready,
            "baseline_trace_qualified": baseline_trace_qualified,
            "action_diff_audit_status": action_diff_audit_status,
            "canonical_strict_metadata_replay_execution_request_ready": canonical_execution_request_ready,
            "two_stage_authorization_packet_ready": two_stage_authorization_packet_ready,
            "stage_b_authorization_request_ready": stage_b_authorization_request_ready,
            "stage_a_overlay_validated": stage_a_overlay_validated,
            "stage_a_overlay_build_validation_complete": stage_a_overlay_build_validation_complete,
            "stage_a_overlay_hash_match": stage_a_overlay_hash_match,
            "stage_b_user_authorized": stage_b_user_authorized,
            "canonical_metadata_replay_authorization_request_ready": canonical_metadata_replay_authorization_request_ready,
            "runtime_provenance_closure_status_provided": runtime_provenance_closure_status_provided,
            "runtime_provenance_closure_implementation_ready": runtime_provenance_closure_implementation_ready,
            "runtime_provenance_trace_export_implemented": runtime_provenance_trace_export_implemented,
            "runtime_provenance_audit_trace_jsonl_supported": runtime_provenance_audit_trace_jsonl_supported,
            "runtime_provenance_audit_trace_csv_supported": runtime_provenance_audit_trace_csv_supported,
            "stage_b_rerun_authorization_required": stage_b_rerun_authorization_required,
            "stage_b_rerun_authorized": stage_b_rerun_authorized,
            "first_wave_expanded_execution_record_provided": expanded_execution_record_provided,
            "first_wave_expanded_metadata_replay_authorized": first_wave_expanded_authorized,
            "first_wave_expanded_scope": expanded_scope,
            "first_wave_expanded_scenario_count": (
                len(expanded_execution_record.get("scenario_ids", []) or [])
                if expanded_execution_record_provided
                else 0
            ),
            "second_wave_expanded_execution_record_provided": expanded_execution_record_provided,
            "second_wave_expanded_metadata_replay_authorized": second_wave_expanded_authorized,
            "second_wave_expanded_scope": expanded_scope,
            "second_wave_expanded_scenario_count": (
                len(expanded_execution_record.get("scenario_ids", []) or [])
                if expanded_execution_record_provided and second_wave_expanded_authorized
                else 0
            ),
            "controlled_replay_admission_review_provided": controlled_admission_provided,
            "controlled_replay_admission_pass": (
                bool(controlled_replay_admission_review.get("minimal_controlled_canary_admission_pass", False))
                if controlled_admission_provided
                else False
            ),
            "minimal_controlled_canary_allowed": minimal_controlled_canary_allowed,
            "minimal_controlled_canary_executed": minimal_controlled_canary_executed,
            "minimal_controlled_canary_pass": controlled_canary_pass,
            "controlled_replay_admission_next_action": controlled_admission_next_action,
            "selected_cache_path": (
                cache_scenario_plan.get("selected_cache_path", "not_provided")
                if isinstance(cache_scenario_plan, Mapping)
                else "not_provided"
            ),
            "cache_fill_run": (
                bool(cache_coverage.get("cache_fill_run", False))
                if isinstance(cache_coverage, Mapping)
                else False
            ),
            "online_llm_called": (
                bool(cache_coverage.get("online_llm_called", False))
                if isinstance(cache_coverage, Mapping)
                else False
            ),
            "controller_sensitive_diff_present": any(
                "llm_agent.py" in str(path)
                for path in closure.get("blocking_paths", []) or []
            ),
            "default_path_call_count": int(default_path.get("default_path_call_count", 0) or 0),
        },
        "targets": {
            "action_diff_steps": 0,
            "max_abs_delta": 0,
            "runtime_provenance_record_count": ">0",
            "runtime_reason_missing_count": 0,
            "unknown_post_guardrail_rewrite_count": 0,
            "joint_prediction_missing_fields": 0,
        },
        "blocked_until": [
            item
            for item, ok in (
                ("strict metadata replay run plan can be drafted", strict_metadata_replay_planning_allowed),
                ("metadata replay execution requires separate authorization", effective_execution_allowed),
            )
            if not ok
        ],
        "planning_blocked_until": [
            item
            for item, ok in (
                ("expert_distillation clean or explicitly authorized", expert_resolved),
                ("protocol delta classification has no unknown hunks", protocol_delta_no_unknown),
                ("protocol v2 user decision complete", protocol_user_decision_complete),
                ("blocked protocol categories resolved for v1 preflight", blocked_categories_resolved_for_v1_preflight),
                ("safety-boundary test oracle review complete", safety_oracle_review_complete),
                ("protocol v1 snapshot manifest available", protocol_v1_snapshot_available),
                ("protocol v1 snapshot manifest is source-hash manifest", protocol_v1_snapshot_manifest_available),
                ("old-vs-new protocol audit design ready", old_vs_new_design_ready),
                ("old-vs-new protocol audit readiness pass", old_vs_new_audit_ready),
                ("default-path action invariance plan ready", default_path_plan_ready),
                ("cache coverage plan ready", cache_plan_ready),
                ("metadata replay cache/scenario plan ready", cache_scenario_plan_ready or not isinstance(cache_scenario_plan, Mapping)),
                ("cache coverage check pass", cache_coverage_pass),
            )
            if not ok
        ],
        "execution_blocked_until": [
            item
            for item, ok in (
                ("protocol v2 baseline authorization remains false", protocol_authorized),
                ("protocol v2 blocked categories resolved", blocked_categories_resolved_for_protocol_v2),
                ("default-path action invariance evidence pass", default_path_pass),
                ("mainline diff authorization complete", authorization_complete),
                ("strict metadata replay run plan ready", strict_metadata_replay_run_plan_ready),
                ("protocol v1 ephemeral overlay validated", protocol_v1_overlay_validated),
                ("strict metadata replay execution authorization packet ready", execution_authorization_packet_ready),
                ("canonical action-diff baseline trace manifest ready", baseline_trace_manifest_ready),
                ("canonical strict metadata replay execution request ready", canonical_execution_request_ready),
                ("strict metadata replay two-stage authorization packet ready", two_stage_authorization_packet_ready),
                ("canonical Stage-B authorization request ready", stage_b_authorization_request_ready),
                ("explicit Stage B user authorization for canonical strict metadata replay", stage_b_user_authorized),
                (
                    "explicit Stage-B rerun authorization after runtime provenance closure",
                    (not stage_b_rerun_authorization_required) or stage_b_rerun_authorized,
                ),
                ("separate metadata replay execution authorization", metadata_replay_execution_allowed),
            )
            if not ok
        ],
        "authorization_request_blocked_until": [
            item
            for item, ok in (
                ("strict metadata replay planning allowed", strict_metadata_replay_planning_allowed),
                ("strict metadata replay run plan ready", strict_metadata_replay_run_plan_ready),
                ("protocol v1 ephemeral overlay validated", protocol_v1_overlay_validated),
                ("strict metadata replay execution authorization packet ready", execution_authorization_packet_ready),
                ("canonical action-diff baseline trace manifest ready", baseline_trace_manifest_ready),
                ("canonical strict metadata replay execution request ready", canonical_execution_request_ready),
                ("strict metadata replay two-stage authorization packet ready", two_stage_authorization_packet_ready),
                ("canonical Stage-B authorization request ready", stage_b_authorization_request_ready),
            )
            if not ok
        ],
        "next_action": next_action_value,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Metadata Replay Readiness Checklist {str(report.get('schema_version', 'v14')).rsplit('_', 1)[-1]}",
        "",
        "- Mode: audit-only / readiness checklist",
        f"- Metadata replay allowed: {report.get('metadata_replay_allowed', False)}",
        f"- Strict metadata replay planning allowed: {report.get('strict_metadata_replay_planning_allowed', False)}",
        f"- Strict metadata replay run plan ready: {report.get('strict_metadata_replay_run_plan_ready', False)}",
        f"- Metadata replay execution allowed: {report.get('metadata_replay_execution_allowed', False)}",
        f"- Canonical metadata replay authorization request ready: {report.get('canonical_metadata_replay_authorization_request_ready', False)}",
        "- Controlled replay allowed: false",
        f"- Minimal controlled canary allowed: {report.get('minimal_controlled_canary_allowed', False)}",
        f"- Minimal controlled canary executed: {report.get('minimal_controlled_canary_executed', False)}",
        f"- Minimal controlled canary pass: {report.get('minimal_controlled_canary_pass', False)}",
        "- Performance claim allowed: false",
        f"- Stage-B replay executed: {report.get('stage_b_replay_executed', False)}",
        f"- Canonical strict metadata replay pass: {report.get('canonical_strict_metadata_replay_pass', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Checks",
        "",
        "| check | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("checks", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Planning Blocked Until", ""])
    for item in report.get("planning_blocked_until", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Execution Blocked Until", ""])
    for item in report.get("execution_blocked_until", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Authorization Request Blocked Until", ""])
    for item in report.get("authorization_request_blocked_until", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Failure Taxonomy", ""])
    for item in report.get("failure_taxonomy", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Post-Run Metrics", "", "| metric | value |", "| --- | --- |"])
    for key, value in dict(report.get("post_run_metrics", {})).items():
        lines.append(f"| {key} | `{value}` |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closure-json", required=True)
    parser.add_argument("--protocol-json", required=True)
    parser.add_argument("--default-path-json", required=True)
    parser.add_argument("--expert-status-json", default="")
    parser.add_argument("--default-path-plan-json", default="")
    parser.add_argument("--cache-plan-json", default="")
    parser.add_argument("--delta-classification-json", default="")
    parser.add_argument("--protocol-authorization-json", default="")
    parser.add_argument("--protocol-user-decision-json", default="")
    parser.add_argument("--blocked-categories-plan-json", default="")
    parser.add_argument("--safety-oracle-review-json", default="")
    parser.add_argument("--old-vs-new-design-json", default="")
    parser.add_argument("--old-vs-new-readiness-json", default="")
    parser.add_argument("--cache-scenario-plan-json", default="")
    parser.add_argument("--cache-coverage-json", default="")
    parser.add_argument("--expanded-cache-coverage-plan-json", default="")
    parser.add_argument("--strict-metadata-replay-run-plan-json", default="")
    parser.add_argument("--protocol-v1-overlay-preflight-json", default="")
    parser.add_argument("--execution-authorization-packet-json", default="")
    parser.add_argument("--baseline-trace-manifest-json", default="")
    parser.add_argument("--canonical-execution-request-json", default="")
    parser.add_argument("--two-stage-authorization-packet-json", default="")
    parser.add_argument("--stage-b-authorization-request-json", default="")
    parser.add_argument("--strict-metadata-replay-summary-json", default="")
    parser.add_argument("--trace-action-diff-audit-json", default="")
    parser.add_argument("--runtime-provenance-audit-json", default="")
    parser.add_argument("--joint-prediction-readiness-json", default="")
    parser.add_argument("--runtime-provenance-closure-status-json", default="")
    parser.add_argument("--expanded-execution-record-json", default="")
    parser.add_argument("--controlled-replay-admission-json", default="")
    parser.add_argument("--controlled-canary-execution-record-json", default="")
    parser.add_argument("--controlled-canary-summary-audit-json", default="")
    parser.add_argument("--controlled-canary-trace-audit-json", default="")
    parser.add_argument("--controlled-canary-strict-effect-audit-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        closure=_load(args.closure_json),
        protocol=_load(args.protocol_json),
        default_path=_load(args.default_path_json),
        expert_status=_load(args.expert_status_json) if args.expert_status_json else None,
        default_path_plan=_load(args.default_path_plan_json) if args.default_path_plan_json else None,
        cache_plan=_load(args.cache_plan_json) if args.cache_plan_json else None,
        delta_classification=_load(args.delta_classification_json) if args.delta_classification_json else None,
        protocol_authorization=_load(args.protocol_authorization_json) if args.protocol_authorization_json else None,
        protocol_user_decision=_load(args.protocol_user_decision_json) if args.protocol_user_decision_json else None,
        blocked_categories_plan=_load(args.blocked_categories_plan_json) if args.blocked_categories_plan_json else None,
        safety_oracle_review=_load(args.safety_oracle_review_json) if args.safety_oracle_review_json else None,
        old_vs_new_design=_load(args.old_vs_new_design_json) if args.old_vs_new_design_json else None,
        old_vs_new_readiness=_load(args.old_vs_new_readiness_json) if args.old_vs_new_readiness_json else None,
        cache_scenario_plan=_load(args.cache_scenario_plan_json) if args.cache_scenario_plan_json else None,
        cache_coverage=_load(args.cache_coverage_json) if args.cache_coverage_json else None,
        expanded_cache_coverage_plan=_load(args.expanded_cache_coverage_plan_json) if args.expanded_cache_coverage_plan_json else None,
        strict_metadata_replay_run_plan=_load(args.strict_metadata_replay_run_plan_json) if args.strict_metadata_replay_run_plan_json else None,
        protocol_v1_overlay_preflight=_load(args.protocol_v1_overlay_preflight_json) if args.protocol_v1_overlay_preflight_json else None,
        execution_authorization_packet=_load(args.execution_authorization_packet_json) if args.execution_authorization_packet_json else None,
        baseline_trace_manifest=_load(args.baseline_trace_manifest_json) if args.baseline_trace_manifest_json else None,
        canonical_execution_request=_load(args.canonical_execution_request_json) if args.canonical_execution_request_json else None,
        two_stage_authorization_packet=_load(args.two_stage_authorization_packet_json) if args.two_stage_authorization_packet_json else None,
        stage_b_authorization_request=_load(args.stage_b_authorization_request_json) if args.stage_b_authorization_request_json else None,
        strict_metadata_replay_summary=_load(args.strict_metadata_replay_summary_json) if args.strict_metadata_replay_summary_json else None,
        trace_action_diff_audit=_load(args.trace_action_diff_audit_json) if args.trace_action_diff_audit_json else None,
        runtime_provenance_audit=_load(args.runtime_provenance_audit_json) if args.runtime_provenance_audit_json else None,
        joint_prediction_readiness=_load(args.joint_prediction_readiness_json) if args.joint_prediction_readiness_json else None,
        runtime_provenance_closure_status=(
            _load(args.runtime_provenance_closure_status_json)
            if args.runtime_provenance_closure_status_json
            else None
        ),
        expanded_execution_record=(
            _load(args.expanded_execution_record_json)
            if args.expanded_execution_record_json
            else None
        ),
        controlled_replay_admission_review=(
            _load(args.controlled_replay_admission_json)
            if args.controlled_replay_admission_json
            else None
        ),
        controlled_canary_execution_record=(
            _load(args.controlled_canary_execution_record_json)
            if args.controlled_canary_execution_record_json
            else None
        ),
        controlled_canary_summary_audit=(
            _load(args.controlled_canary_summary_audit_json)
            if args.controlled_canary_summary_audit_json
            else None
        ),
        controlled_canary_trace_audit=(
            _load(args.controlled_canary_trace_audit_json)
            if args.controlled_canary_trace_audit_json
            else None
        ),
        controlled_canary_strict_effect_audit=(
            _load(args.controlled_canary_strict_effect_audit_json)
            if args.controlled_canary_strict_effect_audit_json
            else None
        ),
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
    print(f"metadata_replay_allowed={report['metadata_replay_allowed']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
