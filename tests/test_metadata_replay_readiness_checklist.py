import unittest

from gl_gym.experiments.metadata_replay_readiness_checklist import build_report


class TestMetadataReplayReadinessChecklist(unittest.TestCase):
    def test_protocol_isolation_blocks_before_default_path(self):
        report = build_report(
            closure={
                "mainline_diff_authorization_complete": False,
                "blocking_paths": ["gl_gym/agent/llm_agent.py"],
            },
            protocol={"protocol_isolation_pass": False},
            default_path={"default_path_evidence_pass": False, "default_path_call_count": 3},
        )

        self.assertFalse(report["metadata_replay_allowed"])
        self.assertEqual(report["next_action"], "protocol_delta_classification_required")
        self.assertIn("protocol delta classification has no unknown hunks", report["planning_blocked_until"])
        self.assertIn("protocol v1 snapshot manifest available", report["planning_blocked_until"])

    def test_default_path_blocks_after_protocol_passes(self):
        report = build_report(
            closure={
                "mainline_diff_authorization_complete": False,
                "blocking_paths": ["gl_gym/agent/llm_agent.py"],
            },
            protocol={
                "protocol_isolation_pass": True,
                "requires_protocol_baseline_authorization": False,
                "protocol_delta_explained": True,
            },
            default_path={"default_path_evidence_pass": False, "default_path_call_count": 3},
            delta_classification={"unknown_hunk_count": 0},
            protocol_user_decision={"protocol_user_decision_complete": True},
            blocked_categories_plan={"blocked_categories_resolved": True},
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            old_vs_new_design={"design_ready": True},
            old_vs_new_readiness={
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
        )

        self.assertFalse(report["metadata_replay_allowed"])
        self.assertEqual(report["next_action"], "default_path_action_invariance_plan_required")

    def test_all_readiness_inputs_allow_metadata_replay_planning_only(self):
        report = build_report(
            closure={
                "mainline_diff_authorization_complete": True,
                "blocking_paths": [],
            },
            protocol={
                "protocol_isolation_pass": True,
                "requires_protocol_baseline_authorization": False,
                "protocol_delta_explained": True,
            },
            default_path={"default_path_evidence_pass": True, "default_path_call_count": 0},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            delta_classification={"unknown_hunk_count": 0},
            protocol_user_decision={"protocol_user_decision_complete": True},
            blocked_categories_plan={"blocked_categories_resolved": True},
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            old_vs_new_design={"design_ready": True},
            old_vs_new_readiness={
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
            cache_coverage={"cache_coverage_pass": True},
        )

        self.assertFalse(report["metadata_replay_allowed"])
        self.assertTrue(report["strict_metadata_replay_planning_allowed"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertEqual(report["next_action"], "strict_metadata_replay_plan_can_be_drafted")

    def test_protocol_baseline_authorization_blocks_execution_not_v1_planning(self):
        report = build_report(
            closure={"mainline_diff_authorization_complete": True, "blocking_paths": []},
            protocol={
                "protocol_isolation_pass": True,
                "requires_protocol_baseline_authorization": True,
                "protocol_baseline_authorized": False,
                "protocol_delta_explained": True,
            },
            default_path={"default_path_evidence_pass": True, "default_path_call_count": 0},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            delta_classification={"unknown_hunk_count": 0},
            protocol_authorization={
                "recommended_decision": "partial_accept_review_required",
                "protocol_baseline_authorized": False,
                "requires_user_authorization": True,
            },
            protocol_user_decision={"protocol_user_decision_complete": True},
            blocked_categories_plan={"blocked_categories_resolved": True},
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            old_vs_new_design={"design_ready": True},
            old_vs_new_readiness={
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
            cache_coverage={"cache_coverage_pass": True},
        )

        self.assertTrue(report["strict_metadata_replay_planning_allowed"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertEqual(report["next_action"], "strict_metadata_replay_plan_can_be_drafted")
        self.assertIn("protocol v2 baseline authorization remains false", report["execution_blocked_until"])

    def test_unknown_protocol_delta_blocks_even_with_authorization_packet(self):
        report = build_report(
            closure={"mainline_diff_authorization_complete": True, "blocking_paths": []},
            protocol={
                "protocol_isolation_pass": True,
                "requires_protocol_baseline_authorization": False,
                "protocol_delta_explained": False,
            },
            default_path={"default_path_evidence_pass": True, "default_path_call_count": 0},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            delta_classification={"unknown_hunk_count": 1},
            protocol_authorization={"protocol_baseline_authorized": False},
            protocol_user_decision={"protocol_user_decision_complete": True},
            blocked_categories_plan={"blocked_categories_resolved": True},
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            old_vs_new_design={"design_ready": True},
            old_vs_new_readiness={
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
            cache_coverage={"cache_coverage_pass": True},
        )

        self.assertFalse(report["strict_metadata_replay_planning_allowed"])
        self.assertEqual(report["next_action"], "protocol_delta_classification_required")

    def test_old_vs_new_design_blocks_after_protocol_ready(self):
        report = build_report(
            closure={"mainline_diff_authorization_complete": True, "blocking_paths": []},
            protocol={
                "protocol_isolation_pass": True,
                "requires_protocol_baseline_authorization": True,
                "protocol_baseline_authorized": True,
                "protocol_delta_explained": True,
            },
            default_path={"default_path_evidence_pass": True, "default_path_call_count": 0},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            delta_classification={
                "unknown_hunk_count": 0,
                "safety_boundary_test_oracle_changed": True,
                "safety_boundary_test_oracle_hunk_count": 4,
            },
            protocol_authorization={
                "recommended_decision": "accept_full_protocol_v2",
                "protocol_baseline_authorized": True,
                "requires_user_authorization": True,
            },
            protocol_user_decision={"protocol_user_decision_complete": True},
            blocked_categories_plan={"blocked_categories_resolved": True},
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            cache_coverage={"cache_coverage_pass": True},
        )

        self.assertFalse(report["strict_metadata_replay_planning_allowed"])
        self.assertEqual(report["next_action"], "old_vs_new_protocol_audit_design_required")
        self.assertTrue(report["checks"]["safety_boundary_test_oracle_changed"])
        self.assertEqual(report["checks"]["safety_boundary_test_oracle_hunk_count"], 4)

    def test_old_vs_new_readiness_blocks_after_design_ready(self):
        report = build_report(
            closure={"mainline_diff_authorization_complete": True, "blocking_paths": []},
            protocol={
                "protocol_isolation_pass": True,
                "requires_protocol_baseline_authorization": True,
                "protocol_baseline_authorized": True,
                "protocol_delta_explained": True,
            },
            default_path={"default_path_evidence_pass": True, "default_path_call_count": 0},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            delta_classification={"unknown_hunk_count": 0},
            protocol_authorization={
                "recommended_decision": "accept_full_protocol_v2",
                "protocol_baseline_authorized": True,
                "requires_user_authorization": True,
            },
            protocol_user_decision={"protocol_user_decision_complete": True},
            blocked_categories_plan={"blocked_categories_resolved": True},
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            old_vs_new_design={"design_ready": True},
            old_vs_new_readiness={"old_vs_new_protocol_audit_ready": False, "old_protocol_snapshot_available": False},
            cache_coverage={"cache_coverage_pass": True},
        )

        self.assertFalse(report["strict_metadata_replay_planning_allowed"])
        self.assertEqual(report["next_action"], "old_vs_new_protocol_audit_readiness_required")
        self.assertFalse(report["checks"]["old_protocol_snapshot_available"])

    def test_v1_preflight_allows_run_plan_but_not_execution(self):
        report = build_report(
            closure={
                "mainline_diff_authorization_complete": False,
                "blocking_paths": ["gl_gym/agent/llm_agent.py"],
            },
            protocol={
                "protocol_isolation_pass": False,
                "requires_protocol_baseline_authorization": True,
                "protocol_baseline_authorized": False,
                "protocol_delta_explained": True,
            },
            default_path={"default_path_evidence_pass": False, "default_path_call_count": 3},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            cache_scenario_plan={
                "cache_scenario_plan_ready": True,
                "selected_cache_path": "gl_gym/result/plan_cache/holdout_seed42_43_hot_dry_qwen_20260517_merged.json",
            },
            delta_classification={"unknown_hunk_count": 0},
            protocol_authorization={
                "recommended_decision": "partial_accept_review_required",
                "protocol_baseline_authorized": False,
                "requires_user_authorization": True,
            },
            protocol_user_decision={
                "protocol_user_decision_complete": True,
                "selected_decision": "partial_accept_protocol_v2",
            },
            blocked_categories_plan={
                "blocked_categories_resolved": False,
                "blocked_categories_resolved_for_v1_preflight": True,
                "blocked_categories_resolved_for_protocol_v2": False,
            },
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            old_vs_new_design={"design_ready": True},
            old_vs_new_readiness={
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
            cache_coverage={"cache_coverage_pass": True, "cache_fill_run": False, "online_llm_called": False},
        )

        self.assertEqual(report["schema_version"], "metadata_replay_readiness_checklist_v14")
        self.assertTrue(report["strict_metadata_replay_planning_allowed"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertFalse(report["checks"]["protocol_baseline_authorized"])
        self.assertTrue(report["checks"]["blocked_categories_resolved_for_v1_preflight"])
        self.assertFalse(report["checks"]["blocked_categories_resolved_for_protocol_v2"])

    def test_run_plan_ready_changes_next_action_but_not_execution_permission(self):
        report = build_report(
            closure={
                "mainline_diff_authorization_complete": False,
                "blocking_paths": ["gl_gym/agent/llm_agent.py"],
            },
            protocol={
                "protocol_isolation_pass": False,
                "requires_protocol_baseline_authorization": True,
                "protocol_baseline_authorized": False,
                "protocol_delta_explained": True,
            },
            default_path={"default_path_evidence_pass": False, "default_path_call_count": 3},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            cache_scenario_plan={
                "cache_scenario_plan_ready": True,
                "scenario_list_deduplicated": True,
                "coverage_claim_scope": "canonical_failure_only",
                "cache_filename_seed_mismatch_explained": True,
                "selected_cache_path": "cache.json",
            },
            delta_classification={"unknown_hunk_count": 0},
            protocol_authorization={
                "recommended_decision": "partial_accept_review_required",
                "protocol_baseline_authorized": False,
                "requires_user_authorization": True,
            },
            protocol_user_decision={
                "protocol_user_decision_complete": True,
                "selected_decision": "partial_accept_protocol_v2",
            },
            blocked_categories_plan={
                "blocked_categories_resolved": False,
                "blocked_categories_resolved_for_v1_preflight": True,
                "blocked_categories_resolved_for_protocol_v2": False,
            },
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            old_vs_new_design={"design_ready": True},
            old_vs_new_readiness={
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
            cache_coverage={"cache_coverage_pass": True, "cache_fill_run": False, "online_llm_called": False},
            expanded_cache_coverage_plan={
                "expanded_cache_coverage_plan_ready": True,
                "expanded_cache_coverage_ready": False,
            },
            strict_metadata_replay_run_plan={"strict_metadata_replay_run_plan_ready": True},
        )

        self.assertTrue(report["strict_metadata_replay_planning_allowed"])
        self.assertTrue(report["strict_metadata_replay_run_plan_ready"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertEqual(report["next_action"], "protocol_v1_ephemeral_overlay_preflight_required")
        self.assertIn("separate metadata replay execution authorization", report["execution_blocked_until"])
        self.assertIn("protocol v1 ephemeral overlay validated", report["authorization_request_blocked_until"])

    def test_overlay_and_request_require_two_stage_packet_before_authorization_ready(self):
        report = build_report(
            closure={
                "mainline_diff_authorization_complete": False,
                "blocking_paths": ["gl_gym/agent/llm_agent.py"],
            },
            protocol={
                "protocol_isolation_pass": False,
                "requires_protocol_baseline_authorization": True,
                "protocol_baseline_authorized": False,
                "protocol_delta_explained": True,
            },
            default_path={"default_path_evidence_pass": False, "default_path_call_count": 3},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            cache_scenario_plan={
                "cache_scenario_plan_ready": True,
                "scenario_list_deduplicated": True,
                "coverage_claim_scope": "canonical_failure_only",
                "cache_filename_seed_mismatch_explained": True,
                "selected_cache_path": "cache.json",
            },
            delta_classification={"unknown_hunk_count": 0},
            protocol_authorization={
                "recommended_decision": "partial_accept_review_required",
                "protocol_baseline_authorized": False,
                "requires_user_authorization": True,
            },
            protocol_user_decision={
                "protocol_user_decision_complete": True,
                "selected_decision": "partial_accept_protocol_v2",
            },
            blocked_categories_plan={
                "blocked_categories_resolved": False,
                "blocked_categories_resolved_for_v1_preflight": True,
                "blocked_categories_resolved_for_protocol_v2": False,
            },
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            old_vs_new_design={"design_ready": True},
            old_vs_new_readiness={
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
            cache_coverage={"cache_coverage_pass": True, "cache_fill_run": False, "online_llm_called": False},
            expanded_cache_coverage_plan={
                "expanded_cache_coverage_plan_ready": True,
                "expanded_cache_coverage_ready": False,
            },
            strict_metadata_replay_run_plan={"strict_metadata_replay_run_plan_ready": True},
            protocol_v1_overlay_preflight={
                "protocol_v1_overlay_available": True,
                "protocol_v1_overlay_validated": True,
                "protocol_v1_overlay_root": "overlay",
            },
            execution_authorization_packet={"authorization_packet_ready": True},
            baseline_trace_manifest={"baseline_trace_manifest_ready": True, "baseline_trace_qualified": True},
            canonical_execution_request={"canonical_strict_metadata_replay_execution_request_ready": True},
        )

        self.assertFalse(report["canonical_metadata_replay_authorization_request_ready"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertEqual(
            report["next_action"],
            "strict_metadata_replay_two_stage_authorization_packet_required",
        )
        self.assertTrue(report["checks"]["baseline_trace_manifest_ready"])
        self.assertTrue(report["checks"]["canonical_strict_metadata_replay_execution_request_ready"])
        self.assertFalse(report["checks"]["two_stage_authorization_packet_ready"])

    def test_two_stage_packet_requires_stage_b_authorization_request_before_ready(self):
        report = build_report(
            closure={
                "mainline_diff_authorization_complete": False,
                "blocking_paths": ["gl_gym/agent/llm_agent.py"],
            },
            protocol={
                "protocol_isolation_pass": False,
                "requires_protocol_baseline_authorization": True,
                "protocol_baseline_authorized": False,
                "protocol_delta_explained": True,
            },
            default_path={"default_path_evidence_pass": False, "default_path_call_count": 3},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            cache_scenario_plan={
                "cache_scenario_plan_ready": True,
                "scenario_list_deduplicated": True,
                "coverage_claim_scope": "canonical_failure_only",
                "cache_filename_seed_mismatch_explained": True,
                "selected_cache_path": "cache.json",
            },
            delta_classification={"unknown_hunk_count": 0},
            protocol_authorization={
                "recommended_decision": "partial_accept_review_required",
                "protocol_baseline_authorized": False,
                "requires_user_authorization": True,
            },
            protocol_user_decision={
                "protocol_user_decision_complete": True,
                "selected_decision": "partial_accept_protocol_v2",
            },
            blocked_categories_plan={
                "blocked_categories_resolved": False,
                "blocked_categories_resolved_for_v1_preflight": True,
                "blocked_categories_resolved_for_protocol_v2": False,
            },
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            old_vs_new_design={"design_ready": True},
            old_vs_new_readiness={
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
            cache_coverage={"cache_coverage_pass": True, "cache_fill_run": False, "online_llm_called": False},
            expanded_cache_coverage_plan={
                "expanded_cache_coverage_plan_ready": True,
                "expanded_cache_coverage_ready": False,
            },
            strict_metadata_replay_run_plan={"strict_metadata_replay_run_plan_ready": True},
            protocol_v1_overlay_preflight={
                "protocol_v1_overlay_available": True,
                "protocol_v1_overlay_validated": True,
                "protocol_v1_overlay_root": "overlay",
            },
            execution_authorization_packet={"authorization_packet_ready": True},
            baseline_trace_manifest={"baseline_trace_manifest_ready": True, "baseline_trace_qualified": True},
            canonical_execution_request={"canonical_strict_metadata_replay_execution_request_ready": True},
            two_stage_authorization_packet={
                "two_stage_authorization_packet_ready": True,
                "stage_a": {
                    "overlay_validated": True,
                    "overlay_build_validation_complete": True,
                    "overlay_hash_match": True,
                },
                "stage_b": {"canonical_replay_user_authorized": False},
            },
        )

        self.assertEqual(report["schema_version"], "metadata_replay_readiness_checklist_v14")
        self.assertFalse(report["canonical_metadata_replay_authorization_request_ready"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertTrue(report["checks"]["stage_a_overlay_validated"])
        self.assertFalse(report["checks"]["stage_b_user_authorized"])
        self.assertFalse(report["checks"]["stage_b_authorization_request_ready"])
        self.assertEqual(
            report["next_action"],
            "canonical_stage_b_authorization_request_required",
        )

    def test_stage_b_request_ready_without_user_authorization_blocks_execution(self):
        base_kwargs = {
            "closure": {
                "mainline_diff_authorization_complete": False,
                "blocking_paths": ["gl_gym/agent/llm_agent.py"],
            },
            "protocol": {
                "protocol_isolation_pass": False,
                "requires_protocol_baseline_authorization": True,
                "protocol_baseline_authorized": False,
                "protocol_delta_explained": True,
            },
            "default_path": {"default_path_evidence_pass": False, "default_path_call_count": 3},
            "expert_status": {"hard_blocker": False},
            "default_path_plan": {"plan_ready": True},
            "cache_plan": {"schema_version": "cache_coverage_plan_v1"},
            "cache_scenario_plan": {"cache_scenario_plan_ready": True},
            "delta_classification": {"unknown_hunk_count": 0},
            "protocol_user_decision": {"protocol_user_decision_complete": True},
            "blocked_categories_plan": {
                "blocked_categories_resolved_for_v1_preflight": True,
                "blocked_categories_resolved_for_protocol_v2": False,
            },
            "safety_oracle_review": {"safety_boundary_test_oracle_review_complete": True},
            "old_vs_new_design": {"design_ready": True},
            "old_vs_new_readiness": {
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
            "cache_coverage": {"cache_coverage_pass": True},
            "strict_metadata_replay_run_plan": {"strict_metadata_replay_run_plan_ready": True},
            "protocol_v1_overlay_preflight": {
                "protocol_v1_overlay_available": True,
                "protocol_v1_overlay_validated": True,
            },
            "execution_authorization_packet": {"authorization_packet_ready": True},
            "baseline_trace_manifest": {"baseline_trace_manifest_ready": True, "baseline_trace_qualified": True},
            "canonical_execution_request": {"canonical_strict_metadata_replay_execution_request_ready": True},
            "two_stage_authorization_packet": {
                "two_stage_authorization_packet_ready": True,
                "stage_a": {
                    "overlay_validated": True,
                    "overlay_build_validation_complete": True,
                    "overlay_hash_match": True,
                },
                "stage_b": {"canonical_replay_user_authorized": False},
            },
        }

        report = build_report(
            **base_kwargs,
            stage_b_authorization_request={
                "stage_b_authorization_request_ready": True,
                "stage_b_user_authorized": False,
                "metadata_replay_execution_allowed": False,
            },
        )

        self.assertTrue(report["canonical_metadata_replay_authorization_request_ready"])
        self.assertTrue(report["checks"]["stage_b_authorization_request_ready"])
        self.assertFalse(report["checks"]["stage_b_user_authorized"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertEqual(
            report["next_action"],
            "await_explicit_stage_b_user_authorization_for_canonical_strict_metadata_replay",
        )

        authorized = build_report(
            **base_kwargs,
            stage_b_authorization_request={
                "stage_b_authorization_request_ready": True,
                "stage_b_user_authorized": True,
                "metadata_replay_execution_allowed": True,
            },
        )

        self.assertTrue(authorized["metadata_replay_execution_allowed"])
        self.assertTrue(authorized["metadata_replay_allowed"])
        self.assertEqual(authorized["next_action"], "execute_single_canonical_strict_metadata_replay")

        post_run = build_report(
            **base_kwargs,
            stage_b_authorization_request={
                "stage_b_authorization_request_ready": True,
                "stage_b_user_authorized": True,
                "metadata_replay_execution_allowed": True,
            },
            strict_metadata_replay_summary={
                "aggregate": {"decision": "pass"},
                "rows": [
                    {
                        "plan_cache_enabled_steps": 240,
                        "plan_cache_hit_steps": 240,
                        "runtime_error_steps": 0,
                        "strict_cache_miss_runtime_error_steps": 0,
                    }
                ],
            },
            trace_action_diff_audit={
                "trace_count": 1,
                "action_diff_steps": 0,
                "missing_step_count": 0,
                "max_abs_delta": 0,
            },
            runtime_provenance_audit={
                "record_count": 0,
                "runtime_reason_missing_count": 8,
                "unknown_post_guardrail_rewrite_count": 8,
            },
            joint_prediction_readiness={
                "ready_for_shadow_audit": True,
                "row_count": 240,
                "missing_field_counts": {},
            },
        )

        self.assertEqual(post_run["schema_version"], "metadata_replay_readiness_checklist_v15")
        self.assertTrue(post_run["stage_b_replay_executed"])
        self.assertFalse(post_run["metadata_replay_execution_allowed"])
        self.assertFalse(post_run["canonical_strict_metadata_replay_pass"])
        self.assertIn("runtime_provenance_missing", post_run["failure_taxonomy"])
        self.assertEqual(post_run["next_action"], "stage_b_failure_diagnosis_required")

    def test_ready_packet_without_baseline_manifest_blocks_final_request(self):
        report = build_report(
            closure={
                "mainline_diff_authorization_complete": False,
                "blocking_paths": ["gl_gym/agent/llm_agent.py"],
            },
            protocol={
                "protocol_isolation_pass": False,
                "requires_protocol_baseline_authorization": True,
                "protocol_baseline_authorized": False,
                "protocol_delta_explained": True,
            },
            default_path={"default_path_evidence_pass": False, "default_path_call_count": 3},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            cache_scenario_plan={"cache_scenario_plan_ready": True},
            delta_classification={"unknown_hunk_count": 0},
            protocol_user_decision={"protocol_user_decision_complete": True},
            blocked_categories_plan={
                "blocked_categories_resolved_for_v1_preflight": True,
                "blocked_categories_resolved_for_protocol_v2": False,
            },
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            old_vs_new_design={"design_ready": True},
            old_vs_new_readiness={
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
            cache_coverage={"cache_coverage_pass": True},
            strict_metadata_replay_run_plan={"strict_metadata_replay_run_plan_ready": True},
            protocol_v1_overlay_preflight={
                "protocol_v1_overlay_available": True,
                "protocol_v1_overlay_validated": True,
            },
            execution_authorization_packet={"authorization_packet_ready": True},
        )

        self.assertFalse(report["canonical_metadata_replay_authorization_request_ready"])
        self.assertEqual(report["next_action"], "canonical_action_diff_baseline_trace_manifest_required")
        self.assertIn(
            "canonical action-diff baseline trace manifest ready",
            report["authorization_request_blocked_until"],
        )

    def test_runtime_provenance_closure_refresh_requires_stage_b_rerun_authorization(self):
        report = build_report(
            closure={
                "mainline_diff_authorization_complete": False,
                "blocking_paths": ["gl_gym/agent/llm_agent.py"],
            },
            protocol={
                "protocol_isolation_pass": False,
                "requires_protocol_baseline_authorization": True,
                "protocol_baseline_authorized": False,
                "protocol_delta_explained": True,
            },
            default_path={"default_path_evidence_pass": False, "default_path_call_count": 3},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            cache_scenario_plan={
                "cache_scenario_plan_ready": True,
                "coverage_claim_scope": "canonical_failure_only",
                "selected_cache_path": "cache.json",
            },
            delta_classification={"unknown_hunk_count": 0},
            protocol_user_decision={"protocol_user_decision_complete": True},
            blocked_categories_plan={
                "blocked_categories_resolved_for_v1_preflight": True,
                "blocked_categories_resolved_for_protocol_v2": False,
            },
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            old_vs_new_design={"design_ready": True},
            old_vs_new_readiness={
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
            cache_coverage={"cache_coverage_pass": True, "cache_fill_run": False, "online_llm_called": False},
            strict_metadata_replay_run_plan={"strict_metadata_replay_run_plan_ready": True},
            protocol_v1_overlay_preflight={
                "protocol_v1_overlay_available": True,
                "protocol_v1_overlay_validated": True,
            },
            execution_authorization_packet={"authorization_packet_ready": True},
            baseline_trace_manifest={"baseline_trace_manifest_ready": True, "baseline_trace_qualified": True},
            canonical_execution_request={"canonical_strict_metadata_replay_execution_request_ready": True},
            two_stage_authorization_packet={
                "two_stage_authorization_packet_ready": True,
                "stage_a": {
                    "overlay_validated": True,
                    "overlay_build_validation_complete": True,
                    "overlay_hash_match": True,
                },
                "stage_b": {"canonical_replay_user_authorized": True},
            },
            stage_b_authorization_request={
                "stage_b_authorization_request_ready": True,
                "stage_b_user_authorized": True,
                "metadata_replay_execution_allowed": True,
            },
            runtime_provenance_closure_status={
                "runtime_provenance_closure_implementation_ready": True,
                "runtime_provenance_trace_export_implemented": True,
                "audit_supports_trace_jsonl": True,
                "audit_supports_trace_csv": True,
                "stage_b_rerun_authorization_required": True,
            },
        )

        self.assertEqual(report["schema_version"], "metadata_replay_readiness_checklist_v16")
        self.assertTrue(report["checks"]["runtime_provenance_closure_implementation_ready"])
        self.assertTrue(report["checks"]["runtime_provenance_trace_export_implemented"])
        self.assertTrue(report["checks"]["stage_b_rerun_authorization_required"])
        self.assertFalse(report["checks"]["stage_b_rerun_authorized"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertEqual(
            report["next_action"],
            "await_explicit_stage_b_rerun_authorization_after_runtime_provenance_closure",
        )
        self.assertIn(
            "explicit Stage-B rerun authorization after runtime provenance closure",
            report["execution_blocked_until"],
        )

    def test_runtime_provenance_closure_rerun_post_run_uses_v17(self):
        base_kwargs = {
            "closure": {
                "mainline_diff_authorization_complete": False,
                "blocking_paths": ["gl_gym/agent/llm_agent.py"],
            },
            "protocol": {
                "protocol_isolation_pass": False,
                "requires_protocol_baseline_authorization": True,
                "protocol_baseline_authorized": False,
                "protocol_delta_explained": True,
            },
            "default_path": {"default_path_evidence_pass": False, "default_path_call_count": 3},
            "expert_status": {"hard_blocker": False},
            "default_path_plan": {"plan_ready": True},
            "cache_plan": {"schema_version": "cache_coverage_plan_v1"},
            "cache_scenario_plan": {
                "cache_scenario_plan_ready": True,
                "coverage_claim_scope": "canonical_failure_only",
                "selected_cache_path": "cache.json",
            },
            "delta_classification": {"unknown_hunk_count": 0},
            "protocol_user_decision": {"protocol_user_decision_complete": True},
            "blocked_categories_plan": {
                "blocked_categories_resolved_for_v1_preflight": True,
                "blocked_categories_resolved_for_protocol_v2": False,
            },
            "safety_oracle_review": {"safety_boundary_test_oracle_review_complete": True},
            "old_vs_new_design": {"design_ready": True},
            "old_vs_new_readiness": {
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
            "cache_coverage": {"cache_coverage_pass": True, "cache_fill_run": False, "online_llm_called": False},
            "strict_metadata_replay_run_plan": {"strict_metadata_replay_run_plan_ready": True},
            "protocol_v1_overlay_preflight": {
                "protocol_v1_overlay_available": True,
                "protocol_v1_overlay_validated": True,
            },
            "execution_authorization_packet": {"authorization_packet_ready": True},
            "baseline_trace_manifest": {"baseline_trace_manifest_ready": True, "baseline_trace_qualified": True},
            "canonical_execution_request": {"canonical_strict_metadata_replay_execution_request_ready": True},
            "two_stage_authorization_packet": {
                "two_stage_authorization_packet_ready": True,
                "stage_a": {
                    "overlay_validated": True,
                    "overlay_build_validation_complete": True,
                    "overlay_hash_match": True,
                },
                "stage_b": {"canonical_replay_user_authorized": True},
            },
            "stage_b_authorization_request": {
                "stage_b_authorization_request_ready": True,
                "stage_b_user_authorized": True,
                "stage_b_rerun_authorized": True,
                "metadata_replay_execution_allowed": True,
            },
            "runtime_provenance_closure_status": {
                "runtime_provenance_closure_implementation_ready": True,
                "runtime_provenance_trace_export_implemented": True,
                "audit_supports_trace_jsonl": True,
                "audit_supports_trace_csv": True,
                "stage_b_rerun_authorization_required": True,
                "stage_b_rerun_authorized": False,
            },
        }
        report = build_report(
            **base_kwargs,
            strict_metadata_replay_summary={
                "aggregate": {"decision": "pass"},
                "rows": [
                    {
                        "plan_cache_enabled_steps": 240,
                        "plan_cache_hit_steps": 240,
                        "runtime_error_steps": 0,
                        "strict_cache_miss_runtime_error_steps": 0,
                    }
                ],
            },
            trace_action_diff_audit={
                "trace_count": 1,
                "action_diff_steps": 0,
                "missing_step_count": 0,
                "max_abs_delta": 0,
            },
            runtime_provenance_audit={
                "record_count": 258,
                "runtime_reason_missing_count": 0,
                "unknown_post_guardrail_rewrite_count": 0,
            },
            joint_prediction_readiness={
                "ready_for_shadow_audit": True,
                "row_count": 240,
                "missing_field_counts": {},
            },
        )

        self.assertEqual(report["schema_version"], "metadata_replay_readiness_checklist_v17")
        self.assertTrue(report["checks"]["stage_b_rerun_authorized"])
        self.assertTrue(report["canonical_strict_metadata_replay_pass"])
        self.assertEqual(report["failure_taxonomy"], [])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertEqual(report["next_action"], "expanded_metadata_coverage_planning_only")

    def test_first_wave_expanded_post_run_uses_v18(self):
        report = build_report(
            closure={
                "mainline_diff_authorization_complete": False,
                "blocking_paths": ["gl_gym/agent/llm_agent.py"],
            },
            protocol={
                "protocol_isolation_pass": False,
                "requires_protocol_baseline_authorization": True,
                "protocol_baseline_authorized": False,
                "protocol_delta_explained": True,
            },
            default_path={"default_path_evidence_pass": False, "default_path_call_count": 3},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            cache_scenario_plan={"cache_scenario_plan_ready": True, "coverage_claim_scope": "canonical_failure_only"},
            delta_classification={"unknown_hunk_count": 0},
            protocol_user_decision={"protocol_user_decision_complete": True},
            blocked_categories_plan={
                "blocked_categories_resolved_for_v1_preflight": True,
                "blocked_categories_resolved_for_protocol_v2": False,
            },
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": True},
            old_vs_new_design={"design_ready": True},
            old_vs_new_readiness={
                "old_vs_new_protocol_audit_ready": True,
                "old_protocol_snapshot_available": True,
                "protocol_v1_snapshot_manifest_available": True,
            },
            cache_coverage={"cache_coverage_pass": True, "cache_fill_run": False, "online_llm_called": False},
            strict_metadata_replay_run_plan={"strict_metadata_replay_run_plan_ready": True},
            protocol_v1_overlay_preflight={
                "protocol_v1_overlay_available": True,
                "protocol_v1_overlay_validated": True,
            },
            execution_authorization_packet={"authorization_packet_ready": True},
            baseline_trace_manifest={"baseline_trace_manifest_ready": True, "baseline_trace_qualified": True},
            canonical_execution_request={"canonical_strict_metadata_replay_execution_request_ready": True},
            two_stage_authorization_packet={
                "two_stage_authorization_packet_ready": True,
                "stage_a": {
                    "overlay_validated": True,
                    "overlay_build_validation_complete": True,
                    "overlay_hash_match": True,
                },
                "stage_b": {"canonical_replay_user_authorized": True},
            },
            stage_b_authorization_request={
                "stage_b_authorization_request_ready": True,
                "stage_b_user_authorized": True,
                "metadata_replay_execution_allowed": True,
            },
            expanded_execution_record={
                "first_wave_expanded_metadata_replay_authorized": True,
                "scope": "first_wave_expanded_metadata_coverage",
                "scenario_ids": ["y2010_d120_s44_n240", "y2010_d180_s44_n240"],
            },
            strict_metadata_replay_summary={
                "aggregate": {"decision": "pass"},
                "rows": [
                    {
                        "plan_cache_enabled_steps": 240,
                        "plan_cache_hit_steps": 240,
                        "runtime_error_steps": 0,
                        "strict_cache_miss_runtime_error_steps": 0,
                    },
                    {
                        "plan_cache_enabled_steps": 240,
                        "plan_cache_hit_steps": 240,
                        "runtime_error_steps": 0,
                        "strict_cache_miss_runtime_error_steps": 0,
                    },
                ],
            },
            trace_action_diff_audit={
                "trace_count": 2,
                "action_diff_steps": 0,
                "missing_step_count": 0,
                "max_abs_delta": 0,
            },
            runtime_provenance_audit={
                "record_count": 10,
                "runtime_reason_missing_count": 0,
                "unknown_post_guardrail_rewrite_count": 0,
            },
            joint_prediction_readiness={
                "ready_for_shadow_audit": True,
                "row_count": 480,
                "missing_field_counts": {},
            },
        )

        self.assertEqual(report["schema_version"], "metadata_replay_readiness_checklist_v18")
        self.assertTrue(report["first_wave_expanded_metadata_replay_executed"])
        self.assertTrue(report["checks"]["first_wave_expanded_metadata_replay_authorized"])
        self.assertEqual(report["checks"]["first_wave_expanded_scenario_count"], 2)
        self.assertTrue(report["canonical_strict_metadata_replay_pass"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertEqual(
            report["next_action"],
            "expanded_metadata_coverage_planning_continue_or_scenario_cache_discovery",
        )

    def test_user_decision_blocks_after_delta_classification(self):
        report = build_report(
            closure={"mainline_diff_authorization_complete": True, "blocking_paths": []},
            protocol={"protocol_isolation_pass": True, "requires_protocol_baseline_authorization": False},
            default_path={"default_path_evidence_pass": True, "default_path_call_count": 0},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            delta_classification={"unknown_hunk_count": 0},
        )

        self.assertFalse(report["strict_metadata_replay_planning_allowed"])
        self.assertEqual(report["next_action"], "protocol_v2_user_decision_required")
        self.assertIn("protocol v2 user decision complete", report["planning_blocked_until"])

    def test_blocked_categories_and_safety_review_are_explicit_gates(self):
        report = build_report(
            closure={"mainline_diff_authorization_complete": True, "blocking_paths": []},
            protocol={"protocol_isolation_pass": True, "requires_protocol_baseline_authorization": False},
            default_path={"default_path_evidence_pass": True, "default_path_call_count": 0},
            expert_status={"hard_blocker": False},
            default_path_plan={"plan_ready": True},
            cache_plan={"schema_version": "cache_coverage_plan_v1"},
            delta_classification={"unknown_hunk_count": 0},
            protocol_user_decision={"protocol_user_decision_complete": True},
            blocked_categories_plan={"blocked_categories_resolved": False},
            safety_oracle_review={"safety_boundary_test_oracle_review_complete": False},
        )

        self.assertFalse(report["metadata_replay_allowed"])
        self.assertEqual(report["next_action"], "protocol_v2_blocked_categories_resolution_required")
        self.assertFalse(report["checks"]["blocked_categories_resolved"])
        self.assertFalse(report["checks"]["safety_boundary_test_oracle_review_complete"])

    def test_second_wave_expanded_record_refreshes_v19_without_execution_permission(self):
        report = build_report(
            closure={"mainline_diff_authorization_complete": False, "blocking_paths": ["gl_gym/agent/llm_agent.py"]},
            protocol={"protocol_isolation_pass": False, "requires_protocol_baseline_authorization": True},
            default_path={"default_path_evidence_pass": False, "default_path_call_count": 3},
            protocol_v1_overlay_preflight={
                "protocol_v1_overlay_available": True,
                "protocol_v1_overlay_validated": True,
            },
            two_stage_authorization_packet={
                "stage_a": {"overlay_hash_match": True},
            },
            expanded_execution_record={
                "second_wave_expanded_metadata_replay_authorized": True,
                "scope": "second_wave_expanded_metadata_coverage",
                "scenario_ids": ["y2015_d180_s44_n240", "y2020_d180_s44_n240"],
            },
            strict_metadata_replay_summary={
                "aggregate": {"decision": "pass"},
                "rows": [
                    {
                        "plan_cache_enabled_steps": 240,
                        "plan_cache_hit_steps": 240,
                        "runtime_error_steps": 0,
                        "strict_cache_miss_runtime_error_steps": 0,
                    }
                ],
            },
            trace_action_diff_audit={
                "trace_count": 2,
                "action_diff_steps": 0,
                "missing_step_count": 0,
                "max_abs_delta": 0,
            },
            runtime_provenance_audit={
                "record_count": 10,
                "runtime_reason_missing_count": 0,
                "unknown_post_guardrail_rewrite_count": 0,
            },
            joint_prediction_readiness={
                "ready_for_shadow_audit": True,
                "row_count": 480,
                "missing_field_counts": {},
            },
        )

        self.assertEqual(report["schema_version"], "metadata_replay_readiness_checklist_v19")
        self.assertTrue(report["second_wave_expanded_metadata_replay_executed"])
        self.assertTrue(report["checks"]["second_wave_expanded_metadata_replay_authorized"])
        self.assertEqual(report["checks"]["second_wave_expanded_scenario_count"], 2)
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertEqual(
            report["next_action"],
            "expanded_metadata_coverage_consolidation_or_controlled_replay_admission_review",
        )

    def test_v20_controlled_admission_does_not_promote_default_controller(self):
        report = build_report(
            closure={"mainline_diff_authorization_complete": False, "blocking_paths": ["gl_gym/agent/llm_agent.py"]},
            protocol={"protocol_isolation_pass": False, "requires_protocol_baseline_authorization": True},
            default_path={"default_path_evidence_pass": False, "default_path_call_count": 3},
            controlled_replay_admission_review={
                "minimal_controlled_canary_admission_pass": False,
                "minimal_controlled_canary_allowed": False,
                "next_action": "controlled_canary_runner_protocol_overlay_resolution_required",
            },
        )

        self.assertEqual(report["schema_version"], "metadata_replay_readiness_checklist_v20")
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["minimal_controlled_canary_allowed"])
        self.assertEqual(report["next_action"], "controlled_canary_runner_protocol_overlay_resolution_required")

    def test_v21_minimal_controlled_canary_restores_execution_block(self):
        report = build_report(
            closure={"mainline_diff_authorization_complete": False, "blocking_paths": ["gl_gym/agent/llm_agent.py"]},
            protocol={"protocol_isolation_pass": False, "requires_protocol_baseline_authorization": True},
            default_path={"default_path_evidence_pass": False, "default_path_call_count": 3},
            controlled_replay_admission_review={
                "minimal_controlled_canary_admission_pass": True,
                "minimal_controlled_canary_allowed": True,
                "next_action": "generate_minimal_controlled_canary_execution_record",
            },
            controlled_canary_execution_record={"minimal_controlled_canary_authorized": True},
            controlled_canary_summary_audit={
                "aggregate": {"decision": "pass", "failure_count": 0},
                "rows": [
                    {
                        "plan_cache_enabled_steps": 240,
                        "plan_cache_hit_steps": 240,
                        "runtime_error_steps": 0,
                        "strict_cache_miss_runtime_error_steps": 0,
                    }
                ],
                "paired_deltas": [
                    {
                        "d_canopy_dew_margin_lt0_steps": 0,
                        "d_dew_margin_air_lt0_steps": 0,
                    }
                ],
            },
            controlled_canary_trace_audit={
                "aggregate": {
                    "unsafe_preferred_steps": 0,
                    "unsafe_conflict_steps": 0,
                    "unsafe_applied_steps": 0,
                    "applied_steps": 2,
                    "strict_applied_steps": 2,
                }
            },
            controlled_canary_strict_effect_audit={
                "aggregate": {
                    "runtime_error_steps": 0,
                    "unsafe_applied_steps": 0,
                    "warnings": [],
                }
            },
            runtime_provenance_audit={
                "record_count": 10,
                "runtime_reason_missing_count": 0,
                "unknown_post_guardrail_rewrite_count": 0,
            },
            joint_prediction_readiness={
                "ready_for_shadow_audit": True,
                "row_count": 240,
                "missing_field_counts": {},
            },
        )

        self.assertEqual(report["schema_version"], "metadata_replay_readiness_checklist_v21")
        self.assertTrue(report["minimal_controlled_canary_executed"])
        self.assertTrue(report["minimal_controlled_canary_pass"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["minimal_controlled_canary_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertEqual(report["next_action"], "minimal_controlled_canary_result_review_or_expansion_admission")


if __name__ == "__main__":
    unittest.main()
