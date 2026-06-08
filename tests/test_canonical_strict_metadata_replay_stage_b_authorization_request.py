import unittest

from gl_gym.experiments.canonical_strict_metadata_replay_stage_b_authorization_request import build_report


class TestCanonicalStrictMetadataReplayStageBAuthorizationRequest(unittest.TestCase):
    def _report(self, *, authorized: bool = False):
        return build_report(
            two_stage_authorization_packet={
                "two_stage_authorization_packet_ready": True,
                "stage_a": {"overlay_build_validation_complete": True},
                "stage_b": {"stage_b_ready_for_user_decision": True},
            },
            execution_request={
                "canonical_strict_metadata_replay_execution_request_ready": True,
                "scope": "canonical_failure_only",
                "scenario": "y2020_d120_s44_n240",
                "controller": "llm_rspc_v2",
                "max_steps": 240,
                "selected_cache_path": "cache.json",
                "actual_protocol_implementation_for_execution": "protocol_v1_snapshot_ephemeral_overlay",
                "overlay_runner_path_exists": True,
                "baseline_trace_precheck": {"baseline_trace_qualified": True},
                "planned_command": ["python", "run_frozen_benchmark.py", "--plan-cache-mode", "replay"],
                "post_run_audits": [
                    {"name": "strict_metadata_replay_summary_20260523", "status": "planned"},
                    {"name": "metadata_replay_readiness_checklist_20260523_v12", "status": "planned"},
                ],
            },
            readiness={"canonical_metadata_replay_authorization_request_ready": True},
            stage_b_user_authorized=authorized,
            authorization_source="unit_test" if authorized else "not_authorized_in_this_turn",
        )

    def test_stage_b_request_ready_does_not_authorize_by_default(self):
        report = self._report()

        self.assertTrue(report["stage_b_authorization_request_ready"])
        self.assertFalse(report["stage_b_user_authorized"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertEqual(report["next_action"], "await_explicit_stage_b_user_authorization_for_canonical_strict_metadata_replay")
        audit_names = [row["name"] for row in report["post_run_audits"]]
        self.assertIn("strict_metadata_replay_summary_20260524", audit_names)
        self.assertIn("metadata_replay_readiness_checklist_20260524_v14", audit_names)

    def test_explicit_authorization_flag_can_record_execution_permission(self):
        report = self._report(authorized=True)

        self.assertTrue(report["stage_b_user_authorized"])
        self.assertTrue(report["metadata_replay_execution_allowed"])
        self.assertTrue(report["metadata_replay_allowed"])
        self.assertEqual(report["next_action"], "execute_single_canonical_strict_metadata_replay")
        self.assertEqual(report["authorization_source"], "unit_test")

    def test_missing_baseline_trace_keeps_request_not_ready(self):
        report = build_report(
            two_stage_authorization_packet={
                "two_stage_authorization_packet_ready": True,
                "stage_a": {"overlay_build_validation_complete": True},
                "stage_b": {"stage_b_ready_for_user_decision": True},
            },
            execution_request={
                "canonical_strict_metadata_replay_execution_request_ready": True,
                "overlay_runner_path_exists": True,
                "baseline_trace_precheck": {"baseline_trace_qualified": False},
            },
            readiness={"canonical_metadata_replay_authorization_request_ready": True},
        )

        self.assertFalse(report["stage_b_authorization_request_ready"])
        self.assertFalse(report["metadata_replay_execution_allowed"])

    def test_runtime_provenance_closure_requires_fresh_rerun_authorization(self):
        report = build_report(
            two_stage_authorization_packet={
                "two_stage_authorization_packet_ready": True,
                "stage_a": {"overlay_build_validation_complete": True},
                "stage_b": {"stage_b_ready_for_user_decision": True},
            },
            execution_request={
                "canonical_strict_metadata_replay_execution_request_ready": True,
                "scope": "canonical_failure_only",
                "scenario": "y2020_d120_s44_n240",
                "controller": "llm_rspc_v2",
                "max_steps": 240,
                "actual_protocol_implementation_for_execution": "protocol_v1_snapshot_ephemeral_overlay",
                "overlay_runner_path_exists": True,
                "baseline_trace_precheck": {"baseline_trace_qualified": True},
                "post_run_audits": [
                    {"name": "metadata_replay_readiness_checklist_20260523_v12", "status": "planned"},
                ],
            },
            readiness={"canonical_metadata_replay_authorization_request_ready": True},
            runtime_provenance_closure_status={
                "runtime_provenance_closure_implementation_ready": True,
                "stage_b_rerun_authorization_required": True,
                "stage_b_rerun_authorized": False,
            },
            stage_b_user_authorized=True,
            authorization_source="previous_stage_b_authorization_not_rerun_authorization",
        )

        self.assertTrue(report["stage_b_authorization_request_ready"])
        self.assertTrue(report["runtime_provenance_closure_implementation_ready"])
        self.assertTrue(report["stage_b_rerun_authorization_required"])
        self.assertFalse(report["stage_b_rerun_authorized"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertEqual(
            report["next_action"],
            "await_explicit_stage_b_rerun_authorization_after_runtime_provenance_closure",
        )
        audit_names = [row["name"] for row in report["post_run_audits"] if isinstance(row, dict)]
        self.assertIn("metadata_replay_readiness_checklist_20260524_v17", audit_names)


if __name__ == "__main__":
    unittest.main()
