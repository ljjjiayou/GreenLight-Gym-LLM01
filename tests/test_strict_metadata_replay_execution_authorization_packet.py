import unittest

from gl_gym.experiments.strict_metadata_replay_execution_authorization_packet import build_report


class TestStrictMetadataReplayExecutionAuthorizationPacket(unittest.TestCase):
    def _base_report(self, *, overlay_path: str = "", overlay_preflight=None):
        return build_report(
            strict_run_plan={
                "strict_metadata_replay_run_plan_ready": True,
                "allowed_initial_scenarios": [
                    {
                        "scenario_id": "y2020_d120_s44_n240",
                        "env_id": "TomatoEnv_y2020_d120_s44",
                    }
                ],
                "fixed_inputs": {
                    "controller": "llm_rspc_v2",
                    "max_steps": 240,
                    "selected_cache_path": "cache.json",
                    "plan_cache_key_policy": "scenario_timestep",
                },
                "targets": {
                    "action_diff_steps": 0,
                    "max_abs_delta": 0,
                    "runtime_provenance_record_count": ">0",
                },
            },
            protocol_v1_snapshot={
                "snapshot_available": True,
                "source_git_ref": "HEAD",
                "source_git_commit": "abc",
                "tracked_baseline_file_count": 2,
                "files": [
                    {
                        "path": "gl_gym/experiments/frozen_benchmark_protocol.py",
                        "working_tree_status": "M gl_gym/experiments/frozen_benchmark_protocol.py",
                        "sha256": "hash",
                        "object_available": True,
                    }
                ],
            },
            readiness={"next_action": "strict_metadata_replay_execution_authorization_required"},
            overlay_preflight=overlay_preflight,
            v1_overlay_path=overlay_path,
        )

    def test_packet_never_grants_execution_by_default(self):
        report = self._base_report()

        self.assertEqual(report["schema_version"], "strict_metadata_replay_execution_authorization_packet_v2")
        self.assertTrue(report["authorization_packet_ready"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertTrue(report["user_explicit_authorization_required"])
        self.assertFalse(report["cache_fill_run"])
        self.assertFalse(report["online_llm_called"])
        self.assertFalse(report["default_llm_rspc_v2_changed"])
        self.assertTrue(report["rejected_presets_disabled"])

    def test_protocol_implementation_is_explicit_and_not_working_tree(self):
        report = self._base_report()

        self.assertEqual(
            report["actual_protocol_implementation_for_execution"],
            "protocol_v1_snapshot_ephemeral_overlay_required",
        )
        self.assertNotEqual(report["actual_protocol_implementation_for_execution"], "current_working_tree")
        self.assertTrue(report["current_working_tree_protocol_files_modified"])
        self.assertFalse(report["protocol_v1_overlay_available"])
        self.assertFalse(report["protocol_v1_overlay_validated"])
        self.assertIn("do not fall back to current working tree", report["protocol_implementation_execution_blocker"])

    def test_validated_overlay_updates_protocol_boundary_without_granting_execution(self):
        report = self._base_report(
            overlay_preflight={
                "schema_version": "protocol_v1_ephemeral_overlay_preflight_v1",
                "protocol_v1_overlay_available": True,
                "protocol_v1_overlay_validated": True,
                "protocol_v1_overlay_root": "gl_gym/result/audits/protocol_v1_overlay_20260523",
                "next_action": "refresh_strict_metadata_replay_execution_authorization_packet",
            }
        )

        self.assertEqual(
            report["actual_protocol_implementation_for_execution"],
            "protocol_v1_snapshot_ephemeral_overlay",
        )
        self.assertTrue(report["protocol_v1_overlay_available"])
        self.assertTrue(report["protocol_v1_overlay_validated"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertIn("explicit user authorization", report["protocol_implementation_execution_blocker"])

    def test_post_run_audits_and_failure_taxonomy_are_complete(self):
        report = self._base_report()

        for item in (
            "strict_metadata_replay_summary",
            "trace_action_diff_audit",
            "post_guardrail_runtime_provenance_audit",
            "post_guardrail_joint_prediction_readiness_refresh",
            "metadata_replay_readiness_checklist_v12",
        ):
            self.assertIn(item, report["post_run_audits"])
        for item in (
            "action_changed",
            "metadata_missing",
            "runtime_provenance_missing",
            "joint_prediction_missing",
            "cache_mismatch",
            "protocol_implementation_mismatch",
        ):
            self.assertIn(item, report["failure_taxonomy"])


if __name__ == "__main__":
    unittest.main()
