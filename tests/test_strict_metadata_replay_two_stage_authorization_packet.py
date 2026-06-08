import unittest

from gl_gym.experiments.strict_metadata_replay_two_stage_authorization_packet import build_report


class TestStrictMetadataReplayTwoStageAuthorizationPacket(unittest.TestCase):
    def _report(self, *, stage_b_user_authorized: bool = False, overlay_valid: bool = True):
        return build_report(
            overlay_preflight={
                "protocol_v1_overlay_available": overlay_valid,
                "protocol_v1_overlay_validated": overlay_valid,
                "hash_mismatch_count": 0 if overlay_valid else 1,
                "object_missing_count": 0,
                "compile_pass": overlay_valid,
                "import_pass": overlay_valid,
                "protocol_v1_overlay_root": "overlay",
            },
            execution_request={
                "canonical_strict_metadata_replay_execution_request_ready": True,
                "actual_protocol_implementation_for_execution": "protocol_v1_snapshot_ephemeral_overlay",
                "scope": "canonical_failure_only",
                "scenario": "y2020_d120_s44_n240",
                "controller": "llm_rspc_v2",
                "max_steps": 240,
                "selected_cache_path": "cache.json",
                "baseline_trace_precheck": {"baseline_trace_qualified": True},
            },
            readiness={"canonical_metadata_replay_authorization_request_ready": True},
            overlay_preflight_path="overlay.json",
            execution_request_path="request.json",
            stage_b_user_authorized=stage_b_user_authorized,
        )

    def test_stage_a_validated_does_not_grant_stage_b_execution(self):
        report = self._report()

        self.assertTrue(report["two_stage_authorization_packet_ready"])
        self.assertTrue(report["stage_a"]["overlay_build_validation_complete"])
        self.assertTrue(report["stage_a"]["overlay_validated"])
        self.assertFalse(report["stage_b"]["canonical_replay_user_authorized"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertEqual(
            report["actual_protocol_implementation_for_execution"],
            "protocol_v1_snapshot_ephemeral_overlay",
        )

    def test_stage_b_authorization_record_is_still_not_an_execution_tool_call(self):
        report = self._report(stage_b_user_authorized=True)

        self.assertTrue(report["stage_b"]["canonical_replay_user_authorized"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertIn("scope expands beyond canonical_failure_only", report["stage_b_stop_conditions"])

    def test_invalid_overlay_blocks_packet_readiness(self):
        report = self._report(overlay_valid=False)

        self.assertFalse(report["two_stage_authorization_packet_ready"])
        self.assertFalse(report["stage_a"]["overlay_build_validation_complete"])
        self.assertEqual(report["next_action"], "repair_two_stage_authorization_inputs")


if __name__ == "__main__":
    unittest.main()
