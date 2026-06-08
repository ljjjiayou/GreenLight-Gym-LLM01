import unittest

from gl_gym.experiments.canonical_strict_metadata_replay_stage_b_rerun_execution_record import build_report


class TestCanonicalStrictMetadataReplayStageBRerunExecutionRecord(unittest.TestCase):
    def _request(self):
        return {
            "stage_b_authorization_request_ready": True,
            "runtime_provenance_closure_implementation_ready": True,
            "scope": "canonical_failure_only",
            "scenario": "y2020_d120_s44_n240",
            "controller": "llm_rspc_v2",
            "max_steps": 240,
            "actual_protocol_implementation_for_execution": "protocol_v1_snapshot_ephemeral_overlay",
            "overlay_runner_path_exists": True,
            "baseline_trace_precheck": {"baseline_trace_qualified": True},
            "planned_command": [
                "python",
                "run_frozen_benchmark.py",
                "--plan-cache-mode",
                "replay",
                "--plan-cache-strict",
                "--plan-cache-key-policy",
                "scenario_timestep",
            ],
            "cache_fill_run": False,
            "online_llm_called": False,
            "controlled_replay_allowed": False,
            "performance_claim_allowed": False,
        }

    def test_authorizes_only_when_precheck_passes(self):
        report = build_report(
            rerun_request=self._request(),
            authorization_source="unit_test",
        )

        self.assertTrue(report["stage_b_user_authorized"])
        self.assertTrue(report["stage_b_rerun_authorized"])
        self.assertTrue(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["next_action"], "execute_single_canonical_strict_metadata_replay_rerun")

    def test_missing_strict_flag_blocks_execution(self):
        request = self._request()
        request["planned_command"] = ["python", "run_frozen_benchmark.py"]

        report = build_report(
            rerun_request=request,
            authorization_source="unit_test",
        )

        self.assertFalse(report["stage_b_rerun_authorized"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["precheck"]["planned_command_has_strict_replay_flags"])
        self.assertEqual(report["next_action"], "stage_b_rerun_precheck_failure")

    def test_expanded_or_controlled_replay_stays_blocked(self):
        request = self._request()
        request["controlled_replay_allowed"] = True

        report = build_report(
            rerun_request=request,
            authorization_source="unit_test",
        )

        self.assertFalse(report["stage_b_rerun_authorized"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])


if __name__ == "__main__":
    unittest.main()
