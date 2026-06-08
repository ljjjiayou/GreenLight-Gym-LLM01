import unittest

from gl_gym.experiments.canonical_strict_metadata_replay_execution_request import build_report


class TestCanonicalStrictMetadataReplayExecutionRequest(unittest.TestCase):
    def _base_report(self, *, baseline_qualified: bool = True):
        return build_report(
            authorization_packet={
                "authorization_packet_ready": True,
                "actual_protocol_implementation_for_execution": "protocol_v1_snapshot_ephemeral_overlay",
                "controller": "llm_rspc_v2",
                "max_steps": 240,
                "selected_cache_path": "cache.json",
                "plan_cache_key_policy": "scenario_timestep",
            },
            readiness={"canonical_metadata_replay_authorization_request_ready": True},
            strict_run_plan={
                "strict_metadata_replay_run_plan_ready": True,
                "allowed_initial_scenarios": [
                    {
                        "scenario_id": "y2020_d120_s44_n240",
                        "env_id": "TomatoEnv_y2020_d120_s44",
                        "year": 2020,
                        "day": 120,
                        "seed": 44,
                    }
                ],
                "fixed_inputs": {
                    "controller": "llm_rspc_v2",
                    "max_steps": 240,
                    "selected_cache_path": "cache.json",
                    "plan_cache_key_policy": "scenario_timestep",
                },
            },
            overlay_preflight={
                "protocol_v1_overlay_validated": True,
                "protocol_v1_overlay_root": ".",
            },
            cache_coverage={"cache_coverage_pass": True},
            baseline_trace_manifest={
                "baseline_trace_manifest_ready": True,
                "baseline_trace_qualified": baseline_qualified,
                "action_diff_audit_status": (
                    "ready_with_qualified_baseline_trace"
                    if baseline_qualified
                    else "blocked_missing_baseline_trace"
                ),
                "selected_baseline_trace_dir": "baseline",
                "selected_baseline_csv": "baseline.csv",
            },
        )

    def test_request_never_grants_execution(self):
        report = self._base_report()

        self.assertTrue(report["canonical_strict_metadata_replay_execution_request_ready"])
        self.assertTrue(report["user_explicit_authorization_required"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["cache_fill_run"])
        self.assertFalse(report["online_llm_called"])
        self.assertFalse(report["default_llm_rspc_v2_changed"])
        self.assertTrue(report["rejected_presets_disabled"])

    def test_planned_command_is_strict_replay_no_fill_no_online_llm(self):
        report = self._base_report()
        command = report["planned_command"]

        for token in (
            "--plan-cache-mode",
            "replay",
            "--plan-cache-strict",
            "--plan-cache-key-policy",
            "scenario_timestep",
            "--years",
            "2020",
            "--days",
            "120",
            "--seeds",
            "44",
            "--controllers",
            "llm_rspc_v2",
        ):
            self.assertIn(token, command)
        self.assertEqual(report["scope"], "canonical_failure_only")
        self.assertEqual(report["actual_protocol_implementation_for_execution"], "protocol_v1_snapshot_ephemeral_overlay")
        self.assertNotIn("--plan-cache-fill", command)

    def test_missing_baseline_trace_blocks_action_diff_claim_not_request(self):
        report = self._base_report(baseline_qualified=False)

        self.assertTrue(report["canonical_strict_metadata_replay_execution_request_ready"])
        self.assertEqual(
            report["baseline_trace_precheck"]["action_diff_audit_status"],
            "blocked_missing_baseline_trace",
        )
        self.assertEqual(report["execution_targets"]["action_diff_steps"], "blocked_missing_baseline_trace")
        self.assertFalse(report["metadata_replay_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
