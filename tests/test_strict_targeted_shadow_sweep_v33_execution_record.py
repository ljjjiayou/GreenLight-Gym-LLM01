import unittest

from gl_gym.experiments.strict_targeted_shadow_sweep_v33_execution_record import build_report


class TestStrictTargetedShadowSweepV33ExecutionRecord(unittest.TestCase):
    def _inputs(self):
        return {
            "manifest": {
                "strict_targeted_shadow_sweep_v33_manifest_ready": True,
                "scenario_ids": ["y2021_d1_s1_n240"],
                "command_groups": [
                    {
                        "group_id": "v33_y2021_d1_s1_n240",
                        "cache_path": "cache.json",
                        "years": [2021],
                        "days": [1],
                        "seeds": [1],
                        "scenario_ids": ["y2021_d1_s1_n240"],
                        "cartesian_product_safe": True,
                    }
                ],
            },
            "cache_coverage": {
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
            },
            "overlay_preflight": {
                "controlled_canary_overlay_validated": True,
                "controlled_canary_runner_surface_validated": True,
                "protocol_hash_mismatch_count": 0,
                "overlay_runner_path": "gl_gym/experiments/strict_targeted_shadow_sweep_v33_execution_record.py",
                "protocol_v1_overlay_root": "gl_gym/result/audits/protocol_v1_controlled_canary_overlay_20260525",
            },
            "authorization_source": "user_delegated_v33_shadow_only_source_discovery_20260528",
            "output_root": "out",
            "controllers": ["llm_rspc_v2"],
        }

    def test_execution_record_authorizes_shadow_only_controller(self):
        report = build_report(**self._inputs())

        self.assertTrue(report["strict_targeted_shadow_sweep_v33_authorized"])
        self.assertEqual(report["controllers"], ["llm_rspc_v2"])
        self.assertIn("--plan-cache-strict", report["planned_commands"][0])
        self.assertIn("scenario_timestep", report["planned_commands"][0])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_execution_record_blocks_controlled_controller(self):
        inputs = self._inputs()
        inputs["controllers"] = ["llm_rspc_v2", "llm_rspc_v2_hot_dry_proposer_strict"]
        report = build_report(**inputs)

        self.assertFalse(report["strict_targeted_shadow_sweep_v33_authorized"])
        self.assertFalse(report["precheck"]["controller_scope_ok"])
        self.assertFalse(report["controlled_replay_execution_allowed"])

    def test_execution_record_blocks_cache_failure(self):
        inputs = self._inputs()
        inputs["cache_coverage"] = dict(inputs["cache_coverage"], cache_coverage_pass=False, missing_key_count=1)
        report = build_report(**inputs)

        self.assertFalse(report["strict_targeted_shadow_sweep_v33_authorized"])
        self.assertFalse(report["metadata_replay_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
