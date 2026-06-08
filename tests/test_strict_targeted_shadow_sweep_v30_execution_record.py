import unittest

from gl_gym.experiments.strict_targeted_shadow_sweep_v30_execution_record import build_report


class TestStrictTargetedShadowSweepV30ExecutionRecord(unittest.TestCase):
    def _base_inputs(self):
        return {
            "manifest": {
                "strict_targeted_shadow_sweep_v30_manifest_ready": True,
                "shadow_only": True,
                "scenario_count": 28,
                "blocked_cache_repair_backlog_count": 8,
                "scenario_ids": ["y2010_d59_s42_n240"],
                "selected_cache_path": "cache.json",
                "blocked_cache_repair_backlog": [{"scenario_id": "y2020_d59_s42_n240"}],
                "command_groups": [
                    {
                        "group_id": "v30_group_a_2010_2015_all_seeds",
                        "years": [2010, 2015],
                        "days": [59, 120, 180, 240],
                        "seeds": [42, 43, 44],
                        "cartesian_product_safe": True,
                    },
                    {
                        "group_id": "v30_group_b_2020_seed44",
                        "years": [2020],
                        "days": [59, 120, 180, 240],
                        "seeds": [44],
                        "cartesian_product_safe": True,
                    },
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
                "overlay_runner_path": "gl_gym/experiments/strict_targeted_shadow_sweep_v30_execution_record.py",
                "protocol_v1_overlay_root": "gl_gym/result/audits/protocol_v1_controlled_canary_overlay_20260525",
            },
            "authorization_source": "user_delegated_v30_coverage_qualified_shadow_only_sweep_20260528",
            "output_root": "out",
            "controllers": ["llm_rspc_v2"],
        }

    def test_execution_record_authorizes_shadow_only_subset(self):
        report = build_report(**self._base_inputs())

        self.assertTrue(report["strict_targeted_shadow_sweep_v30_authorized"])
        self.assertEqual(report["controllers"], ["llm_rspc_v2"])
        self.assertEqual(len(report["planned_commands"]), 2)
        self.assertIn("--plan-cache-strict", report["planned_commands"][0])
        self.assertIn("scenario_timestep", report["planned_commands"][1])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_execution_record_blocks_when_cache_coverage_fails(self):
        inputs = self._base_inputs()
        inputs["cache_coverage"] = dict(inputs["cache_coverage"], cache_coverage_pass=False, missing_key_count=1)
        report = build_report(**inputs)

        self.assertFalse(report["strict_targeted_shadow_sweep_v30_authorized"])
        self.assertFalse(report["metadata_replay_execution_allowed"])

    def test_execution_record_rejects_controlled_controller(self):
        inputs = self._base_inputs()
        inputs["controllers"] = ["llm_rspc_v2", "llm_rspc_v2_hot_dry_proposer_strict"]
        report = build_report(**inputs)

        self.assertFalse(report["strict_targeted_shadow_sweep_v30_authorized"])
        self.assertFalse(report["precheck"]["controller_scope_ok"])


if __name__ == "__main__":
    unittest.main()
