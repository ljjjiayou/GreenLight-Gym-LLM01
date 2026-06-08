import unittest

from gl_gym.experiments.strict_targeted_shadow_sweep_v33_manifest import build_report


class TestStrictTargetedShadowSweepV33Manifest(unittest.TestCase):
    def test_manifest_uses_only_inventory_candidates(self):
        report = build_report(
            inventory={
                "existing_cache_inventory_ready": True,
                "candidate_scenarios": [
                    {
                        "scenario_id": "y2021_d1_s1_n240",
                        "env_id": "TomatoEnv_y2021_d1_s1",
                        "year": 2021,
                        "day": 1,
                        "seed": 1,
                        "max_steps": 240,
                        "cache_path": "cache_a.json",
                        "entry_count": 20,
                        "payload_complete": True,
                    }
                ],
            }
        )

        self.assertTrue(report["strict_targeted_shadow_sweep_v33_manifest_ready"])
        self.assertEqual(report["controllers"], ["llm_rspc_v2"])
        self.assertEqual(report["scenario_ids"], ["y2021_d1_s1_n240"])
        self.assertEqual(report["command_groups"][0]["years"], [2021])
        self.assertTrue(report["command_groups"][0]["cartesian_product_safe"])
        self.assertFalse(report["controlled_replay_allowed"])

    def test_manifest_blocks_when_inventory_has_no_candidates(self):
        report = build_report(inventory={"existing_cache_inventory_ready": True, "candidate_scenarios": []})

        self.assertFalse(report["strict_targeted_shadow_sweep_v33_manifest_ready"])
        self.assertEqual(report["next_action"], "existing_cache_exhausted_for_independent_strict_source")


if __name__ == "__main__":
    unittest.main()
