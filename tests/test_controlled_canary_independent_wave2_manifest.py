import unittest

from gl_gym.experiments.controlled_canary_independent_wave2_manifest import build_report


class TestControlledCanaryIndependentWave2Manifest(unittest.TestCase):
    def test_manifest_uses_only_independent_discovery_candidates(self):
        report = build_report(
            discovery={
                "independent_trigger_candidates_found": True,
                "excluded_scenario_ids": ["y2015_d120_s42_n240"],
                "qualified_trigger_scenarios": [
                    {
                        "scenario_id": "y2018_d120_s45_n240",
                        "year": 2018,
                        "day": 120,
                        "seed": 45,
                        "max_steps": 240,
                        "expected_strict_eligible_applied_steps": 2,
                    }
                ],
            },
            selected_cache_path="cache.json",
            output_root="out",
        )

        self.assertTrue(report["wave2_manifest_ready"])
        self.assertEqual(report["scope"], "independent_triggered_holdout_wave2_only")
        self.assertEqual(report["scenario_ids"], ["y2018_d120_s45_n240"])
        self.assertTrue(report["command_groups"][0]["cartesian_product_safe"])
        self.assertFalse(report["performance_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
