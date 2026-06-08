import unittest

from gl_gym.experiments.expanded_metadata_first_wave_scenario_manifest import build_report


class TestExpandedMetadataFirstWaveScenarioManifest(unittest.TestCase):
    def test_manifest_selects_first_wave_scenarios_and_blocks_missing_families(self):
        report = build_report(
            counterexample_suite={
                "windows_by_regime": {
                    "neutral_no_risk": [{"scenario_id": "y2010_d120_s44_n240"}],
                    "radiation_spike": [{"scenario_id": "y2010_d120_s44_n240"}],
                    "wind_stress": [{"scenario_id": "y2010_d120_s44_n240"}],
                    "cold_humid": [{"scenario_id": "y2010_d120_s44_n240"}],
                    "dawn_dew": [{"scenario_id": "y2010_d180_s44_n240"}],
                }
            },
            selected_cache_path="cache.json",
        )

        self.assertTrue(report["first_wave_manifest_ready"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["scenario_count"], 2)
        self.assertEqual(
            [item["scenario_id"] for item in report["selected_scenarios"]],
            ["y2010_d120_s44_n240", "y2010_d180_s44_n240"],
        )
        blocked = {item["family"]: item["status"] for item in report["blocked_families"]}
        self.assertEqual(blocked["pure_hot_dry"], "blocked_missing_cache_or_scenario")
        self.assertEqual(blocked["mixed_dry_dew"], "blocked_missing_cache_or_scenario")


if __name__ == "__main__":
    unittest.main()
