import unittest

from gl_gym.experiments.controlled_canary_shadow_scenario_acquisition_manifest import build_report


class TestControlledCanaryShadowScenarioAcquisitionManifest(unittest.TestCase):
    def test_manifest_never_authorizes_cache_fill_or_replay(self):
        report = build_report(
            weather_inventory={
                "weather_window_inventory_ready": True,
                "candidate_scenarios_for_cache_acquisition": [
                    {
                        "scenario_id": "y2021_d10_s42_n240",
                        "year": 2021,
                        "day": 10,
                        "seed": 42,
                        "max_steps": 240,
                        "weather_site": "Amsterdam",
                        "weather_label": "2021",
                    }
                ],
            },
            requested_cache_path="cache.json",
        )

        self.assertTrue(report["shadow_scenario_acquisition_manifest_ready"])
        self.assertFalse(report["cache_fill_authorized"])
        self.assertFalse(report["online_llm_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["replay_execution_record_generated"])
        self.assertEqual(report["next_action"], "await_explicit_shadow_cache_acquisition_authorization_for_weather_expansion")

    def test_empty_manifest_points_to_external_dataset_design(self):
        report = build_report(
            weather_inventory={
                "weather_window_inventory_ready": True,
                "candidate_scenarios_for_cache_acquisition": [],
            },
            requested_cache_path="cache.json",
        )

        self.assertFalse(report["shadow_scenario_acquisition_manifest_ready"])
        self.assertEqual(report["next_action"], "external_weather_dataset_or_relaxed_shadow_source_design_required")


if __name__ == "__main__":
    unittest.main()
