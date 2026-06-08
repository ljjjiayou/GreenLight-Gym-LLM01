import unittest

from gl_gym.experiments.controlled_canary_weather_space_expansion_readiness import build_report


class TestControlledCanaryWeatherSpaceExpansionReadiness(unittest.TestCase):
    def test_readiness_waits_for_shadow_cache_authorization_when_manifest_ready(self):
        report = build_report(
            target_spec={"target_spec_ready": True, "v33_near_miss_scenario_count": 7, "v33_near_miss_row_count": 44},
            weather_inventory={
                "weather_window_inventory_ready": True,
                "new_weather_source_candidates_found": True,
                "weather_file_count": 20,
                "candidate_weather_day_count": 3,
                "selected_candidate_weather_day_count": 2,
            },
            acquisition_manifest={
                "shadow_scenario_acquisition_manifest_ready": True,
                "requested_scenario_count": 6,
            },
        )

        self.assertEqual(report["next_action"], "await_explicit_shadow_cache_acquisition_authorization_for_weather_expansion")
        self.assertFalse(report["cache_fill_authorized"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_readiness_no_candidate_moves_to_external_dataset_or_relaxed_design(self):
        report = build_report(
            target_spec={"target_spec_ready": True},
            weather_inventory={
                "weather_window_inventory_ready": True,
                "new_weather_source_candidates_found": False,
                "weather_file_count": 20,
                "candidate_weather_day_count": 0,
            },
        )

        self.assertEqual(report["next_action"], "external_weather_dataset_or_relaxed_shadow_source_design_required")
        self.assertIn("weather_strict_proxy_source_not_found", report["failure_taxonomy"])
        self.assertFalse(report["controlled_replay_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
