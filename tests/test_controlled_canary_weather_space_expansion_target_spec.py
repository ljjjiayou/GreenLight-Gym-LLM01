import unittest

from gl_gym.experiments.controlled_canary_weather_space_expansion_target_spec import build_report


class TestControlledCanaryWeatherSpaceExpansionTargetSpec(unittest.TestCase):
    def test_target_spec_ready_after_v34_exhaustion(self):
        report = build_report(
            near_miss_catalog={
                "near_miss_catalog_ready": True,
                "near_miss_scenario_count": 7,
                "near_miss_row_count": 44,
                "excluded_scenario_ids": ["y2015_d120_s42_n240"],
            },
            v34_source_inventory={
                "new_shadow_source_inventory_ready": True,
                "new_shadow_source_candidates_found": False,
                "candidate_scenario_count": 0,
            },
            v34_readiness={"next_action": "external_weather_or_scenario_space_expansion_design"},
        )

        self.assertTrue(report["target_spec_ready"])
        self.assertTrue(report["weather_proxy_source_hint_only"])
        self.assertFalse(report["weather_proxy_is_strict_applied_evidence"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertIn("y2015_d120_s42_n240", report["excluded_scenario_ids"])
        self.assertIn("y2020_d59_s42_n240", report["excluded_scenario_ids"])

    def test_target_spec_not_ready_when_v34_has_candidate(self):
        report = build_report(
            near_miss_catalog={"near_miss_catalog_ready": True},
            v34_source_inventory={
                "new_shadow_source_inventory_ready": True,
                "new_shadow_source_candidates_found": True,
                "candidate_scenario_count": 1,
            },
            v34_readiness={"next_action": "shadow_cache_acquisition_request"},
        )

        self.assertFalse(report["target_spec_ready"])


if __name__ == "__main__":
    unittest.main()
