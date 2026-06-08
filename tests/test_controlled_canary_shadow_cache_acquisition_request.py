import unittest

from gl_gym.experiments.controlled_canary_shadow_cache_acquisition_request import build_report


class TestControlledCanaryShadowCacheAcquisitionRequest(unittest.TestCase):
    def test_request_never_authorizes_cache_fill_or_replay(self):
        report = build_report(
            source_inventory={
                "new_shadow_source_inventory_ready": True,
                "candidate_scenarios": [
                    {
                        "scenario_id": "y2021_d1_s1_n240",
                        "year": 2021,
                        "day": 1,
                        "seed": 1,
                        "max_steps": 240,
                        "expected_strict_eligible_applied_steps": 2,
                        "source_rows": [{"step": 10}],
                    }
                ],
            },
            requested_cache_path="cache.json",
        )

        self.assertTrue(report["shadow_cache_acquisition_request_ready"])
        self.assertFalse(report["cache_fill_authorized"])
        self.assertFalse(report["online_llm_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["replay_execution_record_generated"])
        self.assertEqual(report["requested_scenario_count"], 1)
        self.assertEqual(report["next_action"], "await_explicit_shadow_cache_acquisition_authorization")

    def test_empty_inventory_requests_external_expansion(self):
        report = build_report(
            source_inventory={"new_shadow_source_inventory_ready": True, "candidate_scenarios": []},
            requested_cache_path="cache.json",
        )

        self.assertFalse(report["shadow_cache_acquisition_request_ready"])
        self.assertEqual(report["next_action"], "external_weather_or_scenario_space_expansion_design")


if __name__ == "__main__":
    unittest.main()
