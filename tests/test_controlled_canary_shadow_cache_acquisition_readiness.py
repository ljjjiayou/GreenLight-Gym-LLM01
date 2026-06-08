import unittest

from gl_gym.experiments.controlled_canary_shadow_cache_acquisition_readiness import build_report


class TestControlledCanaryShadowCacheAcquisitionReadiness(unittest.TestCase):
    def test_readiness_waits_for_cache_authorization_when_request_ready(self):
        report = build_report(
            target_spec={"target_spec_ready": True, "v33_near_miss_scenario_count": 7, "v33_near_miss_row_count": 44},
            source_inventory={
                "new_shadow_source_inventory_ready": True,
                "new_shadow_source_candidates_found": True,
                "candidate_scenario_count": 1,
                "trace_count": 10,
            },
            cache_acquisition_request={
                "shadow_cache_acquisition_request_ready": True,
                "requested_scenario_count": 1,
            },
        )

        self.assertEqual(report["next_action"], "await_explicit_shadow_cache_acquisition_authorization")
        self.assertFalse(report["cache_fill_authorized"])
        self.assertFalse(report["online_llm_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_readiness_external_expansion_when_no_source_found(self):
        report = build_report(
            target_spec={"target_spec_ready": True},
            source_inventory={
                "new_shadow_source_inventory_ready": True,
                "new_shadow_source_candidates_found": False,
                "candidate_scenario_count": 0,
            },
        )

        self.assertEqual(report["next_action"], "external_weather_or_scenario_space_expansion_design")
        self.assertIn("new_shadow_source_not_found", report["failure_taxonomy"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_readiness_requires_inventory_after_target_spec(self):
        report = build_report(target_spec={"target_spec_ready": True})

        self.assertEqual(report["next_action"], "run_v34_read_only_trace_source_inventory")
        self.assertFalse(report["controlled_replay_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
