import unittest

from gl_gym.experiments.controlled_canary_weather_shadow_cache_acquisition_readiness import build_report


class TestControlledCanaryWeatherShadowCacheAcquisitionReadiness(unittest.TestCase):
    def test_readiness_reports_credentials_missing_without_controlled_permission(self):
        report = build_report(
            manifest={
                "weather_shadow_cache_acquisition_manifest_ready": True,
                "scenario_count": 9,
                "isolated_cache_path": "gl_gym/result/plan_cache/weather_expansion_shadow_cache_v36_20260529.json",
            },
            execution_record={
                "weather_shadow_cache_acquisition_authorized": True,
                "weather_shadow_cache_acquisition_executable": False,
                "cache_fill_authorized": True,
                "online_llm_allowed": True,
                "failure_taxonomy": ["online_llm_credentials_missing"],
                "next_action": "online_llm_credentials_missing",
            },
        )

        self.assertEqual(report["next_action"], "online_llm_credentials_missing")
        self.assertIn("online_llm_credentials_missing", report["failure_taxonomy"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["controlled_replay_execution_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_readiness_moves_to_shadow_replay_admission_after_coverage_pass(self):
        report = build_report(
            manifest={
                "weather_shadow_cache_acquisition_manifest_ready": True,
                "scenario_count": 9,
                "isolated_cache_path": "gl_gym/result/plan_cache/weather_expansion_shadow_cache_v36_20260529.json",
            },
            execution_record={
                "weather_shadow_cache_acquisition_authorized": True,
                "weather_shadow_cache_acquisition_executable": True,
                "cache_fill_authorized": True,
                "online_llm_allowed": True,
                "failure_taxonomy": [],
            },
            cache_coverage={
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
            },
        )

        self.assertEqual(report["next_action"], "shadow_only_strict_metadata_replay_admission_for_v36_cache")
        self.assertTrue(report["shadow_cache_coverage_pass"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_readiness_blocks_failed_coverage(self):
        report = build_report(
            manifest={"weather_shadow_cache_acquisition_manifest_ready": True, "scenario_count": 9},
            execution_record={
                "weather_shadow_cache_acquisition_authorized": True,
                "weather_shadow_cache_acquisition_executable": True,
                "cache_fill_authorized": True,
                "online_llm_allowed": True,
                "failure_taxonomy": [],
            },
            cache_coverage={
                "cache_coverage_pass": False,
                "coverage_rate": 0.5,
                "missing_key_count": 1,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
            },
        )

        self.assertIn("shadow_cache_coverage_incomplete", report["failure_taxonomy"])
        self.assertEqual(report["next_action"], "shadow_cache_acquisition_failure_diagnosis")
        self.assertFalse(report["shadow_cache_coverage_pass"])


if __name__ == "__main__":
    unittest.main()
