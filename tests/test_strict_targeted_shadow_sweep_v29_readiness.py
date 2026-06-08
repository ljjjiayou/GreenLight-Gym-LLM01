import unittest

from gl_gym.experiments.strict_targeted_shadow_sweep_v29_readiness import build_report


class TestStrictTargetedShadowSweepV29Readiness(unittest.TestCase):
    def test_readiness_blocks_when_cache_coverage_fails(self):
        report = build_report(
            manifest={"strict_targeted_shadow_sweep_manifest_ready": True, "scenario_count": 36},
            cache_coverage={
                "cache_coverage_pass": False,
                "coverage_rate": 1.0,
                "missing_key_count": 3,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
                "expected_env_count": 36,
            },
        )

        self.assertFalse(report["v29_cache_coverage_pass"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertEqual(report["next_action"], "repair_or_expand_selected_cache_before_v29_sweep")

    def test_readiness_requests_cache_precheck_when_strict_source_found(self):
        report = build_report(
            manifest={"strict_targeted_shadow_sweep_manifest_ready": True, "scenario_count": 36},
            cache_coverage={
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
                "expected_env_count": 36,
            },
            execution_record={"strict_targeted_shadow_sweep_v29_authorized": True},
            strict_sourcing={
                "strict_targeted_source_found": True,
                "expected_strict_eligible_applied_steps": 2,
                "would_apply_steps": 2,
                "strict_filtered_steps": 0,
            },
        )

        self.assertTrue(report["strict_targeted_source_found"])
        self.assertEqual(report["next_action"], "strict_targeted_source_manifest_and_no_fill_cache_precheck")
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_readiness_marks_selected_cache_exhaustion_when_no_source_found_after_sweep(self):
        report = build_report(
            manifest={"strict_targeted_shadow_sweep_manifest_ready": True, "scenario_count": 36},
            cache_coverage={
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
                "expected_env_count": 36,
            },
            execution_record={"strict_targeted_shadow_sweep_v29_authorized": True},
            strict_sourcing={
                "strict_targeted_source_found": False,
                "expected_strict_eligible_applied_steps": 0,
                "would_apply_steps": 4,
                "strict_filtered_steps": 4,
                "blocker_taxonomy": ["strict_targeted_source_not_found"],
            },
        )

        self.assertIn("strict_targeted_source_exhausted_in_selected_cache", report["failure_taxonomy"])
        self.assertEqual(report["next_action"], "new_scenario_cache_discovery_for_strict_targeted_source")


if __name__ == "__main__":
    unittest.main()
