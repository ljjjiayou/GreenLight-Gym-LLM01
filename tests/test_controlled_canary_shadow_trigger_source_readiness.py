import unittest

from gl_gym.experiments.controlled_canary_shadow_trigger_source_readiness import build_report


class TestControlledCanaryShadowTriggerSourceReadiness(unittest.TestCase):
    def test_v25_never_allows_controlled_replay_even_with_cache_precheck(self):
        report = build_report(
            readiness_v24={"wave2_controlled_canary_pass": False},
            blocker_diagnosis={"blocker_diagnosis_complete": True},
            sourcing={"shadow_trigger_sources_found": True, "source_scenarios": [{"scenario_id": "s"}]},
            source_manifest={"shadow_trigger_source_manifest_ready": True, "scenario_ids": ["s"]},
            cache_coverage={
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
            },
        )

        self.assertEqual(report["next_action"], "shadow_only_trigger_source_sweep_design_required")
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["controlled_replay_execution_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_no_source_points_to_new_shadow_sweep(self):
        report = build_report(
            readiness_v24={},
            blocker_diagnosis={"blocker_diagnosis_complete": True},
            sourcing={"shadow_trigger_sources_found": False, "blocker_taxonomy": ["no_near_miss_shadow_trigger_source"]},
        )

        self.assertEqual(report["next_action"], "independent_trigger_source_exhausted_or_needs_new_shadow_sweep")
        self.assertFalse(report["controlled_replay_allowed"])


if __name__ == "__main__":
    unittest.main()
