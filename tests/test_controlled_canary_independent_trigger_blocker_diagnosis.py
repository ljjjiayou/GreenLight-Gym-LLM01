import unittest

from gl_gym.experiments.controlled_canary_independent_trigger_blocker_diagnosis import build_report


class TestControlledCanaryIndependentTriggerBlockerDiagnosis(unittest.TestCase):
    def test_diagnosis_explains_all_strict_filtered_steps(self):
        discovery = {
            "discovery_decision": "blocked_no_independent_trigger_candidate",
            "independent_trigger_candidates_found": False,
            "strict_filtered_steps": 2,
            "excluded_scenario_ids": ["y2015_d120_s42_n240"],
            "blocker_taxonomy": ["no_independent_strict_trigger_candidate"],
            "traces": [
                {
                    "scenario_id": "y2015_d240_s42_n240",
                    "controller": "llm_rspc_v2_hot_dry_proposer_strict",
                    "would_apply_rows": 2,
                    "strict_filtered_rows": 2,
                    "expected_strict_eligible_applied_rows": 0,
                    "reason_counts": {"margin_below_strict_min": 2},
                    "candidate_counts": {"shadow_hot_dry_shade_preempt": 2},
                }
            ],
            "sample_rows": [
                {
                    "scenario_id": "y2015_d240_s42_n240",
                    "strict_filtered": True,
                    "candidate": "shadow_hot_dry_shade_preempt",
                    "variant": "dry_vpd_x2",
                    "reason": "margin_below_strict_min",
                    "margin": 0.18,
                    "min_margin": 0.2,
                },
                {
                    "scenario_id": "y2015_d240_s42_n240",
                    "strict_filtered": True,
                    "candidate": "shadow_hot_dry_shade_preempt",
                    "variant": "dry_vpd_x2",
                    "reason": "margin_below_strict_min",
                    "margin": 0.19,
                    "min_margin": 0.2,
                },
            ],
        }
        report = build_report(
            readiness_v24={"next_action": "independent_trigger_blocker_diagnosis"},
            discovery=discovery,
        )

        self.assertTrue(report["blocker_diagnosis_complete"])
        self.assertEqual(report["strict_filter_reason_distribution"]["margin_below_strict_min"], 2)
        self.assertEqual(report["next_action"], "near_miss_shadow_trigger_sourcing")
        self.assertFalse(report["controlled_replay_allowed"])

    def test_v32_readiness_next_action_is_accepted(self):
        discovery = {
            "independent_trigger_candidates_found": False,
            "strict_filtered_steps": 0,
            "blocker_taxonomy": ["no_independent_strict_trigger_candidate"],
            "traces": [],
            "sample_rows": [],
        }
        report = build_report(
            readiness_v24={"next_action": "new_shadow_only_scenario_cache_discovery_after_independent_holdout_blocked"},
            discovery=discovery,
        )

        self.assertTrue(report["blocker_diagnosis_complete"])
        self.assertFalse(report["performance_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
