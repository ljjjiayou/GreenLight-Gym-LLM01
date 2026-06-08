import unittest

from gl_gym.experiments.controlled_canary_shadow_trigger_sourcing import build_report


class TestControlledCanaryShadowTriggerSourcing(unittest.TestCase):
    def test_near_miss_sources_are_shadow_only_not_strict_applied(self):
        report = build_report(
            blocker_diagnosis={"blocker_diagnosis_complete": True},
            discovery={
                "excluded_scenario_ids": ["y2015_d120_s42_n240"],
                "sample_rows": [
                    {
                        "scenario_id": "y2015_d240_s42_n240",
                        "strict_filtered": True,
                        "strict_eligible_applied": False,
                        "reason": "margin_below_strict_min",
                        "candidate": "shadow_hot_dry_shade_preempt",
                        "variant": "dry_vpd_x2",
                        "margin": 0.18,
                        "min_margin": 0.2,
                    },
                    {
                        "scenario_id": "y2015_d120_s42_n240",
                        "strict_filtered": True,
                        "strict_eligible_applied": False,
                        "reason": "margin_below_strict_min",
                    },
                ],
            },
        )

        self.assertTrue(report["shadow_trigger_sources_found"])
        self.assertEqual(report["source_scenarios"][0]["scenario_id"], "y2015_d240_s42_n240")
        self.assertEqual(report["source_scenarios"][0]["strict_applied_steps"], 0)
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_no_near_miss_blocks_source_manifest(self):
        report = build_report(
            blocker_diagnosis={"blocker_diagnosis_complete": True},
            discovery={"excluded_scenario_ids": [], "sample_rows": []},
        )

        self.assertFalse(report["shadow_trigger_sources_found"])
        self.assertEqual(report["next_action"], "independent_trigger_source_exhausted_or_needs_new_shadow_sweep")


if __name__ == "__main__":
    unittest.main()
