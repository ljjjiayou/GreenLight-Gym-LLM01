import unittest

from gl_gym.experiments.controlled_canary_near_miss_source_catalog import build_report


class TestControlledCanaryNearMissSourceCatalog(unittest.TestCase):
    def test_near_miss_catalog_does_not_create_strict_applied_evidence(self):
        report = build_report(
            blocker_diagnosis={
                "blocker_diagnosis_complete": True,
                "strict_filtered_steps": 1,
                "strict_filter_explained_steps": 1,
                "blocker_taxonomy": ["strict_candidates_filtered"],
            },
            discovery={
                "strict_filtered_steps": 1,
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
                        "step": 57,
                    }
                ],
            },
        )

        self.assertTrue(report["near_miss_catalog_ready"])
        self.assertFalse(report["strict_applied_evidence"])
        self.assertEqual(report["source_scenarios"][0]["strict_applied_steps"], 0)
        self.assertTrue(report["near_miss_rows"][0]["source_hint_only"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_strict_applied_rows_are_not_near_miss_sources(self):
        report = build_report(
            blocker_diagnosis={"blocker_diagnosis_complete": True, "strict_filtered_steps": 0},
            discovery={
                "strict_filtered_steps": 0,
                "sample_rows": [
                    {
                        "scenario_id": "y2015_d120_s42_n240",
                        "strict_filtered": True,
                        "strict_eligible_applied": True,
                    }
                ],
            },
        )

        self.assertTrue(report["near_miss_catalog_ready"])
        self.assertEqual(report["near_miss_row_count"], 0)
        self.assertEqual(report["near_miss_scenario_count"], 0)


if __name__ == "__main__":
    unittest.main()
