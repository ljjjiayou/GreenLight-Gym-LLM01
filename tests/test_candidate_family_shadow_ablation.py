import unittest

from gl_gym.experiments.candidate_family_shadow_ablation import build_report


class TestCandidateFamilyShadowAblation(unittest.TestCase):
    def test_aggregates_candidate_family_metrics(self):
        suite = {
            "traces": [
                {
                    "records": [
                        {
                            "scenario_id": "case",
                            "preset": "balanced",
                            "step": 1,
                            "repair_class": "repair_post_guardrail_safe",
                            "hypothetical_candidates": [
                                {
                                    "name": "canopy_safe_no_screen_gain",
                                    "candidate_family": "canopy_safe_hold_screen",
                                    "feasible": True,
                                    "dry_relief_useful": True,
                                    "post_guardrail_remains_safe": True,
                                    "dry_tradeoff": False,
                                },
                                {
                                    "name": "canopy_safe_min_vent",
                                    "candidate_family": "canopy_safe_min_vent",
                                    "feasible": True,
                                    "dry_relief_useful": False,
                                    "post_guardrail_remains_safe": True,
                                    "dry_tradeoff": True,
                                },
                            ],
                        },
                        {
                            "repair_class": "no_safe_repair_found",
                            "hypothetical_candidates": [
                                {
                                    "name": "bad",
                                    "candidate_family": "canopy_safe_balanced_repair",
                                    "feasible": False,
                                    "post_guardrail_reintroduces_canopy_risk": True,
                                }
                            ],
                        },
                    ]
                }
            ]
        }
        report = build_report(suite)
        rows = {item["family"]: item for item in report["families"]}

        self.assertEqual(rows["canopy_safe_hold_screen"]["useful_dry_relief_count"], 1)
        self.assertEqual(rows["vent_floor_repair"]["dry_relief_lost_count"], 1)
        self.assertEqual(rows["dry_relief_preserving_repair"]["hard_safety_regression_count"], 1)
        self.assertFalse(report["controlled_replay_allowed"])


if __name__ == "__main__":
    unittest.main()
