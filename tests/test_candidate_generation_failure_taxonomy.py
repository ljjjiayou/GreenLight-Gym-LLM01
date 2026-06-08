import unittest

from gl_gym.experiments.candidate_generation_failure_taxonomy import build_report


class TestCandidateGenerationFailureTaxonomy(unittest.TestCase):
    def test_taxonomizes_repair_classes_and_reasons(self):
        report = build_report(
            {
                "traces": [
                    {
                        "records": [
                            {
                                "scenario_id": "case",
                                "category": "canonical_failure",
                                "preset": "balanced",
                                "step": 1,
                                "repair_class": "repair_post_guardrail_safe",
                                "hypothetical_candidates": [
                                    {"feasibility_reason": "ok", "tradeoff_quality": "useful_dry_relief"}
                                ],
                            },
                            {
                                "scenario_id": "case",
                                "category": "pure_hot_dry_false_positive",
                                "preset": "balanced",
                                "step": 2,
                                "repair_class": "no_safe_repair_found",
                                "hypothetical_candidates": [
                                    {"feasibility_reason": "canopy_risk_v2", "tradeoff_quality": "dry_tradeoff"}
                                ],
                            },
                            {
                                "scenario_id": "case",
                                "category": "pure_hot_dry_false_positive",
                                "preset": "balanced",
                                "step": 3,
                                "repair_class": "repair_dry_relief_lost",
                                "hypothetical_candidates": [
                                    {
                                        "feasibility_reason": "ok",
                                        "tradeoff_quality": "conservative_ineffective",
                                        "conservative_ineffective_action": True,
                                    }
                                ],
                            },
                        ]
                    }
                ]
            }
        )

        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["taxonomy_counts"]["localized_safe_repair_signal"], 1)
        self.assertEqual(report["taxonomy_counts"]["candidate_space_insufficiency"], 1)
        self.assertEqual(report["taxonomy_counts"]["dry_relief_lost"], 1)
        self.assertEqual(report["next_action"], "candidate_space_redesign_shadow")


if __name__ == "__main__":
    unittest.main()
