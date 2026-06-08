import unittest

from gl_gym.experiments.post_guardrail_vent_floor_shadow_policy_audit import build_report


class TestPostGuardrailVentFloorShadowPolicyAudit(unittest.TestCase):
    def test_labels_vent_floor_and_warning_rows(self):
        counterfactual = {
            "rows": [
                {
                    "preset": "balanced",
                    "step": 1,
                    "counterfactual_conclusion": "vent_floor_likely_sufficient",
                    "delta_vent": -0.3,
                    "pre_score_pred_canopy_margin": 1.0,
                    "post_guardrail_pred_canopy_margin": 0.9,
                    "actual_next_canopy_margin": 0.8,
                    "variants": [
                        {"variant": "pre_score_action", "action": {"vent": 0.5}},
                        {"variant": "post_guardrail_action", "action": {"vent": 0.1}},
                        {
                            "variant": "post_guardrail_canopy_aware_vent_floor",
                            "action": {"vent": 0.3},
                            "predicted_canopy_margin": 1.1,
                        },
                    ],
                },
                {
                    "preset": "hot_dry_relief",
                    "step": 2,
                    "counterfactual_conclusion": "proxy_warning_not_actionable",
                    "delta_vent": -0.2,
                    "pre_score_pred_canopy_margin": -0.1,
                    "post_guardrail_pred_canopy_margin": -0.05,
                    "actual_next_canopy_margin": -0.2,
                    "variants": [
                        {"variant": "post_guardrail_action", "action": {"vent": 0.2}},
                        {
                            "variant": "post_guardrail_canopy_aware_vent_floor",
                            "action": {"vent": 0.3},
                            "predicted_canopy_margin": 0.0,
                        },
                    ],
                },
            ]
        }

        report = build_report(counterfactual)

        self.assertEqual(report["vent_floor_shadow_block_rows"], 1)
        self.assertEqual(report["hard_warning_to_action_rows"], 1)
        self.assertEqual(report["would_block_count"], 2)
        self.assertFalse(report["ready_for_controlled_replay"])


if __name__ == "__main__":
    unittest.main()
