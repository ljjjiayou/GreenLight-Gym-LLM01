import unittest

from gl_gym.experiments.proxy_warning_not_actionable_audit import build_report


class TestProxyWarningNotActionableAudit(unittest.TestCase):
    def test_extracts_hard_warning_rows(self):
        counterfactual = {
            "rows": [
                {
                    "preset": "hot_dry_relief",
                    "step": 233,
                    "counterfactual_conclusion": "proxy_warning_not_actionable",
                    "pre_score_pred_canopy_margin": -0.1,
                    "post_guardrail_pred_canopy_margin": -0.05,
                    "actual_next_canopy_margin": -0.2,
                },
                {
                    "preset": "balanced",
                    "step": 221,
                    "counterfactual_conclusion": "vent_floor_likely_sufficient",
                    "pre_score_pred_canopy_margin": 2.0,
                    "post_guardrail_pred_canopy_margin": 2.0,
                    "actual_next_canopy_margin": 1.0,
                },
            ]
        }

        report = build_report(counterfactual)

        self.assertEqual(report["proxy_warning_not_actionable_count"], 1)
        self.assertEqual(report["actual_hard_crossing_count"], 1)
        self.assertTrue(report["requires_warning_to_action_blocker_shadow_design"])


if __name__ == "__main__":
    unittest.main()
