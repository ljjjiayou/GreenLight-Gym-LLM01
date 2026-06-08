import unittest

from gl_gym.experiments.post_guardrail_rule_provenance_audit import build_report


class TestPostGuardrailRuleProvenanceAudit(unittest.TestCase):
    def test_missing_reason_blocks_promotion(self):
        rewrite = {
            "rows": [
                {
                    "scenario_id": "case",
                    "preset": "balanced",
                    "step": 1,
                    "pre_score_action": {"vent": 0.5},
                    "post_guardrail_action": {"vent": 0.1},
                    "delta_vent": -0.4,
                    "delta_heat": 0,
                    "delta_screen": 0,
                    "delta_shade": 0,
                    "tomato_safety_v2_reasons": "",
                    "root_causes": ["guardrail_introduced_risk"],
                }
            ]
        }
        report = build_report(rewrite)

        self.assertTrue(report["unsafe_for_promotion_due_to_missing_provenance"])
        self.assertEqual(report["reason_missing_count"], 1)
        self.assertEqual(report["rows"][0]["which_rule_fired"], "unknown_post_guardrail_rewrite")

    def test_recorded_reason_is_not_missing(self):
        rewrite = {
            "rows": [
                {
                    "delta_vent": -0.2,
                    "tomato_safety_v2_reasons": "dew_protection",
                    "root_causes": ["guardrail_introduced_risk"],
                }
            ]
        }
        report = build_report(rewrite)
        self.assertFalse(report["unsafe_for_promotion_due_to_missing_provenance"])
        self.assertEqual(report["rows"][0]["which_rule_fired"], "tomato_safety_v2")


if __name__ == "__main__":
    unittest.main()
