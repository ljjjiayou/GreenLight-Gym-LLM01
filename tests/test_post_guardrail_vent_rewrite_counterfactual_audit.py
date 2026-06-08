import unittest

from gl_gym.experiments.post_guardrail_vent_rewrite_counterfactual_audit import build_report


class TestPostGuardrailVentRewriteCounterfactualAudit(unittest.TestCase):
    def test_outputs_five_variants_and_detects_vent_floor_signal(self):
        rewrite = {
            "rows": [
                {
                    "scenario_id": "case",
                    "preset": "balanced",
                    "step": 10,
                    "selected_candidate_name": "anchor",
                    "pre_score_action": {"heat": 0, "co2": 0, "screen": 0.7, "vent": 0.5, "lamp": 0, "shade": 0},
                    "post_guardrail_action": {"heat": 0, "co2": 0, "screen": 0.7, "vent": 0.1, "lamp": 0, "shade": 0},
                    "delta_heat": 0,
                    "delta_screen": 0,
                    "delta_vent": -0.4,
                    "delta_shade": 0,
                    "pre_score_pred_canopy_margin": 1.0,
                    "post_guardrail_pred_canopy_margin": 0.8,
                    "actual_next_canopy_margin": 0.2,
                    "fine_grained_root_causes": ["ventilation_decrease_dominant"],
                }
            ]
        }
        report = build_report(rewrite)

        self.assertTrue(report["vent_floor_likely_sufficient"])
        self.assertEqual(report["conclusion_counts"]["vent_floor_likely_sufficient"], 1)
        self.assertEqual(len(report["rows"][0]["variants"]), 5)
        names = {item["variant"] for item in report["rows"][0]["variants"]}
        self.assertIn("post_guardrail_keep_pre_score_vent", names)
        self.assertIn("post_guardrail_canopy_aware_vent_floor", names)

    def test_proxy_warning_not_actionable_blocks_simple_vent_floor_claim(self):
        rewrite = {
            "rows": [
                {
                    "pre_score_action": {"screen": 1.0, "vent": 0.5},
                    "post_guardrail_action": {"screen": 1.0, "vent": 0.2},
                    "delta_vent": -0.3,
                    "pre_score_pred_canopy_margin": -0.1,
                    "post_guardrail_pred_canopy_margin": -0.2,
                    "fine_grained_root_causes": ["proxy_warning_exists_but_not_actionable"],
                }
            ]
        }
        report = build_report(rewrite)
        self.assertTrue(report["proxy_warning_not_actionable"])
        self.assertEqual(report["rows"][0]["counterfactual_conclusion"], "proxy_warning_not_actionable")


if __name__ == "__main__":
    unittest.main()
