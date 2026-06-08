import unittest

from gl_gym.experiments.post_guardrail_joint_prediction_readiness import build_report


class TestPostGuardrailJointPredictionReadiness(unittest.TestCase):
    def test_missing_vpd_blocks_policy_judgment(self):
        vent_policy = {
            "rows": [
                {
                    "preset": "balanced",
                    "step": 221,
                    "policy_label": "vent_floor_shadow_block",
                    "predicted_canopy_margin_after_floor": 1.0,
                    "vpd_prediction_available": False,
                }
            ]
        }

        report = build_report(vent_policy)

        self.assertFalse(report["ready_for_policy_judgment"])
        self.assertIn("predicted_vpd_after_floor", report["missing_field_counts"])
        self.assertIn("predicted_rh_after_floor", report["missing_field_counts"])


if __name__ == "__main__":
    unittest.main()
