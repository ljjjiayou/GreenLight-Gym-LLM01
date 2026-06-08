import unittest

from gl_gym.experiments.post_guardrail_vent_floor_shadow_policy_v2_design import build_report


class TestPostGuardrailVentFloorShadowPolicyV2Design(unittest.TestCase):
    def test_splits_vent_floor_rows_from_hard_warning_rows(self):
        v1 = {
            "rows": [
                {"policy_label": "vent_floor_shadow_block", "preset": "balanced", "step": 221, "shadow_floor_vent": 0.3},
                {"policy_label": "hard_warning_to_action_shadow_needed", "preset": "hot_dry_relief", "step": 233},
            ]
        }

        report = build_report(v1)

        self.assertEqual(report["design_row_count"], 1)
        self.assertEqual(report["excluded_hard_warning_rows"], 1)
        self.assertEqual(report["rows"][0]["trigger_condition"], "post_guardrail_delta_vent_lt0_and_canopy_or_dry_risk_active")
        self.assertEqual(report["excluded_rows"][0]["reason"], "belongs_to_warning_to_action_blocker_not_vent_floor_v2")


if __name__ == "__main__":
    unittest.main()
