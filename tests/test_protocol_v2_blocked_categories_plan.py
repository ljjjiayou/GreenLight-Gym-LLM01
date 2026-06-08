import unittest

from gl_gym.experiments.protocol_v2_blocked_categories_plan import build_report


class TestProtocolV2BlockedCategoriesPlan(unittest.TestCase):
    def test_blocks_high_risk_categories_from_promotion_evidence(self):
        report = build_report(
            classification={
                "category_counts": {
                    "cache_behavior": 2,
                    "runner_control_surface": 3,
                    "summary_aggregation": 1,
                    "test_oracle": 4,
                    "stricter_safety_gate": 1,
                }
            },
            user_decision={"protocol_user_decision_complete": False},
        )

        rows = {row["category"]: row for row in report["rows"]}
        self.assertFalse(report["blocked_categories_resolved"])
        for category in ("cache_behavior", "runner_control_surface", "summary_aggregation", "test_oracle"):
            self.assertTrue(rows[category]["promotion_blocker"])
            self.assertFalse(rows[category]["allowed_for_promotion_evidence"])
            self.assertIn("explicitly_authorize", rows[category]["allowed_next_actions"])
        self.assertTrue(report["stricter_safety_gate_requires_explicit_authorization"])
        self.assertFalse(report["metadata_replay_allowed"])

    def test_option_b_decision_does_not_resolve_blocked_categories(self):
        report = build_report(
            classification={
                "category_counts": {
                    "cache_behavior": 1,
                    "runner_control_surface": 1,
                    "summary_aggregation": 1,
                    "test_oracle": 1,
                }
            },
            user_decision={
                "protocol_user_decision_complete": True,
                "selected_decision": "partial_accept_protocol_v2",
            },
        )

        self.assertTrue(report["protocol_user_decision_complete"])
        self.assertEqual(report["selected_protocol_decision"], "partial_accept_protocol_v2")
        self.assertTrue(report["blocked_categories_resolution_plan_ready"])
        self.assertFalse(report["blocked_categories_resolved"])
        self.assertFalse(report["protocol_baseline_authorized"])

    def test_evidence_boundary_isolation_resolves_v1_preflight_only(self):
        report = build_report(
            classification={
                "category_counts": {
                    "cache_behavior": 1,
                    "runner_control_surface": 1,
                    "summary_aggregation": 1,
                    "test_oracle": 1,
                }
            },
            user_decision={
                "protocol_user_decision_complete": True,
                "selected_decision": "partial_accept_protocol_v2",
            },
            evidence_boundary_isolation=True,
        )

        self.assertEqual(report["schema_version"], "protocol_v2_blocked_categories_resolution_v3")
        self.assertTrue(report["blocked_categories_resolved_for_v1_preflight"])
        self.assertFalse(report["blocked_categories_resolved_for_protocol_v2"])
        self.assertFalse(report["protocol_baseline_authorized"])
        self.assertEqual(report["v1_preflight_resolution_strategy"], "evidence_boundary_isolation")
        for row in report["rows"]:
            self.assertFalse(row["included_in_v1_metadata_replay_baseline"])
            self.assertFalse(row["included_in_protocol_v2_baseline"])


if __name__ == "__main__":
    unittest.main()
