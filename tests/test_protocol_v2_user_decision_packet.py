import unittest

from gl_gym.experiments.protocol_v2_user_decision_packet import build_report


class TestProtocolV2UserDecisionPacket(unittest.TestCase):
    def test_recommends_partial_accept_without_authorizing_baseline(self):
        report = build_report(
            classification={
                "category_counts": {
                    "metadata_report_only": 1,
                    "stricter_safety_gate": 1,
                    "runner_control_surface": 1,
                    "test_oracle": 1,
                },
                "safety_boundary_test_oracle_hunk_count": 7,
            },
            authorization={
                "blocked_categories_before_promotion_evidence": [
                    "runner_control_surface",
                    "test_oracle",
                ]
            },
        )

        self.assertEqual(report["recommended_decision"], "partial_accept_review_required")
        self.assertFalse(report["protocol_baseline_authorized"])
        self.assertFalse(report["protocol_user_decision_complete"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertTrue(report["historical_benchmark_pass_context"]["historical_context_only"])
        self.assertTrue(report["historical_benchmark_pass_context"]["not_current_working_tree_evidence"])
        self.assertIn("partial_accept_protocol_v2", {option["option_id"] for option in report["options"]})

    def test_records_option_b_decision_without_authorizing_baseline(self):
        report = build_report(
            classification={"category_counts": {"metadata_report_only": 1}},
            authorization={"blocked_categories_before_promotion_evidence": []},
            selected_decision="partial_accept_protocol_v2",
        )

        self.assertEqual(report["schema_version"], "protocol_v2_user_decision_record_v1")
        self.assertEqual(report["selected_decision"], "partial_accept_protocol_v2")
        self.assertTrue(report["protocol_user_decision_complete"])
        self.assertFalse(report["protocol_baseline_authorized"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertEqual(report["next_action"], "resolve_protocol_v2_blocked_categories")


if __name__ == "__main__":
    unittest.main()
