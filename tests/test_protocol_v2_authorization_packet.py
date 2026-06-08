import unittest

from gl_gym.experiments.protocol_v2_authorization_packet import build_report


class TestProtocolV2AuthorizationPacket(unittest.TestCase):
    def test_defaults_to_partial_accept_without_authorizing_baseline(self):
        report = build_report(
            {
                "protocol_delta_explained": True,
                "unknown_hunk_count": 0,
                "category_counts": {
                    "metadata_report_only": 1,
                    "stricter_safety_gate": 1,
                    "runner_control_surface": 1,
                    "cache_behavior": 1,
                    "summary_aggregation": 1,
                    "test_oracle": 1,
                },
                "blocked_categories_before_promotion_evidence": [
                    "runner_control_surface",
                    "test_oracle",
                ],
                "safety_boundary_test_oracle_hunk_count": 1,
                "safety_boundary_test_oracle_changed": True,
                "rows": [{"hunk_id": "h001"}],
            }
        )

        self.assertFalse(report["protocol_baseline_authorized"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertEqual(report["recommended_decision"], "partial_accept_review_required")
        self.assertIn("metadata_report_only", report["accepted_candidate_categories_for_review"])
        self.assertIn("stricter_safety_gate", report["accepted_candidate_categories_for_review"])
        self.assertIn("runner_control_surface", report["blocked_categories_before_promotion_evidence"])
        self.assertIn("cache_behavior", report["blocked_categories_before_promotion_evidence"])
        self.assertIn("summary_aggregation", report["blocked_categories_before_promotion_evidence"])
        self.assertTrue(report["safety_boundary_test_oracle_requires_separate_review"])


if __name__ == "__main__":
    unittest.main()
