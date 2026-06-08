import unittest

from gl_gym.experiments.safety_boundary_test_oracle_review import build_report


class TestSafetyBoundaryTestOracleReview(unittest.TestCase):
    def test_reviews_safety_boundary_oracle_hunks_as_independent_suite(self):
        classification = {
            "safety_boundary_test_oracle_hunk_count": 2,
            "rows": [
                {
                    "hunk_id": "h001",
                    "path": "tests/test_planning_extensions.py",
                    "hunk_header": "@@ -1 +1 @@",
                    "safety_boundary_test_oracle": True,
                    "sample_changed_lines": [
                        'def test_tomato_safety_v2_hot_dry_guard(self):',
                        'self.assertIn("hot_dry_cooling_guard", info["reasons"])',
                    ],
                },
                {
                    "hunk_id": "h002",
                    "path": "tests/test_planning_extensions.py",
                    "hunk_header": "@@ -2 +2 @@",
                    "safety_boundary_test_oracle": True,
                    "sample_changed_lines": [
                        'self.assertTrue(info["canopy_dew_buffer"])',
                    ],
                },
            ],
        }

        report = build_report(classification)

        self.assertTrue(report["safety_boundary_test_oracle_review_complete"])
        self.assertEqual(report["safety_boundary_test_oracle_hunk_count"], 2)
        self.assertTrue(report["independent_safety_test_suite_candidate"])
        self.assertFalse(report["allowed_in_replay_protocol_equivalence"])
        self.assertIn("hot_dry", report["rows"][0]["coverage_tags"])
        self.assertIn("tomato_safety_v2", report["rows"][0]["coverage_tags"])
        self.assertEqual(report["rows"][0]["recommended_suite_treatment"], "independent_safety_test_suite_candidate")


if __name__ == "__main__":
    unittest.main()
