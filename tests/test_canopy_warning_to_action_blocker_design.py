import unittest

from gl_gym.experiments.canopy_warning_to_action_blocker_design import build_report


class TestCanopyWarningToActionBlockerDesign(unittest.TestCase):
    def test_covers_actual_hard_crossing_warning_rows(self):
        warning = {
            "rows": [
                {"preset": "conservative_dry", "step": 233, "actual_hard_crossing": True},
                {"preset": "hot_dry_relief", "step": 233, "actual_hard_crossing": True},
            ]
        }

        report = build_report(warning)

        self.assertEqual(report["design_row_count"], 2)
        self.assertEqual(report["actual_hard_crossing_rows_covered"], 2)
        self.assertTrue(report["rows"][0]["requires_safe_alternative_check"])
        self.assertFalse(report["controlled_replay_allowed"])


if __name__ == "__main__":
    unittest.main()
