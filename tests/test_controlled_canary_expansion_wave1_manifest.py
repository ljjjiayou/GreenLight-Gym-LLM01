import unittest

from gl_gym.experiments.controlled_canary_expansion_wave1_manifest import build_report


class TestControlledCanaryExpansionWave1Manifest(unittest.TestCase):
    def test_manifest_contains_only_fixed_five_scenarios(self):
        report = build_report(
            admission_review={"controlled_canary_expansion_admission_pass": True},
            selected_cache_path="cache.json",
            output_root="out",
        )

        self.assertTrue(report["wave1_manifest_ready"])
        self.assertEqual(
            report["scenario_ids"],
            [
                "y2020_d120_s44_n240",
                "y2010_d120_s44_n240",
                "y2010_d180_s44_n240",
                "y2015_d180_s44_n240",
                "y2020_d180_s44_n240",
            ],
        )
        self.assertEqual(len(report["command_groups"]), 5)
        self.assertTrue(all(group["cartesian_product_safe"] for group in report["command_groups"]))
        self.assertFalse(report["controlled_replay_allowed"])

    def test_manifest_blocks_without_admission(self):
        report = build_report(
            admission_review={"controlled_canary_expansion_admission_pass": False},
            selected_cache_path="cache.json",
            output_root="out",
        )

        self.assertFalse(report["wave1_manifest_ready"])
        self.assertEqual(report["scope"], "blocked")


if __name__ == "__main__":
    unittest.main()
