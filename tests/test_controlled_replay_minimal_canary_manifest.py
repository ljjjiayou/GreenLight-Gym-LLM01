import unittest

from gl_gym.experiments.controlled_replay_minimal_canary_manifest import build_report


class TestControlledReplayMinimalCanaryManifest(unittest.TestCase):
    def test_manifest_groups_avoid_extra_cartesian_scenarios(self):
        report = build_report(selected_cache_path="cache.json", output_root="out")

        self.assertTrue(report["minimal_canary_manifest_ready"])
        self.assertEqual(
            report["scenario_ids"],
            ["y2015_d180_s44_n240", "y2020_d120_s44_n240", "y2020_d180_s44_n240"],
        )
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertTrue(all(group["cartesian_product_safe"] for group in report["command_groups"]))


if __name__ == "__main__":
    unittest.main()
