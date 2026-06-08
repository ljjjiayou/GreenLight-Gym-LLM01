import unittest

from gl_gym.experiments.controlled_canary_triggered_scenario_manifest import build_report


class TestControlledCanaryTriggeredScenarioManifest(unittest.TestCase):
    def test_manifest_uses_only_discovered_scenarios(self):
        report = build_report(
            discovery={
                "trigger_candidates_found": True,
                "qualified_trigger_scenarios": [
                    {"scenario_id": "y2020_d180_s44_n240", "year": 2020, "day": 180, "seed": 44, "max_steps": 240}
                ],
            },
            selected_cache_path="cache.json",
            output_root="out",
        )

        self.assertTrue(report["triggered_manifest_ready"])
        self.assertEqual(report["scenario_ids"], ["y2020_d180_s44_n240"])
        self.assertEqual(report["command_groups"][0]["expected_scenario_ids"], ["y2020_d180_s44_n240"])
        self.assertFalse(report["controlled_replay_allowed"])

    def test_manifest_blocks_without_discovery(self):
        report = build_report(
            discovery={"trigger_candidates_found": False, "qualified_trigger_scenarios": []},
            selected_cache_path="cache.json",
            output_root="out",
        )

        self.assertFalse(report["triggered_manifest_ready"])
        self.assertEqual(report["next_action"], "strict_trigger_discovery_required")


if __name__ == "__main__":
    unittest.main()
