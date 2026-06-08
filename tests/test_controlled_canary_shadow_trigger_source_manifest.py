import unittest

from gl_gym.experiments.controlled_canary_shadow_trigger_source_manifest import build_report


class TestControlledCanaryShadowTriggerSourceManifest(unittest.TestCase):
    def test_manifest_is_shadow_only_and_has_no_execution_authorization(self):
        report = build_report(
            sourcing={
                "shadow_trigger_sources_found": True,
                "source_scenarios": [
                    {
                        "scenario_id": "y2015_d240_s42_n240",
                        "year": 2015,
                        "day": 240,
                        "seed": 42,
                        "max_steps": 240,
                        "near_miss_rows": 6,
                    }
                ],
            },
            selected_cache_path="cache.json",
        )

        self.assertTrue(report["shadow_trigger_source_manifest_ready"])
        self.assertTrue(report["shadow_only"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["controlled_replay_execution_allowed"])
        self.assertEqual(report["scenario_ids"], ["y2015_d240_s42_n240"])


if __name__ == "__main__":
    unittest.main()
