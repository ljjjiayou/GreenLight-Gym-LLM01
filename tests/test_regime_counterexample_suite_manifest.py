import unittest

from gl_gym.experiments.regime_counterexample_suite_manifest import build_report


class TestRegimeCounterexampleSuiteManifest(unittest.TestCase):
    def test_builds_manifest_from_existing_inventory_only(self):
        inventory = {
            "examples": {
                "cold_humid": [
                    {
                        "scenario_id": "cold_case",
                        "preset": "balanced",
                        "step": 3,
                        "temp_air": 15,
                        "rh_air": 85,
                    }
                ],
                "neutral_no_risk": [
                    {
                        "scenario_id": "neutral_case",
                        "preset": "balanced",
                        "step": 100,
                    }
                ],
            }
        }
        report = build_report(inventory, max_windows_per_regime=1, radius=2)

        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["regime_counts"]["cold_humid"], 1)
        self.assertEqual(report["windows_by_regime"]["cold_humid"][0]["start_step"], 1)
        self.assertEqual(report["windows_by_regime"]["cold_humid"][0]["end_step"], 5)
        self.assertEqual(report["regime_counts"]["dawn_dew"], 0)


if __name__ == "__main__":
    unittest.main()
