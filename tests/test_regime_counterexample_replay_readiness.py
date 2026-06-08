import unittest

from gl_gym.experiments.regime_counterexample_replay_readiness import REPLAY_ORDER, build_report


class TestRegimeCounterexampleReplayReadiness(unittest.TestCase):
    def test_readiness_has_fixed_order_and_does_not_run_replay(self):
        manifest = {
            "windows": [
                {
                    "scenario_id": "neutral",
                    "preset": "balanced",
                    "regime": "neutral_no_risk",
                    "start_step": 1,
                    "end_step": 3,
                    "center_step": 2,
                    "state_summary": {},
                }
            ]
        }

        report = build_report(manifest)

        self.assertEqual(report["recommended_replay_order"], REPLAY_ORDER)
        self.assertTrue(report["ready_for_future_shadow_replay_input"])
        self.assertFalse(report["replay_run"])
        self.assertFalse(report["cache_fill_run"])


if __name__ == "__main__":
    unittest.main()
