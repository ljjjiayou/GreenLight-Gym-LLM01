import unittest

from gl_gym.experiments.regime_counterexample_shadow_audit import build_report


class TestRegimeCounterexampleShadowAudit(unittest.TestCase):
    def test_converts_manifest_without_claiming_replay(self):
        manifest = {
            "windows": [
                {
                    "scenario_id": "neutral",
                    "preset": "balanced",
                    "regime": "neutral_no_risk",
                    "center_step": 10,
                    "state_summary": {"canopy_dew_margin": 5.0, "dew_margin_air": 4.0, "rh_air": 60, "vpd_air": 1.0},
                },
                {
                    "scenario_id": "cold",
                    "preset": "balanced",
                    "regime": "cold_humid",
                    "center_step": 11,
                    "state_summary": {"temp_air": 14.0, "rh_air": 86.0, "wind_speed": 2.0},
                },
            ]
        }

        report = build_report(manifest)

        self.assertFalse(report["replay_run"])
        self.assertEqual(report["regime_counts"]["neutral_no_risk"], 1)
        self.assertIn("neutral_should_not_trigger_blocker", report["shadow_audit_flag_counts"])
        self.assertIn("cold_humid_overvent_risk", report["shadow_audit_flag_counts"])


if __name__ == "__main__":
    unittest.main()
