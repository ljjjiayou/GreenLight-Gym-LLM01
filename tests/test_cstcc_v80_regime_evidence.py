import unittest

from gl_gym.cstcc.contracts import asdict_clean
from gl_gym.cstcc.regime_evidence import regime_evidence_score
from gl_gym.cstcc.temporal_context import build_temporal_context


class TestCSTCCV80RegimeEvidence(unittest.TestCase):
    def test_high_vpd_evidence_is_regime_specific(self):
        context = build_temporal_context(
            [
                {"temp_air": 28.0, "rh_air": 52.0, "vpd_air": 2.1},
                {"temp_air": 29.0, "rh_air": 48.0, "vpd_air": 2.5},
                {"temp_air": 30.0, "rh_air": 44.0, "vpd_air": 2.9},
            ],
            action_history=[],
        )
        score, reasons = regime_evidence_score("HIGH_VPD_DRY_STRESS", asdict_clean(context), {})

        self.assertGreater(score, 0.5)
        self.assertIn("vpd_slope_score", reasons)
        self.assertIn("rh_decline_score", reasons)

    def test_unrelated_low_light_regime_gets_low_evidence_without_light_data(self):
        context = build_temporal_context(
            [
                {"temp_air": 28.0, "rh_air": 52.0, "vpd_air": 2.1},
                {"temp_air": 29.0, "rh_air": 48.0, "vpd_air": 2.5},
                {"temp_air": 30.0, "rh_air": 44.0, "vpd_air": 2.9},
            ],
            action_history=[],
        )
        score, reasons = regime_evidence_score("LOW_LIGHT_ENERGY_SAVING", asdict_clean(context), {})

        self.assertLess(score, 0.5)
        self.assertIn("weather_forecast_available", reasons)

    def test_invalid_regime_scores_zero(self):
        score, reasons = regime_evidence_score("NEW_LLM_FREEFORM_REGIME", {}, {})

        self.assertEqual(score, 0.0)
        self.assertEqual(reasons["invalid_regime"], "NEW_LLM_FREEFORM_REGIME")


if __name__ == "__main__":
    unittest.main()
