import unittest

from gl_gym.agent.ppo_strategy_labeler import label_strategy


def _row(**overrides):
    row = {
        "temp_air": 17.0,
        "rh_air": 88.0,
        "co2_air": 430.0,
        "glob_rad": 0.0,
        "hour_of_day": 4.0,
        "dew_margin_air": 0.8,
        "canopy_dew_margin": 0.8,
        "forecast_humidity_risk": 0.5,
        "u_heating": 0.0,
        "u_co2": 0.0,
        "u_screen": 0.6,
        "u_ventilation": 0.0,
        "u_lighting": 0.0,
        "u_shading": 0.0,
    }
    row.update(overrides)
    return row


class TestPPOStrategyLabeler(unittest.TestCase):
    def test_labels_heat_vent_dehumidification_only_when_humidity_risk_is_clear(self):
        label = label_strategy(
            _row(
                temp_air=16.0,
                rh_air=92.0,
                u_heating=0.28,
                u_ventilation=0.70,
                u_screen=0.35,
            )
        )

        self.assertEqual(label.label, "heat_vent_dehumidify")
        self.assertGreaterEqual(label.confidence, 0.55)
        self.assertIn("humidity risk", " ".join(label.reasons))

    def test_labels_vent_only_dehumidification_when_heat_is_low(self):
        label = label_strategy(
            _row(
                temp_air=17.0,
                rh_air=90.0,
                u_heating=0.02,
                u_ventilation=0.62,
                u_screen=0.55,
            )
        )

        self.assertEqual(label.label, "vent_only_dehumidify")
        self.assertGreaterEqual(label.confidence, 0.55)

    def test_labels_heat_preservation_for_cold_closed_screen_low_vent(self):
        label = label_strategy(
            _row(
                temp_air=13.5,
                rh_air=78.0,
                hour_of_day=2.0,
                u_heating=0.35,
                u_ventilation=0.02,
                u_screen=0.95,
            )
        )

        self.assertEqual(label.label, "heat_preservation")
        self.assertGreaterEqual(label.confidence, 0.55)

    def test_labels_free_air_exchange_for_low_cost_venting(self):
        label = label_strategy(
            _row(
                temp_air=16.5,
                rh_air=84.0,
                u_heating=0.0,
                u_co2=0.0,
                u_screen=0.0,
                u_ventilation=1.0,
                u_lighting=0.0,
            )
        )

        self.assertEqual(label.label, "free_air_exchange")
        self.assertGreaterEqual(label.confidence, 0.55)

    def test_rejects_ambiguous_or_weak_mixed_action(self):
        label = label_strategy(
            _row(
                temp_air=20.0,
                rh_air=80.0,
                glob_rad=140.0,
                hour_of_day=12.0,
                u_heating=0.10,
                u_co2=0.04,
                u_screen=0.50,
                u_ventilation=0.24,
                u_lighting=0.02,
            )
        )

        self.assertIn(label.label, {"ambiguous", "unknown"})
        self.assertLess(label.confidence, 0.75)


if __name__ == "__main__":
    unittest.main()
