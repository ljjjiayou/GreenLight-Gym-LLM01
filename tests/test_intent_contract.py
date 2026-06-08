import unittest
from dataclasses import dataclass

from gl_gym.agent.intent_contract import (
    coerce_intent_contract,
    intent_contract_from_setpoint_plan,
)


@dataclass
class DummyState:
    timestep: int = 12
    hour_of_day: float = 12.0
    temp_air: float = 20.0
    rh_air: float = 70.0
    co2_air: float = 430.0
    glob_rad: float = 200.0
    forecast_rad_mean_1h: float = 200.0
    forecast_rad_peak_2h: float = 200.0
    temp_air_delta_1h: float = 0.0
    temp_out: float = 18.0
    rh_out: float = 70.0
    wind_speed: float = 1.0
    dew_margin_air: float = 3.0
    canopy_dew_margin: float = 3.0


def _plan(**overrides):
    plan = {
        "target_temp": 18.0,
        "target_co2": 430.0,
        "target_rh": 76.0,
        "created_timestep": 12,
        "target_profile": {
            "target_temp": [18.0] * 12,
            "target_co2": [430.0] * 12,
            "target_rh": [76.0] * 12,
        },
    }
    plan.update(overrides)
    if "target_temp" in overrides:
        plan["target_profile"]["target_temp"] = [float(overrides["target_temp"])] * 12
    if "target_co2" in overrides:
        plan["target_profile"]["target_co2"] = [float(overrides["target_co2"])] * 12
    if "target_rh" in overrides:
        plan["target_profile"]["target_rh"] = [float(overrides["target_rh"])] * 12
    return plan


class TestIntentContract(unittest.TestCase):
    def test_setpoint_plan_adapts_to_hot_dry_contract(self):
        state = DummyState(temp_air=30.5, rh_air=49.0, glob_rad=720.0)

        contract = intent_contract_from_setpoint_plan(state, _plan(target_rh=78.0))
        data = contract.to_dict()

        self.assertEqual(data["regime"], "hot_dry_relief")
        self.assertIn("humidity", data["priority"])
        self.assertEqual(data["profile_shape"]["rh"], "hot_dry_protect")
        self.assertEqual(data["profile_shape"]["temp"], "shade_cooling")
        self.assertLessEqual(data["constraints"]["vpd_max"], 1.45)
        self.assertEqual(data["target_range"]["rh"], [75.0, 81.0])

    def test_dawn_high_humidity_uses_predehumidify_shape(self):
        state = DummyState(hour_of_day=5.5, rh_air=91.0, dew_margin_air=0.8, canopy_dew_margin=1.0)

        contract = intent_contract_from_setpoint_plan(state, _plan(target_rh=72.0, target_temp=16.0))
        data = contract.to_dict()

        self.assertEqual(data["regime"], "dawn_predehumidify")
        self.assertEqual(data["profile_shape"]["rh"], "strict_dehumidify_then_relax")
        self.assertEqual(data["profile_shape"]["temp"], "dawn_ramp")
        self.assertEqual(data["constraints"]["rh_hard_max"], 88.0)

    def test_hot_humid_has_temperature_and_humidity_priority(self):
        state = DummyState(temp_air=29.0, rh_air=88.0, glob_rad=550.0, dew_margin_air=1.5)

        contract = intent_contract_from_setpoint_plan(state, _plan(target_rh=74.0, target_temp=23.0))
        data = contract.to_dict()

        self.assertEqual(data["regime"], "hot_humid_relief")
        self.assertEqual(data["profile_shape"]["temp"], "shade_cooling")
        self.assertEqual(data["profile_shape"]["rh"], "strict_dehumidify_then_relax")
        self.assertIn("temperature", data["priority"])
        self.assertIn("humidity", data["priority"])

    def test_cold_humid_and_cold_dry_are_distinct(self):
        cold_humid = DummyState(temp_air=15.8, rh_air=90.0, dew_margin_air=0.9, glob_rad=20.0)
        cold_dry = DummyState(temp_air=15.8, rh_air=50.0, glob_rad=20.0)

        humid_contract = intent_contract_from_setpoint_plan(cold_humid, _plan(target_temp=17.0, target_rh=72.0))
        dry_contract = intent_contract_from_setpoint_plan(cold_dry, _plan(target_temp=17.0, target_rh=72.0))

        self.assertEqual(humid_contract.regime, "cold_humid_recovery")
        self.assertEqual(humid_contract.profile_shape["rh"], "strict_dehumidify_then_relax")
        self.assertEqual(dry_contract.regime, "cold_dry_protect")
        self.assertEqual(dry_contract.profile_shape["rh"], "cold_dry_protect")

    def test_radiation_spike_and_wind_limited_safety_are_covered(self):
        spike = DummyState(
            temp_air=28.8,
            rh_air=76.0,
            glob_rad=500.0,
            forecast_rad_peak_2h=720.0,
            temp_air_delta_1h=0.8,
        )
        wind = DummyState(temp_air=31.0, rh_air=78.0, glob_rad=500.0, wind_speed=6.5)

        spike_contract = intent_contract_from_setpoint_plan(spike, _plan(target_temp=27.5, target_rh=76.0))
        wind_contract = intent_contract_from_setpoint_plan(wind, _plan(target_temp=25.0, target_rh=74.0))

        self.assertEqual(spike_contract.regime, "radiation_spike_relief")
        self.assertEqual(spike_contract.profile_shape["temp"], "shade_cooling")
        self.assertEqual(wind_contract.regime, "wind_limited_safety")
        self.assertEqual(wind_contract.constraints["wind_vent_soft_cap"], 0.55)

    def test_growth_support_regimes_cover_co2_and_lighting(self):
        co2_state = DummyState(temp_air=20.0, rh_air=70.0, co2_air=430.0, glob_rad=250.0)
        low_light_state = DummyState(
            temp_air=16.5,
            rh_air=72.0,
            glob_rad=40.0,
            forecast_rad_mean_1h=40.0,
            forecast_rad_peak_2h=40.0,
        )

        co2_contract = intent_contract_from_setpoint_plan(co2_state, _plan(target_temp=20.0, target_co2=650.0))
        light_contract = intent_contract_from_setpoint_plan(
            low_light_state,
            _plan(target_temp=16.5, target_co2=430.0, target_rh=76.0),
        )

        self.assertEqual(co2_contract.regime, "co2_day_boost")
        self.assertEqual(co2_contract.profile_shape["co2"], "co2_day_boost")
        self.assertEqual(light_contract.regime, "lighting_assist")
        self.assertEqual(light_contract.profile_shape["temp"], "low_light_hold")

    def test_missing_intent_falls_back_to_setpoint_adapter(self):
        state = DummyState(temp_air=14.0, hour_of_day=22.0)

        contract, diagnostics = coerce_intent_contract(None, state, _plan(target_temp=17.0))

        self.assertTrue(diagnostics["fallback"])
        self.assertEqual(diagnostics["source"], "setpoint_adapter")
        self.assertEqual(contract.regime, "night_heat_hold")

    def test_invalid_intent_is_clipped_and_uses_safe_fallbacks(self):
        state = DummyState(rh_air=91.0, dew_margin_air=0.6)
        payload = {
            "regime": "unknown",
            "target_range": {"temp": [40.0, 5.0], "co2": ["bad", 1000.0]},
            "priority": [],
            "constraints": {"rh_hard_max": "bad", "temp_max": 31.0},
            "profile_shape": "bad",
            "confidence": 2.0,
        }

        contract, diagnostics = coerce_intent_contract(payload, state, _plan(target_rh=72.0))
        data = contract.to_dict()

        self.assertTrue(diagnostics["fallback"])
        self.assertIn("regime:fallback", diagnostics["corrected"])
        self.assertEqual(data["regime"], "high_humidity_recovery")
        self.assertEqual(data["target_range"]["temp"], [11.5, 24.0])
        self.assertEqual(data["target_range"]["rh"], [69.0, 75.0])
        self.assertEqual(data["confidence"], 1.0)
        self.assertEqual(data["constraints"]["temp_max"], 31.0)
        self.assertNotIn("bad", data["constraints"])


if __name__ == "__main__":
    unittest.main()
