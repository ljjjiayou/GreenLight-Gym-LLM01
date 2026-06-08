from types import SimpleNamespace
import unittest

import numpy as np

from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector


def _director(**config_overrides):
    director = object.__new__(RuleBasedLLMDirector)
    director.config = AgentConfig(**config_overrides)
    director.last_control = None
    return director


def _state(**overrides):
    data = {
        "timestep": 0,
        "temp_air": 29.0,
        "rh_air": 50.0,
        "co2_air": 430.0,
        "hour_of_day": 12.0,
        "glob_rad": 600.0,
        "forecast_rad_mean_1h": 600.0,
        "forecast_rad_peak_2h": 620.0,
        "temp_air_delta_1h": 0.0,
        "temp_violation": 0.0,
        "dew_margin_air": 4.0,
        "canopy_dew_margin": 4.0,
        "fruit_weight": 0.0,
        "u_boil": 0.0,
        "u_co2": 0.0,
        "u_th_scr": 0.2,
        "u_vent": 0.7,
        "u_lamp": 0.0,
        "u_bl_scr": 0.0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _candidate(name, *, temp, rh, co2=430.0):
    return {
        "name": name,
        "regime": "hot_dry_relief",
        "target_profile": {
            "target_temp": [float(temp)] * 12,
            "target_co2": [float(co2)] * 12,
            "target_rh": [float(rh)] * 12,
        },
        "priority": ["safety", "humidity", "temperature"],
        "constraints": {},
        "reason": f"{name} test candidate",
    }


def _plan():
    return {
        "created_timestep": 0,
        "target_temp": 27.0,
        "target_co2": 430.0,
        "target_rh": 70.0,
        "target_profile": {
            "target_temp": [27.0] * 12,
            "target_co2": [430.0] * 12,
            "target_rh": [70.0] * 12,
        },
        "intent_contract": {
            "regime": "hot_dry_relief",
            "target_range": {"temp": [25.0, 29.0], "co2": [400.0, 500.0], "rh": [70.0, 82.0]},
            "priority": ["safety", "humidity", "temperature"],
            "constraints": {},
            "profile_shape": {"temp": "shade_cooling", "rh": "hot_dry_protect"},
            "confidence": 0.8,
        },
        "profile_generator_diagnostics": {"horizon_steps": 12},
        "profile_candidates": [
            _candidate("constant_hold", temp=27.0, rh=70.0),
            _candidate("hot_dry_protect", temp=28.0, rh=82.0),
            _candidate("shade_cooling", temp=24.0, rh=68.0),
        ],
    }


class TestProfileRspcShadow(unittest.TestCase):
    def test_shadow_helper_does_not_mutate_control_or_plan(self):
        director = _director(tomato_safety_v2_enabled=True)
        plan = _plan()
        original_plan_candidates = [dict(item) for item in plan["profile_candidates"]]
        baseline = np.array([0.0, 0.0, 0.2, 0.75, 0.0, 0.0], dtype=np.float32)
        baseline_before = baseline.copy()

        shadow = director._evaluate_profile_rspc_shadow(_state(), plan, baseline)

        self.assertTrue(shadow["enabled"])
        np.testing.assert_allclose(baseline, baseline_before)
        self.assertEqual(plan["profile_candidates"], original_plan_candidates)
        self.assertEqual(shadow["candidate_count"], 3)
        self.assertIn("top_candidates", shadow)

    def test_safe_hot_dry_can_give_hot_dry_profile_positive_margin(self):
        director = _director(tomato_safety_v2_enabled=True)
        baseline = np.array([0.0, 0.0, 0.2, 0.85, 0.0, 0.0], dtype=np.float32)

        shadow = director._evaluate_profile_rspc_shadow(_state(), _plan(), baseline)

        self.assertEqual(shadow["safety_gate_reason"], "none")
        self.assertEqual(shadow["best_profile"], "hot_dry_protect")
        self.assertGreater(shadow["margin"], 0.0)
        self.assertTrue(shadow["would_improve"])
        self.assertEqual(shadow["safety_alignment"], "dry_benefit")

    def test_hot_dry_profile_is_not_best_under_temp_gate(self):
        director = _director(tomato_safety_v2_enabled=True)
        state = _state(temp_air=33.0, rh_air=50.0, glob_rad=700.0, forecast_rad_peak_2h=720.0)
        baseline = np.array([0.0, 0.0, 0.2, 0.45, 0.0, 0.2], dtype=np.float32)

        shadow = director._evaluate_profile_rspc_shadow(state, _plan(), baseline)

        self.assertEqual(shadow["safety_gate_reason"], "temp_high_gate")
        self.assertNotEqual(shadow["best_profile"], "hot_dry_protect")
        hot_dry = next(item for item in shadow["top_candidates"] if item["name"] == "hot_dry_protect")
        self.assertFalse(hot_dry["eligible"])
        self.assertEqual(hot_dry["ineligible_reason"], "temp_high_gate:profile_forbidden")

    def test_hot_dry_profile_is_not_best_under_canopy_gate(self):
        director = _director(tomato_safety_v2_enabled=True)
        state = _state(rh_air=64.0, canopy_dew_margin=0.6, dew_margin_air=2.5)
        baseline = np.array([0.0, 0.0, 0.4, 0.35, 0.0, 0.2], dtype=np.float32)

        shadow = director._evaluate_profile_rspc_shadow(state, _plan(), baseline)

        self.assertEqual(shadow["safety_gate_reason"], "canopy_gate")
        self.assertNotEqual(shadow["best_profile"], "hot_dry_protect")

    def test_rh_high_dehumidify_positive_margin_is_safe_relief(self):
        director = _director(tomato_safety_v2_enabled=True)
        plan = _plan()
        plan["profile_candidates"] = [
            _candidate("hot_dry_protect", temp=28.0, rh=82.0),
            _candidate("strict_dehumidify_then_relax", temp=18.0, rh=62.0),
        ]
        state = _state(temp_air=17.0, rh_air=92.0, dew_margin_air=1.4, canopy_dew_margin=4.0)
        baseline = np.array([0.0, 0.0, 0.8, 0.05, 0.0, 0.0], dtype=np.float32)

        shadow = director._evaluate_profile_rspc_shadow(state, plan, baseline)

        self.assertEqual(shadow["safety_gate_reason"], "rh_high_gate")
        self.assertEqual(shadow["best_profile"], "strict_dehumidify_then_relax")
        self.assertGreater(shadow["margin"], 0.0)
        self.assertEqual(shadow["safety_alignment"], "safe_relief")


if __name__ == "__main__":
    unittest.main()
