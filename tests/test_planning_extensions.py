import unittest
from dataclasses import dataclass

import numpy as np

try:
    from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector
    from gl_gym.agent.tools import ControlAction

    IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional agent deps may be absent
    IMPORT_ERROR = exc


@dataclass
class DummyState:
    timestep: int = 0
    hour_of_day: float = 4.0
    temp_air: float = 16.0
    rh_air: float = 96.0
    co2_air: float = 420.0
    glob_rad: float = 0.0
    temp_out: float = 8.0
    rh_out: float = 88.0
    fruit_weight: float = 0.1
    dew_margin_air: float = 0.4
    canopy_dew_margin: float = 0.3


class DummyEnv:
    nu = 6

    def __init__(self):
        self.u = np.zeros(6, dtype=np.float32)


class DummyInterface:
    def __init__(self):
        self.env = DummyEnv()


@unittest.skipIf(IMPORT_ERROR is not None, f"agent imports unavailable: {IMPORT_ERROR}")
class TestPlanningExtensions(unittest.TestCase):
    def _director(self):
        director = object.__new__(RuleBasedLLMDirector)
        director.config = AgentConfig()
        director.rules = RuleBasedLLMDirector._init_rules(director)
        director.interface = DummyInterface()
        director.rule_controller = None
        director.last_control = None
        director.last_fallback_selection = {}
        director.rh_violation_debt = 0.0
        return director

    def test_explicit_action_set_allows_zero_anchor(self):
        action = ControlAction()
        self.assertTrue(action.is_empty())
        action.action_set = True
        self.assertFalse(action.is_empty())
        self.assertTrue(np.allclose(action.to_array(), np.zeros(6)))

    def test_target_profile_extends_and_clips(self):
        director = self._director()
        profiles, contract = director._build_target_profiles(
            {
                "target_temp_profile": [14.0, 25.0],
                "target_rh_profile": [60.0, 70.0],
            },
            target_temp=14.0,
            target_co2=430.0,
            target_rh=76.0,
            horizon=4,
        )

        self.assertEqual(len(profiles["target_temp"]), 4)
        self.assertEqual(profiles["target_temp"], [14.0, 24.0, 24.0, 24.0])
        self.assertEqual(profiles["target_co2"], [430.0, 430.0, 430.0, 430.0])
        self.assertEqual(profiles["target_rh"], [62.0, 70.0, 70.0, 70.0])
        self.assertEqual(contract["target_temp_profile"]["source"], "llm_profile")
        self.assertEqual(contract["target_co2_profile"]["source"], "contract_constant")

    def test_high_rh_fallback_prefers_dehumidification(self):
        director = self._director()
        control = director._select_fallback_control(state=DummyState(), analysis={"is_critical": True})

        self.assertEqual(director.last_fallback_selection["source"], "emergency_dehumidify")
        self.assertGreaterEqual(float(control[3]), 0.5)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_setpoint_contract_tightens_rh_target_under_dew_risk(self):
        director = self._director()
        _, _, target_rh, _, corrected = director._enforce_setpoint_contract(
            DummyState(rh_air=91.0, dew_margin_air=0.4, canopy_dew_margin=0.3),
            np.zeros(6, dtype=np.float32),
            target_temp=18.0,
            target_co2=600.0,
            target_rh=88.0,
        )

        self.assertLessEqual(target_rh, director.config.rh_target_extreme_cap)
        self.assertIn("target_rh:risk_cap", corrected)

    def test_rh_violation_debt_accumulates_high_humidity_risk(self):
        director = self._director()
        debt = director._update_rh_violation_debt(DummyState(rh_air=96.0))

        self.assertGreater(debt, 0.0)
        self.assertEqual(director.rh_violation_debt, debt)


if __name__ == "__main__":
    unittest.main()
