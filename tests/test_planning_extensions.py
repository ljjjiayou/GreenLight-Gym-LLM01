import unittest
from dataclasses import dataclass

import numpy as np

try:
    from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector, apply_safety_guardrails, apply_tomato_safety_v2
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
    wind_speed: float = 0.0
    fruit_weight: float = 0.1
    dew_margin_air: float = 0.4
    canopy_dew_margin: float = 0.3
    forecast_rad_mean_1h: float = 0.0
    forecast_rad_peak_2h: float = 0.0
    forecast_temp_out_delta_1h: float = 0.0
    temp_air_delta_1h: float = 0.0
    temp_air_slope_c_per_step: float = 0.0
    u_th_scr: float = 0.0
    u_vent: float = 0.0
    u_bl_scr: float = 0.0


class DummyEnv:
    nu = 6
    nd = 10

    def __init__(self):
        self.u = np.zeros(6, dtype=np.float32)
        self.x = np.zeros(4, dtype=np.float32)
        self.timestep = 0
        self.weather_data = [np.zeros(self.nd, dtype=np.float32)]


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
        director.expert_policy = None
        director.humidity_memory = None
        director.last_control = None
        director.last_fallback_selection = {}
        director.last_expert_prediction = {}
        director.last_humidity_memory_prediction = {}
        director.last_humidity_memory_selected_step = -10**9
        director.last_llm_trigger_step = -10**9
        director.emergency_replan_cooldown_steps = max(6, director.config.control_interval // 2)
        director.current_plan = None
        director.last_rollout_selection = {}
        director.rh_emergency_streak_steps = 0
        director.rh_violation_debt = 0.0
        director.pending_control = None
        director.dehumidify_mode = "normal"
        director.dehumidify_mode_hold_steps = 0
        director.dehumidify_exit_confirm_counter = 0
        director._lamp_budget_day_marker = None
        director._lamp_budget_integral = 0.0
        director.last_lamp_budget_remaining = director.config.lamp_daily_budget
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

    def test_setpoint_contract_raises_dry_rh_target(self):
        director = self._director()
        target_temp, target_co2, target_rh, _, corrected = director._enforce_setpoint_contract(
            DummyState(temp_air=24.0, rh_air=45.0, glob_rad=260.0),
            np.array([0.2, 0.5, 0.0, 0.25, 0.3, 0.0], dtype=np.float32),
            target_temp=23.0,
            target_co2=700.0,
            target_rh=62.0,
        )

        self.assertGreaterEqual(target_rh, director.config.dry_target_rh_floor)
        self.assertLessEqual(target_temp, director.config.dry_temp_target_cap)
        self.assertEqual(target_co2, 430.0)
        self.assertIn("target_rh:dry_recovery_floor", corrected)
        self.assertIn("target_temp:dry_recovery_cap", corrected)
        self.assertIn("target_co2:dry_recovery_cap", corrected)

    def test_rh_violation_debt_accumulates_high_humidity_risk(self):
        director = self._director()
        debt = director._update_rh_violation_debt(DummyState(rh_air=96.0))

        self.assertGreater(debt, 0.0)
        self.assertEqual(director.rh_violation_debt, debt)

    def test_guardrails_apply_heat_vent_pulse_for_extreme_rh(self):
        control = apply_safety_guardrails(
            DummyState(temp_air=18.4, rh_air=95.2, hour_of_day=14.0),
            np.zeros(6, dtype=np.float32),
        )

        self.assertGreaterEqual(float(control[0]), 0.239)
        self.assertGreaterEqual(float(control[3]), 0.699)
        self.assertLessEqual(float(control[2]), 0.32)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_moderate_rh_low_temp_vpd_does_not_trigger_dehumid_pulse(self):
        control = apply_safety_guardrails(
            DummyState(
                temp_air=14.0,
                rh_air=80.0,
                hour_of_day=14.0,
                dew_margin_air=2.0,
                canopy_dew_margin=2.0,
            ),
            np.zeros(6, dtype=np.float32),
        )

        self.assertLess(float(control[0]), 0.05)
        self.assertLess(float(control[3]), 0.15)

    def test_guardrails_limit_dry_high_vpd_actions(self):
        control = apply_safety_guardrails(
            DummyState(
                temp_air=25.0,
                rh_air=42.0,
                glob_rad=320.0,
                hour_of_day=13.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            ),
            np.array([0.3, 0.5, 0.0, 0.8, 0.6, 0.0], dtype=np.float32),
        )

        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)
        self.assertLessEqual(float(control[3]), 0.180001)
        self.assertGreaterEqual(float(control[5]), 0.5)

    def test_guardrails_preempt_hot_dry_heat_load_before_32c(self):
        control = apply_safety_guardrails(
            DummyState(
                temp_air=30.8,
                rh_air=49.0,
                glob_rad=690.0,
                forecast_rad_peak_2h=820.0,
                temp_out=27.0,
                hour_of_day=12.0,
                dew_margin_air=4.0,
                canopy_dew_margin=4.0,
            ),
            np.array([0.2, 0.5, 1.0, 0.20, 0.6, 0.0], dtype=np.float32),
        )

        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)
        self.assertGreaterEqual(float(control[3]), 0.719)
        self.assertLessEqual(float(control[3]), 0.850001)
        self.assertGreaterEqual(float(control[2]), 0.549)
        self.assertLessEqual(float(control[2]), 0.720001)
        self.assertGreaterEqual(float(control[5]), 0.819)

    def test_guardrails_hot_dry_numerical_band_caps_non_extreme_vent(self):
        control = apply_safety_guardrails(
            DummyState(
                temp_air=34.2,
                rh_air=45.0,
                glob_rad=805.0,
                forecast_rad_peak_2h=850.0,
                temp_out=29.8,
                hour_of_day=12.0,
                dew_margin_air=5.0,
                canopy_dew_margin=5.0,
            ),
            np.array([0.2, 0.5, 1.0, 0.20, 0.6, 0.0], dtype=np.float32),
        )

        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)
        self.assertGreaterEqual(float(control[2]), 0.349)
        self.assertLessEqual(float(control[2]), 0.350001)
        self.assertGreaterEqual(float(control[3]), 0.899)
        self.assertLess(float(control[3]), 0.95)
        self.assertGreaterEqual(float(control[5]), 0.849)
        self.assertLessEqual(float(control[5]), 0.850001)

    def test_guardrails_extreme_hot_dry_solver_band_reduces_shade(self):
        control = apply_safety_guardrails(
            DummyState(
                temp_air=34.4,
                rh_air=43.0,
                glob_rad=825.0,
                forecast_rad_peak_2h=850.0,
                temp_out=30.2,
                hour_of_day=12.0,
                dew_margin_air=5.0,
                canopy_dew_margin=5.0,
            ),
            np.array([0.2, 0.5, 0.05, 0.20, 0.6, 0.85], dtype=np.float32),
        )

        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)
        self.assertGreaterEqual(float(control[2]), 0.349)
        self.assertLessEqual(float(control[2]), 0.350001)
        self.assertGreaterEqual(float(control[3]), 0.949)
        self.assertGreaterEqual(float(control[5]), 0.749)
        self.assertLessEqual(float(control[5]), 0.750001)

    def test_guardrails_raw_extreme_hot_dry_uses_stability_fallback(self):
        control = apply_safety_guardrails(
            DummyState(
                temp_air=34.1,
                rh_air=38.5,
                glob_rad=847.0,
                forecast_rad_peak_2h=850.0,
                temp_out=30.5,
                wind_speed=6.2,
                hour_of_day=12.0,
                dew_margin_air=5.0,
                canopy_dew_margin=5.0,
            ),
            np.array([0.2, 0.5, 0.35, 0.20, 0.6, 0.85], dtype=np.float32),
        )

        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)
        self.assertGreaterEqual(float(control[2]), 0.899)
        self.assertLessEqual(float(control[2]), 1.000001)
        self.assertGreaterEqual(float(control[3]), 0.949)
        self.assertGreaterEqual(float(control[5]), 0.499)
        self.assertLessEqual(float(control[5]), 0.500001)

    def test_guardrails_raw_extreme_hot_dry_holds_above_35c(self):
        control = apply_safety_guardrails(
            DummyState(
                temp_air=35.1,
                rh_air=42.3,
                glob_rad=842.0,
                forecast_rad_peak_2h=847.0,
                temp_out=30.7,
                wind_speed=6.3,
                hour_of_day=12.0,
                dew_margin_air=5.0,
                canopy_dew_margin=5.0,
            ),
            np.array([0.2, 0.5, 0.05, 0.20, 0.6, 0.85], dtype=np.float32),
        )

        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)
        self.assertGreaterEqual(float(control[2]), 0.899)
        self.assertGreaterEqual(float(control[3]), 0.949)
        self.assertGreaterEqual(float(control[5]), 0.499)
        self.assertLessEqual(float(control[5]), 0.500001)

    def test_guardrails_extreme_dew_hot_state_keeps_full_vent(self):
        control = apply_safety_guardrails(
            DummyState(
                temp_air=34.2,
                rh_air=95.0,
                glob_rad=805.0,
                forecast_rad_peak_2h=850.0,
                temp_out=29.8,
                hour_of_day=12.0,
                dew_margin_air=0.2,
                canopy_dew_margin=0.2,
            ),
            np.array([0.2, 0.5, 1.0, 0.20, 0.6, 0.0], dtype=np.float32),
        )

        self.assertGreaterEqual(float(control[3]), 0.949)
        self.assertLessEqual(float(control[2]), 0.350001)

    def test_tomato_safety_v2_disabled_is_noop(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=False)
        original = np.array([0.3, 0.5, 0.0, 0.8, 0.6, 0.0], dtype=np.float32)

        control, info = apply_tomato_safety_v2(
            DummyState(temp_air=25.0, rh_air=42.0, glob_rad=320.0, hour_of_day=13.0),
            original,
            config=cfg,
            target_rh=72.0,
        )

        self.assertTrue(np.allclose(control, original))
        self.assertFalse(info["enabled"])
        self.assertFalse(info["applied"])

    def test_tomato_safety_v2_caps_dry_high_vpd_ventilation(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(temp_air=25.0, rh_air=42.0, glob_rad=320.0, hour_of_day=13.0),
            np.array([0.3, 0.5, 0.0, 0.8, 0.6, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=72.0,
        )

        self.assertTrue(info["enabled"])
        self.assertTrue(info["applied"])
        self.assertIn("dry_vpd_guard", info["reasons"])
        self.assertIn("hard_dry_vpd_guard", info["reasons"])
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)
        self.assertLessEqual(float(control[3]), 0.140001)
        self.assertGreaterEqual(float(control[5]), 0.649)

    def test_tomato_safety_v2_warm_hard_dry_buffers_screen_below_hot_dry(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=23.5,
                rh_air=38.0,
                glob_rad=780.0,
                hour_of_day=12.0,
                forecast_rad_peak_2h=840.0,
            ),
            np.array([0.0, 0.4, 0.0, 0.5, 0.4, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=72.0,
        )

        self.assertIn("dry_vpd_guard", info["reasons"])
        self.assertIn("hard_dry_vpd_guard", info["reasons"])
        self.assertNotIn("hot_dry_cooling_guard", info["reasons"])
        self.assertGreaterEqual(float(control[2]), 0.749)
        self.assertGreaterEqual(float(control[5]), 0.649)
        self.assertLessEqual(float(control[3]), 0.140001)

    def test_tomato_safety_v2_pre_hot_dry_uses_shade_first_vent_budget(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=30.0,
                rh_air=56.0,
                glob_rad=760.0,
                hour_of_day=12.0,
                dew_margin_air=2.0,
                canopy_dew_margin=2.0,
            ),
            np.array([0.2, 0.5, 0.0, 0.12, 0.6, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=72.0,
        )

        self.assertTrue(info["applied"])
        self.assertIn("hot_dry_cooling_guard", info["reasons"])
        self.assertIn("hot_dry_radiation_relief", info["reasons"])
        self.assertIn("pre_hot_dry_vent_budget", info["reasons"])
        self.assertIn("hard_dry_vpd_guard", info["reasons"])
        self.assertGreaterEqual(float(control[3]), 0.599)
        self.assertLessEqual(float(control[3]), 0.820001)
        self.assertGreaterEqual(float(control[2]), 0.649)
        self.assertLessEqual(float(control[2]), 0.850001)
        self.assertGreaterEqual(float(control[5]), 0.819)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_cooldown_recovery_reduces_vent_and_restores_screen(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=26.2,
                rh_air=38.0,
                glob_rad=760.0,
                hour_of_day=13.0,
                dew_margin_air=5.0,
                canopy_dew_margin=5.0,
            ),
            np.array([0.0, 0.2, 0.0, 0.18, 0.0, 0.50], dtype=np.float32),
            config=cfg,
            target_rh=72.0,
        )

        self.assertIn("hot_dry_cooling_guard", info["reasons"])
        self.assertIn("cooldown_dry_recovery", info["reasons"])
        self.assertNotIn("severe_dry_recovery", info["reasons"])
        self.assertGreaterEqual(float(control[2]), 0.819)
        self.assertLessEqual(float(control[3]), 0.280001)
        self.assertGreaterEqual(float(control[5]), 0.449)

    def test_tomato_safety_v2_limits_pre_hot_dry_vent_under_rising_heat_load(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=28.5,
                rh_air=52.0,
                glob_rad=585.0,
                forecast_rad_mean_1h=692.0,
                forecast_rad_peak_2h=825.0,
                forecast_temp_out_delta_1h=1.5,
                hour_of_day=8.0,
                dew_margin_air=5.0,
                canopy_dew_margin=5.0,
            ),
            np.array([0.0, 0.2, 1.0, 0.18, 0.0, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=72.0,
        )

        self.assertIn("hot_dry_cooling_guard", info["reasons"])
        self.assertIn("hot_dry_radiation_relief", info["reasons"])
        self.assertIn("pre_hot_dry_heat_relief", info["reasons"])
        self.assertIn("pre_hot_dry_vent_budget", info["reasons"])
        self.assertNotIn("severe_dry_recovery", info["reasons"])
        self.assertGreaterEqual(float(control[2]), 0.699)
        self.assertLessEqual(float(control[2]), 0.820001)
        self.assertGreaterEqual(float(control[3]), 0.599)
        self.assertLessEqual(float(control[3]), 0.820001)
        self.assertGreaterEqual(float(control[5]), 0.879)

    def test_tomato_safety_v2_hot_override_prioritizes_cooling_over_dry_screen(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=33.0,
                rh_air=45.0,
                glob_rad=780.0,
                temp_out=24.0,
                hour_of_day=12.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            ),
            np.array([0.2, 0.5, 1.0, 0.12, 0.6, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=72.0,
        )

        self.assertTrue(info["applied"])
        self.assertTrue(info["hot_override"])
        self.assertIn("hot_temperature_override", info["reasons"])
        self.assertIn("dry_vpd_heat_deprioritized", info["reasons"])
        self.assertIn("hot_dry_shade_limited_for_stability", info["reasons"])
        self.assertNotIn("hot_dry_cooling_guard", info["reasons"])
        self.assertLessEqual(float(control[2]), 0.350001)
        self.assertGreaterEqual(float(control[5]), 0.849)
        self.assertLessEqual(float(control[5]), 0.850001)
        self.assertGreaterEqual(float(control[3]), 0.899)
        self.assertLess(float(control[3]), 0.95)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_high_rad_rapid_rise_preempts_before_32c(self):
        cfg = AgentConfig(
            tomato_safety_v2_enabled=True,
            tomato_safety_v2_hot_rapid_rise_temp_threshold=29.0,
        )

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=29.4,
                rh_air=48.0,
                glob_rad=720.0,
                temp_out=26.0,
                hour_of_day=12.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
                temp_air_delta_1h=0.7,
            ),
            np.array([0.1, 0.3, 1.0, 0.12, 0.2, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertTrue(info["hot_override"])
        self.assertIn("high_rad_rapid_rise_override", info["reasons"])
        self.assertIn("hot_dry_shade_limited_for_stability", info["reasons"])
        self.assertGreaterEqual(float(control[2]), 0.349)
        self.assertLessEqual(float(control[2]), 0.600001)
        self.assertGreaterEqual(float(control[5]), 0.849)
        self.assertLessEqual(float(control[5]), 0.850001)
        self.assertGreaterEqual(float(control[3]), 0.849)
        self.assertLess(float(control[3]), 0.90)

    def test_tomato_safety_v2_non_extreme_hot_dry_delays_full_vent(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=34.2,
                rh_air=45.0,
                glob_rad=805.0,
                forecast_rad_peak_2h=850.0,
                temp_out=29.8,
                hour_of_day=12.0,
                dew_margin_air=5.0,
                canopy_dew_margin=5.0,
            ),
            np.array([0.0, 0.0, 0.2, 0.95, 0.0, 0.85], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertTrue(info["hot_override"])
        self.assertIn("hot_temperature_override", info["reasons"])
        self.assertIn("hot_dry_solver_screen_buffer", info["reasons"])
        self.assertNotIn("extreme_hot_dry_stability_guard", info["reasons"])
        self.assertGreaterEqual(float(control[3]), 0.899)
        self.assertLess(float(control[3]), 0.95)
        self.assertGreaterEqual(float(control[2]), 0.349)
        self.assertLessEqual(float(control[2]), 0.350001)
        self.assertGreaterEqual(float(control[5]), 0.849)
        self.assertLessEqual(float(control[5]), 0.850001)

    def test_tomato_safety_v2_extreme_solver_band_uses_mid_shade(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=34.4,
                rh_air=43.0,
                glob_rad=825.0,
                forecast_rad_peak_2h=850.0,
                temp_out=30.2,
                hour_of_day=12.0,
                dew_margin_air=5.0,
                canopy_dew_margin=5.0,
            ),
            np.array([0.0, 0.0, 0.05, 0.90, 0.0, 0.85], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertTrue(info["hot_override"])
        self.assertIn("hot_dry_extreme_solver_relief", info["reasons"])
        self.assertNotIn("extreme_hot_dry_stability_guard", info["reasons"])
        self.assertGreaterEqual(float(control[2]), 0.349)
        self.assertLessEqual(float(control[2]), 0.350001)
        self.assertGreaterEqual(float(control[3]), 0.949)
        self.assertGreaterEqual(float(control[5]), 0.749)
        self.assertLessEqual(float(control[5]), 0.750001)

    def test_tomato_safety_v2_forecast_rad_promotes_hot_dry_cooling(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=25.9,
                rh_air=55.4,
                glob_rad=477.0,
                forecast_rad_mean_1h=602.0,
                forecast_temp_out_delta_1h=1.8,
                hour_of_day=7.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            ),
            np.array([0.0, 0.0, 0.0, 0.08, 0.0, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertIn("hot_dry_cooling_guard", info["reasons"])
        self.assertIn("hot_dry_radiation_relief", info["reasons"])
        self.assertIn("pre_hot_dry_vent_budget", info["reasons"])
        self.assertNotIn("target_rh_action_mismatch", info["reasons"])
        self.assertGreaterEqual(float(control[2]), 0.699)
        self.assertLessEqual(float(control[2]), 0.820001)
        self.assertGreaterEqual(float(control[3]), 0.599)
        self.assertLessEqual(float(control[3]), 0.820001)
        self.assertGreaterEqual(float(control[5]), 0.879)

    def test_tomato_safety_v2_pre_hot_non_hard_dry_uses_shade_first_budget(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=28.6,
                rh_air=60.5,
                glob_rad=476.0,
                forecast_rad_mean_1h=550.0,
                forecast_rad_peak_2h=793.0,
                forecast_temp_out_delta_1h=0.7,
                hour_of_day=9.5,
                dew_margin_air=8.4,
                canopy_dew_margin=1.2,
            ),
            np.array([0.0, 0.0, 0.50, 0.35, 0.0, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=75.0,
        )

        self.assertIn("hot_dry_cooling_guard", info["reasons"])
        self.assertIn("hot_dry_radiation_relief", info["reasons"])
        self.assertIn("pre_hot_dry_heat_relief", info["reasons"])
        self.assertIn("pre_hot_dry_vent_budget", info["reasons"])
        self.assertIn("canopy_dew_hot_dry_conflict_buffer", info["reasons"])
        self.assertGreaterEqual(float(control[3]), 0.459)
        self.assertLessEqual(float(control[3]), 0.640001)
        self.assertLessEqual(float(control[2]), 0.600001)
        self.assertLessEqual(float(control[5]), 0.500001)

    def test_tomato_safety_v2_warm_severe_dry_avoids_screen_trap_under_heat_load(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=29.4,
                rh_air=53.0,
                glob_rad=330.0,
                forecast_rad_mean_1h=340.0,
                forecast_rad_peak_2h=345.0,
                hour_of_day=10.0,
                dew_margin_air=9.5,
                canopy_dew_margin=4.0,
            ),
            np.array([0.0, 0.2, 1.0, 0.18, 0.0, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=72.0,
        )

        self.assertIn("hot_dry_cooling_guard", info["reasons"])
        self.assertIn("hot_dry_radiation_relief", info["reasons"])
        self.assertIn("cooldown_dry_recovery", info["reasons"])
        self.assertNotIn("severe_dry_recovery", info["reasons"])
        self.assertGreaterEqual(float(control[2]), 0.819)
        self.assertLessEqual(float(control[2]), 0.850001)
        self.assertLessEqual(float(control[3]), 0.280001)
        self.assertGreaterEqual(float(control[5]), 0.449)

    def test_tomato_safety_v2_hot_override_hold_prevents_screen_sawtooth(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=30.2,
                rh_air=48.0,
                glob_rad=560.0,
                temp_out=25.0,
                hour_of_day=12.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
                u_th_scr=0.2,
                u_vent=0.95,
                u_bl_scr=0.85,
            ),
            np.array([0.1, 0.3, 1.0, 0.12, 0.2, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertTrue(info["hot_override"])
        self.assertIn("hot_override_hold", info["reasons"])
        self.assertIn("hot_dry_shade_limited_for_stability", info["reasons"])
        self.assertNotIn("hot_dry_cooling_guard", info["reasons"])
        self.assertGreaterEqual(float(control[2]), 0.349)
        self.assertLessEqual(float(control[2]), 0.600001)
        self.assertGreaterEqual(float(control[5]), 0.799)
        self.assertLessEqual(float(control[5]), 0.850001)
        self.assertGreaterEqual(float(control[3]), 0.849)
        self.assertLess(float(control[3]), 0.90)

    def test_tomato_safety_v2_extreme_hot_dry_adds_stability_buffer(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=34.5,
                rh_air=35.0,
                glob_rad=830.0,
                temp_out=30.0,
                hour_of_day=11.5,
                dew_margin_air=12.0,
                canopy_dew_margin=3.0,
            ),
            np.array([0.0, 0.3, 0.2, 0.95, 0.2, 0.65], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertTrue(info["hot_override"])
        self.assertIn("hot_temperature_override", info["reasons"])
        self.assertIn("hot_dry_shade_limited_for_stability", info["reasons"])
        self.assertIn("extreme_hot_dry_stability_guard", info["reasons"])
        self.assertGreaterEqual(float(control[2]), 0.899)
        self.assertGreaterEqual(float(control[3]), 0.949)
        self.assertGreaterEqual(float(control[5]), 0.499)
        self.assertLessEqual(float(control[5]), 0.500001)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_extreme_hot_dry_holds_after_humidity_rebound(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=38.2,
                rh_air=54.0,
                glob_rad=825.0,
                temp_out=31.5,
                hour_of_day=12.0,
                dew_margin_air=11.0,
                canopy_dew_margin=-2.5,
            ),
            np.array([0.0, 0.3, 0.2, 0.95, 0.2, 0.65], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertTrue(info["hot_override"])
        self.assertIn("extreme_hot_dry_stability_guard", info["reasons"])
        self.assertIn("extreme_hot_dry_temperature_hold", info["reasons"])
        self.assertGreaterEqual(float(control[2]), 0.899)
        self.assertGreaterEqual(float(control[3]), 0.949)
        self.assertGreaterEqual(float(control[5]), 0.499)
        self.assertLessEqual(float(control[5]), 0.500001)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_canopy_dew_relief_blocks_humid_hot_dry_screen(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=30.4,
                rh_air=70.0,
                glob_rad=530.0,
                temp_out=23.0,
                hour_of_day=12.0,
                dew_margin_air=6.0,
                canopy_dew_margin=-1.0,
            ),
            np.array([0.1, 0.3, 1.0, 0.12, 0.2, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertTrue(info["canopy_dew_relief"])
        self.assertIn("canopy_dew_relief_guard", info["reasons"])
        self.assertIn("dry_vpd_deprioritized_for_canopy_dew", info["reasons"])
        self.assertNotIn("hot_dry_cooling_guard", info["reasons"])
        self.assertLessEqual(float(control[2]), 0.350001)
        self.assertGreaterEqual(float(control[3]), 0.419)
        self.assertGreaterEqual(float(control[5]), 0.499)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_canopy_dew_temperate_buffer_avoids_overcooling(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=25.3,
                rh_air=66.5,
                glob_rad=628.0,
                forecast_rad_mean_1h=620.0,
                temp_out=22.0,
                hour_of_day=11.0,
                dew_margin_air=6.7,
                canopy_dew_margin=0.6,
            ),
            np.array([0.0, 0.0, 1.0, 0.20, 0.0, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertTrue(info["canopy_dew_relief"])
        self.assertTrue(info["canopy_dew_temperate_buffer"])
        self.assertIn("canopy_dew_relief_guard", info["reasons"])
        self.assertIn("canopy_dew_temperature_buffer", info["reasons"])
        self.assertLessEqual(float(control[2]), 0.750001)
        self.assertGreaterEqual(float(control[2]), 0.749)
        self.assertGreaterEqual(float(control[3]), 0.199)
        self.assertLessEqual(float(control[3]), 0.200001)
        self.assertGreaterEqual(float(control[5]), 0.499)
        self.assertLessEqual(float(control[5]), 0.500001)

    def test_tomato_safety_v2_target_rh_preempt_avoids_canopy_dew_conflict(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=23.4,
                rh_air=63.7,
                glob_rad=549.0,
                forecast_rad_mean_1h=560.0,
                temp_out=22.0,
                hour_of_day=10.75,
                dew_margin_air=7.2,
                canopy_dew_margin=1.8,
            ),
            np.array([0.0, 0.0, 1.0, 0.08, 0.0, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=75.0,
        )

        self.assertFalse(info["canopy_dew_relief"])
        self.assertTrue(info["canopy_dew_target_preempt"])
        self.assertIn("target_rh_action_mismatch", info["reasons"])
        self.assertIn("canopy_dew_target_preempt", info["reasons"])
        self.assertLessEqual(float(control[2]), 0.750001)
        self.assertGreaterEqual(float(control[3]), 0.179)
        self.assertLessEqual(float(control[3]), 0.240001)
        self.assertLessEqual(float(control[5]), 0.350001)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_canopy_dew_buffer_limits_hot_dry_cooldown_oscillation(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=28.9,
                rh_air=63.0,
                glob_rad=520.0,
                temp_out=23.0,
                hour_of_day=13.0,
                dew_margin_air=7.5,
                canopy_dew_margin=1.5,
            ),
            np.array([0.0, 0.0, 0.82, 0.18, 0.0, 0.50], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertTrue(info["canopy_dew_buffer"])
        self.assertIn("canopy_dew_hot_dry_conflict_buffer", info["reasons"])
        self.assertNotIn("canopy_dew_relief_guard", info["reasons"])
        self.assertLessEqual(float(control[2]), 0.600001)
        self.assertGreaterEqual(float(control[3]), 0.419)
        self.assertGreaterEqual(float(control[5]), 0.499)
        self.assertLessEqual(float(control[5]), 0.500001)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_cooldown_canopy_reserve_buffer_limits_strong_rad_screen(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=26.1,
                rh_air=64.5,
                glob_rad=705.0,
                forecast_rad_mean_1h=705.0,
                forecast_rad_peak_2h=705.0,
                temp_out=24.0,
                hour_of_day=12.0,
                dew_margin_air=8.5,
                canopy_dew_margin=2.5,
            ),
            np.array([0.0, 0.0, 0.82, 0.17, 0.0, 0.50], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertIn("hot_dry_cooling_guard", info["reasons"])
        self.assertIn("cooldown_dry_recovery", info["reasons"])
        self.assertIn("hot_dry_radiation_relief", info["reasons"])
        self.assertTrue(info["cooldown_canopy_reserve_buffer"])
        self.assertIn("cooldown_canopy_reserve_buffer", info["reasons"])
        self.assertLessEqual(float(control[2]), 0.600001)
        self.assertGreaterEqual(float(control[3]), 0.419)
        self.assertAlmostEqual(float(control[5]), 0.50, places=5)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_cooldown_canopy_reserve_does_not_override_hard_dry(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=26.1,
                rh_air=48.5,
                glob_rad=705.0,
                forecast_rad_mean_1h=705.0,
                forecast_rad_peak_2h=705.0,
                temp_out=24.0,
                hour_of_day=12.0,
                dew_margin_air=10.0,
                canopy_dew_margin=2.5,
            ),
            np.array([0.0, 0.0, 0.82, 0.17, 0.0, 0.50], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertIn("hot_dry_cooling_guard", info["reasons"])
        self.assertIn("cooldown_dry_recovery", info["reasons"])
        self.assertFalse(info["cooldown_canopy_reserve_buffer"])
        self.assertNotIn("cooldown_canopy_reserve_buffer", info["reasons"])
        self.assertGreaterEqual(float(control[2]), 0.819)
        self.assertLessEqual(float(control[3]), 0.280001)
        self.assertGreaterEqual(float(control[5]), 0.449)

    def test_tomato_safety_v2_canopy_dew_buffer_applies_to_pre_hot_dry_relief(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=28.8,
                rh_air=60.3,
                glob_rad=700.0,
                forecast_rad_mean_1h=705.0,
                temp_out=24.0,
                hour_of_day=12.25,
                dew_margin_air=8.4,
                canopy_dew_margin=1.3,
            ),
            np.array([0.0, 0.0, 0.82, 0.18, 0.0, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertIn("hot_dry_cooling_guard", info["reasons"])
        self.assertIn("pre_hot_dry_heat_relief", info["reasons"])
        self.assertTrue(info["canopy_dew_buffer"])
        self.assertIn("canopy_dew_hot_dry_conflict_buffer", info["reasons"])
        self.assertLessEqual(float(control[2]), 0.600001)
        self.assertGreaterEqual(float(control[3]), 0.599)
        self.assertLessEqual(float(control[5]), 0.500001)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_canopy_dew_buffer_covers_moderate_rh_radiation_relief(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=26.8,
                rh_air=57.5,
                glob_rad=660.0,
                forecast_rad_mean_1h=660.0,
                temp_out=24.0,
                hour_of_day=13.0,
                dew_margin_air=9.1,
                canopy_dew_margin=1.1,
            ),
            np.array([0.0, 0.0, 0.82, 0.18, 0.0, 0.5], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertTrue(info["canopy_dew_buffer"])
        self.assertIn("canopy_dew_hot_dry_conflict_buffer", info["reasons"])
        self.assertLessEqual(float(control[2]), 0.600001)
        self.assertGreaterEqual(float(control[3]), 0.419)
        self.assertLessEqual(float(control[5]), 0.500001)

    def test_tomato_safety_v2_canopy_dew_buffer_catches_hard_dry_rh_edge(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=27.9,
                rh_air=56.9,
                glob_rad=511.0,
                forecast_rad_mean_1h=511.0,
                temp_out=24.0,
                hour_of_day=13.0,
                dew_margin_air=9.3,
                canopy_dew_margin=1.9,
            ),
            np.array([0.0, 0.0, 0.82, 0.18, 0.0, 0.50], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertTrue(info["canopy_dew_buffer"])
        self.assertIn("canopy_dew_hot_dry_conflict_buffer", info["reasons"])
        self.assertLessEqual(float(control[2]), 0.600001)
        self.assertGreaterEqual(float(control[3]), 0.419)
        self.assertLessEqual(float(control[5]), 0.500001)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_severe_dry_canopy_reserve_avoids_full_screen_no_shade(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=26.9,
                rh_air=51.8,
                glob_rad=615.0,
                forecast_rad_mean_1h=615.0,
                forecast_rad_peak_2h=650.0,
                temp_out=24.0,
                hour_of_day=13.0,
                dew_margin_air=10.7,
                canopy_dew_margin=3.9,
                temp_air_delta_1h=0.8,
                u_vent=0.18,
            ),
            np.array([0.0, 0.0, 0.82, 0.18, 0.0, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertIn("hot_dry_cooling_guard", info["reasons"])
        self.assertIn("severe_dry_recovery", info["reasons"])
        self.assertIn("severe_dry_canopy_reserve", info["reasons"])
        self.assertLessEqual(float(control[2]), 0.820001)
        self.assertGreaterEqual(float(control[2]), 0.699)
        self.assertGreaterEqual(float(control[3]), 0.599)
        self.assertAlmostEqual(float(control[5]), 0.50, places=5)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_hot_dry_overrides_low_canopy_dew_when_air_is_dry(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=28.5,
                rh_air=60.0,
                glob_rad=800.0,
                hour_of_day=11.0,
                dew_margin_air=8.0,
                canopy_dew_margin=-0.2,
            ),
            np.array([0.0, 0.2, 0.0, 0.12, 0.1, 0.5], dtype=np.float32),
            config=cfg,
            target_rh=70.0,
        )

        self.assertIn("hot_dry_cooling_guard", info["reasons"])
        self.assertIn("hot_dry_radiation_relief", info["reasons"])
        self.assertIn("pre_hot_dry_vent_budget", info["reasons"])
        self.assertGreaterEqual(float(control[3]), 0.459)
        self.assertLessEqual(float(control[3]), 0.640001)
        self.assertGreaterEqual(float(control[2]), 0.699)
        self.assertLessEqual(float(control[2]), 0.820001)
        self.assertGreaterEqual(float(control[5]), 0.819)

    def test_tomato_safety_v2_buffers_low_temp_without_extreme_dew(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=11.5,
                rh_air=88.0,
                glob_rad=0.0,
                hour_of_day=3.0,
                dew_margin_air=1.2,
                canopy_dew_margin=1.1,
            ),
            np.array([0.0, 0.5, 0.2, 0.8, 0.6, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=75.0,
        )

        self.assertTrue(info["applied"])
        self.assertIn("cold_buffer_guard", info["reasons"])
        self.assertGreaterEqual(float(control[0]), 0.899)
        self.assertGreaterEqual(float(control[2]), 0.899)
        self.assertLessEqual(float(control[3]), 0.080001)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_does_not_block_extreme_dew_venting(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=14.0,
                rh_air=95.0,
                glob_rad=0.0,
                hour_of_day=3.0,
                dew_margin_air=0.3,
                canopy_dew_margin=0.3,
            ),
            np.array([0.2, 0.0, 0.2, 0.7, 0.0, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=72.0,
        )

        self.assertFalse(info["applied"])
        self.assertGreaterEqual(float(control[3]), 0.699)

    def test_tomato_safety_v2_hot_override_keeps_extreme_dew_venting_active(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=33.0,
                rh_air=95.0,
                glob_rad=780.0,
                temp_out=24.0,
                hour_of_day=12.0,
                dew_margin_air=0.2,
                canopy_dew_margin=0.2,
            ),
            np.array([0.2, 0.3, 1.0, 0.12, 0.4, 0.0], dtype=np.float32),
            config=cfg,
            target_rh=72.0,
        )

        self.assertTrue(info["hot_override"])
        self.assertIn("extreme_dew_hot_override", info["reasons"])
        self.assertLessEqual(float(control[2]), 0.350001)
        self.assertGreaterEqual(float(control[3]), 0.949)
        self.assertGreaterEqual(float(control[5]), 0.799)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_tomato_safety_v2_suppresses_strict_replay_emergency_replans_only(self):
        director = object.__new__(RuleBasedLLMDirector)
        director.config = AgentConfig(
            tomato_safety_v2_enabled=True,
            plan_cache_mode="replay",
            plan_cache_key_policy="scenario_timestep",
            plan_cache_strict=True,
        )

        self.assertTrue(director._should_suppress_strict_replay_emergency_replan())
        self.assertTrue(director._should_suppress_tomato_v2_replay_emergency_replan())

        director.config.plan_cache_key_policy = "prompt"
        self.assertFalse(director._should_suppress_strict_replay_emergency_replan())
        self.assertFalse(director._should_suppress_tomato_v2_replay_emergency_replan())

        director.config.plan_cache_key_policy = "scenario_timestep"
        director.config.tomato_safety_v2_enabled = False
        self.assertTrue(director._should_suppress_strict_replay_emergency_replan())
        self.assertFalse(director._should_suppress_tomato_v2_replay_emergency_replan())

    def test_cold_non_extreme_humidity_caps_ventilation(self):
        control = apply_safety_guardrails(
            DummyState(
                temp_air=11.5,
                rh_air=88.0,
                glob_rad=0.0,
                hour_of_day=3.0,
                dew_margin_air=1.2,
                canopy_dew_margin=1.1,
            ),
            np.array([0.0, 0.5, 0.2, 0.8, 0.6, 0.0], dtype=np.float32),
        )

        self.assertGreaterEqual(float(control[0]), 0.90)
        self.assertGreaterEqual(float(control[2]), 0.899)
        self.assertLessEqual(float(control[3]), 0.120001)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_rh_emergency_replan_respects_extended_cooldown(self):
        director = self._director()
        director.last_llm_trigger_step = 10
        director.rh_emergency_streak_steps = director.config.emergency_rh_confirm_steps - 1

        self.assertFalse(director._should_rh_emergency_replan(DummyState(timestep=17, rh_air=95.0)))
        self.assertTrue(director._should_rh_emergency_replan(DummyState(timestep=18, rh_air=95.0)))

    def test_high_rh_low_vpd_relaxes_wind_vent_cap(self):
        control = apply_safety_guardrails(
            DummyState(
                temp_air=19.0,
                rh_air=92.5,
                temp_out=10.0,
                wind_speed=7.0,
                hour_of_day=14.0,
            ),
            np.zeros(6, dtype=np.float32),
        )

        self.assertGreaterEqual(float(control[3]), 0.549)

    def test_rollout_candidate_sharing_avoids_bad_rule_action(self):
        class WastefulRule:
            def predict(self, *_args):
                return np.array([0.70, 0.50, 1.00, 0.70, 0.80, 0.0], dtype=np.float32)

        director = self._director()
        director.rule_controller = WastefulRule()
        director.current_plan = {
            "anchor_control": np.zeros(6, dtype=np.float32),
            "created_timestep": 0,
            "expires_timestep": 12,
            "target_temp": 18.0,
            "target_co2": 430.0,
            "target_rh": 70.0,
            "target_profile": {},
        }

        control, _rule, _anchor, rule_weight = director._plan_control_step(
            DummyState(
                temp_air=18.0,
                rh_air=70.0,
                co2_air=430.0,
                glob_rad=300.0,
                hour_of_day=12.0,
                dew_margin_air=2.0,
                canopy_dew_margin=2.0,
            )
        )

        self.assertEqual(director.last_rollout_selection["source"], "anchor")
        self.assertEqual(rule_weight, 0.0)
        self.assertLess(float(control[0]), 0.05)
        self.assertLess(float(control[3]), 0.15)
        self.assertEqual(float(control[4]), 0.0)

    def test_dry_fallback_prefers_recovery_candidate(self):
        director = self._director()
        director.interface.env.u = np.array([0.4, 0.5, 0.0, 0.8, 0.6, 0.0], dtype=np.float32)

        control = director._select_fallback_control(
            state=DummyState(
                temp_air=25.0,
                rh_air=43.0,
                co2_air=400.0,
                glob_rad=320.0,
                temp_out=18.0,
                rh_out=35.0,
                hour_of_day=12.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            )
        )

        self.assertEqual(director.last_fallback_selection["source"], "dry_recovery")
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)
        self.assertLessEqual(float(control[3]), director.config.dry_warm_vent_cap + 1e-6)
        self.assertGreaterEqual(float(control[5]), 0.5)

    def test_hot_dry_rspc_candidates_are_tomato_v2_opt_in(self):
        director = self._director()
        director.config.rspc_hot_dry_candidates_enabled = False
        director.config.tomato_safety_v2_enabled = True
        director.interface.env.u = np.array([0.0, 0.0, 1.0, 0.10, 0.0, 0.0], dtype=np.float32)

        candidates = director._build_fallback_candidates(
            DummyState(
                temp_air=31.0,
                rh_air=48.0,
                glob_rad=720.0,
                temp_out=25.0,
                rh_out=35.0,
                hour_of_day=12.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            )
        )

        self.assertFalse(any(item.name.startswith("hot_dry_") for item in candidates))

        director.config.rspc_hot_dry_candidates_enabled = True
        director.config.tomato_safety_v2_enabled = False
        candidates = director._build_fallback_candidates(
            DummyState(
                temp_air=31.0,
                rh_air=48.0,
                glob_rad=720.0,
                temp_out=25.0,
                rh_out=35.0,
                hour_of_day=12.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            )
        )

        self.assertFalse(any(item.name.startswith("hot_dry_") for item in candidates))

        director.config.tomato_safety_v2_enabled = True
        candidates = director._build_fallback_candidates(
            DummyState(
                temp_air=31.0,
                rh_air=48.0,
                glob_rad=720.0,
                temp_out=25.0,
                rh_out=35.0,
                hour_of_day=12.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            )
        )

        self.assertTrue(any(item.name.startswith("hot_dry_") for item in candidates))

    def test_fallback_score_breakdown_exposes_standard_terms(self):
        director = self._director()
        state = DummyState(
            temp_air=25.0,
            rh_air=43.0,
            glob_rad=320.0,
            temp_out=18.0,
            rh_out=35.0,
            hour_of_day=12.0,
            dew_margin_air=3.0,
            canopy_dew_margin=3.0,
        )
        score, details = director._score_fallback_candidate(
            state,
            np.array([0.2, 0.1, 0.4, 0.5, 0.2, 0.0], dtype=np.float32),
        )

        required = {
            "total_score",
            "temp_penalty",
            "rh_penalty",
            "vpd_high_penalty",
            "vpd_low_penalty",
            "dew_penalty",
            "canopy_dew_penalty",
            "energy_penalty",
            "smooth_penalty",
            "heat_vent_conflict_penalty",
            "co2_leak_penalty",
            "dry_vent_penalty",
            "forecast_risk_penalty",
        }
        self.assertTrue(required.issubset(details))
        self.assertAlmostEqual(float(score), float(details["total_score"]))
        subset = director._score_term_subset(details)
        self.assertTrue(required.issubset(subset))

    def test_fallback_selection_records_candidate_score_breakdown(self):
        director = self._director()
        director.interface.env.u = np.array([0.0, 0.0, 0.2, 0.6, 0.0, 0.0], dtype=np.float32)
        director._select_fallback_control(
            state=DummyState(
                temp_air=25.0,
                rh_air=43.0,
                glob_rad=320.0,
                temp_out=18.0,
                rh_out=35.0,
                hour_of_day=12.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            )
        )

        selection = director.last_fallback_selection
        self.assertEqual(selection["selected_fallback_candidate"], selection["source"])
        self.assertGreater(len(selection["fallback_candidate_scores"]), 0)
        self.assertIn("total_score", selection["selected_fallback_score_breakdown"])
        self.assertIn("action", selection["fallback_candidate_scores"][0])
        self.assertIn("score_breakdown", selection["fallback_candidate_scores"][0])

    def test_hot_dry_rspc_candidates_prefer_humidity_preserving_cooling(self):
        director = self._director()
        director.config.tomato_safety_v2_enabled = True
        director.config.rspc_hot_dry_candidates_enabled = True
        director.interface.env.u = np.array([0.0, 0.0, 1.0, 0.10, 0.0, 0.0], dtype=np.float32)

        candidates = director._build_fallback_candidates(
            DummyState(
                temp_air=31.0,
                rh_air=48.0,
                glob_rad=720.0,
                temp_out=25.0,
                rh_out=35.0,
                hour_of_day=12.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            )
        )

        self.assertTrue(candidates[0].name.startswith("hot_dry_"))
        self.assertLessEqual(float(candidates[0].control[2]), 0.850001)
        self.assertGreaterEqual(float(candidates[0].control[3]), 0.50)
        self.assertGreaterEqual(float(candidates[0].control[5]), 0.50)
        self.assertEqual(float(candidates[0].control[0]), 0.0)
        self.assertEqual(float(candidates[0].control[1]), 0.0)
        self.assertEqual(float(candidates[0].control[4]), 0.0)
        self.assertLess(candidates[0].details["hot_dry_penalty"], candidates[1].details["hot_dry_penalty"])

    def test_hot_dry_rspc_rollout_keeps_vent_relief_after_dry_guard(self):
        director = self._director()
        director.config.tomato_safety_v2_enabled = True
        director.config.rspc_hot_dry_candidates_enabled = True
        director.interface.env.u = np.array([0.0, 0.0, 1.0, 0.10, 0.0, 0.0], dtype=np.float32)
        director.current_plan = {
            "anchor_control": np.array([0.0, 0.0, 1.0, 0.10, 0.0, 0.0], dtype=np.float32),
            "created_timestep": 0,
            "expires_timestep": 12,
            "target_temp": 28.0,
            "target_co2": 430.0,
            "target_rh": 70.0,
            "target_profile": {},
        }

        control, _rule, _anchor, _rule_weight = director._plan_control_step(
            DummyState(
                temp_air=31.0,
                rh_air=48.0,
                co2_air=430.0,
                glob_rad=720.0,
                temp_out=25.0,
                rh_out=35.0,
                hour_of_day=12.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            )
        )

        self.assertTrue(director.last_rollout_selection["source"].startswith("hot_dry_"))
        self.assertTrue(director.last_rollout_selection["dry_recovery_override"]["hot_dry_vent_relief"])
        self.assertGreaterEqual(float(control[3]), 0.50)
        self.assertLessEqual(float(control[2]), 0.850001)
        self.assertGreaterEqual(float(control[5]), 0.50)
        self.assertEqual(float(control[0]), 0.0)
        self.assertEqual(float(control[1]), 0.0)
        self.assertEqual(float(control[4]), 0.0)

    def test_hot_dry_rspc_candidates_do_not_bypass_extreme_dew(self):
        director = self._director()
        director.config.tomato_safety_v2_enabled = True
        director.config.rspc_hot_dry_candidates_enabled = True
        director.interface.env.u = np.array([0.0, 0.0, 1.0, 0.10, 0.0, 0.0], dtype=np.float32)

        candidates = director._build_fallback_candidates(
            DummyState(
                temp_air=31.0,
                rh_air=95.0,
                glob_rad=720.0,
                temp_out=25.0,
                rh_out=80.0,
                hour_of_day=12.0,
                dew_margin_air=0.2,
                canopy_dew_margin=0.2,
            )
        )

        self.assertFalse(any(item.name.startswith("hot_dry_") for item in candidates))
        self.assertIn("emergency_dehumidify", {item.name for item in candidates})
        self.assertGreaterEqual(float(candidates[0].control[3]), 0.65)
        self.assertGreater(candidates[0].details["rh_penalty"], 0.0)
        self.assertGreater(candidates[0].details["dew_penalty"], 0.0)

    def test_humidity_memory_can_be_selected_as_rollout_candidate(self):
        class WastefulRule:
            def predict(self, *_args):
                return np.array([0.35, 0.0, 0.80, 0.65, 0.0, 0.0], dtype=np.float32)

        class Memory:
            def retrieve(self, *_args, **_kwargs):
                return [
                    {
                        "candidate_action": [0.0, 0.0, 0.05, 1.0, 0.0, 0.0],
                        "case_id": "case-a",
                        "distance": 0.2,
                        "trust": 0.8,
                        "support_count": 12,
                        "status": "pending",
                        "score": 0.76,
                    }
                ]

        director = self._director()
        director.config.humidity_memory_enabled = True
        director.config.humidity_memory_teacher_policy_id = "ppo_a"
        director.config.humidity_memory_baseline_controller_id = "rspc_a"
        director.rule_controller = WastefulRule()
        director.humidity_memory = Memory()
        director.current_plan = {
            "anchor_control": np.array([0.35, 0.0, 0.80, 0.65, 0.0, 0.0], dtype=np.float32),
            "created_timestep": 0,
            "expires_timestep": 12,
            "target_temp": 18.0,
            "target_co2": 430.0,
            "target_rh": 75.0,
            "target_profile": {},
        }

        control, _rule, _anchor, _rule_weight = director._plan_control_step(
            DummyState(
                temp_air=18.5,
                rh_air=82.0,
                co2_air=430.0,
                glob_rad=120.0,
                temp_out=15.0,
                rh_out=70.0,
                hour_of_day=10.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            )
        )

        self.assertEqual(director.last_rollout_selection["source"], "humidity_memory")
        self.assertTrue(director.last_rollout_selection["humidity_memory_prediction"]["accepted"])
        self.assertEqual(director.last_rollout_selection["humidity_memory_prediction"]["case_id"], "case-a")
        self.assertLess(float(control[0]), 0.05)
        self.assertGreater(float(control[3]), 0.95)

    def test_humidity_memory_is_rejected_in_high_risk_night_window(self):
        class WastefulRule:
            def predict(self, *_args):
                return np.array([0.30, 0.0, 0.80, 0.55, 0.0, 0.0], dtype=np.float32)

        class Memory:
            def retrieve(self, *_args, **_kwargs):
                return [
                    {
                        "candidate_action": [0.0, 0.0, 0.05, 1.0, 0.0, 0.0],
                        "case_id": "night-risk",
                        "distance": 0.2,
                        "trust": 0.8,
                        "support_count": 12,
                        "status": "pending",
                        "score": 0.76,
                    }
                ]

        director = self._director()
        director.config.humidity_memory_enabled = True
        director.rule_controller = WastefulRule()
        director.humidity_memory = Memory()
        director.current_plan = {
            "anchor_control": np.array([0.30, 0.0, 0.80, 0.55, 0.0, 0.0], dtype=np.float32),
            "created_timestep": 0,
            "expires_timestep": 12,
            "target_temp": 18.0,
            "target_co2": 430.0,
            "target_rh": 75.0,
            "target_profile": {},
        }

        director._plan_control_step(
            DummyState(
                timestep=20,
                temp_air=17.0,
                rh_air=82.0,
                co2_air=430.0,
                glob_rad=0.0,
                temp_out=11.0,
                rh_out=65.0,
                hour_of_day=23.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            )
        )

        self.assertFalse(director.last_rollout_selection["source"].startswith("humidity_memory"))
        hem_info = director.last_rollout_selection["humidity_memory_prediction"]
        self.assertEqual(hem_info["reject_reason"], "night_temp_buffer")
        self.assertTrue(hem_info["horizon_filter_rejected"])

    def test_humidity_memory_night_gap_blocks_repeated_pulses(self):
        class WastefulRule:
            def predict(self, *_args):
                return np.array([0.30, 0.0, 0.80, 0.55, 0.0, 0.0], dtype=np.float32)

        class Memory:
            def retrieve(self, *_args, **_kwargs):
                return [
                    {
                        "candidate_action": [0.0, 0.0, 0.05, 1.0, 0.0, 0.0],
                        "case_id": "night-gap",
                        "distance": 0.2,
                        "trust": 0.8,
                        "support_count": 12,
                        "status": "pending",
                        "score": 0.76,
                    }
                ]

        director = self._director()
        director.config.humidity_memory_enabled = True
        director.rule_controller = WastefulRule()
        director.humidity_memory = Memory()
        director.last_humidity_memory_selected_step = 10
        director.current_plan = {
            "anchor_control": np.array([0.30, 0.0, 0.80, 0.55, 0.0, 0.0], dtype=np.float32),
            "created_timestep": 0,
            "expires_timestep": 12,
            "target_temp": 18.0,
            "target_co2": 430.0,
            "target_rh": 75.0,
            "target_profile": {},
        }

        director._plan_control_step(
            DummyState(
                timestep=14,
                temp_air=18.1,
                rh_air=82.0,
                co2_air=430.0,
                glob_rad=0.0,
                temp_out=12.0,
                rh_out=65.0,
                hour_of_day=23.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            )
        )

        self.assertFalse(director.last_rollout_selection["source"].startswith("humidity_memory"))
        self.assertEqual(director.last_rollout_selection["humidity_memory_prediction"]["reject_reason"], "night_gap")

    def test_humidity_memory_disabled_does_not_query_memory(self):
        class FailingMemory:
            def retrieve(self, *_args, **_kwargs):
                raise AssertionError("disabled HEM should not be queried")

        director = self._director()
        director.config.humidity_memory_enabled = False
        director.humidity_memory = FailingMemory()
        director.current_plan = {
            "anchor_control": np.zeros(6, dtype=np.float32),
            "created_timestep": 0,
            "expires_timestep": 12,
            "target_temp": 18.0,
            "target_co2": 430.0,
            "target_rh": 75.0,
            "target_profile": {},
        }

        director._plan_control_step(
            DummyState(temp_air=18.5, rh_air=82.0, glob_rad=120.0, hour_of_day=10.0)
        )

        self.assertFalse(director.last_rollout_selection["humidity_memory_prediction"]["enabled"])
        self.assertFalse(director.last_rollout_selection["humidity_memory_prediction"]["available"])

    def test_humidity_memory_shape_survives_night_guardrail_when_safe(self):
        director = self._director()
        director.config.humidity_memory_enabled = True
        director.last_rollout_selection = {"source": "humidity_memory"}
        director.last_humidity_memory_prediction = {"accepted": True}
        director.pending_control = np.array([0.0, 0.0, 0.05, 1.0, 0.0, 0.0], dtype=np.float32)

        final = director._flush_buffered_control(
            DummyState(
                temp_air=17.8,
                rh_air=82.0,
                co2_air=430.0,
                glob_rad=0.0,
                temp_out=12.0,
                rh_out=65.0,
                hour_of_day=4.0,
                dew_margin_air=3.0,
                canopy_dew_margin=3.0,
            )
        )

        self.assertTrue(director.last_rollout_selection["humidity_memory_post_guardrail_shape"]["applied"])
        self.assertGreaterEqual(float(final[2]), 0.449)
        self.assertGreaterEqual(float(final[3]), 0.55)
        self.assertLessEqual(float(final[3]), 0.58)
        self.assertLessEqual(float(final[0]), 0.10)
        self.assertGreaterEqual(float(final[0]), 0.059)

    def test_distilled_expert_can_be_selected_as_rollout_candidate(self):
        class WastefulRule:
            def predict(self, *_args):
                return np.array([0.70, 0.50, 1.00, 0.70, 0.80, 0.0], dtype=np.float32)

        class CalmExpert:
            def predict(self, *_args, **_kwargs):
                return np.zeros(6, dtype=np.float32), {
                    "feature_distance": 0.5,
                    "max_abs_z": 1.0,
                    "model": "test_expert",
                    "train_cases": 24,
                }

        director = self._director()
        director.config.expert_rollout_enabled = True
        director.config.expert_candidate_max_distance = 4.0
        director.rule_controller = WastefulRule()
        director.expert_policy = CalmExpert()
        director.current_plan = {
            "anchor_control": np.ones(6, dtype=np.float32) * 0.75,
            "created_timestep": 0,
            "expires_timestep": 12,
            "target_temp": 18.0,
            "target_co2": 430.0,
            "target_rh": 70.0,
            "target_profile": {},
        }

        control, _rule, _anchor, _rule_weight = director._plan_control_step(
            DummyState(
                temp_air=18.0,
                rh_air=70.0,
                co2_air=430.0,
                glob_rad=300.0,
                hour_of_day=12.0,
                dew_margin_air=2.0,
                canopy_dew_margin=2.0,
            )
        )

        self.assertEqual(director.last_rollout_selection["source"], "distilled_expert")
        self.assertTrue(director.last_rollout_selection["expert_prediction"]["accepted"])
        self.assertLess(float(control[0]), 0.05)
        self.assertLess(float(control[3]), 0.15)
        self.assertEqual(float(control[4]), 0.0)


if __name__ == "__main__":
    unittest.main()
