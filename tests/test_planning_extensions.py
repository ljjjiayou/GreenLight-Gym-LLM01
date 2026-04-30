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

    def test_tomato_safety_v2_hot_dry_prefers_cooling_ventilation(self):
        cfg = AgentConfig(tomato_safety_v2_enabled=True)

        control, info = apply_tomato_safety_v2(
            DummyState(
                temp_air=30.0,
                rh_air=43.0,
                glob_rad=760.0,
                hour_of_day=12.0,
                dew_margin_air=2.0,
                canopy_dew_margin=2.0,
            ),
            np.array([0.2, 0.5, 0.0, 0.12, 0.6, 0.65], dtype=np.float32),
            config=cfg,
            target_rh=72.0,
        )

        self.assertTrue(info["applied"])
        self.assertIn("hot_dry_cooling_guard", info["reasons"])
        self.assertIn("hard_dry_vpd_guard", info["reasons"])
        self.assertGreaterEqual(float(control[3]), 0.779)
        self.assertGreaterEqual(float(control[2]), 0.999)
        self.assertLessEqual(float(control[5]), 0.001)
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
        self.assertGreaterEqual(float(control[3]), 0.599)
        self.assertGreaterEqual(float(control[2]), 0.999)

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

    def test_tomato_safety_v2_suppresses_strict_replay_emergency_replans_only(self):
        director = object.__new__(RuleBasedLLMDirector)
        director.config = AgentConfig(
            tomato_safety_v2_enabled=True,
            plan_cache_mode="replay",
            plan_cache_key_policy="scenario_timestep",
            plan_cache_strict=True,
        )

        self.assertTrue(director._should_suppress_tomato_v2_replay_emergency_replan())

        director.config.plan_cache_key_policy = "prompt"
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
