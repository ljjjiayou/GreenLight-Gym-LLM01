import unittest
from dataclasses import dataclass

import numpy as np

from gl_gym.agent.expert_distillation import DistilledExpertPolicy, FEATURE_NAMES
from gl_gym.agent.plan_intent import (
    action_record_for_labeling,
    default_setpoint_contract,
    infer_plan_intent,
    make_plan_from_targets,
    strategy_intent_alignment,
    target_tracking_baseline_control,
)
from gl_gym.agent.ppo_strategy_labeler import label_strategy


@dataclass
class DummyState:
    timestep: int = 0
    hour_of_day: float = 12.0
    day_of_year: float = 180.0
    temp_air: float = 17.0
    rh_air: float = 84.0
    co2_air: float = 430.0
    glob_rad: float = 120.0
    temp_out: float = 12.0
    rh_out: float = 75.0
    wind_speed: float = 1.5
    pipe_temp: float = 30.0
    fruit_weight: float = 0.2
    canopy_temp_24h: float = 17.0
    temperature_sum: float = 100.0
    dli: float = 3.0
    dew_margin_air: float = 1.5
    canopy_dew_margin: float = 1.5
    forecast_humidity_risk: float = 0.2
    time_to_sunrise_steps: float = 4.0
    u_boil: float = 0.0
    u_co2: float = 0.0
    u_th_scr: float = 0.0
    u_vent: float = 0.0
    u_lamp: float = 0.0
    u_bl_scr: float = 0.0


class TestPlanIntentDistillation(unittest.TestCase):
    def test_contract_caps_rh_target_under_extreme_humidity(self):
        state = DummyState(temp_air=16.0, rh_air=93.0, dew_margin_air=0.5, canopy_dew_margin=0.5)

        plan, diagnostics = default_setpoint_contract(state, horizon=12)
        intent = infer_plan_intent(state, plan)

        self.assertLessEqual(plan["target_rh"], 72.0)
        self.assertEqual(intent.label, "safe_dehumidify")
        self.assertGreaterEqual(diagnostics["humidity_risk"], 0.5)

    def test_moderate_rh_target_gap_prefers_economic_dehumidification(self):
        state = DummyState(temp_air=17.0, rh_air=84.0, dew_margin_air=1.6, canopy_dew_margin=1.6)
        plan = make_plan_from_targets(state, target_temp=17.0, target_co2=430.0, target_rh=76.0)

        intent = infer_plan_intent(state, plan)
        base = target_tracking_baseline_control(state, plan, intent)

        self.assertEqual(intent.label, "economic_dehumidify")
        self.assertGreater(float(base[3]), 0.25)
        self.assertLess(float(base[0]), 0.10)

    def test_free_air_exchange_aligns_with_economic_dehumidification(self):
        state = DummyState(temp_air=17.0, rh_air=84.0, dew_margin_air=1.6, canopy_dew_margin=1.6)
        plan = make_plan_from_targets(state, target_temp=17.0, target_co2=430.0, target_rh=76.0)
        intent = infer_plan_intent(state, plan)
        strategy = label_strategy(
            action_record_for_labeling(
                state,
                np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            )
        )

        alignment = strategy_intent_alignment(strategy, intent)

        self.assertEqual(strategy.label, "free_air_exchange")
        self.assertTrue(alignment["aligned"])
        self.assertGreater(alignment["score"], 0.5)

    def test_heat_vent_strategy_is_not_economic_dehumidification(self):
        state = DummyState(temp_air=17.0, rh_air=84.0, dew_margin_air=1.6, canopy_dew_margin=1.6)
        plan = make_plan_from_targets(state, target_temp=17.0, target_co2=430.0, target_rh=76.0)
        intent = infer_plan_intent(state, plan)
        strategy = label_strategy(
            action_record_for_labeling(
                state,
                np.asarray([0.35, 0.0, 0.2, 0.9, 0.0, 0.0], dtype=np.float32),
            )
        )

        alignment = strategy_intent_alignment(strategy, intent)

        self.assertEqual(intent.label, "economic_dehumidify")
        self.assertFalse(alignment["aligned"])

    def test_residual_expert_adds_prediction_to_target_tracking_baseline(self):
        state = DummyState(temp_air=17.0, rh_air=84.0, dew_margin_air=1.6, canopy_dew_margin=1.6)
        plan = make_plan_from_targets(state, target_temp=17.0, target_co2=430.0, target_rh=76.0)
        feature_count = len(FEATURE_NAMES)
        policy = DistilledExpertPolicy(
            feature_names=FEATURE_NAMES,
            feature_mean=np.zeros(feature_count, dtype=np.float32),
            feature_std=np.ones(feature_count, dtype=np.float32),
            coef=np.zeros((feature_count, 6), dtype=np.float32),
            intercept=np.asarray([0.01, 0.0, 0.0, 0.05, 0.0, 0.0], dtype=np.float32),
            metadata={"target_mode": "residual", "model": "unit_test"},
        )

        action, info = policy.predict(state, plan=plan)
        base = np.asarray(info["base_action"], dtype=np.float32)

        self.assertEqual(info["target_mode"], "residual")
        self.assertAlmostEqual(float(action[0]), float(base[0] + 0.01), places=5)
        self.assertAlmostEqual(float(action[3]), float(base[3] + 0.05), places=5)


if __name__ == "__main__":
    unittest.main()
