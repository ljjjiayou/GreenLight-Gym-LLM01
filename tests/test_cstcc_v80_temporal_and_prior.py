import unittest

from gl_gym.cstcc.prior_confidence import (
    candidate_count_from_confidence,
    ppo_prior_confidence,
    previous_plan_confidence,
)
from gl_gym.cstcc.temporal_context import build_temporal_context


class TestCSTCCV80TemporalAndPrior(unittest.TestCase):
    def test_debt_uses_real_time_normalization_and_saturation(self):
        states = [
            {"temp_air": 30.0, "rh_air": 45.0, "vpd_air": 2.6},
            {"temp_air": 31.0, "rh_air": 44.0, "vpd_air": 2.8},
            {"temp_air": 32.0, "rh_air": 43.0, "vpd_air": 3.0},
        ]

        context = build_temporal_context(
            states,
            action_history=[],
            delta_t_hours=0.25,
            gamma=1.0,
            debt_refs={"vpd_debt_kpa_h": 0.1, "heat_debt_c_h": 0.1, "dryness_debt_rh_h": 0.1},
        )

        self.assertGreater(context.debt_features["vpd_debt_kpa_h"], 0.0)
        self.assertEqual(context.debt_features["vpd_debt_norm"], 1.0)
        self.assertEqual(context.window_hours, 0.75)
        self.assertIn("reversal_count_norm", context.action_smoothness_features)

    def test_ppo_confidence_has_floor_but_emergency_can_zero(self):
        low_conf, _ = ppo_prior_confidence(
            base_conf=0.8,
            ood_score=1.0,
            rewrite_pressure=1.0,
            reversal_risk=1.0,
            min_conf=0.05,
        )
        zero_conf, reason = ppo_prior_confidence(solver_sensitive_emergency=True)

        self.assertEqual(low_conf, 0.05)
        self.assertEqual(zero_conf, 0.0)
        self.assertEqual(reason["emergency_zeroed"], "solver_sensitive_emergency")

    def test_previous_plan_emergency_shift_zeroes_confidence(self):
        conf, reason = previous_plan_confidence(emergency_regime_shift=True)

        self.assertEqual(conf, 0.0)
        self.assertEqual(reason["emergency_zeroed"], "emergency_regime_shift")

    def test_candidate_count_bins(self):
        self.assertEqual(candidate_count_from_confidence(0.0), 0)
        self.assertEqual(candidate_count_from_confidence(0.2), 1)
        self.assertEqual(candidate_count_from_confidence(0.5), 2)
        self.assertEqual(candidate_count_from_confidence(0.7), 3)
        self.assertEqual(candidate_count_from_confidence(0.9), 4)


if __name__ == "__main__":
    unittest.main()
