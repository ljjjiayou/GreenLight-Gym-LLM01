import unittest
from types import SimpleNamespace

import numpy as np

from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector


def _director():
    director = object.__new__(RuleBasedLLMDirector)
    director.config = AgentConfig()
    director.last_control = None
    director._canopy_margin_history = None
    director._last_canopy_history_timestep = None
    return director


def _state(**extra):
    data = {
        "temp_air": 24.0,
        "rh_air": 82.0,
        "co2_air": 500.0,
        "glob_rad": 320.0,
        "temp_out": 18.0,
        "rh_out": 65.0,
        "hour_of_day": 14.0,
        "fruit_weight": 10.0,
        "dew_margin_air": 1.2,
        "canopy_dew_margin": 0.35,
        "u_boil": 0.0,
        "u_co2": 0.0,
        "u_th_scr": 0.4,
        "u_vent": 0.5,
        "u_lamp": 0.0,
        "u_bl_scr": 0.0,
    }
    data.update(extra)
    return SimpleNamespace(**data)


class TestCanopyShadowMetadata(unittest.TestCase):
    def test_candidate_score_terms_include_proxy_predictions_without_score_change(self):
        director = _director()
        control = np.asarray([0.0, 0.0, 0.8, 0.1, 0.0, 0.0], dtype=np.float32)

        score, details = director._score_fallback_candidate(_state(), control)
        subset = director._score_term_subset(details)

        self.assertAlmostEqual(score, details["total_score"])
        self.assertEqual(subset["prediction_model"], "proxy_v1")
        self.assertEqual(subset["prediction_schema_version"], "canopy_proxy_schema_v2")
        self.assertEqual(subset["prediction_model_v2"], "proxy_v2_conservative")
        self.assertIn("predicted_canopy_dew_margin_next", subset)
        self.assertIn("predicted_canopy_dew_margin_next_v2", subset)
        self.assertIn("predicted_canopy_warning_v2", subset)
        self.assertIn("canopy_proxy_v2_risk_buffer", subset)
        self.assertIn("canopy_proxy_v2_risk_terms", subset)
        self.assertIn("predicted_dew_margin_air_next", subset)

    def test_boundary_shadow_marks_screen_up_vent_down_near_canopy_risk(self):
        director = _director()
        before = np.asarray([0.0, 0.0, 0.4, 0.5, 0.0, 0.0], dtype=np.float32)
        final = np.asarray([0.0, 0.0, 0.8, 0.1, 0.0, 0.0], dtype=np.float32)

        record = director._canopy_boundary_shadow_record(_state(), before, final)

        self.assertTrue(record["canopy_dew_boundary_active"])
        self.assertTrue(record["screen_increase_under_canopy_risk"])
        self.assertTrue(record["ventilation_decrease_under_canopy_risk"])
        self.assertTrue(record["would_reject_for_canopy_dew_boundary"])
        self.assertEqual(record["boundary_reject_reason"], "canopy_dew_screen_vent_conflict")
        self.assertTrue(record["canopy_boundary_warning_v2"])
        self.assertTrue(record["would_reject_for_canopy_dew_boundary_v2"])

    def test_proxy_v2_warns_for_step_233_like_case_without_changing_score(self):
        director = _director()
        state = _state(
            timestep=233,
            rh_air=88.0,
            dew_margin_air=0.9,
            canopy_dew_margin=0.55,
            u_th_scr=0.55,
            u_vent=0.25,
        )
        control = np.asarray([0.0, 0.0, 0.68, 0.18, 0.0, 0.0], dtype=np.float32)

        score, details = director._score_fallback_candidate(state, control)

        self.assertAlmostEqual(score, details["total_score"])
        self.assertTrue(details["predicted_canopy_warning_v2"])
        self.assertEqual(details["prediction_model_v2"], "proxy_v2_conservative")


if __name__ == "__main__":
    unittest.main()
