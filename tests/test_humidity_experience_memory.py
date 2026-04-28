import unittest

import numpy as np

from gl_gym.agent.humidity_experience_memory import (
    HumidityExperienceMemory,
    action_from_row,
    assess_free_air_exchange_pair,
    build_experience_from_pair,
    dehumidification_potential,
)


def _llm_row(**overrides):
    row = {
        "algo": "llm_director",
        "year": 2020,
        "day": 240,
        "seed": 42,
        "step": 8,
        "timestep": 8,
        "hour_of_day": 9.0,
        "day_of_year": 240.0,
        "temp_air": 17.0,
        "rh_air": 84.0,
        "co2_air": 430.0,
        "glob_rad": 120.0,
        "temp_out": 9.0,
        "rh_out": 65.0,
        "wind_speed": 2.0,
        "dew_margin_air": 2.0,
        "canopy_dew_margin": 1.5,
        "forecast_humidity_risk": 0.25,
        "target_temp": 17.0,
        "target_co2": 430.0,
        "target_rh": 76.0,
        "profit": -0.0020,
        "heat_cost": 0.0005,
        "co2_cost": 0.0,
        "elec_cost": 0.0,
        "rh_violation": 0.0,
        "temp_violation": 0.0,
        "u_heating": 0.24,
        "u_co2": 0.0,
        "u_screen": 0.50,
        "u_ventilation": 0.40,
        "u_lighting": 0.0,
        "u_shading": 0.0,
    }
    row.update(overrides)
    return row


def _ppo_row(**overrides):
    row = _llm_row(
        algo="ppo",
        profit=-0.0013,
        heat_cost=0.0001,
        u_heating=0.02,
        u_screen=0.0,
        u_ventilation=0.78,
        strategy_label="free_air_exchange",
        strategy_confidence=0.78,
    )
    row.update(overrides)
    return row


class TestHumidityExperienceMemory(unittest.TestCase):
    def test_dehumidification_potential_uses_absolute_humidity_not_outer_rh_only(self):
        row = _llm_row(temp_air=17.0, rh_air=84.0, temp_out=9.0, rh_out=85.0)

        potential = dehumidification_potential(row)

        self.assertGreater(potential["absolute_humidity_advantage"], 0.5)
        self.assertGreater(potential["dew_point_advantage"], 1.0)

    def test_accepts_safe_economic_free_air_exchange_pair(self):
        assessment = assess_free_air_exchange_pair(_ppo_row(), _llm_row())

        self.assertTrue(assessment.accepted, assessment.reasons)
        self.assertEqual(assessment.intent_label, "economic_dehumidify")
        self.assertEqual(assessment.strategy_label, "free_air_exchange")
        self.assertGreater(assessment.metrics["operating_cost_saving_llm_minus_ppo"], 0.0)

    def test_allows_narrow_risk_override_when_llm_target_is_over_aggressive(self):
        llm = _llm_row(target_rh=72.0)
        ppo = _ppo_row(target_rh=72.0)

        assessment = assess_free_air_exchange_pair(ppo, llm)

        self.assertTrue(assessment.accepted, assessment.reasons)
        self.assertEqual(assessment.intent_label, "economic_dehumidify")
        self.assertEqual(assessment.metrics["plan_intent_overridden"], 1.0)

    def test_rejects_severe_humidity_as_not_economic_memory(self):
        llm = _llm_row(rh_air=91.0, target_rh=72.0, dew_margin_air=0.8, canopy_dew_margin=0.6)
        ppo = _ppo_row(rh_air=91.0, target_rh=72.0, dew_margin_air=0.8, canopy_dew_margin=0.6)

        assessment = assess_free_air_exchange_pair(ppo, llm)

        self.assertFalse(assessment.accepted)
        self.assertIn("not_moderate_rh", assessment.reasons)
        self.assertIn("intent_is_not_economic_dehumidify", assessment.reasons)

    def test_rejects_when_outside_air_is_not_drier(self):
        llm = _llm_row(temp_out=17.0, rh_out=90.0)
        ppo = _ppo_row(temp_out=17.0, rh_out=90.0)

        assessment = assess_free_air_exchange_pair(ppo, llm)

        self.assertFalse(assessment.accepted)
        self.assertIn("outside_air_has_no_dehumidification_potential", assessment.reasons)

    def test_memory_retrieves_and_builds_candidate_from_residual(self):
        experience, assessment = build_experience_from_pair(_ppo_row(), _llm_row())
        self.assertIsNotNone(experience)
        self.assertTrue(assessment.accepted)
        memory = HumidityExperienceMemory([experience])

        matches = memory.retrieve(_llm_row(step=12, timestep=12), target_row=_llm_row(step=12, timestep=12))

        self.assertTrue(matches)
        candidate = np.asarray(matches[0]["candidate_action"], dtype=np.float32)
        self.assertEqual(candidate.shape[0], 6)
        self.assertGreater(float(candidate[3]), float(action_from_row(_llm_row())[3]))
        self.assertLess(float(candidate[0]), float(action_from_row(_llm_row())[0]))

    def test_metadata_filter_prevents_cross_version_reuse(self):
        experience, _ = build_experience_from_pair(
            _ppo_row(),
            _llm_row(),
            source_metadata={
                "teacher_policy_id": "ppo_a",
                "baseline_controller_id": "rspc_a",
                "memory_schema_version": "hem_rspc_v1",
            },
        )
        memory = HumidityExperienceMemory([experience])

        matching = memory.retrieve(
            _llm_row(),
            target_row=_llm_row(),
            required_metadata={"teacher_policy_id": "ppo_a"},
        )
        mismatching = memory.retrieve(
            _llm_row(),
            target_row=_llm_row(),
            required_metadata={"teacher_policy_id": "ppo_b"},
        )

        self.assertTrue(matching)
        self.assertFalse(mismatching)


if __name__ == "__main__":
    unittest.main()
