import unittest
import copy

from gl_gym.cstcc.config import load_config
from gl_gym.cstcc.audit import build_audit_record
from gl_gym.cstcc.contracts import SafetyProjectionReport, SemanticSuggestion, asdict_clean
from gl_gym.cstcc.dual_scorer import score_dual
from gl_gym.cstcc.sequence_templates import generate_sequence


LAST_ACTION = {
    "u_heating": 0.2,
    "u_co2": 0.2,
    "u_screen": 0.4,
    "u_ventilation": 0.4,
    "u_lighting": 0.0,
    "u_shading": 0.2,
}


class TestCSTCCV80DualScorerAndAudit(unittest.TestCase):
    def test_dual_score_separates_projected_score_and_rewrite_penalty(self):
        raw = generate_sequence("hold_all", last_action=LAST_ACTION, horizon=4)
        projected = generate_sequence("hold_all", last_action=LAST_ACTION, horizon=4)
        report = SafetyProjectionReport(
            rewrite_magnitude=0.4,
            rewrite_reasons=["vpd_boundary", "solver_sensitive"],
            rewrite_reason_severity={"vpd_boundary": "high", "solver_sensitive": "critical"},
            severe_rewrite_count=2,
            projected_safe=True,
        )

        score = score_dual(raw, projected, report, lambda_rewrite=0.25)

        self.assertGreater(score.rewrite_penalty, 0.0)
        self.assertAlmostEqual(score.final_score, score.projected_score - score.rewrite_penalty)
        self.assertEqual(score.score_validity, "sequence_only_no_rollout")
        self.assertEqual(report.projection_level, "placeholder_no_real_projection")

    def test_audit_record_is_complete_and_shadow_only(self):
        state_history = [
            {"temp_air": 29.0, "rh_air": 50.0, "vpd_air": 2.1},
            {"temp_air": 30.0, "rh_air": 48.0, "vpd_air": 2.4},
            {"temp_air": 31.0, "rh_air": 46.0, "vpd_air": 2.7},
        ]
        action_history = [LAST_ACTION, {**LAST_ACTION, "u_ventilation": 0.45}]
        suggestion = SemanticSuggestion(
            regime="HIGH_VPD_DRY_STRESS",
            regime_confidence=0.95,
            llm_reported_confidence=0.95,
            priority_weight_suggestions={"vpd_risk": 0.50, "humidity_recovery": 0.30},
            control_intent="reduce dry stress without abrupt ventilation reversal",
            risk_factors=["vpd_rising", "rh_declining"],
        )

        record = build_audit_record(
            state_history=state_history,
            action_history=action_history,
            semantic_suggestion=suggestion,
            current_runtime_final_action=action_history[-1],
            active_regime="NORMAL_BALANCED",
            regime_age_steps=10,
            ppo_action={**LAST_ACTION, "u_ventilation": 0.50},
        )
        payload = asdict_clean(record)

        self.assertFalse(payload["controller_changed"])
        self.assertFalse(payload["online_llm_called"])
        self.assertFalse(payload["final_action_changed"])
        self.assertFalse(payload["predictive_rollout_executed"])
        self.assertEqual(payload["prediction_level"], 0)
        self.assertEqual(payload["active_regime_after"], "HIGH_VPD_DRY_STRESS")
        self.assertTrue(payload["sequence_candidates"])
        self.assertTrue(payload["tomato_safety_dual_scores"])
        self.assertEqual(payload["scoring_metadata"]["score_validity"], "sequence_only_no_rollout")
        self.assertEqual(payload["scoring_metadata"]["tomato_safety_projection_level"], "placeholder_no_real_projection")
        self.assertFalse(payload["scoring_metadata"]["real_tomato_safety_projection"])
        self.assertIsNotNone(payload["selected_sequence_id"])
        self.assertEqual(payload["selected_sequence_id"], payload["shadow_selected_sequence_id"])
        self.assertTrue(payload["selected_action_is_shadow_only"])
        self.assertIn("action_difference_from_runtime", payload)
        self.assertIn("shadow_action_difference_from_runtime", payload)

    def test_low_regime_specific_evidence_suppresses_llm_influence(self):
        state_history = [
            {"temp_air": 29.0, "rh_air": 50.0, "vpd_air": 2.1},
            {"temp_air": 30.0, "rh_air": 48.0, "vpd_air": 2.4},
            {"temp_air": 31.0, "rh_air": 46.0, "vpd_air": 2.7},
        ]
        suggestion = SemanticSuggestion(
            regime="LOW_LIGHT_ENERGY_SAVING",
            regime_confidence=0.99,
            llm_reported_confidence=0.99,
            priority_weight_suggestions={"energy": 0.25},
        )

        record = build_audit_record(
            state_history=state_history,
            action_history=[LAST_ACTION, {**LAST_ACTION, "u_ventilation": 0.45}],
            semantic_suggestion=suggestion,
            current_runtime_final_action=LAST_ACTION,
            active_regime="NORMAL_BALANCED",
            regime_age_steps=20,
        )
        payload = asdict_clean(record)

        self.assertLess(payload["evidence_confidence"], 0.5)
        self.assertLess(payload["effective_confidence"], 0.5)
        self.assertLess(payload["weight_governor_report"]["llm_influence_alpha"], 0.10)

    def test_inline_config_rejects_final_action_change_and_prediction_level(self):
        bad_config = copy.deepcopy(load_config())
        bad_config["final_action_changed"] = True

        with self.assertRaises(ValueError):
            build_audit_record(state_history=[], action_history=[], config=bad_config)

        bad_config = copy.deepcopy(load_config())
        bad_config["prediction_level"] = 1

        with self.assertRaises(ValueError):
            build_audit_record(state_history=[], action_history=[], config=bad_config)

    def test_no_generated_candidate_triggers_fallback(self):
        config = copy.deepcopy(load_config())
        config["sequence_generation"]["max_candidates"] = 0

        record = build_audit_record(
            state_history=[{"temp_air": 25.0, "rh_air": 60.0, "vpd_air": 1.2}],
            action_history=[LAST_ACTION],
            semantic_suggestion=SemanticSuggestion(regime="NORMAL_BALANCED", regime_confidence=0.8),
            current_runtime_final_action=LAST_ACTION,
            config=config,
        )
        payload = asdict_clean(record)

        self.assertTrue(payload["fallback_triggered"])
        self.assertEqual(payload["fallback_reason"], "no_feasible_candidate")
        self.assertEqual(payload["shadow_selected_sequence_id"], "fallback:conservative_stabilize:0")


if __name__ == "__main__":
    unittest.main()
