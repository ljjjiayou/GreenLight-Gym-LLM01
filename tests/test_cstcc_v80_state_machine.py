import unittest

from gl_gym.cstcc.contracts import SemanticSuggestion
from gl_gym.cstcc.semantic_state_machine import REGIME_SPECS, evaluate_transition
from gl_gym.cstcc.weight_governor import effective_confidence


class TestCSTCCV80StateMachine(unittest.TestCase):
    def test_invalid_regime_is_not_added_to_state_machine(self):
        suggestion = SemanticSuggestion(
            regime="VERY_HOT_DANGEROUS_DRY_WEATHER",
            regime_confidence=0.99,
            priority_weight_suggestions={},
        )

        state = evaluate_transition(
            "NORMAL_BALANCED",
            20,
            suggestion,
            evidence_confidence=1.0,
            data_quality_score=1.0,
        )

        self.assertEqual(state.active_regime, "NORMAL_BALANCED")
        self.assertEqual(state.invalid_regime, "VERY_HOT_DANGEROUS_DRY_WEATHER")
        self.assertIn("NORMAL_BALANCED", REGIME_SPECS)

    def test_min_hold_prevents_non_emergency_transition(self):
        suggestion = SemanticSuggestion(
            regime="HIGH_VPD_DRY_STRESS",
            regime_confidence=0.95,
            priority_weight_suggestions={"vpd_risk": 0.5},
        )

        state = evaluate_transition(
            "NORMAL_BALANCED",
            1,
            suggestion,
            evidence_confidence=1.0,
            data_quality_score=1.0,
        )

        self.assertEqual(state.active_regime, "NORMAL_BALANCED")
        self.assertEqual(state.transition_reason, "min_hold_not_satisfied")

    def test_emergency_transition_overrides_hold_time(self):
        suggestion = SemanticSuggestion(
            regime="SOLVER_SENSITIVE_EMERGENCY",
            regime_confidence=0.80,
            priority_weight_suggestions={"solver_risk": 0.35},
        )

        state = evaluate_transition(
            "LOW_LIGHT_ENERGY_SAVING",
            0,
            suggestion,
            evidence_confidence=0.80,
            data_quality_score=1.0,
        )

        self.assertEqual(state.active_regime, "SOLVER_SENSITIVE_EMERGENCY")
        self.assertEqual(state.transition_reason, "emergency_override")

    def test_effective_confidence_does_not_blindly_trust_llm(self):
        self.assertEqual(effective_confidence(0.95, 0.30, mode="min"), 0.30)
        self.assertAlmostEqual(effective_confidence(0.95, 0.30, mode="weighted"), 0.56)


if __name__ == "__main__":
    unittest.main()
