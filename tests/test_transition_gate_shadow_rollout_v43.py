import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gl_gym.agent.expert_distillation import ACTION_NAMES
from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector
from gl_gym.experiments.transition_gate_shadow_rollout_v43 import (
    CONTROLLED_CONTROLLER,
    FAILURE_SCENARIOS,
    build_execution_record,
    build_readiness,
)


def _director(config: AgentConfig) -> RuleBasedLLMDirector:
    director = object.__new__(RuleBasedLLMDirector)
    director.config = config
    director.last_transition_gate = {}
    director.last_tomato_safety_v2 = {"applied": False}
    director._transition_gate_previous_action = None
    director._transition_gate_sign_history = {name: deque(maxlen=6) for name in ACTION_NAMES}
    director._transition_gate_soft_limits = None
    director._transition_gate_soft_limit_source = ""
    director._transition_gate_soft_limit_error = ""
    return director


def _state(**overrides):
    defaults = {
        "temp_air": 24.0,
        "rh_air": 60.0,
        "dew_margin_air": 8.0,
        "canopy_dew_margin": 5.0,
        "dew_risk": False,
        "dry_risk": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _envelope(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "transition_stats_by_regime": [
                    {
                        "label": "stable_positive",
                        "regime": "neutral_or_mild",
                        "action_delta_stats": {
                            "u_heating": {"p95": 0.1},
                            "u_ventilation": {"p95": 0.1},
                            "u_screen": {"p95": 0.1},
                            "u_shading": {"p95": 0.1},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


class TestTransitionGateRuntimeV43(unittest.TestCase):
    def test_default_disabled_keeps_action_unchanged(self):
        director = _director(AgentConfig(transition_gate_enabled=False))
        action = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        result = director._apply_transition_gate(action, _state())
        np.testing.assert_allclose(result, action)
        self.assertFalse(director.last_transition_gate["enabled"])

    def test_non_safety_large_delta_is_limited_to_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            envelope = Path(tmp) / "envelope.json"
            _envelope(envelope)
            director = _director(
                AgentConfig(
                    transition_gate_enabled=True,
                    transition_gate_soft_limit_path=str(envelope),
                    transition_gate_reversal_window_steps=6,
                )
            )
            director._transition_gate_previous_action = np.asarray([0.1, 0.0, 0.1, 0.2, 0.0, 0.0], dtype=np.float32)
            action = np.asarray([0.1, 0.0, 0.1, 0.9, 0.0, 0.0], dtype=np.float32)
            result = director._apply_transition_gate(action, _state())

        self.assertAlmostEqual(float(result[ACTION_NAMES.index("ventilation")]), 0.3, places=5)
        self.assertTrue(director.last_transition_gate["applied"])
        self.assertIn("ventilation", director.last_transition_gate["adjusted_fields"])

    def test_hard_safety_bypass_is_not_clipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            envelope = Path(tmp) / "envelope.json"
            _envelope(envelope)
            director = _director(
                AgentConfig(
                    transition_gate_enabled=True,
                    transition_gate_soft_limit_path=str(envelope),
                    transition_gate_hard_safety_bypass=True,
                )
            )
            director._transition_gate_previous_action = np.asarray([0.1, 0.0, 0.1, 0.2, 0.0, 0.0], dtype=np.float32)
            director.last_tomato_safety_v2 = {"applied": True}
            action = np.asarray([0.1, 0.0, 0.1, 0.9, 0.0, 0.0], dtype=np.float32)
            result = director._apply_transition_gate(action, _state())

        np.testing.assert_allclose(result, action)
        self.assertTrue(director.last_transition_gate["bypassed"])
        self.assertIn("tomato_safety_v2_applied", director.last_transition_gate["bypass_reasons"])


class TestV43ExecutionRecord(unittest.TestCase):
    def test_execution_record_scope_is_fixed_and_shadow_only(self):
        record = build_execution_record()
        self.assertEqual(record["scenario_ids"], list(FAILURE_SCENARIOS))
        self.assertEqual(record["controller"], "llm_rspc_v2")
        self.assertFalse(record["controlled_controller_present"])
        self.assertNotIn(CONTROLLED_CONTROLLER, json.dumps(record))
        self.assertIn("transition_gate_shadow_v43_20260530.json", record["isolated_cache_path"])

    def test_readiness_never_allows_promotion_or_controlled_replay(self):
        readiness = build_readiness(build_execution_record(), {})
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
