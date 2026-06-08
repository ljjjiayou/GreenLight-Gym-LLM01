import json
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from gl_gym.agent.llm_agent import AgentConfig, FallbackCandidate, RuleBasedLLMDirector
from gl_gym.experiments.diagnose_ppo_vs_llm import profile_template_patch_to_record
from gl_gym.experiments.qwen37_recovery_anchor_opt_in_shadow_rollout_v56 import (
    AGENT_OVERRIDES,
    FAILURE_SCENARIOS,
    MODEL_NAME,
    build_execution_record,
    build_readiness,
)


def _director(**config_overrides):
    director = object.__new__(RuleBasedLLMDirector)
    director.config = AgentConfig(**config_overrides)
    director.last_profile_template_patch = {
        "enabled": bool(getattr(director.config, "profile_template_patch_enabled", False)),
        "applied": False,
        "fallback_veto_applied": False,
        "fallback_veto_no_alternative": False,
        "recovery_anchor_enabled": bool(getattr(director.config, "recovery_anchor_enabled", False)),
        "recovery_anchor_applied": False,
    }
    director.current_plan = {}
    return director


def _state(**values):
    base = {
        "rh_air": 63.5,
        "temp_air": 26.7,
        "glob_rad": 405.9,
        "dew_margin_air": 7.5,
        "canopy_dew_margin": 1.9,
        "temp_violation": 0.0,
    }
    base.update(values)
    return SimpleNamespace(**base)


class TestRecoveryAnchorRuntimeV56(unittest.TestCase):
    def test_default_disabled_keeps_selected_action(self):
        director = _director(
            profile_template_patch_enabled=True,
            fallback_post_selection_veto_enabled=True,
            recovery_anchor_enabled=False,
        )
        bad = FallbackCandidate("bad_anchor", np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), score=0.1)

        def fake_prediction(self, state, control):
            return {
                "hard_safety_rewrite": True,
                "tomato_safety_applied": True,
                "safety_reasons": ["hot_dry_cooling_guard"],
                "rewrite_fields": ["heat", "screen", "vent", "shade"],
                "after": [0.0, 0.0, 0.6, 0.6, 0.0, 0.5],
            }

        director._fallback_candidate_safety_prediction = MethodType(fake_prediction, director)
        selected, record = director._apply_fallback_post_selection_veto(_state(), [bad])

        self.assertEqual(selected.name, "bad_anchor")
        self.assertFalse(record["recovery_anchor_applied"])
        self.assertTrue(record["no_alternative"])

    def test_projected_recovery_anchor_created_when_no_compatible_candidate(self):
        director = _director(
            profile_template_patch_enabled=True,
            fallback_post_selection_veto_enabled=True,
            recovery_anchor_enabled=True,
        )
        bad = FallbackCandidate("bad_anchor", np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), score=0.1)

        def fake_prediction(self, state, control):
            is_projected = float(control[2]) >= 0.59 and float(control[3]) >= 0.59
            return {
                "hard_safety_rewrite": not is_projected,
                "tomato_safety_applied": not is_projected,
                "safety_reasons": [] if is_projected else ["hot_dry_cooling_guard"],
                "rewrite_fields": [] if is_projected else ["heat", "screen", "vent", "shade"],
                "after": [0.0, 0.0, 0.6, 0.6, 0.0, 0.5],
            }

        director._fallback_candidate_safety_prediction = MethodType(fake_prediction, director)
        selected, record = director._apply_fallback_post_selection_veto(_state(), [bad])

        self.assertEqual(selected.name, "tomato_safety_projected_anchor")
        self.assertTrue(record["recovery_anchor_applied"])
        self.assertTrue(record["recovery_anchor_no_compatible_existing_candidate"])
        self.assertFalse(record["selected_after_hard_safety_rewrite"])
        self.assertFalse(record["no_alternative"])

    def test_existing_compatible_candidate_wins_over_recovery_anchor(self):
        director = _director(
            profile_template_patch_enabled=True,
            fallback_post_selection_veto_enabled=True,
            recovery_anchor_enabled=True,
        )
        bad = FallbackCandidate("bad_anchor", np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), score=0.1)
        good = FallbackCandidate("compatible_anchor", np.asarray([0.0, 0.0, 0.6, 0.6, 0.0, 0.5], dtype=np.float32), score=0.2)

        def fake_prediction(self, state, control):
            hard = float(control[0]) > 0.5
            return {
                "hard_safety_rewrite": hard,
                "tomato_safety_applied": hard,
                "safety_reasons": ["hard"] if hard else [],
                "rewrite_fields": ["heat"] if hard else [],
                "after": [0.0, 0.0, 0.6, 0.6, 0.0, 0.5],
            }

        director._fallback_candidate_safety_prediction = MethodType(fake_prediction, director)
        selected, record = director._apply_fallback_post_selection_veto(_state(), [bad, good])

        self.assertEqual(selected.name, "compatible_anchor")
        self.assertFalse(record["recovery_anchor_applied"])
        self.assertTrue(record["applied"])

    def test_recovery_anchor_trace_record_flattens_without_secrets(self):
        record = profile_template_patch_to_record(
            {
                "recovery_anchor_enabled": True,
                "recovery_anchor_applied": True,
                "recovery_anchor_source": "tomato_safety_projected_anchor",
                "recovery_anchor_reason": "fallback_veto_no_alternative_projected_anchor",
                "recovery_anchor_no_compatible_existing_candidate": True,
                "recovery_anchor_action": {"heat": 0.0, "vent": 0.6},
                "recovery_anchor_prediction": {"hard_safety_rewrite": False, "tomato_safety_applied": False},
                "recovery_anchor_selected_before": "bad_anchor",
                "recovery_anchor_selected_after": "tomato_safety_projected_anchor",
            }
        )

        self.assertTrue(record["recovery_anchor_enabled"])
        self.assertTrue(record["recovery_anchor_applied"])
        self.assertFalse(record["recovery_anchor_hard_safety_rewrite"])
        self.assertNotIn("BAILIAN_API_KEY", json.dumps(record))


class TestV56ExecutionAndReadiness(unittest.TestCase):
    def test_execution_record_scope_is_fixed_and_shadow_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            v55 = Path(tmp) / "v55.json"
            v55.write_text(
                json.dumps(
                    {
                        "v55_recovery_anchor_design_complete": True,
                        "recovery_anchor_coverage_rate": 1.0,
                        "no_recovery_anchor_available_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            record = build_execution_record(
                {
                    "model_name": MODEL_NAME,
                    "online_llm_credentials_present": True,
                    "online_llm_accessible": True,
                    "provider_error_detected": False,
                },
                v55_readiness_json=v55,
            )

        self.assertEqual(record["scenarios"], list(FAILURE_SCENARIOS))
        self.assertEqual(record["controller"], "llm_rspc_v2")
        self.assertEqual(record["model_name"], MODEL_NAME)
        self.assertFalse(record["controlled_controller_included"])
        self.assertTrue(record["profile_template_patch_enabled"])
        self.assertTrue(record["fallback_post_selection_veto_enabled"])
        self.assertTrue(record["recovery_anchor_enabled"])
        self.assertFalse(record["transition_gate_enabled"])
        self.assertFalse(AGENT_OVERRIDES["profile_feasibility_gate_enabled"])
        self.assertIn("qwen37_recovery_anchor_opt_in_shadow_v56_20260601.json", record["cache_path"])
        for command in record["commands"]:
            self.assertIn("--llm-model qwen3.7-max", command["command"])
            self.assertIn('"recovery_anchor_enabled": true', command["command"])

    def test_readiness_never_allows_replay_or_claims(self):
        readiness = build_readiness(
            {
                "totals": {"recovery_anchor_applied_steps": 1},
                "acceptance": {"v56_acceptance_pass": True, "all_v56_traces_present": True},
            },
            execution_record={"executable": True, "online_llm_accessible": True},
        )

        self.assertEqual(readiness["next_action"], "recovery_anchor_h720_shadow_expansion_plan")
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
