import json
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from gl_gym.agent.llm_agent import AgentConfig, FallbackCandidate, RuleBasedLLMDirector
from gl_gym.experiments.diagnose_ppo_vs_llm import profile_template_patch_to_record
from gl_gym.experiments.profile_template_opt_in_shadow_rollout_v51 import (
    AGENT_OVERRIDES,
    FAILURE_SCENARIOS,
    build_execution_record,
    build_readiness,
)


def _director(**config_overrides):
    director = object.__new__(RuleBasedLLMDirector)
    director.config = AgentConfig(**config_overrides)
    director.last_tomato_safety_v2 = {"enabled": True, "applied": False}
    director.last_profile_template_patch = {
        "enabled": bool(getattr(director.config, "profile_template_patch_enabled", False)),
        "applied": False,
        "fallback_veto_applied": False,
        "fallback_veto_no_alternative": False,
    }
    director.current_plan = {}
    return director


def _state(**values):
    base = {
        "rh_air": 48.0,
        "temp_air": 29.5,
        "glob_rad": 500.0,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "temp_violation": 0.0,
    }
    base.update(values)
    return SimpleNamespace(**base)


class TestProfileTemplatePatchRuntimeV51(unittest.TestCase):
    def test_default_disabled_returns_targets_unchanged(self):
        director = _director(profile_template_patch_enabled=False)
        target = (24.0, 900.0, 82.0)
        repaired = director._apply_profile_template_patch_targets(
            _state(rh_air=92.0, dew_margin_air=0.5, canopy_dew_margin=0.6),
            np.asarray([0.0, 0.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32),
            *target,
            source="test",
        )

        self.assertEqual(repaired[:3], target)
        self.assertFalse(repaired[3]["enabled"])
        self.assertFalse(repaired[3]["applied"])

    def test_opt_in_template_blocks_high_rh_high_co2_during_safety_vent(self):
        director = _director(profile_template_patch_enabled=True)
        repaired_temp, repaired_co2, repaired_rh, record = director._apply_profile_template_patch_targets(
            _state(rh_air=92.0, dew_margin_air=0.5, canopy_dew_margin=0.6),
            np.asarray([0.0, 0.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32),
            24.0,
            900.0,
            82.0,
            source="test",
        )

        self.assertTrue(record["enabled"])
        self.assertTrue(record["applied"])
        self.assertTrue(record["mode"]["vent_required_by_safety"])
        self.assertLess(repaired_rh, 82.0)
        self.assertEqual(repaired_co2, 430.0)
        self.assertIn("safety_vent_blocks_high_rh_high_co2", record["forbidden_combinations"])

    def test_fallback_post_selection_veto_switches_to_compatible_candidate(self):
        director = _director(
            profile_template_patch_enabled=True,
            fallback_post_selection_veto_enabled=True,
        )
        bad = FallbackCandidate("bad_fallback", np.asarray([0.0, 0.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32), score=0.1)
        good = FallbackCandidate("good_anchor", np.asarray([0.0, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float32), score=0.2)

        def fake_prediction(self, state, control):
            hard = float(control[3]) > 0.5
            return {
                "hard_safety_rewrite": hard,
                "tomato_safety_applied": hard,
                "safety_reasons": ["hard"] if hard else [],
                "rewrite_fields": ["vent"] if hard else [],
            }

        director._fallback_candidate_safety_prediction = MethodType(fake_prediction, director)
        selected, record = director._apply_fallback_post_selection_veto(_state(), [bad, good])

        self.assertEqual(selected.name, "good_anchor")
        self.assertTrue(record["applied"])
        self.assertEqual(record["selected_before"], "bad_fallback")
        self.assertEqual(record["selected_after"], "good_anchor")

    def test_trace_record_flattens_profile_template_patch_without_secrets(self):
        record = profile_template_patch_to_record(
            {
                "enabled": True,
                "applied": True,
                "corrections": ["target_co2:vent_or_radiation_cap"],
                "forbidden_combinations": ["safety_vent_blocks_high_rh_high_co2"],
                "mode": {"vent_required_by_safety": True, "dry_side": False},
                "original_targets": {"target_temp": 24.0, "target_co2": 900.0, "target_rh": 82.0},
                "repaired_targets": {"target_temp": 24.0, "target_co2": 430.0, "target_rh": 72.0},
                "fallback_veto_enabled": True,
                "fallback_veto_applied": True,
                "fallback_veto_reason": "fallback_post_selection_hard_safety_rewrite",
            }
        )

        self.assertTrue(record["profile_template_patch_enabled"])
        self.assertTrue(record["profile_template_patch_applied"])
        self.assertTrue(record["profile_template_patch_fallback_veto_applied"])
        self.assertEqual(record["profile_template_patch_repaired_target_co2"], 430.0)
        self.assertNotIn("BAILIAN_API_KEY", json.dumps(record))


class TestV51ExecutionRecord(unittest.TestCase):
    def test_execution_record_scope_is_fixed_and_shadow_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            v50 = Path(tmp) / "v50.json"
            v50.write_text(json.dumps({"profile_template_shadow_patch_pass": True}), encoding="utf-8")
            record = build_execution_record(
                {
                    "online_llm_credentials_present": True,
                    "online_llm_accessible": True,
                    "provider_error_detected": False,
                },
                v50_readiness_json=v50,
            )

        self.assertEqual(record["scenarios"], list(FAILURE_SCENARIOS))
        self.assertEqual(record["controller"], "llm_rspc_v2")
        self.assertFalse(record["controlled_controller_included"])
        self.assertTrue(record["profile_template_patch_enabled"])
        self.assertTrue(record["fallback_post_selection_veto_enabled"])
        self.assertFalse(record["transition_gate_enabled"])
        self.assertFalse(AGENT_OVERRIDES["profile_feasibility_gate_enabled"])

    def test_readiness_never_allows_promotion_or_performance_claim(self):
        readiness = build_readiness(
            {"scenario_reports": [], "acceptance": {"v51_acceptance_pass": True}},
            execution_record={"executable": True, "online_llm_accessible": True},
        )

        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])

    def test_blocked_online_access_sets_blocked_next_action(self):
        readiness = build_readiness(
            {"scenario_reports": [], "acceptance": {"v51_acceptance_pass": False}},
            execution_record={
                "executable": False,
                "online_llm_accessible": False,
                "blocked_reason": "online_llm_accessibility_blocked",
            },
        )

        self.assertEqual(readiness["next_action"], "online_llm_accessibility_blocked")
        self.assertIn("online_llm_accessibility_blocked", readiness["stop_taxonomy"])


if __name__ == "__main__":
    unittest.main()
