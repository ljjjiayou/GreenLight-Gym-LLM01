import unittest
from types import SimpleNamespace

import numpy as np

from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector
from gl_gym.experiments.diagnose_ppo_vs_llm import profile_feasibility_gate_to_record
from gl_gym.experiments.profile_feasibility_gate_shadow_rollout_v48 import (
    AGENT_OVERRIDES,
    FAILURE_SCENARIOS,
    V48_CACHE_PATH,
    build_execution_record,
    build_readiness,
)


def _director(**config_overrides):
    director = object.__new__(RuleBasedLLMDirector)
    director.config = AgentConfig(**config_overrides)
    director.last_tomato_safety_v2 = {"enabled": True, "applied": False}
    director.last_profile_feasibility_gate = {
        "enabled": bool(getattr(director.config, "profile_feasibility_gate_enabled", False)),
        "applied": False,
        "hard_safety_veto_count": 0,
    }
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


class TestProfileFeasibilityGateV48(unittest.TestCase):
    def test_default_disabled_returns_targets_unchanged(self):
        director = _director(profile_feasibility_gate_enabled=False)
        target = (24.0, 900.0, 80.0)
        repaired = director._apply_profile_feasibility_gate_targets(
            _state(rh_air=91.0, dew_margin_air=0.6, canopy_dew_margin=0.8),
            np.asarray([0.0, 0.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32),
            *target,
            source="test",
        )

        self.assertEqual(repaired[:3], target)
        self.assertFalse(repaired[3]["enabled"])
        self.assertFalse(repaired[3]["applied"])

    def test_opt_in_repairs_current_step_without_plan_mutation(self):
        director = _director(profile_feasibility_gate_enabled=True)
        plan = {"target_temp": 24.0, "target_co2": 900.0, "target_rh": 80.0}
        repaired_temp, repaired_co2, repaired_rh, record = director._apply_profile_feasibility_gate_targets(
            _state(rh_air=91.0, dew_margin_air=0.6, canopy_dew_margin=0.8),
            np.asarray([0.0, 0.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32),
            plan["target_temp"],
            plan["target_co2"],
            plan["target_rh"],
            source="test",
        )

        self.assertTrue(record["enabled"])
        self.assertTrue(record["applied"])
        self.assertLess(repaired_rh, plan["target_rh"])
        self.assertEqual(repaired_co2, 430.0)
        self.assertEqual(plan["target_rh"], 80.0)
        self.assertTrue(record["mode"]["vent_required_by_safety"])

    def test_candidate_adjustment_does_not_penalize_safety_required_vent_as_dry_conflict(self):
        director = _director(profile_feasibility_gate_enabled=True)
        score, details = director._profile_feasibility_candidate_adjustment(
            _state(rh_air=48.0, temp_air=33.0),
            np.asarray([0.0, 0.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32),
            {"target_temp": 20.0, "target_co2": 430.0, "target_rh": 70.0},
        )

        self.assertGreaterEqual(score, 0.0)
        self.assertTrue(details["gate_record"]["mode"]["vent_required_by_safety"])
        self.assertEqual(details["dry_risk_high_vent_conflict"], 0)

    def test_candidate_adjustment_penalizes_non_safety_dry_high_vent(self):
        director = _director(profile_feasibility_gate_enabled=True)
        score, details = director._profile_feasibility_candidate_adjustment(
            _state(rh_air=45.0, temp_air=25.0, dew_margin_air=3.0, canopy_dew_margin=3.0),
            np.asarray([0.0, 0.0, 0.0, 0.9, 0.0, 0.0], dtype=np.float32),
            {"target_temp": 20.0, "target_co2": 430.0, "target_rh": 70.0},
        )

        self.assertGreater(score, 0.0)
        self.assertFalse(details["gate_record"]["mode"]["vent_required_by_safety"])
        self.assertEqual(details["dry_risk_high_vent_conflict"], 1)

    def test_trace_record_flattens_profile_gate_without_secret_values(self):
        record = profile_feasibility_gate_to_record(
            {
                "enabled": True,
                "applied": True,
                "corrections": ["target_rh:safety_cap"],
                "source": "plan_info_current_targets",
                "mode": {"vent_required_by_safety": True, "dry_side": False},
                "original_targets": {"target_temp": 24.0, "target_co2": 900.0, "target_rh": 80.0},
                "repaired_targets": {"target_temp": 24.0, "target_co2": 430.0, "target_rh": 72.0},
                "hard_safety_veto_count": 2,
            }
        )

        self.assertTrue(record["profile_feasibility_gate_enabled"])
        self.assertTrue(record["profile_feasibility_gate_applied"])
        self.assertEqual(record["profile_feasibility_gate_hard_safety_veto_count"], 2)
        self.assertEqual(record["profile_feasibility_gate_repaired_target_co2"], 430.0)

    def test_execution_record_scope_is_fixed_and_blocks_controlled_replay(self):
        record = build_execution_record()

        self.assertEqual(record["scenarios"], list(FAILURE_SCENARIOS))
        self.assertEqual(record["controller"], "llm_rspc_v2")
        self.assertFalse(record["controlled_controller_included"])
        self.assertFalse(record["controlled_replay_allowed"])
        self.assertFalse(record["performance_claim_allowed"])
        self.assertTrue(record["cache_path"].replace("\\", "/").endswith("gl_gym/result/plan_cache/profile_feasibility_gate_shadow_v48_20260531.json"))
        self.assertEqual(AGENT_OVERRIDES["transition_gate_enabled"], False)

    def test_readiness_never_allows_promotion_or_performance_claim(self):
        readiness = build_readiness({"scenario_reports": [], "acceptance": {"v48_acceptance_pass": True}})

        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
