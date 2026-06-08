import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gl_gym.agent.llm_agent import AgentConfig
from gl_gym.agent.structured_anchor_profile_bridge import (
    build_structured_anchor_profile_bridge_shadow_payload,
)
from gl_gym.experiments import qwen37plus_structured_anchor_profile_bridge_runtime_v64 as v64
from gl_gym.experiments.diagnose_ppo_vs_llm import structured_anchor_profile_bridge_to_record


def _state(**overrides):
    data = {
        "timestep": 12,
        "hour_of_day": 12.0,
        "temp_air": 25.0,
        "rh_air": 62.0,
        "co2_air": 430.0,
        "glob_rad": 500.0,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _structured_anchor(**overrides):
    shadow = {
        "profile_intent": "hot dry radiation relief",
        "target_temp": 23.0,
        "target_co2": 430.0,
        "target_rh": 68.0,
        "risk_flags": ["dry", "high_vpd"],
        "forbidden_intents": [],
        "planning_horizon_steps": 12,
        "confidence": 0.85,
    }
    shadow.update(overrides)
    return {
        "enabled": True,
        "attempted": True,
        "valid": True,
        "clean_planning_evidence": True,
        "shadow_plan": shadow,
    }


class TestQwen37PlusStructuredAnchorProfileBridgeRuntimeV64(unittest.TestCase):
    def test_agent_config_defaults_keep_runtime_bridge_disabled(self):
        cfg = AgentConfig()

        self.assertFalse(cfg.structured_anchor_profile_bridge_enabled)
        self.assertTrue(cfg.structured_anchor_profile_bridge_shadow_only)
        self.assertTrue(cfg.structured_anchor_profile_bridge_record_provenance)
        self.assertEqual(structured_anchor_profile_bridge_to_record({}), {})

    def test_valid_anchor_generates_shadow_profile_without_control_action(self):
        bridge, candidates = build_structured_anchor_profile_bridge_shadow_payload(
            _structured_anchor(),
            _state(),
            current_plan={"target_temp": 24.0, "target_co2": 430.0, "target_rh": 65.0},
            model_name="qwen3.7-plus",
        )

        self.assertTrue(bridge["bridgeable"])
        self.assertTrue(bridge["applied"])
        self.assertGreater(bridge["profile_candidate_count"], 0)
        self.assertGreater(len(candidates), 0)
        self.assertFalse(bridge["final_control_change"])
        self.assertFalse(bridge["current_plan_modified"])
        self.assertFalse(bridge["low_level_action_generated"])

    def test_invalid_anchor_does_not_bridge(self):
        bridge, candidates = build_structured_anchor_profile_bridge_shadow_payload(
            {"attempted": True, "valid": False, "shadow_plan": {}},
            _state(),
            model_name="qwen3.7-plus",
        )

        self.assertFalse(bridge["bridgeable"])
        self.assertEqual(bridge["failure_reason"], "valid_structured_anchor_missing")
        self.assertEqual(candidates, [])

    def test_execution_record_fixed_scope_and_runtime_bridge_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            readiness = Path(tmp) / "v63.json"
            readiness.write_text(
                json.dumps(
                    {
                        "model_name": "qwen3.7-plus",
                        "structured_anchor_prompt_retry_shadow_rollout_pass": True,
                        "next_action": "structured_anchor_to_profile_generator_runtime_shadow_bridge_plan",
                    }
                ),
                encoding="utf-8",
            )
            record = v64.build_execution_record(
                {
                    "model_name": "qwen3.7-plus",
                    "online_llm_credentials_present": True,
                    "online_llm_accessible": True,
                },
                v63_readiness_json=readiness,
            )

        self.assertTrue(record["executable"])
        self.assertEqual(record["controller"], "llm_rspc_v2")
        self.assertFalse(record["controlled_controller_included"])
        self.assertEqual(len(record["scenarios"]), 3)
        self.assertTrue(record["agent_config_overrides"]["structured_anchor_profile_bridge_enabled"])
        self.assertTrue(record["agent_config_overrides"]["structured_anchor_profile_bridge_shadow_only"])
        self.assertFalse(record["agent_config_overrides"]["transition_gate_enabled"])
        self.assertIn("qwen37plus_structured_anchor_profile_bridge_shadow_v64_20260602.json", record["cache_path"])
        for item in record["commands"]:
            argv = item["argv"]
            self.assertIn("--llm-model", argv)
            self.assertIn("qwen3.7-plus", argv)
            self.assertNotIn("llm_rspc_v2_hot_dry_proposer_strict", argv)

    def test_runtime_bridge_audit_reads_cache_bridge_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            cache.write_text(
                json.dumps(
                    {
                        "entries": {
                            "k": {
                                "env_id": "TomatoEnv_y2010_d180_s43",
                                "timestep": 12,
                                "model_name": "qwen3.7-plus",
                                "structured_anchor": _structured_anchor(),
                                "parsed_plan": {
                                    "structured_anchor_profile_bridge": {
                                        "enabled": True,
                                        "shadow_only": True,
                                        "attempted": True,
                                        "valid_structured_anchor": True,
                                        "bridgeable": True,
                                        "applied": True,
                                        "profile_candidate_count": 2,
                                        "selected_shadow_profile_name": "hot_dry_protect",
                                        "hard_safety_profile_violation": False,
                                        "final_control_change": False,
                                        "current_plan_modified": False,
                                        "low_level_action_generated": False,
                                    }
                                },
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            audit = v64.build_runtime_bridge_audit(cache)

        self.assertTrue(audit["runtime_bridge_contract_pass"])
        self.assertEqual(audit["valid_structured_anchor_count"], 1)
        self.assertEqual(audit["runtime_bridge_attempt_rate"], 1.0)
        self.assertEqual(audit["bridgeable_rate"], 1.0)
        self.assertEqual(audit["profile_candidate_count_min"], 2)

    def test_readiness_separates_bridge_contract_from_runtime_stability(self):
        readiness = v64.build_readiness(
            {
                "runtime_bridge_contract_pass": True,
                "runtime_stability_pass": False,
                "model_consistency_pass": True,
                "provider_error_steps": 0,
                "structured_anchor_attempt_count": 10,
                "valid_structured_anchor_rate": 1.0,
                "runtime_bridge_attempt_rate": 1.0,
                "bridgeable_rate": 1.0,
                "profile_candidate_generation_error_count": 0,
                "profile_candidate_count_min": 1,
                "hard_safety_profile_violation_count": 0,
                "runtime_error_steps": 1,
                "short_trace_scenario_count": 1,
                "acceptance": {
                    "model_consistency_pass": True,
                    "provider_error_steps_zero": True,
                    "valid_structured_anchor_rate_ge_0_95": True,
                    "runtime_bridge_attempt_rate_ge_0_95": True,
                    "bridgeable_rate_ge_0_95": True,
                    "profile_candidate_generation_error_count_zero": True,
                    "profile_candidate_count_positive": True,
                    "hard_safety_profile_violation_count_zero": True,
                    "final_control_change_count_zero": True,
                    "current_plan_modified_count_zero": True,
                    "low_level_action_generated_count_zero": True,
                    "runtime_stability_pass": False,
                },
            },
            {"executable": True},
        )

        self.assertEqual(readiness["next_action"], "runtime_failure_attribution_after_structured_bridge")
        self.assertIn("runtime_or_short_trace_persists", readiness["stop_taxonomy"])
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
