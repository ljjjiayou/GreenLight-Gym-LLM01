import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector
from gl_gym.experiments import qwen37plus_structured_anchor_prompt_retry_shadow_rollout_v63 as v63


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        day_of_year=180.0,
        hour_of_day=12.0,
        temp_air=26.0,
        rh_air=62.0,
        co2_air=430.0,
        glob_rad=500.0,
        temp_out=22.0,
        rh_out=70.0,
        fruit_weight=0.001,
        canopy_temp_24h=22.0,
        temperature_sum=1200.0,
        u_boil=0.0,
        u_co2=0.0,
        u_th_scr=0.1,
        u_vent=0.2,
        u_lamp=0.0,
        u_bl_scr=0.0,
    )


def _cache_entry(*, env_id: str, timestep: int, attempt: int, valid: bool, raw: str, errors=None, empty=False):
    return {
        "env_id": env_id,
        "timestep": timestep,
        "model_name": "qwen3.7-plus",
        "raw_response": raw,
        "structured_anchor": {
            "attempt": attempt,
            "attempted": True,
            "valid": valid,
            "clean_planning_evidence": valid,
            "empty": empty,
            "errors": list(errors or []),
            "final_control_field_leak_count": 0,
            "retry_provenance": {
                "retry_attempt": attempt > 1,
                "retry_source_errors": ["invalid_json_anchor"] if attempt > 1 else [],
            },
            "shadow_plan": (
                {
                    "profile_intent": "hot_dry_relief",
                    "target_temp": 24.0,
                    "target_co2": 430.0,
                    "target_rh": 65.0,
                    "risk_flags": ["dry"],
                    "forbidden_intents": [],
                    "planning_horizon_steps": 12,
                    "confidence": 0.8,
                }
                if valid
                else {}
            ),
        },
    }


class TestQwen37PlusStructuredAnchorPromptRetryShadowRolloutV63(unittest.TestCase):
    def test_agent_config_defaults_keep_compact_retry_disabled(self):
        cfg = AgentConfig()

        self.assertFalse(cfg.structured_anchor_compact_json_prompt_enabled)
        self.assertFalse(cfg.structured_anchor_retry_invalid_or_empty_enabled)
        self.assertEqual(cfg.structured_anchor_retry_max_attempts, 1)
        self.assertEqual(cfg.structured_anchor_list_max_items, 3)

    def test_compact_prompt_is_json_only_contract_and_avoids_tool_anchor(self):
        director = RuleBasedLLMDirector.__new__(RuleBasedLLMDirector)
        director.config = AgentConfig(
            structured_anchor_parser_enabled=True,
            structured_anchor_compact_json_prompt_enabled=True,
            structured_anchor_list_max_items=3,
        )
        director.active_control_interval = 12

        prompt = director._build_compact_prompt(
            _state(),
            {"suggestions": ["hot dry risk", "avoid co2 vent conflict"]},
            "test",
            status_brief="status ok",
            horizon=12,
        )

        self.assertIn('"structured_planning_anchor"', prompt)
        self.assertIn("Return exactly one minified JSON object", prompt)
        self.assertNotIn("set_all_controls", prompt)
        self.assertNotIn("final_control", prompt)
        self.assertIn("risk_flags max 3", prompt)

    def test_parse_structured_anchor_records_retry_provenance(self):
        director = RuleBasedLLMDirector.__new__(RuleBasedLLMDirector)
        director.config = AgentConfig(
            structured_anchor_parser_enabled=True,
            structured_anchor_retry_invalid_or_empty_enabled=True,
            structured_anchor_compact_json_prompt_enabled=True,
        )

        record = director._parse_structured_anchor_output(
            '{"structured_planning_anchor":',
            attempt=2,
            prompt_hash="abc",
            planning_horizon=12,
            legacy_tool_action_present=False,
            retry_attempt=True,
            retry_source_errors=["invalid_json_anchor"],
        )

        self.assertFalse(record["valid"])
        self.assertFalse(record["clean_planning_evidence"])
        self.assertTrue(record["retry_provenance"]["retry_attempt"])
        self.assertEqual(record["retry_provenance"]["retry_source_errors"], ["invalid_json_anchor"])

    def test_execution_record_fixed_scope_and_retry_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            readiness = Path(tmp) / "v62.json"
            readiness.write_text(
                json.dumps(
                    {
                        "model_name": "qwen3.7-plus",
                        "repair_design_ready": True,
                        "next_action": "qwen37plus_structured_anchor_prompt_retry_opt_in_shadow_rollout_plan",
                    }
                ),
                encoding="utf-8",
            )
            record = v63.build_execution_record(
                {
                    "model_name": "qwen3.7-plus",
                    "online_llm_credentials_present": True,
                    "online_llm_accessible": True,
                },
                v62_readiness_json=readiness,
            )

        self.assertTrue(record["executable"])
        self.assertEqual(record["model_name"], "qwen3.7-plus")
        self.assertEqual(record["controller"], "llm_rspc_v2")
        self.assertFalse(record["controlled_controller_included"])
        self.assertIn("qwen37plus_structured_anchor_prompt_retry_shadow_v63_20260602.json", record["cache_path"])
        self.assertTrue(record["agent_config_overrides"]["structured_anchor_compact_json_prompt_enabled"])
        self.assertTrue(record["agent_config_overrides"]["structured_anchor_retry_invalid_or_empty_enabled"])
        for item in record["commands"]:
            argv = item["argv"]
            self.assertIn("--llm-model", argv)
            self.assertIn("qwen3.7-plus", argv)
            self.assertIn("--llm-max-iterations", argv)
            self.assertIn("2", argv)
            self.assertNotIn("llm_rspc_v2_hot_dry_proposer_strict", argv)

    def test_retry_outcome_counts_final_event_success_not_failed_first_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            cache.write_text(
                json.dumps(
                    {
                        "entries": {
                            "attempt1": _cache_entry(
                                env_id="TomatoEnv_y2010_d180_s43",
                                timestep=10,
                                attempt=1,
                                valid=False,
                                raw='{"structured_planning_anchor":',
                                errors=["invalid_json_anchor"],
                            ),
                            "attempt2": _cache_entry(
                                env_id="TomatoEnv_y2010_d180_s43",
                                timestep=10,
                                attempt=2,
                                valid=True,
                                raw='{"structured_planning_anchor":{"profile_intent":"hot_dry_relief","target_temp":24,"target_co2":430,"target_rh":65,"risk_flags":["dry"],"forbidden_intents":[],"planning_horizon_steps":12,"confidence":0.8}}',
                            ),
                        }
                    }
                ),
                encoding="utf-8",
            )

            audit = v63.build_retry_outcome_audit(cache)

        self.assertEqual(audit["planning_event_count"], 1)
        self.assertEqual(audit["structured_anchor_attempt_count"], 2)
        self.assertEqual(audit["retry_success_count"], 1)
        self.assertEqual(audit["valid_structured_anchor_rate"], 1.0)
        self.assertEqual(audit["invalid_json_truncated_count"], 0)
        self.assertEqual(audit["attempt_failure_type_counts"]["invalid_json_truncated"], 1)

    def test_readiness_never_allows_replay_or_claims(self):
        readiness = v63.build_readiness(
            {
                "acceptance": {"v63_acceptance_pass": True},
                "structured_anchor_attempt_count": 10,
                "model_consistency_pass": True,
                "provider_error_steps": 0,
                "valid_structured_anchor_rate": 1.0,
                "empty_structured_anchor_rate": 0.0,
                "invalid_json_truncated_count": 0,
                "bridge_input_coverage_rate": 1.0,
                "hard_safety_profile_violation_count": 0,
            },
            {"executable": True},
        )

        self.assertEqual(readiness["next_action"], "structured_anchor_to_profile_generator_runtime_shadow_bridge_plan")
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
