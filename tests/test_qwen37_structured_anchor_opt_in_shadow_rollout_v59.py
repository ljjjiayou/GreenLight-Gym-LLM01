import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector
from gl_gym.experiments.qwen37_structured_anchor_opt_in_shadow_rollout_v59 import (
    AGENT_OVERRIDES,
    MODEL_NAME,
    _structured_anchor_cache_audit,
    build_execution_record,
    build_readiness,
)


class TestStructuredAnchorOptInV59(unittest.TestCase):
    def test_opt_in_prompt_uses_structured_anchor_not_set_all_controls(self):
        dummy = SimpleNamespace(config=AgentConfig(structured_anchor_parser_enabled=True), active_control_interval=12)
        state = SimpleNamespace(
            hour_of_day=12.0,
            day_of_year=180.0,
            temp_air=22.0,
            rh_air=74.0,
            co2_air=430.0,
            glob_rad=400.0,
            temp_out=18.0,
            rh_out=80.0,
            fruit_weight=0.1,
            canopy_temp_24h=21.0,
            temperature_sum=1000.0,
            u_boil=0.0,
            u_co2=0.0,
            u_th_scr=0.0,
            u_vent=0.2,
            u_lamp=0.0,
            u_bl_scr=0.0,
        )

        prompt = RuleBasedLLMDirector._build_compact_prompt(
            dummy,
            state,
            {"suggestions": ["watch dew"]},
            "test",
            status_brief="status",
            horizon=12,
        )

        self.assertIn("structured_planning_anchor", prompt)
        self.assertIn("Do not output low-level actuator", prompt)
        self.assertNotIn("set_all_controls", prompt)

    def test_execution_record_is_fixed_scope_and_no_controlled_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            readiness = Path(tmp) / "v58.json"
            readiness.write_text(
                json.dumps(
                    {
                        "structured_anchor_parser_ready": True,
                        "next_action": "structured_anchor_opt_in_qwen37_shadow_rollout_plan",
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
                v58_readiness_json=readiness,
            )

        self.assertTrue(record["executable"])
        self.assertEqual(record["controller"], "llm_rspc_v2")
        self.assertFalse(record["controlled_controller_included"])
        self.assertEqual(record["model_name"], "qwen3.7-max")
        self.assertEqual(len(record["scenarios"]), 3)
        self.assertTrue(record["structured_anchor_parser_enabled"])
        self.assertFalse(record["profile_template_patch_enabled"])
        self.assertIn("qwen37_structured_anchor_opt_in_shadow_v59_20260601.json", record["cache_path"])
        for command in record["commands"]:
            argv = command["argv"]
            self.assertIn("--llm-model", argv)
            self.assertIn("qwen3.7-max", argv)
            self.assertIn("--controllers", argv)
            self.assertIn("llm_rspc_v2", argv)
            self.assertNotIn("llm_rspc_v2_hot_dry_proposer_strict", argv)

    def test_agent_overrides_keep_shadow_path_isolated(self):
        self.assertTrue(AGENT_OVERRIDES["structured_anchor_parser_enabled"])
        self.assertTrue(AGENT_OVERRIDES["structured_anchor_shadow_only"])
        self.assertEqual(AGENT_OVERRIDES["max_tokens"], 2048)
        self.assertFalse(AGENT_OVERRIDES["transition_gate_enabled"])
        self.assertFalse(AGENT_OVERRIDES["profile_feasibility_gate_enabled"])
        self.assertFalse(AGENT_OVERRIDES["profile_template_patch_enabled"])
        self.assertFalse(AGENT_OVERRIDES["fallback_post_selection_veto_enabled"])
        self.assertFalse(AGENT_OVERRIDES["recovery_anchor_enabled"])

    def test_cache_audit_counts_valid_and_invalid_structured_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            cache.write_text(
                json.dumps(
                    {
                        "entries": {
                            "valid": {
                                "raw_response": "{}",
                                "structured_anchor": {
                                    "enabled": True,
                                    "valid": True,
                                    "clean_planning_evidence": True,
                                    "empty": False,
                                    "final_control_field_leak_count": 0,
                                    "shadow_plan": {"profile_intent": "hot_dry"},
                                },
                            },
                            "invalid": {
                                "raw_response": "{}",
                                "structured_anchor": {
                                    "enabled": True,
                                    "valid": False,
                                    "clean_planning_evidence": False,
                                    "empty": True,
                                    "errors": ["empty_anchor"],
                                    "final_control_field_leak_count": 0,
                                },
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            audit = _structured_anchor_cache_audit(cache)

        self.assertEqual(audit["structured_anchor_attempt_count"], 2)
        self.assertEqual(audit["valid_structured_anchor_count"], 1)
        self.assertEqual(audit["empty_structured_anchor_count"], 1)
        self.assertEqual(audit["invalid_anchor_clean_planning_evidence_count"], 0)

    def test_readiness_never_allows_replay_or_claims(self):
        readiness = build_readiness(
            {
                "acceptance": {
                    "v59_acceptance_pass": True,
                    "model_consistency_pass": True,
                },
                "structured_anchor_attempt_count": 3,
                "valid_structured_anchor_rate": 1.0,
                "final_control_field_leak_count": 0,
                "empty_structured_anchor_rate": 0.0,
                "provider_error_steps": 0,
            },
            {"executable": True, "online_llm_accessible": True},
        )

        self.assertEqual(readiness["next_action"], "structured_anchor_to_profile_generator_shadow_bridge_plan")
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
