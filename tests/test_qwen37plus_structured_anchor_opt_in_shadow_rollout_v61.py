import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments import qwen37plus_structured_anchor_opt_in_shadow_rollout_v61 as v61


class TestQwen37PlusStructuredAnchorOptInShadowRolloutV61(unittest.TestCase):
    def test_execution_record_fixed_scope_and_qwen37plus_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            readiness = Path(tmp) / "v60.json"
            readiness.write_text(
                json.dumps(
                    {
                        "model_name": "qwen3.7-plus",
                        "qwen37plus_precheck_pass": True,
                        "bridge_pass": True,
                        "next_action": "qwen37plus_structured_anchor_opt_in_shadow_rollout_plan",
                    }
                ),
                encoding="utf-8",
            )
            record = v61.build_execution_record(
                {
                    "model_name": "qwen3.7-plus",
                    "online_llm_credentials_present": True,
                    "online_llm_accessible": True,
                    "provider_error_detected": False,
                },
                v60_readiness_json=readiness,
            )

        self.assertTrue(record["executable"])
        self.assertEqual(record["model_name"], "qwen3.7-plus")
        self.assertEqual(record["controller"], "llm_rspc_v2")
        self.assertFalse(record["controlled_controller_included"])
        self.assertEqual(len(record["scenarios"]), 3)
        self.assertIn("qwen37plus_structured_anchor_shadow_v61_20260602.json", record["cache_path"])
        self.assertIn("qwen37plus_structured_anchor_shadow_rollout_v61_20260602", record["output_dir"])
        self.assertNotIn("qwen3.7-max", json.dumps(record, sort_keys=True))
        for item in record["commands"]:
            argv = item["argv"]
            self.assertIn("--llm-model", argv)
            self.assertIn("qwen3.7-plus", argv)
            self.assertIn("--controllers", argv)
            self.assertIn("llm_rspc_v2", argv)
            self.assertNotIn("llm_rspc_v2_hot_dry_proposer_strict", argv)

    def test_execution_record_blocks_when_v60_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            readiness = Path(tmp) / "v60.json"
            readiness.write_text(
                json.dumps(
                    {
                        "model_name": "qwen3.7-plus",
                        "qwen37plus_precheck_pass": False,
                        "bridge_pass": True,
                        "next_action": "online_llm_accessibility_blocked",
                    }
                ),
                encoding="utf-8",
            )
            record = v61.build_execution_record(
                {
                    "model_name": "qwen3.7-plus",
                    "online_llm_credentials_present": True,
                    "online_llm_accessible": True,
                },
                v60_readiness_json=readiness,
            )

        self.assertFalse(record["executable"])
        self.assertEqual(record["blocked_reason"], "v60_qwen37plus_model_switch_or_bridge_not_ready")

    def test_agent_overrides_keep_shadow_path_isolated(self):
        self.assertTrue(v61.AGENT_OVERRIDES["structured_anchor_parser_enabled"])
        self.assertTrue(v61.AGENT_OVERRIDES["structured_anchor_shadow_only"])
        self.assertFalse(v61.AGENT_OVERRIDES["transition_gate_enabled"])
        self.assertFalse(v61.AGENT_OVERRIDES["profile_feasibility_gate_enabled"])
        self.assertFalse(v61.AGENT_OVERRIDES["profile_template_patch_enabled"])
        self.assertFalse(v61.AGENT_OVERRIDES["fallback_post_selection_veto_enabled"])
        self.assertFalse(v61.AGENT_OVERRIDES["recovery_anchor_enabled"])

    def test_cache_audit_counts_contract_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            cache.write_text(
                json.dumps(
                    {
                        "entries": {
                            "valid": {
                                "model_name": "qwen3.7-plus",
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
                                "model_name": "qwen3.7-plus",
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

            audit = v61._structured_anchor_cache_audit(cache)

        self.assertEqual(audit["model_name"], "qwen3.7-plus")
        self.assertEqual(audit["structured_anchor_attempt_count"], 2)
        self.assertEqual(audit["valid_structured_anchor_count"], 1)
        self.assertEqual(audit["empty_structured_anchor_count"], 1)
        self.assertEqual(audit["final_control_field_leak_count"], 0)

    def test_readiness_never_allows_replay_or_claims(self):
        result = {
            "acceptance": {
                "v61_acceptance_pass": True,
                "model_consistency_pass": True,
            },
            "structured_anchor_attempt_count": 90,
            "valid_structured_anchor_rate": 1.0,
            "final_control_field_leak_count": 0,
            "empty_structured_anchor_rate": 0.0,
            "provider_error_steps": 0,
            "bridge_input_coverage_rate": 1.0,
            "profile_candidate_generation_error_count": 0,
            "profile_candidate_count_min": 1,
            "hard_safety_profile_violation_count": 0,
        }
        readiness = v61.build_readiness(result, {"executable": True, "online_llm_accessible": True})

        self.assertEqual(readiness["next_action"], "qwen37plus_structured_anchor_to_profile_generator_shadow_bridge_plan")
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])

    def test_stop_taxonomy_covers_bridge_failures(self):
        taxonomy = v61._stop_taxonomy(
            {
                "acceptance": {
                    "model_consistency_pass": True,
                    "provider_error_steps_zero": True,
                    "structured_anchor_attempt_count_positive": True,
                    "valid_structured_anchor_rate_ge_0_80": True,
                    "final_control_field_leak_count_zero": True,
                    "empty_structured_anchor_rate_le_0_10": True,
                    "invalid_anchor_not_clean_evidence": True,
                    "bridge_input_coverage_ge_0_90": False,
                    "profile_candidate_generation_error_count_zero": False,
                    "hard_safety_profile_violation_count_zero": False,
                }
            }
        )

        self.assertIn("bridge_input_coverage_low", taxonomy)
        self.assertIn("profile_candidate_generation_error", taxonomy)
        self.assertIn("hard_safety_profile_violation", taxonomy)


if __name__ == "__main__":
    unittest.main()
