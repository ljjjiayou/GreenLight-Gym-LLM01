import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.profile_template_opt_in_shadow_rollout_v52 import (
    MODEL_NAME,
    _model_consistency,
    build_execution_record,
    build_readiness,
)


class TestV52Qwen37ExecutionRecord(unittest.TestCase):
    def test_execution_record_is_fixed_scope_and_explicit_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            v50 = Path(tmp) / "v50.json"
            v50.write_text(json.dumps({"profile_template_shadow_patch_pass": True}), encoding="utf-8")
            record = build_execution_record(
                {
                    "model_name": MODEL_NAME,
                    "online_llm_credentials_present": True,
                    "online_llm_accessible": True,
                    "provider_error_detected": False,
                },
                v50_readiness_json=v50,
            )

        self.assertTrue(record["executable"])
        self.assertEqual(record["model_name"], "qwen3.7-max")
        self.assertEqual(record["controller"], "llm_rspc_v2")
        self.assertFalse(record["controlled_controller_included"])
        self.assertIn("profile_template_opt_in_shadow_v52_qwen37_20260601.json", record["cache_path"])
        self.assertTrue(record["profile_template_patch_enabled"])
        self.assertTrue(record["fallback_post_selection_veto_enabled"])
        self.assertFalse(record["transition_gate_enabled"])
        self.assertFalse(record["profile_feasibility_gate_enabled"])
        command_text = "\n".join(item["command"] for item in record["commands"])
        self.assertIn("--llm-model qwen3.7-max", command_text)
        self.assertNotIn("llm_rspc_v2_hot_dry_proposer_strict", command_text)

    def test_model_mismatch_blocks_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            v50 = Path(tmp) / "v50.json"
            v50.write_text(json.dumps({"profile_template_shadow_patch_pass": True}), encoding="utf-8")
            record = build_execution_record(
                {
                    "model_name": "qwen-max-latest",
                    "online_llm_credentials_present": True,
                    "online_llm_accessible": True,
                    "provider_error_detected": False,
                },
                v50_readiness_json=v50,
            )

        self.assertFalse(record["executable"])
        self.assertEqual(record["blocked_reason"], "model_name_mismatch")


class TestV52Qwen37ModelConsistency(unittest.TestCase):
    def test_model_consistency_passes_only_for_clean_qwen37_rows(self):
        report = _model_consistency(
            [
                {"model_name": MODEL_NAME, "source": "rspc_rollout"},
                {"model_name": MODEL_NAME, "source": "rule_anchor"},
            ]
        )

        self.assertTrue(report["model_consistency_pass"])
        self.assertEqual(report["model_name_values"], [MODEL_NAME])

    def test_model_consistency_fails_for_missing_provider_or_fallback_only(self):
        missing = _model_consistency([{"source": "rspc_rollout"}])
        provider = _model_consistency([{"model_name": MODEL_NAME, "runtime_error": "403 Access denied"}])
        fallback = _model_consistency(
            [
                {"model_name": MODEL_NAME, "source": "fallback_rule"},
                {"model_name": MODEL_NAME, "source": "fallback_anchor"},
            ]
        )

        self.assertFalse(missing["model_consistency_pass"])
        self.assertFalse(provider["model_consistency_pass"])
        self.assertFalse(fallback["model_consistency_pass"])

    def test_readiness_never_allows_replay_promotion_or_performance_claim(self):
        readiness = build_readiness(
            {
                "acceptance": {
                    "v52_acceptance_pass": True,
                    "all_v51_traces_present": True,
                    "qwen37_model_consistency_pass": True,
                }
            },
            execution_record={"executable": True, "online_llm_accessible": True},
        )

        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
