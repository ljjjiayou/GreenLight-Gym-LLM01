import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.controlled_canary_expansion_admission_review import build_report


def valid_inputs(root: Path):
    runner = root / "gl_gym" / "experiments" / "run_frozen_benchmark.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("llm_rspc_v2_hot_dry_proposer_strict\n", encoding="utf-8")
    return {
        "consolidation": {"expanded_metadata_consolidation_pass": True},
        "readiness_v22": {
            "triggered_controlled_canary_pass": True,
            "promotion_evidence": False,
            "performance_claim_allowed": False,
        },
        "protocol": {
            "protocol_v1_overlay_root": str(root),
            "overlay_runner_path": str(runner),
            "controlled_canary_overlay_validated": True,
            "protocol_v1_overlay_validated": True,
            "protocol_hash_mismatch_count": 0,
            "controlled_canary_runner_surface_validated": True,
        },
        "summary": {"aggregate": {"decision": "pass"}},
        "trace": {
            "aggregate": {
                "strict_applied_steps": 1,
                "unsafe_preferred_steps": 0,
                "unsafe_conflict_steps": 0,
                "unsafe_applied_steps": 0,
            }
        },
        "effect": {"aggregate": {"runtime_error_steps": 0, "unsafe_applied_steps": 0}},
        "runtime": {
            "audit_status": "runtime_provenance_complete",
            "runtime_reason_missing_count": 0,
            "unknown_post_guardrail_rewrite_count": 0,
        },
        "joint": {"ready_for_policy_judgment": True},
    }


class TestControlledCanaryExpansionAdmissionReview(unittest.TestCase):
    def test_admission_allows_wave1_when_current_canary_evidence_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = valid_inputs(Path(tmp))
            report = build_report(
                consolidation=inputs["consolidation"],
                readiness_v22=inputs["readiness_v22"],
                protocol_v1_overlay_preflight=inputs["protocol"],
                triggered_summary=inputs["summary"],
                triggered_trace=inputs["trace"],
                triggered_effect=inputs["effect"],
                triggered_runtime=inputs["runtime"],
                triggered_joint=inputs["joint"],
            )

        self.assertTrue(report["wave1_controlled_canary_allowed"])
        self.assertEqual(report["next_action"], "generate_controlled_canary_expansion_wave1_manifest")
        self.assertFalse(report["performance_claim_allowed"])

    def test_admission_blocks_without_v22_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = valid_inputs(Path(tmp))
            inputs["readiness_v22"]["triggered_controlled_canary_pass"] = False
            report = build_report(
                consolidation=inputs["consolidation"],
                readiness_v22=inputs["readiness_v22"],
                protocol_v1_overlay_preflight=inputs["protocol"],
                triggered_summary=inputs["summary"],
                triggered_trace=inputs["trace"],
                triggered_effect=inputs["effect"],
                triggered_runtime=inputs["runtime"],
                triggered_joint=inputs["joint"],
            )

        self.assertFalse(report["wave1_controlled_canary_allowed"])
        self.assertIn("readiness_v22_triggered_canary_pass", report["blockers"])

    def test_overlay_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = valid_inputs(Path(tmp))
            Path(inputs["protocol"]["overlay_runner_path"]).write_text("llm_rspc_v2\n", encoding="utf-8")
            report = build_report(
                consolidation=inputs["consolidation"],
                readiness_v22=inputs["readiness_v22"],
                protocol_v1_overlay_preflight=inputs["protocol"],
                triggered_summary=inputs["summary"],
                triggered_trace=inputs["trace"],
                triggered_effect=inputs["effect"],
                triggered_runtime=inputs["runtime"],
                triggered_joint=inputs["joint"],
            )

        self.assertFalse(report["wave1_controlled_canary_allowed"])
        self.assertIn("overlay_runner_supports_candidate_controller", report["blockers"])


if __name__ == "__main__":
    unittest.main()
