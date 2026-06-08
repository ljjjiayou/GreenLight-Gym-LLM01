import unittest

from gl_gym.experiments.controlled_canary_shadow_sourced_admission_review import build_report, build_v31_report


class TestControlledCanaryShadowSourcedAdmissionReview(unittest.TestCase):
    def _base_inputs(self):
        return {
            "readiness_v26": {"shadow_trigger_source_sweep_pass": True},
            "shadow_replay": {"aggregate": {"replay_window_count": 2, "signal_steps": 5, "unsafe_preferred_steps": 0}},
            "controlled_shadow": {"aggregate": {"unsafe_preferred_steps": 0, "unsafe_applied_steps": 0}},
            "summary_audit": {"aggregate": {"decision": "pass"}},
            "runtime_provenance_audit": {
                "audit_status": "runtime_provenance_complete",
                "record_count": 12,
                "runtime_reason_missing_count": 0,
                "unknown_post_guardrail_rewrite_count": 0,
            },
            "joint_prediction_readiness": {
                "ready_for_policy_judgment": True,
                "row_count": 10,
                "missing_field_counts": {"a": 0},
            },
            "overlay_preflight": {
                "controlled_canary_overlay_validated": True,
                "controlled_canary_runner_surface_validated": True,
                "protocol_hash_mismatch_count": 0,
                "overlay_runner_path": "gl_gym/experiments/controlled_canary_shadow_sourced_admission_review.py",
                "protocol_v1_overlay_root": "gl_gym/result/audits/protocol_v1_controlled_canary_overlay_20260525",
            },
        }

    def test_admission_passes_on_complete_v26_evidence(self):
        report = build_report(**self._base_inputs())

        self.assertTrue(report["shadow_sourced_admission_pass"])
        self.assertEqual(report["blocker_taxonomy"], [])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_admission_blocks_unsafe_or_missing_v26(self):
        inputs = self._base_inputs()
        inputs["controlled_shadow"] = {"aggregate": {"unsafe_preferred_steps": 1, "unsafe_applied_steps": 0}}
        report = build_report(**inputs)

        self.assertFalse(report["shadow_sourced_admission_pass"])
        self.assertIn("unsafe_preferred_steps_zero", report["blocker_taxonomy"])
        self.assertFalse(report["promotion_evidence"])

    def test_v31_admission_passes_on_v30_strict_source_evidence(self):
        inputs = self._base_inputs()
        report = build_v31_report(
            readiness_v30={
                "v30_shadow_sweep_executed": True,
                "post_run_audits_pass": True,
                "runtime_provenance_complete": True,
                "joint_prediction_complete": True,
            },
            strict_sourcing={
                "strict_targeted_source_found": True,
                "expected_strict_eligible_applied_steps": 9,
                "would_apply_steps": 68,
                "strict_filtered_steps": 59,
                "sample_rows": [
                    {"strict_targeted_source": True},
                    {"strict_targeted_source": True},
                ],
            },
            summary_audit=inputs["summary_audit"],
            runtime_provenance_audit=inputs["runtime_provenance_audit"],
            joint_prediction_readiness=inputs["joint_prediction_readiness"],
            overlay_preflight=inputs["overlay_preflight"],
        )

        self.assertTrue(report["shadow_sourced_admission_pass"])
        self.assertEqual(report["metrics"]["expected_strict_eligible_applied_steps"], 9)
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_v31_admission_blocks_without_positive_strict_projection(self):
        inputs = self._base_inputs()
        report = build_v31_report(
            readiness_v30={
                "v30_shadow_sweep_executed": True,
                "post_run_audits_pass": True,
                "runtime_provenance_complete": True,
                "joint_prediction_complete": True,
            },
            strict_sourcing={
                "strict_targeted_source_found": False,
                "expected_strict_eligible_applied_steps": 0,
                "sample_rows": [],
            },
            summary_audit=inputs["summary_audit"],
            runtime_provenance_audit=inputs["runtime_provenance_audit"],
            joint_prediction_readiness=inputs["joint_prediction_readiness"],
            overlay_preflight=inputs["overlay_preflight"],
        )

        self.assertFalse(report["shadow_sourced_admission_pass"])
        self.assertIn("strict_targeted_source_found", report["blocker_taxonomy"])
        self.assertIn("expected_strict_eligible_applied_steps_positive", report["blocker_taxonomy"])


if __name__ == "__main__":
    unittest.main()
