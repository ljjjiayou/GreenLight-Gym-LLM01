import unittest

from gl_gym.experiments.minimal_controlled_canary_result_review import build_report


class TestMinimalControlledCanaryResultReview(unittest.TestCase):
    def test_strict_not_triggered_routes_to_discovery(self):
        report = build_report(
            readiness_v21={
                "minimal_controlled_canary_pass": True,
                "post_run_metrics": {
                    "controlled_canary_runtime_error_steps": 0,
                    "controlled_canary_strict_cache_miss_runtime_error_steps": 0,
                },
            },
            summary_audit={"aggregate": {"decision": "pass"}},
            trace_audit={
                "aggregate": {
                    "would_apply_steps": 6,
                    "applied_steps": 0,
                    "strict_applied_steps": 0,
                    "strict_filtered_steps": 3,
                    "unsafe_preferred_steps": 0,
                    "unsafe_conflict_steps": 0,
                    "unsafe_applied_steps": 0,
                }
            },
            strict_effect_audit={
                "aggregate": {
                    "runtime_error_steps": 0,
                    "unsafe_applied_steps": 0,
                    "warnings": [],
                    "recommendation": {"decision": "strict_not_triggered"},
                }
            },
            runtime_provenance_audit={"audit_status": "runtime_provenance_complete", "record_count": 10},
            joint_prediction_readiness={"ready_for_policy_judgment": True, "missing_field_counts": {}},
        )

        self.assertTrue(report["canary_safety_pass"])
        self.assertFalse(report["control_effect_observed"])
        self.assertTrue(report["strict_not_triggered"])
        self.assertEqual(report["next_action"], "strict_trigger_discovery_required")
        self.assertFalse(report["performance_claim_allowed"])

    def test_safety_failure_blocks_trigger_discovery(self):
        report = build_report(
            readiness_v21={"minimal_controlled_canary_pass": False, "post_run_metrics": {}},
            summary_audit={"aggregate": {"decision": "pass"}},
            trace_audit={"aggregate": {"unsafe_applied_steps": 1}},
            strict_effect_audit={"aggregate": {"recommendation": {"decision": "strict_not_triggered"}}},
            runtime_provenance_audit={"record_count": 0},
            joint_prediction_readiness={"ready_for_policy_judgment": False, "missing_field_counts": {"x": 1}},
        )

        self.assertFalse(report["canary_safety_pass"])
        self.assertEqual(report["next_action"], "minimal_controlled_canary_failure_diagnosis_required")


if __name__ == "__main__":
    unittest.main()
