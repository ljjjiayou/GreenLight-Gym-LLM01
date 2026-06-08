import unittest

from gl_gym.experiments.controlled_canary_shadow_sourced_readiness import build_report


class TestControlledCanaryShadowSourcedReadiness(unittest.TestCase):
    def test_readiness_stops_on_strict_projection_blocker(self):
        report = build_report(
            admission_review={"shadow_sourced_admission_pass": True},
            manifest={
                "shadow_sourced_manifest_ready": False,
                "blocker_taxonomy": ["no_shadow_sourced_strict_trigger_candidate"],
                "strict_projection": {"expected_strict_eligible_applied_steps": 0},
            },
        )

        self.assertEqual(report["next_action"], "shadow_sourced_strict_projection_blocker_diagnosis")
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["controlled_replay_execution_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertIn("no_shadow_sourced_strict_trigger_candidate", report["failure_taxonomy"])

    def test_readiness_requires_positive_strict_applied_after_execution(self):
        report = build_report(
            admission_review={"shadow_sourced_admission_pass": True},
            manifest={
                "shadow_sourced_manifest_ready": True,
                "strict_projection": {"expected_strict_eligible_applied_steps": 1},
            },
            cache_coverage={
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
            },
            execution_record={"shadow_sourced_controlled_canary_authorized": True},
            summary_audit={"aggregate": {"decision": "pass", "runtime_error_steps": 0, "strict_cache_miss_runtime_error_steps": 0}},
            trace_audit={"aggregate": {"strict_applied_steps": 0, "unsafe_preferred_steps": 0, "unsafe_applied_steps": 0}},
            strict_effect_audit={"aggregate": {"strict_applied_steps": 0}},
            runtime_provenance_audit={"audit_status": "runtime_provenance_complete", "record_count": 1},
            joint_prediction_readiness={"ready_for_policy_judgment": True, "row_count": 1},
        )

        self.assertFalse(report["shadow_sourced_controlled_canary_pass"])
        self.assertIn("strict_not_triggered", report["failure_taxonomy"])
        self.assertFalse(report["promotion_evidence"])

    def test_v31_readiness_never_promotes_even_when_canary_passes(self):
        report = build_report(
            admission_review={"shadow_sourced_admission_pass": True},
            manifest={
                "shadow_sourced_manifest_ready": True,
                "canary_scope": "strict_source_controlled_canary_v31_only",
                "readiness_schema_version": "metadata_replay_readiness_checklist_v31",
                "strict_projection": {"expected_strict_eligible_applied_steps": 9},
            },
            cache_coverage={
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
            },
            execution_record={"shadow_sourced_controlled_canary_authorized": True},
            summary_audit={"aggregate": {"decision": "pass", "runtime_error_steps": 0, "strict_cache_miss_runtime_error_steps": 0}},
            trace_audit={
                "aggregate": {
                    "strict_applied_steps": 9,
                    "unsafe_preferred_steps": 0,
                    "unsafe_conflict_steps": 0,
                    "unsafe_applied_steps": 0,
                }
            },
            strict_effect_audit={"aggregate": {"strict_applied_steps": 9}},
            runtime_provenance_audit={"audit_status": "runtime_provenance_complete", "record_count": 12},
            joint_prediction_readiness={"ready_for_policy_judgment": True, "row_count": 12},
        )

        self.assertEqual(report["schema_version"], "metadata_replay_readiness_checklist_v31")
        self.assertTrue(report["shadow_sourced_controlled_canary_pass"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["controlled_replay_execution_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
