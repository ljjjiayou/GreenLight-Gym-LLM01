import unittest

from gl_gym.experiments.controlled_canary_expansion_wave1_result_review import build_report


class TestControlledCanaryExpansionWave1ResultReview(unittest.TestCase):
    def test_wave1_no_trigger_pass_requests_independent_discovery(self):
        report = build_report(
            readiness_v23={
                "wave1_controlled_canary_pass": True,
                "control_effect_observed": False,
                "effect_decision": "expansion_no_trigger",
                "performance_claim_allowed": False,
                "promotion_evidence": False,
                "post_run_metrics": {"scenario_count": 5},
            },
            wave1_manifest={"scenario_ids": ["y2020_d120_s44_n240"]},
            triggered_manifest={"scenario_ids": ["y2015_d120_s42_n240"]},
            summary_audit={"aggregate": {"decision": "pass"}},
            trace_audit={
                "aggregate": {
                    "strict_applied_steps": 0,
                    "would_apply_steps": 4,
                    "strict_filtered_steps": 2,
                    "unsafe_preferred_steps": 0,
                    "unsafe_conflict_steps": 0,
                    "unsafe_applied_steps": 0,
                }
            },
            strict_effect_audit={"aggregate": {"runtime_error_steps": 0, "unsafe_applied_steps": 0}},
            runtime_provenance_audit={"audit_status": "runtime_provenance_complete", "record_count": 10},
            joint_prediction_readiness={"ready_for_policy_judgment": True, "row_count": 240},
        )

        self.assertTrue(report["wave1_result_review_pass"])
        self.assertEqual(report["next_action"], "independent_triggered_holdout_discovery_required")
        self.assertIn("y2015_d120_s42_n240", report["excluded_scenario_ids"])
        self.assertIn("y2020_d120_s44_n240", report["excluded_scenario_ids"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_wave1_with_effect_observed_does_not_match_independent_no_trigger_path(self):
        report = build_report(
            readiness_v23={"wave1_controlled_canary_pass": True, "control_effect_observed": True},
            wave1_manifest={},
            triggered_manifest={},
            summary_audit={"aggregate": {"decision": "pass"}},
            trace_audit={"aggregate": {"strict_applied_steps": 1}},
            strict_effect_audit={"aggregate": {"runtime_error_steps": 0, "unsafe_applied_steps": 0}},
            runtime_provenance_audit={"audit_status": "runtime_provenance_complete"},
            joint_prediction_readiness={"ready_for_policy_judgment": True},
        )

        self.assertFalse(report["wave1_result_review_pass"])
        self.assertIn("wave1_result_review_not_passed", report["blockers"])


if __name__ == "__main__":
    unittest.main()
