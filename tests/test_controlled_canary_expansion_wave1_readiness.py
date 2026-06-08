import unittest

from gl_gym.experiments.controlled_canary_expansion_wave1_readiness import build_report


def prereq():
    return {
        "admission_review": {"controlled_canary_expansion_admission_pass": True},
        "manifest": {"wave1_manifest_ready": True, "scenario_ids": ["y2020_d120_s44_n240"]},
        "cache_coverage": {
            "cache_coverage_pass": True,
            "coverage_rate": 1.0,
            "missing_key_count": 0,
            "missing_buffered_action_count": 0,
            "missing_parsed_plan_count": 0,
        },
    }


class TestControlledCanaryExpansionWave1Readiness(unittest.TestCase):
    def test_no_trigger_guard_scenario_can_pass_safety_gate(self):
        args = prereq()
        report = build_report(
            **args,
            execution_record={"wave1_controlled_canary_authorized": True},
            summary_audit={
                "aggregate": {"decision": "pass", "runtime_error_steps": 0, "strict_cache_miss_runtime_error_steps": 0},
                "paired_deltas": [
                    {
                        "scenario_id": "y2020_d120_s44_n240",
                        "d_canopy_dew_margin_lt0_steps": 0,
                        "d_dew_margin_air_lt0_steps": 0,
                    }
                ],
            },
            trace_audit={"aggregate": {"strict_applied_steps": 0, "unsafe_preferred_steps": 0, "unsafe_conflict_steps": 0, "unsafe_applied_steps": 0}},
            strict_effect_audit={"aggregate": {"runtime_error_steps": 0, "unsafe_applied_steps": 0}},
            runtime_provenance_audit={"audit_status": "runtime_provenance_complete", "record_count": 2, "runtime_reason_missing_count": 0, "unknown_post_guardrail_rewrite_count": 0},
            joint_prediction_readiness={"ready_for_policy_judgment": True, "row_count": 240},
        )

        self.assertTrue(report["wave1_controlled_canary_pass"])
        self.assertEqual(report["effect_decision"], "expansion_no_trigger")
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_action_change_outside_strict_eligible_fails(self):
        args = prereq()
        report = build_report(
            **args,
            execution_record={"wave1_controlled_canary_authorized": True},
            summary_audit={"aggregate": {"decision": "pass"}, "paired_deltas": []},
            trace_audit={"aggregate": {"strict_applied_steps": 1, "unsafe_preferred_steps": 0, "unsafe_conflict_steps": 1, "unsafe_applied_steps": 0}},
            strict_effect_audit={"aggregate": {"runtime_error_steps": 0, "unsafe_applied_steps": 0}},
            runtime_provenance_audit={"audit_status": "runtime_provenance_complete", "record_count": 2, "runtime_reason_missing_count": 0, "unknown_post_guardrail_rewrite_count": 0},
            joint_prediction_readiness={"ready_for_policy_judgment": True, "row_count": 240},
        )

        self.assertFalse(report["wave1_controlled_canary_pass"])
        self.assertIn("unsafe_conflict", report["failure_taxonomy"])

    def test_new_hard_safety_violation_fails(self):
        args = prereq()
        report = build_report(
            **args,
            execution_record={"wave1_controlled_canary_authorized": True},
            summary_audit={
                "aggregate": {"decision": "pass"},
                "paired_deltas": [
                    {
                        "scenario_id": "y2020_d120_s44_n240",
                        "d_canopy_dew_margin_lt0_steps": 1,
                        "d_dew_margin_air_lt0_steps": 0,
                    }
                ],
            },
            trace_audit={"aggregate": {"strict_applied_steps": 1, "unsafe_preferred_steps": 0, "unsafe_conflict_steps": 0, "unsafe_applied_steps": 0}},
            strict_effect_audit={"aggregate": {"runtime_error_steps": 0, "unsafe_applied_steps": 0}},
            runtime_provenance_audit={"audit_status": "runtime_provenance_complete", "record_count": 2, "runtime_reason_missing_count": 0, "unknown_post_guardrail_rewrite_count": 0},
            joint_prediction_readiness={"ready_for_policy_judgment": True, "row_count": 240},
        )

        self.assertFalse(report["wave1_controlled_canary_pass"])
        self.assertIn("hard_safety_regression", report["failure_taxonomy"])


if __name__ == "__main__":
    unittest.main()
