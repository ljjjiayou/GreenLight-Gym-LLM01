import unittest

from gl_gym.experiments.controlled_canary_shadow_sourced_result_review import build_report


class TestControlledCanaryShadowSourcedResultReview(unittest.TestCase):
    def _base_inputs(self):
        return {
            "readiness_v31": {
                "shadow_sourced_controlled_canary_pass": True,
                "control_effect_observed": True,
                "controlled_replay_allowed": False,
                "controlled_replay_execution_allowed": False,
                "performance_claim_allowed": False,
                "promotion_evidence": False,
                "post_run_metrics": {
                    "expected_strict_eligible_applied_steps": 9,
                    "strict_applied_steps": 9,
                },
            },
            "v31_manifest": {
                "scenario_ids": [
                    "y2015_d120_s42_n240",
                    "y2015_d120_s43_n240",
                    "y2015_d120_s44_n240",
                ]
            },
            "summary_audit": {
                "aggregate": {"decision": "pass", "scenario_count": 3},
                "paired_deltas": [
                    {
                        "scenario_id": "y2015_d120_s42_n240",
                        "d_canopy_dew_margin_lt0_steps": 0,
                        "d_dew_margin_air_lt0_steps": 0,
                    }
                ],
            },
            "trace_audit": {
                "aggregate": {
                    "strict_applied_steps": 9,
                    "would_apply_steps": 20,
                    "strict_filtered_steps": 1,
                    "unsafe_preferred_steps": 0,
                    "unsafe_conflict_steps": 0,
                    "unsafe_applied_steps": 0,
                }
            },
            "strict_effect_audit": {"aggregate": {"runtime_error_steps": 0, "unsafe_applied_steps": 0}},
            "runtime_provenance_audit": {
                "audit_status": "runtime_provenance_complete",
                "record_count": 1278,
                "runtime_reason_missing_count": 0,
                "unknown_post_guardrail_rewrite_count": 0,
            },
            "joint_prediction_readiness": {
                "ready_for_policy_judgment": True,
                "row_count": 1440,
                "missing_field_counts": {},
            },
        }

    def test_v31_positive_canary_signal_requests_independent_discovery_without_promotion(self):
        report = build_report(
            **self._base_inputs(),
            extra_excluded_scenario_ids=["y2020_d120_s44_n240"],
        )

        self.assertTrue(report["shadow_sourced_canary_result_review_pass"])
        self.assertTrue(report["independent_holdout_admission_review_pass"])
        self.assertEqual(report["effect_decision"], "positive_canary_signal")
        self.assertEqual(report["next_action"], "independent_triggered_holdout_discovery_required")
        self.assertIn("y2015_d120_s42_n240", report["excluded_scenario_ids"])
        self.assertIn("y2020_d120_s44_n240", report["excluded_scenario_ids"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_hard_safety_regression_blocks_review(self):
        inputs = self._base_inputs()
        inputs["summary_audit"] = {
            "aggregate": {"decision": "pass"},
            "paired_deltas": [
                {
                    "scenario_id": "y2015_d120_s42_n240",
                    "d_canopy_dew_margin_lt0_steps": 1,
                    "d_dew_margin_air_lt0_steps": 0,
                }
            ],
        }

        report = build_report(**inputs)

        self.assertFalse(report["shadow_sourced_canary_result_review_pass"])
        self.assertIn("shadow_sourced_canary_result_review_not_passed", report["blockers"])
        self.assertFalse(report["performance_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
