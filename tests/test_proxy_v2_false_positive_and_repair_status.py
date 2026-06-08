import unittest

from gl_gym.experiments.proxy_v2_false_positive_and_repair_status import build_report


class TestProxyV2FalsePositiveAndRepairStatus(unittest.TestCase):
    def test_blocks_controlled_replay_when_false_positive_remains(self):
        report = build_report(
            shadow_validation_reports=[
                {
                    "affected_pure_hot_dry_steps": 3,
                    "pure_hot_dry_warning_without_hard_event_next": 3,
                    "false_safe_v1_unwarned_by_v2": 0,
                    "warning_classification_counts": {"over_conservative_warning": 3},
                    "warning_action_pattern_counts": {"other": 3},
                    "buffer_sensitivity_summary": {
                        "buffer_scale_1": {
                            "v2_unwarned_false_safe": 0,
                            "pure_hot_dry_affected_steps": 3,
                        }
                    },
                }
            ],
            candidate_response_reports=[
                {
                    "traces": [
                        {
                            "false_safe_canopy_count": 1,
                            "v2_unwarned_false_safe_count": 0,
                            "missing_prediction_metadata": False,
                            "proxy_v2_buffer_sensitivity": {
                                "buffer_scale_1": {
                                    "v2_unwarned_false_safe": 0,
                                    "pure_hot_dry_affected_steps": 0,
                                }
                            },
                        }
                    ]
                }
            ],
            candidate_repair_reports=[
                {
                    "candidate_pool_gap_steps": 2,
                    "repair_candidate_available_steps": 2,
                    "candidate_pool_repair_conclusion": "new_candidate_needed",
                    "repair_quality_counts": {"feasible_useful_dry_relief": 1},
                }
            ],
            post_guardrail_reports=[
                {
                    "harmful_override_count": 2,
                    "harmful_override_preventable_by_proxy_v2_count": 1,
                    "harmful_override_explainable_by_proxy_v2_or_candidate_repair_count": 2,
                    "root_cause_summary": {"guardrail_introduced_risk": 2},
                    "harmful_root_cause_combinations": {"guardrail_introduced_risk": 2},
                }
            ],
            action_diff_reports=[
                {
                    "trace_count": 1,
                    "action_diff_steps": 0,
                    "missing_step_count": 0,
                    "max_abs_delta": 0.0,
                }
            ],
        )

        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["next_action"], "proxy_v2_false_positive_calibration")
        self.assertEqual(report["false_positive_decomposition"]["affected_pure_hot_dry_steps"], 3)
        self.assertEqual(report["candidate_repair"]["candidate_pool_repair_conclusion"], "new_candidate_needed")
        self.assertTrue(report["gates"]["metadata_only_action_diff_zero"])


if __name__ == "__main__":
    unittest.main()
