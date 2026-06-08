import unittest

from gl_gym.experiments.proxy_v2_1_shadow_calibration_status import build_report


class TestProxyV21ShadowCalibrationStatus(unittest.TestCase):
    def test_recommends_candidate_generation_when_gap_has_useful_shadow_repair(self):
        report = build_report(
            shadow_validation_reports=[
                {
                    "proxy_v2_1_shadow_candidate": {
                        "candidate_name": "proxy_v2_1_shadow_candidate",
                        "buffer_scale": 0.75,
                        "v2_unwarned_false_safe": 0,
                        "pure_hot_dry_affected_steps": 12,
                        "baseline_proxy_v2_pure_hot_dry_affected_steps": 45,
                        "accepted_for_shadow_followup": True,
                        "rejected_scales": {
                            "buffer_scale_0.5": {
                                "reason": "unsafe_due_to_false_safe",
                                "v2_unwarned_false_safe": 2,
                                "pure_hot_dry_affected_steps": 0,
                            }
                        },
                    }
                }
            ],
            candidate_repair_reports=[
                {
                    "candidate_pool_gap_steps": 6,
                    "repair_candidate_available_steps": 6,
                    "candidate_pool_repair_conclusion": "new_candidate_needed_shadow_evidence",
                    "candidate_generation_shadow_conclusion": "new_candidate_needed_shadow_evidence_post_guardrail_safe",
                    "repair_quality_counts": {"feasible_useful_dry_relief": 6},
                }
            ],
            post_guardrail_reports=[
                {
                    "harmful_override_count": 8,
                    "harmful_override_preventable_by_proxy_v2_1_count": 2,
                    "root_cause_summary": {"guardrail_introduced_risk": 8},
                    "fine_grained_root_cause_summary": {"screen_vent_coupled_risk": 8},
                    "harmful_fine_grained_root_cause_summary": {"pre_score_candidate_safe_but_post_guardrail_unsafe": 8},
                }
            ],
            action_diff_reports=[
                {"trace_count": 45, "action_diff_steps": 0, "missing_step_count": 0, "max_abs_delta": 0.0}
            ],
        )

        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["next_action"], "candidate_generation_shadow_design")
        self.assertTrue(report["gates"]["buffer_scale_0_5_rejected_due_to_false_safe"])
        self.assertTrue(report["gates"]["proxy_v2_1_reduces_pure_hot_dry_affected"])
        self.assertEqual(
            report["candidate_repair"]["candidate_generation_shadow_conclusion"],
            "new_candidate_needed_shadow_evidence_post_guardrail_safe",
        )
        self.assertIn("screen_vent_coupled_risk", report["post_guardrail"]["fine_grained_root_cause_summary"])
        self.assertIn(
            "pre_score_candidate_safe_but_post_guardrail_unsafe",
            report["post_guardrail"]["harmful_fine_grained_root_cause_summary"],
        )

    def test_prioritizes_post_guardrail_when_repair_is_blocked_after_rewrite(self):
        report = build_report(
            shadow_validation_reports=[
                {
                    "proxy_v2_1_shadow_candidate": {
                        "buffer_scale": 0.75,
                        "v2_unwarned_false_safe": 0,
                        "pure_hot_dry_affected_steps": 12,
                        "baseline_proxy_v2_pure_hot_dry_affected_steps": 45,
                        "accepted_for_shadow_followup": True,
                        "rejected_scales": {"buffer_scale_0.5": {"reason": "unsafe_due_to_false_safe"}},
                    }
                }
            ],
            candidate_repair_reports=[
                {
                    "candidate_pool_gap_steps": 6,
                    "repair_candidate_available_steps": 6,
                    "candidate_pool_repair_conclusion": "new_candidate_needed_shadow_evidence",
                    "candidate_generation_shadow_conclusion": "post_guardrail_policy_blocks_candidate_repair",
                }
            ],
            post_guardrail_reports=[
                {"harmful_override_count": 8, "harmful_override_preventable_by_proxy_v2_1_count": 2}
            ],
            action_diff_reports=[
                {"trace_count": 45, "action_diff_steps": 0, "missing_step_count": 0, "max_abs_delta": 0.0}
            ],
        )

        self.assertEqual(report["next_action"], "post_guardrail_root_cause_deepening")


if __name__ == "__main__":
    unittest.main()
