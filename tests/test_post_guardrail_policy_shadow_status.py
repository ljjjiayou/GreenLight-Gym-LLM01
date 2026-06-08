import unittest

from gl_gym.experiments.post_guardrail_policy_shadow_status import build_report


class TestPostGuardrailPolicyShadowStatus(unittest.TestCase):
    def test_cleanup_blocks_controlled_replay(self):
        report = build_report(
            decision_packet={"blocking_paths": ["gl_gym/agent/expert_distillation.py"], "decision_counts": {}},
            counterfactual={"vent_floor_likely_sufficient": True, "conclusion_counts": {"vent_floor_likely_sufficient": 1}},
            provenance={"unsafe_for_promotion_due_to_missing_provenance": True, "reason_missing_count": 1, "rule_counts": {}},
            family_ablation={
                "candidate_family_ablation_conclusion": "candidate_space_redesign_shadow",
                "useful_shadow_families": ["canopy_safe_hold_screen"],
                "hard_safety_regression_present": False,
            },
            counterexample_manifest={"suite_name": "regime_counterexample_suite_v1", "regime_counts": {"cold_humid": 1}, "windows": [{}]},
            action_diff={"trace_count": 1, "action_diff_steps": 0, "missing_step_count": 0, "max_abs_delta": 0.0},
        )

        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["next_action"], "mainline_diff_cleanup_required")
        self.assertFalse(report["gates"]["mainline_diff_cleanup_complete"])
        self.assertTrue(report["gates"]["metadata_only_action_diff_zero"])

    def test_without_cleanup_prioritizes_post_guardrail_policy(self):
        report = build_report(
            decision_packet={"blocking_paths": [], "decision_counts": {}},
            counterfactual={"vent_floor_likely_sufficient": True, "conclusion_counts": {"vent_floor_likely_sufficient": 1}},
            provenance={"unsafe_for_promotion_due_to_missing_provenance": True, "reason_missing_count": 1, "rule_counts": {}},
            family_ablation={"candidate_family_ablation_conclusion": "candidate_space_redesign_shadow"},
            counterexample_manifest={"windows": [{}]},
            action_diff={"trace_count": 1, "action_diff_steps": 0, "missing_step_count": 0},
        )
        self.assertEqual(report["next_action"], "post_guardrail_policy_shadow_design")


if __name__ == "__main__":
    unittest.main()
