import unittest

from gl_gym.experiments.candidate_generation_shadow_status import build_report


class TestCandidateGenerationShadowStatus(unittest.TestCase):
    def test_blocks_on_mainline_diff_first(self):
        report = build_report(
            mainline_diff_report={
                "controlled_replay_blocked": True,
                "controlled_replay_block_reasons": ["mainline_blocking_diff_present"],
                "expert_distillation_status": "unrelated_diff_present",
            },
            post_guardrail_report={"harmful_override_count": 8, "harmful_override_preventable_by_proxy_v2_1_count": 2},
            candidate_suite_report={
                "candidate_pool_gap_steps": 6,
                "post_guardrail_safe_useful_repair_count": 6,
                "pure_hot_dry_false_positive_blocker_count": 0,
                "candidate_generation_shadow_conclusion": "candidate_generation_shadow_expand",
            },
            action_diff_report={"trace_count": 45, "action_diff_steps": 0, "missing_step_count": 0},
            proxy_status_report={"controlled_replay_allowed": False},
        )

        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["next_action"], "mainline_diff_cleanup_required")
        self.assertFalse(report["gates"]["mainline_diff_clean_enough_for_controlled_replay"])

    def test_prioritizes_post_guardrail_when_diff_clean_but_harmful_unexplained(self):
        report = build_report(
            mainline_diff_report={"controlled_replay_blocked": False},
            post_guardrail_report={"harmful_override_count": 8, "harmful_override_preventable_by_proxy_v2_1_count": 2},
            candidate_suite_report={
                "candidate_pool_gap_steps": 6,
                "post_guardrail_safe_useful_repair_count": 6,
                "pure_hot_dry_false_positive_blocker_count": 0,
                "candidate_generation_shadow_conclusion": "candidate_generation_shadow_expand",
            },
            action_diff_report={"trace_count": 45, "action_diff_steps": 0, "missing_step_count": 0},
        )

        self.assertEqual(report["next_action"], "post_guardrail_policy_shadow_design")
        self.assertFalse(report["gates"]["post_guardrail_harmful_override_fully_explained"])


if __name__ == "__main__":
    unittest.main()
