import unittest

from gl_gym.experiments.mainline_and_post_guardrail_status import build_report


class TestMainlineAndPostGuardrailStatus(unittest.TestCase):
    def test_blocks_on_cleanup_before_post_guardrail_followup(self):
        report = build_report(
            cleanup_plan={"blocking_paths": ["gl_gym/agent/expert_distillation.py"], "cleanup_action_counts": {}},
            rewrite_audit={
                "harmful_override_count": 8,
                "root_cause_layer_counts": {"post_guardrail_destruction": 8},
            },
            taxonomy_report={"taxonomy_counts": {"candidate_space_insufficiency": 2}, "next_action": "candidate_space_redesign_shadow"},
            inventory_report={"regime_available": {"cold_humid": True}, "regime_step_counts": {"cold_humid": 1}},
            action_diff_report={"trace_count": 45, "action_diff_steps": 0, "missing_step_count": 0},
        )

        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["next_action"], "mainline_diff_cleanup_required")
        self.assertFalse(report["gates"]["mainline_diff_cleanup_complete"])

    def test_prioritizes_post_guardrail_when_cleanup_is_clear(self):
        report = build_report(
            cleanup_plan={"blocking_paths": [], "cleanup_action_counts": {}},
            rewrite_audit={
                "harmful_override_count": 8,
                "root_cause_layer_counts": {"post_guardrail_destruction": 8},
            },
            taxonomy_report={"taxonomy_counts": {"candidate_space_insufficiency": 2}, "next_action": "candidate_space_redesign_shadow"},
            inventory_report={"regime_available": {"cold_humid": True}, "regime_step_counts": {"cold_humid": 1}},
            action_diff_report={"trace_count": 45, "action_diff_steps": 0, "missing_step_count": 0},
        )

        self.assertEqual(report["next_action"], "post_guardrail_policy_shadow_design")


if __name__ == "__main__":
    unittest.main()
