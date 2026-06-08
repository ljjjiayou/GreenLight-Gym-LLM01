import unittest

from gl_gym.experiments.post_guardrail_policy_next_status import build_report


class TestPostGuardrailPolicyNextStatus(unittest.TestCase):
    def test_mainline_diff_blocks_next_action_first(self):
        report = build_report(
            decision_packet={"blocking_paths": ["gl_gym/agent/expert_distillation.py"]},
            static_provenance={"runtime_provenance_instrumentation_required": True, "runtime_reason_missing_count": 8},
            vent_policy={"vent_floor_shadow_block_rows": 6, "would_block_count": 8},
            warning_audit={"requires_warning_to_action_blocker_shadow_design": True, "proxy_warning_not_actionable_count": 2},
            family_ablation={"hard_safety_regression_present": False},
            counterexample_shadow={"rows": [{}], "regime_counts": {"neutral_no_risk": 1}},
            action_diff={"action_diff_steps": 0, "missing_step_count": 0, "max_abs_delta": 0.0},
        )

        self.assertEqual(report["next_action"], "mainline_diff_cleanup_required")
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertTrue(report["gates"]["runtime_provenance_instrumentation_required"])


if __name__ == "__main__":
    unittest.main()
