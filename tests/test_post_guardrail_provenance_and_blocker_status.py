import unittest

from gl_gym.experiments.post_guardrail_provenance_and_blocker_status import build_report


class TestPostGuardrailProvenanceAndBlockerStatus(unittest.TestCase):
    def test_mainline_authorization_blocks_before_other_actions(self):
        report = build_report(
            authorization={
                "blocking_paths": ["gl_gym/agent/expert_distillation.py"],
                "authorization_counts": {"restore_or_user_authorization_required": 1},
            },
            runtime_design={"runtime_provenance_instrumentation_pending": True, "current_runtime_reason_missing_count": 8},
            vent_v2={"design_row_count": 6, "excluded_hard_warning_rows": 2},
            warning_design={"design_row_count": 2, "actual_hard_crossing_rows_covered": 2},
            joint_readiness={"ready_for_policy_judgment": False, "missing_field_counts": {"predicted_vpd_after_floor": 8}},
            replay_readiness={"ready_for_future_shadow_replay_input": True, "recommended_replay_order": ["neutral_no_risk"], "replay_run": False},
            action_diff={"action_diff_steps": 0},
        )

        self.assertEqual(report["next_action"], "mainline_diff_authorization_required")
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["gates"]["mainline_diff_authorization_complete"])
        self.assertTrue(report["gates"]["runtime_provenance_instrumentation_pending"])


if __name__ == "__main__":
    unittest.main()
