import unittest

from gl_gym.experiments.post_guardrail_shadow_policy_instrumentation_status import build_report


class TestPostGuardrailShadowPolicyInstrumentationStatus(unittest.TestCase):
    def test_mainline_authorization_blocks_first(self):
        report = build_report(
            authorization_status={
                "authorization_complete": False,
                "blocking_paths": ["gl_gym/agent/expert_distillation.py"],
                "status_counts": {"not_authorized_for_promotion_evidence": 1},
            },
            runtime_provenance_audit={
                "audit_status": "instrumented_but_needs_metadata_replay",
                "record_count": 0,
                "unknown_post_guardrail_rewrite_count": 8,
                "runtime_reason_missing_count": 8,
                "metadata_replay_required": True,
            },
            joint_prediction_v2={
                "ready_for_policy_judgment": False,
                "row_count": 8,
                "missing_field_counts": {"predicted_vpd_after_floor": 8},
            },
            action_diff={"action_diff_steps": 0, "max_abs_delta": 0.0},
            replay_readiness={
                "ready_for_future_shadow_replay_input": True,
                "replay_run": False,
                "recommended_replay_order": ["neutral_no_risk"],
            },
        )

        self.assertEqual(report["next_action"], "mainline_diff_authorization_required")
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["gates"]["mainline_diff_authorization_complete"])
        self.assertTrue(report["gates"]["metadata_only_action_diff_zero"])


if __name__ == "__main__":
    unittest.main()
