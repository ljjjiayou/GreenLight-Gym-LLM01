import unittest

from gl_gym.experiments.controlled_canary_shadow_trigger_source_sweep_readiness import build_report


def readiness_v25():
    return {
        "shadow_trigger_sources_found": True,
        "shadow_trigger_source_manifest_ready": True,
        "shadow_trigger_source_cache_precheck_pass": True,
    }


def summary_audit():
    return {
        "aggregate": {"decision": "pass", "scenario_count": 5},
        "rows": [
            {
                "scenario_id": "y2015_d180_s42_n240",
                "plan_cache_enabled_steps": 240,
                "plan_cache_hit_steps": 240,
                "runtime_error_steps": 0,
                "strict_cache_miss_runtime_error_steps": 0,
            }
        ],
    }


class TestControlledCanaryShadowTriggerSourceSweepReadiness(unittest.TestCase):
    def test_pass_with_shadow_windows_resets_all_execution_and_promotion_flags(self):
        report = build_report(
            readiness_v25=readiness_v25(),
            execution_record={"shadow_trigger_source_sweep_authorized": True},
            summary_audit=summary_audit(),
            controlled_shadow_audit={"aggregate": {"unsafe_preferred_steps": 0, "unsafe_applied_steps": 0, "would_apply_steps": 5}},
            shadow_replay_audit={"aggregate": {"signal_steps": 5, "replay_window_count": 2}},
            runtime_provenance_audit={
                "audit_status": "runtime_provenance_complete",
                "record_count": 8,
                "runtime_reason_missing_count": 0,
                "unknown_post_guardrail_rewrite_count": 0,
            },
            joint_prediction_readiness={"ready_for_policy_judgment": True, "missing_field_counts": {}, "row_count": 240},
        )

        self.assertTrue(report["shadow_trigger_source_sweep_pass"])
        self.assertTrue(report["near_miss_window"])
        self.assertEqual(report["next_action"], "independent_shadow_source_expansion_admission_planning")
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["controlled_replay_execution_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_pass_without_windows_goes_to_exhaustion_diagnosis(self):
        report = build_report(
            readiness_v25=readiness_v25(),
            execution_record={"shadow_trigger_source_sweep_authorized": True},
            summary_audit=summary_audit(),
            controlled_shadow_audit={"aggregate": {"unsafe_preferred_steps": 0, "unsafe_applied_steps": 0}},
            shadow_replay_audit={"aggregate": {"signal_steps": 0, "replay_window_count": 0}},
            runtime_provenance_audit={
                "audit_status": "runtime_provenance_complete",
                "record_count": 8,
                "runtime_reason_missing_count": 0,
                "unknown_post_guardrail_rewrite_count": 0,
            },
            joint_prediction_readiness={"ready_for_policy_judgment": True, "missing_field_counts": {}, "row_count": 240},
        )

        self.assertTrue(report["shadow_trigger_source_sweep_pass"])
        self.assertFalse(report["near_miss_window"])
        self.assertEqual(report["next_action"], "shadow_trigger_source_exhaustion_diagnosis")

    def test_missing_runtime_provenance_blocks_pass(self):
        report = build_report(
            readiness_v25=readiness_v25(),
            execution_record={"shadow_trigger_source_sweep_authorized": True},
            summary_audit=summary_audit(),
            controlled_shadow_audit={"aggregate": {"unsafe_preferred_steps": 0, "unsafe_applied_steps": 0}},
            shadow_replay_audit={"aggregate": {"signal_steps": 1, "replay_window_count": 1}},
            runtime_provenance_audit={"audit_status": "runtime_provenance_complete", "record_count": 0},
            joint_prediction_readiness={"ready_for_policy_judgment": True, "missing_field_counts": {}, "row_count": 240},
        )

        self.assertFalse(report["shadow_trigger_source_sweep_pass"])
        self.assertIn("runtime_provenance_missing", report["failure_taxonomy"])


if __name__ == "__main__":
    unittest.main()
