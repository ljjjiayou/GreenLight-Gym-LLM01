import unittest

from gl_gym.experiments.controlled_canary_strict_trigger_readiness import build_report, build_v28_report


class TestControlledCanaryStrictTriggerReadiness(unittest.TestCase):
    def test_readiness_blocks_when_no_trigger_candidate(self):
        report = build_report(
            result_review={"canary_safety_pass": True},
            trigger_discovery={
                "trigger_candidates_found": False,
                "expected_strict_eligible_applied_steps": 0,
                "would_apply_steps": 6,
                "strict_filtered_steps": 3,
                "blocker_taxonomy": ["strict_not_triggered_under_current_rules"],
            },
        )

        self.assertEqual(report["schema_version"], "metadata_replay_readiness_checklist_v22")
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertEqual(report["next_action"], "strict_trigger_blocker_diagnosis_or_scenario_discovery_continue")

    def test_readiness_accepts_triggered_canary_only_with_strict_applied(self):
        report = build_report(
            result_review={"canary_safety_pass": True},
            trigger_discovery={"trigger_candidates_found": True, "expected_strict_eligible_applied_steps": 2},
            triggered_manifest={"triggered_manifest_ready": True},
            cache_coverage={
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
            },
            execution_record={"triggered_controlled_canary_authorized": True},
            summary_audit={"aggregate": {"decision": "pass"}},
            trace_audit={
                "aggregate": {
                    "strict_applied_steps": 1,
                    "unsafe_preferred_steps": 0,
                    "unsafe_conflict_steps": 0,
                    "unsafe_applied_steps": 0,
                }
            },
            strict_effect_audit={"aggregate": {"runtime_error_steps": 0, "unsafe_applied_steps": 0, "warnings": []}},
            runtime_provenance_audit={"audit_status": "runtime_provenance_complete"},
            joint_prediction_readiness={"ready_for_policy_judgment": True},
        )

        self.assertTrue(report["triggered_controlled_canary_pass"])
        self.assertEqual(report["next_action"], "controlled_canary_expansion_admission_review")
        self.assertFalse(report["controlled_replay_allowed"])

    def test_v28_readiness_blocks_controlled_replay_when_no_strict_source(self):
        report = build_v28_report(
            trigger_discovery={
                "strict_targeted_source_found": False,
                "expected_strict_eligible_applied_steps": 0,
                "would_apply_steps": 44,
                "strict_filtered_steps": 22,
                "trace_count": 5,
                "blocker_taxonomy": ["strict_targeted_source_not_found"],
            }
        )

        self.assertEqual(report["schema_version"], "metadata_replay_readiness_checklist_v28")
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["controlled_replay_execution_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])
        self.assertEqual(report["next_action"], "design_larger_shadow_only_sweep")

    def test_v28_readiness_only_requests_cache_precheck_when_source_found(self):
        report = build_v28_report(
            trigger_discovery={
                "strict_targeted_source_found": True,
                "expected_strict_eligible_applied_steps": 2,
                "would_apply_steps": 2,
                "strict_filtered_steps": 0,
                "trace_count": 1,
                "strict_targeted_source_scenarios": [{"scenario_id": "y2015_d240_s42_n240"}],
            }
        )

        self.assertTrue(report["strict_targeted_source_found"])
        self.assertTrue(report["strict_targeted_source_manifest_ready"])
        self.assertTrue(report["no_fill_cache_precheck_requested"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["next_action"], "strict_targeted_source_manifest_and_no_fill_cache_precheck")


if __name__ == "__main__":
    unittest.main()
