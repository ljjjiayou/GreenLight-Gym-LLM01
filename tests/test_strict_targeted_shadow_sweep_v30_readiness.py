import unittest

from gl_gym.experiments.strict_targeted_shadow_sweep_v30_readiness import build_report


class TestStrictTargetedShadowSweepV30Readiness(unittest.TestCase):
    def test_readiness_blocks_when_cache_coverage_fails(self):
        report = build_report(
            manifest={"strict_targeted_shadow_sweep_v30_manifest_ready": True, "scenario_count": 28},
            cache_coverage={
                "cache_coverage_pass": False,
                "coverage_rate": 0.9,
                "missing_key_count": 1,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
                "expected_env_count": 28,
            },
        )

        self.assertFalse(report["v30_cache_coverage_pass"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertEqual(report["next_action"], "v30_subset_cache_coverage_blocked")

    def test_readiness_requests_cache_precheck_when_strict_source_found(self):
        report = build_report(
            manifest={"strict_targeted_shadow_sweep_v30_manifest_ready": True, "scenario_count": 28},
            cache_coverage={
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
                "expected_env_count": 28,
            },
            execution_record={"strict_targeted_shadow_sweep_v30_authorized": True},
            strict_sourcing={
                "strict_targeted_source_found": True,
                "expected_strict_eligible_applied_steps": 2,
                "would_apply_steps": 2,
                "strict_filtered_steps": 0,
            },
            summary_audit={"aggregate": {"decision": "pass", "failure_count": 0}},
            runtime_audit={
                "audit_status": "runtime_provenance_complete",
                "record_count": 1,
                "runtime_reason_missing_count": 0,
                "unknown_post_guardrail_rewrite_count": 0,
            },
            joint_readiness={"ready_for_policy_judgment": True, "missing_field_counts": {}},
        )

        self.assertTrue(report["strict_targeted_source_found"])
        self.assertTrue(report["post_run_audits_pass"])
        self.assertEqual(report["next_action"], "strict_targeted_source_manifest_and_no_fill_cache_precheck")
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_readiness_marks_subset_exhaustion_when_no_source_found_after_sweep(self):
        report = build_report(
            manifest={"strict_targeted_shadow_sweep_v30_manifest_ready": True, "scenario_count": 28},
            cache_coverage={
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
                "expected_env_count": 28,
            },
            execution_record={"strict_targeted_shadow_sweep_v30_authorized": True},
            strict_sourcing={
                "strict_targeted_source_found": False,
                "expected_strict_eligible_applied_steps": 0,
                "would_apply_steps": 4,
                "strict_filtered_steps": 4,
                "blocker_taxonomy": ["strict_targeted_source_not_found"],
            },
            summary_audit={"aggregate": {"decision": "pass", "failure_count": 0}},
            runtime_audit={
                "audit_status": "runtime_provenance_complete",
                "record_count": 1,
                "runtime_reason_missing_count": 0,
                "unknown_post_guardrail_rewrite_count": 0,
            },
            joint_readiness={"ready_for_policy_judgment": True, "missing_field_counts": {}},
        )

        self.assertIn("strict_targeted_source_not_found_in_coverage_qualified_subset", report["failure_taxonomy"])
        self.assertEqual(report["next_action"], "new_scenario_cache_discovery_for_strict_targeted_source")


if __name__ == "__main__":
    unittest.main()
