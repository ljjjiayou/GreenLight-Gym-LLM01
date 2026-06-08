import unittest

from gl_gym.experiments.strict_targeted_shadow_sweep_v33_readiness import build_report


class TestStrictTargetedShadowSweepV33Readiness(unittest.TestCase):
    def test_readiness_turns_no_inventory_candidate_into_new_cache_design(self):
        report = build_report(
            near_miss_catalog={"near_miss_catalog_ready": True, "near_miss_scenario_count": 7, "near_miss_row_count": 44},
            inventory={"existing_cache_inventory_ready": True, "candidate_scenario_count": 0},
        )

        self.assertEqual(report["next_action"], "new_scenario_cache_acquisition_design")
        self.assertIn("existing_cache_exhausted_for_independent_strict_source", report["failure_taxonomy"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_readiness_requests_canary_admission_when_strict_source_found(self):
        report = build_report(
            near_miss_catalog={"near_miss_catalog_ready": True, "near_miss_scenario_count": 7, "near_miss_row_count": 44},
            inventory={"existing_cache_inventory_ready": True, "candidate_scenario_count": 1},
            manifest={"strict_targeted_shadow_sweep_v33_manifest_ready": True},
            cache_coverage={
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
            },
            execution_record={"strict_targeted_shadow_sweep_v33_authorized": True},
            strict_sourcing={"strict_targeted_source_found": True, "expected_strict_eligible_applied_steps": 1},
            summary_audit={"aggregate": {"decision": "pass"}},
            runtime_audit={
                "audit_status": "runtime_provenance_complete",
                "record_count": 1,
                "runtime_reason_missing_count": 0,
                "unknown_post_guardrail_rewrite_count": 0,
            },
            joint_readiness={"ready_for_policy_judgment": True, "missing_field_counts": {}},
        )

        self.assertEqual(report["next_action"], "strict_source_manifest_and_minimal_canary_admission_plan")
        self.assertTrue(report["strict_targeted_source_found"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_readiness_keeps_promotion_disabled_when_source_found_without_audits(self):
        report = build_report(
            near_miss_catalog={"near_miss_catalog_ready": True},
            inventory={"existing_cache_inventory_ready": True, "candidate_scenario_count": 1},
            manifest={"strict_targeted_shadow_sweep_v33_manifest_ready": True},
            cache_coverage={
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
            },
            execution_record={"strict_targeted_shadow_sweep_v33_authorized": True},
            strict_sourcing={"strict_targeted_source_found": True, "expected_strict_eligible_applied_steps": 1},
        )

        self.assertFalse(report["promotion_evidence"])
        self.assertFalse(report["controlled_replay_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
