import unittest

from gl_gym.experiments.controlled_canary_independent_wave2_readiness import build_report


def prereq():
    return {
        "result_review": {"wave1_result_review_pass": True},
        "discovery": {"independent_trigger_candidates_found": True},
        "manifest": {"wave2_manifest_ready": True, "scenario_ids": ["y2018_d120_s45_n240"]},
        "cache_coverage": {
            "cache_coverage_pass": True,
            "coverage_rate": 1.0,
            "missing_key_count": 0,
            "missing_buffered_action_count": 0,
            "missing_parsed_plan_count": 0,
        },
    }


class TestControlledCanaryIndependentWave2Readiness(unittest.TestCase):
    def test_wave2_pass_requires_strict_applied_and_resets_promotion_flags(self):
        report = build_report(
            **prereq(),
            execution_record={"wave2_controlled_canary_authorized": True},
            summary_audit={
                "aggregate": {"decision": "pass", "runtime_error_steps": 0, "strict_cache_miss_runtime_error_steps": 0},
                "paired_deltas": [
                    {
                        "scenario_id": "y2018_d120_s45_n240",
                        "d_canopy_dew_margin_lt0_steps": 0,
                        "d_dew_margin_air_lt0_steps": 0,
                    }
                ],
            },
            trace_audit={
                "aggregate": {
                    "strict_applied_steps": 2,
                    "unsafe_preferred_steps": 0,
                    "unsafe_conflict_steps": 0,
                    "unsafe_applied_steps": 0,
                }
            },
            strict_effect_audit={"aggregate": {"runtime_error_steps": 0, "unsafe_applied_steps": 0}},
            runtime_provenance_audit={"audit_status": "runtime_provenance_complete", "record_count": 5},
            joint_prediction_readiness={"ready_for_policy_judgment": True, "row_count": 240},
        )

        self.assertTrue(report["wave2_controlled_canary_pass"])
        self.assertTrue(report["control_effect_observed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_no_strict_applied_fails_wave2(self):
        report = build_report(
            **prereq(),
            execution_record={"wave2_controlled_canary_authorized": True},
            summary_audit={"aggregate": {"decision": "pass"}, "paired_deltas": []},
            trace_audit={
                "aggregate": {
                    "strict_applied_steps": 0,
                    "unsafe_preferred_steps": 0,
                    "unsafe_conflict_steps": 0,
                    "unsafe_applied_steps": 0,
                }
            },
            strict_effect_audit={"aggregate": {"runtime_error_steps": 0, "unsafe_applied_steps": 0}},
            runtime_provenance_audit={"audit_status": "runtime_provenance_complete"},
            joint_prediction_readiness={"ready_for_policy_judgment": True},
        )

        self.assertFalse(report["wave2_controlled_canary_pass"])
        self.assertIn("strict_not_triggered", report["failure_taxonomy"])

    def test_v32_readiness_accepts_shadow_sourced_review_and_never_promotes(self):
        report = build_report(
            result_review={"shadow_sourced_canary_result_review_pass": True},
            discovery={"independent_trigger_candidates_found": False},
            schema_version="metadata_replay_readiness_checklist_v32",
            mode_label="v32 independent holdout admission readiness",
        )

        self.assertEqual(report["schema_version"], "metadata_replay_readiness_checklist_v32")
        self.assertFalse(report["wave2_pre_run_ready"])
        self.assertEqual(report["next_action"], "new_shadow_only_scenario_cache_discovery_after_independent_holdout_blocked")
        self.assertIn("no_independent_strict_trigger_candidate_after_v31", report["failure_taxonomy"])
        self.assertTrue(report["acceptance"]["result_review_pass"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
