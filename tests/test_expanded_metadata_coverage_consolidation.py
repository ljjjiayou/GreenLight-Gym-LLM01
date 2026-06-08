import unittest

from gl_gym.experiments.expanded_metadata_coverage_consolidation import build_report


def readiness(schema_version: str, *, phase_flag: str = ""):
    checks = {}
    if phase_flag:
        checks[phase_flag] = True
    return {
        "schema_version": schema_version,
        "canonical_strict_metadata_replay_pass": True,
        "failure_taxonomy": [],
        "checks": checks,
        "post_run_metrics": {
            "cache_hit_rate": 1.0,
            "runtime_error_steps": 0,
            "strict_cache_miss_runtime_error_steps": 0,
            "action_diff_steps": 0,
            "max_abs_delta": 0,
            "runtime_provenance_record_count": 4,
            "runtime_reason_missing_count": 0,
            "unknown_post_guardrail_rewrite_count": 0,
            "joint_prediction_missing_fields": 0,
            "joint_prediction_row_count": 8,
        },
    }


class TestExpandedMetadataCoverageConsolidation(unittest.TestCase):
    def test_all_three_phases_pass_for_admission_evidence(self):
        report = build_report(
            [
                readiness("metadata_replay_readiness_checklist_v17", phase_flag="stage_b_rerun_authorized"),
                readiness("metadata_replay_readiness_checklist_v18", phase_flag="first_wave_expanded_metadata_replay_authorized"),
                readiness("metadata_replay_readiness_checklist_v19", phase_flag="second_wave_expanded_metadata_replay_authorized"),
            ]
        )

        self.assertTrue(report["expanded_metadata_consolidation_pass"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertEqual(report["next_action"], "controlled_replay_admission_review")

    def test_missing_phase_blocks_consolidation(self):
        report = build_report(
            [
                readiness("metadata_replay_readiness_checklist_v17", phase_flag="stage_b_rerun_authorized"),
                readiness("metadata_replay_readiness_checklist_v19", phase_flag="second_wave_expanded_metadata_replay_authorized"),
            ]
        )

        self.assertFalse(report["expanded_metadata_consolidation_pass"])
        self.assertIn("first_wave_expanded_metadata", report["missing_phases"])


if __name__ == "__main__":
    unittest.main()
