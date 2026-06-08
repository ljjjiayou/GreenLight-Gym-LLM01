import unittest

from gl_gym.experiments.strict_metadata_replay_run_plan import build_report


class TestStrictMetadataReplayRunPlan(unittest.TestCase):
    def test_drafts_canonical_plan_without_execution_permission(self):
        report = build_report(
            protocol_v1_snapshot={"snapshot_available": True, "source_git_ref": "HEAD", "source_git_commit": "abc"},
            blocked_categories={
                "blocked_categories_resolved_for_v1_preflight": True,
                "blocked_categories_resolved_for_protocol_v2": False,
            },
            cache_scenario_plan={
                "cache_scenario_plan_ready": True,
                "coverage_claim_scope": "canonical_failure_only",
                "controller": "llm_rspc_v2",
                "selected_cache_path": "cache.json",
                "scenario_list": [
                    {
                        "scenario_id": "y2020_d120_s44_n240",
                        "scenario_family": "canonical_failure",
                        "env_id": "TomatoEnv_y2020_d120_s44",
                    }
                ],
            },
            cache_coverage={"cache_coverage_pass": True},
            default_path_plan={"plan_ready": True},
            expanded_cache_plan={
                "rows": [
                    {
                        "family": "neutral_no_risk",
                        "coverage_status": "blocked_missing_cache_or_scenario",
                        "scenario_count": 1,
                    }
                ]
            },
        )

        self.assertTrue(report["strict_metadata_replay_run_plan_ready"])
        self.assertFalse(report["replay_run"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertEqual(report["initial_run_scope"], "canonical_failure_only")
        self.assertEqual(report["targets"]["action_diff_steps"], 0)
        self.assertEqual(report["expanded_family_blockers"][0]["family"], "neutral_no_risk")

    def test_missing_canonical_cache_blocks_plan(self):
        report = build_report(
            protocol_v1_snapshot={"snapshot_available": True},
            blocked_categories={"blocked_categories_resolved_for_v1_preflight": True},
            cache_scenario_plan={"cache_scenario_plan_ready": True, "scenario_list": []},
            cache_coverage={"cache_coverage_pass": True},
            default_path_plan={"plan_ready": True},
        )

        self.assertFalse(report["strict_metadata_replay_run_plan_ready"])
        self.assertEqual(report["next_action"], "repair_strict_metadata_replay_run_plan_inputs")


if __name__ == "__main__":
    unittest.main()
