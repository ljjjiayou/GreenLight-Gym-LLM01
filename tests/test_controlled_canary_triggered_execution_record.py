import unittest

from gl_gym.experiments.controlled_canary_triggered_execution_record import build_report


class TestControlledCanaryTriggeredExecutionRecord(unittest.TestCase):
    def test_execution_record_requires_manifest_coverage_and_overlay(self):
        manifest = {
            "triggered_manifest_ready": True,
            "scenario_ids": ["y2020_d180_s44_n240"],
            "controllers": ["llm_rspc_v2", "llm_rspc_v2_hot_dry_proposer_strict"],
            "selected_cache_path": "cache.json",
            "command_groups": [
                {
                    "group_id": "y2020_d180_s44_n240",
                    "years": [2020],
                    "days": [180],
                    "seeds": [44],
                    "expected_scenario_ids": ["y2020_d180_s44_n240"],
                }
            ],
        }
        coverage = {
            "cache_coverage_pass": True,
            "coverage_rate": 1.0,
            "missing_key_count": 0,
            "missing_buffered_action_count": 0,
            "missing_parsed_plan_count": 0,
        }
        report = build_report(
            manifest=manifest,
            coverage=coverage,
            admission_review={
                "minimal_controlled_canary_admission_pass": True,
                "overlay_runner_path": "overlay/run_frozen_benchmark.py",
                "protocol_v1_overlay_root": "overlay",
            },
            authorization_source="user_delegated_triggered_controlled_canary_authorization_20260525",
            output_root="out",
        )

        self.assertTrue(report["triggered_controlled_canary_authorized"])
        self.assertEqual(report["controlled_replay_scope"], "triggered_controlled_canary_only")
        self.assertEqual(len(report["planned_commands"]), 1)
        self.assertIn("llm_rspc_v2,llm_rspc_v2_hot_dry_proposer_strict", report["planned_commands"][0])
        self.assertFalse(report["performance_claim_allowed"])

    def test_execution_record_blocks_on_missing_cache_coverage(self):
        report = build_report(
            manifest={"triggered_manifest_ready": True},
            coverage={"cache_coverage_pass": False},
            admission_review={"minimal_controlled_canary_admission_pass": True},
            authorization_source="user_delegated_triggered_controlled_canary_authorization_20260525",
            output_root="out",
        )

        self.assertFalse(report["triggered_controlled_canary_authorized"])
        self.assertEqual(report["next_action"], "triggered_canary_not_authorized")


if __name__ == "__main__":
    unittest.main()
