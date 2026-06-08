import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.controlled_canary_independent_wave2_execution_record import build_report


class TestControlledCanaryIndependentWave2ExecutionRecord(unittest.TestCase):
    def test_execution_record_fixes_scope_and_strict_replay_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            report = build_report(
                result_review={"wave1_result_review_pass": True},
                discovery={"independent_trigger_candidates_found": True},
                manifest={
                    "wave2_manifest_ready": True,
                    "scenario_ids": ["y2018_d120_s45_n240"],
                    "controllers": ["llm_rspc_v2", "llm_rspc_v2_hot_dry_proposer_strict"],
                    "selected_cache_path": "cache.json",
                    "command_groups": [
                        {
                            "group_id": "y2018_d120_s45_n240",
                            "years": [2018],
                            "days": [120],
                            "seeds": [45],
                        }
                    ],
                },
                coverage={
                    "cache_coverage_pass": True,
                    "coverage_rate": 1.0,
                    "missing_key_count": 0,
                    "missing_buffered_action_count": 0,
                    "missing_parsed_plan_count": 0,
                },
                admission_review={
                    "controlled_canary_expansion_admission_pass": True,
                    "overlay_runner_path": str(runner),
                    "protocol_v1_overlay_root": "overlay",
                },
                authorization_source="user_delegated_independent_triggered_holdout_wave2_authorization_20260525",
                output_root="out",
            )

        self.assertTrue(report["wave2_controlled_canary_authorized"])
        self.assertEqual(report["controlled_replay_scope"], "independent_triggered_holdout_wave2_only")
        command = report["planned_commands"][0]
        self.assertIn("--plan-cache-mode", command)
        self.assertIn("replay", command)
        self.assertIn("--plan-cache-strict", command)
        self.assertIn("--plan-cache-key-policy", command)
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_execution_record_blocks_when_cache_missing(self):
        report = build_report(
            result_review={"wave1_result_review_pass": True},
            discovery={"independent_trigger_candidates_found": True},
            manifest={"wave2_manifest_ready": True},
            coverage={"cache_coverage_pass": False},
            admission_review={"controlled_canary_expansion_admission_pass": True},
            authorization_source="source",
            output_root="out",
        )

        self.assertFalse(report["wave2_controlled_canary_authorized"])
        self.assertEqual(report["next_action"], "wave2_not_authorized")

    def test_execution_record_accepts_v32_shadow_sourced_result_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            report = build_report(
                result_review={"shadow_sourced_canary_result_review_pass": True},
                discovery={"independent_trigger_candidates_found": True},
                manifest={
                    "wave2_manifest_ready": True,
                    "scenario_ids": ["y2018_d120_s45_n240"],
                    "controllers": ["llm_rspc_v2", "llm_rspc_v2_hot_dry_proposer_strict"],
                    "selected_cache_path": "cache.json",
                    "command_groups": [
                        {
                            "group_id": "y2018_d120_s45_n240",
                            "years": [2018],
                            "days": [120],
                            "seeds": [45],
                        }
                    ],
                },
                coverage={
                    "cache_coverage_pass": True,
                    "coverage_rate": 1.0,
                    "missing_key_count": 0,
                    "missing_buffered_action_count": 0,
                    "missing_parsed_plan_count": 0,
                },
                admission_review={
                    "controlled_canary_expansion_admission_pass": True,
                    "overlay_runner_path": str(runner),
                    "protocol_v1_overlay_root": "overlay",
                },
                authorization_source="user_delegated_independent_triggered_holdout_wave2_authorization_20260528",
                output_root="out",
            )

        self.assertTrue(report["wave2_controlled_canary_authorized"])
        self.assertTrue(report["precheck"]["result_review_pass"])


if __name__ == "__main__":
    unittest.main()
