import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.sweep_rspc_scoring_weights import (
    PRESETS,
    build_report,
    build_run_command,
    main,
    parse_args,
)


class TestSweepRspcScoringWeights(unittest.TestCase):
    def test_build_command_includes_agent_config_overrides(self):
        args = parse_args(["--preset", "hot_dry_relief", "--dry-run"])
        command = build_run_command(args, "hot_dry_relief", PRESETS["hot_dry_relief"])

        self.assertIn("--agent-config-overrides", command)
        overrides = json.loads(command[command.index("--agent-config-overrides") + 1])
        self.assertEqual(overrides["fallback_dry_penalty_weight"], 1.8)
        self.assertIn("holdout_seed42_43_hot_dry_qwen_20260517_merged.json", " ".join(command))
        self.assertIn("--plan-cache-strict", command)

    def test_dry_run_writes_plan_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_json = Path(tmp) / "sweep.json"
            output_md = Path(tmp) / "sweep.md"
            code = main(
                [
                    "--preset",
                    "balanced",
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                    "--dry-run",
                ]
            )

            self.assertEqual(code, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["presets"], ["balanced"])
            self.assertIn("RSPC Scoring Weight Sweep", output_md.read_text(encoding="utf-8"))

    def test_dry_run_supports_selected_scenario_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scenario_json = tmp_path / "selected.json"
            output_json = tmp_path / "sweep.json"
            output_md = tmp_path / "sweep.md"
            scenario_json.write_text(
                json.dumps(
                    {
                        "selected_scenarios": [
                            {
                                "scenario_id": "y2015_d120_s42_n240",
                                "year": 2015,
                                "day": 120,
                                "seed": 42,
                                "max_steps": 240,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            code = main(
                [
                    "--preset",
                    "balanced",
                    "--scenario-list-json",
                    str(scenario_json),
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                    "--dry-run",
                ]
            )

            self.assertEqual(code, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["commands"]), 1)
            command = payload["commands"][0]["command"]
            self.assertIn("--years", command)
            self.assertIn("2015", command)
            self.assertIn("y2015_d120_s42_n240.json", " ".join(command))

    def test_report_includes_metrics_when_rows_exist(self):
        report = build_report(
            {
                "dry_run": False,
                "presets": ["balanced"],
                "commands": [],
                "rows": [
                    {
                        "preset": "balanced",
                        "scenario_id": "s1",
                        "controller": "llm_rspc_v2",
                        "total_reward": 1.0,
                        "total_profit": -0.1,
                        "total_rh_low_violation": 2.0,
                        "total_vpd_high_excess": 0.5,
                        "total_temp_violation": 0.0,
                        "plan_cache_hit_rate": 1.0,
                        "runtime_error_steps": 0,
                    }
                ],
            }
        )

        self.assertIn("Metrics", report)
        self.assertIn("llm_rspc_v2", report)


if __name__ == "__main__":
    unittest.main()
