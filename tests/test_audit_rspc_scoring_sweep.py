import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.audit_rspc_scoring_sweep import (
    audit_sweep,
    build_markdown_report,
    main,
)


def _row(preset, scenario="s1", **metrics):
    defaults = {
        "preset": preset,
        "scenario_id": scenario,
        "controller": "llm_rspc_v2",
        "total_reward": 100.0,
        "total_profit": 0.0,
        "total_rh_low_violation": 10.0,
        "total_vpd_high_excess": 5.0,
        "total_temp_violation": 0.0,
        "dew_margin_air_lt0_steps": 0,
        "canopy_dew_margin_lt0_steps": 0,
        "canopy_dew_margin_lt1_steps": 0,
        "runtime_error_steps": 0,
        "strict_cache_miss_runtime_error_steps": 0,
        "plan_cache_enabled_steps": 240,
        "plan_cache_hit_steps": 240,
        "plan_cache_hit_rate": 1.0,
    }
    defaults.update(metrics)
    return defaults


def _write_sweep(path, rows):
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")


class TestAuditRspcScoringSweep(unittest.TestCase):
    def test_recommends_clean_improving_preset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.json"
            _write_sweep(
                path,
                [
                    _row("balanced"),
                    _row("hot_dry_relief", total_reward=101.0, total_rh_low_violation=8.0, total_vpd_high_excess=4.0),
                ],
            )

            audit = audit_sweep(input_paths=[path])

        self.assertEqual(audit["aggregate"]["decision"], "pass")
        self.assertEqual(audit["recommendation"]["best_safe_preset"], "hot_dry_relief")
        self.assertFalse(audit["recommendation"]["merge_recommendation"])
        self.assertEqual(audit["recommendation"]["evidence_level"], "small_holdout_positive_signal")
        self.assertEqual(audit["recommendation"]["next_action"], "expand_validation")
        self.assertEqual(audit["recommendation"]["recommended_next_validation_candidate"], "hot_dry_relief")
        self.assertGreater(audit["recommendation"]["positive_scenario_ratio"], 0.0)
        self.assertAlmostEqual(audit["paired_deltas"][0]["d_total_rh_low_violation"], -2.0)
        self.assertIn("relative_d_total_rh_low_violation_pct", audit["paired_deltas"][0])
        report = build_markdown_report(audit)
        self.assertIn("RSPC Scoring Sweep Preset Audit", report)
        self.assertIn("Recommended next validation candidate", report)

    def test_rejects_safety_and_protocol_regressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.json"
            _write_sweep(
                path,
                [
                    _row("balanced"),
                    _row(
                        "hot_dry_relief",
                        runtime_error_steps=1,
                        plan_cache_hit_steps=239,
                        canopy_dew_margin_lt0_steps=1,
                    ),
                ],
            )

            audit = audit_sweep(input_paths=[path])

        self.assertEqual(audit["aggregate"]["decision"], "fail")
        reasons = {item["reason"] for item in audit["gate_failures"]}
        self.assertIn("runtime_error_steps", reasons)
        self.assertIn("plan_cache_hit_rate_below_1", reasons)
        self.assertIn("canopy_dew_margin_lt0_regression", reasons)
        self.assertEqual(audit["recommendation"]["best_safe_preset"], "balanced")
        self.assertEqual(audit["recommendation"]["evidence_level"], "rejected_due_to_gate")

    def test_supports_benchmark_json_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            balanced = tmp_path / "balanced.json"
            dry = tmp_path / "conservative_dry.json"
            balanced.write_text(
                json.dumps(
                    {
                        "summaries": [
                            {
                                "scenario_id": "s1",
                                "controller": "llm_rspc_v2",
                                "aggregate": _row("ignored"),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            dry.write_text(
                json.dumps(
                    {
                        "summaries": [
                            {
                                "scenario_id": "s1",
                                "controller": "llm_rspc_v2",
                                "aggregate": _row("ignored", total_rh_low_violation=9.0),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_sweep(benchmark_json_paths=[balanced, dry])

        self.assertEqual(audit["aggregate"]["paired_delta_count"], 1)
        self.assertEqual(audit["paired_deltas"][0]["preset"], "conservative_dry")

    def test_soft_warning_for_per_scenario_vpd_tradeoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.json"
            _write_sweep(
                path,
                [
                    _row("balanced"),
                    _row(
                        "hot_dry_relief",
                        total_reward=101.0,
                        total_rh_low_violation=8.0,
                        total_vpd_high_excess=5.1,
                    ),
                ],
            )

            audit = audit_sweep(input_paths=[path])

        reasons = {item["reason"] for item in audit["soft_warnings"]}
        self.assertIn("per_scenario_vpd_tradeoff", reasons)
        self.assertEqual(audit["aggregate"]["decision"], "pass")

    def test_small_effect_size_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sweep.json"
            _write_sweep(
                path,
                [
                    _row("balanced", total_reward=1000.0, total_rh_low_violation=1000.0, total_vpd_high_excess=1000.0),
                    _row(
                        "hot_dry_relief",
                        total_reward=1000.1,
                        total_rh_low_violation=999.8,
                        total_vpd_high_excess=999.9,
                    ),
                ],
            )

            audit = audit_sweep(input_paths=[path])

        self.assertTrue(audit["recommendation"]["effect_size_too_small"])
        self.assertEqual(audit["recommendation"]["next_action"], "expand_validation_with_effect_size_caution")
        self.assertFalse(audit["recommendation"]["performance_claim_allowed"])
        self.assertIn("Do not claim performance improvement", build_markdown_report(audit))

    def test_cli_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "sweep.json"
            output_json = tmp_path / "audit.json"
            output_md = tmp_path / "audit.md"
            _write_sweep(input_path, [_row("balanced"), _row("dew_safe")])

            code = main(
                [
                    "--input",
                    str(input_path),
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue(output_json.exists())
            self.assertTrue(output_md.exists())


if __name__ == "__main__":
    unittest.main()
