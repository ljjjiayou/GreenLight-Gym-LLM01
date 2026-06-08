import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.long_horizon_shadow_feasibility_result_audit import build_report


class TestLongHorizonShadowFeasibilityResultAudit(unittest.TestCase):
    def test_runtime_error_and_short_trace_block_horizon_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            trace_path = trace_dir / "y2006_d182_s42_n240_llm_rspc_v2.csv"
            with trace_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["step", "timestep", "runtime_error", "runtime_error_type", "replan_reason"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "step": "38",
                        "timestep": "38.0",
                        "runtime_error": 'CVode returned "CV_CONV_FAILURE".',
                        "runtime_error_type": "simulator_or_controller_error",
                        "replan_reason": "runtime_error",
                    }
                )
            report = build_report(
                execution_record={
                    "horizon_steps": 240,
                    "eligible_scenarios": [{"scenario_id": "y2006_d182_s42_n240", "year": 2006, "day": 182, "seed": 42}],
                },
                coverage={
                    "cache_coverage_pass": False,
                    "envs": {
                        "TomatoEnv_y2006_d182_s42": {
                            "ok": False,
                            "issues": ["last_timestep_too_early"],
                        }
                    },
                },
                trace_dir=trace_dir,
            )

        self.assertEqual(report["metrics"]["horizon_pass_count"], 0)
        self.assertIn("runtime_error", report["failure_taxonomy"])
        self.assertIn("short_trace_before_horizon", report["failure_taxonomy"])
        self.assertIn("cvodes_or_casadi_failure", report["failure_taxonomy"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_complete_trace_with_strict_source_counts_as_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            trace_path = trace_dir / "y2006_d182_s42_n240_llm_rspc_v2.csv"
            with trace_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "step",
                        "rspc_hot_dry_replay_would_apply",
                        "rspc_hot_dry_replay_safe_hot_dry",
                        "rspc_hot_dry_replay_unsafe_preferred",
                        "rspc_hot_dry_replay_unsafe_conflict",
                        "rspc_hot_dry_replay_best_margin",
                        "rspc_hot_dry_replay_best_candidate_name",
                        "rh_air",
                        "vpd_air",
                        "temp_air",
                        "canopy_dew_margin",
                    ],
                )
                writer.writeheader()
                for step in range(240):
                    writer.writerow(
                        {
                            "step": str(step),
                            "rspc_hot_dry_replay_would_apply": "true" if step == 10 else "false",
                            "rspc_hot_dry_replay_safe_hot_dry": "true",
                            "rspc_hot_dry_replay_unsafe_preferred": "false",
                            "rspc_hot_dry_replay_unsafe_conflict": "false",
                            "rspc_hot_dry_replay_best_margin": "0.25",
                            "rspc_hot_dry_replay_best_candidate_name": "shadow_hot_dry_humidity_retention",
                            "rh_air": "40.0",
                            "vpd_air": "2.5",
                            "temp_air": "28.0",
                            "canopy_dew_margin": "4.0",
                        }
                    )
            report = build_report(
                execution_record={
                    "horizon_steps": 240,
                    "eligible_scenarios": [{"scenario_id": "y2006_d182_s42_n240", "year": 2006, "day": 182, "seed": 42}],
                },
                coverage={
                    "cache_coverage_pass": True,
                    "envs": {"TomatoEnv_y2006_d182_s42": {"ok": True, "issues": []}},
                },
                trace_dir=trace_dir,
            )

        self.assertEqual(report["metrics"]["horizon_pass_count"], 1)
        self.assertEqual(report["metrics"]["shadow_signal_count"], 1)
        self.assertEqual(report["metrics"]["strict_source_candidate_count"], 240)
        self.assertEqual(report["next_action"], "build_v37_h720_execution_record")


if __name__ == "__main__":
    unittest.main()
