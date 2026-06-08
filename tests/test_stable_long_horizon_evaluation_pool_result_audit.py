import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.stable_long_horizon_evaluation_pool_result_audit import build_report


def _write_trace(path: Path, steps: int, *, runtime_error_step: int | None = None, cvodes: bool = False):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "runtime_error", "runtime_error_type", "replan_reason"])
        writer.writeheader()
        for step in range(steps):
            row = {"step": str(step), "runtime_error": "", "runtime_error_type": "", "replan_reason": ""}
            if runtime_error_step is not None and step == runtime_error_step:
                row["runtime_error"] = 'CVode returned "CV_CONV_FAILURE".' if cvodes else "runtime error"
                row["runtime_error_type"] = "simulator_or_controller_error"
                row["replan_reason"] = "runtime_error"
            writer.writerow(row)


class TestStableLongHorizonEvaluationPoolResultAudit(unittest.TestCase):
    def test_attribution_distinguishes_failure_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            _write_trace(trace_dir / "y2010_d59_s42_n240_ppo.csv", 240)
            _write_trace(trace_dir / "y2010_d59_s42_n240_llm_rspc_v2.csv", 240)
            _write_trace(trace_dir / "y2006_d160_s42_n240_ppo.csv", 40, runtime_error_step=39, cvodes=True)
            _write_trace(trace_dir / "y2006_d160_s42_n240_llm_rspc_v2.csv", 38, runtime_error_step=37, cvodes=True)
            _write_trace(trace_dir / "y2006_d161_s42_n240_ppo.csv", 240)
            _write_trace(trace_dir / "y2006_d161_s42_n240_llm_rspc_v2.csv", 30, runtime_error_step=29)
            report = build_report(
                execution_record={
                    "horizon_steps": 240,
                    "eligible_scenarios": [
                        {"scenario_id": "y2010_d59_s42_n240", "year": 2010, "day": 59, "seed": 42},
                        {"scenario_id": "y2006_d160_s42_n240", "year": 2006, "day": 160, "seed": 42},
                        {"scenario_id": "y2006_d161_s42_n240", "year": 2006, "day": 161, "seed": 42},
                    ],
                },
                trace_dir=trace_dir,
            )

        attributions = {item["scenario_id"]: item["attribution"] for item in report["scenario_reports"]}
        self.assertEqual(attributions["y2010_d59_s42_n240"], "stable_long_horizon_evaluation_candidate")
        self.assertEqual(attributions["y2006_d160_s42_n240"], "simulator_or_weather_numerical_instability")
        self.assertEqual(attributions["y2006_d161_s42_n240"], "default_controller_action_induced_instability")
        self.assertEqual(report["metrics"]["stable_scenario_count"], 1)
        self.assertEqual(report["metrics"]["cvodes_failure_count"], 2)
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_cache_coverage_failure_blocks_stable_pool_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            _write_trace(trace_dir / "y2010_d59_s42_n240_ppo.csv", 240)
            _write_trace(trace_dir / "y2010_d59_s42_n240_llm_rspc_v2.csv", 240)
            report = build_report(
                execution_record={
                    "horizon_steps": 240,
                    "eligible_scenarios": [
                        {"scenario_id": "y2010_d59_s42_n240", "year": 2010, "day": 59, "seed": 42},
                    ],
                },
                trace_dir=trace_dir,
                coverage_reports=[
                    {
                        "envs": {
                            "TomatoEnv_y2010_d59_s42": {
                                "ok": False,
                                "issues": ["last_timestep_too_early"],
                            }
                        }
                    }
                ],
            )

        self.assertEqual(report["metrics"]["stable_scenario_count"], 0)
        self.assertEqual(report["metrics"]["cache_coverage_incomplete_count"], 1)
        self.assertIn("cache_coverage_incomplete", report["failure_taxonomy"])


if __name__ == "__main__":
    unittest.main()
