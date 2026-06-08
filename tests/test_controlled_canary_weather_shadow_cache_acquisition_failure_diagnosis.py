import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.controlled_canary_weather_shadow_cache_acquisition_failure_diagnosis import (
    build_report,
)


class TestControlledCanaryWeatherShadowCacheAcquisitionFailureDiagnosis(unittest.TestCase):
    def test_runtime_error_and_short_trace_block_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            trace_path = trace_dir / "y2006_d197_s42_n240_llm_rspc_v2.csv"
            with trace_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "step",
                        "timestep",
                        "replan_reason",
                        "runtime_error",
                        "runtime_error_type",
                        "rollout_source_before_error",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "step": "38",
                        "timestep": "38.0",
                        "replan_reason": "runtime_error",
                        "runtime_error": 'CVode returned "CV_CONV_FAILURE".',
                        "runtime_error_type": "simulator_or_controller_error",
                        "rollout_source_before_error": "rule",
                    }
                )

            report = build_report(
                coverage={
                    "cache_coverage_pass": False,
                    "coverage_rate": 1.0,
                    "missing_key_count": 1,
                    "missing_buffered_action_count": 0,
                    "missing_parsed_plan_count": 0,
                    "envs": {
                        "TomatoEnv_y2006_d197_s42": {
                            "entry_count": 4,
                            "required_min_entries": 20,
                            "max_timestep": 36,
                            "issues": ["last_timestep_too_early", "entry_count_below_expected"],
                        }
                    },
                },
                trace_dir=trace_dir,
                max_steps=240,
            )

        self.assertFalse(report["strict_metadata_replay_admission_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])
        self.assertIn("runtime_error_during_shadow_cache_acquisition", report["failure_taxonomy"])
        self.assertIn("cvodes_convergence_failure", report["failure_taxonomy"])
        self.assertIn("short_trace_before_max_steps", report["failure_taxonomy"])
        self.assertEqual(report["metrics"]["runtime_error_trace_count"], 1)


if __name__ == "__main__":
    unittest.main()
