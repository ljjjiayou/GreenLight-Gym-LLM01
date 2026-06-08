import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.expanded_strict_metadata_replay_first_wave_execution_record import build_report


class TestExpandedStrictMetadataReplayFirstWaveExecutionRecord(unittest.TestCase):
    def _overlay(self, root: Path):
        runner = root / "gl_gym" / "experiments" / "run_frozen_benchmark.py"
        runner.parent.mkdir(parents=True)
        runner.write_text("print('ok')\n", encoding="utf-8")
        return {"protocol_v1_overlay_root": str(root), "protocol_v1_overlay_validated": True}

    def _manifest(self):
        return {
            "first_wave_manifest_ready": True,
            "scope": "first_wave_expanded_metadata_coverage",
            "selected_cache_path": "cache.json",
            "selected_scenarios": [
                {"scenario_id": "y2010_d120_s44_n240"},
                {"scenario_id": "y2010_d180_s44_n240"},
            ],
        }

    def _coverage(self):
        return {
            "cache_coverage_pass": True,
            "coverage_rate": 1.0,
            "missing_key_count": 0,
            "missing_buffered_action_count": 0,
            "missing_parsed_plan_count": 0,
            "cache_fill_run": False,
            "online_llm_called": False,
        }

    def test_precheck_authorizes_only_first_wave_metadata_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            baseline.mkdir()
            for scenario in ["y2010_d120_s44_n240", "y2010_d180_s44_n240"]:
                (baseline / f"{scenario}_llm_rspc_v2.csv").write_text("step,u_heating\n0,0\n", encoding="utf-8")

            report = build_report(
                scenario_manifest=self._manifest(),
                cache_coverage=self._coverage(),
                protocol_v1_overlay_preflight=self._overlay(root / "overlay"),
                authorization_source="user_delegated_first_wave_expanded_metadata_authorization_20260525",
                output_json_path="out.json",
                output_trace_dir="traces",
                baseline_trace_dir=str(baseline),
            )

        self.assertTrue(report["first_wave_expanded_metadata_replay_authorized"])
        self.assertTrue(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["counterexample_replay_allowed"])
        self.assertIn("--plan-cache-strict", report["planned_command"])

    def test_coverage_failure_blocks_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            baseline.mkdir()
            for scenario in ["y2010_d120_s44_n240", "y2010_d180_s44_n240"]:
                (baseline / f"{scenario}_llm_rspc_v2.csv").write_text("step,u_heating\n0,0\n", encoding="utf-8")
            coverage = self._coverage()
            coverage["cache_coverage_pass"] = False

            report = build_report(
                scenario_manifest=self._manifest(),
                cache_coverage=coverage,
                protocol_v1_overlay_preflight=self._overlay(root / "overlay"),
                authorization_source="user_delegated_first_wave_expanded_metadata_authorization_20260525",
                output_json_path="out.json",
                output_trace_dir="traces",
                baseline_trace_dir=str(baseline),
            )

        self.assertFalse(report["first_wave_expanded_metadata_replay_authorized"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertEqual(report["next_action"], "first_wave_precheck_failure")


if __name__ == "__main__":
    unittest.main()
