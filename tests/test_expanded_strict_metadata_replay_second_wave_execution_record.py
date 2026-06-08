import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.expanded_strict_metadata_replay_second_wave_execution_record import build_report


class TestExpandedStrictMetadataReplaySecondWaveExecutionRecord(unittest.TestCase):
    def _overlay(self, root: Path):
        runner = root / "gl_gym" / "experiments" / "run_frozen_benchmark.py"
        runner.parent.mkdir(parents=True)
        runner.write_text("print('ok')\n", encoding="utf-8")
        return {"protocol_v1_overlay_root": str(root), "protocol_v1_overlay_validated": True}

    def _manifest(self):
        return {
            "second_wave_manifest_ready": True,
            "scope": "second_wave_expanded_metadata_coverage",
            "covered_families": ["mixed_dry_dew", "pure_hot_dry"],
            "selected_cache_path": "cache.json",
            "selected_scenarios": [
                {"scenario_id": "y2015_d180_s44_n240", "year": 2015, "day": 180, "seed": 44},
                {"scenario_id": "y2020_d180_s44_n240", "year": 2020, "day": 180, "seed": 44},
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

    def test_precheck_authorizes_only_second_wave_metadata_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            baseline.mkdir()
            for scenario in ["y2015_d180_s44_n240", "y2020_d180_s44_n240"]:
                (baseline / f"{scenario}_llm_rspc_v2.csv").write_text("step,u_heating\n0,0\n", encoding="utf-8")

            report = build_report(
                scenario_manifest=self._manifest(),
                cache_coverage=self._coverage(),
                protocol_v1_overlay_preflight=self._overlay(root / "overlay"),
                authorization_source="user_delegated_second_wave_expanded_metadata_authorization_20260525",
                output_json_path="out.json",
                output_trace_dir="traces",
                baseline_trace_dir=str(baseline),
            )

        self.assertTrue(report["second_wave_expanded_metadata_replay_authorized"])
        self.assertTrue(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["counterexample_replay_allowed"])
        self.assertTrue(report["runner_cartesian_product_safe"])
        self.assertIn("--plan-cache-strict", report["planned_command"])

    def test_cartesian_mismatch_blocks_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            baseline.mkdir()
            manifest = self._manifest()
            manifest["selected_scenarios"].append(
                {"scenario_id": "y2020_d240_s44_n240", "year": 2020, "day": 240, "seed": 44}
            )
            for scenario in [item["scenario_id"] for item in manifest["selected_scenarios"]]:
                (baseline / f"{scenario}_llm_rspc_v2.csv").write_text("step,u_heating\n0,0\n", encoding="utf-8")

            report = build_report(
                scenario_manifest=manifest,
                cache_coverage=self._coverage(),
                protocol_v1_overlay_preflight=self._overlay(root / "overlay"),
                authorization_source="user_delegated_second_wave_expanded_metadata_authorization_20260525",
                output_json_path="out.json",
                output_trace_dir="traces",
                baseline_trace_dir=str(baseline),
            )

        self.assertFalse(report["runner_cartesian_product_safe"])
        self.assertFalse(report["second_wave_expanded_metadata_replay_authorized"])
        self.assertEqual(report["next_action"], "second_wave_precheck_failure")


if __name__ == "__main__":
    unittest.main()
