import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.expanded_metadata_second_wave_scenario_manifest import build_report


class TestExpandedMetadataSecondWaveScenarioManifest(unittest.TestCase):
    def _write_trace(self, root: Path, scenario_id: str):
        rows = [
            "step,temp_air,rh_air,vpd_air,dew_margin_air,canopy_dew_margin,glob_rad,wind_speed,dry_risk,dew_risk\n",
            "0,18.5,90.0,0.2,1.6,3.6,0,0,False,False\n",
            "29,25.4,56.0,1.4,9.2,3.5,300,4,True,False\n",
        ]
        (root / f"{scenario_id}_llm_rspc_v2.csv").write_text("".join(rows), encoding="utf-8")
        (root / f"{scenario_id}_llm_rspc_v2.jsonl").write_text("{}\n", encoding="utf-8")

    def test_manifest_selects_second_wave_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_trace(root, "y2015_d180_s44_n240")

            report = build_report(
                baseline_trace_dir=root,
                selected_cache_path="cache.json",
                candidate_scenario_ids=["y2015_d180_s44_n240"],
            )

        self.assertTrue(report["second_wave_manifest_ready"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["covered_families"], ["mixed_dry_dew", "pure_hot_dry"])
        self.assertEqual(report["scenario_count"], 1)
        self.assertGreater(report["selected_scenarios"][0]["pure_hot_dry_window_count"], 0)
        self.assertGreater(report["selected_scenarios"][0]["dew_or_high_humidity_window_count"], 0)

    def test_missing_dew_window_blocks_mixed_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "y2015_d180_s44_n240_llm_rspc_v2.csv").write_text(
                "step,temp_air,rh_air,vpd_air,dew_margin_air,canopy_dew_margin,dry_risk,dew_risk\n"
                "29,25.4,56.0,1.4,9.2,3.5,True,False\n",
                encoding="utf-8",
            )

            report = build_report(
                baseline_trace_dir=root,
                selected_cache_path="cache.json",
                candidate_scenario_ids=["y2015_d180_s44_n240"],
            )

        self.assertFalse(report["second_wave_manifest_ready"])
        self.assertIn("mixed_dry_dew", report["missing_families"])


if __name__ == "__main__":
    unittest.main()
