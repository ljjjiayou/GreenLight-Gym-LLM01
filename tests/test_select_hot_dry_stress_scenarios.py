import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.select_hot_dry_stress_scenarios import select_hot_dry_scenarios


def _write_cache(path: Path, scenarios):
    entries = {}
    idx = 0
    for year, day, seed in scenarios:
        for timestep in range(0, 240, 12):
            entries[f"k{idx}"] = {"env_id": f"TomatoEnv_y{year}_d{day}_s{seed}", "timestep": timestep}
            idx += 1
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")


def _write_trace(path: Path, hot_steps: int, rh_area: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "temp_air", "rh_air", "vpd_air", "glob_rad", "rh_low_violation", "vpd_high_excess", "temp_violation", "dew_margin_air", "canopy_dew_margin"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(20):
            hot = step < hot_steps
            writer.writerow(
                {
                    "step": step,
                    "temp_air": 31 if hot else 20,
                    "rh_air": 45 if hot else 70,
                    "vpd_air": 2.0 if hot else 0.8,
                    "glob_rad": 700 if hot else 100,
                    "rh_low_violation": rh_area / max(hot_steps, 1) if hot else 0,
                    "vpd_high_excess": 1 if hot else 0,
                    "temp_violation": 0,
                    "dew_margin_air": 2,
                    "canopy_dew_margin": 2,
                }
            )


class TestSelectHotDryStressScenarios(unittest.TestCase):
    def test_selection_is_ranked_and_requires_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            cache = root / "cache.json"
            trace_dir = root / "traces"
            manifest.write_text(json.dumps({"scenarios": []}), encoding="utf-8")
            _write_trace(trace_dir / "y2015_d120_s42_n240_llm_rspc_v2.csv", hot_steps=5, rh_area=10)
            _write_trace(trace_dir / "y2015_d180_s42_n240_llm_rspc_v2.csv", hot_steps=10, rh_area=20)
            _write_cache(cache, [(2015, 120, 42), (2015, 180, 42)])

            report = select_hot_dry_scenarios(
                manifest_path=manifest,
                trace_dirs=[trace_dir],
                plan_cache_path=cache,
                min_candidates=2,
                max_candidates=2,
            )

        self.assertTrue(report["can_execute_strict_replay"])
        self.assertEqual(report["selected_scenarios"][0]["scenario_id"], "y2015_d180_s42_n240")
        self.assertEqual(report["executable_count"], 2)
        self.assertIn("pure_hot_dry", report["selected_scenarios"][0]["regime_conflict_tags"])
        self.assertIn("radiation_dry", report["selected_scenarios"][0]["regime_conflict_tags"])


if __name__ == "__main__":
    unittest.main()
