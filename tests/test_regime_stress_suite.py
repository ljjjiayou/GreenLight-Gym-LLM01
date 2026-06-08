import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.build_regime_stress_suite import (
    DEFAULT_CONFIG_PATH,
    build_manifest,
    load_config,
    merge_regime_windows,
    tag_rows,
)
from gl_gym.experiments.run_regime_stress_suite import build_run_plan


def _row(step, **overrides):
    row = {
        "step": step,
        "hour_of_day": step / 4.0,
        "temp_air": 20.0,
        "rh_air": 70.0,
        "vpd_air": 0.8,
        "glob_rad": 0.0,
        "forecast_rad_peak_2h": 0.0,
        "wind_speed": 1.0,
        "dew_margin_min": 3.0,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "rh_low_violation": 0.0,
        "rh_high_violation": 0.0,
        "vpd_high_excess": 0.0,
        "temp_violation": 0.0,
        "source": "rule",
    }
    row.update(overrides)
    return row


def _write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestRegimeStressSuite(unittest.TestCase):
    def setUp(self):
        self.config = load_config(DEFAULT_CONFIG_PATH)

    def test_synthetic_trace_tags_core_extreme_regimes(self):
        rows = [
            _row(0, temp_air=31.0, rh_air=50.0, vpd_air=2.0, glob_rad=720.0),
            _row(1, temp_air=16.0, rh_air=91.0, vpd_air=0.30, dew_margin_min=0.8),
            _row(24, hour_of_day=6.0, rh_air=89.0, canopy_dew_margin=1.0),
            _row(40, temp_air=29.0, wind_speed=6.5, rh_air=70.0),
        ]

        tags = tag_rows(rows, self.config)

        self.assertIn("hot_dry", tags[0])
        self.assertIn("cold_humid", tags[1])
        self.assertIn("dawn_dew", tags[2])
        self.assertIn("wind_stress", tags[3])

    def test_window_merge_allows_one_step_gap(self):
        rows = [
            _row(0),
            _row(1, temp_air=31.0, rh_air=50.0, vpd_air=2.0, glob_rad=720.0),
            _row(2),
            _row(3, temp_air=32.0, rh_air=49.0, vpd_air=2.1, glob_rad=730.0),
            _row(4),
        ]
        tags = tag_rows(rows, self.config)

        windows = [w for w in merge_regime_windows(rows, tags, self.config) if w["regime"] == "hot_dry"]

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["start_step"], 1)
        self.assertEqual(windows[0]["end_step"], 3)
        self.assertEqual(windows[0]["duration_steps"], 3)

    def test_manifest_generation_is_deterministic_and_assigns_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_csv(
                tmp_path / "y2015_d180_s44_n240_llm_rspc_v2.csv",
                [_row(0, temp_air=31.0, rh_air=50.0, vpd_air=2.0, glob_rad=720.0)],
            )
            _write_csv(
                tmp_path / "y2015_d180_s44_n240_llm.csv",
                [_row(0, temp_air=31.0, rh_air=50.0, vpd_air=2.0, glob_rad=720.0)],
            )
            _write_csv(
                tmp_path / "y2015_d120_s42_n240_llm_rspc_v2.csv",
                [_row(0, temp_air=31.0, rh_air=50.0, vpd_air=2.0, glob_rad=720.0)],
            )

            first = build_manifest([tmp_path], config=self.config)
            second = build_manifest([tmp_path], config=self.config)

        self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
        by_id = {entry["scenario_id"]: entry for entry in first["scenarios"]}
        self.assertEqual(by_id["y2015_d180_s44_n240"]["split"], "regression")
        self.assertEqual(by_id["y2015_d120_s42_n240"]["split"], "holdout")
        self.assertEqual(by_id["y2015_d180_s44_n240"]["source_controller"], "llm_rspc_v2")

    def test_empty_and_missing_field_traces_warn_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _write_csv(tmp_path / "y2015_d120_s43_n240_llm_rspc_v2.csv", [], fieldnames=["step", "temp_air"])
            _write_csv(tmp_path / "y2015_d120_s44_n240_llm_rspc_v2.csv", [{"step": 0, "temp_air": 20.0}], fieldnames=["step", "temp_air"])

            manifest = build_manifest([tmp_path], config=self.config)

        warnings = "\n".join(manifest["warnings"])
        self.assertIn("empty_trace", warnings)
        self.assertIn("missing_fields", warnings)
        self.assertEqual(manifest["scenario_count"], 2)

    def test_generated_run_commands_include_strict_replay_plan_cache_and_controllers(self):
        manifest = {
            "schema_version": "regime_stress_manifest_v1",
            "scenarios": [
                {
                    "scenario_id": "y2015_d180_s44_n240",
                    "year": 2015,
                    "day": 180,
                    "seed": 44,
                    "max_steps": 240,
                    "split": "regression",
                    "regime_tags": ["hot_dry", "radiation_spike", "wind_stress"],
                }
            ],
        }

        plan = build_run_plan(
            manifest,
            manifest_path="manifest.json",
            output_dir="out",
            splits=["regression"],
            regimes=["hot_dry"],
            plan_cache_path="cache.json",
        )

        command = plan["groups"][0]["benchmark_commands"][0]["command"]
        self.assertIn("--plan-cache-strict", command)
        self.assertIn("--plan-cache-key-policy scenario_timestep", command)
        self.assertIn("--plan-cache-path cache.json", command)
        self.assertIn("--controllers llm,llm_rspc_v2,llm_sero_shadow,ppo", command)


if __name__ == "__main__":
    unittest.main()
