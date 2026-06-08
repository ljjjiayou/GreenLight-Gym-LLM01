import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.extract_rspc_scoring_delta_windows import extract_delta_windows, main


FIELDS = [
    "step",
    "temp_air",
    "rh_air",
    "vpd_air",
    "dew_margin_air",
    "canopy_dew_margin",
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
    "rh_low_violation",
    "vpd_high_excess",
    "temp_violation",
    "selected_fallback_candidate",
    "selected_fallback_score_breakdown",
    "tomato_safety_v2_applied",
    "tomato_safety_v2_reasons",
]


def _write_trace(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            full = {field: "" for field in FIELDS}
            full.update(row)
            writer.writerow(full)


class TestExtractRspcScoringDeltaWindows(unittest.TestCase):
    def test_extracts_useful_and_tradeoff_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "balanced"
            compare = root / "hot_dry_relief"
            trace_name = "y2015_d120_s42_n240_llm_rspc_v2.csv"
            base_rows = [
                {
                    "step": 1,
                    "u_ventilation": 0.5,
                    "u_shading": 0.0,
                    "rh_low_violation": 2.0,
                    "vpd_high_excess": 1.0,
                    "temp_violation": 0.0,
                    "selected_fallback_candidate": "base",
                    "selected_fallback_score_breakdown": json.dumps({"dry": 1.0}),
                },
                {
                    "step": 2,
                    "u_ventilation": 0.4,
                    "vpd_high_excess": 0.1,
                    "temp_violation": 0.0,
                },
            ]
            comp_rows = [
                {
                    "step": 1,
                    "u_ventilation": 0.3,
                    "u_shading": 0.2,
                    "rh_low_violation": 1.0,
                    "vpd_high_excess": 0.5,
                    "temp_violation": 0.0,
                    "selected_fallback_candidate": "dry_relief",
                    "selected_fallback_score_breakdown": json.dumps({"dry": 0.5}),
                },
                {
                    "step": 2,
                    "u_ventilation": 0.35,
                    "vpd_high_excess": 0.3,
                    "temp_violation": 0.0,
                },
            ]
            _write_trace(baseline / trace_name, base_rows)
            _write_trace(compare / trace_name, comp_rows)

            report = extract_delta_windows(baseline_trace_dir=baseline, compare_trace_dir=compare)

        self.assertEqual(report["trace_count"], 1)
        labels = {window["interpretation_label"] for window in report["windows"]}
        self.assertIn("useful_hot_dry_relief", labels)
        self.assertIn("vpd_tradeoff", labels)

    def test_cli_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "balanced"
            compare = root / "hot_dry_relief"
            output_json = root / "windows.json"
            output_md = root / "windows.md"
            trace_name = "trace.csv"
            _write_trace(baseline / trace_name, [{"step": 1, "u_ventilation": 0.5}])
            _write_trace(compare / trace_name, [{"step": 1, "u_ventilation": 0.3}])

            code = main(
                [
                    "--baseline-trace-dir",
                    str(baseline),
                    "--compare-trace-dir",
                    str(compare),
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue(output_json.exists())
            self.assertIn("RSPC Scoring Delta Windows", output_md.read_text(encoding="utf-8"))

    def test_derives_canopy_lt0_delta_from_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "balanced"
            compare = root / "hot_dry_relief"
            trace_name = "trace.csv"
            _write_trace(baseline / trace_name, [{"step": 1, "canopy_dew_margin": 0.2}])
            _write_trace(compare / trace_name, [{"step": 1, "canopy_dew_margin": -0.1}])

            report = extract_delta_windows(baseline_trace_dir=baseline, compare_trace_dir=compare)

            self.assertEqual(report["windows"][0]["local_metric_delta"]["canopy_dew_margin_lt0_steps"], 1.0)


if __name__ == "__main__":
    unittest.main()
