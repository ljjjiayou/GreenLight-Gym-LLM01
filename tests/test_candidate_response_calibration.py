import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.candidate_response_calibration import build_report, calibrate_trace


def _selected_candidate(include_canopy_prediction=False, include_full_prediction=False):
    terms = {"temp_next": 21.0, "rh_next": 71.0, "vpd_next": 0.8}
    if include_canopy_prediction:
        terms["canopy_dew_margin_next"] = 0.2
    if include_full_prediction:
        terms.update(
            {
                "prediction_schema_version": "canopy_proxy_schema_v2",
                "prediction_horizon_steps": 1,
                "predicted_temp_next": 21.0,
                "predicted_rh_next": 71.0,
                "predicted_vpd_next": 0.8,
                "predicted_dew_margin_air_next": 1.5,
                "predicted_canopy_dew_margin_next": 0.2,
                "predicted_dew_lt0": False,
                "predicted_canopy_lt0": False,
                "prediction_model": "proxy_v1",
                "prediction_model_v2": "proxy_v2_conservative",
                "predicted_canopy_dew_margin_next_v2": -0.05,
                "predicted_canopy_lt0_v2": True,
                "predicted_canopy_warning_v2": True,
            }
        )
    return {"name": "selected", "selected": True, "score_terms": terms}


def _write_trace(path: Path, *, include_canopy_prediction=False, include_full_prediction=False):
    fields = [
        "step",
        "rspc_action_selected_name",
        "rspc_action_candidates_json",
        "temp_air",
        "rh_air",
        "vpd_air",
        "dew_margin_air",
        "canopy_dew_margin",
        "u_screen",
        "u_ventilation",
        "u_shading",
        "u_heating",
        "u_co2",
        "u_lighting",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(4):
            writer.writerow(
                {
                    "step": step,
                    "rspc_action_selected_name": "selected",
                    "rspc_action_candidates_json": json.dumps(
                        [_selected_candidate(include_canopy_prediction, include_full_prediction)]
                    ),
                    "temp_air": 20 + step,
                    "rh_air": 70,
                    "vpd_air": 0.9,
                    "dew_margin_air": 2,
                    "canopy_dew_margin": -0.1 if step == 1 and (include_canopy_prediction or include_full_prediction) else 2,
                    "u_screen": 0.8 if step == 1 else 0.2,
                    "u_ventilation": 0.1 if step == 1 else 0.4,
                    "u_shading": 0,
                    "u_heating": 0,
                    "u_co2": 0,
                    "u_lighting": 0,
                }
            )


class TestCandidateResponseCalibration(unittest.TestCase):
    def test_reports_missing_canopy_prediction_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            _write_trace(path, include_canopy_prediction=False)
            report = calibrate_trace(trace_path=path, preset="p", scenario_id="s", start_step=0, end_step=2)

        self.assertTrue(report["missing_prediction_metadata"])
        self.assertEqual(report["metric_summary"]["canopy_dew_margin"]["prediction_count"], 0)

    def test_counts_false_safe_canopy_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            _write_trace(path, include_canopy_prediction=True)
            report = calibrate_trace(trace_path=path, preset="p", scenario_id="s", start_step=0, end_step=2)

        self.assertEqual(report["predicted_canopy_count"], 3)
        self.assertGreaterEqual(report["false_safe_canopy_count"], 1)

    def test_reads_new_proxy_prediction_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            _write_trace(path, include_full_prediction=True)
            report = calibrate_trace(trace_path=path, preset="p", scenario_id="s", start_step=0, end_step=2)

        self.assertFalse(report["missing_prediction_metadata"])
        self.assertEqual(report["prediction_schema_counts"]["canonical"], 3)
        self.assertEqual(report["metric_summary"]["canopy_dew_margin"]["prediction_count"], 3)
        self.assertEqual(report["metric_summary"]["dew_margin"]["prediction_count"], 3)
        self.assertIn("near_boundary_summary", report)
        self.assertIn("proxy_error_distribution", report)
        self.assertIn("risk_buffer_adequacy", report)
        self.assertIn("proxy_v2_buffer_sensitivity", report)
        self.assertIn("buffer_scale_1", report["proxy_v2_buffer_sensitivity"])

    def test_legacy_prediction_fields_do_not_count_as_complete_canonical_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            _write_trace(path, include_canopy_prediction=True)
            report = calibrate_trace(trace_path=path, preset="p", scenario_id="s", start_step=0, end_step=2)

        self.assertTrue(report["missing_prediction_metadata"])
        self.assertEqual(report["prediction_schema_counts"]["legacy"], 3)

    def test_false_safe_details_include_v2_warning_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            _write_trace(path, include_full_prediction=True)
            report = calibrate_trace(trace_path=path, preset="p", scenario_id="s", start_step=0, end_step=2)

        self.assertGreaterEqual(report["false_safe_canopy_count"], 1)
        detail = report["false_safe_details"][0]
        self.assertIn("predicted_canopy_dew_margin_next_v2", detail)
        self.assertTrue(detail["predicted_canopy_warning_v2"])
        self.assertIn("post_guardrail_outcome", detail)
        self.assertIn("canopy_proxy_v2_risk_buffer", detail)
        self.assertEqual(report["v2_unwarned_false_safe_count"], 0)

    def test_build_report_reads_nested_preset_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "traces"
            for preset in ("balanced", "hot_dry_relief"):
                preset_dir = root / preset
                preset_dir.mkdir(parents=True)
                _write_trace(preset_dir / "case.csv", include_full_prediction=True)

            report = build_report(trace_dirs=[root], scenario_id="case", start_step=0, end_step=2)

        self.assertEqual(report["trace_count"], 2)
        self.assertEqual({trace["preset"] for trace in report["traces"]}, {"balanced", "hot_dry_relief"})


if __name__ == "__main__":
    unittest.main()
