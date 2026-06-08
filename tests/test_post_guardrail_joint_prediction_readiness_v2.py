import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.post_guardrail_joint_prediction_readiness_v2 import build_report, build_trace_report, main


class TestPostGuardrailJointPredictionReadinessV2(unittest.TestCase):
    def test_missing_joint_fields_block_policy_judgment(self):
        report = build_report(
            vent_v2={
                "rows": [
                    {
                        "scenario_id": "y2020_d120_s44_n240",
                        "preset": "balanced",
                        "step": 221,
                        "predicted_canopy_margin_after_floor": 1.0,
                    }
                ]
            },
            warning_design={
                "rows": [
                    {
                        "scenario_id": "y2020_d120_s44_n240",
                        "preset": "hot_dry_relief",
                        "step": 233,
                    }
                ]
            },
        )

        self.assertFalse(report["ready_for_policy_judgment"])
        self.assertEqual(report["row_count"], 2)
        self.assertIn("predicted_vpd_after_floor", report["missing_field_counts"])
        self.assertIn("regime_conflict_flag", report["missing_field_counts"])
        self.assertIn("joint_prediction", report["rows"][0])

    def test_complete_joint_fields_only_allows_shadow_readiness_not_controlled_replay(self):
        complete = {
            "scenario_id": "x",
            "preset": "balanced",
            "step": 1,
            "predicted_canopy_margin_after_floor": 1.0,
            "predicted_dew_margin_after_floor": 2.0,
            "predicted_rh_after_floor": 60.0,
            "predicted_vpd_after_floor": 1.2,
            "predicted_temp_after_floor": 24.0,
            "predicted_dry_risk_after_floor": False,
            "regime_conflict_flag": False,
        }

        report = build_report(vent_v2={"rows": [complete]}, warning_design={"rows": []})

        self.assertTrue(report["ready_for_policy_judgment"])
        self.assertTrue(report["ready_for_shadow_audit"])
        self.assertFalse(report["controlled_replay_allowed"])

    def test_trace_rows_map_final_action_prediction_fields(self):
        report = build_trace_report(
            trace_rows=[
                {
                    "step": "1",
                    "rspc_action_selected_predicted_canopy_dew_margin_next": "3.2",
                    "rspc_action_selected_predicted_dew_margin_air_next": "1.4",
                    "rspc_action_selected_predicted_rh_next": "82.0",
                    "rspc_action_selected_predicted_vpd_next": "0.4",
                    "rspc_action_selected_predicted_temp_next": "17.5",
                    "dry_risk": "False",
                    "final_action_screen_vent_conflict": "False",
                }
            ]
        )

        self.assertTrue(report["ready_for_shadow_audit"])
        self.assertEqual(report["missing_field_counts"], {})
        self.assertEqual(report["row_count"], 1)

    def test_cli_reads_trace_dir_csv_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "traces"
            trace_dir.mkdir()
            output_json = root / "readiness.json"
            output_md = root / "readiness.md"
            fields = [
                "step",
                "rspc_action_selected_predicted_canopy_dew_margin_next",
                "rspc_action_selected_predicted_dew_margin_air_next",
                "rspc_action_selected_predicted_rh_next",
                "rspc_action_selected_predicted_vpd_next",
                "rspc_action_selected_predicted_temp_next",
                "dry_risk",
                "final_action_screen_vent_conflict",
            ]
            for name in ["a.csv", "b.csv"]:
                with (trace_dir / name).open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "step": "1",
                            "rspc_action_selected_predicted_canopy_dew_margin_next": "3.2",
                            "rspc_action_selected_predicted_dew_margin_air_next": "1.4",
                            "rspc_action_selected_predicted_rh_next": "82.0",
                            "rspc_action_selected_predicted_vpd_next": "0.4",
                            "rspc_action_selected_predicted_temp_next": "17.5",
                            "dry_risk": "False",
                            "final_action_screen_vent_conflict": "False",
                        }
                    )

            rc = main(
                [
                    "--trace-dir",
                    str(trace_dir),
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ]
            )

            self.assertEqual(rc, 0)
            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(report["ready_for_shadow_audit"])
            self.assertEqual(report["row_count"], 2)


if __name__ == "__main__":
    unittest.main()
