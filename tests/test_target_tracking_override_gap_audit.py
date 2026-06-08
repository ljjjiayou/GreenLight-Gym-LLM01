import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.target_tracking_override_gap_audit import (
    audit_trace,
    audit_traces,
    build_report,
    classify_tracking_gap,
)


def _row(**extra):
    data = {
        "step": 0,
        "rspc_action_audit_enabled": True,
        "rspc_action_layer_classification": "post_score_override_gap",
        "rspc_action_selected_heat": 0.0,
        "rspc_action_selected_co2": 0.0,
        "rspc_action_selected_screen": 0.70,
        "rspc_action_selected_vent": 0.25,
        "rspc_action_selected_lamp": 0.0,
        "rspc_action_selected_shade": 0.70,
        "rspc_action_post_shape_selected_heat": 0.0,
        "rspc_action_post_shape_selected_co2": 0.0,
        "rspc_action_post_shape_selected_screen": 0.70,
        "rspc_action_post_shape_selected_vent": 0.25,
        "rspc_action_post_shape_selected_lamp": 0.0,
        "rspc_action_post_shape_selected_shade": 0.70,
        "rspc_action_post_shape_best_heat": 0.0,
        "rspc_action_post_shape_best_co2": 0.0,
        "rspc_action_post_shape_best_screen": 0.70,
        "rspc_action_post_shape_best_vent": 0.25,
        "rspc_action_post_shape_best_lamp": 0.0,
        "rspc_action_post_shape_best_shade": 0.70,
        "rspc_action_post_shape_safety_gate_reason": "none",
        "rspc_action_post_shape_dry_benefit": False,
        "rspc_action_post_shape_best_is_proposer": False,
        "rspc_action_dry_recovery_applied": False,
        "tomato_safety_v2_applied": False,
        "rh_air": 50.0,
        "vpd_air": 1.8,
        "temp_air": 29.0,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "rh_low_violation": 1.0,
        "vpd_high_excess": 0.1,
        "temp_violation": 0.0,
        "rh_high_violation": 0.0,
        "rspc_action_hot_dry_semantic_active": True,
        "u_boil": 0.0,
        "u_co2": 0.0,
        "u_th_scr": 0.70,
        "u_ventilation": 0.25,
        "u_lamp": 0.0,
        "u_bl_scr": 0.70,
    }
    data.update(extra)
    return data


def _write_csv(path, rows):
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestTargetTrackingOverrideGapAudit(unittest.TestCase):
    def test_classifies_vent_only_dry_push_as_calibration_candidate(self):
        result = classify_tracking_gap(
            _row(
                rspc_action_post_shape_selected_vent=0.48,
                u_ventilation=0.48,
            )
        )

        self.assertTrue(result["target"])
        self.assertEqual(result["pattern"], "tracking_vent_only_dry_push")
        self.assertTrue(result["calibration_candidate"])

    def test_classifies_screen_or_shade_release_as_calibration_candidate(self):
        result = classify_tracking_gap(
            _row(
                rspc_action_post_shape_selected_screen=0.54,
                rspc_action_post_shape_selected_vent=0.28,
                u_th_scr=0.54,
                u_ventilation=0.28,
            )
        )

        self.assertEqual(result["pattern"], "tracking_screen_or_shade_release")
        self.assertTrue(result["calibration_candidate"])

    def test_classifies_buffered_relief_as_likely_safe_or_minor(self):
        result = classify_tracking_gap(
            _row(
                rspc_action_post_shape_selected_screen=0.86,
                rspc_action_post_shape_selected_vent=0.50,
                u_th_scr=0.86,
                u_ventilation=0.50,
            )
        )

        self.assertEqual(result["pattern"], "tracking_buffered_relief")
        self.assertTrue(result["likely_safe_or_minor"])
        self.assertFalse(result["calibration_candidate"])

    def test_classifies_energy_or_co2_shift_as_likely_safe_or_minor(self):
        result = classify_tracking_gap(
            _row(
                rspc_action_post_shape_selected_heat=0.18,
                u_boil=0.18,
            )
        )

        self.assertEqual(result["pattern"], "tracking_energy_or_co2_shift")
        self.assertTrue(result["likely_safe_or_minor"])

    def test_non_target_rows_are_ignored(self):
        result = classify_tracking_gap(
            _row(
                rspc_action_layer_classification="candidate_gap",
                rspc_action_post_shape_selected_vent=0.48,
                u_ventilation=0.48,
            )
        )

        self.assertFalse(result["target"])
        self.assertEqual(result["pattern"], "not_target")

    def test_audit_trace_and_report_are_stable(self):
        rows = [
            _row(step=0, rspc_action_post_shape_selected_vent=0.48, u_ventilation=0.48),
            _row(
                step=1,
                rspc_action_post_shape_selected_screen=0.86,
                rspc_action_post_shape_selected_vent=0.50,
                u_th_scr=0.86,
                u_ventilation=0.50,
            ),
            _row(step=2, rspc_action_layer_classification="candidate_gap"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm_rspc_v2.csv"
            _write_csv(path, rows)

            trace = audit_trace(path)
            audit = audit_traces([tmp])

        self.assertEqual(trace["target_steps"], 2)
        self.assertEqual(trace["calibration_candidate_steps"], 1)
        self.assertEqual(trace["likely_safe_or_minor_steps"], 1)
        self.assertIn("Target Tracking Override Gap Audit", build_report(audit))


if __name__ == "__main__":
    unittest.main()
