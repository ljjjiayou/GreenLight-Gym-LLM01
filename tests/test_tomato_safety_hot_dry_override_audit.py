import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.tomato_safety_hot_dry_override_audit import (
    audit_trace,
    audit_traces,
    build_report,
    classify_override_pattern,
)


def _vec(heat=0.0, co2=0.0, screen=0.70, vent=0.25, lamp=0.0, shade=0.70):
    return json.dumps([heat, co2, screen, vent, lamp, shade], separators=(",", ":"))


def _row(**extra):
    data = {
        "step": 0,
        "rspc_action_audit_enabled": True,
        "rspc_action_layer_classification": "post_score_override_gap",
        "rspc_action_post_shape_safety_gate_reason": "none",
        "rspc_action_hot_dry_semantic_active": True,
        "rspc_action_hot_dry_proposer_active": True,
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
        "tomato_safety_v2_applied": True,
        "tomato_safety_v2_reasons": "hot_dry_cooling_guard,hot_dry_radiation_relief",
        "tomato_safety_v2_before": _vec(screen=0.70, vent=0.25, shade=0.70),
        "tomato_safety_v2_after": _vec(screen=0.50, vent=0.50, shade=0.70),
        "rh_air": 50.0,
        "vpd_air": 1.8,
        "temp_air": 29.0,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "rh_low_violation": 1.0,
        "vpd_high_excess": 0.1,
        "temp_violation": 0.0,
        "rh_high_violation": 0.0,
        "u_boil": 0.0,
        "u_co2": 0.0,
        "u_th_scr": 0.50,
        "u_ventilation": 0.50,
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


class TestTomatoSafetyHotDryOverrideAudit(unittest.TestCase):
    def test_true_dry_worsening_release(self):
        result = classify_override_pattern(_row())

        self.assertTrue(result["target"])
        self.assertEqual(result["pattern"], "true_dry_worsening_release")
        self.assertTrue(result["calibration_candidate"])

    def test_vent_only_dry_push(self):
        result = classify_override_pattern(
            _row(
                tomato_safety_v2_before=_vec(screen=0.70, vent=0.25, shade=0.70),
                tomato_safety_v2_after=_vec(screen=0.72, vent=0.48, shade=0.72),
                u_th_scr=0.72,
                u_ventilation=0.48,
                u_bl_scr=0.72,
            )
        )

        self.assertEqual(result["pattern"], "vent_only_dry_push")

    def test_canopy_reserve_relief(self):
        result = classify_override_pattern(
            _row(
                tomato_safety_v2_reasons="hot_dry_cooling_guard,canopy_dew_hot_dry_conflict_buffer",
                tomato_safety_v2_before=_vec(screen=0.45, vent=0.20, shade=0.45),
                tomato_safety_v2_after=_vec(screen=0.62, vent=0.48, shade=0.50),
                u_th_scr=0.62,
                u_ventilation=0.48,
                u_bl_scr=0.50,
            )
        )

        self.assertFalse(result["target"])
        self.assertEqual(result["pattern"], "not_target")

    def test_balanced_hot_dry_relief(self):
        result = classify_override_pattern(
            _row(
                tomato_safety_v2_before=_vec(screen=0.45, vent=0.20, shade=0.40),
                tomato_safety_v2_after=_vec(screen=0.68, vent=0.50, shade=0.78),
                u_th_scr=0.68,
                u_ventilation=0.50,
                u_bl_scr=0.78,
            )
        )

        self.assertFalse(result["target"])
        self.assertEqual(result["pattern"], "not_target")

    def test_non_target_row_is_ignored(self):
        result = classify_override_pattern(_row(rspc_action_layer_classification="candidate_gap"))

        self.assertFalse(result["target"])
        self.assertEqual(result["pattern"], "not_target")

    def test_audit_trace_and_report_are_stable(self):
        rows = [
            _row(step=0),
            _row(
                step=1,
                tomato_safety_v2_before=_vec(screen=0.45, vent=0.20, shade=0.40),
                tomato_safety_v2_after=_vec(screen=0.68, vent=0.50, shade=0.78),
                u_th_scr=0.68,
                u_ventilation=0.50,
                u_bl_scr=0.78,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm_rspc_v2.csv"
            _write_csv(path, rows)

            trace = audit_trace(path)
            audit = audit_traces([tmp])

        self.assertEqual(trace["target_steps"], 1)
        self.assertEqual(trace["pattern_counts"]["true_dry_worsening_release"], 1)
        self.assertIn("Tomato Safety Hot-Dry Override Narrow Audit", build_report(audit))


if __name__ == "__main__":
    unittest.main()
