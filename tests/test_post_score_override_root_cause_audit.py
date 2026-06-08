import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.post_score_override_root_cause_audit import (
    audit_trace,
    audit_traces,
    build_report,
    classify_row,
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
        "rspc_action_dry_recovery_before": "[]",
        "rspc_action_dry_recovery_after": "[]",
        "tomato_safety_v2_applied": False,
        "tomato_safety_v2_before": "[]",
        "tomato_safety_v2_after": "[]",
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
        "rspc_action_hot_dry_proposer_active": True,
        "u_boil": 0.0,
        "u_co2": 0.0,
        "u_th_scr": 0.70,
        "u_ventilation": 0.25,
        "u_lamp": 0.0,
        "u_bl_scr": 0.70,
    }
    data.update(extra)
    return data


def _vec(heat=0.0, co2=0.0, screen=0.70, vent=0.25, lamp=0.0, shade=0.70):
    return json.dumps([heat, co2, screen, vent, lamp, shade], separators=(",", ":"))


def _write_csv(path, rows):
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestPostScoreOverrideRootCauseAudit(unittest.TestCase):
    def test_dry_recovery_overrides_safe_hot_dry(self):
        result = classify_row(
            _row(
                rspc_action_dry_recovery_applied=True,
                rspc_action_dry_recovery_before=_vec(screen=0.72, vent=0.22, shade=0.75),
                rspc_action_dry_recovery_after=_vec(screen=0.50, vent=0.48, shade=0.75),
                u_th_scr=0.50,
                u_ventilation=0.48,
                u_bl_scr=0.75,
            )
        )

        self.assertEqual(result["classification"], "dry_recovery_overrides_safe_hot_dry")
        self.assertTrue(result["safe_hot_dry"])

    def test_tomato_safety_required_under_safety_gate(self):
        result = classify_row(
            _row(
                temp_air=33.0,
                temp_violation=1.0,
                rspc_action_post_shape_safety_gate_reason="temp_high_gate",
                tomato_safety_v2_applied=True,
                tomato_safety_v2_before=_vec(screen=0.80, vent=0.20, shade=0.20),
                tomato_safety_v2_after=_vec(screen=0.35, vent=0.90, shade=0.85),
                u_th_scr=0.35,
                u_ventilation=0.90,
                u_bl_scr=0.85,
            )
        )

        self.assertEqual(result["classification"], "tomato_safety_required_safety_relief")

    def test_tomato_safety_overrides_safe_hot_dry_without_gate(self):
        result = classify_row(
            _row(
                tomato_safety_v2_applied=True,
                tomato_safety_v2_before=_vec(screen=0.76, vent=0.24, shade=0.72),
                tomato_safety_v2_after=_vec(screen=0.54, vent=0.50, shade=0.72),
                u_th_scr=0.54,
                u_ventilation=0.50,
                u_bl_scr=0.72,
            )
        )

        self.assertEqual(result["classification"], "tomato_safety_overrides_safe_hot_dry")

    def test_buffered_tomato_hot_dry_relief_is_not_dry_worsening(self):
        result = classify_row(
            _row(
                tomato_safety_v2_applied=True,
                tomato_safety_v2_before=_vec(screen=0.45, vent=0.20, shade=0.40),
                tomato_safety_v2_after=_vec(screen=0.68, vent=0.50, shade=0.78),
                u_th_scr=0.68,
                u_ventilation=0.50,
                u_bl_scr=0.78,
            )
        )

        self.assertEqual(result["classification"], "no_material_override")

    def test_vent_only_tomato_push_remains_dry_worsening(self):
        result = classify_row(
            _row(
                tomato_safety_v2_applied=True,
                tomato_safety_v2_before=_vec(screen=0.70, vent=0.25, shade=0.70),
                tomato_safety_v2_after=_vec(screen=0.72, vent=0.48, shade=0.72),
                u_th_scr=0.72,
                u_ventilation=0.48,
                u_bl_scr=0.72,
            )
        )

        self.assertEqual(result["classification"], "tomato_safety_overrides_safe_hot_dry")

    def test_final_action_matches_shadow_benefit(self):
        result = classify_row(
            _row(
                rspc_action_post_shape_dry_benefit=True,
                rspc_action_post_shape_best_is_proposer=True,
                rspc_action_post_shape_best_screen=0.66,
                rspc_action_post_shape_best_vent=0.28,
                rspc_action_post_shape_best_shade=0.78,
                u_th_scr=0.65,
                u_ventilation=0.29,
                u_bl_scr=0.78,
            )
        )

        self.assertEqual(result["classification"], "final_action_matches_shadow_benefit")

    def test_missing_metadata_warns(self):
        result = classify_row({"step": 0})

        self.assertEqual(result["classification"], "missing_metadata")

    def test_audit_trace_and_report_are_stable(self):
        rows = [
            _row(step=0),
            _row(
                step=1,
                rspc_action_dry_recovery_applied=True,
                rspc_action_dry_recovery_before=_vec(screen=0.72, vent=0.22),
                rspc_action_dry_recovery_after=_vec(screen=0.50, vent=0.48),
                u_th_scr=0.50,
                u_ventilation=0.48,
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm_rspc_v2.csv"
            _write_csv(path, rows)

            trace = audit_trace(path)
            audit = audit_traces([tmp])

        self.assertEqual(trace["override_audit_steps"], 2)
        self.assertEqual(trace["classification_counts"]["dry_recovery_overrides_safe_hot_dry"], 1)
        self.assertIn("Post-Score Override Root Cause Audit", build_report(audit))


if __name__ == "__main__":
    unittest.main()
