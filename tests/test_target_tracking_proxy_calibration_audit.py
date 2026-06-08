import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector
from gl_gym.experiments.target_tracking_proxy_calibration_audit import (
    audit_trace,
    audit_traces,
    build_report,
    classify_row,
)


def _director(**config_overrides):
    director = object.__new__(RuleBasedLLMDirector)
    director.config = AgentConfig(**config_overrides)
    director.last_control = None
    return director


def _state(**overrides):
    data = {
        "temp_air": 29.0,
        "rh_air": 50.0,
        "co2_air": 430.0,
        "hour_of_day": 12.0,
        "glob_rad": 650.0,
        "temp_out": 24.0,
        "rh_out": 45.0,
        "temp_violation": 0.0,
        "rh_high_violation": 0.0,
        "rh_low_violation": 1.0,
        "vpd_high_excess": 0.1,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "fruit_weight": 0.0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _row(**extra):
    data = {
        "step": 0,
        "rspc_tt_calibration_enabled": True,
        "rspc_tt_calibration_shadow_only": True,
        "rspc_tt_calibration_triggered": True,
        "rspc_tt_calibration_reason": "target_tracking_vent_only_dry_push",
        "rspc_tt_calibration_safety_gate_reason": "none",
        "rspc_tt_calibration_safe_hot_dry": True,
        "rspc_tt_calibration_variant_count": 2,
        "rspc_tt_calibration_best_variant_name": "vent_cap",
        "rspc_tt_calibration_best_improvement_margin": 0.25,
        "rspc_tt_calibration_best_alignment": "dry_benefit",
        "rspc_tt_calibration_would_improve": True,
        "rspc_tt_calibration_unsafe_conflict": False,
        "rspc_tt_calibration_has_unsafe_variant": False,
        "rspc_tt_calibration_unsafe_variant_count": 0,
        "rspc_tt_calibration_variants_json": json.dumps(
            [
                {"name": "vent_cap", "improvement_margin": 0.25, "alignment": "dry_benefit"},
                {"name": "shade_buffer", "improvement_margin": 0.05, "alignment": "neutral_hold"},
            ],
            separators=(",", ":"),
        ),
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


class TestTargetTrackingProxyCalibrationShadow(unittest.TestCase):
    def test_helper_triggers_on_safe_hot_dry_vent_only_push_without_mutation(self):
        director = _director()
        state = _state()
        before = np.array([0.0, 0.0, 0.70, 0.25, 0.0, 0.70], dtype=np.float32)
        current = np.array([0.0, 0.0, 0.72, 0.48, 0.0, 0.72], dtype=np.float32)
        before_copy = before.copy()
        current_copy = current.copy()
        current_score, current_details = director._score_fallback_candidate(state, current)

        shadow = director._evaluate_target_tracking_calibration_shadow(
            state,
            before,
            current,
            current_score,
            current_details,
            gate_reason="none",
        )

        self.assertTrue(shadow["triggered"])
        self.assertEqual(shadow["reason"], "target_tracking_vent_only_dry_push")
        self.assertEqual(shadow["variant_count"], 2)
        self.assertEqual({item["name"] for item in shadow["variants"]}, {"vent_cap", "shade_buffer"})
        self.assertEqual(shadow["unsafe_conflict"], bool(shadow["best_alignment"] == "unsafe_conflict"))
        np.testing.assert_allclose(before, before_copy)
        np.testing.assert_allclose(current, current_copy)

    def test_helper_does_not_trigger_under_safety_gate(self):
        director = _director()
        current = np.array([0.0, 0.0, 0.72, 0.48, 0.0, 0.72], dtype=np.float32)
        current_score, current_details = director._score_fallback_candidate(_state(temp_air=33.0), current)

        shadow = director._evaluate_target_tracking_calibration_shadow(
            _state(temp_air=33.0),
            np.array([0.0, 0.0, 0.70, 0.25, 0.0, 0.70], dtype=np.float32),
            current,
            current_score,
            current_details,
            gate_reason="temp_high_gate",
        )

        self.assertFalse(shadow["triggered"])
        self.assertEqual(shadow["reason"], "safety_gate_active")
        self.assertEqual(shadow["variants"], [])

    def test_helper_does_not_trigger_when_relief_is_buffered(self):
        director = _director()
        current = np.array([0.0, 0.0, 0.84, 0.48, 0.0, 0.90], dtype=np.float32)
        current_score, current_details = director._score_fallback_candidate(_state(), current)

        shadow = director._evaluate_target_tracking_calibration_shadow(
            _state(),
            np.array([0.0, 0.0, 0.70, 0.25, 0.0, 0.70], dtype=np.float32),
            current,
            current_score,
            current_details,
            gate_reason="none",
        )

        self.assertFalse(shadow["triggered"])
        self.assertEqual(shadow["reason"], "no_vent_only_dry_push")


class TestTargetTrackingProxyCalibrationAudit(unittest.TestCase):
    def test_classifies_improving_safe_hot_dry_calibration(self):
        result = classify_row(_row())

        self.assertEqual(result["classification"], "calibrated_improves_safe_hot_dry")
        self.assertTrue(result["triggered"])
        self.assertEqual(result["best_variant_name"], "vent_cap")

    def test_classifies_no_gain(self):
        result = classify_row(
            _row(
                rspc_tt_calibration_best_improvement_margin=-0.05,
                rspc_tt_calibration_would_improve=False,
            )
        )

        self.assertEqual(result["classification"], "calibrated_no_gain")

    def test_classifies_unsafe_conflict(self):
        result = classify_row(
            _row(
                rspc_tt_calibration_unsafe_conflict=True,
                rspc_tt_calibration_has_unsafe_variant=True,
                rspc_tt_calibration_unsafe_variant_count=1,
                rspc_tt_calibration_variants_json=json.dumps(
                    [{"name": "vent_cap", "unsafe_conflict": True}],
                    separators=(",", ":"),
                ),
            )
        )

        self.assertEqual(result["classification"], "calibrated_unsafe_conflict")

    def test_nonbest_unsafe_variant_does_not_fail_row(self):
        result = classify_row(
            _row(
                rspc_tt_calibration_best_variant_name="shade_buffer",
                rspc_tt_calibration_best_improvement_margin=0.04,
                rspc_tt_calibration_would_improve=False,
                rspc_tt_calibration_unsafe_conflict=False,
                rspc_tt_calibration_has_unsafe_variant=True,
                rspc_tt_calibration_unsafe_variant_count=1,
                rspc_tt_calibration_variants_json=json.dumps(
                    [
                        {"name": "vent_cap", "unsafe_conflict": True, "improvement_margin": 7.0},
                        {"name": "shade_buffer", "unsafe_conflict": False, "improvement_margin": 0.04},
                    ],
                    separators=(",", ":"),
                ),
            )
        )

        self.assertEqual(result["classification"], "calibrated_no_gain")
        self.assertFalse(result["unsafe_conflict"])
        self.assertTrue(result["has_unsafe_variant"])

    def test_classifies_not_triggered_safety_gate(self):
        result = classify_row(
            _row(
                rspc_tt_calibration_triggered=False,
                rspc_tt_calibration_reason="safety_gate_active",
                rspc_tt_calibration_safety_gate_reason="temp_high_gate",
                rspc_tt_calibration_variant_count=0,
                rspc_tt_calibration_variants_json="[]",
            )
        )

        self.assertEqual(result["classification"], "not_triggered_safety_gate")

    def test_classifies_not_triggered_no_dry_push(self):
        result = classify_row(
            _row(
                rspc_tt_calibration_triggered=False,
                rspc_tt_calibration_reason="no_vent_only_dry_push",
                rspc_tt_calibration_variant_count=0,
                rspc_tt_calibration_variants_json="[]",
            )
        )

        self.assertEqual(result["classification"], "not_triggered_no_dry_push")

    def test_audit_trace_and_report_are_stable(self):
        rows = [
            _row(step=0),
            _row(step=1, rspc_tt_calibration_best_improvement_margin=-0.05, rspc_tt_calibration_would_improve=False),
            _row(step=2, rspc_tt_calibration_triggered=False, rspc_tt_calibration_reason="no_vent_only_dry_push"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm_rspc_v2.csv"
            _write_csv(path, rows)

            trace = audit_trace(path)
            audit = audit_traces([tmp])

        self.assertEqual(trace["metadata_steps"], 3)
        self.assertEqual(trace["triggered_steps"], 2)
        self.assertEqual(trace["improvement_steps"], 1)
        self.assertEqual(audit["aggregate"]["recommendation"]["decision"], "calibration_signal_detected")
        self.assertIn("Target Tracking Proxy Calibration Shadow Audit", build_report(audit))

    def test_missing_metadata_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm.csv"
            _write_csv(path, [{"step": 0, "reward": 0.0}])

            trace = audit_trace(path)

        self.assertEqual(trace["metadata_steps"], 0)
        self.assertIn("missing_target_tracking_calibration_metadata", trace["warnings"])
        self.assertEqual(trace["recommendation"]["decision"], "needs_trace_metadata")


if __name__ == "__main__":
    unittest.main()
