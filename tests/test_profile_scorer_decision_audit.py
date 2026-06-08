import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.profile_scorer_decision_audit import (
    audit_trace,
    audit_traces,
    build_report,
    risk_flags,
)


def _write_csv(path, rows):
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(step, regime, selector, scorer, **extra):
    data = {
        "step": step,
        "intent_regime": regime,
        "profile_generator_selected_shadow_profile": selector,
        "profile_scorer_selected_shadow_profile": scorer,
        "profile_scorer_selected_score": 0.5,
        "profile_scorer_margin_to_second": 0.2,
        "profile_scorer_safety_gate_reason": "none",
        "profile_scorer_delta_target_temp": 0.0,
        "profile_scorer_delta_target_co2": 0.0,
        "profile_scorer_delta_target_rh": 0.0,
        "profile_scorer_dew_risk": 0.0,
        "profile_scorer_hot_dry_retention_gap": 0.0,
        "profile_scorer_humid_relief_gap": 0.0,
        "rh_air": 70.0,
        "vpd_air": 0.8,
        "temp_air": 24.0,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "rh_low_violation": 0.0,
        "vpd_high_excess": 0.0,
        "temp_violation": 0.0,
        "rh_high_violation": 0.0,
    }
    data.update(extra)
    return data


class TestProfileScorerDecisionAudit(unittest.TestCase):
    def test_risk_flags_merge_violation_metrics_and_state_thresholds(self):
        flags = risk_flags(
            {
                "rh_air": 60.0,
                "vpd_air": 1.8,
                "temp_air": 33.0,
                "rh_high_violation": 0.0,
                "dew_margin_air": 0.5,
                "canopy_dew_margin": 0.8,
            }
        )

        self.assertTrue(flags["rh_low"])
        self.assertTrue(flags["vpd_high"])
        self.assertTrue(flags["temp_high"])
        self.assertTrue(flags["dew_air"])
        self.assertTrue(flags["canopy_dew"])
        self.assertTrue(flags["dry_performance"])
        self.assertTrue(flags["safety"])

    def test_audit_detects_hot_dry_alignment_and_disagreement_windows(self):
        rows = []
        for step in range(12):
            rows.append(
                _row(
                    step,
                    "hot_dry_relief",
                    "hot_dry_protect",
                    "hot_dry_protect",
                    rh_air=55.0,
                    vpd_air=1.9,
                    profile_scorer_delta_target_rh=2.5,
                )
            )
        for step in range(12, 18):
            rows.append(
                _row(
                    step,
                    "hot_dry_relief",
                    "hot_dry_protect",
                    "constant_hold",
                    temp_air=33.0,
                    dew_margin_air=0.7,
                    canopy_dew_margin=0.6,
                    profile_scorer_dew_risk=3.0,
                    profile_scorer_safety_gate_reason="canopy_gate",
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d180_s44_n240_llm_rspc_v2.csv"
            _write_csv(path, rows)
            trace = audit_trace(path)
            audit = audit_traces([tmp])

        self.assertEqual(trace["scorer_steps"], 18)
        self.assertEqual(trace["scorer_selected_counts"]["hot_dry_protect"], 12)
        self.assertEqual(trace["risk_counts"]["dry_performance"], 12)
        self.assertEqual(trace["risk_counts"]["safety"], 6)
        self.assertEqual(len(trace["disagreement_windows"]), 1)
        self.assertEqual(trace["disagreement_windows"][0]["start_step"], 12)
        self.assertEqual(trace["risk_matrix"]["dry_performance"]["hot_dry_protect_steps"], 12)
        self.assertEqual(trace["safety_gate_counts"]["none"], 12)
        self.assertEqual(trace["safety_gate_counts"]["canopy_gate"], 6)
        self.assertEqual(trace["safety_gate_matrix"]["canopy_gate"]["steps"], 6)
        self.assertEqual(audit["scorer_trace_count"], 1)
        self.assertEqual(audit["aggregate"]["safety_gate_counts"]["canopy_gate"], 6)
        self.assertIn("Profile Scorer Decision Audit", build_report(audit))

    def test_warns_when_trace_has_no_scorer_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm.csv"
            _write_csv(path, [{"step": 0, "reward": 0.0}])

            trace = audit_trace(path)

        self.assertEqual(trace["scorer_steps"], 0)
        self.assertIn("missing_profile_scorer_metadata", trace["warnings"])
        self.assertEqual(trace["recommendation"]["decision"], "needs_trace_metadata")


if __name__ == "__main__":
    unittest.main()
