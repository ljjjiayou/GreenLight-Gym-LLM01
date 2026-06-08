import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.profile_rspc_shadow_audit import audit_trace, audit_traces, build_report


def _write_csv(path, rows):
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(step, **extra):
    data = {
        "step": step,
        "intent_regime": "hot_dry_relief",
        "profile_rspc_shadow_enabled": True,
        "profile_rspc_shadow_candidate_count": 3,
        "profile_rspc_shadow_best_profile": "hot_dry_protect",
        "profile_rspc_shadow_best_score": 1.0,
        "profile_rspc_shadow_baseline_score": 1.5,
        "profile_rspc_shadow_margin": 0.5,
        "profile_rspc_shadow_would_improve": True,
        "profile_rspc_shadow_raw_best_profile": "hot_dry_protect",
        "profile_rspc_shadow_raw_best_margin": 0.5,
        "profile_rspc_shadow_raw_best_would_improve": True,
        "profile_rspc_shadow_safety_gate_reason": "none",
        "profile_rspc_shadow_safety_alignment": "dry_benefit",
        "rh_air": 55.0,
        "vpd_air": 1.8,
        "temp_air": 29.0,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "rh_low_violation": 0.0,
        "vpd_high_excess": 0.0,
        "temp_violation": 0.0,
        "rh_high_violation": 0.0,
    }
    data.update(extra)
    return data


class TestProfileRspcShadowAudit(unittest.TestCase):
    def test_audit_detects_safe_hot_dry_signal(self):
        rows = [_row(step) for step in range(8)]
        rows.extend(
            _row(
                step,
                profile_rspc_shadow_best_profile="shade_cooling",
                profile_rspc_shadow_margin=-0.2,
                profile_rspc_shadow_would_improve=False,
                profile_rspc_shadow_raw_best_profile="shade_cooling",
                profile_rspc_shadow_raw_best_margin=-0.2,
                profile_rspc_shadow_raw_best_would_improve=False,
                profile_rspc_shadow_safety_gate_reason="temp_high_gate",
                temp_air=33.0,
            )
            for step in range(8, 12)
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d180_s44_n240_llm_rspc_v2.csv"
            _write_csv(path, rows)

            trace = audit_trace(path)
            audit = audit_traces([tmp])

        self.assertEqual(trace["shadow_steps"], 12)
        self.assertEqual(trace["safe_hot_dry_steps"], 8)
        self.assertEqual(trace["safe_hot_dry_positive_margin_steps"], 8)
        self.assertEqual(trace["positive_margin_safety_steps"], 0)
        self.assertEqual(audit["aggregate"]["recommendation"]["decision"], "profile_signal_detected")
        self.assertIn("Profile RSPC Shadow Audit", build_report(audit))

    def test_audit_allows_safety_aligned_relief_positive_margin(self):
        rows = [
            _row(
                step,
                profile_rspc_shadow_best_profile="strict_dehumidify_then_relax",
                profile_rspc_shadow_raw_best_profile="strict_dehumidify_then_relax",
                profile_rspc_shadow_safety_gate_reason="rh_high_gate",
                profile_rspc_shadow_safety_alignment="safe_relief",
                rh_air=92.0,
                vpd_air=0.2,
                dew_margin_air=1.5,
                canopy_dew_margin=4.0,
            )
            for step in range(4)
        ]
        rows.extend(_row(step) for step in range(4, 8))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm_rspc_v2.csv"
            _write_csv(path, rows)

            trace = audit_trace(path)
            audit = audit_traces([tmp])

        self.assertEqual(trace["positive_margin_safety_steps"], 4)
        self.assertEqual(trace["safety_aligned_positive_margin_steps"], 4)
        self.assertEqual(trace["unsafe_conflict_positive_margin_steps"], 0)
        self.assertEqual(audit["aggregate"]["recommendation"]["decision"], "profile_signal_detected")

    def test_audit_flags_hot_dry_positive_under_safety_gate(self):
        rows = [
            _row(
                step,
                profile_rspc_shadow_safety_gate_reason="canopy_gate",
                profile_rspc_shadow_safety_alignment="unsafe_conflict",
                canopy_dew_margin=0.5,
            )
            for step in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d240_s42_n240_llm_rspc_v2.csv"
            _write_csv(path, rows)

            trace = audit_trace(path)

        self.assertEqual(trace["hot_dry_positive_under_safety_gate_steps"], 3)
        self.assertEqual(trace["recommendation"]["decision"], "unsafe_to_connect")

    def test_audit_flags_unsafe_conflict_alignment(self):
        rows = [
            _row(
                step,
                profile_rspc_shadow_best_profile="constant_hold",
                profile_rspc_shadow_raw_best_profile="constant_hold",
                profile_rspc_shadow_safety_gate_reason="temp_high_gate",
                profile_rspc_shadow_safety_alignment="unsafe_conflict",
                temp_air=33.0,
            )
            for step in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2010_d180_s44_n240_llm.csv"
            _write_csv(path, rows)

            trace = audit_trace(path)

        self.assertEqual(trace["unsafe_conflict_positive_margin_steps"], 3)
        self.assertEqual(trace["recommendation"]["decision"], "unsafe_to_connect")

    def test_warns_when_shadow_metadata_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm.csv"
            _write_csv(path, [{"step": 0, "reward": 0.0}])

            trace = audit_trace(path)

        self.assertEqual(trace["shadow_steps"], 0)
        self.assertIn("missing_profile_rspc_shadow_metadata", trace["warnings"])
        self.assertEqual(trace["recommendation"]["decision"], "needs_trace_metadata")


if __name__ == "__main__":
    unittest.main()
