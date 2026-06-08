import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.profile_to_rspc_shadow_replay import (
    audit_trace,
    audit_traces,
    build_replay_windows,
    build_report,
)


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
        "profile_rspc_shadow_best_eligible": True,
        "profile_rspc_shadow_best_profile": "hot_dry_protect",
        "profile_rspc_shadow_margin": 0.4,
        "profile_rspc_shadow_would_improve": True,
        "profile_rspc_shadow_safety_alignment": "dry_benefit",
        "profile_rspc_shadow_safety_gate_reason": "none",
        "profile_rspc_shadow_raw_best_profile": "hot_dry_protect",
        "profile_rspc_shadow_raw_best_margin": 0.4,
        "profile_rspc_shadow_raw_best_would_improve": True,
        "profile_rspc_shadow_delta_vent": -0.08,
        "profile_rspc_shadow_delta_shade": 0.1,
        "profile_rspc_shadow_delta_screen": 0.0,
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


class TestProfileToRspcShadowReplay(unittest.TestCase):
    def test_builds_replay_window_with_one_step_gap(self):
        rows = [_row(0), _row(1)]
        rows.append(_row(2, profile_rspc_shadow_would_improve=False, profile_rspc_shadow_margin=-0.1))
        rows.append(_row(3))

        windows = build_replay_windows(rows, min_signal_steps=3, max_gap_steps=1)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["start_step"], 0)
        self.assertEqual(windows[0]["end_step"], 3)
        self.assertEqual(windows[0]["signal_steps"], 3)
        self.assertEqual(windows[0]["gap_steps"], 1)

    def test_single_step_signal_is_filtered_by_hysteresis(self):
        windows = build_replay_windows([_row(0)], min_signal_steps=2, max_gap_steps=1)

        self.assertEqual(windows, [])

    def test_audit_detects_replay_signal(self):
        rows = [_row(step) for step in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm_rspc_v2.csv"
            _write_csv(path, rows)

            trace = audit_trace(path, min_signal_steps=2, coverage_threshold=0.20)
            audit = audit_traces([tmp], min_signal_steps=2, coverage_threshold=0.20)

        self.assertEqual(trace["replay_window_count"], 1)
        self.assertEqual(trace["safe_hot_dry_signal_steps"], 4)
        self.assertEqual(audit["aggregate"]["recommendation"]["decision"], "ready_for_profile_shadow_replay")
        self.assertIn("Profile-to-RSPC Shadow Replay", build_report(audit))

    def test_safe_relief_signal_is_allowed_under_safety_gate(self):
        rows = [
            _row(
                step,
                profile_rspc_shadow_best_profile="strict_dehumidify_then_relax",
                profile_rspc_shadow_raw_best_profile="strict_dehumidify_then_relax",
                profile_rspc_shadow_safety_gate_reason="rh_high_gate",
                profile_rspc_shadow_safety_alignment="safe_relief",
                rh_air=92.0,
                vpd_air=0.2,
            )
            for step in range(2)
        ]

        windows = build_replay_windows(rows, min_signal_steps=2, max_gap_steps=0)

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["profile"], "strict_dehumidify_then_relax")
        self.assertEqual(windows[0]["alignment_counts"], {"safe_relief": 2})

    def test_unsafe_conflict_signal_blocks_replay(self):
        rows = [
            _row(
                step,
                profile_rspc_shadow_safety_gate_reason="temp_high_gate",
                profile_rspc_shadow_safety_alignment="unsafe_conflict",
                temp_air=33.0,
            )
            for step in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d180_s44_n240_llm.csv"
            _write_csv(path, rows)

            trace = audit_trace(path)

        self.assertEqual(trace["signal_steps"], 0)
        self.assertEqual(trace["unsafe_signal_steps"], 3)
        self.assertEqual(trace["recommendation"]["decision"], "unsafe_to_replay")

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
