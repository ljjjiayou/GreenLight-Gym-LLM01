import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.canopy_boundary_shadow_audit import audit_trace, build_report


def _write_trace(path: Path):
    fields = [
        "step",
        "temp_air",
        "rh_air",
        "vpd_air",
        "dew_margin_air",
        "canopy_dew_margin",
        "rh_low_violation",
        "vpd_high_excess",
        "rspc_action_hot_dry_active",
        "canopy_boundary_shadow_active",
        "canopy_boundary_shadow_would_reject",
        "canopy_boundary_shadow_would_reject_v2",
        "canopy_boundary_shadow_warning_v2",
        "canopy_boundary_shadow_reason",
        "canopy_boundary_shadow_reason_v2",
        "canopy_boundary_shadow_predicted_canopy_dew_margin_next",
        "canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2",
        "canopy_boundary_shadow_screen_delta",
        "canopy_boundary_shadow_vent_delta",
        "canopy_boundary_shadow_screen_increase_under_canopy_risk",
        "canopy_boundary_shadow_ventilation_decrease_under_canopy_risk",
        "canopy_boundary_shadow_high_screen_under_canopy_risk",
        "canopy_boundary_shadow_low_ventilation_under_canopy_risk",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "step": 233,
                "temp_air": 24,
                "rh_air": 55,
                "vpd_air": 1.7,
                "dew_margin_air": 2.0,
                "canopy_dew_margin": 1.2,
                "rh_low_violation": 0.1,
                "vpd_high_excess": 0.2,
                "rspc_action_hot_dry_active": True,
                "canopy_boundary_shadow_active": True,
                "canopy_boundary_shadow_would_reject": True,
                "canopy_boundary_shadow_would_reject_v2": True,
                "canopy_boundary_shadow_warning_v2": True,
                "canopy_boundary_shadow_reason": "canopy_dew_screen_vent_conflict",
                "canopy_boundary_shadow_reason_v2": "canopy_dew_lead_time_screen_vent_conflict",
                "canopy_boundary_shadow_predicted_canopy_dew_margin_next": 0.1,
                "canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2": -0.2,
                "canopy_boundary_shadow_screen_delta": 0.08,
                "canopy_boundary_shadow_vent_delta": -0.05,
                "canopy_boundary_shadow_screen_increase_under_canopy_risk": True,
                "canopy_boundary_shadow_ventilation_decrease_under_canopy_risk": True,
                "canopy_boundary_shadow_high_screen_under_canopy_risk": True,
                "canopy_boundary_shadow_low_ventilation_under_canopy_risk": True,
            }
        )


class TestCanopyBoundaryShadowAudit(unittest.TestCase):
    def test_marks_failure_before_failure_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            _write_trace(path)
            report = audit_trace(
                trace_path=path,
                preset="hot_dry_relief",
                scenario_id="case",
                start_step=232,
                end_step=234,
                failure_step=234,
            )

        self.assertTrue(report["would_block_failure_case"])
        self.assertEqual(report["first_boundary_warning_step"], 233)
        self.assertEqual(report["lead_time_steps"], 1)
        self.assertTrue(report["catches_pre_failure_warning"])
        self.assertTrue(report["prevention_window_available"])
        self.assertTrue(report["controlled_repair_allowed"])
        self.assertEqual(report["counts"]["would_reject"], 1)
        self.assertEqual(report["counts"]["warning_v2"], 1)
        self.assertEqual(report["counts"]["would_reject_v2"], 1)
        self.assertEqual(report["counts"]["high_screen_under_canopy_risk"], 1)
        self.assertEqual(report["counts"]["low_ventilation_under_canopy_risk"], 1)

    def test_build_report_reads_nested_preset_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "traces"
            hot = root / "hot_dry_relief"
            balanced = root / "balanced"
            hot.mkdir(parents=True)
            balanced.mkdir(parents=True)
            _write_trace(hot / "case_llm_rspc_v2.csv")
            _write_trace(balanced / "case_llm_rspc_v2.csv")

            report = build_report(
                trace_dirs=[root],
                scenario_id="case_llm_rspc_v2",
                start_step=232,
                end_step=234,
                failure_step=234,
            )

        self.assertEqual(report["trace_count"], 2)
        self.assertTrue(report["would_block_any_failure_case"])
        self.assertTrue(report["prevention_window_available"])
        self.assertEqual(report["aggregate_counts"]["would_reject"], 2)
        self.assertEqual(report["aggregate_counts"]["warning_v2"], 2)
        self.assertEqual({trace["preset"] for trace in report["traces"]}, {"balanced", "hot_dry_relief"})


if __name__ == "__main__":
    unittest.main()
