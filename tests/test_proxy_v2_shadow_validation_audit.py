import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.proxy_v2_shadow_validation_audit import build_report


FIELDS = [
    "step",
    "canopy_dew_margin",
    "dew_margin_air",
    "temp_air",
    "rh_low_violation",
    "vpd_high_excess",
    "canopy_boundary_shadow_predicted_canopy_dew_margin_next",
    "canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2",
    "canopy_boundary_shadow_warning_v2",
    "final_action_predicted_canopy_warning_v2",
]


def _write_trace(path: Path, *, warned: bool):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "step": 0,
                "canopy_dew_margin": 1.2,
                "dew_margin_air": 2.0,
                "temp_air": 25,
                "rh_low_violation": 0.1,
                "vpd_high_excess": 0.2,
                "canopy_boundary_shadow_predicted_canopy_dew_margin_next": 0.5,
                "canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2": 0.1,
                "canopy_boundary_shadow_warning_v2": warned,
                "final_action_predicted_canopy_warning_v2": warned,
            }
        )
        writer.writerow(
            {
                "step": 1,
                "canopy_dew_margin": -0.1,
                "dew_margin_air": 2.0,
                "temp_air": 25,
                "rh_low_violation": 0.0,
                "vpd_high_excess": 0.0,
                "canopy_boundary_shadow_predicted_canopy_dew_margin_next": 0.0,
                "canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2": 0.0,
                "canopy_boundary_shadow_warning_v2": False,
                "final_action_predicted_canopy_warning_v2": False,
            }
        )


class TestProxyV2ShadowValidationAudit(unittest.TestCase):
    def test_counts_unwarned_false_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "traces" / "hot_dry_relief"
            root.mkdir(parents=True)
            _write_trace(root / "case_llm_rspc_v2.csv", warned=False)

            report = build_report(trace_roots=[root.parent])

        self.assertEqual(report["false_safe_v1_unwarned_by_v2"], 1)
        self.assertEqual(report["trace_count"], 1)
        self.assertIn("warning_classification_counts", report)
        self.assertIn("buffer_sensitivity_summary", report)
        self.assertIn("proxy_v2_1_shadow_candidate", report)
        self.assertEqual(
            report["proxy_v2_1_shadow_candidate"]["rejected_scales"]["buffer_scale_0.5"]["reason"],
            "unsafe_due_to_false_safe",
        )

    def test_warned_false_safe_not_counted_as_unwarned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "traces" / "hot_dry_relief"
            root.mkdir(parents=True)
            _write_trace(root / "case_llm_rspc_v2.csv", warned=True)

            report = build_report(trace_roots=[root.parent])

        self.assertEqual(report["false_safe_v1_unwarned_by_v2"], 0)
        self.assertEqual(report["affected_pure_hot_dry_steps"], 1)
        self.assertEqual(report["warning_classification_counts"]["reasonable_near_boundary_warning"], 1)
        self.assertTrue(report["proxy_v2_1_shadow_candidate"]["accepted_for_shadow_followup"])

    def test_classifies_over_conservative_pure_hot_dry_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "traces" / "balanced"
            root.mkdir(parents=True)
            path = root / "case_llm_rspc_v2.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "step": 0,
                        "canopy_dew_margin": 2.0,
                        "dew_margin_air": 2.0,
                        "temp_air": 25,
                        "rh_low_violation": 0.1,
                        "vpd_high_excess": 0.2,
                        "canopy_boundary_shadow_predicted_canopy_dew_margin_next": 1.0,
                        "canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2": 0.1,
                        "canopy_boundary_shadow_warning_v2": True,
                        "final_action_predicted_canopy_warning_v2": True,
                    }
                )
                writer.writerow(
                    {
                        "step": 1,
                        "canopy_dew_margin": 2.1,
                        "dew_margin_air": 2.0,
                        "temp_air": 25,
                        "rh_low_violation": 0.0,
                        "vpd_high_excess": 0.0,
                        "canopy_boundary_shadow_predicted_canopy_dew_margin_next": 0.0,
                        "canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2": 0.0,
                        "canopy_boundary_shadow_warning_v2": False,
                        "final_action_predicted_canopy_warning_v2": False,
                    }
                )

            report = build_report(trace_roots=[root.parent])

        self.assertEqual(report["warning_classification_counts"]["over_conservative_warning"], 1)
        self.assertEqual(report["pure_hot_dry_warning_without_hard_event_next"], 1)


if __name__ == "__main__":
    unittest.main()
