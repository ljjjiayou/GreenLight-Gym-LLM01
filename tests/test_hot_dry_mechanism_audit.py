import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.hot_dry_mechanism_audit import (
    analyze_scenario,
    audit_from_trace_dirs,
    build_report,
    find_trace,
)


def _row(step, **overrides):
    row = {
        "step": step,
        "temp_air": 30.0,
        "rh_air": 52.0,
        "vpd_air": 1.7,
        "glob_rad": 720.0,
        "forecast_rad_mean_1h": 700.0,
        "reward": 0.0,
        "profit": 0.0,
        "rh_low_violation": 0.0,
        "rh_high_violation": 0.0,
        "vpd_high_excess": 0.0,
        "temp_violation": 0.0,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "u_heating": 0.0,
        "u_co2": 0.0,
        "u_screen": 0.2,
        "u_ventilation": 0.2,
        "u_lighting": 0.0,
        "u_shading": 0.0,
        "target_temp": 22.0,
        "target_co2": 450.0,
        "target_rh": 75.0,
        "source": "rspc",
        "tomato_safety_v2_applied": False,
        "tomato_safety_v2_reasons": "",
        "mc_sero_would_select": False,
        "mc_sero_margin": 0.0,
        "mc_sero_best_candidate": "",
        "mc_sero_reject_reason": "",
    }
    row.update(overrides)
    return row


def _write_csv(path, rows):
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestHotDryMechanismAudit(unittest.TestCase):
    def test_summarizes_v2_post_guard_dry_degradation_window(self):
        baseline = [_row(i) for i in range(4)]
        candidate = [
            _row(0),
            _row(
                1,
                rh_low_violation=2.0,
                vpd_high_excess=0.3,
                u_ventilation=0.5,
                u_screen=0.0,
                tomato_safety_v2_applied=True,
                tomato_safety_v2_reasons="cooldown_dry_recovery,hot_dry_radiation_relief",
            ),
            _row(
                2,
                rh_low_violation=1.0,
                vpd_high_excess=0.2,
                u_ventilation=0.48,
                u_screen=0.0,
                tomato_safety_v2_applied=True,
                tomato_safety_v2_reasons="cooldown_dry_recovery",
            ),
            _row(3),
        ]
        shadow = [
            _row(0),
            _row(1, mc_sero_would_select=True, mc_sero_margin=0.16, mc_sero_best_candidate="dry_recovery"),
            _row(2, mc_sero_would_select=True, mc_sero_margin=0.12, mc_sero_best_candidate="dry_recovery"),
            _row(3),
        ]

        result = analyze_scenario("case", baseline, candidate, shadow, merge_gap=0, horizon_steps=3)

        self.assertEqual(result["window_count"], 1)
        window = result["windows"][0]
        self.assertEqual(window["start_step"], 1)
        self.assertEqual(window["end_step"], 2)
        self.assertAlmostEqual(window["delta"]["rh_low"], 3.0)
        self.assertGreater(window["action_delta"]["u_ventilation"], 0.25)
        self.assertEqual(window["tomato_safety_v2_reason_counts"]["cooldown_dry_recovery"], 2)
        self.assertEqual(window["shadow"]["best_candidate_counts"], {"dry_recovery": 2})
        self.assertEqual(window["diagnosis"]["primary_layer"], "tomato_safety_v2_post_guard")

    def test_setpoint_delta_points_to_contract_layer(self):
        baseline = [_row(0, target_rh=76.0), _row(1, target_rh=76.0)]
        candidate = [
            _row(0, target_rh=68.0, rh_low_violation=1.0),
            _row(1, target_rh=68.0, rh_low_violation=1.0),
        ]

        result = analyze_scenario("case", baseline, candidate, [], merge_gap=0)

        self.assertEqual(result["window_count"], 1)
        self.assertEqual(result["windows"][0]["diagnosis"]["primary_layer"], "llm_setpoint_or_contract")

    def test_finds_nested_trace_and_audit_warns_on_missing_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested"
            nested.mkdir()
            _write_csv(nested / "y2015_d120_s44_n240_llm.csv", [_row(0)])

            found = find_trace([root], "y2015_d120_s44_n240", "llm")
            audit = audit_from_trace_dirs(
                [root],
                ["y2015_d120_s44_n240"],
                baseline="llm",
                candidate="llm_rspc_v2",
                shadow="llm_sero_shadow",
            )

        self.assertEqual(found.name, "y2015_d120_s44_n240_llm.csv")
        self.assertEqual(audit["scenario_count"], 0)
        self.assertIn("missing_trace", "\n".join(audit["warnings"]))

    def test_report_names_next_optimization_rule(self):
        audit = {
            "baseline": "llm",
            "candidate": "llm_rspc_v2",
            "shadow": "llm_sero_shadow",
            "scenario_count": 1,
            "window_count": 0,
            "warnings": [],
            "scenarios": [
                {
                    "scenario_id": "case",
                    "aligned_steps": 2,
                    "aggregate_delta": {"reward": 0.0, "rh_low": 0.0, "vpd_high": 0.0, "temp": 0.0},
                    "main_layer": "no_degradation_window",
                    "windows": [],
                }
            ],
        }

        report = build_report(audit)

        self.assertIn("Hot-Dry Mechanism Audit v1", report)
        self.assertIn("Next Optimization Rule", report)
        self.assertIn("case", report)


if __name__ == "__main__":
    unittest.main()
