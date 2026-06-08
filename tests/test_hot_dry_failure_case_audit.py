import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.hot_dry_failure_case_audit import audit_failure_case


def _write_trace(path: Path, canopy_values):
    fields = [
        "step",
        "canopy_dew_margin",
        "dew_margin_air",
        "temp_air",
        "rh_air",
        "vpd_air",
        "u_screen",
        "u_ventilation",
        "rspc_action_selected_name",
        "tomato_safety_v2_applied",
        "tomato_safety_v2_reasons",
        "tomato_safety_v2_before",
        "tomato_safety_v2_after",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, canopy in enumerate(canopy_values):
            writer.writerow(
                {
                    "step": 220 + idx,
                    "canopy_dew_margin": canopy,
                    "dew_margin_air": 3.0,
                    "temp_air": 20.0,
                    "rh_air": 80.0,
                    "vpd_air": 0.5,
                    "u_screen": 0.6 + idx * 0.01,
                    "u_ventilation": 0.2,
                    "rspc_action_selected_name": "anchor",
                    "tomato_safety_v2_applied": False,
                    "tomato_safety_v2_reasons": "",
                    "tomato_safety_v2_before": "[0,0,0.6,0.2,0,0]",
                    "tomato_safety_v2_after": "[0,0,0.6,0.2,0,0]",
                }
            )


class TestHotDryFailureCaseAudit(unittest.TestCase):
    def test_detects_first_extra_canopy_lt0(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.csv"
            compare = root / "compare.csv"
            _write_trace(baseline, [1.0, 0.5, 0.3])
            _write_trace(compare, [1.0, -0.1, 0.2])

            report = audit_failure_case(
                baseline_trace=baseline,
                compare_trace=compare,
                scenario_id="case",
                start_step=220,
                end_step=222,
            )

        self.assertTrue(report["hard_safety_regression"])
        self.assertEqual(report["first_extra_canopy_lt0_step"], 221)
        self.assertEqual(report["extra_canopy_lt0_steps"], 1)
        self.assertEqual(report["baseline_canopy_lt0_steps"], 0)


if __name__ == "__main__":
    unittest.main()
