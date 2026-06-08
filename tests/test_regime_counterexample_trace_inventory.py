import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.regime_counterexample_trace_inventory import build_report


FIELDS = [
    "step",
    "hour_of_day",
    "temp_air",
    "rh_air",
    "vpd_air",
    "dew_margin_air",
    "canopy_dew_margin",
    "glob_rad",
    "wind_speed",
    "rh_low_violation",
    "vpd_high_excess",
]


class TestRegimeCounterexampleTraceInventory(unittest.TestCase):
    def test_detects_counterexample_regimes_from_existing_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "traces" / "balanced"
            root.mkdir(parents=True)
            rows = [
                {
                    "step": 0,
                    "hour_of_day": 6,
                    "temp_air": 16,
                    "rh_air": 88,
                    "vpd_air": 0.2,
                    "dew_margin_air": 0.8,
                    "canopy_dew_margin": 0.9,
                    "glob_rad": 100,
                    "wind_speed": 1,
                    "rh_low_violation": 0,
                    "vpd_high_excess": 0,
                },
                {
                    "step": 1,
                    "hour_of_day": 12,
                    "temp_air": 24,
                    "rh_air": 65,
                    "vpd_air": 0.8,
                    "dew_margin_air": 3,
                    "canopy_dew_margin": 3,
                    "glob_rad": 600,
                    "wind_speed": 6,
                    "rh_low_violation": 0,
                    "vpd_high_excess": 0,
                },
                {
                    "step": 2,
                    "hour_of_day": 15,
                    "temp_air": 24,
                    "rh_air": 65,
                    "vpd_air": 0.8,
                    "dew_margin_air": 3,
                    "canopy_dew_margin": 3,
                    "glob_rad": 100,
                    "wind_speed": 1,
                    "rh_low_violation": 0,
                    "vpd_high_excess": 0,
                },
            ]
            with (root / "case_llm_rspc_v2.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            report = build_report(trace_dirs=[root.parent])

        self.assertTrue(report["regime_available"]["cold_humid"])
        self.assertTrue(report["regime_available"]["dawn_dew"])
        self.assertTrue(report["regime_available"]["radiation_spike"])
        self.assertTrue(report["regime_available"]["wind_stress"])
        self.assertTrue(report["regime_available"]["neutral_no_risk"])


if __name__ == "__main__":
    unittest.main()
