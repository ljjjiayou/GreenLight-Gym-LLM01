import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.controlled_canary_weather_window_inventory import build_report


FIELDNAMES = [
    "time",
    "global radiation",
    "wind speed",
    "air temperature",
    "sky temperature",
    "??",
    "CO2 concentration",
    "day number",
    "RH",
]


def _write_weather(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_rows(day: int = 10):
    return [
        {
            "time": str((day * 86400) + (step * 300)),
            "global radiation": "650.0",
            "wind speed": "3.0",
            "air temperature": "28.0",
            "sky temperature": "18.0",
            "??": "0.0",
            "CO2 concentration": "400.0",
            "day number": str(day + step / 288),
            "RH": "35.0",
        }
        for step in [30, 31, 32]
    ]


class TestControlledCanaryWeatherWindowInventory(unittest.TestCase):
    def test_inventory_finds_amsterdam_weather_proxy_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weather_path = root / "Amsterdam" / "2021.csv"
            _write_weather(weather_path, _candidate_rows())
            report = build_report(
                target_spec={
                    "target_spec_ready": True,
                    "weather_sources": [str(root)],
                    "candidate_seeds": [42, 43],
                    "excluded_scenario_ids": [],
                    "max_steps": 240,
                    "max_candidate_days": 2,
                    "strict_weather_proxy_gate": {
                        "candidate": "shadow_hot_dry_humidity_retention",
                        "rh_max": 45.0,
                        "vpd_min": 2.25,
                        "temp_min": 20.0,
                        "temp_max": 30.5,
                        "dew_point_spread_min": 3.0,
                        "global_radiation_min": 200.0,
                    },
                }
            )

        self.assertTrue(report["new_weather_source_candidates_found"])
        self.assertEqual(report["requested_scenario_count"], 2)
        self.assertFalse(report["weather_proxy_is_strict_applied_evidence"])
        self.assertFalse(report["cache_fill_authorized"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(
            sorted(item["scenario_id"] for item in report["candidate_scenarios_for_cache_acquisition"]),
            ["y2021_d10_s42_n240", "y2021_d10_s43_n240"],
        )

    def test_inventory_excludes_used_scenarios_from_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_weather(root / "Amsterdam" / "2021.csv", _candidate_rows())
            report = build_report(
                target_spec={
                    "target_spec_ready": True,
                    "weather_sources": [str(root)],
                    "candidate_seeds": [42],
                    "excluded_scenario_ids": ["y2021_d10_s42_n240"],
                    "max_steps": 240,
                    "max_candidate_days": 2,
                    "strict_weather_proxy_gate": {
                        "rh_max": 45.0,
                        "vpd_min": 2.25,
                        "temp_min": 20.0,
                        "temp_max": 30.5,
                        "dew_point_spread_min": 3.0,
                        "global_radiation_min": 200.0,
                    },
                }
            )

        self.assertFalse(report["new_weather_source_candidates_found"])
        self.assertEqual(report["requested_scenario_count"], 0)


if __name__ == "__main__":
    unittest.main()
