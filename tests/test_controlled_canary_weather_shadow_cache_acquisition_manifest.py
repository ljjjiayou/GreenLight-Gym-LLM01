import unittest

from gl_gym.experiments.controlled_canary_weather_shadow_cache_acquisition_manifest import build_report


def _window(year: int, day: int, score: float):
    return {
        "weather_site": "Amsterdam",
        "weather_label": str(year),
        "year": year,
        "day": day,
        "max_steps": 240,
        "strict_proxy_row_count": 90,
        "weather_proxy_score": score,
        "acquisition_ready_under_current_runner": True,
        "scenario_ids_for_cache_acquisition": [
            f"y{year}_d{day}_s42_n240",
            f"y{year}_d{day}_s43_n240",
            f"y{year}_d{day}_s44_n240",
        ],
    }


class TestControlledCanaryWeatherShadowCacheAcquisitionManifest(unittest.TestCase):
    def test_manifest_selects_top_three_weather_days_and_nine_scenarios(self):
        report = build_report(
            weather_inventory={
                "new_weather_source_candidates_found": True,
                "candidate_windows": [
                    _window(2006, 197, 995.1),
                    _window(2006, 183, 965.2),
                    _window(2005, 169, 955.4),
                    _window(2006, 182, 944.9),
                ],
            },
            acquisition_manifest={"shadow_scenario_acquisition_manifest_ready": True},
        )

        self.assertTrue(report["weather_shadow_cache_acquisition_manifest_ready"])
        self.assertEqual(report["top_weather_day_count"], 3)
        self.assertEqual(report["scenario_count"], 9)
        self.assertEqual(report["scenario_ids"][0:3], [
            "y2006_d197_s42_n240",
            "y2006_d197_s43_n240",
            "y2006_d197_s44_n240",
        ])
        self.assertFalse(report["cache_fill_authorized"])
        self.assertFalse(report["online_llm_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])

    def test_manifest_requires_exact_top_three_days(self):
        report = build_report(
            weather_inventory={
                "new_weather_source_candidates_found": True,
                "candidate_windows": [_window(2006, 197, 995.1), _window(2006, 183, 965.2)],
            },
            acquisition_manifest={"shadow_scenario_acquisition_manifest_ready": True},
        )

        self.assertFalse(report["weather_shadow_cache_acquisition_manifest_ready"])


if __name__ == "__main__":
    unittest.main()
