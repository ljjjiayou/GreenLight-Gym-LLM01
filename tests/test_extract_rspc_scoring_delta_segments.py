import unittest

from gl_gym.experiments.extract_rspc_scoring_delta_segments import build_segments_from_windows


class TestExtractRspcScoringDeltaSegments(unittest.TestCase):
    def test_groups_contiguous_windows_and_scores_segment(self):
        windows = {
            "trace_count": 1,
            "window_count": 3,
            "windows": [
                {
                    "scenario_id": "s1",
                    "step": 1,
                    "interpretation_label": "useful_hot_dry_relief",
                    "local_metric_delta": {"rh_low_violation": -1.0, "vpd_high_excess": -0.2, "temp_violation": 0.0},
                    "action_delta": {"u_ventilation": -0.1},
                    "candidate_changed": True,
                },
                {
                    "scenario_id": "s1",
                    "step": 2,
                    "interpretation_label": "useful_hot_dry_relief",
                    "local_metric_delta": {"rh_low_violation": -0.5, "vpd_high_excess": 0.0, "temp_violation": 0.0},
                    "action_delta": {"u_ventilation": -0.05},
                    "candidate_changed": False,
                },
                {
                    "scenario_id": "s1",
                    "step": 8,
                    "interpretation_label": "vpd_tradeoff",
                    "local_metric_delta": {"rh_low_violation": 0.0, "vpd_high_excess": 0.3, "temp_violation": 0.0},
                    "action_delta": {"u_screen": 0.2},
                    "candidate_changed": False,
                },
            ],
        }

        report = build_segments_from_windows(windows, max_gap=2)

        self.assertEqual(report["segment_count"], 2)
        self.assertEqual(report["segments"][0]["interpretation"], "useful_humidity_retention")
        self.assertEqual(report["segments"][0]["dominant_actuator_delta"], "u_ventilation")
        self.assertEqual(report["segments"][1]["interpretation"], "vpd_tradeoff_segment")

    def test_guardrail_dominated_segment(self):
        windows = {
            "windows": [
                {
                    "scenario_id": "s1",
                    "step": 1,
                    "interpretation_label": "guardrail_overrode_scorer",
                    "tomato_safety_changed": True,
                    "local_metric_delta": {"rh_low_violation": 0.0, "vpd_high_excess": 0.0, "temp_violation": 0.0},
                },
                {
                    "scenario_id": "s1",
                    "step": 2,
                    "interpretation_label": "guardrail_overrode_scorer",
                    "tomato_safety_changed": True,
                    "local_metric_delta": {"rh_low_violation": 0.0, "vpd_high_excess": 0.0, "temp_violation": 0.0},
                },
            ]
        }

        report = build_segments_from_windows(windows)

        self.assertEqual(report["segments"][0]["interpretation"], "guardrail_dominated")


if __name__ == "__main__":
    unittest.main()
