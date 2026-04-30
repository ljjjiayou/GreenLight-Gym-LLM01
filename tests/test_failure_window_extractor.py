import unittest

from gl_gym.experiments.failure_window_extractor import extract_failure_windows


def _row(step, **overrides):
    row = {
        "step": step,
        "hour_of_day": step / 4.0,
        "temp_air": 20.0,
        "rh_air": 70.0,
        "vpd_air": 0.8,
        "rh_low_violation": 0.0,
        "rh_high_violation": 0.0,
        "vpd_high_excess": 0.0,
        "temp_violation": 0.0,
        "u_heating": 0.0,
        "u_co2": 0.0,
        "u_screen": 0.0,
        "u_ventilation": 0.1,
        "u_lighting": 0.0,
        "u_shading": 0.0,
        "source": "anchor",
        "target_rh": 76.0,
        "cold_vent_risk": False,
        "mc_sero_would_select": False,
        "mc_sero_margin": 0.0,
        "mc_sero_best_candidate": "",
    }
    row.update(overrides)
    return row


class TestFailureWindowExtractor(unittest.TestCase):
    def test_extracts_dry_stress_window(self):
        rows = [
            _row(0),
            _row(1, rh_air=48.0, vpd_air=1.4, rh_low_violation=2.0, vpd_high_excess=0.2, u_ventilation=0.3),
            _row(2, rh_air=46.0, vpd_air=1.6, rh_low_violation=4.0, vpd_high_excess=0.4, u_ventilation=0.2),
            _row(3),
        ]

        windows = extract_failure_windows(rows, trace_id="case", context_steps=1, merge_gap=0)
        dry = [w for w in windows if w["kind"] == "dry_stress_window"]

        self.assertEqual(len(dry), 1)
        self.assertEqual(dry[0]["start_step"], 1)
        self.assertEqual(dry[0]["end_step"], 2)
        self.assertAlmostEqual(dry[0]["rh_low_area"], 6.0)
        self.assertAlmostEqual(dry[0]["vpd_high_area"], 0.6)
        self.assertEqual(len(dry[0]["pre_context"]), 1)
        self.assertEqual(len(dry[0]["post_context"]), 1)

    def test_extracts_target_action_mismatch(self):
        rows = [
            _row(0, rh_air=48.0, target_rh=72.0, u_ventilation=0.25),
            _row(1, rh_air=47.0, target_rh=72.0, u_ventilation=0.24),
        ]

        windows = extract_failure_windows(rows, trace_id="case", merge_gap=0)
        mismatch = [w for w in windows if w["kind"] == "target_action_mismatch_window"]

        self.assertEqual(len(mismatch), 1)
        self.assertEqual(mismatch[0]["duration_steps"], 2)
        self.assertEqual(mismatch[0]["source_counts"], {"anchor": 2})

    def test_extracts_cold_and_cold_vent_windows(self):
        rows = [
            _row(0, temp_air=11.5, temp_violation=0.5, u_ventilation=0.4, cold_vent_risk=True),
            _row(1, temp_air=16.0),
        ]

        windows = extract_failure_windows(rows, trace_id="case", merge_gap=0)
        kinds = {w["kind"] for w in windows}

        self.assertIn("cold_recovery_window", kinds)
        self.assertIn("cold_vent_risk_window", kinds)

    def test_extracts_shadow_opportunity(self):
        rows = [
            _row(
                0,
                source="anchor",
                mc_sero_would_select=True,
                mc_sero_margin=0.22,
                mc_sero_best_candidate="dry_recovery",
            ),
            _row(1),
        ]

        windows = extract_failure_windows(rows, trace_id="case", merge_gap=0)
        shadow = [w for w in windows if w["kind"] == "shadow_opportunity_window"]

        self.assertEqual(len(shadow), 1)
        self.assertEqual(shadow[0]["mc_sero_best_candidate_counts"], {"dry_recovery": 1})


if __name__ == "__main__":
    unittest.main()

