import unittest

from gl_gym.experiments.diagnose_ppo_vs_llm import finalize_trace_row, safety_patterns, sum_metrics
from gl_gym.experiments.run_frozen_benchmark import build_jobs, parse_controller_list


class TestFrozenBenchmark(unittest.TestCase):
    def test_builds_full_job_matrix(self):
        jobs = build_jobs(
            years=[2010, 2020],
            days=[59, 240],
            seeds=[42],
            controllers=["llm", "llm_sero_shadow", "llm_hem", "ppo"],
            max_steps=240,
        )

        self.assertEqual(len(jobs), 16)
        self.assertEqual(jobs[0].scenario_id, "y2010_d59_s42_n240")
        self.assertEqual(jobs[-1].controller, "ppo")

    def test_accepts_sero_shadow_controller(self):
        self.assertEqual(parse_controller_list("llm,llm_sero_shadow"), ["llm", "llm_sero_shadow"])

    def test_accepts_rspc_v2_controller(self):
        self.assertEqual(parse_controller_list("llm,llm_rspc_v2"), ["llm", "llm_rspc_v2"])

    def test_rejects_unknown_controller(self):
        with self.assertRaises(ValueError):
            parse_controller_list("llm,bad")

    def _row(self, **overrides):
        row = {
            "algo": "llm_director",
            "source": "anchor",
            "reward": 0.0,
            "profit": 0.0,
            "revenue": 0.0,
            "heat_cost": 0.0,
            "co2_cost": 0.0,
            "elec_cost": 0.0,
            "fixed_cost": 0.0,
            "temp_violation": 0.0,
            "co2_violation": 0.0,
            "rh_violation": 0.0,
            "lamp_violation": 0.0,
            "hour_of_day": 12.0,
            "temp_air": 20.0,
            "rh_air": 70.0,
            "co2_air": 430.0,
            "glob_rad": 120.0,
            "dew_margin_air": 2.0,
            "canopy_dew_margin": 2.0,
            "u_heating": 0.0,
            "u_co2": 0.0,
            "u_screen": 0.0,
            "u_ventilation": 0.0,
            "u_lighting": 0.0,
            "u_shading": 0.0,
        }
        row.update(overrides)
        return finalize_trace_row(row)

    def test_sums_low_and_high_rh_violations_separately(self):
        rows = [
            self._row(rh_air=45.0, temp_air=25.0, u_ventilation=0.20),
            self._row(rh_air=95.0, temp_air=18.0, dew_margin_air=0.3, canopy_dew_margin=0.3),
        ]

        summary = sum_metrics(rows)

        self.assertAlmostEqual(summary["total_rh_low_violation"], 5.0)
        self.assertAlmostEqual(summary["total_rh_high_violation"], 5.0)
        self.assertEqual(summary["dry_risk_steps"], 1)
        self.assertEqual(summary["dew_risk_steps"], 1)
        self.assertEqual(summary["source_counts"], {"anchor": 2})

    def test_safety_patterns_include_dry_and_cold_vent_risks(self):
        rows = [
            self._row(temp_air=11.5, rh_air=86.0, u_ventilation=0.30, dew_margin_air=1.2, canopy_dew_margin=1.1),
            self._row(temp_air=25.0, rh_air=43.0, u_ventilation=0.30),
        ]

        patterns = safety_patterns(rows)["llm_director"]

        self.assertEqual(patterns["cold_vent_risk_steps"], 1)
        self.assertEqual(patterns["dry_vent_risk_steps"], 1)
        self.assertEqual(patterns["dry_risk_steps"], 1)
        self.assertGreater(patterns["mean_vent_when_dry_risk"], 0.0)

    def test_sums_mc_sero_shadow_diagnostics(self):
        rows = [
            self._row(
                mc_sero_enabled=True,
                mc_sero_available=True,
                mc_sero_would_select=True,
                mc_sero_best_candidate="dry_recovery",
                mc_sero_margin=0.30,
            ),
            self._row(
                mc_sero_enabled=True,
                mc_sero_available=True,
                mc_sero_would_select=False,
                mc_sero_best_candidate="anchor",
                mc_sero_margin=0.05,
                mc_sero_reject_reason="insufficient_margin",
            ),
        ]

        summary = sum_metrics(rows)

        self.assertEqual(summary["mc_sero_enabled_steps"], 2)
        self.assertEqual(summary["mc_sero_available_steps"], 2)
        self.assertEqual(summary["mc_sero_would_select_steps"], 1)
        self.assertAlmostEqual(summary["mean_mc_sero_margin"], 0.175)
        self.assertEqual(summary["mc_sero_best_candidate_counts"], {"anchor": 1, "dry_recovery": 1})
        self.assertEqual(summary["mc_sero_reject_reason_counts"], {"insufficient_margin": 1})

    def test_sums_tomato_safety_v2_suppressed_replans(self):
        rows = [
            self._row(
                tomato_safety_v2_enabled=True,
                tomato_safety_v2_applied=True,
                tomato_safety_v2_reasons="hot_dry_cooling_guard",
                tomato_safety_v2_suppressed_replan=True,
            ),
            self._row(
                tomato_safety_v2_enabled=True,
                tomato_safety_v2_applied=False,
                tomato_safety_v2_reasons="",
                tomato_safety_v2_suppressed_replan=False,
            ),
        ]

        summary = sum_metrics(rows)

        self.assertEqual(summary["tomato_safety_v2_enabled_steps"], 2)
        self.assertEqual(summary["tomato_safety_v2_applied_steps"], 1)
        self.assertEqual(summary["tomato_safety_v2_suppressed_replan_steps"], 1)
        self.assertEqual(summary["tomato_safety_v2_reason_counts"], {"hot_dry_cooling_guard": 1})


if __name__ == "__main__":
    unittest.main()
