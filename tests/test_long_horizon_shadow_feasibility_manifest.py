import unittest

from gl_gym.experiments.long_horizon_shadow_feasibility_manifest import build_report


def _requested(year: int, day: int, seed: int):
    return {"year": year, "day": day, "seed": seed, "max_steps": 240}


class TestLongHorizonShadowFeasibilityManifest(unittest.TestCase):
    def test_manifest_excludes_v36_failed_and_includes_primary_candidates(self):
        v35 = {
            "requested_scenarios": [
                *[_requested(2006, 197, seed) for seed in [42, 43, 44]],
                *[_requested(2006, 183, seed) for seed in [42, 43, 44]],
                *[_requested(2005, 169, seed) for seed in [42, 43, 44]],
                *[_requested(2006, 182, seed) for seed in [42, 43, 44]],
                *[_requested(2006, 162, seed) for seed in [42, 43, 44]],
                *[_requested(2006, 161, seed) for seed in [42, 43, 44]],
            ]
        }
        env_reports = {}
        for year, day in [(2006, 197), (2006, 183), (2005, 169)]:
            for seed in [42, 43, 44]:
                env_reports[f"TomatoEnv_y{year}_d{day}_s{seed}"] = {
                    "last_step": 40,
                    "runtime_error_count": 1,
                    "trace_path": f"traces/y{year}_d{day}_s{seed}_n240_llm_rspc_v2.csv",
                    "runtime_errors": [{"error_contains_cvodes_convergence_failure": True}],
                }

        report = build_report(v35_manifest=v35, v36_diagnosis={"shadow_cache_coverage_pass": False, "env_reports": env_reports})

        self.assertTrue(report["long_horizon_shadow_feasibility_manifest_ready"])
        self.assertEqual(len(report["blocked_runtime_failed_scenarios"]), 9)
        self.assertEqual(report["primary_scenario_ids"], [
            "y2006_d182_s42_n240",
            "y2006_d182_s43_n240",
            "y2006_d182_s44_n240",
            "y2006_d162_s42_n240",
            "y2006_d162_s43_n240",
            "y2006_d162_s44_n240",
            "y2006_d161_s42_n240",
            "y2006_d161_s43_n240",
            "y2006_d161_s44_n240",
        ])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
