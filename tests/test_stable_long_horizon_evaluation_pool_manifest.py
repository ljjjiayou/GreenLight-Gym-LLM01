import unittest

from gl_gym.experiments.stable_long_horizon_evaluation_pool_manifest import build_report


def _scenario(year, day, seed):
    return {"year": year, "day": day, "seed": seed, "max_steps": 240, "scenario_id": f"y{year}_d{day}_s{seed}_n240"}


class TestStableLongHorizonEvaluationPoolManifest(unittest.TestCase):
    def test_manifest_excludes_failed_backlog_and_used_scenarios(self):
        v30 = {
            "strict_targeted_shadow_sweep_v30_manifest_ready": True,
            "selected_scenarios": [
                _scenario(2010, 59, 42),
                _scenario(2015, 120, 42),
                _scenario(2020, 120, 42),
            ],
            "blocked_cache_repair_backlog": [_scenario(2020, 120, 42)],
        }
        v35 = {
            "shadow_scenario_acquisition_manifest_ready": True,
            "requested_scenarios": [
                _scenario(2006, 197, 42),
                _scenario(2006, 182, 42),
                _scenario(2006, 160, 42),
                _scenario(2006, 159, 42),
            ],
        }
        v36 = {
            "env_reports": {
                "TomatoEnv_y2006_d197_s42": {
                    "trace_path": "traces/y2006_d197_s42_n240_llm_rspc_v2.csv",
                }
            }
        }
        v37_manifest = {"primary_scenarios": [_scenario(2006, 182, 42)]}
        v37_result = {"scenario_reports": [{"year": 2006, "day": 182, "seed": 42, "horizon_pass": False}]}

        report = build_report(
            v30_manifest=v30,
            v35_manifest=v35,
            v36_diagnosis=v36,
            v37_manifest=v37_manifest,
            v37_result=v37_result,
            tier_b_limit=12,
        )

        self.assertTrue(report["stable_long_horizon_manifest_ready"])
        self.assertEqual([item["scenario_id"] for item in report["tier_a_scenarios"]], ["y2010_d59_s42_n240"])
        self.assertEqual([item["scenario_id"] for item in report["tier_b_scenarios"]], ["y2006_d160_s42_n240", "y2006_d159_s42_n240"])
        excluded_ids = {item["scenario_id"] for item in report["excluded_scenarios"]}
        self.assertIn("y2015_d120_s42_n240", excluded_ids)
        self.assertIn("y2020_d120_s42_n240", excluded_ids)
        self.assertIn("y2006_d197_s42_n240", excluded_ids)
        self.assertIn("y2006_d182_s42_n240", excluded_ids)
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
