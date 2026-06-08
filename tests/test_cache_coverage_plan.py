import unittest

from gl_gym.experiments.cache_coverage_plan import build_report


class TestCacheCoveragePlan(unittest.TestCase):
    def test_cache_plan_does_not_mark_coverage_passed(self):
        report = build_report(
            plan_cache_path="gl_gym/result/plan_cache/llm_plan_cache.json",
            plan_cache_key_policy="scenario_timestep",
            max_steps=240,
        )

        self.assertFalse(report["cache_coverage_pass"])
        self.assertFalse(report["cache_check_run"])
        self.assertFalse(report["cache_fill_run"])
        self.assertFalse(report["metadata_replay_allowed"])
        families = {row["family"] for row in report["scenario_families"]}
        self.assertIn("canonical_failure", families)


if __name__ == "__main__":
    unittest.main()
