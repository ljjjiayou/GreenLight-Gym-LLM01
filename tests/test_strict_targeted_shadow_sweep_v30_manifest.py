import unittest

from gl_gym.experiments.strict_targeted_shadow_sweep_v30_manifest import build_report


def _coverage_payload(fail_subset=False):
    envs = {}
    for year in [2010, 2015, 2020]:
        for day in [59, 120, 180, 240]:
            for seed in [42, 43, 44]:
                env_id = f"TomatoEnv_y{year}_d{day}_s{seed}"
                ok = not (year == 2020 and seed in {42, 43})
                if fail_subset and env_id == "TomatoEnv_y2010_d59_s42":
                    ok = False
                envs[env_id] = {
                    "entry_count": 20 if ok else 2,
                    "ok": ok,
                    "issues": [] if ok else ["last_timestep_too_early"],
                }
    return {
        "schema_version": "plan_cache_coverage_v1",
        "plan_cache_path": "cache.json",
        "cache_coverage_pass": False,
        "missing_key_count": 8,
        "envs": envs,
    }


class TestStrictTargetedShadowSweepV30Manifest(unittest.TestCase):
    def test_manifest_contains_coverage_qualified_28_scenarios(self):
        report = build_report(v29_cache_coverage=_coverage_payload())

        self.assertTrue(report["strict_targeted_shadow_sweep_v30_manifest_ready"])
        self.assertEqual(report["scenario_count"], 28)
        self.assertEqual(report["blocked_cache_repair_backlog_count"], 8)
        self.assertIn("y2020_d59_s44_n240", report["scenario_ids"])
        self.assertNotIn("y2020_d59_s42_n240", report["scenario_ids"])
        self.assertEqual(report["controllers"], ["llm_rspc_v2"])
        self.assertFalse(report["controlled_replay_allowed"])

    def test_manifest_blocks_when_subset_env_is_not_coverage_qualified(self):
        report = build_report(v29_cache_coverage=_coverage_payload(fail_subset=True))

        self.assertFalse(report["strict_targeted_shadow_sweep_v30_manifest_ready"])
        self.assertIn("TomatoEnv_y2010_d59_s42", report["subset_missing_or_failed_envs"])


if __name__ == "__main__":
    unittest.main()
