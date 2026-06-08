import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.strict_targeted_shadow_sweep_v29_manifest import build_report


def _cache_payload(include_extra=False):
    entries = {}
    idx = 0
    for year in [2010, 2015, 2020]:
        for day in [59, 120, 180, 240]:
            for seed in [42, 43, 44]:
                entries[f"k{idx}"] = {
                    "env_id": f"TomatoEnv_y{year}_d{day}_s{seed}",
                    "timestep": 0,
                    "buffered_action": {},
                    "parsed_plan": {},
                }
                idx += 1
    if include_extra:
        entries["extra"] = {"env_id": "TomatoEnv_y2001_d1_s1", "timestep": 0}
    return {"schema_version": "test", "entries": entries}


class TestStrictTargetedShadowSweepV29Manifest(unittest.TestCase):
    def test_manifest_contains_only_cache_covered_36_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            cache.write_text(json.dumps(_cache_payload(include_extra=True)), encoding="utf-8")
            report = build_report(selected_cache_path=str(cache))

        self.assertTrue(report["strict_targeted_shadow_sweep_manifest_ready"])
        self.assertEqual(report["scenario_count"], 36)
        self.assertEqual(len(report["scenario_ids"]), 36)
        self.assertNotIn("y2001_d1_s1_n240", report["scenario_ids"])
        self.assertEqual(report["controllers"], ["llm_rspc_v2"])
        self.assertFalse(report["controlled_replay_allowed"])

    def test_manifest_blocks_when_selected_cache_missing_expected_scenario(self):
        payload = _cache_payload()
        payload["entries"].pop("k0")
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            cache.write_text(json.dumps(payload), encoding="utf-8")
            report = build_report(selected_cache_path=str(cache))

        self.assertFalse(report["strict_targeted_shadow_sweep_manifest_ready"])
        self.assertIn("TomatoEnv_y2010_d59_s42", report["missing_envs"])


if __name__ == "__main__":
    unittest.main()
