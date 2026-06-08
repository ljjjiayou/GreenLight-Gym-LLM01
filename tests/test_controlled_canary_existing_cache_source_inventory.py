import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.controlled_canary_existing_cache_source_inventory import build_report


def _entry(env_id, timestep, with_payload=True):
    item = {"env_id": env_id, "timestep": timestep}
    if with_payload:
        item["buffered_action"] = {"u_vent": 0.0}
        item["parsed_plan"] = {"target_rh": 70.0}
    return item


def _write_cache(path: Path, env_ids):
    entries = {}
    idx = 0
    for env_id in env_ids:
        for timestep in range(0, 240, 12):
            entries[f"k{idx}"] = _entry(env_id, timestep)
            idx += 1
    path.write_text(json.dumps({"schema_version": "test", "entries": entries}), encoding="utf-8")


class TestControlledCanaryExistingCacheSourceInventory(unittest.TestCase):
    def test_inventory_excludes_used_scenarios_and_incomplete_backlog(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            _write_cache(
                cache,
                [
                    "TomatoEnv_y2021_d1_s1",
                    "TomatoEnv_y2010_d59_s42",
                    "TomatoEnv_y2020_d59_s42",
                ],
            )
            report = build_report(
                near_miss_catalog={
                    "near_miss_catalog_ready": True,
                    "excluded_scenario_ids": ["y2015_d120_s42_n240"],
                },
                cache_dir=tmp,
                require_payloads=True,
            )

        self.assertTrue(report["existing_cache_inventory_ready"])
        self.assertEqual(report["candidate_scenario_ids"], ["y2021_d1_s1_n240"])
        self.assertNotIn("y2010_d59_s42_n240", report["candidate_scenario_ids"])
        self.assertNotIn("y2020_d59_s42_n240", report["candidate_scenario_ids"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_inventory_requires_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = {f"k{i}": _entry("TomatoEnv_y2021_d2_s1", timestep, with_payload=False) for i, timestep in enumerate(range(0, 240, 12))}
            (Path(tmp) / "cache.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")
            report = build_report(
                near_miss_catalog={"near_miss_catalog_ready": True, "excluded_scenario_ids": []},
                cache_dir=tmp,
                require_payloads=True,
            )

        self.assertFalse(report["existing_cache_source_candidates_found"])
        self.assertEqual(report["candidate_scenario_count"], 0)
        self.assertFalse(report["cache_fill_run"])


if __name__ == "__main__":
    unittest.main()
