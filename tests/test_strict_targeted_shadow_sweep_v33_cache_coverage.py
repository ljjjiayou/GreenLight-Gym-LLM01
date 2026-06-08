import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.strict_targeted_shadow_sweep_v33_cache_coverage import build_report


def _write_cache(path: Path, *, with_payload=True):
    entries = {}
    for idx, timestep in enumerate(range(0, 240, 12)):
        entry = {"env_id": "TomatoEnv_y2021_d1_s1", "timestep": timestep}
        if with_payload:
            entry["buffered_action"] = {"u_vent": 0.0}
            entry["parsed_plan"] = {"target_rh": 70.0}
        entries[f"k{idx}"] = entry
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")


class TestStrictTargetedShadowSweepV33CacheCoverage(unittest.TestCase):
    def test_grouped_cache_coverage_passes_with_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            _write_cache(cache)
            report = build_report(
                manifest={
                    "strict_targeted_shadow_sweep_v33_manifest_ready": True,
                    "command_groups": [
                        {
                            "group_id": "g1",
                            "cache_path": str(cache),
                            "scenario_ids": ["y2021_d1_s1_n240"],
                            "selected_scenarios": [
                                {
                                    "scenario_id": "y2021_d1_s1_n240",
                                    "year": 2021,
                                    "day": 1,
                                    "seed": 1,
                                    "max_steps": 240,
                                }
                            ],
                        }
                    ],
                },
                require_payloads=True,
            )

        self.assertTrue(report["cache_coverage_pass"])
        self.assertEqual(report["missing_key_count"], 0)
        self.assertFalse(report["controlled_replay_allowed"])

    def test_grouped_cache_coverage_fails_when_payload_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            _write_cache(cache, with_payload=False)
            report = build_report(
                manifest={
                    "strict_targeted_shadow_sweep_v33_manifest_ready": True,
                    "command_groups": [
                        {
                            "group_id": "g1",
                            "cache_path": str(cache),
                            "scenario_ids": ["y2021_d1_s1_n240"],
                            "selected_scenarios": [{"year": 2021, "day": 1, "seed": 1, "max_steps": 240}],
                        }
                    ],
                },
                require_payloads=True,
            )

        self.assertFalse(report["cache_coverage_pass"])
        self.assertGreater(report["missing_buffered_action_count"], 0)


if __name__ == "__main__":
    unittest.main()
