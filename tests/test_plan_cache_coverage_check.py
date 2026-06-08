import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.check_plan_cache_coverage import audit_cache_coverage, main


def _write_cache(path: Path, env_timesteps: dict[str, list[int]]) -> None:
    entries = {}
    idx = 0
    for env_id, timesteps in env_timesteps.items():
        for timestep in timesteps:
            entries[f"k{idx}"] = {"env_id": env_id, "timestep": timestep}
            idx += 1
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")


def _write_payload_cache(path: Path, env_id: str, timesteps: list[int], *, include_payloads: bool) -> None:
    entries = {}
    for idx, timestep in enumerate(timesteps):
        entry = {"env_id": env_id, "timestep": timestep}
        if include_payloads:
            entry["buffered_action"] = {"action_set": True}
            entry["parsed_plan"] = {"target_temp": 20.0}
        entries[f"k{idx}"] = entry
    path.write_text(json.dumps({"entries": entries}), encoding="utf-8")


class TestPlanCacheCoverageCheck(unittest.TestCase):
    def test_full_scenario_coverage_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            _write_cache(cache, {"TomatoEnv_y2015_d120_s42": list(range(0, 240, 12))})

            audit = audit_cache_coverage(
                plan_cache_path=cache,
                years=[2015],
                days=[120],
                seeds=[42],
                max_steps=240,
                control_interval=12,
            )

        self.assertTrue(audit["can_strict_replay"])
        self.assertEqual(audit["coverage_rate"], 1.0)

    def test_missing_env_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            _write_cache(cache, {"TomatoEnv_y2015_d120_s42": list(range(0, 240, 12))})

            audit = audit_cache_coverage(
                plan_cache_path=cache,
                years=[2015],
                days=[120],
                seeds=[42, 43],
                max_steps=240,
                control_interval=12,
            )

        self.assertFalse(audit["can_strict_replay"])
        self.assertIn("TomatoEnv_y2015_d120_s43", audit["missing_envs"])

    def test_short_last_timestep_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            _write_cache(cache, {"TomatoEnv_y2015_d120_s42": list(range(0, 120, 12))})

            audit = audit_cache_coverage(
                plan_cache_path=cache,
                years=[2015],
                days=[120],
                seeds=[42],
                max_steps=240,
                control_interval=12,
                min_entries_per_env=1,
            )

        self.assertFalse(audit["can_strict_replay"])
        self.assertIn("last_timestep_too_early", audit["envs"]["TomatoEnv_y2015_d120_s42"]["issues"])

    def test_cli_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache = tmp_path / "cache.json"
            out_json = tmp_path / "coverage.json"
            out_md = tmp_path / "coverage.md"
            _write_cache(cache, {"TomatoEnv_y2015_d120_s42": list(range(0, 240, 12))})

            code = main(
                [
                    "--plan-cache-path",
                    str(cache),
                    "--years",
                    "2015",
                    "--days",
                    "120",
                    "--seeds",
                    "42",
                    "--max-steps",
                    "240",
                    "--output-json",
                    str(out_json),
                    "--output-md",
                    str(out_md),
                ]
            )

            self.assertEqual(code, 0)
            self.assertTrue(out_json.exists())
            self.assertIn("Plan Cache Coverage Check", out_md.read_text(encoding="utf-8"))

    def test_scenario_specs_avoid_cross_product(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            _write_cache(
                cache,
                {
                    "TomatoEnv_y2015_d120_s42": list(range(0, 240, 12)),
                    "TomatoEnv_y2020_d180_s43": list(range(0, 240, 12)),
                },
            )

            audit = audit_cache_coverage(
                plan_cache_path=cache,
                years=[],
                days=[],
                seeds=[],
                max_steps=240,
                control_interval=12,
                scenario_specs=[
                    {"year": 2015, "day": 120, "seed": 42, "max_steps": 240},
                    {"year": 2020, "day": 180, "seed": 43, "max_steps": 240},
                ],
            )

        self.assertTrue(audit["can_strict_replay"])
        self.assertEqual(audit["expected_env_count"], 2)

    def test_require_payloads_blocks_missing_buffered_action_or_parsed_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            _write_payload_cache(
                cache,
                "TomatoEnv_y2020_d120_s44",
                list(range(0, 240, 12)),
                include_payloads=False,
            )

            audit = audit_cache_coverage(
                plan_cache_path=cache,
                years=[],
                days=[],
                seeds=[],
                max_steps=240,
                control_interval=12,
                scenario_specs=[{"year": 2020, "day": 120, "seed": 44, "max_steps": 240}],
                require_payloads=True,
            )

        self.assertFalse(audit["cache_coverage_pass"])
        self.assertGreater(audit["missing_buffered_action_count"], 0)
        self.assertGreater(audit["missing_parsed_plan_count"], 0)

    def test_require_payloads_passes_when_required_payloads_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            _write_payload_cache(
                cache,
                "TomatoEnv_y2020_d120_s44",
                list(range(0, 240, 12)),
                include_payloads=True,
            )

            audit = audit_cache_coverage(
                plan_cache_path=cache,
                years=[],
                days=[],
                seeds=[],
                max_steps=240,
                control_interval=12,
                scenario_specs=[{"year": 2020, "day": 120, "seed": 44, "max_steps": 240}],
                require_payloads=True,
            )

        self.assertTrue(audit["cache_coverage_pass"])
        self.assertEqual(audit["missing_buffered_action_count"], 0)
        self.assertEqual(audit["missing_parsed_plan_count"], 0)


if __name__ == "__main__":
    unittest.main()
