import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.stable_long_horizon_evaluation_pool_execution_record import build_report


def _scenario(year, day, seed, tier="tier_a_selected_cache", regime="tier_a_selected_cache_d59", cache="selected_cache_replay"):
    return {
        "scenario_id": f"y{year}_d{day}_s{seed}_n240",
        "year": year,
        "day": day,
        "seed": seed,
        "max_steps": 240,
        "source_tier": tier,
        "regime": regime,
        "cache_strategy_h240": cache,
    }


def _manifest():
    return {
        "stable_long_horizon_manifest_ready": True,
        "candidate_scenarios": [
            _scenario(2010, 59, 42),
            _scenario(2010, 59, 43),
            _scenario(2010, 59, 44),
            _scenario(2006, 160, 42, "tier_b_weather_source", "tier_b_weather_source_d160", "isolated_record"),
        ],
    }


def _overlay(runner: str):
    return {
        "controlled_canary_overlay_validated": True,
        "controlled_canary_runner_surface_validated": True,
        "protocol_hash_mismatch_count": 0,
        "overlay_runner_path": runner,
        "overlay_root": "overlay",
    }


class TestStableLongHorizonEvaluationPoolExecutionRecord(unittest.TestCase):
    def test_h240_blocks_controlled_controller_and_uses_expected_cache_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            report = build_report(
                manifest=_manifest(),
                overlay_preflight=_overlay(str(runner)),
                authorization_source="source",
                horizon_steps=240,
                api_key_present=True,
            )

        self.assertTrue(report["stable_long_horizon_execution_executable"])
        self.assertEqual(report["controllers"], ["ppo", "llm_rspc_v2"])
        self.assertNotEqual(report["isolated_cache_path"], report["selected_cache_path"])
        controllers = {record["controller"] for record in report["planned_command_records"]}
        self.assertEqual(controllers, {"ppo", "llm_rspc_v2"})
        for record in report["planned_command_records"]:
            command = record["command"]
            self.assertNotIn("llm_rspc_v2_hot_dry_proposer_strict", command)
            if record["cache_strategy"] == "isolated_record":
                self.assertIn("record", command)
                self.assertIn(report["isolated_cache_path"], command)
            if record["cache_strategy"] == "selected_cache_replay":
                self.assertIn("replay", command)
                self.assertIn("--plan-cache-strict", command)

    def test_h720_uses_previous_stable_and_caps_two_per_regime(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            report = build_report(
                manifest=_manifest(),
                overlay_preflight=_overlay(str(runner)),
                authorization_source="source",
                horizon_steps=720,
                previous_result={
                    "horizon_steps": 240,
                    "stable_base_keys": [
                        [2010, 59, 42],
                        [2010, 59, 43],
                        [2010, 59, 44],
                        [2006, 160, 42],
                    ],
                },
                api_key_present=True,
            )

        self.assertTrue(report["stable_long_horizon_execution_executable"])
        self.assertEqual(
            [item["scenario_id"] for item in report["eligible_scenarios"]],
            ["y2010_d59_s42_n720", "y2010_d59_s43_n720", "y2006_d160_s42_n720"],
        )

    def test_horizon_blocks_without_previous_stable_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            report = build_report(
                manifest=_manifest(),
                overlay_preflight=_overlay(str(runner)),
                authorization_source="source",
                horizon_steps=720,
                previous_result={"horizon_steps": 240, "stable_base_keys": []},
                api_key_present=True,
            )

        self.assertFalse(report["stable_long_horizon_execution_authorized"])
        self.assertIn("no_eligible_scenarios_for_horizon", report["failure_taxonomy"])


if __name__ == "__main__":
    unittest.main()
