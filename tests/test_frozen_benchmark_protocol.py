import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.agent.plan_cache import SCHEMA_VERSION
from gl_gym.experiments.frozen_benchmark_protocol import (
    ScenarioSpec,
    audit_plan_cache,
    build_controller_delta_report,
    build_canary_commands,
    build_scenarios,
    compare_controller_deltas,
    compare_record_replay,
    make_manifest,
    validate_shadow_invariance,
    validate_strict_replay_result,
)


def _summary(scenario_id, controller, **aggregate_overrides):
    aggregate = {
        "steps": 24,
        "total_reward": 1.0,
        "total_profit": -0.1,
        "total_revenue": 0.0,
        "total_heat_cost": 0.1,
        "total_co2_cost": 0.0,
        "total_elec_cost": 0.0,
        "total_temp_violation": 0.0,
        "total_rh_violation": 0.0,
        "total_rh_low_violation": 0.0,
        "total_rh_high_violation": 0.0,
        "total_vpd_high_excess": 0.0,
        "rh_ge_90_steps": 0,
        "rh_ge_94_steps": 0,
        "dew_margin_air_lt1_steps": 0,
        "dew_margin_air_lt0_steps": 0,
        "canopy_dew_margin_lt1_steps": 0,
        "canopy_dew_margin_lt0_steps": 0,
        "mean_u_heating": 0.1,
        "mean_u_co2": 0.0,
        "mean_u_screen": 0.5,
        "mean_u_ventilation": 0.2,
        "mean_u_lighting": 0.0,
        "mean_u_shading": 0.0,
        "plan_cache_enabled_steps": 24,
        "plan_cache_hit_steps": 24,
        "source_counts": {"anchor": 24},
    }
    aggregate.update(aggregate_overrides)
    return {
        "scenario_id": scenario_id,
        "controller": controller,
        "aggregate": aggregate,
    }


class TestFrozenBenchmarkProtocol(unittest.TestCase):
    def test_build_scenarios_and_manifest(self):
        scenarios = build_scenarios(years=[2010], days=[59, 120], seeds=[42], max_steps=240)

        self.assertEqual([s.scenario_id for s in scenarios], ["y2010_d59_s42_n240", "y2010_d120_s42_n240"])
        manifest = make_manifest(scenarios=scenarios, cache_path="cache.json", commit="abc123")
        self.assertEqual(manifest["commit"], "abc123")
        self.assertEqual(manifest["scenarios"][0]["status"], "pending")
        self.assertEqual(manifest["scenarios"][1]["env_id"], "TomatoEnv_y2010_d120_s42")

    def test_audit_plan_cache_detects_missing_and_low_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "entries": {
                            "a": {
                                "env_id": "TomatoEnv_y2010_d59_s42",
                                "buffered_action": {"u_boil": 0.1},
                                "parsed_plan": {"target_rh": 76.0},
                            },
                            "b": {
                                "env_id": "TomatoEnv_y2010_d59_s42",
                                "buffered_action": {"u_boil": 0.2},
                                "parsed_plan": {"target_rh": 75.0},
                            },
                            "c": {
                                "env_id": "TomatoEnv_y2010_d120_s42",
                                "llm_action_found": False,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = audit_plan_cache(
                path,
                scenarios=[
                    ScenarioSpec(2010, 59, 42),
                    ScenarioSpec(2010, 120, 42),
                    ScenarioSpec(2020, 240, 44),
                ],
                min_entries_per_env=2,
            )

        self.assertFalse(result["ok"])
        self.assertEqual(result["entry_count"], 3)
        self.assertIn("TomatoEnv_y2020_d240_s44", result["missing_envs"])
        self.assertIn("TomatoEnv_y2010_d120_s42", result["low_coverage_envs"])
        self.assertEqual(result["written_empty_count"], 1)
        self.assertEqual(result["missing_buffered_action_count"], 1)

    def test_validate_strict_replay_result_flags_cache_miss(self):
        result = {
            "summaries": [
                _summary(
                    "y2010_d59_s42_n24",
                    "llm",
                    plan_cache_hit_steps=12,
                    source_counts={"anchor": 23, "unknown": 1},
                ),
                _summary("y2010_d59_s42_n24", "ppo"),
            ]
        }

        check = validate_strict_replay_result(result)

        self.assertFalse(check["ok"])
        self.assertEqual(check["failure_count"], 1)
        self.assertIn("cache_not_hit_all_steps", check["failures"][0]["reasons"])
        self.assertIn("unknown_source_steps", check["failures"][0]["reasons"])

    def test_shadow_invariance_accepts_diagnostics_only(self):
        result = {
            "summaries": [
                _summary("y2010_d59_s42_n24", "llm"),
                _summary(
                    "y2010_d59_s42_n24",
                    "llm_sero_shadow",
                    mc_sero_available_steps=24,
                    mc_sero_would_select_steps=7,
                    mc_sero_best_candidate_counts={"heat_buffer": 4, "safe_dehumidify": 3},
                    mc_sero_reject_reason_counts={"insufficient_margin": 2},
                ),
            ]
        }

        check = validate_shadow_invariance(result)

        self.assertTrue(check["ok"])
        self.assertEqual(check["shadow_stats"]["total_available_steps"], 24)
        self.assertEqual(check["shadow_stats"]["total_would_select_steps"], 7)
        self.assertEqual(check["shadow_stats"]["best_candidate_counts"]["heat_buffer"], 4)

    def test_shadow_invariance_rejects_behavior_delta(self):
        result = {
            "summaries": [
                _summary("y2010_d59_s42_n24", "llm"),
                _summary("y2010_d59_s42_n24", "llm_sero_shadow", total_reward=2.0),
            ]
        }

        check = validate_shadow_invariance(result)

        self.assertFalse(check["ok"])
        self.assertEqual(check["failures"][0]["reason"], "shadow_changed_behavior")
        self.assertIn("total_reward", check["failures"][0]["deltas"])

    def test_record_replay_compare_flags_aggregate_delta(self):
        record = {"summaries": [_summary("y2010_d59_s42_n24", "llm", total_profit=-0.1)]}
        replay = {"summaries": [_summary("y2010_d59_s42_n24", "llm", total_profit=-0.2)]}

        check = compare_record_replay(record, replay)

        self.assertFalse(check["ok"])
        self.assertEqual(check["failures"][0]["reason"], "aggregate_delta")
        self.assertAlmostEqual(check["failures"][0]["deltas"]["total_profit"], -0.1)

    def test_controller_delta_report_pairs_scenarios_and_blocks(self):
        result = {
            "summaries": [
                _summary("y2010_d59_s44_n240", "llm", total_reward=100.0, total_rh_low_violation=10.0, total_vpd_high_excess=4.0),
                _summary(
                    "y2010_d59_s44_n240",
                    "llm_rspc_v2",
                    total_reward=104.0,
                    total_rh_low_violation=2.0,
                    total_vpd_high_excess=1.0,
                    canopy_dew_margin_lt1_steps=3,
                    tomato_safety_v2_applied_steps=12,
                ),
                _summary("y2010_d59_s44_n240", "llm_sero_shadow", total_reward=100.0, total_rh_low_violation=10.0, total_vpd_high_excess=4.0),
            ]
        }

        comparison = compare_controller_deltas(result)
        report = build_controller_delta_report(comparison)

        self.assertTrue(comparison["ok"])
        self.assertEqual(comparison["scenario_count"], 1)
        self.assertEqual(comparison["block_count"], 1)
        self.assertAlmostEqual(comparison["scenarios"][0]["delta"]["total_rh_low_violation"], -8.0)
        self.assertAlmostEqual(comparison["blocks"][0]["mean_delta"]["total_vpd_high_excess"], -3.0)
        self.assertIn("y2010_d59_s44_n240", report)

    def test_canary_commands_are_single_scenario_and_strict_replay(self):
        commands = build_canary_commands(
            [ScenarioSpec(2020, 240, 42, 24)],
            cache_path="plans.json",
            output_dir="out",
        )

        self.assertEqual(len(commands), 1)
        self.assertIn("--controllers llm", commands[0]["record"])
        self.assertIn("--controllers llm,llm_sero_shadow", commands[0]["replay"])
        self.assertIn("--plan-cache-strict", commands[0]["replay"])
        self.assertIn("--years 2020", commands[0]["record"])


if __name__ == "__main__":
    unittest.main()
