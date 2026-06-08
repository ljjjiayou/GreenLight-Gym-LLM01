import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.long_horizon_shadow_feasibility_execution_record import build_report


def _manifest():
    return {
        "long_horizon_shadow_feasibility_manifest_ready": True,
        "primary_scenarios": [
            {"year": 2006, "day": 182, "seed": 42, "max_steps": 240},
            {"year": 2006, "day": 182, "seed": 43, "max_steps": 240},
            {"year": 2006, "day": 182, "seed": 44, "max_steps": 240},
            {"year": 2006, "day": 162, "seed": 42, "max_steps": 240},
        ],
    }


def _overlay(runner: str):
    return {
        "controlled_canary_overlay_validated": True,
        "controlled_canary_runner_surface_validated": True,
        "protocol_hash_mismatch_count": 0,
        "overlay_runner_path": runner,
        "protocol_v1_overlay_root": "overlay",
    }


class TestLongHorizonShadowFeasibilityExecutionRecord(unittest.TestCase):
    def test_h240_record_only_allows_default_controller_and_isolated_cache(self):
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

        self.assertTrue(report["long_horizon_shadow_feasibility_executable"])
        self.assertEqual(report["controllers"], ["llm_rspc_v2"])
        self.assertIn("long_horizon_shadow_feasibility_v37_h240", report["isolated_cache_path"])
        self.assertNotEqual(report["isolated_cache_path"], report["selected_cache_path"])
        for command in report["planned_commands"]:
            self.assertIn("--plan-cache-mode", command)
            self.assertIn("record", command)
            self.assertIn("--plan-cache-key-policy", command)
            self.assertIn("scenario_timestep", command)
            self.assertNotIn("--plan-cache-strict", command)
            self.assertNotIn("llm_rspc_v2_hot_dry_proposer_strict", command)

    def test_h720_uses_only_previous_passes_and_caps_at_three(self):
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
                    "horizon_pass_base_keys": [
                        [2006, 182, 42],
                        [2006, 182, 43],
                        [2006, 182, 44],
                        [2006, 162, 42],
                    ],
                },
                api_key_present=True,
            )

        self.assertTrue(report["long_horizon_shadow_feasibility_executable"])
        self.assertEqual(report["scenario_ids"], [
            "y2006_d182_s42_n720",
            "y2006_d182_s43_n720",
            "y2006_d182_s44_n720",
        ])

    def test_runtime_ladder_blocks_without_previous_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            report = build_report(
                manifest=_manifest(),
                overlay_preflight=_overlay(str(runner)),
                authorization_source="source",
                horizon_steps=720,
                previous_result={"horizon_steps": 240, "horizon_pass_base_keys": []},
                api_key_present=True,
            )

        self.assertFalse(report["long_horizon_shadow_feasibility_authorized"])
        self.assertIn("no_eligible_scenarios_for_horizon", report["failure_taxonomy"])


if __name__ == "__main__":
    unittest.main()
