import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.controlled_canary_shadow_trigger_source_sweep_execution_record import (
    EXPECTED_SCENARIOS,
    build_report,
)


def prereq(runner: str):
    return {
        "readiness_v25": {
            "shadow_trigger_source_manifest_ready": True,
            "shadow_trigger_source_cache_precheck_pass": True,
        },
        "source_manifest": {
            "shadow_trigger_source_manifest_ready": True,
            "shadow_only": True,
            "scenario_ids": EXPECTED_SCENARIOS,
            "selected_cache_path": "cache.json",
        },
        "cache_coverage": {
            "cache_coverage_pass": True,
            "coverage_rate": 1.0,
            "missing_key_count": 0,
            "missing_buffered_action_count": 0,
            "missing_parsed_plan_count": 0,
        },
        "overlay_preflight": {
            "controlled_canary_overlay_validated": True,
            "controlled_canary_runner_surface_validated": True,
            "protocol_hash_mismatch_count": 0,
            "overlay_runner_path": runner,
            "protocol_v1_overlay_root": "overlay",
        },
    }


class TestControlledCanaryShadowTriggerSourceSweepExecutionRecord(unittest.TestCase):
    def test_authorizes_only_llm_rspc_v2_with_strict_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            report = build_report(
                **prereq(str(runner)),
                authorization_source="user_delegated_shadow_trigger_source_sweep_authorization_20260526",
                output_root="out",
            )

        self.assertTrue(report["shadow_trigger_source_sweep_authorized"])
        self.assertEqual(report["controllers"], ["llm_rspc_v2"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["controlled_replay_execution_allowed"])
        self.assertIn("llm_rspc_v2_hot_dry_proposer_strict", report["blocked_controllers"])
        for command in report["planned_commands"]:
            self.assertIn("--controllers", command)
            self.assertIn("llm_rspc_v2", command)
            self.assertNotIn("llm_rspc_v2_hot_dry_proposer_strict", command)
            self.assertIn("--plan-cache-mode", command)
            self.assertIn("replay", command)
            self.assertIn("--plan-cache-strict", command)
            self.assertIn("--plan-cache-key-policy", command)
            self.assertIn("scenario_timestep", command)

    def test_blocks_non_default_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            report = build_report(
                **prereq(str(runner)),
                authorization_source="source",
                output_root="out",
                controllers=["llm_rspc_v2", "llm_rspc_v2_hot_dry_proposer_strict"],
            )

        self.assertFalse(report["shadow_trigger_source_sweep_authorized"])
        self.assertFalse(report["precheck"]["controller_scope_ok"])

    def test_blocks_when_cache_coverage_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            inputs = prereq(str(runner))
            inputs["cache_coverage"] = {"cache_coverage_pass": False}
            report = build_report(
                **inputs,
                authorization_source="source",
                output_root="out",
            )

        self.assertFalse(report["shadow_trigger_source_sweep_authorized"])
        self.assertEqual(report["next_action"], "shadow_trigger_source_sweep_not_authorized")


if __name__ == "__main__":
    unittest.main()
