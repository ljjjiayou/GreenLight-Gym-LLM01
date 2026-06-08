import shutil
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.protocol_v1_controlled_canary_overlay_preflight import PROJECT_ROOT, build_report


RUNNER_WITH_CONTROLLED_SURFACE = """
from dataclasses import dataclass, fields

def parse_agent_config_overrides(value):
    return value

def main():
    allowed = {
        "llm_rspc_v2",
        "llm_rspc_v2_hot_dry_proposer",
        "llm_rspc_v2_hot_dry_proposer_strict",
    }
    tomato = "llm_rspc_v2_hot_dry_proposer_strict"
    rspc_hot_dry_proposer_control_enabled = True
    rspc_hot_dry_proposer_control_strict_enabled = True
    agent_config_overrides = {}
    return allowed, tomato, rspc_hot_dry_proposer_control_enabled, rspc_hot_dry_proposer_control_strict_enabled, agent_config_overrides
"""


class TestProtocolV1ControlledCanaryOverlayPreflight(unittest.TestCase):
    def setUp(self):
        self.overlay_root = (
            PROJECT_ROOT
            / "gl_gym"
            / "result"
            / "audits"
            / "protocol_v1_controlled_canary_overlay_test_case"
        )
        shutil.rmtree(self.overlay_root, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.overlay_root, ignore_errors=True)

    def _source_overlay(self, tmp: Path) -> Path:
        source = tmp / "source_overlay"
        runner = source / "gl_gym" / "experiments" / "run_frozen_benchmark.py"
        protocol = source / "gl_gym" / "experiments" / "frozen_benchmark_protocol.py"
        runner.parent.mkdir(parents=True)
        runner.write_text('allowed = {"llm_rspc_v2"}\n', encoding="utf-8")
        protocol.write_text("PROTOCOL_VERSION = 'v1'\n", encoding="utf-8")
        return source

    def test_builds_controlled_canary_overlay_without_protocol_hash_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = self._source_overlay(tmp)
            current_runner = tmp / "run_frozen_benchmark.py"
            current_runner.write_text(RUNNER_WITH_CONTROLLED_SURFACE, encoding="utf-8")

            report = build_report(
                protocol_v1_overlay_preflight={
                    "protocol_v1_overlay_root": str(source),
                    "protocol_v1_overlay_validated": True,
                },
                overlay_root=self.overlay_root,
                current_runner_path=current_runner,
                validate_compile=False,
                validate_import=False,
            )

        self.assertTrue(report["controlled_canary_overlay_validated"])
        self.assertEqual(report["actual_protocol_implementation_for_execution"], "protocol_v1_controlled_canary_overlay")
        self.assertEqual(report["protocol_hash_mismatch_count"], 0)
        self.assertTrue(report["controlled_canary_runner_surface_validated"])
        self.assertTrue(report["runner_delta"]["new_runner_supports_candidate_controller"])

    def test_blocks_when_source_overlay_is_not_validated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = self._source_overlay(tmp)
            current_runner = tmp / "run_frozen_benchmark.py"
            current_runner.write_text(RUNNER_WITH_CONTROLLED_SURFACE, encoding="utf-8")

            report = build_report(
                protocol_v1_overlay_preflight={
                    "protocol_v1_overlay_root": str(source),
                    "protocol_v1_overlay_validated": False,
                },
                overlay_root=self.overlay_root,
                current_runner_path=current_runner,
                validate_compile=False,
                validate_import=False,
            )

        self.assertFalse(report["controlled_canary_overlay_validated"])
        self.assertIn("source_protocol_v1_overlay_not_validated", report["errors"])


if __name__ == "__main__":
    unittest.main()
