import unittest

from gl_gym.experiments.controlled_replay_minimal_canary_execution_record import build_report
from gl_gym.experiments.controlled_replay_minimal_canary_manifest import build_report as build_manifest


class TestControlledReplayMinimalCanaryExecutionRecord(unittest.TestCase):
    def test_execution_record_uses_scope_and_command_groups(self):
        report = build_report(
            admission_review={
                "minimal_controlled_canary_allowed": True,
                "overlay_runner_path": "overlay/run_frozen_benchmark.py",
                "protocol_v1_overlay_root": "overlay",
                "actual_protocol_implementation_for_execution": "protocol_v1_controlled_canary_overlay",
            },
            canary_manifest=build_manifest(selected_cache_path="cache.json", output_root="out"),
            authorization_source="user_delegated_minimal_controlled_canary_authorization_20260525",
            output_root="out",
        )

        self.assertTrue(report["minimal_controlled_canary_authorized"])
        self.assertEqual(report["controlled_replay_scope"], "minimal_controlled_canary_only")
        self.assertEqual(report["actual_protocol_implementation_for_execution"], "protocol_v1_controlled_canary_overlay")
        self.assertFalse(report["performance_claim_allowed"])
        self.assertEqual(len(report["planned_commands"]), 2)
        self.assertIn("llm_rspc_v2,llm_rspc_v2_hot_dry_proposer_strict", report["planned_commands"][0])

    def test_execution_record_blocks_without_admission(self):
        report = build_report(
            admission_review={"minimal_controlled_canary_allowed": False},
            canary_manifest=build_manifest(selected_cache_path="cache.json", output_root="out"),
            authorization_source="user_delegated_minimal_controlled_canary_authorization_20260525",
            output_root="out",
        )

        self.assertFalse(report["minimal_controlled_canary_authorized"])
        self.assertEqual(report["next_action"], "controlled_canary_not_authorized")


if __name__ == "__main__":
    unittest.main()
