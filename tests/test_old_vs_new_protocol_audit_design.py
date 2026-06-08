import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.old_vs_new_protocol_audit_design import (
    build_protocol_v1_snapshot_manifest,
    build_readiness_report,
    build_report,
)


class TestOldVsNewProtocolAuditDesign(unittest.TestCase):
    def test_design_only_and_replay_blocked(self):
        report = build_report(
            {
                "protocol_baseline_authorized": False,
                "recommended_decision": "partial_accept_review_required",
                "safety_boundary_test_oracle_hunk_count": 2,
            },
            plan_cache_path="gl_gym/result/plan_cache/llm_plan_cache.json",
            controller="llm_rspc_v2",
            scenario_list=["y2020_d120_s44_n240"],
            max_steps=240,
            plan_cache_key_policy="scenario_timestep",
        )

        self.assertFalse(report["replay_run"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertTrue(report["design_ready"])
        self.assertEqual(report["fixed_inputs"]["controller"], "llm_rspc_v2")
        self.assertEqual(report["safety_boundary_test_oracle_hunk_count"], 2)
        self.assertIn("gate failure count and reasons", report["compare_fields"])

    def test_readiness_requires_old_protocol_snapshot(self):
        design = build_report(
            {
                "protocol_baseline_authorized": False,
                "protocol_user_decision_complete": True,
                "recommended_decision": "partial_accept_review_required",
            },
            plan_cache_path="cache.json",
            controller="llm_rspc_v2",
            scenario_list=["y2020_d120_s44_n240"],
            max_steps=240,
            plan_cache_key_policy="scenario_timestep",
        )
        design["protocol_user_decision_complete"] = True

        report = build_readiness_report(design, old_protocol_snapshot_path="")

        self.assertFalse(report["old_vs_new_protocol_audit_ready"])
        self.assertFalse(report["old_protocol_snapshot_available"])
        self.assertEqual(report["next_action"], "old_protocol_snapshot_required_before_audit")

    def test_readiness_passes_when_snapshot_exists_and_decision_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            snapshot = Path(tmp) / "old_protocol.json"
            snapshot.write_text("{}", encoding="utf-8")
            design = {"design_ready": True, "protocol_user_decision_complete": True}

            report = build_readiness_report(design, old_protocol_snapshot_path=str(snapshot))

        self.assertTrue(report["old_vs_new_protocol_audit_ready"])
        self.assertTrue(report["old_protocol_snapshot_available"])

    def test_head_snapshot_manifest_hashes_tracked_file_without_replay(self):
        report = build_protocol_v1_snapshot_manifest(
            source_git_ref="HEAD",
            tracked_paths=["gl_gym/experiments/frozen_benchmark_protocol.py"],
        )

        self.assertTrue(report["snapshot_available"])
        self.assertFalse(report["replay_run"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertEqual(report["files"][0]["path"], "gl_gym/experiments/frozen_benchmark_protocol.py")
        self.assertTrue(report["files"][0]["sha256"])

    def test_readiness_accepts_snapshot_manifest(self):
        manifest = {
            "snapshot_available": True,
            "source_git_ref": "HEAD",
            "source_git_commit": "abc",
        }
        design = {"design_ready": True, "protocol_user_decision_complete": True}

        report = build_readiness_report(design, snapshot_manifest=manifest)

        self.assertEqual(report["schema_version"], "old_vs_new_protocol_audit_readiness_v2")
        self.assertTrue(report["old_vs_new_protocol_audit_ready"])
        self.assertTrue(report["protocol_v1_snapshot_manifest_available"])
        self.assertEqual(report["protocol_v1_snapshot_source_git_ref"], "HEAD")


if __name__ == "__main__":
    unittest.main()
