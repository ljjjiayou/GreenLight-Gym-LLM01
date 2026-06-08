import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.canonical_action_diff_baseline_trace_manifest import build_report


class TestCanonicalActionDiffBaselineTraceManifest(unittest.TestCase):
    def _write_trace_pair(self, root: Path, *, rows: int = 240, subdir: str = "balanced") -> Path:
        trace_dir = root / subdir
        trace_dir.mkdir(parents=True)
        jsonl = trace_dir / "y2020_d120_s44_n240_llm_rspc_v2.jsonl"
        csv = trace_dir / "y2020_d120_s44_n240_llm_rspc_v2.csv"
        jsonl.write_text("\n".join("{}" for _ in range(rows)) + "\n", encoding="utf-8")
        csv.write_text("step,u_heating,u_co2,u_screen,u_ventilation,u_lighting,u_shading\n", encoding="utf-8")
        return trace_dir

    def test_qualifies_same_scenario_controller_and_max_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = self._write_trace_pair(root)

            report = build_report(candidate_trace_roots=[str(root)], preferred_trace_dirs=[str(trace_dir)])

        self.assertTrue(report["baseline_trace_manifest_ready"])
        self.assertTrue(report["baseline_trace_qualified"])
        self.assertTrue(report["action_diff_audit_allowed"])
        self.assertEqual(report["selected_baseline_row_count"], 240)
        self.assertEqual(report["action_diff_audit_status"], "ready_with_qualified_baseline_trace")
        self.assertFalse(report["metadata_replay_execution_allowed"])

    def test_missing_or_wrong_trace_blocks_action_diff_claim_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_trace_pair(root, rows=239)

            report = build_report(candidate_trace_roots=[str(root)])

        self.assertTrue(report["baseline_trace_manifest_ready"])
        self.assertFalse(report["baseline_trace_qualified"])
        self.assertFalse(report["action_diff_audit_allowed"])
        self.assertEqual(report["action_diff_audit_status"], "blocked_missing_baseline_trace")
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_rejected_preset_trace_path_is_not_qualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_trace_pair(root, subdir="hot_dry_relief")

            report = build_report(candidate_trace_roots=[str(root)])

        self.assertFalse(report["baseline_trace_qualified"])
        self.assertEqual(report["qualified_candidate_count"], 0)
        self.assertEqual(report["candidates"][0]["reject_reason"], "rejected_preset_trace_path")


if __name__ == "__main__":
    unittest.main()
