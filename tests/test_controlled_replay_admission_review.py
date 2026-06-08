import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.controlled_replay_admission_review import build_report
from gl_gym.experiments.controlled_replay_minimal_canary_manifest import build_report as build_manifest


def consolidation():
    return {"expanded_metadata_consolidation_pass": True}


def coverage():
    return {
        "cache_coverage_pass": True,
        "coverage_rate": 1.0,
        "missing_key_count": 0,
        "missing_buffered_action_count": 0,
        "missing_parsed_plan_count": 0,
        "cache_fill_run": False,
        "online_llm_called": False,
    }


def controlled_trace():
    return {
        "aggregate": {
            "metadata_steps": 240,
            "would_apply_steps": 3,
            "strict_applied_steps": 0,
            "unsafe_preferred_steps": 0,
            "unsafe_conflict_steps": 0,
            "unsafe_applied_steps": 0,
        }
    }


def controlled_shadow():
    return {"aggregate": {"metadata_steps": 240, "safe_gain_steps": 4, "unsafe_preferred_steps": 0}}


class TestControlledReplayAdmissionReview(unittest.TestCase):
    def _overlay(self, root: Path, *, supports_candidate: bool):
        runner = root / "gl_gym" / "experiments" / "run_frozen_benchmark.py"
        runner.parent.mkdir(parents=True)
        text = "allowed = {'llm_rspc_v2'}\n"
        if supports_candidate:
            text += "llm_rspc_v2_hot_dry_proposer_strict\n"
        runner.write_text(text, encoding="utf-8")
        return {
            "protocol_v1_overlay_root": str(root),
            "protocol_v1_overlay_validated": True,
            "actual_protocol_implementation_for_execution": "protocol_v1_controlled_canary_overlay",
        }

    def test_overlay_candidate_support_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_report(
                consolidation=consolidation(),
                canary_manifest=build_manifest(selected_cache_path="cache.json", output_root="out"),
                cache_coverage=coverage(),
                protocol_v1_overlay_preflight=self._overlay(Path(tmp), supports_candidate=False),
                historical_controlled_trace=controlled_trace(),
                historical_controlled_shadow=controlled_shadow(),
            )

        self.assertFalse(report["minimal_controlled_canary_allowed"])
        self.assertIn("overlay_runner_supports_candidate_controller", report["blockers"])

    def test_all_gates_allow_minimal_canary(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_report(
                consolidation=consolidation(),
                canary_manifest=build_manifest(selected_cache_path="cache.json", output_root="out"),
                cache_coverage=coverage(),
                protocol_v1_overlay_preflight=self._overlay(Path(tmp), supports_candidate=True),
                historical_controlled_trace=controlled_trace(),
                historical_controlled_shadow=controlled_shadow(),
            )

        self.assertTrue(report["minimal_controlled_canary_allowed"])
        self.assertEqual(report["next_action"], "generate_minimal_controlled_canary_execution_record")
        self.assertEqual(report["actual_protocol_implementation_for_execution"], "protocol_v1_controlled_canary_overlay")
        self.assertFalse(report["performance_claim_allowed"])

    def test_controlled_canary_runner_surface_validation_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            overlay = self._overlay(Path(tmp), supports_candidate=True)
            overlay["controlled_canary_runner_surface_validated"] = False
            report = build_report(
                consolidation=consolidation(),
                canary_manifest=build_manifest(selected_cache_path="cache.json", output_root="out"),
                cache_coverage=coverage(),
                protocol_v1_overlay_preflight=overlay,
                historical_controlled_trace=controlled_trace(),
                historical_controlled_shadow=controlled_shadow(),
            )

        self.assertFalse(report["minimal_controlled_canary_allowed"])
        self.assertIn("controlled_canary_runner_surface_validated", report["blockers"])


if __name__ == "__main__":
    unittest.main()
