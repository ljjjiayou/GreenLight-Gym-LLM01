import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.metadata_replay_cache_scenario_plan import build_report


def _write_cache(path: Path, env_id: str) -> None:
    path.write_text(
        json.dumps({"entries": {"k0": {"env_id": env_id, "timestep": 0}}}),
        encoding="utf-8",
    )


class TestMetadataReplayCacheScenarioPlan(unittest.TestCase):
    def test_rejects_ambiguous_default_cache_without_explicit_allowance(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "llm_plan_cache.json"
            _write_cache(cache, "TomatoEnv_y2020_d120_s44")

            report = build_report(
                selected_cache_path=str(cache),
                scenario_ids=["y2020_d120_s44_n240"],
                candidate_cache_paths=[str(cache)],
            )

        self.assertFalse(report["cache_scenario_plan_ready"])
        self.assertFalse(report["selected_cache_accepted_for_coverage_check"])
        self.assertEqual(
            report["cache_path_selection_status"],
            "blocked_ambiguous_default_cache_requires_explicit_coverage_proof",
        )
        self.assertFalse(report["metadata_replay_allowed"])

    def test_accepts_non_default_cache_for_coverage_check_when_env_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "holdout_seed42_43_hot_dry_qwen_20260517_merged.json"
            _write_cache(cache, "TomatoEnv_y2020_d120_s44")

            report = build_report(
                selected_cache_path=str(cache),
                scenario_ids=["y2020_d120_s44_n240"],
                candidate_cache_paths=[str(cache)],
            )

        self.assertTrue(report["cache_scenario_plan_ready"])
        self.assertTrue(report["selected_cache_accepted_for_coverage_check"])
        self.assertEqual(report["next_action"], "run_no_fill_cache_coverage_check")
        self.assertEqual(report["cache_fill_policy"], "no_fill")
        self.assertEqual(report["online_llm_policy"], "no_call")
        self.assertEqual(report["schema_version"], "metadata_replay_cache_scenario_plan_v2")
        self.assertEqual(report["coverage_claim_scope"], "canonical_failure_only")

    def test_deduplicates_canonical_scenario_and_explains_seed_filename_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "holdout_seed42_43_hot_dry_qwen_20260517_merged.json"
            _write_cache(cache, "TomatoEnv_y2020_d120_s44")

            report = build_report(
                selected_cache_path=str(cache),
                scenario_ids=["y2020_d120_s44_n240", "y2020_d120_s44_n240"],
                candidate_cache_paths=[str(cache)],
            )

        self.assertTrue(report["scenario_list_deduplicated"])
        self.assertEqual(report["duplicate_scenario_ids"], ["y2020_d120_s44_n240"])
        self.assertEqual(report["input_scenario_count"], 2)
        self.assertEqual(report["unique_scenario_count"], 1)
        self.assertTrue(report["cache_filename_seed_mismatch"])
        self.assertTrue(report["cache_filename_seed_mismatch_explained"])
        self.assertIn("cache contents include all requested env_id", report["cache_provenance"])


if __name__ == "__main__":
    unittest.main()
