import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.metadata_replay_cache_scenario_plan import build_report as build_cache_plan
from gl_gym.experiments.metadata_replay_expanded_cache_coverage_plan import build_report


def _write_cache(path: Path, env_id: str) -> None:
    path.write_text(json.dumps({"entries": {"k0": {"env_id": env_id, "timestep": 0}}}), encoding="utf-8")


class TestMetadataReplayExpandedCacheCoveragePlan(unittest.TestCase):
    def test_only_canonical_enters_initial_plan_when_other_families_lack_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "holdout_seed42_43_hot_dry_qwen_20260517_merged.json"
            _write_cache(cache, "TomatoEnv_y2020_d120_s44")
            cache_plan = build_cache_plan(
                selected_cache_path=str(cache),
                scenario_ids=["y2020_d120_s44_n240"],
                candidate_cache_paths=[str(cache)],
            )

            report = build_report(
                cache_scenario_plan=cache_plan,
                counterexample_suite={
                    "windows": [
                        {
                            "regime": "neutral_no_risk",
                            "scenario_id": "y2010_d120_s44_n240",
                        }
                    ]
                },
                default_path_plan={"scenario_families": ["pure hot-dry", "mixed dry-dew"]},
                cache_coverage={"cache_coverage_pass": True},
            )

        rows = {row["family"]: row for row in report["rows"]}
        self.assertTrue(report["expanded_cache_coverage_plan_ready"])
        self.assertFalse(report["expanded_cache_coverage_ready"])
        self.assertEqual(rows["canonical_failure"]["coverage_status"], "canonical_coverage_passed")
        self.assertTrue(rows["canonical_failure"]["can_enter_initial_strict_metadata_replay_plan"])
        self.assertEqual(rows["neutral_no_risk"]["coverage_status"], "blocked_missing_cache_or_scenario")
        self.assertEqual(rows["pure_hot_dry"]["coverage_status"], "blocked_missing_cache_or_scenario")
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertFalse(report["cache_fill_run"])
        self.assertFalse(report["online_llm_called"])


if __name__ == "__main__":
    unittest.main()
