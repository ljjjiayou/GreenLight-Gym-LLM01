import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments import qwen37plus_structured_anchor_failure_diagnosis_v62 as v62


def _entry(
    *,
    raw_response: str,
    errors: list[str],
    empty: bool = False,
    env_id: str = "TomatoEnv_y2010_d180_s43",
    timestep: int = 1,
    attempt: int = 1,
) -> dict[str, object]:
    return {
        "env_id": env_id,
        "timestep": timestep,
        "model_name": "qwen3.7-plus",
        "raw_response": raw_response,
        "llm_duration_seconds": 37.4,
        "structured_anchor": {
            "attempt": attempt,
            "attempted": True,
            "valid": False,
            "clean_planning_evidence": False,
            "empty": empty,
            "errors": errors,
        },
    }


class TestQwen37PlusStructuredAnchorFailureDiagnosisV62(unittest.TestCase):
    def test_real_v61_cache_has_exact_invalid_catalog(self):
        catalog = v62.build_invalid_response_catalog()

        self.assertEqual(catalog["invalid_or_empty_entry_count"], 13)
        self.assertTrue(catalog["catalog_count_matches_v61_failure"])
        self.assertEqual(catalog["primary_failure_type_counts"].get("invalid_json_truncated"), 7)
        self.assertEqual(catalog["primary_failure_type_counts"].get("empty_raw_response"), 6)
        self.assertEqual(catalog["salvaged_or_truncated_clean_evidence_count"], 0)
        for item in catalog["invalid_entries"]:
            self.assertIn(item["scenario_id"], v62.FAILURE_SCENARIOS)
            self.assertFalse(item["clean_planning_evidence"])

    def test_classifies_truncated_json_and_never_counts_salvage_as_clean(self):
        primary, secondary, diagnostics = v62.classify_invalid_entry(
            _entry(
                raw_response='{"structured_planning_anchor":{"profile_intent":"hot_dry","target_temp":26.0,',
                errors=["invalid_json_anchor"],
            )
        )

        self.assertEqual(primary, "invalid_json_truncated")
        self.assertIn("retry_not_invoked_or_not_effective", secondary)
        self.assertTrue(diagnostics["looks_truncated"])
        self.assertTrue(diagnostics["salvage_must_not_count_as_clean_evidence"])

    def test_classifies_empty_response_as_empty_and_provider_empty(self):
        primary, secondary, diagnostics = v62.classify_invalid_entry(
            _entry(raw_response="", errors=["empty_anchor"], empty=True)
        )

        self.assertEqual(primary, "empty_raw_response")
        self.assertIn("provider_returned_empty", secondary)
        self.assertIn("retry_not_invoked_or_not_effective", secondary)
        self.assertEqual(diagnostics["raw_length"], 0)

    def test_catalog_filters_to_fixed_v61_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            cache.write_text(
                json.dumps(
                    {
                        "entries": {
                            "fixed": _entry(
                                raw_response='{"structured_planning_anchor":{"profile_intent":"x"',
                                errors=["invalid_json_anchor"],
                                env_id="TomatoEnv_y2018_d181_s42",
                                timestep=12,
                            ),
                            "outside": _entry(
                                raw_response="",
                                errors=["empty_anchor"],
                                empty=True,
                                env_id="TomatoEnv_y2015_d120_s42",
                                timestep=12,
                            ),
                        }
                    }
                ),
                encoding="utf-8",
            )

            catalog = v62.build_invalid_response_catalog(cache_path=cache)

        self.assertEqual(catalog["invalid_or_empty_entry_count"], 1)
        self.assertEqual(catalog["invalid_entries"][0]["scenario_id"], "y2018_d181_s42_n720")

    def test_diagnosis_and_readiness_do_not_authorize_execution_or_claims(self):
        catalog = v62.build_invalid_response_catalog()
        diagnosis = v62.build_failure_diagnosis(catalog)
        repair = v62.build_repair_design(diagnosis)
        readiness = v62.build_readiness(diagnosis, catalog, repair)

        self.assertEqual(
            diagnosis["dominant_root_cause"],
            "structured_anchor_format_truncation_and_empty_response_with_missing_compact_retry",
        )
        self.assertTrue(repair["repair_design_ready"])
        self.assertFalse(repair["rollout_execution_record_generated"])
        self.assertEqual(readiness["next_action"], "qwen37plus_structured_anchor_prompt_retry_opt_in_shadow_rollout_plan")
        for key in (
            "controlled_replay_allowed",
            "controlled_replay_execution_allowed",
            "metadata_replay_execution_allowed",
            "performance_claim_allowed",
            "promotion_evidence",
        ):
            self.assertFalse(readiness[key])


if __name__ == "__main__":
    unittest.main()
