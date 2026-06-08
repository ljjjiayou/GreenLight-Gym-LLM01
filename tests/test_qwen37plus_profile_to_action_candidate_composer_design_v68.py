import csv
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments import qwen37plus_profile_to_action_candidate_composer_design_v68 as v68


def _gap(*, profile_available: int = 2, eligible: int = 2) -> dict[str, object]:
    return {
        "artifact": "qwen37plus_profile_to_action_candidate_gap_audit_20260603_v67",
        "row_count": 2,
        "profile_candidate_available_steps": profile_available,
        "profile_rspc_shadow_eligible_steps": eligible,
        "profile_low_level_action_generated_steps": 0,
        "normal_path_profile_action_candidate_steps": 0,
        "candidate_composer_missing_steps": 2,
        "fallback_should_not_have_been_primary_steps": 2,
        "profile_to_action_gap_confirmed": True,
        "dominant_gap": "profile_to_action_candidate_composer_missing",
        "next_action": "profile_to_action_candidate_composer_design_plan",
    }


def _v67_readiness() -> dict[str, object]:
    return {
        "artifact": "metadata_replay_readiness_checklist_20260603_v67",
        "metadata_sufficient_for_offline_audit": True,
        "next_action": "profile_to_action_candidate_composer_design_plan",
    }


def _write_trace(path: Path, *, omit: set[str] | None = None) -> None:
    omit = omit or set()
    fields = [field for field in v68.REQUIRED_TRACE_METADATA_FIELDS if field not in omit]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: "" for field in fields})


def _metadata_inventory(*, omit: set[str] | None = None) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        trace_dir = Path(tmp)
        scenario = v68.FAILURE_SCENARIOS[0]
        _write_trace(trace_dir / f"{scenario}_llm_rspc_v2.csv", omit=omit)
        return v68.build_metadata_inventory(trace_dir=trace_dir, failure_scenarios=[scenario])


class TestQwen37PlusProfileToActionCandidateComposerDesignV68(unittest.TestCase):
    def test_contract_schema_covers_required_profile_action_fields(self):
        contract = v68.build_candidate_contract()
        output = contract["output_contract"]
        required = set(output["required_fields"])
        score_fields = set(output["field_schema"]["score_terms"]["fields"])

        self.assertIn("profile_name", required)
        self.assertIn("raw_action", required)
        self.assertIn("post_tomato_action", required)
        self.assertIn("score_terms", required)
        self.assertIn("tomato_safety_v2_applied", required)
        self.assertIn("rejection_reason", required)
        self.assertEqual(output["candidate_source_value"], "normal_path_profile_candidate")
        self.assertTrue(set(v68.SCORE_TERM_FIELDS) <= score_fields)

    def test_v67_like_gap_routes_to_minimal_shadow_patch_not_fallback_patch(self):
        contract = v68.build_candidate_contract()
        metadata = _metadata_inventory()
        gap = _gap()
        design = v68.build_composer_design(
            gap=gap,
            v67_readiness=_v67_readiness(),
            contract=contract,
            metadata_inventory=metadata,
        )
        readiness = v68.build_readiness(
            design=design,
            contract=contract,
            gap=gap,
            v67_readiness=_v67_readiness(),
            metadata_inventory=metadata,
        )

        self.assertEqual(readiness["next_action"], "minimal_normal_path_profile_arbitration_shadow_patch_plan")
        self.assertEqual(design["design_decision"], "define_profile_to_action_candidate_composer_contract_not_fallback_patch")
        self.assertNotEqual(readiness["next_action"], "fallback_patch_plan")

    def test_missing_metadata_routes_to_instrumentation_plan(self):
        contract = v68.build_candidate_contract()
        metadata = _metadata_inventory(omit={"profile_rspc_shadow_best_score"})
        design = v68.build_composer_design(
            gap=_gap(),
            v67_readiness=_v67_readiness(),
            contract=contract,
            metadata_inventory=metadata,
        )
        readiness = v68.build_readiness(
            design=design,
            contract=contract,
            gap=_gap(),
            v67_readiness=_v67_readiness(),
            metadata_inventory=metadata,
        )

        self.assertEqual(readiness["next_action"], "profile_action_composer_instrumentation_plan")
        self.assertIn("profile_rspc_shadow_best_score", readiness["missing_required_metadata_fields"])

    def test_missing_profile_payload_routes_to_profile_generator_repair(self):
        contract = v68.build_candidate_contract()
        metadata = _metadata_inventory()
        gap = _gap(profile_available=0, eligible=0)
        design = v68.build_composer_design(
            gap=gap,
            v67_readiness=_v67_readiness(),
            contract=contract,
            metadata_inventory=metadata,
        )
        readiness = v68.build_readiness(
            design=design,
            contract=contract,
            gap=gap,
            v67_readiness=_v67_readiness(),
            metadata_inventory=metadata,
        )

        self.assertEqual(readiness["next_action"], "profile_generator_payload_schema_repair_plan")

    def test_artifacts_never_authorize_replay_promotion_or_rollout_commands(self):
        contract = v68.build_candidate_contract()
        metadata = _metadata_inventory()
        design = v68.build_composer_design(
            gap=_gap(),
            v67_readiness=_v67_readiness(),
            contract=contract,
            metadata_inventory=metadata,
        )
        readiness = v68.build_readiness(
            design=design,
            contract=contract,
            gap=_gap(),
            v67_readiness=_v67_readiness(),
            metadata_inventory=metadata,
        )

        for payload in (contract, design, readiness):
            self.assertFalse(payload["controlled_replay_allowed"])
            self.assertFalse(payload["metadata_replay_execution_allowed"])
            self.assertFalse(payload["performance_claim_allowed"])
            self.assertFalse(payload["promotion_evidence"])
            self.assertFalse(payload["online_llm_called"])
            self.assertFalse(payload["new_rollout_run"])
            self.assertFalse(payload["rollout_command_generated"])
            self.assertNotIn("commands", payload)


if __name__ == "__main__":
    unittest.main()
