import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.qwen37plus_profile_action_candidate_shadow_patch_v69 import (
    BOUNDARY_FALSE_FIELDS,
    audit_trace,
    audit_traces,
    build_report,
)


def _write_csv(path, rows):
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _candidate(**overrides):
    data = {
        "name": "profile_action:hot_dry_protect",
        "profile_name": "hot_dry_protect",
        "candidate_source": "normal_path_profile_candidate",
        "raw_action": {"heat": 0.1, "co2": 0.0, "screen": 0.3, "vent": 0.4, "lamp": 0.0, "shade": 0.5},
        "post_tomato_action": {"heat": 0.1, "co2": 0.0, "screen": 0.35, "vent": 0.45, "lamp": 0.0, "shade": 0.5},
        "score_terms": {
            "raw_score": 2.0,
            "post_tomato_score": 1.5,
            "selection_score": 1.5,
            "profile_target_error": 0.2,
            "action_delta_penalty": 0.1,
            "tomato_safety_penalty": 1.0,
            "compatibility_penalty": 0.0,
            "hard_safety_rewrite_predicted": True,
        },
        "tomato_safety_v2_applied": True,
        "eligible": True,
        "rejection_reason": "",
    }
    data.update(overrides)
    return data


def _row(step=0, **overrides):
    candidates = overrides.pop("candidates", [_candidate()])
    data = {
        "step": step,
        "profile_action_candidate_shadow_enabled": True,
        "profile_action_candidate_shadow_candidate_count": len(candidates),
        "profile_action_candidate_shadow_eligible_candidate_count": 1,
        "profile_action_candidate_shadow_best_name": "profile_action:hot_dry_protect",
        "profile_action_candidate_shadow_best_profile": "hot_dry_protect",
        "profile_action_candidate_shadow_best_score": 1.5,
        "profile_action_candidate_shadow_best_eligible": True,
        "profile_action_candidate_shadow_best_rejection_reason": "",
        "profile_action_candidate_shadow_final_action_changed": False,
        "profile_action_candidate_shadow_candidates_json": json.dumps(candidates, separators=(",", ":")),
    }
    data.update(overrides)
    return data


class TestQwen37PlusProfileActionCandidateShadowPatchV69(unittest.TestCase):
    def test_synthetic_contract_candidates_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2010_d180_s43_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row(step) for step in range(3)])

            trace = audit_trace(path)
            audit = audit_traces([tmp])

        self.assertEqual(trace["normal_path_profile_action_candidate_steps"], 3)
        self.assertEqual(trace["contract_missing_steps"], 0)
        self.assertEqual(trace["tomato_safety_provenance_missing_steps"], 0)
        self.assertEqual(audit["readiness"]["next_action"], "normal_path_profile_candidate_compatibility_shadow_audit_plan")
        self.assertIn("v69 Profile Action Candidate Shadow Patch Audit", build_report(audit))

    def test_action_diff_fails_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2018_d181_s42_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row(profile_action_candidate_shadow_final_action_changed=True)])

            audit = audit_traces([tmp])

        self.assertEqual(audit["aggregate"]["action_diff_steps"], 1)
        self.assertEqual(
            audit["readiness"]["next_action"],
            "stop_runtime_shadow_patch_action_invariance_violation",
        )

    def test_missing_score_fields_enters_instrumentation_plan(self):
        broken_score = dict(_candidate()["score_terms"])
        broken_score.pop("selection_score")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2018_d181_s43_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row(candidates=[_candidate(score_terms=broken_score)])])

            audit = audit_traces([tmp])

        self.assertGreater(audit["aggregate"]["contract_missing_steps"], 0)
        self.assertEqual(
            audit["readiness"]["next_action"],
            "profile_action_candidate_shadow_instrumentation_repair_plan",
        )

    def test_boundary_flags_remain_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2010_d180_s43_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row()])

            audit = audit_traces([tmp])

        for field in BOUNDARY_FALSE_FIELDS:
            self.assertFalse(audit[field])
            self.assertFalse(audit["readiness"][field])


if __name__ == "__main__":
    unittest.main()
