import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.qwen37plus_profile_action_candidate_compatibility_shadow_audit_v70 import (
    BOUNDARY_FALSE_FIELDS,
    audit_trace,
    audit_traces,
    build_authorization_report,
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
        "raw_action": {"heat": 0.10, "co2": 0.00, "screen": 0.30, "vent": 0.40, "lamp": 0.00, "shade": 0.50},
        "post_tomato_action": {"heat": 0.10, "co2": 0.00, "screen": 0.34, "vent": 0.45, "lamp": 0.00, "shade": 0.50},
        "score_terms": {
            "raw_score": 2.0,
            "post_tomato_score": 1.5,
            "selection_score": 1.5,
            "profile_target_error": 0.2,
            "action_delta_penalty": 0.2,
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
        "profile_action_candidate_shadow_final_action_changed": False,
        "profile_action_candidate_shadow_candidates_json": json.dumps(candidates, separators=(",", ":")),
    }
    data.update(overrides)
    return data


class TestQwen37PlusProfileActionCandidateCompatibilityShadowAuditV70(unittest.TestCase):
    def test_compatible_profile_action_candidates_advance_to_scorer_shadow_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2010_d180_s43_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row(step) for step in range(3)])

            trace = audit_trace(path)
            audit = audit_traces([tmp])

        self.assertEqual(trace["compatibility_shadow_ready_steps"], 3)
        self.assertEqual(audit["aggregate"]["profile_action_candidate_available_steps"], 3)
        self.assertEqual(
            audit["hypothesis_check"]["next_action"],
            "minimal_profile_candidate_compatibility_scorer_shadow_plan",
        )
        self.assertIn("v70 Profile Action Candidate Compatibility", build_report(audit))

    def test_missing_v69_trace_generates_opt_in_authorization_plan(self):
        audit = audit_traces([])

        self.assertEqual(audit["trace_count"], 0)
        self.assertEqual(audit["hypothesis_check"]["next_action"], "needs_opt_in_shadow_trace_authorization")
        self.assertIn("authorization_plan", audit)
        self.assertTrue(audit["authorization_plan"]["shadow_only"])
        self.assertIn("Opt-In Shadow Trace Authorization", build_authorization_report(audit["authorization_plan"]))

    def test_trace_without_v69_metadata_requests_authorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2018_d181_s42_n720_llm_rspc_v2.csv"
            _write_csv(path, [{"step": 0, "reward": 0.0}])

            audit = audit_traces([tmp])

        self.assertEqual(audit["aggregate"]["v69_shadow_metadata_steps"], 0)
        self.assertEqual(audit["hypothesis_check"]["next_action"], "needs_opt_in_shadow_trace_authorization")
        self.assertIn("authorization_plan", audit)

    def test_large_tomato_rewrite_enters_action_envelope_design(self):
        rewritten = _candidate(
            post_tomato_action={"heat": 0.10, "co2": 0.00, "screen": 0.90, "vent": 0.95, "lamp": 0.00, "shade": 0.50}
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2018_d181_s43_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row(candidates=[rewritten])])

            audit = audit_traces([tmp])

        self.assertEqual(audit["aggregate"]["tomato_safety_incompatible_steps"], 1)
        self.assertEqual(
            audit["hypothesis_check"]["next_action"],
            "profile_to_action_mapping_repair_or_action_envelope_design",
        )

    def test_profile_target_incompatibility_requests_hypothesis_revision(self):
        score_terms = dict(_candidate()["score_terms"])
        score_terms["profile_target_error"] = 5.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2018_d181_s43_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row(candidates=[_candidate(score_terms=score_terms)])])

            audit = audit_traces([tmp])

        self.assertEqual(audit["aggregate"]["profile_target_incompatible_steps"], 1)
        self.assertEqual(
            audit["hypothesis_check"]["next_action"],
            "profile_generator_or_composer_hypothesis_revision_plan",
        )

    def test_action_diff_stops_shadow_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2018_d181_s42_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row(profile_action_candidate_shadow_final_action_changed=True)])

            audit = audit_traces([tmp])

        self.assertEqual(audit["aggregate"]["action_invariance_violation_steps"], 1)
        self.assertEqual(
            audit["hypothesis_check"]["next_action"],
            "stop_runtime_shadow_patch_action_invariance_violation",
        )

    def test_fallback_candidate_source_fails_source_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2010_d180_s43_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row(candidates=[_candidate(candidate_source="fallback_candidate")])])

            audit = audit_traces([tmp])

        self.assertEqual(audit["aggregate"]["fallback_candidate_source_violation_steps"], 1)
        self.assertEqual(
            audit["hypothesis_check"]["next_action"],
            "profile_action_candidate_source_repair_plan",
        )

    def test_boundary_flags_remain_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2010_d180_s43_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row()])

            audit = audit_traces([tmp])

        for field in BOUNDARY_FALSE_FIELDS:
            self.assertFalse(audit[field])
            self.assertFalse(audit["hypothesis_check"][field])


if __name__ == "__main__":
    unittest.main()
