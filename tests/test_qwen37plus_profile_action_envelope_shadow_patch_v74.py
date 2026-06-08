import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.qwen37plus_profile_action_envelope_shadow_patch_v74 import (
    BOUNDARY_FALSE_FIELDS,
    audit_trace,
    audit_traces,
    build_report,
)


ACTION = {"heat": 0.1, "co2": 0.0, "screen": 0.35, "vent": 0.45, "lamp": 0.0, "shade": 0.5}


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
        "name": "profile_action_envelope:hot_dry_protect",
        "profile_name": "hot_dry_protect",
        "candidate_source": "normal_path_profile_action_envelope",
        "intent": "hot_dry_protection",
        "target_direction": {"target_temp": "increase", "target_co2": "hold", "target_rh": "decrease"},
        "action_bounds": {
            "heat": {"min": 0.1, "max": 0.1},
            "co2": {"min": 0.0, "max": 0.0},
            "screen": {"min": 0.3, "max": 0.35},
            "vent": {"min": 0.4, "max": 0.45},
            "lamp": {"min": 0.0, "max": 0.0},
            "shade": {"min": 0.5, "max": 0.5},
        },
        "preferred_direction": {
            "heat": "hold",
            "co2": "hold",
            "screen": "increase",
            "vent": "increase",
            "lamp": "hold",
            "shade": "hold",
        },
        "priority_terms": {
            "profile_target_alignment": -0.2,
            "tomato_safety_compatibility": 0.0,
            "action_continuity": -0.1,
            "hard_safety_precedence": True,
        },
        "continuity_constraints": {
            "previous_action_source": "baseline_safety_action",
            "max_delta_from_previous_action": {
                "heat": 0.0,
                "co2": 0.0,
                "screen": 0.05,
                "vent": 0.05,
                "lamp": 0.0,
                "shade": 0.0,
            },
        },
        "tomato_safety_projection": {
            "projection_required": True,
            "projected_action": dict(ACTION),
            "projection_applied": True,
            "projection_reasons": ["canopy_dew_buffer"],
            "rewrite_delta_by_field": {
                "heat": 0.0,
                "co2": 0.0,
                "screen": 0.05,
                "vent": 0.05,
                "lamp": 0.0,
                "shade": 0.0,
            },
        },
        "projected_action": dict(ACTION),
        "score_terms": {
            "profile_target_alignment": -0.2,
            "tomato_projection_delta": 0.1,
            "continuity_penalty": 0.1,
            "envelope_width_penalty": 0.1,
            "compatibility_penalty": 0.0,
            "selection_score": 1.5,
        },
        "eligible": True,
        "rejection_reason": "",
        "compatibility_category": "compatibility_shadow_ready",
    }
    data.update(overrides)
    return data


def _row(step=0, **overrides):
    candidates = overrides.pop("candidates", [_candidate()])
    data = {
        "step": step,
        "profile_action_envelope_shadow_enabled": True,
        "profile_action_envelope_shadow_candidate_count": len(candidates),
        "profile_action_envelope_shadow_eligible_candidate_count": 1,
        "profile_action_envelope_shadow_best_name": "profile_action_envelope:hot_dry_protect",
        "profile_action_envelope_shadow_best_profile": "hot_dry_protect",
        "profile_action_envelope_shadow_best_score": 1.5,
        "profile_action_envelope_shadow_best_eligible": True,
        "profile_action_envelope_shadow_best_rejection_reason": "",
        "profile_action_envelope_shadow_final_action_changed": False,
        "profile_action_envelope_shadow_candidates_json": json.dumps(candidates, separators=(",", ":")),
    }
    data.update(overrides)
    return data


class TestQwen37PlusProfileActionEnvelopeShadowPatchV74(unittest.TestCase):
    def test_synthetic_envelope_candidates_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2010_d180_s43_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row(step) for step in range(3)])

            trace = audit_trace(path)
            audit = audit_traces([tmp])

        self.assertEqual(trace["normal_path_profile_action_envelope_steps"], 3)
        self.assertEqual(trace["contract_missing_steps"], 0)
        self.assertEqual(trace["tomato_safety_projection_missing_steps"], 0)
        self.assertEqual(audit["readiness"]["next_action"], "profile_action_envelope_compatibility_shadow_audit_plan")
        self.assertIn("v74 Profile Action Envelope Shadow Patch Audit", build_report(audit))

    def test_missing_tomato_safety_projection_enters_repair(self):
        broken = _candidate()
        broken.pop("tomato_safety_projection")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2018_d181_s42_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row(candidates=[broken])])

            audit = audit_traces([tmp])

        self.assertGreater(audit["aggregate"]["tomato_safety_projection_missing_steps"], 0)
        self.assertEqual(
            audit["readiness"]["next_action"],
            "profile_action_envelope_contract_or_projection_repair_plan",
        )

    def test_final_action_changed_stops_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2018_d181_s43_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row(profile_action_envelope_shadow_final_action_changed=True)])

            audit = audit_traces([tmp])

        self.assertEqual(audit["aggregate"]["action_diff_steps"], 1)
        self.assertEqual(
            audit["readiness"]["next_action"],
            "stop_profile_action_envelope_shadow_action_invariance_violation",
        )

    def test_fallback_candidate_source_enters_source_violation_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2018_d182_s43_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row(candidates=[_candidate(candidate_source="fallback_candidate")])])

            audit = audit_traces([tmp])

        self.assertEqual(audit["aggregate"]["fallback_candidate_source_steps"], 1)
        self.assertEqual(
            audit["readiness"]["next_action"],
            "profile_action_envelope_candidate_source_repair_plan",
        )

    def test_missing_v74_trace_requests_acquisition_without_command_generation(self):
        audit = audit_traces([])

        self.assertEqual(
            audit["readiness"]["next_action"],
            "needs_profile_action_envelope_opt_in_shadow_trace_acquisition",
        )
        self.assertFalse(audit["rollout_command_generated"])
        self.assertNotIn("commands", audit)

    def test_boundary_flags_remain_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2010_d180_s43_n720_llm_rspc_v2.csv"
            _write_csv(path, [_row()])

            audit = audit_traces([tmp])

        for field in BOUNDARY_FALSE_FIELDS:
            self.assertFalse(audit[field])
            self.assertFalse(audit["readiness"][field])
        self.assertFalse(audit["rollout_command_generated"])
        self.assertFalse(audit["readiness"]["rollout_command_generated"])


if __name__ == "__main__":
    unittest.main()
