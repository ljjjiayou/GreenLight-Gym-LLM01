import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.profile_feasibility_repair_v47 import (
    FAILURE_SCENARIOS,
    build_comparison,
    build_readiness,
    profile_feasibility_mode,
    repair_profile_targets,
    score_candidate_with_feasibility,
    shadow_score_row_with_repair,
)


FIELDS = [
    "step",
    "runtime_error",
    "runtime_error_type",
    "replan_reason",
    "temp_air",
    "rh_air",
    "vpd_air",
    "glob_rad",
    "dew_margin_air",
    "canopy_dew_margin",
    "temp_violation",
    "dry_risk",
    "target_temp",
    "target_rh",
    "target_co2",
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
    "tomato_safety_v2_applied",
    "tomato_safety_v2_vent_before",
    "tomato_safety_v2_vent_after",
    "tomato_safety_v2_screen_before",
    "tomato_safety_v2_screen_after",
    "tomato_safety_v2_shade_before",
    "tomato_safety_v2_shade_after",
    "tomato_safety_v2_reasons",
    "rspc_action_candidates_json",
    "rspc_action_actual_candidate_count",
    "rspc_action_candidate_count",
    "profile_scorer_safety_gate_reason",
    "rspc_action_post_shape_safety_gate_reason",
]


def _candidate(
    name: str,
    score: float,
    selected: bool,
    *,
    vent: float,
    screen: float = 0.0,
    shade: float = 0.0,
    heat: float = 0.0,
    co2: float = 0.0,
):
    return {
        "name": name,
        "score": score,
        "selected": selected,
        "action": {
            "heat": heat,
            "co2": co2,
            "screen": screen,
            "vent": vent,
            "lamp": 0.0,
            "shade": shade,
        },
        "score_terms": {"total_score": score},
    }


def _row(
    *,
    candidates=True,
    selected_bad=True,
    runtime=False,
    step=0,
    safety=False,
    dry=True,
    hard_reason=False,
):
    cand = [
        _candidate("aggressive_vent", 1.0, selected_bad, vent=0.9, screen=0.5 if hard_reason else 0.0, shade=0.0),
        _candidate("anchor_hold", 1.2, not selected_bad, vent=0.2, screen=0.0, shade=0.0),
    ]
    row = {
        "step": step,
        "temp_air": 29.5,
        "rh_air": 48.0 if dry else 91.0,
        "vpd_air": 1.8 if dry else 0.25,
        "glob_rad": 500.0,
        "dew_margin_air": 0.6 if safety else 3.0,
        "canopy_dew_margin": 0.8 if safety else 3.0,
        "temp_violation": 0.0,
        "dry_risk": "true" if dry else "false",
        "target_temp": 24.0,
        "target_rh": 80.0,
        "target_co2": 900.0,
        "u_heating": 0.0,
        "u_co2": 0.0,
        "u_screen": 0.0,
        "u_ventilation": 0.9 if selected_bad else 0.2,
        "u_lighting": 0.0,
        "u_shading": 0.0,
        "tomato_safety_v2_applied": "true" if safety else "false",
        "tomato_safety_v2_vent_before": 0.2,
        "tomato_safety_v2_vent_after": 0.9 if safety else 0.2,
        "tomato_safety_v2_screen_before": 0.0,
        "tomato_safety_v2_screen_after": 0.0,
        "tomato_safety_v2_shade_before": 0.0,
        "tomato_safety_v2_shade_after": 0.0,
        "tomato_safety_v2_reasons": "hard_canopy" if hard_reason or safety else "",
        "rspc_action_candidates_json": json.dumps(cand) if candidates else "",
        "rspc_action_actual_candidate_count": 2,
        "rspc_action_candidate_count": 2,
        "profile_scorer_safety_gate_reason": "canopy_gate" if safety else "none",
        "rspc_action_post_shape_safety_gate_reason": "canopy_gate" if safety else "none",
    }
    if runtime:
        row["runtime_error"] = "CV_CONV_FAILURE"
        row["runtime_error_type"] = "simulator_or_controller_error"
        row["replan_reason"] = "runtime_error"
    return row


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {field: "" for field in FIELDS}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


class TestProfileFeasibilityRepairV47(unittest.TestCase):
    def test_feasibility_mode_and_repair_do_not_mutate_original_row(self):
        row = _row(safety=True, dry=False)
        repaired, repair = repair_profile_targets(row)

        self.assertTrue(profile_feasibility_mode(row)["vent_required_by_safety"])
        self.assertFalse(repair["mode"]["humidity_retention_allowed"])
        self.assertLess(repaired["target_rh"], float(row["target_rh"]))
        self.assertEqual(float(row["target_rh"]), 80.0)
        self.assertIn("target_rh:safety_cap", repair["corrections"])

    def test_safety_required_vent_does_not_count_as_dry_high_vent_conflict(self):
        row = _row(safety=True, dry=True)
        candidate = json.loads(row["rspc_action_candidates_json"])[0]
        scored = score_candidate_with_feasibility(row, candidate)

        self.assertTrue(scored["profile_repair"]["mode"]["vent_required_by_safety"])
        self.assertEqual(scored["dry_risk_high_vent_conflict"], 0)
        self.assertNotIn("dry_risk_vs_high_vent", scored["profile_action_conflicts"])

    def test_non_safety_dry_high_vent_is_still_penalized(self):
        row = _row(safety=False, dry=True)
        candidate = json.loads(row["rspc_action_candidates_json"])[0]
        scored = score_candidate_with_feasibility(row, candidate)

        self.assertFalse(scored["profile_repair"]["mode"]["vent_required_by_safety"])
        self.assertEqual(scored["dry_risk_high_vent_conflict"], 1)
        self.assertIn("dry_risk_vs_high_vent", scored["profile_action_conflicts"])

    def test_hard_safety_veto_blocks_new_hard_preferred_candidate(self):
        row = _row(safety=False, dry=False, hard_reason=True)
        result = shadow_score_row_with_repair(row)

        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["original_selected_name"], "aggressive_vent")
        self.assertEqual(result["shadow_selected_name"], "anchor_hold")
        self.assertFalse(result["hard_safety_rewrite_preferred_new"])
        self.assertTrue(result["original_selected"]["hard_safety_vetoed"])

    def test_missing_candidate_metadata_is_not_fabricated(self):
        result = shadow_score_row_with_repair(_row(candidates=False))

        self.assertEqual(result["status"], "candidate_metadata_missing")
        self.assertEqual(result["candidate_count"], 0)

    def test_comparison_and_readiness_keep_replay_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            v43 = root / "v43"
            original.mkdir()
            v43.mkdir()
            for scenario in FAILURE_SCENARIOS:
                rows = [_row(step=i, safety=False, dry=True) for i in range(10)]
                rows[-1] = _row(runtime=True, step=9, safety=False, dry=True)
                _write(original / f"{scenario}_llm_rspc_v2.csv", rows)
            _write(original / "y2010_d59_s42_n720_llm_rspc_v2.csv", [_row(selected_bad=False, step=i, safety=False) for i in range(10)])
            _write(v43 / f"{FAILURE_SCENARIOS[0]}_llm_rspc_v2.csv", [_row(step=i, safety=False) for i in range(5)])

            comparison = build_comparison(
                original_trace_dir=original,
                v43_trace_dir=v43,
                v45_audit_json=root / "missing_v45.json",
                v46_comparison_json=root / "missing_v46.json",
                failure_scenarios=FAILURE_SCENARIOS,
                window_steps=120,
            )
            readiness = build_readiness(comparison)

        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
