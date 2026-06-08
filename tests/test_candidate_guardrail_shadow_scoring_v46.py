import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (
    FAILURE_SCENARIOS,
    build_comparison,
    build_patch_design,
    build_readiness,
    score_candidate,
    shadow_score_row,
)


FIELDS = [
    "step",
    "runtime_error",
    "runtime_error_type",
    "replan_reason",
    "temp_air",
    "rh_air",
    "vpd_air",
    "dry_risk",
    "dry_vent_cap",
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


def _candidate(name: str, score: float, selected: bool, *, vent: float, screen: float = 0.1, shade: float = 0.0, heat: float = 0.0):
    return {
        "name": name,
        "score": score,
        "selected": selected,
        "action": {
            "heat": heat,
            "co2": 0.0,
            "screen": screen,
            "vent": vent,
            "lamp": 0.0,
            "shade": shade,
        },
        "score_terms": {"total_score": score},
    }


def _row(*, candidates=True, selected_bad=True, runtime=False, step=0):
    cand = [
        _candidate("aggressive_vent", 1.0, selected_bad, vent=0.9, screen=0.5, shade=0.85),
        _candidate("anchor_hold", 1.2, not selected_bad, vent=0.2, screen=0.1, shade=0.0),
    ]
    row = {
        "step": step,
        "temp_air": 29.5,
        "rh_air": 48.0,
        "vpd_air": 1.8,
        "dry_risk": "true",
        "dry_vent_cap": 0.6,
        "target_temp": 24.0,
        "target_rh": 80.0,
        "target_co2": 900.0,
        "u_heating": 0.0,
        "u_co2": 0.0,
        "u_screen": 0.5,
        "u_ventilation": 0.9,
        "u_lighting": 0.0,
        "u_shading": 0.85,
        "tomato_safety_v2_applied": "true",
        "tomato_safety_v2_vent_before": 0.2,
        "tomato_safety_v2_vent_after": 0.9,
        "tomato_safety_v2_screen_before": 0.1,
        "tomato_safety_v2_screen_after": 0.5,
        "tomato_safety_v2_shade_before": 0.0,
        "tomato_safety_v2_shade_after": 0.85,
        "tomato_safety_v2_reasons": "hard_dry_vpd_risk",
        "rspc_action_candidates_json": json.dumps(cand) if candidates else "",
        "rspc_action_actual_candidate_count": 2,
        "rspc_action_candidate_count": 2,
        "profile_scorer_safety_gate_reason": "temp_high_gate",
        "rspc_action_post_shape_safety_gate_reason": "temp_high_gate",
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


class TestCandidateGuardrailShadowScoringV46(unittest.TestCase):
    def test_candidate_json_is_reranked_by_adjusted_score(self):
        row = _row()
        result = shadow_score_row(row)

        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["original_selected_name"], "aggressive_vent")
        self.assertEqual(result["shadow_selected_name"], "anchor_hold")
        self.assertTrue(result["shadow_changed"])

    def test_penalties_capture_conflicts_and_missing_candidate_metadata(self):
        row = _row()
        candidate = json.loads(row["rspc_action_candidates_json"])[0]
        scored = score_candidate(row, candidate)

        self.assertGreaterEqual(scored["profile_action_conflict_count"], 2)
        self.assertEqual(scored["dry_risk_high_vent_conflict"], 1)
        self.assertGreaterEqual(scored["actuator_inconsistency_count"], 1)
        self.assertGreaterEqual(scored["expected_rewrite_field_count"], 1)
        self.assertGreaterEqual(scored["candidate_filter_reason_count"], 1)
        self.assertEqual(shadow_score_row(_row(candidates=False))["status"], "candidate_metadata_missing")

    def test_comparison_tracks_pending_v43_and_readiness_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            v43 = root / "v43"
            original.mkdir()
            v43.mkdir()
            for scenario in FAILURE_SCENARIOS:
                rows = [_row(step=i) for i in range(10)]
                rows[-1] = _row(runtime=True, step=9)
                _write(original / f"{scenario}_llm_rspc_v2.csv", rows)
            _write(original / "y2010_d59_s42_n720_llm_rspc_v2.csv", [_row(selected_bad=False, step=i) for i in range(10)])
            _write(v43 / f"{FAILURE_SCENARIOS[0]}_llm_rspc_v2.csv", [_row(step=i) for i in range(5)])

            comparison = build_comparison(
                original_trace_dir=original,
                v43_trace_dir=v43,
                v45_audit_json=root / "missing_v45.json",
                failure_scenarios=FAILURE_SCENARIOS,
                window_steps=120,
            )
            readiness = build_readiness(comparison, build_patch_design(root / "missing_design.json"))

        self.assertEqual(set(comparison["pending_v43_trace_scenarios"]), {FAILURE_SCENARIOS[1], FAILURE_SCENARIOS[2]})
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
