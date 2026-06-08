import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.diagnose_ppo_vs_llm import derive_candidate_guardrail_compatibility
from gl_gym.experiments.profile_candidate_guardrail_compatibility_v45 import (
    FAILURE_SCENARIOS,
    build_compatibility_audit,
    build_readiness,
    build_scoring_patch_design,
)


FIELDS = [
    "step",
    "runtime_error",
    "runtime_error_type",
    "replan_reason",
    "temp_air",
    "target_temp",
    "target_rh",
    "target_co2",
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
    "dry_vent_risk",
    "tomato_safety_v2_applied",
    "tomato_safety_v2_before",
    "tomato_safety_v2_after",
    "tomato_safety_v2_heat_before",
    "tomato_safety_v2_heat_after",
    "tomato_safety_v2_vent_before",
    "tomato_safety_v2_vent_after",
    "tomato_safety_v2_screen_before",
    "tomato_safety_v2_screen_after",
    "tomato_safety_v2_shade_before",
    "tomato_safety_v2_shade_after",
    "tomato_safety_v2_reasons",
    "selected_fallback_candidate",
    "rspc_action_actual_candidate_count",
    "rspc_action_candidate_count",
    "profile_scorer_safety_gate_reason",
    "rspc_action_post_shape_safety_gate_reason",
    "source",
    "buffered_control",
]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {field: "" for field in FIELDS}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


def _rows(count: int, *, runtime_step: int | None = None, conflict: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(count):
        row = {
            "step": step,
            "temp_air": 30.0 if conflict else 22.0,
            "target_temp": 24.0 if conflict else 22.0,
            "target_rh": 80.0 if conflict else 65.0,
            "target_co2": 900.0 if conflict else 400.0,
            "u_heating": 0.0,
            "u_co2": 0.0,
            "u_screen": 0.20,
            "u_ventilation": 0.85 if conflict else 0.20,
            "u_lighting": 0.0,
            "u_shading": 0.0,
            "tomato_safety_v2_applied": "true" if conflict else "false",
            "tomato_safety_v2_heat_before": 0.0,
            "tomato_safety_v2_heat_after": 0.0,
            "tomato_safety_v2_vent_before": 0.20,
            "tomato_safety_v2_vent_after": 0.85 if conflict else 0.20,
            "tomato_safety_v2_screen_before": 0.10,
            "tomato_safety_v2_screen_after": 0.45 if conflict else 0.10,
            "tomato_safety_v2_shade_before": 0.0,
            "tomato_safety_v2_shade_after": 0.0,
            "tomato_safety_v2_reasons": "hard_dry_vpd_risk" if conflict else "",
            "rspc_action_actual_candidate_count": 3,
            "rspc_action_candidate_count": 3,
            "profile_scorer_safety_gate_reason": "dry_vpd_guard" if conflict else "",
            "rspc_action_post_shape_safety_gate_reason": "hot_dry_cooling_guard" if conflict else "",
        }
        if runtime_step is not None and step == runtime_step:
            row["runtime_error"] = "CV_CONV_FAILURE"
            row["runtime_error_type"] = "simulator_or_controller_error"
            row["replan_reason"] = "runtime_error"
        rows.append(row)
    return rows


class TestProfileCandidateGuardrailCompatibilityV45(unittest.TestCase):
    def test_derive_fields_identifies_rewrite_conflict_and_source(self):
        row = _rows(1, conflict=True)[0]
        derived = derive_candidate_guardrail_compatibility(dict(row))

        self.assertEqual(derived["candidate_guardrail_compatibility_label"], "hard_safety_rewrite")
        self.assertIn("ventilation", derived["candidate_guardrail_rewrite_fields"])
        self.assertIn("target_rh_vs_high_vent", derived["profile_action_conflict_label"])
        self.assertIn("co2_target_vs_vent", derived["profile_action_conflict_label"])
        self.assertEqual(derived["candidate_selection_source"], "rollout_candidate")
        self.assertIn("dry_vpd_guard", derived["candidate_filter_reason_summary"])

    def test_v45_readiness_never_authorizes_controlled_or_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            v43 = root / "v43"
            original.mkdir()
            v43.mkdir()
            for scenario in FAILURE_SCENARIOS:
                _write(original / f"{scenario}_llm_rspc_v2.csv", _rows(12, runtime_step=11, conflict=True))
            _write(v43 / f"{FAILURE_SCENARIOS[0]}_llm_rspc_v2.csv", _rows(8, runtime_step=7, conflict=True))

            audit = build_compatibility_audit(
                original_trace_dir=original,
                v43_trace_dir=v43,
                v44_causal_json=root / "missing_v44.json",
                failure_scenarios=FAILURE_SCENARIOS,
                window_steps=120,
            )
            design = build_scoring_patch_design(audit)
            readiness = build_readiness(audit, design)

        self.assertTrue(audit["compatibility_issue_detected"])
        self.assertEqual(
            set(audit["pending_trace_scenarios"]),
            {FAILURE_SCENARIOS[1], FAILURE_SCENARIOS[2]},
        )
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])
        self.assertFalse(design["patch_execution_authorized"])


if __name__ == "__main__":
    unittest.main()
