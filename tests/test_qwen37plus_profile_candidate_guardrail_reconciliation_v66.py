import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments import qwen37plus_profile_candidate_guardrail_reconciliation_v66 as v66


FIELDS = [
    "step",
    "runtime_error",
    "runtime_error_type",
    "temp_air",
    "rh_air",
    "vpd_air",
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
    "tomato_safety_v2_heat_after",
    "tomato_safety_v2_vent_after",
    "tomato_safety_v2_screen_after",
    "tomato_safety_v2_shade_after",
    "tomato_safety_v2_reasons",
    "final_action_would_fail_canopy_boundary_v2",
    "profile_candidate_names",
    "rspc_action_candidates_json",
    "recovery_anchor_action",
]


def _candidate(name: str, action: dict[str, float], *, selected: bool = False, score: float = 1.0) -> dict[str, object]:
    return {
        "name": name,
        "selected": selected,
        "score": score,
        "action": action,
        "post_shape_action": action,
        "post_shape_score": score,
        "post_shape_score_terms": {
            "predicted_canopy_lt0_v2": False,
            "predicted_canopy_warning_v2": False,
            "predicted_dew_lt0": False,
        },
    }


def _row(step: int, *, runtime: str = "", candidates: list[dict[str, object]] | None = None) -> dict[str, object]:
    candidates = candidates or []
    return {
        "step": step,
        "runtime_error": runtime,
        "runtime_error_type": "simulator_or_controller_error" if runtime else "",
        "temp_air": 34.0,
        "rh_air": 45.0,
        "vpd_air": 3.0,
        "dry_risk": "true",
        "target_temp": 18.0,
        "target_rh": 72.0,
        "target_co2": 430.0,
        "u_heating": 0.0,
        "u_co2": 0.0,
        "u_screen": 0.5,
        "u_ventilation": 0.95,
        "u_lighting": 0.0,
        "u_shading": 0.85,
        "tomato_safety_v2_applied": "true",
        "tomato_safety_v2_heat_after": 0.0,
        "tomato_safety_v2_vent_after": 0.95,
        "tomato_safety_v2_screen_after": 0.5,
        "tomato_safety_v2_shade_after": 0.85,
        "tomato_safety_v2_reasons": "hot_temperature_override,hard_dry_vpd_risk",
        "profile_candidate_names": "constant_hold,hot_dry_protect",
        "rspc_action_candidates_json": json.dumps(candidates),
        "recovery_anchor_action": "{}",
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {field: "" for field in FIELDS}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


class TestQwen37PlusProfileCandidateGuardrailReconciliationV66(unittest.TestCase):
    def test_profile_candidates_are_shadow_only_not_executable_actions(self):
        row = _row(
            1,
            candidates=[
                _candidate("selected_bad", {"heat": 0.0, "co2": 0.0, "screen": 0.5, "vent": 0.95, "lamp": 0.0, "shade": 0.85}, selected=True)
            ],
        )

        result = v66.reconcile_row(row)

        self.assertEqual(result["profile_candidate_count"], 2)
        self.assertFalse(result["profile_candidates_shadow_only"][0]["executable_low_level_action"])
        self.assertGreaterEqual(result["executable_candidate_count"], 2)

    def test_reconciler_prefers_compatible_candidate_over_conflicting_final_action(self):
        row = _row(
            1,
            candidates=[
                _candidate("selected_bad", {"heat": 0.0, "co2": 0.0, "screen": 0.5, "vent": 0.95, "lamp": 0.0, "shade": 0.85}, selected=True, score=1.0),
                _candidate("compatible_relief", {"heat": 0.0, "co2": 0.0, "screen": 0.05, "vent": 0.55, "lamp": 0.0, "shade": 0.1}, score=2.0),
            ],
        )

        result = v66.reconcile_row(row)

        self.assertEqual(result["shadow_selected"]["name"], "compatible_relief")
        self.assertTrue(result["shadow_changed"])
        self.assertGreater(result["compatible_candidate_count"], 0)
        self.assertTrue(result["original"]["candidate_guardrail_issue"] if "candidate_guardrail_issue" in result["original"] else True)

    def test_audit_and_readiness_never_authorize_execution_or_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            good = _candidate(
                "compatible_relief",
                {"heat": 0.0, "co2": 0.0, "screen": 0.05, "vent": 0.55, "lamp": 0.0, "shade": 0.1},
                score=2.0,
            )
            bad = _candidate(
                "selected_bad",
                {"heat": 0.0, "co2": 0.0, "screen": 0.5, "vent": 0.95, "lamp": 0.0, "shade": 0.85},
                selected=True,
                score=1.0,
            )
            scenario = v66.FAILURE_SCENARIOS[0]
            rows = [_row(step, candidates=[bad, good]) for step in range(5)]
            rows[-1]["runtime_error"] = "Error in Function::call for 'F' [CvodesInterface]"
            rows[-1]["runtime_error_type"] = "simulator_or_controller_error"
            _write(trace_dir / f"{scenario}_llm_rspc_v2.csv", rows)

            audit = v66.build_reconciliation_audit(
                trace_dir=trace_dir,
                failure_scenarios=[scenario],
                window_steps=120,
                v65_attribution_json=trace_dir / "missing.json",
            )
            selection = v66.build_shadow_selection(audit)
            readiness = v66.build_readiness(audit, selection)

        self.assertGreater(audit["compatible_candidate_found_steps"], 0)
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])
        self.assertFalse(selection["shadow_selection_changes_final_control"])


if __name__ == "__main__":
    unittest.main()
