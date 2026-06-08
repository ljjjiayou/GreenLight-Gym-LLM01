import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments import qwen37plus_normal_path_profile_candidate_arbitration_v67 as v67


FIELDS = list(
    dict.fromkeys(
        list(v67.REQUIRED_METADATA_FIELDS)
        + [
            "runtime_error",
            "source",
            "profile_rspc_shadow_top_candidates",
            "profile_scorer_ranked_candidates",
            "structured_anchor_profile_bridge_requested_shapes",
            "dry_risk",
            "tomato_safety_v2_reasons",
            "final_action_risk_reason_v2",
            "rspc_action_post_shape_best_name",
        ]
    )
)


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


def _row(
    step: int,
    *,
    candidates: list[dict[str, object]] | None = None,
    runtime: bool = False,
    candidate_selection_source: str = "fallback_candidate",
    selected_fallback_candidate: str = "recent_anchor",
    profile_low_level_action_generated: bool = False,
) -> dict[str, object]:
    candidates = candidates or []
    selected = next((item for item in candidates if item.get("selected")), candidates[0] if candidates else None)
    selected_name = str(selected.get("name")) if isinstance(selected, dict) else "anchor"
    return {
        "step": step,
        "runtime_error": "Error in Function::call for 'F' [CvodesInterface]" if runtime else "",
        "runtime_error_type": "simulator_or_controller_error" if runtime else "",
        "source": "runtime_error" if runtime else "anchor",
        "structured_anchor_valid": "True",
        "structured_anchor_profile_bridge_attempted": "True",
        "structured_anchor_profile_bridge_bridgeable": "True",
        "structured_anchor_profile_bridge_profile_candidate_count": "2",
        "structured_anchor_profile_bridge_selected_shadow_profile": "constant_hold",
        "structured_anchor_profile_bridge_low_level_action_generated": str(profile_low_level_action_generated),
        "structured_anchor_profile_bridge_requested_shapes": "constant_hold,shade_cooling",
        "profile_generator_candidate_count": "2",
        "profile_candidate_names": "constant_hold,shade_cooling",
        "profile_rspc_shadow_candidate_count": "2",
        "profile_rspc_shadow_eligible_candidate_count": "2",
        "profile_rspc_shadow_best_eligible": "True",
        "profile_rspc_shadow_best_profile": "constant_hold",
        "profile_rspc_shadow_top_candidates": "constant_hold,shade_cooling",
        "profile_scorer_ranked_candidates": "constant_hold,shade_cooling",
        "candidate_selection_source": candidate_selection_source,
        "selected_fallback_candidate": selected_fallback_candidate,
        "rspc_action_candidates_json": json.dumps(candidates),
        "rspc_action_candidate_count": str(len(candidates)),
        "rspc_action_actual_candidate_count": str(len(candidates)),
        "rspc_action_selected_name": selected_name,
        "rspc_action_post_shape_best_name": selected_name,
        "tomato_safety_v2_applied": "",
        "final_action_would_fail_canopy_boundary_v2": "False",
        "target_temp": "22.0",
        "target_rh": "60.0",
        "target_co2": "420.0",
        "temp_air": "22.0",
        "rh_air": "60.0",
        "vpd_air": "1.2",
        "dry_risk": "False",
        "tomato_safety_v2_reasons": "",
        "final_action_risk_reason_v2": "",
        "u_heating": "0.0",
        "u_co2": "0.0",
        "u_screen": "0.0",
        "u_ventilation": "0.1",
        "u_lighting": "0.0",
        "u_shading": "0.0",
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {field: "" for field in FIELDS}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


class TestQwen37PlusNormalPathProfileCandidateArbitrationV67(unittest.TestCase):
    def test_profile_present_but_not_composed_to_action_routes_to_composer_design(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            scenario = v67.FAILURE_SCENARIOS[0]
            rows = [_row(0), _row(1, runtime=True)]
            _write(trace_dir / f"{scenario}_llm_rspc_v2.csv", rows)

            audit = v67.build_normal_path_arbitration_audit(
                trace_dir=trace_dir,
                failure_scenarios=[scenario],
                window_steps=120,
                v65_attribution_json=trace_dir / "missing_v65.json",
                v66_reconciliation_json=trace_dir / "missing_v66.json",
            )
            gap = v67.build_profile_to_action_gap_audit(audit)
            readiness = v67.build_readiness(audit, gap)

        self.assertEqual(audit["aggregate"]["profile_candidate_available_steps"], 2)
        self.assertEqual(audit["aggregate"]["normal_path_profile_action_candidate_steps"], 0)
        self.assertEqual(audit["next_action"], "profile_to_action_candidate_composer_design_plan")
        self.assertTrue(gap["profile_to_action_gap_confirmed"])
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])

    def test_profile_derived_compatible_action_candidate_routes_to_shadow_patch_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            scenario = v67.FAILURE_SCENARIOS[0]
            profile_candidate = _candidate(
                "constant_hold",
                {"heat": 0.0, "co2": 0.0, "screen": 0.0, "vent": 0.1, "lamp": 0.0, "shade": 0.0},
                selected=True,
            )
            _write(
                trace_dir / f"{scenario}_llm_rspc_v2.csv",
                [
                    _row(
                        0,
                        candidates=[profile_candidate],
                        runtime=True,
                        candidate_selection_source="normal_path_profile_candidate",
                        selected_fallback_candidate="",
                        profile_low_level_action_generated=True,
                    )
                ],
            )

            audit = v67.build_normal_path_arbitration_audit(
                trace_dir=trace_dir,
                failure_scenarios=[scenario],
                window_steps=120,
                v65_attribution_json=trace_dir / "missing_v65.json",
                v66_reconciliation_json=trace_dir / "missing_v66.json",
            )

        self.assertEqual(audit["aggregate"]["normal_path_profile_action_candidate_steps"], 1)
        self.assertEqual(audit["aggregate"]["normal_path_profile_compatible_candidate_steps"], 1)
        self.assertEqual(audit["next_action"], "minimal_normal_path_profile_arbitration_shadow_patch_plan")

    def test_missing_metadata_is_explicit_taxonomy(self):
        result = v67.classify_step({"step": "7"})

        self.assertIn("metadata_missing", result["categories"])
        self.assertEqual(result["primary_category"], "metadata_missing")


if __name__ == "__main__":
    unittest.main()
