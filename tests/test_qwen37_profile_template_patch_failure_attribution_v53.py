import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.qwen37_profile_template_patch_failure_attribution_v53 import (
    FAILURE_SCENARIOS,
    _classify_leak,
    build_hard_safety_leak_audit,
    build_readiness,
)


FIELDS = [
    "step",
    "source",
    "anchor_source",
    "candidate_selection_source",
    "selected_fallback_candidate",
    "model_name",
    "target_temp",
    "target_rh",
    "target_co2",
    "temp_air",
    "dry_vent_risk",
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
    "tomato_safety_v2_applied",
    "tomato_safety_v2_reasons",
    "tomato_safety_v2_heat_before",
    "tomato_safety_v2_heat_after",
    "tomato_safety_v2_vent_before",
    "tomato_safety_v2_vent_after",
    "tomato_safety_v2_screen_before",
    "tomato_safety_v2_screen_after",
    "tomato_safety_v2_shade_before",
    "tomato_safety_v2_shade_after",
    "rspc_action_actual_candidate_count",
    "rspc_action_candidate_count",
    "rspc_action_candidates_json",
    "profile_template_patch_enabled",
    "profile_template_patch_applied",
    "profile_template_patch_corrections",
    "profile_template_patch_vent_required_by_safety",
    "profile_template_patch_fallback_veto_applied",
    "profile_template_patch_fallback_veto_no_alternative",
    "profile_template_patch_fallback_veto_reason",
]


def _candidate_json() -> str:
    return json.dumps(
        [
            {
                "name": "anchor_hold",
                "selected": True,
                "score": 0.0,
                "action": {"heat": 0.0, "vent": 0.2, "screen": 0.0, "shade": 0.0, "co2": 430.0},
            }
        ]
    )


def _row(
    *,
    step: int = 0,
    hard: bool = False,
    source: str = "anchor",
    anchor_source: str = "hold_current",
    selected_fallback_candidate: str = "hold_current",
    candidates: bool = True,
    veto_applied: bool = False,
    veto_no_alternative: bool = False,
    vent_required: bool = False,
) -> dict[str, object]:
    return {
        "step": step,
        "source": source,
        "anchor_source": anchor_source,
        "candidate_selection_source": "fallback_candidate" if selected_fallback_candidate else "rollout_candidate",
        "selected_fallback_candidate": selected_fallback_candidate,
        "model_name": "qwen3.7-max",
        "target_temp": 18.5,
        "target_rh": 70.0,
        "target_co2": 430.0,
        "temp_air": 28.0,
        "dry_vent_risk": "true",
        "u_heating": 0.0,
        "u_co2": 0.0,
        "u_screen": 0.0,
        "u_ventilation": 0.8 if hard else 0.2,
        "u_lighting": 0.0,
        "u_shading": 0.8 if hard else 0.0,
        "tomato_safety_v2_applied": "true" if hard else "false",
        "tomato_safety_v2_reasons": "dry_vpd_guard,hard_dry_vpd_guard" if hard else "",
        "tomato_safety_v2_heat_before": 0.0,
        "tomato_safety_v2_heat_after": 0.0,
        "tomato_safety_v2_vent_before": 0.2,
        "tomato_safety_v2_vent_after": 0.8 if hard else 0.2,
        "tomato_safety_v2_screen_before": 0.0,
        "tomato_safety_v2_screen_after": 0.0,
        "tomato_safety_v2_shade_before": 0.0,
        "tomato_safety_v2_shade_after": 0.8 if hard else 0.0,
        "rspc_action_actual_candidate_count": 1 if candidates else 0,
        "rspc_action_candidate_count": 1 if candidates else 0,
        "rspc_action_candidates_json": _candidate_json() if candidates else "",
        "profile_template_patch_enabled": "true",
        "profile_template_patch_applied": "false",
        "profile_template_patch_corrections": "",
        "profile_template_patch_vent_required_by_safety": "true" if vent_required else "false",
        "profile_template_patch_fallback_veto_applied": "true" if veto_applied else "false",
        "profile_template_patch_fallback_veto_no_alternative": "true" if veto_no_alternative else "false",
        "profile_template_patch_fallback_veto_reason": "",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {field: "" for field in FIELDS}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


class TestQwen37FailureAttributionV53(unittest.TestCase):
    def test_exact_leak_alignment_is_step_based_not_aggregate_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            v52 = root / "v52"
            original.mkdir()
            v52.mkdir()
            scenario = "y2010_d180_s43_n720"
            _write_csv(
                original / f"{scenario}_llm_rspc_v2.csv",
                [_row(step=0, hard=False), _row(step=1, hard=True), _row(step=2, hard=True)],
            )
            _write_csv(
                v52 / f"{scenario}_llm_rspc_v2.csv",
                [_row(step=0, hard=True), _row(step=1, hard=True), _row(step=2, hard=False)],
            )
            result = root / "result.json"
            result.write_text(
                json.dumps(
                    {
                        "totals": {"hard_safety_rewrite_preferred_new_steps": 0},
                        "scenario_reports": [
                            {
                                "scenario_id": scenario,
                                "original": {"hard_safety_rewrite_steps": 2},
                                "v52": {"hard_safety_rewrite_steps": 2},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = build_hard_safety_leak_audit(
                original_trace_dir=original,
                v52_trace_dir=v52,
                v52_result_json=result,
                scenarios=[scenario],
            )

        self.assertEqual(audit["reported_aggregate_net_new_hard_safety_count_from_v52"], 0)
        self.assertEqual(audit["step_aligned_new_hard_safety_leak_count"], 1)
        self.assertEqual(audit["step_aligned_resolved_original_hard_safety_count"], 1)
        self.assertEqual(audit["leak_entries"][0]["step"], 0)

    def test_classification_covers_anchor_and_missing_metadata(self):
        original = _row(hard=False)
        anchor_leak = _row(hard=True, source="anchor", anchor_source="hold_current", selected_fallback_candidate="hold_current")
        missing = _row(hard=True, candidates=False)
        non_fallback = _row(hard=True, source="rollout", anchor_source="", selected_fallback_candidate="")
        no_alt = _row(hard=True, veto_no_alternative=True)

        self.assertEqual(_classify_leak(original, anchor_leak)[0], "planner_anchor_empty_contract_filled")
        self.assertEqual(_classify_leak(original, missing)[0], "candidate_metadata_missing")
        self.assertEqual(_classify_leak(original, non_fallback)[0], "fallback_veto_not_invoked_on_non_fallback_candidate")
        self.assertEqual(_classify_leak(original, no_alt)[0], "veto_no_compatible_alternative")

    def test_default_scope_and_readiness_never_allow_replay_or_claims(self):
        self.assertEqual(
            set(FAILURE_SCENARIOS),
            {"y2010_d180_s43_n720", "y2018_d181_s42_n720", "y2018_d181_s43_n720"},
        )
        readiness = build_readiness(
            {
                "dominant_root_cause": "planner_anchor_empty_contract_filled",
                "next_action": "qwen37_planning_anchor_contract_fix",
                "v52_qwen37_model_consistency_pass": True,
                "v52_acceptance_pass": False,
            },
            {
                "reported_aggregate_net_new_hard_safety_count_from_v52": 27,
                "step_aligned_new_hard_safety_leak_count": 43,
                "candidate_metadata_missing_count": 0,
                "classification_counts": {"unknown_requires_instrumentation": 0},
            },
            {"design_ready": True},
        )

        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])
        self.assertEqual(readiness["next_action"], "qwen37_planning_anchor_contract_fix")


if __name__ == "__main__":
    unittest.main()
