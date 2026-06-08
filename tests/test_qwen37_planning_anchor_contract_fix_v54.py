import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.qwen37_planning_anchor_contract_fix_v54 import (
    FAILURE_SCENARIOS,
    build_anchor_contract_audit,
    build_contract_design,
    build_readiness,
    classify_anchor_contract,
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
    "profile_template_patch_enabled",
    "profile_template_patch_applied",
    "profile_template_patch_mode",
    "profile_template_patch_corrections",
    "profile_template_patch_forbidden_combinations",
    "profile_template_patch_vent_required_by_safety",
    "profile_template_patch_humidity_retention_allowed",
    "profile_template_patch_co2_enrichment_allowed",
    "profile_template_patch_repaired_target_temp",
    "profile_template_patch_repaired_target_rh",
    "profile_template_patch_repaired_target_co2",
    "profile_template_patch_fallback_veto_applied",
    "profile_template_patch_fallback_veto_no_alternative",
    "temp_air",
    "rh_air",
    "co2_air",
    "glob_rad",
    "hour_of_day",
    "dew_margin_air",
    "canopy_dew_margin",
    "forecast_rad_mean_1h",
    "forecast_rad_peak_2h",
    "forecast_temp_out_delta_1h",
    "tomato_safety_v2_before",
    "rspc_action_candidates_json",
]


def _candidate_json(*, include_safe: bool = True) -> str:
    candidates = [
        {
            "name": "rule",
            "score": -1.0,
            "selected": True,
            "action": {"heat": 1.0, "co2": 0.0, "screen": 0.0, "vent": 0.0, "lamp": 0.0, "shade": 0.0},
        }
    ]
    if include_safe:
        candidates.append(
            {
                "name": "safe_anchor",
                "score": 0.0,
                "selected": False,
                "action": {"heat": 0.0, "co2": 0.0, "screen": 0.6, "vent": 0.6, "lamp": 0.0, "shade": 0.5},
            }
        )
    return json.dumps(candidates)


def _row(*, step: int = 5, include_safe: bool = True) -> dict[str, object]:
    return {
        "step": step,
        "source": "rule",
        "anchor_source": "recent_anchor",
        "candidate_selection_source": "fallback_candidate",
        "selected_fallback_candidate": "recent_anchor",
        "model_name": "qwen3.7-max",
        "target_temp": 18.5,
        "target_rh": 72.0,
        "target_co2": 430.0,
        "profile_template_patch_enabled": "true",
        "profile_template_patch_applied": "true",
        "profile_template_patch_mode": "safety_vent",
        "profile_template_patch_corrections": "target_rh:safety_cap",
        "profile_template_patch_forbidden_combinations": "rh_dew_canopy_pressure_blocks_humidity_retention",
        "profile_template_patch_vent_required_by_safety": "true",
        "profile_template_patch_humidity_retention_allowed": "false",
        "profile_template_patch_co2_enrichment_allowed": "false",
        "profile_template_patch_repaired_target_temp": 18.5,
        "profile_template_patch_repaired_target_rh": 72.0,
        "profile_template_patch_repaired_target_co2": 430.0,
        "profile_template_patch_fallback_veto_applied": "false",
        "profile_template_patch_fallback_veto_no_alternative": "false",
        "temp_air": 26.7,
        "rh_air": 63.5,
        "co2_air": 369.0,
        "glob_rad": 405.9,
        "hour_of_day": 7.25,
        "dew_margin_air": 7.5,
        "canopy_dew_margin": 1.9,
        "forecast_rad_mean_1h": 501.0,
        "forecast_rad_peak_2h": 618.0,
        "forecast_temp_out_delta_1h": 0.5,
        "tomato_safety_v2_before": json.dumps([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "rspc_action_candidates_json": _candidate_json(include_safe=include_safe),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {field: "" for field in FIELDS}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


def _write_inputs(root: Path, *, include_safe: bool) -> tuple[Path, Path, Path, str]:
    trace_dir = root / "traces"
    trace_dir.mkdir()
    scenario = "y2010_d180_s43_n720"
    _write_csv(trace_dir / f"{scenario}_llm_rspc_v2.csv", [_row(include_safe=include_safe)])
    leak_path = root / "v53_leak.json"
    leak_path.write_text(
        json.dumps(
            {
                "step_aligned_new_hard_safety_leak_count": 1,
                "leak_entries": [
                    {
                        "scenario_id": scenario,
                        "step": 5,
                        "classification": "planner_anchor_empty_contract_filled",
                        "source": "rule",
                        "anchor_source": "recent_anchor",
                        "candidate_selection_source": "fallback_candidate",
                        "selected_fallback_candidate": "recent_anchor",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    attribution_path = root / "v53_attr.json"
    attribution_path.write_text(
        json.dumps({"dominant_root_cause": "planner_anchor_empty_contract_filled"}),
        encoding="utf-8",
    )
    return trace_dir, leak_path, attribution_path, scenario


class TestQwen37PlanningAnchorContractFixV54(unittest.TestCase):
    def test_contract_audit_finds_compatible_alternative_for_anchor_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir, leak_path, attribution_path, scenario = _write_inputs(root, include_safe=True)
            audit = build_anchor_contract_audit(
                leak_audit_json=leak_path,
                attribution_json=attribution_path,
                v52_trace_dir=trace_dir,
                scenarios=[scenario],
            )

        self.assertEqual(audit["evaluated_leak_count"], 1)
        self.assertTrue(audit["all_v53_leaks_evaluated"])
        self.assertEqual(audit["compatible_alternative_count"], 1)
        self.assertEqual(audit["no_compatible_anchor_available_count"], 0)
        self.assertEqual(audit["contract_rows"][0]["best_compatible_alternative"]["name"], "safe_anchor")
        self.assertEqual(audit["contract_rows"][0]["classification"], "recent_or_hold_anchor_contract_unsafe")

    def test_no_compatible_alternative_is_explicit_recovery_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir, leak_path, attribution_path, scenario = _write_inputs(root, include_safe=False)
            audit = build_anchor_contract_audit(
                leak_audit_json=leak_path,
                attribution_json=attribution_path,
                v52_trace_dir=trace_dir,
                scenarios=[scenario],
            )
            design = build_contract_design(audit)

        self.assertEqual(audit["compatible_alternative_count"], 0)
        self.assertEqual(audit["no_compatible_anchor_available_count"], 1)
        self.assertEqual(audit["contract_rows"][0]["classification"], "fallback_scored_no_compatible_alternative")
        self.assertEqual(design["recommended_next_action"], "recovery_anchor_candidate_design_plan")

    def test_classifier_preserves_missing_and_unknown_paths(self):
        leak = {"anchor_source": "recent_anchor", "classification": "planner_anchor_empty_contract_filled"}
        self.assertEqual(
            classify_anchor_contract(
                leak,
                row_found=False,
                candidate_count=0,
                selected_hard_safety=True,
                compatible_alternative_count=0,
            ),
            "candidate_metadata_or_control_missing",
        )
        self.assertEqual(
            classify_anchor_contract(
                leak,
                row_found=True,
                candidate_count=1,
                selected_hard_safety=False,
                compatible_alternative_count=0,
            ),
            "unknown_requires_instrumentation",
        )

    def test_default_scope_and_readiness_never_allow_replay_or_claims(self):
        self.assertEqual(
            set(FAILURE_SCENARIOS),
            {"y2010_d180_s43_n720", "y2018_d181_s42_n720", "y2018_d181_s43_n720"},
        )
        audit = {
            "all_v53_leaks_evaluated": True,
            "candidate_metadata_or_control_missing_count": 0,
            "unknown_requires_instrumentation_count": 0,
            "evaluated_leak_count": 43,
            "expected_v53_step_aligned_leak_count": 43,
            "compatible_alternative_count": 10,
            "no_compatible_anchor_available_count": 33,
            "v53_dominant_root_cause": "planner_anchor_empty_contract_filled",
        }
        design = {"design_ready": True, "recommended_next_action": "recovery_anchor_candidate_design_plan"}
        readiness = build_readiness(audit, design)

        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])
        self.assertEqual(readiness["next_action"], "recovery_anchor_candidate_design_plan")


if __name__ == "__main__":
    unittest.main()
