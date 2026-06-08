import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.profile_candidate_runtime_consistency_v49 import (
    build_online_llm_accessibility_design,
    build_profile_template_fix_design,
    build_readiness,
    build_runtime_consistency_audit,
    scan_provider_error_evidence,
)


FIELDS = [
    "step",
    "source",
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
    "dry_vent_risk",
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
    "tomato_safety_v2_heat_before",
    "tomato_safety_v2_heat_after",
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
    "profile_feasibility_gate_enabled",
    "profile_feasibility_gate_applied",
    "profile_feasibility_gate_corrections",
    "profile_feasibility_gate_source",
    "profile_feasibility_gate_original_target_temp",
    "profile_feasibility_gate_original_target_co2",
    "profile_feasibility_gate_original_target_rh",
    "profile_feasibility_gate_repaired_target_temp",
    "profile_feasibility_gate_repaired_target_co2",
    "profile_feasibility_gate_repaired_target_rh",
    "profile_feasibility_gate_hard_safety_veto_count",
    "profile_feasibility_gate_vent_required_by_safety",
    "profile_feasibility_gate_humidity_retention_allowed",
    "profile_feasibility_gate_co2_enrichment_allowed",
]


def _candidate(name: str, score: float, selected: bool, *, vent: float, heat: float = 0.0, co2: float = 0.0):
    return {
        "name": name,
        "score": score,
        "selected": selected,
        "action": {"heat": heat, "co2": co2, "screen": 0.0, "vent": vent, "lamp": 0.0, "shade": 0.0},
        "score_terms": {"total_score": score},
    }


def _row(*, step=0, hard=False, source="rollout", gate=False, gate_applied=True, candidates=True):
    cand = [
        _candidate("aggressive_vent", 1.0, True, vent=0.9, heat=0.2, co2=900.0),
        _candidate("anchor_hold", 2.0, False, vent=0.2),
    ]
    row = {
        "step": step,
        "source": source,
        "runtime_error": "",
        "runtime_error_type": "",
        "replan_reason": "",
        "temp_air": 29.5,
        "rh_air": 48.0,
        "vpd_air": 1.8,
        "glob_rad": 500.0,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "temp_violation": 0.0,
        "dry_risk": "true",
        "dry_vent_risk": "true",
        "target_temp": 24.0,
        "target_rh": 80.0,
        "target_co2": 900.0,
        "u_heating": 0.2,
        "u_co2": 0.0,
        "u_screen": 0.0,
        "u_ventilation": 0.9,
        "u_lighting": 0.0,
        "u_shading": 0.0,
        "tomato_safety_v2_applied": "true" if hard else "false",
        "tomato_safety_v2_vent_before": 0.2,
        "tomato_safety_v2_vent_after": 0.9 if hard else 0.2,
        "tomato_safety_v2_heat_before": 0.0,
        "tomato_safety_v2_heat_after": 0.0,
        "tomato_safety_v2_screen_before": 0.0,
        "tomato_safety_v2_screen_after": 0.0,
        "tomato_safety_v2_shade_before": 0.0,
        "tomato_safety_v2_shade_after": 0.0,
        "tomato_safety_v2_reasons": "hard_canopy" if hard else "",
        "rspc_action_candidates_json": json.dumps(cand) if candidates else "",
        "rspc_action_actual_candidate_count": 2 if candidates else 0,
        "rspc_action_candidate_count": 2 if candidates else 0,
        "profile_scorer_safety_gate_reason": "none",
        "rspc_action_post_shape_safety_gate_reason": "canopy_gate" if hard else "none",
        "profile_feasibility_gate_enabled": "true" if gate else "false",
        "profile_feasibility_gate_applied": "true" if gate_applied else "false",
        "profile_feasibility_gate_corrections": "target_rh:safety_cap" if gate and gate_applied else "",
        "profile_feasibility_gate_source": "plan_info_current_targets",
        "profile_feasibility_gate_original_target_temp": 24.0,
        "profile_feasibility_gate_original_target_co2": 900.0,
        "profile_feasibility_gate_original_target_rh": 80.0,
        "profile_feasibility_gate_repaired_target_temp": 20.0,
        "profile_feasibility_gate_repaired_target_co2": 430.0,
        "profile_feasibility_gate_repaired_target_rh": 72.0,
        "profile_feasibility_gate_hard_safety_veto_count": 2 if gate else 0,
        "profile_feasibility_gate_vent_required_by_safety": "true" if hard else "false",
        "profile_feasibility_gate_humidity_retention_allowed": "false" if hard else "true",
        "profile_feasibility_gate_co2_enrichment_allowed": "false" if hard else "true",
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {field: "" for field in FIELDS}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


class TestProfileCandidateRuntimeConsistencyV49(unittest.TestCase):
    def test_runtime_consistency_aligns_v47_offline_and_v48_runtime_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            v48 = root / "v48"
            original.mkdir()
            v48.mkdir()
            scenario = "y2010_d180_s43_n720"
            _write_csv(original / f"{scenario}_llm_rspc_v2.csv", [_row(step=0, hard=False), _row(step=1, hard=False)])
            _write_csv(v48 / f"{scenario}_llm_rspc_v2.csv", [_row(step=0, hard=True, gate=True), _row(step=1, hard=False, gate=True)])
            v47_json = root / "v47.json"
            v48_json = root / "v48.json"
            v47_json.write_text(json.dumps({"acceptance": {"profile_feasibility_acceptance_pass": True}}), encoding="utf-8")
            v48_json.write_text(
                json.dumps(
                    {
                        "acceptance": {"v48_acceptance_pass": False},
                        "scenario_reports": [
                            {
                                "scenario_id": scenario,
                                "comparison_window_steps": 2,
                                "hard_safety_rewrite_preferred_new_steps": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = build_runtime_consistency_audit(
                original_trace_dir=original,
                v48_trace_dir=v48,
                v47_comparison_json=v47_json,
                v48_result_json=v48_json,
                failure_scenarios=[scenario],
            )

        self.assertEqual(audit["scenario_reports"][0]["aligned_step_count"], 2)
        leak_audit = audit["hard_safety_veto_leak_audit"]
        self.assertEqual(leak_audit["expected_leak_count_from_v48"], 1)
        self.assertEqual(leak_audit["located_leak_count"], 1)
        self.assertIn("veto_score_penalty_insufficient", leak_audit["classification_counts"])

    def test_fallback_path_and_missing_candidate_are_classified_without_fabricating_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            v48 = root / "v48"
            original.mkdir()
            v48.mkdir()
            scenario = "y2018_d181_s42_n720"
            _write_csv(original / f"{scenario}_llm_rspc_v2.csv", [_row(step=0, hard=False)])
            _write_csv(v48 / f"{scenario}_llm_rspc_v2.csv", [_row(step=0, hard=True, source="anchor", gate=True, candidates=False)])
            v48_json = root / "v48.json"
            v48_json.write_text(
                json.dumps(
                    {
                        "scenario_reports": [
                            {
                                "scenario_id": scenario,
                                "comparison_window_steps": 1,
                                "hard_safety_rewrite_preferred_new_steps": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            audit = build_runtime_consistency_audit(
                original_trace_dir=original,
                v48_trace_dir=v48,
                v47_comparison_json=root / "missing_v47.json",
                v48_result_json=v48_json,
                failure_scenarios=[scenario],
            )

        leak = audit["hard_safety_veto_leak_audit"]["leak_entries"][0]
        self.assertEqual(leak["classification"], "candidate_metadata_missing")
        self.assertEqual(leak["candidate_metadata_status"], "candidate_metadata_missing")

    def test_provider_error_scan_marks_online_llm_inaccessible(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run.log").write_text("Qwen call failed: 403 Access denied", encoding="utf-8")
            execution = root / "execution.json"
            execution.write_text(
                json.dumps({"online_llm_credentials_present": True, "cache_path": str(root / "cache.json")}),
                encoding="utf-8",
            )

            scan = scan_provider_error_evidence([root])
            design = build_online_llm_accessibility_design(v48_execution_record_json=execution, scan_paths=[root])

        self.assertTrue(scan["provider_error_evidence_found"])
        self.assertFalse(design["online_llm_accessible"])
        self.assertEqual(design["online_llm_accessibility_status"], "provider_error_detected")

    def test_template_design_and_readiness_never_authorize_replay_or_claims(self):
        runtime = {
            "scenario_reports": [{"scenario_id": "s"}],
            "alignment_issue_counts": {"runtime_fallback_or_rule_path": 1},
            "hard_safety_veto_leak_audit": {
                "located_leak_count": 1,
                "candidate_metadata_missing_count": 0,
                "classification_counts": {"fallback_path_bypassed_veto": 1},
            },
        }
        online = {"online_llm_accessible": False, "online_llm_accessibility_status": "provider_error_detected"}
        template = build_profile_template_fix_design(runtime, online)
        readiness = build_readiness(runtime, online, template)

        self.assertFalse(template["implementation_boundary"]["controlled_replay_allowed"])
        self.assertFalse(template["implementation_boundary"]["performance_claim_allowed"])
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])
        self.assertEqual(readiness["next_action"], "minimal_profile_template_shadow_patch_plan")


if __name__ == "__main__":
    unittest.main()
