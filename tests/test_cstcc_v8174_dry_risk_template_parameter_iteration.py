import unittest
from pathlib import Path

from gl_gym.cstcc.constraints import DEFAULT_MAX_DELTA, check_hard_constraints
from gl_gym.cstcc.contracts import ACTION_FIELDS
from gl_gym.experiments.cstcc_v8173_energy_proxy_formula_redesign import enrich_records_with_candidate_deltas
from gl_gym.experiments.cstcc_v8173_risk_type_specific_bonus_template_counterfactual import (
    _build_step_records,
    _load_csv,
    _load_json,
    _load_jsonl_rows,
)
from gl_gym.experiments.cstcc_v8174_dry_risk_template_parameter_iteration import (
    DEFAULT_JSONL_ROOT,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TRACE_CSV,
    DEFAULT_V81731_JSON,
    DEFAULT_V81732_JSON,
    DEFAULT_V81732_POLICY_CURVE_CSV,
    DEFAULT_V81734_JSON,
    DEFAULT_V81735_JSON,
    DEFAULT_V8174_JSON,
    _route_next_action,
    _v8174_golden_issues,
    build_report,
    conflict_resolver_metadata,
    dry_risk_severity,
    generate_pseudo_template_sequence,
    pseudo_candidates_for_record,
)


def _summary_json():
    return {
        "trace_summary": {
            "cstcc_shadow_audit_success_rate": 1.0,
            "cstcc_shadow_final_action_invariant_rate": 1.0,
            "cstcc_shadow_final_action_changed_steps": 0,
            "cstcc_shadow_online_llm_called_steps": 0,
            "cstcc_shadow_predictive_rollout_executed_steps": 0,
            "cstcc_shadow_real_tomato_safety_projection_steps": 0,
            "cstcc_shadow_fallback_rate": 0.0,
            "cstcc_shadow_mean_infeasible_candidate_ratio": 0.0,
            "cstcc_shadow_previous_shift_all_max_delta_count": 0,
            "cstcc_shadow_selected_candidate_violation_count": 0,
        },
        "answers": {"json_size_kb_max": 20.0},
    }


def _candidate(candidate_id, *, delta=None, feasible=True):
    source, template, *_rest = str(candidate_id).split(":") + [""]
    delta = delta or {}
    return {
        "candidate_id": candidate_id,
        "source_prior": source,
        "template_name": template,
        "feasible": feasible,
        "violation_reason_distribution": {},
        "violation_field_distribution": {},
        "first_action_delta_from_reference": {
            "u_heating": delta.get("u_heating", 0.0),
            "u_co2": delta.get("u_co2", 0.0),
            "u_screen": delta.get("u_screen", 0.0),
            "u_ventilation": delta.get("u_ventilation", 0.0),
            "u_lighting": delta.get("u_lighting", 0.0),
            "u_shading": delta.get("u_shading", 0.0),
        },
        "template_metadata": {},
    }


def _components(final_score, *, prior_bonus=0.0, energy=0.75, smoothness=1.0):
    return {
        "base_final_score": final_score - prior_bonus,
        "final_score": final_score,
        "prior_confidence_bonus": prior_bonus,
        "projected_energy_proxy": energy,
        "raw_energy_proxy": energy,
        "projected_smoothness": smoothness,
        "raw_smoothness": smoothness,
        "projected_low_reversal": 1.0,
        "raw_low_reversal": 1.0,
        "risk_static_hold_penalty": 0.0,
        "rewrite_penalty": 0.0,
        "uncertainty_penalty": 0.0,
    }


def _row(
    *,
    step=1,
    risk_flags=None,
    current_action=None,
    state=None,
    outside=None,
    hour=13.0,
    selected_id="previous_plan_prior:previous_shift_all:0",
):
    risk_flags = risk_flags or {
        "any_risk": True,
        "dry_or_high_vpd_risk": True,
        "high_vpd": True,
        "dry_vent_risk": True,
    }
    current_action = current_action or {
        "u_heating": 0.30,
        "u_co2": 0.2,
        "u_screen": 0.50,
        "u_ventilation": 0.85,
        "u_lighting": 0.2,
        "u_shading": 0.25,
    }
    state = state or {
        "temp_air": 29.0,
        "rh_air": 58.0,
        "vpd_air": 1.30,
        "canopy_dew_margin": 5.0,
        "dew_margin_air": 5.0,
    }
    outside = outside or {"temp_out": 25.0, "rh_out": 60.0}
    conservative_id = "conservative_prior:conservative_stabilize:2"
    return {
        "step": step,
        "hour_of_day": hour,
        "audit_success": True,
        "risk_flags": risk_flags,
        "active_regime_after": "DRY_HIGH_VPD",
        "shadow_selected_sequence_id": selected_id,
        "current_runtime_final_action": current_action,
        **state,
        **outside,
        "candidate_violation_provenance": [
            _candidate(selected_id),
            _candidate(conservative_id, delta={"u_ventilation": -0.04}),
        ],
        "score_component_by_candidate": {
            selected_id: _components(1.0, prior_bonus=0.06),
            conservative_id: _components(0.92, prior_bonus=0.004),
        },
    }


def _records(rows):
    trace_rows = []
    for row in rows:
        trace_rows.append(
            {
                "step": row["step"],
                "hour_of_day": row["hour_of_day"],
                **{
                    key: row[key]
                    for key in (
                        "temp_air",
                        "rh_air",
                        "vpd_air",
                        "canopy_dew_margin",
                        "dew_margin_air",
                        "temp_out",
                        "rh_out",
                    )
                    if key in row
                },
            }
        )
    records, issues = _build_step_records(trace_csv_rows=trace_rows, jsonl_rows=rows)
    if issues:
        raise AssertionError(f"synthetic records had schema issues: {issues}")
    records, delta_issues = enrich_records_with_candidate_deltas(records, rows)
    if delta_issues:
        raise AssertionError(f"synthetic records had delta issues: {delta_issues}")
    return records


def _best_vector_row():
    return {
        "policy": "canopy_dew_first_v1",
        "vector_id": "dry=0.06|dry_vent=0.06|dew=0.08|canopy=0.10",
        "dry_or_high_vpd_bonus": 0.06,
        "dry_vent_bonus": 0.06,
        "dew_or_humidity_bonus": 0.08,
        "canopy_dew_bonus": 0.10,
    }


class TestCSTCCV8174DryRiskTemplateParameterIteration(unittest.TestCase):
    def test_dry_severity_gates_mild_moderate_and_severe(self):
        mild = _row(
            risk_flags={"any_risk": True, "dry_or_high_vpd_risk": True},
            state={"temp_air": 25.0, "rh_air": 70.0, "vpd_air": 0.95, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
        )
        moderate = _row(
            risk_flags={"any_risk": True, "dry_or_high_vpd_risk": True, "high_vpd": True, "dry_vent_risk": True},
            state={"temp_air": 29.0, "rh_air": 58.0, "vpd_air": 1.30, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
        )
        severe = _row(
            risk_flags={"any_risk": True, "dry_or_high_vpd_risk": True, "high_vpd": True, "dry_vent_risk": True},
            state={"temp_air": 30.0, "rh_air": 52.0, "vpd_air": 1.55, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
        )
        severe["dry_vent_age_steps"] = 4

        self.assertEqual(dry_risk_severity(mild)["dry_severity"], "mild")
        self.assertFalse(dry_risk_severity(mild)["strong_guard_allowed"])
        self.assertEqual(dry_risk_severity(moderate)["dry_severity"], "moderate")
        self.assertFalse(dry_risk_severity(moderate)["strong_guard_allowed"])
        self.assertEqual(dry_risk_severity(severe)["dry_severity"], "severe")
        self.assertTrue(dry_risk_severity(severe)["strong_guard_allowed"])

    def test_limited_guard_preserves_minimum_ventilation_and_constraints(self):
        record = _row(
            state={"temp_air": 30.0, "rh_air": 52.0, "vpd_air": 1.60, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0}
        )
        record["dry_vent_age_steps"] = 4
        sequence, metadata = generate_pseudo_template_sequence("conservative_dry_vent_guard_limited", record=record)

        self.assertTrue(metadata["template_applicable"])
        self.assertTrue(metadata["ventilation_floor_respected"])
        self.assertGreaterEqual(min(action["u_ventilation"] for action in sequence), 0.10)
        self.assertLessEqual(sequence[-1]["u_ventilation"], 0.20)
        self.assertGreater(sequence[-1]["u_ventilation"], 0.0)
        self.assertLess(sequence[-1]["u_heating"], record["current_runtime_final_action"]["u_heating"])
        self.assertEqual(
            check_hard_constraints(sequence, previous_action=record["current_runtime_final_action"], max_delta=DEFAULT_MAX_DELTA),
            [],
        )

    def test_external_humidification_marker_does_not_add_action_field_or_score_candidate(self):
        record = _row()
        sequence, metadata = generate_pseudo_template_sequence("conservative_dry_external_humidification_needed", record=record)
        candidate = [
            item
            for item in pseudo_candidates_for_record(record)
            if item["template_name"] == "conservative_dry_external_humidification_needed"
        ][0]

        self.assertTrue(metadata["external_humidification_needed"])
        self.assertFalse(metadata["humidification_actuator_available"])
        self.assertTrue(metadata["metadata_marker_only"])
        self.assertFalse(candidate["score_eligible"])
        self.assertEqual(set(sequence[0].keys()), set(ACTION_FIELDS))

    def test_conflict_resolver_prioritizes_severe_dew_canopy_and_blocks_dry_only_dew_template(self):
        mixed = _row(
            risk_flags={
                "any_risk": True,
                "dry_or_high_vpd_risk": True,
                "high_vpd": True,
                "dry_vent_risk": True,
                "canopy_dew_margin_lt0": True,
                "humidity_or_dew_risk": True,
            },
            state={"temp_air": 24.0, "rh_air": 91.0, "vpd_air": 1.20, "canopy_dew_margin": -0.2, "dew_margin_air": 0.2},
        )
        dry_only = _row()
        dry_sequence, dry_meta = generate_pseudo_template_sequence("conservative_dry_vent_guard_limited", record=mixed)
        dew_sequence, dew_meta = generate_pseudo_template_sequence("conservative_dew_stabilize", record=dry_only)

        self.assertEqual(conflict_resolver_metadata(mixed)["resolved_template_family"], "dew_canopy")
        self.assertFalse(dry_meta["template_applicable"])
        self.assertEqual(dry_meta["template_inapplicable_reason"], "blocked_by_severe_dew_canopy_conflict")
        self.assertFalse(dew_meta["template_applicable"])
        self.assertEqual(dew_meta["template_inapplicable_reason"], "dew_template_requires_dew_or_canopy_risk")
        self.assertEqual(dry_sequence[0], mixed["current_runtime_final_action"])
        self.assertEqual(dew_sequence[0], dry_only["current_runtime_final_action"])

    def test_routing_branches(self):
        boundary = {
            "audit_success_rate": 1.0,
            "final_action_invariant_rate": 1.0,
            "final_action_changed_steps": 0,
            "online_llm_called_steps": 0,
            "predictive_rollout_executed_steps": 0,
            "real_tomato_safety_projection_steps": 0,
            "fallback_rate": 0.0,
            "mean_infeasible_candidate_ratio": 0.0,
            "previous_shift_all_max_delta_count": 0,
            "selected_candidate_max_delta_count": 0,
        }
        passing = {
            "canopy_dew_lt0_dew_combo_row_count": 6,
            "canopy_dew_lt0_dew_combo_pseudo_selected_count": 6,
            "dry_vent_row_count": 21,
            "dry_vent_limited_guard_selected_count": 12,
            "mild_dry_limited_guard_selected_count": 0,
            "dry_formula_g_dependency_steps": 0,
            "nonrisk_pseudo_template_selected_count": 0,
        }

        scan_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_curve=passing,
            feasibility_rows=[],
            input_trace_count=1,
        )
        opt_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_curve=passing,
            feasibility_rows=[],
            input_trace_count=3,
        )
        gate_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_curve={**passing, "dry_vent_limited_guard_selected_count": 21},
            feasibility_rows=[],
            input_trace_count=1,
        )
        joint_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_curve={**passing, "dry_formula_g_dependency_steps": 2},
            feasibility_rows=[],
            input_trace_count=1,
        )
        schema_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=["dry_risk_fields_missing:step=1:vpd_air"],
            best_curve=passing,
            feasibility_rows=[],
            input_trace_count=1,
        )

        self.assertEqual(scan_action, "v8174_multi_scenario_existing_trace_scan_or_acquisition_plan")
        self.assertEqual(opt_action, "v8174_1_opt_in_risk_template_shadow_trace_plan")
        self.assertEqual(gate_action, "v8174_dry_severity_gate_iteration_plan")
        self.assertEqual(joint_action, "v8174_scorer_template_joint_calibration_plan")
        self.assertEqual(schema_action, "v8174_trace_schema_dry_risk_repair_plan")

    def test_build_report_keeps_boundaries_false_and_accepts_synthetic_trace(self):
        row = _row(
            state={"temp_air": 30.0, "rh_air": 52.0, "vpd_air": 1.55, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0}
        )
        report, tables, multi = build_report(
            trace_csv_rows=[
                {
                    "step": row["step"],
                    "hour_of_day": row["hour_of_day"],
                    "temp_air": row["temp_air"],
                    "rh_air": row["rh_air"],
                    "vpd_air": row["vpd_air"],
                    "canopy_dew_margin": row["canopy_dew_margin"],
                    "dew_margin_air": row["dew_margin_air"],
                    "temp_out": row["temp_out"],
                    "rh_out": row["rh_out"],
                }
            ],
            summary_json=_summary_json(),
            jsonl_rows=[row],
            v81731_json={},
            v81732_json={},
            v81734_json={},
            v81735_json={},
            v8174_json={},
            v81732_policy_curve_rows=[_best_vector_row()],
            enforce_golden=False,
        )

        self.assertFalse(report["boundaries"]["online_llm_called"])
        self.assertFalse(report["boundaries"]["predictive_rollout_executed"])
        self.assertFalse(report["boundaries"]["real_tomato_safety_projection"])
        self.assertFalse(report["boundaries"]["runtime_template_pool_changed"])
        self.assertFalse(report["boundaries"]["final_action_changed"])
        self.assertTrue(tables["dry_severity_table"])
        self.assertTrue(tables["dry_template_curve"])
        self.assertTrue(tables["dry_vent_saturation_table"])
        self.assertTrue(tables["template_feasibility_table"])
        self.assertTrue(tables["conflict_resolver_table"])
        self.assertEqual(multi["input_trace_count"], 1)

    def test_real_h720_golden_regression_and_new_report(self):
        for path in (
            DEFAULT_TRACE_CSV,
            DEFAULT_SUMMARY_JSON,
            DEFAULT_JSONL_ROOT,
            DEFAULT_V81731_JSON,
            DEFAULT_V81732_JSON,
            DEFAULT_V81734_JSON,
            DEFAULT_V81735_JSON,
            DEFAULT_V8174_JSON,
            DEFAULT_V81732_POLICY_CURVE_CSV,
        ):
            self.assertTrue(Path(path).exists(), f"required H720 artifact missing: {path}")

        v8174 = _load_json(DEFAULT_V8174_JSON)
        self.assertEqual(_v8174_golden_issues(v8174), [])
        report, tables, multi = build_report(
            trace_csv_rows=_load_csv(DEFAULT_TRACE_CSV),
            summary_json=_load_json(DEFAULT_SUMMARY_JSON),
            jsonl_rows=_load_jsonl_rows(DEFAULT_JSONL_ROOT),
            v81731_json=_load_json(DEFAULT_V81731_JSON),
            v81732_json=_load_json(DEFAULT_V81732_JSON),
            v81734_json=_load_json(DEFAULT_V81734_JSON),
            v81735_json=_load_json(DEFAULT_V81735_JSON),
            v8174_json=v8174,
            v81732_policy_curve_rows=_load_csv(DEFAULT_V81732_POLICY_CURVE_CSV),
        )

        best = report["diagnostics"]["best_curve_row"]
        self.assertEqual(report["boundary_runtime_safety"]["jsonl_row_count"], 720)
        self.assertEqual(report["counterfactual_definition"]["dry_pseudo_templates_added_to_runtime_template_names"], False)
        self.assertEqual(report["counterfactual_definition"]["external_humidification_is_metadata_only"], True)
        self.assertEqual(best["canopy_dew_lt0_dew_combo_pseudo_selected_count"], 6)
        self.assertEqual(best["dry_vent_row_count"], 21)
        self.assertLess(best["dry_vent_limited_guard_selected_count"], best["dry_vent_row_count"])
        self.assertEqual(best["mild_dry_limited_guard_selected_count"], 0)
        self.assertEqual(best["nonrisk_pseudo_template_selected_count"], 0)
        self.assertEqual(multi["readiness"], "blocked_single_h720_trace_only")
        self.assertIn(
            report["next_action"],
            {
                "v8174_multi_scenario_existing_trace_scan_or_acquisition_plan",
                "v8174_1_opt_in_risk_template_shadow_trace_plan",
                "v8174_dry_severity_gate_iteration_plan",
                "v8174_scorer_template_joint_calibration_plan",
                "v8174_template_parameter_iteration_plan",
                "v8174_trace_schema_dry_risk_repair_plan",
            },
        )


if __name__ == "__main__":
    unittest.main()
