import unittest
from pathlib import Path

from gl_gym.cstcc.constraints import DEFAULT_MAX_DELTA, check_hard_constraints
from gl_gym.experiments.cstcc_v8173_energy_proxy_formula_redesign import enrich_records_with_candidate_deltas
from gl_gym.experiments.cstcc_v8173_risk_type_specific_bonus_template_counterfactual import (
    _build_step_records,
    _load_csv,
    _load_json,
    _load_jsonl_rows,
)
from gl_gym.experiments.cstcc_v8174_risk_specific_template_design import (
    CANOPY_DEW_DEW_COMBO,
    DEFAULT_JSONL_ROOT,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TRACE_CSV,
    DEFAULT_V81731_JSON,
    DEFAULT_V81732_JSON,
    DEFAULT_V81732_POLICY_CURVE_CSV,
    DEFAULT_V81734_JSON,
    DEFAULT_V81735_JSON,
    _route_next_action,
    _v81735_golden_issues,
    build_report,
    generate_pseudo_template_sequence,
    humidity_gate,
    pseudo_candidates_for_record,
    winner_with_pseudo_templates,
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
    hour=6.0,
    selected_id="previous_plan_prior:previous_shift_all:0",
    conservative_score=0.92,
    previous_score=1.0,
):
    risk_flags = risk_flags or {
        "any_risk": True,
        "canopy_dew_margin_lt0": True,
        "dew_risk": True,
        "humidity_or_dew_risk": True,
    }
    current_action = current_action or {
        "u_heating": 0.18,
        "u_co2": 0.0,
        "u_screen": 0.50,
        "u_ventilation": 0.28,
        "u_lighting": 0.0,
        "u_shading": 0.25,
    }
    state = state or {
        "temp_air": 21.0,
        "rh_air": 94.0,
        "vpd_air": 0.4,
        "canopy_dew_margin": -0.5,
        "dew_margin_air": 0.2,
    }
    outside = outside or {"temp_out": 8.0, "rh_out": 60.0}
    conservative_id = "conservative_prior:conservative_stabilize:2"
    rule_id = "rule_prior:ventilation_ramp_limited:1"
    ppo_id = "ppo_prior:ppo_follow_limited:3"
    return {
        "step": step,
        "hour_of_day": hour,
        "audit_success": True,
        "risk_flags": risk_flags,
        "active_regime_after": "NORMAL_BALANCED",
        "shadow_selected_sequence_id": selected_id,
        "current_runtime_final_action": current_action,
        **state,
        **outside,
        "candidate_violation_provenance": [
            _candidate(selected_id),
            _candidate(rule_id, delta={"u_ventilation": 0.02}),
            _candidate(conservative_id, delta={"u_heating": 0.0, "u_ventilation": 0.0}),
            _candidate(ppo_id),
        ],
        "score_component_by_candidate": {
            selected_id: _components(previous_score, prior_bonus=0.06),
            rule_id: _components(0.94, prior_bonus=0.048),
            conservative_id: _components(conservative_score, prior_bonus=0.004),
            ppo_id: _components(0.90, prior_bonus=0.064),
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


class TestCSTCCV8174RiskSpecificTemplateDesign(unittest.TestCase):
    def test_humidity_gate_distinguishes_dryer_and_wetter_outside_air(self):
        dryer = _records([_row(outside={"temp_out": 8.0, "rh_out": 60.0})])[0]
        wetter = _records([_row(outside={"temp_out": 21.0, "rh_out": 99.0})])[0]

        self.assertTrue(humidity_gate(dryer)["outside_air_dryer"])
        self.assertFalse(humidity_gate(wetter)["outside_air_dryer"])
        self.assertTrue(humidity_gate(dryer)["inside_minus_outside_abs_humidity_g_m3"] > 0)

    def test_dew_template_blocks_vent_when_outside_air_not_dryer(self):
        record = _records([_row(outside={"temp_out": 21.0, "rh_out": 99.0})])[0]
        sequence, metadata = generate_pseudo_template_sequence("conservative_canopy_dew_warmup", record=record)
        last = record["current_runtime_final_action"]

        self.assertFalse(metadata["humidity_gate"]["outside_air_dryer"])
        self.assertTrue(all(action["u_ventilation"] <= last["u_ventilation"] + 1e-9 for action in sequence))

    def test_dew_pulse_uses_screen_slit_before_vent_when_outside_air_is_dryer(self):
        record = _records([_row(outside={"temp_out": 8.0, "rh_out": 60.0}, hour=6.0)])[0]
        sequence, metadata = generate_pseudo_template_sequence("conservative_dew_pulse_heat_then_vent", record=record)
        last = record["current_runtime_final_action"]

        self.assertTrue(metadata["pulse_window_active"])
        self.assertTrue(metadata["humidity_gate"]["outside_air_dryer"])
        self.assertLess(sequence[0]["u_screen"], last["u_screen"])
        self.assertLessEqual(sequence[0]["u_ventilation"], last["u_ventilation"] + 1e-9)
        self.assertTrue(any(action["u_ventilation"] > last["u_ventilation"] for action in sequence[4:]))

    def test_screen_slit_never_closes_an_already_open_screen(self):
        record = _records([
            _row(
                current_action={
                    "u_heating": 0.18,
                    "u_co2": 0.0,
                    "u_screen": 0.0,
                    "u_ventilation": 0.28,
                    "u_lighting": 0.0,
                    "u_shading": 0.25,
                },
                outside={"temp_out": 8.0, "rh_out": 60.0},
            )
        ])[0]
        sequence, _metadata = generate_pseudo_template_sequence("conservative_dew_stabilize", record=record)

        self.assertEqual(sequence[0]["u_screen"], 0.0)

    def test_dry_vent_guard_reduces_ventilation_and_dew_templates_do_not_misfire_on_dry_only(self):
        record = _records([
            _row(
                risk_flags={
                    "any_risk": True,
                    "dry_vent_risk": True,
                    "dry_risk": True,
                    "high_vpd": True,
                    "dry_or_high_vpd_risk": True,
                },
                current_action={
                    "u_heating": 0.2,
                    "u_co2": 0.2,
                    "u_screen": 0.5,
                    "u_ventilation": 0.90,
                    "u_lighting": 0.0,
                    "u_shading": 0.3,
                },
                state={
                    "temp_air": 30.0,
                    "rh_air": 42.0,
                    "vpd_air": 1.8,
                    "canopy_dew_margin": 5.0,
                    "dew_margin_air": 5.0,
                },
                outside={"temp_out": 25.0, "rh_out": 50.0},
            )
        ])[0]
        guard_sequence, guard_meta = generate_pseudo_template_sequence("conservative_dry_vent_guard", record=record)
        dew_sequence, dew_meta = generate_pseudo_template_sequence("conservative_dew_stabilize", record=record)

        self.assertTrue(guard_meta["template_applicable"])
        self.assertLess(guard_sequence[0]["u_ventilation"], record["current_runtime_final_action"]["u_ventilation"])
        self.assertFalse(dew_meta["template_applicable"])
        self.assertEqual(dew_meta["template_inapplicable_reason"], "dew_template_requires_dew_or_canopy_risk")
        self.assertEqual(dew_sequence[0]["u_ventilation"], record["current_runtime_final_action"]["u_ventilation"])

    def test_pseudo_templates_satisfy_max_delta_and_record_metadata(self):
        record = _records([_row(outside={"temp_out": 8.0, "rh_out": 60.0})])[0]
        candidates = pseudo_candidates_for_record(record)
        applicable = [candidate for candidate in candidates if candidate["template_metadata"]["template_applicable"]]

        self.assertTrue(applicable)
        for candidate in applicable:
            violations = check_hard_constraints(
                candidate["raw_sequence"],
                previous_action=record["current_runtime_final_action"],
                max_delta=DEFAULT_MAX_DELTA,
            )
            self.assertEqual(violations, [])
            self.assertIn("phase_plan", candidate["template_metadata"])
            self.assertIn("humidity_gate", candidate["template_metadata"])
            self.assertIn("first_action_delta_from_reference", candidate)

    def test_synthetic_canopy_dew_template_can_compete_without_nonrisk_selection(self):
        risk_record = _records([
            _row(
                selected_id="previous_plan_prior:previous_shift_all:0",
                previous_score=0.82,
                conservative_score=0.70,
                outside={"temp_out": 8.0, "rh_out": 60.0},
            )
        ])[0]
        nonrisk_record = _records([
            _row(
                step=2,
                risk_flags={"any_risk": False},
                state={
                    "temp_air": 23.0,
                    "rh_air": 70.0,
                    "vpd_air": 0.9,
                    "canopy_dew_margin": 4.0,
                    "dew_margin_air": 4.0,
                },
                selected_id="previous_plan_prior:previous_shift_all:0",
                previous_score=1.05,
                outside={"temp_out": 15.0, "rh_out": 70.0},
            )
        ])[0]
        winner, _score, meta = winner_with_pseudo_templates(
            risk_record,
            formula="baseline_current_energy",
            bonus_vector={
                "dry_or_high_vpd_bonus": 0.06,
                "dry_vent_bonus": 0.06,
                "dew_or_humidity_bonus": 0.08,
                "canopy_dew_bonus": 0.10,
            },
        )
        nonrisk_winner, _nonrisk_score, nonrisk_meta = winner_with_pseudo_templates(
            nonrisk_record,
            formula="baseline_current_energy",
            bonus_vector={
                "dry_or_high_vpd_bonus": 0.06,
                "dry_vent_bonus": 0.06,
                "dew_or_humidity_bonus": 0.08,
                "canopy_dew_bonus": 0.10,
            },
        )

        self.assertTrue(winner.startswith("conservative_prior:conservative_"))
        self.assertEqual(meta["candidate_kind"], "v8174_evaluator_only_pseudo_template")
        self.assertNotEqual(nonrisk_meta.get("candidate_kind"), "v8174_evaluator_only_pseudo_template")

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
            "formula": "baseline_current_energy",
            "canopy_dew_lt0_dew_combo_row_count": 6,
            "canopy_dew_lt0_dew_combo_pseudo_selected_count": 2,
            "dry_vent_row_count": 21,
            "dry_vent_pseudo_selected_count": 10,
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
        joint_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_curve={**passing, "formula": "formula_G_combo_specific_two_sided_semantics"},
            feasibility_rows=[],
            input_trace_count=1,
        )
        schema_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=["humidity_gate_fields_missing:step=1:temp_out"],
            best_curve=passing,
            feasibility_rows=[],
            input_trace_count=1,
        )
        repair_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_curve={**passing, "canopy_dew_lt0_dew_combo_pseudo_selected_count": 0},
            feasibility_rows=[],
            input_trace_count=1,
        )

        self.assertEqual(scan_action, "v8174_multi_scenario_existing_trace_scan_or_acquisition_plan")
        self.assertEqual(opt_action, "v8174_1_opt_in_risk_template_shadow_trace_plan")
        self.assertEqual(joint_action, "v8174_scorer_template_joint_calibration_plan")
        self.assertEqual(schema_action, "v8174_trace_schema_humidity_ratio_repair_plan")
        self.assertEqual(repair_action, "v8174_template_parameter_iteration_plan")

    def test_build_report_keeps_boundaries_false_and_schema_repair_for_missing_humidity_gate(self):
        row = _row()
        report, tables, multi = build_report(
            trace_csv_rows=[
                {
                    "step": 1,
                    "hour_of_day": 6.0,
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
            v81732_policy_curve_rows=[_best_vector_row()],
            enforce_golden=False,
        )
        self.assertFalse(report["boundaries"]["online_llm_called"])
        self.assertFalse(report["boundaries"]["predictive_rollout_executed"])
        self.assertFalse(report["boundaries"]["real_tomato_safety_projection"])
        self.assertFalse(report["boundaries"]["runtime_template_pool_changed"])
        self.assertFalse(report["boundaries"]["final_action_changed"])
        self.assertTrue(tables["template_candidate_curve"])
        self.assertEqual(multi["input_trace_count"], 1)

        missing_report, _tables, _multi = build_report(
            trace_csv_rows=[{"step": 1, "hour_of_day": 6.0}],
            summary_json=_summary_json(),
            jsonl_rows=[{key: value for key, value in row.items() if key not in {"temp_out", "rh_out"}}],
            v81731_json={},
            v81732_json={},
            v81734_json={},
            v81735_json={},
            v81732_policy_curve_rows=[_best_vector_row()],
            enforce_golden=False,
        )
        self.assertEqual(missing_report["next_action"], "v8174_trace_schema_humidity_ratio_repair_plan")
        self.assertTrue(any(issue.startswith("humidity_gate_fields_missing") for issue in missing_report["schema_issues"]))

    def test_real_h720_golden_regression_and_report(self):
        for path in (
            DEFAULT_TRACE_CSV,
            DEFAULT_SUMMARY_JSON,
            DEFAULT_JSONL_ROOT,
            DEFAULT_V81731_JSON,
            DEFAULT_V81732_JSON,
            DEFAULT_V81734_JSON,
            DEFAULT_V81735_JSON,
            DEFAULT_V81732_POLICY_CURVE_CSV,
        ):
            self.assertTrue(Path(path).exists(), f"required H720 artifact missing: {path}")

        v81735 = _load_json(DEFAULT_V81735_JSON)
        self.assertEqual(_v81735_golden_issues(v81735), [])
        report, tables, multi = build_report(
            trace_csv_rows=_load_csv(DEFAULT_TRACE_CSV),
            summary_json=_load_json(DEFAULT_SUMMARY_JSON),
            jsonl_rows=_load_jsonl_rows(DEFAULT_JSONL_ROOT),
            v81731_json=_load_json(DEFAULT_V81731_JSON),
            v81732_json=_load_json(DEFAULT_V81732_JSON),
            v81734_json=_load_json(DEFAULT_V81734_JSON),
            v81735_json=v81735,
            v81732_policy_curve_rows=_load_csv(DEFAULT_V81732_POLICY_CURVE_CSV),
        )

        observed = report["diagnostics"]["observed_gap"]
        self.assertEqual(observed["v81735_next_action"], "v8174_risk_specific_template_design_plan")
        self.assertEqual(report["boundary_runtime_safety"]["jsonl_row_count"], 720)
        self.assertEqual(report["counterfactual_definition"]["pseudo_templates_added_to_runtime_template_names"], False)
        self.assertTrue(tables["template_candidate_curve"])
        self.assertTrue(tables["dew_phase_table"])
        self.assertTrue(tables["outside_air_dryness_gate_table"])
        self.assertTrue(tables["template_feasibility_table"])
        self.assertEqual(multi["readiness"], "blocked_single_h720_trace_only")
        self.assertIn(
            report["next_action"],
            {
                "v8174_multi_scenario_existing_trace_scan_or_acquisition_plan",
                "v8174_1_opt_in_risk_template_shadow_trace_plan",
                "v8174_scorer_template_joint_calibration_plan",
                "v8174_template_parameter_iteration_plan",
                "v8174_trace_schema_humidity_ratio_repair_plan",
            },
        )


if __name__ == "__main__":
    unittest.main()
