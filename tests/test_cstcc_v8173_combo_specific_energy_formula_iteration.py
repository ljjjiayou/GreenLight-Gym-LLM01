import unittest
from pathlib import Path

from gl_gym.experiments.cstcc_v8173_combo_specific_energy_formula_iteration import (
    CANOPY_DEW_DEW_COMBO,
    DEFAULT_JSONL_ROOT,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TRACE_CSV,
    DEFAULT_V81731_JSON,
    DEFAULT_V81732_JSON,
    DEFAULT_V81732_POLICY_CURVE_CSV,
    DEFAULT_V81733_JSON,
    DEFAULT_V81734_JSON,
    _route_next_action,
    _v81734_golden_issues,
    build_report,
    evaluate_formulas,
    formula_projected_energy_score,
    risk_state_surrogate_deltas,
)
from gl_gym.experiments.cstcc_v8173_energy_proxy_formula_redesign import (
    enrich_records_with_candidate_deltas,
)
from gl_gym.experiments.cstcc_v8173_risk_type_specific_bonus_template_counterfactual import (
    _build_step_records,
    _load_csv,
    _load_json,
    _load_jsonl_rows,
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


def _components(final_score, *, prior_bonus=0.0, energy=0.7, smoothness=1.0):
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
    selected_id="previous_plan_prior:previous_shift_all:0",
    conservative_delta=None,
    previous_delta=None,
    previous_score=1.0,
    conservative_score=0.92,
    conservative_energy=0.45,
    previous_energy=0.75,
):
    conservative_id = "conservative_prior:conservative_stabilize:2"
    rule_id = "rule_prior:ventilation_ramp_limited:1"
    ppo_id = "ppo_prior:ppo_follow_limited:3"
    risk_flags = risk_flags or {"any_risk": True, "dew_risk": True, "humidity_or_dew_risk": True}
    current_action = current_action or {
        "u_heating": 0.2,
        "u_co2": 0.1,
        "u_screen": 0.5,
        "u_ventilation": 0.4,
        "u_lighting": 0.0,
        "u_shading": 0.3,
    }
    state = state or {
        "temp_air": 22.0,
        "rh_air": 92.0,
        "vpd_air": 0.5,
        "canopy_dew_margin": -0.2,
        "dew_margin_air": 0.4,
    }
    return {
        "step": step,
        "audit_success": True,
        "risk_flags": risk_flags,
        "active_regime_after": "NORMAL_BALANCED",
        "shadow_selected_sequence_id": selected_id,
        "current_runtime_final_action": current_action,
        **state,
        "candidate_violation_provenance": [
            _candidate(selected_id, delta=previous_delta),
            _candidate(rule_id, delta={"u_ventilation": 0.02}),
            _candidate(conservative_id, delta=conservative_delta),
            _candidate(ppo_id, delta={}),
        ],
        "score_component_by_candidate": {
            selected_id: _components(previous_score, prior_bonus=0.06, energy=previous_energy),
            rule_id: _components(0.94, prior_bonus=0.048, energy=0.74),
            conservative_id: _components(conservative_score, prior_bonus=0.004, energy=conservative_energy),
            ppo_id: _components(0.90, prior_bonus=0.064, energy=0.75),
        },
    }


def _records(rows):
    records, issues = _build_step_records(
        trace_csv_rows=[{"step": row["step"], **{k: row[k] for k in ("temp_air", "rh_air", "vpd_air", "canopy_dew_margin", "dew_margin_air") if k in row}} for row in rows],
        jsonl_rows=rows,
    )
    if issues:
        raise AssertionError(f"synthetic records had schema issues: {issues}")
    records, delta_issues = enrich_records_with_candidate_deltas(records, rows)
    if delta_issues:
        raise AssertionError(f"synthetic records had delta issues: {delta_issues}")
    return records


def _best_vectors():
    return {
        "canopy_dew_first_v1": {
            "policy": "canopy_dew_first_v1",
            "vector_id": "dry=0.06|dry_vent=0.06|dew=0.08|canopy=0.10",
            "dry_or_high_vpd_bonus": 0.06,
            "dry_vent_bonus": 0.06,
            "dew_or_humidity_bonus": 0.08,
            "canopy_dew_bonus": 0.10,
        }
    }


class TestCSTCCV8173ComboSpecificEnergyFormulaIteration(unittest.TestCase):
    def test_surrogate_delta_directions_cover_dew_dry_and_dry_vent(self):
        dew_record = _records([
            _row(conservative_delta={"u_heating": 0.08, "u_ventilation": 0.08, "u_screen": -0.08})
        ])[0]
        dry_record = _records([
            _row(
                risk_flags={"any_risk": True, "dry_risk": True, "high_vpd": True, "dry_or_high_vpd_risk": True},
                state={"temp_air": 29.0, "rh_air": 45.0, "vpd_air": 1.8, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
                conservative_delta={"u_heating": 0.08, "u_ventilation": 0.08},
            )
        ])[0]
        dry_vent_record = _records([
            _row(
                risk_flags={
                    "any_risk": True,
                    "dry_vent_risk": True,
                    "dry_risk": True,
                    "high_vpd": True,
                    "dry_or_high_vpd_risk": True,
                },
                current_action={"u_heating": 0.2, "u_co2": 0.0, "u_screen": 0.4, "u_ventilation": 0.9, "u_lighting": 0.0, "u_shading": 0.2},
                state={"temp_air": 30.0, "rh_air": 42.0, "vpd_air": 1.9, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
                conservative_delta={"u_ventilation": -0.08},
            )
        ])[0]
        candidate = dew_record["candidates"]["conservative_prior:conservative_stabilize:2"]
        dew = risk_state_surrogate_deltas(dew_record, candidate)
        dry = risk_state_surrogate_deltas(dry_record, dry_record["candidates"]["conservative_prior:conservative_stabilize:2"])
        dry_vent = risk_state_surrogate_deltas(
            dry_vent_record,
            dry_vent_record["candidates"]["conservative_prior:conservative_stabilize:2"],
        )

        self.assertGreater(dew["delta_canopy_dew_margin_hat"], 0.0)
        self.assertGreater(dew["delta_dew_margin_air_hat"], 0.0)
        self.assertLess(dew["delta_rh_hat"], 0.0)
        self.assertGreater(dry["delta_vpd_hat"], 0.0)
        self.assertGreater(dry_vent["dry_vent_relief_hat"], 0.0)

    def test_formula_e_uses_risk_state_improvement_not_action_direction_alone(self):
        dew_record = _records([
            _row(conservative_delta={"u_heating": 0.08, "u_ventilation": 0.08, "u_screen": -0.08})
        ])[0]
        dry_record = _records([
            _row(
                risk_flags={"any_risk": True, "dry_risk": True, "high_vpd": True, "dry_or_high_vpd_risk": True},
                state={"temp_air": 29.0, "rh_air": 45.0, "vpd_air": 1.8, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
                conservative_delta={"u_heating": 0.08, "u_ventilation": 0.08, "u_screen": -0.08},
            )
        ])[0]
        dew_candidate = dew_record["candidates"]["conservative_prior:conservative_stabilize:2"]
        dry_candidate = dry_record["candidates"]["conservative_prior:conservative_stabilize:2"]
        dew_component = dew_record["components"]["conservative_prior:conservative_stabilize:2"]
        dry_component = dry_record["components"]["conservative_prior:conservative_stabilize:2"]

        dew_score = formula_projected_energy_score(
            "formula_E_risk_state_transition_credit",
            record=dew_record,
            candidate=dew_candidate,
            component=dew_component,
        )
        dry_score = formula_projected_energy_score(
            "formula_E_risk_state_transition_credit",
            record=dry_record,
            candidate=dry_candidate,
            component=dry_component,
        )

        self.assertGreater(dew_score, dew_component["projected_energy_proxy"])
        self.assertLess(dry_score, dry_component["projected_energy_proxy"])

    def test_formula_f_dry_vent_guard_penalizes_high_vent_hold_and_credits_relief(self):
        common = {
            "risk_flags": {
                "any_risk": True,
                "dry_vent_risk": True,
                "dry_risk": True,
                "high_vpd": True,
                "dry_or_high_vpd_risk": True,
            },
            "current_action": {"u_heating": 0.2, "u_co2": 0.0, "u_screen": 0.4, "u_ventilation": 0.9, "u_lighting": 0.0, "u_shading": 0.2},
            "state": {"temp_air": 30.0, "rh_air": 42.0, "vpd_air": 1.9, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
        }
        hold_record = _records([_row(**common, conservative_delta={"u_ventilation": 0.0})])[0]
        relief_record = _records([_row(**common, conservative_delta={"u_ventilation": -0.08})])[0]
        hold_candidate = hold_record["candidates"]["conservative_prior:conservative_stabilize:2"]
        relief_candidate = relief_record["candidates"]["conservative_prior:conservative_stabilize:2"]
        hold_component = hold_record["components"]["conservative_prior:conservative_stabilize:2"]
        relief_component = relief_record["components"]["conservative_prior:conservative_stabilize:2"]

        hold_score = formula_projected_energy_score(
            "formula_F_dry_vent_guard",
            record=hold_record,
            candidate=hold_candidate,
            component=hold_component,
        )
        relief_score = formula_projected_energy_score(
            "formula_F_dry_vent_guard",
            record=relief_record,
            candidate=relief_candidate,
            component=relief_component,
        )

        self.assertLess(hold_score, hold_component["projected_energy_proxy"])
        self.assertGreater(relief_score, relief_component["projected_energy_proxy"])

    def test_formula_g_synthetic_changes_combo_extremes_without_runtime_changes(self):
        dry_row = _row(
            step=1,
            risk_flags={
                "any_risk": True,
                "dry_vent_risk": True,
                "dry_risk": True,
                "high_vpd": True,
                "dry_or_high_vpd_risk": True,
            },
            current_action={"u_heating": 0.2, "u_co2": 0.0, "u_screen": 0.4, "u_ventilation": 0.9, "u_lighting": 0.0, "u_shading": 0.2},
            state={"temp_air": 30.0, "rh_air": 42.0, "vpd_air": 1.9, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
            conservative_delta={"u_heating": 0.03, "u_ventilation": 0.0},
            conservative_score=0.94,
            previous_score=1.0,
        )
        canopy_row = _row(
            step=2,
            risk_flags={"any_risk": True, "canopy_dew_margin_lt0": True, "dew_risk": True, "humidity_or_dew_risk": True},
            state={"temp_air": 21.0, "rh_air": 94.0, "vpd_air": 0.4, "canopy_dew_margin": -0.5, "dew_margin_air": 0.2},
            conservative_delta={"u_heating": 0.09, "u_ventilation": 0.09, "u_screen": -0.08},
            conservative_score=0.90,
            previous_score=1.0,
        )
        curve, _family, _combo, diagnostics = evaluate_formulas(
            _records([dry_row, canopy_row]),
            policy_best_vectors=_best_vectors(),
            formulas=["formula_G_combo_specific_two_sided_semantics"],
        )
        best = diagnostics["best_formula_row"]

        self.assertEqual(best["dry_vent_conservative_selected_count"], 0)
        self.assertEqual(best["canopy_dew_lt0_dew_combo_conservative_selected_count"], 1)
        self.assertEqual(best["nonrisk_conservative_selected_count"], 0)
        self.assertTrue(curve)

    def test_build_report_boundaries_false_and_missing_state_routes_schema_repair(self):
        row = _row(conservative_delta={"u_heating": 0.08, "u_ventilation": 0.08, "u_screen": -0.08})
        report, tables = build_report(
            trace_csv_rows=[{"step": 1}],
            summary_json=_summary_json(),
            jsonl_rows=[row],
            v8173_json={},
            v81731_json={},
            v81732_json={},
            v81733_json={},
            v81734_json={},
            v81732_policy_curve_rows=[_best_vectors()["canopy_dew_first_v1"]],
            formulas=["formula_E_risk_state_transition_credit"],
            enforce_golden=False,
        )

        self.assertFalse(report["boundaries"]["online_llm_called"])
        self.assertFalse(report["boundaries"]["predictive_rollout_executed"])
        self.assertFalse(report["boundaries"]["real_tomato_safety_projection"])
        self.assertFalse(report["boundaries"]["runtime_scoring_changed"])
        self.assertFalse(report["boundaries"]["runtime_template_pool_changed"])
        self.assertFalse(report["boundaries"]["final_action_changed"])
        self.assertFalse(report["counterfactual_definition"]["runtime_level0_sequence_score_changed"])
        self.assertFalse(report["counterfactual_definition"]["bonus_vector_changed"])
        self.assertTrue(tables["combo_failure_table"])

        missing_state = dict(row)
        for key in ("temp_air", "rh_air", "vpd_air", "canopy_dew_margin", "dew_margin_air"):
            missing_state.pop(key, None)
        repair_report, _ = build_report(
            trace_csv_rows=[{"step": 1}],
            summary_json=_summary_json(),
            jsonl_rows=[missing_state],
            v8173_json={},
            v81731_json={},
            v81732_json={},
            v81733_json={},
            v81734_json={},
            v81732_policy_curve_rows=[_best_vectors()["canopy_dew_first_v1"]],
            formulas=["formula_E_risk_state_transition_credit"],
            enforce_golden=False,
        )

        self.assertEqual(repair_report["next_action"], "v8173_schema_or_golden_trace_repair_plan")
        self.assertTrue(any(issue.startswith("risk_state_fields_missing") for issue in repair_report["schema_issues"]))

    def test_routing_branches_cover_expected_next_actions(self):
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
            "formula": "formula_G_combo_specific_two_sided_semantics",
            "conservative_risk_selected_rate": 0.55,
            "risk_type_switch_spread": 0.2,
            "mixed_risk_combo_spread": 0.2,
            "single_risk_spread": 0.2,
            "family_extreme_rate_count": 0,
            "nonrisk_conservative_selected_count": 0,
            "rule_suppression_detected": False,
            "previous_rule_ppo_suppression_detected": False,
            "dominant_component_after_formula": "prior_confidence_bonus",
            "dry_vent_row_count": 21,
            "dry_vent_conservative_selected_count": 10,
            "canopy_dew_lt0_dew_combo_row_count": 6,
            "canopy_dew_lt0_dew_combo_conservative_selected_count": 2,
        }
        opt_action, _ = _route_next_action(boundary=boundary, schema_issues=[], best_formula=passing)
        schema_action, _ = _route_next_action(boundary=boundary, schema_issues=["missing"], best_formula=passing)
        dry_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_formula={**passing, "dry_vent_conservative_selected_count": 21},
        )
        template_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_formula={**passing, "canopy_dew_lt0_dew_combo_conservative_selected_count": 0},
        )
        weight_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_formula={
                **passing,
                "risk_type_switch_spread": 0.7,
                "dominant_component_after_formula": "raw_energy_proxy",
            },
        )
        curve_template_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_formula={**passing, "dry_vent_conservative_selected_count": 21, "canopy_dew_lt0_dew_combo_conservative_selected_count": 0},
            curve_rows=[
                {
                    "formula": "formula_F_dry_vent_guard",
                    "dry_vent_row_count": 21,
                    "dry_vent_conservative_selected_count": 0,
                    "canopy_dew_lt0_dew_combo_row_count": 6,
                    "canopy_dew_lt0_dew_combo_conservative_selected_count": 0,
                }
            ],
        )

        self.assertEqual(opt_action, "v8173_6_opt_in_risk_state_formula_shadow_trace_plan")
        self.assertEqual(schema_action, "v8173_schema_or_golden_trace_repair_plan")
        self.assertEqual(dry_action, "v8173_dry_vent_guard_policy_or_template_design_plan")
        self.assertEqual(template_action, "v8174_risk_specific_template_design_plan")
        self.assertEqual(weight_action, "v8173_score_weight_interface_design_plan")
        self.assertEqual(curve_template_action, "v8174_risk_specific_template_design_plan")

    def test_real_h720_golden_regression_and_v81735_report(self):
        for path in (
            DEFAULT_TRACE_CSV,
            DEFAULT_SUMMARY_JSON,
            DEFAULT_JSONL_ROOT,
            DEFAULT_V81731_JSON,
            DEFAULT_V81732_JSON,
            DEFAULT_V81733_JSON,
            DEFAULT_V81734_JSON,
            DEFAULT_V81732_POLICY_CURVE_CSV,
        ):
            self.assertTrue(Path(path).exists(), f"required H720 artifact missing: {path}")

        v81734 = _load_json(DEFAULT_V81734_JSON)
        self.assertEqual(_v81734_golden_issues(v81734), [])
        report, tables = build_report(
            trace_csv_rows=_load_csv(DEFAULT_TRACE_CSV),
            summary_json=_load_json(DEFAULT_SUMMARY_JSON),
            jsonl_rows=_load_jsonl_rows(DEFAULT_JSONL_ROOT),
            v8173_json={},
            v81731_json=_load_json(DEFAULT_V81731_JSON),
            v81732_json=_load_json(DEFAULT_V81732_JSON),
            v81733_json=_load_json(DEFAULT_V81733_JSON),
            v81734_json=v81734,
            v81732_policy_curve_rows=_load_csv(DEFAULT_V81732_POLICY_CURVE_CSV),
        )

        observed = report["diagnostics"]["observed_gap"]["v81734_best_formula"]
        self.assertEqual(observed["formula"], "baseline_current_energy")
        self.assertAlmostEqual(observed["risk_type_switch_spread"], 0.641025641025641)
        self.assertEqual(observed["dry_vent_conservative_selected_count"], 21)
        self.assertEqual(observed["canopy_dew_lt0_dew_combo_conservative_selected_count"], 0)
        self.assertEqual(report["boundary_runtime_safety"]["jsonl_row_count"], 720)
        self.assertEqual(report["diagnostics"]["surrogate_state_fields"][1], "rh_air")
        self.assertTrue(tables["risk_state_formula_curve"])
        self.assertTrue(tables["combo_failure_table"])
        self.assertTrue(tables["dry_vent_guard_table"])
        self.assertTrue(tables["canopy_dew_recovery_table"])
        self.assertIn(
            report["next_action"],
            {
                "v8173_6_opt_in_risk_state_formula_shadow_trace_plan",
                "v8173_dry_vent_guard_policy_or_template_design_plan",
                "v8174_risk_specific_template_design_plan",
                "v8173_score_weight_interface_design_plan",
                "v8173_combo_specific_energy_formula_iteration_plan",
            },
        )


if __name__ == "__main__":
    unittest.main()
