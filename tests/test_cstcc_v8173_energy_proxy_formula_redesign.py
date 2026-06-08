import unittest
from pathlib import Path

from gl_gym.experiments.cstcc_v8173_energy_proxy_formula_redesign import (
    DEFAULT_JSONL_ROOT,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TRACE_CSV,
    DEFAULT_V81731_JSON,
    DEFAULT_V81732_JSON,
    DEFAULT_V81732_POLICY_CURVE_CSV,
    DEFAULT_V81733_JSON,
    _policy_local_best_vectors,
    _route_next_action,
    _v81733_golden_issues,
    build_report,
    enrich_records_with_candidate_deltas,
    evaluate_formulas,
    formula_projected_energy_score,
    raw_projected_delta_rows,
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


def _candidate(candidate_id, *, feasible=True, delta=None):
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


def _components(final_score, *, prior_bonus=0.0, energy=0.8, smoothness=1.0):
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
    selected_id="previous_plan_prior:previous_shift_all:0",
    conservative_delta=None,
    previous_delta=None,
    previous_score=1.0,
    conservative_score=0.95,
    conservative_energy=0.4,
    previous_energy=0.8,
):
    conservative_id = "conservative_prior:conservative_stabilize:2"
    rule_id = "rule_prior:ventilation_ramp_limited:1"
    ppo_id = "ppo_prior:ppo_follow_limited:3"
    if risk_flags is None:
        risk_flags = {"any_risk": True, "dew_risk": True, "humidity_or_dew_risk": True}
    return {
        "step": step,
        "audit_success": True,
        "risk_flags": risk_flags,
        "active_regime_after": "NORMAL_BALANCED",
        "shadow_selected_sequence_id": selected_id,
        "current_runtime_final_action": {
            "u_heating": 0.2,
            "u_co2": 0.1,
            "u_screen": 0.5,
            "u_ventilation": 0.4,
            "u_lighting": 0.0,
            "u_shading": 0.3,
        },
        "candidate_violation_provenance": [
            _candidate(selected_id, delta=previous_delta),
            _candidate(rule_id, delta={"u_ventilation": 0.03}),
            _candidate(conservative_id, delta=conservative_delta),
            _candidate(ppo_id, delta={}),
        ],
        "score_component_by_candidate": {
            selected_id: _components(previous_score, prior_bonus=0.06, energy=previous_energy),
            rule_id: _components(0.94, prior_bonus=0.048, energy=0.82),
            conservative_id: _components(conservative_score, prior_bonus=0.004, energy=conservative_energy),
            ppo_id: _components(0.90, prior_bonus=0.064, energy=0.86),
        },
    }


def _records(rows):
    records, issues = _build_step_records(
        trace_csv_rows=[{"step": row["step"]} for row in rows],
        jsonl_rows=rows,
    )
    if issues:
        raise AssertionError(f"synthetic records had schema issues: {issues}")
    records, delta_issues = enrich_records_with_candidate_deltas(records, rows)
    if delta_issues:
        raise AssertionError(f"synthetic records had delta issues: {delta_issues}")
    return records


def _best_vector():
    return {
        "policy": "canopy_dew_first_v1",
        "vector_id": "dry=0.06|dry_vent=0.06|dew=0.08|canopy=0.10",
        "dry_or_high_vpd_bonus": 0.06,
        "dry_vent_bonus": 0.06,
        "dew_or_humidity_bonus": 0.08,
        "canopy_dew_bonus": 0.10,
    }


class TestCSTCCV8173EnergyProxyFormulaRedesign(unittest.TestCase):
    def test_formula_a_credit_only_for_dew_canopy_mitigating_directions(self):
        dew_row = _row(conservative_delta={"u_heating": 0.08, "u_ventilation": 0.08, "u_screen": -0.08})
        dry_row = _row(
            risk_flags={"any_risk": True, "dry_risk": True, "dry_or_high_vpd_risk": True},
            conservative_delta={"u_heating": 0.08, "u_ventilation": 0.08, "u_screen": -0.08},
        )
        dew_record = _records([dew_row])[0]
        dry_record = _records([dry_row])[0]
        dew_candidate = dew_record["candidates"]["conservative_prior:conservative_stabilize:2"]
        dry_candidate = dry_record["candidates"]["conservative_prior:conservative_stabilize:2"]
        component = dew_record["components"]["conservative_prior:conservative_stabilize:2"]

        dew_score = formula_projected_energy_score(
            "formula_A_risk_mitigation_credit",
            record=dew_record,
            candidate=dew_candidate,
            component=component,
        )
        dry_score = formula_projected_energy_score(
            "formula_A_risk_mitigation_credit",
            record=dry_record,
            candidate=dry_candidate,
            component=component,
        )

        self.assertGreater(dew_score, component["projected_energy_proxy"])
        self.assertEqual(dry_score, component["projected_energy_proxy"])

    def test_formula_b_penalizes_dry_high_vpd_inconsistent_directions(self):
        row = _row(
            risk_flags={"any_risk": True, "dry_risk": True, "high_vpd": True, "dry_or_high_vpd_risk": True},
            conservative_delta={"u_heating": 0.08, "u_ventilation": 0.08},
        )
        record = _records([row])[0]
        candidate = record["candidates"]["conservative_prior:conservative_stabilize:2"]
        component = record["components"]["conservative_prior:conservative_stabilize:2"]

        score = formula_projected_energy_score(
            "formula_B_risk_inconsistent_penalty",
            record=record,
            candidate=candidate,
            component=component,
        )

        self.assertLess(score, component["projected_energy_proxy"])

    def test_formula_c_distinguishes_canopy_dew_combo_from_dry_vent_combo(self):
        canopy_row = _row(
            risk_flags={"any_risk": True, "canopy_dew_margin_lt0": True, "dew_risk": True, "humidity_or_dew_risk": True},
            conservative_delta={"u_heating": 0.08, "u_ventilation": 0.08, "u_screen": -0.08},
        )
        dry_vent_row = _row(
            risk_flags={
                "any_risk": True,
                "dry_vent_risk": True,
                "dry_risk": True,
                "dry_or_high_vpd_risk": True,
                "high_vpd": True,
            },
            conservative_delta={"u_heating": 0.08, "u_ventilation": 0.08},
        )
        canopy_record = _records([canopy_row])[0]
        dry_vent_record = _records([dry_vent_row])[0]
        canopy_candidate = canopy_record["candidates"]["conservative_prior:conservative_stabilize:2"]
        dry_candidate = dry_vent_record["candidates"]["conservative_prior:conservative_stabilize:2"]
        canopy_component = canopy_record["components"]["conservative_prior:conservative_stabilize:2"]
        dry_component = dry_vent_record["components"]["conservative_prior:conservative_stabilize:2"]

        canopy_score = formula_projected_energy_score(
            "formula_C_combo_specific_energy_semantics",
            record=canopy_record,
            candidate=canopy_candidate,
            component=canopy_component,
        )
        dry_score = formula_projected_energy_score(
            "formula_C_combo_specific_energy_semantics",
            record=dry_vent_record,
            candidate=dry_candidate,
            component=dry_component,
        )

        self.assertGreater(canopy_score, canopy_component["projected_energy_proxy"])
        self.assertLess(dry_score, dry_component["projected_energy_proxy"])

    def test_policy_local_best_vectors_are_selected_from_v81732_curve(self):
        rows = [
            {
                "policy": "canopy_dew_first_v1",
                "vector_id": "weak",
                "dry_or_high_vpd_bonus": 0.0,
                "dry_vent_bonus": 0.0,
                "dew_or_humidity_bonus": 0.0,
                "canopy_dew_bonus": 0.0,
                "conservative_risk_selected_rate": 0.0,
                "family_target_penalty": 1.0,
                "family_extreme_rate_count": 2,
                "risk_type_switch_spread": 0.0,
                "mixed_risk_combo_spread": 0.0,
                "single_risk_spread": 0.0,
            },
            {
                "policy": "canopy_dew_first_v1",
                "vector_id": "better",
                "dry_or_high_vpd_bonus": 0.06,
                "dry_vent_bonus": 0.06,
                "dew_or_humidity_bonus": 0.08,
                "canopy_dew_bonus": 0.10,
                "conservative_risk_selected_rate": 0.55,
                "family_target_penalty": 0.1,
                "family_extreme_rate_count": 0,
                "risk_type_switch_spread": 0.2,
                "mixed_risk_combo_spread": 0.2,
                "single_risk_spread": 0.2,
                "nonrisk_conservative_selected_count": 0,
                "rule_suppression_detected": False,
                "previous_rule_ppo_suppression_detected": False,
            },
        ]

        best = _policy_local_best_vectors(rows)

        self.assertEqual(best["canopy_dew_first_v1"]["vector_id"], "better")

    def test_build_report_keeps_boundaries_false_and_raw_projected_delta_present(self):
        rows = [
            _row(
                step=1,
                conservative_delta={"u_heating": 0.08, "u_ventilation": 0.08, "u_screen": -0.08},
                conservative_score=0.89,
                conservative_energy=0.1,
                previous_energy=0.9,
            )
        ]
        v81732 = {
            "diagnostics": {
                "best_policy": {
                    **_best_vector(),
                    "risk_row_count": 1,
                    "conservative_risk_selected_count": 0,
                    "risk_type_switch_spread": 0.0,
                    "mixed_risk_combo_spread": 0.0,
                    "single_risk_spread": 0.0,
                },
                "best_policy_dominant_nonfinal_component": "raw_energy_proxy",
            },
            "next_action": "v8173_energy_proxy_reweight_design_plan",
        }
        v81733 = {
            "diagnostics": {
                "best_reweight_vector": {
                    "factor_vector_id": "canopy_lt0=1.0|canopy_lt1=1.0|dew=1.0|dry_vent=1.0|dry=1.0",
                    "conservative_risk_selected_rate": 0.0,
                    "risk_type_switch_spread": 0.0,
                    "dominant_component_after_reweight": "raw_energy_proxy",
                }
            }
        }
        report, tables = build_report(
            trace_csv_rows=[{"step": 1}],
            summary_json=_summary_json(),
            jsonl_rows=rows,
            v8173_json={},
            v81731_json={},
            v81732_json=v81732,
            v81733_json=v81733,
            v81732_policy_curve_rows=[_best_vector()],
            formulas=["formula_A_risk_mitigation_credit"],
            enforce_golden=False,
        )

        self.assertFalse(report["boundaries"]["online_llm_called"])
        self.assertFalse(report["boundaries"]["predictive_rollout_executed"])
        self.assertFalse(report["boundaries"]["real_tomato_safety_projection"])
        self.assertFalse(report["boundaries"]["runtime_scoring_changed"])
        self.assertFalse(report["boundaries"]["runtime_template_pool_changed"])
        self.assertFalse(report["boundaries"]["final_action_changed"])
        self.assertFalse(report["boundaries"]["promotion_evidence"])
        self.assertFalse(report["counterfactual_definition"]["bonus_vector_changed"])
        self.assertFalse(report["counterfactual_definition"]["prior_confidence_changed"])
        self.assertTrue(tables["formula_raw_projected_delta"])

    def test_routing_branches_cover_planned_outcomes(self):
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
        schema_action, _ = _route_next_action(boundary=boundary, schema_issues=["v81733_json_missing"], best_formula=passing)
        dry_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_formula={**passing, "dry_vent_conservative_selected_count": 21},
        )
        combo_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_formula={**passing, "mixed_risk_combo_spread": 0.8, "canopy_dew_lt0_dew_combo_conservative_selected_count": 0},
        )
        template_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_formula={**passing, "risk_type_switch_spread": 0.8, "dominant_component_after_formula": "base_final_score"},
        )

        self.assertEqual(opt_action, "v8173_5_opt_in_formula_shadow_trace_plan")
        self.assertEqual(schema_action, "v8173_schema_or_golden_trace_repair_plan")
        self.assertEqual(dry_action, "v8173_dry_vent_guard_formula_iteration_plan")
        self.assertEqual(combo_action, "v8173_combo_specific_energy_formula_iteration_plan")
        self.assertEqual(template_action, "v8174_risk_specific_template_design_plan")

    def test_real_h720_golden_regression_and_formula_result(self):
        for path in (
            DEFAULT_TRACE_CSV,
            DEFAULT_SUMMARY_JSON,
            DEFAULT_JSONL_ROOT,
            DEFAULT_V81731_JSON,
            DEFAULT_V81732_JSON,
            DEFAULT_V81733_JSON,
            DEFAULT_V81732_POLICY_CURVE_CSV,
        ):
            self.assertTrue(Path(path).exists(), f"required H720 artifact missing: {path}")

        v81733 = _load_json(DEFAULT_V81733_JSON)
        self.assertEqual(_v81733_golden_issues(v81733), [])
        report, tables = build_report(
            trace_csv_rows=_load_csv(DEFAULT_TRACE_CSV),
            summary_json=_load_json(DEFAULT_SUMMARY_JSON),
            jsonl_rows=_load_jsonl_rows(DEFAULT_JSONL_ROOT),
            v8173_json={},
            v81731_json=_load_json(DEFAULT_V81731_JSON),
            v81732_json=_load_json(DEFAULT_V81732_JSON),
            v81733_json=v81733,
            v81732_policy_curve_rows=_load_csv(DEFAULT_V81732_POLICY_CURVE_CSV),
        )

        best = report["diagnostics"]["best_formula_row"]
        self.assertEqual(best["policy"], "canopy_dew_first_v1")
        self.assertEqual(best["formula"], "baseline_current_energy")
        self.assertEqual(best["policy_local_bonus_vector_id"], "dry=0.06|dry_vent=0.06|dew=0.08|canopy=0.10")
        self.assertAlmostEqual(best["conservative_risk_selected_rate"], 0.5396825396825397)
        self.assertEqual(best["dry_vent_conservative_selected_count"], 21)
        self.assertEqual(best["canopy_dew_lt0_dew_combo_conservative_selected_count"], 0)
        self.assertTrue(report["diagnostics"]["raw_projected_energy_gap_identical"])
        self.assertEqual(report["diagnostics"]["energy_redesign_layer"], "energy_proxy_body")
        self.assertEqual(report["next_action"], "v8173_combo_specific_energy_formula_iteration_plan")
        self.assertTrue(tables["formula_curve"])
        self.assertTrue(tables["formula_family_rate_table"])
        self.assertTrue(tables["formula_combo_rate_table"])
        self.assertTrue(tables["formula_switch_explainability"])


if __name__ == "__main__":
    unittest.main()
