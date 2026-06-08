import unittest
from pathlib import Path

from gl_gym.experiments.cstcc_v8173_energy_proxy_reweight_design import (
    DEFAULT_JSONL_ROOT,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TRACE_CSV,
    DEFAULT_V81731_JSON,
    DEFAULT_V81732_COMBO_CSV,
    DEFAULT_V81732_COMPONENT_CSV,
    DEFAULT_V81732_FAMILY_CSV,
    DEFAULT_V81732_JSON,
    _route_next_action,
    _v81732_golden_issues,
    aggregate_energy_gaps,
    build_report,
    evaluate_reweight_factors,
    factor_for_family,
    factor_vector_id,
)
from gl_gym.experiments.cstcc_v8173_risk_type_specific_bonus_template_counterfactual import (
    _build_step_records,
    _load_csv,
    _load_json,
    _load_jsonl_rows,
)
from gl_gym.experiments.cstcc_v8173_conflict_resolver_policy_iteration import (
    annotate_records_for_policy,
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


def _candidate(candidate_id, *, feasible=True):
    source, template, *_rest = str(candidate_id).split(":") + [""]
    return {
        "candidate_id": candidate_id,
        "source_prior": source,
        "template_name": template,
        "feasible": feasible,
        "violation_reason_distribution": {},
        "violation_field_distribution": {},
        "first_action_delta_from_reference": {},
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
    previous_score=1.0,
    conservative_score=0.95,
    conservative_energy=0.1,
    selected_energy=0.9,
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
            "u_screen": 0.2,
            "u_ventilation": 0.4,
            "u_lighting": 0.0,
            "u_shading": 0.3,
        },
        "candidate_violation_provenance": [
            _candidate(selected_id),
            _candidate(rule_id),
            _candidate(conservative_id),
            _candidate(ppo_id),
        ],
        "score_component_by_candidate": {
            selected_id: _components(previous_score, prior_bonus=0.06, energy=selected_energy),
            rule_id: _components(0.94, prior_bonus=0.048, energy=0.82),
            conservative_id: _components(
                conservative_score,
                prior_bonus=0.004,
                energy=conservative_energy,
                smoothness=0.96,
            ),
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
    return records


def _best_policy():
    return {
        "policy": "canopy_dew_first_v1",
        "vector_id": "dry=0.06|dry_vent=0.06|dew=0.08|canopy=0.10",
        "dry_or_high_vpd_bonus": 0.06,
        "dry_vent_bonus": 0.06,
        "dew_or_humidity_bonus": 0.08,
        "canopy_dew_bonus": 0.10,
    }


class TestCSTCCV8173EnergyProxyReweightDesign(unittest.TestCase):
    def test_factor_for_family_is_risk_conditioned_not_global(self):
        vector = {
            "canopy_dew_lt0_factor": 0.6,
            "canopy_dew_lt1_factor": 0.8,
            "dew_or_humidity_factor": 0.6,
            "dry_vent_factor": 0.9,
            "dry_or_high_vpd_factor": 0.8,
        }

        self.assertEqual(factor_for_family(vector, "dew_or_humidity", is_risk=True), 0.6)
        self.assertEqual(factor_for_family(vector, "dew_or_humidity", is_risk=False), 1.0)
        self.assertEqual(factor_vector_id(vector), "canopy_lt0=0.6|canopy_lt1=0.8|dew=0.6|dry_vent=0.9|dry=0.8")

    def test_energy_gap_aggregation_by_family_combo_and_target_label(self):
        rows = [
            _row(step=1, conservative_energy=0.1, selected_energy=0.9),
            _row(
                step=2,
                conservative_energy=0.7,
                selected_energy=0.7,
                risk_flags={"any_risk": True, "dry_risk": True, "dry_or_high_vpd_risk": True},
            ),
        ]
        records = annotate_records_for_policy(_records(rows), "canopy_dew_first_v1")
        by_family = aggregate_energy_gaps(
            records,
            group_name="resolved_risk_family",
            group_key=lambda record: str((record.get("resolver") or {}).get("resolved_risk_family_new") or "none"),
        )
        by_mixed = aggregate_energy_gaps(
            records,
            group_name="single_vs_mixed",
            group_key=lambda record: "mixed_risk" if "+" in str((record.get("resolver") or {}).get("risk_combo_id") or "") else "single_risk",
        )

        family_rows = {row["group_id"]: row for row in by_family}
        self.assertGreater(family_rows["dew_or_humidity"]["combined_energy_gap_mean"], 0.0)
        self.assertEqual(family_rows["dry_or_high_vpd"]["combined_energy_gap_mean"], 0.0)
        self.assertEqual(by_mixed[0]["group_name"], "single_vs_mixed")

    def test_reweight_only_changes_risk_conservative_counterfactual_not_nonrisk(self):
        rows = [
            _row(step=1, conservative_score=0.89, conservative_energy=0.0, selected_energy=1.0),
            _row(
                step=2,
                conservative_score=0.89,
                conservative_energy=0.0,
                selected_energy=1.0,
                risk_flags={"any_risk": False},
            ),
        ]
        vector = {
            "canopy_dew_lt0_factor": 1.0,
            "canopy_dew_lt1_factor": 1.0,
            "dew_or_humidity_factor": 0.6,
            "dry_vent_factor": 1.0,
            "dry_or_high_vpd_factor": 1.0,
        }
        curve, _family, _combo, _diagnostics = evaluate_reweight_factors(
            _records(rows),
            best_policy=_best_policy(),
            factor_vectors=[vector],
        )

        self.assertEqual(curve[0]["conservative_risk_selected_count"], 1)
        self.assertEqual(curve[0]["nonrisk_conservative_selected_count"], 0)
        self.assertEqual(curve[0]["switch_from_rule_to_conservative_count"], 0)

    def test_build_report_keeps_runtime_boundaries_false_and_bonus_unchanged(self):
        rows = [_row(step=1, conservative_score=0.89, conservative_energy=0.0, selected_energy=1.0)]
        v81732_json = {
            "diagnostics": {
                "best_policy": {
                    **_best_policy(),
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
        report, tables = build_report(
            trace_csv_rows=[{"step": 1}],
            summary_json=_summary_json(),
            jsonl_rows=rows,
            v8173_json={},
            v81731_json={},
            v81732_json=v81732_json,
            v81732_family_rows=[
                {
                    "policy": "canopy_dew_first_v1",
                    "vector_id": _best_policy()["vector_id"],
                    "resolved_risk_family": "dew_or_humidity",
                    "family_target_label": "under_target",
                }
            ],
            v81732_combo_rows=[],
            v81732_component_rows=[],
            factor_vectors=[
                {
                    "canopy_dew_lt0_factor": 1.0,
                    "canopy_dew_lt1_factor": 1.0,
                    "dew_or_humidity_factor": 0.6,
                    "dry_vent_factor": 1.0,
                    "dry_or_high_vpd_factor": 1.0,
                }
            ],
            enforce_golden=False,
        )

        self.assertFalse(report["boundaries"]["online_llm_called"])
        self.assertFalse(report["boundaries"]["predictive_rollout_executed"])
        self.assertFalse(report["boundaries"]["real_tomato_safety_projection"])
        self.assertFalse(report["boundaries"]["final_action_changed"])
        self.assertFalse(report["boundaries"]["runtime_scoring_changed"])
        self.assertFalse(report["boundaries"]["global_energy_weight_changed"])
        self.assertFalse(report["counterfactual_definition"]["bonus_vector_changed"])
        self.assertTrue(tables["energy_reweight_curve"])

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
            "dominant_component_after_reweight": "prior_confidence_bonus",
        }
        opt_action, _ = _route_next_action(boundary=boundary, schema_issues=[], best_reweight=passing)
        schema_action, _ = _route_next_action(boundary=boundary, schema_issues=["v81732_json_missing"], best_reweight=passing)
        formula_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_reweight={**passing, "risk_type_switch_spread": 0.6, "dominant_component_after_reweight": "raw_energy_proxy"},
        )
        combo_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_reweight={**passing, "mixed_risk_combo_spread": 0.7, "dominant_component_after_reweight": "prior_confidence_bonus"},
        )
        template_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_reweight={**passing, "risk_type_switch_spread": 0.7, "dominant_component_after_reweight": "base_final_score"},
        )

        self.assertEqual(opt_action, "v8173_4_opt_in_energy_reweight_shadow_trace_plan")
        self.assertEqual(schema_action, "v8173_schema_or_golden_trace_repair_plan")
        self.assertEqual(formula_action, "v8173_energy_proxy_formula_redesign_plan")
        self.assertEqual(combo_action, "v8173_combo_specific_energy_policy_iteration_plan")
        self.assertEqual(template_action, "v8174_risk_specific_template_design_plan")

    def test_real_h720_golden_regression_and_energy_reweight_result(self):
        for path in (
            DEFAULT_TRACE_CSV,
            DEFAULT_SUMMARY_JSON,
            DEFAULT_JSONL_ROOT,
            DEFAULT_V81731_JSON,
            DEFAULT_V81732_JSON,
            DEFAULT_V81732_FAMILY_CSV,
            DEFAULT_V81732_COMBO_CSV,
            DEFAULT_V81732_COMPONENT_CSV,
        ):
            self.assertTrue(Path(path).exists(), f"required H720 artifact missing: {path}")

        v81732_json = _load_json(DEFAULT_V81732_JSON)
        self.assertEqual(_v81732_golden_issues(v81732_json), [])
        report, tables = build_report(
            trace_csv_rows=_load_csv(DEFAULT_TRACE_CSV),
            summary_json=_load_json(DEFAULT_SUMMARY_JSON),
            jsonl_rows=_load_jsonl_rows(DEFAULT_JSONL_ROOT),
            v8173_json={},
            v81731_json=_load_json(DEFAULT_V81731_JSON),
            v81732_json=v81732_json,
            v81732_family_rows=_load_csv(DEFAULT_V81732_FAMILY_CSV),
            v81732_combo_rows=_load_csv(DEFAULT_V81732_COMBO_CSV),
            v81732_component_rows=_load_csv(DEFAULT_V81732_COMPONENT_CSV),
        )

        best = report["diagnostics"]["best_reweight_vector"]
        self.assertEqual(best["factor_vector_id"], "canopy_lt0=1.0|canopy_lt1=1.0|dew=1.0|dry_vent=1.0|dry=1.0")
        self.assertAlmostEqual(best["conservative_risk_selected_rate"], 0.5396825396825397)
        self.assertAlmostEqual(best["risk_type_switch_spread"], 0.641025641025641)
        self.assertEqual(best["dominant_component_after_reweight"], "raw_energy_proxy")
        self.assertEqual(report["next_action"], "v8173_energy_proxy_formula_redesign_plan")
        self.assertTrue(tables["energy_gap_by_family"])
        self.assertTrue(tables["energy_gap_by_combo"])
        self.assertTrue(tables["energy_reweight_switch_explainability"])


if __name__ == "__main__":
    unittest.main()
