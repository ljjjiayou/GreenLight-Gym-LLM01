import unittest
from pathlib import Path

from gl_gym.experiments.cstcc_v8173_mixed_risk_conflict_resolver_design import (
    DEFAULT_JSONL_ROOT,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TRACE_CSV,
    DEFAULT_V8173_JSON,
    _golden_issues,
    _load_csv,
    _load_json,
    _load_jsonl_rows,
    _route_next_action,
    active_risk_families,
    build_report,
    conflict_type,
    resolve_risk_family,
    risk_combo_id,
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


def _components(final_score, *, prior_bonus=0.0):
    return {
        "base_final_score": final_score - prior_bonus,
        "final_score": final_score,
        "prior_confidence_bonus": prior_bonus,
        "projected_energy_proxy": 0.8,
        "raw_energy_proxy": 0.8,
        "projected_smoothness": 1.0,
        "raw_smoothness": 1.0,
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
    rule_score=0.94,
    ppo_score=0.90,
):
    conservative_id = "conservative_prior:conservative_stabilize:2"
    rule_id = "rule_prior:ventilation_ramp_limited:1"
    ppo_id = "ppo_prior:ppo_follow_limited:3"
    if risk_flags is None:
        risk_flags = {"any_risk": True, "dry_risk": True, "dry_or_high_vpd_risk": True}
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
            "u_ventilation": 0.8,
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
            selected_id: _components(previous_score, prior_bonus=0.06),
            rule_id: _components(rule_score, prior_bonus=0.048),
            conservative_id: _components(conservative_score, prior_bonus=0.004),
            ppo_id: _components(ppo_score, prior_bonus=0.064),
        },
    }


class TestCSTCCV8173MixedRiskConflictResolverDesign(unittest.TestCase):
    def test_mixed_risk_combo_and_conflict_type_classification(self):
        flags = {
            "any_risk": True,
            "dry_risk": True,
            "dry_or_high_vpd_risk": True,
            "dew_risk": True,
            "humidity_or_dew_risk": True,
            "canopy_dew_margin_lt1": True,
        }

        self.assertEqual(active_risk_families(flags), ["canopy_dew_lt1", "dew_or_humidity", "dry_or_high_vpd"])
        self.assertEqual(risk_combo_id(flags), "canopy_dew_lt1+dew_or_humidity+dry_or_high_vpd")
        self.assertEqual(conflict_type(flags), "multi_3plus")

    def test_severity_aware_resolver_rules(self):
        canopy = resolve_risk_family(
            {"risk_flags": {"any_risk": True, "canopy_dew_margin_lt0": True, "dew_risk": True}}
        )
        dry_vent = resolve_risk_family(
            {
                "risk_flags": {"any_risk": True, "dry_vent_risk": True, "high_vpd": True},
                "current_runtime_final_action": {"u_ventilation": 0.8},
            }
        )
        dew_dry = resolve_risk_family(
            {"risk_flags": {"any_risk": True, "dew_risk": True, "humidity_or_dew_risk": True, "dry_risk": True}}
        )
        multi = resolve_risk_family(
            {
                "risk_flags": {
                    "any_risk": True,
                    "canopy_dew_margin_lt1": True,
                    "dew_risk": True,
                    "humidity_or_dew_risk": True,
                    "dry_risk": True,
                }
            }
        )

        self.assertEqual(canopy["resolved_risk_family_new"], "canopy_dew_lt0")
        self.assertEqual(dry_vent["resolved_risk_family_new"], "dry_vent")
        self.assertEqual(dry_vent["conflict_type"], "single_risk")
        self.assertEqual(dew_dry["resolved_risk_family_new"], "dew_or_humidity")
        self.assertEqual(multi["conflict_type"], "multi_3plus")

    def test_resolved_family_bonus_is_non_additive(self):
        rows = [
            _row(
                step=1,
                conservative_score=0.97,
                risk_flags={
                    "any_risk": True,
                    "dry_risk": True,
                    "dry_or_high_vpd_risk": True,
                    "dew_risk": True,
                    "humidity_or_dew_risk": True,
                },
            )
        ]
        vector = {
            "dry_or_high_vpd_bonus": 0.06,
            "dry_vent_bonus": 0.00,
            "dew_or_humidity_bonus": 0.04,
            "canopy_dew_bonus": 0.00,
        }
        report, tables = build_report(
            trace_csv_rows=[{"step": 1}],
            summary_json=_summary_json(),
            jsonl_rows=rows,
            v8173_json={},
            vectors=[vector],
            enforce_golden=False,
        )

        curve = tables["conflict_resolver_curve"][0]
        self.assertEqual(curve["conservative_risk_selected_count"], 1)
        self.assertEqual(curve["nonrisk_conservative_selected_count"], 0)
        switch = tables["resolved_family_switch_explainability"][0]
        self.assertEqual(switch["assigned_bonus"], 0.04)
        self.assertFalse(report["boundaries"]["final_action_changed"])
        self.assertFalse(report["boundaries"]["performance_claim_allowed"])

    def test_min_spread_and_best_vector_explainability_present(self):
        rows = [
            _row(step=1, conservative_score=0.96),
            _row(
                step=2,
                conservative_score=0.96,
                risk_flags={"any_risk": True, "dew_risk": True, "humidity_or_dew_risk": True},
            ),
        ]
        vectors = [
            {
                "dry_or_high_vpd_bonus": 0.00,
                "dry_vent_bonus": 0.00,
                "dew_or_humidity_bonus": 0.00,
                "canopy_dew_bonus": 0.00,
            },
            {
                "dry_or_high_vpd_bonus": 0.06,
                "dry_vent_bonus": 0.00,
                "dew_or_humidity_bonus": 0.04,
                "canopy_dew_bonus": 0.00,
            },
        ]
        report, tables = build_report(
            trace_csv_rows=[{"step": 1}, {"step": 2}],
            summary_json=_summary_json(),
            jsonl_rows=rows,
            v8173_json={},
            vectors=vectors,
            enforce_golden=False,
        )

        diagnostics = report["diagnostics"]
        self.assertIn("min_spread_vector", diagnostics)
        self.assertIn("why_min_spread_vector_not_selected", diagnostics)
        self.assertTrue(tables["min_spread_vector_explainability"])

    def test_routing_branches_cover_opt_in_policy_iteration_energy_template_and_schema(self):
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
        opt_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            diagnostics={
                "best_balanced_vector": {
                    "risk_type_switch_spread": 0.2,
                    "mixed_risk_combo_spread": 0.2,
                    "nonrisk_conservative_selected_count": 0,
                    "rule_suppression_detected": False,
                    "previous_rule_ppo_suppression_detected": False,
                }
            },
            v8173_json={},
        )
        schema_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=["golden_mismatch:best_vector"],
            diagnostics={},
            v8173_json={},
        )
        template_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            diagnostics={"best_balanced_vector": {"risk_type_switch_spread": 0.4, "mixed_risk_combo_spread": 0.4}},
            v8173_json={"diagnostics": {"template_variant": {"template_variant_material_improvement": True}}},
        )
        energy_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            diagnostics={"best_balanced_vector": {"risk_type_switch_spread": 0.4, "mixed_risk_combo_spread": 0.2}},
            v8173_json={
                "diagnostics": {
                    "zero_minus_conservative_component_quantiles": [
                        {"component": "projected_energy_proxy", "mean": 0.5}
                    ]
                }
            },
        )
        iteration_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            diagnostics={"best_balanced_vector": {"risk_type_switch_spread": 0.4, "mixed_risk_combo_spread": 0.4}},
            v8173_json={},
        )

        self.assertEqual(opt_action, "v8173_2_opt_in_conflict_resolver_bonus_shadow_trace_plan")
        self.assertEqual(schema_action, "v8173_schema_or_golden_trace_repair_plan")
        self.assertEqual(template_action, "v8174_risk_specific_template_design_plan")
        self.assertEqual(energy_action, "v8173_energy_proxy_reweight_design_plan")
        self.assertEqual(iteration_action, "v8173_2_conflict_resolver_policy_iteration")

    def test_real_h720_golden_regression_values_do_not_drift(self):
        for path in (DEFAULT_TRACE_CSV, DEFAULT_SUMMARY_JSON, DEFAULT_JSONL_ROOT, DEFAULT_V8173_JSON):
            self.assertTrue(Path(path).exists(), f"required H720 artifact missing: {path}")
        v8173_json = _load_json(DEFAULT_V8173_JSON)
        self.assertEqual(_golden_issues(v8173_json), [])

        report, _tables = build_report(
            trace_csv_rows=_load_csv(DEFAULT_TRACE_CSV),
            summary_json=_load_json(DEFAULT_SUMMARY_JSON),
            jsonl_rows=_load_jsonl_rows(DEFAULT_JSONL_ROOT),
            v8173_json=v8173_json,
        )

        reference = report["diagnostics"]
        best = (v8173_json["diagnostics"]["best_balanced_vector"])
        self.assertEqual(reference["v8173_reference_best_vector"], "dry=0.06|dry_vent=0.06|dew=0.10|canopy=0.12")
        self.assertEqual(best["conservative_risk_selected_count"], 106)
        self.assertAlmostEqual(best["conservative_risk_selected_rate"], 0.8412698413)
        self.assertAlmostEqual(best["risk_type_switch_spread"], 0.45)
        self.assertEqual(best["selected_id_vs_zero_bonus_winner_mismatch_count"], 248)
        self.assertFalse(reference["v8173_reference_template_variant_material_improvement"])
        self.assertEqual(v8173_json["next_action"], "v8173_mixed_risk_conflict_resolver_design_plan")


if __name__ == "__main__":
    unittest.main()
