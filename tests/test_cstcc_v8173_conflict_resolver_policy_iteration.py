import unittest
from pathlib import Path

from gl_gym.experiments.cstcc_v8173_conflict_resolver_policy_iteration import (
    DEFAULT_JSONL_ROOT,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TRACE_CSV,
    DEFAULT_V8173_JSON,
    DEFAULT_V81731_JSON,
    POLICIES,
    _route_next_action,
    _v81731_golden_issues,
    build_report,
    component_gap_by_family,
    evaluate_policy_vectors,
    resolve_policy_family,
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


def _candidate(candidate_id, *, feasible=True):
    source, template, *_rest = str(candidate_id).split(":") + [""]
    return {
        "candidate_id": candidate_id,
        "source_prior": source,
        "template_name": template,
        "feasible": feasible,
        "violation_reason_distribution": {} if feasible else {"synthetic": 1},
        "violation_field_distribution": {},
        "first_action_delta_from_reference": {},
        "template_metadata": {},
    }


def _components(final_score, *, prior_bonus=0.0, energy=0.8, smoothness=1.0, base=None):
    base_score = final_score - prior_bonus if base is None else base
    return {
        "base_final_score": base_score,
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
    rule_score=0.94,
    ppo_score=0.90,
    ventilation=0.8,
    conservative_energy=0.7,
    selected_energy=0.9,
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
            "u_ventilation": ventilation,
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
            rule_id: _components(rule_score, prior_bonus=0.048, energy=0.82),
            conservative_id: _components(
                conservative_score,
                prior_bonus=0.004,
                energy=conservative_energy,
                smoothness=0.96,
            ),
            ppo_id: _components(ppo_score, prior_bonus=0.064, energy=0.86),
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


class TestCSTCCV8173ConflictResolverPolicyIteration(unittest.TestCase):
    def test_resolver_policy_family_mapping_and_non_additive_bonus(self):
        flags = {
            "any_risk": True,
            "dry_vent_risk": True,
            "high_vpd": True,
            "dry_or_high_vpd_risk": True,
            "humidity_or_dew_risk": True,
        }
        record = _row(risk_flags=flags, ventilation=0.8)

        self.assertEqual(resolve_policy_family(record, "severity_aware_non_additive_v2")["resolved_risk_family_new"], "dry_vent")
        self.assertEqual(resolve_policy_family(record, "dew_before_canopy_lt1_v1")["resolved_risk_family_new"], "dew_or_humidity")
        self.assertEqual(resolve_policy_family(record, "dry_vent_strong_only_v1")["resolved_risk_family_new"], "dry_vent")

        vector = {
            "dry_or_high_vpd_bonus": 0.06,
            "dry_vent_bonus": 0.02,
            "dew_or_humidity_bonus": 0.00,
            "canopy_dew_bonus": 0.00,
        }
        curve, _family, _combo, _diagnostics = evaluate_policy_vectors(
            _records([record]),
            policy="severity_aware_non_additive_v2",
            vectors=[vector],
        )
        self.assertEqual(curve[0]["conservative_risk_selected_count"], 0)
        self.assertEqual(curve[0]["nonrisk_conservative_selected_count"], 0)

    def test_family_rate_table_includes_target_gap_and_labels(self):
        rows = [
            _row(step=1, conservative_score=0.96, risk_flags={"any_risk": True, "dry_risk": True, "dry_or_high_vpd_risk": True}),
            _row(step=2, conservative_score=0.88, risk_flags={"any_risk": True, "dew_risk": True, "humidity_or_dew_risk": True}),
        ]
        vector = {
            "dry_or_high_vpd_bonus": 0.06,
            "dry_vent_bonus": 0.00,
            "dew_or_humidity_bonus": 0.00,
            "canopy_dew_bonus": 0.00,
        }
        _curve, family, _combo, diagnostics = evaluate_policy_vectors(
            _records(rows),
            policy="severity_aware_non_additive_v2",
            vectors=[vector],
        )

        by_family = {row["resolved_risk_family"]: row for row in family}
        self.assertEqual(by_family["dry_or_high_vpd"]["family_row_count"], 1)
        self.assertEqual(by_family["dry_or_high_vpd"]["family_conservative_selected_rate"], 1.0)
        self.assertEqual(by_family["dry_or_high_vpd"]["family_target_label"], "over_target")
        self.assertEqual(by_family["dew_or_humidity"]["family_target_label"], "under_target")
        self.assertIn("why_min_spread_vector_not_selected", diagnostics)

    def test_single_and_mixed_spread_are_separate(self):
        rows = [
            _row(step=1, conservative_score=0.96, risk_flags={"any_risk": True, "dry_risk": True, "dry_or_high_vpd_risk": True}),
            _row(
                step=2,
                conservative_score=0.96,
                risk_flags={
                    "any_risk": True,
                    "dry_risk": True,
                    "dry_or_high_vpd_risk": True,
                    "dew_risk": True,
                    "humidity_or_dew_risk": True,
                },
            ),
            _row(
                step=3,
                conservative_score=0.88,
                risk_flags={
                    "any_risk": True,
                    "canopy_dew_margin_lt1": True,
                    "dew_risk": True,
                    "humidity_or_dew_risk": True,
                },
            ),
        ]
        vector = {
            "dry_or_high_vpd_bonus": 0.06,
            "dry_vent_bonus": 0.00,
            "dew_or_humidity_bonus": 0.06,
            "canopy_dew_bonus": 0.00,
        }
        curve, _family, combo, _diagnostics = evaluate_policy_vectors(
            _records(rows),
            policy="severity_aware_non_additive_v2",
            vectors=[vector],
        )

        self.assertIn("single_risk_spread", curve[0])
        self.assertIn("mixed_risk_combo_spread", curve[0])
        self.assertTrue(any(row["is_mixed_risk"] for row in combo))

    def test_component_gap_by_family_reports_energy_smoothness_prior_and_base(self):
        rows = [
            _row(step=1, conservative_score=0.96),
            _row(step=2, conservative_score=0.92, risk_flags={"any_risk": True, "dew_risk": True, "humidity_or_dew_risk": True}),
        ]

        component_rows = component_gap_by_family(_records(rows), policies=["severity_aware_non_additive_v2"])
        components = {row["component"] for row in component_rows}
        self.assertIn("base_final_score", components)
        self.assertIn("raw_energy_proxy", components)
        self.assertIn("projected_smoothness", components)
        self.assertIn("prior_confidence_bonus", components)
        self.assertIn("final_score", components)
        self.assertTrue(all("p95" in row for row in component_rows))

    def test_build_report_has_best_and_min_spread_explainability_and_false_boundaries(self):
        rows = [
            _row(step=1, conservative_score=0.96, risk_flags={"any_risk": True, "dry_risk": True, "dry_or_high_vpd_risk": True}),
            _row(step=2, conservative_score=0.90, risk_flags={"any_risk": True, "dew_risk": True, "humidity_or_dew_risk": True}),
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
                "dew_or_humidity_bonus": 0.06,
                "canopy_dew_bonus": 0.00,
            },
        ]
        report, tables = build_report(
            trace_csv_rows=[{"step": 1}, {"step": 2}],
            summary_json=_summary_json(),
            jsonl_rows=rows,
            v8173_json={"next_action": "v8173_mixed_risk_conflict_resolver_design_plan"},
            v81731_json={},
            vectors=vectors,
            policies=["severity_aware_non_additive_v2"],
            enforce_golden=False,
        )

        details = report["diagnostics"]["policy_summaries"][0]
        self.assertIn("why_min_spread_vector_not_selected", details)
        self.assertTrue(tables["policy_curve"])
        self.assertFalse(report["boundaries"]["online_llm_called"])
        self.assertFalse(report["boundaries"]["predictive_rollout_executed"])
        self.assertFalse(report["boundaries"]["real_tomato_safety_projection"])
        self.assertFalse(report["boundaries"]["final_action_changed"])
        self.assertFalse(report["boundaries"]["performance_claim_allowed"])

    def test_routing_branches_cover_all_planned_next_actions(self):
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
        passing_policy = {
            "policy": "synthetic_policy",
            "risk_type_switch_spread": 0.2,
            "mixed_risk_combo_spread": 0.2,
            "single_risk_spread": 0.2,
            "conservative_risk_selected_rate": 0.55,
            "nonrisk_conservative_selected_count": 0,
            "family_extreme_rate_count": 0,
            "rule_suppression_detected": False,
            "previous_rule_ppo_suppression_detected": False,
        }
        passing_family = [
            {"family_conservative_selected_rate": 0.5, "family_row_count": 10},
            {"family_conservative_selected_rate": 0.6, "family_row_count": 10},
        ]
        opt_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_policy=passing_policy,
            best_family_rows=passing_family,
            component_rows=[],
        )
        schema_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=["golden_mismatch:v81731_family_spread"],
            best_policy=passing_policy,
            best_family_rows=passing_family,
            component_rows=[],
        )
        energy_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_policy={**passing_policy, "risk_type_switch_spread": 0.5},
            best_family_rows=passing_family,
            component_rows=[
                {"policy": "synthetic_policy", "component": "raw_energy_proxy", "mean": 0.5},
                {"policy": "synthetic_policy", "component": "raw_smoothness", "mean": 0.1},
            ],
        )
        template_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_policy={**passing_policy, "risk_type_switch_spread": 0.5, "policy": "template_policy"},
            best_family_rows=passing_family,
            component_rows=[
                {"policy": "template_policy", "component": "base_final_score", "mean": 0.5},
                {"policy": "template_policy", "component": "raw_energy_proxy", "mean": 0.1},
            ],
        )
        combo_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            best_policy={**passing_policy, "mixed_risk_combo_spread": 0.6, "risk_type_switch_spread": 0.4},
            best_family_rows=passing_family,
            component_rows=[],
        )

        self.assertEqual(opt_action, "v8173_3_opt_in_policy_resolver_bonus_shadow_trace_plan")
        self.assertEqual(schema_action, "v8173_schema_or_golden_trace_repair_plan")
        self.assertEqual(energy_action, "v8173_energy_proxy_reweight_design_plan")
        self.assertEqual(template_action, "v8174_risk_specific_template_design_plan")
        self.assertEqual(combo_action, "v8173_3_combo_specific_policy_refinement_plan")

    def test_real_h720_golden_regression_and_policy_iteration_output(self):
        for path in (DEFAULT_TRACE_CSV, DEFAULT_SUMMARY_JSON, DEFAULT_JSONL_ROOT, DEFAULT_V8173_JSON, DEFAULT_V81731_JSON):
            self.assertTrue(Path(path).exists(), f"required H720 artifact missing: {path}")

        v81731_json = _load_json(DEFAULT_V81731_JSON)
        self.assertEqual(_v81731_golden_issues(v81731_json), [])

        report, tables = build_report(
            trace_csv_rows=_load_csv(DEFAULT_TRACE_CSV),
            summary_json=_load_json(DEFAULT_SUMMARY_JSON),
            jsonl_rows=_load_jsonl_rows(DEFAULT_JSONL_ROOT),
            v8173_json=_load_json(DEFAULT_V8173_JSON),
            v81731_json=v81731_json,
        )

        self.assertEqual(report["diagnostics"]["policy_count"], len(POLICIES))
        self.assertEqual(report["diagnostics"]["bonus_vector_count"], 576)
        self.assertEqual(report["diagnostics"]["v81731_reference_best_vector"], "dry=0.06|dry_vent=0.06|dew=0.10|canopy=0.12")
        self.assertAlmostEqual(report["diagnostics"]["v81731_reference_family_spread"], 0.45)
        self.assertEqual(v81731_json["diagnostics"]["risk_row_count"], 126)
        self.assertEqual(v81731_json["diagnostics"]["mixed_risk_row_count"], 78)
        self.assertEqual(v81731_json["diagnostics"]["single_risk_row_count"], 48)
        self.assertEqual(
            v81731_json["diagnostics"]["best_balanced_vector"]["selected_id_vs_zero_bonus_winner_mismatch_count"],
            248,
        )
        self.assertTrue(tables["policy_family_rate_table"])
        self.assertTrue(tables["policy_combo_rate_table"])
        self.assertTrue(tables["policy_component_gap_by_family"])
        self.assertIn(
            report["next_action"],
            {
                "v8173_3_opt_in_policy_resolver_bonus_shadow_trace_plan",
                "v8173_energy_proxy_reweight_design_plan",
                "v8174_risk_specific_template_design_plan",
                "v8173_3_combo_specific_policy_refinement_plan",
            },
        )


if __name__ == "__main__":
    unittest.main()
