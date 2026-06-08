import tempfile
import unittest
from pathlib import Path

from gl_gym.cstcc.constraints import DEFAULT_MAX_DELTA, check_hard_constraints
from gl_gym.experiments.cstcc_v8173_risk_type_specific_bonus_template_counterfactual import (
    _route_next_action,
    bonus_for_family,
    build_report,
    generate_pseudo_template_sequence,
    primary_risk_family,
    score_pseudo_template,
    vector_id,
    write_outputs,
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
    rule_score=0.96,
    ppo_score=0.90,
    current_action=None,
):
    conservative_id = "conservative_prior:conservative_stabilize:2"
    rule_id = "rule_prior:ventilation_ramp_limited:1"
    ppo_id = "ppo_prior:ppo_follow_limited:3"
    if risk_flags is None:
        risk_flags = {"any_risk": True, "dry_risk": True, "dry_or_high_vpd_risk": True}
    current_action = current_action or {
        "u_heating": 0.2,
        "u_co2": 0.1,
        "u_screen": 0.2,
        "u_ventilation": 0.5,
        "u_lighting": 0.0,
        "u_shading": 0.3,
    }
    return {
        "step": step,
        "audit_success": True,
        "risk_flags": risk_flags,
        "shadow_selected_sequence_id": selected_id,
        "current_runtime_final_action": current_action,
        "candidate_violation_provenance": [
            _candidate(selected_id),
            _candidate(rule_id),
            _candidate(conservative_id),
            _candidate(ppo_id),
        ],
        "score_component_by_candidate": {
            selected_id: _components(previous_score, prior_bonus=0.06),
            rule_id: _components(rule_score, prior_bonus=0.048),
            conservative_id: _components(conservative_score, prior_bonus=0.004, energy=0.7, smoothness=0.96),
            ppo_id: _components(ppo_score, prior_bonus=0.064),
        },
    }


class TestCSTCCV8173RiskTypeSpecificCounterfactual(unittest.TestCase):
    def test_risk_family_priority_and_non_additive_bonus(self):
        flags = {
            "any_risk": True,
            "dry_vent_risk": True,
            "canopy_dew_margin_lt0": True,
            "humidity_or_dew_risk": True,
            "dry_or_high_vpd_risk": True,
        }
        vector = {
            "dry_or_high_vpd_bonus": 0.02,
            "dry_vent_bonus": 0.04,
            "dew_or_humidity_bonus": 0.08,
            "canopy_dew_bonus": 0.12,
        }

        self.assertEqual(primary_risk_family(flags), "dry_vent")
        self.assertEqual(bonus_for_family(vector, primary_risk_family(flags)), 0.04)
        self.assertEqual(vector_id(vector), "dry=0.02|dry_vent=0.04|dew=0.08|canopy=0.12")

    def test_layered_bonus_only_selects_risk_conservative_and_not_nonrisk(self):
        rows = [
            _row(step=1, conservative_score=0.96, risk_flags={"any_risk": True, "dry_risk": True, "dry_or_high_vpd_risk": True}),
            _row(step=2, conservative_score=0.96, risk_flags={"any_risk": False}),
        ]
        vector = {
            "dry_or_high_vpd_bonus": 0.06,
            "dry_vent_bonus": 0.00,
            "dew_or_humidity_bonus": 0.00,
            "canopy_dew_bonus": 0.00,
        }
        report, tables = build_report(
            trace_csv_rows=[{"step": 1}, {"step": 2}],
            summary_json=_summary_json(),
            jsonl_rows=rows,
            vectors=[vector],
        )

        curve = tables["bonus_vector_curve"][0]
        self.assertEqual(curve["conservative_risk_selected_count"], 1)
        self.assertEqual(curve["nonrisk_conservative_selected_count"], 0)
        self.assertFalse(report["boundaries"]["final_action_changed"])
        self.assertFalse(report["boundaries"]["promotion_evidence"])

    def test_zero_bonus_mismatch_and_relative_switches_are_reported(self):
        row = _row(
            step=1,
            selected_id="rule_prior:ventilation_ramp_limited:1",
            rule_score=0.97,
            previous_score=1.0,
            conservative_score=0.96,
        )
        previous_id = "previous_plan_prior:previous_shift_all:0"
        row["candidate_violation_provenance"].append(_candidate(previous_id))
        row["score_component_by_candidate"][previous_id] = _components(1.0, prior_bonus=0.06)
        rows = [row]
        vector = {
            "dry_or_high_vpd_bonus": 0.06,
            "dry_vent_bonus": 0.00,
            "dew_or_humidity_bonus": 0.00,
            "canopy_dew_bonus": 0.00,
        }
        report, tables = build_report(
            trace_csv_rows=[{"step": 1}],
            summary_json=_summary_json(),
            jsonl_rows=rows,
            vectors=[vector],
        )

        curve = tables["bonus_vector_curve"][0]
        self.assertEqual(curve["selected_id_vs_zero_bonus_winner_mismatch_count"], 1)
        self.assertEqual(curve["switch_relative_to_zero_bonus_count"], 1)
        self.assertEqual(curve["switch_from_previous_to_conservative_count"], 1)
        self.assertTrue(tables["switch_explainability"])
        self.assertIn("energy_proxy_delta_on_switch", tables["switch_explainability"][0])

    def test_pseudo_templates_satisfy_constraints_without_runtime_registration(self):
        current = {
            "u_heating": 0.2,
            "u_co2": 0.1,
            "u_screen": 0.8,
            "u_ventilation": 0.9,
            "u_lighting": 0.0,
            "u_shading": 0.8,
        }
        sequence = generate_pseudo_template_sequence(
            "conservative_stabilize_dew",
            current_runtime_final_action=current,
        )
        result = score_pseudo_template(
            "conservative_stabilize_dew",
            current_runtime_final_action=current,
            conservative_prior_bonus=0.004,
        )

        self.assertFalse(check_hard_constraints(sequence, previous_action=current, max_delta=DEFAULT_MAX_DELTA))
        self.assertTrue(result["feasible"])
        self.assertIn("final_score", result["score_components"])

    def test_build_report_routes_to_template_design_when_variant_is_material(self):
        rows = [
            _row(step=1, conservative_score=0.50, previous_score=1.0),
            _row(step=2, conservative_score=0.50, previous_score=1.0),
        ]
        report, _tables = build_report(
            trace_csv_rows=[{"step": 1}, {"step": 2}],
            summary_json=_summary_json(),
            jsonl_rows=rows,
            vectors=[
                {
                    "dry_or_high_vpd_bonus": 0.00,
                    "dry_vent_bonus": 0.00,
                    "dew_or_humidity_bonus": 0.00,
                    "canopy_dew_bonus": 0.00,
                }
            ],
        )

        self.assertEqual(report["next_action"], "v8174_risk_specific_conservative_template_shadow_design_plan")

    def test_routing_branches_cover_bonus_energy_mixed_and_schema(self):
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
                    "nonrisk_conservative_selected_count": 0,
                    "conservative_risk_selected_rate": 0.5,
                    "rule_suppression_detected": False,
                    "previous_rule_ppo_suppression_detected": False,
                },
                "template_variant": {"template_variant_material_improvement": False},
                "zero_minus_conservative_component_quantiles": [],
            },
            records=[],
        )
        schema_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=["missing_field:score_component_by_candidate"],
            diagnostics={},
            records=[],
        )
        energy_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            diagnostics={
                "best_balanced_vector": {
                    "risk_type_switch_spread": 0.7,
                    "nonrisk_conservative_selected_count": 0,
                    "conservative_risk_selected_rate": 0.0,
                },
                "template_variant": {"template_variant_material_improvement": False},
                "zero_minus_conservative_component_quantiles": [
                    {"component": "projected_energy_proxy", "mean": 0.2}
                ],
            },
            records=[],
        )
        mixed_action, _ = _route_next_action(
            boundary=boundary,
            schema_issues=[],
            diagnostics={
                "best_balanced_vector": {
                    "risk_type_switch_spread": 0.6,
                    "nonrisk_conservative_selected_count": 0,
                    "conservative_risk_selected_rate": 0.5,
                },
                "template_variant": {"template_variant_material_improvement": False},
                "zero_minus_conservative_component_quantiles": [],
            },
            records=[{"risk_flags": {"any_risk": True}, "active_risk_flags": ["dew_risk", "dry_risk"]} for _ in range(10)],
        )

        self.assertEqual(opt_action, "v8173_1_opt_in_risk_bonus_shadow_trace_plan")
        self.assertEqual(schema_action, "v8173_schema_or_zero_bonus_reconstruction_repair_plan")
        self.assertEqual(energy_action, "v8173_energy_proxy_reweight_design_plan")
        self.assertEqual(mixed_action, "v8173_mixed_risk_conflict_resolver_design_plan")

    def test_outputs_write_expected_files(self):
        report, tables = build_report(
            trace_csv_rows=[{"step": 1}],
            summary_json=_summary_json(),
            jsonl_rows=[_row(step=1, conservative_score=0.96)],
            vectors=[
                {
                    "dry_or_high_vpd_bonus": 0.06,
                    "dry_vent_bonus": 0.00,
                    "dew_or_humidity_bonus": 0.00,
                    "canopy_dew_bonus": 0.00,
                }
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_outputs(
                report=report,
                tables=tables,
                output_prefix=Path(tmpdir) / "cstcc_v8173_risk_type_specific_bonus_template_counterfactual_test",
            )

            self.assertEqual(
                set(outputs),
                {
                    "json",
                    "md",
                    "bonus_vector_curve_csv",
                    "risk_family_curve_csv",
                    "template_variant_counterfactual_csv",
                    "switch_explainability_csv",
                },
            )
            for path in outputs.values():
                self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
