import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.cstcc_v8172_risk_aligned_bonus_calibration import (
    build_report,
    evaluate_bonus_curve,
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


def _row(
    *,
    step=1,
    risk=True,
    risk_flags=None,
    selected_id="previous_plan_prior:previous_shift_all:0",
    previous_score=1.0,
    conservative_score=0.93,
    rule_score=0.98,
    ppo_score=0.90,
    conservative_feasible=True,
    margin_components=None,
):
    conservative_id = "conservative_prior:conservative_stabilize:2"
    rule_id = "rule_prior:ventilation_ramp_limited:1"
    ppo_id = "ppo_prior:ppo_follow_limited:3"
    if risk_flags is None:
        risk_flags = {"any_risk": risk, "dry_risk": risk, "high_vpd": risk, "dry_or_high_vpd_risk": risk}
    margin = previous_score - conservative_score
    if margin_components is None:
        margin_components = {
            "base_final_score": margin,
            "projected_energy_proxy": max(margin - 0.02, 0.0),
            "projected_smoothness": 0.01,
            "prior_confidence_bonus": 0.01,
            "final_score": margin,
        }
    return {
        "step": step,
        "audit_success": True,
        "final_action_invariant_verified": True,
        "final_action_changed": False,
        "fallback_triggered": False,
        "risk_flags": risk_flags,
        "shadow_selected_sequence_id": selected_id,
        "shadow_selected_source_prior": selected_id.split(":")[0],
        "shadow_selected_template_name": selected_id.split(":")[1],
        "candidate_violation_provenance": [
            _candidate(selected_id),
            _candidate(rule_id),
            _candidate(conservative_id, feasible=conservative_feasible),
            _candidate(ppo_id),
        ],
        "score_component_by_candidate": {
            selected_id: {"final_score": previous_score, "projected_energy_proxy": 0.8},
            rule_id: {"final_score": rule_score, "projected_energy_proxy": 0.7},
            conservative_id: {"final_score": conservative_score, "projected_energy_proxy": 0.6},
            ppo_id: {"final_score": ppo_score, "projected_energy_proxy": 0.7},
        },
        "score_margin_to_selected_by_prior": {
            "conservative_prior": {
                "status": "lower_score" if conservative_feasible else "no_feasible_candidate",
                "total_margin": margin if conservative_feasible else None,
                "selected_candidate_id": selected_id,
                "selected_source_prior": selected_id.split(":")[0],
                "selected_template_name": selected_id.split(":")[1],
                "selected_score": previous_score,
                "best_candidate_id": conservative_id if conservative_feasible else None,
                "best_template_name": "conservative_stabilize" if conservative_feasible else None,
                "best_score": conservative_score if conservative_feasible else None,
                "selected_minus_competitor": margin_components if conservative_feasible else {},
            }
        },
    }


class TestCSTCCV8172RiskAlignedBonusCalibration(unittest.TestCase):
    def test_bonus_applies_only_to_feasible_conservative_in_risk_rows(self):
        rows = [
            _row(step=1, risk=True, conservative_score=0.93),
            _row(step=2, risk=False, conservative_score=0.93),
            _row(step=3, risk=True, conservative_score=0.93, conservative_feasible=False),
        ]
        curve, _risk_type, _switch, _components, diagnostics = evaluate_bonus_curve(
            trace_csv_rows=[
                {"step": 1, "rh_air": 50, "vpd_air": 1.5},
                {"step": 2, "rh_air": 70, "vpd_air": 0.7},
                {"step": 3, "rh_air": 50, "vpd_air": 1.5},
            ],
            jsonl_rows=rows,
            bonus_grid=[0.0, 0.08],
        )

        by_bonus = {row["bonus"]: row for row in curve}
        self.assertEqual(by_bonus[0.08]["conservative_risk_selected_count"], 1)
        self.assertEqual(by_bonus[0.08]["nonrisk_conservative_selected_count"], 0)
        self.assertEqual(diagnostics["conservative_risk_feasible_steps"], 1)

    def test_bonus_curve_and_margin_quantiles_are_computed(self):
        rows = [
            _row(step=1, conservative_score=0.95),
            _row(step=2, conservative_score=0.93),
            _row(step=3, conservative_score=0.89),
        ]
        report, tables = build_report(
            trace_csv_rows=[{"step": i, "rh_air": 50, "vpd_air": 1.5} for i in (1, 2, 3)],
            summary_json=_summary_json(),
            jsonl_rows=rows,
        )

        quantiles = report["diagnostics"]["conservative_margin_quantiles"]
        self.assertEqual(quantiles["count"], 3)
        self.assertIn("p50", quantiles)
        self.assertIn("p75", quantiles)
        self.assertIn("p90", quantiles)
        self.assertIn("p95", quantiles)
        self.assertIn("max", quantiles)
        self.assertTrue(tables["bonus_curve"])
        self.assertTrue(tables["component_gap_quantiles"])

    def test_nonrisk_rows_never_select_conservative_because_of_bonus(self):
        rows = [_row(step=1, risk=False, conservative_score=0.93)]
        curve, _risk_type, _switch, _components, diagnostics = evaluate_bonus_curve(
            trace_csv_rows=[{"step": 1, "rh_air": 70, "vpd_air": 0.7}],
            jsonl_rows=rows,
            bonus_grid=[0.0, 0.12],
        )

        self.assertEqual(max(row["nonrisk_conservative_selected_count"] for row in curve), 0)
        self.assertEqual(diagnostics["nonrisk_conservative_selected_count_max"], 0)

    def test_switch_attribution_separates_previous_rule_and_ppo_sources(self):
        rows = [
            _row(step=1, selected_id="previous_plan_prior:previous_shift_all:0", conservative_score=0.93),
            _row(step=2, selected_id="rule_prior:ventilation_ramp_limited:1", previous_score=0.98, rule_score=1.0, conservative_score=0.93),
            _row(step=3, selected_id="ppo_prior:ppo_follow_limited:3", previous_score=0.98, ppo_score=1.0, conservative_score=0.93),
        ]
        curve, _risk_type, switch_rows, _components, _diagnostics = evaluate_bonus_curve(
            trace_csv_rows=[{"step": i, "rh_air": 50, "vpd_air": 1.5} for i in (1, 2, 3)],
            jsonl_rows=rows,
            bonus_grid=[0.10],
        )

        result = curve[0]
        self.assertEqual(result["switch_from_previous_to_conservative_count"], 1)
        self.assertEqual(result["switch_from_rule_to_conservative_count"], 1)
        self.assertEqual(result["switch_from_ppo_to_conservative_count"], 1)
        self.assertEqual(len(switch_rows), 3)

    def test_risk_type_cross_tabs_use_jsonl_flags_and_csv_fields(self):
        rows = [
            _row(step=1, risk_flags={"any_risk": True, "dew_risk": True, "humidity_or_dew_risk": True}, conservative_score=0.93),
            _row(step=2, risk_flags={"any_risk": True, "dry_risk": True, "dry_or_high_vpd_risk": True}, conservative_score=0.93),
        ]
        _curve, risk_type_rows, _switch, _components, _diagnostics = evaluate_bonus_curve(
            trace_csv_rows=[
                {"step": 1, "rh_air": 91, "vpd_air": 0.2, "canopy_dew_margin": 0.5},
                {"step": 2, "rh_air": 50, "vpd_air": 1.5, "u_ventilation": 0.8},
            ],
            jsonl_rows=rows,
            bonus_grid=[0.08],
        )
        lookup = {(row["risk_type"], row["bonus"]): row for row in risk_type_rows}

        self.assertEqual(lookup[("dew_risk", 0.08)]["risk_row_count"], 1)
        self.assertEqual(lookup[("canopy_dew_margin_lt1", 0.08)]["risk_row_count"], 1)
        self.assertEqual(lookup[("dry_vent_risk", 0.08)]["risk_row_count"], 1)

    def test_routing_to_opt_in_bonus_shadow_trace_for_interpretable_curve(self):
        rows = [_row(step=i, conservative_score=0.93) for i in range(1, 4)]
        report, _tables = build_report(
            trace_csv_rows=[{"step": i, "rh_air": 50, "vpd_air": 1.5} for i in range(1, 4)],
            summary_json=_summary_json(),
            jsonl_rows=rows,
        )

        self.assertEqual(report["next_action"], "v8172_1_opt_in_bonus_shadow_trace_plan")
        self.assertFalse(report["boundaries"]["final_action_changed"])
        self.assertFalse(report["boundaries"]["performance_claim_allowed"])

    def test_routing_to_energy_reweight_when_bonus_grid_cannot_switch(self):
        rows = [_row(step=i, conservative_score=0.70) for i in range(1, 4)]
        report, _tables = build_report(
            trace_csv_rows=[{"step": i, "rh_air": 50, "vpd_air": 1.5} for i in range(1, 4)],
            summary_json=_summary_json(),
            jsonl_rows=rows,
        )

        self.assertEqual(report["next_action"], "v8172_energy_proxy_reweight_design_plan")

    def test_routing_to_risk_type_specific_plan_when_switch_diverges(self):
        rows = [
            _row(step=1, risk_flags={"any_risk": True, "dew_risk": True, "humidity_or_dew_risk": True}, conservative_score=0.95),
            _row(step=2, risk_flags={"any_risk": True, "dry_risk": True, "dry_or_high_vpd_risk": True}, conservative_score=0.70),
        ]
        report, _tables = build_report(
            trace_csv_rows=[
                {"step": 1, "rh_air": 91, "vpd_air": 0.2, "canopy_dew_margin": 0.5},
                {"step": 2, "rh_air": 50, "vpd_air": 1.5},
            ],
            summary_json=_summary_json(),
            jsonl_rows=rows,
        )

        self.assertEqual(report["next_action"], "v8173_risk_type_specific_bonus_or_template_plan")

    def test_routing_to_schema_repair_when_candidate_scores_missing(self):
        row = _row(step=1)
        row.pop("score_component_by_candidate")
        report, _tables = build_report(
            trace_csv_rows=[{"step": 1, "rh_air": 50, "vpd_air": 1.5}],
            summary_json=_summary_json(),
            jsonl_rows=[row],
        )

        self.assertEqual(report["next_action"], "v8172_schema_repair_plan")

    def test_outputs_write_expected_files(self):
        report, tables = build_report(
            trace_csv_rows=[{"step": 1, "rh_air": 50, "vpd_air": 1.5}],
            summary_json=_summary_json(),
            jsonl_rows=[_row(step=1, conservative_score=0.93)],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_outputs(
                report=report,
                tables=tables,
                output_prefix=Path(tmpdir) / "cstcc_v8172_risk_aligned_bonus_calibration_test",
            )

            self.assertEqual(set(outputs), {"json", "md", "bonus_curve_csv", "bonus_curve_by_risk_type_csv", "switch_steps_topk_csv", "component_gap_quantiles_csv"})
            for path in outputs.values():
                self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
