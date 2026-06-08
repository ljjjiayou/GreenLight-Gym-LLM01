import tempfile
import unittest
from pathlib import Path

from gl_gym.cstcc.audit import build_audit_record
from gl_gym.cstcc.audit_writer import compact_audit_record
from gl_gym.cstcc.config import load_config
from gl_gym.cstcc.contracts import ACTION_FIELDS, SemanticSuggestion, asdict_clean
from gl_gym.experiments.cstcc_v817_level0_scorer_diagnostic import build_report, write_outputs
from gl_gym.experiments.diagnose_ppo_vs_llm import cstcc_shadow_to_record, sum_metrics


RUNTIME_ACTION = {
    "u_heating": 0.25,
    "u_co2": 0.2,
    "u_screen": 0.35,
    "u_ventilation": 0.42,
    "u_lighting": 0.0,
    "u_shading": 0.2,
}


def _diagnostic_config():
    return load_config("gl_gym/cstcc/config/cstcc_v8170_diagnostic.yaml")


def _normal_state_history():
    return [
        {"temp_air": 23.0, "rh_air": 70.0, "vpd_air": 0.8, "canopy_dew_margin": 4.0, "dew_margin_air": 4.0},
        {"temp_air": 23.2, "rh_air": 69.0, "vpd_air": 0.82, "canopy_dew_margin": 4.0, "dew_margin_air": 4.0},
        {"temp_air": 23.1, "rh_air": 70.0, "vpd_air": 0.8, "canopy_dew_margin": 4.0, "dew_margin_air": 4.0},
    ]


def _dry_state_history():
    return [
        {"temp_air": 30.0, "rh_air": 48.0, "vpd_air": 1.8, "canopy_dew_margin": 4.0, "dew_margin_air": 4.0},
        {"temp_air": 31.0, "rh_air": 45.0, "vpd_air": 2.1, "canopy_dew_margin": 4.0, "dew_margin_air": 4.0},
        {"temp_air": 32.0, "rh_air": 43.0, "vpd_air": 2.3, "canopy_dew_margin": 4.0, "dew_margin_air": 4.0},
    ]


def _suggestion(regime="HIGH_VPD_DRY_STRESS"):
    return SemanticSuggestion(
        regime=regime,
        regime_confidence=0.95,
        llm_reported_confidence=0.95,
        priority_weight_suggestions={"vpd_risk": 0.50, "humidity_recovery": 0.30},
        control_intent="diagnose scorer competition without changing action",
        risk_factors=["vpd_rising"],
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
        "answers": {"json_size_kb_max": 12.0},
    }


def _jsonl_row(*, conservative_reason="confidence_below_candidate_threshold", margin=0.05):
    conservative_diag = {
        "confidence": 0.05,
        "candidate_count": 0 if conservative_reason else 1,
        "candidate_count_threshold": 0.10,
        "generated_template_count": 0 if conservative_reason else 1,
        "no_candidate_reason": conservative_reason or None,
        "no_candidate_reasons": [conservative_reason] if conservative_reason else [],
        "confidence_reasons": {"raw_confidence": 0.05},
    }
    return {
        "step": 1,
        "audit_success": True,
        "final_action_invariant_verified": True,
        "final_action_changed": False,
        "fallback_triggered": False,
        "candidate_count": 4,
        "feasible_candidate_count": 4,
        "infeasible_candidate_ratio": 0.0,
        "hard_constraint_violation_count": 0,
        "hard_constraint_violation_reason_distribution": {},
        "hard_constraint_violation_field_distribution": {},
        "risk_flags": {"dry_risk": True, "high_vpd": True, "any_risk": True},
        "active_regime_after": "HIGH_VPD_DRY_STRESS",
        "shadow_selected_sequence_id": "previous_plan_prior:previous_shift_all:0",
        "shadow_selected_source_prior": "previous_plan_prior",
        "shadow_selected_template_name": "previous_shift_all",
        "prior_generation_diagnostics": {
            "rule_prior": {
                "confidence": 0.60,
                "candidate_count": 3,
                "candidate_count_threshold": 0.60,
                "generated_template_count": 3,
                "no_candidate_reason": None,
                "confidence_reasons": {"fixed_rule_confidence": 0.60},
            },
            "conservative_prior": conservative_diag,
        },
        "score_margin_to_selected_by_prior": {
            "rule_prior": {
                "status": "lower_score",
                "total_margin": margin,
                "selected_candidate_id": "previous_plan_prior:previous_shift_all:0",
                "selected_source_prior": "previous_plan_prior",
                "selected_template_name": "previous_shift_all",
                "selected_score": 0.92,
                "best_candidate_id": "rule_prior:ventilation_ramp_limited:3",
                "best_template_name": "ventilation_ramp_limited",
                "best_score": 0.92 - margin,
                "selected_minus_competitor": {
                    "base_final_score": margin,
                    "projected_smoothness": 0.03,
                    "projected_low_reversal": 0.01,
                    "projected_energy_proxy": 0.01,
                    "prior_confidence_bonus": 0.0,
                    "risk_static_hold_penalty": 0.0,
                    "final_score": margin,
                },
            },
            "conservative_prior": {
                "status": "no_candidate" if conservative_reason else "lower_score",
                "total_margin": None if conservative_reason else margin,
                "best_candidate_id": None if conservative_reason else "conservative_prior:conservative_stabilize:4",
                "best_template_name": None if conservative_reason else "conservative_stabilize",
                "no_candidate_reason": conservative_reason or None,
                "selected_minus_competitor": {},
            },
        },
        "score_component_by_candidate": {
            "previous_plan_prior:previous_shift_all:0": {"final_score": 0.92},
            "rule_prior:ventilation_ramp_limited:3": {"final_score": 0.92 - margin},
        },
        "rule_conservative_competitiveness": {
            "rule_prior": {
                "candidate_count": 3,
                "feasible_candidate_count": 3,
                "status": "lower_score",
                "selected_score_margin": margin,
                "best_template_name": "ventilation_ramp_limited",
            },
            "conservative_prior": {
                "candidate_count": 0 if conservative_reason else 1,
                "feasible_candidate_count": 0 if conservative_reason else 1,
                "status": "no_candidate" if conservative_reason else "lower_score",
                "selected_score_margin": None if conservative_reason else 0.04,
            },
        },
    }


class TestCSTCCV817Level0ScorerDiagnostic(unittest.TestCase):
    def test_conservative_low_confidence_records_no_candidate_reason(self):
        record = build_audit_record(
            state_history=_normal_state_history(),
            action_history=[RUNTIME_ACTION],
            current_runtime_final_action=RUNTIME_ACTION,
            active_regime="NORMAL_BALANCED",
            regime_age_steps=4,
            semantic_suggestion=SemanticSuggestion(regime="NORMAL_BALANCED", regime_confidence=0.8),
            ppo_action=RUNTIME_ACTION,
            rule_action=RUNTIME_ACTION,
            config=_diagnostic_config(),
        )
        payload = asdict_clean(record)
        conservative = payload["scoring_metadata"]["prior_generation_diagnostics"]["conservative_prior"]

        self.assertEqual(conservative["candidate_count"], 0)
        self.assertEqual(conservative["no_candidate_reason"], "confidence_below_candidate_threshold")
        self.assertEqual(conservative["candidate_count_threshold"], 0.0)
        self.assertFalse(payload["final_action_changed"])

    def test_rule_margin_has_component_breakdown_without_score_change(self):
        record = build_audit_record(
            state_history=_dry_state_history(),
            action_history=[RUNTIME_ACTION],
            current_runtime_final_action=RUNTIME_ACTION,
            active_regime="HIGH_VPD_DRY_STRESS",
            regime_age_steps=10,
            semantic_suggestion=_suggestion(),
            previous_plan_sequence=[dict(RUNTIME_ACTION), {**RUNTIME_ACTION, "u_ventilation": 0.62}],
            ppo_action={**RUNTIME_ACTION, "u_ventilation": 0.8},
            rule_action={**RUNTIME_ACTION, "u_ventilation": 0.7},
            config=_diagnostic_config(),
        )
        payload = asdict_clean(record)
        metadata = payload["scoring_metadata"]
        rule_margin = metadata["score_margin_to_selected_by_prior"]["rule_prior"]

        self.assertIn(rule_margin["status"], {"selected", "lower_score"})
        self.assertIn("total_margin", rule_margin)
        self.assertIn("selected_minus_competitor", rule_margin)
        self.assertIn("final_score", rule_margin["selected_minus_competitor"])
        self.assertIn(payload["selected_sequence_id"], metadata["score_component_by_candidate"])
        self.assertEqual(payload["current_runtime_final_action"], RUNTIME_ACTION)
        self.assertFalse(payload["final_action_changed"])

    def test_compact_jsonl_and_csv_flattening_include_v817_fields(self):
        record = build_audit_record(
            state_history=_dry_state_history(),
            action_history=[RUNTIME_ACTION],
            current_runtime_final_action=RUNTIME_ACTION,
            active_regime="HIGH_VPD_DRY_STRESS",
            regime_age_steps=10,
            semantic_suggestion=_suggestion(),
            ppo_action={**RUNTIME_ACTION, "u_ventilation": 0.8},
            rule_action={**RUNTIME_ACTION, "u_ventilation": 0.7},
            config=_diagnostic_config(),
        )
        row = compact_audit_record(record, step=1, episode_id="test")
        flat = cstcc_shadow_to_record({"rollout_selection": {"cstcc_shadow": row}})
        summary = sum_metrics([{"reward": 0.0, **flat}])

        self.assertIn("prior_generation_diagnostics", row)
        self.assertIn("score_margin_to_selected_by_prior", row)
        self.assertIn("score_component_by_candidate", row)
        self.assertIn("cstcc_shadow_prior_generation_diagnostics_json", flat)
        self.assertIn("cstcc_shadow_score_margin_to_selected_by_prior_json", flat)
        self.assertIn("cstcc_shadow_score_component_by_candidate_json", flat)
        self.assertIn("cstcc_shadow_conservative_no_candidate_reason_distribution", summary)
        self.assertNotIn("sequence_candidates", row)

    def test_report_routes_conservative_no_candidate_to_allocation_plan(self):
        report, tables = build_report(
            trace_csv_rows=[{"step": 1, "cstcc_shadow_enabled": True, "cstcc_shadow_risk_flags_json": "{\"any_risk\": true}"}],
            summary_json=_summary_json(),
            jsonl_rows=[_jsonl_row(conservative_reason="confidence_below_candidate_threshold", margin=0.04)],
        )

        self.assertEqual(report["next_action"], "v8171_conservative_risk_candidate_allocation_plan")
        self.assertIn("conservative_prior_has_no_candidate", report["readiness_reasons"])
        self.assertTrue(tables["top_loss"])
        self.assertTrue(tables["conservative"])

    def test_report_routes_small_rule_margin_to_tiebreaker_when_conservative_exists(self):
        report, _tables = build_report(
            trace_csv_rows=[{"step": 1, "cstcc_shadow_enabled": True, "cstcc_shadow_risk_flags_json": "{\"any_risk\": true}"}],
            summary_json=_summary_json(),
            jsonl_rows=[_jsonl_row(conservative_reason="", margin=0.01)],
        )

        self.assertEqual(report["next_action"], "v8172_risk_window_tiebreaker_plan")

    def test_report_routes_large_rule_margin_to_risk_aligned_bonus(self):
        report, _tables = build_report(
            trace_csv_rows=[{"step": 1, "cstcc_shadow_enabled": True, "cstcc_shadow_risk_flags_json": "{\"any_risk\": true}"}],
            summary_json=_summary_json(),
            jsonl_rows=[_jsonl_row(conservative_reason="", margin=0.08)],
        )

        self.assertEqual(report["next_action"], "v8172_risk_aligned_bonus_calibration_plan")

    def test_boundary_violation_stops_before_calibration(self):
        summary = _summary_json()
        summary["trace_summary"]["cstcc_shadow_online_llm_called_steps"] = 1
        report, _tables = build_report(
            trace_csv_rows=[{"step": 1, "cstcc_shadow_enabled": True}],
            summary_json=summary,
            jsonl_rows=[_jsonl_row(conservative_reason="", margin=0.08)],
        )

        self.assertEqual(report["next_action"], "stop_and_repair_cstcc_shadow_boundary")
        self.assertFalse(report["boundaries"]["online_llm_called"])

    def test_outputs_write_all_report_files(self):
        report, tables = build_report(
            trace_csv_rows=[{"step": 1, "cstcc_shadow_enabled": True, "cstcc_shadow_risk_flags_json": "{\"any_risk\": true}"}],
            summary_json=_summary_json(),
            jsonl_rows=[_jsonl_row(conservative_reason="confidence_below_candidate_threshold", margin=0.04)],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_outputs(
                report=report,
                tables=tables,
                output_prefix=Path(tmpdir) / "cstcc_v8170_level0_scorer_diagnostic_test",
            )

            for path in outputs.values():
                self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
