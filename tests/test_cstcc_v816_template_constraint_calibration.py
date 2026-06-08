import tempfile
import unittest
from pathlib import Path

from gl_gym.cstcc.audit import build_audit_record
from gl_gym.cstcc.config import load_config
from gl_gym.cstcc.contracts import ACTION_FIELDS, SemanticSuggestion, asdict_clean
from gl_gym.experiments.cstcc_v816_template_constraint_calibration import (
    build_calibration_report,
    write_outputs,
)


LAST_ACTION = {
    "u_heating": 0.2,
    "u_co2": 0.2,
    "u_screen": 0.4,
    "u_ventilation": 0.4,
    "u_lighting": 0.0,
    "u_shading": 0.2,
}

RUNTIME_ACTION = {
    "u_heating": 0.25,
    "u_co2": 0.2,
    "u_screen": 0.35,
    "u_ventilation": 0.42,
    "u_lighting": 0.0,
    "u_shading": 0.2,
}


def _dry_state_history():
    return [
        {"temp_air": 30.0, "rh_air": 48.0, "vpd_air": 1.8, "canopy_dew_margin": 4.0, "dew_margin_air": 4.0},
        {"temp_air": 31.0, "rh_air": 45.0, "vpd_air": 2.1, "canopy_dew_margin": 4.0, "dew_margin_air": 4.0},
        {"temp_air": 32.0, "rh_air": 43.0, "vpd_air": 2.3, "canopy_dew_margin": 4.0, "dew_margin_air": 4.0},
    ]


def _suggestion():
    return SemanticSuggestion(
        regime="HIGH_VPD_DRY_STRESS",
        regime_confidence=0.95,
        llm_reported_confidence=0.95,
        priority_weight_suggestions={"vpd_risk": 0.50, "humidity_recovery": 0.30},
        control_intent="reduce dry stress without abrupt ventilation reversal",
        risk_factors=["vpd_rising", "rh_declining"],
    )


def _post_analytics(next_action="v82_simulator_shadow_rollout_plan"):
    return {
        "next_action": next_action,
        "readiness_reasons": [],
        "boundary_runtime_safety": {
            "audit_success_rate": 1.0,
            "final_action_invariant_rate": 1.0,
            "final_action_changed_steps": 0,
            "online_llm_called_steps": 0,
            "predictive_rollout_executed_steps": 0,
            "real_tomato_safety_projection_steps": 0,
            "runtime_error_steps": 0,
        },
        "max_delta_attribution": {
            "attribution_mode": "candidate_level_attribution",
            "max_delta_count": 0,
        },
        "risk_aligned_behavior": {
            "any_risk": {"hold_all_share": 0.25},
        },
    }


def _pre_analytics():
    return {
        "next_action": "v816_template_constraint_calibration_plan",
        "max_delta_attribution": {
            "attribution_mode": "selected_step_coincidence_attribution",
            "max_delta_count": 220,
        },
        "risk_aligned_behavior": {
            "any_risk": {"hold_all_share": 1.0},
        },
    }


def _row(*, selected_violation=False, risk=True, rule_status="selected", include_bad=False):
    runtime = dict(RUNTIME_ACTION)
    selected_id = "rule_prior:ventilation_ramp_limited:1" if rule_status == "selected" else "ppo_prior:ppo_follow_limited:0"
    bad_id = selected_id if selected_violation else "previous_plan_prior:previous_shift_all:3"
    candidate_violation_provenance = [
        {
            "candidate_id": "rule_prior:ventilation_ramp_limited:1",
            "source_prior": "rule_prior",
            "template_name": "ventilation_ramp_limited",
            "feasible": True,
            "violation_reason_distribution": {},
            "violation_field_distribution": {},
            "first_action_delta_from_reference": {field: 0.0 for field in ACTION_FIELDS},
        },
        {
            "candidate_id": "ppo_prior:hold_all:2",
            "source_prior": "ppo_prior",
            "template_name": "hold_all",
            "feasible": True,
            "violation_reason_distribution": {},
            "violation_field_distribution": {},
            "first_action_delta_from_reference": {field: 0.0 for field in ACTION_FIELDS},
        },
    ]
    if include_bad or selected_violation:
        candidate_violation_provenance.insert(
            1,
            {
                "candidate_id": bad_id,
                "source_prior": "previous_plan_prior" if not selected_violation else "rule_prior",
                "template_name": "previous_shift_all" if not selected_violation else "ventilation_ramp_limited",
                "feasible": False,
                "violation_reason_distribution": {"max_delta": 2},
                "violation_field_distribution": {"u_ventilation": 1, "u_screen": 1},
                "first_action_delta_from_reference": {
                    **{field: 0.0 for field in ACTION_FIELDS},
                    "u_ventilation": 0.6,
                    "u_screen": 0.5,
                },
            },
        )
    return {
        "audit_success": True,
        "candidate_attribution_mode": "candidate_level_attribution",
        "shadow_selected_sequence_id": selected_id,
        "selected_candidate_violation_count": 2 if selected_violation else 0,
        "current_runtime_final_action": runtime,
        "hold_all_reference_kind": "current_runtime_final_action",
        "hold_all_reference": runtime,
        "constraint_reference_kind": "current_runtime_final_action",
        "constraint_reference_action": runtime,
        "candidate_violation_provenance": candidate_violation_provenance,
        "risk_flags": {"dry_risk": risk, "high_vpd": risk, "any_risk": risk},
        "rule_conservative_competitiveness": {
            "rule_prior": {
                "candidate_count": 2,
                "feasible_candidate_count": 1,
                "status": rule_status,
                "selected_score_margin": 0.0 if rule_status == "selected" else 0.12,
            },
            "conservative_prior": {
                "candidate_count": 2,
                "feasible_candidate_count": 1,
                "status": "lower_score",
                "selected_score_margin": 0.08,
            },
        },
    }


class TestCSTCCV816TemplateConstraintCalibration(unittest.TestCase):
    def test_hold_all_uses_current_runtime_final_action_reference(self):
        record = build_audit_record(
            state_history=_dry_state_history(),
            action_history=[LAST_ACTION],
            current_runtime_final_action=RUNTIME_ACTION,
            active_regime="HIGH_VPD_DRY_STRESS",
            regime_age_steps=10,
            semantic_suggestion=_suggestion(),
            ppo_action={**RUNTIME_ACTION, "u_ventilation": 0.8},
            rule_action={**RUNTIME_ACTION, "u_ventilation": 0.7},
        )
        payload = asdict_clean(record)
        metadata = payload["scoring_metadata"]
        hold_candidates = [
            candidate
            for candidate in metadata["candidate_violation_provenance"]
            if candidate["template_name"] == "hold_all"
        ]

        self.assertEqual(metadata["constraint_reference_kind"], "current_runtime_final_action")
        self.assertEqual(metadata["hold_all_reference"], payload["current_runtime_final_action"])
        self.assertTrue(hold_candidates)
        self.assertTrue(
            all(
                abs(candidate["first_action_delta_from_reference"][field]) <= 1e-12
                for candidate in hold_candidates
                for field in ACTION_FIELDS
            )
        )

    def test_previous_shift_projection_removes_candidate_max_delta_and_records_metadata(self):
        previous_sequence = [
            dict(RUNTIME_ACTION),
            {**RUNTIME_ACTION, "u_ventilation": 1.0, "u_screen": 0.9},
        ]
        record = build_audit_record(
            state_history=_dry_state_history(),
            action_history=[LAST_ACTION],
            current_runtime_final_action=RUNTIME_ACTION,
            active_regime="HIGH_VPD_DRY_STRESS",
            regime_age_steps=10,
            semantic_suggestion=_suggestion(),
            previous_plan_sequence=previous_sequence,
            ppo_action={**RUNTIME_ACTION, "u_ventilation": 0.8},
            rule_action={**RUNTIME_ACTION, "u_ventilation": 0.7},
        )
        payload = asdict_clean(record)
        metadata = payload["scoring_metadata"]
        previous_shift_candidates = [
            candidate
            for candidate in metadata["candidate_violation_provenance"]
            if candidate["template_name"] == "previous_shift_all"
        ]

        self.assertTrue(previous_shift_candidates)
        self.assertEqual(metadata["selected_candidate_violation_count"], 0)
        self.assertFalse(
            any(candidate["violation_reason_distribution"].get("max_delta") for candidate in previous_shift_candidates)
        )
        first = previous_shift_candidates[0]
        template_metadata = first["template_metadata"]
        self.assertTrue(template_metadata["previous_shift_projection_applied"])
        self.assertEqual(
            template_metadata["previous_shift_projection_reference_action"],
            payload["current_runtime_final_action"],
        )
        self.assertLessEqual(
            abs(template_metadata["previous_shift_first_delta_after_projection_by_field"]["u_ventilation"]),
            template_metadata["previous_shift_projection_max_delta_by_field"]["u_ventilation"],
        )
        self.assertGreater(
            sum(abs(value) for value in template_metadata["previous_shift_projection_delta_by_field"].values()),
            0.0,
        )

    def test_risk_hold_penalty_and_prior_confidence_term_are_scored_shadow_only(self):
        config_zero = load_config()
        config_zero["scoring"]["prior_confidence_score_weight"] = 0.0
        config_high = load_config()
        config_high["scoring"]["prior_confidence_score_weight"] = 2.0
        ranking_action = dict(LAST_ACTION)
        kwargs = {
            "state_history": _dry_state_history(),
            "action_history": [ranking_action],
            "current_runtime_final_action": ranking_action,
            "active_regime": "HIGH_VPD_DRY_STRESS",
            "regime_age_steps": 10,
            "semantic_suggestion": _suggestion(),
            "previous_plan_sequence": [
                dict(ranking_action),
                {**ranking_action, "u_ventilation": 1.0, "u_screen": 0.9},
            ],
            "ppo_action": {**ranking_action, "u_ventilation": 0.9},
            "rule_action": {**ranking_action, "u_ventilation": 0.7},
        }

        zero = asdict_clean(build_audit_record(**kwargs, config=config_zero))
        high = asdict_clean(build_audit_record(**kwargs, config=config_high))
        hold_scores = [
            row["score_breakdown"]
            for row in high["tomato_safety_dual_scores"]
            if row["template_name"] == "hold_all"
        ]

        self.assertFalse(high["final_action_changed"])
        self.assertEqual(high["current_runtime_final_action"], ranking_action)
        self.assertTrue(any(score["risk_static_hold_penalty_applied"] for score in hold_scores))
        self.assertTrue(
            any(score["final_score"] != score["base_final_score"] for score in hold_scores)
        )

        normal_state = [
            {"temp_air": 24.0, "rh_air": 70.0, "vpd_air": 0.8, "canopy_dew_margin": 4.0, "dew_margin_air": 4.0},
            {"temp_air": 24.2, "rh_air": 69.0, "vpd_air": 0.85, "canopy_dew_margin": 4.0, "dew_margin_air": 4.0},
            {"temp_air": 24.1, "rh_air": 70.0, "vpd_air": 0.8, "canopy_dew_margin": 4.0, "dew_margin_air": 4.0},
        ]
        normal_kwargs = {
            **kwargs,
            "state_history": normal_state,
            "active_regime": "NORMAL_BALANCED",
            "semantic_suggestion": SemanticSuggestion(regime="NORMAL_BALANCED", regime_confidence=0.8),
            "previous_plan_sequence": [dict(ranking_action), dict(ranking_action)],
        }
        normal_zero = asdict_clean(build_audit_record(**normal_kwargs, config=config_zero))
        normal_high = asdict_clean(build_audit_record(**normal_kwargs, config=config_high))
        zero_scores = {
            row["candidate_id"]: row["score_breakdown"]["final_score"]
            for row in normal_zero["tomato_safety_dual_scores"]
        }
        high_scores = {
            row["candidate_id"]: row["score_breakdown"]["final_score"]
            for row in normal_high["tomato_safety_dual_scores"]
        }
        self.assertGreater(
            zero_scores["previous_plan_prior:previous_shift_all:3"],
            zero_scores["ppo_prior:ppo_follow_limited:0"],
        )
        self.assertGreater(
            high_scores["ppo_prior:ppo_follow_limited:0"],
            high_scores["previous_plan_prior:previous_shift_all:3"],
        )

    def test_calibration_report_routes_clean_candidate_level_trace_to_v82(self):
        report, csv_rows = build_calibration_report(
            pre_analytics=_pre_analytics(),
            post_analytics=_post_analytics(),
            post_jsonl_rows=[_row(rule_status="selected", include_bad=False)],
        )

        self.assertEqual(report["next_action"], "v82_simulator_shadow_rollout_plan")
        self.assertFalse(report["boundaries"]["online_llm_called"])
        self.assertFalse(report["boundaries"]["final_action_changed"])
        self.assertEqual(
            report["hold_all_reference_answer"]["hold_all_reference_matches_runtime_rate"],
            1.0,
        )
        self.assertEqual(
            report["candidate_level_max_delta_attribution"]["non_selected_candidate_max_delta_count"],
            0,
        )
        self.assertTrue(csv_rows)

    def test_calibration_routes_selected_violation_to_template_iteration(self):
        report, _csv_rows = build_calibration_report(
            pre_analytics=_pre_analytics(),
            post_analytics=_post_analytics(),
            post_jsonl_rows=[_row(selected_violation=True, rule_status="selected")],
        )

        self.assertEqual(report["next_action"], "v816_template_constraint_calibration_iteration")
        self.assertIn("selected_candidate_has_max_delta_violation", report["readiness_reasons"])

    def test_calibration_routes_previous_shift_max_delta_to_projection_iteration(self):
        report, _csv_rows = build_calibration_report(
            pre_analytics=_pre_analytics(),
            post_analytics=_post_analytics(),
            post_jsonl_rows=[_row(rule_status="selected", include_bad=True)],
        )

        self.assertEqual(report["next_action"], "v816_previous_shift_projection_iteration")
        self.assertIn("previous_shift_all_candidate_has_max_delta_violation", report["readiness_reasons"])
        self.assertEqual(
            report["previous_shift_projection_calibration"]["previous_shift_all_max_delta_count"],
            2,
        )

    def test_calibration_routes_feasible_but_never_selected_rule_to_v817(self):
        report, _csv_rows = build_calibration_report(
            pre_analytics=_pre_analytics(),
            post_analytics=_post_analytics(),
            post_jsonl_rows=[_row(rule_status="lower_score", include_bad=False) for _ in range(3)],
        )

        self.assertEqual(report["next_action"], "v817_level0_scorer_calibration_plan")
        self.assertIn("rule_conservative_feasible_but_never_selected_in_risk_window", report["readiness_reasons"])

    def test_outputs_do_not_claim_online_rollout_or_promotion(self):
        report, csv_rows = build_calibration_report(
            pre_analytics=_pre_analytics(),
            post_analytics=_post_analytics(),
            post_jsonl_rows=[_row(rule_status="selected", include_bad=False)],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = write_outputs(
                report=report,
                csv_rows=csv_rows,
                output_prefix=Path(tmpdir) / "cstcc_v816_template_constraint_calibration_test",
            )

            self.assertTrue(Path(outputs["json"]).exists())
            self.assertTrue(Path(outputs["md"]).exists())
            self.assertTrue(Path(outputs["csv"]).exists())

        self.assertFalse(report["boundaries"]["online_llm_called"])
        self.assertFalse(report["boundaries"]["simulator_rollout_executed"])
        self.assertFalse(report["boundaries"]["promotion_evidence"])
        self.assertFalse(report["boundaries"]["performance_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
