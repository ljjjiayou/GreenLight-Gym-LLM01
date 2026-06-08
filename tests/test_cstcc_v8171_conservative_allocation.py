import copy
import unittest

from gl_gym.cstcc.audit import build_audit_record
from gl_gym.cstcc.audit_writer import compact_audit_record
from gl_gym.cstcc.config import load_config
from gl_gym.cstcc.contracts import SemanticSuggestion, asdict_clean
from gl_gym.experiments.cstcc_v817_level0_scorer_diagnostic import build_report
from gl_gym.experiments.diagnose_ppo_vs_llm import cstcc_shadow_to_record, sum_metrics


RUNTIME_ACTION = {
    "u_heating": 0.24,
    "u_co2": 0.20,
    "u_screen": 0.32,
    "u_ventilation": 0.40,
    "u_lighting": 0.00,
    "u_shading": 0.20,
}


def _v8171_config():
    return load_config("gl_gym/cstcc/config/cstcc_v8171_conservative_allocation.yaml")


def _risk_state_history():
    return [
        {"temp_air": 24.0, "rh_air": 54.0, "vpd_air": 0.70, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
        {"temp_air": 24.0, "rh_air": 53.0, "vpd_air": 0.75, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
        {"temp_air": 24.1, "rh_air": 52.0, "vpd_air": 0.80, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
    ]


def _normal_state_history():
    return [
        {"temp_air": 23.0, "rh_air": 70.0, "vpd_air": 0.70, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
        {"temp_air": 23.0, "rh_air": 70.0, "vpd_air": 0.75, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
        {"temp_air": 23.1, "rh_air": 70.0, "vpd_air": 0.80, "canopy_dew_margin": 5.0, "dew_margin_air": 5.0},
    ]


def _suggestion(regime="HIGH_VPD_DRY_STRESS"):
    return SemanticSuggestion(
        regime=regime,
        regime_confidence=0.95,
        llm_reported_confidence=0.95,
        priority_weight_suggestions={"vpd_risk": 0.45, "humidity_recovery": 0.25},
        control_intent="shadow-only conservative allocation diagnostic",
        risk_factors=["dry_risk"],
    )


def _record(config, *, risk=True):
    return build_audit_record(
        state_history=_risk_state_history() if risk else _normal_state_history(),
        action_history=[RUNTIME_ACTION],
        current_runtime_final_action=copy.deepcopy(RUNTIME_ACTION),
        active_regime="HIGH_VPD_DRY_STRESS" if risk else "NORMAL_BALANCED",
        regime_age_steps=8,
        semantic_suggestion=_suggestion() if risk else SemanticSuggestion(regime="NORMAL_BALANCED", regime_confidence=0.8),
        previous_plan_sequence=[dict(RUNTIME_ACTION), {**RUNTIME_ACTION, "u_ventilation": 0.45}],
        ppo_action={**RUNTIME_ACTION, "u_ventilation": 0.55},
        rule_action={**RUNTIME_ACTION, "u_ventilation": 0.50},
        config=config,
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


class TestCSTCCV8171ConservativeAllocation(unittest.TestCase):
    def test_risk_window_conservative_floor_allocates_one_candidate(self):
        payload = asdict_clean(_record(_v8171_config(), risk=True))
        conservative = payload["scoring_metadata"]["prior_generation_diagnostics"]["conservative_prior"]

        self.assertLess(conservative["confidence"], conservative["first_candidate_threshold"])
        self.assertEqual(conservative["candidate_count_from_confidence"], 0)
        self.assertEqual(conservative["candidate_count_before_override"], 0)
        self.assertEqual(conservative["candidate_count_after_override"], 1)
        self.assertEqual(conservative["candidate_count"], 1)
        self.assertTrue(conservative["candidate_count_override_applied"])
        self.assertEqual(conservative["candidate_count_override_reason"], "risk_triggered_conservative_floor")
        self.assertEqual(conservative["generated_template_count"], 1)
        self.assertEqual(conservative["generated_templates"], ["conservative_stabilize"])
        self.assertIsNone(conservative["no_candidate_reason"])

    def test_normal_window_does_not_trigger_conservative_floor(self):
        payload = asdict_clean(_record(_v8171_config(), risk=False))
        conservative = payload["scoring_metadata"]["prior_generation_diagnostics"]["conservative_prior"]

        self.assertEqual(conservative["candidate_count_before_override"], 0)
        self.assertEqual(conservative["candidate_count_after_override"], 0)
        self.assertFalse(conservative["candidate_count_override_applied"])
        self.assertEqual(conservative["generated_template_count"], 0)
        self.assertEqual(conservative["no_candidate_reason"], "confidence_below_candidate_threshold")

    def test_floor_preserves_confidence_and_final_action_invariant(self):
        config = _v8171_config()
        payload = asdict_clean(_record(config, risk=True))
        conservative_report = payload["prior_confidence_report"]["conservative_prior"]
        conservative_diag = payload["scoring_metadata"]["prior_generation_diagnostics"]["conservative_prior"]

        self.assertEqual(conservative_report["confidence"], conservative_diag["confidence"])
        self.assertEqual(conservative_report["confidence"], 0.05)
        self.assertFalse(payload["final_action_changed"])
        self.assertEqual(payload["current_runtime_final_action"], RUNTIME_ACTION)

    def test_compact_jsonl_and_csv_flattening_include_override_metadata(self):
        row = compact_audit_record(_record(_v8171_config(), risk=True), step=1, episode_id="v8171")
        flat = cstcc_shadow_to_record({"rollout_selection": {"cstcc_shadow": row}})
        summary = sum_metrics([{"reward": 0.0, **flat}])

        self.assertTrue(row["conservative_prior_candidate_count_override_applied"])
        self.assertEqual(row["conservative_prior_candidate_count_override_reason"], "risk_triggered_conservative_floor")
        self.assertEqual(flat["cstcc_shadow_conservative_prior_candidate_count_before_override"], 0)
        self.assertEqual(flat["cstcc_shadow_conservative_prior_candidate_count_after_override"], 1)
        self.assertTrue(flat["cstcc_shadow_conservative_prior_candidate_count_override_applied"])
        self.assertEqual(summary["cstcc_shadow_conservative_risk_candidate_steps"], 1)
        self.assertEqual(summary["cstcc_shadow_conservative_risk_feasible_steps"], 1)
        self.assertEqual(summary["cstcc_shadow_conservative_risk_no_candidate_count"], 0)

    def test_conservative_margin_diagnostic_does_not_change_runtime_final_action(self):
        payload = asdict_clean(_record(_v8171_config(), risk=True))
        margin = payload["scoring_metadata"]["score_margin_to_selected_by_prior"]["conservative_prior"]

        self.assertIn(margin["status"], {"selected", "lower_score"})
        self.assertIsNotNone(margin["total_margin"])
        self.assertEqual(payload["current_runtime_final_action"], RUNTIME_ACTION)
        self.assertFalse(payload["final_action_changed"])

    def test_compact_scan_omits_per_candidate_component_payload(self):
        config = _v8171_config()
        config["diagnostics"]["scorer_diagnostic_log_mode"] = "compact_scan"
        row = compact_audit_record(_record(config, risk=True), step=1, episode_id="v8171")
        flat = cstcc_shadow_to_record({"rollout_selection": {"cstcc_shadow": row}})

        self.assertEqual(row["scorer_diagnostic_log_mode"], "compact_scan")
        self.assertEqual(row["score_component_by_candidate"], {})
        self.assertIn("conservative_prior", row["score_margin_to_selected_by_prior"])
        self.assertEqual(flat["cstcc_shadow_score_component_by_candidate_json"], "{}")

    def test_v8171_report_routes_after_successful_allocation(self):
        row = compact_audit_record(_record(_v8171_config(), risk=True), step=1, episode_id="v8171")
        report, _tables = build_report(
            trace_csv_rows=[
                {"step": 1, "cstcc_shadow_enabled": True, "cstcc_shadow_risk_flags_json": "{\"any_risk\": true}"}
            ],
            summary_json=_summary_json(),
            jsonl_rows=[row],
        )

        self.assertEqual(report["scorer_diagnostics"]["conservative_risk_candidate_steps"], 1)
        self.assertEqual(report["scorer_diagnostics"]["conservative_risk_feasible_steps"], 1)
        self.assertEqual(report["scorer_diagnostics"]["conservative_risk_no_candidate_count"], 0)
        self.assertIn(
            report["next_action"],
            {
                "v8172_risk_window_tiebreaker_plan",
                "v8172_risk_aligned_bonus_calibration_plan",
                "multi_scenario_v817_scan",
            },
        )

    def test_boundaries_remain_false(self):
        payload = asdict_clean(_record(_v8171_config(), risk=True))
        metadata = payload["scoring_metadata"]

        self.assertFalse(metadata["predictive_rollout_executed"])
        self.assertFalse(metadata["real_tomato_safety_projection"])
        self.assertFalse(metadata["final_action_changed"])
        self.assertTrue(metadata["selected_action_is_shadow_only"])
        self.assertEqual(metadata["prediction_level"], 0)
        self.assertFalse(payload["fallback_triggered"])


if __name__ == "__main__":
    unittest.main()
