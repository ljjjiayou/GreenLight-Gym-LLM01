import copy
import tempfile
import unittest
from pathlib import Path

from gl_gym.cstcc.audit_writer import (
    append_audit_record,
    compact_audit_record,
    default_audit_jsonl_path,
    parse_selected_sequence_id,
    read_jsonl,
    should_sample_step,
    summarize_shadow_rows,
)
from gl_gym.cstcc.config import load_config
from gl_gym.cstcc.contracts import CSTCCAuditRecord, SemanticSuggestion, asdict_clean
from gl_gym.cstcc.runtime_shadow import (
    run_cstcc_shadow_step,
    safe_run_cstcc_shadow_step,
    verify_final_action_invariant,
)


LAST_ACTION = {
    "u_heating": 0.2,
    "u_co2": 0.2,
    "u_screen": 0.4,
    "u_ventilation": 0.4,
    "u_lighting": 0.0,
    "u_shading": 0.2,
}


def _state_history():
    return [
        {"temp_air": 29.0, "rh_air": 50.0, "vpd_air": 2.1},
        {"temp_air": 30.0, "rh_air": 48.0, "vpd_air": 2.4},
        {"temp_air": 31.0, "rh_air": 46.0, "vpd_air": 2.7},
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


class TestCSTCCV81RuntimeShadow(unittest.TestCase):
    def test_shadow_step_returns_audit_only_and_preserves_final_action(self):
        runtime_final_action = dict(LAST_ACTION)

        audit = run_cstcc_shadow_step(
            state_history=_state_history(),
            action_history=[LAST_ACTION, {**LAST_ACTION, "u_ventilation": 0.45}],
            weather_history=[{"radiation": 500.0}],
            current_runtime_final_action=runtime_final_action,
            active_regime="NORMAL_BALANCED",
            regime_age_steps=10,
            semantic_suggestion=_suggestion(),
            ppo_action={**LAST_ACTION, "u_ventilation": 0.50},
        )
        payload = asdict_clean(audit)

        self.assertIsInstance(audit, CSTCCAuditRecord)
        self.assertEqual(runtime_final_action, LAST_ACTION)
        self.assertFalse(payload["final_action_changed"])
        self.assertTrue(payload["final_action_invariant_verified"])
        self.assertTrue(payload["selected_action_is_shadow_only"])
        self.assertEqual(payload["scoring_metadata"]["runtime_shadow_version"], "v81")
        self.assertEqual(payload["prediction_level"], 0)
        self.assertFalse(payload["online_llm_called"])
        self.assertFalse(payload["scoring_metadata"]["real_tomato_safety_projection"])

    def test_final_action_invariant_helper_detects_diff(self):
        ok, diff = verify_final_action_invariant(LAST_ACTION, {**LAST_ACTION, "u_ventilation": 0.5})

        self.assertFalse(ok)
        self.assertAlmostEqual(diff["u_ventilation"], 0.1)

    def test_safe_shadow_step_isolates_audit_failure(self):
        bad_config = copy.deepcopy(load_config())
        bad_config["online_llm_enabled"] = True

        audit, failure = safe_run_cstcc_shadow_step(
            state_history=_state_history(),
            action_history=[LAST_ACTION],
            current_runtime_final_action=LAST_ACTION,
            active_regime="NORMAL_BALANCED",
            regime_age_steps=0,
            semantic_suggestion=_suggestion(),
            config=bad_config,
        )

        self.assertIsNone(audit)
        self.assertIsNotNone(failure)
        self.assertFalse(failure["audit_success"])
        self.assertFalse(failure["final_action_changed"])
        self.assertFalse(failure["online_llm_called"])

    def test_compact_writer_outputs_jsonl_without_raw_sequences_by_default(self):
        audit = run_cstcc_shadow_step(
            state_history=_state_history(),
            action_history=[LAST_ACTION, {**LAST_ACTION, "u_ventilation": 0.45}],
            current_runtime_final_action=LAST_ACTION,
            active_regime="NORMAL_BALANCED",
            regime_age_steps=10,
            semantic_suggestion=_suggestion(),
        )

        compact = compact_audit_record(audit, step=7, episode_id="episode/a", save_full_candidates=False)

        self.assertEqual(compact["step"], 7)
        self.assertEqual(compact["episode_id"], "episode/a")
        self.assertTrue(compact["audit_success"])
        self.assertNotIn("sequence_candidates", compact)
        self.assertEqual(compact["score_validity"], "sequence_only_no_rollout")
        self.assertEqual(compact["projection_level"], "placeholder_no_real_projection")
        self.assertIn("shadow_action_difference_from_runtime", compact)
        self.assertIn("shadow_selected_source_prior", compact)
        self.assertIn("shadow_selected_template_name", compact)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "v81_episode_test.jsonl"
            append_audit_record(path, audit, step=7, episode_id="episode/a")
            rows = read_jsonl(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 7)
        self.assertTrue(rows[0]["final_action_invariant_verified"])

    def test_selected_sequence_parsing_and_violation_field_distribution(self):
        source_prior, template_name = parse_selected_sequence_id("ppo_prior:ppo_follow_limited:0")

        self.assertEqual(source_prior, "ppo_prior")
        self.assertEqual(template_name, "ppo_follow_limited")

        compact = compact_audit_record(
            {
                "version": "v81",
                "prediction_level": 0,
                "online_llm_called": False,
                "final_action_changed": False,
                "final_action_invariant_verified": True,
                "selected_action_is_shadow_only": True,
                "active_regime_before": "NORMAL_BALANCED",
                "active_regime_after": "HIGH_VPD_DRY_STRESS",
                "shadow_selected_sequence_id": "ppo_prior:ppo_follow_limited:0",
                "shadow_selected_first_action": LAST_ACTION,
                "current_runtime_final_action": LAST_ACTION,
                "shadow_action_difference_from_runtime": {field: 0.0 for field in LAST_ACTION},
                "sequence_candidates": [
                    {"candidate_id": "ppo_prior:ppo_follow_limited:0", "feasible": False},
                ],
                "hard_constraint_violations": [
                    {"field": "u_co2/u_ventilation", "reason": "co2_injection_with_high_ventilation"},
                    {"field": "u_co2/u_ventilation", "reason": "co2_injection_with_high_ventilation"},
                    {"field": "u_screen", "reason": "max_delta"},
                ],
                "scoring_metadata": {
                    "score_validity": "sequence_only_no_rollout",
                    "real_tomato_safety_projection": False,
                    "tomato_safety_projection_level": "placeholder_no_real_projection",
                },
            },
            step=3,
            episode_id="e1",
        )
        summary = summarize_shadow_rows([compact])

        self.assertEqual(compact["shadow_selected_source_prior"], "ppo_prior")
        self.assertEqual(compact["shadow_selected_template_name"], "ppo_follow_limited")
        self.assertEqual(compact["hard_constraint_violation_field_distribution"]["u_co2/u_ventilation"], 2)
        self.assertEqual(compact["hard_constraint_violation_reason_distribution"]["max_delta"], 1)
        self.assertEqual(summary["selected_source_prior_distribution"]["ppo_prior"], 1)
        self.assertEqual(summary["selected_template_name_distribution"]["ppo_follow_limited"], 1)
        self.assertEqual(summary["hard_constraint_violation_field_distribution"]["u_co2/u_ventilation"], 2)

    def test_compact_writer_can_include_candidates_without_sequences(self):
        audit = run_cstcc_shadow_step(
            state_history=_state_history(),
            action_history=[LAST_ACTION],
            current_runtime_final_action=LAST_ACTION,
            active_regime="NORMAL_BALANCED",
            regime_age_steps=10,
            semantic_suggestion=_suggestion(),
        )

        compact = compact_audit_record(audit, save_full_candidates=True, save_raw_sequences=False, save_projected_sequences=False)

        self.assertTrue(compact["sequence_candidates"])
        first = compact["sequence_candidates"][0]
        self.assertNotIn("raw_sequence", first)
        self.assertNotIn("projected_sequence", first)

    def test_sampling_default_path_and_summary_metrics(self):
        self.assertTrue(should_sample_step(4, 1.0))
        self.assertFalse(should_sample_step(1, 0.5))
        self.assertTrue(should_sample_step(2, 0.5))
        path = default_audit_jsonl_path("episode/a b")
        self.assertEqual(path.name, "v81_episode_episode_a_b.jsonl")

        audit = run_cstcc_shadow_step(
            state_history=_state_history(),
            action_history=[LAST_ACTION],
            current_runtime_final_action=LAST_ACTION,
            active_regime="NORMAL_BALANCED",
            regime_age_steps=10,
            semantic_suggestion=_suggestion(),
        )
        row = compact_audit_record(audit, step=1, episode_id="e1")
        summary = summarize_shadow_rows([row])

        self.assertEqual(summary["row_count"], 1)
        self.assertEqual(summary["audit_success_rate"], 1.0)
        self.assertEqual(summary["final_action_invariant_rate"], 1.0)
        self.assertIn("u_ventilation", summary["mean_abs_action_diff_by_field"])
        self.assertIn("HIGH_VPD_DRY_STRESS", summary["active_regime_distribution"])


if __name__ == "__main__":
    unittest.main()
