import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector
from gl_gym.cstcc.audit_writer import default_audit_jsonl_path, read_jsonl
from gl_gym.experiments.cstcc_v811_smoke_acquisition import build_smoke_summary
from gl_gym.experiments.diagnose_ppo_vs_llm import cstcc_shadow_to_record, sum_metrics


def _director(config: AgentConfig) -> RuleBasedLLMDirector:
    director = RuleBasedLLMDirector.__new__(RuleBasedLLMDirector)
    director.config = config
    director.env_id = "TomatoEnv_test"
    director.last_rollout_selection = {}
    director.last_cstcc_shadow = {}
    return director


def _state(step: int = 12) -> SimpleNamespace:
    return SimpleNamespace(
        timestep=step,
        day_of_year=180,
        hour_of_day=13.5,
        temp_air=30.5,
        rh_air=48.0,
        co2_air=620.0,
        glob_rad=650.0,
        pipe_temp=32.0,
        canopy_dew_margin=3.2,
        dew_margin_air=3.0,
    )


class TestCSTCCV811RuntimeTraceIntegration(unittest.TestCase):
    def test_default_flag_is_closed_and_does_not_emit_rollout_shadow(self):
        director = _director(AgentConfig())
        final_control = np.asarray([0.2, 0.1, 0.3, 0.4, 0.0, 0.2], dtype=np.float32)
        before = final_control.copy()

        director._run_cstcc_shadow_runtime_hook(_state(), final_control)

        np.testing.assert_allclose(final_control, before)
        self.assertFalse(director.last_cstcc_shadow["enabled"])
        self.assertNotIn("cstcc_shadow", director.last_rollout_selection)

    def test_opt_in_hook_writes_compact_jsonl_and_preserves_final_action(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AgentConfig(
                cstcc_shadow_enabled=True,
                cstcc_shadow_audit_log_root=tmpdir,
                cstcc_shadow_sample_rate=1.0,
                cstcc_shadow_save_full_candidates=False,
            )
            director = _director(cfg)
            final_control = np.asarray([0.2, 0.1, 0.3, 0.4, 0.0, 0.2], dtype=np.float32)
            before = final_control.copy()

            director._run_cstcc_shadow_runtime_hook(
                _state(),
                final_control,
                rule_control=np.asarray([0.2, 0.0, 0.3, 0.35, 0.0, 0.2], dtype=np.float32),
                anchor_control=np.asarray([0.25, 0.1, 0.3, 0.45, 0.0, 0.2], dtype=np.float32),
            )

            np.testing.assert_allclose(final_control, before)
            shadow = director.last_rollout_selection["cstcc_shadow"]
            self.assertTrue(shadow["enabled"])
            self.assertTrue(shadow["audit_success"])
            self.assertTrue(shadow["final_action_invariant_verified"])
            self.assertFalse(shadow["final_action_changed"])
            self.assertFalse(shadow["online_llm_called"])
            self.assertFalse(shadow["predictive_rollout_executed"])
            self.assertFalse(shadow["real_tomato_safety_projection"])
            self.assertTrue(shadow["selected_action_is_shadow_only"])
            self.assertEqual(shadow["prediction_level"], 0)
            self.assertEqual(shadow["score_validity"], "sequence_only_no_rollout")
            self.assertIn("shadow_action_difference_from_runtime", shadow)
            self.assertNotIn("sequence_candidates", shadow)

            path = default_audit_jsonl_path("TomatoEnv_test_day180", root=tmpdir)
            rows = read_jsonl(path)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["audit_success"])
            self.assertEqual(rows[0]["runtime_shadow_version"], "v81")

    def test_audit_failure_is_logged_without_interrupting_when_fail_open(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AgentConfig(cstcc_shadow_enabled=True, cstcc_shadow_audit_log_root=tmpdir)
            director = _director(cfg)
            final_control = np.asarray([0.2, 0.1, 0.3, 0.4, 0.0, 0.2], dtype=np.float32)
            before = final_control.copy()
            failure = {
                "version": "v81",
                "audit_success": False,
                "failure_type": "SyntheticAuditFailure",
                "failure_message": "synthetic failure",
                "final_action_changed": False,
                "final_action_invariant_verified": False,
                "online_llm_called": False,
                "predictive_rollout_executed": False,
                "selected_action_is_shadow_only": True,
            }

            with mock.patch("gl_gym.agent.llm_agent.safe_run_cstcc_shadow_step", return_value=(None, failure)):
                director._run_cstcc_shadow_runtime_hook(_state(step=13), final_control)

            np.testing.assert_allclose(final_control, before)
            shadow = director.last_rollout_selection["cstcc_shadow"]
            self.assertFalse(shadow["audit_success"])
            self.assertEqual(shadow["failure_type"], "SyntheticAuditFailure")
            self.assertFalse(shadow["final_action_changed"])

    def test_trace_flattening_and_summary_include_v811_metrics(self):
        shadow = {
            "enabled": True,
            "shadow_only": True,
            "audit_success": True,
            "final_action_changed": False,
            "final_action_invariant_verified": True,
            "online_llm_called": False,
            "predictive_rollout_executed": False,
            "real_tomato_safety_projection": False,
            "selected_action_is_shadow_only": True,
            "prediction_level": 0,
            "score_validity": "sequence_only_no_rollout",
            "projection_level": "placeholder_no_real_projection",
            "candidate_count": 4,
            "feasible_candidate_count": 3,
            "infeasible_candidate_ratio": 0.25,
            "hard_constraint_violation_count": 3,
            "hard_constraint_violation_reason_distribution": {
                "co2_injection_with_high_ventilation": 2,
                "max_delta": 1,
            },
            "hard_constraint_violation_field_distribution": {
                "u_co2/u_ventilation": 2,
                "u_screen": 1,
            },
            "previous_shift_all_candidate_count": 1,
            "previous_shift_all_max_delta_count": 0,
            "previous_shift_all_selected_count": 0,
            "previous_shift_projection_applied_count": 1,
            "previous_shift_first_delta_after_projection_max": 0.2,
            "previous_shift_projection_mean_abs_delta_by_field": {
                "u_heating": 0.0,
                "u_co2": 0.0,
                "u_screen": 0.1,
                "u_ventilation": 0.2,
                "u_lighting": 0.0,
                "u_shading": 0.0,
            },
            "previous_shift_internal_max_delta_after_projection_by_field": {
                "u_heating": 0.0,
                "u_co2": 0.0,
                "u_screen": 0.2,
                "u_ventilation": 0.2,
                "u_lighting": 0.0,
                "u_shading": 0.0,
            },
            "fallback_triggered": False,
            "active_regime_before": "NORMAL_BALANCED",
            "active_regime_after": "HIGH_VPD_DRY_STRESS",
            "evidence_confidence": 0.8,
            "effective_confidence": 0.6,
            "llm_influence_alpha": 0.2,
            "audit_latency_ms": 3.5,
            "audit_json_size_kb": 2.0,
            "shadow_selected_sequence_id": "ppo_prior:ppo_follow_limited:0",
            "shadow_action_difference_from_runtime": {
                "u_heating": 0.1,
                "u_co2": 0.0,
                "u_screen": 0.0,
                "u_ventilation": -0.2,
                "u_lighting": 0.0,
                "u_shading": 0.0,
            },
        }

        record = cstcc_shadow_to_record({"rollout_selection": {"cstcc_shadow": shadow}})
        self.assertTrue(record["cstcc_shadow_enabled"])
        self.assertTrue(record["cstcc_shadow_final_action_invariant_verified"])
        self.assertFalse(record["cstcc_shadow_online_llm_called"])
        self.assertIn("u_ventilation", record["cstcc_shadow_action_diff_json"])
        self.assertEqual(record["cstcc_shadow_selected_source_prior"], "ppo_prior")
        self.assertEqual(record["cstcc_shadow_selected_template_name"], "ppo_follow_limited")
        self.assertIn("u_co2/u_ventilation", record["cstcc_shadow_hard_constraint_violation_field_distribution_json"])
        self.assertEqual(record["cstcc_shadow_previous_shift_all_candidate_count"], 1)
        self.assertEqual(record["cstcc_shadow_previous_shift_all_max_delta_count"], 0)
        self.assertIn(
            "u_ventilation",
            record["cstcc_shadow_previous_shift_projection_mean_abs_delta_by_field_json"],
        )

        row = {
            "reward": 0.0,
            "u_heating": 0.2,
            "u_co2": 0.1,
            "u_screen": 0.3,
            "u_ventilation": 0.4,
            "u_lighting": 0.0,
            "u_shading": 0.2,
            "rh_air": 48.0,
            "temp_air": 30.5,
            "vpd_air": 2.4,
            "dew_margin_air": 3.0,
            "canopy_dew_margin": 3.2,
            **record,
        }
        summary = sum_metrics([row])

        self.assertEqual(summary["cstcc_shadow_enabled_steps"], 1)
        self.assertEqual(summary["cstcc_shadow_audit_success_rate"], 1.0)
        self.assertEqual(summary["cstcc_shadow_final_action_invariant_rate"], 1.0)
        self.assertEqual(summary["cstcc_shadow_online_llm_called_steps"], 0)
        self.assertEqual(summary["cstcc_shadow_predictive_rollout_executed_steps"], 0)
        self.assertEqual(
            summary["cstcc_shadow_hard_constraint_violation_reason_distribution"]["co2_injection_with_high_ventilation"],
            2,
        )
        self.assertEqual(
            summary["cstcc_shadow_hard_constraint_violation_field_distribution"]["u_co2/u_ventilation"],
            2,
        )
        self.assertEqual(summary["cstcc_shadow_selected_source_prior_distribution"]["ppo_prior"], 1)
        self.assertEqual(summary["cstcc_shadow_selected_template_name_distribution"]["ppo_follow_limited"], 1)
        self.assertEqual(summary["cstcc_shadow_previous_shift_all_candidate_count"], 1)
        self.assertEqual(summary["cstcc_shadow_previous_shift_all_max_delta_count"], 0)
        self.assertEqual(summary["cstcc_shadow_previous_shift_projection_applied_count"], 1)
        self.assertAlmostEqual(
            summary["cstcc_shadow_previous_shift_projection_mean_abs_delta_by_field"]["u_ventilation"],
            0.2,
        )
        self.assertAlmostEqual(
            summary["cstcc_shadow_mean_abs_action_diff_by_field"]["u_ventilation"],
            0.2,
        )

    def test_full_episode_summary_routes_clean_trace_to_v815(self):
        shadow = {
            "enabled": True,
            "shadow_only": True,
            "audit_success": True,
            "final_action_changed": False,
            "final_action_invariant_verified": True,
            "online_llm_called": False,
            "predictive_rollout_executed": False,
            "real_tomato_safety_projection": False,
            "selected_action_is_shadow_only": True,
            "prediction_level": 0,
            "score_validity": "sequence_only_no_rollout",
            "projection_level": "placeholder_no_real_projection",
            "candidate_count": 4,
            "feasible_candidate_count": 4,
            "infeasible_candidate_ratio": 0.0,
            "hard_constraint_violation_count": 0,
            "hard_constraint_violation_reason_distribution": {},
            "hard_constraint_violation_field_distribution": {},
            "fallback_triggered": False,
            "active_regime_before": "NORMAL_BALANCED",
            "active_regime_after": "NORMAL_BALANCED",
            "evidence_confidence": 0.8,
            "effective_confidence": 0.6,
            "llm_influence_alpha": 0.0,
            "audit_latency_ms": 2.0,
            "audit_json_size_kb": 2.0,
            "shadow_selected_sequence_id": "rule_prior:hold_steady:0",
            "shadow_action_difference_from_runtime": {
                "u_heating": 0.0,
                "u_co2": 0.0,
                "u_screen": 0.0,
                "u_ventilation": 0.0,
                "u_lighting": 0.0,
                "u_shading": 0.0,
            },
        }
        record = cstcc_shadow_to_record({"rollout_selection": {"cstcc_shadow": shadow}})
        base_row = {
            "reward": 0.0,
            "u_heating": 0.2,
            "u_co2": 0.1,
            "u_screen": 0.3,
            "u_ventilation": 0.4,
            "u_lighting": 0.0,
            "u_shading": 0.2,
            **record,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "v81_episode_test.jsonl"
            jsonl_row = {
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
                "shadow_selected_sequence_id": "rule_prior:hold_steady:0",
                "shadow_selected_source_prior": "rule_prior",
                "shadow_selected_template_name": "hold_steady",
                "shadow_action_difference_from_runtime": shadow["shadow_action_difference_from_runtime"],
                "active_regime_after": "NORMAL_BALANCED",
                "projection_level": "placeholder_no_real_projection",
                "score_validity": "sequence_only_no_rollout",
                "prior_confidence_report": {},
            }
            rows = []
            with jsonl_path.open("w", encoding="utf-8") as handle:
                for step in range(720):
                    rows.append({**base_row, "step": step})
                    handle.write(json.dumps({**jsonl_row, "step": step}) + "\n")

            summary = build_smoke_summary(
                rows=rows,
                metadata={"online_llm_called": False},
                year=2020,
                day=240,
                seed=42,
                max_steps=720,
                log_root=tmpdir,
                stage_label="full_episode_trace",
            )

        self.assertEqual(summary["next_action"], "v815_shadow_analytics")
        self.assertFalse(summary["blockers"])
        self.assertEqual(summary["answers"]["selected_source_prior_distribution"]["rule_prior"], 720)
        self.assertEqual(summary["answers"]["selected_template_name_distribution"]["hold_steady"], 720)


if __name__ == "__main__":
    unittest.main()
