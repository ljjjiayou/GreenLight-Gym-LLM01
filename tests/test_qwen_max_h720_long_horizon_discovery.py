import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.qwen_max_h720_long_horizon_discovery import build_artifacts, build_report


def _write_trace(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "step",
        "runtime_error",
        "runtime_error_type",
        "replan_reason",
        "target_temp",
        "target_rh",
        "target_co2",
        "profile_selected_delta_target_temp",
        "profile_selected_delta_target_rh",
        "profile_selected_delta_target_co2",
        "dew_risk",
        "vpd_high_excess",
        "u_heating",
        "u_ventilation",
        "u_co2",
        "u_lighting",
        "u_screen",
        "u_shading",
        "rspc_action_candidate_count",
        "rspc_action_actual_candidate_count",
        "rspc_action_post_shape_best_eligible",
        "rspc_action_post_shape_safety_gate_reason",
        "rspc_action_post_shape_unsafe_conflict",
        "selected_fallback_candidate",
        "tomato_safety_v2_applied",
        "tomato_safety_v2_heat_before",
        "tomato_safety_v2_heat_after",
        "tomato_safety_v2_vent_before",
        "tomato_safety_v2_vent_after",
        "tomato_safety_v2_screen_before",
        "tomato_safety_v2_screen_after",
        "tomato_safety_v2_shade_before",
        "tomato_safety_v2_shade_after",
        "tomato_safety_v2_reasons",
        "final_action_would_fail_canopy_boundary_v2",
        "final_action_predicted_canopy_lt0_v2",
        "post_guardrail_runtime_provenance_count",
        "post_guardrail_runtime_reason_missing_count",
        "unknown_post_guardrail_rewrite_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            base = {key: "" for key in fieldnames}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


class TestQwenMaxH720LongHorizonDiscovery(unittest.TestCase):
    def test_builds_profile_candidate_and_stability_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            _write_trace(
                trace_dir / "y2010_d59_s42_n3_ppo.csv",
                [
                    {"step": 0},
                    {"step": 1},
                    {"step": 2},
                ],
            )
            _write_trace(
                trace_dir / "y2010_d59_s42_n3_llm_rspc_v2.csv",
                [
                    {
                        "step": 0,
                        "target_temp": 20,
                        "target_rh": 70,
                        "target_co2": 900,
                        "u_heating": 0.1,
                        "u_ventilation": 0.4,
                        "u_co2": 0.0,
                        "u_lighting": 0.0,
                        "u_screen": 0.1,
                        "u_shading": 0.0,
                        "rspc_action_candidate_count": 1,
                        "rspc_action_actual_candidate_count": 1,
                        "post_guardrail_runtime_provenance_count": 1,
                    },
                    {
                        "step": 1,
                        "target_temp": 23.5,
                        "target_rh": 82,
                        "target_co2": 1150,
                        "profile_selected_delta_target_temp": 3.5,
                        "profile_selected_delta_target_rh": 12,
                        "profile_selected_delta_target_co2": 250,
                        "dew_risk": "true",
                        "vpd_high_excess": 1,
                        "u_heating": 0.6,
                        "u_ventilation": 0.0,
                        "u_co2": 0.0,
                        "u_lighting": 0.0,
                        "u_screen": 0.1,
                        "u_shading": 0.0,
                        "rspc_action_candidate_count": 0,
                        "rspc_action_actual_candidate_count": 0,
                        "tomato_safety_v2_applied": "true",
                        "tomato_safety_v2_reasons": "dew_guard",
                        "post_guardrail_runtime_provenance_count": 1,
                    },
                    {
                        "step": 2,
                        "target_temp": 21,
                        "target_rh": 60,
                        "target_co2": 900,
                        "profile_selected_delta_target_temp": -2.5,
                        "profile_selected_delta_target_rh": -22,
                        "profile_selected_delta_target_co2": -250,
                        "u_heating": 0.0,
                        "u_ventilation": 0.5,
                        "u_co2": 0.0,
                        "u_lighting": 0.0,
                        "u_screen": 0.1,
                        "u_shading": 0.0,
                        "rspc_action_candidate_count": 1,
                        "rspc_action_actual_candidate_count": 1,
                        "rspc_action_post_shape_best_eligible": "false",
                        "rspc_action_post_shape_safety_gate_reason": "safety_gate",
                        "selected_fallback_candidate": "safe_hold",
                        "post_guardrail_runtime_provenance_count": 1,
                    },
                ],
            )
            report = build_report(
                execution_record={
                    "horizon_steps": 3,
                    "controllers": ["ppo", "llm_rspc_v2"],
                    "eligible_scenarios": [
                        {"scenario_id": "y2010_d59_s42_n3", "year": 2010, "day": 59, "seed": 42}
                    ],
                    "planned_command_records": [{"controller": "llm_rspc_v2"}],
                },
                trace_dir=trace_dir,
            )

        artifacts = build_artifacts(report)
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertGreater(report["metrics"]["profile_issue_count"], 0)
        self.assertGreater(report["metrics"]["candidate_gap_count"], 0)
        self.assertGreater(report["metrics"]["stability_issue_count"], 0)
        self.assertIn("metadata_replay_readiness_checklist_20260529_v39", artifacts)
        self.assertFalse(artifacts["metadata_replay_readiness_checklist_20260529_v39"]["promotion_evidence"])

    def test_rejects_controlled_controller_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            _write_trace(trace_dir / "y2010_d59_s42_n1_ppo.csv", [{"step": 0}])
            _write_trace(trace_dir / "y2010_d59_s42_n1_llm_rspc_v2.csv", [{"step": 0}])
            report = build_report(
                execution_record={
                    "horizon_steps": 1,
                    "controllers": ["ppo", "llm_rspc_v2_hot_dry_proposer_strict"],
                    "eligible_scenarios": [
                        {"scenario_id": "y2010_d59_s42_n1", "year": 2010, "day": 59, "seed": 42}
                    ],
                    "planned_command_records": [{"controller": "llm_rspc_v2_hot_dry_proposer_strict"}],
                },
                trace_dir=trace_dir,
            )

        self.assertFalse(report["controlled_scope_ok"])
        self.assertIn("controlled_controller_scope_violation", report["failure_taxonomy"])
        self.assertFalse(report["controlled_replay_allowed"])


if __name__ == "__main__":
    unittest.main()
