import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.h720_failure_window_attribution import (
    build_guard_design,
    build_readiness,
    build_report,
)


FIELDS = [
    "step",
    "runtime_error",
    "runtime_error_type",
    "replan_reason",
    "temp_air",
    "rh_air",
    "vpd_air",
    "canopy_dew_margin",
    "dew_margin_air",
    "dry_risk",
    "dew_risk",
    "target_temp",
    "target_rh",
    "target_co2",
    "profile_selected_delta_target_rh",
    "vpd_high_excess",
    "u_heating",
    "u_ventilation",
    "u_co2",
    "u_lighting",
    "u_screen",
    "u_shading",
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
    "post_guardrail_runtime_provenance_count",
    "post_guardrail_runtime_reason_missing_count",
    "unknown_post_guardrail_rewrite_count",
    "rspc_action_candidate_count",
    "rspc_action_post_shape_best_eligible",
    "rspc_action_post_shape_safety_gate_reason",
    "rspc_action_selected_name",
    "selected_fallback_candidate",
    "final_action_would_fail_canopy_boundary_v2",
    "final_action_predicted_canopy_lt0_v2",
]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {field: "" for field in FIELDS}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


def _rows(count: int, *, cvodes_step: int | None = None, unstable: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(count):
        vent = 0.9 if unstable and step % 2 == 0 else 0.2 if unstable else 0.3
        screen = 0.5 if unstable else 0.1
        row = {
            "step": step,
            "temp_air": 30 + step * 0.1,
            "rh_air": 50,
            "vpd_air": 2.0 + step * 0.02,
            "canopy_dew_margin": 4.0,
            "dew_margin_air": 10.0,
            "dry_risk": "true",
            "target_temp": 20,
            "target_rh": 70,
            "target_co2": 400,
            "u_heating": 0.0,
            "u_ventilation": vent,
            "u_co2": 0,
            "u_lighting": 0,
            "u_screen": screen,
            "u_shading": 0.9 if unstable else 0.0,
            "rspc_action_candidate_count": 5,
            "rspc_action_post_shape_best_eligible": "true",
            "rspc_action_post_shape_safety_gate_reason": "",
            "rspc_action_selected_name": "smooth",
            "post_guardrail_runtime_provenance_count": 1,
        }
        if unstable:
            row.update(
                {
                    "tomato_safety_v2_applied": "true",
                    "tomato_safety_v2_vent_before": 0.2,
                    "tomato_safety_v2_vent_after": vent,
                    "tomato_safety_v2_screen_before": 0.1,
                    "tomato_safety_v2_screen_after": screen,
                    "tomato_safety_v2_reasons": "hot_temperature_override,hard_dry_vpd_risk",
                }
            )
        if cvodes_step is not None and step == cvodes_step:
            row.update(
                {
                    "runtime_error": 'CVode returned "CV_CONV_FAILURE".',
                    "runtime_error_type": "simulator_or_controller_error",
                    "replan_reason": "runtime_error",
                }
            )
        rows.append(row)
    return rows


class TestH720FailureWindowAttribution(unittest.TestCase):
    def test_fixed_failure_window_counts_action_and_guardrail_pressure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "y2010_d180_s43_n720_ppo.csv", _rows(80))
            _write(root / "y2010_d180_s43_n720_llm_rspc_v2.csv", _rows(80, cvodes_step=79, unstable=True))

            report = build_report(
                trace_dir=root,
                v39_readiness={"schema_version": "metadata_replay_readiness_checklist_20260529_v39"},
                failure_scenarios=["y2010_d180_s43_n720"],
                window_steps=60,
            )

        scenario = report["scenario_reports"][0]
        self.assertEqual(scenario["window_start_step"], 20)
        self.assertEqual(scenario["window_end_step"], 79)
        self.assertEqual(scenario["llm_window_row_count"], 60)
        self.assertEqual(scenario["ppo_window_row_count"], 60)
        self.assertGreater(scenario["action_instability"]["action_oscillation_count"], 0)
        self.assertGreater(scenario["guardrail_rewrite_pressure"]["guardrail_rewrite_count"], 0)
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])

    def test_guard_design_and_readiness_never_authorize_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "y2010_d180_s43_n720_ppo.csv", _rows(10))
            _write(root / "y2010_d180_s43_n720_llm_rspc_v2.csv", _rows(10, cvodes_step=9, unstable=True))
            report = build_report(trace_dir=root, failure_scenarios=["y2010_d180_s43_n720"], window_steps=5)
            guard = build_guard_design(report)
            readiness = build_readiness(report, guard)

        self.assertFalse(guard["controlled_replay_allowed"])
        self.assertFalse(guard["performance_claim_allowed"])
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["promotion_evidence"])
        self.assertIn(readiness["next_action"], {"shadow_rate_guard_minimal_patch_plan", "additional_runtime_instrumentation_before_fix"})


if __name__ == "__main__":
    unittest.main()
