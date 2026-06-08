import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments import qwen37plus_structured_bridge_runtime_failure_attribution_v65 as v65


FIELDS = [
    "step",
    "runtime_error",
    "runtime_error_type",
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
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
    "tomato_safety_v2_applied",
    "tomato_safety_v2_vent_before",
    "tomato_safety_v2_vent_after",
    "tomato_safety_v2_screen_before",
    "tomato_safety_v2_screen_after",
    "tomato_safety_v2_shade_before",
    "tomato_safety_v2_shade_after",
    "tomato_safety_v2_heat_before",
    "tomato_safety_v2_heat_after",
    "tomato_safety_v2_reasons",
    "candidate_guardrail_compatibility_label",
    "candidate_guardrail_rewrite_fields",
    "candidate_selection_source",
    "profile_action_conflict_label",
    "final_action_risk_reason_v2",
    "final_action_would_fail_canopy_boundary_v2",
    "final_action_predicted_canopy_lt0_v2",
    "final_action_predicted_canopy_warning_v2",
    "canopy_boundary_shadow_warning_v2",
    "rh_low_violation",
    "rh_high_violation",
    "temp_violation",
    "structured_anchor_profile_bridge_attempted",
    "structured_anchor_profile_bridge_bridgeable",
    "structured_anchor_profile_bridge_applied",
    "structured_anchor_profile_bridge_valid_structured_anchor",
    "structured_anchor_profile_bridge_profile_candidate_count",
    "structured_anchor_profile_bridge_hard_safety_profile_violation",
    "structured_anchor_profile_bridge_final_control_change",
    "structured_anchor_profile_bridge_current_plan_modified",
    "structured_anchor_profile_bridge_low_level_action_generated",
]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {field: "" for field in FIELDS}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


def _rows(count: int, *, runtime: str = "", unstable: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(count):
        row = {
            "step": step,
            "temp_air": 35.0 if unstable else 22.0,
            "rh_air": 46.0 if unstable else 70.0,
            "vpd_air": 3.1 if unstable else 1.1,
            "canopy_dew_margin": 2.0 if unstable else 4.0,
            "dew_margin_air": 10.0 if unstable else 4.0,
            "dry_risk": "true" if unstable else "false",
            "dew_risk": "false",
            "target_temp": 18.0 if unstable else 22.0,
            "target_rh": 75.0 if unstable else 68.0,
            "target_co2": 430.0,
            "profile_selected_delta_target_rh": 3.0 if unstable else 0.0,
            "vpd_high_excess": 1.8 if unstable else 0.0,
            "u_heating": 0.0,
            "u_co2": 0.0,
            "u_screen": 0.6 if unstable and step % 2 == 0 else 0.05,
            "u_ventilation": 0.95 if unstable else 0.25,
            "u_lighting": 0.0,
            "u_shading": 0.85 if unstable else 0.0,
            "tomato_safety_v2_applied": "true" if unstable else "false",
            "tomato_safety_v2_vent_before": 0.2,
            "tomato_safety_v2_vent_after": 0.95 if unstable else 0.2,
            "tomato_safety_v2_screen_before": 0.1,
            "tomato_safety_v2_screen_after": 0.6 if unstable else 0.1,
            "tomato_safety_v2_reasons": "hot_temperature_override" if unstable else "",
            "candidate_guardrail_compatibility_label": "hard_rewrite_predicted" if unstable else "compatible",
            "candidate_guardrail_rewrite_fields": "vent,screen" if unstable else "",
            "candidate_selection_source": "recent_anchor" if unstable else "profile_candidate",
            "profile_action_conflict_label": "target_rh_vs_high_vent" if unstable else "compatible",
            "final_action_risk_reason_v2": "hot_temperature_override" if unstable else "",
            "final_action_predicted_canopy_warning_v2": "false",
            "rh_low_violation": 2.0 if unstable else 0.0,
            "temp_violation": 1.0 if unstable else 0.0,
            "structured_anchor_profile_bridge_attempted": "true",
            "structured_anchor_profile_bridge_bridgeable": "true",
            "structured_anchor_profile_bridge_applied": "true",
            "structured_anchor_profile_bridge_valid_structured_anchor": "true",
            "structured_anchor_profile_bridge_profile_candidate_count": 2,
            "structured_anchor_profile_bridge_hard_safety_profile_violation": "false",
            "structured_anchor_profile_bridge_final_control_change": "false",
            "structured_anchor_profile_bridge_current_plan_modified": "false",
            "structured_anchor_profile_bridge_low_level_action_generated": "false",
        }
        if runtime and step == count - 1:
            row["runtime_error"] = runtime
            row["runtime_error_type"] = "simulator_or_controller_error"
        rows.append(row)
    return rows


class TestQwen37PlusStructuredBridgeRuntimeFailureAttributionV65(unittest.TestCase):
    def test_runtime_error_classification_separates_cvodes_and_invalid_argument(self):
        self.assertEqual(v65.classify_runtime_error("Error in Function::call for 'F' [CvodesInterface]"), "cvodes_or_casadi_failure")
        self.assertEqual(v65.classify_runtime_error("[Errno 22] Invalid argument"), "simulator_io_invalid_argument")

    def test_fixed_scenarios_window_and_readiness_never_authorize_controlled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v64_dir = root / "v64"
            v39_dir = root / "v39"
            v64_dir.mkdir()
            v39_dir.mkdir()
            runtimes = [
                "Error in Function::call for 'F' [CvodesInterface]",
                "Error in Function::call for 'F' [CvodesInterface]",
                "[Errno 22] Invalid argument",
            ]
            for scenario, runtime in zip(v65.FAILURE_SCENARIOS, runtimes):
                _write(v64_dir / f"{scenario}_llm_rspc_v2.csv", _rows(30, runtime=runtime, unstable=True))
                _write(v39_dir / f"{scenario}_llm_rspc_v2.csv", _rows(30, runtime=runtime, unstable=True))
            _write(v64_dir / "y2099_d1_s1_n720_llm_rspc_v2.csv", _rows(10, runtime="[Errno 22] Invalid argument", unstable=True))

            attribution = v65.build_runtime_failure_attribution(
                v64_trace_dir=v64_dir,
                v39_trace_dir=v39_dir,
                v64_result_json=root / "missing_result.json",
                v64_consistency_json=root / "missing_consistency.json",
                failure_scenarios=v65.FAILURE_SCENARIOS,
                window_steps=120,
            )
            delta = v65.build_profile_candidate_guardrail_delta_audit(attribution)
            simulator = v65.build_simulator_boundary_audit(attribution)
            repair = v65.build_runtime_repair_design(attribution, delta, simulator)
            readiness = v65.build_readiness(attribution, delta, simulator, repair)

        self.assertEqual(attribution["failure_scenarios"], list(v65.FAILURE_SCENARIOS))
        self.assertEqual(len(attribution["scenario_reports"]), 3)
        self.assertEqual(attribution["runtime_error_kind_counts"]["cvodes_or_casadi_failure"], 2)
        self.assertEqual(attribution["runtime_error_kind_counts"]["simulator_io_invalid_argument"], 1)
        for report in attribution["scenario_reports"]:
            self.assertTrue(report["failure_window"]["short_window"])
            self.assertTrue(report["control_path_unchanged_failure"])
            self.assertEqual(report["bridge_metrics"]["final_control_change_count"], 0)
            self.assertEqual(report["bridge_metrics"]["current_plan_modified_count"], 0)
            self.assertEqual(report["bridge_metrics"]["low_level_action_generated_count"], 0)
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])

    def test_delta_and_simulator_audits_expose_profile_pressure_and_invalid_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v64_dir = root / "v64"
            v39_dir = root / "v39"
            v64_dir.mkdir()
            v39_dir.mkdir()
            scenario = v65.FAILURE_SCENARIOS[0]
            _write(v64_dir / f"{scenario}_llm_rspc_v2.csv", _rows(130, runtime="[Errno 22] Invalid argument", unstable=True))
            _write(v39_dir / f"{scenario}_llm_rspc_v2.csv", _rows(130, runtime="[Errno 22] Invalid argument", unstable=False))

            attribution = v65.build_runtime_failure_attribution(
                v64_trace_dir=v64_dir,
                v39_trace_dir=v39_dir,
                v64_result_json=root / "missing_result.json",
                v64_consistency_json=root / "missing_consistency.json",
                failure_scenarios=[scenario],
                window_steps=120,
            )
            delta = v65.build_profile_candidate_guardrail_delta_audit(attribution)
            simulator = v65.build_simulator_boundary_audit(attribution)

        self.assertTrue(delta["profile_candidate_guardrail_pressure_detected"])
        self.assertGreater(delta["scenario_reports"][0]["candidate_guardrail_issue_steps"], 0)
        self.assertEqual(simulator["simulator_io_invalid_argument_count"], 1)
        self.assertEqual(simulator["cvodes_or_casadi_failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
