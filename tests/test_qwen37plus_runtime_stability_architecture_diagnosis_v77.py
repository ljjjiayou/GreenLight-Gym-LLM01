import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments import qwen37plus_runtime_stability_architecture_diagnosis_v77 as v77


BASE_FIELDS = [
    "step",
    "runtime_error",
    "runtime_error_type",
    "temp_air",
    "rh_air",
    "vpd_air",
    "co2_air",
    "pipe_temp",
    "canopy_dew_margin",
    "dew_margin_air",
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
    "tomato_safety_v2_applied",
    "tomato_safety_v2_heat_before",
    "tomato_safety_v2_heat_after",
    "tomato_safety_v2_screen_before",
    "tomato_safety_v2_screen_after",
    "tomato_safety_v2_vent_before",
    "tomato_safety_v2_vent_after",
    "tomato_safety_v2_shade_before",
    "tomato_safety_v2_shade_after",
    "tomato_safety_v2_reasons",
    "profile_action_envelope_shadow_enabled",
    "profile_action_envelope_shadow_candidate_count",
    "profile_action_envelope_shadow_eligible_candidate_count",
    "profile_action_envelope_shadow_best_name",
    "profile_action_envelope_shadow_final_action_changed",
    "profile_action_candidate_shadow_final_action_changed",
]


def _write(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    fieldnames = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _row(step: int, **overrides) -> dict[str, object]:
    row = {
        "step": step,
        "runtime_error": "",
        "runtime_error_type": "",
        "temp_air": 24.0,
        "rh_air": 70.0,
        "vpd_air": 1.1,
        "co2_air": 420.0,
        "pipe_temp": 25.0,
        "canopy_dew_margin": 4.0,
        "dew_margin_air": 5.0,
        "u_heating": 0.0,
        "u_co2": 0.0,
        "u_screen": 0.1,
        "u_ventilation": 0.25,
        "u_lighting": 0.0,
        "u_shading": 0.0,
        "tomato_safety_v2_applied": "false",
        "tomato_safety_v2_heat_before": 0.0,
        "tomato_safety_v2_heat_after": 0.0,
        "tomato_safety_v2_screen_before": 0.1,
        "tomato_safety_v2_screen_after": 0.1,
        "tomato_safety_v2_vent_before": 0.25,
        "tomato_safety_v2_vent_after": 0.25,
        "tomato_safety_v2_shade_before": 0.0,
        "tomato_safety_v2_shade_after": 0.0,
        "tomato_safety_v2_reasons": "",
        "profile_action_envelope_shadow_enabled": "true",
        "profile_action_envelope_shadow_candidate_count": 3,
        "profile_action_envelope_shadow_eligible_candidate_count": 3,
        "profile_action_envelope_shadow_best_name": "profile_action_envelope:hot_dry_protect",
        "profile_action_envelope_shadow_final_action_changed": "false",
        "profile_action_candidate_shadow_final_action_changed": "false",
    }
    row.update(overrides)
    return row


def _runtime_error() -> str:
    return (
        "Error in Function::call for 'F' [CvodesInterface]: "
        'CVode returned "CV_CONV_FAILURE". Consult CVODES documentation.'
    )


def _catalog_for_rows(rows: list[dict[str, object]], *, window_steps: int = 4, fields: list[str] | None = None):
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    _write(root / "y2010_d180_s43_n720_llm_rspc_v2.csv", rows, fields=fields)
    catalog = v77.build_snapshot_catalog(
        trace_dir=root,
        v76_execution_json=root / "missing_v76_execution.json",
        v76_readiness_json=root / "missing_v76_readiness.json",
        v44_causal_json=root / "missing_v44.json",
        v65_attribution_json=root / "missing_v65.json",
        window_steps=window_steps,
    )
    return tmp, catalog


class TestQwen37PlusRuntimeStabilityArchitectureDiagnosisV77(unittest.TestCase):
    def test_cvodes_trace_produces_failure_snapshot_with_pre_failure_window(self):
        rows = [_row(step) for step in range(10)]
        rows[-1]["runtime_error"] = _runtime_error()
        rows[-1]["runtime_error_type"] = "simulator_or_controller_error"
        tmp, catalog = _catalog_for_rows(rows, window_steps=4, fields=BASE_FIELDS)
        self.addCleanup(tmp.cleanup)

        snapshot = catalog["snapshots"][0]

        self.assertEqual(catalog["failure_trace_count"], 1)
        self.assertEqual(snapshot["runtime_error_kind"], "cvodes_or_casadi_failure")
        self.assertEqual(snapshot["failure_step"], 9)
        self.assertEqual(snapshot["pre_failure_window"]["row_count"], 4)
        self.assertEqual(snapshot["pre_failure_window"]["window_start_step"], 6)
        self.assertEqual(snapshot["pre_failure_window"]["window_end_step"], 9)

    def test_v76_like_trace_attributes_away_from_envelope_action_change(self):
        rows = [_row(step) for step in range(5)]
        rows[-1]["runtime_error"] = _runtime_error()
        rows[-1]["runtime_error_type"] = "simulator_or_controller_error"
        tmp, catalog = _catalog_for_rows(rows, window_steps=5, fields=BASE_FIELDS)
        self.addCleanup(tmp.cleanup)

        attribution = v77.build_root_cause_attribution(catalog)
        report = attribution["scenario_reports"][0]

        self.assertTrue(report["envelope_action_invariance_preserved"])
        self.assertEqual(report["profile_action_envelope_shadow_final_action_changed_steps"], 0)
        self.assertNotIn("envelope_induced_action_change", report["categories"])

    def test_high_temp_high_vpd_routes_to_hot_dry_solver_sensitive_regime(self):
        rows = [_row(step, temp_air=35.0, rh_air=47.0, vpd_air=3.1) for step in range(8)]
        rows[-1]["runtime_error"] = _runtime_error()
        rows[-1]["runtime_error_type"] = "simulator_or_controller_error"
        tmp, catalog = _catalog_for_rows(rows, window_steps=8, fields=BASE_FIELDS)
        self.addCleanup(tmp.cleanup)

        attribution = v77.build_root_cause_attribution(catalog)
        report = attribution["scenario_reports"][0]

        self.assertIn("hot_dry_high_vpd_solver_sensitive_regime", report["categories"])
        self.assertEqual(report["primary_attribution"], "hot_dry_high_vpd_solver_sensitive_regime")

    def test_large_action_jump_routes_to_action_transition_sensitive(self):
        rows = [_row(step) for step in range(8)]
        rows[5]["u_screen"] = 0.65
        rows[6]["u_screen"] = 0.05
        rows[-1]["runtime_error"] = _runtime_error()
        rows[-1]["runtime_error_type"] = "simulator_or_controller_error"
        tmp, catalog = _catalog_for_rows(rows, window_steps=8, fields=BASE_FIELDS)
        self.addCleanup(tmp.cleanup)

        attribution = v77.build_root_cause_attribution(catalog)
        report = attribution["scenario_reports"][0]

        self.assertIn("action_transition_or_rate_sensitive", report["categories"])
        self.assertEqual(report["primary_attribution"], "action_transition_or_rate_sensitive")

    def test_missing_required_fields_routes_to_reproducer_instrumentation(self):
        fields = [field for field in BASE_FIELDS if field not in {"vpd_air", "u_screen"}]
        rows = [{key: value for key, value in _row(step).items() if key in fields} for step in range(6)]
        rows[-1]["runtime_error"] = _runtime_error()
        rows[-1]["runtime_error_type"] = "simulator_or_controller_error"
        tmp, catalog = _catalog_for_rows(rows, window_steps=6, fields=fields)
        self.addCleanup(tmp.cleanup)

        attribution = v77.build_root_cause_attribution(catalog)
        report = attribution["scenario_reports"][0]

        self.assertEqual(report["primary_attribution"], "insufficient_reproducer_state")
        self.assertEqual(attribution["next_action"], "one_step_reproducer_instrumentation_plan")

    def test_boundary_flags_remain_false_for_all_v77_artifacts(self):
        rows = [_row(step, temp_air=35.0, vpd_air=3.0) for step in range(5)]
        rows[-1]["runtime_error"] = _runtime_error()
        rows[-1]["runtime_error_type"] = "simulator_or_controller_error"
        tmp, catalog = _catalog_for_rows(rows, window_steps=5, fields=BASE_FIELDS)
        self.addCleanup(tmp.cleanup)
        attribution = v77.build_root_cause_attribution(catalog)
        design = v77.build_architecture_design(attribution, v65_repair_json=Path(tmp.name) / "missing_repair.json")

        for payload in (catalog, attribution, design):
            for field in v77.BOUNDARY_FALSE_FIELDS:
                self.assertFalse(payload[field])

    def test_architecture_design_contains_required_contracts_and_later_opt_in_boundary(self):
        rows = [_row(step, temp_air=35.0, vpd_air=3.0) for step in range(5)]
        rows[-1]["runtime_error"] = _runtime_error()
        rows[-1]["runtime_error_type"] = "simulator_or_controller_error"
        tmp, catalog = _catalog_for_rows(rows, window_steps=5, fields=BASE_FIELDS)
        self.addCleanup(tmp.cleanup)
        attribution = v77.build_root_cause_attribution(catalog)
        design = v77.build_architecture_design(attribution, v65_repair_json=Path(tmp.name) / "missing_repair.json")

        self.assertEqual(
            set(design["contracts"]),
            {"failure_snapshot", "one_step_reproducer_request", "pre_step_domain_guard", "runtime_stability_policy"},
        )
        self.assertTrue(
            design["contracts"]["runtime_stability_policy"]["future_behavior_changes_require_later_opt_in_stage"]
        )
        self.assertIn("Future behavior changes require a later opt-in stage", v77.build_design_report(design))


if __name__ == "__main__":
    unittest.main()
