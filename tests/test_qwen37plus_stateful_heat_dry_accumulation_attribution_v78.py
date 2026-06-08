import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments import qwen37plus_stateful_heat_dry_accumulation_attribution_v78 as v78


FIELDS = [
    "step",
    "runtime_error",
    "runtime_error_type",
    "temp_air",
    "rh_air",
    "vpd_air",
    "canopy_dew_margin",
    "dew_margin_air",
    "temp_out",
    "rh_out",
    "glob_rad",
    "forecast_temp_out_delta_1h",
    "forecast_rh_out_mean_1h",
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
]


def _runtime_error() -> str:
    return 'Error in Function::call for F [CvodesInterface]: CVode returned "CV_CONV_FAILURE".'


def _write(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    fieldnames = fields or FIELDS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _row(step: int, *, weather: bool = False, control: bool = False, tomato: bool = False) -> dict[str, object]:
    fraction = step / 29.0 if step else 0.0
    temp_air = 23.0
    rh_air = 70.0
    vpd_air = 1.0
    temp_out = 20.0
    rh_out = 75.0
    glob_rad = 150.0
    if weather:
        temp_air = 24.0 + 10.0 * fraction
        rh_air = 72.0 - 28.0 * fraction
        vpd_air = 1.0 + 2.0 * fraction
        temp_out = 19.0 + 8.0 * fraction
        rh_out = 82.0 - 35.0 * fraction
        glob_rad = 100.0 + 700.0 * fraction
    if control:
        temp_air = max(temp_air, 30.0)
        rh_air = min(rh_air, 48.0)
        vpd_air = max(vpd_air, 2.6)
    screen = 0.1
    vent = 0.25
    shade = 0.0
    if control:
        screen = 0.05 if step % 2 else 0.70
        vent = 0.95 if step % 3 else 0.15
        shade = 0.85 if step % 2 else 0.35
    row = {
        "step": step,
        "runtime_error": "",
        "runtime_error_type": "",
        "temp_air": temp_air,
        "rh_air": rh_air,
        "vpd_air": vpd_air,
        "canopy_dew_margin": 3.0,
        "dew_margin_air": 8.0,
        "temp_out": temp_out,
        "rh_out": rh_out,
        "glob_rad": glob_rad,
        "forecast_temp_out_delta_1h": 2.0 if weather else 0.0,
        "forecast_rh_out_mean_1h": rh_out,
        "u_heating": 0.0,
        "u_co2": 0.0,
        "u_screen": screen,
        "u_ventilation": vent,
        "u_lighting": 0.0,
        "u_shading": shade,
        "tomato_safety_v2_applied": "true" if tomato else "false",
        "tomato_safety_v2_heat_before": 0.0,
        "tomato_safety_v2_heat_after": 0.0,
        "tomato_safety_v2_screen_before": 0.1,
        "tomato_safety_v2_screen_after": 0.7 if tomato else 0.1,
        "tomato_safety_v2_vent_before": 0.2,
        "tomato_safety_v2_vent_after": 0.9 if tomato else 0.2,
        "tomato_safety_v2_shade_before": 0.0,
        "tomato_safety_v2_shade_after": 0.8 if tomato else 0.0,
        "tomato_safety_v2_reasons": "hot_dry_rewrite" if tomato else "",
    }
    return row


def _catalog(rows: list[dict[str, object]], *, fields: list[str] | None = None):
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    rows = list(rows)
    rows[-1]["runtime_error"] = _runtime_error()
    rows[-1]["runtime_error_type"] = "simulator_or_controller_error"
    _write(root / "y2010_d180_s43_n720_llm_rspc_v2.csv", rows, fields=fields)
    attribution = v78.build_accumulation_attribution(
        trace_dir=root,
        v77_snapshot_json=root / "missing_snapshot.json",
        v77_attribution_json=root / "missing_attribution.json",
        window_steps=(30,),
    )
    return tmp, attribution


class TestQwen37PlusStatefulHeatDryAccumulationAttributionV78(unittest.TestCase):
    def test_weather_forcing_dominant_without_control_history(self):
        tmp, attribution = _catalog([_row(step, weather=True, control=False) for step in range(30)])
        self.addCleanup(tmp.cleanup)

        report = attribution["scenario_reports"][0]

        self.assertEqual(report["dominant_accumulation_attribution"], "weather_forcing_dominant")
        self.assertEqual(attribution["dominant_accumulation_attribution"], "weather_forcing_dominant")

    def test_control_history_accumulation_dominant_without_weather_signal(self):
        tmp, attribution = _catalog([_row(step, weather=False, control=True) for step in range(30)])
        self.addCleanup(tmp.cleanup)

        report = attribution["scenario_reports"][0]

        self.assertEqual(report["dominant_accumulation_attribution"], "control_history_accumulation_dominant")
        self.assertGreater(report["window_reports"][0]["action_history"]["large_action_delta_count"], 0)

    def test_mixed_weather_control_accumulation_when_both_signals_present(self):
        tmp, attribution = _catalog([_row(step, weather=True, control=True) for step in range(30)])
        self.addCleanup(tmp.cleanup)

        report = attribution["scenario_reports"][0]

        self.assertEqual(report["dominant_accumulation_attribution"], "mixed_weather_control_accumulation")
        self.assertEqual(attribution["next_action"], "stateful_runtime_stability_supervisor_shadow_instrumentation_plan")

    def test_tomato_rewrite_coupled_instability_when_rewrite_signal_is_primary(self):
        tmp, attribution = _catalog([_row(step, weather=False, control=False, tomato=True) for step in range(30)])
        self.addCleanup(tmp.cleanup)

        report = attribution["scenario_reports"][0]

        self.assertEqual(report["dominant_accumulation_attribution"], "tomato_safety_rewrite_coupled_instability")
        self.assertGreater(report["window_reports"][0]["tomato_rewrite"]["rewrite_steps"], 0)

    def test_missing_core_history_fields_routes_to_instrumentation(self):
        fields = [field for field in FIELDS if field not in {"vpd_air", "u_ventilation"}]
        tmp, attribution = _catalog(
            [{key: value for key, value in _row(step, weather=True, control=True).items() if key in fields} for step in range(30)],
            fields=fields,
        )
        self.addCleanup(tmp.cleanup)

        self.assertEqual(attribution["dominant_accumulation_attribution"], "insufficient_history_for_attribution")
        self.assertEqual(attribution["next_action"], "stateful_history_instrumentation_plan")

    def test_supervisor_design_and_readiness_keep_boundary_flags_false(self):
        tmp, attribution = _catalog([_row(step, weather=True, control=True) for step in range(30)])
        self.addCleanup(tmp.cleanup)
        design = v78.build_supervisor_design(attribution)
        readiness = v78.build_readiness(attribution, design)

        self.assertIn("supervisor_contract", design)
        self.assertEqual(design["supervisor_contract"]["risk_states"], ["watch", "warning", "solver_sensitive", "abort_for_diagnosis"])
        self.assertTrue(design["supervisor_contract"]["future_behavior_changes_require_later_opt_in_stage"])
        self.assertTrue(readiness["history_attribution_complete"])
        for payload in (attribution, design, readiness):
            for field in v78.BOUNDARY_FALSE_FIELDS:
                self.assertFalse(payload[field])


if __name__ == "__main__":
    unittest.main()
