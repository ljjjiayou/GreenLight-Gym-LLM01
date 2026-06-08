import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.cvodes_failure_causal_attribution_v44 import (
    FAILURE_SCENARIOS,
    build_action_counterfactual_design,
    build_causal_attribution,
    build_profile_guardrail_compatibility,
    build_readiness,
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
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
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
    "rspc_action_candidate_count",
    "rspc_action_post_shape_best_eligible",
    "rspc_action_post_shape_safety_gate_reason",
    "rspc_action_selected_name",
    "source",
    "transition_gate_applied",
    "transition_gate_bypassed",
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


def _rows(count: int, *, runtime_step: int | None = None, unstable: bool = False, gated: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(count):
        row = {
            "step": step,
            "temp_air": 24 + (0.2 * step if unstable else 0.02 * step),
            "rh_air": 86 - step * 0.15,
            "vpd_air": 0.6 + (0.04 * step if unstable else 0.005 * step),
            "canopy_dew_margin": 4.0 - (0.04 * step if unstable else 0.0),
            "dew_margin_air": 5.0,
            "dry_risk": "true" if unstable else "false",
            "target_temp": 20 if step < count // 2 else 24 if unstable else 20,
            "target_rh": 80,
            "target_co2": 400,
            "profile_selected_delta_target_rh": -1.0 if unstable else 0.0,
            "vpd_high_excess": 1.0 if unstable else 0.0,
            "u_heating": 0.2 if unstable and step % 3 == 0 else 0.0,
            "u_co2": 0.0,
            "u_screen": 0.5 if unstable and step % 2 == 0 else 0.1,
            "u_ventilation": 0.9 if unstable and step % 2 == 0 else 0.25,
            "u_lighting": 0.0,
            "u_shading": 0.85 if unstable else 0.0,
            "tomato_safety_v2_applied": "true" if unstable else "false",
            "tomato_safety_v2_vent_before": 0.2,
            "tomato_safety_v2_vent_after": 0.9 if unstable else 0.2,
            "tomato_safety_v2_screen_before": 0.1,
            "tomato_safety_v2_screen_after": 0.5 if unstable else 0.1,
            "tomato_safety_v2_reasons": "hard_dry_vpd_risk,hot_temperature_override" if unstable else "",
            "rspc_action_candidate_count": 4,
            "rspc_action_post_shape_best_eligible": "false" if unstable else "true",
            "rspc_action_post_shape_safety_gate_reason": "temp_high_gate" if unstable else "",
            "rspc_action_selected_name": "anchor" if unstable else "smooth",
            "source": "anchor" if unstable else "ppo",
            "transition_gate_applied": "true" if gated and step % 2 == 0 else "false",
            "transition_gate_bypassed": "true" if gated and step % 5 == 0 else "false",
        }
        if runtime_step is not None and step == runtime_step:
            row["runtime_error"] = '[Errno 22] Invalid argument'
            row["runtime_error_type"] = "simulator_or_controller_error"
            row["replan_reason"] = "runtime_error"
        rows.append(row)
    return rows


class TestCvodesFailureCausalAttributionV44(unittest.TestCase):
    def test_fixed_scenarios_and_pending_v43_traces_are_not_counted_as_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            v43 = root / "v43"
            original.mkdir()
            v43.mkdir()
            for scenario in FAILURE_SCENARIOS:
                _write(original / f"{scenario}_ppo.csv", _rows(30))
                _write(original / f"{scenario}_llm_rspc_v2.csv", _rows(30, runtime_step=29, unstable=True))
            _write(v43 / f"{FAILURE_SCENARIOS[0]}_llm_rspc_v2.csv", _rows(20, runtime_step=19, unstable=True, gated=True))

            report = build_causal_attribution(
                original_trace_dir=original,
                v43_trace_dir=v43,
                failure_scenarios=FAILURE_SCENARIOS,
                window_steps=120,
            )

        self.assertEqual(report["failure_scenarios"], list(FAILURE_SCENARIOS))
        pending = {item["scenario_id"] for item in report["scenario_reports"] if item["status"] == "pending_trace_not_evaluated"}
        self.assertEqual(pending, {FAILURE_SCENARIOS[1], FAILURE_SCENARIOS[2]})
        first = report["scenario_reports"][0]
        self.assertTrue(first["windows"]["v43_pre_failure"]["short_window"])
        self.assertIn("pending_trace_not_evaluated", report["failure_taxonomy"])

    def test_profile_guardrail_and_readiness_never_authorize_controlled_or_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            v43 = root / "v43"
            original.mkdir()
            v43.mkdir()
            scenario = FAILURE_SCENARIOS[0]
            _write(original / f"{scenario}_ppo.csv", _rows(12))
            _write(original / f"{scenario}_llm_rspc_v2.csv", _rows(12, runtime_step=11, unstable=True))
            _write(v43 / f"{scenario}_llm_rspc_v2.csv", _rows(8, runtime_step=7, unstable=True, gated=True))
            report = build_causal_attribution(
                original_trace_dir=original,
                v43_trace_dir=v43,
                failure_scenarios=[scenario],
                window_steps=120,
            )
            compatibility = build_profile_guardrail_compatibility(report)
            counterfactual = build_action_counterfactual_design(report)
            readiness = build_readiness(report, compatibility, counterfactual)

        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])
        self.assertFalse(counterfactual["counterfactual_execution_authorized"])
        self.assertTrue(compatibility["profile_guardrail_compatibility_issue_detected"])


if __name__ == "__main__":
    unittest.main()
