import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments import qwen37plus_control_chain_trajectory_architecture_redesign_v79 as v79


FIELDS = [
    "step",
    "runtime_error",
    "runtime_error_type",
    "temp_air",
    "rh_air",
    "vpd_air",
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


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def _row(step: int, *, oscillating: bool = True, tomato: bool = True) -> dict[str, object]:
    screen = 0.05 if step % 2 else 0.75
    vent = 0.95 if step % 3 else 0.15
    shade = 0.85 if step % 2 else 0.35
    if not oscillating:
        screen = 0.4
        vent = 0.55
        shade = 0.55
    row = {
        "step": step,
        "runtime_error": "",
        "runtime_error_type": "",
        "temp_air": 30.0 + step * 0.1,
        "rh_air": 48.0,
        "vpd_air": 2.5,
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


def _gap_for_rows(rows: list[dict[str, object]], *, v78_dominant: str = "mixed_weather_control_accumulation"):
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    rows = list(rows)
    rows[-1]["runtime_error"] = 'CVode returned "CV_CONV_FAILURE".'
    rows[-1]["runtime_error_type"] = "simulator_or_controller_error"
    _write(root / "y2010_d180_s43_n720_llm_rspc_v2.csv", rows)
    v78_attr = root / "v78_attr.json"
    v78_ready = root / "v78_ready.json"
    v78_attr.write_text(
        json.dumps(
            {
                "artifact": "synthetic_v78",
                "dominant_accumulation_attribution": v78_dominant,
            }
        ),
        encoding="utf-8",
    )
    v78_ready.write_text(
        json.dumps({"artifact": "synthetic_ready", "next_action": "stateful_runtime_stability_supervisor_shadow_instrumentation_plan"}),
        encoding="utf-8",
    )
    gap = v79.build_control_chain_gap_audit(
        trace_dir=root,
        v78_attribution_json=v78_attr,
        v78_readiness_json=v78_ready,
    )
    return tmp, gap


class TestQwen37PlusControlChainTrajectoryArchitectureRedesignV79(unittest.TestCase):
    def test_mixed_accumulation_triggers_step_local_chain_gap(self):
        tmp, gap = _gap_for_rows([_row(step) for step in range(40)])
        self.addCleanup(tmp.cleanup)

        self.assertTrue(gap["step_local_chain_insufficient"])
        self.assertEqual(gap["required_architecture_shift"], "state_history_to_trajectory_action_envelope_chain")
        self.assertGreater(gap["aggregate_metrics"]["mean_action_reversal_count"], 0)
        self.assertEqual(gap["next_action"], "trajectory_action_envelope_shadow_design_plan")

    def test_trajectory_contract_is_sequence_based_and_in_loop_with_tomato_safety(self):
        tmp, gap = _gap_for_rows([_row(step) for step in range(40)])
        self.addCleanup(tmp.cleanup)

        contract = v79.build_trajectory_action_envelope_contract(gap)
        output = contract["trajectory_output_contract"]

        self.assertEqual(contract["candidate_source"], "normal_path_trajectory_action_envelope_shadow")
        self.assertIn("action_envelope_sequence", output["required_fields"])
        self.assertIn("tomato_safety_projection_sequence", output["required_fields"])
        self.assertTrue(contract["tomato_safety_in_loop"]["projection_before_scoring_required"])
        self.assertTrue(contract["tomato_safety_in_loop"]["final_shield_still_required"])
        self.assertEqual(len(contract["paper_innovation_candidates"]), 3)

    def test_readiness_routes_to_shadow_instrumentation_when_contract_complete(self):
        tmp, gap = _gap_for_rows([_row(step) for step in range(40)])
        self.addCleanup(tmp.cleanup)
        contract = v79.build_trajectory_action_envelope_contract(gap)

        readiness = v79.build_readiness(gap, contract)

        self.assertTrue(readiness["architecture_hypothesis_ready_for_shadow_instrumentation"])
        self.assertEqual(readiness["next_action"], "trajectory_action_envelope_shadow_instrumentation_plan")
        self.assertFalse(readiness["default_controller_changed"])
        self.assertFalse(readiness["online_llm_needed_next"])

    def test_readiness_requires_gap_evidence_not_contract_alone(self):
        tmp, gap = _gap_for_rows([_row(step, oscillating=False, tomato=False) for step in range(40)], v78_dominant="weather_forcing_dominant")
        self.addCleanup(tmp.cleanup)
        contract = v79.build_trajectory_action_envelope_contract(gap)

        readiness = v79.build_readiness(gap, contract)

        self.assertFalse(readiness["architecture_hypothesis_ready_for_shadow_instrumentation"])
        self.assertEqual(readiness["next_action"], "trajectory_action_envelope_contract_repair_plan")

    def test_boundary_flags_remain_false(self):
        tmp, gap = _gap_for_rows([_row(step) for step in range(40)])
        self.addCleanup(tmp.cleanup)
        contract = v79.build_trajectory_action_envelope_contract(gap)
        readiness = v79.build_readiness(gap, contract)

        for payload in (gap, contract, readiness):
            for field in v79.BOUNDARY_FALSE_FIELDS:
                self.assertFalse(payload[field])


if __name__ == "__main__":
    unittest.main()
