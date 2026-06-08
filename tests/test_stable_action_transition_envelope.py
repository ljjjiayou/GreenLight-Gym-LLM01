import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.stable_action_transition_envelope import (
    build_envelope,
    build_readiness,
    build_shadow_gate_design,
    build_transition_rows,
    discover_trace_pairs,
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
]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {field: "" for field in FIELDS}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


def _rows(count: int, *, unstable: bool = False, cvodes_step: int | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(count):
        if unstable:
            vent = 0.95 if step % 2 == 0 else 0.15
            screen = 0.55 if step % 2 == 0 else 0.05
            shade = 0.9 if step % 3 == 0 else 0.0
        else:
            vent = 0.30 + 0.01 * step
            screen = 0.10 + 0.002 * step
            shade = 0.0
        row = {
            "step": step,
            "temp_air": 29 if unstable else 22,
            "rh_air": 42 if unstable else 65,
            "vpd_air": 2.4 if unstable else 1.2,
            "canopy_dew_margin": 5,
            "dew_margin_air": 8,
            "dry_risk": "true" if unstable else "false",
            "dew_risk": "false",
            "u_heating": 0.1,
            "u_ventilation": vent,
            "u_co2": 0.2,
            "u_lighting": 0.0,
            "u_screen": screen,
            "u_shading": shade,
        }
        if unstable:
            row.update(
                {
                    "tomato_safety_v2_applied": "true",
                    "tomato_safety_v2_vent_before": 0.15,
                    "tomato_safety_v2_vent_after": vent,
                    "tomato_safety_v2_screen_before": 0.05,
                    "tomato_safety_v2_screen_after": screen,
                    "tomato_safety_v2_reasons": "hard_dry_vpd_risk",
                }
            )
        if cvodes_step is not None and step == cvodes_step:
            row.update({"runtime_error": 'CVode returned "CV_CONV_FAILURE".'})
        rows.append(row)
    return rows


class TestStableActionTransitionEnvelope(unittest.TestCase):
    def test_trace_pairs_and_transition_labels_are_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "y2010_d180_s43_n720_ppo.csv", _rows(8))
            _write(root / "y2010_d180_s43_n720_llm_rspc_v2.csv", _rows(8, unstable=True, cvodes_step=7))

            pairs = discover_trace_pairs(root)
            stable, unstable, meta = build_transition_rows(
                trace_dir=root,
                failure_scenarios=["y2010_d180_s43_n720"],
                failure_window_steps=3,
            )

        self.assertIn("y2010_d180_s43_n720", pairs)
        self.assertEqual(meta["missing_pairs"], [])
        self.assertTrue(all(row["label"] == "failure_negative" for row in unstable))
        self.assertEqual([row["step"] for row in unstable], [5, 6, 7])
        self.assertTrue(any(row["controller"] == "ppo" for row in stable))
        self.assertFalse(any(row["scenario_id"] == "y2010_d180_s43_n720" and row["controller"] == "llm_rspc_v2" and row["step"] == 7 for row in stable))

    def test_envelope_detects_transition_gap_and_keeps_readiness_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "y2010_d180_s43_n720_ppo.csv", _rows(12))
            _write(root / "y2010_d180_s43_n720_llm_rspc_v2.csv", _rows(12, unstable=True, cvodes_step=11))
            _write(root / "y2010_d120_s42_n720_ppo.csv", _rows(12))
            _write(root / "y2010_d120_s42_n720_llm_rspc_v2.csv", _rows(12))
            stable, unstable, _ = build_transition_rows(
                trace_dir=root,
                failure_scenarios=["y2010_d180_s43_n720"],
                failure_window_steps=4,
            )
            envelope = build_envelope(
                stable_rows=stable,
                unstable_rows=unstable,
                v40_report={"main_attribution": "action_instability"},
            )
            design = build_shadow_gate_design(envelope)
            readiness = build_readiness(envelope, design)

        self.assertTrue(envelope["stable_envelope_exists"])
        self.assertTrue(envelope["transition_gap_detected"])
        self.assertTrue(envelope["transition_pressure_detected"])
        self.assertEqual(design["next_action"], "shadow_transition_gate_patch_plan")
        self.assertTrue(design["shadow_only_design"])
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])
        self.assertEqual(readiness["next_action"], "shadow_transition_gate_patch_plan")


if __name__ == "__main__":
    unittest.main()
