import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.shadow_transition_gate_patch import (
    build_audit,
    build_comparison,
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
    "u_heating",
    "u_ventilation",
    "u_co2",
    "u_lighting",
    "u_screen",
    "u_shading",
    "tomato_safety_v2_applied",
    "tomato_safety_v2_reasons",
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


def _row(step: int, *, vent: float, screen: float, safety: bool = False, cvodes: bool = False) -> dict[str, object]:
    row: dict[str, object] = {
        "step": step,
        "temp_air": 24,
        "rh_air": 60,
        "vpd_air": 1.2,
        "canopy_dew_margin": 5,
        "dew_margin_air": 8,
        "dry_risk": "false",
        "dew_risk": "false",
        "u_heating": 0.1,
        "u_ventilation": vent,
        "u_co2": 0.2,
        "u_lighting": 0.0,
        "u_screen": screen,
        "u_shading": 0.0,
    }
    if safety:
        row.update(
            {
                "tomato_safety_v2_applied": "true",
                "tomato_safety_v2_reasons": "hard_safety_test",
            }
        )
    if cvodes:
        row["runtime_error"] = 'CVode returned "CV_CONV_FAILURE".'
    return row


def _envelope() -> dict[str, object]:
    return {
        "transition_stats_by_regime": [
            {
                "label": "stable_positive",
                "regime": "neutral_or_mild",
                "action_delta_stats": {
                    "u_heating": {"p95": 0.1},
                    "u_ventilation": {"p95": 0.1},
                    "u_screen": {"p95": 0.1},
                    "u_shading": {"p95": 0.1},
                },
            }
        ]
    }


class TestShadowTransitionGatePatch(unittest.TestCase):
    def test_shadow_gate_limits_large_delta_and_detects_reversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ppo_rows = [_row(i, vent=0.2 + i * 0.01, screen=0.1) for i in range(8)]
            llm_rows = [
                _row(0, vent=0.2, screen=0.1),
                _row(1, vent=0.95, screen=0.55),
                _row(2, vent=0.05, screen=0.05),
                _row(3, vent=0.95, screen=0.55, cvodes=True),
            ]
            _write(root / "y2010_d180_s43_n720_ppo.csv", ppo_rows)
            _write(root / "y2010_d180_s43_n720_llm_rspc_v2.csv", llm_rows)

            audit = build_audit(
                trace_dir=root,
                envelope=_envelope(),
                failure_scenarios=["y2010_d180_s43_n720"],
                window_steps=3,
                reversal_window_steps=6,
            )
            scenario = audit["scenario_reports"][0]

        self.assertGreater(scenario["original_metrics"]["large_action_delta_count"], scenario["shadow_metrics"]["large_action_delta_count"])
        self.assertGreater(scenario["original_metrics"]["action_oscillation_count"], scenario["shadow_metrics"]["action_oscillation_count"])
        self.assertGreater(scenario["shadow_gate_counters"]["delta_limited_count"], 0)
        self.assertGreater(scenario["shadow_gate_counters"]["reversal_projected_count"], 0)
        self.assertTrue(audit["transition_gate_shadow_effective"])

    def test_hard_safety_bypass_is_not_clipped_and_readiness_blocks_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ppo_rows = [_row(i, vent=0.2, screen=0.1) for i in range(5)]
            llm_rows = [
                _row(0, vent=0.2, screen=0.1),
                _row(1, vent=0.95, screen=0.55, safety=True),
                _row(2, vent=0.95, screen=0.55, cvodes=True),
            ]
            _write(root / "y2010_d180_s43_n720_ppo.csv", ppo_rows)
            _write(root / "y2010_d180_s43_n720_llm_rspc_v2.csv", llm_rows)
            audit = build_audit(
                trace_dir=root,
                envelope=_envelope(),
                failure_scenarios=["y2010_d180_s43_n720"],
                window_steps=2,
                reversal_window_steps=6,
            )
            comparison = build_comparison(audit)
            readiness = build_readiness(audit)

        safety_record = audit["scenario_reports"][0]["shadow_records"][0]
        self.assertTrue(safety_record["hard_safety_bypass"])
        self.assertEqual(safety_record["original_action"]["u_ventilation"], safety_record["shadow_action"]["u_ventilation"])
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])
        self.assertEqual(comparison["controlled_replay_allowed"], False)


if __name__ == "__main__":
    unittest.main()
