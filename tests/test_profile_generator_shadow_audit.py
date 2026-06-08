import csv
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from gl_gym.experiments.diagnose_ppo_vs_llm import profile_generator_shadow_to_record
from gl_gym.experiments.profile_generator_shadow_audit import audit_trace, audit_traces, build_report


def _write_csv(path, rows):
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@dataclass
class DummyState:
    timestep: int = 12
    temp_air: float = 33.0
    rh_air: float = 55.0
    co2_air: float = 430.0
    glob_rad: float = 760.0
    forecast_rad_mean_1h: float = 760.0
    forecast_rad_peak_2h: float = 820.0
    temp_air_delta_1h: float = 0.8
    dew_margin_air: float = 4.0
    canopy_dew_margin: float = 4.0


class TestProfileGeneratorShadowAudit(unittest.TestCase):
    def test_plan_metadata_is_compressed_to_step_trace_fields(self):
        plan = {
            "created_timestep": 12,
            "current_target_temp": 18.0,
            "current_target_co2": 430.0,
            "current_target_rh": 76.0,
            "target_profile": {
                "target_temp": [18.0] * 4,
                "target_co2": [430.0] * 4,
                "target_rh": [76.0] * 4,
            },
            "intent_contract": {"regime": "hot_dry_relief", "confidence": 0.8},
            "profile_candidates": [
                {
                    "name": "hot_dry_protect",
                    "target_profile": {
                        "target_temp": [18.0, 17.8, 17.6, 17.5],
                        "target_co2": [430.0, 430.0, 430.0, 430.0],
                        "target_rh": [78.0, 78.0, 79.0, 79.0],
                    },
                }
            ],
            "profile_generator_diagnostics": {
                "shadow_only": True,
                "candidate_count": 1,
                "selected_shadow_profile_name": "hot_dry_protect",
                "selected_shadow_profile_reason": "test",
                "requested_shapes": ["constant_hold", "hot_dry_protect"],
                "score_shadow_only": True,
                "score_selected_shadow_profile_name": "hot_dry_protect",
                "score_selected_shadow_profile_score": 1.25,
                "score_margin_to_second": 0.5,
                "score_safety_gate_reason": "none",
                "score_selector_agreement": True,
                "score_ranked_candidates": ["hot_dry_protect"],
                "score_by_candidate": {"hot_dry_protect": 1.25},
                "score_selected_breakdown": {
                    "dew_risk": 0.0,
                    "hot_dry_retention_gap": 0.2,
                    "humid_relief_gap": 0.0,
                    "temperature_high_risk": 0.1,
                },
            },
        }

        record = profile_generator_shadow_to_record(plan, step=14)

        self.assertEqual(record["intent_regime"], "hot_dry_relief")
        self.assertEqual(record["profile_generator_selected_shadow_profile"], "hot_dry_protect")
        self.assertAlmostEqual(record["profile_selected_target_temp"], 17.6)
        self.assertAlmostEqual(record["profile_selected_delta_target_temp"], -0.4)
        self.assertAlmostEqual(record["profile_selected_delta_target_rh"], 3.0)
        self.assertGreater(record["profile_selected_abs_delta_sum"], 3.0)
        self.assertEqual(record["profile_scorer_selected_shadow_profile"], "hot_dry_protect")
        self.assertTrue(record["profile_scorer_agrees_with_selector"])
        self.assertEqual(record["profile_scorer_safety_gate_reason"], "none")
        self.assertAlmostEqual(record["profile_scorer_selected_score"], 1.25)
        self.assertAlmostEqual(record["profile_scorer_hot_dry_retention_gap"], 0.2)

    def test_step_trace_rescores_profile_scorer_with_current_state(self):
        plan = {
            "created_timestep": 12,
            "current_target_temp": 18.0,
            "current_target_co2": 430.0,
            "current_target_rh": 76.0,
            "target_profile": {
                "target_temp": [18.0] * 4,
                "target_co2": [430.0] * 4,
                "target_rh": [76.0] * 4,
            },
            "intent_contract": {
                "regime": "hot_dry_relief",
                "target_range": {"temp": [17.0, 19.0], "co2": [430.0, 650.0], "rh": [72.0, 78.0]},
                "priority": ["safety", "humidity", "temperature"],
                "constraints": {},
                "profile_shape": {"temp": "shade_cooling", "rh": "hot_dry_protect"},
                "confidence": 0.8,
            },
            "profile_candidates": [
                {
                    "name": "hot_dry_protect",
                    "target_profile": {
                        "target_temp": [17.4] * 4,
                        "target_co2": [430.0] * 4,
                        "target_rh": [77.5] * 4,
                    },
                },
                {
                    "name": "shade_cooling",
                    "target_profile": {
                        "target_temp": [18.0, 17.6, 17.2, 17.0],
                        "target_co2": [430.0] * 4,
                        "target_rh": [75.0] * 4,
                    },
                },
            ],
            "profile_generator_diagnostics": {
                "shadow_only": True,
                "horizon_steps": 4,
                "candidate_count": 2,
                "selected_shadow_profile_name": "hot_dry_protect",
                "score_shadow_only": True,
                "score_selected_shadow_profile_name": "hot_dry_protect",
                "score_safety_gate_reason": "none",
            },
        }

        record = profile_generator_shadow_to_record(plan, step=13, state=DummyState())

        self.assertEqual(record["profile_scorer_safety_gate_reason"], "temp_high_gate")
        self.assertEqual(record["profile_scorer_selected_shadow_profile"], "shade_cooling")
        self.assertFalse(record["profile_scorer_agrees_with_selector"])

    def test_audits_profile_shadow_counts_and_large_delta_windows(self):
        rows = [
            {
                "step": 0,
                "intent_regime": "hot_dry_relief",
                "profile_generator_candidate_count": 3,
                "profile_generator_shadow_only": True,
                "profile_generator_selected_shadow_profile": "hot_dry_protect",
                "profile_candidate_names": "constant_hold,hot_dry_protect,shade_cooling",
                "profile_selected_abs_delta_sum": 0.5,
                "profile_scorer_selected_shadow_profile": "hot_dry_protect",
                "profile_scorer_agrees_with_selector": True,
                "profile_scorer_selected_score": 1.0,
                "profile_scorer_margin_to_second": 0.3,
                "profile_scorer_abs_delta_sum": 0.5,
            },
            {
                "step": 1,
                "intent_regime": "hot_dry_relief",
                "profile_generator_candidate_count": 3,
                "profile_generator_shadow_only": True,
                "profile_generator_selected_shadow_profile": "hot_dry_protect",
                "profile_candidate_names": "constant_hold,hot_dry_protect,shade_cooling",
                "profile_selected_abs_delta_sum": 2.5,
                "profile_selected_delta_target_rh": 2.2,
                "profile_scorer_selected_shadow_profile": "hot_dry_protect",
                "profile_scorer_agrees_with_selector": True,
                "profile_scorer_selected_score": 0.8,
                "profile_scorer_margin_to_second": 0.2,
                "profile_scorer_abs_delta_sum": 2.6,
                "profile_scorer_delta_target_rh": 2.3,
                "profile_scorer_hot_dry_retention_gap": 0.4,
            },
            {
                "step": 2,
                "intent_regime": "hot_dry_relief",
                "profile_generator_candidate_count": 3,
                "profile_generator_shadow_only": True,
                "profile_generator_selected_shadow_profile": "shade_cooling",
                "profile_candidate_names": "constant_hold,hot_dry_protect,shade_cooling",
                "profile_selected_abs_delta_sum": 3.0,
                "profile_selected_delta_target_temp": -1.0,
                "profile_scorer_selected_shadow_profile": "shade_cooling",
                "profile_scorer_agrees_with_selector": True,
                "profile_scorer_selected_score": 0.6,
                "profile_scorer_margin_to_second": 0.1,
                "profile_scorer_abs_delta_sum": 3.1,
                "profile_scorer_delta_target_temp": -1.0,
            },
            {
                "step": 3,
                "intent_regime": "economy_hold",
                "profile_generator_candidate_count": 1,
                "profile_generator_shadow_only": True,
                "profile_generator_selected_shadow_profile": "constant_hold",
                "profile_candidate_names": "constant_hold",
                "profile_selected_abs_delta_sum": 0.0,
                "profile_scorer_selected_shadow_profile": "constant_hold",
                "profile_scorer_agrees_with_selector": True,
                "profile_scorer_selected_score": 0.0,
                "profile_scorer_margin_to_second": 0.0,
                "profile_scorer_abs_delta_sum": 0.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm_rspc_v2.csv"
            _write_csv(path, rows)
            trace = audit_trace(path, delta_threshold=2.0)
            audit = audit_traces([tmp], delta_threshold=2.0)

        self.assertEqual(trace["profile_steps"], 4)
        self.assertEqual(trace["intent_regime_counts"]["hot_dry_relief"], 3)
        self.assertEqual(trace["selected_profile_counts"]["hot_dry_protect"], 2)
        self.assertEqual(trace["candidate_name_counts"]["shade_cooling"], 3)
        self.assertEqual(trace["scorer_steps"], 4)
        self.assertEqual(trace["scorer_selected_counts"]["hot_dry_protect"], 2)
        self.assertEqual(trace["scorer_selector_agreement_steps"], 4)
        self.assertEqual(len(trace["large_delta_windows"]), 1)
        self.assertEqual(len(trace["scorer_large_delta_windows"]), 1)
        self.assertEqual(trace["large_delta_windows"][0]["start_step"], 1)
        self.assertEqual(trace["large_delta_windows"][0]["end_step"], 2)
        self.assertEqual(audit["profile_trace_count"], 1)
        self.assertEqual(audit["scorer_trace_count"], 1)
        self.assertIn("Profile Generator Shadow Audit", build_report(audit))

    def test_warns_when_trace_has_no_profile_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm.csv"
            _write_csv(path, [{"step": 0, "reward": 0.0}])

            trace = audit_trace(path)

        self.assertEqual(trace["profile_steps"], 0)
        self.assertIn("missing_profile_generator_metadata", trace["warnings"])


if __name__ == "__main__":
    unittest.main()
