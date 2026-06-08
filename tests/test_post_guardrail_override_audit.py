import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.post_guardrail_override_audit import build_report, classify_override


def _row(**extra):
    candidate = {
        "name": "selected",
        "selected": True,
        "post_shape_action": {"heat": 0, "co2": 0, "screen": 0.3, "vent": 0.5, "lamp": 0, "shade": 0},
    }
    data = {
        "rspc_action_selected_name": "selected",
        "rspc_action_candidates_json": json.dumps([candidate]),
        "temp_air": 20,
        "rh_air": 80,
        "vpd_air": 0.5,
        "dew_margin_air": 2,
        "canopy_dew_margin": 0.5,
        "u_screen": 0.3,
        "u_ventilation": 0.5,
        "u_shading": 0,
        "u_heating": 0,
        "u_co2": 0,
        "u_lighting": 0,
        "tomato_safety_v2_applied": False,
        "tomato_safety_v2_reasons": "",
    }
    data.update(extra)
    return data


class TestPostGuardrailOverrideAudit(unittest.TestCase):
    def test_harmful_override_when_screen_up_vent_down_and_canopy_lt0(self):
        result = classify_override(
            _row(u_screen=0.8, u_ventilation=0.1),
            {"canopy_dew_margin": -0.1, "rh_air": 82, "vpd_air": 0.4, "temp_air": 20},
        )

        self.assertEqual(result["outcome_label"], "harmful_override")
        self.assertTrue(result["override_increased_screen"])
        self.assertTrue(result["override_decreased_ventilation"])

    def test_protective_override_when_screen_down_or_vent_up(self):
        result = classify_override(
            _row(u_screen=0.1, u_ventilation=0.8, tomato_safety_v2_applied=True),
            {"canopy_dew_margin": 0.7, "rh_air": 78, "vpd_air": 0.6, "temp_air": 20},
        )

        self.assertEqual(result["outcome_label"], "protective_override")

    def test_final_checker_marks_harmful_override_caught(self):
        result = classify_override(
            _row(
                u_screen=0.8,
                u_ventilation=0.1,
                final_action_would_fail_canopy_boundary=True,
                final_action_would_fail_canopy_boundary_v2=True,
                final_action_predicted_canopy_warning_v2=True,
                final_action_predicted_canopy_lt0=True,
                final_action_predicted_canopy_dew_margin_next_v2=-0.1,
                final_action_canopy_proxy_v2_risk_buffer=1.0,
                final_action_screen_vent_conflict=True,
                final_action_screen_vent_conflict_v2=True,
                final_action_high_screen_under_canopy_risk=True,
                final_action_low_ventilation_under_canopy_risk=True,
                final_action_risk_reason="canopy_dew_screen_vent_conflict",
                final_action_risk_reason_v2="canopy_dew_lead_time_screen_vent_conflict",
            ),
            {"canopy_dew_margin": -0.1, "rh_air": 82, "vpd_air": 0.4, "temp_air": 20},
        )

        self.assertEqual(result["outcome_label"], "harmful_override")
        self.assertTrue(result["harmful_override_caught_by_final_checker"])
        self.assertTrue(result["final_checker_would_fail_canopy_boundary"])
        self.assertTrue(result["harmful_override_preventable_by_proxy_v2"])
        self.assertTrue(result["harmful_override_preventable_by_proxy_v2_1"])
        self.assertTrue(result["harmful_override_explainable_by_proxy_v2_or_candidate_repair"])
        self.assertTrue(result["final_checker_warning_v2"])
        self.assertTrue(result["final_action_predicted_canopy_warning_v2_1"])
        self.assertIn("guardrail_introduced_risk", result["root_causes"])
        self.assertIn("pre_score_candidate_safe_but_post_guardrail_unsafe", result["fine_grained_root_causes"])
        self.assertIn("screen_vent_coupled_risk", result["fine_grained_root_causes"])
        self.assertTrue(result["pre_score_candidate_safe_but_post_guardrail_unsafe"])
        self.assertTrue(result["screen_vent_coupled_risk"])
        self.assertTrue(result["final_checker_high_screen_under_canopy_risk"])
        self.assertTrue(result["final_checker_low_ventilation_under_canopy_risk"])
        self.assertEqual(result["root_cause_layer"], "post_guardrail")
        self.assertIn("pre_score_selected_action", result)
        self.assertIn("post_guardrail_action", result)
        self.assertIn("pre_score_proxy_prediction", result)
        self.assertIn("post_guardrail_proxy_prediction", result)
        self.assertIn("actual_next_metrics", result)
        self.assertEqual(result["actual_next_metrics"]["canopy_dew_margin"], -0.1)

    def test_build_report_reads_nested_preset_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "traces"
            rows = [
                {"step": 220, **_row(u_screen=0.8, u_ventilation=0.1)},
                {"step": 221, **_row(canopy_dew_margin=-0.1)},
            ]
            fields = sorted(set().union(*(row.keys() for row in rows)))
            for preset in ("balanced", "hot_dry_relief"):
                preset_dir = root / preset
                preset_dir.mkdir(parents=True)
                with (preset_dir / "case.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)

            report = build_report(trace_dirs=[root], scenario_id="case", start_step=220, end_step=220)

        self.assertEqual(report["trace_count"], 2)
        self.assertEqual(report["aggregate_outcome_counts"]["harmful_override"], 2)
        self.assertIn("guardrail_introduced_risk", report["root_cause_summary"])
        self.assertEqual(report["harmful_override_count"], 2)
        self.assertEqual(report["harmful_override_preventable_by_proxy_v2_count"], 0)
        self.assertEqual(report["harmful_override_preventable_by_proxy_v2_1_count"], 0)
        self.assertIn("harmful_root_cause_combinations", report)
        self.assertIn("fine_grained_root_cause_summary", report)
        self.assertIn("harmful_fine_grained_root_cause_summary", report)
        self.assertIn("root_cause_layer", report["traces"][0]["rows"][0])
        self.assertIn("pre_score_proxy_prediction", report["traces"][0]["rows"][0])
        self.assertIn("pre_score_candidate_safe_but_post_guardrail_unsafe", report["fine_grained_root_cause_summary"])
        self.assertIn(
            "pre_score_candidate_safe_but_post_guardrail_unsafe",
            report["harmful_fine_grained_root_cause_summary"],
        )
        self.assertEqual({trace["preset"] for trace in report["traces"]}, {"balanced", "hot_dry_relief"})


if __name__ == "__main__":
    unittest.main()
