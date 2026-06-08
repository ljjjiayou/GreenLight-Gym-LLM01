import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.cstcc_v815_shadow_analytics import (
    build_shadow_analytics,
    write_outputs,
)


ACTION_FIELDS = (
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
)


def _json(value):
    return json.dumps(value, sort_keys=True)


def _base_row(step=0, *, source_prior="rule_prior", template="relief_shift", risk=False):
    diff = {field: 0.0 for field in ACTION_FIELDS}
    runtime = {field: 0.2 for field in ACTION_FIELDS}
    shadow = {field: 0.2 for field in ACTION_FIELDS}
    return {
        "step": step,
        "hour_of_day": float(step) / 12.0,
        "phase": "day",
        "source": "rule",
        "temp_air": 22.0,
        "rh_air": 92.0 if risk else 70.0,
        "vpd_air": 0.30 if risk else 0.75,
        "dew_margin_air": 2.0,
        "canopy_dew_margin": 0.5 if risk else 4.0,
        "dew_risk": str(bool(risk)),
        "dry_risk": "False",
        "dry_vent_risk": "False",
        "cstcc_shadow_enabled": "True",
        "cstcc_shadow_audit_success": "True",
        "cstcc_shadow_final_action_invariant_verified": "True",
        "cstcc_shadow_final_action_changed": "False",
        "cstcc_shadow_online_llm_called": "False",
        "cstcc_shadow_predictive_rollout_executed": "False",
        "cstcc_shadow_real_tomato_safety_projection": "False",
        "cstcc_shadow_candidate_count": "5",
        "cstcc_shadow_feasible_candidate_count": "5",
        "cstcc_shadow_infeasible_candidate_ratio": "0.0",
        "cstcc_shadow_fallback_triggered": "False",
        "cstcc_shadow_audit_latency_ms": "5.0",
        "cstcc_shadow_audit_json_size_kb": "3.0",
        "cstcc_shadow_active_regime_after": "HUMIDITY_EXCESS_DISEASE_RISK" if risk else "NORMAL_BALANCED",
        "cstcc_shadow_effective_confidence": "0.8",
        "cstcc_shadow_llm_influence_alpha": "0.1",
        "cstcc_shadow_selected_source_prior": source_prior,
        "cstcc_shadow_selected_template_name": template,
        "cstcc_shadow_action_diff_json": _json(diff),
        "cstcc_shadow_current_runtime_final_action_json": _json(runtime),
        "cstcc_shadow_selected_first_action_json": _json(shadow),
        "cstcc_shadow_hard_constraint_violation_count": "0",
        "cstcc_shadow_hard_constraint_violation_reason_distribution_json": "{}",
        "cstcc_shadow_hard_constraint_violation_field_distribution_json": "{}",
    }


def _write_inputs(tmpdir, rows):
    root = Path(tmpdir)
    trace_csv = root / "trace.csv"
    summary_json = root / "summary.json"
    jsonl_root = root / "jsonl"
    jsonl_root.mkdir()
    keys = list(rows[0].keys())
    with trace_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary_json.write_text(
        _json({"trace_summary": {"runtime_error_steps": 0}}) + "\n",
        encoding="utf-8",
    )
    with (jsonl_root / "v81_episode_test.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json({"step": row["step"], "audit_success": True}) + "\n")
    return trace_csv, summary_json, jsonl_root


class TestCSTCCV815ShadowAnalytics(unittest.TestCase):
    def test_clean_synthetic_trace_routes_to_v82_and_builds_sections(self):
        rows = [
            _base_row(step=i, source_prior="rule_prior" if i % 2 else "conservative_prior", template="relief_shift")
            for i in range(12)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_csv, summary_json, jsonl_root = _write_inputs(tmpdir, rows)
            report, topk_rows, cross_rows = build_shadow_analytics(
                trace_csv=trace_csv,
                summary_json=summary_json,
                jsonl_root=jsonl_root,
            )

        self.assertEqual(report["next_action"], "v82_simulator_shadow_rollout_plan")
        self.assertEqual(report["boundary_runtime_safety"]["audit_success_rate"], 1.0)
        self.assertEqual(report["candidate_health"]["fallback_rate"], 0.0)
        self.assertIn("active_regime_distribution", report["regime_behavior"])
        self.assertEqual(report["action_diff_topk_summary"]["topk_row_count"], len(topk_rows))
        self.assertTrue(cross_rows)

    def test_max_delta_cross_tabs_and_template_concentration_route_to_constraint_calibration(self):
        rows = []
        for i in range(10):
            row = _base_row(step=i, source_prior="ppo_prior", template="jump_template")
            row["cstcc_shadow_hard_constraint_violation_count"] = "2"
            row["cstcc_shadow_hard_constraint_violation_reason_distribution_json"] = _json({"max_delta": 2})
            row["cstcc_shadow_hard_constraint_violation_field_distribution_json"] = _json({"u_ventilation": 2})
            rows.append(row)
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_csv, summary_json, jsonl_root = _write_inputs(tmpdir, rows)
            report, _topk_rows, cross_rows = build_shadow_analytics(
                trace_csv=trace_csv,
                summary_json=summary_json,
                jsonl_root=jsonl_root,
            )

        self.assertEqual(report["next_action"], "v816_template_constraint_calibration_plan")
        self.assertEqual(report["max_delta_attribution"]["by_template"]["jump_template"], 20)
        self.assertEqual(report["max_delta_attribution"]["by_field"]["u_ventilation"], 20)
        self.assertTrue(
            any(
                row["table"] == "max_delta_field_by_template"
                and row["key_1"] == "u_ventilation"
                and row["key_2"] == "jump_template"
                and row["count"] == 20
                for row in cross_rows
            )
        )

    def test_risk_hold_all_routes_to_level0_scorer_calibration(self):
        rows = [
            _base_row(step=i, source_prior="ppo_prior", template="hold_all", risk=True)
            for i in range(12)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_csv, summary_json, jsonl_root = _write_inputs(tmpdir, rows)
            report, _topk_rows, _cross_rows = build_shadow_analytics(
                trace_csv=trace_csv,
                summary_json=summary_json,
                jsonl_root=jsonl_root,
            )

        self.assertEqual(report["next_action"], "v816_level0_scorer_calibration_plan")
        self.assertGreater(report["risk_aligned_behavior"]["any_risk"]["hold_all_share"], 0.70)
        self.assertIn("risk_steps_hold_all_share_above_0_70", report["readiness_reasons"])

    def test_forbidden_boundary_event_stops_v82_admission(self):
        row = _base_row(step=0)
        row["cstcc_shadow_online_llm_called"] = "True"
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_csv, summary_json, jsonl_root = _write_inputs(tmpdir, [row])
            report, _topk_rows, _cross_rows = build_shadow_analytics(
                trace_csv=trace_csv,
                summary_json=summary_json,
                jsonl_root=jsonl_root,
            )

        self.assertEqual(report["next_action"], "v811_trace_schema_repair_or_rerun_plan")
        self.assertIn("forbidden_boundary:online_llm_called_steps=1", report["readiness_reasons"])

    def test_outputs_do_not_require_raw_or_projected_sequences(self):
        rows = [_base_row(step=i, source_prior="rule_prior", template="relief_shift") for i in range(3)]
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_csv, summary_json, jsonl_root = _write_inputs(tmpdir, rows)
            report, topk_rows, cross_rows = build_shadow_analytics(
                trace_csv=trace_csv,
                summary_json=summary_json,
                jsonl_root=jsonl_root,
            )
            outputs = write_outputs(
                report=report,
                topk_rows=topk_rows,
                cross_rows=cross_rows,
                output_prefix=Path(tmpdir) / "cstcc_v815_shadow_analytics_test",
            )
            with Path(outputs["topk_csv"]).open("r", encoding="utf-8", newline="") as handle:
                fields = csv.DictReader(handle).fieldnames or []
            self.assertTrue(Path(outputs["json"]).exists())
            self.assertTrue(Path(outputs["md"]).exists())

        self.assertNotIn("raw_sequence", fields)
        self.assertNotIn("projected_sequence", fields)


if __name__ == "__main__":
    unittest.main()
