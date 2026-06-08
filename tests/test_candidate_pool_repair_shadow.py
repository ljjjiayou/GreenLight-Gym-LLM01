import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.candidate_pool_repair_shadow import audit_trace, build_hypothetical_candidates, build_report


def _candidate():
    return {
        "name": "selected",
        "selected": True,
        "score": 1.0,
        "action": {"heat": 0, "co2": 0, "screen": 0.8, "vent": 0.1, "lamp": 0, "shade": 0},
        "post_shape_action": {"heat": 0, "co2": 0, "screen": 0.8, "vent": 0.1, "lamp": 0, "shade": 0},
        "score_terms": {
            "temp_next": 24,
            "rh_next": 58,
            "vpd_next": 1.4,
            "dry_penalty": 1.0,
            "vpd_high_penalty": 0.3,
            "temp_penalty": 0,
            "dew_penalty": 0,
            "canopy_dew_penalty": 0,
        },
    }


def _row():
    return {
        "step": 220,
        "rspc_action_selected_name": "selected",
        "rspc_action_candidates_json": json.dumps([_candidate()]),
        "rh_air": 58,
        "vpd_air": 1.7,
        "temp_air": 24,
        "dew_margin_air": 1.2,
        "canopy_dew_margin": 0.4,
        "rh_low_violation": 0,
        "vpd_high_excess": 0.1,
        "glob_rad": 300,
        "u_screen": 0.8,
        "u_ventilation": 0.1,
        "u_heating": 0,
        "u_co2": 0,
        "u_lighting": 0,
        "u_shading": 0,
        "tomato_safety_v2_applied": False,
    }


def _write_trace(path: Path):
    rows = [_row(), {**_row(), "step": 221, "canopy_dew_margin": -0.1}]
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TestCandidatePoolRepairShadow(unittest.TestCase):
    def test_builds_feasible_hypothetical_candidate(self):
        candidates = build_hypothetical_candidates({**_row(), "canopy_dew_margin": 2.0, "dew_margin_air": 2.0})

        self.assertTrue(any(item["feasible"] for item in candidates))
        self.assertTrue(any(item["name"] == "canopy_safe_hold_screen" for item in candidates))
        self.assertTrue(any(item["name"] == "canopy_safe_balanced_repair" for item in candidates))
        self.assertIn("predicted_canopy_dew_margin_next_v2", candidates[0]["prediction"])
        self.assertIn("predicted_VPDhi_delta", candidates[0]["prediction"])
        self.assertIn("feasibility_reason", candidates[0])
        self.assertIn("candidate_family", candidates[0])
        self.assertIn("post_guardrail_shadow_prediction", candidates[0])
        self.assertIn("post_guardrail_remains_safe", candidates[0])

    def test_audit_reports_candidate_pool_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            _write_trace(path)
            report = audit_trace(trace_path=path, preset="p", scenario_id="s", start_step=220, end_step=220)

        self.assertEqual(report["counts"]["candidate_pool_gap"], 1)
        self.assertEqual(report["records"][0]["source_label"], "no_safe_candidate_available")
        self.assertIn("best_canopy_safe_candidate", report["records"][0])
        self.assertIn("candidate_pool_gap_confirmed", report["records"][0])
        self.assertIn("best_candidate_tradeoff_quality", report["records"][0])
        self.assertIn("post_guardrail_safe_repair_available", report["records"][0])
        self.assertIn("candidate_generation_shadow_conclusion", report)
        self.assertEqual(report["records"][0]["repair_conclusion"], "no_safe_repair_found")
        self.assertEqual(report["repair_conclusion"], "no_safe_repair_found")

    def test_useful_feasible_candidate_locks_shadow_evidence_conclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            row = {**_row(), "canopy_dew_margin": 2.0, "dew_margin_air": 2.0, "rh_air": 58, "vpd_air": 1.7}
            rows = [row, {**row, "step": 221, "canopy_dew_margin": 2.1}]
            fields = sorted(set().union(*(item.keys() for item in rows)))
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            report = audit_trace(trace_path=path, preset="p", scenario_id="s", start_step=220, end_step=220)

        self.assertEqual(report["repair_conclusion"], "new_candidate_needed_shadow_evidence")
        self.assertIn("candidate_generation_shadow_conclusion", report)
        self.assertEqual(report["counts"]["feasible_useful_dry_relief"], 1)

    def test_build_report_reads_nested_preset_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "traces"
            for preset in ("balanced", "hot_dry_relief"):
                preset_dir = root / preset
                preset_dir.mkdir(parents=True)
                _write_trace(preset_dir / "case.csv")

            report = build_report(trace_dirs=[root], scenario_id="case", start_step=220, end_step=220)

        self.assertEqual(report["trace_count"], 2)
        self.assertEqual(report["candidate_pool_gap_steps"], 2)
        self.assertTrue(report["candidate_pool_gap_confirmed"])
        self.assertIn("repair_quality_counts", report)
        self.assertIn("candidate_generation_shadow_conclusion", report)
        self.assertEqual(report["candidate_pool_repair_conclusion"], "no_safe_repair_found")
        self.assertEqual({trace["preset"] for trace in report["traces"]}, {"balanced", "hot_dry_relief"})


if __name__ == "__main__":
    unittest.main()
