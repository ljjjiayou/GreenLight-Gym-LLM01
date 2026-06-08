import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.trace_action_diff_audit import build_report, build_suite_report


FIELDS = ["step", "u_heating", "u_co2", "u_screen", "u_ventilation", "u_lighting", "u_shading"]


def _write_trace(path: Path, screen_delta: float = 0.0):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for step in range(3):
            writer.writerow(
                {
                    "step": step,
                    "u_heating": 0,
                    "u_co2": 0,
                    "u_screen": 0.2 + screen_delta,
                    "u_ventilation": 0.4,
                    "u_lighting": 0,
                    "u_shading": 0,
                }
            )


class TestTraceActionDiffAudit(unittest.TestCase):
    def test_reports_zero_action_diff_for_matching_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base" / "balanced"
            cmp = Path(tmp) / "cmp" / "balanced"
            base.mkdir(parents=True)
            cmp.mkdir(parents=True)
            _write_trace(base / "case.csv")
            _write_trace(cmp / "case.csv")

            report = build_report(
                baseline_trace_dir=base.parent,
                compare_trace_dir=cmp.parent,
                scenario_id="case",
                tolerance=1e-7,
            )

        self.assertEqual(report["action_diff_steps"], 0)
        self.assertEqual(report["missing_step_count"], 0)
        self.assertEqual(report["trace_count"], 1)

    def test_reports_action_diff_for_changed_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base" / "balanced"
            cmp = Path(tmp) / "cmp" / "balanced"
            base.mkdir(parents=True)
            cmp.mkdir(parents=True)
            _write_trace(base / "case.csv")
            _write_trace(cmp / "case.csv", screen_delta=0.1)

            report = build_report(
                baseline_trace_dir=base.parent,
                compare_trace_dir=cmp.parent,
                scenario_id="case",
                tolerance=1e-7,
            )

        self.assertEqual(report["action_diff_steps"], 3)
        self.assertGreater(report["max_abs_delta"], 0.0)

    def test_single_trace_pair_compares_when_parent_labels_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base" / "balanced"
            cmp = Path(tmp) / "cmp" / "traces"
            base.mkdir(parents=True)
            cmp.mkdir(parents=True)
            _write_trace(base / "case.csv")
            _write_trace(cmp / "case.csv")

            report = build_report(
                baseline_trace_dir=base.parent,
                compare_trace_dir=cmp,
                scenario_id="case",
                tolerance=1e-7,
            )

        self.assertEqual(report["trace_count"], 1)
        self.assertEqual(report["traces"][0]["label"], "single_trace_pair")
        self.assertEqual(report["traces"][0]["compared_steps"], 3)

    def test_build_suite_report_sums_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            for scenario in ("case_a", "case_b"):
                base = Path(tmp) / "base" / "balanced"
                cmp = Path(tmp) / "cmp" / "balanced"
                base.mkdir(parents=True, exist_ok=True)
                cmp.mkdir(parents=True, exist_ok=True)
                _write_trace(base / f"{scenario}.csv")
                _write_trace(cmp / f"{scenario}.csv")

            report = build_suite_report(
                baseline_trace_dir=Path(tmp) / "base",
                compare_trace_dir=Path(tmp) / "cmp",
                scenario_ids=["case_a", "case_b"],
                tolerance=1e-7,
            )

        self.assertEqual(report["scenario_count"], 2)
        self.assertEqual(report["action_diff_steps"], 0)


if __name__ == "__main__":
    unittest.main()
