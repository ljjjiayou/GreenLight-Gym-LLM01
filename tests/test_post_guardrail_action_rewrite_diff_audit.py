import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.post_guardrail_action_rewrite_diff_audit import build_report


FIELDS = [
    "step",
    "rspc_action_selected_name",
    "rspc_action_candidates_json",
    "temp_air",
    "rh_air",
    "vpd_air",
    "dew_margin_air",
    "canopy_dew_margin",
    "u_screen",
    "u_ventilation",
    "u_heating",
    "u_co2",
    "u_lighting",
    "u_shading",
    "tomato_safety_v2_applied",
    "tomato_safety_v2_reasons",
]


def _candidate():
    return {
        "name": "selected",
        "selected": True,
        "post_shape_action": {"heat": 0, "co2": 0, "screen": 0.2, "vent": 0.7, "lamp": 0, "shade": 0},
        "score_terms": {"predicted_canopy_dew_margin_next": 1.0},
    }


def _row(step, **extra):
    data = {
        "step": step,
        "rspc_action_selected_name": "selected",
        "rspc_action_candidates_json": json.dumps([_candidate()]),
        "temp_air": 20,
        "rh_air": 80,
        "vpd_air": 0.5,
        "dew_margin_air": 2,
        "canopy_dew_margin": 0.5,
        "u_screen": 0.8,
        "u_ventilation": 0.1,
        "u_heating": 0,
        "u_co2": 0,
        "u_lighting": 0,
        "u_shading": 0,
        "tomato_safety_v2_applied": True,
        "tomato_safety_v2_reasons": "test_override",
    }
    data.update(extra)
    return data


class TestPostGuardrailActionRewriteDiffAudit(unittest.TestCase):
    def test_outputs_harmful_rewrite_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "traces" / "balanced"
            root.mkdir(parents=True)
            rows = [_row(220), _row(221, canopy_dew_margin=-0.1)]
            with (root / "case_llm_rspc_v2.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            report = build_report(trace_dirs=[root.parent], scenario_id="case", start_step=220, end_step=220)

        self.assertEqual(report["harmful_override_count"], 1)
        self.assertEqual(report["root_cause_layer_counts"]["post_guardrail_destruction"], 1)
        row = report["rows"][0]
        self.assertEqual(row["root_cause_layer"], "post_guardrail_destruction")
        self.assertIn("pre_score_action", row)
        self.assertIn("post_guardrail_action", row)
        self.assertLess(row["delta_vent"], 0)
        self.assertEqual(row["actual_next_canopy_margin"], -0.1)


if __name__ == "__main__":
    unittest.main()
