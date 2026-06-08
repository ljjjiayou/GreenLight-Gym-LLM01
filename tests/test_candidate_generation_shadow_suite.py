import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.candidate_generation_shadow_suite import build_report


FIELDS = [
    "step",
    "rspc_action_selected_name",
    "rspc_action_candidates_json",
    "rh_air",
    "vpd_air",
    "temp_air",
    "dew_margin_air",
    "canopy_dew_margin",
    "rh_low_violation",
    "vpd_high_excess",
    "glob_rad",
    "u_screen",
    "u_ventilation",
    "u_heating",
    "u_co2",
    "u_lighting",
    "u_shading",
    "canopy_boundary_shadow_predicted_canopy_dew_margin_next",
    "final_action_predicted_canopy_dew_margin_next_v2",
]


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
            "canopy_dew_penalty": 0,
            "dew_penalty": 0,
        },
    }


def _safe_candidate():
    return {
        "name": "safe",
        "selected": False,
        "score": 0.9,
        "action": {"heat": 0, "co2": 0, "screen": 0.1, "vent": 0.1, "lamp": 0, "shade": 0},
        "post_shape_action": {"heat": 0, "co2": 0, "screen": 0.1, "vent": 0.1, "lamp": 0, "shade": 0},
        "score_terms": {
            "temp_next": 24,
            "rh_next": 59,
            "vpd_next": 1.3,
            "dry_penalty": 0.5,
            "vpd_high_penalty": 0.1,
            "canopy_dew_penalty": 0,
            "dew_penalty": 0,
        },
    }


def _row(step: int, **extra):
    data = {
        "step": step,
        "rspc_action_selected_name": "selected",
        "rspc_action_candidates_json": json.dumps([_candidate()]),
        "rh_air": 58,
        "vpd_air": 1.7,
        "temp_air": 24,
        "dew_margin_air": 2.0,
        "canopy_dew_margin": 2.0,
        "rh_low_violation": 0.1,
        "vpd_high_excess": 0.2,
        "glob_rad": 300,
        "u_screen": 0.5,
        "u_ventilation": 0.0,
        "u_heating": 0,
        "u_co2": 0,
        "u_lighting": 0,
        "u_shading": 0,
        "canopy_boundary_shadow_predicted_canopy_dew_margin_next": 0.3,
        "final_action_predicted_canopy_dew_margin_next_v2": 0.1,
    }
    data.update(extra)
    return data


def _write_trace(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class TestCandidateGenerationShadowSuite(unittest.TestCase):
    def test_reports_canonical_gap_and_pure_hot_dry_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "traces" / "balanced"
            root.mkdir(parents=True)
            _write_trace(
                root / "canonical_llm_rspc_v2.csv",
                [
                    _row(0, canopy_dew_margin=2.0, dew_margin_air=2.0),
                    _row(1, canopy_dew_margin=2.1, dew_margin_air=2.0),
                ],
            )
            _write_trace(
                root / "pure_llm_rspc_v2.csv",
                [
                    _row(
                        0,
                        canopy_dew_margin=2.0,
                        dew_margin_air=2.0,
                        rspc_action_candidates_json=json.dumps([_candidate(), _safe_candidate()]),
                    ),
                    _row(
                        1,
                        canopy_dew_margin=2.1,
                        dew_margin_air=2.0,
                        rspc_action_candidates_json=json.dumps([_candidate(), _safe_candidate()]),
                    ),
                ],
            )

            report = build_report(
                trace_dirs=[root.parent],
                windows=[
                    ("canonical_failure", "canonical", 0, 0),
                    ("pure_hot_dry_false_positive", "pure", 0, 0),
                ],
            )

        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertIn("candidate_pool_gap", report["aggregate_counts"])
        self.assertIn("repair_false_positive_blocker", report["aggregate_counts"])
        self.assertGreaterEqual(report["pure_hot_dry_false_positive_blocker_count"], 1)
        self.assertIn("candidate_generation_shadow_conclusion", report)


if __name__ == "__main__":
    unittest.main()
