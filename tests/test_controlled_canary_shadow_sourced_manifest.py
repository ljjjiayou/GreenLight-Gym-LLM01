import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.controlled_canary_shadow_sourced_manifest import (
    PRIORITY_WINDOWS,
    build_report,
    build_v31_report,
)


FIELDNAMES = [
    "step",
    "rspc_hot_dry_replay_safe_hot_dry",
    "rspc_hot_dry_replay_safety_gate_reason",
    "rspc_hot_dry_replay_would_apply",
    "rspc_hot_dry_replay_best_alignment",
    "rspc_hot_dry_replay_unsafe_preferred",
    "rspc_hot_dry_replay_unsafe_conflict",
    "rspc_hot_dry_replay_best_margin",
    "rspc_hot_dry_replay_best_candidate_name",
    "rspc_hot_dry_proposer_control_strict_candidate",
    "rh_air",
    "vpd_air",
    "temp_air",
    "canopy_dew_margin",
    "rspc_hot_dry_replay_best_heat",
]


def _shadow_replay_with_priority_windows():
    return {
        "aggregate": {
            "top_replay_windows": [
                {"scenario_id": scenario, "start_step": start, "end_step": end}
                for scenario, start, end in PRIORITY_WINDOWS
            ]
        }
    }


def _write_trace(trace_dir: Path, scenario_id: str, rows):
    path = trace_dir / f"{scenario_id}_llm_rspc_v2.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


class TestControlledCanaryShadowSourcedManifest(unittest.TestCase):
    def test_manifest_selects_strict_eligible_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            _write_trace(
                trace_dir,
                "y2015_d240_s42_n240",
                [
                    {
                        "step": 57,
                        "rspc_hot_dry_replay_safe_hot_dry": "True",
                        "rspc_hot_dry_replay_safety_gate_reason": "none",
                        "rspc_hot_dry_replay_would_apply": "True",
                        "rspc_hot_dry_replay_best_alignment": "dry_benefit",
                        "rspc_hot_dry_replay_unsafe_preferred": "False",
                        "rspc_hot_dry_replay_unsafe_conflict": "False",
                        "rspc_hot_dry_replay_best_margin": "0.25",
                        "rspc_hot_dry_replay_best_candidate_name": "shadow_hot_dry_humidity_retention",
                        "rspc_hot_dry_proposer_control_strict_candidate": "shadow_hot_dry_humidity_retention",
                        "rh_air": "44.0",
                        "vpd_air": "2.4",
                        "temp_air": "30.0",
                        "canopy_dew_margin": "3.2",
                        "rspc_hot_dry_replay_best_heat": "1.0",
                    }
                ],
            )

            report = build_report(
                admission_review={"shadow_sourced_admission_pass": True},
                shadow_replay=_shadow_replay_with_priority_windows(),
                trace_dir=str(trace_dir),
                selected_cache_path="cache.json",
                output_root="out",
            )

        self.assertTrue(report["shadow_sourced_manifest_ready"])
        self.assertEqual(report["scenario_ids"], ["y2015_d240_s42_n240"])
        self.assertEqual(report["strict_projection"]["expected_strict_eligible_applied_steps"], 1)
        self.assertFalse(report["performance_claim_allowed"])

    def test_manifest_blocks_when_windows_are_only_near_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            _write_trace(
                trace_dir,
                "y2015_d180_s42_n240",
                [
                    {
                        "step": 40,
                        "rspc_hot_dry_replay_safe_hot_dry": "True",
                        "rspc_hot_dry_replay_safety_gate_reason": "none",
                        "rspc_hot_dry_replay_would_apply": "True",
                        "rspc_hot_dry_replay_best_alignment": "dry_benefit",
                        "rspc_hot_dry_replay_unsafe_preferred": "False",
                        "rspc_hot_dry_replay_unsafe_conflict": "False",
                        "rspc_hot_dry_replay_best_margin": "0.25",
                        "rspc_hot_dry_replay_best_candidate_name": "shadow_hot_dry_humidity_retention",
                        "rspc_hot_dry_proposer_control_strict_candidate": "shadow_hot_dry_humidity_retention",
                        "rh_air": "49.0",
                        "vpd_air": "2.3",
                        "temp_air": "31.0",
                        "canopy_dew_margin": "1.8",
                        "rspc_hot_dry_replay_best_heat": "1.0",
                    }
                ],
            )

            report = build_report(
                admission_review={"shadow_sourced_admission_pass": True},
                shadow_replay=_shadow_replay_with_priority_windows(),
                trace_dir=str(trace_dir),
                selected_cache_path="cache.json",
                output_root="out",
            )

        self.assertFalse(report["shadow_sourced_manifest_ready"])
        self.assertIn("no_shadow_sourced_strict_trigger_candidate", report["blocker_taxonomy"])
        self.assertIn("not_severe_dry", report["blocker_taxonomy"])
        self.assertIn("canopy_reserve_low", report["blocker_taxonomy"])

    def test_v31_manifest_accepts_only_v30_strict_source_rows(self):
        strict_rows = [
            {"scenario_id": "y2015_d120_s42_n240", "step": 147, "strict_targeted_source": True, "candidate": "shadow_hot_dry_humidity_retention"},
            {"scenario_id": "y2015_d120_s42_n240", "step": 149, "strict_targeted_source": True, "candidate": "shadow_hot_dry_humidity_retention"},
            {"scenario_id": "y2015_d120_s43_n240", "step": 147, "strict_targeted_source": True, "candidate": "shadow_hot_dry_humidity_retention"},
            {"scenario_id": "y2015_d120_s43_n240", "step": 149, "strict_targeted_source": True, "candidate": "shadow_hot_dry_humidity_retention"},
            {"scenario_id": "y2015_d120_s44_n240", "step": 141, "strict_targeted_source": True, "candidate": "shadow_hot_dry_humidity_retention"},
            {"scenario_id": "y2015_d120_s44_n240", "step": 143, "strict_targeted_source": True, "candidate": "shadow_hot_dry_humidity_retention"},
            {"scenario_id": "y2015_d120_s44_n240", "step": 145, "strict_targeted_source": True, "candidate": "shadow_hot_dry_humidity_retention"},
            {"scenario_id": "y2015_d120_s44_n240", "step": 147, "strict_targeted_source": True, "candidate": "shadow_hot_dry_humidity_retention"},
            {"scenario_id": "y2015_d120_s44_n240", "step": 149, "strict_targeted_source": True, "candidate": "shadow_hot_dry_humidity_retention"},
            {"scenario_id": "y2015_d240_s42_n240", "step": 57, "strict_targeted_source": False, "candidate": "shadow_hot_dry_humidity_retention"},
        ]
        report = build_v31_report(
            admission_review={"shadow_sourced_admission_pass": True},
            strict_sourcing={
                "strict_targeted_source_found": True,
                "expected_strict_eligible_applied_steps": 9,
                "sample_rows": strict_rows,
            },
            selected_cache_path="cache.json",
            output_root="out",
        )

        self.assertTrue(report["shadow_sourced_manifest_ready"])
        self.assertEqual(
            report["scenario_ids"],
            ["y2015_d120_s42_n240", "y2015_d120_s43_n240", "y2015_d120_s44_n240"],
        )
        self.assertEqual(len(report["strict_source_rows"]), 9)
        self.assertEqual(report["strict_projection"]["expected_strict_eligible_applied_steps"], 9)
        self.assertEqual(report["command_groups"][0]["years"], [2015])
        self.assertFalse(report["performance_claim_allowed"])

    def test_v31_manifest_blocks_unexpected_backlog_scenario(self):
        strict_rows = [
            {"scenario_id": "y2020_d120_s42_n240", "step": 147, "strict_targeted_source": True},
        ]
        report = build_v31_report(
            admission_review={"shadow_sourced_admission_pass": True},
            strict_sourcing={
                "strict_targeted_source_found": True,
                "expected_strict_eligible_applied_steps": 9,
                "sample_rows": strict_rows,
            },
            selected_cache_path="cache.json",
            output_root="out",
        )

        self.assertFalse(report["shadow_sourced_manifest_ready"])
        self.assertIn("v30_strict_source_rows_not_exact", report["blocker_taxonomy"])


if __name__ == "__main__":
    unittest.main()
