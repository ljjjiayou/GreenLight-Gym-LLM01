import csv
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.controlled_canary_new_shadow_source_inventory import build_report


FIELDNAMES = [
    "step",
    "rspc_hot_dry_replay_enabled",
    "rspc_hot_dry_replay_would_apply",
    "rspc_hot_dry_replay_safe_hot_dry",
    "rspc_hot_dry_replay_safety_gate_reason",
    "rspc_hot_dry_replay_best_alignment",
    "rspc_hot_dry_replay_unsafe_preferred",
    "rspc_hot_dry_replay_unsafe_conflict",
    "rspc_hot_dry_replay_best_margin",
    "rspc_hot_dry_replay_best_candidate_name",
    "rspc_hot_dry_replay_best_variant",
    "rh_air",
    "vpd_air",
    "temp_air",
    "canopy_dew_margin",
]


def _write_trace(root: Path, scenario: str, row: dict):
    path = root / f"{scenario}_llm_rspc_v2.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)
    return path


def _strict_row():
    return {
        "step": 10,
        "rspc_hot_dry_replay_enabled": "true",
        "rspc_hot_dry_replay_would_apply": "true",
        "rspc_hot_dry_replay_safe_hot_dry": "true",
        "rspc_hot_dry_replay_safety_gate_reason": "none",
        "rspc_hot_dry_replay_best_alignment": "dry_benefit",
        "rspc_hot_dry_replay_unsafe_preferred": "false",
        "rspc_hot_dry_replay_unsafe_conflict": "false",
        "rspc_hot_dry_replay_best_margin": "0.25",
        "rspc_hot_dry_replay_best_candidate_name": "shadow_hot_dry_humidity_retention",
        "rspc_hot_dry_replay_best_variant": "dry_vpd_x2",
        "rh_air": "40.0",
        "vpd_air": "2.5",
        "temp_air": "29.0",
        "canopy_dew_margin": "3.5",
    }


class TestControlledCanaryNewShadowSourceInventory(unittest.TestCase):
    def test_inventory_finds_unexcluded_strict_shadow_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_trace(root, "y2021_d1_s1_n240", _strict_row())
            report = build_report(
                target_spec={"target_spec_ready": True, "excluded_scenario_ids": []},
                trace_inputs=[root],
                stress_manifest={"scenarios": [{"scenario_id": "y2021_d1_s1_n240"}]},
            )

        self.assertTrue(report["new_shadow_source_candidates_found"])
        self.assertEqual(report["candidate_scenario_ids"], ["y2021_d1_s1_n240"])
        self.assertEqual(report["candidate_scenarios"][0]["expected_strict_eligible_applied_steps"], 1)
        self.assertTrue(report["candidate_scenarios"][0]["source_hint_only"])
        self.assertFalse(report["cache_fill_run"])
        self.assertFalse(report["controlled_replay_allowed"])

    def test_inventory_excludes_used_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_trace(root, "y2021_d1_s1_n240", _strict_row())
            report = build_report(
                target_spec={"target_spec_ready": True, "excluded_scenario_ids": ["y2021_d1_s1_n240"]},
                trace_inputs=[root],
                stress_manifest={},
            )

        self.assertFalse(report["new_shadow_source_candidates_found"])
        self.assertEqual(report["candidate_scenario_count"], 0)

    def test_inventory_rejects_unsafe_source_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _strict_row()
            row["rspc_hot_dry_replay_unsafe_preferred"] = "true"
            _write_trace(root, "y2021_d1_s1_n240", row)
            report = build_report(
                target_spec={"target_spec_ready": True, "excluded_scenario_ids": []},
                trace_inputs=[root],
                stress_manifest={},
            )

        self.assertFalse(report["new_shadow_source_candidates_found"])


if __name__ == "__main__":
    unittest.main()
