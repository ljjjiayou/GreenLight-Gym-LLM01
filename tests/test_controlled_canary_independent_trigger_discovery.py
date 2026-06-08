import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.controlled_canary_independent_trigger_discovery import discover_independent_triggers


class TestControlledCanaryIndependentTriggerDiscovery(unittest.TestCase):
    def _write_trace(self, root: Path, name: str, rows: list[dict]) -> None:
        (root / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )

    def test_excludes_used_scenario_and_discovers_independent_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {
                "step": 10,
                "rspc_hot_dry_replay_enabled": True,
                "rspc_hot_dry_replay_would_apply": True,
                "rspc_hot_dry_proposer_control_strict_enabled": True,
                "rspc_hot_dry_proposer_control_applied": True,
                "rspc_hot_dry_proposer_control_safe_hot_dry": True,
                "rspc_hot_dry_proposer_control_safety_gate_reason": "none",
                "rspc_hot_dry_proposer_control_reason": "strict_eligible_applied",
            }
            self._write_trace(root, "y2015_d120_s42_n240_llm_rspc_v2_hot_dry_proposer_strict", [row])
            self._write_trace(root, "y2018_d120_s45_n240_llm_rspc_v2_hot_dry_proposer_strict", [row])

            report = discover_independent_triggers(
                [root],
                excluded_scenario_ids={"y2015_d120_s42_n240"},
            )

        self.assertTrue(report["independent_trigger_candidates_found"])
        self.assertEqual(report["qualified_trigger_scenarios"][0]["scenario_id"], "y2018_d120_s45_n240")
        self.assertNotIn("y2015_d120_s42_n240", [s["scenario_id"] for s in report["qualified_trigger_scenarios"]])

    def test_unsafe_source_metadata_blocks_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_trace(
                root,
                "y2018_d120_s45_n240_llm_rspc_v2_hot_dry_proposer_strict",
                [
                    {
                        "step": 10,
                        "rspc_hot_dry_replay_enabled": True,
                        "rspc_hot_dry_replay_would_apply": True,
                        "rspc_hot_dry_replay_unsafe_preferred": True,
                        "rspc_hot_dry_proposer_control_strict_enabled": True,
                        "rspc_hot_dry_proposer_control_applied": True,
                        "rspc_hot_dry_proposer_control_safe_hot_dry": True,
                        "rspc_hot_dry_proposer_control_safety_gate_reason": "none",
                    }
                ],
            )

            report = discover_independent_triggers([root], excluded_scenario_ids=set())

        self.assertFalse(report["independent_trigger_candidates_found"])
        self.assertIn("unsafe_source_metadata_present", report["blocker_taxonomy"])

    def test_strict_targeted_shadow_only_projects_current_strict_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_trace(
                root,
                "y2018_d120_s45_n240_llm_rspc_v2",
                [
                    {
                        "step": 10,
                        "temp_air": 28.8,
                        "rh_air": 40.0,
                        "vpd_air": 2.4,
                        "canopy_dew_margin": 5.0,
                        "rspc_hot_dry_replay_enabled": True,
                        "rspc_hot_dry_replay_would_apply": True,
                        "rspc_hot_dry_replay_safe_hot_dry": True,
                        "rspc_hot_dry_replay_safety_gate_reason": "none",
                        "rspc_hot_dry_replay_best_alignment": "dry_benefit",
                        "rspc_hot_dry_replay_best_margin": 0.25,
                        "rspc_hot_dry_replay_best_candidate_name": "shadow_hot_dry_humidity_retention",
                        "rspc_hot_dry_replay_best_variant": "dry_vpd_x2",
                    }
                ],
            )

            report = discover_independent_triggers(
                [root],
                excluded_scenario_ids=set(),
                strict_targeted_shadow_only=True,
            )

        self.assertTrue(report["independent_trigger_candidates_found"])
        self.assertEqual(report["expected_strict_eligible_applied_steps"], 1)
        self.assertEqual(report["qualified_trigger_scenarios"][0]["scenario_id"], "y2018_d120_s45_n240")

    def test_strict_targeted_shadow_only_blocks_near_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_trace(
                root,
                "y2018_d120_s45_n240_llm_rspc_v2",
                [
                    {
                        "step": 10,
                        "temp_air": 28.8,
                        "rh_air": 40.0,
                        "vpd_air": 2.4,
                        "canopy_dew_margin": 5.0,
                        "rspc_hot_dry_replay_enabled": True,
                        "rspc_hot_dry_replay_would_apply": True,
                        "rspc_hot_dry_replay_safe_hot_dry": True,
                        "rspc_hot_dry_replay_safety_gate_reason": "none",
                        "rspc_hot_dry_replay_best_alignment": "dry_benefit",
                        "rspc_hot_dry_replay_best_margin": 0.19,
                        "rspc_hot_dry_replay_best_candidate_name": "shadow_hot_dry_humidity_retention",
                    }
                ],
            )

            report = discover_independent_triggers(
                [root],
                excluded_scenario_ids=set(),
                strict_targeted_shadow_only=True,
            )

        self.assertFalse(report["independent_trigger_candidates_found"])
        self.assertIn("strict_candidates_filtered", report["blocker_taxonomy"])


if __name__ == "__main__":
    unittest.main()
