import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.controlled_canary_strict_trigger_discovery import discover_strict_triggers


class TestControlledCanaryStrictTriggerDiscovery(unittest.TestCase):
    def _write_trace(self, root: Path, name: str, rows: list[dict]) -> Path:
        path = root / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_no_trigger_blocks_execution_even_with_would_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_trace(
                root,
                "y2015_d180_s44_n240_llm_rspc_v2_hot_dry_proposer_strict",
                [
                    {
                        "step": 12,
                        "rspc_hot_dry_replay_enabled": True,
                        "rspc_hot_dry_replay_would_apply": True,
                        "rspc_hot_dry_replay_reason": "controlled_shadow_replay_candidate",
                        "rspc_hot_dry_proposer_control_strict_enabled": True,
                        "rspc_hot_dry_proposer_control_applied": False,
                        "rspc_hot_dry_proposer_control_reason": "margin_below_strict_min",
                        "rspc_hot_dry_proposer_control_candidate": "shadow_hot_dry_shade_preempt",
                        "rspc_hot_dry_proposer_control_variant": "dry_vpd_x2",
                        "rspc_hot_dry_proposer_control_margin": 0.19,
                        "rspc_hot_dry_proposer_control_min_margin": 0.2,
                        "rspc_hot_dry_proposer_control_safe_hot_dry": True,
                        "rspc_hot_dry_proposer_control_safety_gate_reason": "none",
                    }
                ],
            )

            report = discover_strict_triggers([root])

        self.assertFalse(report["trigger_candidates_found"])
        self.assertEqual(report["expected_strict_eligible_applied_steps"], 0)
        self.assertIn("strict_candidates_filtered", report["blocker_taxonomy"])
        self.assertIn("only_shadow_would_apply_or_non_strict_signal", report["blocker_taxonomy"])

    def test_strict_eligible_applied_candidate_is_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_trace(
                root,
                "y2020_d180_s44_n240_llm_rspc_v2_hot_dry_proposer_strict",
                [
                    {
                        "step": 36,
                        "rspc_hot_dry_replay_enabled": True,
                        "rspc_hot_dry_replay_would_apply": True,
                        "rspc_hot_dry_proposer_control_strict_enabled": True,
                        "rspc_hot_dry_proposer_control_applied": True,
                        "rspc_hot_dry_proposer_control_reason": "strict_eligible_applied",
                        "rspc_hot_dry_proposer_control_candidate": "shadow_hot_dry_shade_preempt",
                        "rspc_hot_dry_proposer_control_variant": "dry_vpd_x2",
                        "rspc_hot_dry_proposer_control_margin": 0.33,
                        "rspc_hot_dry_proposer_control_min_margin": 0.2,
                        "rspc_hot_dry_proposer_control_safe_hot_dry": True,
                        "rspc_hot_dry_proposer_control_safety_gate_reason": "none",
                    }
                ],
            )

            report = discover_strict_triggers([root])

        self.assertTrue(report["trigger_candidates_found"])
        self.assertEqual(report["expected_strict_eligible_applied_steps"], 1)
        self.assertEqual(report["qualified_trigger_scenarios"][0]["scenario_id"], "y2020_d180_s44_n240")
        self.assertEqual(report["next_action"], "triggered_scenario_manifest_and_cache_coverage")

    def test_strict_targeted_shadow_source_is_discovered_from_baseline_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_trace(
                root,
                "y2015_d240_s42_n240_llm_rspc_v2",
                [
                    {
                        "step": 57,
                        "rspc_hot_dry_replay_enabled": True,
                        "rspc_hot_dry_replay_would_apply": True,
                        "rspc_hot_dry_replay_safe_hot_dry": True,
                        "rspc_hot_dry_replay_safety_gate_reason": "none",
                        "rspc_hot_dry_replay_best_alignment": "dry_benefit",
                        "rspc_hot_dry_replay_unsafe_preferred": False,
                        "rspc_hot_dry_replay_unsafe_conflict": False,
                        "rspc_hot_dry_replay_best_margin": 0.25,
                        "rspc_hot_dry_replay_best_candidate_name": "shadow_hot_dry_humidity_retention",
                        "rh_air": 44.0,
                        "vpd_air": 2.4,
                        "temp_air": 30.0,
                        "canopy_dew_margin": 3.2,
                    }
                ],
            )

            report = discover_strict_triggers([root], strict_targeted_shadow_only=True)

        self.assertTrue(report["strict_targeted_source_found"])
        self.assertEqual(report["schema_version"], "controlled_canary_strict_targeted_shadow_sourcing_v1")
        self.assertEqual(report["expected_strict_eligible_applied_steps"], 1)
        self.assertEqual(report["qualified_trigger_scenarios"][0]["discovery_role"], "strict_targeted_shadow_source")
        self.assertEqual(report["next_action"], "strict_targeted_source_manifest_and_cache_precheck")
        self.assertFalse(report["controlled_replay_allowed"])

    def test_strict_targeted_shadow_source_requires_all_strict_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_trace(
                root,
                "y2015_d240_s42_n240_llm_rspc_v2",
                [
                    {
                        "step": 57,
                        "rspc_hot_dry_replay_enabled": True,
                        "rspc_hot_dry_replay_would_apply": True,
                        "rspc_hot_dry_replay_safe_hot_dry": True,
                        "rspc_hot_dry_replay_safety_gate_reason": "none",
                        "rspc_hot_dry_replay_best_alignment": "dry_benefit",
                        "rspc_hot_dry_replay_unsafe_preferred": False,
                        "rspc_hot_dry_replay_unsafe_conflict": False,
                        "rspc_hot_dry_replay_best_margin": 0.19,
                        "rspc_hot_dry_replay_best_candidate_name": "shadow_hot_dry_shade_preempt",
                        "rh_air": 49.0,
                        "vpd_air": 2.1,
                        "temp_air": 31.0,
                        "canopy_dew_margin": 1.9,
                    }
                ],
            )

            report = discover_strict_triggers([root], strict_targeted_shadow_only=True)

        self.assertFalse(report["strict_targeted_source_found"])
        self.assertIn("strict_targeted_source_not_found", report["blocker_taxonomy"])
        row = report["sample_rows"][0]
        self.assertIn("margin_below_strict_min", row["failure_reasons"])
        self.assertIn("strict_candidate_mismatch", row["failure_reasons"])
        self.assertIn("not_severe_dry", row["failure_reasons"])
        self.assertIn("temp_headroom_low", row["failure_reasons"])
        self.assertIn("canopy_reserve_low", row["failure_reasons"])

    def test_strict_targeted_shadow_source_respects_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_trace(
                root,
                "y2015_d240_s42_n240_llm_rspc_v2",
                [
                    {
                        "step": 57,
                        "rspc_hot_dry_replay_enabled": True,
                        "rspc_hot_dry_replay_would_apply": True,
                        "rspc_hot_dry_replay_safe_hot_dry": True,
                        "rspc_hot_dry_replay_safety_gate_reason": "none",
                        "rspc_hot_dry_replay_best_alignment": "dry_benefit",
                        "rspc_hot_dry_replay_best_margin": 0.25,
                        "rspc_hot_dry_replay_best_candidate_name": "shadow_hot_dry_humidity_retention",
                        "rh_air": 44.0,
                        "vpd_air": 2.4,
                        "temp_air": 30.0,
                        "canopy_dew_margin": 3.2,
                    }
                ],
            )

            report = discover_strict_triggers(
                [root],
                strict_targeted_shadow_only=True,
                excluded_scenario_ids=["y2015_d240_s42_n240"],
            )

        self.assertFalse(report["strict_targeted_source_found"])
        self.assertEqual(report["excluded_trace_count"], 1)
        self.assertEqual(report["qualified_trigger_scenarios"], [])


if __name__ == "__main__":
    unittest.main()
