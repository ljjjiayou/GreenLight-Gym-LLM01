import unittest

from gl_gym.experiments.controlled_canary_shadow_sourced_execution_record import build_report


class TestControlledCanaryShadowSourcedExecutionRecord(unittest.TestCase):
    def _base_inputs(self):
        return {
            "admission_review": {"shadow_sourced_admission_pass": True},
            "manifest": {
                "shadow_sourced_manifest_ready": True,
                "controllers": ["llm_rspc_v2", "llm_rspc_v2_hot_dry_proposer_strict"],
                "scenario_ids": ["y2015_d240_s42_n240"],
                "selected_cache_path": "cache.json",
                "command_groups": [
                    {
                        "group_id": "y2015_d240_s42_n240",
                        "years": [2015],
                        "days": [240],
                        "seeds": [42],
                    }
                ],
            },
            "cache_coverage": {
                "cache_coverage_pass": True,
                "coverage_rate": 1.0,
                "missing_key_count": 0,
                "missing_buffered_action_count": 0,
                "missing_parsed_plan_count": 0,
            },
            "overlay_preflight": {
                "controlled_canary_overlay_validated": True,
                "controlled_canary_runner_surface_validated": True,
                "protocol_hash_mismatch_count": 0,
                "overlay_runner_path": "gl_gym/experiments/controlled_canary_shadow_sourced_execution_record.py",
                "protocol_v1_overlay_root": "gl_gym/result/audits/protocol_v1_controlled_canary_overlay_20260525",
            },
            "authorization_source": "user_delegated_shadow_sourced_controlled_canary_authorization_20260526",
            "output_root": "out",
        }

    def test_execution_record_authorizes_only_fixed_scope(self):
        report = build_report(**self._base_inputs())

        self.assertTrue(report["shadow_sourced_controlled_canary_authorized"])
        command = report["planned_commands"][0]
        self.assertIn("--plan-cache-mode", command)
        self.assertIn("replay", command)
        self.assertIn("--plan-cache-strict", command)
        self.assertIn("--plan-cache-key-policy", command)
        self.assertIn("scenario_timestep", command)
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])

    def test_execution_record_blocks_without_manifest_ready(self):
        inputs = self._base_inputs()
        inputs["manifest"] = dict(inputs["manifest"], shadow_sourced_manifest_ready=False)
        report = build_report(**inputs)

        self.assertFalse(report["shadow_sourced_controlled_canary_authorized"])
        self.assertFalse(report["controlled_replay_execution_allowed"])

    def test_v31_execution_record_uses_exact_controller_pair_and_scope(self):
        inputs = self._base_inputs()
        inputs["manifest"] = dict(
            inputs["manifest"],
            canary_scope="strict_source_controlled_canary_v31_only",
            readiness_schema_version="metadata_replay_readiness_checklist_v31",
            scenario_ids=["y2015_d120_s42_n240", "y2015_d120_s43_n240", "y2015_d120_s44_n240"],
            command_groups=[
                {
                    "group_id": "v31_y2015_d120_s42_s43_s44",
                    "years": [2015],
                    "days": [120],
                    "seeds": [42, 43, 44],
                    "output_json_name": "strict_source_controlled_canary_v31_y2015_d120_s42_s43_s44_20260528.json",
                }
            ],
        )
        report = build_report(**inputs)

        self.assertTrue(report["shadow_sourced_controlled_canary_authorized"])
        self.assertEqual(report["schema_version"], "controlled_canary_shadow_sourced_execution_record_v31")
        self.assertEqual(report["controlled_replay_scope"], "strict_source_controlled_canary_v31_only")
        self.assertEqual(report["controllers"], ["llm_rspc_v2", "llm_rspc_v2_hot_dry_proposer_strict"])
        self.assertIn(
            "strict_source_controlled_canary_v31_y2015_d120_s42_s43_s44_20260528.json",
            " ".join(report["planned_commands"][0]),
        )

    def test_execution_record_rejects_extra_controller(self):
        inputs = self._base_inputs()
        inputs["manifest"] = dict(inputs["manifest"], controllers=["llm_rspc_v2", "extra_controller"])
        report = build_report(**inputs)

        self.assertFalse(report["shadow_sourced_controlled_canary_authorized"])
        self.assertFalse(report["controlled_replay_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
