import unittest

from gl_gym.experiments.controlled_canary_expansion_wave1_execution_record import build_report


def manifest():
    return {
        "wave1_manifest_ready": True,
        "scenario_ids": ["y2020_d120_s44_n240"],
        "controllers": ["llm_rspc_v2", "llm_rspc_v2_hot_dry_proposer_strict"],
        "selected_cache_path": "cache.json",
        "command_groups": [
            {
                "group_id": "y2020_d120_s44_n240",
                "years": [2020],
                "days": [120],
                "seeds": [44],
                "expected_scenario_ids": ["y2020_d120_s44_n240"],
                "cartesian_product_safe": True,
            }
        ],
    }


def coverage(pass_value=True):
    return {
        "cache_coverage_pass": pass_value,
        "coverage_rate": 1.0 if pass_value else 0.5,
        "missing_key_count": 0,
        "missing_buffered_action_count": 0,
        "missing_parsed_plan_count": 0,
    }


class TestControlledCanaryExpansionWave1ExecutionRecord(unittest.TestCase):
    def test_execution_record_requires_coverage_and_fixed_scope(self):
        report = build_report(
            admission_review={
                "controlled_canary_expansion_admission_pass": True,
                "overlay_runner_path": "overlay/run_frozen_benchmark.py",
                "protocol_v1_overlay_root": "overlay",
            },
            manifest=manifest(),
            coverage=coverage(),
            authorization_source="user_delegated_controlled_canary_expansion_wave1_authorization_20260525",
            output_root="out",
        )

        self.assertTrue(report["wave1_controlled_canary_authorized"])
        self.assertEqual(report["controlled_replay_scope"], "controlled_canary_expansion_wave1_only")
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])
        self.assertIn("--plan-cache-strict", report["planned_commands"][0])

    def test_execution_record_blocks_on_missing_coverage(self):
        report = build_report(
            admission_review={"controlled_canary_expansion_admission_pass": True},
            manifest=manifest(),
            coverage=coverage(False),
            authorization_source="user_delegated_controlled_canary_expansion_wave1_authorization_20260525",
            output_root="out",
        )

        self.assertFalse(report["wave1_controlled_canary_authorized"])
        self.assertEqual(report["next_action"], "wave1_not_authorized")


if __name__ == "__main__":
    unittest.main()
