import unittest

from gl_gym.experiments.controlled_canary_shadow_cache_acquisition_target_spec import build_report


class TestControlledCanaryShadowCacheAcquisitionTargetSpec(unittest.TestCase):
    def test_target_spec_keeps_near_miss_as_source_hint_only(self):
        report = build_report(
            near_miss_catalog={
                "near_miss_catalog_ready": True,
                "near_miss_scenario_count": 7,
                "near_miss_row_count": 44,
                "excluded_scenario_ids": ["y2015_d120_s42_n240"],
                "source_scenarios": [{"scenario_id": "y2010_d180_s42_n240", "near_miss_rows": 11}],
            },
            existing_cache_inventory={
                "existing_cache_inventory_ready": True,
                "existing_cache_source_candidates_found": False,
                "candidate_scenario_count": 0,
            },
        )

        self.assertTrue(report["target_spec_ready"])
        self.assertTrue(report["source_hint_only"])
        self.assertTrue(report["near_miss_not_strict_applied_evidence"])
        self.assertFalse(report["cache_fill_authorized"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertIn("y2015_d120_s42_n240", report["excluded_scenario_ids"])
        self.assertIn("y2020_d59_s42_n240", report["excluded_scenario_ids"])

    def test_target_spec_blocks_when_existing_cache_has_candidate(self):
        report = build_report(
            near_miss_catalog={"near_miss_catalog_ready": True},
            existing_cache_inventory={
                "existing_cache_inventory_ready": True,
                "existing_cache_source_candidates_found": True,
                "candidate_scenario_count": 1,
            },
        )

        self.assertFalse(report["target_spec_ready"])


if __name__ == "__main__":
    unittest.main()
