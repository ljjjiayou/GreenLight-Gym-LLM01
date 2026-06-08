import unittest
from types import SimpleNamespace

import numpy as np

from gl_gym.agent.llm_agent import build_post_guardrail_runtime_provenance_record


class TestPostGuardrailRuntimeProvenanceMetadata(unittest.TestCase):
    def test_dry_recovery_record_has_required_runtime_fields_without_mutating_actions(self):
        state = SimpleNamespace(
            timestep=233,
            temp_air=28.0,
            rh_air=50.0,
            dew_margin_air=3.0,
            canopy_dew_margin=0.2,
            glob_rad=500.0,
            hour_of_day=12.0,
            temp_out=25.0,
            wind_speed=1.0,
        )
        before = np.array([0.1, 0.0, 0.3, 0.55, 0.0, 0.2], dtype=np.float32)
        after = np.array([0.1, 0.0, 0.3, 0.30, 0.0, 0.2], dtype=np.float32)
        before_copy = before.copy()
        after_copy = after.copy()

        record = build_post_guardrail_runtime_provenance_record(
            state=state,
            hook_id="dry_recovery_override",
            source_function="_plan_control_step",
            pre_rule_action=before,
            post_rule_action=after,
            info={"vent_cap": 0.3, "hot_dry_vent_relief": False},
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["rule_id"], "rspc_post_score_dry_recovery_vent_cap")
        self.assertEqual(record["rule_family"], "rspc_post_score_dry_recovery")
        self.assertTrue(record["rule_reason"])
        self.assertEqual(record["source_function"], "_plan_control_step")
        self.assertAlmostEqual(record["delta_action"]["ventilation"], -0.25, places=6)
        self.assertFalse(record["control_action_changed_by_metadata"])
        np.testing.assert_array_equal(before, before_copy)
        np.testing.assert_array_equal(after, after_copy)

    def test_unknown_rewrite_is_explicitly_marked_as_blocking_missing_reason(self):
        state = SimpleNamespace(temp_air=20.0, rh_air=70.0)
        record = build_post_guardrail_runtime_provenance_record(
            state=state,
            hook_id="unknown_hook",
            source_function="unknown_source",
            pre_rule_action=np.zeros(6, dtype=np.float32),
            post_rule_action=np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float32),
        )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["rule_id"], "unknown_post_guardrail_rewrite")
        self.assertTrue(record["reason_missing"])
        self.assertEqual(record["rule_reason"], "missing_runtime_reason_blocker")


if __name__ == "__main__":
    unittest.main()
