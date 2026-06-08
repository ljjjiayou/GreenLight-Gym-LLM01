import unittest

from gl_gym.experiments.long_horizon_shadow_feasibility_readiness import build_report


class TestLongHorizonShadowFeasibilityReadiness(unittest.TestCase):
    def test_readiness_never_allows_controlled_or_promotion(self):
        report = build_report(
            manifest={
                "long_horizon_shadow_feasibility_manifest_ready": True,
                "primary_scenarios": [{} for _ in range(9)],
                "blocked_runtime_failed_scenarios": [{} for _ in range(9)],
            },
            execution_record={
                "long_horizon_shadow_feasibility_executable": True,
                "horizon_steps": 240,
                "next_action": "execute_v37_h240_shadow_feasibility",
                "failure_taxonomy": [],
            },
        )

        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["controlled_replay_execution_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])
        self.assertEqual(report["next_action"], "execute_v37_h240_shadow_feasibility")

    def test_result_audit_drives_next_action(self):
        report = build_report(
            manifest={"long_horizon_shadow_feasibility_manifest_ready": True},
            result_audit={
                "horizon_steps": 240,
                "metrics": {"horizon_pass_count": 0, "runtime_error_count": 2, "short_trace_count": 2},
                "failure_taxonomy": ["runtime_error"],
                "next_action": "long_horizon_shadow_feasibility_blocked_by_runtime_or_coverage",
            },
        )

        self.assertEqual(report["next_action"], "long_horizon_shadow_feasibility_blocked_by_runtime_or_coverage")
        self.assertIn("runtime_error", report["failure_taxonomy"])


if __name__ == "__main__":
    unittest.main()
