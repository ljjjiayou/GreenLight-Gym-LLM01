import unittest

from gl_gym.experiments.stable_long_horizon_evaluation_pool_readiness import build_report


class TestStableLongHorizonEvaluationPoolReadiness(unittest.TestCase):
    def test_readiness_never_allows_controlled_replay_or_claims(self):
        report = build_report(
            manifest={"stable_long_horizon_manifest_ready": True, "metrics": {"candidate_count": 3}},
            execution_record={"stable_long_horizon_execution_executable": True, "horizon_steps": 240},
            result_audit={
                "horizon_steps": 240,
                "metrics": {"stable_scenario_count": 2, "cvodes_failure_count": 1},
                "failure_taxonomy": ["simulator_or_weather_numerical_instability"],
                "next_action": "build_v38_h720_execution_record",
            },
        )

        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["controlled_replay_execution_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertFalse(report["promotion_evidence"])
        self.assertEqual(report["stable_scenario_count"], 2)
        self.assertEqual(report["next_action"], "build_v38_h720_execution_record")


if __name__ == "__main__":
    unittest.main()
