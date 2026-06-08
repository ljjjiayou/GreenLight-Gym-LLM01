import unittest

from gl_gym.experiments.evaluation_protocol_isolation_readiness import build_report


class TestEvaluationProtocolIsolationReadiness(unittest.TestCase):
    def test_modified_evaluation_protocol_blocks_promotion_evidence(self):
        report = build_report(
            {
                "gl_gym/experiments/frozen_benchmark_protocol.py": " M",
                "tests/test_frozen_benchmark.py": " M",
            }
        )

        self.assertFalse(report["metadata_replay_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["protocol_isolation_pass"])
        self.assertEqual(report["next_action"], "evaluation_protocol_isolation_required")
        self.assertIn("gl_gym/experiments/frozen_benchmark_protocol.py", report["pending_paths"])
        self.assertIn("action-diff field definitions and values", report["fields_must_match"])

    def test_clean_paths_do_not_create_protocol_pending_paths(self):
        report = build_report({})

        self.assertEqual(report["pending_paths"], [])
        self.assertEqual(report["next_action"], "controller_invariance_metadata_replay_required")
        self.assertFalse(report["metadata_replay_allowed"])


if __name__ == "__main__":
    unittest.main()
