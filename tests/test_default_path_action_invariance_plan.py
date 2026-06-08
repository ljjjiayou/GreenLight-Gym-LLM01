import unittest

from gl_gym.experiments.default_path_action_invariance_plan import build_report


class TestDefaultPathActionInvariancePlan(unittest.TestCase):
    def test_plan_ready_does_not_prove_action_invariance(self):
        report = build_report(
            {
                "calls": [
                    {"callee": "gl_gym.agent.intent_contract.intent_contract_from_setpoint_plan"},
                    {"callee": "gl_gym.agent.profile_generator.build_profile_generator_shadow_payload"},
                ],
                "target_paths": [
                    {"path": "gl_gym/agent/intent_contract.py"},
                    {"path": "gl_gym/agent/profile_generator.py"},
                ],
            }
        )

        self.assertTrue(report["plan_ready"])
        self.assertFalse(report["default_path_action_invariance_proven"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertIn("action_diff_steps = 0", report["required_evidence"])


if __name__ == "__main__":
    unittest.main()
