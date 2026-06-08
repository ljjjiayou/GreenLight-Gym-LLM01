import unittest

from gl_gym.experiments.mainline_diff_authorization_packet import build_report


class TestMainlineDiffAuthorizationPacket(unittest.TestCase):
    def test_expert_distillation_requires_user_authorization_without_cleanup(self):
        decision = {
            "decisions": [
                {
                    "path": "gl_gym/agent/expert_distillation.py",
                    "category": "mainline_blocking",
                    "git_status": "M",
                    "decision_required": "requires_user_restore_or_authorization",
                    "blocks_controlled_replay": True,
                }
            ]
        }

        report = build_report(decision)

        self.assertFalse(report["automated_cleanup_performed"])
        self.assertEqual(report["next_action"], "mainline_diff_authorization_required")
        self.assertEqual(report["authorization_counts"]["restore_or_user_authorization_required"], 1)
        self.assertIn("gl_gym/agent/expert_distillation.py", report["blocking_paths"])

    def test_mainline_core_authorized_is_propagated(self):
        decision = {
            "decisions": [
                {
                    "path": "docs/project_mainline_guidance_20260519.md",
                    "category": "mainline_core",
                    "git_status": "??",
                    "decision_required": "mainline_core_authorized",
                    "blocks_controlled_replay": False,
                }
            ]
        }

        report = build_report(decision)

        self.assertEqual(report["authorization_counts"]["mainline_core_authorized"], 1)
        self.assertEqual(report["blocking_paths"], [])


if __name__ == "__main__":
    unittest.main()
