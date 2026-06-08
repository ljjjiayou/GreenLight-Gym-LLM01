import unittest

from gl_gym.experiments.mainline_diff_authorization_status import build_report


class TestMainlineDiffAuthorizationStatus(unittest.TestCase):
    def test_expert_distillation_is_not_authorized_for_promotion_evidence(self):
        packet = {
            "rows": [
                {
                    "path": "gl_gym/agent/expert_distillation.py",
                    "category": "mainline_blocking",
                    "git_status": "M",
                    "authorization_required": "restore_or_user_authorization_required",
                    "blocks_controlled_replay": True,
                },
                {
                    "path": "gl_gym/agent/llm_agent.py",
                    "category": "controller_sensitive",
                    "git_status": "M",
                    "authorization_required": "controller_invariance_required",
                    "blocks_controlled_replay": True,
                },
            ]
        }

        report = build_report(packet)

        self.assertFalse(report["authorization_complete"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["next_action"], "mainline_diff_authorization_required")
        self.assertIn("gl_gym/agent/expert_distillation.py", report["blocking_paths"])
        self.assertEqual(report["status_counts"]["not_authorized_for_promotion_evidence"], 1)
        self.assertFalse(report["automated_cleanup_performed"])

    def test_mainline_core_authorized_is_not_blocking(self):
        packet = {
            "rows": [
                {
                    "path": "docs/project_mainline_guidance_20260519.md",
                    "category": "mainline_core",
                    "git_status": "??",
                    "authorization_required": "mainline_core_authorized",
                    "blocks_controlled_replay": False,
                }
            ]
        }

        report = build_report(packet)

        self.assertTrue(report["authorization_complete"])
        self.assertEqual(report["status_counts"]["mainline_core_authorized"], 1)
        self.assertEqual(report["blocking_paths"], [])


if __name__ == "__main__":
    unittest.main()
