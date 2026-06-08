import unittest

from gl_gym.experiments.mainline_diff_cleanup_plan import build_report


class TestMainlineDiffCleanupPlan(unittest.TestCase):
    def test_marks_expert_distillation_as_user_cleanup_required(self):
        report = build_report(
            {
                "controlled_replay_blocked": True,
                "controlled_replay_block_reasons": ["mainline_blocking_diff_present"],
                "expert_distillation_status": "unrelated_diff_present",
                "changed_files": [
                    {
                        "path": "gl_gym/agent/expert_distillation.py",
                        "category": "mainline_blocking",
                        "git_status": " M",
                    }
                ],
            }
        )

        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["next_action"], "mainline_diff_cleanup_required")
        self.assertEqual(report["blocking_paths"], ["gl_gym/agent/expert_distillation.py"])
        rec = report["cleanup_items"][0]["cleanup_recommendation"]
        self.assertEqual(rec["cleanup_action"], "restore_or_explicitly_authorize")
        self.assertTrue(rec["blocks_controlled_replay"])

    def test_untracked_shadow_and_skill_files_do_not_block_controlled_replay(self):
        report = build_report(
            {
                "controlled_replay_blocked": False,
                "changed_files": [
                    {
                        "path": "gl_gym/experiments/example_audit.py",
                        "category": "shadow_audit",
                        "git_status": "??",
                    },
                    {
                        "path": "tests/test_example_audit.py",
                        "category": "test_only",
                        "git_status": "??",
                    },
                    {
                        "path": "codex_skills/greenhouse-mainline-guard/SKILL.md",
                        "category": "codex_skill",
                        "git_status": "??",
                    },
                ],
            }
        )

        self.assertFalse(report["blocking_paths"])
        actions = {
            item["path"]: item["cleanup_recommendation"]["cleanup_action"]
            for item in report["cleanup_items"]
        }
        self.assertEqual(actions["gl_gym/experiments/example_audit.py"], "keep_as_shadow_audit_candidate")
        self.assertEqual(actions["tests/test_example_audit.py"], "keep_as_test_only_candidate")
        self.assertEqual(actions["codex_skills/greenhouse-mainline-guard/SKILL.md"], "keep_as_codex_skill_tooling")


if __name__ == "__main__":
    unittest.main()
