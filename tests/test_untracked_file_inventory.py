import unittest

from gl_gym.experiments.untracked_file_inventory import build_report


class TestUntrackedFileInventory(unittest.TestCase):
    def test_classifies_untracked_artifacts_and_blocks_promotion_evidence(self):
        report = build_report(
            [
                "docs/project_mainline_guidance_20260519.md",
                "gl_gym/agent/profile_generator.py",
                "gl_gym/experiments/example_audit.py",
                "tests/test_example.py",
                "gl_gym/result/audits/example.json",
                "codex_skills/greenhouse-mainline-guard/SKILL.md",
                "gl_gym/experiments/example_replay.py",
            ]
        )

        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["metadata_replay_allowed"])
        rows = {row["path"]: row for row in report["rows"]}
        self.assertEqual(rows["docs/project_mainline_guidance_20260519.md"]["category"], "mainline_core")
        self.assertEqual(rows["gl_gym/agent/profile_generator.py"]["category"], "controller_sensitive")
        self.assertEqual(rows["gl_gym/experiments/example_audit.py"]["category"], "shadow_audit")
        self.assertEqual(rows["tests/test_example.py"]["category"], "test_only")
        self.assertEqual(rows["gl_gym/result/audits/example.json"]["category"], "report_only")
        self.assertEqual(rows["codex_skills/greenhouse-mainline-guard/SKILL.md"]["category"], "codex_skill")
        self.assertEqual(rows["gl_gym/experiments/example_replay.py"]["category"], "deprecated_or_hold")
        self.assertFalse(rows["gl_gym/experiments/example_audit.py"]["can_be_promotion_evidence"])
        self.assertTrue(rows["gl_gym/agent/profile_generator.py"]["may_enter_default_import_path"])


if __name__ == "__main__":
    unittest.main()
