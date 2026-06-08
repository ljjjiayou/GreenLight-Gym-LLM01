import unittest

from gl_gym.experiments.mainline_diff_boundary_audit import build_report, classify_path


class TestMainlineDiffBoundaryAudit(unittest.TestCase):
    def test_classifies_mainline_sensitive_paths(self):
        self.assertEqual(classify_path("gl_gym/agent/expert_distillation.py"), "mainline_blocking")
        self.assertEqual(classify_path("gl_gym/agent/llm_agent.py"), "controller_sensitive")
        self.assertEqual(
            classify_path("gl_gym/experiments/run_frozen_benchmark.py"),
            "evaluation_sensitive",
        )
        self.assertEqual(classify_path("gl_gym/experiments/candidate_generation_shadow_suite.py"), "shadow_audit")

    def test_classifies_untracked_file_groups(self):
        self.assertEqual(classify_path("docs/project_mainline_guidance_20260519.md"), "mainline_core")
        self.assertEqual(classify_path("codex_skills/greenhouse-mainline-guard/SKILL.md"), "codex_skill")
        self.assertEqual(classify_path("scripts/check_greenhouse_skills.py"), "codex_skill")
        self.assertEqual(classify_path("tests/test_check_greenhouse_skills.py"), "codex_skill")
        self.assertEqual(classify_path("tests/test_candidate_response_calibration.py"), "test_only")
        self.assertEqual(classify_path("gl_gym/result/audits/example.json"), "report_only")
        self.assertEqual(classify_path("gl_gym/experiments/example_replay.py"), "deprecated_or_hold")
        self.assertEqual(classify_path("gl_gym/experiments/example_audit.py"), "shadow_audit")

    def test_blocks_controlled_replay_for_sensitive_diffs_without_reverting(self):
        report = build_report(
            paths=[
                {"path": "gl_gym/agent/expert_distillation.py", "git_status": " M"},
                {"path": "gl_gym/experiments/run_frozen_benchmark.py", "git_status": " M"},
                {"path": "gl_gym/experiments/candidate_generation_shadow_suite.py", "git_status": "??"},
            ]
        )

        self.assertTrue(report["controlled_replay_blocked"])
        self.assertIn("mainline_blocking_diff_present", report["controlled_replay_block_reasons"])
        self.assertIn("evaluation_sensitive_diff_present", report["controlled_replay_block_reasons"])
        self.assertEqual(report["expert_distillation_status"], "unrelated_diff_present")
        self.assertEqual(report["evaluation_diff_status"], "requires_isolation_or_confirmation")


if __name__ == "__main__":
    unittest.main()
