import unittest

from gl_gym.experiments.mainline_diff_decision_packet import (
    build_expert_diff_review,
    build_metadata_replay_readiness_checklist,
    build_report,
)


class TestMainlineDiffDecisionPacket(unittest.TestCase):
    def test_marks_sensitive_diffs_without_modifying_files(self):
        cleanup = {
            "boundary_summary": {},
            "cleanup_items": [
                {
                    "path": "gl_gym/agent/expert_distillation.py",
                    "git_status": " M",
                    "category": "mainline_blocking",
                    "cleanup_recommendation": {"blocks_controlled_replay": True},
                },
                {
                    "path": "gl_gym/agent/llm_agent.py",
                    "git_status": " M",
                    "category": "controller_sensitive",
                    "cleanup_recommendation": {"blocks_controlled_replay": True},
                },
                {
                    "path": "gl_gym/experiments/run_frozen_benchmark.py",
                    "git_status": " M",
                    "category": "evaluation_sensitive",
                    "cleanup_recommendation": {"blocks_controlled_replay": True},
                },
            ],
        }
        report = build_report(cleanup)

        self.assertFalse(report["controlled_replay_allowed"])
        self.assertIn("gl_gym/agent/expert_distillation.py", report["blocking_paths"])
        decisions = {item["path"]: item["decision_required"] for item in report["decisions"]}
        self.assertEqual(decisions["gl_gym/agent/expert_distillation.py"], "requires_user_restore_or_authorization")
        self.assertEqual(decisions["gl_gym/agent/llm_agent.py"], "requires_controller_invariance_evidence")
        self.assertEqual(
            decisions["gl_gym/experiments/run_frozen_benchmark.py"],
            "requires_evaluation_protocol_isolation",
        )
        rows = {item["path"]: item for item in report["decisions"]}
        self.assertTrue(rows["gl_gym/agent/expert_distillation.py"]["requires_restore"])
        self.assertTrue(rows["gl_gym/agent/expert_distillation.py"]["requires_user_authorization"])
        self.assertTrue(rows["gl_gym/agent/llm_agent.py"]["requires_metadata_replay"])
        self.assertTrue(rows["gl_gym/experiments/run_frozen_benchmark.py"]["requires_protocol_isolation"])

    def test_expert_diff_review_records_restore_decision(self):
        report = build_report(
            {
                "boundary_summary": {},
                "cleanup_items": [
                    {
                        "path": "gl_gym/agent/expert_distillation.py",
                        "git_status": " M",
                        "category": "mainline_blocking",
                        "diff_excerpt": ["-old docstring", "+new docstring"],
                        "cleanup_recommendation": {"blocks_controlled_replay": True},
                    }
                ],
            }
        )

        review = build_expert_diff_review(report)

        self.assertTrue(review["diff_present"])
        self.assertEqual(review["recommended_decision"], "restore_to_head")
        self.assertFalse(review["belongs_to_current_mainline_closure_task"])

    def test_metadata_readiness_blocks_on_mainline_core(self):
        report = build_report(
            {
                "boundary_summary": {},
                "cleanup_items": [
                    {
                        "path": "docs/project_mainline_guidance_20260519.md",
                        "git_status": "??",
                        "category": "mainline_core",
                        "cleanup_recommendation": {"blocks_controlled_replay": True},
                    }
                ],
            }
        )

        checklist = build_metadata_replay_readiness_checklist(report)

        self.assertFalse(checklist["checks"]["mainline_core_authorized"])
        self.assertEqual(checklist["next_action"], "mainline_core_authorization_required")

    def test_authorized_mainline_core_no_longer_blocks_readiness(self):
        report = build_report(
            {
                "boundary_summary": {},
                "cleanup_items": [
                    {
                        "path": "docs/project_mainline_guidance_20260519.md",
                        "git_status": "??",
                        "category": "mainline_core",
                        "cleanup_recommendation": {"blocks_controlled_replay": True},
                    },
                    {
                        "path": "gl_gym/experiments/run_frozen_benchmark.py",
                        "git_status": " M",
                        "category": "evaluation_sensitive",
                        "cleanup_recommendation": {"blocks_controlled_replay": True},
                    },
                ],
            },
            authorized_mainline_core_paths={"docs/project_mainline_guidance_20260519.md"},
        )

        rows = {item["path"]: item for item in report["decisions"]}
        self.assertEqual(rows["docs/project_mainline_guidance_20260519.md"]["decision_required"], "mainline_core_authorized")
        self.assertFalse(rows["docs/project_mainline_guidance_20260519.md"]["blocks_controlled_replay"])
        checklist = build_metadata_replay_readiness_checklist(report)
        self.assertTrue(checklist["checks"]["mainline_core_authorized"])
        self.assertEqual(checklist["next_action"], "evaluation_protocol_isolation_required")


if __name__ == "__main__":
    unittest.main()
