import unittest

from gl_gym.experiments.mainline_diff_authorization_closure import build_report


def _instrumentation_status(metadata_replay_required=True):
    return {
        "runtime_provenance": {
            "audit_status": "instrumented_but_needs_metadata_replay",
            "metadata_replay_required": metadata_replay_required,
            "record_count": 0,
        }
    }


class TestMainlineDiffAuthorizationClosure(unittest.TestCase):
    def test_expert_distillation_blocks_closure_and_metadata_replay(self):
        report = build_report(
            authorization_status={
                "rows": [
                    {
                        "path": "gl_gym/agent/expert_distillation.py",
                        "category": "mainline_blocking",
                        "authorization_status": "not_authorized_for_promotion_evidence",
                        "blocks_controlled_replay": True,
                    }
                ]
            },
            instrumentation_status=_instrumentation_status(),
            action_diff={"action_diff_steps": 0, "max_abs_delta": 0.0},
        )

        self.assertFalse(report["mainline_diff_authorization_complete"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertEqual(report["next_action"], "expert_distillation_user_decision_required")
        self.assertEqual(
            report["rows"][0]["closure_status"],
            "not_closed_requires_restore_or_explicit_user_authorization",
        )

    def test_llm_agent_requires_metadata_replay_invariance(self):
        report = build_report(
            authorization_status={
                "rows": [
                    {
                        "path": "gl_gym/agent/llm_agent.py",
                        "category": "controller_sensitive",
                        "authorization_status": "controller_invariance_required",
                        "blocks_controlled_replay": True,
                    }
                ]
            },
            instrumentation_status=_instrumentation_status(metadata_replay_required=True),
            action_diff={"action_diff_steps": 0, "max_abs_delta": 0.0},
        )

        self.assertFalse(report["mainline_diff_authorization_complete"])
        self.assertEqual(report["next_action"], "controller_invariance_metadata_replay_required")
        self.assertEqual(
            report["rows"][0]["closure_status"],
            "controller_invariance_pending_metadata_replay",
        )

    def test_mainline_core_requires_authorization(self):
        report = build_report(
            authorization_status={
                "rows": [
                    {
                        "path": "docs/project_mainline_guidance_20260519.md",
                        "category": "mainline_core",
                        "authorization_status": "mainline_core_authorization_required",
                        "blocks_controlled_replay": True,
                    }
                ]
            },
            instrumentation_status=_instrumentation_status(metadata_replay_required=False),
            action_diff={"action_diff_steps": 0, "max_abs_delta": 0.0},
        )

        self.assertFalse(report["mainline_diff_authorization_complete"])
        self.assertEqual(report["next_action"], "mainline_core_authorization_required")
        self.assertEqual(report["rows"][0]["closure_status"], "mainline_core_authorization_pending")

    def test_authorized_mainline_core_closes(self):
        report = build_report(
            authorization_status={
                "rows": [
                    {
                        "path": "docs/project_mainline_guidance_20260519.md",
                        "category": "mainline_core",
                        "authorization_status": "mainline_core_authorized",
                        "blocks_controlled_replay": False,
                    }
                ]
            },
            instrumentation_status=_instrumentation_status(metadata_replay_required=False),
            action_diff={"action_diff_steps": 0, "max_abs_delta": 0.0},
        )

        self.assertTrue(report["mainline_diff_authorization_complete"])
        self.assertEqual(report["rows"][0]["closure_status"], "closed")

    def test_evaluation_sensitive_diff_requires_protocol_isolation(self):
        report = build_report(
            authorization_status={
                "rows": [
                    {
                        "path": "gl_gym/experiments/run_frozen_benchmark.py",
                        "category": "evaluation_sensitive",
                        "authorization_status": "evaluation_protocol_isolation_required",
                        "blocks_controlled_replay": True,
                    }
                ]
            },
            instrumentation_status=_instrumentation_status(metadata_replay_required=False),
            action_diff={"action_diff_steps": 0, "max_abs_delta": 0.0},
        )

        self.assertFalse(report["mainline_diff_authorization_complete"])
        self.assertFalse(report["metadata_replay_allowed"])
        self.assertEqual(report["next_action"], "evaluation_protocol_isolation_required")
        self.assertEqual(report["rows"][0]["closure_status"], "evaluation_protocol_isolation_pending")

    def test_evaluation_is_prioritized_before_controller_replay(self):
        report = build_report(
            authorization_status={
                "rows": [
                    {
                        "path": "gl_gym/agent/llm_agent.py",
                        "category": "controller_sensitive",
                        "authorization_status": "controller_invariance_required",
                        "blocks_controlled_replay": True,
                    },
                    {
                        "path": "gl_gym/experiments/frozen_benchmark_protocol.py",
                        "category": "evaluation_sensitive",
                        "authorization_status": "evaluation_protocol_isolation_required",
                        "blocks_controlled_replay": True,
                    },
                ]
            },
            instrumentation_status=_instrumentation_status(metadata_replay_required=True),
            action_diff={"action_diff_steps": 0, "max_abs_delta": 0.0},
        )

        self.assertFalse(report["mainline_diff_authorization_complete"])
        self.assertEqual(report["next_action"], "evaluation_protocol_isolation_required")

    def test_shadow_only_intent_profile_paths_block_until_default_path_evidence(self):
        report = build_report(
            authorization_status={
                "rows": [
                    {
                        "path": "gl_gym/agent/profile_generator.py",
                        "category": "controller_sensitive",
                        "authorization_status": "controller_invariance_required",
                        "blocks_controlled_replay": True,
                    }
                ]
            },
            instrumentation_status=_instrumentation_status(metadata_replay_required=False),
            action_diff={"action_diff_steps": 0, "max_abs_delta": 0.0},
        )

        self.assertFalse(report["mainline_diff_authorization_complete"])
        self.assertEqual(report["next_action"], "shadow_default_path_evidence_required")
        self.assertEqual(
            report["rows"][0]["closure_status"],
            "shadow_only_path_pending_default_path_evidence",
        )


if __name__ == "__main__":
    unittest.main()
