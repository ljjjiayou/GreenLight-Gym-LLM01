import unittest

from gl_gym.experiments.default_path_shadow_import_audit import build_report


class TestDefaultPathShadowImportAudit(unittest.TestCase):
    def test_current_llm_agent_imports_intent_profile_modules(self):
        report = build_report(["gl_gym/agent/llm_agent.py"])

        self.assertFalse(report["metadata_replay_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["default_path_evidence_pass"])
        self.assertTrue(report["shadow_only_path_pending_default_path_evidence"])
        self.assertGreaterEqual(report["default_path_import_count"], 2)
        self.assertGreaterEqual(report["default_path_call_count"], 1)
        paths = {row["path"] for row in report["target_paths"]}
        self.assertIn("gl_gym/agent/intent_contract.py", paths)
        self.assertIn("gl_gym/agent/profile_generator.py", paths)

    def test_no_target_imports_passes_static_default_path_check(self):
        report = build_report(["gl_gym/agent/plan_intent.py"])

        self.assertTrue(report["default_path_evidence_pass"])
        self.assertFalse(report["needs_action_diff_invariance"])
        self.assertEqual(report["default_path_import_count"], 0)


if __name__ == "__main__":
    unittest.main()
