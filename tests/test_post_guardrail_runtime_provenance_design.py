import unittest

from gl_gym.experiments.post_guardrail_runtime_provenance_design import REQUIRED_FIELDS, build_report


class TestPostGuardrailRuntimeProvenanceDesign(unittest.TestCase):
    def test_design_lists_hooks_and_required_fields(self):
        report = build_report({"runtime_reason_missing_count": 8})

        self.assertTrue(report["runtime_provenance_instrumentation_pending"])
        self.assertIn("rule_id", report["required_fields"])
        self.assertIn("expected_safety_benefit", report["required_fields"])
        self.assertEqual(set(REQUIRED_FIELDS), set(report["required_fields"]))
        hook_ids = {hook["hook_id"] for hook in report["hook_map"]}
        self.assertIn("apply_safety_guardrails", hook_ids)
        self.assertIn("tomato_safety_v2_wrapper", hook_ids)
        self.assertFalse(report["controller_files_modified_by_this_design"])


if __name__ == "__main__":
    unittest.main()
