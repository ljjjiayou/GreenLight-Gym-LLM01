import unittest

from gl_gym.experiments.post_guardrail_static_provenance_audit import build_report


class TestPostGuardrailStaticProvenanceAudit(unittest.TestCase):
    def test_infers_heat_vent_conflict_and_requires_runtime_provenance(self):
        rewrite = {
            "rows": [
                {
                    "preset": "balanced",
                    "step": 221,
                    "pre_score_action": {"vent": 0.34, "heat": 0.25, "screen": 0.67},
                    "post_guardrail_action": {"vent": 0.0, "heat": 0.25, "screen": 0.67},
                    "delta_vent": -0.34,
                    "tomato_safety_v2_reasons": "",
                }
            ]
        }
        source = "if vpd >= 0.4:\n    guarded[3] = 0.0\n    guarded[3] = min(guarded[3], 0.1)\n"

        report = build_report(rewrite, source)

        self.assertTrue(report["runtime_provenance_instrumentation_required"])
        self.assertEqual(report["runtime_reason_missing_count"], 1)
        self.assertIn("apply_safety_guardrails_heat_vent_conflict", report["rows"][0]["likely_source_families"])
        self.assertIn("runtime_rule_reason_missing", report["rows"][0]["likely_source_families"])


if __name__ == "__main__":
    unittest.main()
