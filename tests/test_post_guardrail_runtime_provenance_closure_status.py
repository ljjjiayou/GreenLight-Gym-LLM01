import unittest

from gl_gym.experiments.post_guardrail_runtime_provenance_closure_status import build_report


class TestPostGuardrailRuntimeProvenanceClosureStatus(unittest.TestCase):
    def test_closure_status_is_implementation_ready_but_blocks_rerun_execution(self):
        report = build_report(
            previous_readiness={
                "schema_version": "metadata_replay_readiness_checklist_v15",
                "stage_b_replay_executed": True,
                "failure_taxonomy": ["runtime_provenance_missing", "unknown_post_guardrail_rewrite"],
                "post_run_metrics": {
                    "runtime_provenance_record_count": 0,
                    "runtime_reason_missing_count": 8,
                    "unknown_post_guardrail_rewrite_count": 8,
                },
            },
            previous_runtime_audit={
                "audit_status": "instrumented_but_needs_metadata_replay",
                "record_count": 0,
            },
        )

        self.assertTrue(report["runtime_provenance_closure_implementation_ready"])
        self.assertTrue(report["runtime_provenance_trace_export_implemented"])
        self.assertTrue(report["audit_supports_trace_jsonl"])
        self.assertTrue(report["audit_supports_trace_csv"])
        self.assertFalse(report["metadata_replay_execution_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])
        self.assertFalse(report["performance_claim_allowed"])
        self.assertTrue(report["stage_b_rerun_authorization_required"])
        self.assertEqual(
            report["next_action"],
            "await_explicit_stage_b_rerun_authorization_after_runtime_provenance_closure",
        )
        self.assertEqual(report["previous_stage_b_failure"]["runtime_reason_missing_count"], 8)


if __name__ == "__main__":
    unittest.main()
