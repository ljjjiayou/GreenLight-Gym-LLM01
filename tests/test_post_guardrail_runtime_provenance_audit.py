import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.post_guardrail_runtime_provenance_audit import build_report, main


def _complete_record():
    return {
        "hook_id": "dry_recovery_override",
        "rule_id": "rspc_post_score_dry_recovery_vent_cap",
        "rule_family": "rspc_post_score_dry_recovery",
        "rule_reason": "dry side risk caps ventilation",
        "source_function": "_plan_control_step",
        "pre_rule_action": {"ventilation": 0.5},
        "post_rule_action": {"ventilation": 0.3},
        "delta_action": {"ventilation": -0.2},
        "input_features": {"rh_air": 50.0},
    }


class TestPostGuardrailRuntimeProvenanceAudit(unittest.TestCase):
    def test_no_new_trace_reports_metadata_replay_required(self):
        report = build_report(trace_payload=None, expected_missing_count=8, instrumentation_present=True)

        self.assertEqual(report["audit_status"], "instrumented_but_needs_metadata_replay")
        self.assertTrue(report["metadata_replay_required"])
        self.assertEqual(report["runtime_reason_missing_count"], 8)
        self.assertFalse(report["controlled_replay_allowed"])

    def test_complete_records_close_unknown_and_missing_reason_counts(self):
        trace = {
            "post_guardrail_runtime_provenance": [_complete_record()]
        }

        report = build_report(trace_payload=trace, expected_missing_count=8, instrumentation_present=True)

        self.assertEqual(report["audit_status"], "runtime_provenance_complete")
        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["unknown_post_guardrail_rewrite_count"], 0)
        self.assertEqual(report["runtime_reason_missing_count"], 0)

    def test_direct_records_are_not_counted_twice(self):
        report = build_report(
            trace_payload={"post_guardrail_runtime_provenance": [_complete_record()]},
            expected_missing_count=8,
            instrumentation_present=True,
        )

        self.assertEqual(report["record_count"], 1)
        self.assertEqual(len(report["rows"]), 1)

    def test_json_encoded_trace_record_list_is_parsed(self):
        report = build_report(
            trace_payload={"rows": [{"post_guardrail_runtime_provenance": json.dumps([_complete_record()])}]},
            expected_missing_count=8,
            instrumentation_present=True,
        )

        self.assertEqual(report["audit_status"], "runtime_provenance_complete")
        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["runtime_reason_missing_count"], 0)

    def test_missing_reason_blocks_runtime_provenance_closure(self):
        record = dict(_complete_record())
        record["rule_reason"] = ""

        report = build_report(
            trace_payload={"rows": [{"post_guardrail_runtime_provenance": [record]}]},
            expected_missing_count=8,
            instrumentation_present=True,
        )

        self.assertEqual(report["audit_status"], "runtime_provenance_incomplete")
        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["runtime_reason_missing_count"], 1)

    def test_empty_record_list_still_requires_metadata_replay(self):
        report = build_report(
            trace_payload={"rows": [{"post_guardrail_runtime_provenance": []}]},
            expected_missing_count=8,
            instrumentation_present=True,
        )

        self.assertEqual(report["audit_status"], "instrumented_but_needs_metadata_replay")
        self.assertTrue(report["metadata_replay_required"])
        self.assertEqual(report["record_count"], 0)

    def test_cli_reads_jsonl_trace_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.jsonl"
            output_json = root / "audit.json"
            output_md = root / "audit.md"
            trace.write_text(
                json.dumps({"post_guardrail_runtime_provenance": [_complete_record()]}) + "\n",
                encoding="utf-8",
            )

            rc = main(
                [
                    "--trace-jsonl",
                    str(trace),
                    "--instrumentation-present",
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ]
            )

            self.assertEqual(rc, 0)
            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(report["audit_status"], "runtime_provenance_complete")
            self.assertEqual(report["record_count"], 1)

    def test_cli_reads_csv_trace_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.csv"
            output_json = root / "audit.json"
            output_md = root / "audit.md"
            with trace.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["post_guardrail_runtime_provenance"])
                writer.writeheader()
                writer.writerow({"post_guardrail_runtime_provenance": json.dumps([_complete_record()])})

            rc = main(
                [
                    "--trace-csv",
                    str(trace),
                    "--instrumentation-present",
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ]
            )

            self.assertEqual(rc, 0)
            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(report["audit_status"], "runtime_provenance_complete")
            self.assertEqual(report["record_count"], 1)

    def test_cli_reads_trace_dir_jsonl_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "traces"
            trace_dir.mkdir()
            output_json = root / "audit.json"
            output_md = root / "audit.md"
            for name in ["a.jsonl", "b.jsonl"]:
                (trace_dir / name).write_text(
                    json.dumps({"post_guardrail_runtime_provenance": [_complete_record()]}) + "\n",
                    encoding="utf-8",
                )

            rc = main(
                [
                    "--trace-dir",
                    str(trace_dir),
                    "--instrumentation-present",
                    "--output-json",
                    str(output_json),
                    "--output-md",
                    str(output_md),
                ]
            )

            self.assertEqual(rc, 0)
            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(report["audit_status"], "runtime_provenance_complete")
            self.assertEqual(report["record_count"], 2)


if __name__ == "__main__":
    unittest.main()
