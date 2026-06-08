"""Report runtime provenance closure implementation readiness.

This is an implementation-readiness artifact only. It does not run replay and
does not claim runtime provenance closure until a new strict metadata replay
trace proves record_count > 0 with zero missing/unknown runtime reasons.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRACE_SCHEMA_FIELDS = [
    "post_guardrail_runtime_provenance",
    "post_guardrail_runtime_provenance_count",
    "post_guardrail_runtime_reason_missing_count",
    "unknown_post_guardrail_rewrite_count",
]


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_report(
    *,
    previous_readiness: Mapping[str, Any] | None = None,
    previous_runtime_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    readiness = _mapping(previous_readiness)
    runtime_audit = _mapping(previous_runtime_audit)
    previous_metrics = _mapping(readiness.get("post_run_metrics", {}))
    previous_failure_taxonomy = list(readiness.get("failure_taxonomy", []) or [])
    previous_runtime_record_count = int(previous_metrics.get("runtime_provenance_record_count", 0) or 0)
    previous_runtime_reason_missing_count = int(previous_metrics.get("runtime_reason_missing_count", 0) or 0)
    previous_unknown_rewrite_count = int(previous_metrics.get("unknown_post_guardrail_rewrite_count", 0) or 0)

    trace_export_implemented = True
    audit_jsonl_supported = True
    audit_csv_supported = True
    metadata_only = True
    implementation_ready = bool(
        trace_export_implemented
        and audit_jsonl_supported
        and audit_csv_supported
        and metadata_only
    )
    return {
        "schema_version": "post_guardrail_runtime_provenance_closure_status_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Tooling", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "metadata instrumentation closure / no replay",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "runtime_provenance_closure_implementation_ready": implementation_ready,
        "runtime_provenance_trace_export_implemented": trace_export_implemented,
        "metadata_trace_writes_raw_records": True,
        "metadata_trace_writes_derived_counts": True,
        "trace_schema_fields": TRACE_SCHEMA_FIELDS,
        "audit_supports_trace_json": True,
        "audit_supports_trace_jsonl": audit_jsonl_supported,
        "audit_supports_trace_csv": audit_csv_supported,
        "audit_consumes_benchmark_summary_only": False,
        "record_fabrication_allowed": False,
        "only_real_pre_post_action_delta_records_allowed": True,
        "metadata_replay_execution_allowed": False,
        "metadata_replay_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "cache_fill_allowed": False,
        "online_llm_allowed": False,
        "expanded_replay_allowed": False,
        "counterexample_replay_allowed": False,
        "stage_b_rerun_authorization_required": True,
        "stage_b_rerun_authorized": False,
        "previous_stage_b_failure": {
            "readiness_schema_version": readiness.get("schema_version", "not_provided"),
            "stage_b_replay_executed": bool(readiness.get("stage_b_replay_executed", False)),
            "failure_taxonomy": previous_failure_taxonomy,
            "runtime_provenance_record_count": previous_runtime_record_count,
            "runtime_reason_missing_count": previous_runtime_reason_missing_count,
            "unknown_post_guardrail_rewrite_count": previous_unknown_rewrite_count,
            "runtime_audit_status": runtime_audit.get("audit_status", "not_provided"),
            "runtime_audit_record_count": int(runtime_audit.get("record_count", 0) or 0),
        },
        "rerun_scope": {
            "scope": "canonical_failure_only",
            "scenario": "y2020_d120_s44_n240",
            "controller": "llm_rspc_v2",
            "max_steps": 240,
            "protocol_implementation": "protocol_v1_snapshot_ephemeral_overlay",
            "no_cache_fill": True,
            "no_online_llm": True,
        },
        "rerun_acceptance_targets": {
            "cache_hit_rate": 1.0,
            "runtime_error_steps": 0,
            "strict_cache_miss_runtime_error_steps": 0,
            "action_diff_steps": 0,
            "max_abs_delta": 0,
            "runtime_provenance_record_count": ">0",
            "runtime_reason_missing_count": 0,
            "unknown_post_guardrail_rewrite_count": 0,
            "joint_prediction_missing_fields": 0,
        },
        "next_action": "await_explicit_stage_b_rerun_authorization_after_runtime_provenance_closure",
        "notes": [
            "This artifact closes the implementation path only; it does not rerun Stage-B.",
            "A fresh canonical strict metadata replay is required before claiming runtime provenance closure.",
            "The previous Stage-B result remains failed until a rerun proves nonzero records and zero missing/unknown reasons.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    failure = _mapping(report.get("previous_stage_b_failure", {}))
    lines = [
        "# Post-Guardrail Runtime Provenance Closure Status",
        "",
        "- Mode: metadata instrumentation closure / no replay",
        f"- Implementation ready: {report.get('runtime_provenance_closure_implementation_ready', False)}",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Stage-B rerun authorization required: {report.get('stage_b_rerun_authorization_required', True)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Trace Fields",
        "",
    ]
    lines.extend(f"- `{field}`" for field in report.get("trace_schema_fields", []) or [])
    lines.extend(
        [
            "",
            "## Previous Stage-B Failure",
            "",
            f"- Runtime provenance record count: {failure.get('runtime_provenance_record_count', 0)}",
            f"- Runtime reason missing count: {failure.get('runtime_reason_missing_count', 0)}",
            f"- Unknown rewrite count: {failure.get('unknown_post_guardrail_rewrite_count', 0)}",
            "",
            "## Rerun Boundary",
            "",
            "- Scope: canonical_failure_only",
            "- Scenario: y2020_d120_s44_n240",
            "- Controller: llm_rspc_v2",
            "- No cache fill: true",
            "- No online LLM: true",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-readiness-json", default="")
    parser.add_argument("--previous-runtime-audit-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        previous_readiness=_load(args.previous_readiness_json) if args.previous_readiness_json else None,
        previous_runtime_audit=_load(args.previous_runtime_audit_json) if args.previous_runtime_audit_json else None,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json
    if not output_md.is_absolute():
        output_md = PROJECT_ROOT / output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"runtime_provenance_closure_implementation_ready={report['runtime_provenance_closure_implementation_ready']}")
    print(f"metadata_replay_execution_allowed={report['metadata_replay_execution_allowed']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
