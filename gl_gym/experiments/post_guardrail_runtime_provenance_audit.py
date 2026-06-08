"""Audit runtime post-guardrail provenance metadata.

When no metadata replay trace is provided, the report explicitly stays at
``instrumented_but_needs_metadata_replay`` instead of claiming runtime closure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_RECORD_FIELDS = [
    "hook_id",
    "rule_id",
    "rule_family",
    "rule_reason",
    "source_function",
    "pre_rule_action",
    "post_rule_action",
    "delta_action",
    "input_features",
]


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_csv(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_trace_dir(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    rows: list[dict[str, Any]] = []
    for trace in sorted(p.glob("*.jsonl")):
        rows.extend(_load_jsonl(trace))
    return rows


def _decode_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return [item for item in decoded if isinstance(item, dict)]
    return []


def _collect_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        direct = value.get("post_guardrail_runtime_provenance")
        records.extend(_decode_records(direct))
        if isinstance(value.get("records"), list):
            records.extend(item for item in value["records"] if isinstance(item, dict))
        for key, item in value.items():
            if key in {"post_guardrail_runtime_provenance", "records"}:
                continue
            if isinstance(item, (Mapping, list)):
                records.extend(_collect_records(item))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (Mapping, list)):
                records.extend(_collect_records(item))
    return records


def _record_missing_fields(record: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for field in REQUIRED_RECORD_FIELDS:
        value = record.get(field)
        if value is None or value == "" or value == {}:
            missing.append(field)
    return missing


def build_report(
    *,
    trace_payload: Mapping[str, Any] | None = None,
    expected_missing_count: int = 8,
    instrumentation_present: bool = True,
) -> dict[str, Any]:
    records = _collect_records(trace_payload or {}) if trace_payload is not None else []
    row_reports: list[dict[str, Any]] = []
    unknown_count = 0
    reason_missing_count = 0
    for record in records:
        missing = _record_missing_fields(record)
        rule_id = str(record.get("rule_id", "") or "")
        reason_missing = bool(record.get("reason_missing", False)) or bool(missing)
        if rule_id == "unknown_post_guardrail_rewrite" or str(record.get("rule_family", "")).endswith("_unknown"):
            unknown_count += 1
        if reason_missing:
            reason_missing_count += 1
        row_reports.append(
            {
                "hook_id": record.get("hook_id", ""),
                "rule_id": rule_id,
                "rule_family": record.get("rule_family", ""),
                "rule_reason": record.get("rule_reason", ""),
                "source_function": record.get("source_function", ""),
                "missing_fields": missing,
                "reason_missing": reason_missing,
            }
        )
    if not records:
        audit_status = (
            "instrumented_but_needs_metadata_replay"
            if instrumentation_present
            else "instrumentation_not_confirmed"
        )
        reason_missing_count = int(expected_missing_count)
        unknown_count = int(expected_missing_count)
    else:
        audit_status = "runtime_provenance_complete" if reason_missing_count == 0 and unknown_count == 0 else "runtime_provenance_incomplete"
    return {
        "schema_version": "post_guardrail_runtime_provenance_audit_v1",
        "mainline_alignment": {
            "affected_layers": ["Safety Boundary", "Evaluation"],
            "default_llm_rspc_v2_changed": False,
            "mode": "metadata-only provenance audit",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "audit_status": audit_status,
        "instrumentation_present": bool(instrumentation_present),
        "metadata_replay_required": not bool(records),
        "record_count": len(records),
        "unknown_post_guardrail_rewrite_count": int(unknown_count),
        "runtime_reason_missing_count": int(reason_missing_count),
        "target_unknown_post_guardrail_rewrite_count": 0,
        "target_runtime_reason_missing_count": 0,
        "ready_for_promotion_evidence": False,
        "rows": row_reports,
        "notes": [
            "Runtime provenance fields are metadata-only and must not change the final action.",
            "If record_count is zero, a strict metadata replay is still required before claiming runtime closure.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Post-Guardrail Runtime Provenance Audit",
            "",
            "- Mode: metadata-only provenance audit",
            "- Controlled replay allowed: false",
            "- Performance claim allowed: false",
            f"- Audit status: `{report.get('audit_status', '')}`",
            f"- Record count: {report.get('record_count', 0)}",
            f"- Unknown rewrite count: {report.get('unknown_post_guardrail_rewrite_count', 0)}",
            f"- Runtime reason missing count: {report.get('runtime_reason_missing_count', 0)}",
            f"- Metadata replay required: {report.get('metadata_replay_required', True)}",
            "",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", default="")
    parser.add_argument("--trace-jsonl", default="")
    parser.add_argument("--trace-csv", default="")
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--expected-missing-count", type=int, default=8)
    parser.add_argument("--instrumentation-present", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    provided = [bool(args.trace_json), bool(args.trace_jsonl), bool(args.trace_csv), bool(args.trace_dir)]
    if sum(provided) > 1:
        raise SystemExit("provide only one of --trace-json, --trace-jsonl, --trace-csv, or --trace-dir")
    if args.trace_jsonl:
        payload = {"rows": _load_jsonl(args.trace_jsonl)}
    elif args.trace_csv:
        payload = {"rows": _load_csv(args.trace_csv)}
    elif args.trace_dir:
        payload = {"rows": _load_trace_dir(args.trace_dir)}
    else:
        payload = _load(args.trace_json) if args.trace_json else None
    report = build_report(
        trace_payload=payload,
        expected_missing_count=int(args.expected_missing_count),
        instrumentation_present=bool(args.instrumentation_present),
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
    print(f"audit_status={report['audit_status']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
