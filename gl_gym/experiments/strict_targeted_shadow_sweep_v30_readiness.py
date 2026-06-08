"""Build readiness v30 for the coverage-qualified strict-targeted shadow sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load_optional(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _coverage_pass(coverage: Mapping[str, Any] | None) -> bool:
    return bool(
        coverage
        and coverage.get("cache_coverage_pass", False)
        and _num(coverage.get("coverage_rate")) == 1.0
        and int(_num(coverage.get("missing_key_count"))) == 0
        and int(_num(coverage.get("missing_buffered_action_count"))) == 0
        and int(_num(coverage.get("missing_parsed_plan_count"))) == 0
    )


def _summary_pass(summary_audit: Mapping[str, Any] | None) -> bool:
    if not isinstance(summary_audit, Mapping):
        return False
    aggregate = summary_audit.get("aggregate", {})
    return isinstance(aggregate, Mapping) and aggregate.get("decision") == "pass"


def _runtime_complete(runtime_audit: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(runtime_audit, Mapping)
        and runtime_audit.get("audit_status") == "runtime_provenance_complete"
        and int(_num(runtime_audit.get("record_count"))) > 0
        and int(_num(runtime_audit.get("runtime_reason_missing_count"))) == 0
        and int(_num(runtime_audit.get("unknown_post_guardrail_rewrite_count"))) == 0
    )


def _joint_complete(joint_readiness: Mapping[str, Any] | None) -> bool:
    return bool(
        isinstance(joint_readiness, Mapping)
        and joint_readiness.get("ready_for_policy_judgment", False)
        and not dict(joint_readiness.get("missing_field_counts", {}) or {})
    )


def build_report(
    *,
    manifest: Mapping[str, Any],
    cache_coverage: Mapping[str, Any] | None = None,
    execution_record: Mapping[str, Any] | None = None,
    strict_sourcing: Mapping[str, Any] | None = None,
    summary_audit: Mapping[str, Any] | None = None,
    runtime_audit: Mapping[str, Any] | None = None,
    joint_readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_ready = bool(manifest.get("strict_targeted_shadow_sweep_v30_manifest_ready", False))
    cache_ok = _coverage_pass(cache_coverage)
    execution_authorized = bool(execution_record and execution_record.get("strict_targeted_shadow_sweep_v30_authorized", False))
    sourcing_run = isinstance(strict_sourcing, Mapping)
    source_found = bool(strict_sourcing and strict_sourcing.get("strict_targeted_source_found", False))
    expected_steps = int(_num(strict_sourcing.get("expected_strict_eligible_applied_steps") if strict_sourcing else 0))
    summary_ok = _summary_pass(summary_audit)
    runtime_ok = _runtime_complete(runtime_audit)
    joint_ok = _joint_complete(joint_readiness)
    post_run_audits_provided = bool(summary_audit or runtime_audit or joint_readiness)
    post_run_audits_pass = bool(summary_ok and runtime_ok and joint_ok) if post_run_audits_provided else False

    failure_taxonomy: list[str] = []
    if not manifest_ready:
        failure_taxonomy.append("manifest_missing_or_incomplete")
    if cache_coverage and not cache_ok:
        failure_taxonomy.append("cache_mismatch")
    if post_run_audits_provided and not summary_ok:
        failure_taxonomy.append("summary_gate_failed")
    if post_run_audits_provided and not runtime_ok:
        failure_taxonomy.append("runtime_provenance_missing")
    if post_run_audits_provided and not joint_ok:
        failure_taxonomy.append("joint_prediction_missing")
    if sourcing_run and not source_found:
        failure_taxonomy.append("strict_targeted_source_not_found_in_coverage_qualified_subset")
        failure_taxonomy.extend(list(strict_sourcing.get("blocker_taxonomy", []) or []))

    if not manifest_ready:
        next_action = "v30_subset_manifest_required"
    elif not cache_coverage:
        next_action = "run_v30_no_fill_cache_coverage_precheck"
    elif not cache_ok:
        next_action = "v30_subset_cache_coverage_blocked"
    elif not execution_authorized:
        next_action = "v30_shadow_sweep_execution_record_required"
    elif not sourcing_run:
        next_action = "execute_v30_shadow_sweep_and_strict_targeted_sourcing"
    elif post_run_audits_provided and not post_run_audits_pass:
        next_action = "diagnose_v30_post_run_audit_failure"
    elif source_found:
        next_action = "strict_targeted_source_manifest_and_no_fill_cache_precheck"
    else:
        next_action = "new_scenario_cache_discovery_for_strict_targeted_source"

    return {
        "schema_version": "metadata_replay_readiness_checklist_v30",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "coverage-qualified strict-targeted shadow-only sweep readiness",
            "reopens_rejected_preset": False,
        },
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "v30_manifest_ready": manifest_ready,
        "v30_cache_coverage_pass": cache_ok,
        "v30_shadow_sweep_authorized": execution_authorized,
        "v30_shadow_sweep_executed": bool(execution_authorized and sourcing_run),
        "strict_targeted_source_found": source_found,
        "strict_targeted_source_manifest_ready": source_found,
        "post_run_audits_provided": post_run_audits_provided,
        "post_run_audits_pass": post_run_audits_pass,
        "runtime_provenance_complete": runtime_ok,
        "joint_prediction_complete": joint_ok,
        "post_run_metrics": {
            "scenario_count": int(_num(manifest.get("scenario_count"))),
            "blocked_cache_repair_backlog_count": int(_num(manifest.get("blocked_cache_repair_backlog_count"))),
            "expected_env_count": int(_num(cache_coverage.get("expected_env_count") if cache_coverage else 0)),
            "coverage_rate": _num(cache_coverage.get("coverage_rate") if cache_coverage else 0.0),
            "missing_key_count": int(_num(cache_coverage.get("missing_key_count") if cache_coverage else 0)),
            "missing_buffered_action_count": int(_num(cache_coverage.get("missing_buffered_action_count") if cache_coverage else 0)),
            "missing_parsed_plan_count": int(_num(cache_coverage.get("missing_parsed_plan_count") if cache_coverage else 0)),
            "summary_failure_count": int(_num((summary_audit or {}).get("aggregate", {}).get("failure_count") if isinstance((summary_audit or {}).get("aggregate", {}), Mapping) else 0)),
            "runtime_provenance_record_count": int(_num((runtime_audit or {}).get("record_count") if runtime_audit else 0)),
            "runtime_reason_missing_count": int(_num((runtime_audit or {}).get("runtime_reason_missing_count") if runtime_audit else 0)),
            "unknown_post_guardrail_rewrite_count": int(_num((runtime_audit or {}).get("unknown_post_guardrail_rewrite_count") if runtime_audit else 0)),
            "joint_prediction_missing_fields": sum(int(_num(value)) for value in dict((joint_readiness or {}).get("missing_field_counts", {}) if joint_readiness else {}).values()),
            "expected_strict_eligible_applied_steps": expected_steps,
            "would_apply_steps": int(_num(strict_sourcing.get("would_apply_steps") if strict_sourcing else 0)),
            "strict_filtered_steps": int(_num(strict_sourcing.get("strict_filtered_steps") if strict_sourcing else 0)),
        },
        "failure_taxonomy": sorted(set(failure_taxonomy)),
        "next_action": next_action,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Metadata Replay Readiness Checklist v30",
        "",
        "- Mode: coverage-qualified strict-targeted shadow-only sweep readiness",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Controlled replay execution allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Cache coverage pass: {report.get('v30_cache_coverage_pass', False)}",
        f"- Post-run audits pass: {report.get('post_run_audits_pass', False)}",
        f"- Strict-targeted source found: {report.get('strict_targeted_source_found', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("post_run_metrics", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Failure Taxonomy", ""])
    failures = report.get("failure_taxonomy", []) or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--cache-coverage-json", default="")
    parser.add_argument("--execution-record-json", default="")
    parser.add_argument("--strict-sourcing-json", default="")
    parser.add_argument("--summary-audit-json", default="")
    parser.add_argument("--runtime-audit-json", default="")
    parser.add_argument("--joint-readiness-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        manifest=_load_optional(args.manifest_json) or {},
        cache_coverage=_load_optional(args.cache_coverage_json),
        execution_record=_load_optional(args.execution_record_json),
        strict_sourcing=_load_optional(args.strict_sourcing_json),
        summary_audit=_load_optional(args.summary_audit_json),
        runtime_audit=_load_optional(args.runtime_audit_json),
        joint_readiness=_load_optional(args.joint_readiness_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"v30_cache_coverage_pass={report['v30_cache_coverage_pass']}")
    print(f"strict_targeted_source_found={report['strict_targeted_source_found']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
