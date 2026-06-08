"""Build readiness v22 for controlled-canary trigger discovery and reruns."""

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


def build_report(
    *,
    result_review: Mapping[str, Any],
    trigger_discovery: Mapping[str, Any],
    triggered_manifest: Mapping[str, Any] | None = None,
    cache_coverage: Mapping[str, Any] | None = None,
    execution_record: Mapping[str, Any] | None = None,
    summary_audit: Mapping[str, Any] | None = None,
    trace_audit: Mapping[str, Any] | None = None,
    strict_effect_audit: Mapping[str, Any] | None = None,
    runtime_provenance_audit: Mapping[str, Any] | None = None,
    joint_prediction_readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    review_pass = bool(result_review.get("canary_safety_pass", False))
    trigger_found = bool(trigger_discovery.get("trigger_candidates_found", False))
    manifest_ready = bool(triggered_manifest and triggered_manifest.get("triggered_manifest_ready", False))
    coverage_pass = bool(
        cache_coverage
        and cache_coverage.get("cache_coverage_pass", False)
        and _num(cache_coverage.get("coverage_rate")) == 1.0
        and int(_num(cache_coverage.get("missing_key_count"))) == 0
        and int(_num(cache_coverage.get("missing_buffered_action_count"))) == 0
        and int(_num(cache_coverage.get("missing_parsed_plan_count"))) == 0
    )
    execution_authorized = bool(execution_record and execution_record.get("triggered_controlled_canary_authorized", False))
    post_run_inputs = all(
        isinstance(item, Mapping)
        for item in (summary_audit, trace_audit, strict_effect_audit, runtime_provenance_audit, joint_prediction_readiness)
    )

    trace_aggregate = trace_audit.get("aggregate", {}) if isinstance(trace_audit, Mapping) else {}
    effect_aggregate = strict_effect_audit.get("aggregate", {}) if isinstance(strict_effect_audit, Mapping) else {}
    summary_aggregate = summary_audit.get("aggregate", {}) if isinstance(summary_audit, Mapping) else {}
    strict_applied = int(_num(trace_aggregate.get("strict_applied_steps")))
    unsafe_preferred = int(_num(trace_aggregate.get("unsafe_preferred_steps")))
    unsafe_applied = int(_num(trace_aggregate.get("unsafe_applied_steps")))
    unsafe_conflict = int(_num(trace_aggregate.get("unsafe_conflict_steps")))
    runtime_complete = bool(runtime_provenance_audit and runtime_provenance_audit.get("audit_status") == "runtime_provenance_complete")
    joint_complete = bool(joint_prediction_readiness and joint_prediction_readiness.get("ready_for_policy_judgment", False))
    summary_pass = bool(summary_aggregate.get("decision") == "pass")
    effect_safety_pass = bool(
        strict_effect_audit
        and int(_num(effect_aggregate.get("runtime_error_steps"))) == 0
        and int(_num(effect_aggregate.get("unsafe_applied_steps"))) == 0
        and not effect_aggregate.get("warnings", [])
    )
    triggered_canary_pass = bool(
        post_run_inputs
        and execution_authorized
        and summary_pass
        and strict_applied > 0
        and unsafe_preferred == 0
        and unsafe_conflict == 0
        and unsafe_applied == 0
        and runtime_complete
        and joint_complete
        and effect_safety_pass
    )

    if not review_pass:
        next_action = "minimal_controlled_canary_failure_diagnosis_required"
    elif not trigger_found:
        next_action = "strict_trigger_blocker_diagnosis_or_scenario_discovery_continue"
    elif trigger_found and not manifest_ready:
        next_action = "triggered_scenario_manifest_required"
    elif manifest_ready and not coverage_pass:
        next_action = "triggered_canary_cache_coverage_required"
    elif coverage_pass and not execution_authorized:
        next_action = "triggered_canary_execution_authorization_required"
    elif execution_authorized and not post_run_inputs:
        next_action = "execute_triggered_controlled_canary"
    elif triggered_canary_pass:
        next_action = "controlled_canary_expansion_admission_review"
    else:
        next_action = "triggered_controlled_canary_failure_diagnosis_required"

    return {
        "schema_version": "metadata_replay_readiness_checklist_v22",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "controlled canary trigger readiness",
            "reopens_rejected_preset": False,
        },
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "minimal_controlled_canary_result_review_pass": review_pass,
        "strict_trigger_discovery_run": True,
        "trigger_candidates_found": trigger_found,
        "triggered_manifest_ready": manifest_ready,
        "triggered_cache_coverage_pass": coverage_pass,
        "triggered_controlled_canary_authorized": execution_authorized,
        "triggered_controlled_canary_executed": bool(post_run_inputs),
        "triggered_controlled_canary_pass": triggered_canary_pass,
        "post_run_metrics": {
            "expected_strict_eligible_applied_steps": int(
                _num(trigger_discovery.get("expected_strict_eligible_applied_steps"))
            ),
            "would_apply_steps": int(_num(trigger_discovery.get("would_apply_steps"))),
            "strict_filtered_steps": int(_num(trigger_discovery.get("strict_filtered_steps"))),
            "strict_applied_steps": strict_applied,
            "unsafe_preferred_steps": unsafe_preferred,
            "unsafe_conflict_steps": unsafe_conflict,
            "unsafe_applied_steps": unsafe_applied,
        },
        "blocker_taxonomy": list(trigger_discovery.get("blocker_taxonomy", []) or []),
        "next_action": next_action,
    }


def build_v28_report(*, trigger_discovery: Mapping[str, Any]) -> dict[str, Any]:
    source_found = bool(trigger_discovery.get("strict_targeted_source_found", False))
    expected_steps = int(_num(trigger_discovery.get("expected_strict_eligible_applied_steps")))
    blockers = list(trigger_discovery.get("blocker_taxonomy", []) or [])
    if not source_found and "strict_targeted_source_not_found" not in blockers:
        blockers.insert(0, "strict_targeted_source_not_found")
    return {
        "schema_version": "metadata_replay_readiness_checklist_v28",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "strict-targeted shadow-only sourcing readiness",
            "reopens_rejected_preset": False,
        },
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "strict_targeted_shadow_sourcing_run": True,
        "strict_targeted_source_found": source_found,
        "strict_targeted_source_manifest_ready": source_found,
        "no_fill_cache_precheck_requested": source_found,
        "post_run_metrics": {
            "expected_strict_eligible_applied_steps": expected_steps,
            "would_apply_steps": int(_num(trigger_discovery.get("would_apply_steps"))),
            "strict_filtered_steps": int(_num(trigger_discovery.get("strict_filtered_steps"))),
            "trace_count": int(_num(trigger_discovery.get("trace_count"))),
            "excluded_trace_count": int(_num(trigger_discovery.get("excluded_trace_count"))),
        },
        "strict_target_gates": dict(trigger_discovery.get("strict_target_gates", {})),
        "qualified_trigger_scenarios": list(trigger_discovery.get("strict_targeted_source_scenarios", []) or []),
        "blocker_taxonomy": [] if source_found else blockers,
        "next_action": (
            "strict_targeted_source_manifest_and_no_fill_cache_precheck"
            if source_found
            else "design_larger_shadow_only_sweep"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    if report.get("schema_version") == "metadata_replay_readiness_checklist_v28":
        lines = [
            "# Metadata Replay Readiness Checklist v28",
            "",
            "- Mode: strict-targeted shadow-only sourcing readiness",
            "- Metadata replay execution allowed: false",
            "- Controlled replay allowed: false",
            "- Controlled replay execution allowed: false",
            "- Performance claim allowed: false",
            "- Promotion evidence: false",
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
        lines.extend(["", "## Blockers", ""])
        blockers = report.get("blocker_taxonomy", []) or []
        if blockers:
            lines.extend(f"- `{item}`" for item in blockers)
        else:
            lines.append("- none")
        return "\n".join(lines) + "\n"

    lines = [
        "# Metadata Replay Readiness Checklist v22",
        "",
        "- Mode: controlled canary trigger readiness",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Trigger candidates found: {report.get('trigger_candidates_found', False)}",
        f"- Triggered canary pass: {report.get('triggered_controlled_canary_pass', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("post_run_metrics", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Blockers", ""])
    blockers = report.get("blocker_taxonomy", []) or []
    lines.extend(f"- `{item}`" for item in blockers)
    if not blockers:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-version", choices=["v22", "v28"], default="v22")
    parser.add_argument("--result-review-json", default="")
    parser.add_argument("--trigger-discovery-json", required=True)
    parser.add_argument("--triggered-manifest-json", default="")
    parser.add_argument("--cache-coverage-json", default="")
    parser.add_argument("--execution-record-json", default="")
    parser.add_argument("--summary-audit-json", default="")
    parser.add_argument("--trace-audit-json", default="")
    parser.add_argument("--strict-effect-audit-json", default="")
    parser.add_argument("--runtime-provenance-audit-json", default="")
    parser.add_argument("--joint-prediction-readiness-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    trigger_discovery = _load_optional(args.trigger_discovery_json) or {}
    if args.readiness_version == "v28":
        report = build_v28_report(trigger_discovery=trigger_discovery)
    else:
        if not args.result_review_json:
            raise SystemExit("--result-review-json is required for readiness v22")
        report = build_report(
            result_review=_load_optional(args.result_review_json) or {},
            trigger_discovery=trigger_discovery,
            triggered_manifest=_load_optional(args.triggered_manifest_json),
            cache_coverage=_load_optional(args.cache_coverage_json),
            execution_record=_load_optional(args.execution_record_json),
            summary_audit=_load_optional(args.summary_audit_json),
            trace_audit=_load_optional(args.trace_audit_json),
            strict_effect_audit=_load_optional(args.strict_effect_audit_json),
            runtime_provenance_audit=_load_optional(args.runtime_provenance_audit_json),
            joint_prediction_readiness=_load_optional(args.joint_prediction_readiness_json),
        )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"trigger_candidates_found={report.get('trigger_candidates_found', report.get('strict_targeted_source_found', False))}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
