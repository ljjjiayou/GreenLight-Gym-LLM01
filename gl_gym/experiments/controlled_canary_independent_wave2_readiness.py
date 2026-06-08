"""Build readiness v24 for independent triggered holdout Wave-2."""

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


def _aggregate(report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(report, Mapping):
        return {}
    value = report.get("aggregate", {})
    return value if isinstance(value, Mapping) else {}


def _hard_regressions(summary_audit: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(summary_audit, Mapping):
        return []
    regressions: list[str] = []
    for row in summary_audit.get("paired_deltas", []) or []:
        if not isinstance(row, Mapping):
            continue
        scenario_id = str(row.get("scenario_id", "") or "")
        if _num(row.get("d_canopy_dew_margin_lt0_steps")) > 0:
            regressions.append(f"{scenario_id}:canopy_dew_margin_lt0")
        if _num(row.get("d_dew_margin_air_lt0_steps")) > 0:
            regressions.append(f"{scenario_id}:dew_margin_air_lt0")
    return regressions


def _coverage_pass(coverage: Mapping[str, Any] | None) -> bool:
    return bool(
        coverage
        and coverage.get("cache_coverage_pass", False)
        and _num(coverage.get("coverage_rate")) == 1.0
        and int(_num(coverage.get("missing_key_count"))) == 0
        and int(_num(coverage.get("missing_buffered_action_count"))) == 0
        and int(_num(coverage.get("missing_parsed_plan_count"))) == 0
    )


def _result_review_pass(result_review: Mapping[str, Any]) -> bool:
    return bool(
        result_review.get("wave1_result_review_pass", False)
        or result_review.get("shadow_sourced_canary_result_review_pass", False)
        or result_review.get("independent_holdout_admission_review_pass", False)
    )


def build_report(
    *,
    result_review: Mapping[str, Any],
    discovery: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
    cache_coverage: Mapping[str, Any] | None = None,
    execution_record: Mapping[str, Any] | None = None,
    summary_audit: Mapping[str, Any] | None = None,
    trace_audit: Mapping[str, Any] | None = None,
    strict_effect_audit: Mapping[str, Any] | None = None,
    runtime_provenance_audit: Mapping[str, Any] | None = None,
    joint_prediction_readiness: Mapping[str, Any] | None = None,
    schema_version: str = "metadata_replay_readiness_checklist_v24",
    mode_label: str = "independent triggered holdout wave2 readiness",
) -> dict[str, Any]:
    summary_agg = _aggregate(summary_audit)
    trace_agg = _aggregate(trace_audit)
    effect_agg = _aggregate(strict_effect_audit)
    hard_regressions = _hard_regressions(summary_audit)
    post_inputs = all(
        isinstance(item, Mapping)
        for item in (execution_record, summary_audit, trace_audit, strict_effect_audit, runtime_provenance_audit, joint_prediction_readiness)
    )
    strict_applied = int(_num(trace_agg.get("strict_applied_steps")))
    unsafe_preferred = int(_num(trace_agg.get("unsafe_preferred_steps")))
    unsafe_conflict = int(_num(trace_agg.get("unsafe_conflict_steps")))
    unsafe_applied = int(_num(trace_agg.get("unsafe_applied_steps")))
    runtime_errors = int(_num(summary_agg.get("runtime_error_steps"), _num(effect_agg.get("runtime_error_steps"))))
    strict_cache_miss = int(_num(summary_agg.get("strict_cache_miss_runtime_error_steps")))
    runtime_complete = bool(runtime_provenance_audit and runtime_provenance_audit.get("audit_status") == "runtime_provenance_complete")
    joint_complete = bool(joint_prediction_readiness and joint_prediction_readiness.get("ready_for_policy_judgment", False))
    pre_run_ready = bool(
        _result_review_pass(result_review)
        and discovery.get("independent_trigger_candidates_found", False)
        and manifest
        and manifest.get("wave2_manifest_ready", False)
        and _coverage_pass(cache_coverage)
    )
    execution_authorized = bool(execution_record and execution_record.get("wave2_controlled_canary_authorized", False))
    failure_taxonomy: list[str] = []
    review_pass = _result_review_pass(result_review)
    if post_inputs:
        if summary_agg.get("decision") != "pass":
            failure_taxonomy.append("summary_gate_failed")
        if runtime_errors != 0:
            failure_taxonomy.append("runtime_error")
        if strict_cache_miss != 0:
            failure_taxonomy.append("strict_cache_miss")
        if unsafe_preferred:
            failure_taxonomy.append("unsafe_preferred")
        if unsafe_conflict:
            failure_taxonomy.append("unsafe_conflict")
        if unsafe_applied:
            failure_taxonomy.append("unsafe_applied")
        if strict_applied <= 0:
            failure_taxonomy.append("strict_not_triggered")
        if hard_regressions:
            failure_taxonomy.append("hard_safety_regression")
        if not runtime_complete:
            failure_taxonomy.append("metadata_missing")
        if not joint_complete:
            failure_taxonomy.append("joint_prediction_missing")
    elif (
        schema_version.endswith("_v32")
        and review_pass
        and not discovery.get("independent_trigger_candidates_found", False)
    ):
        failure_taxonomy.append("no_independent_strict_trigger_candidate_after_v31")
    wave2_pass = bool(post_inputs and pre_run_ready and execution_authorized and not failure_taxonomy)
    if not review_pass:
        next_action = "independent_holdout_result_review_required"
    elif not discovery.get("independent_trigger_candidates_found", False):
        next_action = (
            "new_shadow_only_scenario_cache_discovery_after_independent_holdout_blocked"
            if schema_version.endswith("_v32")
            else "independent_trigger_blocker_diagnosis"
        )
    elif not (manifest and manifest.get("wave2_manifest_ready", False)):
        next_action = "independent_wave2_manifest_required"
    elif not _coverage_pass(cache_coverage):
        next_action = "independent_wave2_cache_coverage_required"
    elif not execution_authorized:
        next_action = "independent_wave2_execution_record_required"
    elif not post_inputs:
        next_action = "execute_independent_triggered_holdout_wave2"
    elif wave2_pass:
        next_action = "wave2_result_review_and_independent_holdout_admission_planning"
    else:
        next_action = "independent_wave2_failure_diagnosis"
    return {
        "schema_version": schema_version,
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": mode_label,
            "reopens_rejected_preset": False,
        },
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "wave2_pre_run_ready": pre_run_ready,
        "wave2_controlled_canary_executed": bool(post_inputs),
        "wave2_controlled_canary_pass": wave2_pass,
        "control_effect_observed": strict_applied > 0 if post_inputs else False,
        "acceptance": {
            "result_review_pass": review_pass,
            "independent_trigger_candidates_found": bool(discovery.get("independent_trigger_candidates_found", False)),
            "manifest_ready": bool(manifest and manifest.get("wave2_manifest_ready", False)),
            "cache_coverage_pass": _coverage_pass(cache_coverage),
            "execution_authorized": execution_authorized,
            "summary_pass": summary_agg.get("decision") == "pass" if post_inputs else False,
            "runtime_error_steps_zero": runtime_errors == 0 if post_inputs else False,
            "strict_cache_miss_zero": strict_cache_miss == 0 if post_inputs else False,
            "strict_applied_steps_positive": strict_applied > 0 if post_inputs else False,
            "unsafe_preferred_zero": unsafe_preferred == 0 if post_inputs else False,
            "unsafe_conflict_zero": unsafe_conflict == 0 if post_inputs else False,
            "unsafe_applied_zero": unsafe_applied == 0 if post_inputs else False,
            "hard_safety_non_regression": not hard_regressions if post_inputs else False,
            "runtime_provenance_complete": runtime_complete,
            "joint_prediction_complete": joint_complete,
        },
        "post_run_metrics": {
            "scenario_count": int(_num(summary_agg.get("scenario_count"), len((manifest or {}).get("scenario_ids", []) or []))),
            "strict_applied_steps": strict_applied,
            "unsafe_preferred_steps": unsafe_preferred,
            "unsafe_conflict_steps": unsafe_conflict,
            "unsafe_applied_steps": unsafe_applied,
            "runtime_error_steps": runtime_errors,
            "strict_cache_miss_runtime_error_steps": strict_cache_miss,
            "runtime_provenance_record_count": int(_num(runtime_provenance_audit.get("record_count") if runtime_provenance_audit else 0)),
            "runtime_reason_missing_count": int(_num(runtime_provenance_audit.get("runtime_reason_missing_count") if runtime_provenance_audit else 0)),
            "unknown_post_guardrail_rewrite_count": int(_num(runtime_provenance_audit.get("unknown_post_guardrail_rewrite_count") if runtime_provenance_audit else 0)),
            "joint_prediction_row_count": int(_num(joint_prediction_readiness.get("row_count") if joint_prediction_readiness else 0)),
        },
        "hard_safety_regressions": hard_regressions,
        "failure_taxonomy": failure_taxonomy,
        "next_action": next_action,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Metadata Replay Readiness Checklist {str(report.get('schema_version', '')).rsplit('_', 1)[-1]}",
        "",
        f"- Mode: {report.get('mainline_alignment', {}).get('mode', 'independent triggered holdout wave2 readiness')}",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Wave-2 pass: {report.get('wave2_controlled_canary_pass', False)}",
        f"- Control effect observed: {report.get('control_effect_observed', False)}",
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
    if failures:
        lines.extend(f"- `{item}`" for item in failures)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-review-json", required=True)
    parser.add_argument("--discovery-json", required=True)
    parser.add_argument("--manifest-json", default="")
    parser.add_argument("--cache-coverage-json", default="")
    parser.add_argument("--execution-record-json", default="")
    parser.add_argument("--summary-audit-json", default="")
    parser.add_argument("--trace-audit-json", default="")
    parser.add_argument("--strict-effect-audit-json", default="")
    parser.add_argument("--runtime-provenance-audit-json", default="")
    parser.add_argument("--joint-prediction-readiness-json", default="")
    parser.add_argument("--schema-version", default="metadata_replay_readiness_checklist_v24")
    parser.add_argument("--mode-label", default="independent triggered holdout wave2 readiness")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        result_review=_load_optional(args.result_review_json) or {},
        discovery=_load_optional(args.discovery_json) or {},
        manifest=_load_optional(args.manifest_json),
        cache_coverage=_load_optional(args.cache_coverage_json),
        execution_record=_load_optional(args.execution_record_json),
        summary_audit=_load_optional(args.summary_audit_json),
        trace_audit=_load_optional(args.trace_audit_json),
        strict_effect_audit=_load_optional(args.strict_effect_audit_json),
        runtime_provenance_audit=_load_optional(args.runtime_provenance_audit_json),
        joint_prediction_readiness=_load_optional(args.joint_prediction_readiness_json),
        schema_version=args.schema_version,
        mode_label=args.mode_label,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"wave2_controlled_canary_pass={report['wave2_controlled_canary_pass']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
