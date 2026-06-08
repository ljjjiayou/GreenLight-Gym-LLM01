"""Build readiness v26 for the shadow-only trigger source sweep."""

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


def _summary_metrics(summary_audit: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary_audit, Mapping):
        return {
            "summary_pass": False,
            "scenario_count": 0,
            "cache_hit_rate": 0.0,
            "runtime_error_steps": 0,
            "strict_cache_miss_runtime_error_steps": 0,
        }
    rows = [row for row in summary_audit.get("rows", []) or [] if isinstance(row, Mapping)]
    plan_enabled = sum(_num(row.get("plan_cache_enabled_steps")) for row in rows)
    plan_hits = sum(_num(row.get("plan_cache_hit_steps")) for row in rows)
    cache_hit_rate = round(float(plan_hits / plan_enabled), 6) if plan_enabled else 0.0
    runtime_errors = int(sum(_num(row.get("runtime_error_steps")) for row in rows))
    strict_cache_miss = int(sum(_num(row.get("strict_cache_miss_runtime_error_steps")) for row in rows))
    aggregate = _aggregate(summary_audit)
    return {
        "summary_pass": aggregate.get("decision") == "pass",
        "scenario_count": int(_num(aggregate.get("scenario_count"), len({row.get("scenario_id") for row in rows}))),
        "cache_hit_rate": cache_hit_rate,
        "runtime_error_steps": runtime_errors,
        "strict_cache_miss_runtime_error_steps": strict_cache_miss,
    }


def _runtime_complete(runtime_provenance_audit: Mapping[str, Any] | None) -> bool:
    return bool(
        runtime_provenance_audit
        and runtime_provenance_audit.get("audit_status") == "runtime_provenance_complete"
        and int(_num(runtime_provenance_audit.get("record_count"))) > 0
        and int(_num(runtime_provenance_audit.get("runtime_reason_missing_count"))) == 0
        and int(_num(runtime_provenance_audit.get("unknown_post_guardrail_rewrite_count"))) == 0
    )


def _joint_complete(joint_prediction_readiness: Mapping[str, Any] | None) -> bool:
    if not isinstance(joint_prediction_readiness, Mapping):
        return False
    missing_counts = joint_prediction_readiness.get("missing_field_counts", {})
    missing_total = sum(int(_num(value)) for value in missing_counts.values()) if isinstance(missing_counts, Mapping) else 0
    return bool(joint_prediction_readiness.get("ready_for_policy_judgment", False) and missing_total == 0)


def build_report(
    *,
    readiness_v25: Mapping[str, Any],
    execution_record: Mapping[str, Any] | None = None,
    summary_audit: Mapping[str, Any] | None = None,
    controlled_shadow_audit: Mapping[str, Any] | None = None,
    shadow_replay_audit: Mapping[str, Any] | None = None,
    runtime_provenance_audit: Mapping[str, Any] | None = None,
    joint_prediction_readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_ready = bool(
        readiness_v25.get("shadow_trigger_sources_found", False)
        and readiness_v25.get("shadow_trigger_source_manifest_ready", False)
        and readiness_v25.get("shadow_trigger_source_cache_precheck_pass", False)
    )
    execution_authorized = bool(execution_record and execution_record.get("shadow_trigger_source_sweep_authorized", False))
    post_inputs = all(
        isinstance(item, Mapping)
        for item in (
            execution_record,
            summary_audit,
            controlled_shadow_audit,
            shadow_replay_audit,
            runtime_provenance_audit,
            joint_prediction_readiness,
        )
    )
    summary = _summary_metrics(summary_audit)
    controlled_agg = _aggregate(controlled_shadow_audit)
    shadow_agg = _aggregate(shadow_replay_audit)
    unsafe_preferred = int(_num(controlled_agg.get("unsafe_preferred_steps")))
    unsafe_applied = int(_num(controlled_agg.get("unsafe_applied_steps")))
    missing_metadata = int(_num(controlled_agg.get("missing_metadata_steps")))
    shadow_window_count = int(_num(shadow_agg.get("replay_window_count")))
    shadow_signal_steps = int(_num(shadow_agg.get("signal_steps")))
    runtime_complete = _runtime_complete(runtime_provenance_audit)
    joint_complete = _joint_complete(joint_prediction_readiness)
    failure_taxonomy: list[str] = []
    if post_inputs:
        if not summary["summary_pass"]:
            failure_taxonomy.append("summary_gate_failed")
        if summary["cache_hit_rate"] != 1.0:
            failure_taxonomy.append("cache_mismatch")
        if summary["runtime_error_steps"] != 0:
            failure_taxonomy.append("runtime_error")
        if summary["strict_cache_miss_runtime_error_steps"] != 0:
            failure_taxonomy.append("strict_cache_miss")
        if unsafe_preferred != 0:
            failure_taxonomy.append("unsafe_preferred")
        if unsafe_applied != 0:
            failure_taxonomy.append("unsafe_applied")
        if missing_metadata != 0:
            failure_taxonomy.append("metadata_missing")
        if not runtime_complete:
            failure_taxonomy.append("runtime_provenance_missing")
        if not joint_complete:
            failure_taxonomy.append("joint_prediction_missing")
    sweep_pass = bool(post_inputs and source_ready and execution_authorized and not failure_taxonomy)
    if not source_ready:
        next_action = "shadow_trigger_source_manifest_or_cache_required"
    elif not execution_authorized:
        next_action = "shadow_trigger_source_sweep_execution_record_required"
    elif not post_inputs:
        next_action = "execute_shadow_trigger_source_sweep"
    elif sweep_pass and shadow_window_count > 0:
        next_action = "independent_shadow_source_expansion_admission_planning"
    elif sweep_pass:
        next_action = "shadow_trigger_source_exhaustion_diagnosis"
    else:
        next_action = "shadow_trigger_source_sweep_failure_diagnosis"
    return {
        "schema_version": "metadata_replay_readiness_checklist_v26",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only trigger source sweep readiness",
            "reopens_rejected_preset": False,
        },
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "source_precheck_ready": source_ready,
        "shadow_trigger_source_sweep_authorized": execution_authorized,
        "shadow_trigger_source_sweep_executed": bool(post_inputs),
        "shadow_trigger_source_sweep_pass": sweep_pass,
        "shadow_trigger_source_candidate": bool(sweep_pass and shadow_signal_steps > 0),
        "near_miss_window": bool(sweep_pass and shadow_window_count > 0),
        "ready_for_shadow_scenario_sourcing": bool(sweep_pass and (shadow_signal_steps > 0 or shadow_window_count > 0)),
        "acceptance": {
            "source_precheck_ready": source_ready,
            "execution_authorized": execution_authorized,
            "summary_pass": summary["summary_pass"] if post_inputs else False,
            "cache_hit_rate_one": summary["cache_hit_rate"] == 1.0 if post_inputs else False,
            "runtime_error_steps_zero": summary["runtime_error_steps"] == 0 if post_inputs else False,
            "strict_cache_miss_zero": summary["strict_cache_miss_runtime_error_steps"] == 0 if post_inputs else False,
            "unsafe_preferred_zero": unsafe_preferred == 0 if post_inputs else False,
            "unsafe_applied_zero": unsafe_applied == 0 if post_inputs else False,
            "runtime_provenance_complete": runtime_complete,
            "joint_prediction_complete": joint_complete,
        },
        "post_run_metrics": {
            "scenario_count": summary["scenario_count"],
            "cache_hit_rate": summary["cache_hit_rate"],
            "runtime_error_steps": summary["runtime_error_steps"],
            "strict_cache_miss_runtime_error_steps": summary["strict_cache_miss_runtime_error_steps"],
            "would_apply_steps": int(_num(controlled_agg.get("would_apply_steps"))),
            "controlled_applied_steps": int(_num(controlled_agg.get("applied_steps"))),
            "strict_applied_steps": int(_num(controlled_agg.get("strict_applied_steps"))),
            "unsafe_preferred_steps": unsafe_preferred,
            "unsafe_applied_steps": unsafe_applied,
            "shadow_signal_steps": shadow_signal_steps,
            "shadow_replay_window_count": shadow_window_count,
            "runtime_provenance_record_count": int(_num(runtime_provenance_audit.get("record_count") if runtime_provenance_audit else 0)),
            "runtime_reason_missing_count": int(_num(runtime_provenance_audit.get("runtime_reason_missing_count") if runtime_provenance_audit else 0)),
            "unknown_post_guardrail_rewrite_count": int(_num(runtime_provenance_audit.get("unknown_post_guardrail_rewrite_count") if runtime_provenance_audit else 0)),
            "joint_prediction_row_count": int(_num(joint_prediction_readiness.get("row_count") if joint_prediction_readiness else 0)),
        },
        "failure_taxonomy": failure_taxonomy,
        "next_action": next_action,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Metadata Replay Readiness Checklist v26",
        "",
        "- Mode: shadow-only trigger source sweep readiness",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Controlled replay execution allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Sweep pass: {report.get('shadow_trigger_source_sweep_pass', False)}",
        f"- Near-miss window: {report.get('near_miss_window', False)}",
        f"- Ready for shadow scenario sourcing: {report.get('ready_for_shadow_scenario_sourcing', False)}",
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
    parser.add_argument("--readiness-v25-json", required=True)
    parser.add_argument("--execution-record-json", default="")
    parser.add_argument("--summary-audit-json", default="")
    parser.add_argument("--controlled-shadow-audit-json", default="")
    parser.add_argument("--shadow-replay-audit-json", default="")
    parser.add_argument("--runtime-provenance-audit-json", default="")
    parser.add_argument("--joint-prediction-readiness-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        readiness_v25=_load_optional(args.readiness_v25_json) or {},
        execution_record=_load_optional(args.execution_record_json),
        summary_audit=_load_optional(args.summary_audit_json),
        controlled_shadow_audit=_load_optional(args.controlled_shadow_audit_json),
        shadow_replay_audit=_load_optional(args.shadow_replay_audit_json),
        runtime_provenance_audit=_load_optional(args.runtime_provenance_audit_json),
        joint_prediction_readiness=_load_optional(args.joint_prediction_readiness_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"shadow_trigger_source_sweep_pass={report['shadow_trigger_source_sweep_pass']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
