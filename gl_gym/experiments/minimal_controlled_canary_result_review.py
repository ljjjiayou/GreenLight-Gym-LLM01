"""Review minimal controlled canary results before trigger discovery."""

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


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _runtime_complete(runtime_audit: Mapping[str, Any]) -> bool:
    if runtime_audit.get("audit_status") == "runtime_provenance_complete":
        return True
    return bool(
        _num(runtime_audit.get("record_count")) > 0
        and _num(runtime_audit.get("runtime_reason_missing_count")) == 0
        and _num(runtime_audit.get("unknown_post_guardrail_rewrite_count")) == 0
    )


def build_report(
    *,
    readiness_v21: Mapping[str, Any],
    summary_audit: Mapping[str, Any],
    trace_audit: Mapping[str, Any],
    strict_effect_audit: Mapping[str, Any],
    runtime_provenance_audit: Mapping[str, Any],
    joint_prediction_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    post_metrics = readiness_v21.get("post_run_metrics", {})
    trace_aggregate = trace_audit.get("aggregate", {})
    effect_aggregate = strict_effect_audit.get("aggregate", {})
    summary_aggregate = summary_audit.get("aggregate", {})

    unsafe_preferred = int(_num(trace_aggregate.get("unsafe_preferred_steps")))
    unsafe_applied = int(_num(trace_aggregate.get("unsafe_applied_steps")))
    unsafe_conflict = int(_num(trace_aggregate.get("unsafe_conflict_steps")))
    strict_applied = int(_num(trace_aggregate.get("strict_applied_steps")))
    applied_steps = int(_num(trace_aggregate.get("applied_steps")))
    would_apply = int(_num(trace_aggregate.get("would_apply_steps")))
    strict_filtered = int(_num(trace_aggregate.get("strict_filtered_steps")))
    runtime_error_steps = int(_num(post_metrics.get("controlled_canary_runtime_error_steps")))
    strict_cache_miss_steps = int(_num(post_metrics.get("controlled_canary_strict_cache_miss_runtime_error_steps")))

    runtime_complete = _runtime_complete(runtime_provenance_audit)
    joint_complete = bool(joint_prediction_readiness.get("ready_for_policy_judgment", False)) and int(
        sum(int(v) for v in joint_prediction_readiness.get("missing_field_counts", {}).values())
    ) == 0
    summary_pass = str(summary_aggregate.get("decision", "")) == "pass"
    effect_decision = str(effect_aggregate.get("recommendation", {}).get("decision", "unknown"))
    effect_safety_pass = bool(
        int(_num(effect_aggregate.get("runtime_error_steps"))) == 0
        and int(_num(effect_aggregate.get("unsafe_applied_steps"))) == 0
        and not effect_aggregate.get("warnings", [])
    )

    canary_safety_pass = bool(
        readiness_v21.get("minimal_controlled_canary_pass", False)
        and summary_pass
        and effect_safety_pass
        and runtime_error_steps == 0
        and strict_cache_miss_steps == 0
        and unsafe_preferred == 0
        and unsafe_conflict == 0
        and unsafe_applied == 0
        and runtime_complete
        and joint_complete
    )
    control_effect_observed = bool(strict_applied > 0 and applied_steps > 0)
    strict_not_triggered = bool(canary_safety_pass and not control_effect_observed and effect_decision == "strict_not_triggered")
    if not canary_safety_pass:
        next_action = "minimal_controlled_canary_failure_diagnosis_required"
    elif strict_not_triggered:
        next_action = "strict_trigger_discovery_required"
    elif control_effect_observed:
        next_action = "controlled_canary_expansion_admission_review"
    else:
        next_action = "strict_trigger_diagnosis_required"

    return {
        "schema_version": "minimal_controlled_canary_result_review_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "controlled canary diagnosis",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "canary_safety_pass": canary_safety_pass,
        "control_effect_observed": control_effect_observed,
        "strict_not_triggered": strict_not_triggered,
        "effect_decision": effect_decision,
        "metrics": {
            "would_apply_steps": would_apply,
            "applied_steps": applied_steps,
            "strict_applied_steps": strict_applied,
            "strict_filtered_steps": strict_filtered,
            "unsafe_preferred_steps": unsafe_preferred,
            "unsafe_conflict_steps": unsafe_conflict,
            "unsafe_applied_steps": unsafe_applied,
            "runtime_error_steps": runtime_error_steps,
            "strict_cache_miss_runtime_error_steps": strict_cache_miss_steps,
            "runtime_provenance_record_count": int(_num(runtime_provenance_audit.get("record_count"))),
            "runtime_provenance_complete": runtime_complete,
            "joint_prediction_missing_fields": int(
                sum(int(v) for v in joint_prediction_readiness.get("missing_field_counts", {}).values())
            ),
            "joint_prediction_complete": joint_complete,
        },
        "blockers": [] if not strict_not_triggered else ["strict_not_triggered"],
        "next_action": next_action,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), Mapping) else {}
    lines = [
        "# Minimal Controlled Canary Result Review",
        "",
        f"- Canary safety pass: {report.get('canary_safety_pass', False)}",
        f"- Control effect observed: {report.get('control_effect_observed', False)}",
        f"- Strict not triggered: {report.get('strict_not_triggered', False)}",
        f"- Controlled replay allowed: {report.get('controlled_replay_allowed', False)}",
        f"- Performance claim allowed: {report.get('performance_claim_allowed', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | `{value}` |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-v21-json", required=True)
    parser.add_argument("--summary-audit-json", required=True)
    parser.add_argument("--trace-audit-json", required=True)
    parser.add_argument("--strict-effect-audit-json", required=True)
    parser.add_argument("--runtime-provenance-audit-json", required=True)
    parser.add_argument("--joint-prediction-readiness-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        readiness_v21=_load(args.readiness_v21_json),
        summary_audit=_load(args.summary_audit_json),
        trace_audit=_load(args.trace_audit_json),
        strict_effect_audit=_load(args.strict_effect_audit_json),
        runtime_provenance_audit=_load(args.runtime_provenance_audit_json),
        joint_prediction_readiness=_load(args.joint_prediction_readiness_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"canary_safety_pass={report['canary_safety_pass']}")
    print(f"control_effect_observed={report['control_effect_observed']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["canary_safety_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
