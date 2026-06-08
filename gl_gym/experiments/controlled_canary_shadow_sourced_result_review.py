"""Review v31 shadow-sourced controlled canary results before independent holdout admission."""

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


def _aggregate(report: Mapping[str, Any]) -> Mapping[str, Any]:
    value = report.get("aggregate", {})
    return value if isinstance(value, Mapping) else {}


def _scenario_ids(report: Mapping[str, Any]) -> list[str]:
    ids = report.get("scenario_ids", [])
    if isinstance(ids, list):
        return sorted(str(item) for item in ids if str(item or ""))
    scenarios = report.get("selected_scenarios", [])
    if isinstance(scenarios, list):
        return sorted(str(item.get("scenario_id", "")) for item in scenarios if isinstance(item, Mapping))
    return []


def _runtime_complete(runtime_audit: Mapping[str, Any]) -> bool:
    return bool(
        runtime_audit.get("audit_status") == "runtime_provenance_complete"
        and int(_num(runtime_audit.get("record_count"))) > 0
        and int(_num(runtime_audit.get("runtime_reason_missing_count"))) == 0
        and int(_num(runtime_audit.get("unknown_post_guardrail_rewrite_count"))) == 0
    )


def _joint_complete(joint_readiness: Mapping[str, Any]) -> bool:
    return bool(
        joint_readiness.get("ready_for_policy_judgment", False)
        and sum(int(_num(value)) for value in dict(joint_readiness.get("missing_field_counts", {}) or {}).values())
        == 0
    )


def _hard_regressions(summary_audit: Mapping[str, Any]) -> list[str]:
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


def build_report(
    *,
    readiness_v31: Mapping[str, Any],
    v31_manifest: Mapping[str, Any],
    summary_audit: Mapping[str, Any],
    trace_audit: Mapping[str, Any],
    strict_effect_audit: Mapping[str, Any],
    runtime_provenance_audit: Mapping[str, Any],
    joint_prediction_readiness: Mapping[str, Any],
    extra_excluded_scenario_ids: Sequence[str] = (),
) -> dict[str, Any]:
    summary_agg = _aggregate(summary_audit)
    trace_agg = _aggregate(trace_audit)
    effect_agg = _aggregate(strict_effect_audit)
    post_metrics = readiness_v31.get("post_run_metrics", {})
    strict_applied = int(_num(trace_agg.get("strict_applied_steps"), post_metrics.get("strict_applied_steps", 0)))
    unsafe_preferred = int(_num(trace_agg.get("unsafe_preferred_steps"), post_metrics.get("unsafe_preferred_steps", 0)))
    unsafe_conflict = int(_num(trace_agg.get("unsafe_conflict_steps"), post_metrics.get("unsafe_conflict_steps", 0)))
    unsafe_applied = int(_num(trace_agg.get("unsafe_applied_steps"), post_metrics.get("unsafe_applied_steps", 0)))
    runtime_errors = int(_num(summary_agg.get("runtime_error_steps"), effect_agg.get("runtime_error_steps", 0)))
    strict_cache_miss = int(_num(summary_agg.get("strict_cache_miss_runtime_error_steps"), 0))
    expected_strict = int(_num(post_metrics.get("expected_strict_eligible_applied_steps")))
    hard_regressions = _hard_regressions(summary_audit)
    runtime_complete = _runtime_complete(runtime_provenance_audit)
    joint_complete = _joint_complete(joint_prediction_readiness)
    v31_pass = bool(readiness_v31.get("shadow_sourced_controlled_canary_pass", False))
    control_effect_observed = bool(readiness_v31.get("control_effect_observed", False) and strict_applied > 0)
    review_pass = bool(
        v31_pass
        and control_effect_observed
        and summary_agg.get("decision") == "pass"
        and strict_applied > 0
        and expected_strict > 0
        and runtime_errors == 0
        and strict_cache_miss == 0
        and unsafe_preferred == 0
        and unsafe_conflict == 0
        and unsafe_applied == 0
        and int(_num(effect_agg.get("unsafe_applied_steps"))) == 0
        and not hard_regressions
        and runtime_complete
        and joint_complete
        and not readiness_v31.get("controlled_replay_allowed", False)
        and not readiness_v31.get("controlled_replay_execution_allowed", False)
        and not readiness_v31.get("performance_claim_allowed", False)
        and not readiness_v31.get("promotion_evidence", False)
    )
    excluded = sorted(set(_scenario_ids(v31_manifest) + [str(item) for item in extra_excluded_scenario_ids if str(item)]))
    return {
        "schema_version": "controlled_canary_shadow_sourced_result_review_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "v31 shadow-sourced controlled canary result review",
            "reopens_rejected_preset": False,
        },
        "shadow_sourced_canary_result_review_pass": review_pass,
        "independent_holdout_admission_review_pass": review_pass,
        "v31_controlled_canary_pass": v31_pass,
        "control_effect_observed": control_effect_observed,
        "effect_decision": "positive_canary_signal" if control_effect_observed else "no_control_effect_observed",
        "independent_triggered_holdout_discovery_required": review_pass,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "excluded_scenario_ids": excluded,
        "metrics": {
            "expected_strict_eligible_applied_steps": expected_strict,
            "strict_applied_steps": strict_applied,
            "would_apply_steps": int(_num(trace_agg.get("would_apply_steps"))),
            "strict_filtered_steps": int(_num(trace_agg.get("strict_filtered_steps"))),
            "unsafe_preferred_steps": unsafe_preferred,
            "unsafe_conflict_steps": unsafe_conflict,
            "unsafe_applied_steps": unsafe_applied,
            "runtime_error_steps": runtime_errors,
            "strict_cache_miss_runtime_error_steps": strict_cache_miss,
            "runtime_provenance_record_count": int(_num(runtime_provenance_audit.get("record_count"))),
            "joint_prediction_row_count": int(_num(joint_prediction_readiness.get("row_count"))),
            "scenario_count": int(_num(summary_agg.get("scenario_count"), len(_scenario_ids(v31_manifest)))),
        },
        "hard_safety_regressions": hard_regressions,
        "blockers": [] if review_pass else ["shadow_sourced_canary_result_review_not_passed"],
        "next_action": (
            "independent_triggered_holdout_discovery_required"
            if review_pass
            else "shadow_sourced_canary_failure_review"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Shadow-Sourced Controlled Canary Result Review v32",
        "",
        f"- Review pass: {report.get('shadow_sourced_canary_result_review_pass', False)}",
        f"- v31 controlled canary pass: {report.get('v31_controlled_canary_pass', False)}",
        f"- Control effect observed: {report.get('control_effect_observed', False)}",
        f"- Effect decision: `{report.get('effect_decision', '')}`",
        "- Controlled replay allowed: false",
        "- Controlled replay execution allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Excluded Scenarios",
        "",
    ]
    for scenario_id in report.get("excluded_scenario_ids", []) or []:
        lines.append(f"- `{scenario_id}`")
    lines.extend(["", "## Metrics", "", "| metric | value |", "| --- | --- |"])
    for key, value in dict(report.get("metrics", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Blockers", ""])
    blockers = report.get("blockers", []) or []
    lines.extend(f"- `{item}`" for item in blockers) if blockers else lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-v31-json", required=True)
    parser.add_argument("--v31-manifest-json", required=True)
    parser.add_argument("--summary-audit-json", required=True)
    parser.add_argument("--trace-audit-json", required=True)
    parser.add_argument("--strict-effect-audit-json", required=True)
    parser.add_argument("--runtime-provenance-audit-json", required=True)
    parser.add_argument("--joint-prediction-readiness-json", required=True)
    parser.add_argument("--exclude-scenario-id", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        readiness_v31=_load(args.readiness_v31_json),
        v31_manifest=_load(args.v31_manifest_json),
        summary_audit=_load(args.summary_audit_json),
        trace_audit=_load(args.trace_audit_json),
        strict_effect_audit=_load(args.strict_effect_audit_json),
        runtime_provenance_audit=_load(args.runtime_provenance_audit_json),
        joint_prediction_readiness=_load(args.joint_prediction_readiness_json),
        extra_excluded_scenario_ids=args.exclude_scenario_id,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"shadow_sourced_canary_result_review_pass={report['shadow_sourced_canary_result_review_pass']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["shadow_sourced_canary_result_review_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
