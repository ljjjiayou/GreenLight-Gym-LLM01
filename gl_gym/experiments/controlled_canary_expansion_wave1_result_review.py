"""Review Wave-1 controlled canary expansion before independent holdout discovery."""

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


def build_report(
    *,
    readiness_v23: Mapping[str, Any],
    wave1_manifest: Mapping[str, Any],
    triggered_manifest: Mapping[str, Any],
    summary_audit: Mapping[str, Any],
    trace_audit: Mapping[str, Any],
    strict_effect_audit: Mapping[str, Any],
    runtime_provenance_audit: Mapping[str, Any],
    joint_prediction_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    summary_agg = _aggregate(summary_audit)
    trace_agg = _aggregate(trace_audit)
    effect_agg = _aggregate(strict_effect_audit)
    strict_applied = int(_num(trace_agg.get("strict_applied_steps")))
    unsafe_preferred = int(_num(trace_agg.get("unsafe_preferred_steps")))
    unsafe_conflict = int(_num(trace_agg.get("unsafe_conflict_steps")))
    unsafe_applied = int(_num(trace_agg.get("unsafe_applied_steps")))
    runtime_complete = bool(runtime_provenance_audit.get("audit_status") == "runtime_provenance_complete")
    joint_complete = bool(joint_prediction_readiness.get("ready_for_policy_judgment", False))
    wave1_pass = bool(readiness_v23.get("wave1_controlled_canary_pass", False))
    no_effect = bool(
        not readiness_v23.get("control_effect_observed", True)
        and readiness_v23.get("effect_decision") == "expansion_no_trigger"
        and strict_applied == 0
    )
    review_pass = bool(
        wave1_pass
        and no_effect
        and summary_agg.get("decision") == "pass"
        and int(_num(effect_agg.get("runtime_error_steps"))) == 0
        and int(_num(effect_agg.get("unsafe_applied_steps"))) == 0
        and unsafe_preferred == 0
        and unsafe_conflict == 0
        and unsafe_applied == 0
        and runtime_complete
        and joint_complete
        and not readiness_v23.get("performance_claim_allowed", False)
        and not readiness_v23.get("promotion_evidence", False)
    )
    excluded = sorted(set(_scenario_ids(wave1_manifest) + _scenario_ids(triggered_manifest)))
    return {
        "schema_version": "controlled_canary_expansion_wave1_result_review_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "controlled canary expansion result review",
            "reopens_rejected_preset": False,
        },
        "wave1_result_review_pass": review_pass,
        "wave1_controlled_canary_pass": wave1_pass,
        "control_effect_observed": False,
        "effect_decision": "expansion_no_trigger" if no_effect else str(readiness_v23.get("effect_decision", "")),
        "independent_triggered_holdout_discovery_required": review_pass,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "excluded_scenario_ids": excluded,
        "metrics": {
            "wave1_scenario_count": int(_num(readiness_v23.get("post_run_metrics", {}).get("scenario_count"))),
            "strict_applied_steps": strict_applied,
            "would_apply_steps": int(_num(trace_agg.get("would_apply_steps"))),
            "strict_filtered_steps": int(_num(trace_agg.get("strict_filtered_steps"))),
            "unsafe_preferred_steps": unsafe_preferred,
            "unsafe_conflict_steps": unsafe_conflict,
            "unsafe_applied_steps": unsafe_applied,
            "runtime_provenance_record_count": int(_num(runtime_provenance_audit.get("record_count"))),
            "joint_prediction_row_count": int(_num(joint_prediction_readiness.get("row_count"))),
        },
        "blockers": [] if review_pass else ["wave1_result_review_not_passed"],
        "next_action": (
            "independent_triggered_holdout_discovery_required"
            if review_pass
            else "controlled_canary_expansion_wave1_failure_review"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Wave-1 Controlled Canary Expansion Result Review",
        "",
        f"- Review pass: {report.get('wave1_result_review_pass', False)}",
        f"- Wave-1 pass: {report.get('wave1_controlled_canary_pass', False)}",
        f"- Control effect observed: {report.get('control_effect_observed', False)}",
        f"- Effect decision: `{report.get('effect_decision', '')}`",
        "- Controlled replay allowed: false",
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
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-v23-json", required=True)
    parser.add_argument("--wave1-manifest-json", required=True)
    parser.add_argument("--triggered-manifest-json", required=True)
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
        readiness_v23=_load(args.readiness_v23_json),
        wave1_manifest=_load(args.wave1_manifest_json),
        triggered_manifest=_load(args.triggered_manifest_json),
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
    print(f"wave1_result_review_pass={report['wave1_result_review_pass']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["wave1_result_review_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
