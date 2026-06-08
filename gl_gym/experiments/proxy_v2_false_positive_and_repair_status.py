"""Summarize proxy-v2 false positives and candidate repair readiness.

The report is read-only and intentionally conservative: it never recommends
controlled replay while pure-hot-dry warnings or unresolved candidate-pool gaps
remain.
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


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _loads(paths: Sequence[str] | None) -> list[dict[str, Any]]:
    return [_load(path) for path in (paths or [])]


def _first_report(reports: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return reports[0] if reports else {}


def _aggregate_candidate_response(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sensitivity: dict[str, dict[str, int]] = {}
    false_safe = 0
    v2_unwarned = 0
    missing = 0
    for report in reports:
        for trace in report.get("traces", []) or []:
            if not isinstance(trace, Mapping):
                continue
            false_safe += int(_num(trace.get("false_safe_canopy_count")))
            v2_unwarned += int(_num(trace.get("v2_unwarned_false_safe_count")))
            missing += int(bool(trace.get("missing_prediction_metadata")))
            for scale, item in dict(trace.get("proxy_v2_buffer_sensitivity", {})).items():
                if not isinstance(item, Mapping):
                    continue
                target = sensitivity.setdefault(
                    str(scale),
                    {"v2_unwarned_false_safe": 0, "v2_warned_false_safe": 0, "pure_hot_dry_affected_steps": 0},
                )
                target["v2_unwarned_false_safe"] += int(_num(item.get("v2_unwarned_false_safe")))
                target["v2_warned_false_safe"] += int(_num(item.get("v2_warned_false_safe")))
                target["pure_hot_dry_affected_steps"] += int(_num(item.get("pure_hot_dry_affected_steps")))
    return {
        "false_safe_count": false_safe,
        "v2_unwarned_false_safe_count": v2_unwarned,
        "missing_prediction_metadata_trace_count": missing,
        "buffer_sensitivity": sensitivity,
    }


def build_report(
    *,
    shadow_validation_reports: Sequence[Mapping[str, Any]],
    candidate_response_reports: Sequence[Mapping[str, Any]] = (),
    candidate_repair_reports: Sequence[Mapping[str, Any]] = (),
    post_guardrail_reports: Sequence[Mapping[str, Any]] = (),
    action_diff_reports: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    shadow = _first_report(shadow_validation_reports)
    candidate_response = _aggregate_candidate_response(candidate_response_reports)
    repair = _first_report(candidate_repair_reports)
    post_guardrail = _first_report(post_guardrail_reports)
    action_diff = _first_report(action_diff_reports)

    false_positive_steps = int(_num(shadow.get("affected_pure_hot_dry_steps")))
    v2_unwarned = int(_num(shadow.get("false_safe_v1_unwarned_by_v2")))
    action_diff_steps = int(_num(action_diff.get("action_diff_steps")))
    missing_step_count = int(_num(action_diff.get("missing_step_count")))
    candidate_gap_steps = int(_num(repair.get("candidate_pool_gap_steps")))
    repair_available_steps = int(_num(repair.get("repair_candidate_available_steps")))
    harmful = int(_num(post_guardrail.get("harmful_override_count")))
    explainable = int(_num(post_guardrail.get("harmful_override_explainable_by_proxy_v2_or_candidate_repair_count")))
    buffer_sensitivity = dict(shadow.get("buffer_sensitivity_summary", {}))

    gates = {
        "metadata_only_action_diff_zero": bool(action_diff.get("trace_count", 0) and action_diff_steps == 0 and missing_step_count == 0),
        "selected_suite_v2_unwarned_false_safe_zero": bool(v2_unwarned == 0),
        "pure_hot_dry_false_positive_explained": bool(false_positive_steps == 0 or shadow.get("warning_classification_counts")),
        "candidate_repair_not_empty": bool(candidate_gap_steps == 0 or repair_available_steps > 0),
        "harmful_override_root_cause_classified": bool(harmful == 0 or post_guardrail.get("root_cause_summary")),
    }
    if false_positive_steps > 0:
        next_action = "proxy_v2_false_positive_calibration"
    elif candidate_gap_steps > 0:
        next_action = "candidate_generation_shadow_design"
    elif harmful and explainable < harmful:
        next_action = "post_guardrail_root_cause_deepening"
    elif all(gates.values()):
        next_action = "plan_controlled_replay_only"
    else:
        next_action = "repair_proxy_or_candidate_pool"

    return {
        "schema_version": "proxy_v2_false_positive_and_repair_status_v1",
        "mainline_alignment": {
            "affected_layers": ["Response Estimate", "Safety Boundary", "Candidate Pool", "Evaluation"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only / metadata-only",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "next_action": next_action,
        "gates": gates,
        "false_positive_decomposition": {
            "affected_pure_hot_dry_steps": false_positive_steps,
            "pure_hot_dry_warning_without_hard_event_next": int(
                _num(shadow.get("pure_hot_dry_warning_without_hard_event_next"))
            ),
            "warning_classification_counts": dict(shadow.get("warning_classification_counts", {})),
            "warning_action_pattern_counts": dict(shadow.get("warning_action_pattern_counts", {})),
            "warning_reason_counts": dict(shadow.get("warning_reason_counts", {})),
            "top_pure_hot_dry_false_positive_scenarios": list(
                shadow.get("top_pure_hot_dry_false_positive_scenarios", [])
            ),
        },
        "buffer_sensitivity": buffer_sensitivity,
        "candidate_response": candidate_response,
        "candidate_repair": {
            "candidate_pool_gap_steps": candidate_gap_steps,
            "repair_candidate_available_steps": repair_available_steps,
            "candidate_pool_repair_conclusion": str(repair.get("candidate_pool_repair_conclusion", "")),
            "repair_quality_counts": dict(repair.get("repair_quality_counts", {})),
        },
        "post_guardrail": {
            "harmful_override_count": harmful,
            "harmful_override_preventable_by_proxy_v2_count": int(
                _num(post_guardrail.get("harmful_override_preventable_by_proxy_v2_count"))
            ),
            "harmful_override_explainable_by_proxy_v2_or_candidate_repair_count": explainable,
            "root_cause_summary": dict(post_guardrail.get("root_cause_summary", {})),
            "harmful_root_cause_combinations": dict(post_guardrail.get("harmful_root_cause_combinations", {})),
        },
        "action_diff": {
            "trace_count": int(_num(action_diff.get("trace_count"))),
            "action_diff_steps": action_diff_steps,
            "missing_step_count": missing_step_count,
            "max_abs_delta": _num(action_diff.get("max_abs_delta")),
        },
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    fp = report.get("false_positive_decomposition", {}) if isinstance(report.get("false_positive_decomposition"), Mapping) else {}
    repair = report.get("candidate_repair", {}) if isinstance(report.get("candidate_repair"), Mapping) else {}
    post = report.get("post_guardrail", {}) if isinstance(report.get("post_guardrail"), Mapping) else {}
    action = report.get("action_diff", {}) if isinstance(report.get("action_diff"), Mapping) else {}
    lines = [
        "# Proxy v2 False-Positive and Repair Status",
        "",
        "## Mainline Alignment",
        "",
        "- Mode: shadow-only / metadata-only",
        "- Default `llm_rspc_v2` changed: false",
        "- Reopens rejected preset: false",
        "- Controlled replay allowed: false",
        "",
        "## Decision",
        "",
        f"- Next action: `{report.get('next_action', '')}`",
        f"- Performance claim allowed: {report.get('performance_claim_allowed', False)}",
        "",
        "## Gates",
        "",
        "| gate | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("gates", {})).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## False-Positive Decomposition",
            "",
            f"- Affected pure-hot-dry steps: {fp.get('affected_pure_hot_dry_steps', 0)}",
            f"- Pure-hot-dry warning without next hard event: {fp.get('pure_hot_dry_warning_without_hard_event_next', 0)}",
            f"- Classification counts: {json.dumps(fp.get('warning_classification_counts', {}), sort_keys=True)}",
            f"- Action-pattern counts: {json.dumps(fp.get('warning_action_pattern_counts', {}), sort_keys=True)}",
            "",
            "### Top Affected Scenarios",
            "",
            "| scenario | affected_pure_hot_dry_steps | warning_v2 |",
            "| --- | ---: | ---: |",
        ]
    )
    for item in fp.get("top_pure_hot_dry_false_positive_scenarios", []) or []:
        if isinstance(item, Mapping):
            lines.append(
                f"| {item.get('scenario_id', '')} | {item.get('affected_pure_hot_dry_steps', 0)} | {item.get('warning_v2', 0)} |"
            )
    lines.extend(
        [
            "",
            "## Buffer Sensitivity",
            "",
            "| scale | v2_unwarned_false_safe | pure_hot_dry_affected |",
            "| --- | ---: | ---: |",
        ]
    )
    for scale, item in dict(report.get("buffer_sensitivity", {})).items():
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"| {scale} | {int(_num(item.get('v2_unwarned_false_safe')))} | {int(_num(item.get('pure_hot_dry_affected_steps')))} |"
        )
    lines.extend(
        [
            "",
            "## Candidate Repair",
            "",
            f"- Candidate-pool gap steps: {repair.get('candidate_pool_gap_steps', 0)}",
            f"- Repair-candidate available steps: {repair.get('repair_candidate_available_steps', 0)}",
            f"- Conclusion: `{repair.get('candidate_pool_repair_conclusion', '')}`",
            f"- Repair quality counts: {json.dumps(repair.get('repair_quality_counts', {}), sort_keys=True)}",
            "",
            "## Post-Guardrail",
            "",
            f"- Harmful override count: {post.get('harmful_override_count', 0)}",
            f"- Explainable by proxy v2 or candidate repair: {post.get('harmful_override_explainable_by_proxy_v2_or_candidate_repair_count', 0)}",
            f"- Root causes: {json.dumps(post.get('root_cause_summary', {}), sort_keys=True)}",
            "",
            "## Action Diff",
            "",
            f"- Trace count: {action.get('trace_count', 0)}",
            f"- Action diff steps: {action.get('action_diff_steps', 0)}",
            f"- Missing step count: {action.get('missing_step_count', 0)}",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-validation-json", action="append", default=[])
    parser.add_argument("--candidate-response-json", action="append", default=[])
    parser.add_argument("--candidate-repair-json", action="append", default=[])
    parser.add_argument("--post-guardrail-json", action="append", default=[])
    parser.add_argument("--action-diff-json", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        shadow_validation_reports=_loads(args.shadow_validation_json),
        candidate_response_reports=_loads(args.candidate_response_json),
        candidate_repair_reports=_loads(args.candidate_repair_json),
        post_guardrail_reports=_loads(args.post_guardrail_json),
        action_diff_reports=_loads(args.action_diff_json),
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
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
