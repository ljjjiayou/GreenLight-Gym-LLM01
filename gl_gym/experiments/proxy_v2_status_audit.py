"""Aggregate proxy-v2 shadow audit outputs into one promotion-gate status report.

This script is read-only.  It does not inspect or modify controller behavior;
it combines existing audit JSON files into a compact status summary that can be
attached to the project mainline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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


def _paths(paths: Sequence[str] | None) -> list[dict[str, Any]]:
    return [_load(path) for path in (paths or [])]


def _normalize_scenario_id(value: str) -> str:
    text = str(value or "")
    return text[: -len("_llm_rspc_v2")] if text.endswith("_llm_rspc_v2") else text


def _scenario_ids(reports: Sequence[Mapping[str, Any]]) -> list[str]:
    out: set[str] = set()
    for report in reports:
        scenario = _normalize_scenario_id(str(report.get("scenario_id", "") or ""))
        if scenario:
            out.add(scenario)
        for trace in report.get("traces", []) or []:
            if isinstance(trace, Mapping):
                scenario = _normalize_scenario_id(str(trace.get("scenario_id", "") or ""))
                if scenario:
                    out.add(scenario)
    return sorted(out)


def _candidate_response_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    false_safe = 0
    warned = 0
    predicted = 0
    missing = 0
    near_boundary_unwarned = Counter()
    per_trace: list[dict[str, Any]] = []
    for report in reports:
        for trace in report.get("traces", []) or []:
            if not isinstance(trace, Mapping):
                continue
            fs = int(_num(trace.get("false_safe_canopy_count")))
            w = int(_num(trace.get("v2_warning_false_safe_count")))
            u = int(_num(trace.get("v2_unwarned_false_safe_count"), max(0, fs - w)))
            false_safe += fs
            warned += w
            predicted += int(_num(trace.get("predicted_canopy_count")))
            missing += int(bool(trace.get("missing_prediction_metadata")))
            for bucket, summary in dict(trace.get("near_boundary_summary", {})).items():
                if isinstance(summary, Mapping):
                    near_boundary_unwarned[str(bucket)] += int(_num(summary.get("v2_unwarned_false_safe_count")))
            per_trace.append(
                {
                    "scenario_id": str(trace.get("scenario_id", "")),
                    "preset": str(trace.get("preset", "")),
                    "false_safe_count": fs,
                    "v2_warned_false_safe_count": w,
                    "v2_unwarned_false_safe_count": u,
                    "false_safe_rate": trace.get("false_safe_rate"),
                    "missing_prediction_metadata": bool(trace.get("missing_prediction_metadata")),
                }
            )
    return {
        "false_safe_count": false_safe,
        "v2_warned_false_safe_count": warned,
        "v2_unwarned_false_safe_count": int(max(0, false_safe - warned)),
        "predicted_canopy_count": predicted,
        "missing_prediction_metadata_trace_count": missing,
        "near_boundary_unwarned_false_safe": dict(sorted(near_boundary_unwarned.items())),
        "per_trace": per_trace,
    }


def _boundary_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    traces = []
    for report in reports:
        for trace in report.get("traces", []) or []:
            if isinstance(trace, Mapping):
                traces.append(trace)
    return {
        "trace_count": len(traces),
        "prevention_window_trace_count": sum(1 for t in traces if bool(t.get("prevention_window_available"))),
        "hard_event_only_trace_count": sum(1 for t in traces if bool(t.get("hard_event_only"))),
        "affected_pure_hot_dry_steps": int(sum(int(_num(t.get("affected_pure_hot_dry_steps"))) for t in traces)),
        "first_warning_steps": [
            {
                "scenario_id": str(t.get("scenario_id", "")),
                "preset": str(t.get("preset", "")),
                "first_boundary_warning_step": t.get("first_boundary_warning_step"),
                "lead_time_steps": t.get("lead_time_steps"),
                "affected_pure_hot_dry_steps": int(_num(t.get("affected_pure_hot_dry_steps"))),
            }
            for t in traces
        ],
    }


def _post_guardrail_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outcome_counts: Counter[str] = Counter()
    root_causes: Counter[str] = Counter()
    preventable = 0
    harmful = 0
    for report in reports:
        outcome_counts.update({str(k): int(v) for k, v in dict(report.get("aggregate_outcome_counts", {})).items()})
        root_causes.update({str(k): int(v) for k, v in dict(report.get("root_cause_summary", {})).items()})
        for trace in report.get("traces", []) or []:
            if not isinstance(trace, Mapping):
                continue
            for row in trace.get("rows", []) or []:
                if not isinstance(row, Mapping):
                    continue
                if str(row.get("outcome_label", "")) == "harmful_override":
                    harmful += 1
                    if bool(row.get("harmful_override_preventable_by_proxy_v2")):
                        preventable += 1
    return {
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "root_cause_summary": dict(sorted(root_causes.items())),
        "harmful_override_count": harmful,
        "harmful_override_preventable_by_proxy_v2_count": preventable,
        "harmful_override_preventable_by_proxy_v2_rate": float(preventable / harmful) if harmful else None,
    }


def _repair_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gap = 0
    available = 0
    conclusions: Counter[str] = Counter()
    for report in reports:
        gap += int(_num(report.get("candidate_pool_gap_steps")))
        available += int(_num(report.get("repair_candidate_available_steps")))
        conclusion = str(report.get("candidate_pool_repair_conclusion", "") or "")
        if conclusion:
            conclusions[conclusion] += 1
    if gap <= 0:
        aggregate_conclusion = "guardrail_filter_sufficient"
    elif available >= gap:
        aggregate_conclusion = "new_candidate_needed"
    elif available <= 0:
        aggregate_conclusion = "no_safe_repair_found"
    else:
        aggregate_conclusion = "response_model_unreliable"
    return {
        "candidate_pool_gap_steps": gap,
        "repair_candidate_available_steps": available,
        "candidate_pool_repair_conclusion": aggregate_conclusion,
        "source_conclusions": dict(sorted(conclusions.items())),
    }


def _action_diff_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diff_steps = 0
    missing_steps = 0
    trace_count = 0
    max_abs_delta = 0.0
    for report in reports:
        diff_steps += int(_num(report.get("action_diff_steps")))
        missing_steps += int(_num(report.get("missing_step_count")))
        trace_count += int(_num(report.get("trace_count")))
        max_abs_delta = max(max_abs_delta, _num(report.get("max_abs_delta")))
    return {
        "trace_count": trace_count,
        "action_diff_steps": diff_steps,
        "missing_step_count": missing_steps,
        "max_abs_delta": max_abs_delta,
        "verified_zero": bool(trace_count > 0 and diff_steps == 0 and missing_steps == 0),
    }


def _shadow_validation_summary(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trace_count = 0
    scenario_count = 0
    false_safe_unwarned = 0
    affected_pure_hot_dry = 0
    pure_warning_without_hard = 0
    for report in reports:
        trace_count += int(_num(report.get("trace_count")))
        scenario_count += int(_num(report.get("scenario_count")))
        false_safe_unwarned += int(_num(report.get("false_safe_v1_unwarned_by_v2")))
        affected_pure_hot_dry += int(_num(report.get("affected_pure_hot_dry_steps")))
        pure_warning_without_hard += int(_num(report.get("pure_hot_dry_warning_without_hard_event_next")))
    return {
        "trace_count": trace_count,
        "scenario_count": scenario_count,
        "false_safe_v1_unwarned_by_v2": false_safe_unwarned,
        "affected_pure_hot_dry_steps": affected_pure_hot_dry,
        "pure_hot_dry_warning_without_hard_event_next": pure_warning_without_hard,
        "available": bool(trace_count > 0),
    }


def build_status(
    *,
    candidate_response_reports: Sequence[Mapping[str, Any]],
    boundary_reports: Sequence[Mapping[str, Any]],
    post_guardrail_reports: Sequence[Mapping[str, Any]],
    candidate_repair_reports: Sequence[Mapping[str, Any]],
    action_diff_reports: Sequence[Mapping[str, Any]] = (),
    shadow_validation_reports: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    reports = (
        list(candidate_response_reports)
        + list(boundary_reports)
        + list(post_guardrail_reports)
        + list(candidate_repair_reports)
        + list(action_diff_reports)
        + list(shadow_validation_reports)
    )
    scenarios = _scenario_ids(reports)
    candidate_response = _candidate_response_summary(candidate_response_reports)
    boundary = _boundary_summary(boundary_reports)
    post_guardrail = _post_guardrail_summary(post_guardrail_reports)
    repair = _repair_summary(candidate_repair_reports)
    action_diff = _action_diff_summary(action_diff_reports)
    shadow_validation = _shadow_validation_summary(shadow_validation_reports)
    selected_suite_unwarned_ok = (
        bool(shadow_validation.get("false_safe_v1_unwarned_by_v2", 0) == 0)
        if shadow_validation.get("available")
        else "not_checked"
    )
    selected_pure_hot_dry_affected = int(shadow_validation.get("affected_pure_hot_dry_steps", 0))
    boundary_pure_hot_dry_affected = int(boundary.get("affected_pure_hot_dry_steps", 0))
    gates = {
        "canonical_failure_lead_time_warning": bool(boundary.get("prevention_window_trace_count", 0) > 0),
        "v2_unwarned_false_safe_zero": bool(candidate_response.get("v2_unwarned_false_safe_count", 0) == 0),
        "missing_prediction_metadata_zero": bool(candidate_response.get("missing_prediction_metadata_trace_count", 0) == 0),
        "pure_hot_dry_false_positive_zero": bool(
            boundary_pure_hot_dry_affected + selected_pure_hot_dry_affected == 0
        ),
        "harmful_override_explained_by_v2": bool(
            post_guardrail.get("harmful_override_count", 0) == 0
            or post_guardrail.get("harmful_override_preventable_by_proxy_v2_count", 0) > 0
        ),
        "candidate_repair_shadow_available": bool(
            repair.get("candidate_pool_gap_steps", 0) == 0
            or repair.get("repair_candidate_available_steps", 0) > 0
        ),
        "action_diff_verified_zero": bool(action_diff.get("verified_zero", False)),
        "selected_suite_v2_unwarned_false_safe_zero": selected_suite_unwarned_ok,
    }
    ready_without_action_diff = all(value is True for key, value in gates.items() if key != "action_diff_verified_zero")
    if len(scenarios) <= 1:
        next_action = "expand_proxy_v2_shadow_validation"
    elif ready_without_action_diff and not bool(action_diff.get("verified_zero", False)):
        next_action = "run_action_diff_before_controlled_replay_planning"
    elif ready_without_action_diff and bool(action_diff.get("verified_zero", False)):
        next_action = "prepare_controlled_replay_plan_only"
    else:
        next_action = "repair_proxy_or_candidate_pool"
    return {
        "schema_version": "proxy_v2_status_audit_v1",
        "mainline_alignment": {
            "affected_layers": ["Response Estimate", "Safety Gate", "Evaluation", "Candidate Pool"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only / metadata-only",
            "reopens_rejected_preset": False,
        },
        "scenario_ids": scenarios,
        "scenario_count": len(scenarios),
        "candidate_response": candidate_response,
        "canopy_boundary": boundary,
        "post_guardrail": post_guardrail,
        "candidate_repair": repair,
        "action_diff": action_diff,
        "shadow_validation": shadow_validation,
        "gates": gates,
        "controlled_replay_ready": False,
        "ready_without_action_diff_gate": bool(ready_without_action_diff),
        "next_action": next_action,
        "performance_claim_allowed": False,
        "merge_recommendation": False,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    cr = report.get("candidate_response", {}) if isinstance(report.get("candidate_response"), Mapping) else {}
    cb = report.get("canopy_boundary", {}) if isinstance(report.get("canopy_boundary"), Mapping) else {}
    pg = report.get("post_guardrail", {}) if isinstance(report.get("post_guardrail"), Mapping) else {}
    repair = report.get("candidate_repair", {}) if isinstance(report.get("candidate_repair"), Mapping) else {}
    action_diff = report.get("action_diff", {}) if isinstance(report.get("action_diff"), Mapping) else {}
    shadow_validation = report.get("shadow_validation", {}) if isinstance(report.get("shadow_validation"), Mapping) else {}
    lines = [
        "# Proxy v2 Shadow Status",
        "",
        "## Mainline Alignment",
        "",
        "- Mode: shadow-only / metadata-only",
        "- Default `llm_rspc_v2` changed: false",
        "- Reopens rejected preset: false",
        "- Performance claim allowed: false",
        "",
        "## Summary",
        "",
        f"- Scenario count: {report.get('scenario_count', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        f"- Controlled replay ready: **{str(report.get('controlled_replay_ready')).upper()}**",
        f"- Ready without action-diff gate: {report.get('ready_without_action_diff_gate', False)}",
        "",
        "## Gate Snapshot",
        "",
        "| gate | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("gates", {})).items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- False-safe count: {cr.get('false_safe_count', 0)}",
            f"- V2 warned false-safe count: {cr.get('v2_warned_false_safe_count', 0)}",
            f"- V2 unwarned false-safe count: {cr.get('v2_unwarned_false_safe_count', 0)}",
            f"- Prevention-window trace count: {cb.get('prevention_window_trace_count', 0)}",
            f"- Affected pure-hot-dry steps: {cb.get('affected_pure_hot_dry_steps', 0)}",
            f"- Harmful override count: {pg.get('harmful_override_count', 0)}",
            f"- Harmful override preventable by v2: {pg.get('harmful_override_preventable_by_proxy_v2_count', 0)}",
            f"- Candidate-pool gap steps: {repair.get('candidate_pool_gap_steps', 0)}",
            f"- Repair-candidate available steps: {repair.get('repair_candidate_available_steps', 0)}",
            f"- Candidate repair conclusion: `{repair.get('candidate_pool_repair_conclusion', '')}`",
            f"- Action diff steps: {action_diff.get('action_diff_steps', 0)}",
            f"- Action diff max abs delta: {_num(action_diff.get('max_abs_delta')):.8f}",
            f"- Selected-suite v2 unwarned false-safe: {shadow_validation.get('false_safe_v1_unwarned_by_v2', 0)}",
            f"- Selected-suite affected pure-hot-dry steps: {shadow_validation.get('affected_pure_hot_dry_steps', 0)}",
            "",
            "## Scenario IDs",
            "",
        ]
    )
    for scenario in report.get("scenario_ids", []) or []:
        lines.append(f"- `{scenario}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-response-json", action="append", default=[])
    parser.add_argument("--boundary-json", action="append", default=[])
    parser.add_argument("--post-guardrail-json", action="append", default=[])
    parser.add_argument("--candidate-repair-json", action="append", default=[])
    parser.add_argument("--action-diff-json", action="append", default=[])
    parser.add_argument("--shadow-validation-json", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_status(
        candidate_response_reports=_paths(args.candidate_response_json),
        boundary_reports=_paths(args.boundary_json),
        post_guardrail_reports=_paths(args.post_guardrail_json),
        candidate_repair_reports=_paths(args.candidate_repair_json),
        action_diff_reports=_paths(args.action_diff_json),
        shadow_validation_reports=_paths(args.shadow_validation_json),
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
