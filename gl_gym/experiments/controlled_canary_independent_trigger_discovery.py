"""Discover unused strict-trigger holdout scenarios from existing traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.controlled_canary_strict_trigger_discovery import (  # noqa: E402
    BASELINE_CONTROLLER,
    _counts,
    _num,
    _row_record,
    _scenario_parts,
    _strict_targeted_row_record,
    _truthy,
)
from gl_gym.experiments.profile_scorer_decision_audit import discover_traces, read_trace, trace_identity  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _excluded_from_review(review: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(review, Mapping):
        return set()
    return {str(item) for item in review.get("excluded_scenario_ids", []) or [] if str(item or "")}


def _is_unsafe_source_row(row: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    control_applied = bool(record.get("control_applied", False))
    control_safe = bool(record.get("control_safe_hot_dry", False))
    safety_gate = str(record.get("safety_gate_reason", "none") or "none")
    return bool(
        _truthy(row, "rspc_hot_dry_replay_unsafe_preferred")
        or _truthy(row, "rspc_hot_dry_replay_unsafe_conflict")
        or (control_applied and ((not control_safe) or safety_gate not in {"", "none"}))
    )


def _scenario_report(scenario_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        **_scenario_parts(scenario_id),
        "discovery_role": "independent_strict_trigger_candidate",
        "expected_strict_eligible_applied_steps": int(sum(bool(row.get("strict_eligible_applied", False)) for row in rows)),
        "would_apply_steps": int(sum(bool(row.get("would_apply", False)) for row in rows)),
        "strict_filtered_steps": int(sum(bool(row.get("strict_filtered", False)) for row in rows)),
        "unsafe_source_rows": int(sum(bool(row.get("unsafe_source_row", False)) for row in rows)),
        "candidate_counts": _counts(str(row.get("candidate", "")) for row in rows),
        "variant_counts": _counts(str(row.get("variant", "")) for row in rows),
    }


def discover_independent_triggers(
    inputs: Sequence[str | Path],
    *,
    excluded_scenario_ids: Iterable[str],
    max_records: int = 200,
    strict_targeted_shadow_only: bool = False,
) -> dict[str, Any]:
    excluded = {str(item) for item in excluded_scenario_ids if str(item or "")}
    relevant_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    excluded_trace_count = 0
    for path in discover_traces(inputs):
        identity = trace_identity(path)
        scenario_id = str(identity.get("scenario_id", "") or "")
        if scenario_id in excluded:
            excluded_trace_count += 1
            continue
        controller = str(identity.get("controller", "") or "")
        if strict_targeted_shadow_only and controller != BASELINE_CONTROLLER:
            continue
        rows = read_trace(path)
        trace_rows: list[dict[str, Any]] = []
        for row in rows:
            if "rspc_hot_dry_replay_enabled" not in row and "rspc_hot_dry_proposer_control_enabled" not in row:
                continue
            record = (
                _strict_targeted_row_record(path, identity, row)
                if strict_targeted_shadow_only
                else _row_record(path, identity, row)
            )
            record["unsafe_source_row"] = _is_unsafe_source_row(row, record)
            if record["would_apply"] or record["strict_filtered"] or record["strict_eligible_applied"]:
                trace_rows.append(record)
                by_scenario.setdefault(scenario_id, []).append(record)
                if len(relevant_rows) < max_records:
                    relevant_rows.append(record)
        traces.append(
            {
                **identity,
                "rows": len(rows),
                "relevant_rows": len(trace_rows),
                "would_apply_rows": sum(1 for item in trace_rows if item["would_apply"]),
                "strict_filtered_rows": sum(1 for item in trace_rows if item["strict_filtered"]),
                "expected_strict_eligible_applied_rows": sum(
                    1 for item in trace_rows if item["strict_eligible_applied"]
                ),
                "unsafe_source_rows": sum(1 for item in trace_rows if item.get("unsafe_source_row", False)),
                "reason_counts": _counts(str(item.get("reason", "")) for item in trace_rows),
                "candidate_counts": _counts(str(item.get("candidate", "")) for item in trace_rows),
            }
        )

    scenario_reports = [_scenario_report(scenario_id, rows) for scenario_id, rows in sorted(by_scenario.items())]
    qualified = [
        item
        for item in scenario_reports
        if int(item.get("expected_strict_eligible_applied_steps", 0) or 0) > 0
        and int(item.get("unsafe_source_rows", 0) or 0) == 0
    ]
    unsafe_candidates = [
        item["scenario_id"]
        for item in scenario_reports
        if int(item.get("expected_strict_eligible_applied_steps", 0) or 0) > 0
        and int(item.get("unsafe_source_rows", 0) or 0) > 0
    ]
    expected_count = sum(int(item.get("expected_strict_eligible_applied_steps", 0) or 0) for item in qualified)
    would_apply_count = sum(int(trace.get("would_apply_rows", 0) or 0) for trace in traces)
    strict_filtered_count = sum(int(trace.get("strict_filtered_rows", 0) or 0) for trace in traces)
    blockers: list[str] = []
    if not qualified:
        blockers.append("no_independent_strict_trigger_candidate")
        if unsafe_candidates:
            blockers.append("unsafe_source_metadata_present")
        if strict_filtered_count > 0:
            blockers.append("strict_candidates_filtered")
        if would_apply_count > 0:
            blockers.append("only_shadow_would_apply_or_non_strict_signal")
    return {
        "schema_version": "controlled_canary_independent_trigger_discovery_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only independent strict trigger discovery",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "strict_targeted_shadow_only": bool(strict_targeted_shadow_only),
        "excluded_scenario_ids": sorted(excluded),
        "excluded_trace_count": int(excluded_trace_count),
        "independent_trigger_candidates_found": bool(qualified),
        "discovery_decision": "independent_trigger_candidates_found" if qualified else "blocked_no_independent_trigger_candidate",
        "expected_strict_eligible_applied_steps": int(expected_count),
        "would_apply_steps": int(would_apply_count),
        "strict_filtered_steps": int(strict_filtered_count),
        "unsafe_source_scenarios": sorted(unsafe_candidates),
        "qualified_trigger_scenarios": qualified,
        "blocker_taxonomy": blockers,
        "trace_count": len(traces),
        "traces": traces,
        "sample_rows": relevant_rows,
        "next_action": (
            "independent_wave2_manifest_and_cache_coverage"
            if qualified
            else "independent_trigger_blocker_diagnosis"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Independent Triggered Holdout Discovery",
        "",
        f"- Decision: `{report.get('discovery_decision', '')}`",
        f"- Candidates found: {report.get('independent_trigger_candidates_found', False)}",
        f"- Expected strict-eligible applied steps: {report.get('expected_strict_eligible_applied_steps', 0)}",
        f"- Excluded trace count: {report.get('excluded_trace_count', 0)}",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Qualified Scenarios",
        "",
        "| scenario | expected strict steps | unsafe source rows |",
        "| --- | ---: | ---: |",
    ]
    for scenario in report.get("qualified_trigger_scenarios", []) or []:
        if isinstance(scenario, Mapping):
            lines.append(
                f"| {scenario.get('scenario_id', '')} | {scenario.get('expected_strict_eligible_applied_steps', 0)} | {scenario.get('unsafe_source_rows', 0)} |"
            )
    lines.extend(["", "## Blockers", ""])
    blockers = report.get("blocker_taxonomy", []) or []
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-trace", nargs="+", required=True)
    parser.add_argument("--wave1-result-review-json", default="")
    parser.add_argument("--result-review-json", default="")
    parser.add_argument("--exclude-scenario-id", action="append", default=[])
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--strict-targeted-shadow-only", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    review_json = args.result_review_json or args.wave1_result_review_json
    review = _load(review_json) if review_json else {}
    excluded = _excluded_from_review(review) | {str(item) for item in args.exclude_scenario_id}
    report = discover_independent_triggers(
        args.input_trace,
        excluded_scenario_ids=excluded,
        max_records=int(args.max_records),
        strict_targeted_shadow_only=bool(args.strict_targeted_shadow_only),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"discovery_decision={report['discovery_decision']}")
    print(f"expected_strict_eligible_applied_steps={report['expected_strict_eligible_applied_steps']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
