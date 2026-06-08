"""Taxonomize candidate-generation shadow failures from suite output."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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


def _candidate_reason_counts(record: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for candidate in record.get("hypothetical_candidates", []) or []:
        if not isinstance(candidate, Mapping):
            continue
        reason = str(candidate.get("feasibility_reason", "") or "")
        quality = str(candidate.get("tradeoff_quality", "") or "")
        if reason:
            for part in reason.split(","):
                if part:
                    counts[f"reason:{part}"] += 1
        if quality:
            counts[f"quality:{quality}"] += 1
        if bool(candidate.get("post_guardrail_reintroduces_canopy_risk")):
            counts["post_guardrail_reintroduces_canopy_risk"] += 1
        if bool(candidate.get("conservative_ineffective_action")):
            counts["conservative_ineffective_action"] += 1
    return counts


def _taxonomy(record: Mapping[str, Any]) -> str:
    klass = str(record.get("repair_class", "") or "")
    if klass == "repair_post_guardrail_safe":
        return "localized_safe_repair_signal"
    if klass == "repair_becomes_unsafe_after_post_guardrail":
        return "post_guardrail_destruction"
    if klass == "repair_dry_relief_lost":
        return "dry_relief_lost"
    if klass == "repair_false_positive_blocker":
        return "false_positive_blocker"
    if klass == "no_safe_repair_found":
        reasons = _candidate_reason_counts(record)
        if reasons.get("reason:canopy_risk_v2", 0) or reasons.get("reason:dew_risk", 0):
            return "candidate_space_insufficiency"
        if reasons.get("quality:conservative_ineffective", 0):
            return "conservative_noop"
        return "no_safe_repair_found"
    return "unclassified"


def build_report(candidate_suite_report: Mapping[str, Any]) -> dict[str, Any]:
    taxonomy_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    by_scenario: dict[str, Counter[str]] = defaultdict(Counter)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)
    rows: list[dict[str, Any]] = []
    for trace in candidate_suite_report.get("traces", []) or []:
        if not isinstance(trace, Mapping):
            continue
        for record in trace.get("records", []) or []:
            if not isinstance(record, Mapping):
                continue
            taxonomy = _taxonomy(record)
            reasons = _candidate_reason_counts(record)
            taxonomy_counts[taxonomy] += 1
            reason_counts.update(reasons)
            scenario = str(record.get("scenario_id", trace.get("scenario_id", "")) or "")
            category = str(record.get("category", trace.get("category", "")) or "")
            by_scenario[scenario][taxonomy] += 1
            by_category[category][taxonomy] += 1
            rows.append(
                {
                    "scenario_id": scenario,
                    "category": category,
                    "preset": record.get("preset", trace.get("preset", "")),
                    "step": record.get("step"),
                    "repair_class": record.get("repair_class", ""),
                    "taxonomy": taxonomy,
                    "best_candidate": record.get("best_candidate", ""),
                    "best_candidate_family": record.get("best_candidate_family", ""),
                    "feasible_count": int(_num(record.get("feasible_count"))),
                    "useful_count": int(_num(record.get("useful_count"))),
                    "useful_post_guardrail_safe_count": int(_num(record.get("useful_post_guardrail_safe_count"))),
                    "reason_counts": dict(sorted(reasons.items())),
                }
            )
    safe_signal = int(taxonomy_counts.get("localized_safe_repair_signal", 0))
    broad_fail = int(
        taxonomy_counts.get("candidate_space_insufficiency", 0)
        + taxonomy_counts.get("dry_relief_lost", 0)
        + taxonomy_counts.get("no_safe_repair_found", 0)
        + taxonomy_counts.get("post_guardrail_destruction", 0)
    )
    next_action = "candidate_space_redesign_shadow" if broad_fail >= safe_signal else "post_guardrail_policy_shadow_design"
    return {
        "schema_version": "candidate_generation_failure_taxonomy_v1",
        "mainline_alignment": {
            "affected_layers": ["Candidate Pool", "Safety Boundary"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only taxonomy",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "next_action": next_action,
        "taxonomy_counts": dict(sorted(taxonomy_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "scenario_taxonomy_counts": {key: dict(sorted(value.items())) for key, value in sorted(by_scenario.items())},
        "category_taxonomy_counts": {key: dict(sorted(value.items())) for key, value in sorted(by_category.items())},
        "rows": rows,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate Generation Failure Taxonomy",
        "",
        "## Mainline Alignment",
        "",
        "- Mode: shadow-only taxonomy",
        "- Default `llm_rspc_v2` changed: false",
        "- Reopens rejected preset: false",
        "",
        "## Decision",
        "",
        f"- Controlled replay allowed: {report.get('controlled_replay_allowed', False)}",
        f"- Performance claim allowed: {report.get('performance_claim_allowed', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Taxonomy Counts",
        "",
        "| taxonomy | count |",
        "| --- | ---: |",
    ]
    for label, count in dict(report.get("taxonomy_counts", {})).items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Reason Counts", "", "| reason | count |", "| --- | ---: |"])
    for label, count in dict(report.get("reason_counts", {})).items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Scenario Taxonomy", "", "| scenario | counts |", "| --- | --- |"])
    for scenario, counts in dict(report.get("scenario_taxonomy_counts", {})).items():
        lines.append(f"| {scenario} | {json.dumps(counts, sort_keys=True)} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-suite-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.candidate_suite_json))
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
