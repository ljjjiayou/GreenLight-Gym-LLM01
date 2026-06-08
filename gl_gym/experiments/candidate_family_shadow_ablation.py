"""Aggregate hypothetical repair candidates by family for shadow ablation."""

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

EXPECTED_FAMILIES = (
    "canopy_safe_hold_screen",
    "vent_floor_repair",
    "screen_cap_repair",
    "shade_preempt_repair",
    "dry_relief_preserving_repair",
    "dew_safe_hybrid_repair",
)
FAMILY_ALIASES = {
    "canopy_safe_min_vent": "vent_floor_repair",
    "canopy_safe_balanced_repair": "dry_relief_preserving_repair",
}


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _family(value: Any) -> str:
    raw = str(value or "unknown")
    return FAMILY_ALIASES.get(raw, raw)


def build_report(candidate_suite_report: Mapping[str, Any]) -> dict[str, Any]:
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in candidate_suite_report.get("traces", []) or []:
        if not isinstance(trace, Mapping):
            continue
        for record in trace.get("records", []) or []:
            if not isinstance(record, Mapping):
                continue
            record_class = str(record.get("repair_class", ""))
            for candidate in record.get("hypothetical_candidates", []) or []:
                if not isinstance(candidate, Mapping):
                    continue
                family = _family(candidate.get("candidate_family"))
                counts = family_counts[family]
                counts["candidate_count"] += 1
                if bool(candidate.get("feasible")):
                    counts["feasible_count"] += 1
                if bool(candidate.get("dry_relief_useful")):
                    counts["useful_dry_relief_count"] += 1
                if bool(candidate.get("post_guardrail_remains_safe")):
                    counts["post_guardrail_safe_count"] += 1
                if bool(candidate.get("dry_tradeoff")) or str(candidate.get("tradeoff_quality", "")) == "dry_tradeoff":
                    counts["dry_relief_lost_count"] += 1
                if record_class == "no_safe_repair_found":
                    counts["no_safe_repair_count"] += 1
                if record_class == "repair_false_positive_blocker":
                    counts["false_positive_blocker_count"] += 1
                if bool(candidate.get("post_guardrail_reintroduces_canopy_risk")):
                    counts["hard_safety_regression_count"] += 1
                if len(family_examples[family]) < 8:
                    family_examples[family].append(
                        {
                            "scenario_id": record.get("scenario_id", trace.get("scenario_id", "")),
                            "preset": record.get("preset", trace.get("preset", "")),
                            "step": record.get("step"),
                            "category": record.get("category", trace.get("category", "")),
                            "candidate": candidate.get("name", ""),
                            "record_class": record_class,
                            "feasible": bool(candidate.get("feasible")),
                            "dry_relief_useful": bool(candidate.get("dry_relief_useful")),
                            "post_guardrail_remains_safe": bool(candidate.get("post_guardrail_remains_safe")),
                            "tradeoff_quality": candidate.get("tradeoff_quality", ""),
                            "feasibility_reason": candidate.get("feasibility_reason", ""),
                        }
                    )
    for family in EXPECTED_FAMILIES:
        family_counts.setdefault(family, Counter())
        family_examples.setdefault(family, [])
    rows: list[dict[str, Any]] = []
    for family in sorted(family_counts):
        counts = family_counts[family]
        row = {
            "family": family,
            "candidate_count": int(counts.get("candidate_count", 0)),
            "feasible_count": int(counts.get("feasible_count", 0)),
            "useful_dry_relief_count": int(counts.get("useful_dry_relief_count", 0)),
            "post_guardrail_safe_count": int(counts.get("post_guardrail_safe_count", 0)),
            "dry_relief_lost_count": int(counts.get("dry_relief_lost_count", 0)),
            "no_safe_repair_count": int(counts.get("no_safe_repair_count", 0)),
            "false_positive_blocker_count": int(counts.get("false_positive_blocker_count", 0)),
            "hard_safety_regression_count": int(counts.get("hard_safety_regression_count", 0)),
            "examples": family_examples[family],
        }
        rows.append(row)
    any_hard = any(row["hard_safety_regression_count"] > 0 for row in rows)
    useful_families = [row["family"] for row in rows if row["useful_dry_relief_count"] and row["post_guardrail_safe_count"]]
    return {
        "schema_version": "candidate_family_shadow_ablation_v1",
        "mainline_alignment": {
            "affected_layers": ["Candidate Pool", "Safety Boundary"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only family ablation",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "useful_shadow_families": useful_families,
        "hard_safety_regression_present": any_hard,
        "candidate_family_ablation_conclusion": (
            "candidate_space_redesign_shadow"
            if useful_families and not any_hard
            else "no_safe_family_found"
        ),
        "families": rows,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate Family Shadow Ablation",
        "",
        "- Mode: shadow-only family ablation",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Conclusion: `{report.get('candidate_family_ablation_conclusion', '')}`",
        "",
        "| family | candidates | feasible | useful_dry_relief | post_guardrail_safe | dry_relief_lost | no_safe_repair | false_positive_blocker | hard_safety_regression |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("families", []) or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("family", "")),
                    str(row.get("candidate_count", 0)),
                    str(row.get("feasible_count", 0)),
                    str(row.get("useful_dry_relief_count", 0)),
                    str(row.get("post_guardrail_safe_count", 0)),
                    str(row.get("dry_relief_lost_count", 0)),
                    str(row.get("no_safe_repair_count", 0)),
                    str(row.get("false_positive_blocker_count", 0)),
                    str(row.get("hard_safety_regression_count", 0)),
                ]
            )
            + " |"
        )
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
    print(f"conclusion={report['candidate_family_ablation_conclusion']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
