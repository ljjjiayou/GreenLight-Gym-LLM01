"""Find read-only near-miss sources for future shadow-only trigger sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NEAR_MISS_REASONS = {
    "margin_below_strict_min",
    "canopy_reserve_low",
    "temp_headroom_low",
    "not_severe_dry",
}


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


def _scenario_parts(scenario_id: str) -> dict[str, int]:
    parts = {"year": 0, "day": 0, "seed": 0, "max_steps": 240}
    for token in str(scenario_id).split("_"):
        try:
            if token.startswith("y"):
                parts["year"] = int(token[1:])
            elif token.startswith("d"):
                parts["day"] = int(token[1:])
            elif token.startswith("s"):
                parts["seed"] = int(token[1:])
            elif token.startswith("n"):
                parts["max_steps"] = int(token[1:])
        except ValueError:
            continue
    return parts


def _rows_by_scenario(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        scenario_id = str(row.get("scenario_id", "") or "")
        if scenario_id:
            out.setdefault(scenario_id, []).append(row)
    return out


def build_report(*, blocker_diagnosis: Mapping[str, Any], discovery: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {str(item) for item in discovery.get("excluded_scenario_ids", []) or [] if str(item or "")}
    near_rows = [
        row
        for row in discovery.get("sample_rows", []) or []
        if isinstance(row, Mapping)
        and bool(row.get("strict_filtered", False))
        and not bool(row.get("strict_eligible_applied", False))
        and str(row.get("reason", "") or "") in NEAR_MISS_REASONS
        and str(row.get("scenario_id", "") or "") not in excluded
    ]
    scenarios = []
    for scenario_id, rows in sorted(_rows_by_scenario(near_rows).items()):
        reasons: dict[str, int] = {}
        candidates: dict[str, int] = {}
        variants: dict[str, int] = {}
        gaps: list[float] = []
        for row in rows:
            reason = str(row.get("reason", "") or "unknown")
            candidate = str(row.get("candidate", "") or "unknown")
            variant = str(row.get("variant", "") or "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
            candidates[candidate] = candidates.get(candidate, 0) + 1
            variants[variant] = variants.get(variant, 0) + 1
            gaps.append(_num(row.get("min_margin")) - _num(row.get("margin")))
        scenarios.append(
            {
                "scenario_id": scenario_id,
                **_scenario_parts(scenario_id),
                "role": "near_miss_shadow_trigger_source",
                "near_miss_rows": len(rows),
                "strict_applied_steps": 0,
                "reason_counts": dict(sorted(reasons.items())),
                "candidate_counts": dict(sorted(candidates.items())),
                "variant_counts": dict(sorted(variants.items())),
                "min_margin_gap": round(min(gaps), 6) if gaps else None,
                "max_margin_gap": round(max(gaps), 6) if gaps else None,
                "avg_margin_gap": round(sum(gaps) / len(gaps), 6) if gaps else None,
            }
        )
    found = bool(scenarios)
    return {
        "schema_version": "controlled_canary_shadow_trigger_sourcing_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only near-miss shadow trigger sourcing",
            "reopens_rejected_preset": False,
        },
        "shadow_trigger_sources_found": found,
        "source_scope": "shadow_trigger_source_candidates_only" if found else "blocked_no_source",
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "strict_threshold_changed": False,
        "controller_logic_changed": False,
        "near_miss_reasons": sorted(NEAR_MISS_REASONS),
        "excluded_scenario_ids": sorted(excluded),
        "source_scenarios": scenarios,
        "blocker_taxonomy": [] if found else ["no_near_miss_shadow_trigger_source"],
        "next_action": (
            "generate_shadow_trigger_source_manifest"
            if found
            else "independent_trigger_source_exhausted_or_needs_new_shadow_sweep"
        ),
        "source_from_blocker_diagnosis_complete": bool(
            blocker_diagnosis.get("blocker_diagnosis_complete", False)
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Near-Miss Shadow Trigger Sourcing",
        "",
        f"- Sources found: {report.get('shadow_trigger_sources_found', False)}",
        f"- Source scope: `{report.get('source_scope', '')}`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "- Strict threshold changed: false",
        "- Controller logic changed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Source Scenarios",
        "",
        "| scenario | near-miss rows | reasons |",
        "| --- | ---: | --- |",
    ]
    for scenario in report.get("source_scenarios", []) or []:
        if isinstance(scenario, Mapping):
            reasons = ", ".join(f"{k}:{v}" for k, v in dict(scenario.get("reason_counts", {})).items())
            lines.append(f"| {scenario.get('scenario_id', '')} | {scenario.get('near_miss_rows', 0)} | {reasons} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocker-diagnosis-json", required=True)
    parser.add_argument("--discovery-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        blocker_diagnosis=_load(args.blocker_diagnosis_json),
        discovery=_load(args.discovery_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"shadow_trigger_sources_found={report['shadow_trigger_sources_found']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
