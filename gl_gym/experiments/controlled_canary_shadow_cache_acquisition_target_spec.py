"""Build the v34 shadow-only cache acquisition target specification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAX_STEPS = 240


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _scenario_id(year: int, day: int, seed: int) -> str:
    return f"y{year}_d{day}_s{seed}_n{MAX_STEPS}"


def _v30_swept_scenarios() -> set[str]:
    scenarios: set[str] = set()
    for year in [2010, 2015]:
        for day in [59, 120, 180, 240]:
            for seed in [42, 43, 44]:
                scenarios.add(_scenario_id(year, day, seed))
    for day in [59, 120, 180, 240]:
        scenarios.add(_scenario_id(2020, day, 44))
    return scenarios


def _incomplete_backlog() -> set[str]:
    return {_scenario_id(2020, day, seed) for day in [59, 120, 180, 240] for seed in [42, 43]}


def build_report(*, near_miss_catalog: Mapping[str, Any], existing_cache_inventory: Mapping[str, Any]) -> dict[str, Any]:
    catalog_ready = bool(near_miss_catalog.get("near_miss_catalog_ready", False))
    inventory_exhausted = bool(
        existing_cache_inventory.get("existing_cache_inventory_ready", False)
        and not existing_cache_inventory.get("existing_cache_source_candidates_found", False)
    )
    v32_v31_used = {str(item) for item in near_miss_catalog.get("excluded_scenario_ids", []) or [] if str(item or "")}
    v30_swept = _v30_swept_scenarios()
    backlog = _incomplete_backlog()
    excluded = sorted(v32_v31_used | v30_swept | backlog)
    source_hint_scenarios = list(near_miss_catalog.get("source_scenarios", []) or [])
    return {
        "schema_version": "controlled_canary_shadow_cache_acquisition_target_spec_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only scenario/cache acquisition design",
            "reopens_rejected_preset": False,
        },
        "target_spec_ready": bool(catalog_ready and inventory_exhausted),
        "shadow_only": True,
        "cache_fill_authorized": False,
        "online_llm_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "strict_gate": {
            "candidate": "shadow_hot_dry_humidity_retention",
            "min_margin": 0.20,
            "rh_max": 45.0,
            "vpd_min": 2.25,
            "temp_max": 30.5,
            "canopy_dew_margin_min": 3.0,
            "unsafe_preferred_allowed": False,
            "unsafe_conflict_allowed": False,
            "safety_gate_block_allowed": False,
        },
        "source_hint_only": True,
        "near_miss_not_strict_applied_evidence": True,
        "v33_near_miss_scenario_count": int(near_miss_catalog.get("near_miss_scenario_count", 0) or 0),
        "v33_near_miss_row_count": int(near_miss_catalog.get("near_miss_row_count", 0) or 0),
        "source_hint_scenarios": source_hint_scenarios,
        "excluded_scenario_ids": excluded,
        "excluded_groups": {
            "v32_v31_used_or_wave1": sorted(v32_v31_used),
            "v30_swept_subset": sorted(v30_swept),
            "incomplete_cache_backlog": sorted(backlog),
        },
        "inventory_candidate_scenario_count": int(existing_cache_inventory.get("candidate_scenario_count", 0) or 0),
        "next_action": "run_v34_read_only_trace_source_inventory",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Shadow Cache Acquisition Target Spec v34",
        "",
        f"- Target spec ready: {report.get('target_spec_ready', False)}",
        "- Shadow only: true",
        "- Cache fill authorized: false",
        "- Online LLM allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Near-miss rows: {report.get('v33_near_miss_row_count', 0)}",
        f"- Excluded scenarios: {len(report.get('excluded_scenario_ids', []) or [])}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Strict Gate",
        "",
        "| gate | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("strict_gate", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Source Hints", "", "| scenario | rows | reasons |", "| --- | ---: | --- |"])
    for item in report.get("source_hint_scenarios", []) or []:
        if isinstance(item, Mapping):
            reasons = ",".join(dict(item.get("reason_counts", {}) or {}).keys())
            lines.append(f"| {item.get('scenario_id', '')} | {item.get('near_miss_rows', 0)} | {reasons} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near-miss-catalog-json", required=True)
    parser.add_argument("--existing-cache-inventory-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        near_miss_catalog=_load(args.near_miss_catalog_json),
        existing_cache_inventory=_load(args.existing_cache_inventory_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"target_spec_ready={report['target_spec_ready']}")
    print(f"excluded_scenario_count={len(report['excluded_scenario_ids'])}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["target_spec_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
