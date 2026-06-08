"""Build the minimal controlled canary scenario manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASELINE_CONTROLLER = "llm_rspc_v2"
CANDIDATE_CONTROLLER = "llm_rspc_v2_hot_dry_proposer_strict"
SCENARIOS = (
    {"scenario_id": "y2020_d120_s44_n240", "year": 2020, "day": 120, "seed": 44, "max_steps": 240, "role": "canonical_failure"},
    {"scenario_id": "y2015_d180_s44_n240", "year": 2015, "day": 180, "seed": 44, "max_steps": 240, "role": "pure_hot_dry_mixed_dry_dew"},
    {"scenario_id": "y2020_d180_s44_n240", "year": 2020, "day": 180, "seed": 44, "max_steps": 240, "role": "pure_hot_dry_mixed_dry_dew"},
)
COMMAND_GROUPS = (
    {"group_id": "canonical_y2020_d120", "years": [2020], "days": [120], "seeds": [44]},
    {"group_id": "day180_y2015_y2020", "years": [2015, 2020], "days": [180], "seeds": [44]},
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _expected_ids(group: Mapping[str, Any]) -> set[str]:
    return {
        f"y{year}_d{day}_s{seed}_n240"
        for year in group.get("years", []) or []
        for day in group.get("days", []) or []
        for seed in group.get("seeds", []) or []
    }


def build_report(*, selected_cache_path: str, output_root: str, baseline_trace_dir: str = "") -> dict[str, Any]:
    scenario_ids = {item["scenario_id"] for item in SCENARIOS}
    groups: list[dict[str, Any]] = []
    for group in COMMAND_GROUPS:
        expected = _expected_ids(group)
        groups.append(
            {
                **group,
                "expected_scenario_ids": sorted(expected),
                "cartesian_product_safe": expected.issubset(scenario_ids),
            }
        )
    manifest_ready = all(group["cartesian_product_safe"] for group in groups)
    return {
        "schema_version": "controlled_replay_minimal_canary_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC"],
            "default_llm_rspc_v2_changed": False,
            "mode": "minimal controlled canary manifest",
            "reopens_rejected_preset": False,
        },
        "minimal_canary_manifest_ready": manifest_ready,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "scope": "minimal_controlled_canary_only",
        "baseline_controller": BASELINE_CONTROLLER,
        "candidate_controller": CANDIDATE_CONTROLLER,
        "controllers": [BASELINE_CONTROLLER, CANDIDATE_CONTROLLER],
        "selected_cache_path": selected_cache_path,
        "baseline_trace_dir": baseline_trace_dir,
        "output_root": output_root,
        "scenarios": list(SCENARIOS),
        "selected_scenarios": list(SCENARIOS),
        "scenario_ids": sorted(scenario_ids),
        "command_groups": groups,
        "next_action": "run_minimal_canary_cache_coverage_check" if manifest_ready else "fix_minimal_canary_manifest",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Minimal Controlled Canary Manifest",
        "",
        "- Mode: minimal controlled canary manifest",
        "- Default `llm_rspc_v2` changed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Manifest ready: {report.get('minimal_canary_manifest_ready', False)}",
        f"- Candidate controller: `{report.get('candidate_controller', '')}`",
        "",
        "## Scenarios",
        "",
        "| scenario | role |",
        "| --- | --- |",
    ]
    for scenario in report.get("scenarios", []) or []:
        if isinstance(scenario, Mapping):
            lines.append(f"| {scenario.get('scenario_id', '')} | {scenario.get('role', '')} |")
    lines.extend(["", "## Command Groups", "", "| group | expected scenarios | safe |", "| --- | --- | --- |"])
    for group in report.get("command_groups", []) or []:
        if isinstance(group, Mapping):
            lines.append(
                f"| {group.get('group_id', '')} | {', '.join(group.get('expected_scenario_ids', []) or [])} | {group.get('cartesian_product_safe', False)} |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-cache-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-trace-dir", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        selected_cache_path=args.selected_cache_path,
        output_root=args.output_root,
        baseline_trace_dir=args.baseline_trace_dir,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"minimal_canary_manifest_ready={report['minimal_canary_manifest_ready']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["minimal_canary_manifest_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
