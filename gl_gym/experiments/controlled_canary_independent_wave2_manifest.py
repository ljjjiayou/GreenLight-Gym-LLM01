"""Build a Wave-2 independent triggered holdout manifest."""

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


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def build_report(*, discovery: Mapping[str, Any], selected_cache_path: str, output_root: str) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    seen: set[str] = set()
    excluded = {str(item) for item in discovery.get("excluded_scenario_ids", []) or []}
    for item in discovery.get("qualified_trigger_scenarios", []) or []:
        if not isinstance(item, Mapping):
            continue
        scenario_id = str(item.get("scenario_id", "") or "")
        if not scenario_id or scenario_id in seen or scenario_id in excluded:
            continue
        seen.add(scenario_id)
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "year": int(item.get("year", 0) or 0),
                "day": int(item.get("day", 0) or 0),
                "seed": int(item.get("seed", 0) or 0),
                "max_steps": int(item.get("max_steps", 240) or 240),
                "role": "independent_strict_trigger_holdout",
                "expected_strict_eligible_applied_steps": int(
                    item.get("expected_strict_eligible_applied_steps", 0) or 0
                ),
            }
        )
    command_groups = [
        {
            "group_id": scenario["scenario_id"],
            "years": [scenario["year"]],
            "days": [scenario["day"]],
            "seeds": [scenario["seed"]],
            "expected_scenario_ids": [scenario["scenario_id"]],
            "cartesian_product_safe": True,
        }
        for scenario in scenarios
    ]
    ready = bool(discovery.get("independent_trigger_candidates_found", False) and scenarios)
    return {
        "schema_version": "controlled_canary_independent_wave2_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC"],
            "default_llm_rspc_v2_changed": False,
            "mode": "independent triggered controlled canary wave2 manifest",
            "reopens_rejected_preset": False,
        },
        "wave2_manifest_ready": ready,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "scope": "independent_triggered_holdout_wave2_only" if ready else "blocked",
        "baseline_controller": BASELINE_CONTROLLER,
        "candidate_controller": CANDIDATE_CONTROLLER,
        "controllers": [BASELINE_CONTROLLER, CANDIDATE_CONTROLLER],
        "selected_cache_path": selected_cache_path,
        "output_root": output_root,
        "excluded_scenario_ids": sorted(excluded),
        "selected_scenarios": scenarios,
        "scenario_ids": sorted(seen),
        "command_groups": command_groups,
        "next_action": "run_wave2_cache_coverage_check" if ready else "independent_trigger_discovery_required",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Independent Triggered Holdout Wave-2 Manifest",
        "",
        f"- Ready: {report.get('wave2_manifest_ready', False)}",
        f"- Scope: `{report.get('scope', '')}`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Scenarios",
        "",
        "| scenario | expected strict steps |",
        "| --- | ---: |",
    ]
    for scenario in report.get("selected_scenarios", []) or []:
        if isinstance(scenario, Mapping):
            lines.append(
                f"| {scenario.get('scenario_id', '')} | {scenario.get('expected_strict_eligible_applied_steps', 0)} |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-json", required=True)
    parser.add_argument("--selected-cache-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        discovery=_load(args.discovery_json),
        selected_cache_path=args.selected_cache_path,
        output_root=args.output_root,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"wave2_manifest_ready={report['wave2_manifest_ready']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["wave2_manifest_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
