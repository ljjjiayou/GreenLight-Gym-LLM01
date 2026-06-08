"""Plan-only cache coverage checklist for future strict metadata replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_report(
    *,
    plan_cache_path: str,
    plan_cache_key_policy: str,
    max_steps: int,
) -> dict[str, Any]:
    scenario_families = [
        {
            "family": "canonical_failure",
            "scenario_ids": ["y2020_d120_s44_n240"],
            "purpose": "known canopy hard-crossing family and post-guardrail provenance target",
        },
        {
            "family": "neutral_no_risk",
            "scenario_ids": [],
            "purpose": "future false-positive blocker check; exact windows come from counterexample manifest",
        },
        {
            "family": "pure_hot_dry",
            "scenario_ids": [],
            "purpose": "future useful dry-relief interference check",
        },
        {
            "family": "mixed_dry_dew",
            "scenario_ids": [],
            "purpose": "future false-safe and dew/canopy boundary recall check",
        },
    ]
    return {
        "schema_version": "cache_coverage_plan_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "plan-only / no cache check",
            "reopens_rejected_preset": False,
        },
        "metadata_replay_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "cache_coverage_pass": False,
        "cache_check_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "plan_cache_path": plan_cache_path,
        "plan_cache_key_policy": plan_cache_key_policy,
        "max_steps": int(max_steps),
        "expected_planning_boundaries": "scenario_timestep strict replay boundaries plus any recorded emergency replans",
        "expected_entries": "all selected scenario/timestep planning keys needed for strict metadata replay",
        "missing_key_handling": "block metadata replay; do not call online LLM and do not fill cache in this stage",
        "scenario_families": scenario_families,
        "required_future_check": [
            "plan cache exists",
            "schema version accepted",
            "all scenario env ids covered",
            "all required planning timesteps covered",
            "missing_buffered_action_count = 0",
            "missing_parsed_plan_count = 0",
        ],
        "next_action": "cache_coverage_check_after_protocol_and_default_path_plans",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Cache Coverage Plan",
        "",
        "- Mode: plan-only / no cache check",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Cache coverage pass: {report.get('cache_coverage_pass', False)}",
        f"- Plan cache path: `{report.get('plan_cache_path', '')}`",
        f"- Key policy: `{report.get('plan_cache_key_policy', '')}`",
        "",
        "## Scenario Families",
        "",
        "| family | purpose |",
        "| --- | --- |",
    ]
    for row in report.get("scenario_families", []) or []:
        if isinstance(row, Mapping):
            lines.append(f"| {row.get('family', '')} | {row.get('purpose', '')} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-cache-path", default="gl_gym/result/plan_cache/llm_plan_cache.json")
    parser.add_argument("--plan-cache-key-policy", default="scenario_timestep")
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        plan_cache_path=args.plan_cache_path,
        plan_cache_key_policy=args.plan_cache_key_policy,
        max_steps=args.max_steps,
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
    print(f"cache_coverage_pass={report['cache_coverage_pass']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
