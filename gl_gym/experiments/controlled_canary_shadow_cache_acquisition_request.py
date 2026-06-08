"""Build a non-authorizing shadow cache acquisition request from v34 source inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def build_report(*, source_inventory: Mapping[str, Any], requested_cache_path: str) -> dict[str, Any]:
    candidates = [item for item in source_inventory.get("candidate_scenarios", []) or [] if isinstance(item, Mapping)]
    request_ready = bool(source_inventory.get("new_shadow_source_inventory_ready", False) and candidates)
    requested_scenarios: list[dict[str, Any]] = []
    for item in candidates:
        requested_scenarios.append(
            {
                "scenario_id": str(item.get("scenario_id", "")),
                "year": int(item.get("year", 0) or 0),
                "day": int(item.get("day", 0) or 0),
                "seed": int(item.get("seed", 0) or 0),
                "max_steps": int(item.get("max_steps", 240) or 240),
                "expected_strict_eligible_applied_steps": int(item.get("expected_strict_eligible_applied_steps", 0) or 0),
                "rationale": "existing_trace_strict_targeted_shadow_source_without_current_cache_coverage",
                "source_rows": list(item.get("source_rows", []) or []),
            }
        )
    command_groups = [
        {
            "group_id": f"cache_acquisition_{item['scenario_id']}",
            "years": [item["year"]],
            "days": [item["day"]],
            "seeds": [item["seed"]],
            "max_steps": item["max_steps"],
            "scenario_ids": [item["scenario_id"]],
            "cartesian_product_safe": True,
        }
        for item in requested_scenarios
    ]
    return {
        "schema_version": "controlled_canary_shadow_cache_acquisition_request_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow cache acquisition request only",
            "reopens_rejected_preset": False,
        },
        "shadow_cache_acquisition_request_ready": request_ready,
        "cache_fill_authorized": False,
        "online_llm_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "replay_execution_record_generated": False,
        "requested_cache_path": requested_cache_path,
        "requested_scenario_count": len(requested_scenarios),
        "requested_scenarios": requested_scenarios,
        "command_groups_for_future_authorized_cache_acquisition": command_groups,
        "next_action": (
            "await_explicit_shadow_cache_acquisition_authorization"
            if request_ready
            else "external_weather_or_scenario_space_expansion_design"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Shadow Cache Acquisition Request v34",
        "",
        f"- Request ready: {report.get('shadow_cache_acquisition_request_ready', False)}",
        "- Cache fill authorized: false",
        "- Online LLM allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Requested scenarios: {report.get('requested_scenario_count', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Requested Scenarios",
        "",
        "| scenario | expected strict steps | rationale |",
        "| --- | ---: | --- |",
    ]
    for item in report.get("requested_scenarios", []) or []:
        if isinstance(item, Mapping):
            lines.append(f"| {item.get('scenario_id', '')} | {item.get('expected_strict_eligible_applied_steps', 0)} | {item.get('rationale', '')} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-inventory-json", required=True)
    parser.add_argument("--requested-cache-path", default="gl_gym/result/plan_cache/holdout_seed42_43_hot_dry_qwen_20260517_merged.json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(source_inventory=_load(args.source_inventory_json), requested_cache_path=args.requested_cache_path)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"shadow_cache_acquisition_request_ready={report['shadow_cache_acquisition_request_ready']}")
    print(f"requested_scenario_count={report['requested_scenario_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
