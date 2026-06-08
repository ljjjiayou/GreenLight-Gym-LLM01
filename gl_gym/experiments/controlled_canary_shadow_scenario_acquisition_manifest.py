"""Build a non-authorizing v35 shadow scenario/cache acquisition manifest."""

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


def build_report(*, weather_inventory: Mapping[str, Any], requested_cache_path: str) -> dict[str, Any]:
    requested = [
        dict(item)
        for item in weather_inventory.get("candidate_scenarios_for_cache_acquisition", []) or []
        if isinstance(item, Mapping)
    ]
    manifest_ready = bool(weather_inventory.get("weather_window_inventory_ready", False) and requested)
    command_groups = [
        {
            "group_id": f"future_shadow_cache_acquisition_{item['scenario_id']}",
            "years": [int(item.get("year", 0))],
            "days": [int(item.get("day", 0))],
            "seeds": [int(item.get("seed", 0))],
            "max_steps": int(item.get("max_steps", 240)),
            "scenario_ids": [str(item.get("scenario_id", ""))],
            "controller": "llm_rspc_v2",
            "cartesian_product_safe": True,
            "execution_authorized": False,
        }
        for item in requested
    ]
    return {
        "schema_version": "controlled_canary_shadow_scenario_acquisition_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow scenario/cache acquisition manifest only",
            "reopens_rejected_preset": False,
        },
        "shadow_scenario_acquisition_manifest_ready": manifest_ready,
        "weather_proxy_source_hint_only": True,
        "weather_proxy_is_strict_applied_evidence": False,
        "cache_fill_authorized": False,
        "online_llm_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "replay_execution_record_generated": False,
        "requested_cache_path": requested_cache_path,
        "requested_scenario_count": len(requested),
        "requested_scenarios": requested,
        "command_groups_for_future_authorized_shadow_cache_acquisition": command_groups,
        "next_action": (
            "await_explicit_shadow_cache_acquisition_authorization_for_weather_expansion"
            if manifest_ready
            else "external_weather_dataset_or_relaxed_shadow_source_design_required"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Shadow Scenario Acquisition Manifest v35",
        "",
        f"- Manifest ready: {report.get('shadow_scenario_acquisition_manifest_ready', False)}",
        "- Cache fill authorized: false",
        "- Online LLM allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "- Weather proxy is strict-applied evidence: false",
        f"- Requested scenarios: {report.get('requested_scenario_count', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Requested Scenarios",
        "",
        "| scenario | weather source | rationale |",
        "| --- | --- | --- |",
    ]
    for item in report.get("requested_scenarios", []) or []:
        if isinstance(item, Mapping):
            source = f"{item.get('weather_site', '')}/{item.get('weather_label', '')}"
            lines.append(f"| {item.get('scenario_id', '')} | {source} | {item.get('expected_strict_source_rationale', '')} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-inventory-json", required=True)
    parser.add_argument("--requested-cache-path", default="gl_gym/result/plan_cache/holdout_seed42_43_hot_dry_qwen_20260517_merged.json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(weather_inventory=_load(args.weather_inventory_json), requested_cache_path=args.requested_cache_path)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"shadow_scenario_acquisition_manifest_ready={report['shadow_scenario_acquisition_manifest_ready']}")
    print(f"requested_scenario_count={report['requested_scenario_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
