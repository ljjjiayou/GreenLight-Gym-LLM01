"""Build the v33 strict-targeted shadow-only sweep manifest from cache inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONTROLLER = "llm_rspc_v2"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _scenario_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": str(raw.get("scenario_id", "")),
        "env_id": str(raw.get("env_id", "")),
        "year": int(raw.get("year", 0) or 0),
        "day": int(raw.get("day", 0) or 0),
        "seed": int(raw.get("seed", 0) or 0),
        "max_steps": int(raw.get("max_steps", 240) or 240),
        "cache_path": str(raw.get("cache_path", "")),
        "cache_entry_count": int(raw.get("entry_count", 0) or 0),
        "payload_complete": bool(raw.get("payload_complete", False)),
    }


def build_report(*, inventory: Mapping[str, Any], max_scenarios: int = 0) -> dict[str, Any]:
    candidates = [_scenario_record(item) for item in inventory.get("candidate_scenarios", []) or [] if isinstance(item, Mapping)]
    if max_scenarios and max_scenarios > 0:
        candidates = candidates[:max_scenarios]
    selected = [item for item in candidates if item["scenario_id"] and item["cache_path"]]
    command_groups = [
        {
            "group_id": f"v33_{item['scenario_id']}",
            "cache_path": item["cache_path"],
            "years": [item["year"]],
            "days": [item["day"]],
            "seeds": [item["seed"]],
            "scenario_ids": [item["scenario_id"]],
            "selected_scenarios": [item],
            "cartesian_product_safe": True,
        }
        for item in selected
    ]
    ready = bool(inventory.get("existing_cache_inventory_ready", False) and selected)
    return {
        "schema_version": "strict_targeted_shadow_sweep_v33_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "existing-cache strict-targeted shadow-only manifest",
            "reopens_rejected_preset": False,
        },
        "strict_targeted_shadow_sweep_v33_manifest_ready": ready,
        "shadow_only": True,
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "controllers": [CONTROLLER],
        "blocked_controllers": ["llm_rspc_v2_hot_dry_proposer_strict"],
        "scenario_count": len(selected),
        "selected_scenarios": selected,
        "scenarios": selected,
        "scenario_ids": [item["scenario_id"] for item in selected],
        "cache_group_count": len({item["cache_path"] for item in selected}),
        "command_groups": command_groups,
        "next_action": "run_v33_no_fill_cache_coverage_precheck" if ready else "existing_cache_exhausted_for_independent_strict_source",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strict-Targeted Shadow Sweep v33 Manifest",
        "",
        f"- Ready: {report.get('strict_targeted_shadow_sweep_v33_manifest_ready', False)}",
        "- Shadow only: true",
        "- Controller: `llm_rspc_v2`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Scenario count: {report.get('scenario_count', 0)}",
        f"- Cache groups: {report.get('cache_group_count', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Selected Scenarios",
        "",
        "| scenario | cache | entries |",
        "| --- | --- | ---: |",
    ]
    for item in report.get("selected_scenarios", []) or []:
        if isinstance(item, Mapping):
            lines.append(f"| {item.get('scenario_id', '')} | {Path(str(item.get('cache_path', ''))).name} | {item.get('cache_entry_count', 0)} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-json", required=True)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(inventory=_load(args.inventory_json), max_scenarios=args.max_scenarios)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"strict_targeted_shadow_sweep_v33_manifest_ready={report['strict_targeted_shadow_sweep_v33_manifest_ready']}")
    print(f"scenario_count={report['scenario_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["strict_targeted_shadow_sweep_v33_manifest_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
