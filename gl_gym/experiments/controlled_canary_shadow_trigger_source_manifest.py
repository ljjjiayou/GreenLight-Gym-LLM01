"""Build a shadow-only trigger source manifest from near-miss sourcing."""

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


def build_report(*, sourcing: Mapping[str, Any], selected_cache_path: str) -> dict[str, Any]:
    scenarios = []
    seen: set[str] = set()
    for item in sourcing.get("source_scenarios", []) or []:
        if not isinstance(item, Mapping):
            continue
        scenario_id = str(item.get("scenario_id", "") or "")
        if not scenario_id or scenario_id in seen:
            continue
        seen.add(scenario_id)
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "year": int(item.get("year", 0) or 0),
                "day": int(item.get("day", 0) or 0),
                "seed": int(item.get("seed", 0) or 0),
                "max_steps": int(item.get("max_steps", 240) or 240),
                "role": "shadow_only_near_miss_trigger_source",
                "near_miss_rows": int(item.get("near_miss_rows", 0) or 0),
            }
        )
    ready = bool(sourcing.get("shadow_trigger_sources_found", False) and scenarios)
    return {
        "schema_version": "controlled_canary_shadow_trigger_source_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only trigger source manifest",
            "reopens_rejected_preset": False,
        },
        "shadow_trigger_source_manifest_ready": ready,
        "scope": "shadow_trigger_source_candidates_only" if ready else "blocked",
        "shadow_only": True,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "selected_cache_path": selected_cache_path,
        "selected_scenarios": scenarios,
        "scenario_ids": sorted(seen),
        "next_action": "run_shadow_trigger_source_cache_precheck" if ready else "shadow_trigger_source_sourcing_required",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Shadow Trigger Source Manifest",
        "",
        f"- Ready: {report.get('shadow_trigger_source_manifest_ready', False)}",
        f"- Scope: `{report.get('scope', '')}`",
        "- Shadow only: true",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Scenarios",
        "",
        "| scenario | near-miss rows |",
        "| --- | ---: |",
    ]
    for scenario in report.get("selected_scenarios", []) or []:
        if isinstance(scenario, Mapping):
            lines.append(f"| {scenario.get('scenario_id', '')} | {scenario.get('near_miss_rows', 0)} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sourcing-json", required=True)
    parser.add_argument("--selected-cache-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        sourcing=_load(args.sourcing_json),
        selected_cache_path=args.selected_cache_path,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"shadow_trigger_source_manifest_ready={report['shadow_trigger_source_manifest_ready']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["shadow_trigger_source_manifest_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
