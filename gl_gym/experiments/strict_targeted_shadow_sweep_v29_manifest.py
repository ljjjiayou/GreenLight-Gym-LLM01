"""Build the cache-covered strict-targeted shadow-only sweep v29 manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

YEARS = [2010, 2015, 2020]
DAYS = [59, 120, 180, 240]
SEEDS = [42, 43, 44]
MAX_STEPS = 240
CONTROLLER = "llm_rspc_v2"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _expected_scenarios() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for year in YEARS:
        for day in DAYS:
            for seed in SEEDS:
                scenarios.append(
                    {
                        "scenario_id": f"y{year}_d{day}_s{seed}_n{MAX_STEPS}",
                        "env_id": f"TomatoEnv_y{year}_d{day}_s{seed}",
                        "year": year,
                        "day": day,
                        "seed": seed,
                        "max_steps": MAX_STEPS,
                        "role": "cache_covered_strict_targeted_shadow_source_candidate",
                    }
                )
    return scenarios


def _cache_env_counts(plan_cache: Mapping[str, Any]) -> dict[str, int]:
    entries = plan_cache.get("entries", {}) if isinstance(plan_cache, Mapping) else {}
    iterable = entries.values() if isinstance(entries, Mapping) else entries if isinstance(entries, list) else []
    counts: dict[str, int] = {}
    for entry in iterable:
        if not isinstance(entry, Mapping):
            continue
        env_id = str(entry.get("env_id", "") or "")
        if re.match(r"^TomatoEnv_y\d+_d\d+_s\d+$", env_id):
            counts[env_id] = counts.get(env_id, 0) + 1
    return counts


def build_report(*, selected_cache_path: str) -> dict[str, Any]:
    cache = _load(selected_cache_path)
    counts = _cache_env_counts(cache)
    scenarios = []
    missing_envs: list[str] = []
    for scenario in _expected_scenarios():
        count = counts.get(scenario["env_id"], 0)
        item = {**scenario, "cache_entry_count": count, "present_in_selected_cache": count > 0}
        scenarios.append(item)
        if count <= 0:
            missing_envs.append(scenario["env_id"])
    ready = not missing_envs
    command_groups = [
        {
            "group_id": "cache_covered_36_scenarios",
            "years": YEARS,
            "days": DAYS,
            "seeds": SEEDS,
            "expected_scenario_ids": [item["scenario_id"] for item in scenarios],
            "cartesian_product_safe": ready and len(scenarios) == len(YEARS) * len(DAYS) * len(SEEDS),
        }
    ]
    return {
        "schema_version": "strict_targeted_shadow_sweep_v29_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "cache-covered strict-targeted shadow-only sweep manifest",
            "reopens_rejected_preset": False,
        },
        "strict_targeted_shadow_sweep_manifest_ready": ready,
        "scope": "cache_covered_strict_targeted_shadow_only_sweep_v29" if ready else "blocked",
        "shadow_only": True,
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "controller": CONTROLLER,
        "controllers": [CONTROLLER],
        "selected_cache_path": selected_cache_path,
        "scenario_count": len(scenarios),
        "expected_scenario_count": len(YEARS) * len(DAYS) * len(SEEDS),
        "years": YEARS,
        "days": DAYS,
        "seeds": SEEDS,
        "selected_scenarios": scenarios,
        "scenarios": scenarios,
        "scenario_ids": [item["scenario_id"] for item in scenarios],
        "missing_envs": missing_envs,
        "command_groups": command_groups,
        "next_action": "run_v29_no_fill_cache_coverage_precheck" if ready else "selected_cache_missing_expected_v29_scenarios",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strict-Targeted Shadow Sweep v29 Manifest",
        "",
        f"- Ready: {report.get('strict_targeted_shadow_sweep_manifest_ready', False)}",
        "- Shadow only: true",
        "- Controller: `llm_rspc_v2`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Scenario count: {report.get('scenario_count', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Scenarios",
        "",
        "| scenario | env | cache entries |",
        "| --- | --- | ---: |",
    ]
    for item in report.get("selected_scenarios", []) or []:
        if isinstance(item, Mapping):
            lines.append(f"| {item.get('scenario_id', '')} | {item.get('env_id', '')} | {item.get('cache_entry_count', 0)} |")
    missing = report.get("missing_envs", []) or []
    if missing:
        lines.extend(["", "## Missing Envs", ""])
        lines.extend(f"- `{item}`" for item in missing)
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-cache-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(selected_cache_path=args.selected_cache_path)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"strict_targeted_shadow_sweep_manifest_ready={report['strict_targeted_shadow_sweep_manifest_ready']}")
    print(f"scenario_count={report['scenario_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["strict_targeted_shadow_sweep_manifest_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
