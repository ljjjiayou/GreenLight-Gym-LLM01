"""Build the v30 coverage-qualified strict-targeted shadow-only sweep manifest."""

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

MAX_STEPS = 240
CONTROLLER = "llm_rspc_v2"
GROUP_A_YEARS = [2010, 2015]
GROUP_A_DAYS = [59, 120, 180, 240]
GROUP_A_SEEDS = [42, 43, 44]
GROUP_B_YEARS = [2020]
GROUP_B_DAYS = [59, 120, 180, 240]
GROUP_B_SEEDS = [44]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _scenario_id(year: int, day: int, seed: int) -> str:
    return f"y{year}_d{day}_s{seed}_n{MAX_STEPS}"


def _env_id(year: int, day: int, seed: int) -> str:
    return f"TomatoEnv_y{year}_d{day}_s{seed}"


def _scenario_record(year: int, day: int, seed: int, *, role: str, coverage: Mapping[str, Any] | None) -> dict[str, Any]:
    coverage = coverage or {}
    return {
        "scenario_id": _scenario_id(year, day, seed),
        "env_id": _env_id(year, day, seed),
        "year": int(year),
        "day": int(day),
        "seed": int(seed),
        "max_steps": MAX_STEPS,
        "role": role,
        "cache_entry_count": int(coverage.get("entry_count", 0) or 0),
        "cache_coverage_ok": bool(coverage.get("ok", False)),
        "cache_coverage_issues": list(coverage.get("issues", []) or []),
    }


def _expected_subset() -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for year in GROUP_A_YEARS:
        for day in GROUP_A_DAYS:
            for seed in GROUP_A_SEEDS:
                triples.append((year, day, seed))
    for year in GROUP_B_YEARS:
        for day in GROUP_B_DAYS:
            for seed in GROUP_B_SEEDS:
                triples.append((year, day, seed))
    return triples


def _parse_env_id(env_id: str) -> tuple[int, int, int] | None:
    match = re.match(r"^TomatoEnv_y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)$", str(env_id))
    if not match:
        return None
    return int(match.group("year")), int(match.group("day")), int(match.group("seed"))


def _coverage_envs(v29_cache_coverage: Mapping[str, Any]) -> Mapping[str, Any]:
    envs = v29_cache_coverage.get("envs", {})
    return envs if isinstance(envs, Mapping) else {}


def build_report(*, v29_cache_coverage: Mapping[str, Any]) -> dict[str, Any]:
    envs = _coverage_envs(v29_cache_coverage)
    selected_scenarios: list[dict[str, Any]] = []
    subset_missing_or_failed: list[str] = []
    for year, day, seed in _expected_subset():
        env = _env_id(year, day, seed)
        coverage = envs.get(env, {})
        item = _scenario_record(
            year,
            day,
            seed,
            role="coverage_qualified_strict_targeted_shadow_source_candidate",
            coverage=coverage if isinstance(coverage, Mapping) else {},
        )
        selected_scenarios.append(item)
        if not item["cache_coverage_ok"]:
            subset_missing_or_failed.append(env)

    selected_ids = {item["env_id"] for item in selected_scenarios}
    blocked_backlog: list[dict[str, Any]] = []
    for env_id, coverage in sorted(envs.items()):
        if env_id in selected_ids or not isinstance(coverage, Mapping) or bool(coverage.get("ok", False)):
            continue
        parts = _parse_env_id(str(env_id))
        if parts is None:
            continue
        year, day, seed = parts
        blocked_backlog.append(
            _scenario_record(
                year,
                day,
                seed,
                role="blocked_cache_repair_backlog",
                coverage=coverage,
            )
        )

    ready = len(selected_scenarios) == 28 and not subset_missing_or_failed
    command_groups = [
        {
            "group_id": "v30_group_a_2010_2015_all_seeds",
            "years": GROUP_A_YEARS,
            "days": GROUP_A_DAYS,
            "seeds": GROUP_A_SEEDS,
            "expected_scenario_ids": [
                _scenario_id(year, day, seed)
                for year in GROUP_A_YEARS
                for day in GROUP_A_DAYS
                for seed in GROUP_A_SEEDS
            ],
            "cartesian_product_safe": ready,
        },
        {
            "group_id": "v30_group_b_2020_seed44",
            "years": GROUP_B_YEARS,
            "days": GROUP_B_DAYS,
            "seeds": GROUP_B_SEEDS,
            "expected_scenario_ids": [
                _scenario_id(year, day, seed)
                for year in GROUP_B_YEARS
                for day in GROUP_B_DAYS
                for seed in GROUP_B_SEEDS
            ],
            "cartesian_product_safe": ready,
        },
    ]
    return {
        "schema_version": "strict_targeted_shadow_sweep_v30_subset_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "coverage-qualified strict-targeted shadow-only subset manifest",
            "reopens_rejected_preset": False,
        },
        "strict_targeted_shadow_sweep_v30_manifest_ready": ready,
        "scope": "coverage_qualified_strict_targeted_shadow_only_sweep_v30" if ready else "blocked",
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
        "selected_cache_path": str(v29_cache_coverage.get("plan_cache_path", "")),
        "source_v29_cache_coverage_pass": bool(v29_cache_coverage.get("cache_coverage_pass", False)),
        "source_v29_missing_key_count": int(v29_cache_coverage.get("missing_key_count", 0) or 0),
        "scenario_count": len(selected_scenarios),
        "expected_scenario_count": 28,
        "selected_scenarios": selected_scenarios,
        "scenarios": selected_scenarios,
        "scenario_ids": [item["scenario_id"] for item in selected_scenarios],
        "subset_missing_or_failed_envs": subset_missing_or_failed,
        "blocked_cache_repair_backlog": blocked_backlog,
        "blocked_cache_repair_backlog_count": len(blocked_backlog),
        "command_groups": command_groups,
        "next_action": "run_v30_no_fill_cache_coverage_precheck" if ready else "v30_subset_cache_coverage_not_qualified",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strict-Targeted Shadow Sweep v30 Subset Manifest",
        "",
        f"- Ready: {report.get('strict_targeted_shadow_sweep_v30_manifest_ready', False)}",
        "- Shadow only: true",
        "- Controller: `llm_rspc_v2`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Scenario count: {report.get('scenario_count', 0)}",
        f"- Blocked cache repair backlog: {report.get('blocked_cache_repair_backlog_count', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Selected Scenarios",
        "",
        "| scenario | env | cache entries |",
        "| --- | --- | ---: |",
    ]
    for item in report.get("selected_scenarios", []) or []:
        if isinstance(item, Mapping):
            lines.append(f"| {item.get('scenario_id', '')} | {item.get('env_id', '')} | {item.get('cache_entry_count', 0)} |")
    lines.extend(["", "## Blocked Cache Repair Backlog", "", "| scenario | env | issues |", "| --- | --- | --- |"])
    for item in report.get("blocked_cache_repair_backlog", []) or []:
        if isinstance(item, Mapping):
            issues = ",".join(str(value) for value in item.get("cache_coverage_issues", []) or [])
            lines.append(f"| {item.get('scenario_id', '')} | {item.get('env_id', '')} | {issues} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v29-cache-coverage-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(v29_cache_coverage=_load(args.v29_cache_coverage_json))
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"strict_targeted_shadow_sweep_v30_manifest_ready={report['strict_targeted_shadow_sweep_v30_manifest_ready']}")
    print(f"scenario_count={report['scenario_count']}")
    print(f"blocked_cache_repair_backlog_count={report['blocked_cache_repair_backlog_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["strict_targeted_shadow_sweep_v30_manifest_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
