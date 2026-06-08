"""Inventory existing local plan caches for unused strict-replay shadow sources."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAX_STEPS = 240
CONTROL_INTERVAL = 12
MIN_REQUIRED_LAST_TIMESTEP = MAX_STEPS - CONTROL_INTERVAL
MIN_REQUIRED_ENTRIES = (MAX_STEPS + CONTROL_INTERVAL - 1) // CONTROL_INTERVAL
SELECTED_CACHE = "gl_gym/result/plan_cache/holdout_seed42_43_hot_dry_qwen_20260517_merged.json"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _parse_env_id(env_id: str) -> tuple[int, int, int] | None:
    match = re.match(r"^TomatoEnv_y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)$", str(env_id))
    if not match:
        return None
    return int(match.group("year")), int(match.group("day")), int(match.group("seed"))


def _scenario_id(year: int, day: int, seed: int) -> str:
    return f"y{year}_d{day}_s{seed}_n{MAX_STEPS}"


def _env_id_from_scenario(scenario_id: str) -> str | None:
    match = re.match(r"^y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)_n\d+$", str(scenario_id))
    if not match:
        return None
    return f"TomatoEnv_y{match.group('year')}_d{match.group('day')}_s{match.group('seed')}"


def _default_v30_swept_scenarios() -> set[str]:
    scenarios: set[str] = set()
    for year in [2010, 2015]:
        for day in [59, 120, 180, 240]:
            for seed in [42, 43, 44]:
                scenarios.add(_scenario_id(year, day, seed))
    for day in [59, 120, 180, 240]:
        scenarios.add(_scenario_id(2020, day, 44))
    return scenarios


def _default_incomplete_backlog() -> set[str]:
    return {_scenario_id(2020, day, seed) for day in [59, 120, 180, 240] for seed in [42, 43]}


def _load_entries(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = payload.get("entries", {}) if isinstance(payload, Mapping) else {}
    if isinstance(entries, Mapping):
        return [entry for entry in entries.values() if isinstance(entry, Mapping)]
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, Mapping)]
    return []


def _cache_env_reports(cache_path: Path, *, require_payloads: bool) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in _load_entries(cache_path):
        env_id = str(entry.get("env_id", "") or "")
        if _parse_env_id(env_id):
            grouped[env_id].append(entry)

    reports: dict[str, dict[str, Any]] = {}
    for env_id, entries in grouped.items():
        parts = _parse_env_id(env_id)
        if parts is None:
            continue
        year, day, seed = parts
        timesteps = sorted({int(round(float(entry.get("timestep", -1)))) for entry in entries if entry.get("timestep", -1) is not None and float(entry.get("timestep", -1)) >= 0})
        issues: list[str] = []
        if 0 not in timesteps:
            issues.append("missing_timestep_zero")
        if not timesteps:
            issues.append("missing_env")
        if timesteps and max(timesteps) < MIN_REQUIRED_LAST_TIMESTEP:
            issues.append("last_timestep_too_early")
        if len(timesteps) < MIN_REQUIRED_ENTRIES:
            issues.append("entry_count_below_expected")
        missing_buffered_action_count = sum(1 for entry in entries if not isinstance(entry.get("buffered_action"), Mapping))
        missing_parsed_plan_count = sum(1 for entry in entries if not isinstance(entry.get("parsed_plan"), Mapping))
        if require_payloads and missing_buffered_action_count:
            issues.append("missing_buffered_action_payload")
        if require_payloads and missing_parsed_plan_count:
            issues.append("missing_parsed_plan_payload")
        reports[env_id] = {
            "scenario_id": _scenario_id(year, day, seed),
            "env_id": env_id,
            "year": year,
            "day": day,
            "seed": seed,
            "max_steps": MAX_STEPS,
            "cache_path": str(cache_path),
            "entry_count": len(timesteps),
            "raw_entry_count": len(entries),
            "has_timestep_zero": 0 in timesteps,
            "min_timestep": min(timesteps) if timesteps else None,
            "max_timestep": max(timesteps) if timesteps else None,
            "missing_buffered_action_count": missing_buffered_action_count,
            "missing_parsed_plan_count": missing_parsed_plan_count,
            "payload_complete": missing_buffered_action_count == 0 and missing_parsed_plan_count == 0,
            "strict_replay_coverage_ok": not issues,
            "issues": sorted(set(issues)),
        }
    return reports


def _candidate_sort_key(item: Mapping[str, Any]) -> tuple[int, int, str]:
    path = str(item.get("cache_path", ""))
    selected = 0 if Path(path).name == Path(SELECTED_CACHE).name else 1
    merged = 0 if "merged" in Path(path).name else 1
    return (selected, merged, str(item.get("scenario_id", "")))


def build_report(
    *,
    near_miss_catalog: Mapping[str, Any],
    cache_dir: str | Path,
    require_payloads: bool = True,
    additional_excluded_scenario_ids: Sequence[str] = (),
) -> dict[str, Any]:
    cache_root = _resolve(cache_dir)
    catalog_excluded = {str(item) for item in near_miss_catalog.get("excluded_scenario_ids", []) or []}
    excluded = set(catalog_excluded)
    excluded.update(_default_v30_swept_scenarios())
    excluded.update(additional_excluded_scenario_ids)
    incomplete_backlog = _default_incomplete_backlog()
    excluded.update(incomplete_backlog)

    cache_groups: list[dict[str, Any]] = []
    raw_candidates: list[dict[str, Any]] = []
    cache_paths = sorted(cache_root.glob("*.json")) if cache_root.exists() else []
    for cache_path in cache_paths:
        reports = _cache_env_reports(cache_path, require_payloads=require_payloads)
        candidates: list[dict[str, Any]] = []
        blocked: list[dict[str, Any]] = []
        for report in reports.values():
            scenario_id = str(report["scenario_id"])
            status = dict(report)
            status["excluded"] = scenario_id in excluded
            status["excluded_reason"] = (
                "incomplete_cache_repair_backlog"
                if scenario_id in incomplete_backlog
                else "used_or_already_swept"
                if scenario_id in excluded
                else ""
            )
            if report["strict_replay_coverage_ok"] and report["payload_complete"] and scenario_id not in excluded:
                candidates.append(status)
                raw_candidates.append(status)
            else:
                blocked.append(status)
        cache_groups.append(
            {
                "cache_path": str(cache_path),
                "env_count": len(reports),
                "candidate_count": len(candidates),
                "blocked_count": len(blocked),
                "candidates": sorted(candidates, key=lambda item: str(item.get("scenario_id", ""))),
                "blocked_sample": sorted(blocked, key=lambda item: str(item.get("scenario_id", "")))[:20],
            }
        )

    deduped_by_scenario: dict[str, dict[str, Any]] = {}
    for candidate in sorted(raw_candidates, key=_candidate_sort_key):
        scenario_id = str(candidate.get("scenario_id", ""))
        deduped_by_scenario.setdefault(scenario_id, candidate)
    deduped_candidates = [deduped_by_scenario[key] for key in sorted(deduped_by_scenario)]
    return {
        "schema_version": "controlled_canary_existing_cache_source_inventory_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only existing-cache source inventory",
            "reopens_rejected_preset": False,
        },
        "existing_cache_inventory_ready": bool(near_miss_catalog.get("near_miss_catalog_ready", False)),
        "read_only_cache_inventory": True,
        "cache_fill_run": False,
        "online_llm_called": False,
        "require_payloads": bool(require_payloads),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_dir": str(cache_root),
        "scanned_cache_count": len(cache_paths),
        "cache_groups": cache_groups,
        "excluded_scenario_ids": sorted(excluded),
        "excluded_scenario_count": len(excluded),
        "incomplete_cache_repair_backlog": sorted(incomplete_backlog),
        "v30_already_swept_scenario_count": len(_default_v30_swept_scenarios()),
        "candidate_scenarios": deduped_candidates,
        "candidate_scenario_ids": [item["scenario_id"] for item in deduped_candidates],
        "candidate_scenario_count": len(deduped_candidates),
        "existing_cache_source_candidates_found": bool(deduped_candidates),
        "next_action": (
            "strict_targeted_shadow_sweep_v33_manifest_and_cache_precheck"
            if deduped_candidates
            else "new_scenario_cache_acquisition_design"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Controlled Canary Existing Cache Source Inventory v33",
        "",
        f"- Inventory ready: {report.get('existing_cache_inventory_ready', False)}",
        "- Read-only cache inventory: true",
        "- Cache fill run: false",
        "- Online LLM called: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Scanned caches: {report.get('scanned_cache_count', 0)}",
        f"- Candidate scenarios: {report.get('candidate_scenario_count', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Candidate Scenarios",
        "",
        "| scenario | cache | entries |",
        "| --- | --- | ---: |",
    ]
    for item in report.get("candidate_scenarios", []) or []:
        if isinstance(item, Mapping):
            lines.append(f"| {item.get('scenario_id', '')} | {Path(str(item.get('cache_path', ''))).name} | {item.get('entry_count', 0)} |")
    lines.extend(["", "## Cache Groups", "", "| cache | envs | candidates | blocked |", "| --- | ---: | ---: | ---: |"])
    for group in report.get("cache_groups", []) or []:
        if isinstance(group, Mapping):
            lines.append(f"| {Path(str(group.get('cache_path', ''))).name} | {group.get('env_count', 0)} | {group.get('candidate_count', 0)} | {group.get('blocked_count', 0)} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near-miss-catalog-json", required=True)
    parser.add_argument("--cache-dir", default="gl_gym/result/plan_cache")
    parser.add_argument("--exclude-scenario-id", action="append", default=[])
    parser.add_argument("--require-payloads", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        near_miss_catalog=_load(args.near_miss_catalog_json),
        cache_dir=args.cache_dir,
        require_payloads=bool(args.require_payloads),
        additional_excluded_scenario_ids=args.exclude_scenario_id,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"existing_cache_inventory_ready={report['existing_cache_inventory_ready']}")
    print(f"candidate_scenario_count={report['candidate_scenario_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
