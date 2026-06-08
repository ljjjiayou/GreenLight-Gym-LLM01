"""Build an explicit cache/scenario plan before metadata replay coverage checks.

This report is a preflight artifact only.  It reads cache metadata to avoid an
ambiguous default cache choice, but it does not run replay, fill cache, or call
an online LLM.
"""

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

DEFAULT_SCENARIOS = ["y2020_d120_s44_n240"]
DEFAULT_CACHE_CANDIDATES = [
    "gl_gym/result/plan_cache/holdout_seed42_43_hot_dry_qwen_20260517_merged.json",
    "gl_gym/result/plan_cache/frozen_36x240_qwen_mc_sero_shadow.json",
    "gl_gym/result/plan_cache/llm_plan_cache.json",
]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _parse_scenario_id(scenario_id: str, *, max_steps: int) -> dict[str, Any]:
    match = re.match(r"^y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)(?:_n(?P<steps>\d+))?$", scenario_id)
    if not match:
        raise ValueError(f"unsupported scenario_id format: {scenario_id}")
    steps = int(match.group("steps") or max_steps)
    return {
        "scenario_id": scenario_id,
        "scenario_family": "canonical_failure" if scenario_id == "y2020_d120_s44_n240" else "selected_metadata_replay",
        "year": int(match.group("year")),
        "day": int(match.group("day")),
        "seed": int(match.group("seed")),
        "max_steps": steps,
        "env_id": f"TomatoEnv_y{match.group('year')}_d{match.group('day')}_s{match.group('seed')}",
    }


def _dedupe_scenario_ids(scenario_ids: Sequence[str] | None) -> tuple[list[str], list[str], str]:
    raw_ids = list(scenario_ids or DEFAULT_SCENARIOS)
    seen: set[str] = set()
    unique: list[str] = []
    duplicates: list[str] = []
    for scenario_id in raw_ids:
        if scenario_id in seen:
            if scenario_id not in duplicates:
                duplicates.append(scenario_id)
            continue
        seen.add(scenario_id)
        unique.append(scenario_id)
    reason = (
        "duplicate scenario ids removed before cache coverage planning"
        if duplicates
        else "no duplicate scenario ids in requested scenario list"
    )
    return unique, duplicates, reason


def _filename_seed_tokens(path_text: str | Path) -> list[int]:
    seeds: set[int] = set()
    for match in re.finditer(r"seed(\d+(?:_\d+)*)", Path(path_text).name):
        for token in match.group(1).split("_"):
            if token:
                seeds.add(int(token))
    return sorted(seeds)


def _load_cache_env_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", {}) if isinstance(payload, Mapping) else {}
    if isinstance(entries, Mapping):
        values = entries.values()
    elif isinstance(entries, list):
        values = entries
    else:
        values = []
    return {str(entry.get("env_id", "")) for entry in values if isinstance(entry, Mapping) and entry.get("env_id")}


def build_report(
    *,
    selected_cache_path: str,
    scenario_ids: Sequence[str] | None = None,
    candidate_cache_paths: Sequence[str] | None = None,
    plan_cache_key_policy: str = "scenario_timestep",
    max_steps: int = 240,
    controller: str = "llm_rspc_v2",
    allow_default_cache: bool = False,
) -> dict[str, Any]:
    unique_scenario_ids, duplicate_scenario_ids, duplicate_reason = _dedupe_scenario_ids(scenario_ids)
    scenarios = [_parse_scenario_id(sid, max_steps=max_steps) for sid in unique_scenario_ids]
    candidate_paths = list(candidate_cache_paths or DEFAULT_CACHE_CANDIDATES)
    if selected_cache_path not in candidate_paths:
        candidate_paths.insert(0, selected_cache_path)
    selected_path = _resolve(selected_cache_path)
    selected_env_ids = _load_cache_env_ids(selected_path)
    default_cache_selected = Path(selected_cache_path).name == "llm_plan_cache.json"
    env_presence = {
        row["scenario_id"]: {
            "env_id": row["env_id"],
            "present_in_selected_cache": row["env_id"] in selected_env_ids,
        }
        for row in scenarios
    }
    all_envs_present = all(item["present_in_selected_cache"] for item in env_presence.values())
    selected_cache_accepted = bool(
        selected_path.exists() and all_envs_present and (allow_default_cache or not default_cache_selected)
    )
    required_seeds = sorted({int(row["seed"]) for row in scenarios})
    filename_seeds = _filename_seed_tokens(selected_cache_path)
    filename_seed_mismatch = bool(filename_seeds) and not set(required_seeds).issubset(filename_seeds)
    filename_seed_mismatch_explained = bool(not filename_seed_mismatch or all_envs_present)
    if filename_seed_mismatch and all_envs_present:
        cache_provenance = (
            "selected cache filename seed tokens do not list every requested seed, "
            "but cache contents include all requested env_id values"
        )
    elif filename_seed_mismatch:
        cache_provenance = "selected cache filename seed tokens and cache contents do not cover requested seeds"
    else:
        cache_provenance = "selected cache filename seed tokens are consistent with requested scenario seeds or absent"
    scenario_families = sorted({str(row["scenario_family"]) for row in scenarios})
    cache_scope = (
        "canonical_failure_only"
        if len(scenarios) == 1 and scenario_families == ["canonical_failure"]
        else "selected_metadata_replay_subset"
    )
    if not selected_path.exists():
        status = "blocked_selected_cache_missing"
    elif default_cache_selected and not allow_default_cache:
        status = "blocked_ambiguous_default_cache_requires_explicit_coverage_proof"
    elif not all_envs_present:
        status = "blocked_selected_cache_missing_required_envs"
    else:
        status = "selected_non_default_candidate_pending_coverage_check"
    return {
        "schema_version": "metadata_replay_cache_scenario_plan_v2",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / cache scenario plan",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "metadata_replay_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "cache_fill_policy": "no_fill",
        "online_llm_policy": "no_call",
        "cache_fill_run": False,
        "online_llm_called": False,
        "controller": controller,
        "plan_cache_key_policy": plan_cache_key_policy,
        "max_steps": int(max_steps),
        "input_scenario_count": len(list(scenario_ids or DEFAULT_SCENARIOS)),
        "unique_scenario_count": len(scenarios),
        "scenario_list_deduplicated": True,
        "duplicate_scenario_ids": duplicate_scenario_ids,
        "duplicate_scenario_reason": duplicate_reason,
        "scenario_list": scenarios,
        "cache_scope": cache_scope,
        "coverage_claim_scope": cache_scope,
        "candidate_cache_paths": candidate_paths,
        "selected_cache_path": selected_cache_path,
        "selected_cache_exists": selected_path.exists(),
        "selected_cache_is_default_llm_plan_cache": default_cache_selected,
        "selected_cache_filename_seed_tokens": filename_seeds,
        "required_scenario_seeds": required_seeds,
        "cache_filename_seed_mismatch": filename_seed_mismatch,
        "cache_filename_seed_mismatch_explained": filename_seed_mismatch_explained,
        "cache_provenance": cache_provenance,
        "selected_cache_env_presence": env_presence,
        "selected_cache_accepted_for_coverage_check": selected_cache_accepted,
        "cache_scenario_plan_ready": selected_cache_accepted,
        "expected_planning_boundaries": "scenario_timestep strict replay boundaries for selected scenarios plus recorded emergency replans",
        "buffered_action_expectation": "all replay-required entries contain buffered_action",
        "parsed_plan_expectation": "all replay-required entries contain parsed_plan",
        "missing_key_handling": "block metadata replay; do not fill cache and do not call online LLM",
        "next_action": "run_no_fill_cache_coverage_check" if selected_cache_accepted else "resolve_cache_path_or_scenario_list",
        "cache_path_selection_status": status,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Metadata Replay Cache Scenario Plan",
        "",
        "- Mode: audit-only / cache scenario plan",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Selected cache: `{report.get('selected_cache_path', '')}`",
        f"- Cache scenario plan ready: {report.get('cache_scenario_plan_ready', False)}",
        f"- Selection status: `{report.get('cache_path_selection_status', '')}`",
        f"- Coverage claim scope: `{report.get('coverage_claim_scope', '')}`",
        f"- Scenario list deduplicated: {report.get('scenario_list_deduplicated', False)}",
        f"- Cache filename seed mismatch explained: {report.get('cache_filename_seed_mismatch_explained', False)}",
        f"- Cache provenance: {report.get('cache_provenance', '')}",
        "",
        "## Scenarios",
        "",
        "| scenario | env_id | family | max_steps |",
        "| --- | --- | --- | ---: |",
    ]
    for row in report.get("scenario_list", []) or []:
        if isinstance(row, Mapping):
            lines.append(
                f"| `{row.get('scenario_id', '')}` | `{row.get('env_id', '')}` | `{row.get('scenario_family', '')}` | {row.get('max_steps', '')} |"
            )
    lines.extend(["", "## Selected Cache Env Presence", "", "| scenario | env_id | present |", "| --- | --- | --- |"])
    for scenario_id, item in dict(report.get("selected_cache_env_presence", {})).items():
        if isinstance(item, Mapping):
            lines.append(f"| `{scenario_id}` | `{item.get('env_id', '')}` | `{item.get('present_in_selected_cache', False)}` |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-cache-path", default=DEFAULT_CACHE_CANDIDATES[0])
    parser.add_argument("--candidate-cache-path", action="append", default=[])
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--plan-cache-key-policy", default="scenario_timestep")
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--controller", default="llm_rspc_v2")
    parser.add_argument("--allow-default-cache", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        selected_cache_path=args.selected_cache_path,
        scenario_ids=args.scenario,
        candidate_cache_paths=args.candidate_cache_path or None,
        plan_cache_key_policy=args.plan_cache_key_policy,
        max_steps=args.max_steps,
        controller=args.controller,
        allow_default_cache=args.allow_default_cache,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"cache_scenario_plan_ready={report['cache_scenario_plan_ready']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["cache_scenario_plan_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
