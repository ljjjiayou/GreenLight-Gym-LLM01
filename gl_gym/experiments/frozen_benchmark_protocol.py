"""Protocol helpers for safe frozen LLM benchmark runs.

This module is deliberately side-effect-light: by default it audits existing
cache/result files and prints per-scenario commands. It does not call the LLM or
run the simulator unless a user explicitly runs the printed commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gl_gym.agent.plan_cache import SCHEMA_VERSION
from gl_gym.experiments.diagnose_ppo_vs_llm import parse_int_list


LLM_AGGREGATE_KEYS = (
    "total_reward",
    "total_profit",
    "total_revenue",
    "total_heat_cost",
    "total_co2_cost",
    "total_elec_cost",
    "total_temp_violation",
    "total_rh_violation",
    "total_rh_low_violation",
    "total_rh_high_violation",
    "total_vpd_high_excess",
    "mean_u_heating",
    "mean_u_co2",
    "mean_u_screen",
    "mean_u_ventilation",
    "mean_u_lighting",
    "mean_u_shading",
)


@dataclass(frozen=True)
class ScenarioSpec:
    year: int
    day: int
    seed: int
    max_steps: int = 240

    @property
    def scenario_id(self) -> str:
        return f"y{self.year}_d{self.day}_s{self.seed}_n{self.max_steps}"

    @property
    def env_id(self) -> str:
        return f"TomatoEnv_y{self.year}_d{self.day}_s{self.seed}"


def build_scenarios(
    *,
    years: Iterable[int],
    days: Iterable[int],
    seeds: Iterable[int],
    max_steps: int,
) -> List[ScenarioSpec]:
    return [
        ScenarioSpec(year=int(year), day=int(day), seed=int(seed), max_steps=int(max_steps))
        for year in years
        for day in days
        for seed in seeds
    ]


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def file_sha256(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry_env_id(entry: Mapping[str, Any]) -> str:
    env = entry.get("env_id")
    if env:
        return str(env)
    key = entry.get("key_payload")
    if isinstance(key, Mapping) and key.get("env_id"):
        return str(key["env_id"])
    return "unknown"


def _entry_status(entry: Mapping[str, Any]) -> str:
    if entry.get("parsed_plan"):
        return "parsed_plan"
    if entry.get("llm_action_found") is False:
        return "written_empty"
    return str(entry.get("status") or "unknown")


def audit_plan_cache(
    cache_path: str | Path,
    scenarios: Optional[Sequence[ScenarioSpec]] = None,
    *,
    min_entries_per_env: int = 1,
) -> Dict[str, Any]:
    path = Path(cache_path)
    data = read_json(path)
    entries = data.get("entries", {})
    if not isinstance(entries, dict):
        raise ValueError(f"Plan cache entries must be an object: {path}")

    envs: Dict[str, Dict[str, Any]] = {}
    missing_action_count = 0
    missing_parsed_plan_count = 0
    written_empty_count = 0
    status_counts: Dict[str, int] = {}

    for key, raw_entry in entries.items():
        if not isinstance(raw_entry, dict):
            continue
        env_id = _entry_env_id(raw_entry)
        env = envs.setdefault(
            env_id,
            {
                "entry_count": 0,
                "keys": [],
                "statuses": {},
                "missing_buffered_action": 0,
                "missing_parsed_plan": 0,
                "written_empty": 0,
            },
        )
        status = _entry_status(raw_entry)
        env["entry_count"] += 1
        env["keys"].append(str(key))
        env["statuses"][status] = int(env["statuses"].get(status, 0)) + 1
        status_counts[status] = status_counts.get(status, 0) + 1

        if not raw_entry.get("buffered_action"):
            missing_action_count += 1
            env["missing_buffered_action"] += 1
        if not raw_entry.get("parsed_plan"):
            missing_parsed_plan_count += 1
            env["missing_parsed_plan"] += 1
        if status == "written_empty":
            written_empty_count += 1
            env["written_empty"] += 1

    required_envs = [scenario.env_id for scenario in scenarios or []]
    missing_envs = [env_id for env_id in required_envs if env_id not in envs]
    low_coverage_envs = [
        env_id
        for env_id in required_envs
        if env_id in envs and int(envs[env_id].get("entry_count", 0)) < int(min_entries_per_env)
    ]

    return {
        "cache_path": str(path),
        "cache_sha256": file_sha256(path),
        "schema_version": str(data.get("schema_version", "")),
        "schema_ok": data.get("schema_version") == SCHEMA_VERSION,
        "entry_count": int(len(entries)),
        "env_count": int(len(envs)),
        "envs": dict(sorted(envs.items())),
        "required_env_count": int(len(required_envs)),
        "missing_envs": missing_envs,
        "low_coverage_envs": low_coverage_envs,
        "min_entries_per_env": int(min_entries_per_env),
        "missing_buffered_action_count": int(missing_action_count),
        "missing_parsed_plan_count": int(missing_parsed_plan_count),
        "written_empty_count": int(written_empty_count),
        "status_counts": dict(sorted(status_counts.items())),
        "ok": bool(
            data.get("schema_version") == SCHEMA_VERSION
            and not missing_envs
            and not low_coverage_envs
            and missing_action_count == 0
            and missing_parsed_plan_count == 0
        ),
    }


def _summary_key(summary: Mapping[str, Any]) -> Tuple[str, str]:
    return str(summary.get("scenario_id", "")), str(summary.get("controller", ""))


def summaries_by_scenario_controller(result: Mapping[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    summaries = result.get("summaries", [])
    if not isinstance(summaries, list):
        raise ValueError("Benchmark result must contain a summaries list")
    indexed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in summaries:
        if isinstance(item, dict):
            indexed[_summary_key(item)] = item
    return indexed


def validate_strict_replay_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    failures: List[Dict[str, Any]] = []
    for summary in result.get("summaries", []):
        if not isinstance(summary, dict):
            continue
        controller = str(summary.get("controller", ""))
        if not controller.startswith("llm"):
            continue
        aggregate = summary.get("aggregate", {})
        if not isinstance(aggregate, dict):
            continue
        steps = int(aggregate.get("steps", 0) or 0)
        enabled = int(aggregate.get("plan_cache_enabled_steps", 0) or 0)
        hits = int(aggregate.get("plan_cache_hit_steps", 0) or 0)
        source_counts = aggregate.get("source_counts", {})
        unknown_steps = int(source_counts.get("unknown", 0)) if isinstance(source_counts, dict) else 0
        reasons: List[str] = []
        if steps <= 0:
            reasons.append("zero_steps")
        if enabled != steps:
            reasons.append("cache_not_enabled_all_steps")
        if hits != steps:
            reasons.append("cache_not_hit_all_steps")
        if unknown_steps:
            reasons.append("unknown_source_steps")
        if reasons:
            failures.append(
                {
                    "scenario_id": summary.get("scenario_id"),
                    "controller": controller,
                    "steps": steps,
                    "cache_enabled_steps": enabled,
                    "cache_hit_steps": hits,
                    "unknown_source_steps": unknown_steps,
                    "reasons": reasons,
                }
            )
    return {"ok": not failures, "failure_count": len(failures), "failures": failures}


def compare_record_replay(
    record_result: Mapping[str, Any],
    replay_result: Mapping[str, Any],
    *,
    tolerance: float = 1e-8,
) -> Dict[str, Any]:
    record_index = summaries_by_scenario_controller(record_result)
    replay_index = summaries_by_scenario_controller(replay_result)
    failures: List[Dict[str, Any]] = []

    for (scenario_id, controller), record_summary in sorted(record_index.items()):
        if controller != "llm":
            continue
        replay_summary = replay_index.get((scenario_id, "llm"))
        if not replay_summary:
            failures.append({"scenario_id": scenario_id, "reason": "missing_replay_llm"})
            continue
        record_aggregate = record_summary.get("aggregate", {})
        replay_aggregate = replay_summary.get("aggregate", {})
        deltas: Dict[str, float] = {}
        for key in LLM_AGGREGATE_KEYS:
            delta = float(replay_aggregate.get(key, 0.0)) - float(record_aggregate.get(key, 0.0))
            if abs(delta) > float(tolerance):
                deltas[key] = delta
        if deltas:
            failures.append({"scenario_id": scenario_id, "reason": "aggregate_delta", "deltas": deltas})
    return {"ok": not failures, "failure_count": len(failures), "failures": failures}


def validate_shadow_invariance(
    replay_result: Mapping[str, Any],
    *,
    tolerance: float = 1e-8,
) -> Dict[str, Any]:
    index = summaries_by_scenario_controller(replay_result)
    scenario_ids = sorted({scenario for scenario, _controller in index})
    failures: List[Dict[str, Any]] = []
    shadow_stats: Dict[str, Any] = {
        "scenario_count": 0,
        "total_available_steps": 0,
        "total_would_select_steps": 0,
        "best_candidate_counts": {},
        "reject_reason_counts": {},
    }

    for scenario_id in scenario_ids:
        llm = index.get((scenario_id, "llm"))
        shadow = index.get((scenario_id, "llm_sero_shadow"))
        if not llm or not shadow:
            continue
        shadow_stats["scenario_count"] += 1
        llm_aggregate = llm.get("aggregate", {})
        shadow_aggregate = shadow.get("aggregate", {})
        deltas: Dict[str, float] = {}
        for key in LLM_AGGREGATE_KEYS:
            delta = float(shadow_aggregate.get(key, 0.0)) - float(llm_aggregate.get(key, 0.0))
            if abs(delta) > float(tolerance):
                deltas[key] = delta
        if deltas:
            failures.append({"scenario_id": scenario_id, "reason": "shadow_changed_behavior", "deltas": deltas})

        shadow_stats["total_available_steps"] += int(shadow_aggregate.get("mc_sero_available_steps", 0) or 0)
        shadow_stats["total_would_select_steps"] += int(shadow_aggregate.get("mc_sero_would_select_steps", 0) or 0)
        for key, value in (shadow_aggregate.get("mc_sero_best_candidate_counts", {}) or {}).items():
            shadow_stats["best_candidate_counts"][str(key)] = int(shadow_stats["best_candidate_counts"].get(str(key), 0)) + int(value)
        for key, value in (shadow_aggregate.get("mc_sero_reject_reason_counts", {}) or {}).items():
            shadow_stats["reject_reason_counts"][str(key)] = int(shadow_stats["reject_reason_counts"].get(str(key), 0)) + int(value)

    shadow_stats["best_candidate_counts"] = dict(sorted(shadow_stats["best_candidate_counts"].items()))
    shadow_stats["reject_reason_counts"] = dict(sorted(shadow_stats["reject_reason_counts"].items()))
    return {"ok": not failures, "failure_count": len(failures), "failures": failures, "shadow_stats": shadow_stats}


def make_manifest(
    *,
    scenarios: Sequence[ScenarioSpec],
    cache_path: str,
    commit: str,
    llm_model: str = "qwen-max-latest",
    key_policy: str = "scenario_timestep",
    llm_interval: int = 12,
) -> Dict[str, Any]:
    return {
        "schema_version": "frozen_benchmark_manifest_v1",
        "commit": str(commit),
        "llm_model": str(llm_model),
        "plan_cache_path": str(cache_path),
        "plan_cache_key_policy": str(key_policy),
        "llm_interval": int(llm_interval),
        "scenarios": [
            {
                **asdict(scenario),
                "scenario_id": scenario.scenario_id,
                "env_id": scenario.env_id,
                "status": "pending",
                "record_json": "",
                "replay_json": "",
                "notes": "",
            }
            for scenario in scenarios
        ],
    }


def scenario_command(
    scenario: ScenarioSpec,
    *,
    mode: str,
    controllers: str,
    cache_path: str,
    output_json: str,
    strict: bool = False,
) -> str:
    parts = [
        "python",
        "gl_gym\\experiments\\run_frozen_benchmark.py",
        "--years",
        str(scenario.year),
        "--days",
        str(scenario.day),
        "--seeds",
        str(scenario.seed),
        "--controllers",
        str(controllers),
        "--max-steps",
        str(scenario.max_steps),
        "--plan-cache-mode",
        str(mode),
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--plan-cache-path",
        str(cache_path),
        "--output-json",
        str(output_json),
    ]
    if strict:
        parts.append("--plan-cache-strict")
    return " ".join(shlex.quote(part) for part in parts)


def build_canary_commands(
    scenarios: Sequence[ScenarioSpec],
    *,
    cache_path: str,
    output_dir: str,
) -> List[Dict[str, str]]:
    commands: List[Dict[str, str]] = []
    for scenario in scenarios:
        stem = scenario.scenario_id
        record_json = str(Path(output_dir) / f"{stem}_record.json")
        replay_json = str(Path(output_dir) / f"{stem}_replay.json")
        commands.append(
            {
                "scenario_id": scenario.scenario_id,
                "record": scenario_command(
                    scenario,
                    mode="record",
                    controllers="llm",
                    cache_path=cache_path,
                    output_json=record_json,
                ),
                "replay": scenario_command(
                    scenario,
                    mode="replay",
                    controllers="llm,llm_sero_shadow",
                    cache_path=cache_path,
                    output_json=replay_json,
                    strict=True,
                ),
            }
        )
    return commands


def _scenarios_from_args(args: argparse.Namespace) -> List[ScenarioSpec]:
    return build_scenarios(
        years=parse_int_list(args.years),
        days=parse_int_list(args.days),
        seeds=parse_int_list(args.seeds),
        max_steps=int(args.max_steps),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and plan frozen benchmark runs.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_grid(p: argparse.ArgumentParser) -> None:
        p.add_argument("--years", type=str, default="2010,2015,2020")
        p.add_argument("--days", type=str, default="59,120,180,240")
        p.add_argument("--seeds", type=str, default="42,43,44")
        p.add_argument("--max-steps", type=int, default=240)

    init = sub.add_parser("init-manifest")
    add_grid(init)
    init.add_argument("--commit", type=str, default="")
    init.add_argument("--cache-path", type=str, required=True)
    init.add_argument("--output-json", type=str, required=True)

    audit = sub.add_parser("audit-cache")
    add_grid(audit)
    audit.add_argument("--cache-path", type=str, required=True)
    audit.add_argument("--min-entries-per-env", type=int, default=1)
    audit.add_argument("--output-json", type=str, default="")

    check = sub.add_parser("check-replay")
    check.add_argument("--replay-json", type=str, required=True)
    check.add_argument("--record-json", type=str, default="")
    check.add_argument("--tolerance", type=float, default=1e-8)
    check.add_argument("--output-json", type=str, default="")
    check.add_argument("--fail-on-error", action="store_true")

    commands = sub.add_parser("commands")
    add_grid(commands)
    commands.add_argument("--cache-path", type=str, required=True)
    commands.add_argument("--output-dir", type=str, default="gl_gym/result/benchmarks/frozen_shards")

    args = parser.parse_args()
    if args.command == "init-manifest":
        manifest = make_manifest(
            scenarios=_scenarios_from_args(args),
            cache_path=args.cache_path,
            commit=args.commit,
        )
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {output}")
        return

    if args.command == "audit-cache":
        result = audit_plan_cache(
            args.cache_path,
            scenarios=_scenarios_from_args(args),
            min_entries_per_env=int(args.min_entries_per_env),
        )
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output_json:
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            print(f"wrote {output}")
        else:
            print(text)
        return

    if args.command == "check-replay":
        replay = read_json(args.replay_json)
        result: Dict[str, Any] = {
            "strict_replay": validate_strict_replay_result(replay),
            "shadow_invariance": validate_shadow_invariance(replay, tolerance=float(args.tolerance)),
        }
        if args.record_json:
            record = read_json(args.record_json)
            result["record_replay"] = compare_record_replay(record, replay, tolerance=float(args.tolerance))
        result["ok"] = all(item.get("ok", False) for item in result.values() if isinstance(item, dict))
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output_json:
            output = Path(args.output_json)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            print(f"wrote {output}")
        else:
            print(text)
        if args.fail_on_error and not result["ok"]:
            raise SystemExit(1)
        return

    if args.command == "commands":
        for item in build_canary_commands(_scenarios_from_args(args), cache_path=args.cache_path, output_dir=args.output_dir):
            print(f"# {item['scenario_id']}")
            print(item["record"])
            print(item["replay"])
        return


if __name__ == "__main__":
    main()
