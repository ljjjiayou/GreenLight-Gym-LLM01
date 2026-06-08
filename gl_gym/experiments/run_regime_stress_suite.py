"""Run or plan frozen benchmarks from a regime stress-suite manifest."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gl_gym.agent.llm_agent import AgentConfig


DEFAULT_MANIFEST_PATH = project_root / "gl_gym" / "result" / "stress_suites" / "regime_stress_v1.json"
DEFAULT_OUTPUT_DIR = project_root / "gl_gym" / "result" / "stress_suites" / "regime_stress_v1_runs"
DEFAULT_PLAN_CACHE_PATH = "gl_gym/result/plan_cache/llm_plan_cache.json"
DEFAULT_CONTROLLERS = "llm,llm_rspc_v2,llm_sero_shadow,ppo"

REGIME_DELTA_KEYS = (
    "runtime_error_steps",
    "total_reward",
    "total_profit",
    "total_rh_low_violation",
    "total_vpd_high_excess",
    "total_temp_violation",
    "total_rh_high_violation",
    "dew_margin_air_lt1_steps",
    "dew_margin_air_lt0_steps",
    "canopy_dew_margin_lt1_steps",
    "canopy_dew_margin_lt0_steps",
)


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _split_csv(text: str) -> List[str]:
    if str(text).strip().lower() in {"", "all", "*"}:
        return []
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _scenario_sort_key(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(entry.get("split", "")),
        int(entry.get("year", 0) or 0),
        int(entry.get("day", 0) or 0),
        int(entry.get("seed", 0) or 0),
        int(entry.get("max_steps", 0) or 0),
        str(entry.get("scenario_id", "")),
    )


def filter_entries(
    manifest: Mapping[str, Any],
    *,
    splits: Sequence[str] | None = None,
    regimes: Sequence[str] | None = None,
    scenario_ids: Sequence[str] | None = None,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    split_set = {str(item) for item in splits or []}
    regime_set = {str(item) for item in regimes or []}
    scenario_set = {str(item) for item in scenario_ids or []}
    selected: List[Dict[str, Any]] = []
    for raw in manifest.get("scenarios", []) or []:
        if not isinstance(raw, dict):
            continue
        entry = dict(raw)
        if split_set and str(entry.get("split", "")) not in split_set:
            continue
        tags = {str(tag) for tag in entry.get("regime_tags", []) or []}
        if regime_set and not (tags & regime_set):
            continue
        if scenario_set and str(entry.get("scenario_id", "")) not in scenario_set:
            continue
        selected.append(entry)
    selected = sorted(selected, key=_scenario_sort_key)
    if limit > 0:
        selected = selected[: int(limit)]
    return selected


def command_to_string(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def benchmark_command_parts(
    entry: Mapping[str, Any],
    *,
    python_executable: str,
    controllers: str,
    plan_cache_path: str,
    output_json: str | Path,
    output_trace_dir: str | Path,
    config: str,
    model_path: str,
    vecnorm_path: str,
    plan_cache_mode: str,
    llm_model: str,
    llm_interval: int,
    llm_max_iterations: int,
    llm_max_tokens: int,
) -> List[str]:
    return [
        str(python_executable),
        "gl_gym\\experiments\\run_frozen_benchmark.py",
        "--config",
        str(config),
        "--model-path",
        str(model_path),
        "--vecnorm-path",
        str(vecnorm_path),
        "--years",
        str(entry["year"]),
        "--days",
        str(entry["day"]),
        "--seeds",
        str(entry["seed"]),
        "--controllers",
        str(controllers),
        "--max-steps",
        str(entry.get("max_steps", 240)),
        "--llm-model",
        str(llm_model),
        "--llm-interval",
        str(llm_interval),
        "--llm-max-iterations",
        str(llm_max_iterations),
        "--llm-max-tokens",
        str(llm_max_tokens),
        "--plan-cache-mode",
        str(plan_cache_mode),
        "--plan-cache-strict",
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--plan-cache-path",
        str(plan_cache_path),
        "--output-json",
        str(output_json),
        "--output-trace-dir",
        str(output_trace_dir),
    ]


def check_replay_command_parts(
    *,
    python_executable: str,
    replay_json: str | Path,
    output_json: str | Path,
    fail_on_error: bool,
) -> List[str]:
    parts = [
        str(python_executable),
        "gl_gym\\experiments\\frozen_benchmark_protocol.py",
        "check-replay",
        "--replay-json",
        str(replay_json),
        "--output-json",
        str(output_json),
    ]
    if fail_on_error:
        parts.append("--fail-on-error")
    return parts


def compare_command_parts(
    *,
    python_executable: str,
    replay_json: str | Path,
    output_json: str | Path,
    output_report: str | Path,
    baseline: str,
    candidate: str,
    shadow: str,
    fail_on_error: bool,
) -> List[str]:
    parts = [
        str(python_executable),
        "gl_gym\\experiments\\frozen_benchmark_protocol.py",
        "compare-controllers",
        "--replay-json",
        str(replay_json),
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--shadow",
        str(shadow),
        "--output-json",
        str(output_json),
        "--output-report",
        str(output_report),
    ]
    if fail_on_error:
        parts.append("--fail-on-error")
    return parts


def _group_by_split(entries: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(str(entry.get("split", "unknown")), []).append(dict(entry))
    return {split: sorted(items, key=_scenario_sort_key) for split, items in sorted(grouped.items())}


def build_run_plan(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    splits: Sequence[str] | None = None,
    regimes: Sequence[str] | None = None,
    scenario_ids: Sequence[str] | None = None,
    limit: int = 0,
    python_executable: str = "python",
    controllers: str = DEFAULT_CONTROLLERS,
    plan_cache_path: str = DEFAULT_PLAN_CACHE_PATH,
    config: str = "gl_gym/configs/envs/TomatoEnv.yml",
    model_path: str = "train_data/AgriControl/ppo/deterministic/models/None/best_model.zip",
    vecnorm_path: str = "train_data/AgriControl/ppo/deterministic/envs/None/best_vecnormalize.pkl",
    plan_cache_mode: str = "replay",
    llm_model: str = AgentConfig.model_name,
    llm_interval: int = 12,
    llm_max_iterations: int = 1,
    llm_max_tokens: int = 260,
    baseline: str = "llm",
    candidate: str = "llm_rspc_v2",
    shadow: str = "llm_sero_shadow",
    fail_on_error: bool = False,
) -> Dict[str, Any]:
    entries = filter_entries(manifest, splits=splits, regimes=regimes, scenario_ids=scenario_ids, limit=limit)
    output_dir = Path(output_dir)
    groups: List[Dict[str, Any]] = []
    for split, split_entries in _group_by_split(entries).items():
        split_dir = output_dir / split
        trace_dir = split_dir / "traces"
        scenario_outputs: List[str] = []
        benchmark_commands: List[Dict[str, Any]] = []
        for entry in split_entries:
            scenario_id = str(entry.get("scenario_id", "scenario"))
            output_json = split_dir / f"{scenario_id}_replay.json"
            scenario_outputs.append(str(output_json))
            parts = benchmark_command_parts(
                entry,
                python_executable=python_executable,
                controllers=controllers,
                plan_cache_path=plan_cache_path,
                output_json=output_json,
                output_trace_dir=trace_dir,
                config=config,
                model_path=model_path,
                vecnorm_path=vecnorm_path,
                plan_cache_mode=plan_cache_mode,
                llm_model=llm_model,
                llm_interval=llm_interval,
                llm_max_iterations=llm_max_iterations,
                llm_max_tokens=llm_max_tokens,
            )
            benchmark_commands.append(
                {
                    "scenario_id": scenario_id,
                    "parts": parts,
                    "command": command_to_string(parts),
                    "output_json": str(output_json),
                }
            )

        combined_json = split_dir / f"{split}_benchmark_replay.json"
        check_json = split_dir / f"{split}_check_replay.json"
        compare_json = split_dir / f"{split}_compare_controllers.json"
        compare_report = split_dir / f"{split}_compare_controllers.md"
        regime_delta_json = split_dir / f"{split}_regime_delta_summary.json"
        regime_delta_report = split_dir / f"{split}_regime_delta_report.md"
        check_parts = check_replay_command_parts(
            python_executable=python_executable,
            replay_json=combined_json,
            output_json=check_json,
            fail_on_error=fail_on_error,
        )
        compare_parts = compare_command_parts(
            python_executable=python_executable,
            replay_json=combined_json,
            output_json=compare_json,
            output_report=compare_report,
            baseline=baseline,
            candidate=candidate,
            shadow=shadow,
            fail_on_error=fail_on_error,
        )
        groups.append(
            {
                "split": split,
                "scenario_count": len(split_entries),
                "scenario_ids": [str(entry.get("scenario_id", "")) for entry in split_entries],
                "trace_dir": str(trace_dir),
                "scenario_output_jsons": scenario_outputs,
                "combined_replay_json": str(combined_json),
                "check_replay_json": str(check_json),
                "compare_json": str(compare_json),
                "compare_report": str(compare_report),
                "regime_delta_json": str(regime_delta_json),
                "regime_delta_report": str(regime_delta_report),
                "benchmark_commands": benchmark_commands,
                "postprocess_commands": {
                    "check_replay": command_to_string(check_parts),
                    "compare_controllers": command_to_string(compare_parts),
                },
                "postprocess_parts": {
                    "check_replay": check_parts,
                    "compare_controllers": compare_parts,
                },
            }
        )

    return {
        "schema_version": "regime_stress_run_plan_v1",
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "selected_scenario_count": len(entries),
        "controllers": str(controllers),
        "plan_cache_path": str(plan_cache_path),
        "plan_cache_mode": str(plan_cache_mode),
        "plan_cache_key_policy": "scenario_timestep",
        "strict_replay": True,
        "filters": {
            "splits": list(splits or []),
            "regimes": list(regimes or []),
            "scenario_ids": list(scenario_ids or []),
            "limit": int(limit),
        },
        "groups": groups,
    }


def merge_benchmark_outputs(paths: Sequence[str | Path]) -> Dict[str, Any]:
    summaries: List[Dict[str, Any]] = []
    benchmark_meta: List[Dict[str, Any]] = []
    for path in paths:
        data = read_json(path)
        if isinstance(data.get("benchmark"), dict):
            benchmark_meta.append(dict(data["benchmark"]))
        for item in data.get("summaries", []) or []:
            if isinstance(item, dict):
                summaries.append(dict(item))
    return {
        "benchmark": {
            "merged_from": [str(path) for path in paths],
            "source_benchmark_count": len(benchmark_meta),
        },
        "summaries": summaries,
    }


def _summary_index(result: Mapping[str, Any]) -> Dict[tuple[str, str], Dict[str, Any]]:
    indexed: Dict[tuple[str, str], Dict[str, Any]] = {}
    for item in result.get("summaries", []) or []:
        if isinstance(item, dict):
            indexed[(str(item.get("scenario_id", "")), str(item.get("controller", "")))] = item
    return indexed


def build_regime_delta_summary(
    manifest_entries: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
    *,
    baseline: str = "llm",
    candidate: str = "llm_rspc_v2",
) -> Dict[str, Any]:
    index = _summary_index(result)
    manifest_by_id = {str(entry.get("scenario_id", "")): entry for entry in manifest_entries}
    regimes: Dict[str, Dict[str, Any]] = {}
    scenario_rows: List[Dict[str, Any]] = []
    for scenario_id, entry in sorted(manifest_by_id.items()):
        base = index.get((scenario_id, baseline))
        cand = index.get((scenario_id, candidate))
        if not base or not cand:
            continue
        base_agg = base.get("aggregate", {}) if isinstance(base.get("aggregate"), Mapping) else {}
        cand_agg = cand.get("aggregate", {}) if isinstance(cand.get("aggregate"), Mapping) else {}
        delta = {
            key: float(cand_agg.get(key, 0.0) or 0.0) - float(base_agg.get(key, 0.0) or 0.0)
            for key in REGIME_DELTA_KEYS
        }
        row = {
            "scenario_id": scenario_id,
            "split": entry.get("split", ""),
            "regime_tags": list(entry.get("regime_tags", []) or []),
            "delta": delta,
            "safety_regression": bool(
                delta.get("runtime_error_steps", 0.0) > 0.0
                or delta.get("total_temp_violation", 0.0) > 0.0
                or delta.get("total_rh_high_violation", 0.0) > 0.0
                or delta.get("dew_margin_air_lt1_steps", 0.0) > 0.0
                or delta.get("canopy_dew_margin_lt1_steps", 0.0) > 0.0
            ),
            "performance_regression": bool(
                delta.get("total_reward", 0.0) < 0.0 or delta.get("total_profit", 0.0) < 0.0
            ),
        }
        scenario_rows.append(row)
        for regime in row["regime_tags"]:
            bucket = regimes.setdefault(
                str(regime),
                {"scenario_count": 0, "delta_sum": {key: 0.0 for key in REGIME_DELTA_KEYS}, "safety_regressions": [], "performance_regressions": []},
            )
            bucket["scenario_count"] += 1
            for key in REGIME_DELTA_KEYS:
                bucket["delta_sum"][key] += float(delta.get(key, 0.0))
            if row["safety_regression"]:
                bucket["safety_regressions"].append(scenario_id)
            if row["performance_regression"]:
                bucket["performance_regressions"].append(scenario_id)

    for bucket in regimes.values():
        count = max(int(bucket["scenario_count"]), 1)
        bucket["delta_mean"] = {key: float(value) / count for key, value in bucket["delta_sum"].items()}
    return {
        "baseline": baseline,
        "candidate": candidate,
        "scenario_count": len(scenario_rows),
        "regimes": dict(sorted(regimes.items())),
        "scenarios": scenario_rows,
    }


def build_regime_delta_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Regime Delta Report",
        "",
        f"- Baseline: `{summary.get('baseline')}`",
        f"- Candidate: `{summary.get('candidate')}`",
        f"- Scenarios: {summary.get('scenario_count', 0)}",
        "",
        "## Regime Aggregates",
        "",
        "| regime | scenarios | d_runtime | d_reward | d_profit | d_RHlow | d_VPDhi | d_temp | d_RHhigh | d_dew<1 | d_canopy<1 | safety regressions | performance regressions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for regime, item in (summary.get("regimes", {}) or {}).items():
        delta = item.get("delta_sum", {}) if isinstance(item.get("delta_sum"), Mapping) else {}
        lines.append(
            "| {regime} | {count} | {runtime:.0f} | {reward:.4f} | {profit:.4f} | {rhlow:.4f} | {vpd:.4f} | {temp:.4f} | {rhhigh:.4f} | {dew:.0f} | {canopy:.0f} | {safety} | {perf} |".format(
                regime=regime,
                count=int(item.get("scenario_count", 0) or 0),
                runtime=float(delta.get("runtime_error_steps", 0.0)),
                reward=float(delta.get("total_reward", 0.0)),
                profit=float(delta.get("total_profit", 0.0)),
                rhlow=float(delta.get("total_rh_low_violation", 0.0)),
                vpd=float(delta.get("total_vpd_high_excess", 0.0)),
                temp=float(delta.get("total_temp_violation", 0.0)),
                rhhigh=float(delta.get("total_rh_high_violation", 0.0)),
                dew=float(delta.get("dew_margin_air_lt1_steps", 0.0)),
                canopy=float(delta.get("canopy_dew_margin_lt1_steps", 0.0)),
                safety=", ".join(item.get("safety_regressions", []) or []) or "-",
                perf=", ".join(item.get("performance_regressions", []) or []) or "-",
            )
        )
    return "\n".join(lines) + "\n"


def _run(parts: Sequence[str]) -> None:
    print(command_to_string(parts))
    subprocess.run([str(part) for part in parts], cwd=project_root, check=True)


def execute_run_plan(plan: Mapping[str, Any], manifest: Mapping[str, Any], *, baseline: str, candidate: str) -> None:
    entries_by_id = {str(entry.get("scenario_id", "")): entry for entry in manifest.get("scenarios", []) if isinstance(entry, Mapping)}
    for group in plan.get("groups", []) or []:
        if not isinstance(group, Mapping):
            continue
        for command in group.get("benchmark_commands", []) or []:
            _run(command["parts"])
        combined = merge_benchmark_outputs(group.get("scenario_output_jsons", []) or [])
        write_json(group["combined_replay_json"], combined)
        _run(group["postprocess_parts"]["check_replay"])
        _run(group["postprocess_parts"]["compare_controllers"])
        selected_entries = [entries_by_id[scenario_id] for scenario_id in group.get("scenario_ids", []) if scenario_id in entries_by_id]
        regime_summary = build_regime_delta_summary(selected_entries, combined, baseline=baseline, candidate=candidate)
        write_json(group["regime_delta_json"], regime_summary)
        Path(group["regime_delta_report"]).parent.mkdir(parents=True, exist_ok=True)
        Path(group["regime_delta_report"]).write_text(build_regime_delta_report(regime_summary), encoding="utf-8")
        print(f"wrote {group['regime_delta_json']}")
        print(f"wrote {group['regime_delta_report']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or execute regime stress-suite frozen benchmarks.")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--regime", type=str, default="all")
    parser.add_argument("--scenario-id", type=str, default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--controllers", type=str, default=DEFAULT_CONTROLLERS)
    parser.add_argument("--plan-cache-path", type=str, default=DEFAULT_PLAN_CACHE_PATH)
    parser.add_argument("--plan-cache-mode", type=str, choices=["replay", "record", "refresh", "off"], default="replay")
    parser.add_argument("--python-executable", type=str, default="python")
    parser.add_argument("--config", type=str, default="gl_gym/configs/envs/TomatoEnv.yml")
    parser.add_argument("--model-path", type=str, default="train_data/AgriControl/ppo/deterministic/models/None/best_model.zip")
    parser.add_argument("--vecnorm-path", type=str, default="train_data/AgriControl/ppo/deterministic/envs/None/best_vecnormalize.pkl")
    parser.add_argument("--llm-model", type=str, default=AgentConfig.model_name)
    parser.add_argument("--llm-interval", type=int, default=12)
    parser.add_argument("--llm-max-iterations", type=int, default=1)
    parser.add_argument("--llm-max-tokens", type=int, default=260)
    parser.add_argument("--baseline", type=str, default="llm")
    parser.add_argument("--candidate", type=str, default="llm_rspc_v2")
    parser.add_argument("--shadow", type=str, default="llm_sero_shadow")
    parser.add_argument("--plan-json", type=str, default="")
    parser.add_argument("--execute", action="store_true", help="Actually run benchmarks. Without this, only print the run plan.")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    plan = build_run_plan(
        manifest,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        splits=_split_csv(args.split),
        regimes=_split_csv(args.regime),
        scenario_ids=_split_csv(args.scenario_id),
        limit=int(args.limit),
        python_executable=args.python_executable,
        controllers=args.controllers,
        plan_cache_path=args.plan_cache_path,
        config=args.config,
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
        plan_cache_mode=args.plan_cache_mode,
        llm_model=args.llm_model,
        llm_interval=int(args.llm_interval),
        llm_max_iterations=int(args.llm_max_iterations),
        llm_max_tokens=int(args.llm_max_tokens),
        baseline=args.baseline,
        candidate=args.candidate,
        shadow=args.shadow,
        fail_on_error=bool(args.fail_on_error),
    )
    text = json.dumps(plan, ensure_ascii=False, indent=2)
    if args.plan_json:
        write_json(args.plan_json, plan)
        print(f"wrote {args.plan_json}")
    print(text)
    if args.execute:
        execute_run_plan(plan, manifest, baseline=args.baseline, candidate=args.candidate)


if __name__ == "__main__":
    main()
