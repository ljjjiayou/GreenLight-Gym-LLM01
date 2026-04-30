"""Run frozen LLM-plan benchmark scenarios.

This runner is intentionally small and explicit. It lets us first record LLM
plans, then replay those exact plans while comparing rollout/controller changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
from dotenv import load_dotenv
from stable_baselines3 import PPO

from gl_gym.agent.llm_agent import AgentConfig
from gl_gym.experiments.diagnose_ppo_vs_llm import (
    build_env,
    parse_int_list,
    run_llm_trace,
    run_ppo_trace,
    safety_patterns,
    sum_metrics,
    write_rows_csv,
)

load_dotenv()


@dataclass(frozen=True)
class BenchmarkJob:
    controller: str
    year: int
    day: int
    seed: int
    max_steps: int

    @property
    def scenario_id(self) -> str:
        return f"y{self.year}_d{self.day}_s{self.seed}_n{self.max_steps}"


def parse_controller_list(text: str) -> List[str]:
    allowed = {"ppo", "llm", "llm_hem", "llm_sero_shadow", "llm_rspc_v2"}
    controllers = [part.strip() for part in str(text).split(",") if part.strip()]
    invalid = [name for name in controllers if name not in allowed]
    if invalid:
        raise ValueError(f"Unknown controllers: {invalid}; allowed={sorted(allowed)}")
    return controllers


def build_jobs(years: Iterable[int], days: Iterable[int], seeds: Iterable[int], controllers: Iterable[str], max_steps: int) -> List[BenchmarkJob]:
    jobs: List[BenchmarkJob] = []
    for year in years:
        for day in days:
            for seed in seeds:
                for controller in controllers:
                    jobs.append(
                        BenchmarkJob(
                            controller=str(controller),
                            year=int(year),
                            day=int(day),
                            seed=int(seed),
                            max_steps=int(max_steps),
                        )
                    )
    return jobs


def summarize_trace(job: BenchmarkJob, rows: List[Dict[str, Any]], elapsed_seconds: float) -> Dict[str, Any]:
    return {
        "job": asdict(job),
        "scenario_id": job.scenario_id,
        "controller": job.controller,
        "elapsed_seconds": float(elapsed_seconds),
        "aggregate": sum_metrics(rows),
        "safety_patterns": safety_patterns(rows).get("ppo" if job.controller == "ppo" else "llm_director", {}),
    }


def run_job(
    job: BenchmarkJob,
    *,
    config: Dict[str, Any],
    model: PPO | None,
    vecnorm_path: Path,
    api_key: str,
    uncertainty_scale: float,
    llm_model: str,
    llm_interval: int,
    llm_max_iterations: int,
    llm_max_tokens: int,
    plan_cache_mode: str,
    plan_cache_path: str,
    plan_cache_strict: bool,
    plan_cache_key_policy: str,
    humidity_memory_path: str,
    humidity_memory_teacher_policy_id: str,
    humidity_memory_baseline_controller_id: str,
    humidity_memory_version: str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    start = time.perf_counter()
    if job.controller == "ppo":
        if model is None:
            raise RuntimeError("PPO controller requested but model was not loaded.")
        raw_env = build_env(config, job.year, job.day, job.seed, uncertainty_scale)
        rows = run_ppo_trace(model, vecnorm_path, raw_env, job.year, job.day, job.seed, job.max_steps)
    else:
        rows = run_llm_trace(
            config=config,
            api_key=api_key,
            year=job.year,
            day=job.day,
            seed=job.seed,
            max_steps=job.max_steps,
            llm_model=llm_model,
            llm_interval=llm_interval,
            llm_max_iterations=llm_max_iterations,
            llm_max_tokens=llm_max_tokens,
            expert_rollout=False,
            expert_policy_path=AgentConfig.expert_policy_path,
            expert_max_distance=AgentConfig.expert_candidate_max_distance,
            humidity_memory_enabled=(job.controller == "llm_hem"),
            humidity_memory_path=humidity_memory_path,
            humidity_memory_max_distance=AgentConfig.humidity_memory_max_distance,
            humidity_memory_min_trust=AgentConfig.humidity_memory_min_trust,
            humidity_memory_teacher_policy_id=humidity_memory_teacher_policy_id,
            humidity_memory_baseline_controller_id=humidity_memory_baseline_controller_id,
            humidity_memory_version=humidity_memory_version,
            mc_sero_mode="shadow" if job.controller == "llm_sero_shadow" else "off",
            tomato_safety_v2_enabled=(job.controller == "llm_rspc_v2"),
            plan_cache_mode=plan_cache_mode,
            plan_cache_path=plan_cache_path,
            plan_cache_strict=plan_cache_strict,
            plan_cache_key_policy=plan_cache_key_policy,
            uncertainty_scale=uncertainty_scale,
        )
    return summarize_trace(job, rows, time.perf_counter() - start), rows


def _trace_json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_trace_outputs(trace_dir: str, job: BenchmarkJob, rows: List[Dict[str, Any]]) -> None:
    if not trace_dir:
        return
    trace_path = Path(trace_dir)
    trace_path.mkdir(parents=True, exist_ok=True)
    stem = f"{job.scenario_id}_{job.controller}"
    write_rows_csv(trace_path / f"{stem}.csv", rows)
    jsonl_path = trace_path / f"{stem}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_trace_json_default) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen LLM-plan benchmark matrix.")
    parser.add_argument("--config", type=str, default="gl_gym/configs/envs/TomatoEnv.yml")
    parser.add_argument("--model-path", type=str, default="train_data/AgriControl/ppo/deterministic/models/None/best_model.zip")
    parser.add_argument("--vecnorm-path", type=str, default="train_data/AgriControl/ppo/deterministic/envs/None/best_vecnormalize.pkl")
    parser.add_argument("--years", type=str, default="2010,2015,2020")
    parser.add_argument("--days", type=str, default="59,120,180,240")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--controllers", type=str, default="llm,llm_hem,ppo")
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--uncertainty-scale", type=float, default=0.0)
    parser.add_argument("--llm-model", type=str, default="qwen-max-latest")
    parser.add_argument("--llm-interval", type=int, default=12)
    parser.add_argument("--llm-max-iterations", type=int, default=1)
    parser.add_argument("--llm-max-tokens", type=int, default=260)
    parser.add_argument("--plan-cache-mode", type=str, choices=["off", "record", "replay", "refresh"], default="replay")
    parser.add_argument("--plan-cache-path", type=str, default=AgentConfig.plan_cache_path)
    parser.add_argument("--plan-cache-strict", action="store_true")
    parser.add_argument("--plan-cache-key-policy", type=str, choices=["prompt", "scenario_timestep"], default="scenario_timestep")
    parser.add_argument("--humidity-memory-path", type=str, default=AgentConfig.humidity_memory_path)
    parser.add_argument("--humidity-memory-teacher-policy-id", type=str, default="")
    parser.add_argument("--humidity-memory-baseline-controller-id", type=str, default="")
    parser.add_argument("--humidity-memory-version", type=str, default=AgentConfig.humidity_memory_version)
    parser.add_argument("--output-json", type=str, default="gl_gym/result/benchmarks/frozen_benchmark_summary.json")
    parser.add_argument("--output-trace-dir", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    controllers = parse_controller_list(args.controllers)
    jobs = build_jobs(
        years=parse_int_list(args.years),
        days=parse_int_list(args.days),
        seeds=parse_int_list(args.seeds),
        controllers=controllers,
        max_steps=args.max_steps,
    )
    if args.dry_run:
        print(json.dumps({"jobs": [asdict(job) for job in jobs], "job_count": len(jobs)}, ensure_ascii=False, indent=2))
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = PPO.load(args.model_path) if "ppo" in controllers else None
    api_key = os.getenv("BAILIAN_API_KEY") or ("plan-cache-replay" if args.plan_cache_mode == "replay" else "")
    if any(c.startswith("llm") for c in controllers) and not api_key:
        raise RuntimeError("BAILIAN_API_KEY is required unless --plan-cache-mode replay is used.")

    summaries: List[Dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {job.controller} {job.scenario_id}")
        summary, rows = run_job(
            job,
            config=config,
            model=model,
            vecnorm_path=Path(args.vecnorm_path),
            api_key=api_key,
            uncertainty_scale=args.uncertainty_scale,
            llm_model=args.llm_model,
            llm_interval=args.llm_interval,
            llm_max_iterations=args.llm_max_iterations,
            llm_max_tokens=args.llm_max_tokens,
            plan_cache_mode=args.plan_cache_mode,
            plan_cache_path=args.plan_cache_path,
            plan_cache_strict=args.plan_cache_strict,
            plan_cache_key_policy=args.plan_cache_key_policy,
            humidity_memory_path=args.humidity_memory_path,
            humidity_memory_teacher_policy_id=args.humidity_memory_teacher_policy_id,
            humidity_memory_baseline_controller_id=args.humidity_memory_baseline_controller_id,
            humidity_memory_version=args.humidity_memory_version,
        )
        write_trace_outputs(args.output_trace_dir, job, rows)
        summaries.append(summary)

    output = {
        "benchmark": {
            "years": parse_int_list(args.years),
            "days": parse_int_list(args.days),
            "seeds": parse_int_list(args.seeds),
            "controllers": controllers,
            "max_steps": int(args.max_steps),
            "plan_cache_mode": str(args.plan_cache_mode),
            "plan_cache_path": str(args.plan_cache_path),
            "plan_cache_key_policy": str(args.plan_cache_key_policy),
            "output_trace_dir": str(args.output_trace_dir),
        },
        "summaries": summaries,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved frozen benchmark summary to {output_path}")


if __name__ == "__main__":
    main()
