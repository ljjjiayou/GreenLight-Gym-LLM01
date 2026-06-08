import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import dotenv
import yaml

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
experiments_dir = os.path.dirname(current_file_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if experiments_dir not in sys.path:
    sys.path.insert(0, experiments_dir)
os.chdir(project_root)

from gl_gym.common.utils import load_env_params, load_model_hyperparams
from gl_gym.agent.llm_agent import AgentConfig
from compare_ppo_rule_based import build_markdown_report, parse_int_list, run_llm_case, summarize


def find_algo_summary(summary: dict, algo: str) -> dict:
    for item in summary.get("summaries", []):
        if item.get("algo") == algo:
            return item
    return {}


def passes_constraints(candidate: dict, baseline: dict) -> bool:
    return (
        float(candidate.get("sum_total_reward", 0.0)) >= float(baseline.get("sum_total_reward", 0.0))
        and float(candidate.get("sum_total_profit", -1e9)) >= float(baseline.get("sum_total_profit", -1e9))
        and float(candidate.get("sum_violation_temp", 1e9)) <= float(baseline.get("sum_violation_temp", 1e9))
        and float(candidate.get("sum_violation_co2", 1e9)) <= float(baseline.get("sum_violation_co2", 1e9))
        and float(candidate.get("sum_violation_rh", 1e9)) <= float(baseline.get("sum_violation_rh", 1e9))
    )


def _clip_if_needed(key: str, value: float) -> float:
    if "rh" in key.lower():
        return max(0.0, min(100.0, value))
    return value


def build_rule_search_space(base_rule_params: dict) -> dict[str, list]:
    # Focus on high-impact control strengths: heating / ventilation / humidity / CO2.
    defaults = {
        "temp_setpoint_day": [-2.0, 0.0, 2.0],
        "temp_setpoint_night": [-2.0, 0.0, 2.0],
        "heat_correction": [-0.5, 0.0, 0.5],
        "heat_deadzone": [-1.0, 0.0, 1.0],
        "co2_day": [-150.0, 0.0, 150.0],
        "vent_heat_Pband": [-1.5, 0.0, 1.5],
        "rh_max": [-5.0, 0.0, 5.0],
        "mech_dehumid_Pband": [-1.0, 0.0, 1.0],
        "vent_rh_Pband": [-1.5, 0.0, 1.5],
        "t_vent_off": [-1.0, 0.0, 1.0],
        "vent_cold_Pband": [-1.0, 0.0, 1.0],
        "thScrSpDay": [-2.0, 0.0, 2.0],
        "thScrSpNight": [-2.0, 0.0, 2.0],
        "thScrPband": [-2.0, 0.0, 2.0],
        "thScrDeadZone": [-1.5, 0.0, 1.5],
        "thScrRh": [-2.0, 0.0, 2.0],
        "thScrRhPband": [-1.0, 0.0, 1.0],
        "lampExtraHeat": [-1.0, 0.0, 1.0],
        "blScrExtraRh": [-20.0, 0.0, 20.0],
        "rhMax": [-5.0, 0.0, 5.0],
        "tHeatBand": [-1.0, 0.0, 1.0],
        "co2Band": [-100.0, 0.0, 100.0],
        "lamps_off_sun": [-120.0, 0.0, 120.0],
        "lamp_rad_sum_limit": [-4.0, 0.0, 4.0],
    }

    space: dict[str, list] = {}
    for key, deltas in defaults.items():
        if key not in base_rule_params:
            continue
        base = float(base_rule_params[key])
        vals = sorted({_clip_if_needed(key, base + d) for d in deltas})
        if vals:
            space[key] = vals
    return space


def sample_rule_params(base_rule_params: dict, rule_space: dict[str, list], rng: random.Random) -> dict:
    out = dict(base_rule_params)
    for key, values in rule_space.items():
        out[key] = rng.choice(values)
    return out


def build_llm_candidates(interval_grid: list[int], tokens_grid: list[int], iter_grid: list[int]) -> list[dict]:
    out = []
    for interval in interval_grid:
        for max_tokens in tokens_grid:
            for max_iter in iter_grid:
                out.append(
                    {
                        "llm_interval": interval,
                        "llm_max_tokens": max_tokens,
                        "llm_max_iterations": max_iter,
                    }
                )
    return out


def evaluate_trial(
    idx: int,
    trial_settings: dict,
    args,
    years: list[int],
    days: list[int],
    env_cfg: dict,
    api_key: str,
    baseline_llm: dict,
    baseline_ppo: dict,
    baseline_rb: dict,
    reports_dir: Path,
) -> dict:
    rows = []
    t0 = time.perf_counter()
    for repeat in range(args.repeats):
        seed = args.base_seed + repeat
        for year in years:
            for day in days:
                try:
                    llm_result = run_llm_case(
                        env_cfg=env_cfg,
                        api_key=api_key,
                        year=year,
                        day=day,
                        seed=seed,
                        max_steps=args.max_steps,
                        llm_model=args.llm_model,
                        llm_interval=int(trial_settings["llm_interval"]),
                        llm_max_iterations=int(trial_settings["llm_max_iterations"]),
                        llm_max_tokens=int(trial_settings["llm_max_tokens"]),
                        rule_params=trial_settings["rule_params"],
                    )
                except BaseException as exc:
                    # Keep search progressing even when remote API/network is unstable.
                    llm_result = {
                        "steps": 0,
                        "total_reward": -10000.0,
                        "mean_reward": -10000.0,
                        "runtime_error": f"trial_failed: {type(exc).__name__}: {exc}",
                        "total_profit": -10000.0,
                        "total_revenue": 0.0,
                        "total_heat_cost": 0.0,
                        "total_co2_cost": 0.0,
                        "total_elec_cost": 0.0,
                        "total_fixed_cost": 0.0,
                        "violation_temp": 1e6,
                        "violation_co2": 1e6,
                        "violation_rh": 1e6,
                        "violation_lamp": 1e6,
                    }
                rows.append(
                    {
                        "algo": "llm_director",
                        "year": year,
                        "day": day,
                        "seed": seed,
                        "repeat": repeat,
                        "elapsed_time": 0.0,
                        **llm_result,
                    }
                )

    elapsed = time.perf_counter() - t0
    llm_summary = summarize(rows, "llm_director")
    llm_summary["sum_elapsed_time"] = float(elapsed)
    llm_summary["mean_elapsed_time"] = float(elapsed / max(len(rows), 1))
    llm_summary["std_elapsed_time"] = 0.0

    trial = {
        "trial_index": idx,
        "llm_interval": int(trial_settings["llm_interval"]),
        "llm_max_tokens": int(trial_settings["llm_max_tokens"]),
        "llm_max_iterations": int(trial_settings["llm_max_iterations"]),
        "rule_params": trial_settings["rule_params"],
        "passed_constraints": passes_constraints(llm_summary, baseline_llm),
        "summary_llm": llm_summary,
    }

    report_json_path = reports_dir / f"trial_{idx:03d}.json"
    report_md_path = reports_dir / f"trial_{idx:03d}.md"
    report_payload = {
        "project": args.project,
        "years": years,
        "days": days,
        "repeats": args.repeats,
        "base_seed": args.base_seed,
        "max_steps": args.max_steps,
        "uncertainty_scale": args.uncertainty_scale,
        "llm_model": args.llm_model,
        "llm_interval": trial["llm_interval"],
        "llm_max_iterations": trial["llm_max_iterations"],
        "llm_max_tokens": trial["llm_max_tokens"],
        "rule_params": trial["rule_params"],
        "summaries": [baseline_ppo, baseline_rb, llm_summary],
        "pairwise_analysis": [],
        "results": rows,
    }
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown_report(report_payload, report_json_path))

    return trial


def score_trial(trial: dict) -> tuple:
    s = trial["summary_llm"]
    return (
        float(s.get("sum_total_profit", -1e12)),
        float(s.get("sum_total_reward", -1e12)),
        -float(s.get("sum_violation_rh", 1e12)),
        -float(s.get("sum_elapsed_time", 1e12)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl")
    parser.add_argument("--years", type=str, default="2010,2015,2020")
    parser.add_argument("--days", type=str, default="59,180,240")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=666)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--uncertainty-scale", type=float, default=0.0)
    parser.add_argument("--llm-model", type=str, default=AgentConfig.model_name)
    parser.add_argument("--interval-grid", type=str, default="7,8,9")
    parser.add_argument("--max-tokens-grid", type=str, default="450,500,550")
    parser.add_argument("--max-iterations-grid", type=str, default="3,4,6")

    # Efficient search controls
    parser.add_argument("--search-trials", type=int, default=36)
    parser.add_argument("--refine-trials", type=int, default=18)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-seed", type=int, default=20260324)

    parser.add_argument("--baseline-json", type=str, default="gl_gym/result/ppo_multi_season_metrics.json")
    parser.add_argument("--output-json", type=str, default="gl_gym/result/pareto_search_llm_rule.json")
    parser.add_argument("--output-md", type=str, default="gl_gym/result/pareto_search_llm_rule.md")
    parser.add_argument("--reports-dir", type=str, default="gl_gym/result/pareto_reports_rule")
    args = parser.parse_args()

    dotenv.load_dotenv()
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 BAILIAN_API_KEY")

    years = parse_int_list(args.years)
    days = parse_int_list(args.days)
    interval_grid = parse_int_list(args.interval_grid)
    tokens_grid = parse_int_list(args.max_tokens_grid)
    iter_grid = parse_int_list(args.max_iterations_grid)
    rng = random.Random(args.search_seed)

    baseline_path = Path(args.baseline_json)
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_summary = json.load(f)

    baseline_llm = find_algo_summary(baseline_summary, "llm_director")
    baseline_ppo = find_algo_summary(baseline_summary, "ppo")
    baseline_rb = find_algo_summary(baseline_summary, "rule_based")
    if not baseline_llm:
        raise RuntimeError("baseline json 中未找到 llm_director 汇总")

    load_env_params("TomatoEnv", "gl_gym/configs/envs/")
    base_rule_params = load_model_hyperparams("rule_based", "TomatoEnv")
    rule_space = build_rule_search_space(base_rule_params)
    llm_candidates = build_llm_candidates(interval_grid, tokens_grid, iter_grid)

    if not llm_candidates:
        raise RuntimeError("LLM 参数网格为空，请检查 interval/tokens/iterations")

    with open("gl_gym/configs/envs/TomatoEnv.yml", "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)
    env_cfg["TomatoEnv"]["uncertainty_scale"] = args.uncertainty_scale

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    trials = []

    # Stage 1: coarse random search
    for idx in range(1, args.search_trials + 1):
        llm_pick = rng.choice(llm_candidates)
        trial_settings = dict(llm_pick)
        trial_settings["rule_params"] = sample_rule_params(base_rule_params, rule_space, rng)

        trial = evaluate_trial(
            idx=idx,
            trial_settings=trial_settings,
            args=args,
            years=years,
            days=days,
            env_cfg=env_cfg,
            api_key=api_key,
            baseline_llm=baseline_llm,
            baseline_ppo=baseline_ppo,
            baseline_rb=baseline_rb,
            reports_dir=reports_dir,
        )
        trials.append(trial)
        s = trial["summary_llm"]
        print(
            f"[coarse {idx}/{args.search_trials}] pass={trial['passed_constraints']}, "
            f"interval={trial['llm_interval']}, tok={trial['llm_max_tokens']}, it={trial['llm_max_iterations']}, "
            f"profit={float(s.get('sum_total_profit', 0.0)):.3f}, rh={float(s.get('sum_violation_rh', 0.0)):.3f}"
        )

    # Stage 2: refine near top candidates
    top_candidates = sorted(trials, key=score_trial, reverse=True)[: max(1, args.top_k)]
    next_index = len(trials) + 1
    for i in range(args.refine_trials):
        anchor = rng.choice(top_candidates)
        llm_pick = {
            "llm_interval": anchor["llm_interval"],
            "llm_max_tokens": anchor["llm_max_tokens"],
            "llm_max_iterations": anchor["llm_max_iterations"],
        }
        if rng.random() < 0.5:
            llm_pick = rng.choice(llm_candidates)

        mutated_rule = dict(anchor["rule_params"])
        for key, values in rule_space.items():
            if rng.random() < 0.7:
                mutated_rule[key] = rng.choice(values)

        trial_settings = dict(llm_pick)
        trial_settings["rule_params"] = mutated_rule

        trial = evaluate_trial(
            idx=next_index,
            trial_settings=trial_settings,
            args=args,
            years=years,
            days=days,
            env_cfg=env_cfg,
            api_key=api_key,
            baseline_llm=baseline_llm,
            baseline_ppo=baseline_ppo,
            baseline_rb=baseline_rb,
            reports_dir=reports_dir,
        )
        trials.append(trial)
        s = trial["summary_llm"]
        print(
            f"[refine {i + 1}/{args.refine_trials}] pass={trial['passed_constraints']}, "
            f"interval={trial['llm_interval']}, tok={trial['llm_max_tokens']}, it={trial['llm_max_iterations']}, "
            f"profit={float(s.get('sum_total_profit', 0.0)):.3f}, rh={float(s.get('sum_violation_rh', 0.0)):.3f}"
        )
        next_index += 1

    passed_trials = [x for x in trials if x["passed_constraints"]]
    ranked_trials = sorted(trials, key=score_trial, reverse=True)
    top = ranked_trials[0] if ranked_trials else None
    top_passed = sorted(passed_trials, key=score_trial, reverse=True)[0] if passed_trials else None

    output_payload = {
        "baseline_json": str(baseline_path),
        "baseline_llm": baseline_llm,
        "search_space": {
            "interval_grid": interval_grid,
            "max_tokens_grid": tokens_grid,
            "max_iterations_grid": iter_grid,
            "rule_space": rule_space,
        },
        "search_config": {
            "search_trials": args.search_trials,
            "refine_trials": args.refine_trials,
            "top_k": args.top_k,
            "search_seed": args.search_seed,
        },
        "trials": trials,
        "passed_count": len(passed_trials),
        "top_overall": top,
        "top_passed": top_passed,
    }

    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# LLM + Rule 高效参数搜索报告",
        "",
        f"- baseline: {baseline_path.as_posix()}",
        f"- 搜索 trial 数: {len(trials)}",
        f"- 满足约束 trial 数: {len(passed_trials)}",
        "",
        "## 约束",
        "",
        f"- 总奖励 >= {float(baseline_llm.get('sum_total_reward', 0.0)):.4f}",
        f"- 总利润 >= {float(baseline_llm.get('sum_total_profit', 0.0)):.4f}",
        f"- 温度违规 <= {float(baseline_llm.get('sum_violation_temp', 0.0)):.4f}",
        f"- CO2违规 <= {float(baseline_llm.get('sum_violation_co2', 0.0)):.4f}",
        f"- 湿度违规 <= {float(baseline_llm.get('sum_violation_rh', 0.0)):.4f}",
        "",
        "## Top Overall",
        "",
    ]

    if top:
        s = top["summary_llm"]
        lines.extend(
            [
                f"- interval={top['llm_interval']}, max_tokens={top['llm_max_tokens']}, max_iterations={top['llm_max_iterations']}",
                f"- rule_params={json.dumps(top.get('rule_params', {}), ensure_ascii=False)}",
                f"- 总奖励={float(s.get('sum_total_reward', 0.0)):.4f}, 总利润={float(s.get('sum_total_profit', 0.0)):.4f}, 湿度违规={float(s.get('sum_violation_rh', 0.0)):.4f}, 耗时={float(s.get('sum_elapsed_time', 0.0)):.4f}",
            ]
        )
    else:
        lines.append("- 无可用结果")

    if top_passed:
        ps = top_passed["summary_llm"]
        lines.extend(
            [
                "",
                "## Top Passed",
                "",
                f"- interval={top_passed['llm_interval']}, max_tokens={top_passed['llm_max_tokens']}, max_iterations={top_passed['llm_max_iterations']}",
                f"- rule_params={json.dumps(top_passed.get('rule_params', {}), ensure_ascii=False)}",
                f"- 总奖励={float(ps.get('sum_total_reward', 0.0)):.4f}, 总利润={float(ps.get('sum_total_profit', 0.0)):.4f}, 湿度违规={float(ps.get('sum_violation_rh', 0.0)):.4f}, 耗时={float(ps.get('sum_elapsed_time', 0.0)):.4f}",
            ]
        )

    output_md_path = Path(args.output_md)
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"saved search json: {output_json_path}")
    print(f"saved search report: {output_md_path}")


if __name__ == "__main__":
    main()
