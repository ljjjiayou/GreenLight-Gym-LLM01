import os
import sys
import json
import argparse
import yaml
import dotenv
from statistics import mean, stdev

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector
from gl_gym.agent.interface import GreenhouseAgentInterface
from gl_gym.agent.tools import create_langchain_tools
from gl_gym.agent.control_loop import GreenhouseControlLoop, ControlLoopConfig


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_one_case(
    config,
    api_key: str,
    year: int,
    day: int,
    max_steps: int = 10,
    interval: int = 12,
    model_name: str = AgentConfig.model_name,
    seed: int | None = None,
):
    base_env_params = dict(config["GreenLightEnv"])
    base_env_params["training"] = False
    tomato_cfg = dict(config["TomatoEnv"])
    eval_options = dict(tomato_cfg["eval_options"])
    eval_options["eval_years"] = [year]
    eval_options["eval_days"] = [day]
    tomato_cfg["eval_options"] = eval_options

    env = TomatoEnv(
        reward_function=tomato_cfg["reward_function"],
        observation_modules=tomato_cfg["observation_modules"],
        constraints=tomato_cfg["constraints"],
        eval_options=tomato_cfg["eval_options"],
        reward_params=tomato_cfg["reward_params"],
        base_env_params=base_env_params,
        uncertainty_scale=tomato_cfg.get("uncertainty_scale", 0.0),
    )
    if seed is not None:
        env.set_seed(seed)

    interface = GreenhouseAgentInterface(env)
    if hasattr(create_langchain_tools, "instance"):
        delattr(create_langchain_tools, "instance")
    tools = create_langchain_tools(interface)

    agent_config = AgentConfig(
        model_name=model_name,
        verbose=False,
        max_iterations=2,
        max_tokens=260,
        api_key=api_key,
        control_interval=interval
    )
    agent = RuleBasedLLMDirector(
        agent_interface=interface,
        tools=tools,
        config=agent_config,
    )

    loop_config = ControlLoopConfig(max_steps=max_steps, log_freq=1, verbose=False)
    control_loop = GreenhouseControlLoop(agent=agent, agent_interface=interface, config=loop_config)
    results = control_loop.run()

    return {
        "year": year,
        "day": day,
        "seed": seed,
        "steps": results["steps"],
        "total_reward": float(results["total_reward"]),
        "avg_reward": float(results["avg_reward"]),
        "elapsed_time": float(results["elapsed_time"]),
        "interval": interval
    }


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def summarize_by_interval(results: list[dict], intervals: list[int]) -> list[dict]:
    summaries = []
    for interval in intervals:
        rows = [x for x in results if x["interval"] == interval]
        rewards = [x["total_reward"] for x in rows]
        times = [x["elapsed_time"] for x in rows]
        summaries.append({
            "interval": interval,
            "cases": len(rows),
            "mean_total_reward": float(mean(rewards)) if rewards else 0.0,
            "std_total_reward": float(stdev(rewards)) if len(rewards) > 1 else 0.0,
            "mean_elapsed_time": float(mean(times)) if times else 0.0,
            "std_elapsed_time": float(stdev(times)) if len(times) > 1 else 0.0,
        })
    return summaries


def paired_interval_analysis(results: list[dict], interval_a: int, interval_b: int) -> dict:
    table = {}
    for row in results:
        key = (row["year"], row["day"], row["repeat"])
        if key not in table:
            table[key] = {}
        table[key][row["interval"]] = row
    deltas = []
    wins_b = 0
    wins_a = 0
    ties = 0
    for _, pair in table.items():
        if interval_a in pair and interval_b in pair:
            da = pair[interval_a]["total_reward"]
            db = pair[interval_b]["total_reward"]
            diff = db - da
            deltas.append(diff)
            if diff > 1e-9:
                wins_b += 1
            elif diff < -1e-9:
                wins_a += 1
            else:
                ties += 1
    n = len(deltas)
    mean_delta = float(mean(deltas)) if n else 0.0
    std_delta = float(stdev(deltas)) if n > 1 else 0.0
    ci95_half = float(1.96 * (std_delta / (n ** 0.5))) if n > 1 else 0.0
    return {
        "base_interval": interval_a,
        "compare_interval": interval_b,
        "paired_count": n,
        "mean_reward_delta_compare_minus_base": mean_delta,
        "std_reward_delta": std_delta,
        "ci95_delta": [mean_delta - ci95_half, mean_delta + ci95_half],
        "wins_compare": wins_b,
        "wins_base": wins_a,
        "ties": ties
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=str, default="2010,2020")
    parser.add_argument("--days", type=str, default="59,180,240")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--interval", type=int, default=12, help="LLM control interval")
    parser.add_argument("--intervals", type=str, default="", help="comma-separated intervals for batch eval")
    parser.add_argument("--model-name", type=str, default=AgentConfig.model_name)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=666)
    parser.add_argument("--save-json", type=str, default="")
    args = parser.parse_args()

    dotenv.load_dotenv()
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        raise RuntimeError("未找到 BAILIAN_API_KEY")

    config_path = os.path.join(project_root, "gl_gym", "configs", "envs", "TomatoEnv.yml")
    config = load_config(config_path)

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    days = [int(x.strip()) for x in args.days.split(",") if x.strip()]
    max_steps = int(args.max_steps)
    intervals = parse_int_list(args.intervals) if args.intervals else [int(args.interval)]
    model_name = args.model_name
    repeats = int(args.repeats)
    base_seed = int(args.base_seed)

    all_results = []
    skipped = []
    for repeat in range(repeats):
        for interval in intervals:
            for year in years:
                for day in days:
                    seed = base_seed + repeat
                    try:
                        case = run_one_case(
                            config,
                            api_key,
                            year,
                            day,
                            max_steps=max_steps,
                            interval=interval,
                            model_name=model_name,
                            seed=seed,
                        )
                        case["repeat"] = repeat
                        all_results.append(case)
                        print(
                            f"[Case] rep={repeat}, seed={seed}, year={year}, day={day}, "
                            f"interval={interval}, model={model_name}, "
                            f"total_reward={case['total_reward']:.4f}, time={case['elapsed_time']:.2f}s"
                        )
                    except FileNotFoundError as e:
                        skipped.append({"repeat": repeat, "year": year, "day": day, "interval": interval, "reason": str(e)})
                        print(f"[Skip] rep={repeat}, year={year}, day={day}, interval={interval}, reason={e}")

    rewards = [x["total_reward"] for x in all_results]
    times = [x["elapsed_time"] for x in all_results]
    interval_summaries = summarize_by_interval(all_results, intervals)
    paired_analysis = None
    if len(intervals) == 2:
        paired_analysis = paired_interval_analysis(all_results, intervals[0], intervals[1])
    summary = {
        "cases": len(all_results),
        "years": years,
        "days": days,
        "max_steps": max_steps,
        "intervals": intervals,
        "model_name": model_name,
        "repeats": repeats,
        "base_seed": base_seed,
        "mean_total_reward": float(mean(rewards)) if rewards else 0.0,
        "mean_elapsed_time": float(mean(times)) if times else 0.0,
        "interval_summaries": interval_summaries,
        "paired_analysis": paired_analysis,
        "results": all_results,
        "skipped": skipped,
    }

    print("\n=== Generalization Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
