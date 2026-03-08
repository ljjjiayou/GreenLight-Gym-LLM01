import os
import sys
import json
import argparse
import yaml
import dotenv
from statistics import mean, pstdev

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


def run_one_case(config, api_key: str, year: int, day: int, max_steps: int = 10):
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

    interface = GreenhouseAgentInterface(env)
    if hasattr(create_langchain_tools, "instance"):
        delattr(create_langchain_tools, "instance")
    tools = create_langchain_tools(interface)

    agent_config = AgentConfig(
        model_name="qwen-plus",
        verbose=False,
        max_iterations=10,
        api_key=api_key,
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
        "steps": results["steps"],
        "total_reward": float(results["total_reward"]),
        "avg_reward": float(results["avg_reward"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=str, default="2010,2020")
    parser.add_argument("--days", type=str, default="59,180,240")
    parser.add_argument("--max-steps", type=int, default=10)
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

    all_results = []
    skipped = []
    for year in years:
        for day in days:
            try:
                case = run_one_case(config, api_key, year, day, max_steps=max_steps)
                all_results.append(case)
                print(f"[Case] year={year}, day={day}, total_reward={case['total_reward']:.4f}, steps={case['steps']}")
            except FileNotFoundError as e:
                skipped.append({"year": year, "day": day, "reason": str(e)})
                print(f"[Skip] year={year}, day={day}, reason={e}")

    rewards = [x["total_reward"] for x in all_results]
    summary = {
        "cases": len(all_results),
        "years": years,
        "days": days,
        "max_steps": max_steps,
        "mean_total_reward": float(mean(rewards)) if rewards else 0.0,
        "std_total_reward": float(pstdev(rewards)) if len(rewards) > 1 else 0.0,
        "min_total_reward": float(min(rewards)) if rewards else 0.0,
        "max_total_reward": float(max(rewards)) if rewards else 0.0,
        "results": all_results,
        "skipped": skipped,
    }

    print("\n=== Generalization Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
