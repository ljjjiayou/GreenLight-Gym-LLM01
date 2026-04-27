import os
import time
import json
import yaml
import dotenv
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gl_gym.environments.tomato_env import TomatoEnv
import gl_gym.agent.llm_agent as llm_mod
from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector
from gl_gym.agent.interface import GreenhouseAgentInterface
from gl_gym.agent.tools import create_langchain_tools

# Reduce logging overhead during profiling.
llm_mod.print = lambda *args, **kwargs: None
llm_mod.log_control_tracking = lambda *args, **kwargs: None

dotenv.load_dotenv()
api_key = os.getenv("BAILIAN_API_KEY")
if not api_key:
    raise SystemExit("BAILIAN_API_KEY missing")

with open("gl_gym/configs/envs/TomatoEnv.yml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

base_env_params = dict(cfg["GreenLightEnv"])
base_env_params["training"] = False
spec = dict(cfg["TomatoEnv"])
eval_options = dict(spec["eval_options"])
eval_options["eval_years"] = [2020]
eval_options["eval_days"] = [240]
spec["eval_options"] = eval_options


def build_env(seed: int):
    env = TomatoEnv(
        reward_function=spec["reward_function"],
        observation_modules=spec["observation_modules"],
        constraints=spec["constraints"],
        eval_options=spec["eval_options"],
        reward_params=spec["reward_params"],
        base_env_params=base_env_params,
        uncertainty_scale=spec.get("uncertainty_scale", 0.0),
    )
    env.reset(seed=seed)
    return env


def run_case(name: str, interval: int, max_tokens: int, max_steps: int = 60, seed: int = 42):
    env = build_env(seed)
    interface = GreenhouseAgentInterface(env)
    if hasattr(create_langchain_tools, "instance"):
        delattr(create_langchain_tools, "instance")
    tools = create_langchain_tools(interface)

    cfg = AgentConfig(
        model_name="qwen-max-latest",
        api_key=api_key,
        verbose=False,
        max_iterations=1,
        max_tokens=max_tokens,
        control_interval=interval,
    )
    agent = RuleBasedLLMDirector(
        agent_interface=interface,
        tools=tools,
        config=cfg,
    )

    t0 = time.perf_counter()
    replan = 0
    rollout = 0
    steps = 0
    total_reward = 0.0
    for _ in range(max_steps):
        r = agent.step_with_rules()
        act = r.get("action", "")
        if act == "llm_replan":
            replan += 1
        elif act == "plan_rollout":
            rollout += 1
        total_reward += float(r.get("reward", 0.0))
        steps += 1
        if bool(r.get("done", False)):
            break

    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "steps": steps,
        "interval": interval,
        "max_tokens": max_tokens,
        "llm_replan_steps": replan,
        "plan_rollout_steps": rollout,
        "elapsed_s": elapsed,
        "avg_step_s": elapsed / max(1, steps),
        "proxy_max_output_tokens": replan * max_tokens,
        "total_reward": total_reward,
    }


if __name__ == "__main__":
    cases = [
        ("baseline_i12_t260", 12, 260),
        ("fast_i60_t128", 60, 128),
    ]
    outs = []
    for name, interval, tok in cases:
        outs.append(run_case(name, interval, tok, max_steps=60, seed=42))

    print(json.dumps({"profile": outs}, ensure_ascii=False, indent=2))
