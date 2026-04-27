
import argparse
import json
import time
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import yaml
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.agent.llm_agent import create_langchain_tools, AgentConfig, RuleBasedLLMDirector
from gl_gym.agent.interface import GreenhouseAgentInterface
from gl_gym.environments.baseline import RuleBasedController as GreenhouseRuleController

# Load env vars
load_dotenv()

def build_tomato_env(base_params, specific_params, year, day, seed, uncertainty_scale=0.0):
    base = base_params.copy()
    specific = specific_params.copy()
    eval_opts = specific.get("eval_options", {}).copy()
    eval_opts["eval_years"] = [year]
    eval_opts["eval_days"] = [day]
    specific["eval_options"] = eval_opts
    
    env = TomatoEnv(
        reward_function=specific["reward_function"],
        observation_modules=specific["observation_modules"],
        constraints=specific["constraints"],
        eval_options=specific["eval_options"],
        reward_params=specific["reward_params"],
        base_env_params=base,
        uncertainty_scale=uncertainty_scale
    )
    env.reset(seed=seed)
    return env

def run_agent_detailed(agent_type, agent, vecnorm_path, env, max_steps):
    """Runs agent and collects detailed time-series data."""
    logs = {
        "step": [], "temp_air": [], "rh_air": [], "co2_air": [], 
        "u_heat": [], "u_vent": [], "u_co2": [], "u_screen": [], "u_lamp": [],
        "reward": [], "profit": []
    }
    
    # Setup PPO env wrapper if needed
    if agent_type == "ppo":
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        obs = env.reset()
    else:
        # LLM or Rule
        obs, _ = env.reset()
        
    for step in range(max_steps):
        if agent_type == "ppo":
            action, _ = agent.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            real_env = env.envs[0]
            info = infos[0]
            reward = rewards[0]
            done = dones[0]
        elif agent_type == "llm":
            try:
                result = agent.step_with_rules()
                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", False))
                real_env = env
                info = env._get_info()
            except Exception as e:
                print(f"LLM Error: {e}")
                break
        
        # Log data
        logs["step"].append(step)
        logs["temp_air"].append(real_env.x[0]) # Assuming x[0] is temp
        logs["rh_air"].append(real_env.x[2])   # Assuming x[2] is RH (need check)
        logs["co2_air"].append(real_env.x[1])  # Assuming x[1] is CO2
        
        # Controls
        logs["u_heat"].append(real_env.u[0])
        logs["u_co2"].append(real_env.u[1])
        logs["u_screen"].append(real_env.u[2])
        logs["u_vent"].append(real_env.u[3])
        logs["u_lamp"].append(real_env.u[4])
        
        logs["reward"].append(reward)
        logs["profit"].append(info.get("EPI", 0.0))
        
        if done:
            break
            
    return logs

def plot_comparison(results, days, output_dir):
    """Generates comparison plots for each day."""
    metrics = ["temp_air", "u_heat", "u_vent", "u_co2"]
    titles = ["Air Temperature (°C)", "Heating Control", "Ventilation Control", "CO2 Control"]
    
    for day in days:
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Agent Comparison - Day {day} (Winter/Spring/Autumn)", fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            for agent_name, day_data in results.items():
                if day in day_data:
                    data = day_data[day]
                    ax.plot(data["step"], data[metric], label=agent_name, alpha=0.8)
            
            ax.set_ylabel(titles[i])
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            
            # Add threshold lines for temp
            if metric == "temp_air":
                ax.axhline(y=12.0, color='r', linestyle='--', alpha=0.5, label="Min Limit (12°C)")
                
        plt.xlabel("Steps (15 min intervals)")
        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_day_{day}.png")
        plt.close()

def main():
    # Config
    year = 2010
    days = [1, 120, 240]
    max_steps = 96 * 3 # Run for 3 days to see patterns
    output_dir = Path("gl_gym/result/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load env config
    with open("gl_gym/configs/envs/TomatoEnv.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    env_base = config["GreenLightEnv"]
    env_specific = config["TomatoEnv"]
    
    # Load PPO
    model_path = Path("train_data/AgriControl/ppo/deterministic/models/None/best_model.zip")
    vecnorm_path = Path("train_data/AgriControl/ppo/deterministic/envs/None/best_vecnormalize.pkl")
    ppo_model = PPO.load(model_path)
    
    # Load LLM Agent
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        print("API Key missing")
        return
        
    env_llm = build_tomato_env(env_base, env_specific, year, days[0], 42)
    interface = GreenhouseAgentInterface(env_llm)
    tools = create_langchain_tools(interface)
    agent_cfg = AgentConfig(
        model_name="qwen-max-latest",
        api_key=api_key,
        verbose=True,
        control_interval=12,
        max_iterations=1,
        max_tokens=260
    )
    llm_agent = RuleBasedLLMDirector(interface, tools, agent_cfg)
    
    results = {"PPO": {}, "LLM": {}}
    
    for day in days:
        print(f"Generating data for Day {day}...")
        
        # Run PPO
        env_ppo = build_tomato_env(env_base, env_specific, year, day, 42)
        results["PPO"][day] = run_agent_detailed("ppo", ppo_model, vecnorm_path, env_ppo, max_steps)
        
        # Run LLM
        # Re-init env and agent to clear state
        env_llm = build_tomato_env(env_base, env_specific, year, day, 42)
        llm_agent.interface.env = env_llm
        # We need to re-bind tools to new env interface? 
        # Actually create_langchain_tools uses singleton, need to update it
        # But simpler: just update interface.env reference
        results["LLM"][day] = run_agent_detailed("llm", llm_agent, None, env_llm, max_steps)
        
    print("Plotting results...")
    plot_comparison(results, days, output_dir)
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
