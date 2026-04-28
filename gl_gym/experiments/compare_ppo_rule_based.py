import argparse
import json
import time
import os
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 设置 KMP_DUPLICATE_LIB_OK 以避免 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
from datetime import datetime
from statistics import mean, stdev

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from dotenv import load_dotenv

# 导入所需模块
from gl_gym.environments.tomato_env import TomatoEnv
# from gl_gym.envs.gl_env import GreenLightEnv
from gl_gym.agent.llm_agent import create_langchain_tools, AgentConfig, RuleBasedLLMDirector
from gl_gym.agent.interface import GreenhouseAgentInterface
from gl_gym.environments.baseline import RuleBasedController as GreenhouseRuleController
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# 加载环境变量
load_dotenv()

# --- 配置项 ---
METRIC_KEYS = [
    "total_reward", "mean_reward", "total_profit", "total_revenue",
    "total_heat_cost", "total_co2_cost", "total_elec_cost", "total_fixed_cost",
    "violation_temp", "violation_co2", "violation_rh", "violation_lamp"
]

# 全局安全阈值（可通过命令行参数覆盖）。
SAFETY_THRESHOLDS = {
    "temp": 100.0,
    "co2": 10000.0,
    "rh": 30.0,
}


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


class RuleBasedController:
    """用于对比的模拟规则控制器。"""
    def __init__(self, params):
        self.params = params

    def predict(self, obs, deterministic=True):
        # 规则逻辑占位实现
        # 当前返回一个哑动作（例如全零或保持）
        # 实际场景中这里应实现真正的规则控制逻辑。
        return np.zeros(6, dtype=np.float32), None 
    
    # 这里需要提供与 run_rule_case 期望签名一致的 predict 方法
    # 旧代码调用的是 agent.rule_controller.predict(env.x, weather, env)
    # run_rule_case 看起来假设可调用 predict(obs)
    # 因此可以在 run_rule_case 内部适配具体调用逻辑。
    
    # 进一步看 run_rule_case，实际是 rule_agent.predict(obs, deterministic=True)
    # 但 llm_agent.py 中提供的是 GreenhouseRuleController
    # 优先使用该实现，或在此处兼容其调用方式。
    pass

def build_tomato_env(base_params, specific_params, year, day, seed, uncertainty_scale=0.0):
    """创建并配置 TomatoEnv。"""
    # 浅拷贝字典，避免污染传入参数
    base = base_params.copy()
    specific = specific_params.copy()
    
    # 设置评估选项
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

def init_metric_totals():
    return {k: 0.0 for k in METRIC_KEYS if k not in ["total_reward", "mean_reward"]}

def add_info_metrics(totals, info):
    """从 info 字典累计各项指标。"""
    # info 字段到指标字段的映射
    # 字段来源于 TomatoEnv._get_info()
    # "EPI": profit
    # "revenue": revenue
    # "heat_cost": heat_cost
    # "co2_cost": co2_cost
    # "elec_cost": elec_cost
    # "fixed_costs": fixed_cost
    # "temp_violation": violation_temp
    # "co2_violation": violation_co2
    # "rh_violation": violation_rh
    # "lamp_violation": violation_lamp
    
    totals["total_profit"] += info.get("EPI", 0.0)
    totals["total_revenue"] += info.get("revenue", 0.0)
    totals["total_heat_cost"] += info.get("heat_cost", 0.0)
    totals["total_co2_cost"] += info.get("co2_cost", 0.0)
    totals["total_elec_cost"] += info.get("elec_cost", 0.0)
    totals["total_fixed_cost"] += info.get("fixed_costs", 0.0)
    totals["violation_temp"] += info.get("temp_violation", 0.0)
    totals["violation_co2"] += info.get("co2_violation", 0.0)
    totals["violation_rh"] += info.get("rh_violation", 0.0)
    totals["violation_lamp"] += info.get("lamp_violation", 0.0)

def run_ppo_case(model, vecnorm_path, env, max_steps):
    """执行 PPO 智能体评估。"""
    # 包装为 DummyVecEnv 并加载 VecNormalize
    env = DummyVecEnv([lambda: env])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False # 评估阶段必须关闭奖励归一化
    
    obs = env.reset()
    total_reward = 0.0
    totals = init_metric_totals()
    steps = 0
    done = False
    
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        # VecEnv 返回数组结构
        reward = rewards[0]
        done = dones[0]
        info = infos[0]
        
        total_reward += reward
        add_info_metrics(totals, info)
        steps += 1
        
    return {
        "steps": steps,
        "total_reward": total_reward,
        "mean_reward": total_reward / max(steps, 1),
        **totals
    }

def run_rule_case(rule_controller, env, max_steps, seed):
    """执行规则控制器评估。"""
    # reset 可在外部完成，也可在这里再次调用
    # build_tomato_env 已执行 reset，但为安全起见可重复 reset
    # 这里默认传入的是新环境实例。
    
    # 需要适配：llm_agent 中的 RuleBasedController 期望 (state, weather, env)
    # 但这里使用标准 gym 循环
    # 因此在循环内手动调用规则逻辑。
    
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    totals = init_metric_totals()
    steps = 0
    done = False
    
    while not done and steps < max_steps:
        # 为规则控制器提取必要输入
        # env.x 为状态向量
        # weather 也必需；llm_agent 中由 self._get_weather_vector() 获取
        # 这里可复现该逻辑或直接访问环境内部字段。
        
        # 通过 env.weather_data[env.timestep] 获取天气
        # 若可行优先复用 predict 调用
        # llm_agent.py 中的调用为：
        # control = self.rule_controller.predict(env.x, weather, env)
        
        try:
            weather = env.weather_data[env.timestep]
            action = rule_controller.predict(env.x, weather, env)
            
            # 规则控制器输出的是原始控制量 u，而非 [-1, 1] 归一化动作
            # 若使用 step()，TomatoEnv 期望输入为归一化动作
            # 因此优先使用 step_raw_control()（若可用）。
            
            if hasattr(env, "step_raw_control"):
                obs, reward, terminated, truncated, info = env.step_raw_control(np.array(action, dtype=np.float32))
            else:
                # 兜底分支（TomatoEnv 正常情况下不会走到这里）
                obs, reward, terminated, truncated, info = env.step(np.zeros(6))
                
            done = terminated or truncated
            total_reward += reward
            add_info_metrics(totals, info)
            steps += 1
        except Exception as e:
            print(f"Rule-Based Controller crashed at step {steps}: {e}")
            # 异常时施加较大惩罚，避免错误轨迹被误判为有效
            total_reward -= 10000.0
            done = True
            break
        
    return {
        "steps": steps,
        "total_reward": total_reward,
        "mean_reward": total_reward / max(steps, 1),
        **totals
    }

def run_llm_case(
    env_cfg: dict,
    api_key: str,
    year: int,
    day: int,
    seed: int,
    max_steps: int,
    llm_model: str,
    llm_interval: int,
    llm_max_iterations: int,
    llm_max_tokens: int,
    llm_fallback_max_steps: int,
    rule_params: dict,
    expert_rollout_enabled: bool = False,
    expert_policy_path: str = "",
    expert_candidate_max_distance: float = 0.0,
) -> dict:
    base_env_params = dict(env_cfg["GreenLightEnv"])
    base_env_params["training"] = False
    tomato_cfg = dict(env_cfg["TomatoEnv"])
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
    env.set_seed(seed)
    interface = GreenhouseAgentInterface(env)
    if hasattr(create_langchain_tools, "instance"):
        delattr(create_langchain_tools, "instance")
    tools = create_langchain_tools(interface)
    agent_cfg = AgentConfig(
        model_name=llm_model,
        api_key=api_key,
        verbose=False,
        control_interval=llm_interval,
        max_iterations=llm_max_iterations,
        max_tokens=llm_max_tokens,
        expert_rollout_enabled=bool(expert_rollout_enabled),
        expert_policy_path=expert_policy_path or AgentConfig.expert_policy_path,
        expert_candidate_max_distance=float(expert_candidate_max_distance),
    )
    agent = RuleBasedLLMDirector(
        agent_interface=interface,
        tools=tools,
        config=agent_cfg,
        rule_params=rule_params,
    )
    env.reset(seed=seed)
    total_reward = 0.0
    totals = init_metric_totals()
    steps = 0
    done = False
    runtime_error = ""
    fallback_steps = 0
    rollout_source_counts: dict[str, int] = {}
    expert_available_steps = 0
    expert_accepted_steps = 0
    expert_selected_steps = 0
    expert_distances: list[float] = []
    while (not done) and steps < max_steps:
        try:
            result = agent.step_with_rules()
        except Exception as exc:
            # 兜底路径：避免单次求解失败直接终止整条 episode。
            runtime_error = str(exc)
            if fallback_steps >= max(0, int(llm_fallback_max_steps)):
                done = True
                break
            env_ref = interface.env
            fallback_applied = False
            try:
                weather = env_ref.weather_data[env_ref.timestep]
                raw_control = agent.rule_controller.predict(env_ref.x, weather, env_ref)
                safe_control = np.asarray(raw_control, dtype=np.float32)
                if hasattr(env_ref, "step_raw_control"):
                    _, reward, terminated, truncated, info = env_ref.step_raw_control(safe_control)
                else:
                    zero_action = np.zeros(getattr(env_ref, "nu", 6), dtype=np.float32)
                    _, reward, terminated, truncated, info = env_ref.step(zero_action)
                total_reward += float(reward)
                add_info_metrics(totals, info)
                done = bool(terminated or truncated)
                steps += 1
                fallback_steps += 1
                fallback_applied = True
            except Exception:
                done = True
            if fallback_applied:
                continue
            break
        reward = float(result.get("reward", 0.0))
        total_reward += reward
        info = env._get_info()
        add_info_metrics(totals, info)
        plan_info = result.get("plan", {}) if isinstance(result, dict) else {}
        rollout_info = plan_info.get("rollout_selection", {}) if isinstance(plan_info, dict) else {}
        rollout_source = str(rollout_info.get("source", "unknown"))
        rollout_source_counts[rollout_source] = rollout_source_counts.get(rollout_source, 0) + 1
        if rollout_source.startswith("expert") or rollout_source == "distilled_expert":
            expert_selected_steps += 1
        expert_info = rollout_info.get("expert_prediction", {}) if isinstance(rollout_info, dict) else {}
        if isinstance(expert_info, dict):
            if expert_info.get("available"):
                expert_available_steps += 1
            if expert_info.get("accepted"):
                expert_accepted_steps += 1
            if expert_info.get("feature_distance") is not None:
                try:
                    expert_distances.append(float(expert_info.get("feature_distance")))
                except Exception:
                    pass
        done = bool(result.get("done", False))
        steps += 1
    return {
        "steps": steps,
        "total_reward": total_reward,
        "mean_reward": total_reward / max(steps, 1),
        "runtime_error": runtime_error,
        "fallback_steps": fallback_steps,
        "rollout_source_counts": rollout_source_counts,
        "expert_available_steps": expert_available_steps,
        "expert_accepted_steps": expert_accepted_steps,
        "expert_selected_steps": expert_selected_steps,
        "expert_mean_feature_distance": float(mean(expert_distances)) if expert_distances else 0.0,
        "expert_max_feature_distance": float(max(expert_distances)) if expert_distances else 0.0,
        **totals,
    }


def summarize(rows: list[dict], algo: str) -> dict:
    selected = [x for x in rows if x["algo"] == algo]
    rewards = [x["total_reward"] for x in selected]
    times = [x["elapsed_time"] for x in selected]
    steps = [x["steps"] for x in selected]
    business_scores = [compute_business_score(x) for x in selected]
    safety_flags = [1 if is_safety_pass(x) else 0 for x in selected]
    summary = {
        "algo": algo,
        "cases": len(selected),
        "sum_total_reward": float(sum(rewards)) if rewards else 0.0,
        "mean_total_reward": float(mean(rewards)) if rewards else 0.0,
        "std_total_reward": float(stdev(rewards)) if len(rewards) > 1 else 0.0,
        "mean_reward_per_step": float(sum(rewards) / max(sum(steps), 1)) if rewards else 0.0,
        "sum_elapsed_time": float(sum(times)) if times else 0.0,
        "mean_elapsed_time": float(mean(times)) if times else 0.0,
        "std_elapsed_time": float(stdev(times)) if len(times) > 1 else 0.0,
        "sum_business_score": float(sum(business_scores)) if business_scores else 0.0,
        "mean_business_score": float(mean(business_scores)) if business_scores else 0.0,
        "std_business_score": float(stdev(business_scores)) if len(business_scores) > 1 else 0.0,
        "safety_pass_rate": float(sum(safety_flags) / len(safety_flags)) if safety_flags else 0.0,
    }
    for k in METRIC_KEYS:
        vals = [x[k] for x in selected]
        summary[f"sum_{k}"] = float(sum(vals)) if vals else 0.0
        summary[f"mean_{k}"] = float(mean(vals)) if vals else 0.0
    return summary

def compute_business_score(row):
    """简单商业得分：收入 - （供热 + CO2 + 电力 + 固定成本）。"""
    cost = row["total_heat_cost"] + row["total_co2_cost"] + row["total_elec_cost"] + row["total_fixed_cost"]
    return row["total_revenue"] - cost

def is_safety_pass(row):
    """仅当主要违规项均低于硬阈值时判定为通过（温度/CO2/湿度）。"""
    return (
        (row["violation_temp"] < SAFETY_THRESHOLDS["temp"]) and
        (row["violation_co2"] < SAFETY_THRESHOLDS["co2"]) and
        (row["violation_rh"] < SAFETY_THRESHOLDS["rh"])
    )

def paired_compare(rows, algo_a, algo_b):
    """按相同种子做配对比较（A - B）。"""
    data_a = { (x["year"], x["day"], x["seed"]): x for x in rows if x["algo"] == algo_a }
    data_b = { (x["year"], x["day"], x["seed"]): x for x in rows if x["algo"] == algo_b }
    
    keys = set(data_a.keys()) & set(data_b.keys())
    diffs_reward = []
    diffs_profit = []
    
    for k in keys:
        ra = data_a[k]["total_reward"]
        rb = data_b[k]["total_reward"]
        diffs_reward.append(ra - rb)
        
        pa = compute_business_score(data_a[k])
        pb = compute_business_score(data_b[k])
        diffs_profit.append(pa - pb)
        
    return {
        "pair": f"{algo_a} vs {algo_b}",
        "count": len(keys),
        "mean_diff_reward": float(mean(diffs_reward)) if diffs_reward else 0.0,
        "mean_diff_profit": float(mean(diffs_profit)) if diffs_profit else 0.0,
        "win_rate_reward": float(sum(1 for d in diffs_reward if d > 0) / len(diffs_reward)) if diffs_reward else 0.0,
        "win_rate_profit": float(sum(1 for d in diffs_profit if d > 0) / len(diffs_profit)) if diffs_profit else 0.0,
    }

def export_model_params(model, rule_params, model_path, vecnorm_path, output_dir):
    """导出模型与规则参数，便于复现实验。"""
    info = {
        "ppo_model": str(model_path),
        "vecnorm": str(vecnorm_path),
        "rule_params": rule_params,
        "timestamp": datetime.now().isoformat()
    }
    with open(output_dir / "experiment_params.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

def build_markdown_report(summary, output_path):
    """生成 Markdown 报告文本。"""
    lines = []
    lines.append(f"# 三算法统一评估报告")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**评估场景**: {summary['years']} years, {summary['days']} days, {summary['repeats']} repeats")
    lines.append(f"**总步数**: {summary['max_steps']}")
    lines.append(f"**不确定性**: {summary['uncertainty_scale']}")
    lines.append(
        f"**安全阈值(硬约束)**: "
        f"temp<{summary['safety_thresholds']['temp']}, "
        f"co2<{summary['safety_thresholds']['co2']}, "
        f"rh<{summary['safety_thresholds']['rh']}"
    )
    lines.append("")
    
    lines.append("## 1. 总体表现摘要")
    lines.append("| 算法 | 总奖励 | 平均每步奖励 | 总利润 | 总收入 | 总供热成本 | 总CO2成本 | 总电力成本 | 温度违规 | CO2违规 | 湿度违规 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    
    for s in summary["summaries"]:
        # 这里商业得分与利润大体一致（收入 - 成本）
        # 注意：s["mean_total_profit"] 来自环境的 EPI，可能与手动计算略有差异
        # 此处统一使用聚合后的统计指标。
        lines.append(f"| {s['algo']} | {s['mean_total_reward']:.2f} | {s['mean_reward_per_step']:.4f} | {s['mean_total_profit']:.2f} | {s['mean_total_revenue']:.2f} | {s['mean_total_heat_cost']:.2f} | {s['mean_total_co2_cost']:.2f} | {s['mean_total_elec_cost']:.2f} | {s['mean_violation_temp']:.2f} | {s['mean_violation_co2']:.2f} | {s['mean_violation_rh']:.2f} |")
        
    lines.append("")
    lines.append("## 2. 配对胜率分析")
    lines.append("| 对比组合 | 奖励差异(均值) | 利润差异(均值) | 奖励胜率 | 利润胜率 |")
    lines.append("|---|---|---|---|---|")
    for p in summary["pairwise_analysis"]:
        lines.append(f"| {p['pair']} | {p['mean_diff_reward']:.2f} | {p['mean_diff_profit']:.2f} | {p['win_rate_reward']:.1%} | {p['win_rate_profit']:.1%} |")
        
    lines.append("")
    lines.append("## 3. 详细统计")
    for s in summary["summaries"]:
        lines.append(f"### {s['algo']}")
        lines.append(f"- **耗时**: {s['mean_elapsed_time']:.2f}s (std: {s['std_elapsed_time']:.2f})")
        lines.append(f"- **商业得分**: {s['mean_business_score']:.2f} (std: {s['std_business_score']:.2f})")
        lines.append(f"- **安全通过率**: {s['safety_pass_rate']:.1%}")
        
    report = "\n".join(lines)
    with open(output_path.with_suffix(".md"), "w", encoding="utf-8") as f:
        f.write(report)
    return report

def main():
    parser = argparse.ArgumentParser(description="Compare PPO, Rule-Based, and LLM Director.")
    parser.add_argument("--project", type=str, default="AgriControl", help="W&B project name")
    parser.add_argument("--years", type=str, default="2020", help="Comma-separated years")
    parser.add_argument("--days", type=str, default="120,240", help="Comma-separated start days")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per scenario")
    parser.add_argument("--max-steps", type=int, default=1920, help="Max steps per episode")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--uncertainty-scale", type=float, default=0.0, help="Env uncertainty")
    parser.add_argument("--output-json", type=str, default="gl_gym/result/i12_hybrid_mild_120_240.json", help="Output JSON path")
    parser.add_argument("--output-report", type=str, default="gl_gym/result/i12_hybrid_mild_120_240.md", help="Output Markdown report path")
    
    # LLM 参数
    parser.add_argument("--llm-model", type=str, default="qwen-max-latest", help="LLM model name")
    parser.add_argument("--llm-interval", type=int, default=12, help="Control interval steps")
    parser.add_argument("--llm-max-iterations", type=int, default=1, help="Max reasoning steps")
    parser.add_argument("--llm-max-tokens", type=int, default=260, help="Max output tokens")
    parser.add_argument("--llm-fallback-max-steps", type=int, default=2, help="Max fallback control steps after LLM runtime exceptions")
    parser.add_argument("--llm-expert-rollout", action="store_true", help="Enable PPO-distilled expert rollout candidate")
    parser.add_argument(
        "--llm-expert-policy-path",
        type=str,
        default=AgentConfig.expert_policy_path,
        help="Path to distilled expert NPZ model",
    )
    parser.add_argument(
        "--llm-expert-max-distance",
        type=float,
        default=AgentConfig.expert_candidate_max_distance,
        help="OOD cutoff for expert feature distance; <=0 uses model metadata p99 * 1.2",
    )
    parser.add_argument("--safety-temp-threshold", type=float, default=100.0, help="Hard threshold for temperature violation")
    parser.add_argument("--safety-co2-threshold", type=float, default=10000.0, help="Hard threshold for CO2 violation")
    parser.add_argument("--safety-rh-threshold", type=float, default=30.0, help="Hard threshold for RH violation")
    # 可选规则参数覆盖（用于定向消融）
    parser.add_argument("--rule-co2-day", type=float, default=None, help="Override co2_day setpoint")
    parser.add_argument("--rule-rh-max", type=float, default=None, help="Override rh_max threshold")
    parser.add_argument("--rule-vent-rh-pband", type=float, default=None, help="Override vent_rh_Pband")
    parser.add_argument("--rule-mech-dehumid-pband", type=float, default=None, help="Override mech_dehumid_Pband")
    parser.add_argument("--rule-co2-band", type=float, default=None, help="Override co2Band")
    
    args = parser.parse_args()

    global SAFETY_THRESHOLDS
    SAFETY_THRESHOLDS = {
        "temp": float(args.safety_temp_threshold),
        "co2": float(args.safety_co2_threshold),
        "rh": float(args.safety_rh_threshold),
    }
    
    years = [int(y) for y in args.years.split(",")]
    days = [int(d) for d in args.days.split(",")]
    
    # 加载配置
    with open("gl_gym/configs/envs/TomatoEnv.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    env_base_params = config["GreenLightEnv"]
    env_specific_params = config["TomatoEnv"]
    
    # 模型与归一化器路径
    model_path = Path("train_data/AgriControl/ppo/deterministic/models/None/best_model.zip")
    vecnorm_path = Path("train_data/AgriControl/ppo/deterministic/envs/None/best_vecnormalize.pkl")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    # 初始化 PPO 模型
    print("Loading PPO model...")
    model = PPO.load(model_path)
    
    # 初始化规则控制器
    # Hybrid-mild 默认参数（仍可被下方命令行参数覆盖）
    rule_params = {
        "lamps_on": 6.0,
        "lamps_off": 20.0,
        "lamps_day_start": 1,
        "lamps_day_stop": 365,
        "lamps_off_sun": 400.0,
        "lamp_rad_sum_limit": 1000.0,
        "temp_setpoint_day": 20.0,
        "temp_setpoint_night": 16.0,
        "heat_correction": 1.0,
        "heat_deadzone": 0.5,
        "co2_day": 850.0,
        "vent_heat_Pband": 4.0,
        "rh_max": 83.0,
        "mech_dehumid_Pband": 7.0,
        "vent_rh_Pband": 7.0,
        "t_vent_off": 10.0,
        "vent_cold_Pband": 5.0,
        "thScrSpDay": 25.0,
        "thScrSpNight": 10.0,
        "thScrPband": 5.0,
        "thScrDeadZone": 2.0,
        "thScrRh": 90.0,
        "thScrRhPband": 5.0,
        "lampExtraHeat": 2.0,
        "blScrExtraRh": 5.0,
        "rhMax": 90.0,
        "tHeatBand": 2.0,
        "co2Band": 180.0,
        "useBlScr": False
    }
    overrides = {
        "co2_day": args.rule_co2_day,
        "rh_max": args.rule_rh_max,
        "vent_rh_Pband": args.rule_vent_rh_pband,
        "mech_dehumid_Pband": args.rule_mech_dehumid_pband,
        "co2Band": args.rule_co2_band,
    }
    for k, v in overrides.items():
        if v is not None:
            rule_params[k] = v
    rule = GreenhouseRuleController(**rule_params)
    
    # 获取 LLM API Key

    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        print("Warning: BAILIAN_API_KEY not found. LLM agent might fail if invoked.")
    
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_model_params(model, rule_params, model_path, vecnorm_path, output_path.parent)
    with open("gl_gym/configs/envs/TomatoEnv.yml", "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)
    env_cfg["TomatoEnv"]["uncertainty_scale"] = args.uncertainty_scale

    rows = []
    for repeat in range(args.repeats):
        seed = args.base_seed + repeat
        for year in years:
            for day in days:
                print(f"Running PPO for year {year}, day {day}, repeat {repeat}...")
                env_ppo = build_tomato_env(env_base_params, env_specific_params, year, day, seed, args.uncertainty_scale)
                env_rb = build_tomato_env(env_base_params, env_specific_params, year, day, seed, args.uncertainty_scale)

                t0 = time.perf_counter()
                ppo_result = run_ppo_case(model, vecnorm_path, env_ppo, args.max_steps)
                ppo_elapsed = time.perf_counter() - t0
                ppo_row = {
                    "algo": "ppo",
                    "year": year,
                    "day": day,
                    "seed": seed,
                    "repeat": repeat,
                    "elapsed_time": float(ppo_elapsed),
                    **ppo_result,
                }
                rows.append(ppo_row)

                t1 = time.perf_counter()
                print(f"Running Rule-Based for year {year}, day {day}, repeat {repeat}...")
                try:
                    rb_result = run_rule_case(rule, env_rb, args.max_steps, seed)
                    rb_elapsed = time.perf_counter() - t1
                    rb_row = {
                        "algo": "rule_based",
                        "year": year,
                        "day": day,
                        "seed": seed,
                        "repeat": repeat,
                        "elapsed_time": float(rb_elapsed),
                        **rb_result,
                    }
                    rows.append(rb_row)
                except Exception as e:
                    # 规则控制器失败时不中断整体实验，确保 PPO/LLM 结果仍可产出。
                    print(f"Warning: Rule-Based failed for year {year}, day {day}, repeat {repeat}: {e}")
                
                t2 = time.perf_counter()
                if api_key:
                    print(f"Running LLM Director for year {year}, day {day}, repeat {repeat}...")
                    llm_result = run_llm_case(
                        env_cfg, api_key, year, day, seed, args.max_steps,
                        args.llm_model, args.llm_interval, args.llm_max_iterations, args.llm_max_tokens,
                        args.llm_fallback_max_steps,
                        rule_params,
                        expert_rollout_enabled=args.llm_expert_rollout,
                        expert_policy_path=args.llm_expert_policy_path,
                        expert_candidate_max_distance=args.llm_expert_max_distance,
                    )
                    llm_elapsed = time.perf_counter() - t2
                    llm_row = {
                        "algo": "llm_director",
                        "year": year,
                        "day": day,
                        "seed": seed,
                        "repeat": repeat,
                        "elapsed_time": float(llm_elapsed),
                        **llm_result,
                    }
                    rows.append(llm_row)
                else:
                    print("Skipping LLM Director due to missing API Key.")


    summary = {
        "project": args.project,
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnorm_path),
        "years": years,
        "days": days,
        "repeats": args.repeats,
        "base_seed": args.base_seed,
        "max_steps": args.max_steps,
        "uncertainty_scale": args.uncertainty_scale,
        "llm_model": args.llm_model,
        "llm_interval": args.llm_interval,
        "llm_max_iterations": args.llm_max_iterations,
        "llm_max_tokens": args.llm_max_tokens,
        "llm_expert_rollout": bool(args.llm_expert_rollout),
        "llm_expert_policy_path": args.llm_expert_policy_path,
        "llm_expert_max_distance": float(args.llm_expert_max_distance),
        "safety_thresholds": SAFETY_THRESHOLDS,
        "summaries": [
            summarize(rows, "ppo"),
            summarize(rows, "rule_based"),
            summarize(rows, "llm_director"),
        ],
        "pairwise_analysis": [
            paired_compare(rows, "ppo", "rule_based"),
            paired_compare(rows, "ppo", "llm_director"),
            paired_compare(rows, "llm_director", "rule_based"),
        ],
        "results": rows,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    report_path = Path(args.output_report) if args.output_report else output_path.with_suffix(".md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = build_markdown_report(summary, output_path)
    print(f"Metrics saved to {output_path}")
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
