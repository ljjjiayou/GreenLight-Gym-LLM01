import argparse
import os
from os.path import join
from tqdm import tqdm
import pandas as pd
import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from gl_gym.RL.utils import make_vec_env
from gl_gym.common.results import Results
from gl_gym.common.utils import load_env_params, load_model_hyperparams

# 算法映射表
ALG = {"ppo": PPO,
       "sac": SAC}


def load_env(env_id, model_name, env_base_params, env_specific_params, load_path):
    """
    加载评估环境并恢复标准化统计参数。
    """
    # 设置为非训练模式
    env_base_params["training"] = False

    # 创建向量化环境（单环境实例）
    env = make_vec_env(
        env_id,
        env_base_params,
        env_specific_params,
        seed=666,
        n_envs=1,
        monitor_filename=None,
        vec_norm_kwargs=None,
        eval_env=True
    )

    # 从保存的路径中加载 VecNormalize 统计信息
    env = VecNormalize.load(join(load_path + f"/envs", f"{model_name}/best_vecnormalize.pkl"), env)
    # 确保评估时不更新归一化参数，且不归一化奖励
    env.training = False
    env.norm_reward = False

    return env


def evaluate(model, env):
    """
    执行单次模拟评估并收集所有步骤的数据。
    """
    N = env.get_attr("N")[0]  # 获取总步数
    # 初始化用于存储各项指标的数组
    epi, revenue, heat_cost, co2_cost, elec_cost = np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1), np.zeros(
        N + 1), np.zeros(N + 1)
    temp_violation, co2_violation, rh_violation = np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1)
    episodic_obs = np.zeros((N + 1, 23))  # 存储前 23 个观测变量
    episode_rewards = np.zeros(N + 1)

    dones = np.zeros((1,), dtype=bool)
    episode_starts = np.ones((1,), dtype=bool)

    # 重置环境获取初始观测值
    observations = env.reset()
    timestep = 0
    states = None

    for timestep in range(N):
        # 模型预测动作
        actions, states = model.predict(
            observations,  # 类型忽略
            state=states,
            episode_start=episode_starts,
            deterministic=True,  # 使用确定性动作进行评估
        )
        # 执行动作
        observations, rewards, dones, infos = env.step(actions)

        # 记录奖励和反标准化后的观测值
        episode_rewards[timestep] += rewards[0]
        episodic_obs[timestep] += env.unnormalize_obs(observations)[0, :23]

        # 从 info 字典中提取经济和违规指标
        epi[timestep] += infos[0]["EPI"]
        revenue[timestep] += infos[0]["revenue"]
        heat_cost[timestep] += infos[0]["heat_cost"]
        elec_cost[timestep] += infos[0]["elec_cost"]
        co2_cost[timestep] += infos[0]["co2_cost"]
        temp_violation[timestep] += infos[0]["temp_violation"]
        co2_violation[timestep] += infos[0]["co2_violation"]
        rh_violation[timestep] += infos[0]["rh_violation"]
        timestep += 1

    # 拼接所有结果列
    result_data = np.column_stack(
        (episodic_obs, episode_rewards, epi, revenue, heat_cost, co2_cost, elec_cost, temp_violation, co2_violation,
         rh_violation))
    return result_data[:-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl", help="项目名称（W&B 中）")
    parser.add_argument("--env_id", type=str, default="TomatoEnv", help="环境 ID")
    parser.add_argument("--model_name", type=str, default="cosmic-music-45", help="训练好的模型文件夹名称")
    parser.add_argument("--algorithm", type=str, default="ppo", help="算法名称 (ppo 或 sac)")
    parser.add_argument("--uncertainty_scale", type=float, help="参数不确定性缩放比例", required=True)
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True)
    args = parser.parse_args()

    # 检查：如果是确定性模式，不确定性缩放必须为 0.0
    assert not (args.mode == "deterministic" and args.uncertainty_scale != 0.0), \
        "确定性模式下，不确定性缩放必须为 0.0"

    env_config_path = f"gl_gym/configs/envs/"
    load_path = f"train_data/{args.project}/{args.algorithm}/{args.mode}/"

    # 设置保存目录及模拟次数
    if args.mode == "stochastic":
        save_dir = f"data/{args.project}/{args.mode}/{args.algorithm}/{args.uncertainty_scale}/"
        n_sims = 30  # 随机模式下运行 30 次模拟以观察分布
    else:
        save_dir = f"data/{args.project}/{args.mode}/{args.algorithm}/"
        n_sims = 1  # 确定性模式下只运行 1 次
    os.makedirs(save_dir, exist_ok=True)

    # 加载环境参数和模型超参数
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    model_params = load_model_hyperparams(args.algorithm, args.env_id)

    # 将用户输入的随机性缩放应用到环境参数中
    env_specific_params["uncertainty_scale"] = args.uncertainty_scale
    eval_env = load_env(args.env_id, args.model_name, env_base_params, env_specific_params, load_path)

    # 加载训练好的 SB3 模型
    model = ALG[args.algorithm].load(join(load_path + f"models", f"{args.model_name}/best_model.zip"), device="cpu")

    # 构建结果文件的列名
    result_columns = eval_env.env_method("get_obs_names")[0][:23]
    result_columns.extend(["Rewards", "EPI", "Revenue", "Heat costs", "CO2 costs", "Elec costs"])
    result_columns.extend(["temp_violation", "co2_violation", "rh_violation"])
    result_columns.extend(["episode"])
    result = Results(result_columns)

    # 开始循环模拟
    for sim in tqdm(range(n_sims)):
        # 为环境设置不同的种子以获得不同的随机参数采样
        eval_env.env_method("set_seed", 666 + sim)
        result_data = evaluate(model, eval_env)

        # 添加一列标识当前的模拟序号 (episode)
        sim_column = np.full((result_data.shape[0], 1), sim)
        result_data = np.column_stack((result_data, sim_column))

        # 更新结果集
        result.update_result(result_data)

    # 获取环境的具体属性用于生成文件名
    start_day = eval_env.get_attr("start_day")[0]
    growth_year = eval_env.get_attr("growth_year")[0]
    location = eval_env.get_attr("location")[0]

    # 保存结果到 CSV 文件
    save_name = f"{args.model_name}-{growth_year}{start_day}-{location}.csv"
    print("正在保存结果至", save_name)
    result.save(f"{save_dir}/{save_name}")