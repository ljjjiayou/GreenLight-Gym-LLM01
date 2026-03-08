import argparse
import os
from os.path import join
from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.environments.baseline import RuleBasedController
from gl_gym.common.utils import load_env_params, load_model_hyperparams
from gl_gym.common.results import Results
import numpy as np
from tqdm import tqdm


def evaluate_controller(env, controller, rank=0):
    """
    执行单次模拟评估并收集所有步骤的数据。
    """
    # 初始化用于存储各项指标的数组
    epi, revenue, heat_cost, co2_cost, elec_cost = np.zeros(env.N + 1), np.zeros(env.N + 1), np.zeros(
        env.N + 1), np.zeros(env.N + 1), np.zeros(env.N + 1)
    temp_violation, co2_violation, rh_violation = np.zeros(env.N + 1), np.zeros(env.N + 1), np.zeros(env.N + 1)
    rewards = np.zeros(env.N + 1)
    episodic_obs = np.zeros((env.N + 1, 23))  # 存储前 23 个观测变量

    # 重置环境，根据 rank 设置随机种子
    obs = env.reset(seed=666 + rank)
    done = False
    timestep = 0

    while not done:
        # 基于规则的控制器根据当前状态和天气预测控制输出
        control = controller.predict(env.x, env.weather_data[env.timestep], env)

        # 执行原始控制量（不经过 RL 动作缩放）
        obs, r, done, _, info = env.step_raw_control(control)

        # 记录数据
        rewards[timestep] += r
        episodic_obs[timestep] += obs[:23]
        epi[timestep] += info["EPI"]
        revenue[timestep] += info["revenue"]
        heat_cost[timestep] += info["heat_cost"]
        elec_cost[timestep] += info["elec_cost"]
        co2_cost[timestep] += info["co2_cost"]
        temp_violation[timestep] += info["temp_violation"]
        co2_violation[timestep] += info["co2_violation"]
        rh_violation[timestep] += info["rh_violation"]
        timestep += 1

    # 将所有结果数据拼接成矩阵
    result_data = np.column_stack(
        (episodic_obs, rewards, epi, revenue, heat_cost, co2_cost, elec_cost, temp_violation, co2_violation,
         rh_violation))
    return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl", help="项目名称（W&B 中）")
    parser.add_argument("--env_id", type=str, default="TomatoEnv", help="环境 ID")
    parser.add_argument("--uncertainty_scale", type=float, help="参数不确定性缩放比例", required=True)
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True)
    args = parser.parse_args()

    # 定义结果保存路径
    save_dir = f"data/{args.project}/{args.mode}/rb_baseline/"
    env_config_path = f"gl_gym/configs/envs/"

    # 设置模拟次数和最终保存子目录
    if args.mode == "stochastic":
        save_dir = f"data/{args.project}/{args.mode}/rb_baseline/{args.uncertainty_scale}/"
        n_sims = 30  # 随机模式下运行 30 次以观察统计分布
    else:
        # 注意：此处原代码存在一个小 bug，通常应为 args.mode 而非不存在的 args.algorithm
        save_dir = f"data/{args.project}/{args.mode}/rb_baseline/"
        n_sims = 1  # 确定性模式下运行 1 次

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 加载基于规则的控制器超参数
    rb_params = load_model_hyperparams('rule_based', args.env_id)
    rb_controller = RuleBasedController(**rb_params)

    # 加载环境参数并初始化 TomatoEnv
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    env_base_params['training'] = True  # 设置为 True 以允许在重置时随机采样年份/天数
    eval_env = TomatoEnv(base_env_params=env_base_params, uncertainty_scale=args.uncertainty_scale,
                         **env_specific_params)

    # 构建结果文件的列名
    result_columns = eval_env.get_obs_names()[:23]
    result_columns.extend(["Rewards", "EPI", "Revenue", "Heat costs", "CO2 costs", "Elec costs"])
    result_columns.extend(["temp_violation", "co2_violation", "rh_violation"])
    result_columns.extend(["episode"])
    result = Results(result_columns)

    # 开始循环评估
    for sim in tqdm(range(n_sims)):
        result_data = evaluate_controller(eval_env, rb_controller, rank=sim)

        # 添加一列标识当前的模拟序号 (episode)
        sim_column = np.full((result_data.shape[0], 1), sim)
        result_data = np.column_stack((result_data, sim_column))

        # 更新结果集
        result.update_result(result_data)

    # 获取当前环境的具体属性用于生成文件名
    start_day = eval_env.start_day
    growth_year = eval_env.growth_year
    location = eval_env.location

    # 保存结果到 CSV 文件
    save_name = f"rb_baseline-{growth_year}{start_day}-{location}.csv"
    print("正在保存结果至", save_name)
    result.save(f"{save_dir}/{save_name}")