import os
import argparse

import wandb
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # 配置命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default="bartvlaatum", help="W&B 的实体名称，通常是你的用户名")
    parser.add_argument("--project", type=str, help="你想要提取记录数据的项目名称")
    parser.add_argument("--group", type=str, help="你想要提取记录数据的实验组名称")
    args = parser.parse_args()

    # 创建本地保存目录
    save_dir = f"data/{args.project}/deterministic/{args.group}/"
    os.makedirs(save_dir, exist_ok=True)

    # 初始化 W&B API 并根据项目名和组名过滤运行记录
    api = wandb.Api()
    runs = api.runs(args.entity + "/" + args.project, filters={"group": args.group})
    print(runs)

    summary_list, config_list, name_list = [], [], []
    train_rew_mean = []  # 存储训练奖励均值
    step = []  # 存储本地步数
    global_step = []  # 存储全局步数

    for run in runs:
        # run.summary 包含指标的输出键/值（如准确率等）
        # 我们调用 ._json_dict 来忽略大型文件并转换为字典格式
        summary_list.append(run.summary._json_dict)
        summary_dict = run.summary._json_dict
        # 获取该次运行的历史记录数据
        history_dict = run.history()

        # 提取训练过程中的回合平均奖励
        train_rew_mean.append(history_dict['rollout/ep_rew_mean'])

        # 计算数据点的数量
        N = len(history_dict['rollout/ep_rew_mean'])

        # 生成对应的步数索引和实验名称
        step.append(np.arange(1, N + 1))
        name_list.append([run.name] * N)
        global_step.append(history_dict['global_step'])

        # run.config 包含超参数
        # 我们移除以 "_" 开头的特殊内部参数
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # 将嵌套列表展平为一维数组，方便创建 DataFrame
    # 注意：这里从 [1:] 开始提取是为了过滤或处理特定数据的示例逻辑
    step = np.array(step[1:]).flatten()
    train_rew_mean = np.array(train_rew_mean[1:]).flatten()
    name_list = np.array(name_list[1:]).flatten()
    global_step = np.array(global_step[1:]).flatten()

    print(
        f"数据量统计: 步数:{len(step)}, 奖励:{len(train_rew_mean)}, 名称:{len(name_list)}, 全局步数:{len(global_step)}")

    # 封装为 Pandas DataFrame
    train_df = pd.DataFrame({
        'global step': global_step,
        "step": step,
        "train reward": train_rew_mean,
        "name": name_list
    })

    # 清洗数据：移除空值并保存为 CSV
    train_df = train_df.dropna()
    train_df.to_csv(f"data/{args.project}/deterministic/{args.group}/rollout.csv", index=False)