import os
import argparse
import gc
import numpy as np

# 屏蔽 TensorFlow 的冗余日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from torch.nn.modules.activation import ReLU, SiLU, Tanh, ELU
from torch.optim import Adam, RMSprop
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from gl_gym.common.utils import load_model_hyperparams
from gl_gym.RL.utils import (
    load_env_params,
    wandb_init,
    make_vec_env,
    create_callbacks,
    load_sweep_config
)

import wandb

# 映射表：将配置字符串映射为 PyTorch 激活函数和优化器类
ACTIVATION_FN = {"relu": ReLU, "silu": SiLU, "tanh": Tanh, "elu": ELU}
OPTIMIZER = {"adam": Adam, "rmsprop": RMSprop}
# 动作噪声映射（主要用于 SAC 算法）
ACTION_NOISE = {"normalactionnoise": NormalActionNoise, "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise}


def get_obs_names(env):
    """
    从给定的环境中提取并返回观测变量名称列表。
    """
    obs_names = []
    # 遍历环境中所有观测模块
    for obs_module in env.get_attr("observation_modules")[0]:
        for name in obs_module.obs_names:
            # 将下划线替换为空格以方便展示
            obs_names.append(name.replace("_", " "))
    return obs_names


class ExperimentManager:
    """
    强化学习运行全生命周期管理器。

    主要职责：
    - 加载环境和智能体配置
    - 构建带标准化的向量化训练/评估环境
    - 实例化 SB3 模型（PPO/SAC/RecurrentPPO）
    - 执行训练、周期性评估及持久化保存
    - 可选：执行 W&B 超参数搜索 (Sweeps)
    """

    def __init__(
            self,
            env_id,
            project,
            env_base_params,
            env_specific_params,
            hyperparameters,
            group,
            n_eval_episodes,
            n_evals,
            algorithm,
            env_seed,
            model_seed,
            stochastic,
            save_model=True,
            save_env=True,
            hp_tuning=False,
            device="cpu"
    ):
        """初始化管理器。"""
        self.env_id = env_id
        self.project = project
        self.env_base_params = env_base_params
        self.env_specific_params = env_specific_params
        self.n_envs = hyperparameters["n_envs"]
        self.total_timesteps = hyperparameters["total_timesteps"]
        self.stochastic = stochastic

        # 提取总步数和环境数量后，剩余的作为算法超参数
        del hyperparameters["total_timesteps"]
        del hyperparameters["n_envs"]
        self.hyperparameters = hyperparameters

        self.n_eval_episodes = n_eval_episodes
        self.group = group
        self.n_evals = n_evals
        self.algorithm = algorithm
        self.env_seed = env_seed
        self.model_seed = model_seed
        self.save_model = save_model
        self.save_env = save_env
        self.device = device
        self.hp_tuning = hp_tuning

        # 支持的算法字典
        self.models = {"ppo": PPO, "sac": SAC, "recurrentppo": RecurrentPPO}
        self.model_class = self.models[self.algorithm.lower()]

        # 配置文件路径
        self.hyp_config_path = f"gl_gym/configs/sweeps/"

        # 初始化环境和模型
        print("是否开启调优:", self.hp_tuning)
        if not self.hp_tuning:
            # 初始化 WandB 会话
            self.run, self.config = wandb_init(
                self.hyperparameters,
                self.env_seed,
                self.model_seed,
                project=self.project,
                group=self.group,
                save_code=False
            )
            # 初始化向量化环境
            self.init_envs(self.hyperparameters["gamma"])
            # 构建模型所需参数
            self.model_params = self.build_model_parameters()
            # 实例化模型
            self.initialise_model()

    def init_envs(self, gamma):
        """
        创建带有 VecNormalize 的向量化训练和评估环境。
        标准化使用指定的折扣因子 `gamma` 来计算回报（Returns）。
        """
        self.monitor_filename = None
        vec_norm_kwargs = {
            "norm_obs": True,  # 开启观测值标准化
            "norm_reward": True,  # 开启奖励标准化
            "clip_obs": 10,  # 裁剪极端观测值
            "gamma": gamma  # 用于回报标准化的折扣因子
        }

        # 设置训练环境
        self.env_base_params["training"] = True
        self.env = make_vec_env(
            self.env_id,
            self.env_base_params,
            self.env_specific_params,
            seed=self.env_seed,
            n_envs=self.n_envs,
            monitor_filename=self.monitor_filename,
            vec_norm_kwargs=vec_norm_kwargs
        )

        # 设置评估环境（固定为单环境）
        self.env_base_params["training"] = False
        self.eval_env = make_vec_env(
            self.env_id,
            self.env_base_params,
            self.env_specific_params,
            seed=self.env_seed,
            n_envs=1,
            monitor_filename=self.monitor_filename,
            vec_norm_kwargs=vec_norm_kwargs,
            eval_env=True,
        )

    def initialise_model(self):
        """实例化 SB3 模型并配置 TensorBoard 日志路径。"""
        # 根据是否是随机模式确定日志目录
        mode_str = "stochastic" if self.stochastic else "deterministic"
        tensorboard_log = f"train_data/{self.project}/{self.algorithm}/{mode_str}/logs/{self.run.name}"

        # 创建 SB3 智能体
        self.model = self.model_class(
            env=self.env,
            seed=self.model_seed,
            verbose=1,
            **self.model_params,
            tensorboard_log=tensorboard_log,
            device=self.device
        )

    def build_model_parameters(self):
        """
        从配置中准备 SB3 构造函数的关键字参数。

        - 将 policy_kwargs 中的字符串（激活函数、优化器）映射为类
        - 解析 log_std_init 表达式
        - 对于 SAC，构造配置的动作噪声
        """
        model_params = self.hyperparameters.copy()

        if "policy_kwargs" in self.hyperparameters:
            policy_kwargs = self.hyperparameters["policy_kwargs"].copy()
            # 解析字符串形式的初始标准差
            policy_kwargs["log_std_init"] = eval(policy_kwargs["log_std_init"])

            # 处理激活函数映射
            if "activation_fn" in policy_kwargs:
                activation_fn_str = policy_kwargs["activation_fn"]
                if activation_fn_str in ACTIVATION_FN:
                    policy_kwargs["activation_fn"] = ACTIVATION_FN[activation_fn_str]
                else:
                    raise ValueError(f"不支持的激活函数: {activation_fn_str}")

            # 处理优化器类映射
            if "optimizer_class" in policy_kwargs:
                optimizer_str = policy_kwargs["optimizer_class"]
                if optimizer_str in OPTIMIZER:
                    policy_kwargs["optimizer_class"] = OPTIMIZER[optimizer_str]
                else:
                    raise ValueError(f"不支持的优化器: {optimizer_str}")
            model_params["policy_kwargs"] = policy_kwargs

        # 为 SAC 算法配置动作噪声
        if self.algorithm == "sac":
            if "action_noise" in self.hyperparameters:
                action_noise_key, noise_params = next(iter(self.hyperparameters["action_noise"].items()))
                if action_noise_key in ACTION_NOISE:
                    action_noise = ACTION_NOISE[action_noise_key](
                        mean=np.zeros(self.env.action_space.shape),
                        sigma=noise_params["sigma"] * np.ones(self.env.action_space.shape)
                    )
                model_params["action_noise"] = action_noise
        return model_params

    def build_model_hyperparameters(self, config):
        """
        将 W&B Sweep 采样的配置转换为 SB3 参数。
        处理 gamma 偏移、网络架构映射以及 SAC 的动作噪声。
        """
        self.model_params = dict(config).copy()
        self.model_params["gamma"] = 1.0 - config["gamma_offset"]

        policy_kwargs = {}
        policy_kwargs["net_arch"] = {}
        policy_kwargs["optimizer_kwargs"] = config["optimizer_kwargs"]
        policy_kwargs["activation_fn"] = ACTIVATION_FN[config["activation_fn"]]

        # 清理 Sweep 特有键名
        del self.model_params["optimizer_kwargs"], self.model_params["activation_fn"], self.model_params["gamma_offset"]

        if self.algorithm == "ppo":
            policy_kwargs["net_arch"]["pi"] = [config["pi"]] * 3
            policy_kwargs["net_arch"]["vf"] = [config["vf"]] * 3
            del self.model_params["vf"]

        elif self.algorithm == "sac":
            policy_kwargs["net_arch"]["pi"] = [config["pi"]] * 3
            policy_kwargs["net_arch"]["qf"] = [config["qf"]] * 3
            # 配置 SAC 的动作噪声
            action_noise_key = config["action_noise_type"]
            action_std = config["action_sigma"]
            action_noise = ACTION_NOISE[action_noise_key](
                mean=np.zeros(self.env.action_space.shape),
                sigma=action_std * np.ones(self.env.action_space.shape)
            )
            self.model_params["action_noise"] = action_noise
            del self.model_params["action_noise_type"], self.model_params["action_sigma"], self.model_params["qf"]

        elif self.algorithm == "recurrentppo":
            policy_kwargs["net_arch"]["pi"] = [config["pi"]] * 2
            policy_kwargs["net_arch"]["vf"] = [config["vf"]] * 2
            policy_kwargs["lstm_hidden_size"] = config["lstm_hidden_size"]
            policy_kwargs["enable_critic_lstm"] = config["enable_critic_lstm"]
            del self.model_params["vf"]
            # 共享或独立 LSTM
            policy_kwargs["shared_lstm"] = not policy_kwargs["enable_critic_lstm"]
            del self.model_params["lstm_hidden_size"], self.model_params["enable_critic_lstm"]

        del self.model_params["pi"]
        self.model_params["policy_kwargs"] = policy_kwargs

    def run_single_sweep(self):
        """运行单次 Sweep 试验：初始化、构建参数并开始训练。"""
        with wandb.init(sync_tensorboard=True) as run:
            self.run = run
            self.config = wandb.config
            self.init_envs(1 - self.config["gamma_offset"])
            self.build_model_hyperparameters(self.config)
            self.initialise_model()
            self.run_experiment()

    def hyperparameter_tuning(self):
        """启动 W&B Sweep 任务，运行多次采样试验。"""
        self.total_timesteps = 1.5e6  # 标准调优运行 1.5M 步
        continue_sweep = False
        sweep_config = load_sweep_config(self.hyp_config_path, self.env_id, self.algorithm)
        if continue_sweep:
            # 继续之前的 Sweep 任务
            wandb.agent("puk5fznz", project="dwarf-env", function=self.run_single_sweep, count=100)
        else:
            # 创建新 Sweep 并运行
            sweep_id = wandb.sweep(sweep=sweep_config, project=self.project)
            wandb.agent(sweep_id, function=self.run_single_sweep, count=100)

    def run_experiment(self):
        """
        开始训练过程。执行指定步数的学习，定期评估并持久化结果。
        包含日志目录设置、评估回调创建以及最后的模型与环境统计保存。
        """
        mode_str = "stochastic" if self.stochastic else "deterministic"
        model_log_dir = f"train_data/{self.project}/{self.algorithm}/{mode_str}/models/{self.run.name}/" if self.save_model else None
        env_log_dir = f"train_data/{self.project}/{self.algorithm}/{mode_str}/envs/{self.run.name}/" if self.save_env else None

        # 计算评估频率
        eval_freq = self.total_timesteps // self.n_evals // self.n_envs
        save_name = "vec_norm"

        # 创建训练回调（处理评估、最佳模型保存等）
        callbacks = create_callbacks(
            self.n_eval_episodes,
            eval_freq,
            env_log_dir,
            save_name,
            model_log_dir,
            self.eval_env,
            run=self.run,
            results=None,
            save_env=self.save_env,
            verbose=1
        )

        # 开始模型学习
        self.model.learn(total_timesteps=self.total_timesteps, callback=callbacks, reset_num_timesteps=False)

        if model_log_dir:
            self.model.save(os.path.join(model_log_dir, "last_model"))

        # 保存环境标准化统计参数
        if env_log_dir:
            env_save_path = os.path.join(env_log_dir, "last_vecnormalize.pkl")
            self.model.get_vec_normalize_env().save(env_save_path)

        # 清理并结束运行
        self.run.finish()
        self.env.close()
        self.eval_env.close()
        del self.model, self.env, self.eval_env
        gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="AgriControl", help="Wandb 项目名称")
    parser.add_argument("--env_id", type=str, default="TomatoEnv", help="环境 ID")
    parser.add_argument("--algorithm", type=str, default="ppo", help="使用的 RL 算法")
    parser.add_argument("--group", type=str, default="group1", help="Wandb 组名称")
    parser.add_argument("--n_eval_episodes", type=int, default=1, help="每次评估执行的轮数")
    parser.add_argument("--n_evals", type=int, default=10, help="训练期间评估的总次数")
    parser.add_argument("--env_seed", type=int, default=666, help="环境随机种子")
    parser.add_argument("--model_seed", type=int, default=666, help="模型随机种子")
    parser.add_argument("--stochastic", action="store_true", help="是否以随机模式运行（针对环境参数）")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备 (cpu/cuda)")
    parser.add_argument("--save_model", default=True, action=argparse.BooleanOptionalAction, help="是否保存模型")
    parser.add_argument("--save_env", default=True, action=argparse.BooleanOptionalAction, help="是否保存环境状态")
    parser.add_argument("--hyperparameter_tuning", default=False, action=argparse.BooleanOptionalAction,
                        help="是否执行超参数搜索")
    args = parser.parse_args()

    # 加载环境和超参数配置
    env_config_path = f"gl_gym/configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    hyperparameters = load_model_hyperparams(args.algorithm, args.env_id)

    # 实例化实验管理器
    experiment_manager = ExperimentManager(
        env_id=args.env_id,
        project=args.project,
        env_base_params=env_base_params,
        env_specific_params=env_specific_params,
        hyperparameters=hyperparameters,
        group=args.group,
        n_eval_episodes=args.n_eval_episodes,
        n_evals=args.n_evals,
        algorithm=args.algorithm,
        env_seed=args.env_seed,
        model_seed=args.model_seed,
        stochastic=args.stochastic,
        save_model=args.save_model,
        save_env=args.save_env,
        hp_tuning=args.hyperparameter_tuning,
        device=args.device
    )

    if args.hyperparameter_tuning:
        experiment_manager.hyperparameter_tuning()
    else:
        experiment_manager.run_experiment()


if __name__ == "__main__":
    main()