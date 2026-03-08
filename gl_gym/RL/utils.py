import os
import yaml
from os.path import join
from typing import Dict, Any, Callable, List, Optional, Union, Tuple

import wandb
import numpy as np
import pandas as pd

from torch.optim.adam import Adam
from torch.nn.modules.activation import ReLU, SiLU, Tanh, ELU
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor, VecEnv

from gl_gym.common.callbacks import CustomWandbCallback, SaveVecNormalizeCallback, BaseCallback
from gl_gym.environments.tomato_env import TomatoEnv

from gl_gym.common.results import Results

# 映射配置中的字符串到具体的 PyTorch 激活函数和优化器类
ACTIVATION_FN = {"ReLU": ReLU, "SiLU": SiLU, "Tanh": Tanh, "ELU": ELU}
OPTIMIZER = {"ADAM": Adam}
# 环境注册字典
ENVS = {"TomatoEnv": TomatoEnv}


def make_env(env_id, rank, seed, env_base_params, env_specific_params, eval_env):
    '''
    用于多进程环境的实用工厂函数。

    参数:
        env_id: (str) 环境 ID
        rank: (int) 进程排名/索引
        seed: (int) 随机种子
        env_base_params: 基础环境参数
        env_specific_params: 特定任务参数
        eval_env: 是否为评估环境
    返回:
        _init: (function) 返回一个初始化环境的函数
    '''

    def _init():
        # 根据 env_id 实例化具体的环境类（如 TomatoEnv）
        env = ENVS[env_id](**env_specific_params, base_env_params=env_base_params)
        # 按照 gymnasium 的新语法，在重置时传入带偏移的种子
        env.reset(seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _init


def make_vec_env(
        env_id: str,
        env_base_params: Dict[str, Any],
        env_specific_params: Dict[str, Any],
        seed: int,
        n_envs: int,
        monitor_filename: str | None = None,
        vec_norm_kwargs: Dict[str, Any] | None = None,
        eval_env: bool = False
) -> VecEnv:
    """
    创建包含 n 个独立环境的向量化环境。
    """
    # 如果指定了监控日志文件路径，确保对应的文件夹存在
    if monitor_filename is not None and not os.path.exists(os.path.dirname(monitor_filename)):
        os.makedirs(os.path.dirname(monitor_filename), exist_ok=True)

    # 创建并行子进程环境
    env = SubprocVecEnv(
        [make_env(env_id, rank, seed, env_base_params, env_specific_params, eval_env=eval_env) for rank in
         range(n_envs)])
    # 包装监控装饰器，记录训练统计数据
    env = VecMonitor(env, filename=monitor_filename)

    # 如果提供了标准化参数，应用 VecNormalize
    if vec_norm_kwargs is not None:
        env = VecNormalize(env, **vec_norm_kwargs)
        if eval_env:
            # 评估环境不需要更新标准化参数，也不标准化奖励
            env.training = False
            env.norm_reward = False
    return env


def load_env_params(env_id: str, path: str) -> Tuple[Dict, Dict]:
    '''
    加载环境配置变量的函数。
    返回通用基类 GreenLightEnv 的参数，以及特定环境（如 TomatoEnv）的参数。
    参数:
        env_id (str): 环境 ID
        path (str): YAML 配置文件所在的路径
    '''
    # 显式使用 utf-8 编码读取，防止 Windows 系统报错
    with open(join(path, env_id + ".yml"), "r", encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if env_id != "GreenLightEnv":
        env_specific_params = params[env_id]
    else:
        env_specific_params = {}

    env_base_params = params["GreenLightEnv"]

    return env_base_params, env_specific_params


def load_sweep_config(path: str, env_id: str, algorithm: str) -> Dict[str, Any]:
    """加载 W&B 超参数搜索 (Sweep) 的配置文件"""
    with open(join(path, algorithm + ".yml"), "r") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)
    return sweep_config[env_id]


def loadParameters(env_id: str, path: str, filename: str, algorithm: Optional[str] = None):
    """从 YAML 文件中加载完整的仿真、状态和模型参数"""
    with open(join(path, filename), "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if env_id != "GreenLightEnv":
        envSpecificParams = params[env_id]
    else:
        envSpecificParams = {}

    envBaseParams = params["GreenLightEnv"]
    options = params["options"]

    state_columns = params["state_columns"]
    action_columns = params["action_columns"]

    # 如果提供了算法名称，加载并解析算法相关的超参数
    if algorithm is not None:
        model_params = params[algorithm]

        if "policy_kwargs" in model_params.keys():
            # 转换激活函数、优化器和解析 log_std_init
            model_params["policy_kwargs"]["activation_fn"] = \
                ACTIVATION_FN[model_params["policy_kwargs"]["activation_fn"]]
            model_params["policy_kwargs"]["optimizer_class"] = \
                OPTIMIZER[model_params["policy_kwargs"]["optimizer_class"]]
            model_params["policy_kwargs"]["log_std_init"] = \
                eval(model_params["policy_kwargs"]["log_std_init"])
    else:
        model_params = None
    return envBaseParams, envSpecificParams, model_params, options, state_columns, action_columns


def set_model_params(config):
    """根据 W&B config 配置模型字典（主要用于 Sweep 调优）"""
    model_params = {}
    policy_kwargs = {}
    policy_kwargs['activation_fn'] = ACTIVATION_FN[config['activation_fn']]
    # 设置网络架构层级大小
    policy_kwargs['net_arch'] = {"pi": [config["pi_size"]], "vf": [config["vf_size"]]}
    policy_kwargs['optimizer_class'] = OPTIMIZER[config["optimizer_class"]]
    policy_kwargs['optimizer_kwargs'] = config['optimizer_kwargs']
    policy_kwargs['log_std_init'] = np.log(config['std_init'])

    model_params["policy_kwargs"] = policy_kwargs

    # 设置训练相关的超参数
    model_params['batch_size'] = config['batch_size']
    model_params['n_steps'] = config['n_steps']
    model_params['n_epochs'] = config['n_epochs']
    model_params['learning_rate'] = config['learning_rate']
    model_params['gamma'] = config['gamma']
    model_params['gae_lambda'] = config['gae_lambda']
    model_params['policy'] = config['policy']
    model_params['normalize_advantage'] = config['normalize_advantage']
    model_params['ent_coef'] = config['ent_coef']
    model_params['vf_coef'] = config['vf_coef']
    model_params['max_grad_norm'] = config['max_grad_norm']
    model_params['use_sde'] = config['use_sde']
    model_params['sde_sample_freq'] = config['sde_sample_freq']
    model_params['target_kl'] = None

    return model_params


def wandb_init(hyperparameters: Dict[str, Any],
               env_seed: int,
               model_seed: int,
               project: str,
               group: str,
               save_code: bool = False,
               ):
    """初始化 WandB 实验跟踪记录"""
    config = {
        "env_seed": env_seed,
        "model_seed": model_seed,
        **hyperparameters,
    }

    run = wandb.init(
        project=project,
        config=config,
        group=group,
        sync_tensorboard=True,  # 同步 TensorBoard 日志到 WandB 云端
        save_code=save_code,
        allow_val_change=True,
    )
    return run, config


def create_callbacks(n_eval_episodes: int,
                     eval_freq: int,
                     env_log_dir: str | None,
                     save_name: str,
                     model_log_dir: str | None,
                     eval_env: VecEnv,
                     run: Any | None = None,
                     results: Results | None = None,
                     save_env: bool = True,
                     verbose: int = 1,
                     ) -> List[BaseCallback]:
    """创建训练期间的回调函数列表"""
    # 如果提供了路径，创建环境标准化的保存回调
    if env_log_dir:
        save_vec_best = SaveVecNormalizeCallback(save_freq=1, save_path=env_log_dir, verbose=2)
    else:
        save_vec_best = None

    # 创建自定义的 WandB 评估回调，负责定期测试智能体并保存最佳模型
    eval_callback = CustomWandbCallback(eval_env,
                                        n_eval_episodes=n_eval_episodes,
                                        eval_freq=eval_freq,
                                        best_model_save_path=model_log_dir,
                                        name_vec_env=save_name,
                                        path_vec_env=env_log_dir,
                                        deterministic=True,
                                        callback_on_new_best=save_vec_best,
                                        run=run,
                                        results=results,
                                        verbose=verbose)
    # SB3 官方的 WandB 回调
    wandbcallback = WandbCallback(verbose=verbose)
    return [eval_callback, wandbcallback]


def controlScheme(GL, nightValue, dayValue):
    """
    用于测试特定变量控制效果的函数（如全天固定值控制）。
    """
    obs, info = GL.reset()
    GL.GLModel.setNightCo2(nightValue)
    N = GL.N  # 运行步数
    states = np.zeros((N + 1, GL.modelObsVars))  # 保存状态的数组
    controlSignals = np.zeros((N + 1, GL.GLModel.nu))  # 保存控制信号的数组
    states[0, :] = obs[:GL.modelObsVars]  # 获取初始状态
    timevec = np.zeros((N + 1,))  # 时间向量
    timevec[0] = GL.GLModel.time
    i = 1

    while not GL.terminated:
        # 根据天气数据判断是白天还是黑夜
        if GL.weatherData[GL.GLModel.timestep * GL.solverSteps, 9] > 0:
            controls = np.ones((GL.action_space.shape[0],)) * dayValue
        else:
            controls = np.ones((GL.action_space.shape[0],)) * nightValue
        obs, r, terminated, _, info = GL.step(controls.astype(np.float32))
        states[i, :] += obs[:GL.modelObsVars]
        controlSignals[i, :] += info["controls"]
        timevec[i] = info["Time"]
        i += 1

    # 将结果转换为 Pandas DataFrame 格式以便后续处理
    states = np.insert(states, 0, timevec, axis=1)
    states = pd.DataFrame(data=states[:],
                          columns=["Time", "空气温度", "CO2浓度", "湿度", "果实重量", "果实采收", "PAR"])
    controlSignals = pd.DataFrame(data=controlSignals,
                                  columns=["加热", "CO2", "遮阳网", "通风", "补光", "间层灯", "生长管", "黑幕"])
    weatherData = pd.DataFrame(
        data=GL.weatherData[[int(ts * GL.solverSteps) for ts in range(0, GL.Np + 1)], :GL.weatherObsVars],
        columns=["温度", "湿度", "PAR", "CO2浓度", "风速"])

    return states, controlSignals, weatherData


def runRuleBasedController(GL, stateColumns, actionColumns):
    """运行基于规则的基础控制器"""
    obs, info = GL.reset()
    N = GL.N
    states = np.zeros((N + 1, GL.modelObsVars))
    controlSignals = np.zeros((N + 1, GL.GLModel.nu))
    states[0, :] = obs[:GL.modelObsVars]
    timevec = np.zeros((N + 1,))
    timevec[0] = GL.GLModel.time
    i = 1
    while not GL.terminated:
        # 这里使用 0.5 作为演示，实际可根据逻辑定义
        controls = np.ones((GL.action_space.shape[0],)) * 0.5
        obs, r, terminated, _, info = GL.step(controls.astype(np.float32))
        states[i, :] += obs[:GL.modelObsVars]
        controlSignals[i, :] += info["controls"]
        timevec[i] = info["Time"]
        i += 1

    states = np.insert(states, 0, timevec, axis=1)
    states = pd.DataFrame(data=states[:], columns=["Time"] + stateColumns)
    controlSignals = pd.DataFrame(data=controlSignals, columns=actionColumns)
    return states, controlSignals


def runSimulationDefinedControls(GL, matlabControls, stateNames, matlabStates, nx):
    """根据预先定义好的（如来自 Matlab）的控制信号运行仿真"""
    obs, info = GL.reset()
    N = matlabControls.shape[0]

    cythonStates = np.zeros((N, nx))
    cyhtonControls = np.zeros((N, GL.GLModel.nu))
    cythonStates[0, :] = GL.GLModel.getStatesArray()

    for i in range(1, N):
        controls = matlabControls.iloc[i, :].values
        obs, reward, terminated, truncated, info = GL.step(controls)
        cythonStates[i, :] += GL.GLModel.getStatesArray()
        cyhtonControls[i, :] += info["controls"]

        if terminated:
            break

    cythonStates = pd.DataFrame(data=cythonStates, columns=stateNames[:])
    return cythonStates, cyhtonControls