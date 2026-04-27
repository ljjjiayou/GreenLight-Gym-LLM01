from datetime import date
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

# 导入自定义的观测和奖励基类
from gl_gym.environments.observations import BaseObservations
from gl_gym.environments.rewards import BaseReward


class GreenLightEnv(gym.Env):
    """
    Python 基类，作为 Python 与 C++ 绑定（或原生 Python 模型）之间的包装器环境。

    参数:
        weather_data_dir (str): 天气数据存放路径
        location (str): 天气记录的具体地点名称
        nx (int): 状态变量的数量
        nd (int): 外部干扰项的数量
        h (float): [秒] GreenLight 求解器的时间步长
        time_interval (int): [秒] 两次观测之间的时间间隔
        pred_horizon (int): [天] 未来天气预测的天数
        n_setpoints (int): 控制的设定点数量
        season_length (int): 模拟的总天数（一个生长季的长度）。默认为 350 天。
        start_train_year (int, 可选): 训练的起始年份。默认为 2023。
        end_train_year (int, 可选): 训练的结束年份。默认为 2023。
        start_train_day (int, 可选): 训练在一年中的起始天数。默认为 265。
        end_train_day (int, 可选): 训练在一年中的结束天数。默认为 284。
        reward_function (str, 可选): 使用的奖励函数名称。
        training (bool, 可选): 标识当前是处于训练阶段还是测试阶段。默认为 True。
        options (Dict[str, Any], 可选): 传递给 GreenLight 模型的配置选项。
    """

    # 类型提示：模型、生长年份、起始天数和奖励对象
    # gl_model: GreenLight 仿真模型实例
    growth_year: int
    start_day: int
    reward_function: BaseReward

    def __init__(
            self,
            weather_data_dir: str,  # 天气数据目录
            location: str,  # 地点
            num_params: int,  # 模型参数的数量
            nx: int,  # 状态变量数量
            nu: int,  # 控制输入数量
            nd: int,  # 干扰项数量
            dt: float,  # [秒] 底层求解器的步长
            u_min: List[float],  # 控制输入的下限
            u_max: List[float],  # 控制输入的上限
            delta_u_max: Any,  # 控制输入的单步最大变化量
            pred_horizon: int,  # [天] 天气预测长度
            season_length: int = 60,  # 生长季总天数
            start_train_year: int = 2023,  # 训练起始年
            end_train_year: int = 2023,  # 训练结束年
            start_train_day: int = 265,  # 训练起始天
            end_train_day: int = 284,  # 训练结束天
            training: bool = True,  # 是否处于训练模式
    ) -> None:
        super(GreenLightEnv, self).__init__()

        # 一天的总秒数
        self.c = 86400

        # 在不同模拟中保持不变的基础参数
        self.num_params = num_params
        self.nx = nx
        self.nu = nu
        self.nd = nd
        self.u_min = np.array(u_min, dtype=np.float32)
        self.u_max = np.array(u_max, dtype=np.float32)
        if isinstance(delta_u_max, (list, tuple, np.ndarray)):
            delta_u_array = np.array(delta_u_max, dtype=np.float32).reshape(-1)
            if delta_u_array.shape[0] != self.nu:
                raise ValueError(f"delta_u_max 长度应为 {self.nu}，实际为 {delta_u_array.shape[0]}")
            self.delta_u_max = delta_u_array
        else:
            self.delta_u_max = np.ones(self.nu, dtype=np.float32) * float(delta_u_max)
        self.weather_data_dir = weather_data_dir
        self.location = location
        self.dt = dt
        self.pred_horizon = pred_horizon
        # 将预测天数转换为对应的模型步数 (Np)
        self.Np = int(self.pred_horizon * self.c / self.dt)

        # 训练年份和天数的列表
        self.train_years = list(range(start_train_year, end_train_year + 1))
        self.train_days = list(range(start_train_day, end_train_day + 1))

        self.training = training
        self.eval_idx = 0  # 评估计数器

        self.season_length = season_length
        # 将生长季长度转换为总步数 (N)
        self.N = int(self.season_length * self.c / self.dt)

        # 观测值的上下限（空气温度、CO2浓度、湿度等）
        self.obs_low = None
        self.obs_high = None

    @abstractmethod
    def step(self, action: np.ndarray
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        环境推进函数。
        输入：动作 (action)
        输出：观测值 (obs), 奖励 (reward), 是否终止 (done), 是否截断 (truncated), 附加信息 (info)
        """
        pass

    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """
        获取当前环境状态的详细附加信息。
        """
        pass

    @abstractmethod
    def _init_rewards(self) -> BaseReward:
        """
        初始化环境的奖励函数对象。
        """
        pass

    @abstractmethod
    def _init_observations(
            self,
            model_obs_vars: Optional[List[str]] = None,
            weather_obs_vars: Optional[List[str]] = None,
            Np: Optional[int] = None
    ) -> List[BaseObservations]:
        """
        初始化环境的观测模块。
        """
        pass

    @abstractmethod
    def _generate_observation_space(self) -> spaces.Dict:
        """
        生成 Gymnasium 标准的观测空间定义。
        """
        pass

    @abstractmethod
    def _get_obs(self):
        """
        获取当前步的观测数据。
        """
        pass

    @abstractmethod
    def _terminalState(self) -> bool:
        """
        判断是否到达终止状态。
        """
        pass

    def _get_time(self):
        """
        从底层的 GreenLight 模型获取当前时间（秒）。
        """
        return self.gl_model.get_time()

    def _get_time_in_days(self) -> float:
        """
        获取从公元 0001-01-01 到模拟起始日期的总天数。
        用于对齐时间尺度。
        """
        d0 = date(1, 1, 1)
        d1 = date(self.growth_year, 1, 1)
        delta = d1 - d0
        return delta.days + self.start_day

    def _scale(self, action, action_min, action_max):
        """
        最小-最大缩放器，将动作映射到 [0, 1] 区间。常用于标准化动作空间。
        """
        return (action - action_min) / (action_max - action_min)

    def _reset_eval_idx(self):
        """重置评估计数器。"""
        self.eval_idx = 0

    def increase_eval_idx(self):
        """递增评估计数器。"""
        self.eval_idx += 1

    def set_seed(self, seed):
        """
        设置环境的随机种子。
        """
        self._np_random, self._np_random_seed = seeding.np_random(seed)

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境的容器函数。
        调用 super().reset(seed=seed) 来确保 Gymnasium 内部状态重置。
        """
        super().reset(seed=seed)
        return np.array([]), {}
