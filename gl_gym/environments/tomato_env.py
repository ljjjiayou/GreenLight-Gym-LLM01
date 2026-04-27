from typing import Any, Dict, List, Optional, Tuple, SupportsFloat

import numpy as np
import casadi as ca
from gymnasium import spaces

from gl_gym.environments.base_env import GreenLightEnv
from gl_gym.environments.observations import *
from gl_gym.environments.rewards import BaseReward, GreenhouseReward
# from gl_gym.environments.models.greenlight_model import GreenLight

from gl_gym.environments.models.utils import define_model

from gl_gym.environments.utils import load_weather_data, init_state
from gl_gym.environments.parameters import init_default_params
from gl_gym.environments.noise import parametric_crop_uncertainty

# 可选的奖励函数映射表
REWARDS = {"GreenhouseReward": GreenhouseReward}

# 可选的观测模块映射表
OBSERVATION_MODULES = {
    "StateObservations": StateObservations,
    "IndoorClimateObservations": IndoorClimateObservations,
    "BasicCropObservations": BasicCropObservations,
    "ControlObservations": ControlObservations,
    "WeatherObservations": WeatherObservations,
    "WeatherForecastObservations": WeatherForecastObservations,
    "TimeObservations": TimeObservations
}


class TomatoEnv(GreenLightEnv):
    def __init__(self,
                 reward_function: str,  # 奖励函数类型
                 observation_modules: List[str],  # 观测模块列表
                 constraints: Dict[str, Any],  # 环境约束条件（温度、CO2、湿度等）
                 eval_options: Dict[str, Any],  # 评估相关的配置选项（天数、年份等）
                 reward_params: Dict[str, Any] = {},  # 奖励函数的具体参数
                 base_env_params: Dict[str, Any] = {},  # 基类环境的参数
                 uncertainty_scale=0.0  # 参数不确定性的缩放比例
                 ) -> None:
        # 调用基类构造函数
        super(TomatoEnv, self).__init__(**base_env_params)

        self.uncertainty_scale = uncertainty_scale

        # 设置评估环境的年份和天数
        self.eval_options = eval_options

        # 初始化观测空间和动作空间
        self.observation_modules = self._init_observations(observation_modules)
        self.observation_space = self._generate_observation_space()
        self.action_space = self._generate_action_space()

        # 定义物理模型（使用 CasADi 加载 ODE 模型）
        self.F = define_model(
            nx=self.nx,
            nu=self.nu,
            nd=self.nd,
            n_params=self.num_params,
            dt=self.dt,
        )

        # 设置环境约束的下限
        self.constraints_low = np.array([
            constraints["co2_min"],
            constraints["temp_min"],
            constraints["rh_min"],
        ])

        # 设置环境约束的上限
        self.constraints_high = np.array([
            constraints["co2_max"],
            constraints["temp_max"],
            constraints["rh_max"],
        ])

        # 初始化默认模型参数
        self.p = init_default_params(self.num_params)

        # 初始化奖励函数对象
        self.reward = self._init_rewards(reward_function, reward_params)

    def _terminalState(self) -> bool:
        """
        检查模拟是否达到终止状态。
        当模拟步数达到生长季结束步数时，返回 True。
        """
        if self.timestep >= self.N:
            return True
        return False

    def _init_observations(
            self,
            observation_modules: List[str],
    ) -> List[BaseObservations]:
        """
        根据配置列表初始化观测模块。
        """
        return [OBSERVATION_MODULES[module](self) for module in observation_modules]

    def _generate_observation_space(self) -> spaces.Box:
        """
        遍历所有观测模块，拼接各自的范围，生成统一的 Box 观测空间。
        """
        spaces_low_list = []
        spaces_high_list = []

        for module in self.observation_modules:
            module_obs_space = module.observation_space()
            spaces_low_list.append(module_obs_space.low)
            spaces_high_list.append(module_obs_space.high)

        low = np.concatenate(spaces_low_list, axis=0)  # 拼接下限
        high = np.concatenate(spaces_high_list, axis=0)  # 拼接上限

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _generate_action_space(self) -> spaces.Box:
        """
        生成动作空间，范围通常在 [-1, 1] 之间。
        """
        return spaces.Box(low=-1, high=1, shape=(self.nu,), dtype=np.float32)

    def _init_rewards(self, reward_function: str, reward_params: Dict[str, Any]) -> BaseReward:
        """
        初始化奖励函数。
        """
        return REWARDS[reward_function](self, **reward_params)

    def _get_reward(self) -> SupportsFloat:
        """
        计算并返回当前步的奖励。
        """
        return self.reward.compute_reward()

    def _scale(self, action, action_min, action_max):
        """
        将控制器动作（[-1, 1]）线性映射到实际控制范围（[min, max]）。
        """
        return (action + 1) * (action_max - action_min) / 2 + action_min

    def action_to_control(self, action: np.ndarray) -> np.ndarray:
        """
        将智能体输出的动作转换为环境的实际控制输入。
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.nu:
            raise ValueError(f"action 维度应为 {self.nu}，实际为 {action.shape[0]}")
        delta_u_max = np.asarray(self.delta_u_max, dtype=np.float32).reshape(-1)
        if delta_u_max.shape[0] != self.nu:
            raise ValueError(f"delta_u_max 维度应为 {self.nu}，实际为 {delta_u_max.shape[0]}")
        return np.clip(self.u + action * delta_u_max, self.u_min, self.u_max)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        执行一个时间步。
        """
        # 将智能体的动作（[-1, 1]）转换为实际控制量（如阀门开度、CO2流量等）
        self.u = self.action_to_control(action)
        # 根据不确定性缩放比例随机生成作物参数，用于模拟现实中的变异
        params = parametric_crop_uncertainty(self.p, self.uncertainty_scale, self._np_random)
        try:
            # 拼接当前时刻的天气干扰数据和模型参数
            p_dyn = ca.vertcat(ca.DM(self.weather_data[self.timestep]), params)
            # 运行物理引擎求解下一时刻状态
            res = self.F(x0=ca.DM(self.x), u=ca.DM(self.u), p=p_dyn)
            self.x = res["xf"].full().flatten()
        except:
            print("ODE 近似求解出错")
            self.terminated = True

        # 更新模拟时间（日期和小时）
        self.day_of_year += (self.dt / self.c) % 365
        self.hour_of_day += (self.dt / 3600)
        self.hour_of_day = self.hour_of_day % 24

        # 获取当前步的观测值
        self.obs = self._get_obs()
        if self._terminalState():
            self.terminated = True
        # 计算奖励值
        reward = self._get_reward()
        # 获取用于日志记录或 W&B 监控的附加信息
        info = self._get_info()
        self.timestep += 1
        self.x_prev = np.copy(self.x)

        return (
            self.obs,
            reward,
            self.terminated,
            False,
            info
        )

    def step_raw_control(self, control: np.ndarray):
        """
        使用原始控制量直接执行步进（不通过 [-1, 1] 映射），常用于基准控制器（如 Rule-based）。
        """
        self.u = control
        params = parametric_crop_uncertainty(self.p, self.uncertainty_scale, self._np_random)
        p_dyn = ca.vertcat(ca.DM(self.weather_data[self.timestep]), params)
        res = self.F(x0=ca.DM(self.x), u=ca.DM(self.u), p=p_dyn)
        self.x = res["xf"].full().flatten()

        # 更新时间
        self.day_of_year += (self.dt / self.c) % 365
        self.hour_of_day += (self.dt / 3600)
        self.hour_of_day = self.hour_of_day % 24

        self.obs = self._get_obs()

        if self._terminalState():
            self.terminated = True
        # 计算奖励
        reward = self._get_reward()
        self.timestep += 1
        self.x_prev = np.copy(self.x)
        # 返回评估所需的完整信息
        return (
            self.obs,
            reward,
            self.terminated,
            False,
            self._get_info()
        )

    def step_raw_control_pipeinput(self, control: np.ndarray):
        """
        用于特定的管道化输入模式下的步进逻辑。
        """
        self.u = control

        self.x = self.F(self.x, self.u, self.weather_data[self.timestep], self.p)

        if self._terminalState():
            self.terminated = True

        reward = self._get_reward()

        self.timestep += 1
        return (
            self.x,
            self.terminated,
        )

    def _get_obs(self):
        """
        从所有已加载的观测模块中提取当前步的数据，并合并为一个向量。
        """
        obs = []
        for module in self.observation_modules:
            obs.append(module.compute_obs())
        obs = np.concatenate(obs, axis=0)
        return obs

    def get_obs_names(self):
        """
        获取当前观测空间中所有变量的名称（用于绘图或数据分析）。
        """
        obs_names = []
        for module in self.observation_modules:
            obs_names.extend(module.obs_names)
        return obs_names

    def _get_info(self) -> Dict[str, Any]:
        """
        生成当前状态的详细信息字典，包含利润、收益、各项成本及环境约束违规情况。
        """
        return {
            "EPI": self.reward.profit,  # 经济利润指标
            "revenue": self.reward.gains,  # 作物销售带来的收益
            "variable_costs": self.reward.variable_costs,  # 可变成本总计
            "fixed_costs": self.reward.fixed_costs,  # 固定成本
            "co2_cost": self.reward.co2_costs,  # CO2 增施成本
            "heat_cost": self.reward.heat_costs,  # 加热能源成本
            "elec_cost": self.reward.elec_costs,  # 电力成本
            "temp_violation": self.reward.temp_violation,  # 温度约束违规程度
            "co2_violation": self.reward.co2_violation,  # CO2 约束违规程度
            "rh_violation": self.reward.rh_violation,  # 相对湿度约束违规程度
            "lamp_violation": self.reward.lamp_violation,  # 补光时长违规程度
            "controls": self.u,  # 当前执行的实际控制量向量
        }

    def set_crop_state(
            self,
            cBuf: float,
            cLeaf: float,
            cStem: float,
            cFruit: float,
            tCanSum: float
    ) -> None:
        """
        手动设置作物的生长状态变量（用于特定场景下的初始化）。
        """
        self.x[22] = cBuf  # 缓冲碳水化合物
        self.x[23] = cLeaf  # 叶片干重
        self.x[24] = cStem  # 茎干干重
        self.x[25] = cFruit  # 果实干重
        self.x[26] = tCanSum  # 冠层积温

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境至初始状态。
        """
        super().reset(seed=seed)

        # 确保随机数生成器已初始化
        if self._np_random is None:
            if hasattr(self, 'np_random') and self.np_random is not None:
                self._np_random = self.np_random   # 使用 Gymnasium 的标准属性
            else:
                self._np_random = np.random.default_rng(seed)

        # 如果在训练模式，随机选择一个训练年份和起始天
        if self.training:
            self.growth_year = self._np_random.choice(self.train_years)
            self.start_day = self._np_random.choice(self.train_days)
        else:
            # 如果在评估模式，根据配置选择评估年份和天数
            self.growth_year = self._np_random.choice(self.eval_options["eval_years"])
            self.start_day = self._np_random.choice(self.eval_options["eval_days"])
            self.location = self.eval_options["location"]
            self.increase_eval_idx()

        self.day_of_year = self.start_day
        self.hour_of_day = 0

        # 为当前模拟加载具体的天气数据
        self.weather_data = load_weather_data(
            self.weather_data_dir,
            self.location,
            self.growth_year,
            self.start_day,
            self.season_length,
            self.Np + 1,
            self.dt,
            self.nd
        )

        # 初始化控制量、系统状态和观测值
        self.u = np.zeros(self.nu)
        self.x = init_state(self.weather_data[0])
        self.x_prev = np.copy(self.x)
        self.timestep = 0
        self.obs = self._get_obs()

        self.terminated = False
        return self.obs, {}
