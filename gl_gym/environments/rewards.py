"""
rewards.py 定义了强化学习智能体的“目标函数”。它将温室的物理状态和经济投入转化为一个数值信号（奖励），
指导智能体学习如何在最大化收益的同时减少能源消耗并遵守环境约束。
"""
from abc import ABC, abstractmethod
from typing import SupportsFloat, List, Optional

import numpy as np


class BaseReward(ABC):
    """奖励函数基类"""
    profit: float  # 利润
    fixed_costs: float  # 固定成本
    variable_costs: float  # 可变成本
    gains: float  # 收益

    @abstractmethod
    def compute_reward(self) -> SupportsFloat:
        """子类必须实现的计算奖励的方法"""
        pass

    def scale_reward(self, r: float, min_r, max_r) -> SupportsFloat:
        """将奖励信号标准化到 [0, 1] 附近的区间"""
        return (r - min_r) / (max_r - min_r)


class GreenhouseReward(BaseReward):
    """
    GreenLight 环境的经济奖励函数。
    奖励计算为收益与成本之差。
    收益根据每天每盆果实的生长量乘以果实价格计算。
    成本计算为加热、二氧化碳施用、高峰和非高峰电费的总和。
    同时考虑了温室、二氧化碳设备、灯具、屏幕和间距的固定成本。

    参数:
        fixed_greenhouse_cost (float): 温室固定成本 [€/m2/年]
        fixed_co2_cost (float): 二氧化碳设备固定成本 [€/m2/年]
        fixed_lamp_cost (float): 灯具固定成本 [€/m2/年]
        fixed_screen_cost (float): 遮阳网固定成本 [€/m2/年]
        elec_price (float): 电价 [€/kWh]
        heating_price (float): 加热价格 [€/kWh]
        co2_price (float): 二氧化碳价格 [€/kg]
        fruit_price (float): 果实价格 [€/kg]
        pen_weights (List[float]): 各项约束违规的惩罚权重
        pen_lamp (float): 违规用灯的惩罚
        dmfm (float): 干物质到鲜物质的转换率
        env: 环境对象引用
    """

    def __init__(
            self,
            env,
            fixed_greenhouse_cost: float,
            fixed_co2_cost: float,
            fixed_lamp_cost: float,
            fixed_screen_cost: float,
            elec_price: float,
            heating_price: float,
            co2_price: float,
            fruit_price: float,
            pen_weights: List[float],
            pen_lamp: float,
            dmfm: float,
            include_fixed_costs_in_reward: bool = False,
    ) -> None:
        super(GreenhouseReward, self).__init__()

        self.env = env
        # 设置各项固定成本
        self.fixed_greenhouse_cost = fixed_greenhouse_cost
        self.fixed_co2_cost = fixed_co2_cost
        self.fixed_lamp_cost = fixed_lamp_cost * 116  # 乘以每盏灯的最大强度
        self.fixed_screen_cost = fixed_screen_cost

        # 计算年度总固定成本并折算到每个时间步
        yearly_fixed_costs = sum(
            [self.fixed_greenhouse_cost, self.fixed_co2_cost, self.fixed_lamp_cost, self.fixed_screen_cost])
        self.fixed_costs = self._fixed_costs_timestep(yearly_fixed_costs)

        # 能源和物资的变量价格
        self.elec_price = elec_price  # €/kWh
        self.heating_price = heating_price  # €/kWh
        self.co2_price = co2_price  # €/kg

        self.fruit_price = fruit_price  # €/kg
        self.dmfm = dmfm  # 干鲜比转化系数
        self.pen_weights = np.array(pen_weights)  # 惩罚权重数组
        self.pen_lamp = pen_lamp  # 灯具违规惩罚值
        self.include_fixed_costs_in_reward = include_fixed_costs_in_reward
        self._obs_name_to_idx = {}
        if hasattr(self.env, "get_obs_names"):
            names = self.env.get_obs_names()
            self._obs_name_to_idx = {name: idx for idx, name in enumerate(names)}

        # 初始化状态
        self._init_costs()
        self._init_violations()

        # 计算理论上的最大/最小利润及违规界限，用于标准化
        self.max_profit = self.max_profit_reward()
        self.min_profit = self.min_profit_reward()
        self.min_state_violations = self.min_violations()
        self.max_state_violations = self.max_violations()

    def min_violations(self):
        """最小违规量定义为 0"""
        return np.zeros(3)

    def max_violations(self):
        """定义各环境指标的最大可能违规程度"""
        co2_violation = 2500  # CO2 浓度偏离上限/下限的最大值
        temp_violation = 15  # 温度偏离值
        rh_violation = 15  # 湿度偏离值
        return np.array([co2_violation, temp_violation, rh_violation])

    def max_profit_reward(self):
        """
        计算当前时间步可能达到的最大奖励（最大收益 - 最小成本）。
        """
        # 基于模型参数 p[154] 计算最大的理论果实生长收益
        max_gains = self.env.p[154] * self.env.dt * 1e-6 / self.dmfm * self.fruit_price
        return max_gains

    def min_profit_reward(self):
        """
        计算当前时间步可能达到的最小奖励（最小收益 - 最大成本）。
        """
        # 计算全负荷运行时的最大供暖、电力和二氧化碳成本
        max_heating = self.env.p[108] / self.env.p[46] * self.env.dt / 3600 * 1e-3 * self.heating_price
        max_elec = self.env.p[172] * self.env.dt / 3600 * 1e-3 * self.elec_price
        max_cost = self.env.p[109] / self.env.p[46] * self.env.dt * 1e-6 * self.co2_price
        max_costs = sum([max_heating, max_elec, max_cost])
        return -max_costs

    def _init_violations(self):
        """重置所有违规计数器"""
        self.temp_violation = 0
        self.co2_violation = 0
        self.rh_violation = 0
        self.lamp_violation = 0

    def _init_costs(self):
        """重置所有成本和收益记录"""
        self.variable_costs = 0
        self.gains = 0
        self.profit = 0
        self.heat_costs = 0
        self.co2_costs = 0
        self.elec_costs = 0

    def _fixed_costs_daily(self):
        """将年度固定成本转换为每日成本"""
        return self.yearly_fixed_costs / 365.

    def _fixed_costs_timestep(self, yearly_fixed_costs):
        """将年度固定成本转换为每个模拟步长的成本"""
        return yearly_fixed_costs / 365 / (86400 // self.env.dt)

    def _variable_costs(self):
        """
        根据 GreenLight 环境的当前控制动作计算变量成本。
        计算包括加热能耗、电力使用和二氧化碳投放量。
        返回总变量成本（单位：€/m2/时间步）。
        """
        # u[0] 是加热，u[4] 是补光灯，u[1] 是 CO2
        heating_energy = self.env.u[0] * self.env.p[108] / self.env.p[46] * self.env.dt / 3600 * 1e-3
        elec_use = self.env.u[4] * self.env.p[172] * self.env.dt / 3600 * 1e-3
        co2_dosing = self.env.u[1] * self.env.p[109] / self.env.p[46] * self.env.dt * 1e-6

        self.heat_costs = heating_energy * self.heating_price
        self.co2_costs = co2_dosing * self.co2_price
        self.elec_costs = elec_use * self.elec_price
        return sum([self.heat_costs, self.co2_costs, self.elec_costs])

    def _gains(self):
        """
        计算当前时间步的收益。
        1. 计算果实干重（DW）的变化（x[25] 是果实碳含量）
        2. 使用 dmfm 因子将干重转换为鲜重（FFW）
        3. 鲜重增长量乘以果实单价得出收益
        """
        fruit_growth_dm = self.env.x[25] - self.env.x_prev[25]
        fruit_growth_ffw = fruit_growth_dm * 1e-6 / self.dmfm
        return fruit_growth_ffw * self.fruit_price

    def output_violations(self):
        """
        计算违反环境约束（温度、CO2、湿度）的绝对数值惩罚。
        对比环境观测值与预设的上下限。
        """
        co2_air = self._obs_value("co2_air", 0)
        temp_air = self._obs_value("temp_air", 1)
        rh_air = self._obs_value("rh_air", 2)
        state_vec = np.array([co2_air, temp_air, rh_air], dtype=np.float32)

        lowerbound = self.env.constraints_low[:] - state_vec
        lowerbound[lowerbound < 0] = 0
        upperbound = state_vec - self.env.constraints_high[:]
        upperbound[upperbound < 0] = 0

        self.co2_violation = lowerbound[0] + upperbound[0]
        self.temp_violation = lowerbound[1] + upperbound[1]
        self.rh_violation = lowerbound[2] + upperbound[2]
        return lowerbound + upperbound

    def output_penalty_reward(self, violations):
        """计算加权后的违规总惩罚"""
        return np.dot(self.pen_weights, violations)

    def control_violation(self):
        """
        控制策略检查：检查是否在夜间（晚上 8 点后）使用了灯具。
        """
        if self.env.hour_of_day >= 20:
            if self.env.u[4] > 0:
                self.lamp_violation = 1
                return
        self.lamp_violation = 0

    def _obs_value(self, name: str, fallback_idx: int) -> float:
        idx = self._obs_name_to_idx.get(name, fallback_idx)
        obs = np.asarray(self.env.obs)
        if obs.size == 0:
            return 0.0
        idx = min(max(int(idx), 0), obs.size - 1)
        return float(obs[idx])

    def control_penalty(self):
        """返回控制违规的惩罚值"""
        self.control_violation()
        return self.lamp_violation * self.pen_lamp

    def compute_reward(self) -> SupportsFloat:
        """
        执行完整的奖励计算逻辑：
        1. 计算利润 = 收益 - 变量成本
        2. 计算环境和策略惩罚
        3. 标准化利润和惩罚并返回最终奖励值
        """
        self.variable_costs = self._variable_costs()
        self.gains = self._gains()
        self.profit = self.gains - self.variable_costs
        if self.include_fixed_costs_in_reward:
            self.profit -= self.fixed_costs

        violations = self.output_violations()
        self.penalty = self.output_penalty_reward(violations)
        self.control_pen = self.control_penalty()

        # 标准化奖励信号，使其更适合神经网络训练
        scaled_profit = self.scale_reward(self.profit, self.min_profit, self.max_profit)
        scaled_pen = np.sum(self.scale_reward(violations, self.min_state_violations, self.max_state_violations))

        return scaled_profit - scaled_pen - self.control_pen
