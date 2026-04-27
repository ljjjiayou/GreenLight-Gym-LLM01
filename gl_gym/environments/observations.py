"""
observations.py 定义了强化学习智能体的“眼睛”。它负责从复杂的温室物理模拟中提取出最有用的信息，
并将其转化为神经网络能够理解的格式（通常是标准化的数值向量）。
"""
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from gymnasium import spaces

from gl_gym.environments.utils import co2dens2ppm, vaporPres2rh

class BaseObservations(ABC):
    """
    观测者基类，用于控制强化学习智能体的观测值（即输入）。
    可以从 GreenLight 模型中构造观测值，也可以从当前和未来的天气数据中构造。
    模型观测值在 GreenLight 中计算，天气观测值从天气数据数组中提取。
    """
    def __init__(self,
                ) -> None:
        self.n_obs = None       # 观测值的数量
        self.low = None         # 观测值的下限
        self.high = None        # 观测值的上限
        self.obs_names = None   # 观测变量的名称

    @abstractmethod
    def observation_space(self) -> spaces.Box:
        """定义 Gymnasium 标准的观测空间"""
        pass

    @abstractmethod
    def compute_obs(self) -> np.ndarray:
        """
        计算并从 GreenLight 模型和天气数据中检索观测值。
        """
        pass


class StateObservations(BaseObservations):
    """
    全状态观测类，提供对所有模型状态变量的访问。
    """
    def __init__(self) -> None:
        self.obs_names = ["co2_air", "co2_top", "temp_air", "temp_top", "can_temp", "covin_temp", "covex_temp",
                                "thScr_temp", "flr_temp", "pipe_temp", "soil1_temp", "soil2_temp", "soil3_temp", "soil4_temp", "soil5_temp",
                                "vp_air", "vp_top", "lamp_temp", "intlamp_temp", "grow_pipe_temp", "blscr_temp", "24_can_temp",
                                "cBuf", "cleaves", "cstem", "cFruit", "tsum"]
        self.n_obs = len(self.obs_names)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        计算并检索全状态观测值（当前实现为随机值，用于示例）。
        """
        return np.random.rand(self.n_obs)

class IndoorClimateObservations(BaseObservations):
    """
    室内气候观测模块，关注空气温度、CO2、湿度和加热管温度。
    """
    def __init__(self, env) -> None:
        self.env = env
        self.obs_names = ["co2_air", "temp_air","rh_air",  "pipe_temp"]
        self.n_obs = len(self.obs_names)

    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        从物理模型状态中提取数据，并将 CO2 密度转换为 ppm，水蒸气压转换为相对湿度。
        """
        climate_obs = np.array(self.env.x)[[0, 2, 15, 9]]
        # 转换 CO2 单位
        climate_obs[0] = co2dens2ppm(climate_obs[1], climate_obs[0]*1e-6)
        # 转换湿度单位
        climate_obs[2] = vaporPres2rh(climate_obs[1], climate_obs[2])
        return climate_obs

class BasicCropObservations(BaseObservations):
    """
    基础作物观测模块，关注 24 小时平均温、果实重量和积温。
    """
    def __init__(self, env) -> None:
        self.env = env
        self.obs_names = ["24CanTemp", "cFruit", "tSum"]
        self.n_obs = len(self.obs_names)

    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        从状态向量 x 中提取作物生长相关的变量。
        """
        crop_obs = np.array(self.env.x)[[21, 25, 26]]
        return crop_obs

class ControlObservations(BaseObservations):
    """
    控制变量观测模块，让智能体知道上一步的执行动作。
    """
    def __init__(self, env) -> None:
        self.env = env
        self.obs_names = ["uBoil", "uCo2", "uThScr", "uVent", "uLamp", "uBlScr"]
        self.n_obs = len(self.obs_names)

    def observation_space(self):
        return spaces.Box(low=0., high=1., shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        直接返回环境中的控制向量 u。
        """
        return self.env.u

class WeatherObservations(BaseObservations):
    """
    室外天气观测模块，实时提取当前的室外气候数据。
    """
    def __init__(self, env) -> None:
        self.env = env
        self.obs_names = ["glob_rad", "temp_out", "rh_out", "co2_out", "wind_speed", "dli"]
        self.n_obs = len(self.obs_names)

    def observation_space(self) -> spaces.Box:
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        从天气数组中提取当前步的数据，并进行单位转换。
        """
        # data indices: 0:glob_rad, 1:temp, 2:vp, 3:co2, 4:wind, ..., 7:dli
        data = self.env.weather_data[self.env.timestep]
        weather_obs = np.array([data[0], data[1], data[2], data[3], data[4], data[7]])
        
        weather_obs[2] = vaporPres2rh(weather_obs[1], weather_obs[2])
        weather_obs[3] = co2dens2ppm(weather_obs[1], weather_obs[3]*1e-6)
        return weather_obs

class TimeObservations(BaseObservations):
    """
    时间观测模块，利用正余弦变换编码周期性时间。
    """
    def __init__(self, env) -> None:
        self.env = env
        self.obs_names = ["day of year sin", "day of year cos", "hour of day sin", "hour of day cos"]
        self.n_obs = len(self.obs_names)

    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        计算基于时间的观测值。将一年中的日期和一天中的小时归一化到 [0, 1]，
        并使用 sin 和 cos 编码，以便模型能够学习日期和年份的周期性。
        timestep 表示当前的模拟步数。
        """
        day_of_year_sin = np.sin(2 * np.pi * self.env.day_of_year / 365.0)
        day_of_year_cos = np.cos(2 * np.pi * self.env.day_of_year / 365.0)

        hour_of_day_sin = np.sin(2 * np.pi * self.env.hour_of_day / 24.0)
        hour_of_day_cos = np.cos(2 * np.pi * self.env.hour_of_day / 24.0)

        return np.array([day_of_year_sin, day_of_year_cos, hour_of_day_sin, hour_of_day_cos])

class WeatherForecastObservations(BaseObservations):
    """
    天气预报观测模块，提供未来 Np 个步长的预报数据。
    """
    def __init__(self, env) -> None:
        self.env = env
        self.n_obs = 5*self.env.Np
        base_names = ["glob_rad", "temp_out", "rh_out", "co2_out", "wind_speed"]
        self.obs_names = []
        for i in range(1, self.env.Np + 1):
            self.obs_names.extend([f"{name}_t+{i}" for name in base_names])


    def observation_space(self):
        return spaces.Box(low=-1e-4, high=1e4, shape=(self.n_obs,), dtype=np.float32)

    def compute_obs(self) -> np.ndarray:
        """
        从天气数据数组中提取未来步长的预测信息（仅提取前 5 个主要天气变量）。
        """
        forecast = []
        for i in range(1, self.env.Np+1):
            forecast.extend(self.env.weather_data[self.env.timestep+i][0:5])
        return np.array(forecast)