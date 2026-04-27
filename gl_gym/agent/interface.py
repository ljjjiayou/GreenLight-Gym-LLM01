"""
温室智能体接口模块

该模块充当智能体与温室环境（TomatoEnv）之间的适配器层。
它负责将环境的原始数值观测转换为结构化的数据对象，
并提供语义化的状态描述，便于 LLM 理解。
"""

from typing import Any, Dict, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

from gymnasium import spaces
from gl_gym.common.utils import load_model_hyperparams, calculate_vpd_kpa, calculate_dew_point_c
from gl_gym.environments.utils import vaporPres2rh


@dataclass
class GreenhouseState:
    """
    温室环境状态数据类
    
    用于存储从环境观测中解析出的关键指标，提供类型安全的访问方式。
    包含环境状态、作物状态、执行器状态和气象数据。
    """
    timestep: int  # 当前仿真步数
    day_of_year: float  # 一年中的第几天 (1-365)
    hour_of_day: float  # 一天中的小时 (0-24)
    
    # --- 室内环境指标 ---
    co2_air: float  # 室内空气 CO2 浓度 (ppm)
    temp_air: float  # 室内空气温度 (°C)
    rh_air: float  # 室内相对湿度 (%)
    pipe_temp: float  # 加热管道温度 (°C)
    
    # --- 作物生长指标 ---
    fruit_weight: float  # 果实干重/累计产量 (kg/m²)
    canopy_temp_24h: float  # 24小时平均冠层温度 (°C)
    temperature_sum: float  # 积温/累计温度 (°C·day)
    
    # --- 执行器当前状态 (上一时刻的控制量) ---
    u_boil: float  # 加热锅炉开度 (0-1)
    u_co2: float  # CO2 阀门开度 (0-1)
    u_th_scr: float  # 保温幕位置 (0=开, 1=关)
    u_vent: float  # 通风窗开度 (0-1)
    u_lamp: float  # 补光灯强度 (0-1)
    u_bl_scr: float  # 遮阳网位置 (0-1)
    
    # --- 室外气象数据 ---
    glob_rad: float  # 全球辐射 (W/m²)
    temp_out: float  # 室外温度 (°C)
    rh_out: float  # 室外相对湿度 (%)
    co2_out: float  # 室外 CO2 浓度 (ppm)
    wind_speed: float  # 风速 (m/s)
    dli: float # 日累积光量 (MJ/m2)
    dew_point_air: float
    dew_margin_air: float
    canopy_dew_margin: float
    forecast_rad_mean_1h: float
    forecast_rad_peak_2h: float
    forecast_temp_out_delta_1h: float
    forecast_rh_out_mean_1h: float
    forecast_wind_peak_1h: float
    forecast_humidity_risk: float
    time_to_sunrise_steps: int
    
    # --- 强化学习反馈 ---
    reward: float = 0.0  # 当前步获得的奖励值
    terminated: bool = False # 是否达到终止条件 (如完成目标)
    truncated: bool = False # 是否被截断 (如达到最大步数)


@dataclass
class ControlAction:
    """
    控制动作数据类
    
    封装了智能体发出的控制指令，用于标准化动作传递。
    所有控制量范围均为 [0, 1]。
    """
    u_boil: float = 0.0  # 加热
    u_co2: float = 0.0   # CO2
    u_th_scr: float = 0.0 # 保温幕
    u_vent: float = 0.0  # 通风
    u_lamp: float = 0.0  # 补光
    u_bl_scr: float = 0.0 # 遮阳
    
    def to_array(self) -> np.ndarray:
        """转换为 numpy 数组，顺序与环境 action_space 一致"""
        return np.array([
            self.u_boil, self.u_co2, self.u_th_scr,
            self.u_vent, self.u_lamp, self.u_bl_scr
        ], dtype=np.float32)
    
    def to_action_space(self, u_min: np.ndarray, u_max: np.ndarray) -> np.ndarray:
        """
        将 [0, 1] 范围的动作映射到环境实际需要的动作空间范围 (通常是 [-1, 1])。
        
        Args:
            u_min: 环境动作最小值数组
            u_max: 环境动作最大值数组
            
        Returns:
            映射后的动作数组
        """
        action = self.to_array()
        # 线性映射公式: y = 2 * (x - min) / (max - min) - 1
        # 假设输入 x 在 u_min 和 u_max 之间归一化
        return 2 * (action - u_min) / (u_max - u_min) - 1

class GreenhouseAgentInterface:
    """
    温室智能体接口层
    
    功能：
    1. 动态映射：自动解析环境的 observation_space，建立变量名到索引的映射。
    2. 状态缓存：维护历史奖励和终止状态。
    3. 趋势分析：计算温度变化趋势，为 LLM 提供更丰富的上下文。
    4. 语义转换：生成自然语言描述报告。
    """
    
    def __init__(self, env: Any, window_size: int = 12):
        self.env = env
        # 获取环境动作空间维度和范围
        self.nu = getattr(env, 'nu', 6)
        self.u_min = getattr(env, 'u_min', np.zeros(self.nu))
        self.u_max = getattr(env, 'u_max', np.ones(self.nu))
        
        # 缓存上一时刻的反馈
        self.last_reward = 0.0
        self.last_terminated = False
        self.last_truncated = False
        
        # 用于温度趋势分析的滑动窗口
        self.temp_history = deque(maxlen=window_size)
        # 构建观测变量名到数组索引的映射表
        self._obs_name_to_index = self._build_obs_name_to_index()
        
    def _build_obs_name_to_index(self) -> Dict[str, int]:
        """遍历环境的 observation_modules，构建变量名索引映射"""
        if not hasattr(self.env, "observation_modules"):
            return {}
        names: list[str] = []
        for module in self.env.observation_modules:
            if not hasattr(module, "obs_names") or not module.obs_names:
                continue
            names.extend(module.obs_names)
        return {name: idx for idx, name in enumerate(names)}

    def _get_current_obs(self) -> np.ndarray:
        """安全获取当前环境观测值"""
        if hasattr(self.env, 'obs') and self.env.obs is not None:
            return self.env.obs
        if hasattr(self.env, '_get_obs'):
            return self.env._get_obs()
        return np.zeros(30) # 兜底防止报错

    def _get_future_weather_window(self, steps: int) -> np.ndarray:
        env = self.env
        if not hasattr(env, "weather_data") or not hasattr(env, "timestep"):
            return np.empty((0, 0), dtype=float)

        start = int(getattr(env, "timestep", 0)) + 1
        end = min(start + max(int(steps), 0), len(env.weather_data))
        if start >= end:
            return np.empty((0, 0), dtype=float)
        return np.asarray(env.weather_data[start:end], dtype=float)

    def _compute_forecast_features(self) -> Dict[str, float]:
        steps_per_hour = max(1, int(round(3600.0 / float(getattr(self.env, "dt", 900.0)))))
        window_1h = self._get_future_weather_window(steps_per_hour)
        window_2h = self._get_future_weather_window(2 * steps_per_hour)

        current_rad = 0.0
        if hasattr(self.env, "weather_data") and hasattr(self.env, "timestep"):
            idx = int(getattr(self.env, "timestep", 0))
            if 0 <= idx < len(self.env.weather_data):
                current_rad = float(self.env.weather_data[idx, 0])

        if window_1h.size == 0:
            return {
                "forecast_rad_mean_1h": current_rad,
                "forecast_rad_peak_2h": current_rad,
                "forecast_temp_out_delta_1h": 0.0,
                "forecast_rh_out_mean_1h": 0.0,
                "forecast_wind_peak_1h": 0.0,
                "forecast_humidity_risk": 0.0,
                "time_to_sunrise_steps": steps_per_hour,
            }

        rad_1h = window_1h[:, 0]
        temp_1h = window_1h[:, 1]
        rh_1h = vaporPres2rh(window_1h[:, 1], window_1h[:, 2])
        wind_1h = window_1h[:, 4]

        rad_2h = window_2h[:, 0] if window_2h.size else rad_1h
        sunrise_steps = steps_per_hour
        sunrise_candidates = np.where(rad_2h >= 20.0)[0]
        if sunrise_candidates.size > 0:
            sunrise_steps = int(sunrise_candidates[0]) + 1

        rad_mean = float(np.mean(rad_1h))
        rad_peak = float(np.max(rad_2h))
        rh_mean = float(np.mean(rh_1h))
        wind_peak = float(np.max(wind_1h))
        temp_delta = float(temp_1h[-1] - temp_1h[0]) if len(temp_1h) > 1 else 0.0

        humidity_risk = 0.0
        humidity_risk += np.clip((rh_mean - 82.0) / 12.0, 0.0, 1.0) * 0.55
        humidity_risk += np.clip((8.0 - rad_mean) / 8.0, 0.0, 1.0) * 0.20
        humidity_risk += np.clip((6.0 - temp_1h[0]) / 10.0, 0.0, 1.0) * 0.15
        humidity_risk += np.clip((2.0 - wind_peak) / 2.0, 0.0, 1.0) * 0.10

        return {
            "forecast_rad_mean_1h": rad_mean,
            "forecast_rad_peak_2h": rad_peak,
            "forecast_temp_out_delta_1h": temp_delta,
            "forecast_rh_out_mean_1h": rh_mean,
            "forecast_wind_peak_1h": wind_peak,
            "forecast_humidity_risk": float(np.clip(humidity_risk, 0.0, 1.0)),
            "time_to_sunrise_steps": sunrise_steps,
        }

    def get_state(self) -> GreenhouseState:
        """
        获取当前完整的结构化状态。
        自动从观测数组中提取对应的值填入 GreenhouseState 对象。
        """
        obs = self._get_current_obs()
        
        def _get_obs_value(name: str, default: float) -> float:
            """根据变量名从观测数组中取值"""
            if self._obs_name_to_index:
                idx = self._obs_name_to_index.get(name)
                if idx is not None and idx < len(obs):
                    return float(obs[idx])
            return default

        forecast_features = self._compute_forecast_features()
        temp_air = _get_obs_value("temp_air", 20)
        rh_air = _get_obs_value("rh_air", 70)
        canopy_temp = _get_obs_value("24CanTemp", 20)
        dew_point_air = float(calculate_dew_point_c(temp_air, rh_air))
        dew_margin_air = float(temp_air - dew_point_air)
        canopy_dew_margin = float(canopy_temp - dew_point_air)

        state = GreenhouseState(
            timestep=getattr(self.env, 'timestep', 0),
            day_of_year=getattr(self.env, 'day_of_year', 0),
            hour_of_day=getattr(self.env, 'hour_of_day', 0),
            
            co2_air=_get_obs_value("co2_air", 400),
            temp_air=temp_air,
            rh_air=rh_air,
            pipe_temp=_get_obs_value("pipe_temp", 40),
            
            fruit_weight=_get_obs_value("cFruit", 0) * 1e-6, # mg/m2 -> kg/m2
            canopy_temp_24h=canopy_temp,
            temperature_sum=_get_obs_value("tSum", 0),
            
            # 获取上一时刻的动作状态 (环境通常会记录在 env.u 中)
            u_boil=getattr(self.env, 'u', np.zeros(self.nu))[0] if hasattr(self.env, 'u') else 0,
            u_co2=getattr(self.env, 'u', np.zeros(self.nu))[1] if hasattr(self.env, 'u') else 0,
            u_th_scr=getattr(self.env, 'u', np.zeros(self.nu))[2] if hasattr(self.env, 'u') else 0,
            u_vent=getattr(self.env, 'u', np.zeros(self.nu))[3] if hasattr(self.env, 'u') else 0,
            u_lamp=getattr(self.env, 'u', np.zeros(self.nu))[4] if hasattr(self.env, 'u') else 0,
            u_bl_scr=getattr(self.env, 'u', np.zeros(self.nu))[5] if hasattr(self.env, 'u') else 0,
            
            glob_rad=_get_obs_value("glob_rad", 0),
            temp_out=_get_obs_value("temp_out", 10),
            rh_out=_get_obs_value("rh_out", 70),
            co2_out=_get_obs_value("co2_out", 400),
            wind_speed=_get_obs_value("wind_speed", 3),
            dli=_get_obs_value("dli", 0),
            dew_point_air=dew_point_air,
            dew_margin_air=dew_margin_air,
            canopy_dew_margin=canopy_dew_margin,
            forecast_rad_mean_1h=forecast_features["forecast_rad_mean_1h"],
            forecast_rad_peak_2h=forecast_features["forecast_rad_peak_2h"],
            forecast_temp_out_delta_1h=forecast_features["forecast_temp_out_delta_1h"],
            forecast_rh_out_mean_1h=forecast_features["forecast_rh_out_mean_1h"],
            forecast_wind_peak_1h=forecast_features["forecast_wind_peak_1h"],
            forecast_humidity_risk=forecast_features["forecast_humidity_risk"],
            time_to_sunrise_steps=int(forecast_features["time_to_sunrise_steps"]),
            
            reward=self.last_reward,
            terminated=self.last_terminated,
            truncated=self.last_truncated
        )
        return state

    def _analyze_temp_trend(self, current_temp: float) -> str:
        """
        基于历史温度数据，使用线性回归分析温度变化趋势。
        返回易于理解的自然语言描述（如“快速回升”、“急剧失温”）。
        """
        self.temp_history.append(current_temp)
        if len(self.temp_history) < 5:
            return "正在建立分析基准..."
        
        y = np.array(self.temp_history, dtype=float)
        x = np.arange(len(y), dtype=float)
        slope, _ = np.polyfit(x, y, 1) # 计算斜率
        
        if slope > 0.08:
            return "快速回升 ↑↑"
        elif slope > 0.02:
            return "缓慢上升 ↑"
        elif slope < -0.08:
            return "急剧失温 ↓↓"
        elif slope < -0.02:
            return "缓慢下降 ↓"
        else:
            return "基本恒定 (热平衡)"

    def _get_forecast_summary(self) -> str:
        """获取未来 1 小时 (4步) 天气预报数据"""
        forecast = self._compute_forecast_features()
        rad_mean = forecast["forecast_rad_mean_1h"]
        rad_peak = forecast["forecast_rad_peak_2h"]
        temp_delta = forecast["forecast_temp_out_delta_1h"]
        rh_mean = forecast["forecast_rh_out_mean_1h"]
        sunrise_steps = int(forecast["time_to_sunrise_steps"])

        rad_trend = "平稳"
        if rad_peak - rad_mean > 80.0:
            rad_trend = "增强"
        elif rad_mean - rad_peak > 40.0:
            rad_trend = "减弱"

        return (
            f"1h辐射均值 {rad_mean:.0f} W/m², 2h峰值 {rad_peak:.0f} W/m², "
            f"1h外温变化 {temp_delta:+.1f}°C, 1h外湿均值 {rh_mean:.0f}%, "
            f"约 {sunrise_steps} 步后日出 ({rad_trend})"
        )

    def get_state_description(self) -> str:
        """
        生成温室状态的自然语言报告。
        作为 Prompt 的一部分输入给 LLM，帮助其理解当前局势。
        """
        s = self.get_state()
        temp_trend = self._analyze_temp_trend(s.temp_air)
        forecast_text = self._get_forecast_summary()
        is_day = 6 <= s.hour_of_day <= 18
        t_target = "18-25°C" if is_day else "12-16°C"
        
        # 计算 VPD (kPa)
        vpd = calculate_vpd_kpa(s.temp_air, s.rh_air)
        vpd_status = ""
        if vpd < 0.4: vpd_status = "(过湿! 需除湿)"
        elif vpd > 1.6: vpd_status = "(过干! 需加湿/降温)"
        else: vpd_status = "(适宜)"
        dew_margin = min(float(s.dew_margin_air), float(s.canopy_dew_margin))
        if dew_margin < 0.5:
            dew_status = "(高结露风险)"
        elif dew_margin < 1.5:
            dew_status = "(接近露点)"
        else:
            dew_status = "(露点安全)"

        # DLI 换算
        dli_mol = s.dli * 2.0  # MJ -> mol (approx)
        dli_status = "(低)" if dli_mol < 10 else "(足)"

        return f"""=== 温室环境监测分析报告 (Step: {s.timestep}) ===
时间状态: 第 {s.day_of_year:.1f} 天, {s.hour_of_day:.1f} 时 ({'白天' if is_day else '夜晚'})

【核心环境指标】
- 室内温度: {s.temp_air:.1f}°C (精准走势: {temp_trend})
  * 建议目标范围: {t_target}
- 相对湿度: {s.rh_air:.1f}% (目标范围 50-85%)
- 饱和水汽压差 (VPD): {vpd:.2f} kPa {vpd_status}
  * 建议范围: 0.4 - 1.2 kPa (最佳 0.8-1.0)
- 露点温度: {s.dew_point_air:.1f}°C, 空气露点裕度: {s.dew_margin_air:.1f}°C, 冠层露点裕度: {s.canopy_dew_margin:.1f}°C {dew_status}
- CO2 浓度: {s.co2_air:.0f} ppm (建议 400-1000 ppm)
- 加热管道温度: {s.pipe_temp:.1f}°C

【作物生理状态】
- 果实累计产量(碳量): {s.fruit_weight:.4f}
- 24h平均冠层温度: {s.canopy_temp_24h:.1f}°C
- 当前累计积温: {s.temperature_sum:.1f}°C·d

【当前执行器状态 (0-1开度)】
- 加热锅炉: {s.u_boil:.2f}, CO2 阀门: {s.u_co2:.2f}, 窗口通风: {s.u_vent:.2f}
- 保温幕布: {s.u_th_scr:.2f}, 人工补光: {s.u_lamp:.2f}, 遮阳网: {s.u_bl_scr:.2f}

【外部背景气象】
- 辐射强度: {s.glob_rad:.1f} W/m² (影响光合作用与自然升温)
- 室外环境: 温度 {s.temp_out:.1f}°C, 湿度 {s.rh_out:.1f}%, 风速 {s.wind_speed:.1f} m/s
- 累积光量(DLI): {dli_mol:.1f} mol/m² {dli_status} (目标 > 15)
- 未来1h特征: 辐射均值 {s.forecast_rad_mean_1h:.0f} W/m², 2h峰值 {s.forecast_rad_peak_2h:.0f} W/m², 外湿均值 {s.forecast_rh_out_mean_1h:.0f}%, 风峰值 {s.forecast_wind_peak_1h:.1f} m/s
- 前瞻湿害风险: {s.forecast_humidity_risk:.2f} (0-1), 距离日出约 {s.time_to_sunrise_steps} 步
- 气象预报(1h): {forecast_text}"""
