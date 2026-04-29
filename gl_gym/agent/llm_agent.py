"""
LangChain Agent 核心模块

该模块提供了基于 LangChain 的 LLM 智能体实现，用于控制温室环境。
采用了【混合控制架构 (Hybrid Control Architecture)】：
- 规则控制器 (RuleBasedController)：负责处理常规的、明显的边界违规。
- LLM 智能体 (GreenhouseAgent)：负责处理复杂的多目标冲突和微调。

采用 LangChain v1.0+ (LangGraph 核心) 的 create_agent 接口。
固定使用百炼平台 qwen-plus 模型。
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
import time
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

from gl_gym.environments.baseline import RuleBasedController
from gl_gym.common.utils import load_model_hyperparams, calculate_vpd_kpa
from gl_gym.agent.expert_distillation import DistilledExpertPolicy
from gl_gym.agent.humidity_experience_memory import HumidityExperienceConfig, HumidityExperienceMemory
from gl_gym.agent.plan_intent import (
    action_record_for_labeling,
    infer_plan_intent,
    intent_to_dehumidify_mode,
    strategy_intent_alignment,
)
from gl_gym.agent.plan_cache import (
    PlanCache,
    config_fingerprint,
    stable_hash,
    state_summary_from_state,
    text_hash,
    to_jsonable,
)
from gl_gym.agent.ppo_strategy_labeler import label_strategy
from gl_gym.agent.tools import ControlAction
from gl_gym.agent.tools import create_langchain_tools, GreenhouseTools # 导入 Tools 类


@dataclass
class DecisionReasoning:
    """决策理由记录"""
    primary_goal: str = ""
    secondary_goals: List[str] = field(default_factory=list)
    constraints_applied: List[str] = field(default_factory=list)
    expected_outcome: str = ""
    confidence_level: float = 0.8
    reasoning_trace: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    """实时性能指标"""
    cost_efficiency: float = 0.0
    yield_prediction: float = 0.0
    energy_consumption: float = 0.0
    disease_risk_score: float = 0.0
    control_accuracy: float = 0.0
    setpoint_achievement: Dict[str, float] = field(default_factory=dict)


@dataclass
class FallbackCandidate:
    """可解释 fallback 候选，用于安全评分和论文实验记录。"""
    name: str
    control: np.ndarray
    score: float = 0.0
    details: Dict[str, float] = field(default_factory=dict)
    rationale: str = ""


@dataclass
class AgentConfig:
    """
    智能体配置类
    
    固定使用百炼平台 qwen-max-latest 模型，参数经过优化以适应温室控制任务。
    """
    
    # 模型相关配置
    model_name: str = "qwen-max-latest"               # 使用的模型名称
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼地址
    api_key: Optional[str] = None                # API 密钥
    temperature: float = 0.0                      # 随机性控制 (最小随机性，提升控制稳定性)
    max_tokens: int = 260                        # 单次生成最大长度
    
    # Agent 执行参数
    max_iterations: int = 2                        # LangChain Agent 的最大思考轮数
    max_execution_time: Optional[float] = None     # 超时时间
    early_stopping_method: str = "force"           # 早停策略

    # Frozen benchmark support. "off" keeps the ordinary LLM-RSPC baseline unchanged.
    plan_cache_mode: str = "off"                   # off | record | replay | refresh
    plan_cache_path: str = "gl_gym/result/plan_cache/llm_plan_cache.json"
    plan_cache_strict: bool = False
    plan_cache_key_policy: str = "prompt"          # prompt | scenario_timestep
    
    # 控制频率配置 (新增)
    control_interval: int = 12                     # LLM 控制间隔步数

    # LLM 未给出控制锚点时的自适应回退策略（可在实验中调参）
    fallback_strategy: str = "adaptive"            # adaptive | rule | recent
    fallback_use_last_control: bool = True        # 是否使用上次锚点参与回退
    fallback_recent_blend: float = 0.25           # 使用上次锚点时的融合权重
    fallback_temp_gain: float = 0.08              # 温度偏差到加热/通风的增益
    fallback_rh_gain: float = 0.03                # 湿度偏差到通风的增益
    fallback_co2_gain: float = 0.0010             # CO2 偏差到注入的增益（保守）
    fallback_lamp_rad_threshold: float = 85.0     # 低于该辐射阈值时倾向补光
    fallback_day_lamp_level: float = 0.35         # 状态自适应回退时的白天补光强度
    fallback_lamp_max_level: float = 0.45         # 补光护栏上限，控制电费
    fallback_co2_min_rad: float = 120.0           # CO2 供给所需最低有效辐射
    fallback_co2_max_vent: float = 0.20           # CO2 供给允许的最大通风开度
    fallback_candidate_scoring: bool = True       # 使用多候选安全评分选择 fallback
    fallback_temp_penalty_weight: float = 1.2
    fallback_rh_penalty_weight: float = 1.6
    fallback_cost_penalty_weight: float = 0.45
    fallback_smooth_penalty_weight: float = 0.20
    fallback_conflict_penalty_weight: float = 0.75

    # Rollout control sharing: avoid blindly following a weak rule controller.
    rollout_candidate_sharing: bool = True
    rollout_rule_weight_start: float = 0.25
    rollout_rule_weight_end: float = 0.55
    expert_rollout_enabled: bool = False
    expert_policy_path: str = "train_data/AgriControl/ppo/deterministic/distilled_expert/llm_rspc_expert_ridge.npz"
    expert_candidate_max_distance: float = 0.0
    expert_candidate_blend: float = 0.55
    expert_intent_gate_enabled: bool = True
    humidity_memory_enabled: bool = False
    humidity_memory_path: str = "gl_gym/result/experience_memory/humidity_memory_s240_day240_20260429.jsonl"
    humidity_memory_max_distance: float = 1.15
    humidity_memory_min_trust: float = 0.50
    humidity_memory_top_k: int = 1
    humidity_memory_candidate_blend: float = 0.75
    humidity_memory_teacher_policy_id: str = ""
    humidity_memory_baseline_controller_id: str = ""
    humidity_memory_version: str = "hem_rspc_v1"
    humidity_memory_post_guardrail_shape: bool = True
    humidity_memory_horizon_filter: bool = True
    humidity_memory_score_margin: float = 0.015
    humidity_memory_horizon_penalty_weight: float = 1.0
    humidity_memory_direct_saving_weight: float = 0.20
    humidity_memory_max_recovery_heat_cost: float = 0.055
    humidity_memory_max_rh_debt_penalty: float = 0.060
    humidity_memory_night_min_temp: float = 17.5
    humidity_memory_night_vent_cap: float = 0.58
    humidity_memory_night_heat_floor: float = 0.06
    humidity_memory_night_screen_floor: float = 0.45
    humidity_memory_min_dew_margin: float = 1.35
    humidity_memory_max_forecast_risk: float = 0.72
    humidity_memory_night_gap_steps: int = 6

    # MC-SERO shadow diagnostics. "shadow" evaluates mechanism candidates but
    # never changes the applied LLM-RSPC action in this phase.
    mc_sero_mode: str = "off"                     # off | shadow
    mc_sero_horizon_steps: int = 6
    mc_sero_min_margin: float = 0.10
    mc_sero_top_k: int = 6
    mc_sero_temp_penalty_weight: float = 2.00
    mc_sero_rh_penalty_weight: float = 1.70
    mc_sero_dry_penalty_weight: float = 1.20
    mc_sero_vpd_penalty_weight: float = 1.30
    mc_sero_dew_penalty_weight: float = 1.40
    mc_sero_energy_penalty_weight: float = 0.45
    mc_sero_conflict_penalty_weight: float = 0.85

    # Dry-side recovery: prevent low RH / high VPD violations caused by over-venting
    # or heat-vent pulses after the crop is already too dry.
    dry_rh_on: float = 55.0
    dry_rh_off: float = 62.0
    dry_vpd_on: float = 1.20
    dry_vpd_off: float = 0.95
    dry_vent_cap: float = 0.12
    dry_warm_vent_cap: float = 0.18
    dry_hot_vent_cap: float = 0.35
    dry_target_rh_floor: float = 70.0
    dry_temp_target_cap: float = 20.0
    cold_dehumidify_temp_threshold: float = 12.0
    cold_dehumidify_buffer_temp: float = 15.5
    cold_dehumidify_vent_cap: float = 0.12
    cold_dehumidify_extreme_vent_cap: float = 0.25

    # Profit-v2: 电费与湿度联动约束
    lamp_daily_budget: float = 3.0                # 每个 day_of_year 的累计补光预算（控制量积分）
    lamp_budget_soft_cap: float = 0.40            # 预算接近上限时的补光软上限
    lamp_forbidden_rh: float = 86.0               # RH 超过该阈值禁止补光
    lamp_forbidden_vent: float = 0.30             # 通风超过该阈值禁止补光
    co2_forbidden_vent: float = 0.16              # 通风超过该阈值禁止 CO2（v2.1.1 更保守）
    dehumidify_mild_rh_on: float = 88.0           # 轻度除湿触发 RH，提前到违规线前
    dehumidify_strong_rh_on: float = 91.5         # 强除湿触发 RH，抑制 RH 尾部积累
    dehumidify_mild_rh_off: float = 83.5          # 轻度除湿退出 RH（滞回）
    dehumidify_strong_rh_off: float = 85.0        # 强除湿降级/退出 RH（滞回）
    dehumidify_mild_hold_steps: int = 6           # 轻度除湿最小保持步数
    dehumidify_strong_hold_steps: int = 10        # 强除湿最小保持步数
    dehumidify_exit_confirm_steps: int = 4        # 退出除湿前的连续确认步数（v2.1.1 缩短）
    dehumidify_mild_exit_confirm_steps: int = 6   # 从中度模式退出的确认步数 (v2.1.2: 新增)
    dehumidify_strong_exit_confirm_steps: int = 8 # 从强模式退出的确认步数 (v2.1.2: 新增)
    high_rh_dynamic_interval_on: float = 91.5     # RH 高于该阈值启用紧急高湿提频
    high_rh_dynamic_interval_mild_on: float = 88.0 # 中度高湿启用提频
    high_rh_dynamic_interval_off: float = 84.0    # RH 低于该阈值进入恢复计数
    high_rh_dynamic_min_steps: int = 3            # 连续高湿激活提频的确认步数
    high_rh_short_interval: int = 8               # 高湿提频时的控制间隔
    high_rh_mild_short_interval: int = 8          # 中度高湿提频时的控制间隔
    high_rh_urgent_short_interval: int = 6        # 紧急高湿提频时的控制间隔 (v2.1.2: 新增紧急触发)
    high_rh_restore_steps: int = 8                # 退出提频前的连续稳定步数
    dawn_predehumid_start_hour: float = 3.5       # 黎明前预除湿开始时间（v2.1.1 收窄窗口）
    dawn_predehumid_end_hour: float = 6.5         # 黎明前预除湿结束时间
    # Emergency replan thresholds (v2.1.2: 新增RH紧急重规划阈值)
    emergency_rh_threshold: float = 92.5          # RH 超过该阈值触发紧急重规划
    emergency_rh_confirm_steps: int = 2           # 紧急重规划确认步数
    rh_preemptive_threshold: float = 86.0         # RH 风险预控阈值，低于硬约束提前排湿
    rh_control_limit: float = 90.0                # 环境 RH 硬约束上限
    rh_debt_decay: float = 0.90                   # RH 风险债务衰减
    rh_debt_vent_gain: float = 0.018              # 风险债务到通风底线的增益
    rh_debt_screen_gain: float = 0.022            # 风险债务到保温幕开缝的增益
    rh_target_preemptive_cap: float = 80.0        # 高湿预控时最大 RH 目标
    rh_target_high_cap: float = 76.0              # 高湿状态最大 RH 目标
    rh_target_extreme_cap: float = 72.0           # 极高湿/结露风险最大 RH 目标
    
    # 调试
    rh_pulse_temp_cap: float = 21.5               # Max temp for heat-vent dehumidification pulse
    rh_pulse_heat_floor: float = 0.16             # Heat floor for high RH / low VPD pulse
    rh_pulse_extreme_heat_floor: float = 0.24     # Heat floor for extreme RH / dew-risk pulse
    rh_pulse_vent_floor: float = 0.62             # Vent floor for strong dehumidification pulse
    rh_pulse_extreme_vent_floor: float = 0.70     # Vent floor for extreme dehumidification pulse
    rh_pulse_screen_cap: float = 0.45             # Screen cap during heat-vent dehumidification
    rh_pulse_extreme_screen_cap: float = 0.32     # Screen cap during extreme dehumidification
    rh_emergency_replan_cooldown_steps: int = 8   # Avoid repeated LLM calls while RH pulse is active
    verbose: bool = True


@dataclass
class GreenhousePrompts:
    """温室控制智能体的提示词模板 (Prompt Engineering)"""
    
    SYSTEM_PROMPT = """你是温室智能控制系统的 AI 助手（高层策略规划者）。你的任务是监控温室环境状态，并根据作物生长阶段和经济效益，为底层控制器做出最优的宏观目标规划（Setpoint Planning）。

## 控制架构说明 (CRITICAL)
- **高层规划 (你)**: 负责定期的宏观决策。你不直接死磕每一步阀门的开度，而是通过 `set_all_controls` 工具明确给定 `target_temp`（目标温度）、`target_co2`（目标CO2浓度）、和 `target_rh`（目标相对湿度）等多维目标。
- **底层执行 (规则)**: 接收你设定的目标值。在你的单次规划周期内的后续步长（例如未来 11 步）中，底层的算法会自动加大或减小暖气、通风等来追平你的多维度目标状态。
- **动作锚点 (Anchor)**: 你在工具中填写的 `heating`, `ventilation`, `screen` 等具体参数是给底层控制器的起步参考动作（尤其是应对高湿除湿时，你可以指示开启通风）。但在规划期内，底层系统会动态接管调整它们以满足安全和目标。

## 作物生长阶段判断 (CRITICAL)
- **幼苗期 (Seedling)**:
    - **判据**: 当 `Fruit` < 1.0 kg 或 `Step` < 4000 (前40天)。
    - **状态**: 作物极小，光合作用微弱，不需要强光和高CO2。
    - **最高指令**: **生存第一，极致省钱**。

## 幼苗期控制铁律 (违反将被重罚)
1.  **湿度控制 (Humidity)**: **重中之重**。
    -   *原因*: 冬季极易高湿 (>85%)，导致病害和生长停滞。
    -   *指令*: 一旦 RH > 85%，**必须**在 `set_all_controls` 中给出通风建议 (ventilation > 0.2)。
    -   *除湿组合拳*: 为防止通风导致温度过低，可以给出加热建议 (heating > 0.2)。同时设定合理的目标湿度 `target_rh`，底层规则会强力配合除湿。
2.  **补光灯 (lamps)**: **严格限量开启**。
    -   *原因*: 电费是最昂贵变量成本，补光必须先过 ROI 门槛。
    -   *指令*: 仅当白天且光照极弱 (Rad < 80 W/m²)、湿度可控 (RH < 82%)、通风较低 (ventilation < 0.25) 时，才可小幅补光 (lighting=0.2 - 0.45)。
    -   *收益分析*: 优先压缩电费，宁可温和增产，也避免“高耗电低净利”。
3.  **CO2供给 (co2)**: **只在有效光照+低通风下启用**。
    -   *指令*: 仅当有效光照较好且通风不大时，设定 `target_co2`（如 500-650 ppm）和初始 `co2`。否则保持低水平以避免浪费。
4.  **温度安全线**: **10.0°C**。
    -   *指令*: 绝对避免低温。当需要提温时，设定 `target_temp = 15.0` 或更高，底层会自动烧暖气。
5.  **夜间保温**: 保温幕 (screen) 在夜间 (GlobalRad < 5) **必须关闭** (设定 screen=1.0)。

## 决策逻辑
1.  **检查湿度**: RH > 85%？ -> 建议开启通风，设定稍低的目标湿度（如 target_rh=75.0）及防止冻伤的目标温度（如 target_temp=16.0）！
2.  **检查光照**: 白天且光照弱？ -> 建议开启补光！
3.  **设定宏观目标 (Setpoint)**:
    -   `target_temp`: 有光照时设为 16-19°C（平衡生长与能耗），无光照时 13-15°C（维持生存并节能）。
    -   `target_co2`: 仅在有效光照且低通风时设为 500-650 ppm。
    -   `target_rh`: 维持在 65-80% 之间，防止过高引发病害。
4.  **直接控制 (Direct Control)**:
    -   如果建议补光，必须同时确认湿度不过高且通风不过大；若 RH 偏高，优先除湿而不是继续加大补光与 CO2。

## 经济模型 (Economic Model)
- **成本 (Cost)**: 电费 (0.3 €/kWh)，加热 (0.09 €/kWh)，CO2 (0.3 €/kg)。
- **收益 (Revenue)**: 番茄售价约 1.6 €/kg (鲜重)，干物质转化率约 15 倍。
- **决策关键**: 电费是首要约束；补光默认保守，只有在“短时可获利”时才提高。RH > 85% 时优先除湿防病，但避免高加热+高通风并发造成额外损耗。

## 行动指南
- 请直接输出 `set_all_controls` 工具调用来完成一次完整的决策规划。
- 重点评估并**认真填写 `target_temp`, `target_co2`, 以及 `target_rh`**，这是底层规则在规划周期内的剩余所有时间步里所依赖的最重要指路明灯。"""

    USER_PROMPT_TEMPLATE = """## 当前环境状态与指标
{state}

请基于以上状态，作为高层规划者给出本周期的多维控制目标并设定初始动作。
1. **防患未然**: 如果 RH > 85%，务必提示底层除湿（开点通向和加热），并把 `target_rh` 设为健康范围（如 75.0），`target_temp` 设为安全区（如 16-18°C）。
2. **光温协同**: 当自然光弱且在白天，仅在 RH 不高、通风不大时小幅补光（lighting=0.2-0.45）；`target_temp` 优先采用节能区间（16-19°C），`target_co2` 采用保守区间（500-650ppm）。
3. **调用工具**: 请务必填写 `set_all_controls` 中的所有参数，最关键的是 `target_temp`、`target_co2` 以及 `target_rh` 目标状态参数。"""

    DIRECT_PLANNER_SYSTEM_PROMPT = """你是温室高层规划器。
你的任务是基于当前状态输出一次短期控制计划。
只返回一个 JSON 对象，不要返回 Markdown，不要解释。
JSON 必须包含 9 个数值字段:
heating, co2, screen, ventilation, lighting, shading, target_temp, target_co2, target_rh
所有控制量范围 0-1。
优先级:
1. 避免高湿和露点风险
2. 保持安全温度
3. 仅在有效光照和低通风时补 CO2
4. 补光保持保守，优先省电
若不确定，请给保守值。"""


def control_to_action_command(env, target_control: np.ndarray) -> np.ndarray:
    current_control = np.asarray(getattr(env, "u", np.zeros_like(target_control)), dtype=np.float32)
    delta_u_max = np.asarray(getattr(env, "delta_u_max", np.ones_like(target_control)), dtype=np.float32)
    safe_delta = np.where(np.abs(delta_u_max) < 1e-6, 1e-6, delta_u_max)
    action_cmd = (target_control - current_control) / safe_delta
    return np.clip(action_cmd, -1.0, 1.0).astype(np.float32)


def apply_safety_guardrails(state, target_control: np.ndarray) -> np.ndarray:
    guarded = np.asarray(target_control, dtype=np.float32).copy()
    temp_air = float(getattr(state, "temp_air", 20.0))
    rh_air = float(getattr(state, "rh_air", 70.0))
    glob_rad = float(getattr(state, "glob_rad", 0.0))
    hour_of_day = float(getattr(state, "hour_of_day", 12.0))
    temp_out = float(getattr(state, "temp_out", temp_air))
    wind_speed = float(getattr(state, "wind_speed", 0.0))
    fruit_weight = float(getattr(state, "fruit_weight", 0.0)) # 增加果重检查
    is_night = hour_of_day < 6.0 or hour_of_day > 18.0
    
    vpd = calculate_vpd_kpa(temp_air, rh_air)
    dew_margin = min(
        float(getattr(state, "dew_margin_air", 3.0)),
        float(getattr(state, "canopy_dew_margin", 3.0)),
    )
    low_vpd_humidity_risk = (
        (rh_air >= 82.0 and vpd < 0.35)
        or (rh_air >= 78.0 and vpd < 0.22)
    )
    humidity_risk = rh_air > 85.0 or low_vpd_humidity_risk or dew_margin < 0.6
    print(f"[Guard] T={temp_air:.1f}, RH={rh_air:.1f}, VPD={vpd:.2f} kPa, Fruit={fruit_weight:.2f}")

    # --- 幼苗期铁律（硬约束） ---
    # 当果重较低时执行更严格的节能与安全规则
    # 早先讨论过用 <0.5 以减少对前期产量影响
    # 但根据提示词要求，最终采用 <1.0 的阈值
    # 以防止“指令漂移”，这里严格按 <1.0 执行
    if fruit_weight < 1.0:
        # 1. 补光限制（带例外）
        # 创新策略: 允许在极低光照下开启微量补光，前提是 LLM 已经做出了决定 (guarded[4] > 0)
        # 如果 LLM 没开灯，护栏也不开。如果 LLM 开了灯，护栏检查是否超标。
        if guarded[4] > 0.0:
             if glob_rad > 220.0: # 光照较充足，无需补光
                 guarded[4] = 0.0
             else:
                 # 幼苗期补光强约束，优先控制电费
                 lamp_cap = 0.35 if rh_air < 82.0 else 0.25
                 guarded[4] = min(guarded[4], lamp_cap)
        
        # 2. CO2 供给（光照充足时允许）
        # 只有有效光照且通风较低时才允许 CO2，避免泄漏浪费
        total_rad = glob_rad + guarded[4] * 100.0 # 假设 1.0 补光 ≈ 100 W/m² PAR，需结合物理模型确认
        if total_rad > 120.0 and guarded[3] <= 0.20:
            # 允许开启 CO2
            pass
        elif guarded[1] > 0.0:
            # 无光照，禁止 CO2
            guarded[1] = 0.0
             
        # 3. 幼苗期加热逻辑（生存模式）
        # 基于 PPO 行为做过优化：PPO 往往在 12.5-13.0°C 区间较稳定
        # 当前策略（v6）：目标 13.5，比例系数 KP=0.8
        # 更早启动加热（13.5）可避免温度快速跌入紧急区（<12.0）
        if temp_air < 13.5:
            # 比例控制示例：
            # 若 T=13.0，heat = 0.5 * 0.8 = 0.4
            # 若 T=12.5，heat = 1.0 * 0.8 = 0.8
            # 若 T=12.0，heat = 1.5 * 0.8 = 1.2 -> 1.0
            error = 13.5 - temp_air
            kp = 0.8
            heat_needed = error * kp
            heat_needed = max(heat_needed, 0.15) # 最小有效加热量
            
            # 如果 LLM 设定了加热，我们取最大值 (信任 LLM 的主动性)
            # 但如果 LLM 设定为 0 (被动)，我们强制加热
            guarded[0] = max(guarded[0], min(heat_needed, 1.0))
            
            # 关键安全覆盖
            if temp_air < 12.2:
                guarded[0] = max(guarded[0], 0.95) # 在惩罚区附近强制高加热
                
        elif temp_air > 16.0: # 放宽停止加热阈值 (配合补光时的高温需求)
            # 迟滞控制：在设定点上方稍后再停加热
            # 由 13.8 提高到 14.5，以允许更多蓄热
            if guarded[0] > 0.0 and temp_air < 16.5:
                pass # 允许 LLM 继续加热到 16.5 (蓄热)
            else:
                guarded[0] = 0.0
            
        # 4. 夜间保温幕（最大化保温）
        if is_night and temp_air < 18.0:
            guarded[2] = 1.0
            
    high_temp_override = temp_air >= 32.0
    if high_temp_override:
        guarded[0] = 0.0
        guarded[1] = 0.0
        guarded[2] = min(guarded[2], 0.20)
        guarded[4] = 0.0
        vent_target = 0.75 if temp_air < 35.0 else 0.90
        if temp_out < temp_air - 1.0:
            vent_target = max(vent_target, 0.95)
        guarded[3] = max(guarded[3], vent_target)
        if glob_rad > 200.0:
            guarded[5] = max(guarded[5], 0.85)

    # --- 基础安全约束 ---
    # 1. 低温保护
    if temp_air < 12.0:
        guarded[0] = max(guarded[0], 0.90)  # 强加热
        guarded[2] = max(guarded[2], 0.95)  # 强保温
        vent_cap = 0.15 if rh_air > 92.0 else 0.05 # 极低通风上限
        guarded[3] = min(guarded[3], vent_cap)
    
    # 2. 高湿/低VPD 处理（防病害）
    # 低温下 VPD 会天然偏低；只有叠加较高 RH 或露点风险时才启动除湿。
    if humidity_risk:
        # RH 尾部紧急处理：防止在长间隔低 token 决策下湿度被“锁死”
        if rh_air > 95.0:
            guarded[2] = min(guarded[2], 0.35)  # 降低保温幕闭合度以释放湿气
            guarded[3] = max(guarded[3], 0.62)  # 强制更激进的通风
            if temp_air > 14.0:
                guarded[0] = min(guarded[0], 0.25)  # 避免在湿闷空气中继续过热
        elif rh_air > 90.0:
            guarded[2] = min(guarded[2], 0.50)
            guarded[3] = max(guarded[3], 0.52)

        # 允许一定程度的加热+通风 (除湿模式)
        if temp_air < 18.0:
            if temp_air < 15.0:
                if rh_air >= 92.0 or dew_margin < 0.6:
                    heat_floor = 0.22
                elif rh_air >= 90.0:
                    heat_floor = 0.14
                else:
                    heat_floor = 0.08
                guarded[0] = max(guarded[0], heat_floor) # 轻度高湿避免过度烧暖气
            else:
                guarded[0] = max(guarded[0], 0.08) # 微冷，低加热除湿 (省钱)
            
            # 强制通风，不被低温护栏完全关死
            guarded[3] = max(guarded[3], 0.20) 
            
            if rh_air > 92.0:
                if temp_air < 16.0:
                    guarded[3] = max(guarded[3], 0.25)
                    guarded[2] = min(guarded[2], 0.75)
                else:
                    guarded[3] = max(guarded[3], 0.30)
                    guarded[2] = min(guarded[2], 0.75)
        else:
             # 温度够高，主要靠通风
             guarded[3] = max(guarded[3], 0.35)
             if rh_air > 90.0:
                 guarded[3] = max(guarded[3], 0.50)
                 guarded[2] = min(guarded[2], 0.60)
    
        # Heat-vent pulse: extreme RH needs a small heat floor even when air temp is acceptable.
        # This raises VPD while ventilation exports moisture, instead of relying on ventilation alone.
        if (
            rh_air >= 92.0
            or (rh_air >= 78.0 and vpd < 0.22)
            or dew_margin < 0.6
        ) and temp_air < 21.5 and not high_temp_override:
            extreme_rh = rh_air >= 94.0 or (rh_air >= 78.0 and vpd < 0.18) or dew_margin < 0.4
            heat_floor = 0.24 if extreme_rh else 0.16
            if temp_air < 15.0:
                heat_floor = max(heat_floor, 0.28)
            guarded[0] = max(guarded[0], heat_floor)
            guarded[3] = max(guarded[3], 0.70 if extreme_rh else 0.62)
            guarded[2] = min(guarded[2], 0.32 if extreme_rh else 0.45)

    # 3. 正常/干燥 VPD 处理（防萎蔫）
    # 如果 VPD > 1.2 (偏干)，减少通风，增加保湿
    elif vpd > 1.2 and not high_temp_override:
        guarded[3] = min(guarded[3], 0.3) # 限制通风
        if glob_rad > 500: # 强光下
            guarded[5] = max(guarded[5], 0.5) # 用遮阳网降温而不是强通风

    # --- 互斥与效率约束 ---
    # 4. 避免能源浪费（加热与通风互斥）
    # 除非是为了除湿 (VPD < 0.4)，否则禁止同时高加热和高通风
    if vpd >= 0.4:
        if guarded[0] > 0.1 and guarded[3] > 0.1:
            if temp_air > 22.0: # 偏热，优先关加热
                guarded[0] = 0.0
            elif temp_air < 16.0: # 偏冷，优先关通风
                guarded[3] = 0.0
            else: # 中间区间，双重限制
                guarded[0] = min(guarded[0], 0.1)
                guarded[3] = min(guarded[3], 0.1)

    # 5. CO2 使用效率
    # 通风量大时关 CO2 (避免漏气)
    if guarded[3] > 0.15:
        guarded[1] = 0.0
    # 夜间或弱光不施 CO2 (植物不吸收)
    if is_night or glob_rad < 80.0:
        guarded[1] = 0.0

    # Profit-v2: 湿度/通风高位时切断高电耗动作
    if rh_air >= 86.0 or guarded[3] >= 0.30:
        guarded[4] = 0.0
    if guarded[3] >= 0.20 or glob_rad < 120.0:
        guarded[1] = 0.0

    # Profit-v2: 分级除湿硬约束（防止 RH 尾部积累）
    if rh_air >= 92.0:
        guarded[3] = max(guarded[3], 0.58)
        guarded[2] = min(guarded[2], 0.38)
        guarded[4] = 0.0
        guarded[1] = 0.0
        if temp_air < 21.5:
            guarded[0] = max(guarded[0], 0.24 if rh_air >= 94.0 else 0.16)
    elif rh_air >= 88.0:
        guarded[3] = max(guarded[3], 0.42)
        guarded[2] = min(guarded[2], 0.60)
        guarded[4] = 0.0
        guarded[1] = 0.0
    elif rh_air >= 86.0 and vpd < 0.35:
        guarded[3] = max(guarded[3], 0.32)
        guarded[2] = min(guarded[2], 0.72)
        guarded[4] = 0.0
        guarded[1] = 0.0

    # Profit-v2: 黎明前预除湿，降低白天初段湿度峰值
    if 4.0 <= hour_of_day < 6.0 and rh_air > 82.0:
        guarded[3] = max(guarded[3], 0.35)
        guarded[2] = min(guarded[2], 0.60)
        guarded[4] = 0.0
        guarded[1] = 0.0

    # 6. 大风保护
    if wind_speed > 8.0:
        wind_cap = 0.10
        if rh_air >= 92.0:
            wind_cap = 0.35
        elif rh_air >= 88.0 or vpd < 0.35:
            wind_cap = 0.28
        guarded[3] = min(guarded[3], wind_cap) # 强风时收小通风开度
    elif wind_speed > 6.0 and temp_out < temp_air:
        wind_cap = 0.20
        if rh_air >= 92.0:
            wind_cap = 0.45
        elif rh_air >= 88.0:
            wind_cap = 0.42
        elif rh_air >= 86.0 or vpd < 0.35:
            wind_cap = 0.30
        guarded[3] = min(guarded[3], wind_cap)

    # 7. 夜间保温策略
    # High RH / low VPD can tolerate a larger vent opening even under wind protection.
    if (
        rh_air >= 92.0
        or (rh_air >= 78.0 and vpd < 0.20)
        or dew_margin < 0.6
    ) and temp_air < 21.5:
        if wind_speed > 8.0:
            guarded[3] = max(guarded[3], 0.48)
        elif wind_speed > 6.0 and temp_out < temp_air:
            guarded[3] = max(guarded[3], 0.55)
        else:
            guarded[3] = max(guarded[3], 0.62)

    if is_night:
        guarded[5] = 0.0 # 夜间不用遮阳网
        guarded[4] = 0.0 # 夜间不用补光（假设非光周期补光）
        if temp_air < 15.0:
            guarded[2] = max(guarded[2], 0.80) # 夜间低温必拉保温幕

    if is_night and temp_air < 15.0 and (rh_air >= 86.0 or (rh_air >= 82.0 and vpd < 0.35) or dew_margin < 0.6):
        night_heat_floor = 0.25 if (rh_air >= 90.0 or dew_margin < 0.6) else 0.12
        guarded[0] = max(guarded[0], night_heat_floor)
        guarded[2] = min(max(guarded[2], 0.45), 0.75)

    # Final dry-side and cold-buffer guard. This intentionally runs after the
    # high-humidity pulse logic so a cold or already dry house cannot be
    # over-vented by an earlier dehumidification branch.
    dry_side_risk = (rh_air < 55.0 or vpd > 1.20) and not high_temp_override
    if dry_side_risk:
        if temp_air >= 28.0:
            vent_cap = 0.35
        elif temp_air >= 24.0:
            vent_cap = 0.18
        else:
            vent_cap = 0.12
        guarded[3] = min(guarded[3], vent_cap)
        if temp_air >= 12.0:
            guarded[0] = 0.0
        guarded[1] = 0.0
        guarded[4] = 0.0
        if temp_air < 18.0:
            guarded[2] = max(guarded[2], 0.70)
        if glob_rad > 250.0 or temp_air > 24.0:
            guarded[5] = max(guarded[5], 0.50)

    extreme_dew_risk = rh_air >= 94.0 or dew_margin < 0.4
    if temp_air < 12.0 and not extreme_dew_risk:
        guarded[0] = max(guarded[0], 0.90)
        guarded[2] = max(guarded[2], 0.90)
        guarded[3] = min(guarded[3], 0.12)
        guarded[1] = 0.0
        guarded[4] = 0.0
    elif (
        12.0 <= temp_air < 15.5
        and rh_air < 90.0
        and dew_margin >= 0.8
        and (guarded[3] > 0.25 or rh_air >= 86.0)
    ):
        guarded[0] = max(guarded[0], 0.18 if temp_air < 13.0 else 0.08)
        guarded[2] = max(guarded[2], 0.65)
        guarded[3] = min(guarded[3], 0.25)
        guarded[1] = 0.0
        guarded[4] = 0.0

    return np.clip(guarded, 0.0, 1.0)


def log_control_tracking(prefix: str, target_control: np.ndarray, applied_control: np.ndarray, action_cmd: Optional[np.ndarray] = None) -> None:
    """记录控制跟踪日志：对比目标控制量与环境实际执行量。

    作用:
    1) 将目标控制 `target_control` 与实际控制 `applied_control` 按执行器逐项对齐输出；
    2) 显示偏差 `差值=执行-目标`，用于快速判断执行误差与护栏/限幅影响；
    3) 若提供 `action_cmd`（速率受限动作指令），同步打印每路 cmd，便于排查“目标到执行”的转换过程。
    """
    names = ["加热", "CO2", "保温幕", "通风", "补光", "遮阳"]
    # 统一转为一维 float32，避免列表/矩阵形状差异导致日志错位
    target = np.asarray(target_control, dtype=np.float32).flatten()
    applied = np.asarray(applied_control, dtype=np.float32).flatten()
    cmd = None if action_cmd is None else np.asarray(action_cmd, dtype=np.float32).flatten()
    # 取三者最小长度，保证索引安全且逐项可比
    n = min(len(names), len(target), len(applied))
    records = []
    for i in range(n):
        # 正值表示实际执行高于目标，负值表示低于目标
        diff = applied[i] - target[i]
        if cmd is None or i >= len(cmd):
            records.append(f"{names[i]}(目标:{target[i]:.2f},执行:{applied[i]:.2f},差值:{diff:+.2f})")
        else:
            # cmd 为归一化动作命令（常见于 step(action_cmd) 路径）
            records.append(f"{names[i]}(目标:{target[i]:.2f},执行:{applied[i]:.2f},差值:{diff:+.2f},cmd:{cmd[i]:+.2f})")
    print(f"{prefix} 控制跟踪 -> " + ", ".join(records))


class GreenhouseAgent:
    """
    纯 LLM 智能体 (Pure LLM Agent)
    
    完全依赖 LLM 进行决策，适用于复杂场景或全托管模式。
    基于 LangChain (LangGraph) 构建。
    """
    
    def __init__(
        self,
        agent_interface,
        tools: List,
        config: Optional[AgentConfig] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None
    ):
        self.interface = agent_interface
        self.tools = tools
        self.config = config or AgentConfig()
        self.system_prompt = system_prompt or GreenhousePrompts.SYSTEM_PROMPT
        self.user_prompt_template = user_prompt_template or GreenhousePrompts.USER_PROMPT_TEMPLATE
        
        self.llm = self._create_llm()
        self.agent_graph = self._create_agent_graph()
        
        self.history: List[BaseMessage] = []
        self.total_reward = 0.0
        self.episode_count = 0
        
    def _create_llm(self):
        return ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            verbose=self.config.verbose
        )
    
    def _create_agent_graph(self):
        """创建 LangGraph Agent 执行图"""
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_iterations=self.config.max_iterations
        )
    
    def _build_user_prompt(self, state_description: str) -> str:
        return self.user_prompt_template.format(state=state_description)
    
    def reset(self):
        self.history = []
        self.total_reward = 0.0
        self.episode_count += 1
    
    def get_context(self) -> str:
        return f"这是第 {self.episode_count + 1} 个生长季，当前时间步 {self.interface.get_state().timestep}"
    
    def step(self, context: Optional[str] = None) -> Dict:
        """
        执行一步 LLM 决策（单步闭环）。

        流程分为 5 个阶段：
        1) 准备阶段：重置工具缓冲区、读取状态并构造用户输入。
        2) 推理阶段：调用 LangGraph Agent，让 LLM 产出回复并（可选）写入工具动作缓冲。
        3) 执行阶段：从工具缓冲读取最终控制量，经过护栏与动作映射后执行 env.step。
        4) 记录阶段：更新接口中的 reward/终止标记，并维护会话历史。
        5) 返回阶段：输出统一结构字典；若异常则返回错误信息。

        注意：
        - 这里采用“工具先缓冲、最后统一执行”的模式，避免一次推理中多次直接驱动环境。
        - 若本轮没有可执行工具动作，函数仍会返回 success=True，但 reward 保持默认值 0.0。
        """
        # 1. 重置工具缓冲区
        # 每一步开始都清空上一步残留动作，避免历史工具调用污染当前决策。
        if hasattr(create_langchain_tools, "instance"):
             create_langchain_tools.instance.reset_buffer()

        # 读取当前环境状态与描述文本，作为本轮 prompt 的主要上下文。
        state = self.interface.get_state()
        state_desc = self.interface.get_state_description()
        user_input = self._build_user_prompt(state_desc)
        
        # 可选附加外部上下文（例如“当前实验目标/额外约束”）。
        if context:
            user_input = f"{context}\n\n{user_input}"
        
        try:
            # 2) 构造消息序列：历史对话 + 当前用户输入。
            # 历史上下文有助于 LLM 保持策略连续性。
            current_messages = self.history + [HumanMessage(content=user_input)]
            
            # 调用 LangGraph Agent 执行一次完整推理。
            # LLM 在此阶段可调用工具，但工具只更新缓冲区，不直接 step 环境。
            result_state = self.agent_graph.invoke({"messages": current_messages})
            
            # 读取模型最终文本输出（用于日志或上层展示）。
            messages = result_state.get("messages", [])
            output = ""
            if messages and isinstance(messages[-1], AIMessage):
                output = messages[-1].content
            
            # 3) 统一执行环境步进：仅在工具实例可用时尝试取动作并执行。
            tools_instance = None
            if hasattr(create_langchain_tools, "instance"):
                tools_instance = create_langchain_tools.instance
            
            # 默认返回值（无动作时沿用）。
            reward = 0.0
            done = False
            
            if tools_instance:
                 # 从工具缓冲读取本轮目标动作，先打印“目标动作摘要”方便排查。
                 action = tools_instance.buffered_action
                 action_log = []
                 if action.u_boil > 0: action_log.append(f"加热:{action.u_boil:.2f}")
                 if action.u_vent > 0: action_log.append(f"通风:{action.u_vent:.2f}")
                 if action.u_co2 > 0: action_log.append(f"CO2:{action.u_co2:.2f}")
                 if action.u_th_scr > 0: action_log.append(f"保温幕:{action.u_th_scr:.2f}")
                 if action.u_lamp > 0: action_log.append(f"补光:{action.u_lamp:.2f}")
                 if action.u_bl_scr > 0: action_log.append(f"遮阳:{action.u_bl_scr:.2f}")
                 
                 log_str = ", ".join(action_log) if action_log else "无动作"
                 print(f"[Agent] 正在执行缓冲动作 -> {log_str}")
                 
                 # 工具动作 -> 控制向量 -> 护栏修正。
                 # 护栏负责安全/经济约束，可能改写 LLM 原始控制目标。
                 target_control = tools_instance.buffered_action.to_array()
                 target_control = apply_safety_guardrails(state, target_control)

                 # 将目标控制映射为环境动作命令（考虑当前开度与每步变化限制）。
                 action_cmd = control_to_action_command(self.interface.env, target_control)

                 # 真正推进环境一步；该调用会更新环境内部状态。
                 obs, reward, terminated, truncated, info = self.interface.env.step(action_cmd)

                 # 记录“目标 vs 实际执行”差异，便于调试限速/裁剪/护栏影响。
                 log_control_tracking("[Agent]", target_control, self.interface.env.u, action_cmd)

                 # 4) 将本步执行结果写回接口缓存，供外层统一读取。
                 self.interface.last_reward = reward
                 self.interface.last_terminated = terminated
                 self.interface.last_truncated = truncated
                 done = terminated or truncated
                 print(f"[Agent] 动作执行完毕 -> 奖励: {reward:.4f}, 是否结束: {done}")

            # 维护对话历史：保存“本轮输入 + 模型文本输出”。
            # 仅保留最近 20 条，防止上下文无限增长。
            self.history.append(HumanMessage(content=user_input))
            self.history.append(AIMessage(content=output))
            
            if len(self.history) > 20:
                self.history = self.history[-20:]
            
            # 5) 正常返回：包含是否成功、模型输出、状态快照、奖励与结束标志。
            return {
                "success": True,
                "output": output,
                "state": state,
                "reward": reward,
                "done": done
            }
            
        except Exception as e:
            # 异常返回：保留当前 state 与错误文本，便于上层记录/恢复。
            return {
                "success": False,
                "error": str(e),
                "state": state,
                "reward": state.reward
            }
    
    def run_episode(self, max_steps: Optional[int] = None, callback: Optional[Callable] = None) -> Dict:
        """运行一个完整的 episode"""
        obs, info = self.interface.env.reset()
        self.reset()
        
        step_count = 0
        episode_reward = 0.0
        done = False
        truncated = False
        max_steps = max_steps or getattr(self.interface.env, 'N', 1000)
        
        while not (done or truncated) and step_count < max_steps:
            result = self.step()
            state = self.interface.get_state()
            reward = getattr(self.interface, 'last_reward', state.reward)
            
            if hasattr(self.interface.env, 'terminated'):
                done = self.interface.env.terminated
                truncated = self.interface.env.truncated
            else:
                done = step_count >= max_steps - 1
            
            episode_reward += reward
            step_count += 1
            
            if callback:
                callback(step_count, state, reward, result)
            
            if self.config.verbose and step_count % 100 == 0:
                print(f"Step {step_count}: Reward = {reward:.2f}, Total = {episode_reward:.2f}")
        
        return {
            "steps": step_count,
            "total_reward": episode_reward,
            "avg_reward": episode_reward / max(step_count, 1)
        }
    
    def save_history(self, filepath: str):
        history_data = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                history_data.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history_data.append({"type": "ai", "content": msg.content})
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    
    def load_history(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            history_data = json.load(f)
        self.history = []
        for msg in history_data:
            if msg["type"] == "human":
                self.history.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                self.history.append(AIMessage(content=msg["content"]))


class RuleBasedLLMDirector:
    """
     规则引导的 LLM 导演模式 (Hybrid Director) - 规划执行版

     核心逻辑：
     1. LLM 在激活时输出“阶段性控制规划”（锚点控制 + 可选 setpoints）。
     2. 规划生效期间由规则控制器逐步执行和跟踪该规划。
     3. 当规划到期或出现紧急状态时，提前触发一次 LLM 重规划。
    """
    
    def __init__(
        self,
        agent_interface,
        tools: List,
        config: Optional[AgentConfig] = None,
        env_id: Optional[str] = None,
        rule_params: Optional[Dict[str, Any]] = None
    ):
        self.interface = agent_interface
        self.tools = tools
        self.config = config or AgentConfig()
        self.llm = self._create_llm()
        self.agent_graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=GreenhousePrompts.SYSTEM_PROMPT
        )
        
        # 初始化规则阈值
        self.rules = self._init_rules()
        self.env_id = env_id or getattr(self.interface.env, "env_id", self.interface.env.__class__.__name__)
        # 加载规则控制器 (PID 或其他算法)
        self.rule_controller = self._init_rule_controller(rule_params)
        self.expert_policy = self._init_expert_policy()
        self.humidity_memory = self._init_humidity_memory()
        self.plan_cache = self._init_plan_cache()
        
        # 状态追踪
        self.steps_since_last_llm = 0
        self.last_control: Optional[np.ndarray] = None
        self.last_llm_state = None # 记录上一次 LLM 决策时的环境状态
        self.last_source = "none"
        self.current_plan: Optional[Dict[str, Any]] = None
        self.last_plan_message: str = ""
        self.last_fallback_selection: Dict[str, Any] = {}
        self.last_rollout_selection: Dict[str, Any] = {}
        self.last_expert_prediction: Dict[str, Any] = {}
        self.last_humidity_memory_prediction: Dict[str, Any] = {}
        self.last_plan_cache_event: Dict[str, Any] = {"mode": str(getattr(self.config, "plan_cache_mode", "off"))}
        self.last_humidity_memory_selected_step: int = -10**9
        self.last_llm_trigger_step: int = -10**9
        self.emergency_replan_cooldown_steps: int = max(6, int(self.config.control_interval // 2))
        self.emergency_replan_confirm_steps: int = 2
        self.emergency_replan_min_remaining_steps: int = 2
        self.emergency_streak_steps: int = 0
        self.dehumidify_mode: str = "normal"
        self.dehumidify_mode_hold_steps: int = 0
        self.dehumidify_exit_confirm_counter: int = 0
        self.dynamic_interval_active: bool = False
        self.dynamic_interval_on_counter: int = 0
        self.dynamic_interval_restore_counter: int = 0
        self.active_control_interval: int = int(self.config.control_interval)
        self._lamp_budget_day_marker: Optional[int] = None
        self._lamp_budget_integral: float = 0.0
        self.last_lamp_budget_remaining: float = float(self.config.lamp_daily_budget)
        self.rh_violation_debt: float = 0.0
        self.rh_emergency_streak_steps: int = 0
        self.pending_control: Optional[np.ndarray] = None
        
        # v2.1.2: 决策可解释性增强
        self.decision_history: List[Dict] = []
        self.current_reasoning: Optional[DecisionReasoning] = None
        self.current_metrics: Optional[PerformanceMetrics] = None
        self._rh_history: deque = deque(maxlen=6)
        self._weather_history: deque = deque(maxlen=12)
        
    def _create_llm(self):
        return ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
    
    def _init_rules(self) -> Dict:
        """定义温室环境的安全阈值"""
        return {
            "temp_low": 17.0,
            "temp_high": 25.0,
            "temp_critical_low": 15.0,
            "temp_critical_high": 30.0,
            "co2_low": 400.0,
            "co2_high": 1000.0,
            "rh_low": 50.0,
            "rh_high": 84.0
        }
    
    def _init_rule_controller(self, rule_params: Optional[Dict[str, Any]]) -> Optional[RuleBasedController]:
        if rule_params is None:
            try:
                rule_params = load_model_hyperparams("rule_based", self.env_id)
            except Exception:
                rule_params = None
        if not rule_params:
            return None
        return RuleBasedController(**rule_params)

    def _init_expert_policy(self) -> Optional[DistilledExpertPolicy]:
        if not bool(getattr(self.config, "expert_rollout_enabled", False)):
            return None
        policy_path = str(getattr(self.config, "expert_policy_path", "") or "")
        if not policy_path:
            return None
        try:
            policy = DistilledExpertPolicy.load(policy_path)
            print(f"[Director] Loaded distilled expert rollout policy: {policy_path}")
            return policy
        except FileNotFoundError:
            print(f"[Director] Distilled expert policy not found, disabling expert rollout: {policy_path}")
        except Exception as exc:
            print(f"[Director] Failed to load distilled expert policy, disabling expert rollout: {exc}")
        return None

    def _init_humidity_memory(self) -> Optional[HumidityExperienceMemory]:
        if not bool(getattr(self.config, "humidity_memory_enabled", False)):
            return None
        memory_path = str(getattr(self.config, "humidity_memory_path", "") or "")
        if not memory_path:
            return None
        try:
            memory = HumidityExperienceMemory.load_jsonl(memory_path)
            print(f"[Director] Loaded humidity experience memory: {memory_path}")
            return memory
        except FileNotFoundError:
            print(f"[Director] Humidity memory not found, disabling HEM rollout: {memory_path}")
        except Exception as exc:
            print(f"[Director] Failed to load humidity memory, disabling HEM rollout: {exc}")
        return None

    def _init_plan_cache(self) -> Optional[PlanCache]:
        mode = str(getattr(self.config, "plan_cache_mode", "off") or "off").lower()
        if mode == "off":
            return None
        try:
            cache = PlanCache(
                getattr(self.config, "plan_cache_path", "gl_gym/result/plan_cache/llm_plan_cache.json"),
                mode=mode,
                strict=bool(getattr(self.config, "plan_cache_strict", False)),
            )
            print(f"[Director] LLM plan cache enabled: mode={mode}, path={cache.path}")
            return cache
        except Exception as exc:
            if bool(getattr(self.config, "plan_cache_strict", False)):
                raise
            print(f"[Director] Failed to initialize plan cache, disabling cache: {exc}")
            return None

    def _control_action_record(self, action: ControlAction) -> Dict[str, Any]:
        return {
            "u_boil": float(action.u_boil),
            "u_co2": float(action.u_co2),
            "u_th_scr": float(action.u_th_scr),
            "u_vent": float(action.u_vent),
            "u_lamp": float(action.u_lamp),
            "u_bl_scr": float(action.u_bl_scr),
            "action_set": bool(action.action_set),
        }

    def _restore_cached_action(self, tools_instance: GreenhouseTools, entry: Dict[str, Any]) -> bool:
        action_record = entry.get("buffered_action", {})
        if not isinstance(action_record, dict) or not bool(action_record.get("action_set", False)):
            return False
        tools_instance.buffered_action = ControlAction(
            u_boil=float(action_record.get("u_boil", 0.0)),
            u_co2=float(action_record.get("u_co2", 0.0)),
            u_th_scr=float(action_record.get("u_th_scr", 0.0)),
            u_vent=float(action_record.get("u_vent", 0.0)),
            u_lamp=float(action_record.get("u_lamp", 0.0)),
            u_bl_scr=float(action_record.get("u_bl_scr", 0.0)),
            action_set=True,
        )
        setpoints = entry.get("buffered_setpoints", {})
        tools_instance.buffered_setpoints = dict(setpoints) if isinstance(setpoints, dict) else {}
        return True

    def _make_plan_cache_entry(
        self,
        *,
        key: str,
        state,
        reason: str,
        planning_horizon: int,
        attempt: int,
        prompt_text: str,
        status_brief: str,
        llm_output: str,
        llm_duration: float,
        llm_action_found: bool,
        tools_instance: GreenhouseTools,
    ) -> Dict[str, Any]:
        config_fp = config_fingerprint(self.config)
        return {
            "schema_version": "llm_plan_cache_v1",
            "key": key,
            "mode_written": str(getattr(self.config, "plan_cache_mode", "off")),
            "model_name": str(getattr(self.config, "model_name", "")),
            "env_id": str(getattr(self, "env_id", "")),
            "reason": str(reason),
            "planning_horizon": int(planning_horizon),
            "attempt": int(attempt),
            "timestep": int(getattr(state, "timestep", 0)),
            "state_summary": state_summary_from_state(state),
            "prompt_hash": text_hash(prompt_text),
            "config_hash": stable_hash(config_fp),
            "config_fingerprint": config_fp,
            "status_brief": status_brief,
            "raw_response": str(llm_output or ""),
            "llm_duration_seconds": float(llm_duration),
            "llm_action_found": bool(llm_action_found),
            "buffered_action": self._control_action_record(tools_instance.buffered_action),
            "buffered_setpoints": to_jsonable(getattr(tools_instance, "buffered_setpoints", {})),
        }

    def _state_plan_memory_row(self, state, plan: Optional[Dict[str, Any]]) -> Dict[str, float]:
        plan = plan or {}
        target_temp = self._get_plan_target(plan, "target_temp", state)
        target_co2 = self._get_plan_target(plan, "target_co2", state)
        target_rh = self._get_plan_target(plan, "target_rh", state)
        return {
            "timestep": float(getattr(state, "timestep", 0.0)),
            "step": float(getattr(state, "timestep", 0.0)),
            "hour_of_day": float(getattr(state, "hour_of_day", 12.0)),
            "day_of_year": float(getattr(state, "day_of_year", 180.0)),
            "temp_air": float(getattr(state, "temp_air", 20.0)),
            "rh_air": float(getattr(state, "rh_air", 70.0)),
            "co2_air": float(getattr(state, "co2_air", 430.0)),
            "glob_rad": float(getattr(state, "glob_rad", 0.0)),
            "temp_out": float(getattr(state, "temp_out", getattr(state, "temp_air", 20.0))),
            "rh_out": float(getattr(state, "rh_out", 70.0)),
            "wind_speed": float(getattr(state, "wind_speed", 0.0)),
            "dew_margin_air": float(getattr(state, "dew_margin_air", 3.0)),
            "canopy_dew_margin": float(getattr(state, "canopy_dew_margin", 3.0)),
            "forecast_humidity_risk": float(getattr(state, "forecast_humidity_risk", 0.0)),
            "target_temp": float(target_temp if target_temp is not None else getattr(state, "temp_air", 20.0)),
            "target_co2": float(target_co2 if target_co2 is not None else getattr(state, "co2_air", 430.0)),
            "target_rh": float(target_rh if target_rh is not None else 76.0),
        }

    def _predict_humidity_memory_control(
        self,
        state,
        plan: Optional[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        enabled = bool(getattr(self.config, "humidity_memory_enabled", False))
        self.last_humidity_memory_prediction = {"enabled": enabled}
        if not enabled:
            self.last_humidity_memory_prediction["available"] = False
            return None
        memory = getattr(self, "humidity_memory", None)
        if memory is None:
            self.last_humidity_memory_prediction["available"] = False
            return None
        try:
            row = self._state_plan_memory_row(state, plan)
            required_metadata = {
                key: value
                for key, value in {
                    "teacher_policy_id": getattr(self.config, "humidity_memory_teacher_policy_id", ""),
                    "baseline_controller_id": getattr(self.config, "humidity_memory_baseline_controller_id", ""),
                    "memory_schema_version": getattr(self.config, "humidity_memory_version", ""),
                }.items()
                if value
            }
            matches = memory.retrieve(
                row,
                target_row=row,
                top_k=int(getattr(self.config, "humidity_memory_top_k", 1)),
                max_distance=float(getattr(self.config, "humidity_memory_max_distance", 1.15)),
                min_trust=float(getattr(self.config, "humidity_memory_min_trust", 0.50)),
                required_metadata=required_metadata or None,
            )
            if not matches:
                self.last_humidity_memory_prediction = {
                    "enabled": True,
                    "available": False,
                    "accepted": False,
                    "required_metadata": required_metadata,
                }
                return None
            match = matches[0]
            control = np.clip(np.asarray(match["candidate_action"], dtype=np.float32), 0.0, 1.0)
            strategy = label_strategy(action_record_for_labeling(state, control), action_prefix="u")
            self.last_humidity_memory_prediction = {
                "enabled": True,
                "available": True,
                "accepted": True,
                "case_id": match.get("case_id"),
                "distance": float(match.get("distance", 0.0)),
                "trust": float(match.get("trust", 0.0)),
                "support_count": int(match.get("support_count", 0)),
                "status": match.get("status"),
                "score": float(match.get("score", 0.0)),
                "strategy_label": strategy.label,
                "strategy_confidence": float(strategy.confidence),
                "required_metadata": required_metadata,
            }
            return control
        except Exception as exc:
            self.last_humidity_memory_prediction = {
                "enabled": True,
                "available": True,
                "accepted": False,
                "error": str(exc),
            }
            print(f"[Director] Humidity memory rollout failed: {exc}")
            return None

    def _predict_expert_control(
        self,
        state,
        plan: Optional[Dict[str, Any]],
        rh_debt: float,
        lamp_budget_remaining: float,
        dehumidify_mode: str,
    ) -> Optional[np.ndarray]:
        self.last_expert_prediction = {"enabled": bool(getattr(self.config, "expert_rollout_enabled", False))}
        expert_policy = getattr(self, "expert_policy", None)
        if expert_policy is None:
            self.last_expert_prediction["available"] = False
            return None
        try:
            intent = infer_plan_intent(state, plan)
            effective_dehumidify_mode = str(dehumidify_mode or "normal")
            if effective_dehumidify_mode == "normal":
                effective_dehumidify_mode = intent_to_dehumidify_mode(intent.label)
            control, info = expert_policy.predict(
                state,
                plan=plan,
                rh_violation_debt=rh_debt,
                lamp_budget_remaining=lamp_budget_remaining,
                dehumidify_mode=effective_dehumidify_mode,
            )
            control = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
            strategy = label_strategy(action_record_for_labeling(state, control), action_prefix="u")
            alignment = strategy_intent_alignment(strategy, intent)
            distance = float(info.get("feature_distance", 0.0))
            max_distance = float(getattr(self.config, "expert_candidate_max_distance", 4.0))
            if max_distance <= 0.0:
                metadata_threshold = info.get("distance_threshold_p99", info.get("distance_threshold"))
                try:
                    max_distance = float(metadata_threshold) * 1.20
                except Exception:
                    max_distance = 4.0
            accepted = distance <= max_distance
            reject_reason = "accepted"
            if not accepted:
                reject_reason = "feature_distance"
            intent_gate_enabled = bool(getattr(self.config, "expert_intent_gate_enabled", True))
            if (
                accepted
                and intent_gate_enabled
                and intent.is_confident
                and strategy.is_confident
                and not bool(alignment.get("aligned", False))
            ):
                accepted = False
                reject_reason = "intent_mismatch"
            self.last_expert_prediction = {
                "enabled": True,
                "available": True,
                "accepted": accepted,
                "reject_reason": reject_reason,
                "feature_distance": distance,
                "max_abs_z": float(info.get("max_abs_z", 0.0)),
                "max_distance": max_distance,
                "distance_threshold": info.get("distance_threshold"),
                "distance_threshold_p99": info.get("distance_threshold_p99"),
                "model": info.get("model", "distilled_expert"),
                "train_cases": info.get("train_cases"),
                "target_mode": info.get("target_mode", "action"),
                "plan_intent_label": intent.label,
                "plan_intent_confidence": float(intent.confidence),
                "effective_dehumidify_mode": effective_dehumidify_mode,
                "expert_strategy_label": strategy.label,
                "expert_strategy_confidence": float(strategy.confidence),
                "intent_strategy_alignment": alignment,
            }
            if not accepted:
                return None
            return control
        except Exception as exc:
            self.last_expert_prediction = {
                "enabled": True,
                "available": True,
                "accepted": False,
                "error": str(exc),
            }
            print(f"[Director] Distilled expert rollout failed: {exc}")
            return None
    
    def _get_weather_vector(self) -> np.ndarray:
        env = self.interface.env
        if hasattr(env, "weather_data") and hasattr(env, "timestep"):
            if 0 <= env.timestep < len(env.weather_data):
                return env.weather_data[env.timestep]
        return np.zeros(getattr(env, "nd", 10))

    def _extract_status_brief(self, status_text: str) -> str:
        """从 get_status 文本中提取关键状态，避免提示词过长。"""
        if not status_text:
            return ""

        # 去掉工具返回里的缓冲区说明，仅保留环境状态主体。
        core_text = status_text.split("【本轮待执行控制（缓冲区）】")[0]
        lines = [line.strip() for line in core_text.splitlines() if line.strip()]
        keep_keywords = (
            "Step:",
            "室内温度:",
            "相对湿度:",
            "饱和水汽压差",
            "CO2 浓度:",
            "加热管道温度:",
            "气象预报(1h):",
        )
        selected = [line for line in lines if any(keyword in line for keyword in keep_keywords)]
        if not selected:
            selected = lines[:8]
        return " | ".join(selected[:8])

    def _build_compact_prompt(self, state, analysis: Dict, mode: str, status_brief: Optional[str] = None, horizon: Optional[int] = None) -> str:
        is_day = 6 <= state.hour_of_day <= 18
        effective_horizon = max(1, int(horizon if horizon is not None else self.active_control_interval))
        remaining_steps = effective_horizon - 1
        status_block = ""
        if status_brief:
            status_block = f"强制状态快照(get_status):{status_brief}\\n"

        return (
            f"模式:{mode}\n"
            f"时间:day={state.day_of_year:.1f},hour={state.hour_of_day:.1f},is_day={int(is_day)}\n"
            f"环境:T={state.temp_air:.2f},RH={state.rh_air:.2f},CO2={state.co2_air:.2f},Rad={state.glob_rad:.2f},Tout={state.temp_out:.2f}\n"
            f"作物:Fruit={state.fruit_weight:.6f},Can24={state.canopy_temp_24h:.2f},TSum={state.temperature_sum:.2f}\n"
            f"当前控制:heat={state.u_boil:.2f},co2={state.u_co2:.2f},scr={state.u_th_scr:.2f},vent={state.u_vent:.2f},lamp={state.u_lamp:.2f},blscr={state.u_bl_scr:.2f}\n"
            f"建议:{' | '.join(analysis.get('suggestions', [])[:4]) if analysis.get('suggestions') else '状态平稳，按经济性微调'}\n"
            f"{status_block}"
            f"请为未来 {effective_horizon} 步（即包含当步与随后 {remaining_steps} 步）制定控制规划："
            "先调用一次 set_all_controls 给出本步控制锚点。"
            "同时必须给出未来阶段目标 target_temp/target_co2/target_rh（任何目标不得留空，若不确定请给保守值）。"
            f"若能判断趋势，可额外给出 target_temp_profile/target_co2_profile/target_rh_profile，长度为 {effective_horizon}；"
            "没有把握时只给标量目标即可，系统会自动扩展为保守目标轨迹。"
            "后续时间步将由规则控制器依据这些目标自动追踪与微调。"
            "利润优先：电费是首要成本，除非低辐射且湿度安全，否则不要大补光；"
            "仅在有效光照且低通风时提升 CO2；高湿时优先除湿并避免高加热+高通风并发。"
        )

    def _is_plan_active(self, timestep: int) -> bool:
        if self.current_plan is None:
            return False
        return timestep < int(self.current_plan.get("expires_timestep", -1))

    def _update_lamp_budget(self, state) -> float:
        """按 day_of_year 统计补光预算，超过预算后抑制补光。"""
        day_marker = int(getattr(state, "day_of_year", 0))
        if self._lamp_budget_day_marker != day_marker:
            self._lamp_budget_day_marker = day_marker
            self._lamp_budget_integral = 0.0

        self._lamp_budget_integral += float(getattr(state, "u_lamp", 0.0))
        remaining = max(float(self.config.lamp_daily_budget) - self._lamp_budget_integral, 0.0)
        self.last_lamp_budget_remaining = remaining
        return remaining

    def _update_dynamic_interval(self, state) -> int:
        """分层高湿时缩短重规划间隔，恢复后再回到默认间隔。"""
        rh = float(state.rh_air)
        on_threshold = float(self.config.high_rh_dynamic_interval_on)
        mild_on_threshold = float(self.config.high_rh_dynamic_interval_mild_on)
        off_threshold = float(self.config.high_rh_dynamic_interval_off)
        min_steps = int(max(1, self.config.high_rh_dynamic_min_steps))

        # 直接处理严重高湿情况（紧急模式）
        if rh >= on_threshold:
            self.dynamic_interval_active = True
            self.dynamic_interval_on_counter = 0
            self.dynamic_interval_restore_counter = 0
            short_interval = int(max(1, self.config.high_rh_urgent_short_interval))
            self.active_control_interval = min(int(self.config.control_interval), short_interval)
            return self.active_control_interval
        
        # 中度高湿情况（轻度模式）
        if rh >= mild_on_threshold:
            if not self.dynamic_interval_active:
                self.dynamic_interval_on_counter += 1
                if self.dynamic_interval_on_counter >= min_steps:
                    self.dynamic_interval_active = True
                    self.dynamic_interval_on_counter = 0
                    self.dynamic_interval_restore_counter = 0
            else:
                self.dynamic_interval_on_counter = 0
        else:
            self.dynamic_interval_on_counter = 0

        if self.dynamic_interval_active:
            if rh <= off_threshold:
                self.dynamic_interval_restore_counter += 1
                if self.dynamic_interval_restore_counter >= int(self.config.high_rh_restore_steps):
                    self.dynamic_interval_active = False
                    self.dynamic_interval_on_counter = 0
                    self.dynamic_interval_restore_counter = 0
            else:
                self.dynamic_interval_restore_counter = 0

        if self.dynamic_interval_active:
            # 根据湿度水平选择不同的间隔
            if rh >= on_threshold:
                short_interval = int(max(1, self.config.high_rh_urgent_short_interval))
            elif rh >= mild_on_threshold:
                short_interval = int(max(1, self.config.high_rh_mild_short_interval))
            else:
                short_interval = int(max(1, self.config.high_rh_short_interval))
            self.active_control_interval = min(int(self.config.control_interval), short_interval)
        else:
            self.active_control_interval = int(max(1, self.config.control_interval))

        return self.active_control_interval

    def _update_dehumidify_mode(self, state) -> str:
        """带滞回+持锁+退出确认的湿度分级状态机，减少来回抖动。v2.1.2: 改为双阈值退出逻辑"""
        rh = float(state.rh_air)
        strong_on = float(self.config.dehumidify_strong_rh_on)
        mild_on = float(self.config.dehumidify_mild_rh_on)
        mild_off = float(self.config.dehumidify_mild_rh_off)
        strong_off = float(self.config.dehumidify_strong_rh_off)
        mild_exit_confirm_steps = int(self.config.dehumidify_mild_exit_confirm_steps)
        strong_exit_confirm_steps = int(self.config.dehumidify_strong_exit_confirm_steps)

        if self.dehumidify_mode == "normal":
            if rh >= strong_on:
                self.dehumidify_mode = "strong"
                self.dehumidify_mode_hold_steps = int(self.config.dehumidify_strong_hold_steps)
                self.dehumidify_exit_confirm_counter = 0
            elif rh >= mild_on:
                self.dehumidify_mode = "mild"
                self.dehumidify_mode_hold_steps = int(self.config.dehumidify_mild_hold_steps)
                self.dehumidify_exit_confirm_counter = 0
            return self.dehumidify_mode

        if self.dehumidify_mode == "mild":
            if rh >= strong_on:
                self.dehumidify_mode = "strong"
                self.dehumidify_mode_hold_steps = int(self.config.dehumidify_strong_hold_steps)
                self.dehumidify_exit_confirm_counter = 0
                return self.dehumidify_mode

            if self.dehumidify_mode_hold_steps > 0:
                self.dehumidify_mode_hold_steps -= 1
                return self.dehumidify_mode

            if rh <= mild_off:
                self.dehumidify_exit_confirm_counter += 1
                if self.dehumidify_exit_confirm_counter >= mild_exit_confirm_steps:
                    self.dehumidify_mode = "normal"
                    self.dehumidify_mode_hold_steps = 0
                    self.dehumidify_exit_confirm_counter = 0
            else:
                self.dehumidify_exit_confirm_counter = 0
            return self.dehumidify_mode

        # strong mode
        if self.dehumidify_mode_hold_steps > 0:
            self.dehumidify_mode_hold_steps -= 1
            return self.dehumidify_mode

        if rh <= strong_off:
            self.dehumidify_exit_confirm_counter += 1
            if self.dehumidify_exit_confirm_counter >= strong_exit_confirm_steps:
                self.dehumidify_mode = "normal"
                self.dehumidify_mode_hold_steps = 0
                self.dehumidify_exit_confirm_counter = 0
        elif rh < strong_on:
            self.dehumidify_mode = "mild"
            self.dehumidify_mode_hold_steps = int(self.config.dehumidify_mild_hold_steps)
            self.dehumidify_exit_confirm_counter = 0
        else:
            self.dehumidify_exit_confirm_counter = 0

        if rh >= strong_on:
            self.dehumidify_mode = "strong"
            self.dehumidify_mode_hold_steps = max(self.dehumidify_mode_hold_steps, int(self.config.dehumidify_strong_hold_steps) // 2)
            self.dehumidify_exit_confirm_counter = 0

        return self.dehumidify_mode

    def _update_rh_violation_debt(self, state) -> float:
        """Accumulate near-constraint RH risk so repeated small violations are not ignored."""
        rh = float(getattr(state, "rh_air", 0.0))
        temp = float(getattr(state, "temp_air", 20.0))
        vpd = float(calculate_vpd_kpa(temp, rh))
        dew_margin = min(
            float(getattr(state, "dew_margin_air", 3.0)),
            float(getattr(state, "canopy_dew_margin", 3.0)),
        )

        preemptive_risk = max(rh - float(self.config.rh_preemptive_threshold), 0.0)
        violation_risk = max(rh - float(self.config.rh_control_limit), 0.0)
        if rh >= 82.0:
            vpd_risk_threshold = 0.42
        elif rh >= 78.0:
            vpd_risk_threshold = 0.25
        else:
            vpd_risk_threshold = 0.0
        vpd_risk = max(vpd_risk_threshold - vpd, 0.0) * 8.0
        dew_risk = max(1.0 - dew_margin, 0.0) * 2.0

        debt = float(self.rh_violation_debt) * float(self.config.rh_debt_decay)
        debt += 0.35 * preemptive_risk + 1.25 * violation_risk + vpd_risk + dew_risk

        if rh < 82.0 and vpd > 0.55 and dew_margin > 1.5:
            debt *= 0.35

        self.rh_violation_debt = float(np.clip(debt, 0.0, 30.0))
        return self.rh_violation_debt

    def _should_emergency_replan(self, analysis: Dict[str, Any], state) -> bool:
        # 首先检查一般的紧急重规划条件
        general_emergency = self._should_general_emergency_replan(analysis, state)
        
        # 然后检查RH紧急重规划条件
        rh_emergency = self._should_rh_emergency_replan(state)
        
        # 如果任一条件满足，则触发紧急重规划
        return general_emergency or rh_emergency

    def _should_general_emergency_replan(self, analysis: Dict[str, Any], state) -> bool:
        if not analysis.get("is_critical", False):
            self.emergency_streak_steps = 0
            return False

        current_step = int(state.timestep)
        if self.current_plan is not None:
            remaining_steps = int(self.current_plan.get("expires_timestep", current_step + 1)) - current_step
            if remaining_steps <= self.emergency_replan_min_remaining_steps:
                return False

        if current_step - self.last_llm_trigger_step < self.emergency_replan_cooldown_steps:
            return False

        self.emergency_streak_steps += 1
        critical_level = analysis.get("critical_level", "normal")
        required_streak = 1 if critical_level == "severe" else self.emergency_replan_confirm_steps
        return self.emergency_streak_steps >= required_streak

    def _should_rh_emergency_replan(self, state) -> bool:
        """RH紧急重规划单独通道，比通用cooldown更灵敏"""
        rh = float(state.rh_air)
        threshold = float(self.config.emergency_rh_threshold)
        confirm_steps = int(self.config.emergency_rh_confirm_steps)
        
        current_step = int(state.timestep)
        
        # 检查是否超过RH紧急阈值
        if rh >= threshold:
            self.rh_emergency_streak_steps = getattr(self, 'rh_emergency_streak_steps', 0) + 1
            
            # 检查连续步数是否达到确认要求
            if self.rh_emergency_streak_steps >= confirm_steps:
                # 检查是否满足最小冷却时间（比通用冷却时间更短）
                # Deterministic heat-vent pulses handle RH between LLM calls; avoid token-heavy replans.
                rh_emergency_cooldown = max(
                    int(getattr(self.config, "rh_emergency_replan_cooldown_steps", 8)),
                    int(self.emergency_replan_cooldown_steps),
                )
                if current_step - self.last_llm_trigger_step >= rh_emergency_cooldown:
                    self.rh_emergency_streak_steps = 0  # 重置计数器
                    return True
        else:
            self.rh_emergency_streak_steps = 0  # 重置计数器
            
        return False

    def _predict_rule_control(self) -> Optional[np.ndarray]:
        env = self.interface.env
        if self.rule_controller is not None and hasattr(env, "x"):
            try:
                weather = self._get_weather_vector()
                return np.asarray(self.rule_controller.predict(env.x, weather, env), dtype=np.float32)
            except Exception:
                pass

        return None

    def _predict_rh_trend(self, state) -> str:
        """预测RH趋势"""
        current_rh = float(state.rh_air)
        self._rh_history.append(current_rh)
        
        if len(self._rh_history) < 4:
            return "stable"
        
        # 计算趋势
        recent_trend = (self._rh_history[-1] - self._rh_history[-4]) / 3
        
        if recent_trend > 0.5:
            return "rising"
        elif recent_trend < -0.5:
            return "falling"
        else:
            return "stable"

    def _predict_weather_trend(self, state) -> Dict[str, str]:
        """预测天气趋势"""
        current_weather = {
            "temp": float(state.temp_out),
            "rad": float(state.glob_rad),
            "rh": float(state.rh_out),
            "wind": float(state.wind_speed)
        }
        
        self._weather_history.append(current_weather)
        
        if len(self._weather_history) < 6:
            return {"temp": "unknown", "rad": "unknown", "rh": "unknown"}
        
        # 计算趋势
        trends = {}
        for key in ["temp", "rad", "rh"]:
            recent_avg = sum(w[key] for w in list(self._weather_history)[-3:]) / 3
            earlier_avg = sum(w[key] for w in list(self._weather_history)[-6:-3]) / 3
            diff = recent_avg - earlier_avg
            
            if diff > 0.5:
                trends[key] = "rising"
            elif diff < -0.5:
                trends[key] = "falling"
            else:
                trends[key] = "stable"
        
        return trends

    def _record_decision_reasoning(self, plan, state, analysis):
        """记录决策理由"""
        reasoning = DecisionReasoning()
        
        # 确定主要目标
        if analysis.get("is_critical", False):
            reasoning.primary_goal = "emergency_response"
        elif float(state.rh_air) > 85.0:
            reasoning.primary_goal = "humidity_control"
        elif float(state.fruit_weight) < 1.0:
            reasoning.primary_goal = "survival_mode"
        else:
            reasoning.primary_goal = "balanced_optimization"
        
        # 记录次要目标
        reasoning.secondary_goals = [
            "energy_saving",
            "yield_protection",
            "disease_prevention"
        ]
        
        # 记录应用的约束
        if float(state.rh_air) >= 86.0:
            reasoning.constraints_applied.append("lamp_forbidden_high_rh")
        if float(state.rh_air) >= 88.0:
            reasoning.constraints_applied.append("co2_forbidden_high_rh")
        
        # 记录预期结果
        reasoning.expected_outcome = (
            f"RH reduction to {plan.get('target_rh', 75.0):.1f}% "
            f"within {plan.get('plan_interval', 12)} steps"
        )
        
        self.current_reasoning = reasoning
        # 不在这里修改plan，而是在调用后由调用者处理
        return reasoning

    def _calculate_performance_metrics(self, state, control, plan):
        """计算实时性能指标"""
        metrics = PerformanceMetrics()
        
        # 成本效率
        metrics.cost_efficiency = self._calculate_cost_efficiency(state, control)
        
        # 产量预测
        metrics.yield_prediction = self._predict_yield(state)
        
        # 能源消耗
        metrics.energy_consumption = self._calculate_energy_consumption(control)
        
        # 病害风险评分
        metrics.disease_risk_score = self._calculate_disease_risk(state)
        
        # 控制精度
        metrics.control_accuracy = self._calculate_control_accuracy(state, control, plan)
        
        # 目标达成度
        metrics.setpoint_achievement = self._calculate_setpoint_achievement(state, plan)
        
        self.current_metrics = metrics
        return metrics

    def _calculate_cost_efficiency(self, state, control) -> float:
        """计算成本效率"""
        # 简化版本：基于当前控制量估算成本效率
        lamp_cost = float(control[4]) * 0.3  # 补光电费
        heat_cost = float(control[0]) * 0.09  # 加热电费
        co2_cost = float(control[1]) * 0.3   # CO2成本
        
        total_cost = lamp_cost + heat_cost + co2_cost
        if total_cost < 0.01:
            return 1.0
        
        # 基于当前环境状态评估成本合理性
        rh = float(state.rh_air)
        temp = float(state.temp_air)
        
        # 高湿时高成本是合理的（除湿需要）
        if rh > 85.0:
            return min(1.0, 0.5 / total_cost)
        
        # 低温时加热成本是合理的
        if temp < 15.0:
            return min(1.0, 0.3 / total_cost)
        
        # 正常情况下成本越低越好
        return max(0.0, 1.0 - total_cost)

    def _predict_yield(self, state) -> float:
        """预测产量"""
        # 简化版本：基于当前果重和生长条件预测
        fruit_weight = float(state.fruit_weight)
        temp = float(state.temp_air)
        rh = float(state.rh_air)
        
        # 基础产量
        base_yield = fruit_weight
        
        # 温度影响
        if 18.0 <= temp <= 22.0:
            temp_factor = 1.0
        elif 15.0 <= temp <= 25.0:
            temp_factor = 0.8
        else:
            temp_factor = 0.5
        
        # 湿度影响
        if 60.0 <= rh <= 80.0:
            rh_factor = 1.0
        elif 50.0 <= rh <= 90.0:
            rh_factor = 0.8
        else:
            rh_factor = 0.5
        
        return base_yield * temp_factor * rh_factor

    def _calculate_energy_consumption(self, control) -> float:
        """计算能源消耗"""
        # 简化版本：基于控制量计算能源消耗
        lamp_energy = float(control[4]) * 1.0  # 补光能耗系数
        heat_energy = float(control[0]) * 0.8  # 加热能耗系数
        co2_energy = float(control[1]) * 0.2   # CO2能耗系数
        
        return lamp_energy + heat_energy + co2_energy

    def _calculate_disease_risk(self, state) -> float:
        """计算病害风险评分"""
        rh = float(state.rh_air)
        temp = float(state.temp_air)
        
        # 湿度是主要风险因素
        if rh >= 90.0:
            rh_risk = 1.0
        elif rh >= 85.0:
            rh_risk = 0.7
        elif rh >= 80.0:
            rh_risk = 0.4
        else:
            rh_risk = 0.1
        
        # 温度也是风险因素（过高或过低）
        if temp >= 30.0 or temp <= 12.0:
            temp_risk = 0.8
        elif temp >= 25.0 or temp <= 15.0:
            temp_risk = 0.5
        else:
            temp_risk = 0.2
        
        return max(0.0, min(1.0, (rh_risk + temp_risk) / 2))

    def _calculate_control_accuracy(self, state, control, plan) -> float:
        """计算控制精度"""
        if plan is None:
            return 0.5
        
        # 简化版本：基于目标达成度评估控制精度
        target_temp = plan.get("target_temp")
        target_rh = plan.get("target_rh")
        target_co2 = plan.get("target_co2")
        
        accuracy_scores = []
        
        if target_temp is not None:
            temp_error = abs(float(state.temp_air) - target_temp)
            accuracy_scores.append(max(0.0, 1.0 - temp_error / 5.0))
        
        if target_rh is not None:
            rh_error = abs(float(state.rh_air) - target_rh)
            accuracy_scores.append(max(0.0, 1.0 - rh_error / 10.0))
        
        if target_co2 is not None:
            co2_error = abs(float(state.co2_air) - target_co2)
            accuracy_scores.append(max(0.0, 1.0 - co2_error / 200.0))
        
        return sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.5

    def _calculate_setpoint_achievement(self, state, plan):
        """评估目标达成度"""
        if plan is None:
            return {}
        
        achievement = {}
        
        target_temp = plan.get("target_temp")
        if target_temp is not None:
            temp_error = abs(float(state.temp_air) - target_temp)
            achievement["temp"] = max(0, 1 - temp_error / 5.0)
        
        target_rh = plan.get("target_rh")
        if target_rh is not None:
            rh_error = abs(float(state.rh_air) - target_rh)
            achievement["rh"] = max(0, 1 - rh_error / 10.0)
        
        target_co2 = plan.get("target_co2")
        if target_co2 is not None:
            co2_error = abs(float(state.co2_air) - target_co2)
            achievement["co2"] = max(0, 1 - co2_error / 200.0)
        
        return achievement

    def _track_decision_history(self, decision_data):
        """追踪决策历史"""
        # 限制历史记录长度
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
        
        self.decision_history.append({
            "timestep": int(decision_data.get("timestep", 0)),
            "decision_type": decision_data.get("decision_type", "unknown"),
            "primary_goal": decision_data.get("primary_goal", ""),
            "control_values": decision_data.get("control_values", []),
            "expected_outcome": decision_data.get("expected_outcome", ""),
            "actual_outcome": decision_data.get("actual_outcome", ""),
            "success": decision_data.get("success", False),
            "metrics": decision_data.get("metrics", {})
        })

    def _generate_state_adaptive_control(self, state, base_control: np.ndarray) -> np.ndarray:
        """基于当前环境状态生成自适应控制，避免仅复用历史动作。"""
        control = np.asarray(base_control, dtype=np.float32).copy()

        temp_air = float(state.temp_air)
        rh_air = float(state.rh_air)
        co2_air = float(state.co2_air)
        glob_rad = float(state.glob_rad)
        is_day = 6 <= float(state.hour_of_day) <= 18

        # 温度项：偏冷增热减风，偏热增风降热
        if temp_air < self.rules["temp_low"]:
            temp_err = self.rules["temp_low"] - temp_air
            control[0] = np.clip(control[0] + self.config.fallback_temp_gain * temp_err, 0.0, 1.0)
            control[3] = np.clip(control[3] - 0.5 * self.config.fallback_temp_gain * temp_err, 0.0, 1.0)
        elif temp_air > self.rules["temp_high"]:
            temp_err = temp_air - self.rules["temp_high"]
            control[3] = np.clip(control[3] + self.config.fallback_temp_gain * temp_err, 0.0, 1.0)
            control[0] = np.clip(control[0] - self.config.fallback_temp_gain * temp_err, 0.0, 1.0)

        # 湿度项：高湿优先增风，过干则减风
        if rh_air > self.rules["rh_high"]:
            rh_err = rh_air - self.rules["rh_high"]
            control[3] = np.clip(control[3] + self.config.fallback_rh_gain * rh_err, 0.0, 1.0)
            if temp_air < self.rules["temp_low"] + 1.0:
                control[0] = np.clip(control[0] + 0.5 * self.config.fallback_rh_gain * rh_err, 0.0, 1.0)
        elif rh_air < self.rules["rh_low"]:
            rh_err = self.rules["rh_low"] - rh_air
            control[3] = np.clip(control[3] - self.config.fallback_rh_gain * rh_err, 0.0, 1.0)

        # CO2 项：白天低风低辐射条件满足时补 CO2，夜间/弱光尽量收敛
        if (
            is_day
            and glob_rad > self.config.fallback_co2_min_rad
            and control[3] <= self.config.fallback_co2_max_vent
            and co2_air < self.rules["co2_low"]
        ):
            co2_err = self.rules["co2_low"] - co2_air
            control[1] = np.clip(control[1] + self.config.fallback_co2_gain * co2_err, 0.0, 1.0)
        elif (not is_day) or glob_rad < (0.5 * self.config.fallback_co2_min_rad):
            control[1] = np.clip(control[1], 0.0, 0.05)

        # 补光与夜间保温项
        if (
            is_day
            and glob_rad < self.config.fallback_lamp_rad_threshold
            and temp_air > 14.0
            and rh_air < 82.0
            and control[3] <= 0.25
        ):
            lamp_floor = float(self.config.fallback_day_lamp_level)
            if glob_rad < 40.0:
                lamp_floor = min(lamp_floor + 0.10, float(self.config.fallback_lamp_max_level))
            control[4] = np.clip(max(control[4], lamp_floor), 0.0, float(self.config.fallback_lamp_max_level))
        else:
            control[4] = 0.0

        if not is_day and temp_air < 18.0:
            control[2] = max(control[2], 0.80)

        return np.clip(control, 0.0, 1.0).astype(np.float32)

    def _estimate_candidate_response(self, state, control: np.ndarray) -> Dict[str, float]:
        """轻量级一阶响应估计，用于候选排序而非替代 GreenLight 物理模型。"""
        control = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        temp = float(state.temp_air)
        rh = float(state.rh_air)
        co2 = float(state.co2_air)
        rad = float(state.glob_rad)
        temp_out = float(getattr(state, "temp_out", temp))
        rh_out = float(getattr(state, "rh_out", rh))
        is_day = 6 <= float(state.hour_of_day) <= 18

        heat, co2_u, screen, vent, lamp, shade = [float(x) for x in control[:6]]
        total_rad = rad + 100.0 * lamp

        vent_cooling = vent * max(temp - temp_out, 0.0) * 0.18
        shade_cooling = shade * min(rad / 500.0, 1.0) * 0.18
        screen_retention = screen * (0.10 if not is_day else 0.03)
        temp_next = temp + 0.62 * heat + 0.16 * lamp + screen_retention - vent_cooling - shade_cooling

        outdoor_drier = 1.0 if rh_out <= rh else -0.35
        vent_rh_delta = -5.0 * vent * outdoor_drier
        heat_rh_delta = -1.5 * heat
        lamp_rh_delta = -0.4 * lamp
        screen_rh_delta = 0.45 * screen if not is_day else 0.10 * screen
        rh_next = np.clip(rh + vent_rh_delta + heat_rh_delta + lamp_rh_delta + screen_rh_delta, 35.0, 100.0)
        vpd_next = calculate_vpd_kpa(float(temp_next), float(rh_next))

        co2_assimilation = 20.0 if is_day and total_rad > 120.0 else 4.0
        co2_leak = 190.0 * vent * max((co2 - 410.0) / 500.0, 0.0)
        co2_next = max(320.0, co2 + 170.0 * co2_u - co2_leak - co2_assimilation)

        return {
            "temp_next": float(temp_next),
            "rh_next": float(rh_next),
            "vpd_next": float(vpd_next),
            "co2_next": float(co2_next),
            "total_rad": float(total_rad),
        }

    def _score_fallback_candidate(
        self,
        state,
        control: np.ndarray,
        analysis: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """为 fallback 候选给出可解释风险分数，分数越低越优。"""
        control = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        response = self._estimate_candidate_response(state, control)
        temp_next = response["temp_next"]
        rh_next = response["rh_next"]
        vpd_next = response.get("vpd_next", calculate_vpd_kpa(float(temp_next), float(rh_next)))
        co2_next = response["co2_next"]
        total_rad = response["total_rad"]

        fruit_weight = float(getattr(state, "fruit_weight", 0.0))
        temp_floor = 10.0 if fruit_weight < 1.0 else 12.0
        temp_penalty = max(temp_floor - temp_next, 0.0) ** 2 + 0.35 * max(temp_next - 30.0, 0.0) ** 2
        rh_penalty = (
            0.08 * max(rh_next - 84.0, 0.0) ** 2
            + 0.18 * max(rh_next - 90.0, 0.0) ** 2
            + 0.55 * max(rh_next - 95.0, 0.0) ** 2
        )
        dry_penalty = (
            0.10 * max(55.0 - rh_next, 0.0) ** 2
            + 0.35 * max(50.0 - rh_next, 0.0) ** 2
            + 1.25 * max(45.0 - rh_next, 0.0) ** 2
        )
        vpd_penalty = (
            0.85 * max(vpd_next - 1.20, 0.0) ** 2
            + 1.50 * max(vpd_next - 1.60, 0.0) ** 2
        )
        dew_margin = min(
            float(getattr(state, "dew_margin_air", 3.0)),
            float(getattr(state, "canopy_dew_margin", 3.0)),
        )
        dew_penalty = max(1.2 - dew_margin, 0.0) * (1.0 + max(float(state.rh_air) - 85.0, 0.0) / 10.0)

        energy_penalty = 0.70 * control[0] + 0.22 * control[1] + 1.05 * control[4] + 0.05 * control[3]
        conflict_penalty = 0.0
        if control[0] > 0.18 and control[3] > 0.22 and float(state.rh_air) < 88.0:
            conflict_penalty += (control[0] + control[3])
        if control[1] > 0.05 and (control[3] > 0.18 or total_rad < float(self.config.fallback_co2_min_rad)):
            conflict_penalty += 1.5 * control[1]
        if control[4] > 0.05 and (float(state.rh_air) >= 86.0 or control[3] >= 0.30):
            conflict_penalty += 1.2 * control[4]

        smooth_penalty = 0.0
        if self.last_control is not None:
            smooth_penalty = float(np.mean(np.abs(control - np.asarray(self.last_control, dtype=np.float32))))

        mitigation_bonus = 0.0
        if float(state.rh_air) >= 88.0:
            mitigation_bonus += 0.8 * max(control[3] - 0.25, 0.0)
            mitigation_bonus += 0.25 if control[4] <= 0.01 and control[1] <= 0.01 else 0.0
        if float(state.temp_air) < temp_floor + 1.5:
            mitigation_bonus += 0.55 * control[0] + 0.20 * control[2]
        if float(state.temp_air) > 28.0:
            mitigation_bonus += 0.55 * control[3] + 0.25 * control[5]
        if float(state.rh_air) < float(getattr(self.config, "dry_rh_on", 55.0)) or calculate_vpd_kpa(
            float(state.temp_air), float(state.rh_air)
        ) > float(getattr(self.config, "dry_vpd_on", 1.20)):
            if control[3] <= float(getattr(self.config, "dry_warm_vent_cap", 0.18)):
                mitigation_bonus += 0.45
            if control[1] <= 0.01 and control[4] <= 0.01:
                mitigation_bonus += 0.30
            if float(state.temp_air) < 18.0 and control[2] >= 0.65:
                mitigation_bonus += 0.18

        score = (
            self.config.fallback_temp_penalty_weight * temp_penalty
            + self.config.fallback_rh_penalty_weight * rh_penalty
            + 1.20 * dry_penalty
            + 1.00 * vpd_penalty
            + self.config.fallback_cost_penalty_weight * energy_penalty
            + self.config.fallback_smooth_penalty_weight * smooth_penalty
            + self.config.fallback_conflict_penalty_weight * conflict_penalty
            + 0.65 * dew_penalty
            - mitigation_bonus
        )
        details = {
            "score": float(score),
            "temp_penalty": float(temp_penalty),
            "rh_penalty": float(rh_penalty),
            "dry_penalty": float(dry_penalty),
            "vpd_penalty": float(vpd_penalty),
            "dew_penalty": float(dew_penalty),
            "energy_penalty": float(energy_penalty),
            "conflict_penalty": float(conflict_penalty),
            "smooth_penalty": float(smooth_penalty),
            "mitigation_bonus": float(mitigation_bonus),
            **response,
        }
        return float(score), details

    @staticmethod
    def _candidate_energy_proxy(control: np.ndarray) -> float:
        control = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        return float(0.70 * control[0] + 0.22 * control[1] + 1.05 * control[4] + 0.05 * control[3])

    def _mc_sero_mode(self) -> str:
        mode = str(getattr(self.config, "mc_sero_mode", "off") or "off").strip().lower()
        return mode if mode in {"off", "shadow"} else "off"

    def _mc_sero_dry_vent_cap(self, temp_air: float) -> float:
        if temp_air >= 28.0:
            return float(getattr(self.config, "dry_hot_vent_cap", 0.35))
        if temp_air >= 24.0:
            return float(getattr(self.config, "dry_warm_vent_cap", 0.18))
        return float(getattr(self.config, "dry_vent_cap", 0.12))

    @staticmethod
    def _state_control_vector(state) -> np.ndarray:
        return np.asarray(
            [
                float(getattr(state, "u_boil", 0.0)),
                float(getattr(state, "u_co2", 0.0)),
                float(getattr(state, "u_th_scr", 0.0)),
                float(getattr(state, "u_vent", 0.0)),
                float(getattr(state, "u_lamp", 0.0)),
                float(getattr(state, "u_bl_scr", 0.0)),
            ],
            dtype=np.float32,
        )

    def _build_mc_sero_mechanism_candidates(
        self,
        state,
        baseline_control: np.ndarray,
    ) -> List[Tuple[str, np.ndarray]]:
        """Build mechanism candidates for shadow scoring only."""
        temp_air = float(getattr(state, "temp_air", 20.0))
        rh_air = float(getattr(state, "rh_air", 70.0))
        rh_out = float(getattr(state, "rh_out", rh_air))
        temp_out = float(getattr(state, "temp_out", temp_air))
        hour = float(getattr(state, "hour_of_day", 12.0))
        rad = float(getattr(state, "glob_rad", 0.0))
        vpd_air = float(calculate_vpd_kpa(temp_air, rh_air))
        dry_risk = rh_air < float(getattr(self.config, "dry_rh_on", 55.0)) or vpd_air > float(
            getattr(self.config, "dry_vpd_on", 1.20)
        )
        is_night = bool(hour >= 18.0 or hour < 6.0 or rad < 10.0)
        outdoor_dry_potential = bool(rh_out <= rh_air - 3.0 or temp_out <= temp_air - 2.0)
        current = np.clip(self._state_control_vector(state), 0.0, 1.0)
        baseline = np.clip(np.asarray(baseline_control, dtype=np.float32), 0.0, 1.0)
        candidates: List[Tuple[str, np.ndarray]] = []

        def add(name: str, control: np.ndarray) -> None:
            candidates.append((name, np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)))

        economy = np.minimum(current, baseline).astype(np.float32)
        economy[0] = min(float(economy[0]), 0.08 if temp_air >= 16.0 else 0.16)
        economy[1] = min(float(economy[1]), 0.02)
        economy[4] = 0.0
        if is_night and temp_air < 18.0:
            economy[2] = max(float(economy[2]), 0.65)
        if dry_risk:
            economy[3] = min(float(economy[3]), self._mc_sero_dry_vent_cap(temp_air))
        add("economy_hold", economy)

        economic_dehum = economy.copy()
        economic_vent = 0.26 if outdoor_dry_potential else 0.16
        if rh_air >= 86.0:
            economic_vent += 0.08
        economic_dehum[3] = max(float(economic_dehum[3]), economic_vent)
        if dry_risk:
            economic_dehum[3] = min(float(economic_dehum[3]), self._mc_sero_dry_vent_cap(temp_air))
        economic_dehum[1] = 0.0
        economic_dehum[4] = 0.0
        economic_dehum[2] = min(float(economic_dehum[2]), 0.50 if is_night else 0.42)
        if temp_air < 15.5:
            economic_dehum[0] = max(float(economic_dehum[0]), 0.10)
        add("economic_dehumidify", economic_dehum)

        safe_dehum = economy.copy()
        safe_dehum[0] = max(float(safe_dehum[0]), 0.10 if temp_air < 18.0 else 0.02)
        safe_dehum[1] = 0.0
        safe_dehum[2] = min(float(safe_dehum[2]), 0.35)
        safe_dehum[3] = max(float(safe_dehum[3]), 0.46 if rh_air < 92.0 else 0.54)
        safe_dehum[4] = 0.0
        add("safe_dehumidify", safe_dehum)

        emergency_dehum = economy.copy()
        emergency_dehum[0] = max(float(emergency_dehum[0]), 0.28 if temp_air < 16.0 else 0.14)
        emergency_dehum[1] = 0.0
        emergency_dehum[2] = min(float(emergency_dehum[2]), 0.25)
        emergency_dehum[3] = max(float(emergency_dehum[3]), 0.66)
        emergency_dehum[4] = 0.0
        add("emergency_dehumidify", emergency_dehum)

        dry_recovery = economy.copy()
        dry_recovery[0] = min(float(dry_recovery[0]), 0.08 if temp_air >= 18.0 else 0.16)
        dry_recovery[1] = 0.0
        dry_recovery[3] = min(float(dry_recovery[3]), self._mc_sero_dry_vent_cap(temp_air))
        dry_recovery[4] = 0.0
        if is_night:
            dry_recovery[2] = max(float(dry_recovery[2]), 0.70)
        add("dry_recovery", dry_recovery)

        heat_buffer = economy.copy()
        heat_buffer[0] = max(float(heat_buffer[0]), 0.24 if temp_air < 16.0 else 0.12)
        heat_buffer[1] = 0.0
        heat_buffer[2] = max(float(heat_buffer[2]), 0.82 if is_night else 0.55)
        heat_buffer[3] = min(float(heat_buffer[3]), 0.10)
        heat_buffer[4] = 0.0
        add("heat_buffer", heat_buffer)

        return candidates

    @staticmethod
    def _mc_sero_proxy_step(
        temp: float,
        rh: float,
        co2: float,
        rad: float,
        temp_out: float,
        rh_out: float,
        hour: float,
        control: np.ndarray,
    ) -> Dict[str, float]:
        control = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        heat, co2_u, screen, vent, lamp, shade = [float(x) for x in control[:6]]
        is_day = 6.0 <= float(hour) <= 18.0
        total_rad = float(rad) + 100.0 * lamp

        vent_cooling = vent * max(temp - temp_out, 0.0) * 0.18
        shade_cooling = shade * min(max(rad, 0.0) / 500.0, 1.0) * 0.18
        screen_retention = screen * (0.10 if not is_day else 0.03)
        temp_next = temp + 0.62 * heat + 0.16 * lamp + screen_retention - vent_cooling - shade_cooling

        outdoor_drier = 1.0 if rh_out <= rh else -0.35
        vent_rh_delta = -5.0 * vent * outdoor_drier
        heat_rh_delta = -1.5 * heat
        lamp_rh_delta = -0.4 * lamp
        screen_rh_delta = 0.45 * screen if not is_day else 0.10 * screen
        rh_next = float(np.clip(rh + vent_rh_delta + heat_rh_delta + lamp_rh_delta + screen_rh_delta, 35.0, 100.0))

        co2_assimilation = 20.0 if is_day and total_rad > 120.0 else 4.0
        co2_leak = 190.0 * vent * max((co2 - 410.0) / 500.0, 0.0)
        co2_next = max(320.0, co2 + 170.0 * co2_u - co2_leak - co2_assimilation)
        vpd_next = float(calculate_vpd_kpa(float(temp_next), float(rh_next)))

        return {
            "temp_next": float(temp_next),
            "rh_next": float(rh_next),
            "co2_next": float(co2_next),
            "vpd_next": float(vpd_next),
            "total_rad": float(total_rad),
        }

    def _score_mc_sero_candidate(self, state, control: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Score a candidate over a deterministic short horizon for shadow diagnostics."""
        cfg = self.config
        control = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        horizon = max(1, int(getattr(cfg, "mc_sero_horizon_steps", 6)))
        temp = float(getattr(state, "temp_air", 20.0))
        rh = float(getattr(state, "rh_air", 70.0))
        co2 = float(getattr(state, "co2_air", 430.0))
        rad = float(getattr(state, "glob_rad", 0.0))
        temp_out = float(getattr(state, "temp_out", temp))
        rh_out = float(getattr(state, "rh_out", rh))
        hour = float(getattr(state, "hour_of_day", 12.0))
        fruit_weight = float(getattr(state, "fruit_weight", 0.0))
        temp_floor = 10.0 if fruit_weight < 1.0 else 12.0
        dew_margin_base = min(
            float(getattr(state, "dew_margin_air", 3.0)),
            float(getattr(state, "canopy_dew_margin", 3.0)),
        )
        vpd_base = float(calculate_vpd_kpa(temp, rh))
        terms: Dict[str, float] = {
            "temp_low": 0.0,
            "temp_high": 0.0,
            "rh_high": 0.0,
            "rh_low": 0.0,
            "vpd_high": 0.0,
            "vpd_low": 0.0,
            "dew": 0.0,
            "dry_vent": 0.0,
            "cold_vent": 0.0,
            "energy": 0.0,
            "conflict": 0.0,
        }

        heat, co2_u, _screen, vent, lamp, _shade = [float(x) for x in control[:6]]
        dry_cap = self._mc_sero_dry_vent_cap(temp)
        initial_dry_risk = rh < float(getattr(self.config, "dry_rh_on", 55.0)) or vpd_base > float(
            getattr(self.config, "dry_vpd_on", 1.20)
        )
        initial_extreme_dew_risk = rh >= 94.0 or dew_margin_base < 0.40
        conflict = 0.0
        conflict += max(heat - 0.12, 0.0) * max(vent - 0.18, 0.0) * (4.0 if rh < 88.0 else 1.4)
        conflict += co2_u * max(vent - 0.16, 0.0) * 4.0
        conflict += lamp * max(vent - 0.25, 0.0) * 3.0
        if rh >= 86.0:
            conflict += lamp * 1.2 + co2_u * 0.7
        if initial_dry_risk:
            terms["dry_vent"] = max(vent - dry_cap, 0.0) ** 2
        if temp < 12.0 and vent > 0.18 and not initial_extreme_dew_risk:
            terms["cold_vent"] = (vent - 0.18) ** 2
        terms["energy"] = float(self._candidate_energy_proxy(control))
        terms["conflict"] = float(conflict)

        for idx in range(horizon):
            response = self._mc_sero_proxy_step(temp, rh, co2, rad, temp_out, rh_out, hour + idx / 12.0, control)
            temp = response["temp_next"]
            rh = response["rh_next"]
            co2 = response["co2_next"]
            vpd = response["vpd_next"]
            dew_margin_proxy = max(
                0.0,
                dew_margin_base + 0.90 * (vpd - vpd_base) - 0.025 * max(rh - 85.0, 0.0),
            )
            terms["temp_low"] += max(temp_floor - temp, 0.0) ** 2 + 0.25 * max(15.0 - temp, 0.0) ** 2
            terms["temp_high"] += 0.35 * max(temp - 30.0, 0.0) ** 2
            terms["rh_high"] += (
                0.05 * max(rh - 84.0, 0.0) ** 2
                + 0.18 * max(rh - 90.0, 0.0) ** 2
                + 0.55 * max(rh - 95.0, 0.0) ** 2
            )
            terms["rh_low"] += (
                0.10 * max(55.0 - rh, 0.0) ** 2
                + 0.35 * max(50.0 - rh, 0.0) ** 2
                + 1.20 * max(45.0 - rh, 0.0) ** 2
            )
            terms["vpd_high"] += 0.85 * max(vpd - 1.20, 0.0) ** 2 + 1.50 * max(vpd - 1.60, 0.0) ** 2
            terms["vpd_low"] += 0.55 * max(0.25 - vpd, 0.0) ** 2
            terms["dew"] += max(1.20 - dew_margin_proxy, 0.0) * (1.0 + max(rh - 85.0, 0.0) / 10.0)

        for key in ("temp_low", "temp_high", "rh_high", "rh_low", "vpd_high", "vpd_low", "dew"):
            terms[key] = float(terms[key] / horizon)
        terms["temp_terminal"] = float(temp)
        terms["rh_terminal"] = float(rh)
        terms["vpd_terminal"] = float(calculate_vpd_kpa(temp, rh))

        score = (
            float(getattr(cfg, "mc_sero_temp_penalty_weight", 2.00)) * (terms["temp_low"] + terms["temp_high"])
            + float(getattr(cfg, "mc_sero_rh_penalty_weight", 1.70)) * terms["rh_high"]
            + float(getattr(cfg, "mc_sero_dry_penalty_weight", 1.20)) * terms["rh_low"]
            + float(getattr(cfg, "mc_sero_vpd_penalty_weight", 1.30)) * (terms["vpd_high"] + terms["vpd_low"])
            + float(getattr(cfg, "mc_sero_dew_penalty_weight", 1.40)) * terms["dew"]
            + 160.0 * terms["dry_vent"]
            + 45.0 * terms["cold_vent"]
            + float(getattr(cfg, "mc_sero_energy_penalty_weight", 0.45)) * terms["energy"]
            + float(getattr(cfg, "mc_sero_conflict_penalty_weight", 0.85)) * terms["conflict"]
        )
        terms["score"] = float(score)
        return float(score), terms

    def _mc_sero_candidate_risk_reason(self, state, control: np.ndarray) -> str:
        control = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        temp_air = float(getattr(state, "temp_air", 20.0))
        rh_air = float(getattr(state, "rh_air", 70.0))
        vpd = float(calculate_vpd_kpa(temp_air, rh_air))
        dew_margin = min(
            float(getattr(state, "dew_margin_air", 3.0)),
            float(getattr(state, "canopy_dew_margin", 3.0)),
        )
        dry_risk = rh_air < float(getattr(self.config, "dry_rh_on", 55.0)) or vpd > float(
            getattr(self.config, "dry_vpd_on", 1.20)
        )
        if dry_risk and float(control[3]) > self._mc_sero_dry_vent_cap(temp_air) + 1e-4:
            return "dry_vent_risk"
        extreme_dew_risk = rh_air >= 94.0 or dew_margin < 0.40
        if temp_air < 12.0 and float(control[3]) > 0.18 and not extreme_dew_risk:
            return "cold_vent_risk"
        return ""

    def _evaluate_mc_sero_shadow(
        self,
        state,
        rollout_candidates: List[Tuple[str, np.ndarray]],
        baseline_source: str,
        baseline_control: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate MC-SERO candidates without changing the selected control."""
        mode = self._mc_sero_mode()
        if mode != "shadow":
            return {"enabled": False, "mode": mode}

        candidate_items: List[Tuple[str, np.ndarray]] = []
        seen = set()

        def add(name: str, control: np.ndarray) -> None:
            unique = str(name)
            if unique in seen:
                return
            seen.add(unique)
            candidate_items.append((unique, np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)))

        for name, control in rollout_candidates:
            add(str(name), control)
        for name, control in self._build_mc_sero_mechanism_candidates(state, baseline_control):
            add(str(name), control)

        baseline_control = np.clip(np.asarray(baseline_control, dtype=np.float32), 0.0, 1.0)
        baseline_score, baseline_terms = self._score_mc_sero_candidate(state, baseline_control)
        if not any(name == str(baseline_source) for name, _ in candidate_items):
            add(str(baseline_source or "baseline_actual"), baseline_control)

        scored: List[Dict[str, Any]] = []
        for name, control in candidate_items:
            score, terms = self._score_mc_sero_candidate(state, control)
            scored.append(
                {
                    "name": name,
                    "score": float(score),
                    "control": [float(x) for x in control[:6]],
                    "score_terms": {key: float(value) for key, value in terms.items()},
                }
            )
        scored.sort(key=lambda item: float(item["score"]))
        if not scored:
            return {"enabled": True, "mode": "shadow", "available": False, "reject_reason": "no_candidates"}

        best = scored[0]
        best_control = np.asarray(best["control"], dtype=np.float32)
        margin = float(baseline_score - float(best["score"]))
        reject_reason = self._mc_sero_candidate_risk_reason(state, best_control)
        if not reject_reason:
            if str(best["name"]) == str(baseline_source):
                reject_reason = "baseline_best"
            elif margin < float(getattr(self.config, "mc_sero_min_margin", 0.10)):
                reject_reason = "insufficient_margin"
        would_select = bool(
            str(best["name"]) != str(baseline_source)
            and margin >= float(getattr(self.config, "mc_sero_min_margin", 0.10))
            and not reject_reason
        )
        if would_select:
            reject_reason = ""

        top_k = max(1, int(getattr(self.config, "mc_sero_top_k", 6)))
        return {
            "enabled": True,
            "mode": "shadow",
            "available": True,
            "would_select": bool(would_select),
            "best_candidate": str(best["name"]),
            "best_control": [float(x) for x in best_control[:6]],
            "best_score": float(best["score"]),
            "baseline_source": str(baseline_source),
            "baseline_score": float(baseline_score),
            "margin": float(margin),
            "reject_reason": str(reject_reason),
            "score_terms": dict(best["score_terms"]),
            "baseline_terms": {key: float(value) for key, value in baseline_terms.items()},
            "candidate_count": int(len(scored)),
            "horizon_steps": int(max(1, int(getattr(self.config, "mc_sero_horizon_steps", 6)))),
            "top_candidates": [
                {
                    "name": str(item["name"]),
                    "score": float(item["score"]),
                    "score_terms": dict(item["score_terms"]),
                }
                for item in scored[:top_k]
            ],
        }

    def _is_humidity_memory_night_window(self, state) -> bool:
        hour = float(getattr(state, "hour_of_day", 12.0))
        rad = float(getattr(state, "glob_rad", 0.0))
        return bool(hour >= 18.0 or hour < 6.0 or rad < 10.0)

    def _evaluate_humidity_memory_horizon(
        self,
        state,
        control: np.ndarray,
        raw_score: float,
        details: Dict[str, float],
        best_non_hem: Optional[Tuple[float, np.ndarray, Dict[str, float]]],
        plan: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Apply a HEM-only short-horizon safety score without changing baseline scoring."""
        cfg = self.config
        control = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        is_night = self._is_humidity_memory_night_window(state)
        timestep = int(float(getattr(state, "timestep", 0.0)))
        rh_air = float(getattr(state, "rh_air", 70.0))
        temp_air = float(getattr(state, "temp_air", 20.0))
        temp_out = float(getattr(state, "temp_out", temp_air))
        forecast_risk = float(getattr(state, "forecast_humidity_risk", 0.0))
        dew_margin = min(
            float(getattr(state, "dew_margin_air", 3.0)),
            float(getattr(state, "canopy_dew_margin", 3.0)),
        )
        target_temp = self._get_plan_target(plan, "target_temp", state) if plan is not None else None
        if target_temp is None:
            target_temp = 18.0 if not is_night else 16.5

        best_score = float("inf")
        best_control = None
        best_details: Dict[str, float] = {}
        if best_non_hem is not None:
            best_score, best_control, best_details = best_non_hem
            best_control = np.clip(np.asarray(best_control, dtype=np.float32), 0.0, 1.0)

        direct_saving = 0.0
        if best_control is not None:
            direct_saving = max(self._candidate_energy_proxy(best_control) - self._candidate_energy_proxy(control), 0.0)

        hem_temp_next = float(details.get("temp_next", temp_air))
        hem_rh_next = float(details.get("rh_next", rh_air))
        non_hem_temp_next = float(best_details.get("temp_next", hem_temp_next))
        non_hem_rh_next = float(best_details.get("rh_next", hem_rh_next))

        temp_drop_vs_non_hem = max(non_hem_temp_next - hem_temp_next, 0.0)
        target_temp_gap = max(float(target_temp) - hem_temp_next, 0.0)
        temp_buffer_deficit = max(float(getattr(cfg, "humidity_memory_night_min_temp", 17.5)) - min(temp_air, hem_temp_next), 0.0)
        outdoor_heat_loss = 0.0
        screen_heat_loss = 0.0
        if best_control is not None:
            outdoor_heat_loss = max(float(control[3] - best_control[3]), 0.0) * max(temp_air - temp_out, 0.0) / 10.0
            screen_heat_loss = max(float(best_control[2] - control[2]), 0.0)

        night_multiplier = 1.65 if is_night else 1.0
        recovery_heat_cost = night_multiplier * (
            0.018 * target_temp_gap
            + 0.018 * temp_drop_vs_non_hem
            + 0.020 * temp_buffer_deficit
            + 0.030 * outdoor_heat_loss
            + 0.018 * screen_heat_loss
        )
        rh_debt_penalty = (
            0.010 * max(hem_rh_next - 84.0, 0.0)
            + 0.018 * max(hem_rh_next - 90.0, 0.0)
            + 0.008 * max(hem_rh_next - non_hem_rh_next, 0.0)
            + 0.020 * max(float(getattr(cfg, "humidity_memory_min_dew_margin", 1.35)) - dew_margin, 0.0)
            + 0.020 * max(forecast_risk - 0.55, 0.0)
        )
        horizon_penalty = recovery_heat_cost + rh_debt_penalty
        adjusted_score = (
            float(raw_score)
            + float(getattr(cfg, "humidity_memory_horizon_penalty_weight", 1.0)) * horizon_penalty
            - float(getattr(cfg, "humidity_memory_direct_saving_weight", 0.20)) * direct_saving
        )

        reject_reason = ""
        last_step = int(getattr(self, "last_humidity_memory_selected_step", -10**9))
        gap_steps = int(getattr(cfg, "humidity_memory_night_gap_steps", 6))
        if is_night and timestep - last_step < gap_steps:
            reject_reason = "night_gap"
        elif is_night and rh_air >= 78.0 and temp_air < float(getattr(cfg, "humidity_memory_night_min_temp", 17.5)):
            reject_reason = "night_temp_buffer"
        elif rh_air >= 78.0 and dew_margin < float(getattr(cfg, "humidity_memory_min_dew_margin", 1.35)):
            reject_reason = "low_dew_margin"
        elif rh_air >= 78.0 and forecast_risk > float(getattr(cfg, "humidity_memory_max_forecast_risk", 0.72)):
            reject_reason = "forecast_humidity_risk"
        elif recovery_heat_cost > float(getattr(cfg, "humidity_memory_max_recovery_heat_cost", 0.055)) and recovery_heat_cost > direct_saving:
            reject_reason = "recovery_cost_exceeds_saving"
        elif rh_debt_penalty > float(getattr(cfg, "humidity_memory_max_rh_debt_penalty", 0.060)):
            reject_reason = "rh_debt_penalty"
        elif np.isfinite(best_score) and adjusted_score > best_score - float(getattr(cfg, "humidity_memory_score_margin", 0.015)):
            reject_reason = "insufficient_score_margin"

        info: Dict[str, Any] = {
            "enabled": True,
            "rejected": bool(reject_reason),
            "reject_reason": reject_reason,
            "raw_score": float(raw_score),
            "adjusted_score": float(adjusted_score),
            "horizon_penalty": float(horizon_penalty),
            "direct_saving": float(direct_saving),
            "recovery_heat_cost": float(recovery_heat_cost),
            "rh_debt_penalty": float(rh_debt_penalty),
            "best_non_hem_score": float(best_score) if np.isfinite(best_score) else None,
            "temp_next": float(hem_temp_next),
            "rh_next": float(hem_rh_next),
            "non_hem_temp_next": float(non_hem_temp_next),
            "non_hem_rh_next": float(non_hem_rh_next),
            "is_night": bool(is_night),
            "dew_margin": float(dew_margin),
            "forecast_humidity_risk": float(forecast_risk),
            "last_selected_step": int(last_step),
        }
        if reject_reason:
            return float("inf"), info
        return float(adjusted_score), info

    def _build_fallback_candidates(
        self,
        state,
        analysis: Optional[Dict[str, Any]] = None,
    ) -> List[FallbackCandidate]:
        """生成 fallback 候选池：规则、近期动作、自适应动作和紧急专家动作。"""
        env = self.interface.env
        env_control = np.asarray(getattr(env, "u", np.zeros(getattr(env, "nu", 6))), dtype=np.float32)
        rule_control = self._predict_rule_control()
        candidates: List[FallbackCandidate] = [
            FallbackCandidate("hold_current", np.clip(env_control, 0.0, 1.0), rationale="保持当前执行器状态"),
        ]

        if rule_control is not None:
            candidates.append(FallbackCandidate("rule_controller", np.clip(rule_control, 0.0, 1.0), rationale="基准规则控制器输出"))
            candidates.append(
                FallbackCandidate(
                    "adaptive_rule",
                    self._generate_state_adaptive_control(state, rule_control),
                    rationale="规则输出叠加状态误差修正",
                )
            )
        else:
            candidates.append(
                FallbackCandidate(
                    "adaptive_current",
                    self._generate_state_adaptive_control(state, env_control),
                    rationale="当前控制叠加状态误差修正",
                )
            )

        if self.config.fallback_use_last_control and self.last_control is not None:
            recent = np.asarray(self.last_control, dtype=np.float32)
            candidates.append(FallbackCandidate("recent_anchor", np.clip(recent, 0.0, 1.0), rationale="复用上一规划锚点"))
            blend = float(np.clip(self.config.fallback_recent_blend, 0.0, 1.0))
            base = rule_control if rule_control is not None else env_control
            adaptive = self._generate_state_adaptive_control(state, base)
            candidates.append(
                FallbackCandidate(
                    "adaptive_recent_blend",
                    np.clip((1.0 - blend) * adaptive + blend * recent, 0.0, 1.0),
                    rationale="自适应动作与上一锚点平滑融合",
                )
            )

        rh = float(state.rh_air)
        temp = float(state.temp_air)
        rad = float(state.glob_rad)
        hour = float(state.hour_of_day)
        vpd = calculate_vpd_kpa(temp, rh)
        is_day = 6 <= hour <= 18

        if rh >= 86.0:
            dehum = np.clip(env_control.copy(), 0.0, 1.0)
            if rh >= 94.0:
                dehum[3] = max(dehum[3], 0.70)
                dehum[2] = min(dehum[2], 0.32)
            elif rh >= 90.0:
                dehum[3] = max(dehum[3], 0.58)
                dehum[2] = min(dehum[2], 0.45)
            else:
                dehum[3] = max(dehum[3], 0.32)
                dehum[2] = min(dehum[2], 0.70)
            dehum[1] = 0.0
            dehum[4] = 0.0
            if temp < 21.5:
                heat_floor = 0.24 if rh >= 94.0 else 0.16
                if temp < 15.0:
                    heat_floor = max(heat_floor, 0.28 if temp >= 10.0 else 0.35)
                dehum[0] = max(dehum[0], heat_floor)
            candidates.append(FallbackCandidate("emergency_dehumidify", dehum, rationale="高湿/结露风险优先排湿"))

        if temp < 12.5:
            survival = np.clip(env_control.copy(), 0.0, 1.0)
            survival[0] = max(survival[0], 0.85)
            survival[2] = max(survival[2], 0.90)
            survival[3] = min(survival[3], 0.12 if rh >= 90.0 else 0.05)
            survival[1] = 0.0
            survival[4] = 0.0
            candidates.append(FallbackCandidate("cold_survival", survival, rationale="低温生存保护"))

        if temp > 28.0:
            cooling = np.clip(env_control.copy(), 0.0, 1.0)
            cooling[0] = 0.0
            cooling[1] = 0.0
            cooling[3] = max(cooling[3], 0.65)
            if rad > 250.0:
                cooling[5] = max(cooling[5], 0.75)
            cooling[4] = 0.0
            candidates.append(FallbackCandidate("heat_relief", cooling, rationale="高温降温保护"))

        if rh <= float(self.config.dry_rh_on) or vpd >= float(self.config.dry_vpd_on):
            dry = np.clip(env_control.copy(), 0.0, 1.0)
            dry[1] = 0.0
            dry[4] = 0.0
            if temp >= 28.0:
                vent_cap = float(self.config.dry_hot_vent_cap)
            elif temp >= 24.0:
                vent_cap = float(self.config.dry_warm_vent_cap)
            else:
                vent_cap = float(self.config.dry_vent_cap)
            dry[3] = min(dry[3], vent_cap)
            if temp < 12.0:
                dry[0] = max(dry[0], 0.85)
                dry[2] = max(dry[2], 0.90)
                dry[3] = min(dry[3], float(self.config.cold_dehumidify_vent_cap))
            else:
                dry[0] = 0.0
                if temp < 18.0:
                    dry[2] = max(dry[2], 0.70)
            if temp > 24.0 or rad > 250.0:
                dry[5] = max(dry[5], 0.50)
            candidates.append(FallbackCandidate("dry_recovery", dry, rationale="dry-side recovery"))

        economy = np.clip(env_control.copy(), 0.0, 1.0)
        economy[1] = 0.0 if (not is_day or rad < self.config.fallback_co2_min_rad or economy[3] > 0.16) else economy[1]
        economy[4] = 0.0 if (rh >= 84.0 or rad > 120.0 or economy[3] >= 0.25) else min(economy[4], 0.20)
        candidates.append(FallbackCandidate("economy_hold", economy, rationale="经济保守控制"))

        scored: List[FallbackCandidate] = []
        for candidate in candidates:
            score, details = self._score_fallback_candidate(state, candidate.control, analysis=analysis)
            candidate.score = score
            candidate.details = details
            scored.append(candidate)
        return sorted(scored, key=lambda item: item.score)

    def _select_fallback_control(self, state=None, analysis: Optional[Dict[str, Any]] = None) -> np.ndarray:
        env = self.interface.env
        env_control = np.asarray(getattr(env, "u", np.zeros(getattr(env, "nu", 6))), dtype=np.float32)
        strategy = str(getattr(self.config, "fallback_strategy", "adaptive") or "adaptive").lower()

        rule_control = self._predict_rule_control()
        if strategy == "rule" and rule_control is not None:
            self.last_fallback_selection = {"source": "rule_controller", "score": None, "candidates": []}
            return np.asarray(rule_control, dtype=np.float32).copy()

        if strategy == "recent" and self.config.fallback_use_last_control and self.last_control is not None:
            self.last_fallback_selection = {"source": "recent_anchor", "score": None, "candidates": []}
            return np.asarray(self.last_control, dtype=np.float32).copy()

        if state is not None and bool(getattr(self.config, "fallback_candidate_scoring", True)):
            candidates = self._build_fallback_candidates(state, analysis=analysis)
            if candidates:
                best = candidates[0]
                self.last_fallback_selection = {
                    "source": best.name,
                    "score": float(best.score),
                    "rationale": best.rationale,
                    "details": best.details,
                    "candidates": [
                        {
                            "name": item.name,
                            "score": float(item.score),
                            "rationale": item.rationale,
                            "details": item.details,
                        }
                        for item in candidates[:5]
                    ],
                }
                return np.clip(best.control, 0.0, 1.0).astype(np.float32)

        if state is not None:
            base_control = rule_control if rule_control is not None else env_control
            adaptive_control = self._generate_state_adaptive_control(state, base_control)
            if self.config.fallback_use_last_control and self.last_control is not None:
                blend = float(np.clip(self.config.fallback_recent_blend, 0.0, 1.0))
                adaptive_control = (1.0 - blend) * adaptive_control + blend * np.asarray(self.last_control, dtype=np.float32)
            self.last_fallback_selection = {"source": "adaptive_legacy", "score": None, "candidates": []}
            return np.clip(adaptive_control, 0.0, 1.0).astype(np.float32)

        if rule_control is not None:
            self.last_fallback_selection = {"source": "rule_controller", "score": None, "candidates": []}
            return np.asarray(rule_control, dtype=np.float32).copy()

        if self.config.fallback_use_last_control and self.last_control is not None:
            self.last_fallback_selection = {"source": "recent_anchor", "score": None, "candidates": []}
            return np.asarray(self.last_control, dtype=np.float32).copy()

        self.last_fallback_selection = {"source": "hold_current", "score": None, "candidates": []}
        return env_control.copy()

    def _buffer_control(self, control: np.ndarray) -> np.ndarray:
        """缓存本步候选控制量，等待统一护栏修正后再执行。"""
        self.pending_control = np.asarray(control, dtype=np.float32).copy()
        return self.pending_control.copy()

    def _apply_humidity_memory_final_shape(self, state, control: np.ndarray) -> np.ndarray:
        final = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        if not bool(getattr(self.config, "humidity_memory_post_guardrail_shape", True)):
            return final
        selection = getattr(self, "last_rollout_selection", {}) or {}
        prediction = getattr(self, "last_humidity_memory_prediction", {}) or {}
        source = str(selection.get("source", ""))
        if not source.startswith("humidity_memory") or not bool(prediction.get("accepted", False)):
            return final

        temp_air = float(getattr(state, "temp_air", 20.0))
        rh_air = float(getattr(state, "rh_air", 70.0))
        vpd_now = float(calculate_vpd_kpa(temp_air, rh_air))
        dew_margin = min(
            float(getattr(state, "dew_margin_air", 3.0)),
            float(getattr(state, "canopy_dew_margin", 3.0)),
        )
        wind_speed = float(getattr(state, "wind_speed", 0.0))
        hour_of_day = float(getattr(state, "hour_of_day", 12.0))
        glob_rad = float(getattr(state, "glob_rad", 0.0))
        is_dark = hour_of_day < 6.0 or hour_of_day > 18.0 or glob_rad < 5.0
        safe_economic_band = (
            temp_air >= 16.0
            and 78.0 <= rh_air <= 88.5
            and vpd_now >= 0.25
            and dew_margin >= 1.0
            and wind_speed <= 6.0
        )
        shape_info = {
            "applied": False,
            "source": source,
            "temp_air": temp_air,
            "rh_air": rh_air,
            "vpd_kpa": vpd_now,
            "dew_margin": dew_margin,
            "wind_speed": wind_speed,
            "is_dark": is_dark,
        }
        if not safe_economic_band:
            shape_info["reason"] = "outside_safe_economic_band"
            selection["humidity_memory_post_guardrail_shape"] = shape_info
            return final
        if is_dark and temp_air < float(getattr(self.config, "humidity_memory_night_min_temp", 17.5)):
            shape_info["reason"] = "night_temp_buffer"
            selection["humidity_memory_post_guardrail_shape"] = shape_info
            return final
        if dew_margin < float(getattr(self.config, "humidity_memory_min_dew_margin", 1.35)):
            shape_info["reason"] = "low_dew_margin"
            selection["humidity_memory_post_guardrail_shape"] = shape_info
            return final
        if float(getattr(state, "forecast_humidity_risk", 0.0)) > float(
            getattr(self.config, "humidity_memory_max_forecast_risk", 0.72)
        ):
            shape_info["reason"] = "forecast_humidity_risk"
            selection["humidity_memory_post_guardrail_shape"] = shape_info
            return final

        memory_cfg = getattr(getattr(self, "humidity_memory", None), "config", HumidityExperienceConfig())
        before = final.copy()
        final[0] = min(float(final[0]), float(memory_cfg.free_air_exchange_heat_cap))
        final[1] = min(float(final[1]), float(memory_cfg.free_air_exchange_co2_cap))
        screen_cap = float(memory_cfg.free_air_exchange_screen_cap)
        if is_dark and temp_air < 18.0:
            screen_cap = max(screen_cap, float(getattr(self.config, "humidity_memory_night_screen_floor", 0.45)))
        final[2] = min(float(final[2]), screen_cap)
        final[3] = max(float(final[3]), float(memory_cfg.free_air_exchange_min_ventilation))
        final[4] = min(float(final[4]), float(memory_cfg.free_air_exchange_lighting_cap))
        if is_dark and temp_air < 18.0:
            final[0] = max(float(final[0]), float(getattr(self.config, "humidity_memory_night_heat_floor", 0.06)))
            final[2] = max(float(final[2]), float(getattr(self.config, "humidity_memory_night_screen_floor", 0.45)))
            final[3] = min(float(final[3]), float(getattr(self.config, "humidity_memory_night_vent_cap", 0.58)))
        shape_info.update(
            {
                "applied": True,
                "before": [float(x) for x in before],
                "after": [float(x) for x in final],
            }
        )
        selection["humidity_memory_post_guardrail_shape"] = shape_info
        return np.clip(final, 0.0, 1.0).astype(np.float32)

    def _flush_buffered_control(self, state) -> np.ndarray:
        """将缓存控制量统一过护栏并清空缓存。"""
        if self.pending_control is None:
            self.pending_control = self._select_fallback_control(state=state)
        final_control = apply_safety_guardrails(state, self.pending_control)
        final_control = self._apply_humidity_memory_final_shape(state, final_control)
        self.pending_control = None
        return np.asarray(final_control, dtype=np.float32)

    def _enforce_setpoint_contract(
        self,
        state,
        target_control: np.ndarray,
        target_temp: Optional[float],
        target_co2: Optional[float],
        target_rh: Optional[float],
    ):
        """确保关键 setpoints 永不缺失，并做范围约束。"""
        is_day = 6 <= float(state.hour_of_day) <= 18
        total_rad = float(state.glob_rad) + float(target_control[4]) * 100.0
        vent_level = float(target_control[3])

        missing_fields: List[str] = []
        corrected_fields: List[str] = []

        if target_temp is None:
            if float(state.temp_air) < 12.5:
                target_temp = 15.0
            elif is_day and total_rad > 180.0:
                target_temp = 18.5
            elif is_day:
                target_temp = 17.0
            else:
                target_temp = 14.0
            missing_fields.append("target_temp")
        temp_before = float(target_temp)
        target_temp = float(np.clip(temp_before, 11.5, 24.0))
        if abs(target_temp - temp_before) > 1e-6:
            corrected_fields.append("target_temp")

        if target_co2 is None:
            if is_day and total_rad > 140.0 and vent_level <= 0.25:
                target_co2 = 620.0
            else:
                target_co2 = 430.0
            missing_fields.append("target_co2")
        co2_before = float(target_co2)
        target_co2 = float(np.clip(co2_before, 380.0, 850.0))
        if abs(target_co2 - co2_before) > 1e-6:
            corrected_fields.append("target_co2")

        if target_rh is None:
            rh_now = float(state.rh_air)
            if rh_now > 88.0:
                target_rh = 72.0
            elif rh_now < 60.0:
                target_rh = 70.0
            else:
                target_rh = 76.0
            missing_fields.append("target_rh")
        rh_before = float(target_rh)
        target_rh = float(np.clip(rh_before, 62.0, 88.0))
        if abs(target_rh - rh_before) > 1e-6:
            corrected_fields.append("target_rh")

        rh_now = float(getattr(state, "rh_air", 0.0))
        temp_now = float(getattr(state, "temp_air", 20.0))
        vpd_now = float(calculate_vpd_kpa(temp_now, rh_now))
        dew_margin = min(
            float(getattr(state, "dew_margin_air", 3.0)),
            float(getattr(state, "canopy_dew_margin", 3.0)),
        )
        risk_cap = 88.0
        if (
            rh_now >= float(self.config.rh_control_limit)
            or (rh_now >= 82.0 and vpd_now < 0.30)
            or (rh_now >= 78.0 and vpd_now < 0.20)
            or dew_margin < 0.8
        ):
            risk_cap = float(self.config.rh_target_extreme_cap)
        elif (
            rh_now >= float(self.config.dehumidify_mild_rh_on)
            or (rh_now >= 82.0 and vpd_now < 0.40)
            or dew_margin < 1.2
        ):
            risk_cap = float(self.config.rh_target_high_cap)
        elif rh_now >= float(self.config.rh_preemptive_threshold):
            risk_cap = float(self.config.rh_target_preemptive_cap)

        if target_rh > risk_cap:
            target_rh = max(62.0, risk_cap)
            corrected_fields.append("target_rh:risk_cap")

        dry_side = rh_now < float(self.config.dry_rh_on) or vpd_now > float(self.config.dry_vpd_on)
        if dry_side:
            dry_floor = float(self.config.dry_target_rh_floor)
            if target_rh < dry_floor:
                target_rh = min(88.0, dry_floor)
                corrected_fields.append("target_rh:dry_recovery_floor")
            if target_temp > float(self.config.dry_temp_target_cap):
                target_temp = float(self.config.dry_temp_target_cap)
                corrected_fields.append("target_temp:dry_recovery_cap")
            if target_co2 > 430.0 and (vent_level > 0.12 or total_rad < float(self.config.fallback_co2_min_rad)):
                target_co2 = 430.0
                corrected_fields.append("target_co2:dry_recovery_cap")

        return target_temp, target_co2, target_rh, missing_fields, corrected_fields

    def _coerce_target_profile(
        self,
        raw_profile: Any,
        base_value: float,
        horizon: int,
        lower: float,
        upper: float,
        field_name: str,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """把 LLM 可选目标轨迹清洗成固定长度列表；缺失时由标量 setpoint 扩展。"""
        source = "contract_constant"
        corrections: List[str] = []
        values: List[float] = []

        if isinstance(raw_profile, str):
            try:
                raw_profile = json.loads(raw_profile)
            except Exception:
                raw_profile = None

        if isinstance(raw_profile, (list, tuple, np.ndarray)):
            for item in list(raw_profile):
                try:
                    values.append(float(item))
                except Exception:
                    corrections.append(f"{field_name}:drop_non_numeric")
            if values:
                source = "llm_profile"

        if not values:
            values = [float(base_value)]
            corrections.append(f"{field_name}:filled_from_scalar")

        if len(values) < horizon:
            values.extend([values[-1]] * (horizon - len(values)))
            corrections.append(f"{field_name}:extended_to_horizon")
        elif len(values) > horizon:
            values = values[:horizon]
            corrections.append(f"{field_name}:trimmed_to_horizon")

        clipped_values: List[float] = []
        for value in values:
            clipped = float(np.clip(value, lower, upper))
            if abs(clipped - value) > 1e-6:
                corrections.append(f"{field_name}:clipped")
            clipped_values.append(clipped)

        return clipped_values, {
            "source": source,
            "corrections": sorted(set(corrections)),
            "horizon": int(horizon),
        }

    def _build_target_profiles(
        self,
        raw_setpoints: Dict[str, Any],
        target_temp: float,
        target_co2: float,
        target_rh: float,
        horizon: int,
    ) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
        horizon = max(1, int(horizon))
        temp_profile, temp_contract = self._coerce_target_profile(
            raw_setpoints.get("target_temp_profile"),
            target_temp,
            horizon,
            11.5,
            24.0,
            "target_temp_profile",
        )
        co2_profile, co2_contract = self._coerce_target_profile(
            raw_setpoints.get("target_co2_profile"),
            target_co2,
            horizon,
            380.0,
            850.0,
            "target_co2_profile",
        )
        rh_profile_upper = max(62.0, min(88.0, float(target_rh) + 2.0))
        rh_profile, rh_contract = self._coerce_target_profile(
            raw_setpoints.get("target_rh_profile"),
            target_rh,
            horizon,
            62.0,
            rh_profile_upper,
            "target_rh_profile",
        )
        profile_contract = {
            "target_temp_profile": temp_contract,
            "target_co2_profile": co2_contract,
            "target_rh_profile": rh_contract,
        }
        return {
            "target_temp": temp_profile,
            "target_co2": co2_profile,
            "target_rh": rh_profile,
        }, profile_contract

    def _replan_with_llm(self, state, analysis: Dict[str, Any], reason: str, planning_interval: Optional[int] = None) -> Dict[str, Any]:
        print(f"[Director] 调用 LLM 进行重规划... reason={reason}, max_iterations={self.config.max_iterations}")

        llm_output = ""
        llm_action_found = False
        llm_duration = 0.0
        llm_attempts = 0
        planning_horizon = max(1, int(planning_interval if planning_interval is not None else self.active_control_interval))
        plan_cache = getattr(self, "plan_cache", None)
        cache_mode = str(getattr(plan_cache, "mode", "off")) if plan_cache is not None else "off"
        cache_key = None
        cache_prompt_hash = ""
        cache_config_hash = stable_hash(config_fingerprint(self.config))
        self.last_plan_cache_event = {
            "enabled": bool(plan_cache is not None),
            "mode": cache_mode,
            "hit": False,
            "status": "disabled" if plan_cache is None else "pending",
            "key_policy": str(getattr(self.config, "plan_cache_key_policy", "prompt")),
        }

        try:
            tools_instance = None
            if hasattr(create_langchain_tools, "instance"):
                tools_instance = create_langchain_tools.instance

            if tools_instance is None:
                raise RuntimeError("无法获取 LLM 工具实例 (tools_instance 为 None)")

            for llm_attempts in range(1, max(1, self.config.max_iterations) + 1):
                tools_instance.reset_buffer()

                # 每次 LLM 重规划前强制调用一次 get_status，确保温度趋势等语义信息可用。
                status_snapshot = tools_instance.get_status()
                status_brief = self._extract_status_brief(status_snapshot)
                # 归零计数，让 LLM 在本轮中仍可再次主动调用一次 get_status。
                tools_instance.status_calls_this_round = 0
                prompt_text = self._build_compact_prompt(
                    state,
                    analysis,
                    reason,
                    status_brief=status_brief,
                    horizon=planning_horizon,
                )
                if llm_attempts > 1:
                    retry_hint = (
                        "\n【纠错重试要求】上一轮没有产出可执行锚点。"
                        "本轮必须且仅调用一次 set_all_controls，"
                        "并给出全部 9 个参数的明确数值："
                        "heating/co2/screen/ventilation/lighting/shading/"
                        "target_temp/target_co2/target_rh。"
                        "不要只输出解释文本。"
                    )
                    prompt_text = f"{prompt_text}{retry_hint}"
                    print(f"[Director] LLM iteration {llm_attempts} 使用纠错重试提示")

                cache_entry = None
                cache_prompt_hash = text_hash(prompt_text)
                if plan_cache is not None:
                    key_policy = str(getattr(self.config, "plan_cache_key_policy", "prompt") or "prompt")
                    if key_policy == "scenario_timestep":
                        cache_state_summary = {"timestep": int(getattr(state, "timestep", 0))}
                        cache_reason = "frozen_replan"
                        cache_horizon = 0
                        cache_prompt_key = "scenario_timestep"
                    else:
                        cache_state_summary = state_summary_from_state(state)
                        cache_reason = reason
                        cache_horizon = planning_horizon
                        cache_prompt_key = cache_prompt_hash
                    cache_key = plan_cache.make_key(
                        env_id=str(getattr(self, "env_id", "")),
                        state_summary=cache_state_summary,
                        reason=cache_reason,
                        planning_horizon=cache_horizon,
                        config_hash=cache_config_hash,
                        prompt_hash=cache_prompt_key,
                        attempt=llm_attempts,
                    )
                    cache_entry = plan_cache.get(cache_key)
                    self.last_plan_cache_event = {
                        "enabled": True,
                        "mode": cache_mode,
                        "key": cache_key,
                        "path": str(getattr(plan_cache, "path", "")),
                        "hit": cache_entry is not None,
                        "status": "hit" if cache_entry is not None else "miss",
                        "attempt": int(llm_attempts),
                        "prompt_hash": cache_prompt_hash,
                        "config_hash": cache_config_hash,
                        "key_policy": key_policy,
                    }
                    if cache_entry is not None and cache_mode in {"record", "replay"}:
                        llm_output = str(cache_entry.get("raw_response", ""))
                        llm_action_found = self._restore_cached_action(tools_instance, cache_entry)
                        llm_duration = 0.0
                        print(f"[Director] LLM plan cache hit: mode={cache_mode}, key={cache_key}")
                        if llm_action_found:
                            break
                        print(f"[Director] Cached iteration {llm_attempts} has no planning anchor, retrying...")
                        continue
                    if cache_entry is None and cache_mode == "replay" and bool(getattr(plan_cache, "strict", False)):
                        raise RuntimeError(f"LLM plan cache miss in strict replay mode: key={cache_key}")

                start_time = time.time()
                config = {"max_execution_time": self.config.max_execution_time or 60.0}
                result_state = self.agent_graph.invoke(
                    {"messages": [HumanMessage(content=prompt_text)]},
                    config=config,
                )
                llm_duration = time.time() - start_time

                messages = result_state.get("messages", [])
                if messages and isinstance(messages[-1], AIMessage):
                    llm_output = messages[-1].content
                if plan_cache is not None and cache_key is not None and cache_mode == "replay" and cache_entry is None:
                    self.last_plan_cache_event.update({"hit": False, "status": "miss_fallback"})

                if not tools_instance.buffered_action.is_empty():
                    llm_action_found = True
                    print(f"[Director] LLM iteration {llm_attempts} found planning anchor")
                    if plan_cache is not None and cache_key is not None and cache_mode in {"record", "refresh"}:
                        plan_cache.put(
                            cache_key,
                            self._make_plan_cache_entry(
                                key=cache_key,
                                state=state,
                                reason=reason,
                                planning_horizon=planning_horizon,
                                attempt=llm_attempts,
                                prompt_text=prompt_text,
                                status_brief=status_brief,
                                llm_output=llm_output,
                                llm_duration=llm_duration,
                                llm_action_found=llm_action_found,
                                tools_instance=tools_instance,
                            ),
                        )
                        self.last_plan_cache_event.update({"hit": False, "status": "written"})
                    break

                if plan_cache is not None and cache_key is not None and cache_mode in {"record", "refresh"}:
                    plan_cache.put(
                        cache_key,
                        self._make_plan_cache_entry(
                            key=cache_key,
                            state=state,
                            reason=reason,
                            planning_horizon=planning_horizon,
                            attempt=llm_attempts,
                            prompt_text=prompt_text,
                            status_brief=status_brief,
                            llm_output=llm_output,
                            llm_duration=llm_duration,
                            llm_action_found=llm_action_found,
                            tools_instance=tools_instance,
                        ),
                    )
                    self.last_plan_cache_event.update({"hit": False, "status": "written_empty"})
                elif plan_cache is not None and cache_key is not None and cache_mode == "replay":
                    self.last_plan_cache_event.update({"hit": False, "status": "miss_fallback"})

                print(f"[Director] LLM iteration {llm_attempts} planning anchor empty, retrying...")

            print(
                f"[Director] LLM 重规划耗时: {llm_duration:.2f}s, "
                f"迭代次数: {llm_attempts}, 成功: {llm_action_found}"
            )

            target_control = self._select_fallback_control(state=state, analysis=analysis)
            if llm_action_found:
                target_control = tools_instance.buffered_action.to_array()

            target_control = apply_safety_guardrails(state, target_control)

            target_temp = tools_instance.buffered_setpoints.get("target_temp")
            target_co2 = tools_instance.buffered_setpoints.get("target_co2")
            target_rh = tools_instance.buffered_setpoints.get("target_rh")
            try:
                target_temp = None if target_temp is None else float(target_temp)
            except Exception:
                target_temp = None
            try:
                target_co2 = None if target_co2 is None else float(target_co2)
            except Exception:
                target_co2 = None
            try:
                target_rh = None if target_rh is None else float(target_rh)
            except Exception:
                target_rh = None

            try:
                target_temp, target_co2, target_rh, missing_fields, corrected_fields = self._enforce_setpoint_contract(
                    state,
                    np.asarray(target_control, dtype=np.float32),
                    target_temp,
                    target_co2,
                    target_rh,
                )
            except Exception as e:
                print(f"[Director] Setpoint contract enforcement failed: {e}, using fallback values")
                target_temp = 17.0
                target_co2 = 430.0
                target_rh = 76.0
                missing_fields = ["target_temp", "target_co2", "target_rh"]
                corrected_fields = []
            
            if missing_fields or corrected_fields:
                print(
                    "[Director] Setpoint contract applied: "
                    f"filled={missing_fields if missing_fields else 'none'}, "
                    f"corrected={corrected_fields if corrected_fields else 'none'}"
                )

            target_profile, profile_contract = self._build_target_profiles(
                tools_instance.buffered_setpoints,
                target_temp,
                target_co2,
                target_rh,
                planning_horizon,
            )
            anchor_source = "llm" if llm_action_found else str(
                self.last_fallback_selection.get("source", "fallback_scored")
            )
            plan = {
                "anchor_control": np.asarray(target_control, dtype=np.float32),
                "anchor_source": anchor_source,
                "target_temp": target_temp,
                "target_co2": target_co2,
                "target_rh": target_rh,
                "target_profile": target_profile,
                "profile_contract": profile_contract,
                "setpoint_contract": {
                    "filled": list(missing_fields),
                    "corrected": list(corrected_fields),
                },
                "fallback_selection": dict(self.last_fallback_selection) if not llm_action_found else {},
                "created_timestep": int(state.timestep),
                "expires_timestep": int(state.timestep) + planning_horizon,
                "reason": reason,
                "llm_action_found": llm_action_found,
                "plan_interval": planning_horizon,
                "plan_cache_event": dict(getattr(self, "last_plan_cache_event", {})),
            }

            if plan_cache is not None and cache_key is not None and cache_mode in {"record", "refresh"}:
                plan_cache.update(
                    cache_key,
                    {
                        "parsed_plan": to_jsonable(plan),
                        "setpoint_contract_plan": {
                            "anchor_control": to_jsonable(np.asarray(target_control, dtype=np.float32)),
                            "target_temp": target_temp,
                            "target_co2": target_co2,
                            "target_rh": target_rh,
                            "target_profile": target_profile,
                            "profile_contract": profile_contract,
                            "setpoint_contract": plan.get("setpoint_contract", {}),
                        },
                        "fallback_info": plan.get("fallback_selection", {}),
                        "anchor_source": anchor_source,
                    },
                )

            # v2.1.2: 记录决策理由
            reasoning = self._record_decision_reasoning(plan, state, analysis)
            plan["reasoning"] = reasoning

            self.current_plan = plan
            self.last_control = np.asarray(target_control, dtype=np.float32)
            self.last_llm_state = state
            self.last_plan_message = llm_output

            return {
                "success": True,
                "llm_attempts": llm_attempts,
                "llm_action_found": llm_action_found,
                "llm_output": llm_output,
                "plan": plan,
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "llm_attempts": llm_attempts,
                "llm_action_found": llm_action_found,
                "llm_output": llm_output,
            }

    def _get_plan_target(self, plan: Dict[str, Any], target_key: str, state):
        profile = plan.get("target_profile", {}) if isinstance(plan, dict) else {}
        values = profile.get(target_key) if isinstance(profile, dict) else None
        if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
            created = int(plan.get("created_timestep", int(state.timestep)))
            idx = int(np.clip(int(state.timestep) - created, 0, len(values) - 1))
            try:
                return float(values[idx])
            except Exception:
                pass
        return plan.get(target_key)

    def _plan_control_step(self, state):
        plan = self.current_plan or {}
        if "anchor_control" in plan:
            anchor_control = np.asarray(plan["anchor_control"], dtype=np.float32)
        else:
            anchor_control = np.asarray(self._select_fallback_control(state=state), dtype=np.float32)
        rule_control = anchor_control.copy()
        lamp_budget_remaining = self._update_lamp_budget(state)
        dehumidify_mode = self._update_dehumidify_mode(state)
        rh_debt = self._update_rh_violation_debt(state)

        env = self.interface.env
        if self.rule_controller is not None and hasattr(env, "x"):
            try:
                weather = self._get_weather_vector()
                rule_control = np.asarray(self.rule_controller.predict(env.x, weather, env), dtype=np.float32)
            except Exception as e:
                print(f"[Director] Rule rollout failed, fallback to anchor control: {e}")

        expert_control = self._predict_expert_control(
            state=state,
            plan=plan,
            rh_debt=rh_debt,
            lamp_budget_remaining=lamp_budget_remaining,
            dehumidify_mode=dehumidify_mode,
        )
        humidity_memory_control = self._predict_humidity_memory_control(state=state, plan=plan)

        plan_start = int(plan.get("created_timestep", int(state.timestep)))
        plan_end = int(plan.get("expires_timestep", plan_start + 1))
        plan_len = max(1, plan_end - plan_start)
        progress = float(np.clip((int(state.timestep) - plan_start) / plan_len, 0.0, 1.0))

        # Risk-weighted control sharing. The previous fixed 55%-80% rule blend was too
        # trusting when the rule controller itself performed poorly in this benchmark.
        rule_weight_start = float(np.clip(self.config.rollout_rule_weight_start, 0.0, 1.0))
        rule_weight_end = float(np.clip(self.config.rollout_rule_weight_end, 0.0, 1.0))
        rule_weight = float(np.clip(rule_weight_start + (rule_weight_end - rule_weight_start) * progress, 0.0, 1.0))
        nominal_blend = (1.0 - rule_weight) * anchor_control + rule_weight * rule_control
        control = np.asarray(nominal_blend, dtype=np.float32)

        if bool(getattr(self.config, "rollout_candidate_sharing", True)):
            candidate_controls = [
                ("anchor", anchor_control, 0.0),
                ("anchor_lean_blend", 0.75 * anchor_control + 0.25 * rule_control, 0.25),
                ("nominal_blend", nominal_blend, rule_weight),
                ("rule_lean_blend", 0.25 * anchor_control + 0.75 * rule_control, 0.75),
                ("rule", rule_control, 1.0),
            ]
            if expert_control is not None:
                expert_blend = float(np.clip(getattr(self.config, "expert_candidate_blend", 0.55), 0.0, 1.0))
                candidate_controls.extend(
                    [
                        ("distilled_expert", expert_control, 0.0),
                        (
                            "expert_anchor_blend",
                            expert_blend * expert_control + (1.0 - expert_blend) * anchor_control,
                            0.0,
                        ),
                        (
                            "expert_rule_blend",
                            expert_blend * expert_control + (1.0 - expert_blend) * rule_control,
                            1.0 - expert_blend,
                        ),
                    ]
                )
            if humidity_memory_control is not None:
                memory_blend = float(np.clip(getattr(self.config, "humidity_memory_candidate_blend", 0.75), 0.0, 1.0))
                candidate_controls.extend(
                    [
                        ("humidity_memory", humidity_memory_control, 0.0),
                        (
                            "humidity_memory_anchor_blend",
                            memory_blend * humidity_memory_control + (1.0 - memory_blend) * anchor_control,
                            0.0,
                        ),
                        (
                            "humidity_memory_rule_blend",
                            memory_blend * humidity_memory_control + (1.0 - memory_blend) * rule_control,
                            1.0 - memory_blend,
                        ),
                    ]
                )
            raw_scored_candidates = []
            for name, candidate_control, candidate_rule_weight in candidate_controls:
                clipped = np.clip(np.asarray(candidate_control, dtype=np.float32), 0.0, 1.0)
                score, details = self._score_fallback_candidate(state, clipped)
                raw_scored_candidates.append((score, name, clipped, candidate_rule_weight, details))
            non_hem_candidates = [
                (score, control_item, details)
                for score, name, control_item, _candidate_rule_weight, details in raw_scored_candidates
                if not str(name).startswith("humidity_memory")
            ]
            best_non_hem = None
            if non_hem_candidates:
                best_non_hem = min(non_hem_candidates, key=lambda item: item[0])

            scored_candidates = []
            best_memory_eval: Optional[Dict[str, Any]] = None
            horizon_filter_enabled = bool(getattr(self.config, "humidity_memory_horizon_filter", True))
            for raw_score, name, clipped, candidate_rule_weight, details in raw_scored_candidates:
                score = float(raw_score)
                candidate_details: Dict[str, Any] = dict(details)
                if horizon_filter_enabled and str(name).startswith("humidity_memory"):
                    score, hem_eval = self._evaluate_humidity_memory_horizon(
                        state,
                        clipped,
                        raw_score=float(raw_score),
                        details=details,
                        best_non_hem=best_non_hem,
                        plan=plan,
                    )
                    candidate_details["humidity_memory_horizon_filter"] = hem_eval
                    if best_memory_eval is None or float(hem_eval["adjusted_score"]) < float(
                        best_memory_eval["adjusted_score"]
                    ):
                        best_memory_eval = dict(hem_eval)
                        best_memory_eval["candidate_name"] = name
                scored_candidates.append((score, name, clipped, candidate_rule_weight, candidate_details))
            scored_candidates.sort(key=lambda item: item[0])
            _, selected_name, selected_control, selected_rule_weight, selected_details = scored_candidates[0]
            mc_sero_shadow = self._evaluate_mc_sero_shadow(
                state,
                rollout_candidates=[(name, control_item) for _score, name, control_item, _rw, _details in raw_scored_candidates],
                baseline_source=str(selected_name),
                baseline_control=np.asarray(selected_control, dtype=np.float32),
            )
            if str(selected_name).startswith("humidity_memory"):
                self.last_humidity_memory_selected_step = int(float(getattr(state, "timestep", 0.0)))
                if best_memory_eval is not None:
                    best_memory_eval["rejected"] = False
                    best_memory_eval["reject_reason"] = ""
            humidity_memory_prediction = dict(getattr(self, "last_humidity_memory_prediction", {}))
            if best_memory_eval is not None:
                humidity_memory_prediction.update(
                    {
                        "reject_reason": best_memory_eval.get("reject_reason", ""),
                        "horizon_penalty": float(best_memory_eval.get("horizon_penalty", 0.0)),
                        "direct_saving": float(best_memory_eval.get("direct_saving", 0.0)),
                        "recovery_heat_cost": float(best_memory_eval.get("recovery_heat_cost", 0.0)),
                        "rh_debt_penalty": float(best_memory_eval.get("rh_debt_penalty", 0.0)),
                        "best_non_hem_score": best_memory_eval.get("best_non_hem_score"),
                        "horizon_adjusted_score": float(best_memory_eval.get("adjusted_score", 0.0)),
                        "horizon_filter_rejected": bool(best_memory_eval.get("rejected", False)),
                        "horizon_filter_candidate": best_memory_eval.get("candidate_name"),
                    }
                )
                self.last_humidity_memory_prediction = humidity_memory_prediction
            control = np.asarray(selected_control, dtype=np.float32)
            rule_weight = float(selected_rule_weight)
            self.last_rollout_selection = {
                "source": selected_name,
                "score": float(scored_candidates[0][0]),
                "rule_weight": float(rule_weight),
                "details": selected_details,
                "candidates": [
                    {
                        "name": name,
                        "score": float(score) if np.isfinite(score) else None,
                        "rule_weight": float(candidate_rule_weight),
                        "humidity_memory_rejected": bool(
                            isinstance(details.get("humidity_memory_horizon_filter"), dict)
                            and details["humidity_memory_horizon_filter"].get("rejected", False)
                        ),
                        "humidity_memory_reject_reason": (
                            details.get("humidity_memory_horizon_filter", {}).get("reject_reason", "")
                            if isinstance(details.get("humidity_memory_horizon_filter"), dict)
                            else ""
                        ),
                    }
                    for score, name, _, candidate_rule_weight, details in scored_candidates
                ],
                "expert_prediction": dict(getattr(self, "last_expert_prediction", {})),
                "humidity_memory_prediction": humidity_memory_prediction,
                "humidity_memory_horizon_filter": dict(best_memory_eval or {}),
                "mc_sero_shadow": mc_sero_shadow,
            }
        else:
            mc_sero_shadow = self._evaluate_mc_sero_shadow(
                state,
                rollout_candidates=[
                    ("fixed_blend", control),
                    ("anchor", anchor_control),
                    ("rule", rule_control),
                ],
                baseline_source="fixed_blend",
                baseline_control=np.asarray(control, dtype=np.float32),
            )
            self.last_rollout_selection = {
                "source": "fixed_blend",
                "score": None,
                "rule_weight": float(rule_weight),
                "candidates": [],
                "expert_prediction": dict(getattr(self, "last_expert_prediction", {})),
                "humidity_memory_prediction": dict(getattr(self, "last_humidity_memory_prediction", {})),
                "mc_sero_shadow": mc_sero_shadow,
            }

        target_temp = self._get_plan_target(plan, "target_temp", state)
        if target_temp is not None:
            temp_err = float(target_temp) - float(state.temp_air)
            if temp_err > 0.25:
                control[0] = np.clip(control[0] + 0.10 * temp_err, 0.0, 1.0)
                control[3] = np.clip(control[3] - 0.06 * temp_err, 0.0, 1.0)
            elif temp_err < -0.25:
                cool_err = -temp_err
                control[3] = np.clip(control[3] + 0.08 * cool_err, 0.0, 1.0)
                control[0] = np.clip(control[0] - 0.08 * cool_err, 0.0, 1.0)

        target_co2 = self._get_plan_target(plan, "target_co2", state)
        if target_co2 is not None:
            co2_err = float(target_co2) - float(state.co2_air)
            is_day = 6 <= float(state.hour_of_day) <= 18
            if is_day and co2_err > 20 and control[3] <= 0.30:
                control[1] = np.clip(control[1] + 0.0015 * co2_err, 0.0, 1.0)
            elif co2_err < -20:
                control[1] = np.clip(control[1] - 0.0015 * (-co2_err), 0.0, 1.0)

        target_rh = self._get_plan_target(plan, "target_rh", state)
        if target_rh is not None:
            rh_err = float(target_rh) - float(state.rh_air)
            if rh_err < -5.0: # 当前湿度过高，需要除湿 (通风)
                control[3] = np.clip(control[3] + 0.06 * (-rh_err) / 10.0, 0.0, 1.0)
                if float(state.temp_air) < 15.5: # 仅在偏冷时配合轻微加热除湿
                    control[0] = np.clip(control[0] + 0.01 * (-rh_err) / 10.0, 0.0, 1.0)
            elif rh_err > 10.0: # 当前湿度过低，需要保水 (减少通风)
                control[3] = np.clip(control[3] - 0.05 * rh_err / 10.0, 0.0, 1.0)

        rh_air = float(state.rh_air)
        temp_air = float(state.temp_air)
        vpd_now = float(calculate_vpd_kpa(temp_air, rh_air))
        dew_margin = min(
            float(getattr(state, "dew_margin_air", 3.0)),
            float(getattr(state, "canopy_dew_margin", 3.0)),
        )
        low_vpd_humidity_risk = (
            (rh_air >= 82.0 and vpd_now < 0.42)
            or (rh_air >= 78.0 and vpd_now < 0.25)
        )
        rh_risk_active = (
            rh_air >= float(self.config.rh_preemptive_threshold)
            or low_vpd_humidity_risk
            or dew_margin < 1.1
            or rh_debt > 2.0
        )
        if rh_risk_active:
            severity = max(rh_air - float(self.config.rh_preemptive_threshold), 0.0)
            if low_vpd_humidity_risk:
                severity = max(severity, max(0.42 - vpd_now, 0.0) * 6.0)
            severity = max(severity, max(1.1 - dew_margin, 0.0) * 2.0)
            debt_vent = min(0.22, float(self.config.rh_debt_vent_gain) * rh_debt)
            debt_screen = min(0.28, float(self.config.rh_debt_screen_gain) * rh_debt)
            vent_floor = min(0.68, 0.30 + 0.035 * severity + debt_vent)
            screen_cap = max(0.25, 0.72 - 0.035 * severity - debt_screen)
            control[3] = max(control[3], vent_floor)
            control[2] = min(control[2], screen_cap)
            control[1] = 0.0
            control[4] = 0.0
            if temp_air < 10.0:
                control[0] = max(control[0], 0.35)
            elif temp_air < 15.0:
                control[0] = max(control[0], 0.18)
            elif vpd_now < 0.35 and temp_air < 18.0:
                control[0] = max(control[0], 0.08)

            if (
                rh_air >= 92.0
                or (rh_air >= 78.0 and vpd_now < 0.22)
                or dew_margin < 0.6
            ) and temp_air < float(self.config.rh_pulse_temp_cap):
                extreme_rh = rh_air >= 94.0 or (rh_air >= 78.0 and vpd_now < 0.18) or dew_margin < 0.4
                heat_floor = (
                    float(self.config.rh_pulse_extreme_heat_floor)
                    if extreme_rh
                    else float(self.config.rh_pulse_heat_floor)
                )
                if temp_air < 15.0:
                    heat_floor = max(heat_floor, 0.28)
                control[0] = max(control[0], heat_floor)
                control[3] = max(
                    control[3],
                    float(self.config.rh_pulse_extreme_vent_floor)
                    if extreme_rh
                    else float(self.config.rh_pulse_vent_floor),
                )
                control[2] = min(
                    control[2],
                    float(self.config.rh_pulse_extreme_screen_cap)
                    if extreme_rh
                    else float(self.config.rh_pulse_screen_cap),
                )

        # Profit-v2: 补光预算化与高湿禁补光
        if lamp_budget_remaining <= 0.0:
            control[4] = 0.0
        elif lamp_budget_remaining < 0.8:
            control[4] = min(control[4], float(self.config.lamp_budget_soft_cap))

        if float(state.rh_air) >= float(self.config.lamp_forbidden_rh) or control[3] >= float(self.config.lamp_forbidden_vent):
            control[4] = 0.0

        # v2.1.2: 强化高湿时补光与CO2的联动抑制
        # 在除湿模式下更严格地限制补光和CO2
        if dehumidify_mode != "normal":
            control[4] = 0.0  # 在任何除湿模式下都关闭补光
            control[1] = 0.0  # 在任何除湿模式下都关闭CO2
        elif float(state.rh_air) >= 88.0:
            control[1] = 0.0

        # v2.1.2: RH上升趋势时提前关灯
        if hasattr(self, '_prev_rh_air'):
            rh_trend = float(state.rh_air) - self._prev_rh_air
            if rh_trend > 0.5 and control[4] > 0:  # RH快速上升且正在补光
                control[4] = 0.0
        self._prev_rh_air = float(state.rh_air)

        # Profit-v2: CO2 防浪费
        if control[3] >= float(self.config.co2_forbidden_vent) or float(state.glob_rad) < float(self.config.fallback_co2_min_rad):
            control[1] = 0.0

        # Profit-v2: 黎明前预除湿
        hour = float(state.hour_of_day)
        if (
            float(self.config.dawn_predehumid_start_hour) <= hour < float(self.config.dawn_predehumid_end_hour)
            and float(state.rh_air) > 82.0
        ):
            control[3] = max(control[3], 0.32)
            control[2] = min(control[2], 0.65)
            control[1] = 0.0
            control[4] = 0.0

        # Profit-v2: 分级除湿模式（带滞回）
        if dehumidify_mode == "mild":
            control[3] = max(control[3], 0.40)
            control[2] = min(control[2], 0.65)
            control[1] = 0.0
            control[4] = 0.0
            # v2.1.2: 高湿阶段加热底线做温度分段
            temp_air = float(state.temp_air)
            if temp_air < 10.0:
                control[0] = max(control[0], 0.30)  # 极冷时保留加热底线
            elif temp_air < 15.5:
                control[0] = max(control[0], 0.08)
            elif temp_air < float(self.config.rh_pulse_temp_cap) and (rh_air >= 90.0 or vpd_now < 0.28):
                control[0] = max(control[0], float(self.config.rh_pulse_heat_floor))
            # 温度回到安全区后不设置加热底线，允许自然调节
        elif dehumidify_mode == "strong":
            control[3] = max(control[3], float(self.config.rh_pulse_vent_floor))
            control[2] = min(control[2], float(self.config.rh_pulse_screen_cap))
            control[1] = 0.0
            control[4] = 0.0
            # v2.1.2: 高湿阶段加热底线做温度分段
            temp_air = float(state.temp_air)
            if temp_air < 10.0:
                control[0] = max(control[0], 0.40)  # 极冷时更强的加热底线
            elif temp_air < 15.0:
                control[0] = max(control[0], 0.12)
            elif temp_air < float(self.config.rh_pulse_temp_cap):
                control[0] = max(control[0], float(self.config.rh_pulse_heat_floor))
            # 温度回到安全区后不设置加热底线，避免加热+通风并发成本

        # Profit-v2.1: RH 极高时增加额外硬约束，抑制尾部持续超湿
        if float(state.rh_air) >= 94.0:
            control[3] = max(control[3], float(self.config.rh_pulse_extreme_vent_floor))
            control[2] = min(control[2], float(self.config.rh_pulse_extreme_screen_cap))
            control[1] = 0.0
            control[4] = 0.0
            temp_air = float(state.temp_air)
            if temp_air < 10.0:
                control[0] = max(control[0], 0.35)  # 极冷时加热底线
            elif temp_air < 15.0:
                control[0] = max(control[0], 0.28)
            elif temp_air < float(self.config.rh_pulse_temp_cap):
                control[0] = max(control[0], float(self.config.rh_pulse_extreme_heat_floor))

        dry_side_risk = (
            float(state.rh_air) <= float(self.config.dry_rh_on)
            or vpd_now >= float(self.config.dry_vpd_on)
        )
        if dry_side_risk:
            before = np.asarray(control, dtype=np.float32).copy()
            if temp_air >= 28.0:
                vent_cap = float(self.config.dry_hot_vent_cap)
            elif temp_air >= 24.0:
                vent_cap = float(self.config.dry_warm_vent_cap)
            else:
                vent_cap = float(self.config.dry_vent_cap)
            control[3] = min(control[3], vent_cap)
            control[1] = 0.0
            control[4] = 0.0
            if temp_air < 12.5:
                control[0] = max(control[0], 0.75)
                control[2] = max(control[2], 0.90)
                control[3] = min(control[3], float(self.config.cold_dehumidify_vent_cap))
            elif temp_air < 15.0:
                control[0] = min(max(control[0], 0.10), 0.25)
                control[2] = max(control[2], 0.75)
            else:
                control[0] = min(control[0], 0.05)
                if temp_air < 18.0:
                    control[2] = max(control[2], 0.70)
            if float(state.glob_rad) > 250.0 or temp_air > 24.0:
                control[5] = max(control[5], 0.50)
            self.last_rollout_selection["dry_recovery_override"] = {
                "applied": True,
                "rh_air": float(state.rh_air),
                "vpd": float(vpd_now),
                "before": before.tolist(),
                "after": np.asarray(control, dtype=np.float32).tolist(),
            }
        else:
            self.last_rollout_selection.setdefault("dry_recovery_override", {"applied": False})

        return (
            np.asarray(control, dtype=np.float32),
            np.asarray(rule_control, dtype=np.float32),
            np.asarray(anchor_control, dtype=np.float32),
            float(rule_weight),
        )

    def _execute_control_step(self, control: np.ndarray, log_prefix: str = "[Director]"):
        env = self.interface.env
        if hasattr(env, "step_raw_control"):
            _, reward, terminated, truncated, _ = env.step_raw_control(control)
            log_control_tracking(log_prefix, np.asarray(control, dtype=np.float32), env.u)
        else:
            action_cmd = control_to_action_command(env, control)
            _, reward, terminated, truncated, _ = env.step(action_cmd)
            log_control_tracking(log_prefix, np.asarray(control, dtype=np.float32), env.u, action_cmd)

        self.interface.last_reward = reward
        self.interface.last_terminated = terminated
        self.interface.last_truncated = truncated
        return reward, bool(terminated or truncated)
    
    def analyze_state(self) -> Dict:
        """生成详细的状态分析报告"""
        state = self.interface.get_state()
        suggestions = []
        
        # 严重违规检测 (Critical Detection)
        critical_violations = []
        critical_level = "none"
        
        # Dynamic Thresholds for Seedlings
        temp_crit_low = self.rules["temp_critical_low"]
        if state.fruit_weight < 1.0:
            # For seedlings, we accept lower temperatures (down to 10.0) to save energy
            # We only panic if it drops below 10.0 (Severe Penalty zone)
            # This prevents Rule-Based Controller from hijacking LLM when T=12.5
            temp_crit_low = 10.0
            
        if state.temp_air < temp_crit_low:
            msg = "CRITICAL: 温度过低，需要立即加热!"
            suggestions.append(msg)
            critical_violations.append("temp_low")
        elif state.temp_air > self.rules["temp_critical_high"]:
            msg = "CRITICAL: 温度过高，需要通风降温"
            suggestions.append(msg)
            critical_violations.append("temp_high")

        if state.rh_air >= 94.0:
            suggestions.append("CRITICAL: 湿度极高，需立即强制除湿")
            critical_violations.append("rh_very_high")
            
        # 一般建议
        if state.temp_air < self.rules["temp_low"]:
            suggestions.append("温度偏低，建议增加供暖")
        elif state.temp_air > self.rules["temp_high"]:
            suggestions.append("温度偏高，建议增加通风")
            
        if state.co2_air < self.rules["co2_low"]:
            suggestions.append("CO2浓度过低，建议补充CO2")
        elif state.co2_air > self.rules["co2_high"]:
            suggestions.append("CO2浓度过高，建议减少供给或通风")
            
        if state.rh_air < self.rules["rh_low"]:
            suggestions.append("湿度过低，建议减少通风")
        elif state.rh_air > self.rules["rh_high"]:
            suggestions.append("湿度过高，建议增加通风")
        
        violation_count = sum([
            state.temp_air < self.rules["temp_low"] or state.temp_air > self.rules["temp_high"],
            state.co2_air < self.rules["co2_low"] or state.co2_air > self.rules["co2_high"],
            state.rh_air < self.rules["rh_low"] or state.rh_air > self.rules["rh_high"]
        ])

        if critical_violations:
            critical_level = "normal"
        if (
            state.temp_air < (temp_crit_low - 1.0)
            or state.temp_air > (self.rules["temp_critical_high"] + 2.0)
            or state.rh_air >= 96.0
        ):
            critical_level = "severe"
        
        return {
            "state": state,
            "suggestions": suggestions,
            "needs_action": len(suggestions) > 0,
            "violation_count": violation_count,
            "is_critical": len(critical_violations) > 0,
            "critical_violations": critical_violations,
            "critical_level": critical_level,
        }
    
    def step_with_rules(self) -> Dict:
        """
        执行一步混合控制决策（规划执行版）

        - LLM 只在“初始无规划 / 规划到期 / 紧急状态”时触发。
        - 其余时间由规则控制器执行规划并闭环跟踪 setpoints。
        """
        analysis = self.analyze_state()
        state = analysis["state"]
        current_interval = self._update_dynamic_interval(state)

        replan_reason = None
        if not self._is_plan_active(int(state.timestep)):
            replan_reason = "init_plan" if self.current_plan is None else "plan_expired"
        elif self._should_emergency_replan(analysis, state):
            replan_reason = "emergency_replan"

        llm_attempts = 0
        llm_action_found = False

        if replan_reason is not None:
            replan_result = self._replan_with_llm(state, analysis, replan_reason, planning_interval=current_interval)
            if replan_result.get("success", False):
                llm_attempts = int(replan_result.get("llm_attempts", 0))
                llm_action_found = bool(replan_result.get("llm_action_found", False))
                self.last_llm_trigger_step = int(state.timestep)
                self.emergency_streak_steps = 0
                self.steps_since_last_llm = 0
            else:
                print(f"[Director] LLM 重规划失败，降级为保守规划: {replan_result.get('error', 'unknown error')}")
                fallback_control = apply_safety_guardrails(state, self._select_fallback_control(state=state, analysis=analysis))
                fallback_horizon = max(2, int(current_interval) // 2)
                fallback_temp, fallback_co2, fallback_rh, missing_fields, corrected_fields = self._enforce_setpoint_contract(
                    state,
                    np.asarray(fallback_control, dtype=np.float32),
                    None,
                    None,
                    None,
                )
                target_profile, profile_contract = self._build_target_profiles(
                    {},
                    fallback_temp,
                    fallback_co2,
                    fallback_rh,
                    fallback_horizon,
                )
                self.current_plan = {
                    "anchor_control": np.asarray(fallback_control, dtype=np.float32),
                    "anchor_source": str(self.last_fallback_selection.get("source", "fallback_scored")),
                    "target_temp": fallback_temp,
                    "target_co2": fallback_co2,
                    "target_rh": fallback_rh,
                    "target_profile": target_profile,
                    "profile_contract": profile_contract,
                    "setpoint_contract": {
                        "filled": list(missing_fields),
                        "corrected": list(corrected_fields),
                    },
                    "fallback_selection": dict(self.last_fallback_selection),
                    "created_timestep": int(state.timestep),
                    "expires_timestep": int(state.timestep) + fallback_horizon,
                    "reason": "fallback_plan",
                    "llm_action_found": False,
                    "plan_interval": fallback_horizon,
                }
                self.last_control = np.asarray(fallback_control, dtype=np.float32)
                self.last_plan_message = f"LLM replan failed: {replan_result.get('error', 'unknown error')}"
                self.emergency_streak_steps = 0
                self.steps_since_last_llm += 1
        else:
            self.steps_since_last_llm += 1

        control, rule_control, anchor_control, rule_weight = self._plan_control_step(state)
        buffered_control = self._buffer_control(control)
        final_control = self._flush_buffered_control(state)
        
        # v2.1.2: 计算性能指标
        if self.current_plan is not None:
            self._calculate_performance_metrics(state, final_control, self.current_plan)
        
        reward, done = self._execute_control_step(final_control, log_prefix="[Director-Plan]")

        plan_info = {}
        if self.current_plan is not None:
            expires_timestep = int(self.current_plan.get("expires_timestep", int(state.timestep)))
            current_target_temp = self._get_plan_target(self.current_plan, "target_temp", state)
            current_target_co2 = self._get_plan_target(self.current_plan, "target_co2", state)
            current_target_rh = self._get_plan_target(self.current_plan, "target_rh", state)
            plan_info = {
                "created_timestep": int(self.current_plan.get("created_timestep", int(state.timestep))),
                "expires_timestep": expires_timestep,
                "remaining_steps": max(expires_timestep - int(state.timestep) - 1, 0),
                "target_temp": self.current_plan.get("target_temp"),
                "target_co2": self.current_plan.get("target_co2"),
                "target_rh": self.current_plan.get("target_rh"),
                "current_target_temp": current_target_temp,
                "current_target_co2": current_target_co2,
                "current_target_rh": current_target_rh,
                "target_profile": self.current_plan.get("target_profile", {}),
                "profile_contract": self.current_plan.get("profile_contract", {}),
                "setpoint_contract": self.current_plan.get("setpoint_contract", {}),
                "rule_weight": rule_weight,
                "reason": self.current_plan.get("reason"),
                "anchor_source": self.current_plan.get("anchor_source", "unknown"),
                "fallback_selection": self.current_plan.get("fallback_selection", {}),
                "plan_cache_event": self.current_plan.get("plan_cache_event", dict(getattr(self, "last_plan_cache_event", {}))),
                "rollout_selection": dict(getattr(self, "last_rollout_selection", {})),
                "dehumidify_mode": self.dehumidify_mode,
                "rh_violation_debt": self.rh_violation_debt,
                "lamp_budget_remaining": self.last_lamp_budget_remaining,
                "active_interval": self.active_control_interval,
            }
        
        # v2.1.2: 添加决策理由和性能指标
        reasoning_info = None
        metrics_info = None
        if self.current_reasoning is not None:
            reasoning_info = {
                "primary_goal": self.current_reasoning.primary_goal,
                "secondary_goals": self.current_reasoning.secondary_goals,
                "constraints_applied": self.current_reasoning.constraints_applied,
                "expected_outcome": self.current_reasoning.expected_outcome,
                "confidence_level": self.current_reasoning.confidence_level,
            }
        
        if self.current_metrics is not None:
            metrics_info = {
                "cost_efficiency": self.current_metrics.cost_efficiency,
                "yield_prediction": self.current_metrics.yield_prediction,
                "energy_consumption": self.current_metrics.energy_consumption,
                "disease_risk_score": self.current_metrics.disease_risk_score,
                "control_accuracy": self.current_metrics.control_accuracy,
                "setpoint_achievement": self.current_metrics.setpoint_achievement,
            }

        return {
            "action": "llm_replan" if replan_reason is not None else "plan_rollout",
            "message": self.last_plan_message if self.last_plan_message else "Rule execution under active plan",
            "analysis": analysis,
            "success": True,
            "reward": reward,
            "done": done,
            "replan_reason": replan_reason,
            "llm_attempts": llm_attempts,
            "llm_action_found": llm_action_found,
            "plan": plan_info,
            "reasoning": reasoning_info,
            "metrics": metrics_info,
            "anchor_control": anchor_control.tolist(),
            "rule_control": rule_control.tolist(),
            "buffered_control": np.asarray(buffered_control, dtype=np.float32).tolist(),
            "applied_control": np.asarray(final_control, dtype=np.float32).tolist(),
            "fallback_selection": dict(self.last_fallback_selection),
        }
