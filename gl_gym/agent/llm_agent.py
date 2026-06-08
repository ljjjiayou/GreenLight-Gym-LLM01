"""
LangChain Agent 核心模块

该模块提供了基于 LangChain 的 LLM 智能体实现，用于控制温室环境。
采用了【混合控制架构 (Hybrid Control Architecture)】：
- 规则控制器 (RuleBasedController)：负责处理常规的、明显的边界违规。
- LLM 智能体 (GreenhouseAgent)：负责处理复杂的多目标冲突和微调。

采用 LangChain v1.0+ (LangGraph 核心) 的 create_agent 接口。
固定使用百炼平台 qwen-plus 模型。
"""

from typing import Any, Dict, List, Optional, Callable, Tuple, Mapping, Sequence
from dataclasses import dataclass, field
from collections import deque
import json
import time
from pathlib import Path
import numpy as np

from gl_gym.cstcc.audit_writer import (
    append_jsonl,
    compact_audit_record,
    compact_failure_record,
    default_audit_jsonl_path,
    should_sample_step,
)
from gl_gym.cstcc.config import load_config as load_cstcc_config
from gl_gym.cstcc.contracts import ACTION_FIELDS as CSTCC_ACTION_FIELDS, SemanticSuggestion
from gl_gym.cstcc.runtime_shadow import safe_run_cstcc_shadow_step, verify_final_action_invariant

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

from gl_gym.environments.baseline import RuleBasedController
from gl_gym.common.utils import load_model_hyperparams, calculate_vpd_kpa
from gl_gym.agent.expert_distillation import ACTION_NAMES, DistilledExpertPolicy
from gl_gym.agent.humidity_experience_memory import HumidityExperienceConfig, HumidityExperienceMemory
from gl_gym.agent.plan_intent import (
    action_record_for_labeling,
    infer_plan_intent,
    intent_to_dehumidify_mode,
    strategy_intent_alignment,
)
from gl_gym.agent.intent_contract import intent_contract_from_setpoint_plan
from gl_gym.agent.profile_generator import (
    build_profile_generator_shadow_payload,
    score_profile_candidate_payloads,
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
from gl_gym.agent.structured_anchor import parse_structured_anchor
from gl_gym.agent.structured_anchor_profile_bridge import (
    build_structured_anchor_profile_bridge_shadow_payload,
)
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
    
    固定使用百炼平台 qwen3.7-max 模型，参数经过优化以适应温室控制任务。
    """
    
    # 模型相关配置
    model_name: str = "qwen3.7-max"                   # 使用的模型名称
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
    strict_replay_suppress_uncached_emergency_replans: bool = True

    # Structured planning-anchor migration is default-off and shadow-only. It
    # parses intent/target/risk anchors without changing final control.
    structured_anchor_parser_enabled: bool = False
    structured_anchor_shadow_only: bool = True
    structured_anchor_required_fields: str = (
        "profile_intent,target_temp,target_co2,target_rh,"
        "risk_flags,forbidden_intents,planning_horizon_steps,confidence"
    )
    structured_anchor_compact_json_prompt_enabled: bool = False
    structured_anchor_retry_invalid_or_empty_enabled: bool = False
    structured_anchor_retry_max_attempts: int = 1
    structured_anchor_list_max_items: int = 3
    structured_anchor_profile_bridge_enabled: bool = False
    structured_anchor_profile_bridge_shadow_only: bool = True
    structured_anchor_profile_bridge_record_provenance: bool = True
    
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
    fallback_dry_penalty_weight: float = 1.20
    fallback_vpd_high_penalty_weight: float = 1.00
    fallback_dew_penalty_weight: float = 0.65
    fallback_cost_penalty_weight: float = 0.45
    fallback_smooth_penalty_weight: float = 0.20
    fallback_conflict_penalty_weight: float = 0.75

    # Rollout control sharing: avoid blindly following a weak rule controller.
    rollout_candidate_sharing: bool = True
    rollout_rule_weight_start: float = 0.25
    rollout_rule_weight_end: float = 0.55
    rspc_hot_dry_candidates_enabled: bool = False
    rspc_hot_dry_require_tomato_v2: bool = True
    rspc_hot_dry_temp_threshold: float = 28.5
    rspc_hot_dry_rad_threshold: float = 300.0
    rspc_hot_dry_rh_threshold: float = 60.0
    rspc_hot_dry_vpd_threshold: float = 1.55
    rspc_hot_dry_vent_relief_floor: float = 0.42
    rspc_hot_dry_hot_vent_relief_floor: float = 0.68
    rspc_hot_dry_score_weight: float = 1.35
    rspc_hot_dry_proposer_control_enabled: bool = False
    rspc_hot_dry_proposer_control_min_margin: float = 0.05
    rspc_hot_dry_proposer_control_strict_enabled: bool = False
    rspc_hot_dry_proposer_control_strict_min_margin: float = 0.20
    rspc_hot_dry_proposer_control_strict_rh_max: float = 45.0
    rspc_hot_dry_proposer_control_strict_vpd_min: float = 2.25
    rspc_hot_dry_proposer_control_strict_temp_max: float = 30.5
    rspc_hot_dry_proposer_control_strict_canopy_margin_min: float = 3.0
    rspc_hot_dry_proposer_control_strict_candidate: str = "shadow_hot_dry_humidity_retention"
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

    # Tomato safety v2 is opt-in so frozen baselines stay untouched.
    tomato_safety_v2_enabled: bool = False
    tomato_safety_v2_dry_rh_on: float = 55.0
    tomato_safety_v2_dry_rh_hard: float = 50.0
    tomato_safety_v2_dry_vpd_on: float = 1.20
    tomato_safety_v2_dry_vpd_hard: float = 1.60
    tomato_safety_v2_target_rh_gap: float = 10.0
    tomato_safety_v2_low_temp_threshold: float = 15.5
    tomato_safety_v2_extreme_dew_rh: float = 94.0
    tomato_safety_v2_extreme_dew_margin: float = 0.40
    tomato_safety_v2_canopy_dew_guard_margin: float = 1.00
    tomato_safety_v2_canopy_dew_guard_rh: float = 61.0
    tomato_safety_v2_canopy_dew_screen_cap: float = 0.35
    tomato_safety_v2_canopy_dew_vent_floor: float = 0.42
    tomato_safety_v2_canopy_dew_shade_floor: float = 0.50
    tomato_safety_v2_canopy_dew_temperate_temp_cap: float = 26.5
    tomato_safety_v2_canopy_dew_temperate_vpd_cap: float = 1.25
    tomato_safety_v2_canopy_dew_temperate_air_margin: float = 4.0
    tomato_safety_v2_canopy_dew_temperate_rad: float = 550.0
    tomato_safety_v2_canopy_dew_temperate_screen_cap: float = 0.75
    tomato_safety_v2_canopy_dew_temperate_vent_floor: float = 0.20
    tomato_safety_v2_canopy_dew_preempt_margin: float = 2.00
    tomato_safety_v2_canopy_dew_preempt_rh: float = 61.0
    tomato_safety_v2_canopy_dew_preempt_air_margin: float = 3.0
    tomato_safety_v2_canopy_dew_preempt_temp_min: float = 20.0
    tomato_safety_v2_canopy_dew_preempt_temp_cap: float = 26.5
    tomato_safety_v2_canopy_dew_preempt_rad: float = 300.0
    tomato_safety_v2_canopy_dew_preempt_screen_cap: float = 0.75
    tomato_safety_v2_canopy_dew_preempt_vent_floor: float = 0.18
    tomato_safety_v2_canopy_dew_preempt_vent_cap: float = 0.24
    tomato_safety_v2_canopy_dew_preempt_shade_temp_cap: float = 24.5
    tomato_safety_v2_canopy_dew_preempt_shade_cap: float = 0.35
    tomato_safety_v2_canopy_dew_buffer_margin: float = 2.00
    tomato_safety_v2_canopy_dew_buffer_min_margin: float = 0.50
    tomato_safety_v2_canopy_dew_buffer_rh: float = 56.5
    tomato_safety_v2_canopy_dew_buffer_temp: float = 24.0
    tomato_safety_v2_canopy_dew_buffer_rad: float = 300.0
    tomato_safety_v2_canopy_dew_buffer_screen_cap: float = 0.60
    tomato_safety_v2_canopy_dew_buffer_vent_floor: float = 0.42
    tomato_safety_v2_canopy_dew_buffer_shade_floor: float = 0.50
    tomato_safety_v2_canopy_dew_buffer_shade_cap: float = 0.50
    tomato_safety_v2_hot_temp_threshold: float = 32.0
    tomato_safety_v2_hot_preempt_temp_threshold: float = 31.0
    tomato_safety_v2_hot_rad_threshold: float = 600.0
    tomato_safety_v2_hot_rapid_rise_threshold: float = 0.50
    tomato_safety_v2_hot_rapid_rise_temp_threshold: float = 30.5
    tomato_safety_v2_hot_hold_temp_threshold: float = 29.0
    tomato_safety_v2_hot_hold_rad_threshold: float = 350.0
    tomato_safety_v2_hot_screen_floor: float = 0.00
    tomato_safety_v2_hot_screen_cap: float = 0.35
    tomato_safety_v2_hot_vent_floor: float = 0.85
    tomato_safety_v2_extreme_hot_vent_temp_threshold: float = 35.0
    tomato_safety_v2_extreme_hot_vent_floor: float = 0.95
    tomato_safety_v2_hot_shade_floor: float = 0.80
    tomato_safety_v2_hot_shade_cap: float = 0.90
    tomato_safety_v2_hot_dry_override_vent_floor: float = 0.90
    tomato_safety_v2_near_hot_dry_vent_floor: float = 0.85
    tomato_safety_v2_hot_hold_dry_vent_floor: float = 0.72
    tomato_safety_v2_soft_hot_dry_screen_floor: float = 0.35
    tomato_safety_v2_soft_hot_dry_screen_cap: float = 0.60
    tomato_safety_v2_hot_dry_override_shade_floor: float = 0.85
    tomato_safety_v2_hot_dry_override_shade_cap: float = 0.85
    tomato_safety_v2_hot_dry_solver_screen_floor: float = 0.35
    tomato_safety_v2_hot_dry_solver_temp_threshold: float = 34.0
    tomato_safety_v2_hot_dry_solver_vpd_threshold: float = 2.80
    tomato_safety_v2_hot_dry_solver_rh_threshold: float = 45.0
    tomato_safety_v2_hot_dry_extreme_solver_vpd_threshold: float = 3.05
    tomato_safety_v2_hot_dry_extreme_solver_rh_threshold: float = 43.5
    tomato_safety_v2_hot_dry_extreme_solver_vent_floor: float = 0.95
    tomato_safety_v2_hot_dry_extreme_solver_shade: float = 0.75
    tomato_safety_v2_extreme_hot_dry_vpd_threshold: float = 3.20
    tomato_safety_v2_extreme_hot_dry_rh_threshold: float = 38.0
    tomato_safety_v2_extreme_hot_dry_screen_floor: float = 0.90
    tomato_safety_v2_extreme_hot_dry_screen_cap: float = 1.00
    tomato_safety_v2_extreme_hot_dry_vent_floor: float = 0.95
    tomato_safety_v2_extreme_hot_dry_shade_floor: float = 0.50
    tomato_safety_v2_extreme_hot_dry_shade_cap: float = 0.50
    tomato_safety_v2_extreme_hot_dry_hold_temp_threshold: float = 36.0
    tomato_safety_v2_extreme_hot_dry_hold_vpd_threshold: float = 2.80
    tomato_safety_v2_hot_dry_rad_relief_threshold: float = 350.0
    tomato_safety_v2_hot_dry_recovery_rh: float = 55.0
    tomato_safety_v2_hot_dry_screen_cap: float = 0.75
    tomato_safety_v2_hot_dry_screen_floor: float = 0.45
    tomato_safety_v2_hot_dry_shade_floor: float = 0.50
    tomato_safety_v2_severe_dry_screen_temp_cap: float = 27.0
    tomato_safety_v2_severe_dry_canopy_reserve_margin: float = 4.0
    tomato_safety_v2_severe_dry_canopy_reserve_rad: float = 600.0
    tomato_safety_v2_severe_dry_canopy_reserve_screen_floor: float = 0.70
    tomato_safety_v2_severe_dry_canopy_reserve_screen_cap: float = 0.82
    tomato_safety_v2_severe_dry_canopy_reserve_shade: float = 0.50
    tomato_safety_v2_pre_hot_dry_temp_threshold: float = 28.5
    tomato_safety_v2_pre_hot_dry_vent_floor: float = 0.60
    tomato_safety_v2_pre_hot_dry_warm_vent_temp: float = 29.0
    tomato_safety_v2_pre_hot_dry_warm_vent_floor: float = 0.60
    tomato_safety_v2_pre_hot_dry_hot_vent_temp: float = 30.5
    tomato_safety_v2_pre_hot_dry_hot_vent_floor: float = 0.72
    tomato_safety_v2_pre_hot_dry_vent_cap: float = 0.82
    tomato_safety_v2_pre_hot_dry_rapid_vent_cap: float = 0.82
    tomato_safety_v2_pre_hot_dry_screen_floor: float = 0.70
    tomato_safety_v2_pre_hot_dry_screen_cap: float = 0.82
    tomato_safety_v2_pre_hot_dry_shade_floor: float = 0.88
    tomato_safety_v2_pre_hot_dry_shade_cap: float = 0.92
    tomato_safety_v2_cooldown_dry_temp_threshold: float = 30.5
    tomato_safety_v2_cooldown_dry_vent_cap: float = 0.28
    tomato_safety_v2_cooldown_dry_screen_floor: float = 0.82
    tomato_safety_v2_cooldown_dry_screen_cap: float = 0.85
    tomato_safety_v2_cooldown_dry_shade_floor: float = 0.45
    tomato_safety_v2_cooldown_canopy_reserve_margin: float = 3.0
    tomato_safety_v2_cooldown_canopy_reserve_rad: float = 650.0
    tomato_safety_v2_cooldown_canopy_reserve_rh: float = 58.0
    tomato_safety_v2_cooldown_canopy_reserve_vpd_cap: float = 1.65
    tomato_safety_v2_cooldown_canopy_reserve_screen_cap: float = 0.60
    tomato_safety_v2_cooldown_canopy_reserve_vent_floor: float = 0.42
    tomato_safety_v2_cooldown_canopy_reserve_shade: float = 0.50
    tomato_safety_v2_warm_hard_dry_screen_floor: float = 0.75
    tomato_safety_v2_warm_hard_dry_screen_temp_cap: float = 25.5
    tomato_safety_v2_warm_hard_dry_screen_rad_threshold: float = 350.0
    tomato_safety_v2_suppress_replay_emergency_replans: bool = True

    # v43 opt-in transition gate. Defaults preserve the ordinary llm_rspc_v2 path.
    transition_gate_enabled: bool = False
    transition_gate_soft_limit_path: str = ""
    transition_gate_reversal_window_steps: int = 6
    transition_gate_hard_safety_bypass: bool = True
    profile_feasibility_gate_enabled: bool = False
    profile_feasibility_gate_hard_safety_veto: bool = True
    profile_template_patch_enabled: bool = False
    fallback_post_selection_veto_enabled: bool = False
    profile_template_patch_record_provenance: bool = True
    recovery_anchor_enabled: bool = False
    recovery_anchor_source: str = "tomato_safety_projected_anchor"
    recovery_anchor_record_provenance: bool = True
    profile_action_candidate_shadow_enabled: bool = False
    profile_action_candidate_shadow_record_provenance: bool = True
    profile_action_candidate_shadow_max_candidates: int = 5
    profile_action_envelope_shadow_enabled: bool = False
    profile_action_envelope_shadow_record_provenance: bool = True
    profile_action_envelope_shadow_max_candidates: int = 5
    cstcc_shadow_enabled: bool = False
    cstcc_shadow_version: str = "v81"
    cstcc_shadow_audit_log_root: str = "logs/cstcc_shadow"
    cstcc_shadow_sample_rate: float = 1.0
    cstcc_shadow_save_full_candidates: bool = False
    cstcc_shadow_save_raw_sequences: bool = False
    cstcc_shadow_save_projected_sequences: bool = False
    cstcc_shadow_fail_closed: bool = False
    cstcc_shadow_assert_final_action_invariant: bool = True
    cstcc_shadow_history_steps: int = 60
    cstcc_shadow_config_path: Optional[str] = None

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


CONTROL_TERM_NAMES = ("heating", "co2", "screen", "ventilation", "lighting", "shading")


def _pg_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        numeric = float(value)
        if not np.isfinite(numeric):
            return float(default)
        return numeric
    except Exception:
        return float(default)


def _pg_state_float(state: Any, name: str, default: float = 0.0) -> float:
    return _pg_float(getattr(state, name, default), default)


def _runtime_control_terms(control: Sequence[float] | np.ndarray) -> Dict[str, float]:
    array = np.asarray(control, dtype=np.float32).reshape(-1)
    return {
        name: float(array[idx]) if idx < int(array.size) else 0.0
        for idx, name in enumerate(CONTROL_TERM_NAMES)
    }


def _runtime_delta_terms(
    pre_rule_action: Sequence[float] | np.ndarray,
    post_rule_action: Sequence[float] | np.ndarray,
) -> Dict[str, float]:
    before = np.asarray(pre_rule_action, dtype=np.float32).reshape(-1)
    after = np.asarray(post_rule_action, dtype=np.float32).reshape(-1)
    size = max(int(before.size), int(after.size), len(CONTROL_TERM_NAMES))
    padded_before = np.zeros(size, dtype=np.float32)
    padded_after = np.zeros(size, dtype=np.float32)
    padded_before[: int(before.size)] = before
    padded_after[: int(after.size)] = after
    delta = padded_after - padded_before
    return {name: float(delta[idx]) for idx, name in enumerate(CONTROL_TERM_NAMES)}


def _post_guardrail_runtime_input_features(state: Any) -> Dict[str, float]:
    temp_air = _pg_state_float(state, "temp_air", 20.0)
    rh_air = _pg_state_float(state, "rh_air", 70.0)
    vpd = float(calculate_vpd_kpa(temp_air, rh_air))
    dew_margin_air = _pg_state_float(state, "dew_margin_air", 3.0)
    canopy_margin = _pg_state_float(state, "canopy_dew_margin", dew_margin_air)
    glob_rad = _pg_state_float(state, "glob_rad", 0.0)
    return {
        "timestep": _pg_state_float(state, "timestep", 0.0),
        "temp_air": temp_air,
        "rh_air": rh_air,
        "vpd_air": vpd,
        "dew_margin_air": dew_margin_air,
        "canopy_dew_margin": canopy_margin,
        "combined_dew_margin": min(dew_margin_air, canopy_margin),
        "glob_rad": glob_rad,
        "hour_of_day": _pg_state_float(state, "hour_of_day", 12.0),
        "temp_out": _pg_state_float(state, "temp_out", temp_air),
        "wind_speed": _pg_state_float(state, "wind_speed", 0.0),
        "forecast_rad_mean_1h": _pg_state_float(state, "forecast_rad_mean_1h", glob_rad),
        "forecast_rad_peak_2h": _pg_state_float(state, "forecast_rad_peak_2h", glob_rad),
    }


def _safe_rule_slug(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:80] or "unknown"


def _runtime_rule_from_rewrite(
    *,
    state: Any,
    hook_id: str,
    pre_rule_action: Sequence[float] | np.ndarray,
    post_rule_action: Sequence[float] | np.ndarray,
    info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    info = dict(info or {})
    explicit_rule_id = str(info.get("rule_id", "") or "")
    explicit_family = str(info.get("rule_family", "") or "")
    explicit_reason = str(info.get("rule_reason", "") or "")
    if explicit_rule_id and explicit_family and explicit_reason:
        return {
            "rule_id": explicit_rule_id,
            "rule_family": explicit_family,
            "rule_priority": int(_pg_float(info.get("rule_priority", 50), 50)),
            "rule_reason": explicit_reason,
            "expected_safety_benefit": str(info.get("expected_safety_benefit", "") or "runtime rule documented benefit"),
            "rule_assumed_tradeoff": str(info.get("rule_assumed_tradeoff", "") or "documented by runtime hook"),
            "reason_missing": False,
        }

    delta = _runtime_delta_terms(pre_rule_action, post_rule_action)
    before = _runtime_control_terms(pre_rule_action)
    features = _post_guardrail_runtime_input_features(state)
    hook = str(hook_id)
    if hook == "tomato_safety_v2_wrapper":
        reasons = info.get("reasons", [])
        if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)) and reasons:
            reason_text = "; ".join(str(reason) for reason in reasons if str(reason))
            first = str(reasons[0])
            return {
                "rule_id": f"tomato_safety_v2_{_safe_rule_slug(first)}",
                "rule_family": "tomato_safety_v2",
                "rule_priority": 80,
                "rule_reason": reason_text,
                "expected_safety_benefit": "tomato safety v2 post-guardrail risk reduction",
                "rule_assumed_tradeoff": "may reshape vent, screen, heat, shade, co2, or lighting",
                "reason_missing": False,
            }
        return {
            "rule_id": "unknown_post_guardrail_rewrite",
            "rule_family": "tomato_safety_v2_unknown",
            "rule_priority": 80,
            "rule_reason": "missing_runtime_reason_blocker",
            "expected_safety_benefit": "unknown because runtime reason is missing",
            "rule_assumed_tradeoff": "unknown because runtime reason is missing",
            "reason_missing": True,
        }

    if hook == "humidity_memory_final_shape":
        reason = str(info.get("reason", "") or "")
        if not reason and bool(info.get("applied", False)):
            reason = "humidity_memory_post_guardrail_shape_applied"
        if reason:
            return {
                "rule_id": f"humidity_memory_{_safe_rule_slug(reason)}",
                "rule_family": "humidity_memory_final_shape",
                "rule_priority": 60,
                "rule_reason": reason,
                "expected_safety_benefit": "humidity-memory final shaping tradeoff recorded in metadata",
                "rule_assumed_tradeoff": "may alter screen, ventilation, heat, lighting, or co2",
                "reason_missing": False,
            }

    if hook == "dry_recovery_override":
        vent_cap = info.get("vent_cap")
        hot_dry = bool(info.get("hot_dry_vent_relief", False))
        return {
            "rule_id": "rspc_post_score_dry_recovery_vent_cap",
            "rule_family": "rspc_post_score_dry_recovery",
            "rule_priority": 70,
            "rule_reason": (
                f"dry_side_risk caps ventilation at {float(vent_cap):.3f}"
                if vent_cap is not None
                else "dry_side_risk caps ventilation"
            ),
            "expected_safety_benefit": "reduce excessive dry-side ventilation and preserve humidity",
            "rule_assumed_tradeoff": (
                "hot-dry relief floor active" if hot_dry else "may reduce ventilation near canopy/dew boundary"
            ),
            "reason_missing": False,
        }

    if hook == "apply_safety_guardrails" and float(delta.get("ventilation", 0.0)) < -1e-6:
        temp_air = features["temp_air"]
        rh_air = features["rh_air"]
        vpd = features["vpd_air"]
        wind = features["wind_speed"]
        temp_out = features["temp_out"]
        if vpd > 1.2 and temp_air < 32.0:
            return {
                "rule_id": "apply_safety_guardrails_dry_vpd_vent_cap",
                "rule_family": "apply_safety_guardrails_dry_vpd",
                "rule_priority": 70,
                "rule_reason": "dry-side VPD/RH guard capped ventilation",
                "expected_safety_benefit": "avoid worsening low RH or high VPD",
                "rule_assumed_tradeoff": "may reduce air exchange near canopy/dew risk",
                "reason_missing": False,
            }
        if vpd >= 0.4 and before.get("heating", 0.0) > 0.1 and before.get("ventilation", 0.0) > 0.1:
            cold = temp_air < 16.0
            return {
                "rule_id": (
                    "apply_safety_guardrails_heat_vent_conflict_cold_close_vent"
                    if cold
                    else "apply_safety_guardrails_heat_vent_conflict_limit_vent"
                ),
                "rule_family": "apply_safety_guardrails_heat_vent_conflict",
                "rule_priority": 65,
                "rule_reason": "heat and ventilation conflict guard reduced ventilation",
                "expected_safety_benefit": "avoid simultaneous heating and ventilation waste",
                "rule_assumed_tradeoff": "may reduce ventilation during dry/canopy-risk regimes",
                "reason_missing": False,
            }
        if wind > 8.0 or (wind > 6.0 and temp_out < temp_air):
            return {
                "rule_id": "apply_safety_guardrails_wind_cap",
                "rule_family": "apply_safety_guardrails_wind",
                "rule_priority": 75,
                "rule_reason": "wind safety cap reduced ventilation",
                "expected_safety_benefit": "avoid unsafe vent opening under wind stress",
                "rule_assumed_tradeoff": "may conflict with humidity or dry-risk relief",
                "reason_missing": False,
            }
        if rh_air <= 55.0 or vpd >= 1.2 or temp_air < 15.5:
            return {
                "rule_id": "apply_safety_guardrails_dry_side_or_cold_buffer_vent_cap",
                "rule_family": "apply_safety_guardrails_dry_or_cold_buffer",
                "rule_priority": 55,
                "rule_reason": "dry-side or cold-buffer guard reduced ventilation",
                "expected_safety_benefit": "avoid dry-side or cold-buffer regression",
                "rule_assumed_tradeoff": "may reduce air exchange near canopy/dew boundary",
                "reason_missing": False,
            }

    if hook == "apply_safety_guardrails":
        return {
            "rule_id": "apply_safety_guardrails_general_rewrite",
            "rule_family": "apply_safety_guardrails_general",
            "rule_priority": 50,
            "rule_reason": "generic safety guardrail changed final action",
            "expected_safety_benefit": "generic hard-boundary and resource guard",
            "rule_assumed_tradeoff": "depends on changed actuator",
            "reason_missing": False,
        }

    return {
        "rule_id": "unknown_post_guardrail_rewrite",
        "rule_family": f"{hook}_unknown",
        "rule_priority": int(_pg_float(info.get("rule_priority", 0), 0)),
        "rule_reason": "missing_runtime_reason_blocker",
        "expected_safety_benefit": "unknown because runtime reason is missing",
        "rule_assumed_tradeoff": "unknown because runtime reason is missing",
        "reason_missing": True,
    }


def build_post_guardrail_runtime_provenance_record(
    *,
    state: Any,
    hook_id: str,
    source_function: str,
    pre_rule_action: Sequence[float] | np.ndarray,
    post_rule_action: Sequence[float] | np.ndarray,
    info: Optional[Mapping[str, Any]] = None,
    epsilon: float = 1e-6,
) -> Optional[Dict[str, Any]]:
    before = np.asarray(pre_rule_action, dtype=np.float32).copy()
    after = np.asarray(post_rule_action, dtype=np.float32).copy()
    if before.shape != after.shape:
        size = max(int(before.size), int(after.size))
        padded_before = np.zeros(size, dtype=np.float32)
        padded_after = np.zeros(size, dtype=np.float32)
        padded_before[: int(before.size)] = before.reshape(-1)
        padded_after[: int(after.size)] = after.reshape(-1)
        before = padded_before
        after = padded_after
    if not bool(np.any(np.abs(after - before) > float(epsilon))):
        return None
    rule = _runtime_rule_from_rewrite(
        state=state,
        hook_id=str(hook_id),
        pre_rule_action=before,
        post_rule_action=after,
        info=info,
    )
    return {
        "schema_version": "post_guardrail_runtime_provenance_v1",
        "shadow_only": True,
        "metadata_only": True,
        "control_action_changed_by_metadata": False,
        "hook_id": str(hook_id),
        "rule_id": str(rule.get("rule_id", "")),
        "rule_family": str(rule.get("rule_family", "")),
        "rule_priority": int(_pg_float(rule.get("rule_priority", 0), 0)),
        "rule_reason": str(rule.get("rule_reason", "")),
        "source_function": str(source_function),
        "pre_rule_action": _runtime_control_terms(before),
        "post_rule_action": _runtime_control_terms(after),
        "delta_action": _runtime_delta_terms(before, after),
        "input_features": _post_guardrail_runtime_input_features(state),
        "expected_safety_benefit": str(rule.get("expected_safety_benefit", "")),
        "rule_assumed_tradeoff": str(rule.get("rule_assumed_tradeoff", "")),
        "reason_missing": bool(rule.get("reason_missing", False)),
    }


def apply_safety_guardrails(state, target_control: np.ndarray) -> np.ndarray:
    guarded = np.asarray(target_control, dtype=np.float32).copy()
    temp_air = float(getattr(state, "temp_air", 20.0))
    rh_air = float(getattr(state, "rh_air", 70.0))
    glob_rad = float(getattr(state, "glob_rad", 0.0))
    forecast_rad_mean = _first_finite_state_attr(state, ["forecast_rad_mean_1h"], glob_rad)
    forecast_rad_peak = _first_finite_state_attr(state, ["forecast_rad_peak_2h"], glob_rad)
    rad_heat_load = max(glob_rad, forecast_rad_mean, forecast_rad_peak)
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
    extreme_dew_risk = rh_air >= 94.0 or dew_margin < 0.4
    hot_dry_preemptive = (
        not extreme_dew_risk
        and temp_air >= 29.0
        and rad_heat_load >= 600.0
        and (rh_air < 55.0 or vpd > 1.60)
    )
    hot_dry_solver_sensitive = (
        not extreme_dew_risk
        and temp_air < 35.0
        and rad_heat_load >= 650.0
        and (rh_air <= 52.0 or vpd >= 2.30)
        and temp_out >= 24.0
    )
    hot_dry_ultra_solver_sensitive = (
        hot_dry_solver_sensitive
        and temp_air >= 34.0
        and (vpd >= 2.80 or rh_air <= 45.0)
    )
    hot_dry_extreme_solver_sensitive = (
        hot_dry_ultra_solver_sensitive
        and (vpd >= 3.00 or rh_air <= 43.5)
    )
    hot_dry_raw_extreme_solver_sensitive = (
        not extreme_dew_risk
        and temp_air >= 33.0
        and rad_heat_load >= 600.0
        and temp_out >= 24.0
        and (vpd >= 3.20 or rh_air <= 38.5)
    )
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
        if hot_dry_ultra_solver_sensitive:
            guarded[2] = min(max(guarded[2], 0.35), 0.35)
        else:
            guarded[2] = min(guarded[2], 0.05)
        guarded[4] = 0.0
        vent_target = 0.75 if temp_air < 35.0 else 0.90
        if temp_out < temp_air - 1.0:
            vent_target = max(vent_target, 0.95)
        if hot_dry_extreme_solver_sensitive:
            vent_target = max(vent_target, 0.95)
        elif hot_dry_solver_sensitive:
            vent_target = min(max(vent_target, 0.90), 0.90)
        guarded[3] = max(guarded[3], vent_target)
        if glob_rad > 200.0:
            if hot_dry_extreme_solver_sensitive:
                guarded[5] = min(max(guarded[5], 0.75), 0.75)
            elif hot_dry_solver_sensitive:
                guarded[5] = min(max(guarded[5], 0.85), 0.85)
            else:
                guarded[5] = max(guarded[5], 0.85)
        if hot_dry_raw_extreme_solver_sensitive:
            # CVODES becomes sensitive in very dry, high-radiation heat loads;
            # mirror the opt-in tomato v2 stability fallback without fully
            # closing the screen or dropping shade to zero.
            guarded[2] = min(max(guarded[2], 0.90), 1.00)
            guarded[3] = max(guarded[3], 0.95)
            guarded[5] = min(max(guarded[5], 0.50), 0.50)
    elif hot_dry_preemptive:
        guarded[0] = 0.0
        guarded[1] = 0.0
        guarded[4] = 0.0
        vent_floor = 0.62
        if temp_air >= 30.5 or rad_heat_load >= 650.0:
            vent_floor = max(vent_floor, 0.72)
        if temp_air >= 31.5 or rad_heat_load >= 750.0:
            vent_floor = max(vent_floor, 0.85)
        guarded[3] = min(max(guarded[3], vent_floor), 0.85)
        screen_floor = 0.55 if (rh_air < 50.0 or vpd > 2.0) else 0.45
        guarded[2] = min(max(guarded[2], screen_floor), 0.72)
        shade_floor = 0.75 if rad_heat_load < 750.0 else 0.90
        guarded[5] = max(guarded[5], shade_floor)

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

    if high_temp_override:
        hot_vent_floor = 0.75 if temp_air < 35.0 else 0.90
        if temp_out < temp_air - 1.0:
            hot_vent_floor = max(hot_vent_floor, 0.95)
        if hot_dry_solver_sensitive:
            hot_vent_floor = max(hot_vent_floor, 0.90)
        if hot_dry_extreme_solver_sensitive or hot_dry_raw_extreme_solver_sensitive:
            hot_vent_floor = max(hot_vent_floor, 0.95)
        guarded[3] = max(guarded[3], hot_vent_floor)

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
        vent_floor = 0.0
        if hot_dry_preemptive:
            vent_cap = 0.85
            vent_floor = 0.62
            if temp_air >= 30.5 or rad_heat_load >= 650.0:
                vent_floor = max(vent_floor, 0.72)
            if temp_air >= 31.5 or rad_heat_load >= 750.0:
                vent_floor = max(vent_floor, 0.85)
        elif temp_air >= 28.0:
            vent_cap = 0.35
        elif temp_air >= 24.0:
            vent_cap = 0.18
        else:
            vent_cap = 0.12
        guarded[3] = min(guarded[3], vent_cap)
        if hot_dry_preemptive:
            guarded[3] = max(guarded[3], vent_floor)
        if temp_air >= 12.0:
            guarded[0] = 0.0
        guarded[1] = 0.0
        guarded[4] = 0.0
        if hot_dry_preemptive:
            screen_floor = 0.55 if (rh_air < 50.0 or vpd > 2.0) else 0.45
            guarded[2] = min(max(guarded[2], screen_floor), 0.72)
        elif temp_air < 18.0:
            guarded[2] = max(guarded[2], 0.70)
        if hot_dry_preemptive:
            guarded[5] = max(guarded[5], 0.75 if rad_heat_load < 750.0 else 0.90)
        elif glob_rad > 250.0 or temp_air > 24.0:
            guarded[5] = max(guarded[5], 0.50)

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


def _first_finite_state_attr(state, names: List[str], default: float = 0.0) -> float:
    for name in names:
        if not hasattr(state, name):
            continue
        try:
            value = float(getattr(state, name))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return float(default)


def apply_tomato_safety_v2(
    state,
    target_control: np.ndarray,
    config: Optional[AgentConfig] = None,
    target_rh: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Opt-in tomato dry-side and low-temperature safety shaping.

    This layer is deliberately conservative and disabled by default. It is used
    only by the `llm_rspc_v2` benchmark controller to protect tomato seedlings
    from low RH / high VPD and cold over-ventilation windows without changing the
    ordinary LLM-RSPC baseline.
    """
    cfg = config or AgentConfig()
    shaped = np.asarray(target_control, dtype=np.float32).copy()
    before = shaped.copy()

    temp_air = float(getattr(state, "temp_air", 20.0))
    rh_air = float(getattr(state, "rh_air", 70.0))
    glob_rad = float(getattr(state, "glob_rad", 0.0))
    forecast_rad_mean = _first_finite_state_attr(state, ["forecast_rad_mean_1h"], glob_rad)
    forecast_rad_peak = _first_finite_state_attr(state, ["forecast_rad_peak_2h"], glob_rad)
    forecast_temp_out_delta = _first_finite_state_attr(state, ["forecast_temp_out_delta_1h"], 0.0)
    temp_rise_1h = _first_finite_state_attr(
        state,
        ["temp_air_delta_1h", "temp_air_rise_1h", "temp_air_trend_c_per_hour"],
        0.0,
    )
    temp_slope_step = _first_finite_state_attr(
        state,
        ["temp_air_slope_c_per_step", "temp_air_slope", "temp_trend_slope"],
        0.0,
    )
    hour_of_day = float(getattr(state, "hour_of_day", 12.0))
    air_dew_margin = float(getattr(state, "dew_margin_air", 3.0))
    canopy_dew_margin = float(getattr(state, "canopy_dew_margin", 3.0))
    dew_margin = min(air_dew_margin, canopy_dew_margin)
    vpd = float(calculate_vpd_kpa(temp_air, rh_air))
    is_night = hour_of_day < 6.0 or hour_of_day > 18.0
    extreme_dew_risk = (
        rh_air >= float(getattr(cfg, "tomato_safety_v2_extreme_dew_rh", 94.0))
        or dew_margin < float(getattr(cfg, "tomato_safety_v2_extreme_dew_margin", 0.40))
    )
    hot_temp_threshold = float(getattr(cfg, "tomato_safety_v2_hot_temp_threshold", 32.0))
    hot_preempt_temp = float(getattr(cfg, "tomato_safety_v2_hot_preempt_temp_threshold", 31.0))
    hot_rad_threshold = float(getattr(cfg, "tomato_safety_v2_hot_rad_threshold", 600.0))
    hot_rapid_rise_threshold = float(getattr(cfg, "tomato_safety_v2_hot_rapid_rise_threshold", 0.50))
    hot_rapid_rise_temp = float(getattr(cfg, "tomato_safety_v2_hot_rapid_rise_temp_threshold", 30.5))
    hot_hold_temp_threshold = float(getattr(cfg, "tomato_safety_v2_hot_hold_temp_threshold", 29.0))
    hot_hold_rad_threshold = float(getattr(cfg, "tomato_safety_v2_hot_hold_rad_threshold", 350.0))
    pre_hot_dry_temp_threshold = float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_temp_threshold", 28.5))
    hot_screen_floor = float(getattr(cfg, "tomato_safety_v2_hot_screen_floor", 0.20))
    hot_screen_cap = float(getattr(cfg, "tomato_safety_v2_hot_screen_cap", 0.35))
    hot_shade_floor = float(getattr(cfg, "tomato_safety_v2_hot_shade_floor", 0.80))
    hot_shade_cap = float(getattr(cfg, "tomato_safety_v2_hot_shade_cap", 0.90))
    solver_screen_floor = float(getattr(cfg, "tomato_safety_v2_hot_dry_solver_screen_floor", 0.35))
    solver_temp_threshold = float(getattr(cfg, "tomato_safety_v2_hot_dry_solver_temp_threshold", 34.0))
    solver_vpd_threshold = float(getattr(cfg, "tomato_safety_v2_hot_dry_solver_vpd_threshold", 2.80))
    solver_rh_threshold = float(getattr(cfg, "tomato_safety_v2_hot_dry_solver_rh_threshold", 45.0))
    extreme_solver_vpd_threshold = float(
        getattr(cfg, "tomato_safety_v2_hot_dry_extreme_solver_vpd_threshold", 3.00)
    )
    extreme_solver_rh_threshold = float(
        getattr(cfg, "tomato_safety_v2_hot_dry_extreme_solver_rh_threshold", 43.5)
    )
    extreme_solver_vent_floor = float(
        getattr(cfg, "tomato_safety_v2_hot_dry_extreme_solver_vent_floor", 0.95)
    )
    extreme_solver_shade = float(
        getattr(cfg, "tomato_safety_v2_hot_dry_extreme_solver_shade", 0.75)
    )
    previous_screen = _first_finite_state_attr(state, ["u_th_scr"], float(before[2]) if len(before) > 2 else 0.0)
    previous_shade = _first_finite_state_attr(state, ["u_bl_scr"], float(before[5]) if len(before) > 5 else 0.0)
    previous_vent = _first_finite_state_attr(state, ["u_vent"], float(before[3]) if len(before) > 3 else 0.0)
    rad_heat_load = max(glob_rad, forecast_rad_mean, forecast_rad_peak)
    high_temp_cooling = temp_air >= hot_temp_threshold
    high_rad_near_hot = temp_air >= max(hot_preempt_temp, hot_temp_threshold - 1.0) and glob_rad >= hot_rad_threshold
    rapid_temp_rise = (
        temp_rise_1h >= hot_rapid_rise_threshold
        or temp_slope_step >= 0.08
        or forecast_temp_out_delta >= hot_rapid_rise_threshold
    )
    high_rad_fast_rise = (
        temp_air >= hot_rapid_rise_temp
        and rad_heat_load >= hot_rad_threshold
        and rapid_temp_rise
    )
    continuing_hot_load = (
        temp_air >= hot_hold_temp_threshold
        and rad_heat_load >= hot_hold_rad_threshold
        and previous_screen <= hot_screen_cap + 0.05
        and previous_shade >= hot_shade_floor - 0.05
        and previous_vent >= 0.60
    )
    hot_override_reason = ""
    if high_temp_cooling:
        hot_override_reason = "hot_temperature_override"
    elif high_rad_near_hot:
        hot_override_reason = "high_rad_near_hot_override"
    elif high_rad_fast_rise:
        hot_override_reason = "high_rad_rapid_rise_override"
    elif continuing_hot_load:
        hot_override_reason = "hot_override_hold"
    hot_override = bool(hot_override_reason)

    reasons: List[str] = []
    details: Dict[str, Any] = {
        "enabled": bool(getattr(cfg, "tomato_safety_v2_enabled", False)),
        "applied": False,
        "reasons": reasons,
        "temp_air": temp_air,
        "rh_air": rh_air,
        "vpd_kpa": vpd,
        "dew_margin": dew_margin,
        "air_dew_margin": air_dew_margin,
        "canopy_dew_margin": canopy_dew_margin,
        "target_rh": None if target_rh is None else float(target_rh),
        "forecast_rad_mean_1h": forecast_rad_mean,
        "forecast_rad_peak_2h": forecast_rad_peak,
        "forecast_temp_out_delta_1h": forecast_temp_out_delta,
        "temp_rise_1h": temp_rise_1h,
        "temp_slope_step": temp_slope_step,
        "hot_rapid_rise_temp_threshold": hot_rapid_rise_temp,
        "pre_hot_dry_temp_threshold": pre_hot_dry_temp_threshold,
        "hot_screen_floor": hot_screen_floor,
        "hot_screen_cap": hot_screen_cap,
        "hot_shade_floor": hot_shade_floor,
        "hot_shade_cap": hot_shade_cap,
        "rapid_temp_rise": rapid_temp_rise,
        "continuing_hot_load": continuing_hot_load,
        "canopy_dew_relief": False,
        "previous_screen": previous_screen,
        "previous_shade": previous_shade,
        "previous_vent": previous_vent,
        "hot_override": hot_override,
        "hot_override_reason": hot_override_reason,
        "before": before.tolist(),
    }

    if not bool(getattr(cfg, "tomato_safety_v2_enabled", False)):
        details["after"] = shaped.tolist()
        return np.clip(shaped, 0.0, 1.0), details

    dry_risk = (
        rh_air < float(getattr(cfg, "tomato_safety_v2_dry_rh_on", 55.0))
        or vpd > float(getattr(cfg, "tomato_safety_v2_dry_vpd_on", 1.20))
    )
    hard_dry_risk = (
        rh_air < float(getattr(cfg, "tomato_safety_v2_dry_rh_hard", 50.0))
        or vpd > float(getattr(cfg, "tomato_safety_v2_dry_vpd_hard", 1.60))
    )
    target_mismatch = (
        target_rh is not None
        and float(target_rh) - rh_air > float(getattr(cfg, "tomato_safety_v2_target_rh_gap", 10.0))
    )

    hot_dry_dew_block = extreme_dew_risk and rh_air >= 80.0
    canopy_dew_relief = (
        canopy_dew_margin < float(getattr(cfg, "tomato_safety_v2_canopy_dew_guard_margin", 1.00))
        and rh_air > float(getattr(cfg, "tomato_safety_v2_canopy_dew_guard_rh", 61.0))
        and temp_air >= 24.0
    )
    details["canopy_dew_relief"] = bool(canopy_dew_relief)
    canopy_dew_temperate_buffer = (
        canopy_dew_relief
        and not extreme_dew_risk
        and air_dew_margin >= float(getattr(cfg, "tomato_safety_v2_canopy_dew_temperate_air_margin", 4.0))
        and temp_air <= float(getattr(cfg, "tomato_safety_v2_canopy_dew_temperate_temp_cap", 26.5))
        and vpd <= float(getattr(cfg, "tomato_safety_v2_canopy_dew_temperate_vpd_cap", 1.25))
        and rad_heat_load >= float(getattr(cfg, "tomato_safety_v2_canopy_dew_temperate_rad", 550.0))
    )
    details["canopy_dew_temperate_buffer"] = bool(canopy_dew_temperate_buffer)
    canopy_dew_buffer_candidate = (
        not canopy_dew_relief
        and not hot_override
        and canopy_dew_margin < float(getattr(cfg, "tomato_safety_v2_canopy_dew_buffer_margin", 2.00))
        and canopy_dew_margin >= float(getattr(cfg, "tomato_safety_v2_canopy_dew_buffer_min_margin", 0.50))
        and rh_air > float(getattr(cfg, "tomato_safety_v2_canopy_dew_buffer_rh", 56.5))
        and temp_air >= float(getattr(cfg, "tomato_safety_v2_canopy_dew_buffer_temp", 24.0))
        and rad_heat_load >= float(getattr(cfg, "tomato_safety_v2_canopy_dew_buffer_rad", 300.0))
    )
    canopy_dew_target_preempt = (
        target_mismatch
        and not canopy_dew_relief
        and not hot_override
        and not extreme_dew_risk
        and canopy_dew_margin < float(getattr(cfg, "tomato_safety_v2_canopy_dew_preempt_margin", 2.00))
        and rh_air > float(getattr(cfg, "tomato_safety_v2_canopy_dew_preempt_rh", 61.0))
        and air_dew_margin >= float(getattr(cfg, "tomato_safety_v2_canopy_dew_preempt_air_margin", 3.0))
        and temp_air >= float(getattr(cfg, "tomato_safety_v2_canopy_dew_preempt_temp_min", 20.0))
        and temp_air <= float(getattr(cfg, "tomato_safety_v2_canopy_dew_preempt_temp_cap", 26.5))
        and rad_heat_load >= float(getattr(cfg, "tomato_safety_v2_canopy_dew_preempt_rad", 300.0))
    )
    hot_dry_cooling = (
        dry_risk
        and not hot_override
        and not hot_dry_dew_block
        and not canopy_dew_relief
        and (temp_air >= 27.0 or (temp_air >= 25.5 and rad_heat_load >= hot_rad_threshold))
    )
    hot_dry_ultra_solver_sensitive = (
        hot_override
        and dry_risk
        and not extreme_dew_risk
        and temp_air >= solver_temp_threshold
        and rad_heat_load >= hot_rad_threshold
        and (vpd >= solver_vpd_threshold or rh_air <= solver_rh_threshold)
    )
    hot_dry_extreme_solver_sensitive = (
        hot_dry_ultra_solver_sensitive
        and (vpd >= extreme_solver_vpd_threshold or rh_air <= extreme_solver_rh_threshold)
    )
    soft_hot_dry_override = (
        hot_override
        and dry_risk
        and hot_override_reason != "hot_temperature_override"
        and temp_air < hot_temp_threshold
        and not hot_dry_ultra_solver_sensitive
    )
    hot_dry_radiation_relief = (
        hot_dry_cooling
        and (
            rad_heat_load >= float(getattr(cfg, "tomato_safety_v2_hot_dry_rad_relief_threshold", 350.0))
            or temp_air >= hot_hold_temp_threshold
        )
    )
    pre_hot_dry_heat_load = (
        rad_heat_load >= hot_rad_threshold
        and hard_dry_risk
        and temp_air >= pre_hot_dry_temp_threshold
        and (temp_air >= hot_rapid_rise_temp or rapid_temp_rise)
    )
    pre_hot_dry_phase = (
        hot_dry_cooling
        and not hot_override
        and temp_air < hot_preempt_temp
        and rad_heat_load >= hot_rad_threshold
        and (temp_air >= pre_hot_dry_temp_threshold or (rapid_temp_rise and temp_air >= 25.5))
    )
    cooldown_dry_recovery = (
        dry_risk
        and not hot_override
        and not canopy_dew_relief
        and temp_air < float(getattr(cfg, "tomato_safety_v2_cooldown_dry_temp_threshold", 30.5))
        and not pre_hot_dry_phase
        and (
            temp_air < pre_hot_dry_temp_threshold
            or rad_heat_load < hot_rad_threshold
            or previous_vent >= 0.50
        )
    )
    severe_dry_recovery = (
        hot_dry_cooling
        and rh_air < float(getattr(cfg, "tomato_safety_v2_hot_dry_recovery_rh", 55.0))
        and temp_air < float(getattr(cfg, "tomato_safety_v2_severe_dry_screen_temp_cap", 27.0))
        and not cooldown_dry_recovery
        and not pre_hot_dry_heat_load
    )
    severe_dry_canopy_reserve = (
        severe_dry_recovery
        and not extreme_dew_risk
        and canopy_dew_margin < float(getattr(cfg, "tomato_safety_v2_severe_dry_canopy_reserve_margin", 4.0))
        and rad_heat_load >= float(getattr(cfg, "tomato_safety_v2_severe_dry_canopy_reserve_rad", 600.0))
    )
    cooldown_canopy_reserve_buffer = (
        cooldown_dry_recovery
        and hot_dry_radiation_relief
        and not hard_dry_risk
        and not extreme_dew_risk
        and canopy_dew_margin < float(getattr(cfg, "tomato_safety_v2_cooldown_canopy_reserve_margin", 3.0))
        and rad_heat_load >= float(getattr(cfg, "tomato_safety_v2_cooldown_canopy_reserve_rad", 650.0))
        and rh_air >= float(getattr(cfg, "tomato_safety_v2_cooldown_canopy_reserve_rh", 58.0))
        and vpd <= float(getattr(cfg, "tomato_safety_v2_cooldown_canopy_reserve_vpd_cap", 1.65))
    )
    canopy_dew_buffer = bool(canopy_dew_buffer_candidate and (cooldown_dry_recovery or hot_dry_cooling))
    details["canopy_dew_buffer"] = bool(canopy_dew_buffer)
    details["hot_dry_radiation_relief"] = bool(hot_dry_radiation_relief)
    details["pre_hot_dry_heat_load"] = bool(pre_hot_dry_heat_load)
    details["pre_hot_dry_phase"] = bool(pre_hot_dry_phase)
    details["cooldown_dry_recovery"] = bool(cooldown_dry_recovery)
    details["cooldown_canopy_reserve_buffer"] = bool(cooldown_canopy_reserve_buffer)
    details["severe_dry_recovery"] = bool(severe_dry_recovery)
    details["severe_dry_canopy_reserve"] = bool(severe_dry_canopy_reserve)
    details["canopy_dew_target_preempt"] = bool(canopy_dew_target_preempt)
    details["hot_dry_ultra_solver_sensitive"] = bool(hot_dry_ultra_solver_sensitive)
    details["hot_dry_extreme_solver_sensitive"] = bool(hot_dry_extreme_solver_sensitive)
    details["soft_hot_dry_override"] = bool(soft_hot_dry_override)

    if hot_override:
        reasons.append(hot_override_reason)
        if dry_risk:
            reasons.append("dry_vpd_heat_deprioritized")
        if hard_dry_risk:
            reasons.append("hard_dry_vpd_risk")
        if extreme_dew_risk:
            reasons.append("extreme_dew_hot_override")
        vent_floor = float(getattr(cfg, "tomato_safety_v2_hot_vent_floor", 0.85))
        vent_cap = vent_floor
        temp_out = float(getattr(state, "temp_out", temp_air))
        if dry_risk and rad_heat_load >= hot_rad_threshold:
            soft_vent_floor = float(getattr(cfg, "tomato_safety_v2_near_hot_dry_vent_floor", 0.85))
            if hot_override_reason == "hot_override_hold" and soft_hot_dry_override:
                soft_vent_floor = float(getattr(cfg, "tomato_safety_v2_hot_hold_dry_vent_floor", 0.72))
            vent_floor = max(
                vent_floor,
                soft_vent_floor
                if soft_hot_dry_override
                else float(getattr(cfg, "tomato_safety_v2_hot_dry_override_vent_floor", 0.90)),
            )
            vent_cap = vent_floor
        if hot_dry_extreme_solver_sensitive:
            reasons.append("hot_dry_extreme_solver_relief")
            vent_floor = max(vent_floor, extreme_solver_vent_floor)
            vent_cap = vent_floor
        extreme_hot_vent_temp = float(getattr(cfg, "tomato_safety_v2_extreme_hot_vent_temp_threshold", 35.0))
        if temp_air >= extreme_hot_vent_temp or (
            extreme_dew_risk
            and rh_air >= float(getattr(cfg, "tomato_safety_v2_extreme_dew_rh", 94.0))
        ):
            vent_floor = max(vent_floor, float(getattr(cfg, "tomato_safety_v2_extreme_hot_vent_floor", 0.95)))
            vent_cap = vent_floor
        shaped[3] = min(max(float(shaped[3]), min(vent_floor, 1.0)), min(vent_cap, 1.0))
        shaped[0] = 0.0
        shaped[1] = 0.0
        effective_hot_screen_floor = hot_screen_floor
        effective_hot_screen_cap = hot_screen_cap
        if soft_hot_dry_override:
            effective_hot_screen_floor = max(
                effective_hot_screen_floor,
                float(getattr(cfg, "tomato_safety_v2_soft_hot_dry_screen_floor", 0.35)),
            )
            effective_hot_screen_cap = max(
                effective_hot_screen_cap,
                float(getattr(cfg, "tomato_safety_v2_soft_hot_dry_screen_cap", 0.60)),
            )
        shaped[2] = min(max(float(shaped[2]), effective_hot_screen_floor), effective_hot_screen_cap)
        if hot_dry_ultra_solver_sensitive:
            reasons.append("hot_dry_solver_screen_buffer")
            shaped[2] = min(max(float(shaped[2]), solver_screen_floor), hot_screen_cap)
        shaped[4] = 0.0
        if rad_heat_load >= 200.0:
            shade_floor = hot_shade_floor
            shade_cap = hot_shade_cap
            if dry_risk:
                reasons.append("hot_dry_shade_limited_for_stability")
                hot_dry_shade_floor = float(getattr(cfg, "tomato_safety_v2_hot_dry_override_shade_floor", 0.65))
                hot_dry_shade_cap = float(getattr(cfg, "tomato_safety_v2_hot_dry_override_shade_cap", 0.75))
                if soft_hot_dry_override:
                    hot_dry_shade_cap = max(hot_dry_shade_cap, hot_shade_cap)
                if rad_heat_load >= hot_rad_threshold:
                    shade_floor = max(shade_floor, hot_dry_shade_floor)
                    shade_cap = min(shade_cap, hot_dry_shade_cap)
                else:
                    shade_floor = min(shade_floor, hot_dry_shade_floor)
                    shade_cap = min(shade_cap, hot_dry_shade_cap)
                shade_floor = min(shade_floor, shade_cap)
            shaped[5] = min(max(float(shaped[5]), shade_floor), shade_cap)
            if hot_dry_extreme_solver_sensitive:
                shaped[5] = min(max(float(shaped[5]), extreme_solver_shade), extreme_solver_shade)
        raw_extreme_hot_dry = (
            dry_risk
            and temp_air >= max(hot_temp_threshold + 1.0, 33.0)
            and rad_heat_load >= hot_rad_threshold
            and (
                vpd >= float(getattr(cfg, "tomato_safety_v2_extreme_hot_dry_vpd_threshold", 3.20))
                or rh_air <= float(getattr(cfg, "tomato_safety_v2_extreme_hot_dry_rh_threshold", 38.0))
            )
        )
        very_hot_dry_hold = (
            dry_risk
            and temp_air >= float(getattr(cfg, "tomato_safety_v2_extreme_hot_dry_hold_temp_threshold", 36.0))
            and rad_heat_load >= hot_rad_threshold
            and vpd >= float(getattr(cfg, "tomato_safety_v2_extreme_hot_dry_hold_vpd_threshold", 2.80))
        )
        extreme_hot_dry = raw_extreme_hot_dry or very_hot_dry_hold
        if extreme_hot_dry:
            reasons.append("extreme_hot_dry_stability_guard")
            if very_hot_dry_hold and not raw_extreme_hot_dry:
                reasons.append("extreme_hot_dry_temperature_hold")
            shaped[2] = min(
                max(
                    float(shaped[2]),
                    float(getattr(cfg, "tomato_safety_v2_extreme_hot_dry_screen_floor", 0.90)),
                ),
                float(getattr(cfg, "tomato_safety_v2_extreme_hot_dry_screen_cap", 1.00)),
            )
            shaped[3] = max(
                float(shaped[3]),
                float(getattr(cfg, "tomato_safety_v2_extreme_hot_dry_vent_floor", 0.95)),
            )
            extreme_shade_floor = float(
                getattr(cfg, "tomato_safety_v2_extreme_hot_dry_shade_floor", 0.50)
            )
            extreme_shade_cap = float(
                getattr(cfg, "tomato_safety_v2_extreme_hot_dry_shade_cap", 0.50)
            )
            extreme_shade_floor = min(extreme_shade_floor, extreme_shade_cap)
            shaped[5] = min(max(float(shaped[5]), extreme_shade_floor), extreme_shade_cap)

    if canopy_dew_relief and not hot_override:
        reasons.append("canopy_dew_relief_guard")
        if dry_risk:
            reasons.append("dry_vpd_deprioritized_for_canopy_dew")
        shaped[0] = 0.0
        shaped[1] = 0.0
        shaped[4] = 0.0
        if canopy_dew_temperate_buffer:
            reasons.append("canopy_dew_temperature_buffer")
            screen_cap = float(getattr(cfg, "tomato_safety_v2_canopy_dew_temperate_screen_cap", 0.75))
            vent_floor = float(getattr(cfg, "tomato_safety_v2_canopy_dew_temperate_vent_floor", 0.20))
        else:
            screen_cap = float(getattr(cfg, "tomato_safety_v2_canopy_dew_screen_cap", 0.35))
            vent_floor = float(getattr(cfg, "tomato_safety_v2_canopy_dew_vent_floor", 0.42))
        shaped[2] = min(float(shaped[2]), screen_cap)
        shaped[3] = max(float(shaped[3]), vent_floor)
        if glob_rad > 180.0 or temp_air > 26.0:
            canopy_shade_floor = float(getattr(cfg, "tomato_safety_v2_canopy_dew_shade_floor", 0.50))
            canopy_shade_cap = float(getattr(cfg, "tomato_safety_v2_canopy_dew_shade_cap", 0.50))
            canopy_shade_floor = min(canopy_shade_floor, canopy_shade_cap)
            shaped[5] = min(max(float(shaped[5]), canopy_shade_floor), canopy_shade_cap)

    if hot_dry_cooling:
        reasons.append("hot_dry_cooling_guard")
        if pre_hot_dry_phase:
            reasons.append("pre_hot_dry_heat_relief")
            reasons.append("pre_hot_dry_vent_budget")
        if cooldown_dry_recovery:
            reasons.append("cooldown_dry_recovery")
            if hot_dry_radiation_relief:
                reasons.append("hot_dry_radiation_relief")
        if severe_dry_recovery:
            reasons.append("severe_dry_recovery")
        if cooldown_dry_recovery:
            vent_floor = 0.0
        elif pre_hot_dry_phase:
            vent_floor = float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_vent_floor", 0.60))
            if temp_air >= float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_warm_vent_temp", 29.0)):
                vent_floor = max(
                    vent_floor,
                    float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_warm_vent_floor", 0.60)),
                )
            if temp_air >= float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_hot_vent_temp", 30.5)):
                vent_floor = max(
                    vent_floor,
                    float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_hot_vent_floor", 0.72)),
                )
        elif hard_dry_risk or vpd > 1.8 or temp_air >= 30.0:
            vent_floor = 0.78
        else:
            vent_floor = 0.60
        vent_cap = 1.0
        if pre_hot_dry_phase:
            vent_cap = max(
                vent_floor,
                float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_vent_cap", 0.82)),
            )
            if temp_air >= hot_rapid_rise_temp or rapid_temp_rise:
                vent_cap = min(
                    float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_rapid_vent_cap", 0.82)),
                    max(vent_cap, vent_floor + 0.10),
                )
        elif cooldown_dry_recovery:
            vent_cap = float(getattr(cfg, "tomato_safety_v2_cooldown_dry_vent_cap", 0.28))
        if cooldown_dry_recovery:
            shaped[3] = min(float(shaped[3]), vent_cap)
        else:
            shaped[3] = min(max(float(shaped[3]), vent_floor), vent_cap)
            shaped[3] = min(float(shaped[3]), 1.00)
        shaped[0] = 0.0
        shaped[1] = 0.0
        shaped[4] = 0.0
        if severe_dry_recovery:
            if severe_dry_canopy_reserve:
                reasons.append("severe_dry_canopy_reserve")
                reserve_screen_floor = float(
                    getattr(cfg, "tomato_safety_v2_severe_dry_canopy_reserve_screen_floor", 0.70)
                )
                reserve_screen_cap = float(
                    getattr(cfg, "tomato_safety_v2_severe_dry_canopy_reserve_screen_cap", 0.82)
                )
                reserve_screen_floor = min(reserve_screen_floor, reserve_screen_cap)
                shaped[2] = min(max(float(shaped[2]), reserve_screen_floor), reserve_screen_cap)
                reserve_shade = float(getattr(cfg, "tomato_safety_v2_severe_dry_canopy_reserve_shade", 0.50))
                shaped[5] = min(max(float(shaped[5]), reserve_shade), reserve_shade)
            else:
                shaped[2] = max(float(shaped[2]), 1.00)
                shaped[5] = min(float(shaped[5]), 0.0 if hard_dry_risk else 0.15)
        elif cooldown_dry_recovery:
            cooldown_screen_floor = float(getattr(cfg, "tomato_safety_v2_cooldown_dry_screen_floor", 0.82))
            cooldown_screen_cap = float(getattr(cfg, "tomato_safety_v2_cooldown_dry_screen_cap", 0.85))
            cooldown_screen_floor = min(cooldown_screen_floor, cooldown_screen_cap)
            shaped[2] = min(max(float(shaped[2]), cooldown_screen_floor), cooldown_screen_cap)
            if rad_heat_load > 180.0:
                shaped[5] = max(
                    float(shaped[5]),
                    float(getattr(cfg, "tomato_safety_v2_cooldown_dry_shade_floor", 0.45)),
                )
            if cooldown_canopy_reserve_buffer:
                reasons.append("cooldown_canopy_reserve_buffer")
                shaped[2] = min(
                    float(shaped[2]),
                    float(getattr(cfg, "tomato_safety_v2_cooldown_canopy_reserve_screen_cap", 0.60)),
                )
                shaped[3] = max(
                    float(shaped[3]),
                    float(getattr(cfg, "tomato_safety_v2_cooldown_canopy_reserve_vent_floor", 0.42)),
                )
                reserve_shade = float(getattr(cfg, "tomato_safety_v2_cooldown_canopy_reserve_shade", 0.50))
                shaped[5] = min(max(float(shaped[5]), reserve_shade), reserve_shade)
        elif hot_dry_radiation_relief:
            reasons.append("hot_dry_radiation_relief")
            screen_floor = float(getattr(cfg, "tomato_safety_v2_hot_dry_screen_floor", 0.45))
            screen_cap = float(getattr(cfg, "tomato_safety_v2_hot_dry_screen_cap", 0.75))
            if rh_air < float(getattr(cfg, "tomato_safety_v2_dry_rh_hard", 50.0)):
                screen_floor = max(screen_floor, 0.70)
                screen_cap = max(screen_cap, 0.85)
            elif rh_air < float(getattr(cfg, "tomato_safety_v2_dry_rh_on", 55.0)):
                screen_floor = max(screen_floor, 0.60)
                screen_cap = max(screen_cap, 0.80)
            elif (
                rh_air >= float(getattr(cfg, "tomato_safety_v2_dry_rh_on", 55.0))
                and not pre_hot_dry_phase
                and not pre_hot_dry_heat_load
            ):
                screen_cap = min(screen_cap, 0.65)
            shade_floor = float(getattr(cfg, "tomato_safety_v2_hot_dry_shade_floor", 0.50))
            shade_cap = 1.0
            if pre_hot_dry_phase:
                screen_cap = min(
                    screen_cap,
                    float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_screen_cap", 0.82)),
                )
                screen_floor = max(
                    screen_floor,
                    float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_screen_floor", 0.70)),
                )
                screen_floor = min(screen_floor, screen_cap)
                shade_floor = max(
                    shade_floor,
                    float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_shade_floor", 0.82)),
                )
                shade_cap = float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_shade_cap", 0.92))
                shade_floor = min(shade_floor, shade_cap)
            elif pre_hot_dry_heat_load:
                screen_cap = min(
                    screen_cap,
                    float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_screen_cap", 0.82)),
                )
                screen_floor = max(
                    screen_floor,
                    float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_screen_floor", 0.70)),
                )
                screen_floor = min(screen_floor, screen_cap)
                shade_floor = max(
                    shade_floor,
                    float(getattr(cfg, "tomato_safety_v2_pre_hot_dry_shade_floor", 0.82)),
                )
            shaped[2] = min(max(float(shaped[2]), screen_floor), screen_cap)
            shaped[5] = min(max(float(shaped[5]), shade_floor), shade_cap)
        else:
            shaped[2] = max(float(shaped[2]), 1.00)
            shaped[5] = min(float(shaped[5]), 0.0 if hard_dry_risk else 0.15)
        if hard_dry_risk:
            reasons.append("hard_dry_vpd_guard")

    if canopy_dew_buffer and not hot_override:
        reasons.append("canopy_dew_hot_dry_conflict_buffer")
        shaped[0] = 0.0
        shaped[1] = 0.0
        shaped[4] = 0.0
        shaped[2] = min(
            float(shaped[2]),
            float(getattr(cfg, "tomato_safety_v2_canopy_dew_buffer_screen_cap", 0.60)),
        )
        shaped[3] = max(
            float(shaped[3]),
            float(getattr(cfg, "tomato_safety_v2_canopy_dew_buffer_vent_floor", 0.42)),
        )
        if glob_rad > 180.0 or temp_air > 26.0:
            buffer_shade_floor = float(getattr(cfg, "tomato_safety_v2_canopy_dew_buffer_shade_floor", 0.50))
            buffer_shade_cap = float(getattr(cfg, "tomato_safety_v2_canopy_dew_buffer_shade_cap", 0.50))
            buffer_shade_floor = min(buffer_shade_floor, buffer_shade_cap)
            shaped[5] = min(max(float(shaped[5]), buffer_shade_floor), buffer_shade_cap)

    if dry_risk and not hot_override and not hot_dry_cooling and not canopy_dew_relief and not canopy_dew_buffer:
        reasons.append("dry_vpd_guard")
        if temp_air >= 28.0:
            vent_cap = 0.24 if hard_dry_risk else 0.30
        elif temp_air >= 24.0:
            vent_cap = 0.14 if hard_dry_risk else 0.18
        else:
            vent_cap = 0.08 if hard_dry_risk else 0.10
        shaped[3] = min(float(shaped[3]), vent_cap)
        shaped[1] = 0.0
        shaped[4] = 0.0
        if temp_air >= 12.0:
            shaped[0] = min(float(shaped[0]), 0.02 if hard_dry_risk else 0.05)
        if temp_air < 20.0:
            shaped[2] = max(float(shaped[2]), 0.80 if hard_dry_risk else 0.70)
        elif (
            hard_dry_risk
            and temp_air < float(getattr(cfg, "tomato_safety_v2_warm_hard_dry_screen_temp_cap", 25.5))
            and rad_heat_load >= float(getattr(cfg, "tomato_safety_v2_warm_hard_dry_screen_rad_threshold", 350.0))
        ):
            shaped[2] = max(
                float(shaped[2]),
                float(getattr(cfg, "tomato_safety_v2_warm_hard_dry_screen_floor", 0.75)),
            )
        if glob_rad > 180.0 or temp_air > 24.0 or hard_dry_risk:
            shaped[5] = max(float(shaped[5]), 0.65 if hard_dry_risk else 0.50)
        if hard_dry_risk:
            reasons.append("hard_dry_vpd_guard")

    if target_mismatch and not hot_override and not hot_dry_cooling and not canopy_dew_relief and not canopy_dew_buffer:
        reasons.append("target_rh_action_mismatch")
        if canopy_dew_target_preempt:
            reasons.append("canopy_dew_target_preempt")
            preempt_vent_floor = float(getattr(cfg, "tomato_safety_v2_canopy_dew_preempt_vent_floor", 0.18))
            preempt_vent_cap = float(getattr(cfg, "tomato_safety_v2_canopy_dew_preempt_vent_cap", 0.24))
            preempt_vent_cap = max(preempt_vent_cap, preempt_vent_floor)
            shaped[3] = min(max(float(shaped[3]), preempt_vent_floor), preempt_vent_cap)
            shaped[2] = min(
                float(shaped[2]),
                float(getattr(cfg, "tomato_safety_v2_canopy_dew_preempt_screen_cap", 0.75)),
            )
        else:
            shaped[3] = min(float(shaped[3]), 0.14 if temp_air >= 24.0 else 0.08)
        shaped[1] = 0.0
        shaped[4] = 0.0
        if temp_air < 20.0:
            shaped[2] = max(float(shaped[2]), 0.70)
        if glob_rad > 200.0:
            shaped[5] = max(float(shaped[5]), 0.50)
        if canopy_dew_target_preempt and temp_air <= float(
            getattr(cfg, "tomato_safety_v2_canopy_dew_preempt_shade_temp_cap", 24.5)
        ):
            shaped[5] = min(
                float(shaped[5]),
                float(getattr(cfg, "tomato_safety_v2_canopy_dew_preempt_shade_cap", 0.35)),
            )

    if temp_air < float(getattr(cfg, "tomato_safety_v2_low_temp_threshold", 15.5)) and not extreme_dew_risk:
        reasons.append("cold_buffer_guard")
        if temp_air < 12.0:
            shaped[0] = max(float(shaped[0]), 0.90)
            shaped[2] = max(float(shaped[2]), 0.90)
            shaped[3] = min(float(shaped[3]), 0.08)
        else:
            shaped[0] = max(float(shaped[0]), 0.16 if is_night else 0.08)
            shaped[2] = max(float(shaped[2]), 0.75 if is_night else 0.65)
            shaped[3] = min(float(shaped[3]), 0.18 if rh_air >= 88.0 else 0.12)
        shaped[1] = 0.0
        shaped[4] = 0.0

    shaped = np.clip(shaped, 0.0, 1.0)
    details["applied"] = bool(np.max(np.abs(shaped - before)) > 1e-6)
    details["after"] = shaped.tolist()
    details["vent_before"] = float(before[3]) if len(before) > 3 else 0.0
    details["vent_after"] = float(shaped[3]) if len(shaped) > 3 else 0.0
    details["heat_before"] = float(before[0]) if len(before) > 0 else 0.0
    details["heat_after"] = float(shaped[0]) if len(shaped) > 0 else 0.0
    details["screen_before"] = float(before[2]) if len(before) > 2 else 0.0
    details["screen_after"] = float(shaped[2]) if len(shaped) > 2 else 0.0
    details["shade_before"] = float(before[5]) if len(before) > 5 else 0.0
    details["shade_after"] = float(shaped[5]) if len(shaped) > 5 else 0.0
    return shaped, details


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
        self.last_tomato_safety_v2: Dict[str, Any] = {"enabled": bool(getattr(self.config, "tomato_safety_v2_enabled", False))}
        self.last_tomato_safety_v2_suppressed_replan: Dict[str, Any] = {"applied": False}
        self.tomato_safety_v2_suppressed_replan_steps: int = 0
        self.last_plan_cache_event: Dict[str, Any] = {"mode": str(getattr(self.config, "plan_cache_mode", "off"))}
        self.last_transition_gate: Dict[str, Any] = {
            "enabled": bool(getattr(self.config, "transition_gate_enabled", False)),
            "applied": False,
            "bypassed": False,
        }
        self.last_cstcc_shadow: Dict[str, Any] = {
            "enabled": bool(getattr(self.config, "cstcc_shadow_enabled", False)),
            "shadow_only": True,
            "final_action_changed": False,
            "final_action_invariant_verified": True,
        }
        self.last_profile_feasibility_gate: Dict[str, Any] = {
            "enabled": bool(getattr(self.config, "profile_feasibility_gate_enabled", False)),
            "applied": False,
            "hard_safety_veto_count": 0,
        }
        self.last_profile_template_patch: Dict[str, Any] = {
            "enabled": bool(getattr(self.config, "profile_template_patch_enabled", False)),
            "applied": False,
            "fallback_veto_applied": False,
            "fallback_veto_no_alternative": False,
            "recovery_anchor_enabled": bool(getattr(self.config, "recovery_anchor_enabled", False)),
            "recovery_anchor_applied": False,
        }
        self.last_structured_anchor: Dict[str, Any] = {
            "enabled": bool(getattr(self.config, "structured_anchor_parser_enabled", False)),
            "shadow_only": bool(getattr(self.config, "structured_anchor_shadow_only", True)),
            "attempted": False,
            "valid": False,
            "clean_planning_evidence": False,
            "empty": True,
        }
        self.last_structured_anchor_profile_bridge: Dict[str, Any] = {
            "enabled": bool(getattr(self.config, "structured_anchor_profile_bridge_enabled", False)),
            "shadow_only": bool(getattr(self.config, "structured_anchor_profile_bridge_shadow_only", True)),
            "applied": False,
            "bridgeable": False,
        }
        self._transition_gate_previous_action: Optional[np.ndarray] = None
        reversal_window = max(1, int(getattr(self.config, "transition_gate_reversal_window_steps", 6) or 6))
        self._transition_gate_sign_history: Dict[str, deque] = {
            name: deque(maxlen=reversal_window) for name in ACTION_NAMES
        }
        self._transition_gate_soft_limits: Optional[Dict[str, Dict[str, float]]] = None
        self._transition_gate_soft_limit_source: str = ""
        self._transition_gate_soft_limit_error: str = ""
        cstcc_history_steps = max(1, int(getattr(self.config, "cstcc_shadow_history_steps", 60) or 60))
        self._cstcc_state_history: deque = deque(maxlen=cstcc_history_steps)
        self._cstcc_action_history: deque = deque(maxlen=cstcc_history_steps)
        self._cstcc_weather_history: deque = deque(maxlen=cstcc_history_steps)
        self._cstcc_active_regime: str = "NORMAL_BALANCED"
        self._cstcc_regime_age_steps: int = 0
        self._cstcc_previous_plan_sequence: List[Dict[str, float]] = []
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
        self._canopy_margin_history: deque = deque(maxlen=8)
        self._last_canopy_history_timestep: Optional[int] = None
        
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
            "structured_anchor": to_jsonable(dict(getattr(self, "last_structured_anchor", {}) or {})),
        }

    def _structured_anchor_empty_record(self, *, attempted: bool = False) -> Dict[str, Any]:
        enabled = bool(getattr(self.config, "structured_anchor_parser_enabled", False))
        return {
            "enabled": enabled,
            "shadow_only": bool(getattr(self.config, "structured_anchor_shadow_only", True)),
            "attempted": bool(attempted),
            "valid": False,
            "clean_planning_evidence": False,
            "empty": True,
            "errors": ["disabled"] if not enabled else ["not_attempted"],
            "missing_fields": [],
            "unexpected_fields": [],
            "final_control_field_hits": [],
            "final_control_field_leak_count": 0,
            "legacy_tool_action_present": False,
            "shadow_plan": {},
        }

    def _store_structured_anchor_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        stored = dict(record)
        stored["enabled"] = bool(getattr(self.config, "structured_anchor_parser_enabled", False))
        stored["shadow_only"] = bool(getattr(self.config, "structured_anchor_shadow_only", True))
        self.last_structured_anchor = stored
        return stored

    def _restore_cached_structured_anchor(self, entry: Mapping[str, Any]) -> bool:
        record = entry.get("structured_anchor", {}) if isinstance(entry, Mapping) else {}
        if not isinstance(record, Mapping):
            self._store_structured_anchor_record(self._structured_anchor_empty_record(attempted=False))
            return False
        restored = self._store_structured_anchor_record(record)
        return bool(restored.get("valid", False))

    def _parse_structured_anchor_output(
        self,
        llm_output: Any,
        *,
        attempt: int,
        prompt_hash: str,
        planning_horizon: int,
        legacy_tool_action_present: bool,
        retry_attempt: bool = False,
        retry_source_errors: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        parsed = parse_structured_anchor(llm_output)
        errors = list(parsed.get("errors", []) or [])
        final_hits = list(parsed.get("final_control_field_hits", []) or [])
        record = {
            "enabled": bool(getattr(self.config, "structured_anchor_parser_enabled", False)),
            "shadow_only": bool(getattr(self.config, "structured_anchor_shadow_only", True)),
            "attempted": True,
            "attempt": int(attempt),
            "prompt_hash": str(prompt_hash),
            "planning_horizon": int(planning_horizon),
            "valid": bool(parsed.get("valid", False)),
            "clean_planning_evidence": bool(parsed.get("clean_planning_evidence", False)),
            "empty": bool("empty_anchor" in errors),
            "errors": errors,
            "missing_fields": list(parsed.get("missing_fields", []) or []),
            "unexpected_fields": list(parsed.get("unexpected_fields", []) or []),
            "final_control_field_hits": final_hits,
            "final_control_field_leak_count": int(len(final_hits)),
            "wrapper_used": bool(parsed.get("wrapper_used", False)),
            "raw_anchor_keys": list(parsed.get("raw_anchor_keys", []) or []),
            "legacy_tool_action_present": bool(legacy_tool_action_present),
            "retry_provenance": {
                "retry_attempt": bool(retry_attempt),
                "retry_source_errors": list(retry_source_errors or []),
                "retry_invalid_or_empty_enabled": bool(
                    getattr(self.config, "structured_anchor_retry_invalid_or_empty_enabled", False)
                ),
                "retry_max_attempts": int(getattr(self.config, "structured_anchor_retry_max_attempts", 1) or 1),
                "compact_json_prompt_enabled": bool(
                    getattr(self.config, "structured_anchor_compact_json_prompt_enabled", False)
                ),
            },
            "shadow_plan": dict(parsed.get("shadow_plan", {}) or {}),
        }
        return self._store_structured_anchor_record(record)

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
        if bool(getattr(self.config, "structured_anchor_parser_enabled", False)):
            if status_brief:
                status_block = f"status_brief:{status_brief}\n"
            required = str(getattr(self.config, "structured_anchor_required_fields", "") or "")
            if bool(getattr(self.config, "structured_anchor_compact_json_prompt_enabled", False)):
                list_max = max(1, int(getattr(self.config, "structured_anchor_list_max_items", 3) or 3))
                return (
                    "Return exactly one minified JSON object. No markdown. No prose. No tools. "
                    "No low-level actuator or final-control fields. "
                    "{\"structured_planning_anchor\":{\"profile_intent\":\"<intent>\","
                    "\"target_temp\":<number>,\"target_co2\":<number>,\"target_rh\":<number>,"
                    "\"risk_flags\":[\"<risk>\"],\"forbidden_intents\":[\"<forbidden>\"],"
                    "\"planning_horizon_steps\":<integer>,\"confidence\":<0_to_1>}}. "
                    f"Required fields: {required}. "
                    f"risk_flags max {list_max}; forbidden_intents max {list_max}. "
                    f"Targets are average/band-center values for the next {effective_horizon} steps, "
                    "not actuator commands. "
                    f"mode={mode};day={state.day_of_year:.1f};hour={state.hour_of_day:.1f};"
                    f"is_day={int(is_day)};horizon_steps={effective_horizon};remaining_steps={remaining_steps};"
                    f"T={state.temp_air:.2f};RH={state.rh_air:.2f};CO2={state.co2_air:.2f};"
                    f"Rad={state.glob_rad:.2f};Tout={state.temp_out:.2f};"
                    f"Fruit={state.fruit_weight:.6f};Can24={state.canopy_temp_24h:.2f};"
                    f"TSum={state.temperature_sum:.2f};"
                    f"heat={state.u_boil:.2f};co2={state.u_co2:.2f};screen={state.u_th_scr:.2f};"
                    f"vent={state.u_vent:.2f};lamp={state.u_lamp:.2f};shade={state.u_bl_scr:.2f};"
                    f"suggestions={' | '.join(analysis.get('suggestions', [])[:3]) if analysis.get('suggestions') else 'stable'};"
                    f"{status_block}"
                )
            return (
                "You are a greenhouse high-level planner. Return exactly one JSON object and no tool calls.\n"
                "The object must be wrapped as {\"structured_planning_anchor\": {...}}.\n"
                f"Required fields inside structured_planning_anchor: {required}.\n"
                "Do not output low-level actuator or final-control fields: heating, co2, screen, "
                "ventilation, lighting, shading, blindscreen, u_boil, u_co2, u_th_scr, u_vent, "
                "u_lamp, u_bl_scr, final_control.\n"
                "The anchor describes scenario/intent/targets/risk only; internal controllers will "
                "generate executable actions later.\n"
                f"mode={mode}; day={state.day_of_year:.1f}; hour={state.hour_of_day:.1f}; "
                f"is_day={int(is_day)}; horizon_steps={effective_horizon}; remaining_steps={remaining_steps}\n"
                f"state: temp_air={state.temp_air:.2f}, rh_air={state.rh_air:.2f}, "
                f"co2_air={state.co2_air:.2f}, radiation={state.glob_rad:.2f}, "
                f"temp_out={state.temp_out:.2f}, rh_out={getattr(state, 'rh_out', 70.0):.2f}\n"
                f"crop: fruit_weight={state.fruit_weight:.6f}, canopy_temp_24h={state.canopy_temp_24h:.2f}, "
                f"temperature_sum={state.temperature_sum:.2f}\n"
                f"current_control: heat={state.u_boil:.2f}, co2={state.u_co2:.2f}, "
                f"screen={state.u_th_scr:.2f}, vent={state.u_vent:.2f}, "
                f"lamp={state.u_lamp:.2f}, shading={state.u_bl_scr:.2f}\n"
                f"suggestions: {' | '.join(analysis.get('suggestions', [])[:4]) if analysis.get('suggestions') else 'stable'}\n"
                f"{status_block}"
                "Target semantics: target_temp, target_co2, and target_rh are intended average/band-center "
                "targets for the next planning_horizon_steps, not immediate actuator commands. "
                "Use conservative feasible values when uncertain. Include risk_flags and forbidden_intents "
                "for dew/canopy/RH/VPD/heat/dry/CO2-vent conflicts.\n"
                "Example shape: {\"structured_planning_anchor\":{\"profile_intent\":\"dawn_dew_relief\","
                "\"target_temp\":18.5,\"target_co2\":430,\"target_rh\":76,"
                "\"risk_flags\":[\"dew_risk\"],\"forbidden_intents\":[\"co2_enrichment_high_vent\"],"
                f"\"planning_horizon_steps\":{effective_horizon},\"confidence\":0.7}}"
            )
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

    def _rspc_hot_dry_features(self, state) -> Dict[str, Any]:
        temp_air = float(getattr(state, "temp_air", 20.0))
        rh_air = float(getattr(state, "rh_air", 70.0))
        glob_rad = float(getattr(state, "glob_rad", 0.0))
        forecast_rad_mean = _first_finite_state_attr(state, ["forecast_rad_mean_1h"], glob_rad)
        forecast_rad_peak = _first_finite_state_attr(state, ["forecast_rad_peak_2h"], glob_rad)
        rad_load = max(glob_rad, forecast_rad_mean, forecast_rad_peak)
        vpd = float(calculate_vpd_kpa(temp_air, rh_air))
        dew_margin = min(
            float(getattr(state, "dew_margin_air", 3.0)),
            float(getattr(state, "canopy_dew_margin", 3.0)),
        )
        extreme_dew_risk = (
            rh_air >= float(getattr(self.config, "tomato_safety_v2_extreme_dew_rh", 94.0))
            or dew_margin < float(getattr(self.config, "tomato_safety_v2_extreme_dew_margin", 0.40))
        )
        enabled = bool(getattr(self.config, "rspc_hot_dry_candidates_enabled", False))
        if bool(getattr(self.config, "rspc_hot_dry_require_tomato_v2", True)):
            enabled = enabled and bool(getattr(self.config, "tomato_safety_v2_enabled", False))
        dry_pressure = (
            rh_air <= float(getattr(self.config, "rspc_hot_dry_rh_threshold", 60.0))
            or vpd >= float(getattr(self.config, "rspc_hot_dry_vpd_threshold", 1.55))
        )
        hot_load = (
            temp_air >= float(getattr(self.config, "rspc_hot_dry_temp_threshold", 28.5))
            and rad_load >= float(getattr(self.config, "rspc_hot_dry_rad_threshold", 300.0))
        )
        active = bool(enabled and dry_pressure and hot_load and not extreme_dew_risk)
        hot_danger = temp_air >= float(getattr(self.config, "tomato_safety_v2_hot_temp_threshold", 32.0))
        pre_hot_dry = bool(active and not hot_danger and temp_air < 31.0 and rad_load >= 600.0)
        cooldown_recovery = bool(
            active
            and not hot_danger
            and temp_air < float(getattr(self.config, "tomato_safety_v2_cooldown_dry_temp_threshold", 30.5))
            and rad_load < 600.0
        )
        return {
            "active": active,
            "enabled": bool(enabled),
            "temp_air": float(temp_air),
            "rh_air": float(rh_air),
            "vpd": float(vpd),
            "glob_rad": float(glob_rad),
            "rad_load": float(rad_load),
            "dew_margin": float(dew_margin),
            "extreme_dew_risk": bool(extreme_dew_risk),
            "dry_pressure": bool(dry_pressure),
            "hot_load": bool(hot_load),
            "hot_danger": bool(hot_danger),
            "pre_hot_dry": bool(pre_hot_dry),
            "cooldown_recovery": bool(cooldown_recovery),
        }

    def _rspc_hot_dry_vent_relief_floor(self, state, features: Optional[Dict[str, Any]] = None) -> float:
        info = features if isinstance(features, dict) else self._rspc_hot_dry_features(state)
        temp_air = float(info.get("temp_air", float(getattr(state, "temp_air", 20.0))))
        rad_load = float(info.get("rad_load", float(getattr(state, "glob_rad", 0.0))))
        floor = float(getattr(self.config, "rspc_hot_dry_vent_relief_floor", 0.45))
        if temp_air >= 32.0:
            floor = max(floor, 0.85)
        elif temp_air >= 31.0:
            floor = max(floor, float(getattr(self.config, "rspc_hot_dry_hot_vent_relief_floor", 0.62)))
        elif rad_load >= 650.0:
            floor = max(floor, 0.52)
        if temp_air >= 31.5 and float(getattr(state, "temp_out", temp_air)) < temp_air - 1.0:
            floor = min(0.85, floor + 0.10)
        return float(np.clip(floor, 0.0, 1.0))

    def _build_rspc_hot_dry_candidates(
        self,
        state,
        baseline_control: np.ndarray,
    ) -> List[Tuple[str, np.ndarray, str]]:
        info = self._rspc_hot_dry_features(state)
        if not info["active"]:
            return []

        temp_air = float(info["temp_air"])
        rh_air = float(info["rh_air"])
        vpd = float(info["vpd"])
        rad_load = float(info["rad_load"])
        baseline = np.clip(np.asarray(baseline_control, dtype=np.float32), 0.0, 1.0)
        hot_danger = bool(info.get("hot_danger", False))
        pre_hot_dry = bool(info.get("pre_hot_dry", False))
        cooldown_recovery = bool(info.get("cooldown_recovery", False))
        temp_severity = float(np.clip((temp_air - 28.5) / 4.0, 0.0, 1.0))
        dry_severity = float(np.clip(max(55.0 - rh_air, vpd * 12.0 - 18.0) / 12.0, 0.0, 1.0))
        vent_floor = self._rspc_hot_dry_vent_relief_floor(state, info)
        shade_floor = 0.50 if rad_load < 550.0 else 0.66
        shade_high = 0.72 if rad_load < 700.0 else 0.84
        screen_humid_floor = 0.72 if rh_air < 52.0 else 0.62
        screen_balanced = float(np.clip(0.62 + 0.18 * dry_severity - 0.12 * temp_severity, 0.45, 0.82))

        def candidate(screen: float, vent: float, shade: float) -> np.ndarray:
            shaped = baseline.copy()
            shaped[0] = 0.0
            shaped[1] = 0.0
            shaped[2] = float(np.clip(max(float(baseline[2]), screen), 0.0, 0.85))
            if hot_danger:
                shaped[3] = float(np.clip(max(float(baseline[3]), vent), 0.0, 0.95))
            else:
                shaped[3] = float(np.clip(vent, 0.0, 0.85))
            shaped[4] = 0.0
            shaped[5] = float(np.clip(max(float(baseline[5]), shade), 0.0, 0.85))
            return np.clip(shaped, 0.0, 1.0).astype(np.float32)

        if cooldown_recovery:
            balanced_vent = min(0.34, max(0.22, vent_floor))
            preserving_vent = 0.20
            vent_first = 0.42
            screen_humid_floor = max(screen_humid_floor, 0.82)
        elif pre_hot_dry:
            balanced_vent = min(0.62, max(0.46, vent_floor + 0.08))
            preserving_vent = min(0.48, max(0.34, vent_floor))
            vent_first = min(0.74, max(0.60, vent_floor + 0.16))
            screen_humid_floor = max(screen_humid_floor, 0.74)
            shade_floor = max(shade_floor, 0.78)
            shade_high = max(shade_high, 0.84)
        else:
            balanced_vent = max(vent_floor, 0.46 + 0.16 * temp_severity)
            preserving_vent = max(0.34, min(0.52, vent_floor - 0.08))
            vent_first = max(vent_floor + 0.12, 0.70 if temp_air >= 31.0 else 0.58)
        return [
            (
                "hot_dry_humidity_preserving_cooling",
                candidate(max(screen_humid_floor, screen_balanced), preserving_vent, shade_floor),
                "hot-dry cooling with retained humidity buffer",
            ),
            (
                "hot_dry_balanced_cooling",
                candidate(screen_balanced, balanced_vent, max(shade_floor, 0.56)),
                "balanced hot-dry cooling candidate",
            ),
            (
                "hot_dry_shade_first_cooling",
                candidate(max(0.55, screen_balanced - 0.08), balanced_vent, shade_high),
                "radiation-first hot-dry cooling candidate",
            ),
            (
                "hot_dry_vent_first_cooling",
                candidate(max(0.42, screen_balanced - 0.18), vent_first, shade_high),
                "temperature-first hot-dry cooling candidate",
            ),
        ]

    def _rspc_hot_dry_shadow_features(self, state) -> Dict[str, Any]:
        info = dict(self._rspc_hot_dry_features(state))
        gate_reason = self._rspc_action_safety_gate_reason(state)
        temp_air = float(info.get("temp_air", float(getattr(state, "temp_air", 20.0))))
        rad_load = float(info.get("rad_load", float(getattr(state, "glob_rad", 0.0))))
        hot_danger = temp_air >= float(getattr(self.config, "tomato_safety_v2_hot_temp_threshold", 32.0))
        semantic_active = bool(
            info.get("dry_pressure", False)
            and info.get("hot_load", False)
            and not info.get("extreme_dew_risk", False)
        )
        info.update(
            {
                "semantic_active": bool(semantic_active),
                "shadow_active": bool(semantic_active and gate_reason == "none" and not hot_danger),
                "shadow_gate_reason": str(gate_reason),
                "hot_pressure": bool(info.get("hot_load", False)),
                "strong_rad": bool(rad_load >= 600.0),
                "extreme_dew": bool(info.get("extreme_dew_risk", False)),
                "pre_hot_dry_shadow": bool(semantic_active and not hot_danger and temp_air < 31.0 and rad_load >= 600.0),
                "cooldown_recovery_shadow": bool(
                    semantic_active
                    and not hot_danger
                    and temp_air < float(getattr(self.config, "tomato_safety_v2_cooldown_dry_temp_threshold", 30.5))
                    and rad_load < 600.0
                ),
            }
        )
        return info

    def _build_rspc_hot_dry_shadow_candidates(
        self,
        state,
        baseline_control: np.ndarray,
    ) -> List[Tuple[str, np.ndarray, str]]:
        info = self._rspc_hot_dry_shadow_features(state)
        if not info.get("shadow_active", False):
            return []

        temp_air = float(info["temp_air"])
        rh_air = float(info["rh_air"])
        vpd = float(info["vpd"])
        rad_load = float(info["rad_load"])
        baseline = np.clip(np.asarray(baseline_control, dtype=np.float32), 0.0, 1.0)
        temp_severity = float(np.clip((temp_air - 28.5) / 3.5, 0.0, 1.0))
        dry_severity = float(np.clip(max(55.0 - rh_air, vpd * 12.0 - 18.0) / 12.0, 0.0, 1.0))
        vent_floor = self._rspc_hot_dry_vent_relief_floor(state, info)
        pre_hot_dry = bool(info.get("pre_hot_dry_shadow", False))
        cooldown_recovery = bool(info.get("cooldown_recovery_shadow", False))

        shade_preempt = 0.70 if rad_load < 650.0 else 0.84
        shade_balanced = 0.58 if rad_load < 650.0 else 0.72
        screen_cap = 0.78 if temp_air < 30.0 else 0.68 if temp_air < 31.0 else 0.58
        screen_buffer = float(np.clip(0.58 + 0.18 * dry_severity - 0.10 * temp_severity, 0.42, screen_cap))
        if pre_hot_dry:
            screen_buffer = max(screen_buffer, min(screen_cap, 0.66))
            shade_preempt = max(shade_preempt, 0.84)
            shade_balanced = max(shade_balanced, 0.74)
        if cooldown_recovery:
            screen_buffer = max(screen_buffer, min(screen_cap, 0.72))

        def candidate(screen: float, vent: float, shade: float) -> np.ndarray:
            shaped = baseline.copy()
            shaped[0] = 0.0
            shaped[1] = 0.0
            shaped[2] = float(np.clip(screen, 0.0, screen_cap))
            shaped[3] = float(np.clip(vent, 0.0, 0.88))
            shaped[4] = 0.0
            shaped[5] = float(np.clip(max(float(baseline[5]), shade), 0.0, 0.88))
            return np.clip(shaped, 0.0, 1.0).astype(np.float32)

        preserving_vent = min(0.42, max(0.20, min(vent_floor, 0.36)))
        balanced_vent = min(0.58, max(0.34, vent_floor))
        shade_first_vent = min(0.52, max(0.28, vent_floor - 0.06))
        reserve_vent = max(0.42, min(0.58, vent_floor))
        if temp_air >= 31.0:
            preserving_vent = max(preserving_vent, 0.42)
            balanced_vent = max(balanced_vent, 0.56)
            shade_first_vent = max(shade_first_vent, 0.50)

        candidates: List[Tuple[str, np.ndarray, str]] = [
            (
                "shadow_hot_dry_humidity_retention",
                candidate(screen_buffer, preserving_vent, shade_balanced),
                "shadow-only hot-dry humidity retention candidate",
            ),
            (
                "shadow_hot_dry_shade_preempt",
                candidate(max(0.42, screen_buffer - 0.08), shade_first_vent, shade_preempt),
                "shadow-only radiation preemption candidate",
            ),
            (
                "shadow_hot_dry_balanced_relief",
                candidate(max(0.42, screen_buffer - 0.04), balanced_vent, max(shade_balanced, 0.66)),
                "shadow-only balanced humidity and cooling candidate",
            ),
        ]

        canopy_margin = float(getattr(state, "canopy_dew_margin", 3.0))
        if canopy_margin < 3.0 and rad_load >= 650.0 and rh_air >= 58.0 and vpd <= 1.65:
            candidates.append(
                (
                    "shadow_hot_dry_canopy_reserve",
                    candidate(min(screen_buffer, 0.55), reserve_vent, 0.50),
                    "shadow-only canopy reserve candidate under hot-dry radiation",
                )
            )
        return candidates

    def _score_rspc_hot_dry_candidate(self, state, control: np.ndarray) -> Dict[str, float]:
        info = self._rspc_hot_dry_features(state)
        if not info["active"]:
            return {
                "hot_dry_heat_trap_penalty": 0.0,
                "hot_dry_under_cooling_penalty": 0.0,
                "hot_dry_overvent_dry_penalty": 0.0,
                "hot_dry_mitigation_bonus": 0.0,
            }
        heat, co2_u, screen, vent, lamp, shade = [float(x) for x in np.clip(control[:6], 0.0, 1.0)]
        temp_air = float(info["temp_air"])
        rad_load = float(info["rad_load"])
        rh_air = float(info["rh_air"])
        vpd = float(info["vpd"])
        heat_load = max(temp_air - 28.0, 0.0) + max(rad_load - 300.0, 0.0) / 350.0
        heat_trap = heat_load * (
            1.20 * screen * max(0.60 - shade, 0.0)
            + 2.00 * max(self._rspc_hot_dry_vent_relief_floor(state, info) - vent, 0.0) ** 2
            + 0.35 * max(screen - 0.88, 0.0) ** 2
            + 0.40 * heat
            + 0.25 * lamp
        )
        vent_floor = self._rspc_hot_dry_vent_relief_floor(state, info)
        under_cooling = max(vent_floor - vent, 0.0) ** 2
        if rad_load >= 450.0:
            under_cooling += 0.75 * max(0.50 - shade, 0.0) ** 2
        hot_danger = bool(info.get("hot_danger", False))
        pre_hot_dry = bool(info.get("pre_hot_dry", False))
        cooldown_recovery = bool(info.get("cooldown_recovery", False))
        if hot_danger:
            vent_budget = 1.0
        elif pre_hot_dry:
            vent_budget = float(getattr(self.config, "tomato_safety_v2_pre_hot_dry_vent_cap", 0.82))
        elif cooldown_recovery:
            vent_budget = float(getattr(self.config, "tomato_safety_v2_cooldown_dry_vent_cap", 0.28))
        else:
            vent_budget = 0.62
        dry_severity = float(np.clip(max(55.0 - rh_air, vpd * 12.0 - 18.0) / 12.0, 0.0, 1.0))
        overvent_dry = (1.0 + dry_severity) * max(vent - vent_budget, 0.0) ** 2
        humidity_buffer = 0.22 * min(max(screen, 0.0), 0.85) if rh_air < 52.0 or vpd > 1.80 else 0.0
        vent_mitigation_weight = 0.26 if hot_danger else 0.10
        mitigation = 0.34 * min(max(shade - 0.35, 0.0), 0.45) + vent_mitigation_weight * min(max(vent - 0.30, 0.0), 0.45)
        mitigation += humidity_buffer
        if co2_u <= 0.01 and lamp <= 0.01 and heat <= 0.05:
            mitigation += 0.18
        return {
            "hot_dry_heat_trap_penalty": float(heat_trap),
            "hot_dry_under_cooling_penalty": float(under_cooling),
            "hot_dry_overvent_dry_penalty": float(overvent_dry),
            "hot_dry_mitigation_bonus": float(mitigation),
        }

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

    def _candidate_response_prediction_metadata(
        self,
        state,
        control: np.ndarray,
        response: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Shadow-only next-state risk proxy for candidate diagnostics."""
        control = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        current_control = np.clip(self._state_control_vector(state), 0.0, 1.0)
        temp_air = float(getattr(state, "temp_air", 20.0))
        rh_air = float(getattr(state, "rh_air", 70.0))
        dew_margin_air = float(getattr(state, "dew_margin_air", 3.0))
        canopy_dew_margin = float(getattr(state, "canopy_dew_margin", 3.0))
        canopy_trend = self._update_canopy_proxy_history(state)
        temp_next = float(response.get("temp_next", temp_air) or temp_air)
        rh_next = float(response.get("rh_next", rh_air) or rh_air)
        vpd_next = float(response.get("vpd_next", calculate_vpd_kpa(temp_next, rh_next)) or 0.0)

        heat, _co2_u, screen, vent, _lamp, shade = [float(x) for x in control[:6]]
        screen_delta = screen - float(current_control[2])
        vent_delta = vent - float(current_control[3])
        temp_delta = temp_next - temp_air
        rh_rise = max(rh_next - rh_air, 0.0)
        rh_drop = max(rh_air - rh_next, 0.0)

        dew_margin_next = (
            dew_margin_air
            + 0.06 * temp_delta
            + 0.012 * rh_drop
            + 0.10 * heat
            + 0.04 * shade
            + 0.20 * max(vent_delta, 0.0)
            - 0.025 * rh_rise
            - 0.14 * max(screen_delta, 0.0)
            - 0.10 * max(-vent_delta, 0.0)
        )
        canopy_margin_next = (
            canopy_dew_margin
            + 0.08 * temp_delta
            + 0.010 * rh_drop
            + 0.08 * heat
            + 0.04 * shade
            + 0.24 * max(vent_delta, 0.0)
            - 0.030 * rh_rise
            - 0.32 * max(screen_delta, 0.0)
            - 0.20 * max(-vent_delta, 0.0)
            - 0.05 * max(screen - 0.70, 0.0)
        )
        near_boundary = bool(canopy_dew_margin < 1.5 or canopy_margin_next < 1.0)
        negative_canopy_trend = max(-float(canopy_trend.get("canopy_delta_1", 0.0) or 0.0), 0.0)
        risk_terms = {
            "base": 0.20,
            "screen_delta": 0.35 * max(screen_delta, 0.0),
            "ventilation_delta": 0.25 * max(-vent_delta, 0.0),
            "high_screen": 0.30 if screen >= 0.75 else (0.15 if screen >= 0.55 else 0.0),
            "low_ventilation": 0.25 if vent <= 0.25 else (0.10 if vent <= 0.35 else 0.0),
            "rh_high": 0.15 * max(rh_air - 85.0, 0.0) / 10.0,
            "near_boundary": 0.25 if canopy_dew_margin < 1.5 else (0.10 if canopy_margin_next < 1.5 else 0.0),
            "negative_canopy_trend": min(0.60, 0.30 * negative_canopy_trend),
            "low_air_dew_margin": 0.15 if dew_margin_air < 1.5 else 0.0,
        }
        v2_risk_buffer = float(sum(float(value) for value in risk_terms.values()))
        canopy_margin_next_v2 = float(canopy_margin_next - v2_risk_buffer)
        canopy_warning_v2 = bool(
            canopy_margin_next_v2 < 0.25
            or (
                near_boundary
                and screen >= 0.55
                and vent <= 0.25
                and (vent_delta < -0.02 or negative_canopy_trend > 0.35)
            )
        )
        return {
            "prediction_schema_version": "canopy_proxy_schema_v2",
            "predicted_temp_next": float(temp_next),
            "predicted_rh_next": float(rh_next),
            "predicted_vpd_next": float(vpd_next),
            "predicted_dew_margin_air_next": float(dew_margin_next),
            "predicted_canopy_dew_margin_next": float(canopy_margin_next),
            "predicted_dew_lt0": bool(dew_margin_next < 0.0),
            "predicted_canopy_lt0": bool(canopy_margin_next < 0.0),
            "prediction_horizon_steps": 1,
            "prediction_model": "proxy_v1",
            "prediction_model_v2": "proxy_v2_conservative",
            "predicted_canopy_dew_margin_next_v2": float(canopy_margin_next_v2),
            "predicted_canopy_lt0_v2": bool(canopy_margin_next_v2 < 0.0),
            "predicted_canopy_warning_v2": canopy_warning_v2,
            "canopy_proxy_v2_risk_buffer": float(v2_risk_buffer),
            "canopy_proxy_v2_risk_terms": {key: float(value) for key, value in risk_terms.items()},
            "canopy_proxy_history_available": bool(canopy_trend.get("history_available", False)),
            "canopy_proxy_recent_delta_1": float(canopy_trend.get("canopy_delta_1", 0.0) or 0.0),
            "canopy_proxy_recent_delta_3": float(canopy_trend.get("canopy_delta_3", 0.0) or 0.0),
        }

    def _update_canopy_proxy_history(self, state) -> Dict[str, Any]:
        history = getattr(self, "_canopy_margin_history", None)
        if history is None:
            history = deque(maxlen=8)
            self._canopy_margin_history = history
            self._last_canopy_history_timestep = None
        try:
            timestep = int(getattr(state, "timestep", -1))
        except Exception:
            timestep = -1
        if getattr(self, "_last_canopy_history_timestep", None) != timestep:
            temp_air = float(getattr(state, "temp_air", 20.0))
            rh_air = float(getattr(state, "rh_air", 70.0))
            history.append(
                {
                    "timestep": timestep,
                    "canopy_dew_margin": float(getattr(state, "canopy_dew_margin", 3.0)),
                    "dew_margin_air": float(getattr(state, "dew_margin_air", 3.0)),
                    "rh_air": rh_air,
                    "vpd_air": float(calculate_vpd_kpa(temp_air, rh_air)),
                }
            )
            self._last_canopy_history_timestep = timestep
        items = list(history)
        if len(items) < 2:
            return {"history_available": False, "canopy_delta_1": 0.0, "canopy_delta_3": 0.0}
        current = float(items[-1]["canopy_dew_margin"])
        previous = float(items[-2]["canopy_dew_margin"])
        older = float(items[-4]["canopy_dew_margin"]) if len(items) >= 4 else previous
        return {
            "history_available": True,
            "canopy_delta_1": float(current - previous),
            "canopy_delta_3": float((current - older) / max(1, min(3, len(items) - 1))),
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
        dew_margin_air = float(getattr(state, "dew_margin_air", 3.0))
        canopy_dew_margin = float(getattr(state, "canopy_dew_margin", 3.0))
        dew_margin = min(dew_margin_air, canopy_dew_margin)
        dew_penalty = max(1.2 - dew_margin, 0.0) * (1.0 + max(float(state.rh_air) - 85.0, 0.0) / 10.0)
        canopy_dew_penalty = max(1.2 - canopy_dew_margin, 0.0) * (
            1.0 + max(float(state.rh_air) - 85.0, 0.0) / 10.0
        )

        energy_penalty = 0.70 * control[0] + 0.22 * control[1] + 1.05 * control[4] + 0.05 * control[3]
        conflict_penalty = 0.0
        heat_vent_conflict_penalty = 0.0
        co2_leak_penalty = 0.0
        lamp_risk_penalty = 0.0
        if control[0] > 0.18 and control[3] > 0.22 and float(state.rh_air) < 88.0:
            heat_vent_conflict_penalty = float(control[0] + control[3])
            conflict_penalty += heat_vent_conflict_penalty
        if control[1] > 0.05 and (control[3] > 0.18 or total_rad < float(self.config.fallback_co2_min_rad)):
            co2_leak_penalty = float(1.5 * control[1])
            conflict_penalty += co2_leak_penalty
        if control[4] > 0.05 and (float(state.rh_air) >= 86.0 or control[3] >= 0.30):
            lamp_risk_penalty = float(1.2 * control[4])
            conflict_penalty += lamp_risk_penalty

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

        hot_dry_terms = self._score_rspc_hot_dry_candidate(state, control)
        hot_dry_penalty = (
            hot_dry_terms["hot_dry_heat_trap_penalty"]
            + 18.0 * hot_dry_terms["hot_dry_under_cooling_penalty"]
            + 9.0 * hot_dry_terms["hot_dry_overvent_dry_penalty"]
            - 2.8 * hot_dry_terms["hot_dry_mitigation_bonus"]
        )

        score = (
            self.config.fallback_temp_penalty_weight * temp_penalty
            + self.config.fallback_rh_penalty_weight * rh_penalty
            + float(getattr(self.config, "fallback_dry_penalty_weight", 1.20)) * dry_penalty
            + float(getattr(self.config, "fallback_vpd_high_penalty_weight", 1.00)) * vpd_penalty
            + self.config.fallback_cost_penalty_weight * energy_penalty
            + self.config.fallback_smooth_penalty_weight * smooth_penalty
            + self.config.fallback_conflict_penalty_weight * conflict_penalty
            + float(getattr(self.config, "fallback_dew_penalty_weight", 0.65)) * dew_penalty
            + float(getattr(self.config, "rspc_hot_dry_score_weight", 1.15)) * hot_dry_penalty
            - mitigation_bonus
        )
        details = {
            "score": float(score),
            "total_score": float(score),
            "temp_penalty": float(temp_penalty),
            "rh_penalty": float(rh_penalty),
            "dry_penalty": float(dry_penalty),
            "vpd_penalty": float(vpd_penalty),
            "vpd_high_penalty": float(vpd_penalty),
            "vpd_low_penalty": 0.0,
            "dew_penalty": float(dew_penalty),
            "canopy_dew_penalty": float(canopy_dew_penalty),
            "energy_penalty": float(energy_penalty),
            "conflict_penalty": float(conflict_penalty),
            "heat_vent_conflict_penalty": float(heat_vent_conflict_penalty),
            "co2_leak_penalty": float(co2_leak_penalty),
            "lamp_risk_penalty": float(lamp_risk_penalty),
            "dry_vent_penalty": float(hot_dry_terms["hot_dry_overvent_dry_penalty"]),
            "forecast_risk_penalty": 0.0,
            "smooth_penalty": float(smooth_penalty),
            "mitigation_bonus": float(mitigation_bonus),
            "hot_dry_penalty": float(hot_dry_penalty),
            **hot_dry_terms,
            **response,
            **self._candidate_response_prediction_metadata(state, control, response),
        }
        return float(score), details

    @staticmethod
    def _control_terms(control: np.ndarray) -> Dict[str, float]:
        values = np.clip(np.asarray(control, dtype=np.float32).reshape(-1), 0.0, 1.0)
        names = ("heat", "co2", "screen", "vent", "lamp", "shade")
        return {name: float(values[i]) if i < len(values) else 0.0 for i, name in enumerate(names)}

    @staticmethod
    def _control_from_terms(terms: Mapping[str, Any]) -> Optional[np.ndarray]:
        if not isinstance(terms, Mapping):
            return None
        names = ("heat", "co2", "screen", "vent", "lamp", "shade")
        values: List[float] = []
        try:
            for name in names:
                values.append(float(terms.get(name, 0.0) or 0.0))
        except Exception:
            return None
        return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)

    @staticmethod
    def _cstcc_float(value: Any, default: float = 0.0) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(default)
        if not np.isfinite(number):
            number = float(default)
        return number

    @classmethod
    def _cstcc_action_record(cls, control: Any) -> Dict[str, float]:
        aliases = {
            "u_heating": ("u_heating", "heating", "heat", "u_boil"),
            "u_co2": ("u_co2", "co2"),
            "u_screen": ("u_screen", "screen", "u_th_scr"),
            "u_ventilation": ("u_ventilation", "ventilation", "vent", "u_vent"),
            "u_lighting": ("u_lighting", "lighting", "lamp", "u_lamp"),
            "u_shading": ("u_shading", "shading", "shade", "u_bl_scr"),
        }
        if isinstance(control, Mapping):
            row: Dict[str, float] = {}
            for field, names in aliases.items():
                value = 0.0
                for name in names:
                    if name in control:
                        value = cls._cstcc_float(control.get(name), 0.0)
                        break
                row[field] = float(np.clip(value, 0.0, 1.0))
            return row
        try:
            values = np.asarray(control, dtype=np.float32).reshape(-1)
        except Exception:
            values = np.zeros(len(CSTCC_ACTION_FIELDS), dtype=np.float32)
        return {
            field: float(np.clip(values[idx] if idx < int(values.size) else 0.0, 0.0, 1.0))
            for idx, field in enumerate(CSTCC_ACTION_FIELDS)
        }

    def _cstcc_state_record(self, state: Any) -> Dict[str, Any]:
        temp = self._cstcc_float(getattr(state, "temp_air", 20.0), 20.0)
        rh = self._cstcc_float(getattr(state, "rh_air", 70.0), 70.0)
        vpd = self._cstcc_float(getattr(state, "vpd_air", None), float("nan"))
        if not np.isfinite(vpd):
            vpd = self._cstcc_float(calculate_vpd_kpa(temp, rh), 0.0)
        return {
            "timestep": int(self._cstcc_float(getattr(state, "timestep", 0), 0.0)),
            "day_of_year": int(self._cstcc_float(getattr(state, "day_of_year", 0), 0.0)),
            "hour_of_day": self._cstcc_float(getattr(state, "hour_of_day", 0.0), 0.0),
            "temp_air": temp,
            "rh_air": rh,
            "vpd_air": vpd,
            "co2_air": self._cstcc_float(getattr(state, "co2_air", 400.0), 400.0),
            "pipe_temp": self._cstcc_float(
                getattr(state, "pipe_temp", getattr(state, "t_pipe", getattr(state, "pipe_temperature", temp))),
                temp,
            ),
            "canopy_dew_margin": self._cstcc_float(getattr(state, "canopy_dew_margin", 3.0), 3.0),
            "dew_margin_air": self._cstcc_float(getattr(state, "dew_margin_air", 3.0), 3.0),
            "glob_rad": self._cstcc_float(getattr(state, "glob_rad", getattr(state, "radiation", 0.0)), 0.0),
        }

    def _cstcc_weather_record(self, state: Any) -> Dict[str, Any]:
        return {
            "available": True,
            "timestep": int(self._cstcc_float(getattr(state, "timestep", 0), 0.0)),
            "outdoor_temp": self._cstcc_float(
                getattr(state, "temp_out", getattr(state, "outdoor_temp", getattr(state, "t_out", 20.0))),
                20.0,
            ),
            "outdoor_rh": self._cstcc_float(
                getattr(state, "rh_out", getattr(state, "outdoor_rh", getattr(state, "rh_outside", 70.0))),
                70.0,
            ),
            "radiation": self._cstcc_float(getattr(state, "glob_rad", getattr(state, "radiation", 0.0)), 0.0),
            "co2_outdoor": self._cstcc_float(getattr(state, "co2_out", 400.0), 400.0),
        }

    def _cstcc_semantic_suggestion(
        self,
        state: Any,
        final_action: Mapping[str, Any],
    ) -> SemanticSuggestion:
        temp = self._cstcc_float(getattr(state, "temp_air", 20.0), 20.0)
        rh = self._cstcc_float(getattr(state, "rh_air", 70.0), 70.0)
        vpd = self._cstcc_float(getattr(state, "vpd_air", None), float("nan"))
        if not np.isfinite(vpd):
            vpd = self._cstcc_float(calculate_vpd_kpa(temp, rh), 0.0)
        co2 = self._cstcc_float(getattr(state, "co2_air", 400.0), 400.0)
        dew_margin = min(
            self._cstcc_float(getattr(state, "dew_margin_air", 3.0), 3.0),
            self._cstcc_float(getattr(state, "canopy_dew_margin", 3.0), 3.0),
        )
        vent = self._cstcc_float(final_action.get("u_ventilation", 0.0), 0.0)
        co2_action = self._cstcc_float(final_action.get("u_co2", 0.0), 0.0)

        regime = "NORMAL_BALANCED"
        confidence = 0.35
        weights: Dict[str, float] = {}
        intent = "maintain balanced greenhouse operation"
        risks: List[str] = []
        if temp >= 31.0 and vpd >= 2.4:
            regime = "SOLVER_SENSITIVE_EMERGENCY"
            confidence = 0.75
            weights = {"solver_risk": 0.70, "action_smoothness": 0.60, "vpd_risk": 0.50}
            intent = "avoid abrupt changes in hot dry solver-sensitive conditions"
            risks = ["hot_dry_solver_sensitive_regime", "high_temp", "high_vpd"]
        elif vpd >= 2.0 and rh <= 60.0:
            regime = "HIGH_VPD_DRY_STRESS"
            confidence = min(0.90, 0.55 + 0.12 * max(vpd - 2.0, 0.0) + 0.01 * max(60.0 - rh, 0.0))
            weights = {"vpd_risk": 0.65, "humidity_recovery": 0.45, "action_smoothness": 0.40}
            intent = "reduce dry stress without abrupt ventilation reversal"
            risks = ["high_vpd", "low_rh"]
        elif temp >= 28.5:
            regime = "HEAT_ACCUMULATION"
            confidence = min(0.80, 0.45 + 0.08 * (temp - 28.5))
            weights = {"temperature_tracking": 0.60, "energy": 0.30, "action_smoothness": 0.35}
            intent = "relieve heat accumulation smoothly"
            risks = ["high_temp"]
        elif rh >= 88.0 or dew_margin < 1.0:
            regime = "HUMIDITY_EXCESS_DISEASE_RISK"
            confidence = min(0.85, 0.50 + 0.04 * max(rh - 88.0, 0.0) + 0.10 * max(1.0 - dew_margin, 0.0))
            weights = {"humidity_recovery": 0.65, "rewrite_pressure": 0.40, "action_smoothness": 0.35}
            intent = "reduce humidity and dew risk smoothly"
            risks = ["humidity_excess", "dew_margin_low"]
        elif co2_action >= 0.35 and vent >= 0.45 and co2 <= 700.0:
            regime = "CO2_VENTILATION_CONFLICT"
            confidence = 0.70
            weights = {"energy": 0.35, "action_smoothness": 0.45}
            intent = "avoid wasting CO2 under high ventilation"
            risks = ["co2_ventilation_conflict"]

        return SemanticSuggestion(
            regime=regime,
            regime_confidence=confidence,
            llm_reported_confidence=confidence,
            priority_weight_suggestions=weights,
            control_intent=intent,
            risk_factors=risks,
            uncertain_fields=[],
            suggested_refresh_steps=12,
            source="runtime_rule_semantic_suggestion_v81_1_no_online_llm",
        )

    def _ensure_cstcc_shadow_buffers(self) -> None:
        history_steps = max(1, int(getattr(self.config, "cstcc_shadow_history_steps", 60) or 60))
        if not isinstance(getattr(self, "_cstcc_state_history", None), deque):
            self._cstcc_state_history = deque(maxlen=history_steps)
        if not isinstance(getattr(self, "_cstcc_action_history", None), deque):
            self._cstcc_action_history = deque(maxlen=history_steps)
        if not isinstance(getattr(self, "_cstcc_weather_history", None), deque):
            self._cstcc_weather_history = deque(maxlen=history_steps)
        if not hasattr(self, "_cstcc_previous_plan_sequence"):
            self._cstcc_previous_plan_sequence = []
        if not hasattr(self, "_cstcc_active_regime"):
            self._cstcc_active_regime = "NORMAL_BALANCED"
        if not hasattr(self, "_cstcc_regime_age_steps"):
            self._cstcc_regime_age_steps = 0

    def _cstcc_episode_id(self, state: Any) -> str:
        env_id = str(getattr(self, "env_id", "unknown_env") or "unknown_env")
        day = int(self._cstcc_float(getattr(state, "day_of_year", 0), 0.0))
        return f"{env_id}_day{day}"

    @staticmethod
    def _cstcc_row_json_size_kb(row: Mapping[str, Any]) -> float:
        return float(len(json.dumps(dict(row), ensure_ascii=False, default=str).encode("utf-8")) / 1024.0)

    def _cstcc_shadow_config_payload(self) -> Optional[Dict[str, Any]]:
        config_path = str(getattr(self.config, "cstcc_shadow_config_path", "") or "")
        if not config_path:
            return None
        cached_path = getattr(self, "_cstcc_shadow_loaded_config_path", None)
        cached_payload = getattr(self, "_cstcc_shadow_loaded_config_payload", None)
        if cached_path == config_path and isinstance(cached_payload, dict):
            return dict(cached_payload)
        payload = load_cstcc_config(config_path)
        self._cstcc_shadow_loaded_config_path = config_path
        self._cstcc_shadow_loaded_config_payload = dict(payload)
        return dict(payload)

    def _run_cstcc_shadow_runtime_hook(
        self,
        state: Any,
        final_control: np.ndarray,
        *,
        rule_control: Optional[np.ndarray] = None,
        anchor_control: Optional[np.ndarray] = None,
    ) -> None:
        if not bool(getattr(self.config, "cstcc_shadow_enabled", False)):
            self.last_cstcc_shadow = {
                "enabled": False,
                "shadow_only": True,
                "final_action_changed": False,
                "final_action_invariant_verified": True,
            }
            return

        self._ensure_cstcc_shadow_buffers()
        step = int(self._cstcc_float(getattr(state, "timestep", 0), 0.0))
        episode_id = self._cstcc_episode_id(state)
        before_vector = np.asarray(final_control, dtype=np.float32).reshape(-1).copy()
        before_action = self._cstcc_action_record(before_vector)
        state_record = self._cstcc_state_record(state)
        weather_record = self._cstcc_weather_record(state)
        action_record = dict(before_action)
        state_history = list(self._cstcc_state_history) + [state_record]
        action_history = list(self._cstcc_action_history) + [action_record]
        weather_history = list(self._cstcc_weather_history) + [weather_record]
        semantic_suggestion = self._cstcc_semantic_suggestion(state, before_action)
        log_path = default_audit_jsonl_path(
            episode_id,
            root=str(getattr(self.config, "cstcc_shadow_audit_log_root", "logs/cstcc_shadow") or "logs/cstcc_shadow"),
        )

        started = time.perf_counter()
        audit, failure = safe_run_cstcc_shadow_step(
            state_history=state_history,
            action_history=action_history,
            weather_history=weather_history,
            current_runtime_final_action=dict(before_action),
            active_regime=str(getattr(self, "_cstcc_active_regime", "NORMAL_BALANCED") or "NORMAL_BALANCED"),
            regime_age_steps=int(getattr(self, "_cstcc_regime_age_steps", 0) or 0),
            previous_plan_sequence=list(getattr(self, "_cstcc_previous_plan_sequence", []) or []),
            ppo_action=self._cstcc_action_record(anchor_control) if anchor_control is not None else None,
            rule_action=self._cstcc_action_record(rule_control) if rule_control is not None else None,
            semantic_suggestion=semantic_suggestion,
            config=self._cstcc_shadow_config_payload(),
        )
        latency_ms = float((time.perf_counter() - started) * 1000.0)

        after_action = self._cstcc_action_record(final_control)
        invariant_ok, invariant_diff = verify_final_action_invariant(before_action, after_action)
        if not invariant_ok:
            try:
                np.asarray(final_control, dtype=np.float32).reshape(-1)[: len(before_vector)] = before_vector
            except Exception:
                pass
            if failure is None:
                failure = {
                    "version": str(getattr(self.config, "cstcc_shadow_version", "v81") or "v81"),
                    "audit_success": False,
                    "failure_type": "FinalActionInvariantViolation",
                    "failure_message": "C-STCC runtime hook changed final action before environment step",
                    "online_llm_called": False,
                    "predictive_rollout_executed": False,
                    "selected_action_is_shadow_only": True,
                }
            failure["final_action_changed"] = True
            failure["final_action_invariant_verified"] = False
            failure["final_action_invariant_diff"] = invariant_diff

        if audit is not None and invariant_ok:
            row = compact_audit_record(
                audit,
                step=step,
                episode_id=episode_id,
                save_full_candidates=bool(getattr(self.config, "cstcc_shadow_save_full_candidates", False)),
                save_raw_sequences=bool(getattr(self.config, "cstcc_shadow_save_raw_sequences", False)),
                save_projected_sequences=bool(getattr(self.config, "cstcc_shadow_save_projected_sequences", False)),
            )
            row.update(
                {
                    "enabled": True,
                    "shadow_only": True,
                    "runtime_shadow_version": str(getattr(self.config, "cstcc_shadow_version", "v81") or "v81"),
                    "audit_latency_ms": latency_ms,
                    "final_action_invariant_diff": invariant_diff,
                    "predictive_rollout_executed": False,
                    "real_tomato_safety_projection": False,
                    "audit_log_path": str(log_path),
                }
            )
            row["audit_json_size_kb"] = self._cstcc_row_json_size_kb(row)
            new_regime = str(getattr(audit, "active_regime_after", "NORMAL_BALANCED") or "NORMAL_BALANCED")
            old_regime = str(getattr(self, "_cstcc_active_regime", "NORMAL_BALANCED") or "NORMAL_BALANCED")
            self._cstcc_regime_age_steps = int(getattr(self, "_cstcc_regime_age_steps", 0) or 0) + 1 if new_regime == old_regime else 0
            self._cstcc_active_regime = new_regime
            plan_memory = getattr(audit, "plan_memory_report", {}) or {}
            self._cstcc_previous_plan_sequence = list(plan_memory.get("remaining_sequence", []) or [])
        else:
            row = compact_failure_record(
                failure
                or {
                    "version": str(getattr(self.config, "cstcc_shadow_version", "v81") or "v81"),
                    "audit_success": False,
                    "failure_type": "UnknownCSTCCShadowFailure",
                    "failure_message": "C-STCC shadow hook returned no audit and no failure payload",
                    "online_llm_called": False,
                    "predictive_rollout_executed": False,
                    "selected_action_is_shadow_only": True,
                },
                step=step,
                episode_id=episode_id,
            )
            row.update(
                {
                    "enabled": True,
                    "shadow_only": True,
                    "runtime_shadow_version": str(getattr(self.config, "cstcc_shadow_version", "v81") or "v81"),
                    "audit_latency_ms": latency_ms,
                    "final_action_invariant_diff": invariant_diff,
                    "real_tomato_safety_projection": False,
                    "audit_log_path": str(log_path),
                }
            )
            row["audit_json_size_kb"] = self._cstcc_row_json_size_kb(row)
            self._cstcc_regime_age_steps = int(getattr(self, "_cstcc_regime_age_steps", 0) or 0) + 1

        sampled = should_sample_step(step, float(getattr(self.config, "cstcc_shadow_sample_rate", 1.0) or 1.0))
        row["audit_sampled"] = bool(sampled)
        row["audit_write_success"] = False
        if sampled:
            row_to_write = dict(row)
            row_to_write["audit_write_success"] = True
            row_to_write["audit_json_size_kb"] = self._cstcc_row_json_size_kb(row_to_write)
            try:
                append_jsonl(log_path, row_to_write)
                row.update(row_to_write)
            except Exception as exc:
                row["audit_write_error"] = str(exc)
                if bool(getattr(self.config, "cstcc_shadow_fail_closed", False)):
                    raise
        self._cstcc_state_history.append(state_record)
        self._cstcc_action_history.append(action_record)
        self._cstcc_weather_history.append(weather_record)
        self.last_cstcc_shadow = dict(row)
        if isinstance(getattr(self, "last_rollout_selection", None), dict):
            self.last_rollout_selection["cstcc_shadow"] = dict(row)
        if (not invariant_ok) and bool(getattr(self.config, "cstcc_shadow_assert_final_action_invariant", True)) and bool(
            getattr(self.config, "cstcc_shadow_fail_closed", False)
        ):
            raise RuntimeError("C-STCC v81.1 violated final-action invariant")

    def _canopy_boundary_shadow_record(
        self,
        state,
        before_control: Optional[np.ndarray],
        final_control: np.ndarray,
    ) -> Dict[str, Any]:
        before = (
            np.clip(np.asarray(before_control, dtype=np.float32), 0.0, 1.0)
            if before_control is not None
            else np.clip(self._state_control_vector(state), 0.0, 1.0)
        )
        final = np.clip(np.asarray(final_control, dtype=np.float32), 0.0, 1.0)
        response = self._estimate_candidate_response(state, final)
        prediction = self._candidate_response_prediction_metadata(state, final, response)
        current_canopy = float(getattr(state, "canopy_dew_margin", 3.0))
        predicted_canopy = float(prediction.get("predicted_canopy_dew_margin_next", current_canopy) or current_canopy)
        predicted_canopy_v2 = float(
            prediction.get("predicted_canopy_dew_margin_next_v2", predicted_canopy) or predicted_canopy
        )
        warning_v2 = bool(prediction.get("predicted_canopy_warning_v2", False))
        screen_delta = float(final[2] - before[2])
        vent_delta = float(final[3] - before[3])
        boundary_active = bool(current_canopy < 0.5 or predicted_canopy < 0.25)
        screen_increase = bool(boundary_active and screen_delta > 0.02)
        ventilation_decrease = bool(boundary_active and vent_delta < -0.02)
        high_screen = bool(boundary_active and float(final[2]) >= 0.55)
        low_ventilation = bool(boundary_active and float(final[3]) <= 0.25)
        screen_vent_conflict = bool((screen_increase or high_screen) and (ventilation_decrease or low_ventilation))
        screen_vent_conflict_v2 = bool(
            warning_v2
            and (screen_delta > 0.02 or float(final[2]) >= 0.55)
            and (vent_delta < -0.02 or float(final[3]) <= 0.25)
        )
        reason = "canopy_dew_screen_vent_conflict" if screen_vent_conflict else ""
        reason_v2 = "canopy_dew_lead_time_screen_vent_conflict" if screen_vent_conflict_v2 else ""
        return {
            "enabled": True,
            "shadow_only": True,
            "canopy_dew_boundary_active": boundary_active,
            "current_canopy_dew_margin": float(current_canopy),
            "predicted_canopy_dew_margin_next": float(predicted_canopy),
            "predicted_canopy_dew_margin_next_v2": float(predicted_canopy_v2),
            "predicted_dew_margin_air_next": float(prediction.get("predicted_dew_margin_air_next", 0.0) or 0.0),
            "screen_delta": float(screen_delta),
            "ventilation_delta": float(vent_delta),
            "screen_increase_under_canopy_risk": screen_increase,
            "ventilation_decrease_under_canopy_risk": ventilation_decrease,
            "high_screen_under_canopy_risk": high_screen,
            "low_ventilation_under_canopy_risk": low_ventilation,
            "would_reject_for_canopy_dew_boundary": screen_vent_conflict,
            "boundary_reject_reason": reason,
            "canopy_boundary_warning_v2": warning_v2,
            "would_reject_for_canopy_dew_boundary_v2": screen_vent_conflict_v2,
            "boundary_reject_reason_v2": reason_v2,
            "final_action_predicted_canopy_lt0": bool(prediction.get("predicted_canopy_lt0", False)),
            "final_action_predicted_canopy_lt0_v2": bool(prediction.get("predicted_canopy_lt0_v2", False)),
            "final_action_predicted_canopy_warning_v2": warning_v2,
            "final_action_predicted_dew_lt0": bool(prediction.get("predicted_dew_lt0", False)),
            "final_action_would_fail_canopy_boundary": screen_vent_conflict,
            "final_action_would_fail_canopy_boundary_v2": screen_vent_conflict_v2,
            "final_action_screen_vent_conflict": screen_vent_conflict,
            "final_action_screen_vent_conflict_v2": screen_vent_conflict_v2,
            "final_action_risk_reason": reason,
            "final_action_risk_reason_v2": reason_v2,
            "prediction_model": str(prediction.get("prediction_model", "") or ""),
            "prediction_model_v2": str(prediction.get("prediction_model_v2", "") or ""),
            "canopy_proxy_v2_risk_buffer": float(prediction.get("canopy_proxy_v2_risk_buffer", 0.0) or 0.0),
            "canopy_proxy_v2_risk_terms": dict(prediction.get("canopy_proxy_v2_risk_terms", {}) or {}),
            "canopy_proxy_history_available": bool(prediction.get("canopy_proxy_history_available", False)),
            "before_action": self._control_terms(before),
            "final_action": self._control_terms(final),
        }

    def _store_post_guardrail_final_risk_shadow(
        self,
        state,
        reference_control: Optional[np.ndarray],
        final_control: np.ndarray,
    ) -> Dict[str, Any]:
        """Record shadow-only canopy boundary diagnostics for the final action."""
        boundary_shadow = self._canopy_boundary_shadow_record(
            state,
            reference_control,
            np.asarray(final_control, dtype=np.float32),
        )
        if isinstance(getattr(self, "last_rollout_selection", None), dict):
            self.last_rollout_selection["canopy_boundary_shadow"] = boundary_shadow
            self.last_rollout_selection["post_guardrail_final_risk_shadow"] = {
                "enabled": True,
                "shadow_only": True,
                "final_action_predicted_canopy_lt0": bool(
                    boundary_shadow.get("final_action_predicted_canopy_lt0", False)
                ),
                "final_action_predicted_canopy_lt0_v2": bool(
                    boundary_shadow.get("final_action_predicted_canopy_lt0_v2", False)
                ),
                "final_action_predicted_canopy_warning_v2": bool(
                    boundary_shadow.get("final_action_predicted_canopy_warning_v2", False)
                ),
                "final_action_would_fail_canopy_boundary": bool(
                    boundary_shadow.get("final_action_would_fail_canopy_boundary", False)
                ),
                "final_action_would_fail_canopy_boundary_v2": bool(
                    boundary_shadow.get("final_action_would_fail_canopy_boundary_v2", False)
                ),
                "final_action_screen_vent_conflict": bool(
                    boundary_shadow.get("final_action_screen_vent_conflict", False)
                ),
                "final_action_screen_vent_conflict_v2": bool(
                    boundary_shadow.get("final_action_screen_vent_conflict_v2", False)
                ),
                "final_action_high_screen_under_canopy_risk": bool(
                    boundary_shadow.get("high_screen_under_canopy_risk", False)
                ),
                "final_action_low_ventilation_under_canopy_risk": bool(
                    boundary_shadow.get("low_ventilation_under_canopy_risk", False)
                ),
                "final_action_risk_reason": str(boundary_shadow.get("final_action_risk_reason", "") or ""),
                "final_action_risk_reason_v2": str(boundary_shadow.get("final_action_risk_reason_v2", "") or ""),
                "predicted_canopy_dew_margin_next": float(
                    boundary_shadow.get("predicted_canopy_dew_margin_next", 0.0) or 0.0
                ),
                "predicted_canopy_dew_margin_next_v2": float(
                    boundary_shadow.get("predicted_canopy_dew_margin_next_v2", 0.0) or 0.0
                ),
                "predicted_dew_margin_air_next": float(
                    boundary_shadow.get("predicted_dew_margin_air_next", 0.0) or 0.0
                ),
                "screen_delta": float(boundary_shadow.get("screen_delta", 0.0) or 0.0),
                "ventilation_delta": float(boundary_shadow.get("ventilation_delta", 0.0) or 0.0),
                "prediction_model": str(boundary_shadow.get("prediction_model", "") or ""),
                "prediction_model_v2": str(boundary_shadow.get("prediction_model_v2", "") or ""),
                "canopy_proxy_v2_risk_buffer": float(
                    boundary_shadow.get("canopy_proxy_v2_risk_buffer", 0.0) or 0.0
                ),
            }
        return boundary_shadow

    def _record_post_guardrail_runtime_provenance(
        self,
        state,
        *,
        hook_id: str,
        source_function: str,
        pre_rule_action: Sequence[float] | np.ndarray,
        post_rule_action: Sequence[float] | np.ndarray,
        info: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Store post-guardrail rewrite provenance as metadata only."""
        record = build_post_guardrail_runtime_provenance_record(
            state=state,
            hook_id=hook_id,
            source_function=source_function,
            pre_rule_action=pre_rule_action,
            post_rule_action=post_rule_action,
            info=info,
        )
        if record is None:
            return None
        if isinstance(getattr(self, "last_rollout_selection", None), dict):
            self.last_rollout_selection.setdefault("post_guardrail_runtime_provenance", []).append(record)
        return record

    @staticmethod
    def _score_term_subset(details: Mapping[str, Any]) -> Dict[str, Any]:
        keys = (
            "score",
            "total_score",
            "temp_penalty",
            "rh_penalty",
            "dry_penalty",
            "vpd_penalty",
            "vpd_high_penalty",
            "vpd_low_penalty",
            "dew_penalty",
            "canopy_dew_penalty",
            "energy_penalty",
            "conflict_penalty",
            "heat_vent_conflict_penalty",
            "co2_leak_penalty",
            "lamp_risk_penalty",
            "dry_vent_penalty",
            "forecast_risk_penalty",
            "smooth_penalty",
            "mitigation_bonus",
            "hot_dry_penalty",
            "hot_dry_heat_trap_penalty",
            "hot_dry_under_cooling_penalty",
            "hot_dry_overvent_dry_penalty",
            "hot_dry_mitigation_bonus",
            "temp_next",
            "rh_next",
            "vpd_next",
            "co2_next",
            "total_rad",
            "predicted_temp_next",
            "predicted_rh_next",
            "predicted_vpd_next",
            "predicted_dew_margin_air_next",
            "predicted_canopy_dew_margin_next",
            "predicted_dew_lt0",
            "predicted_canopy_lt0",
            "prediction_horizon_steps",
            "predicted_canopy_dew_margin_next_v2",
            "canopy_proxy_v2_risk_buffer",
            "canopy_proxy_recent_delta_1",
            "canopy_proxy_recent_delta_3",
        )
        out: Dict[str, Any] = {}
        for key in keys:
            try:
                out[key] = float(details.get(key, 0.0) or 0.0)
            except Exception:
                out[key] = 0.0
        for key in (
            "predicted_dew_lt0",
            "predicted_canopy_lt0",
            "predicted_canopy_lt0_v2",
            "predicted_canopy_warning_v2",
            "canopy_proxy_history_available",
        ):
            out[key] = bool(details.get(key, False))
        out["prediction_schema_version"] = str(details.get("prediction_schema_version", "") or "")
        out["prediction_model"] = str(details.get("prediction_model", "") or "")
        out["prediction_model_v2"] = str(details.get("prediction_model_v2", "") or "")
        risk_terms = details.get("canopy_proxy_v2_risk_terms", {})
        if isinstance(risk_terms, Mapping):
            normalized_terms: Dict[str, float] = {}
            for key, value in risk_terms.items():
                try:
                    normalized_terms[str(key)] = float(value or 0.0)
                except Exception:
                    normalized_terms[str(key)] = 0.0
            out["canopy_proxy_v2_risk_terms"] = normalized_terms
        else:
            out["canopy_proxy_v2_risk_terms"] = {}
        return out

    @staticmethod
    def _rspc_controlled_shadow_score(details: Mapping[str, Any], weights: Mapping[str, float]) -> float:
        def term(key: str) -> float:
            try:
                return float(details.get(key, 0.0) or 0.0)
            except Exception:
                return 0.0

        return float(
            1.20 * term("temp_penalty")
            + 1.60 * term("rh_penalty")
            + float(weights.get("dry", 1.20)) * term("dry_penalty")
            + float(weights.get("vpd", 1.00)) * term("vpd_penalty")
            + 0.45 * term("energy_penalty")
            + 0.20 * term("smooth_penalty")
            + 0.75 * term("conflict_penalty")
            + 0.65 * term("dew_penalty")
            + float(weights.get("hot_dry", 1.35)) * term("hot_dry_penalty")
            - term("mitigation_bonus")
        )

    def _evaluate_hot_dry_action_proposer_controlled_replay_shadow(
        self,
        state,
        *,
        selected_name: str,
        selected_post_shape_score: Optional[float],
        selected_post_shape_control: Optional[np.ndarray],
        selected_post_shape_details: Mapping[str, Any],
        post_shape_rows: Sequence[Mapping[str, Any]],
        gate_reason: str,
    ) -> Dict[str, Any]:
        variants = {
            "balanced_hot_dry": {"dry": 1.80, "vpd": 1.60, "hot_dry": 2.00},
            "dry_vpd_x2": {"dry": 2.40, "vpd": 2.00, "hot_dry": 1.35},
        }
        gate = str(gate_reason or "none") or "none"
        safe_hot_dry = bool(self._profile_rspc_safe_hot_dry_state(state))
        proposer_rows = [
            row for row in post_shape_rows if bool(row.get("shadow_proposer", False))
        ]
        selected_control = (
            np.clip(np.asarray(selected_post_shape_control, dtype=np.float32), 0.0, 1.0)
            if selected_post_shape_control is not None
            else None
        )
        base: Dict[str, Any] = {
            "enabled": True,
            "shadow_only": True,
            "safe_hot_dry": safe_hot_dry,
            "safety_gate_reason": gate,
            "selected_name": str(selected_name),
            "selected_score": (
                float(selected_post_shape_score)
                if selected_post_shape_score is not None and np.isfinite(float(selected_post_shape_score))
                else None
            ),
            "selected_action": self._control_terms(selected_control) if selected_control is not None else {},
            "selected_score_terms": self._score_term_subset(selected_post_shape_details),
            "proposer_count": int(len(proposer_rows)),
            "variant_count": int(len(variants)),
            "variants": [],
            "would_apply_shadow": False,
            "best_candidate_name": "",
            "best_variant": "",
            "best_margin": 0.0,
            "best_alignment": "neutral_hold",
            "unsafe_conflict": False,
            "unsafe_preferred": False,
            "unsafe_filtered_count": 0,
        }
        if selected_control is None or selected_post_shape_score is None:
            base["reason"] = "missing_selected_post_shape"
            return base
        if gate != "none":
            base["reason"] = "safety_gate_active"
        elif not safe_hot_dry:
            base["reason"] = "not_safe_hot_dry"
        elif not proposer_rows:
            base["reason"] = "no_hot_dry_proposer"
        else:
            base["reason"] = "no_eligible_proposer"

        variant_results: List[Dict[str, Any]] = []
        best_eligible: Dict[str, Any] = {}
        unsafe_preferred = False
        unsafe_filtered_count = 0
        selected_details = dict(selected_post_shape_details or {})
        for variant_name, weights in variants.items():
            selected_variant_score = self._rspc_controlled_shadow_score(selected_details, weights)
            row_evals: List[Dict[str, Any]] = []
            for row in proposer_rows:
                row_details = row.get("details", {})
                if not isinstance(row_details, Mapping):
                    row_details = {}
                row_score = self._rspc_controlled_shadow_score(row_details, weights)
                margin = float(selected_variant_score - row_score)
                action_delta = dict(row.get("action_delta_terms", {})) if isinstance(row.get("action_delta_terms", {}), Mapping) else {}
                score_delta = dict(row.get("score_delta_terms", {})) if isinstance(row.get("score_delta_terms", {}), Mapping) else {}
                row_control_terms = self._control_terms(np.asarray(row.get("control", np.zeros(6)), dtype=np.float32))
                actuator_unsafe = (
                    float(row_control_terms.get("heat", 0.0)) > 0.05
                    or float(row_control_terms.get("co2", 0.0)) > 0.05
                    or float(row_control_terms.get("lamp", 0.0)) > 0.05
                    or float(score_delta.get("temp", 0.0)) > 1e-6
                    or float(score_delta.get("dew", 0.0)) > 1e-6
                )
                if actuator_unsafe:
                    unsafe_filtered_count += 1
                    row_score += 1000.0
                alignment = self._rspc_action_post_shape_alignment(
                    state,
                    gate_reason=gate,
                    margin=margin,
                    action_delta_terms=action_delta,
                    score_delta_terms=score_delta,
                )
                if margin > 0.05 and actuator_unsafe:
                    alignment = "unsafe_conflict"
                if actuator_unsafe:
                    margin = float(selected_variant_score - row_score)
                row_evals.append(
                    {
                        "name": str(row.get("name", "") or ""),
                        "row": row,
                        "score": float(row_score),
                        "margin": float(margin),
                        "alignment": alignment,
                        "unsafe": bool(actuator_unsafe or (margin > 0.05 and alignment == "unsafe_conflict")),
                        "safety_penalty": bool(actuator_unsafe),
                        "eligible": bool(gate == "none" and safe_hot_dry and margin > 0.05 and alignment == "dry_benefit"),
                    }
                )
            raw_best = min(row_evals, key=lambda item: float(item["score"])) if row_evals else {}
            if (
                raw_best
                and bool(raw_best.get("unsafe", False))
                and float(raw_best.get("margin", 0.0) or 0.0) > 0.05
            ):
                unsafe_preferred = True
            eligible_rows = [item for item in row_evals if bool(item.get("eligible", False))]
            eligible_best = max(
                eligible_rows,
                key=lambda item: float(item.get("margin", 0.0) or 0.0),
            ) if eligible_rows else {}
            if eligible_best and float(eligible_best.get("margin", 0.0) or 0.0) > float(best_eligible.get("margin", 0.0) or 0.0):
                best_eligible = {**eligible_best, "variant": variant_name}
            variant_results.append(
                {
                    "variant": variant_name,
                    "selected_score": float(selected_variant_score),
                    "raw_best_name": str(raw_best.get("name", "") or "") if raw_best else "",
                    "raw_margin": float(raw_best.get("margin", 0.0) or 0.0) if raw_best else 0.0,
                    "raw_alignment": str(raw_best.get("alignment", "neutral_hold") or "neutral_hold") if raw_best else "neutral_hold",
                    "raw_unsafe_conflict": bool(raw_best.get("unsafe", False) and float(raw_best.get("margin", 0.0) or 0.0) > 0.05) if raw_best else False,
                    "raw_safety_penalty": bool(raw_best.get("safety_penalty", False)) if raw_best else False,
                    "eligible_best_name": str(eligible_best.get("name", "") or "") if eligible_best else "",
                    "eligible_margin": float(eligible_best.get("margin", 0.0) or 0.0) if eligible_best else 0.0,
                    "eligible_alignment": str(eligible_best.get("alignment", "neutral_hold") or "neutral_hold") if eligible_best else "neutral_hold",
                    "would_apply_shadow": bool(eligible_best),
                }
            )

        base["variants"] = variant_results
        base["unsafe_preferred"] = bool(unsafe_preferred)
        base["unsafe_filtered_count"] = int(unsafe_filtered_count)
        if unsafe_preferred:
            base["reason"] = "unsafe_preferred_proposer"
        if best_eligible:
            best_row = best_eligible.get("row", {})
            best_control = np.asarray(best_row.get("control", np.zeros(6)), dtype=np.float32)
            best_details = best_row.get("details", {})
            if not isinstance(best_details, Mapping):
                best_details = {}
            base.update(
                {
                    "reason": "controlled_shadow_replay_candidate",
                    "would_apply_shadow": bool(not unsafe_preferred),
                    "best_candidate_name": str(best_eligible.get("name", "") or ""),
                    "best_variant": str(best_eligible.get("variant", "") or ""),
                    "best_score": float(best_eligible.get("score", 0.0) or 0.0),
                    "best_margin": float(best_eligible.get("margin", 0.0) or 0.0),
                    "best_alignment": str(best_eligible.get("alignment", "neutral_hold") or "neutral_hold"),
                    "best_action": self._control_terms(best_control),
                    "best_score_terms": self._score_term_subset(best_details),
                    "best_action_delta": dict(best_row.get("action_delta_terms", {})) if isinstance(best_row.get("action_delta_terms", {}), Mapping) else {},
                    "best_score_delta": dict(best_row.get("score_delta_terms", {})) if isinstance(best_row.get("score_delta_terms", {}), Mapping) else {},
                }
            )
        base["unsafe_conflict"] = bool(base.get("would_apply_shadow", False) and base.get("best_alignment") == "unsafe_conflict")
        return base

    def _apply_hot_dry_proposer_control_access(
        self,
        state,
        control: np.ndarray,
        rspc_action_scoring: Mapping[str, Any],
    ) -> np.ndarray:
        before = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        enabled = bool(getattr(self.config, "rspc_hot_dry_proposer_control_enabled", False))
        strict_enabled = bool(getattr(self.config, "rspc_hot_dry_proposer_control_strict_enabled", False))
        min_margin = float(
            getattr(
                self.config,
                "rspc_hot_dry_proposer_control_strict_min_margin"
                if strict_enabled
                else "rspc_hot_dry_proposer_control_min_margin",
                0.20 if strict_enabled else 0.05,
            )
        )
        replay = (
            rspc_action_scoring.get("hot_dry_action_proposer_controlled_replay", {})
            if isinstance(rspc_action_scoring, Mapping)
            else {}
        )
        if not isinstance(replay, Mapping):
            replay = {}
        temp_air = float(getattr(state, "temp_air", 20.0))
        rh_air = float(getattr(state, "rh_air", 70.0))
        try:
            vpd_air = float(getattr(state, "vpd_air"))
        except Exception:
            vpd_air = float(calculate_vpd_kpa(temp_air, rh_air))
        canopy_margin = float(getattr(state, "canopy_dew_margin", 3.0))
        strict_rh_max = float(getattr(self.config, "rspc_hot_dry_proposer_control_strict_rh_max", 45.0))
        strict_vpd_min = float(getattr(self.config, "rspc_hot_dry_proposer_control_strict_vpd_min", 2.25))
        strict_temp_max = float(getattr(self.config, "rspc_hot_dry_proposer_control_strict_temp_max", 30.5))
        strict_canopy_min = float(
            getattr(self.config, "rspc_hot_dry_proposer_control_strict_canopy_margin_min", 3.0)
        )
        strict_candidate = str(
            getattr(
                self.config,
                "rspc_hot_dry_proposer_control_strict_candidate",
                "shadow_hot_dry_humidity_retention",
            )
            or "shadow_hot_dry_humidity_retention"
        )
        candidate = str(replay.get("best_candidate_name", "") or "")
        info: Dict[str, Any] = {
            "enabled": enabled,
            "strict_enabled": strict_enabled,
            "applied": False,
            "reason": "disabled" if not enabled else "not_evaluated",
            "min_margin": float(min_margin),
            "before_action": self._control_terms(before),
            "after_action": self._control_terms(before),
            "candidate": candidate,
            "variant": str(replay.get("best_variant", "") or ""),
            "margin": float(replay.get("best_margin", 0.0) or 0.0),
            "safe_hot_dry": bool(replay.get("safe_hot_dry", False)),
            "safety_gate_reason": str(replay.get("safety_gate_reason", "none") or "none"),
            "temp_air": float(temp_air),
            "rh_air": float(rh_air),
            "vpd_air": float(vpd_air),
            "canopy_dew_margin": float(canopy_margin),
            "strict_rh_max": float(strict_rh_max),
            "strict_vpd_min": float(strict_vpd_min),
            "strict_temp_max": float(strict_temp_max),
            "strict_canopy_margin_min": float(strict_canopy_min),
            "strict_candidate": strict_candidate,
        }
        if not enabled:
            if isinstance(getattr(self, "last_rollout_selection", None), dict):
                self.last_rollout_selection["hot_dry_proposer_control"] = info
            return before

        gate = str(replay.get("safety_gate_reason", "none") or "none")
        checks = [
            ("missing_replay_metadata", bool(replay)),
            ("not_safe_hot_dry", bool(replay.get("safe_hot_dry", False))),
            ("safety_gate_active", gate == "none"),
            ("shadow_not_eligible", bool(replay.get("would_apply_shadow", False))),
            ("alignment_not_dry_benefit", str(replay.get("best_alignment", "") or "") == "dry_benefit"),
            ("unsafe_preferred", not bool(replay.get("unsafe_preferred", False))),
            ("unsafe_conflict", not bool(replay.get("unsafe_conflict", False))),
        ]
        for reason, passed in checks:
            if not passed:
                info["reason"] = reason
                if isinstance(getattr(self, "last_rollout_selection", None), dict):
                    self.last_rollout_selection["hot_dry_proposer_control"] = info
                return before
        if float(replay.get("best_margin", 0.0) or 0.0) < min_margin:
            info["reason"] = "margin_below_strict_min" if strict_enabled else "margin_below_threshold"
            if isinstance(getattr(self, "last_rollout_selection", None), dict):
                self.last_rollout_selection["hot_dry_proposer_control"] = info
            return before
        if strict_enabled:
            if candidate != strict_candidate:
                info["reason"] = "candidate_not_allowed"
                if isinstance(getattr(self, "last_rollout_selection", None), dict):
                    self.last_rollout_selection["hot_dry_proposer_control"] = info
                return before
            if not (rh_air <= strict_rh_max and vpd_air >= strict_vpd_min):
                info["reason"] = "not_severe_dry"
                if isinstance(getattr(self, "last_rollout_selection", None), dict):
                    self.last_rollout_selection["hot_dry_proposer_control"] = info
                return before
            if temp_air > strict_temp_max:
                info["reason"] = "temp_headroom_low"
                if isinstance(getattr(self, "last_rollout_selection", None), dict):
                    self.last_rollout_selection["hot_dry_proposer_control"] = info
                return before
            if canopy_margin < strict_canopy_min:
                info["reason"] = "canopy_reserve_low"
                if isinstance(getattr(self, "last_rollout_selection", None), dict):
                    self.last_rollout_selection["hot_dry_proposer_control"] = info
                return before

        best_action_terms = replay.get("best_action", {})
        if not isinstance(best_action_terms, Mapping) or not best_action_terms:
            info["reason"] = "missing_best_action"
            if isinstance(getattr(self, "last_rollout_selection", None), dict):
                self.last_rollout_selection["hot_dry_proposer_control"] = info
            return before

        after = self._control_from_terms(best_action_terms)
        if after is None:
            info["reason"] = "missing_best_action"
            if isinstance(getattr(self, "last_rollout_selection", None), dict):
                self.last_rollout_selection["hot_dry_proposer_control"] = info
            return before

        info.update(
            {
                "applied": True,
                "reason": "strict_eligible_applied" if strict_enabled else "applied",
                "after_action": self._control_terms(after),
                "delta_action": self._profile_action_delta_terms(before, after),
                "score_delta": (
                    dict(replay.get("best_score_delta", {}))
                    if isinstance(replay.get("best_score_delta", {}), Mapping)
                    else {}
                ),
            }
        )
        if isinstance(getattr(self, "last_rollout_selection", None), dict):
            self.last_rollout_selection["hot_dry_proposer_control"] = info
        return np.asarray(after, dtype=np.float32)

    def _profile_feasibility_gate_record(
        self,
        *,
        enabled: Optional[bool] = None,
        applied: bool = False,
        corrections: Optional[Sequence[str]] = None,
        mode: Optional[Mapping[str, Any]] = None,
        original_targets: Optional[Mapping[str, Any]] = None,
        repaired_targets: Optional[Mapping[str, Any]] = None,
        hard_safety_veto_count: int = 0,
        source: str = "",
    ) -> Dict[str, Any]:
        return {
            "enabled": bool(getattr(self.config, "profile_feasibility_gate_enabled", False) if enabled is None else enabled),
            "applied": bool(applied),
            "corrections": list(corrections or []),
            "mode": dict(mode or {}),
            "original_targets": dict(original_targets or {}),
            "repaired_targets": dict(repaired_targets or {}),
            "hard_safety_veto_enabled": bool(getattr(self.config, "profile_feasibility_gate_hard_safety_veto", True)),
            "hard_safety_veto_count": int(hard_safety_veto_count),
            "source": str(source or ""),
        }

    def _profile_feasibility_gate_mode(
        self,
        state,
        target_control: np.ndarray,
        target_temp: Optional[float],
        target_co2: Optional[float],
        target_rh: Optional[float],
    ) -> Dict[str, Any]:
        control = np.clip(np.asarray(target_control, dtype=np.float32).reshape(-1), 0.0, 1.0)
        rh_air = float(getattr(state, "rh_air", 70.0))
        temp_air = float(getattr(state, "temp_air", 20.0))
        glob_rad = float(getattr(state, "glob_rad", 0.0))
        vpd_air = float(calculate_vpd_kpa(temp_air, rh_air))
        dew_margin = float(getattr(state, "dew_margin_air", 3.0))
        canopy_margin = float(getattr(state, "canopy_dew_margin", 3.0))
        min_dew_margin = min(dew_margin, canopy_margin)
        vent_level = float(control[3]) if int(control.size) > 3 else 0.0
        tomato = getattr(self, "last_tomato_safety_v2", {}) if isinstance(getattr(self, "last_tomato_safety_v2", {}), dict) else {}
        tomato_reasons = " ".join(str(x) for x in tomato.get("reasons", []) or []).lower()
        tomato_vent_raise = (
            bool(tomato.get("applied", False))
            and isinstance(tomato.get("before_action", {}), Mapping)
            and isinstance(tomato.get("after_action", {}), Mapping)
            and float(tomato.get("after_action", {}).get("ventilation", tomato.get("after_action", {}).get("vent", 0.0)) or 0.0)
            > float(tomato.get("before_action", {}).get("ventilation", tomato.get("before_action", {}).get("vent", 0.0)) or 0.0) + 0.05
        )

        high_temp_pressure = (
            temp_air >= 32.0
            or float(getattr(state, "temp_violation", 0.0)) > 0.0
            or "hot_temperature" in tomato_reasons
            or "temp" in tomato_reasons and "high" in tomato_reasons
        )
        dew_canopy_pressure = (
            min_dew_margin < 1.0
            or "dew" in tomato_reasons
            or "canopy" in tomato_reasons
        )
        rh_hard_pressure = (
            rh_air >= float(getattr(self.config, "rh_control_limit", 90.0))
            or "rh_high" in tomato_reasons
            or "humidity" in tomato_reasons and "hard" in tomato_reasons
        )
        safety_vent_reason = bool(
            tomato_vent_raise
            and any(token in tomato_reasons for token in ("hard", "dew", "canopy", "extreme", "hot_temperature", "rh_high"))
        )
        vent_required = bool(high_temp_pressure or dew_canopy_pressure or rh_hard_pressure or safety_vent_reason)
        dry_side = bool(
            rh_air < float(getattr(self.config, "dry_rh_on", 55.0))
            or vpd_air > float(getattr(self.config, "dry_vpd_on", 1.20))
        )
        co2_allowed = bool(
            not vent_required
            and vent_level <= float(getattr(self.config, "fallback_co2_max_vent", 0.20))
            and glob_rad >= float(getattr(self.config, "fallback_co2_min_rad", 120.0))
        )
        humidity_allowed = bool(not vent_required and not dew_canopy_pressure and not rh_hard_pressure)
        return {
            "vent_required_by_safety": bool(vent_required),
            "humidity_retention_allowed": bool(humidity_allowed),
            "co2_enrichment_allowed": bool(co2_allowed),
            "high_temp_pressure": bool(high_temp_pressure),
            "dew_canopy_pressure": bool(dew_canopy_pressure),
            "rh_hard_pressure": bool(rh_hard_pressure),
            "safety_vent_reason": bool(safety_vent_reason),
            "dry_side": bool(dry_side),
            "target_ventilation": float(vent_level),
            "rh_air": float(rh_air),
            "temp_air": float(temp_air),
            "vpd_air": float(vpd_air),
            "dew_margin": float(min_dew_margin),
            "glob_rad": float(glob_rad),
        }

    def _apply_profile_feasibility_gate_targets(
        self,
        state,
        target_control: np.ndarray,
        target_temp: Optional[float],
        target_co2: Optional[float],
        target_rh: Optional[float],
        *,
        source: str,
        update_last: bool = True,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Dict[str, Any]]:
        enabled = bool(getattr(self.config, "profile_feasibility_gate_enabled", False))
        original = {
            "target_temp": None if target_temp is None else float(target_temp),
            "target_co2": None if target_co2 is None else float(target_co2),
            "target_rh": None if target_rh is None else float(target_rh),
        }
        mode = self._profile_feasibility_gate_mode(state, target_control, target_temp, target_co2, target_rh)
        if not enabled:
            record = self._profile_feasibility_gate_record(
                enabled=False,
                applied=False,
                mode=mode,
                original_targets=original,
                repaired_targets=original,
                source=source,
            )
            if update_last:
                self.last_profile_feasibility_gate = record
            return target_temp, target_co2, target_rh, record

        repaired_temp = float(target_temp) if target_temp is not None else None
        repaired_co2 = float(target_co2) if target_co2 is not None else None
        repaired_rh = float(target_rh) if target_rh is not None else None
        corrections: List[str] = []

        if repaired_rh is not None:
            if mode["dew_canopy_pressure"] or mode["rh_hard_pressure"]:
                cap = float(getattr(self.config, "rh_target_extreme_cap", 72.0))
            elif mode["vent_required_by_safety"]:
                cap = float(getattr(self.config, "rh_target_high_cap", 76.0))
            elif float(mode["rh_air"]) >= float(getattr(self.config, "rh_preemptive_threshold", 86.0)):
                cap = float(getattr(self.config, "rh_target_preemptive_cap", 80.0))
            else:
                cap = 88.0
            if repaired_rh > cap:
                repaired_rh = cap
                corrections.append("target_rh:safety_cap")
            if mode["vent_required_by_safety"] and repaired_rh >= 75.0:
                repaired_rh = min(repaired_rh, float(getattr(self.config, "rh_target_high_cap", 76.0)) - 1.0)
                corrections.append("target_rh:high_vent_compatibility_cap")
            if mode["dry_side"] and not mode["vent_required_by_safety"]:
                floor = float(getattr(self.config, "dry_target_rh_floor", 70.0))
                if repaired_rh < floor:
                    repaired_rh = min(88.0, floor)
                    corrections.append("target_rh:dry_recovery_floor")

        if repaired_temp is not None and mode["dry_side"] and not mode["vent_required_by_safety"]:
            cap = float(getattr(self.config, "dry_temp_target_cap", 20.0))
            if repaired_temp > cap:
                repaired_temp = cap
                corrections.append("target_temp:dry_recovery_cap")

        if repaired_co2 is not None and not mode["co2_enrichment_allowed"] and repaired_co2 > 430.0:
            repaired_co2 = 430.0
            corrections.append("target_co2:vent_or_radiation_cap")

        repaired = {
            "target_temp": repaired_temp,
            "target_co2": repaired_co2,
            "target_rh": repaired_rh,
        }
        unique_corrections = sorted(set(corrections))
        record = self._profile_feasibility_gate_record(
            enabled=True,
            applied=bool(unique_corrections),
            corrections=unique_corrections,
            mode=mode,
            original_targets=original,
            repaired_targets=repaired,
            source=source,
        )
        if update_last:
            self.last_profile_feasibility_gate = record
        return repaired_temp, repaired_co2, repaired_rh, record

    def _profile_template_patch_record(
        self,
        *,
        enabled: bool,
        applied: bool,
        corrections: Optional[Sequence[str]] = None,
        forbidden_combinations: Optional[Sequence[str]] = None,
        mode: Optional[Mapping[str, Any]] = None,
        original_targets: Optional[Mapping[str, Any]] = None,
        repaired_targets: Optional[Mapping[str, Any]] = None,
        source: str = "",
    ) -> Dict[str, Any]:
        return {
            "enabled": bool(enabled),
            "applied": bool(applied),
            "corrections": list(corrections or []),
            "forbidden_combinations": list(forbidden_combinations or []),
            "mode": dict(mode or {}),
            "original_targets": dict(original_targets or {}),
            "repaired_targets": dict(repaired_targets or {}),
            "source": str(source or ""),
            "fallback_veto_enabled": bool(getattr(self.config, "fallback_post_selection_veto_enabled", False)),
            "fallback_veto_applied": False,
            "fallback_veto_no_alternative": False,
            "fallback_veto_reason": "",
            "recovery_anchor_enabled": bool(getattr(self.config, "recovery_anchor_enabled", False)),
            "recovery_anchor_applied": False,
            "recovery_anchor_source": str(getattr(self.config, "recovery_anchor_source", "") or ""),
            "recovery_anchor_reason": "",
        }

    def _store_profile_template_patch_record(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        stored = dict(record)
        existing = dict(getattr(self, "last_profile_template_patch", {}) or {})
        for key in (
            "fallback_veto_applied",
            "fallback_veto_no_alternative",
            "fallback_veto_reason",
            "fallback_veto_selected_before",
            "fallback_veto_selected_after",
            "fallback_veto_safety_reasons",
            "fallback_veto_rewrite_fields",
            "recovery_anchor_enabled",
            "recovery_anchor_applied",
            "recovery_anchor_source",
            "recovery_anchor_reason",
            "recovery_anchor_action",
            "recovery_anchor_prediction",
            "recovery_anchor_no_compatible_existing_candidate",
            "recovery_anchor_selected_before",
            "recovery_anchor_selected_after",
            "recovery_anchor_safety_reasons",
            "recovery_anchor_rewrite_fields",
        ):
            if key in existing and key not in stored:
                stored[key] = existing[key]
            elif key.startswith("recovery_anchor_") and key in existing:
                existing_value = existing.get(key)
                stored_value = stored.get(key)
                if isinstance(existing_value, bool):
                    if existing_value and not bool(stored_value):
                        stored[key] = existing_value
                elif existing_value not in (None, "", [], {}):
                    if stored_value in (None, "", [], {}, False):
                        stored[key] = existing_value
        self.last_profile_template_patch = stored
        return stored

    def _apply_profile_template_patch_targets(
        self,
        state,
        target_control: np.ndarray,
        target_temp: Optional[float],
        target_co2: Optional[float],
        target_rh: Optional[float],
        *,
        source: str,
        update_last: bool = True,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Dict[str, Any]]:
        enabled = bool(getattr(self.config, "profile_template_patch_enabled", False))
        original = {
            "target_temp": None if target_temp is None else float(target_temp),
            "target_co2": None if target_co2 is None else float(target_co2),
            "target_rh": None if target_rh is None else float(target_rh),
        }
        mode = self._profile_feasibility_gate_mode(state, target_control, target_temp, target_co2, target_rh)
        if not enabled:
            record = self._profile_template_patch_record(
                enabled=False,
                applied=False,
                mode=mode,
                original_targets=original,
                repaired_targets=original,
                source=source,
            )
            if update_last:
                record = self._store_profile_template_patch_record(record)
            return target_temp, target_co2, target_rh, record

        repaired_temp = float(target_temp) if target_temp is not None else None
        repaired_co2 = float(target_co2) if target_co2 is not None else None
        repaired_rh = float(target_rh) if target_rh is not None else None
        corrections: List[str] = []
        forbidden: List[str] = []

        if repaired_rh is not None:
            if mode["dew_canopy_pressure"] or mode["rh_hard_pressure"]:
                cap = float(getattr(self.config, "rh_target_extreme_cap", 72.0))
            elif mode["vent_required_by_safety"]:
                cap = float(getattr(self.config, "rh_target_high_cap", 76.0))
            elif float(mode["rh_air"]) >= float(getattr(self.config, "rh_preemptive_threshold", 86.0)):
                cap = float(getattr(self.config, "rh_target_preemptive_cap", 80.0))
            else:
                cap = 88.0
            if repaired_rh > cap:
                repaired_rh = cap
                corrections.append("target_rh:safety_cap")
            if mode["vent_required_by_safety"] and repaired_rh >= 75.0:
                repaired_rh = min(repaired_rh, float(getattr(self.config, "rh_target_high_cap", 76.0)) - 1.0)
                corrections.append("target_rh:high_vent_compatibility_cap")
                forbidden.append("safety_vent_blocks_high_rh")
            if mode["dry_side"] and not mode["vent_required_by_safety"]:
                floor = float(getattr(self.config, "dry_target_rh_floor", 70.0))
                if repaired_rh < floor:
                    repaired_rh = min(88.0, floor)
                    corrections.append("target_rh:dry_recovery_floor")

        if repaired_temp is not None and mode["dry_side"] and not mode["vent_required_by_safety"]:
            cap = float(getattr(self.config, "dry_temp_target_cap", 20.0))
            if repaired_temp > cap:
                repaired_temp = cap
                corrections.append("target_temp:dry_recovery_cap")

        if repaired_co2 is not None and not mode["co2_enrichment_allowed"] and repaired_co2 > 430.0:
            repaired_co2 = 430.0
            corrections.append("target_co2:vent_or_radiation_cap")
            if mode["vent_required_by_safety"]:
                forbidden.append("safety_vent_blocks_high_co2")

        if mode["vent_required_by_safety"]:
            if original.get("target_rh") is not None and float(original["target_rh"]) >= 75.0:
                if original.get("target_co2") is not None and float(original["target_co2"]) >= 800.0:
                    forbidden.append("safety_vent_blocks_high_rh_high_co2")
            if not mode.get("humidity_retention_allowed", True):
                forbidden.append("rh_dew_canopy_pressure_blocks_humidity_retention")
        if mode["dry_side"] and mode["vent_required_by_safety"]:
            forbidden.append("dry_high_vent_penalty_suppressed_by_safety_vent")

        repaired = {
            "target_temp": repaired_temp,
            "target_co2": repaired_co2,
            "target_rh": repaired_rh,
        }
        record = self._profile_template_patch_record(
            enabled=True,
            applied=bool(corrections or forbidden),
            corrections=sorted(set(corrections)),
            forbidden_combinations=sorted(set(forbidden)),
            mode=mode,
            original_targets=original,
            repaired_targets=repaired,
            source=source,
        )
        if update_last:
            record = self._store_profile_template_patch_record(record)
        return repaired_temp, repaired_co2, repaired_rh, record

    def _profile_feasibility_candidate_adjustment(
        self,
        state,
        control: np.ndarray,
        plan: Mapping[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        if not bool(getattr(self.config, "profile_feasibility_gate_enabled", False)):
            return 0.0, {"enabled": False}
        clipped = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        target_temp = self._get_plan_target(dict(plan), "target_temp", state)
        target_co2 = self._get_plan_target(dict(plan), "target_co2", state)
        target_rh = self._get_plan_target(dict(plan), "target_rh", state)
        repaired_temp, repaired_co2, repaired_rh, gate_record = self._apply_profile_feasibility_gate_targets(
            state,
            clipped,
            target_temp,
            target_co2,
            target_rh,
            source="candidate_scoring",
            update_last=False,
        )
        mode = gate_record.get("mode", {}) if isinstance(gate_record.get("mode", {}), Mapping) else {}
        conflicts: List[str] = []
        vent = float(clipped[3]) if int(clipped.size) > 3 else 0.0
        heat = float(clipped[0]) if int(clipped.size) > 0 else 0.0
        if repaired_rh is not None and float(repaired_rh) >= 75.0 and vent >= 0.70:
            conflicts.append("target_rh_vs_high_vent")
        if repaired_co2 is not None and float(repaired_co2) >= 800.0 and vent >= 0.30:
            conflicts.append("co2_target_vs_vent")
        if repaired_temp is not None:
            temp_air = float(getattr(state, "temp_air", 20.0))
            if float(repaired_temp) <= temp_air - 1.0 and heat >= 0.10:
                conflicts.append("cooling_target_vs_heat")
            if float(repaired_temp) >= temp_air + 1.0 and vent >= 0.50:
                conflicts.append("heating_target_vs_vent")
        dry_high_vent = bool(mode.get("dry_side", False) and not mode.get("vent_required_by_safety", False) and vent >= 0.50)
        if dry_high_vent:
            conflicts.append("dry_risk_vs_high_vent")

        safety_control, safety_info = apply_tomato_safety_v2(
            state,
            clipped,
            config=self.config,
            target_rh=repaired_rh,
        )
        reason_text = " ".join(str(x) for x in safety_info.get("reasons", []) or []).lower()
        hard_safety_rewrite = bool(
            safety_info.get("applied", False)
            and any(token in reason_text for token in ("hard", "dew", "canopy", "extreme", "hot_temperature"))
        )
        rewrite_delta = float(np.sum(np.abs(np.asarray(safety_control, dtype=np.float32) - clipped)))
        hard_safety_vetoed = bool(
            hard_safety_rewrite and bool(getattr(self.config, "profile_feasibility_gate_hard_safety_veto", True))
        )
        adjustment = (
            1.25 * len(conflicts)
            + 1.50 * int(dry_high_vent)
            + 0.75 * int(rewrite_delta >= 0.50 or hard_safety_rewrite)
            + (1000.0 if hard_safety_vetoed else 0.0)
        )
        details = {
            "enabled": True,
            "adjustment": float(adjustment),
            "profile_action_conflicts": list(sorted(set(conflicts))),
            "profile_action_conflict_count": int(len(set(conflicts))),
            "dry_risk_high_vent_conflict": int(dry_high_vent),
            "major_or_hard_rewrite_predicted": bool(rewrite_delta >= 0.50 or hard_safety_rewrite),
            "hard_safety_rewrite_predicted": bool(hard_safety_rewrite),
            "hard_safety_vetoed": bool(hard_safety_vetoed),
            "safety_rewrite_delta_abs_sum": float(rewrite_delta),
            "safety_reasons": list(safety_info.get("reasons", []) or []),
            "gate_record": gate_record,
        }
        return float(adjustment), details

    def _apply_rspc_post_score_shadow_shape(
        self,
        state,
        control: np.ndarray,
        plan: Mapping[str, Any],
        *,
        rh_debt: float,
        dehumidify_mode: str,
        lamp_budget_remaining: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        shaped = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        events: Dict[str, Any] = {
            "target_tracking_applied": False,
            "dry_recovery_applied": False,
            "tomato_safety_applied": False,
            "dehumidify_mode": str(dehumidify_mode or "normal"),
        }
        target_temp = self._get_plan_target(dict(plan), "target_temp", state)
        target_co2 = self._get_plan_target(dict(plan), "target_co2", state)
        target_rh = self._get_plan_target(dict(plan), "target_rh", state)
        target_temp, target_co2, target_rh, gate_record = self._apply_profile_feasibility_gate_targets(
            state,
            shaped,
            target_temp,
            target_co2,
            target_rh,
            source="rspc_post_score_shadow_shape",
            update_last=False,
        )
        events["profile_feasibility_gate"] = gate_record
        events["profile_feasibility_gate_applied"] = bool(gate_record.get("applied", False))
        events["profile_feasibility_gate_corrections"] = list(gate_record.get("corrections", []) or [])
        tracked = self._apply_profile_target_tracking_proxy(
            state,
            shaped,
            target_temp=target_temp,
            target_co2=target_co2,
            target_rh=target_rh,
        )
        events["target_tracking_applied"] = bool(float(np.max(np.abs(tracked - shaped))) > 1e-6)
        shaped = np.asarray(tracked, dtype=np.float32)

        rh_air = float(getattr(state, "rh_air", 70.0))
        temp_air = float(getattr(state, "temp_air", 20.0))
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
            or float(rh_debt) > 2.0
        )
        if rh_risk_active:
            before = shaped.copy()
            severity = max(rh_air - float(self.config.rh_preemptive_threshold), 0.0)
            if low_vpd_humidity_risk:
                severity = max(severity, max(0.42 - vpd_now, 0.0) * 6.0)
            severity = max(severity, max(1.1 - dew_margin, 0.0) * 2.0)
            debt_vent = min(0.22, float(self.config.rh_debt_vent_gain) * float(rh_debt))
            debt_screen = min(0.28, float(self.config.rh_debt_screen_gain) * float(rh_debt))
            vent_floor = min(0.68, 0.30 + 0.035 * severity + debt_vent)
            screen_cap = max(0.25, 0.72 - 0.035 * severity - debt_screen)
            shaped[3] = max(shaped[3], vent_floor)
            shaped[2] = min(shaped[2], screen_cap)
            shaped[1] = 0.0
            shaped[4] = 0.0
            events["rh_risk_shape_applied"] = bool(float(np.max(np.abs(shaped - before))) > 1e-6)
            events["rh_risk_vent_floor"] = float(vent_floor)
            events["rh_risk_screen_cap"] = float(screen_cap)
        else:
            events["rh_risk_shape_applied"] = False

        if float(lamp_budget_remaining) <= 0.0:
            shaped[4] = 0.0
        elif float(lamp_budget_remaining) < 0.8:
            shaped[4] = min(shaped[4], float(self.config.lamp_budget_soft_cap))
        if rh_air >= float(self.config.lamp_forbidden_rh) or shaped[3] >= float(self.config.lamp_forbidden_vent):
            shaped[4] = 0.0
        if str(dehumidify_mode or "normal") != "normal":
            shaped[4] = 0.0
            shaped[1] = 0.0
        elif rh_air >= 88.0:
            shaped[1] = 0.0
        if shaped[3] >= float(self.config.co2_forbidden_vent) or float(getattr(state, "glob_rad", 0.0)) < float(self.config.fallback_co2_min_rad):
            shaped[1] = 0.0

        hour = float(getattr(state, "hour_of_day", 12.0))
        if (
            float(self.config.dawn_predehumid_start_hour) <= hour < float(self.config.dawn_predehumid_end_hour)
            and rh_air > 82.0
        ):
            shaped[3] = max(shaped[3], 0.32)
            shaped[2] = min(shaped[2], 0.65)
            shaped[1] = 0.0
            shaped[4] = 0.0
            events["dawn_predehumid_shape_applied"] = True
        else:
            events["dawn_predehumid_shape_applied"] = False

        if str(dehumidify_mode or "normal") == "mild":
            shaped[3] = max(shaped[3], 0.40)
            shaped[2] = min(shaped[2], 0.65)
            shaped[1] = 0.0
            shaped[4] = 0.0
        elif str(dehumidify_mode or "normal") == "strong":
            shaped[3] = max(shaped[3], float(self.config.rh_pulse_vent_floor))
            shaped[2] = min(shaped[2], float(self.config.rh_pulse_screen_cap))
            shaped[1] = 0.0
            shaped[4] = 0.0

        dry_side_risk = (
            rh_air <= float(self.config.dry_rh_on)
            or vpd_now >= float(self.config.dry_vpd_on)
        )
        if dry_side_risk:
            before = shaped.copy()
            hot_dry_features = self._rspc_hot_dry_features(state)
            if temp_air >= 28.0:
                vent_cap = float(self.config.dry_hot_vent_cap)
            elif temp_air >= 24.0:
                vent_cap = float(self.config.dry_warm_vent_cap)
            else:
                vent_cap = float(self.config.dry_vent_cap)
            if hot_dry_features["active"]:
                vent_cap = max(vent_cap, self._rspc_hot_dry_vent_relief_floor(state, hot_dry_features))
            shaped[3] = min(shaped[3], vent_cap)
            shaped[1] = 0.0
            shaped[4] = 0.0
            if temp_air < 12.5:
                shaped[0] = max(shaped[0], 0.75)
                shaped[2] = max(shaped[2], 0.90)
                shaped[3] = min(shaped[3], float(self.config.cold_dehumidify_vent_cap))
            elif temp_air < 15.0:
                shaped[0] = min(max(shaped[0], 0.10), 0.25)
                shaped[2] = max(shaped[2], 0.75)
            else:
                shaped[0] = min(shaped[0], 0.05)
                if temp_air < 18.0:
                    shaped[2] = max(shaped[2], 0.70)
            if float(getattr(state, "glob_rad", 0.0)) > 250.0 or temp_air > 24.0:
                shaped[5] = max(shaped[5], 0.50)
            events["dry_recovery_applied"] = True
            events["dry_recovery_vent_cap"] = float(vent_cap)
            events["dry_recovery_hot_dry_vent_relief"] = bool(hot_dry_features["active"])
            events["dry_recovery_delta_l1"] = float(np.sum(np.abs(shaped - before)))
        else:
            events["dry_recovery_applied"] = False

        safety_control, safety_info = apply_tomato_safety_v2(
            state,
            shaped,
            config=self.config,
            target_rh=target_rh,
        )
        events["tomato_safety_applied"] = bool(safety_info.get("applied", False))
        events["tomato_safety_reasons"] = list(safety_info.get("reasons", [])) if isinstance(safety_info.get("reasons", []), list) else []
        shaped = np.asarray(safety_control, dtype=np.float32)
        return np.clip(shaped, 0.0, 1.0).astype(np.float32), events

    def _rspc_action_candidate_record(
        self,
        *,
        name: str,
        control: np.ndarray,
        score: float,
        rule_weight: float,
        details: Mapping[str, Any],
        selected: bool,
        post_shape_control: Optional[np.ndarray] = None,
        post_shape_score: Optional[float] = None,
        post_shape_details: Optional[Mapping[str, Any]] = None,
        post_shape_events: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "name": str(name),
            "score": float(score) if np.isfinite(score) else None,
            "rule_weight": float(rule_weight),
            "selected": bool(selected),
            "action": self._control_terms(control),
            "score_terms": self._score_term_subset(details),
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
        if post_shape_control is not None and post_shape_details is not None:
            record["post_shape_action"] = self._control_terms(post_shape_control)
            record["post_shape_score"] = (
                float(post_shape_score)
                if post_shape_score is not None and np.isfinite(float(post_shape_score))
                else None
            )
            record["post_shape_score_terms"] = self._score_term_subset(post_shape_details)
            record["post_shape_events"] = dict(post_shape_events or {})
        return record

    @staticmethod
    def _rspc_action_safety_gate_reason(state) -> str:
        temp_air = float(getattr(state, "temp_air", 20.0))
        rh_air = float(getattr(state, "rh_air", 70.0))
        dew_air = float(getattr(state, "dew_margin_air", 3.0))
        canopy_margin = float(getattr(state, "canopy_dew_margin", 3.0))
        if temp_air >= 32.0 or float(getattr(state, "temp_violation", 0.0)) > 0.0:
            return "temp_high_gate"
        if canopy_margin < 1.0:
            return "canopy_gate"
        if dew_air < 1.0:
            return "dew_gate"
        if rh_air >= 90.0 or float(getattr(state, "rh_high_violation", 0.0)) > 0.0:
            return "rh_high_gate"
        return "none"

    def _rspc_action_post_shape_alignment(
        self,
        state,
        *,
        gate_reason: str,
        margin: float,
        action_delta_terms: Mapping[str, float],
        score_delta_terms: Mapping[str, float],
    ) -> str:
        if float(margin) <= 0.05:
            return "neutral_hold"
        gate = str(gate_reason or "none")
        if gate != "none":
            if self._profile_rspc_gate_risk_worsened(gate, score_delta_terms):
                return "unsafe_conflict"
            if self._profile_rspc_action_conflicts(gate, action_delta_terms):
                return "unsafe_conflict"
            return "safe_relief"
        if self._profile_rspc_safe_hot_dry_state(state):
            dry_improved = (
                float(score_delta_terms.get("dry", 0.0)) < -1e-6
                or float(score_delta_terms.get("vpd", 0.0)) < -1e-6
                or float(score_delta_terms.get("hot_dry", 0.0)) < -1e-6
            )
            safety_not_worse = (
                float(score_delta_terms.get("temp", 0.0)) <= 1e-6
                and float(score_delta_terms.get("dew", 0.0)) <= 1e-6
                and float(action_delta_terms.get("heat", 0.0)) <= 0.02
                and float(action_delta_terms.get("co2", 0.0)) <= 0.02
                and float(action_delta_terms.get("lamp", 0.0)) <= 0.02
            )
            if dry_improved and safety_not_worse:
                return "dry_benefit"
        return "neutral_hold"

    @staticmethod
    def _rspc_tt_calibration_safe_hot_dry_state(state) -> bool:
        temp = float(getattr(state, "temp_air", 20.0))
        rh = float(getattr(state, "rh_air", 70.0))
        vpd = float(calculate_vpd_kpa(temp, rh))
        dew_margin_air = float(getattr(state, "dew_margin_air", 3.0))
        canopy_dew_margin = float(getattr(state, "canopy_dew_margin", 3.0))
        dry_pressure = (
            rh <= 55.0
            or vpd >= 1.60
            or float(getattr(state, "rh_low_violation", 0.0)) > 0.0
            or float(getattr(state, "vpd_high_excess", 0.0)) > 0.0
        )
        safety_clear = (
            temp < 32.0
            and float(getattr(state, "temp_violation", 0.0)) <= 0.0
            and float(getattr(state, "rh_high_violation", 0.0)) <= 0.0
            and rh < 90.0
            and dew_margin_air >= 1.0
            and canopy_dew_margin >= 1.0
        )
        return bool(dry_pressure and safety_clear)

    @staticmethod
    def _rspc_tt_calibration_alignment(
        *,
        improvement_margin: float,
        action_delta_terms: Mapping[str, float],
        score_delta_terms: Mapping[str, float],
    ) -> str:
        if float(improvement_margin) <= 1e-6:
            return "neutral_hold"
        safety_worse = (
            float(score_delta_terms.get("temp", 0.0)) > 1e-6
            or float(score_delta_terms.get("dew", 0.0)) > 1e-6
            or float(action_delta_terms.get("heat", 0.0)) > 0.02
            or float(action_delta_terms.get("co2", 0.0)) > 0.02
            or float(action_delta_terms.get("lamp", 0.0)) > 0.02
        )
        if safety_worse:
            return "unsafe_conflict"
        dry_improved = (
            float(score_delta_terms.get("dry", 0.0)) < -1e-6
            or float(score_delta_terms.get("vpd", 0.0)) < -1e-6
            or float(score_delta_terms.get("hot_dry", 0.0)) < -1e-6
        )
        return "dry_benefit" if dry_improved else "neutral_hold"

    def _evaluate_target_tracking_calibration_shadow(
        self,
        state,
        before_control: np.ndarray,
        current_control: Optional[np.ndarray],
        current_score: Optional[float],
        current_details: Mapping[str, Any],
        *,
        gate_reason: str,
    ) -> Dict[str, Any]:
        base = np.clip(np.asarray(before_control, dtype=np.float32), 0.0, 1.0)
        if current_control is None or current_score is None:
            return {
                "enabled": True,
                "shadow_only": True,
                "triggered": False,
                "reason": "missing_current_target_tracking",
                "safety_gate_reason": str(gate_reason or "none"),
                "variants": [],
            }
        current = np.clip(np.asarray(current_control, dtype=np.float32), 0.0, 1.0)
        gate = str(gate_reason or "none") or "none"
        delta = self._profile_action_delta_terms(base, current)
        safe_hot_dry = self._rspc_tt_calibration_safe_hot_dry_state(state)
        reason = "target_tracking_vent_only_dry_push"
        if gate != "none":
            reason = "safety_gate_active"
        elif not safe_hot_dry:
            reason = "not_safe_hot_dry"
        elif not (
            float(delta.get("vent", 0.0)) > 0.10
            and float(delta.get("screen", 0.0)) < 0.10
            and float(delta.get("shade", 0.0)) < 0.10
        ):
            reason = "no_vent_only_dry_push"

        variants: List[Dict[str, Any]] = []
        triggered = reason == "target_tracking_vent_only_dry_push"
        if triggered:
            variant_controls = []
            vent_cap = current.copy()
            vent_cap[3] = min(float(vent_cap[3]), float(base[3]) + 0.08)
            variant_controls.append(("vent_cap", np.clip(vent_cap, 0.0, 1.0)))

            shade_buffer = current.copy()
            shade_buffer[5] = max(float(shade_buffer[5]), min(1.0, float(base[5]) + 0.15))
            variant_controls.append(("shade_buffer", np.clip(shade_buffer, 0.0, 1.0)))

            current_terms = self._score_term_subset(current_details)
            for name, variant_control in variant_controls:
                score, details = self._score_fallback_candidate(state, variant_control)
                score_terms = self._score_term_subset(details)
                action_delta = self._profile_action_delta_terms(current, variant_control)
                score_delta = self._profile_score_delta_terms(current_terms, score_terms)
                margin = float(current_score) - float(score)
                alignment = self._rspc_tt_calibration_alignment(
                    improvement_margin=margin,
                    action_delta_terms=action_delta,
                    score_delta_terms=score_delta,
                )
                variants.append(
                    {
                        "name": str(name),
                        "action": self._control_terms(variant_control),
                        "score": float(score),
                        "score_terms": score_terms,
                        "improvement_margin": float(margin),
                        "action_delta": action_delta,
                        "score_delta": score_delta,
                        "alignment": alignment,
                        "unsafe_conflict": bool(alignment == "unsafe_conflict"),
                    }
                )

        eligible_variants = [item for item in variants if not bool(item.get("unsafe_conflict", False))]
        best_variant = max(
            eligible_variants or variants,
            key=lambda item: float(item.get("improvement_margin", 0.0) or 0.0),
        ) if variants else {}
        best_margin = float(best_variant.get("improvement_margin", 0.0) or 0.0) if best_variant else 0.0
        unsafe_variant_count = int(sum(bool(item.get("unsafe_conflict", False)) for item in variants))
        best_unsafe_conflict = bool(best_variant.get("unsafe_conflict", False)) if best_variant else False
        return {
            "enabled": True,
            "shadow_only": True,
            "triggered": bool(triggered),
            "reason": reason,
            "safety_gate_reason": gate,
            "safe_hot_dry": bool(safe_hot_dry),
            "before_action": self._control_terms(base),
            "current_action": self._control_terms(current),
            "current_score": float(current_score),
            "current_score_terms": self._score_term_subset(current_details),
            "target_tracking_delta": delta,
            "variants": variants,
            "variant_count": int(len(variants)),
            "best_variant_name": str(best_variant.get("name", "") or ""),
            "best_variant_score": (
                float(best_variant.get("score", 0.0) or 0.0)
                if best_variant
                else None
            ),
            "best_improvement_margin": float(best_margin),
            "best_alignment": str(best_variant.get("alignment", "neutral_hold") or "neutral_hold") if best_variant else "neutral_hold",
            "best_action": best_variant.get("action", {}) if best_variant else {},
            "best_action_delta": best_variant.get("action_delta", {}) if best_variant else {},
            "best_score_delta": best_variant.get("score_delta", {}) if best_variant else {},
            "has_unsafe_variant": bool(unsafe_variant_count > 0),
            "unsafe_variant_count": unsafe_variant_count,
            "unsafe_conflict": best_unsafe_conflict,
            "would_improve": bool(best_variant and best_margin > 0.05 and not bool(best_variant.get("unsafe_conflict", False))),
        }

    def _rspc_action_scoring_diagnostics(
        self,
        state,
        scored_candidates: List[Tuple[float, str, np.ndarray, float, Mapping[str, Any]]],
        *,
        plan: Mapping[str, Any],
        rh_debt: float,
        dehumidify_mode: str,
        lamp_budget_remaining: float,
        selected_name: str,
        selected_control: np.ndarray,
        selected_score: float,
    ) -> Dict[str, Any]:
        hot_dry_features = self._rspc_hot_dry_features(state)
        hot_dry_shadow_features = self._rspc_hot_dry_shadow_features(state)
        selected_action = self._control_terms(selected_control)
        selected_terms: Mapping[str, Any] = {}
        candidate_records: List[Dict[str, Any]] = []
        selected_post_shape_score = None
        selected_post_shape_control: Optional[np.ndarray] = None
        selected_post_shape_details: Mapping[str, Any] = {}
        post_shape_rows: List[Dict[str, Any]] = []
        actual_candidate_count = int(len(scored_candidates))
        for score, name, control_item, rule_weight, details in scored_candidates:
            is_selected = str(name) == str(selected_name)
            if is_selected:
                selected_terms = details
            post_control, post_events = self._apply_rspc_post_score_shadow_shape(
                state,
                np.asarray(control_item, dtype=np.float32),
                plan,
                rh_debt=float(rh_debt),
                dehumidify_mode=str(dehumidify_mode or "normal"),
                lamp_budget_remaining=float(lamp_budget_remaining),
            )
            post_score, post_details = self._score_fallback_candidate(state, post_control)
            if is_selected:
                selected_post_shape_score = float(post_score)
                selected_post_shape_control = np.asarray(post_control, dtype=np.float32)
                selected_post_shape_details = post_details
            post_shape_rows.append(
                {
                    "name": str(name),
                    "score": float(post_score),
                    "control": np.asarray(post_control, dtype=np.float32),
                    "details": post_details,
                    "events": dict(post_events or {}),
                    "shadow_proposer": False,
                }
            )
            record = self._rspc_action_candidate_record(
                name=str(name),
                control=np.asarray(control_item, dtype=np.float32),
                score=float(score),
                rule_weight=float(rule_weight),
                details=details,
                selected=is_selected,
                post_shape_control=post_control,
                post_shape_score=float(post_score),
                post_shape_details=post_details,
                post_shape_events=post_events,
            )
            record["candidate_source"] = "actual"
            record["shadow_proposer"] = False
            candidate_records.append(record)

        hot_dry_shadow_candidates = self._build_rspc_hot_dry_shadow_candidates(state, selected_control)
        for name, control_item, rationale in hot_dry_shadow_candidates:
            raw_score, raw_details = self._score_fallback_candidate(state, control_item)
            post_control, post_events = self._apply_rspc_post_score_shadow_shape(
                state,
                np.asarray(control_item, dtype=np.float32),
                plan,
                rh_debt=float(rh_debt),
                dehumidify_mode=str(dehumidify_mode or "normal"),
                lamp_budget_remaining=float(lamp_budget_remaining),
            )
            post_score, post_details = self._score_fallback_candidate(state, post_control)
            post_shape_rows.append(
                {
                    "name": str(name),
                    "score": float(post_score),
                    "control": np.asarray(post_control, dtype=np.float32),
                    "details": post_details,
                    "events": dict(post_events or {}),
                    "shadow_proposer": True,
                    "reason": str(rationale),
                }
            )
            record = self._rspc_action_candidate_record(
                name=str(name),
                control=np.asarray(control_item, dtype=np.float32),
                score=float(raw_score),
                rule_weight=0.0,
                details=raw_details,
                selected=False,
                post_shape_control=post_control,
                post_shape_score=float(post_score),
                post_shape_details=post_details,
                post_shape_events=post_events,
            )
            record["candidate_source"] = "hot_dry_action_proposer_shadow"
            record["shadow_proposer"] = True
            record["reason"] = str(rationale)
            candidate_records.append(record)

        hot_dry_count = sum(1 for item in candidate_records if str(item.get("name", "")).startswith("hot_dry_"))
        hot_dry_shadow_count = sum(1 for item in candidate_records if bool(item.get("shadow_proposer", False)))
        selected_score_terms = self._score_term_subset(selected_terms)
        post_shape_candidates = [
            item for item in candidate_records if item.get("post_shape_score") is not None
        ]
        post_shape_selected_action: Dict[str, float] = {}
        post_shape_selected_score_terms: Dict[str, float] = self._score_term_subset(selected_post_shape_details)
        if selected_post_shape_control is not None:
            post_shape_selected_action = self._control_terms(selected_post_shape_control)
        post_shape_gate_reason = self._rspc_action_safety_gate_reason(state)
        target_tracking_calibration = self._evaluate_target_tracking_calibration_shadow(
            state,
            selected_control,
            selected_post_shape_control,
            selected_post_shape_score,
            selected_post_shape_details,
            gate_reason=post_shape_gate_reason,
        )

        post_shape_rows_by_name: Dict[str, Dict[str, Any]] = {}
        for row in post_shape_rows:
            row_score = float(row.get("score", 0.0) or 0.0)
            row_margin = (
                float(selected_post_shape_score) - row_score
                if selected_post_shape_score is not None
                else 0.0
            )
            row_action_delta: Dict[str, float] = {}
            row_score_delta: Dict[str, float] = {}
            if selected_post_shape_control is not None:
                row_action_delta = self._profile_action_delta_terms(
                    selected_post_shape_control,
                    np.asarray(row.get("control", np.zeros(6)), dtype=np.float32),
                )
            if isinstance(selected_post_shape_details, Mapping) and isinstance(row.get("details", {}), Mapping):
                row_score_delta = self._profile_score_delta_terms(selected_post_shape_details, row.get("details", {}))
            row_alignment = self._rspc_action_post_shape_alignment(
                state,
                gate_reason=post_shape_gate_reason,
                margin=row_margin,
                action_delta_terms=row_action_delta,
                score_delta_terms=row_score_delta,
            )
            row["margin"] = float(row_margin)
            row["action_delta_terms"] = row_action_delta
            row["score_delta_terms"] = row_score_delta
            row["alignment"] = row_alignment
            row["eligible"] = bool(not (row_margin > 0.05 and row_alignment == "unsafe_conflict"))
            post_shape_rows_by_name[str(row.get("name", "") or "")] = row

        for record in candidate_records:
            row = post_shape_rows_by_name.get(str(record.get("name", "") or ""))
            if row:
                record["post_shape_margin"] = float(row.get("margin", 0.0) or 0.0)
                record["post_shape_alignment"] = str(row.get("alignment", "neutral_hold") or "neutral_hold")
                record["post_shape_eligible"] = bool(row.get("eligible", False))
                record["post_shape_action_delta_terms"] = dict(row.get("action_delta_terms", {}))
                record["post_shape_score_delta_terms"] = dict(row.get("score_delta_terms", {}))

        raw_post_shape_best_row = min(
            post_shape_rows,
            key=lambda item: float(item.get("score", 0.0) or 0.0),
        ) if post_shape_rows else {}
        eligible_post_shape_rows = [row for row in post_shape_rows if bool(row.get("eligible", False))]
        post_shape_best_row = min(
            eligible_post_shape_rows,
            key=lambda item: float(item.get("score", 0.0) or 0.0),
        ) if eligible_post_shape_rows else {}
        post_shape_best_name = str(post_shape_best_row.get("name", "") or "")
        post_shape_best = next(
            (
                item for item in post_shape_candidates
                if str(item.get("name", "") or "") == post_shape_best_name
            ),
            {},
        )
        post_shape_best_score = (
            float(post_shape_best_row.get("score", 0.0) or 0.0)
            if post_shape_best_row
            else None
        )
        post_shape_margin = (
            float(post_shape_best_row.get("margin", 0.0) or 0.0)
            if post_shape_best_row
            else 0.0
        )
        post_shape_action_delta_terms: Dict[str, float] = dict(post_shape_best_row.get("action_delta_terms", {})) if post_shape_best_row else {}
        post_shape_score_delta_terms: Dict[str, float] = dict(post_shape_best_row.get("score_delta_terms", {})) if post_shape_best_row else {}
        post_shape_alignment = str(post_shape_best_row.get("alignment", "neutral_hold") or "neutral_hold") if post_shape_best_row else "neutral_hold"
        best_details = post_shape_best_row.get("details", {}) if post_shape_best_row else {}
        post_shape_best_score_terms = self._score_term_subset(best_details) if isinstance(best_details, Mapping) else {}
        raw_post_shape_margin = float(raw_post_shape_best_row.get("margin", 0.0) or 0.0) if raw_post_shape_best_row else 0.0
        raw_post_shape_alignment = str(raw_post_shape_best_row.get("alignment", "neutral_hold") or "neutral_hold") if raw_post_shape_best_row else "neutral_hold"
        raw_post_shape_would_switch = bool(
            raw_post_shape_best_row
            and str(raw_post_shape_best_row.get("name", "") or "") != str(selected_name)
            and raw_post_shape_margin > 0.05
        )
        post_shape_would_switch = bool(
            post_shape_best
            and str(post_shape_best.get("name", "") or "") != str(selected_name)
            and post_shape_margin > 0.05
        )
        controlled_replay = self._evaluate_hot_dry_action_proposer_controlled_replay_shadow(
            state,
            selected_name=str(selected_name),
            selected_post_shape_score=selected_post_shape_score,
            selected_post_shape_control=selected_post_shape_control,
            selected_post_shape_details=selected_post_shape_details,
            post_shape_rows=post_shape_rows,
            gate_reason=post_shape_gate_reason,
        )
        return {
            "enabled": True,
            "shadow_only": True,
            "actual_candidate_count": int(actual_candidate_count),
            "candidate_count": int(len(candidate_records)),
            "selected_name": str(selected_name),
            "selected_score": float(selected_score) if np.isfinite(selected_score) else None,
            "selected_action": selected_action,
            "selected_score_terms": selected_score_terms,
            "candidates": candidate_records,
            "post_shape_enabled": True,
            "post_shape_selected_score": selected_post_shape_score,
            "post_shape_best_name": str(post_shape_best.get("name", "") or ""),
            "post_shape_best_score": post_shape_best_score,
            "post_shape_selected_action": post_shape_selected_action,
            "post_shape_selected_score_terms": post_shape_selected_score_terms,
            "post_shape_best_action": post_shape_best.get("post_shape_action", {}),
            "post_shape_best_score_terms": post_shape_best_score_terms,
            "post_shape_best_events": post_shape_best.get("post_shape_events", {}),
            "post_shape_best_eligible": bool(post_shape_best_row.get("eligible", False)) if post_shape_best_row else False,
            "post_shape_best_is_proposer": bool(str(post_shape_best.get("name", "") or "").startswith("shadow_hot_dry_")),
            "post_shape_raw_best_name": str(raw_post_shape_best_row.get("name", "") or ""),
            "post_shape_raw_best_score": (
                float(raw_post_shape_best_row.get("score", 0.0) or 0.0)
                if raw_post_shape_best_row
                else None
            ),
            "post_shape_raw_best_is_proposer": bool(
                str(raw_post_shape_best_row.get("name", "") or "").startswith("shadow_hot_dry_")
            ),
            "post_shape_raw_best_alignment": raw_post_shape_alignment,
            "post_shape_raw_best_margin": raw_post_shape_margin,
            "post_shape_raw_would_switch": raw_post_shape_would_switch,
            "post_shape_raw_unsafe_conflict": bool(raw_post_shape_would_switch and raw_post_shape_alignment == "unsafe_conflict"),
            "post_shape_action_delta_terms": post_shape_action_delta_terms,
            "post_shape_score_delta_terms": post_shape_score_delta_terms,
            "post_shape_safety_gate_reason": post_shape_gate_reason,
            "post_shape_alignment": post_shape_alignment,
            "post_shape_margin": float(post_shape_margin),
            "post_shape_would_switch": post_shape_would_switch,
            "post_shape_unsafe_conflict": bool(post_shape_would_switch and post_shape_alignment == "unsafe_conflict"),
            "post_shape_dry_benefit": bool(post_shape_would_switch and post_shape_alignment == "dry_benefit"),
            "post_shape_safe_relief": bool(post_shape_would_switch and post_shape_alignment == "safe_relief"),
            "target_tracking_calibration": target_tracking_calibration,
            "hot_dry_action_proposer_controlled_replay": controlled_replay,
            "hot_dry_features": {
                "active": bool(hot_dry_features.get("active", False)),
                "semantic_active": bool(hot_dry_shadow_features.get("semantic_active", False)),
                "shadow_active": bool(hot_dry_shadow_features.get("shadow_active", False)),
                "shadow_gate_reason": str(hot_dry_shadow_features.get("shadow_gate_reason", "none") or "none"),
                "dry_pressure": bool(hot_dry_features.get("dry_pressure", False)),
                "hot_pressure": bool(hot_dry_shadow_features.get("hot_pressure", False)),
                "strong_rad": bool(hot_dry_shadow_features.get("strong_rad", False)),
                "extreme_dew": bool(hot_dry_shadow_features.get("extreme_dew", False)),
                "enabled": bool(hot_dry_features.get("enabled", False)),
            },
            "hot_dry_candidates_enabled": bool(
                getattr(self.config, "rspc_hot_dry_candidates_enabled", False)
            ),
            "hot_dry_candidate_count": int(hot_dry_count),
            "hot_dry_proposer_enabled": True,
            "hot_dry_proposer_active": bool(hot_dry_shadow_features.get("shadow_active", False)),
            "hot_dry_proposer_candidate_count": int(hot_dry_shadow_count),
            "hot_dry_proposer_gate_reason": str(hot_dry_shadow_features.get("shadow_gate_reason", "none") or "none"),
        }

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

        for name, control, rationale in self._build_rspc_hot_dry_candidates(state, env_control):
            candidates.append(FallbackCandidate(name, control, rationale=rationale))

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

    def _fallback_candidate_safety_prediction(
        self,
        state,
        control: np.ndarray,
    ) -> Dict[str, Any]:
        target_temp: Optional[float] = None
        target_co2: Optional[float] = None
        target_rh: Optional[float] = None
        if isinstance(getattr(self, "current_plan", None), dict):
            try:
                target_temp = self._get_plan_target(self.current_plan, "target_temp", state)
                target_co2 = self._get_plan_target(self.current_plan, "target_co2", state)
                target_rh = self._get_plan_target(self.current_plan, "target_rh", state)
            except Exception:
                target_temp = None
                target_co2 = None
                target_rh = None
        _, _, target_rh, template_record = self._apply_profile_template_patch_targets(
            state,
            np.asarray(control, dtype=np.float32),
            target_temp,
            target_co2,
            target_rh,
            source="fallback_post_selection_veto_prediction",
            update_last=False,
        )
        shaped, info = apply_tomato_safety_v2(
            state,
            np.asarray(control, dtype=np.float32),
            config=self.config,
            target_rh=target_rh,
        )
        reason_text = " ".join(str(x) for x in info.get("reasons", []) or []).lower()
        hard_safety = bool(
            info.get("applied", False)
            and any(token in reason_text for token in ("hard", "dew", "canopy", "extreme", "hot_temperature"))
        )
        before = np.asarray(control, dtype=np.float32)
        after = np.asarray(shaped, dtype=np.float32)
        rewrite_fields = [
            name
            for idx, name in enumerate(("heat", "co2", "screen", "vent", "lamp", "shade"))
            if idx < int(after.size) and idx < int(before.size) and abs(float(after[idx]) - float(before[idx])) > 1e-6
        ]
        return {
            "hard_safety_rewrite": bool(hard_safety),
            "tomato_safety_applied": bool(info.get("applied", False)),
            "safety_reasons": list(info.get("reasons", []) or []),
            "rewrite_fields": rewrite_fields,
            "delta_abs_sum": float(np.sum(np.abs(after - before))),
            "template_patch": template_record,
            "before": [float(x) for x in before.tolist()],
            "after": [float(x) for x in after.tolist()],
        }

    def _recovery_anchor_is_enabled(self) -> bool:
        return bool(
            getattr(self.config, "profile_template_patch_enabled", False)
            and getattr(self.config, "fallback_post_selection_veto_enabled", False)
            and getattr(self.config, "recovery_anchor_enabled", False)
            and str(getattr(self.config, "recovery_anchor_source", "") or "")
            == "tomato_safety_projected_anchor"
        )

    def _recovery_anchor_record_base(self) -> Dict[str, Any]:
        enabled = self._recovery_anchor_is_enabled()
        return {
            "recovery_anchor_enabled": bool(enabled),
            "recovery_anchor_applied": False,
            "recovery_anchor_source": str(getattr(self.config, "recovery_anchor_source", "") or ""),
            "recovery_anchor_reason": "disabled" if not enabled else "",
            "recovery_anchor_action": {},
            "recovery_anchor_prediction": {},
            "recovery_anchor_no_compatible_existing_candidate": False,
            "recovery_anchor_selected_before": "",
            "recovery_anchor_selected_after": "",
            "recovery_anchor_safety_reasons": [],
            "recovery_anchor_rewrite_fields": [],
        }

    def _build_recovery_anchor_candidate(
        self,
        state,
        selected: FallbackCandidate,
        selected_prediction: Mapping[str, Any],
    ) -> Tuple[Optional[FallbackCandidate], Dict[str, Any]]:
        record = self._recovery_anchor_record_base()
        if not record["recovery_anchor_enabled"]:
            return None, record
        if not bool(selected_prediction.get("hard_safety_rewrite", False)):
            record["recovery_anchor_reason"] = "selected_candidate_compatible"
            return None, record

        projected_values = selected_prediction.get("after")
        if not isinstance(projected_values, (list, tuple, np.ndarray)):
            record["recovery_anchor_reason"] = "projected_action_missing"
            return None, record
        projected = np.clip(np.asarray(projected_values, dtype=np.float32), 0.0, 1.0)
        if projected.size < 6:
            record["recovery_anchor_reason"] = "projected_action_missing"
            return None, record

        recovery_prediction = self._fallback_candidate_safety_prediction(state, projected)
        record.update(
            {
                "recovery_anchor_no_compatible_existing_candidate": True,
                "recovery_anchor_selected_before": selected.name,
                "recovery_anchor_selected_after": "tomato_safety_projected_anchor",
                "recovery_anchor_action": self._control_terms(projected),
                "recovery_anchor_prediction": {
                    "hard_safety_rewrite": bool(recovery_prediction.get("hard_safety_rewrite", False)),
                    "tomato_safety_applied": bool(recovery_prediction.get("tomato_safety_applied", False)),
                    "safety_reasons": list(recovery_prediction.get("safety_reasons", []) or []),
                    "rewrite_fields": list(recovery_prediction.get("rewrite_fields", []) or []),
                },
                "recovery_anchor_safety_reasons": list(recovery_prediction.get("safety_reasons", []) or []),
                "recovery_anchor_rewrite_fields": list(recovery_prediction.get("rewrite_fields", []) or []),
            }
        )
        if bool(recovery_prediction.get("hard_safety_rewrite", False)):
            record["recovery_anchor_reason"] = "recovery_anchor_still_hard_safety_rewrite"
            return None, record

        recovery = FallbackCandidate(
            "tomato_safety_projected_anchor",
            projected.astype(np.float32),
            score=float(selected.score),
            details={
                "source": "recovery_anchor",
                "selected_before": selected.name,
                "no_compatible_existing_candidate": True,
            },
            rationale="Tomato Safety projected action exposed as an explicit recovery anchor",
        )
        record.update(
            {
                "recovery_anchor_applied": True,
                "recovery_anchor_reason": "fallback_veto_no_alternative_projected_anchor",
            }
        )
        return recovery, record

    def _apply_recovery_anchor_to_selected_control(
        self,
        state,
        *,
        source_name: str,
        control: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        base_control = np.clip(np.asarray(control, dtype=np.float32), 0.0, 1.0)
        record: Dict[str, Any] = {
            "enabled": bool(
                getattr(self.config, "profile_template_patch_enabled", False)
                and getattr(self.config, "fallback_post_selection_veto_enabled", False)
            ),
            "applied": False,
            "selected_before": str(source_name or "selected_control"),
            "selected_after": str(source_name or "selected_control"),
            "reason": "selected_candidate_compatible",
            "no_alternative": False,
            "candidate_predictions": [],
        }
        record.update(self._recovery_anchor_record_base())
        if state is None or not self._recovery_anchor_is_enabled():
            return base_control, record

        selected = FallbackCandidate(
            str(source_name or "selected_control"),
            base_control,
            score=0.0,
            details={"source": "direct_post_selection_contract"},
            rationale="direct selected fallback source",
        )
        prediction = self._fallback_candidate_safety_prediction(state, selected.control)
        record.update(
            {
                "candidate_predictions": [
                    {
                        "name": selected.name,
                        "score": 0.0,
                        "hard_safety_rewrite": bool(prediction.get("hard_safety_rewrite", False)),
                        "tomato_safety_applied": bool(prediction.get("tomato_safety_applied", False)),
                        "safety_reasons": list(prediction.get("safety_reasons", []) or []),
                        "rewrite_fields": list(prediction.get("rewrite_fields", []) or []),
                    }
                ],
                "selected_before_hard_safety_rewrite": bool(prediction.get("hard_safety_rewrite", False)),
                "selected_before_safety_reasons": list(prediction.get("safety_reasons", []) or []),
                "selected_before_rewrite_fields": list(prediction.get("rewrite_fields", []) or []),
            }
        )
        if not bool(prediction.get("hard_safety_rewrite", False)):
            return selected.control, record

        record.update({"reason": "fallback_veto_no_alternative", "no_alternative": True})
        recovery, recovery_record = self._build_recovery_anchor_candidate(state, selected, prediction)
        record.update(recovery_record)
        if recovery is None:
            return selected.control, record
        recovery_prediction = dict(recovery_record.get("recovery_anchor_prediction", {}) or {})
        record.update(
            {
                "applied": True,
                "reason": "fallback_post_selection_recovery_anchor",
                "no_alternative": False,
                "selected_after": recovery.name,
                "selected_after_hard_safety_rewrite": bool(recovery_prediction.get("hard_safety_rewrite", False)),
                "selected_after_safety_reasons": list(recovery_prediction.get("safety_reasons", []) or []),
                "selected_after_rewrite_fields": list(recovery_prediction.get("rewrite_fields", []) or []),
            }
        )
        return recovery.control, record

    def _record_fallback_contract_provenance(self, veto_record: Mapping[str, Any]) -> None:
        if not bool(getattr(self.config, "profile_template_patch_record_provenance", True)):
            return
        existing = dict(getattr(self, "last_profile_template_patch", {}) or {})
        existing.update(
            {
                "enabled": bool(getattr(self.config, "profile_template_patch_enabled", False)),
                "fallback_veto_enabled": bool(getattr(self.config, "fallback_post_selection_veto_enabled", False)),
                "fallback_veto_applied": bool(veto_record.get("applied", False)),
                "fallback_veto_no_alternative": bool(veto_record.get("no_alternative", False)),
                "fallback_veto_reason": str(veto_record.get("reason", "")),
                "fallback_veto_selected_before": str(veto_record.get("selected_before", "")),
                "fallback_veto_selected_after": str(veto_record.get("selected_after", "")),
                "fallback_veto_safety_reasons": list(veto_record.get("selected_before_safety_reasons", []) or []),
                "fallback_veto_rewrite_fields": list(veto_record.get("selected_before_rewrite_fields", []) or []),
                "recovery_anchor_enabled": bool(veto_record.get("recovery_anchor_enabled", False)),
                "recovery_anchor_applied": bool(veto_record.get("recovery_anchor_applied", False)),
                "recovery_anchor_source": str(veto_record.get("recovery_anchor_source", "")),
                "recovery_anchor_reason": str(veto_record.get("recovery_anchor_reason", "")),
                "recovery_anchor_action": dict(veto_record.get("recovery_anchor_action", {}) or {}),
                "recovery_anchor_prediction": dict(veto_record.get("recovery_anchor_prediction", {}) or {}),
                "recovery_anchor_no_compatible_existing_candidate": bool(
                    veto_record.get("recovery_anchor_no_compatible_existing_candidate", False)
                ),
                "recovery_anchor_selected_before": str(veto_record.get("recovery_anchor_selected_before", "")),
                "recovery_anchor_selected_after": str(veto_record.get("recovery_anchor_selected_after", "")),
                "recovery_anchor_safety_reasons": list(veto_record.get("recovery_anchor_safety_reasons", []) or []),
                "recovery_anchor_rewrite_fields": list(veto_record.get("recovery_anchor_rewrite_fields", []) or []),
            }
        )
        self.last_profile_template_patch = existing

    def _apply_fallback_post_selection_veto(
        self,
        state,
        candidates: Sequence[FallbackCandidate],
    ) -> Tuple[FallbackCandidate, Dict[str, Any]]:
        enabled = bool(
            getattr(self.config, "profile_template_patch_enabled", False)
            and getattr(self.config, "fallback_post_selection_veto_enabled", False)
        )
        if not candidates:
            raise ValueError("fallback post-selection veto requires at least one candidate")
        best = candidates[0]
        record: Dict[str, Any] = {
            "enabled": enabled,
            "applied": False,
            "selected_before": best.name,
            "selected_after": best.name,
            "reason": "disabled" if not enabled else "selected_candidate_compatible",
            "no_alternative": False,
            "candidate_predictions": [],
        }
        record.update(self._recovery_anchor_record_base())
        if not enabled or state is None:
            return best, record

        predictions: List[Tuple[FallbackCandidate, Dict[str, Any]]] = []
        for candidate in candidates:
            prediction = self._fallback_candidate_safety_prediction(state, candidate.control)
            predictions.append((candidate, prediction))
            record["candidate_predictions"].append(
                {
                    "name": candidate.name,
                    "score": float(candidate.score),
                    "hard_safety_rewrite": bool(prediction.get("hard_safety_rewrite", False)),
                    "tomato_safety_applied": bool(prediction.get("tomato_safety_applied", False)),
                    "safety_reasons": list(prediction.get("safety_reasons", []) or []),
                    "rewrite_fields": list(prediction.get("rewrite_fields", []) or []),
                }
            )

        best_prediction = predictions[0][1]
        record["selected_before_hard_safety_rewrite"] = bool(best_prediction.get("hard_safety_rewrite", False))
        record["selected_before_safety_reasons"] = list(best_prediction.get("safety_reasons", []) or [])
        record["selected_before_rewrite_fields"] = list(best_prediction.get("rewrite_fields", []) or [])
        if not bool(best_prediction.get("hard_safety_rewrite", False)):
            return best, record

        compatible = next(
            (
                (candidate, prediction)
                for candidate, prediction in predictions[1:]
                if not bool(prediction.get("hard_safety_rewrite", False))
            ),
            None,
        )
        if compatible is None:
            record.update(
                {
                    "reason": "fallback_veto_no_alternative",
                    "no_alternative": True,
                    "selected_after": best.name,
                }
            )
            recovery, recovery_record = self._build_recovery_anchor_candidate(state, best, best_prediction)
            record.update(recovery_record)
            if recovery is not None:
                recovery_prediction = dict(recovery_record.get("recovery_anchor_prediction", {}) or {})
                record.update(
                    {
                        "applied": True,
                        "reason": "fallback_post_selection_recovery_anchor",
                        "no_alternative": False,
                        "selected_after": recovery.name,
                        "selected_after_hard_safety_rewrite": bool(
                            recovery_prediction.get("hard_safety_rewrite", False)
                        ),
                        "selected_after_safety_reasons": list(
                            recovery_prediction.get("safety_reasons", []) or []
                        ),
                        "selected_after_rewrite_fields": list(
                            recovery_prediction.get("rewrite_fields", []) or []
                        ),
                    }
                )
                return recovery, record
            return best, record

        selected, selected_prediction = compatible
        record.update(
            {
                "applied": True,
                "reason": "fallback_post_selection_hard_safety_rewrite",
                "selected_after": selected.name,
                "selected_after_hard_safety_rewrite": bool(selected_prediction.get("hard_safety_rewrite", False)),
                "selected_after_safety_reasons": list(selected_prediction.get("safety_reasons", []) or []),
                "selected_after_rewrite_fields": list(selected_prediction.get("rewrite_fields", []) or []),
            }
        )
        return selected, record

    def _select_fallback_control(self, state=None, analysis: Optional[Dict[str, Any]] = None) -> np.ndarray:
        env = self.interface.env
        env_control = np.asarray(getattr(env, "u", np.zeros(getattr(env, "nu", 6))), dtype=np.float32)
        strategy = str(getattr(self.config, "fallback_strategy", "adaptive") or "adaptive").lower()

        rule_control = self._predict_rule_control()
        if strategy == "rule" and rule_control is not None:
            control, veto_record = self._apply_recovery_anchor_to_selected_control(
                state,
                source_name="rule_controller",
                control=np.asarray(rule_control, dtype=np.float32),
            )
            self.last_fallback_selection = {
                "source": str(veto_record.get("selected_after", "rule_controller")),
                "score": None,
                "candidates": [],
                "fallback_candidate_scores": [],
                "selected_fallback_candidate": str(veto_record.get("selected_after", "rule_controller")),
                "selected_fallback_score_breakdown": {},
                "profile_template_patch_fallback_veto": veto_record,
            }
            self._record_fallback_contract_provenance(veto_record)
            return np.asarray(control, dtype=np.float32).copy()

        if strategy == "recent" and self.config.fallback_use_last_control and self.last_control is not None:
            control, veto_record = self._apply_recovery_anchor_to_selected_control(
                state,
                source_name="recent_anchor",
                control=np.asarray(self.last_control, dtype=np.float32),
            )
            self.last_fallback_selection = {
                "source": str(veto_record.get("selected_after", "recent_anchor")),
                "score": None,
                "candidates": [],
                "fallback_candidate_scores": [],
                "selected_fallback_candidate": str(veto_record.get("selected_after", "recent_anchor")),
                "selected_fallback_score_breakdown": {},
                "profile_template_patch_fallback_veto": veto_record,
            }
            self._record_fallback_contract_provenance(veto_record)
            return np.asarray(control, dtype=np.float32).copy()

        if state is not None and bool(getattr(self.config, "fallback_candidate_scoring", True)):
            candidates = self._build_fallback_candidates(state, analysis=analysis)
            if candidates:
                best, veto_record = self._apply_fallback_post_selection_veto(state, candidates)
                candidate_scores = [
                    {
                        "name": item.name,
                        "score": float(item.score),
                        "rationale": item.rationale,
                        "action": self._control_terms(item.control),
                        "score_breakdown": self._score_term_subset(item.details),
                    }
                    for item in candidates
                ]
                self.last_fallback_selection = {
                    "source": best.name,
                    "score": float(best.score),
                    "rationale": best.rationale,
                    "details": best.details,
                    "selected_fallback_candidate": best.name,
                    "selected_fallback_score_breakdown": self._score_term_subset(best.details),
                    "fallback_candidate_scores": candidate_scores,
                    "candidates": [
                        {
                            "name": item.name,
                            "score": float(item.score),
                            "rationale": item.rationale,
                            "details": item.details,
                            "score_breakdown": self._score_term_subset(item.details),
                        }
                        for item in candidates[:5]
                    ],
                    "profile_template_patch_fallback_veto": veto_record,
                }
                self._record_fallback_contract_provenance(veto_record)
                return np.clip(best.control, 0.0, 1.0).astype(np.float32)

        if state is not None:
            base_control = rule_control if rule_control is not None else env_control
            adaptive_control = self._generate_state_adaptive_control(state, base_control)
            if self.config.fallback_use_last_control and self.last_control is not None:
                blend = float(np.clip(self.config.fallback_recent_blend, 0.0, 1.0))
                adaptive_control = (1.0 - blend) * adaptive_control + blend * np.asarray(self.last_control, dtype=np.float32)
            self.last_fallback_selection = {
                "source": "adaptive_legacy",
                "score": None,
                "candidates": [],
                "fallback_candidate_scores": [],
                "selected_fallback_candidate": "adaptive_legacy",
                "selected_fallback_score_breakdown": {},
            }
            adaptive_control = np.clip(adaptive_control, 0.0, 1.0).astype(np.float32)
            control, veto_record = self._apply_recovery_anchor_to_selected_control(
                state,
                source_name="adaptive_legacy",
                control=adaptive_control,
            )
            self.last_fallback_selection.update(
                {
                    "source": str(veto_record.get("selected_after", "adaptive_legacy")),
                    "selected_fallback_candidate": str(veto_record.get("selected_after", "adaptive_legacy")),
                    "profile_template_patch_fallback_veto": veto_record,
                }
            )
            self._record_fallback_contract_provenance(veto_record)
            return np.asarray(control, dtype=np.float32).copy()

        if rule_control is not None:
            control, veto_record = self._apply_recovery_anchor_to_selected_control(
                state,
                source_name="rule_controller",
                control=np.asarray(rule_control, dtype=np.float32),
            )
            self.last_fallback_selection = {
                "source": str(veto_record.get("selected_after", "rule_controller")),
                "score": None,
                "candidates": [],
                "fallback_candidate_scores": [],
                "selected_fallback_candidate": str(veto_record.get("selected_after", "rule_controller")),
                "selected_fallback_score_breakdown": {},
                "profile_template_patch_fallback_veto": veto_record,
            }
            self._record_fallback_contract_provenance(veto_record)
            return np.asarray(control, dtype=np.float32).copy()

        if self.config.fallback_use_last_control and self.last_control is not None:
            control, veto_record = self._apply_recovery_anchor_to_selected_control(
                state,
                source_name="recent_anchor",
                control=np.asarray(self.last_control, dtype=np.float32),
            )
            self.last_fallback_selection = {
                "source": str(veto_record.get("selected_after", "recent_anchor")),
                "score": None,
                "candidates": [],
                "fallback_candidate_scores": [],
                "selected_fallback_candidate": str(veto_record.get("selected_after", "recent_anchor")),
                "selected_fallback_score_breakdown": {},
                "profile_template_patch_fallback_veto": veto_record,
            }
            self._record_fallback_contract_provenance(veto_record)
            return np.asarray(control, dtype=np.float32).copy()

        self.last_fallback_selection = {
            "source": "hold_current",
            "score": None,
            "candidates": [],
            "fallback_candidate_scores": [],
            "selected_fallback_candidate": "hold_current",
            "selected_fallback_score_breakdown": {},
        }
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
        pre_guardrail_control = np.asarray(self.pending_control, dtype=np.float32)
        final_control = apply_safety_guardrails(state, pre_guardrail_control)
        self._record_post_guardrail_runtime_provenance(
            state,
            hook_id="apply_safety_guardrails",
            source_function="apply_safety_guardrails",
            pre_rule_action=pre_guardrail_control,
            post_rule_action=final_control,
        )
        pre_humidity_memory_control = np.asarray(final_control, dtype=np.float32)
        final_control = self._apply_humidity_memory_final_shape(state, final_control)
        humidity_memory_info = {}
        if isinstance(getattr(self, "last_rollout_selection", None), dict):
            humidity_memory_info = dict(
                self.last_rollout_selection.get("humidity_memory_post_guardrail_shape", {}) or {}
            )
        self._record_post_guardrail_runtime_provenance(
            state,
            hook_id="humidity_memory_final_shape",
            source_function="_apply_humidity_memory_final_shape",
            pre_rule_action=pre_humidity_memory_control,
            post_rule_action=final_control,
            info=humidity_memory_info,
        )
        pre_tomato_safety_control = np.asarray(final_control, dtype=np.float32)
        final_control = self._apply_tomato_safety_v2(state, final_control)
        self._record_post_guardrail_runtime_provenance(
            state,
            hook_id="tomato_safety_v2_wrapper",
            source_function="_apply_tomato_safety_v2",
            pre_rule_action=pre_tomato_safety_control,
            post_rule_action=final_control,
            info=getattr(self, "last_tomato_safety_v2", {}) or {},
        )
        self._store_post_guardrail_final_risk_shadow(
            state,
            np.asarray(self.pending_control, dtype=np.float32),
            np.asarray(final_control, dtype=np.float32),
        )
        self.pending_control = None
        return np.asarray(final_control, dtype=np.float32)

    def _apply_tomato_safety_v2(
        self,
        state,
        control: np.ndarray,
        target_rh: Optional[float] = None,
    ) -> np.ndarray:
        target_temp: Optional[float] = None
        target_co2: Optional[float] = None
        if isinstance(getattr(self, "current_plan", None), dict):
            try:
                if target_rh is None:
                    target_rh = self._get_plan_target(self.current_plan, "target_rh", state)
                target_temp = self._get_plan_target(self.current_plan, "target_temp", state)
                target_co2 = self._get_plan_target(self.current_plan, "target_co2", state)
            except Exception:
                target_temp = None
                target_co2 = None
                target_rh = target_rh
        if bool(getattr(self.config, "profile_feasibility_gate_enabled", False)):
            target_temp, target_co2, target_rh, gate_record = self._apply_profile_feasibility_gate_targets(
                state,
                np.asarray(control, dtype=np.float32),
                target_temp,
                target_co2,
                target_rh,
                source="tomato_safety_v2_target",
                update_last=True,
            )
        if bool(getattr(self.config, "profile_template_patch_enabled", False)):
            target_temp, target_co2, target_rh, _template_record = self._apply_profile_template_patch_targets(
                state,
                np.asarray(control, dtype=np.float32),
                target_temp,
                target_co2,
                target_rh,
                source="tomato_safety_v2_target",
                update_last=True,
            )
        shaped, info = apply_tomato_safety_v2(
            state,
            np.asarray(control, dtype=np.float32),
            config=self.config,
            target_rh=target_rh,
        )
        self.last_tomato_safety_v2 = info
        suppressed_replan = dict(getattr(self, "last_tomato_safety_v2_suppressed_replan", {"applied": False}))
        info["suppressed_replan"] = suppressed_replan
        info["suppressed_replan_step_count"] = int(getattr(self, "tomato_safety_v2_suppressed_replan_steps", 0))
        if isinstance(getattr(self, "last_rollout_selection", None), dict):
            self.last_rollout_selection["tomato_safety_v2"] = info
        return np.asarray(shaped, dtype=np.float32)

    def _should_suppress_tomato_v2_replay_emergency_replan(self) -> bool:
        return (
            bool(getattr(self.config, "tomato_safety_v2_enabled", False))
            and bool(getattr(self.config, "tomato_safety_v2_suppress_replay_emergency_replans", True))
            and str(getattr(self.config, "plan_cache_mode", "off")) == "replay"
            and str(getattr(self.config, "plan_cache_key_policy", "prompt")) == "scenario_timestep"
            and bool(getattr(self.config, "plan_cache_strict", False))
        )

    def _should_suppress_strict_replay_emergency_replan(self) -> bool:
        return (
            bool(getattr(self.config, "strict_replay_suppress_uncached_emergency_replans", True))
            and str(getattr(self.config, "plan_cache_mode", "off")) == "replay"
            and str(getattr(self.config, "plan_cache_key_policy", "prompt")) == "scenario_timestep"
            and bool(getattr(self.config, "plan_cache_strict", False))
        )

    def _strict_replay_plan_key_for_step(self, state) -> Optional[str]:
        plan_cache = getattr(self, "plan_cache", None)
        if plan_cache is None:
            return None
        try:
            cache_key = plan_cache.make_key(
                env_id=str(getattr(self, "env_id", "")),
                state_summary={"timestep": int(getattr(state, "timestep", 0))},
                reason="frozen_replan",
                planning_horizon=0,
                config_hash=stable_hash(config_fingerprint(self.config)),
                prompt_hash="scenario_timestep",
                attempt=1,
            )
            return cache_key if plan_cache.get(cache_key) is not None else None
        except Exception:
            return None

    def _tomato_v2_replay_plan_key_for_step(self, state) -> Optional[str]:
        return self._strict_replay_plan_key_for_step(state)

    def _has_strict_replay_plan_for_step(self, state) -> bool:
        return self._strict_replay_plan_key_for_step(state) is not None

    def _has_tomato_v2_replay_plan_for_step(self, state) -> bool:
        return self._tomato_v2_replay_plan_key_for_step(state) is not None

    def _should_follow_strict_replay_frozen_replan_step(self, state) -> bool:
        if not self._should_suppress_strict_replay_emergency_replan():
            return False
        cache_key = self._strict_replay_plan_key_for_step(state)
        if cache_key is None:
            return False
        plan = self.current_plan if isinstance(getattr(self, "current_plan", None), dict) else {}
        event = plan.get("plan_cache_event", {}) if isinstance(plan, dict) else {}
        current_key = str(event.get("key", "")) if isinstance(event, dict) else ""
        return current_key != str(cache_key)

    def _should_follow_tomato_v2_frozen_replan_step(self, state) -> bool:
        if not self._should_suppress_tomato_v2_replay_emergency_replan():
            return False
        cache_key = self._tomato_v2_replay_plan_key_for_step(state)
        if cache_key is None:
            return False
        plan = self.current_plan if isinstance(getattr(self, "current_plan", None), dict) else {}
        event = plan.get("plan_cache_event", {}) if isinstance(plan, dict) else {}
        current_key = str(event.get("key", "")) if isinstance(event, dict) else ""
        return current_key != str(cache_key)

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
        structured_anchor_mode = bool(getattr(self.config, "structured_anchor_parser_enabled", False))
        self._store_structured_anchor_record(self._structured_anchor_empty_record(attempted=False))
        self.last_structured_anchor_profile_bridge = {
            "enabled": bool(getattr(self.config, "structured_anchor_profile_bridge_enabled", False)),
            "shadow_only": bool(getattr(self.config, "structured_anchor_profile_bridge_shadow_only", True)),
            "applied": False,
            "bridgeable": False,
        }
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

            previous_structured_anchor_errors: List[str] = []
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
                if structured_anchor_mode and llm_attempts > 1:
                    if bool(getattr(self.config, "structured_anchor_compact_json_prompt_enabled", False)):
                        retry_hint = (
                            "\nRETRY_JSON_ONLY: previous structured anchor was invalid or empty "
                            f"({','.join(previous_structured_anchor_errors) if previous_structured_anchor_errors else 'unknown'}). "
                            "Return exactly one minified JSON object: "
                            "{\"structured_planning_anchor\":{\"profile_intent\":\"<intent>\","
                            "\"target_temp\":<number>,\"target_co2\":<number>,\"target_rh\":<number>,"
                            "\"risk_flags\":[\"<risk>\"],\"forbidden_intents\":[\"<forbidden>\"],"
                            "\"planning_horizon_steps\":<integer>,\"confidence\":<0_to_1>}}. "
                            "No markdown, no prose, no tools, no actuator fields."
                        )
                    else:
                        retry_hint = (
                            "\nRetry requirement: return one valid structured_planning_anchor JSON object. "
                            "Do not call set_all_controls or output final-control/actuator fields. "
                            "All required fields must be present and confidence must be in [0, 1]."
                        )
                    prompt_text = f"{prompt_text}{retry_hint}"
                    print(f"[Director] LLM iteration {llm_attempts} using structured-anchor retry prompt")
                elif llm_attempts > 1:
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
                        llm_duration = 0.0
                        print(f"[Director] LLM plan cache hit: mode={cache_mode}, key={cache_key}")
                        if structured_anchor_mode:
                            structured_valid = self._restore_cached_structured_anchor(cache_entry)
                            if structured_valid:
                                self.last_plan_cache_event.update({"status": "hit_structured_anchor"})
                                break
                            print(f"[Director] Cached iteration {llm_attempts} has no valid structured anchor, retrying...")
                            continue
                        llm_action_found = self._restore_cached_action(tools_instance, cache_entry)
                        if llm_action_found:
                            break
                        print(f"[Director] Cached iteration {llm_attempts} has no planning anchor, retrying...")
                        continue
                    if cache_entry is None and cache_mode == "replay" and bool(getattr(plan_cache, "strict", False)):
                        raise RuntimeError(f"LLM plan cache miss in strict replay mode: key={cache_key}")

                start_time = time.time()
                config = {"max_execution_time": self.config.max_execution_time or 60.0}
                if structured_anchor_mode:
                    structured_system = (
                        "Return JSON only. Do not call tools. Do not emit actuator/final-control fields. "
                        "Your sole output is a structured_planning_anchor object for shadow parsing."
                    )
                    direct_response = self.llm.invoke(
                        [SystemMessage(content=structured_system), HumanMessage(content=prompt_text)],
                        config=config,
                    )
                    result_state = {"messages": [direct_response]}
                else:
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

                if structured_anchor_mode:
                    structured_record = self._parse_structured_anchor_output(
                        llm_output,
                        attempt=llm_attempts,
                        prompt_hash=cache_prompt_hash,
                        planning_horizon=planning_horizon,
                        legacy_tool_action_present=not tools_instance.buffered_action.is_empty(),
                        retry_attempt=bool(llm_attempts > 1),
                        retry_source_errors=previous_structured_anchor_errors,
                    )
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
                                llm_action_found=False,
                                tools_instance=tools_instance,
                            ),
                        )
                    if bool(structured_record.get("valid", False)):
                        if plan_cache is not None and cache_key is not None and cache_mode in {"record", "refresh"}:
                            self.last_plan_cache_event.update({"hit": False, "status": "written_structured_anchor"})
                        print(f"[Director] LLM iteration {llm_attempts} produced valid structured planning anchor")
                        llm_action_found = False
                        break
                    if plan_cache is not None and cache_key is not None and cache_mode in {"record", "refresh"}:
                        self.last_plan_cache_event.update({"hit": False, "status": "written_structured_anchor_invalid"})
                    elif plan_cache is not None and cache_key is not None and cache_mode == "replay":
                        self.last_plan_cache_event.update({"hit": False, "status": "miss_fallback"})
                    errors = [str(item) for item in structured_record.get("errors", []) or []]
                    previous_structured_anchor_errors = errors
                    retry_enabled = bool(getattr(self.config, "structured_anchor_retry_invalid_or_empty_enabled", False))
                    retry_max = max(0, int(getattr(self.config, "structured_anchor_retry_max_attempts", 1) or 1))
                    invalid_or_empty = bool(
                        "invalid_json_anchor" in errors
                        or "empty_anchor" in errors
                        or structured_record.get("empty", False)
                    )
                    if retry_enabled:
                        retry_allowed = bool(
                            invalid_or_empty
                            and llm_attempts <= retry_max
                            and llm_attempts < max(1, self.config.max_iterations)
                        )
                    else:
                        retry_allowed = bool(llm_attempts < max(1, self.config.max_iterations))
                    if retry_allowed:
                        print(f"[Director] LLM iteration {llm_attempts} structured anchor invalid, retrying...")
                        continue
                    if plan_cache is not None and cache_key is not None and cache_mode in {"record", "refresh"}:
                        self.last_plan_cache_event.update({"hit": False, "status": "written_structured_anchor_invalid_final"})
                    print(f"[Director] LLM iteration {llm_attempts} structured anchor invalid, using fallback path")
                    break

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

            raw_setpoints = {}
            if not structured_anchor_mode:
                raw_setpoints = getattr(tools_instance, "buffered_setpoints", {}) or {}
            target_temp = raw_setpoints.get("target_temp")
            target_co2 = raw_setpoints.get("target_co2")
            target_rh = raw_setpoints.get("target_rh")
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
                raw_setpoints,
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
                "structured_anchor": dict(getattr(self, "last_structured_anchor", {}) or {}),
            }
            intent_contract = intent_contract_from_setpoint_plan(state, plan)
            plan["intent_contract"] = intent_contract.to_dict()
            plan["intent_contract_diagnostics"] = {
                "schema_version": "intent_contract_mvp_v1",
                "source": "setpoint_adapter",
                "fallback": False,
            }
            try:
                profile_candidates, profile_diagnostics = build_profile_generator_shadow_payload(
                    intent_contract,
                    state,
                    current_plan=plan,
                    horizon_steps=planning_horizon,
                )
                plan["profile_candidates"] = profile_candidates
                plan["profile_generator_diagnostics"] = profile_diagnostics
            except Exception as exc:
                plan["profile_candidates"] = []
                plan["profile_generator_diagnostics"] = {
                    "schema_version": "profile_generator_v1",
                    "shadow_only": True,
                    "error": str(exc),
                }

            if bool(getattr(self.config, "structured_anchor_profile_bridge_enabled", False)):
                try:
                    bridge_record, bridge_candidates = build_structured_anchor_profile_bridge_shadow_payload(
                        dict(getattr(self, "last_structured_anchor", {}) or {}),
                        state,
                        current_plan=plan,
                        model_name=str(getattr(self.config, "model_name", "")),
                        enabled=True,
                        shadow_only=bool(getattr(self.config, "structured_anchor_profile_bridge_shadow_only", True)),
                    )
                except Exception as exc:
                    bridge_record = {
                        "enabled": True,
                        "shadow_only": bool(getattr(self.config, "structured_anchor_profile_bridge_shadow_only", True)),
                        "source": "structured_anchor_profile_bridge_v64",
                        "attempted": bool(getattr(self, "last_structured_anchor", {}).get("attempted", False)),
                        "valid_structured_anchor": bool(getattr(self, "last_structured_anchor", {}).get("valid", False)),
                        "bridgeable": False,
                        "applied": False,
                        "failure_reason": "runtime_bridge_exception",
                        "error": str(exc),
                        "final_control_change": False,
                        "current_plan_modified": False,
                        "low_level_action_generated": False,
                    }
                    bridge_candidates = []
                self.last_structured_anchor_profile_bridge = bridge_record
                if bool(getattr(self.config, "structured_anchor_profile_bridge_record_provenance", True)):
                    plan["structured_anchor_profile_bridge"] = bridge_record
                    plan["structured_anchor_profile_bridge_candidates"] = bridge_candidates

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
                        "structured_anchor": plan.get("structured_anchor", {}),
                        "structured_anchor_profile_bridge": plan.get("structured_anchor_profile_bridge", {}),
                        "structured_anchor_profile_bridge_candidates": plan.get("structured_anchor_profile_bridge_candidates", []),
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
            if (
                str(getattr(self.config, "plan_cache_mode", "off")) == "replay"
                and bool(getattr(self.config, "plan_cache_strict", False))
            ):
                raise
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

    @staticmethod
    def _profile_shadow_target_value(
        profile: Mapping[str, Any],
        target_key: str,
        *,
        created_timestep: int,
        timestep: int,
        default: Optional[float],
    ) -> Optional[float]:
        values = profile.get(target_key) if isinstance(profile, Mapping) else None
        if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
            idx = int(np.clip(int(timestep) - int(created_timestep), 0, len(values) - 1))
            try:
                return float(values[idx])
            except Exception:
                return default
        return default

    @staticmethod
    def _apply_profile_target_tracking_proxy(
        state,
        control: np.ndarray,
        *,
        target_temp: Optional[float],
        target_co2: Optional[float],
        target_rh: Optional[float],
    ) -> np.ndarray:
        shaped = np.asarray(control, dtype=np.float32).copy()
        if target_temp is not None:
            temp_err = float(target_temp) - float(state.temp_air)
            if temp_err > 0.25:
                shaped[0] = np.clip(shaped[0] + 0.10 * temp_err, 0.0, 1.0)
                shaped[3] = np.clip(shaped[3] - 0.06 * temp_err, 0.0, 1.0)
            elif temp_err < -0.25:
                cool_err = -temp_err
                shaped[3] = np.clip(shaped[3] + 0.08 * cool_err, 0.0, 1.0)
                shaped[0] = np.clip(shaped[0] - 0.08 * cool_err, 0.0, 1.0)

        if target_co2 is not None:
            co2_err = float(target_co2) - float(state.co2_air)
            is_day = 6 <= float(state.hour_of_day) <= 18
            if is_day and co2_err > 20 and shaped[3] <= 0.30:
                shaped[1] = np.clip(shaped[1] + 0.0015 * co2_err, 0.0, 1.0)
            elif co2_err < -20:
                shaped[1] = np.clip(shaped[1] - 0.0015 * (-co2_err), 0.0, 1.0)

        if target_rh is not None:
            rh_err = float(target_rh) - float(state.rh_air)
            if rh_err < -5.0:
                shaped[3] = np.clip(shaped[3] + 0.06 * (-rh_err) / 10.0, 0.0, 1.0)
                if float(state.temp_air) < 15.5:
                    shaped[0] = np.clip(shaped[0] + 0.01 * (-rh_err) / 10.0, 0.0, 1.0)
            elif rh_err > 10.0:
                shaped[3] = np.clip(shaped[3] - 0.05 * rh_err / 10.0, 0.0, 1.0)
        return np.clip(shaped, 0.0, 1.0)

    @staticmethod
    def _profile_rspc_shadow_eligibility(candidate_name: str, gate_reason: str) -> Tuple[bool, str]:
        name = str(candidate_name or "").strip()
        gate = str(gate_reason or "none").strip() or "none"
        if gate == "none":
            return True, ""
        banned = {"hot_dry_protect", "co2_day_boost", "lighting_assist"}
        if name in banned:
            return False, f"{gate}:profile_forbidden"
        if gate == "temp_high_gate":
            allowed = {"shade_cooling", "constant_hold", "strict_dehumidify_then_relax"}
            return (name in allowed), ("" if name in allowed else f"{gate}:not_cooling_profile")
        if gate in {"rh_high_gate", "dew_gate", "canopy_gate"}:
            allowed = {
                "strict_dehumidify_then_relax",
                "dawn_predehumidify",
                "cold_humid_recovery",
                "constant_hold",
            }
            return (name in allowed), ("" if name in allowed else f"{gate}:not_dehumidify_profile")
        return True, ""

    @staticmethod
    def _profile_score_delta_terms(
        baseline_details: Mapping[str, Any],
        candidate_details: Mapping[str, Any],
    ) -> Dict[str, float]:
        key_map = {
            "temp": "temp_penalty",
            "rh": "rh_penalty",
            "dry": "dry_penalty",
            "vpd": "vpd_penalty",
            "dew": "dew_penalty",
            "hot_dry": "hot_dry_penalty",
            "energy": "energy_penalty",
        }
        out: Dict[str, float] = {}
        for label, key in key_map.items():
            try:
                out[label] = float(candidate_details.get(key, 0.0)) - float(baseline_details.get(key, 0.0))
            except Exception:
                out[label] = 0.0
        return out

    @staticmethod
    def _profile_action_delta_terms(baseline_control: np.ndarray, candidate_control: np.ndarray) -> Dict[str, float]:
        base = np.asarray(baseline_control, dtype=np.float32)
        cand = np.asarray(candidate_control, dtype=np.float32)
        names = ("heat", "co2", "screen", "vent", "lamp", "shade")
        return {name: float(cand[idx] - base[idx]) for idx, name in enumerate(names)}

    @staticmethod
    def _profile_rspc_gate_risk_worsened(gate_reason: str, score_delta_terms: Mapping[str, float]) -> bool:
        gate = str(gate_reason or "none")
        eps = 1e-6
        if gate == "temp_high_gate":
            return float(score_delta_terms.get("temp", 0.0)) > eps
        if gate == "rh_high_gate":
            return float(score_delta_terms.get("rh", 0.0)) > eps
        if gate in {"dew_gate", "canopy_gate"}:
            return (
                float(score_delta_terms.get("dew", 0.0)) > eps
                or float(score_delta_terms.get("rh", 0.0)) > eps
            )
        return False

    @staticmethod
    def _profile_rspc_action_conflicts(gate_reason: str, action_delta_terms: Mapping[str, float]) -> bool:
        gate = str(gate_reason or "none")
        if gate == "temp_high_gate":
            return (
                float(action_delta_terms.get("heat", 0.0)) > 0.02
                or float(action_delta_terms.get("lamp", 0.0)) > 0.02
                or (
                    float(action_delta_terms.get("vent", 0.0)) < -0.02
                    and float(action_delta_terms.get("shade", 0.0)) < 0.02
                )
            )
        if gate in {"rh_high_gate", "dew_gate", "canopy_gate"}:
            return (
                float(action_delta_terms.get("vent", 0.0)) < -0.02
                or float(action_delta_terms.get("screen", 0.0)) > 0.02
                or float(action_delta_terms.get("co2", 0.0)) > 0.02
                or float(action_delta_terms.get("lamp", 0.0)) > 0.02
            )
        return False

    @staticmethod
    def _profile_rspc_safe_hot_dry_state(state) -> bool:
        temp = float(getattr(state, "temp_air", 20.0))
        rh = float(getattr(state, "rh_air", 70.0))
        vpd = float(calculate_vpd_kpa(temp, rh))
        dew_margin = min(
            float(getattr(state, "dew_margin_air", 3.0)),
            float(getattr(state, "canopy_dew_margin", 3.0)),
        )
        safety = (
            temp >= 32.0
            or float(getattr(state, "temp_violation", 0.0)) > 0.0
            or rh >= 90.0
            or dew_margin < 1.0
        )
        dry = (
            rh < 62.0
            or vpd > 1.60
            or float(getattr(state, "rh_low_violation", 0.0)) > 0.0
            or float(getattr(state, "vpd_high_excess", 0.0)) > 0.0
        )
        return bool(dry and not safety)

    def _profile_rspc_shadow_alignment(
        self,
        state,
        *,
        gate_reason: str,
        candidate_name: str,
        eligible: bool,
        margin: float,
        action_delta_terms: Mapping[str, float],
        score_delta_terms: Mapping[str, float],
    ) -> str:
        if float(margin) <= 1e-6:
            return "neutral_hold"
        gate = str(gate_reason or "none")
        name = str(candidate_name or "")
        if gate != "none":
            if not eligible:
                return "unsafe_conflict"
            if self._profile_rspc_gate_risk_worsened(gate, score_delta_terms):
                return "unsafe_conflict"
            if self._profile_rspc_action_conflicts(gate, action_delta_terms):
                return "unsafe_conflict"
            return "neutral_hold" if name == "constant_hold" else "safe_relief"
        if self._profile_rspc_safe_hot_dry_state(state) and name == "hot_dry_protect":
            return "dry_benefit"
        return "neutral_hold"

    def _evaluate_profile_rspc_shadow(
        self,
        state,
        plan: Mapping[str, Any],
        baseline_control: np.ndarray,
    ) -> Dict[str, Any]:
        candidates = plan.get("profile_candidates", []) if isinstance(plan, Mapping) else []
        if not isinstance(candidates, list) or not candidates:
            return {
                "enabled": False,
                "reason": "missing_profile_candidates",
                "candidate_count": 0,
                "shadow_only": True,
            }

        created_timestep = int(plan.get("created_timestep", int(getattr(state, "timestep", 0))) or 0)
        timestep = int(getattr(state, "timestep", created_timestep))
        baseline = np.clip(np.asarray(baseline_control, dtype=np.float32).copy(), 0.0, 1.0)
        target_temp = self._get_plan_target(dict(plan), "target_temp", state)
        target_co2 = self._get_plan_target(dict(plan), "target_co2", state)
        target_rh = self._get_plan_target(dict(plan), "target_rh", state)
        baseline_raw_score, baseline_raw_details = self._score_fallback_candidate(state, baseline)
        baseline_safety_control, baseline_safety_info = apply_tomato_safety_v2(
            state,
            baseline,
            config=self.config,
            target_rh=target_rh,
        )
        baseline_safety_score, baseline_safety_details = self._score_fallback_candidate(state, baseline_safety_control)

        scorer_diagnostics: Dict[str, Any] = {}
        try:
            _scores, refreshed = score_profile_candidate_payloads(
                candidates,
                plan.get("intent_contract", {}) if isinstance(plan, Mapping) else {},
                state,
                current_plan=plan,
                horizon_steps=int(
                    (plan.get("profile_generator_diagnostics", {}) or {}).get("horizon_steps", 12)
                    if isinstance(plan.get("profile_generator_diagnostics", {}), Mapping)
                    else 12
                ),
            )
            scorer_diagnostics = dict(refreshed)
        except Exception as exc:
            scorer_diagnostics = {"score_refresh_error": str(exc)}
        gate_reason = str(scorer_diagnostics.get("score_safety_gate_reason") or "none")
        if gate_reason not in {"none", "temp_high_gate", "dew_gate", "canopy_gate", "rh_high_gate"}:
            gate_reason = "none"
        scorer_selected = str(scorer_diagnostics.get("score_selected_shadow_profile_name") or "")

        rows: List[Dict[str, Any]] = []
        for item in candidates:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("name") or "").strip()
            profile = item.get("target_profile", {})
            if not name or not isinstance(profile, Mapping):
                continue
            cand_temp = self._profile_shadow_target_value(
                profile,
                "target_temp",
                created_timestep=created_timestep,
                timestep=timestep,
                default=target_temp,
            )
            cand_co2 = self._profile_shadow_target_value(
                profile,
                "target_co2",
                created_timestep=created_timestep,
                timestep=timestep,
                default=target_co2,
            )
            cand_rh = self._profile_shadow_target_value(
                profile,
                "target_rh",
                created_timestep=created_timestep,
                timestep=timestep,
                default=target_rh,
            )
            raw_control = self._apply_profile_target_tracking_proxy(
                state,
                baseline,
                target_temp=cand_temp,
                target_co2=cand_co2,
                target_rh=cand_rh,
            )
            raw_score, raw_details = self._score_fallback_candidate(state, raw_control)
            safety_control, safety_info = apply_tomato_safety_v2(
                state,
                raw_control,
                config=self.config,
                target_rh=cand_rh,
            )
            safety_score, safety_details = self._score_fallback_candidate(state, safety_control)
            eligible, ineligible_reason = self._profile_rspc_shadow_eligibility(name, gate_reason)
            action_delta_terms = self._profile_action_delta_terms(baseline_safety_control, safety_control)
            score_delta_terms = self._profile_score_delta_terms(baseline_safety_details, safety_details)
            if eligible and gate_reason != "none" and self._profile_rspc_gate_risk_worsened(gate_reason, score_delta_terms):
                eligible = False
                ineligible_reason = f"{gate_reason}:gate_risk_worsened"
            if eligible and gate_reason != "none" and self._profile_rspc_action_conflicts(gate_reason, action_delta_terms):
                eligible = False
                ineligible_reason = f"{gate_reason}:action_conflict"
            rows.append(
                {
                    "name": name,
                    "intent": str(item.get("regime") or item.get("intent") or name),
                    "priority": item.get("priority", {}) if isinstance(item.get("priority", {}), Mapping) else {},
                    "constraints": item.get("constraints", {}) if isinstance(item.get("constraints", {}), Mapping) else {},
                    "reason": str(item.get("reason") or ""),
                    "eligible": bool(eligible),
                    "ineligible_reason": str(ineligible_reason),
                    "raw_score": float(raw_score),
                    "safety_score": float(safety_score),
                    "selection_score": float(safety_score),
                    "target_temp": None if cand_temp is None else float(cand_temp),
                    "target_co2": None if cand_co2 is None else float(cand_co2),
                    "target_rh": None if cand_rh is None else float(cand_rh),
                    "raw_control": np.asarray(raw_control, dtype=np.float32).tolist(),
                    "safety_control": np.asarray(safety_control, dtype=np.float32).tolist(),
                    "action_delta_terms": action_delta_terms,
                    "score_delta_terms": score_delta_terms,
                    "raw_details": {key: float(value) for key, value in raw_details.items() if isinstance(value, (int, float))},
                    "safety_details": {
                        key: float(value) for key, value in safety_details.items() if isinstance(value, (int, float))
                    },
                    "tomato_safety_v2_applied": bool(safety_info.get("applied", False)),
                    "tomato_safety_v2_reasons": list(safety_info.get("reasons", []))
                    if isinstance(safety_info.get("reasons", []), list)
                    else [],
                }
            )

        raw_rows = sorted(rows, key=lambda row: (float(row.get("safety_score", 0.0)), str(row.get("name", ""))))
        eligible_rows = [row for row in raw_rows if bool(row.get("eligible", False))]
        raw_best = raw_rows[0] if raw_rows else {}
        best = eligible_rows[0] if eligible_rows else {}
        rows = sorted(rows, key=lambda row: (not bool(row.get("eligible", False)), float(row.get("safety_score", 0.0)), str(row.get("name", ""))))
        margin = float(baseline_safety_score) - float(best.get("safety_score", baseline_safety_score)) if best else 0.0
        raw_margin = float(baseline_safety_score) - float(raw_best.get("safety_score", baseline_safety_score))
        best_alignment = self._profile_rspc_shadow_alignment(
            state,
            gate_reason=gate_reason,
            candidate_name=str(best.get("name", "")),
            eligible=bool(best.get("eligible", False)),
            margin=margin,
            action_delta_terms=best.get("action_delta_terms", {}),
            score_delta_terms=best.get("score_delta_terms", {}),
        )
        raw_alignment = self._profile_rspc_shadow_alignment(
            state,
            gate_reason=gate_reason,
            candidate_name=str(raw_best.get("name", "")),
            eligible=bool(raw_best.get("eligible", False)),
            margin=raw_margin,
            action_delta_terms=raw_best.get("action_delta_terms", {}),
            score_delta_terms=raw_best.get("score_delta_terms", {}),
        )
        return {
            "enabled": True,
            "shadow_only": True,
            "candidate_count": len(rows),
            "eligible_candidate_count": int(len(eligible_rows)),
            "baseline_raw_score": float(baseline_raw_score),
            "baseline_score": float(baseline_safety_score),
            "baseline_control": baseline.tolist(),
            "baseline_safety_control": np.asarray(baseline_safety_control, dtype=np.float32).tolist(),
            "baseline_target_temp": None if target_temp is None else float(target_temp),
            "baseline_target_co2": None if target_co2 is None else float(target_co2),
            "baseline_target_rh": None if target_rh is None else float(target_rh),
            "baseline_raw_details": {
                key: float(value) for key, value in baseline_raw_details.items() if isinstance(value, (int, float))
            },
            "baseline_safety_details": {
                key: float(value) for key, value in baseline_safety_details.items() if isinstance(value, (int, float))
            },
            "baseline_tomato_safety_v2_applied": bool(baseline_safety_info.get("applied", False)),
            "best_profile": str(best.get("name", "")),
            "best_score": float(best.get("safety_score", baseline_safety_score)),
            "best_raw_score": float(best.get("raw_score", baseline_raw_score)),
            "best_safety_score": float(best.get("safety_score", baseline_safety_score)),
            "best_eligible": bool(best.get("eligible", False)),
            "best_ineligible_reason": str(best.get("ineligible_reason", "no_eligible_candidate" if not best else "")),
            "best_action_delta_terms": dict(best.get("action_delta_terms", {})),
            "best_score_delta_terms": dict(best.get("score_delta_terms", {})),
            "margin": float(margin),
            "would_improve": bool(best and margin > 1e-6),
            "raw_best_profile": str(raw_best.get("name", "")),
            "raw_best_score": float(raw_best.get("safety_score", baseline_safety_score)),
            "raw_best_raw_score": float(raw_best.get("raw_score", baseline_raw_score)),
            "raw_best_safety_score": float(raw_best.get("safety_score", baseline_safety_score)),
            "raw_best_eligible": bool(raw_best.get("eligible", False)),
            "raw_best_ineligible_reason": str(raw_best.get("ineligible_reason", "")),
            "raw_best_action_delta_terms": dict(raw_best.get("action_delta_terms", {})),
            "raw_best_score_delta_terms": dict(raw_best.get("score_delta_terms", {})),
            "raw_best_margin": float(raw_margin),
            "raw_best_would_improve": bool(raw_rows and raw_margin > 1e-6),
            "safety_gate_reason": gate_reason,
            "safety_alignment": best_alignment,
            "raw_best_safety_alignment": raw_alignment,
            "scorer_selected_profile": scorer_selected,
            "best_agrees_with_scorer": bool(scorer_selected and str(best.get("name", "")) == scorer_selected),
            "scorer_diagnostics": scorer_diagnostics,
            "top_candidates": rows[:5],
            "raw_top_candidates": raw_rows[:5],
        }

    @staticmethod
    def _profile_action_candidate_control_terms(control: Any) -> Optional[Dict[str, float]]:
        names = ("heat", "co2", "screen", "vent", "lamp", "shade")
        try:
            arr = np.asarray(control, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if arr.size < len(names):
            return None
        arr = arr[: len(names)]
        if not bool(np.all(np.isfinite(arr))):
            return None
        return {name: float(arr[idx]) for idx, name in enumerate(names)}

    @staticmethod
    def _profile_action_candidate_shadow_score_terms(row: Mapping[str, Any]) -> Dict[str, Any]:
        def _num(value: Any, default: float = 0.0) -> float:
            try:
                if value is None:
                    return float(default)
                return float(value)
            except Exception:
                return float(default)

        action_delta = row.get("action_delta_terms", {})
        if not isinstance(action_delta, Mapping):
            action_delta = {}
        safety_details = row.get("safety_details", {})
        if not isinstance(safety_details, Mapping):
            safety_details = {}
        safety_penalty_keys = ("temp_penalty", "rh_penalty", "dry_penalty", "vpd_penalty", "dew_penalty")
        profile_target_error = sum(abs(_num(safety_details.get(key))) for key in safety_penalty_keys)
        action_delta_penalty = sum(
            abs(_num(action_delta.get(key))) for key in ("heat", "co2", "screen", "vent", "lamp", "shade")
        )
        tomato_applied = bool(row.get("tomato_safety_v2_applied", False))
        eligible = bool(row.get("eligible", False))
        return {
            "raw_score": _num(row.get("raw_score")),
            "post_tomato_score": _num(row.get("safety_score")),
            "selection_score": _num(row.get("selection_score", row.get("safety_score"))),
            "profile_target_error": float(profile_target_error),
            "action_delta_penalty": float(action_delta_penalty),
            "tomato_safety_penalty": 1.0 if tomato_applied else 0.0,
            "compatibility_penalty": 0.0 if eligible else 1.0,
            "hard_safety_rewrite_predicted": bool(tomato_applied),
        }

    def _profile_action_candidate_shadow_from_row(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        profile_name = str(row.get("name") or "").strip() or "unknown_profile"
        raw_action = self._profile_action_candidate_control_terms(row.get("raw_control"))
        post_tomato_action = self._profile_action_candidate_control_terms(row.get("safety_control"))
        missing_action = raw_action is None or post_tomato_action is None
        row_eligible = bool(row.get("eligible", False))
        eligible = bool(row_eligible and not missing_action)
        if missing_action:
            rejection_reason = "missing_action_provenance"
        elif eligible:
            rejection_reason = ""
        else:
            rejection_reason = str(
                row.get("ineligible_reason")
                or row.get("rejection_reason")
                or "ineligible_profile_candidate"
            )
        reasons = row.get("tomato_safety_v2_reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        candidate: Dict[str, Any] = {
            "name": f"profile_action:{profile_name}",
            "profile_name": profile_name,
            "candidate_source": "normal_path_profile_candidate",
            "raw_action": raw_action or {},
            "post_tomato_action": post_tomato_action or {},
            "score_terms": self._profile_action_candidate_shadow_score_terms(row),
            "tomato_safety_v2_applied": bool(row.get("tomato_safety_v2_applied", False)),
            "tomato_safety_v2_reasons": [str(item) for item in reasons],
            "eligible": bool(eligible),
            "rejection_reason": str(rejection_reason),
        }
        for key in ("target_temp", "target_co2", "target_rh"):
            value = row.get(key)
            candidate[key] = None if value is None else float(value)
        return candidate

    @staticmethod
    def _profile_action_envelope_num(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _profile_action_envelope_direction(delta: float, *, eps: float = 1e-6) -> str:
        if delta > eps:
            return "increase"
        if delta < -eps:
            return "decrease"
        return "hold"

    def _profile_action_envelope_action_bounds(
        self,
        baseline_action: Mapping[str, Any],
        raw_action: Mapping[str, Any],
        post_tomato_action: Mapping[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        bounds: Dict[str, Dict[str, float]] = {}
        for field in ("heat", "co2", "screen", "vent", "lamp", "shade"):
            values = [
                self._profile_action_envelope_num(baseline_action.get(field)),
                self._profile_action_envelope_num(raw_action.get(field)),
                self._profile_action_envelope_num(post_tomato_action.get(field)),
            ]
            bounds[field] = {
                "min": float(np.clip(min(values), 0.0, 1.0)),
                "max": float(np.clip(max(values), 0.0, 1.0)),
            }
        return bounds

    def _profile_action_envelope_target_direction(
        self,
        row: Mapping[str, Any],
        profile_shadow: Mapping[str, Any],
    ) -> Dict[str, str]:
        directions: Dict[str, str] = {}
        pairs = (
            ("target_temp", "baseline_target_temp"),
            ("target_co2", "baseline_target_co2"),
            ("target_rh", "baseline_target_rh"),
        )
        for target_key, baseline_key in pairs:
            target_value = row.get(target_key)
            baseline_value = profile_shadow.get(baseline_key)
            if target_value is None or baseline_value is None:
                directions[target_key] = "any"
                continue
            directions[target_key] = self._profile_action_envelope_direction(
                self._profile_action_envelope_num(target_value) - self._profile_action_envelope_num(baseline_value)
            )
        return directions

    def _profile_action_envelope_shadow_from_row(
        self,
        row: Mapping[str, Any],
        profile_shadow: Mapping[str, Any],
    ) -> Dict[str, Any]:
        profile_name = str(row.get("name") or "").strip() or "unknown_profile"
        raw_action = self._profile_action_candidate_control_terms(row.get("raw_control"))
        post_tomato_action = self._profile_action_candidate_control_terms(row.get("safety_control"))
        baseline_action = self._profile_action_candidate_control_terms(profile_shadow.get("baseline_safety_control"))
        if baseline_action is None:
            baseline_action = self._profile_action_candidate_control_terms(profile_shadow.get("baseline_control"))
        if baseline_action is None and raw_action is not None:
            baseline_action = dict(raw_action)

        missing_action = raw_action is None or post_tomato_action is None or baseline_action is None
        row_eligible = bool(row.get("eligible", False))
        eligible = bool(row_eligible and not missing_action)
        if missing_action:
            rejection_reason = "missing_action_provenance"
        elif eligible:
            rejection_reason = ""
        else:
            rejection_reason = str(
                row.get("ineligible_reason")
                or row.get("rejection_reason")
                or "ineligible_profile_candidate"
            )

        reasons = row.get("tomato_safety_v2_reasons", [])
        if not isinstance(reasons, list):
            reasons = []
        projected_action = dict(post_tomato_action or {})
        rewrite_delta = {}
        if raw_action is not None and post_tomato_action is not None:
            rewrite_delta = {
                field: float(post_tomato_action[field] - raw_action[field])
                for field in ("heat", "co2", "screen", "vent", "lamp", "shade")
            }
        action_bounds: Dict[str, Dict[str, float]] = {}
        preferred_direction = {field: "any" for field in ("heat", "co2", "screen", "vent", "lamp", "shade")}
        continuity_delta: Dict[str, float] = {}
        if not missing_action and baseline_action is not None and post_tomato_action is not None and raw_action is not None:
            action_bounds = self._profile_action_envelope_action_bounds(baseline_action, raw_action, post_tomato_action)
            preferred_direction = {
                field: self._profile_action_envelope_direction(post_tomato_action[field] - baseline_action[field])
                for field in ("heat", "co2", "screen", "vent", "lamp", "shade")
            }
            continuity_delta = {
                field: float(abs(post_tomato_action[field] - baseline_action[field]))
                for field in ("heat", "co2", "screen", "vent", "lamp", "shade")
            }

        candidate_terms = self._profile_action_candidate_shadow_score_terms(row)
        profile_target_error = self._profile_action_envelope_num(candidate_terms.get("profile_target_error"))
        action_delta_penalty = self._profile_action_envelope_num(candidate_terms.get("action_delta_penalty"))
        compatibility_penalty = self._profile_action_envelope_num(candidate_terms.get("compatibility_penalty"))
        tomato_projection_delta = float(sum(abs(value) for value in rewrite_delta.values()))
        envelope_width_penalty = float(
            sum(abs(bound.get("max", 0.0) - bound.get("min", 0.0)) for bound in action_bounds.values())
        )
        score_terms = {
            "profile_target_alignment": float(-profile_target_error),
            "tomato_projection_delta": tomato_projection_delta,
            "continuity_penalty": action_delta_penalty,
            "envelope_width_penalty": envelope_width_penalty,
            "compatibility_penalty": compatibility_penalty,
            "selection_score": self._profile_action_envelope_num(candidate_terms.get("selection_score")),
        }

        max_rewrite = max((abs(value) for value in rewrite_delta.values()), default=0.0)
        if missing_action:
            compatibility_category = "contract_missing"
        elif max_rewrite > 0.35:
            compatibility_category = "tomato_safety_incompatible"
        elif profile_target_error > 2.0:
            compatibility_category = "profile_target_incompatible"
        elif action_delta_penalty > 1.2:
            compatibility_category = "action_continuity_incompatible"
        elif not eligible:
            compatibility_category = "contract_missing"
        else:
            compatibility_category = "compatibility_shadow_ready"

        priority = row.get("priority", {})
        if not isinstance(priority, Mapping):
            priority = {}
        constraints = row.get("constraints", {})
        if not isinstance(constraints, Mapping):
            constraints = {}

        candidate: Dict[str, Any] = {
            "name": f"profile_action_envelope:{profile_name}",
            "profile_name": profile_name,
            "candidate_source": "normal_path_profile_action_envelope",
            "intent": str(row.get("intent") or row.get("regime") or profile_name),
            "target_direction": self._profile_action_envelope_target_direction(row, profile_shadow),
            "action_bounds": action_bounds,
            "preferred_direction": preferred_direction,
            "priority_terms": {
                "profile_target_alignment": score_terms["profile_target_alignment"],
                "tomato_safety_compatibility": 0.0 if bool(row.get("tomato_safety_v2_applied", False)) else 1.0,
                "action_continuity": float(-action_delta_penalty),
                "hard_safety_precedence": True,
                "profile_priority": dict(priority),
            },
            "continuity_constraints": {
                "previous_action_source": "baseline_safety_action",
                "max_delta_from_previous_action": continuity_delta,
                "profile_constraints": dict(constraints),
            },
            "tomato_safety_projection": {
                "projection_required": True,
                "projected_action": projected_action,
                "projection_applied": bool(row.get("tomato_safety_v2_applied", False)),
                "projection_reasons": [str(item) for item in reasons],
                "rewrite_delta_by_field": rewrite_delta,
            },
            "projected_action": projected_action,
            "score_terms": score_terms,
            "eligible": bool(eligible),
            "rejection_reason": str(rejection_reason),
            "compatibility_category": compatibility_category,
        }
        for key in ("target_temp", "target_co2", "target_rh"):
            value = row.get(key)
            candidate[key] = None if value is None else float(value)
        return candidate

    def _compose_profile_action_envelopes_shadow(self, profile_shadow: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(profile_shadow, Mapping) or not bool(profile_shadow.get("enabled", False)):
            return {
                "enabled": False,
                "shadow_only": True,
                "schema_version": "profile_action_envelope_shadow_v74",
                "reason": str(profile_shadow.get("reason", "profile_rspc_shadow_unavailable"))
                if isinstance(profile_shadow, Mapping)
                else "profile_rspc_shadow_unavailable",
                "final_action_changed": False,
                "candidate_count": 0,
                "eligible_candidate_count": 0,
                "candidates": [],
            }

        try:
            max_candidates = int(getattr(self.config, "profile_action_envelope_shadow_max_candidates", 5) or 0)
        except Exception:
            max_candidates = 5
        max_candidates = max(0, max_candidates)
        candidates: List[Dict[str, Any]] = []
        seen_profiles = set()
        for key in ("top_candidates", "raw_top_candidates"):
            rows = profile_shadow.get(key, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if len(candidates) >= max_candidates:
                    break
                if not isinstance(row, Mapping):
                    continue
                profile_name = str(row.get("name") or "").strip()
                if not profile_name or profile_name in seen_profiles:
                    continue
                seen_profiles.add(profile_name)
                candidates.append(self._profile_action_envelope_shadow_from_row(row, profile_shadow))
            if len(candidates) >= max_candidates:
                break

        eligible_candidates = [item for item in candidates if bool(item.get("eligible", False))]
        best = eligible_candidates[0] if eligible_candidates else (candidates[0] if candidates else {})
        best_score = None
        if isinstance(best, Mapping):
            score_terms = best.get("score_terms", {})
            if isinstance(score_terms, Mapping):
                try:
                    best_score = float(score_terms.get("selection_score", 0.0) or 0.0)
                except Exception:
                    best_score = None
        return {
            "enabled": True,
            "shadow_only": True,
            "schema_version": "profile_action_envelope_shadow_v74",
            "candidate_source": "normal_path_profile_action_envelope",
            "final_action_changed": False,
            "candidate_count": int(len(candidates)),
            "eligible_candidate_count": int(len(eligible_candidates)),
            "best_name": str(best.get("name", "") or "") if isinstance(best, Mapping) else "",
            "best_profile": str(best.get("profile_name", "") or "") if isinstance(best, Mapping) else "",
            "best_score": best_score,
            "best_eligible": bool(best.get("eligible", False)) if isinstance(best, Mapping) else False,
            "best_rejection_reason": str(best.get("rejection_reason", "") or "") if isinstance(best, Mapping) else "",
            "candidates": candidates,
        }

    def _compose_profile_action_candidates_shadow(self, profile_shadow: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(profile_shadow, Mapping) or not bool(profile_shadow.get("enabled", False)):
            return {
                "enabled": False,
                "shadow_only": True,
                "schema_version": "profile_action_candidate_shadow_v69",
                "reason": str(profile_shadow.get("reason", "profile_rspc_shadow_unavailable"))
                if isinstance(profile_shadow, Mapping)
                else "profile_rspc_shadow_unavailable",
                "final_action_changed": False,
                "candidate_count": 0,
                "eligible_candidate_count": 0,
                "candidates": [],
            }

        try:
            max_candidates = int(getattr(self.config, "profile_action_candidate_shadow_max_candidates", 5) or 0)
        except Exception:
            max_candidates = 5
        max_candidates = max(0, max_candidates)
        candidates: List[Dict[str, Any]] = []
        seen_profiles = set()
        for key in ("top_candidates", "raw_top_candidates"):
            rows = profile_shadow.get(key, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if len(candidates) >= max_candidates:
                    break
                if not isinstance(row, Mapping):
                    continue
                profile_name = str(row.get("name") or "").strip()
                if not profile_name or profile_name in seen_profiles:
                    continue
                seen_profiles.add(profile_name)
                candidates.append(self._profile_action_candidate_shadow_from_row(row))
            if len(candidates) >= max_candidates:
                break

        eligible_candidates = [item for item in candidates if bool(item.get("eligible", False))]
        best = eligible_candidates[0] if eligible_candidates else (candidates[0] if candidates else {})
        best_score = None
        if isinstance(best, Mapping):
            score_terms = best.get("score_terms", {})
            if isinstance(score_terms, Mapping):
                try:
                    best_score = float(score_terms.get("selection_score", 0.0) or 0.0)
                except Exception:
                    best_score = None
        return {
            "enabled": True,
            "shadow_only": True,
            "schema_version": "profile_action_candidate_shadow_v69",
            "candidate_source": "normal_path_profile_candidate",
            "final_action_changed": False,
            "candidate_count": int(len(candidates)),
            "eligible_candidate_count": int(len(eligible_candidates)),
            "best_name": str(best.get("name", "") or "") if isinstance(best, Mapping) else "",
            "best_profile": str(best.get("profile_name", "") or "") if isinstance(best, Mapping) else "",
            "best_score": best_score,
            "best_eligible": bool(best.get("eligible", False)) if isinstance(best, Mapping) else False,
            "best_rejection_reason": str(best.get("rejection_reason", "") or "") if isinstance(best, Mapping) else "",
            "candidates": candidates,
        }

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
            for name, hot_dry_control, _rationale in self._build_rspc_hot_dry_candidates(state, nominal_blend):
                candidate_controls.append((name, hot_dry_control, rule_weight))
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
                if bool(getattr(self.config, "profile_feasibility_gate_enabled", False)):
                    adjustment, gate_score_details = self._profile_feasibility_candidate_adjustment(
                        state,
                        clipped,
                        plan if isinstance(plan, Mapping) else {},
                    )
                    score += float(adjustment)
                    candidate_details["profile_feasibility_gate_scoring"] = gate_score_details
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
            rspc_action_scoring = self._rspc_action_scoring_diagnostics(
                state,
                scored_candidates,
                plan=plan if isinstance(plan, Mapping) else {},
                rh_debt=float(rh_debt),
                dehumidify_mode=str(dehumidify_mode or "normal"),
                lamp_budget_remaining=float(lamp_budget_remaining),
                selected_name=str(selected_name),
                selected_control=np.asarray(selected_control, dtype=np.float32),
                selected_score=float(scored_candidates[0][0]),
            )
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
            profile_gate_veto_count = 0
            for item in scored_candidates:
                details_for_veto = item[4] if len(item) > 4 and isinstance(item[4], Mapping) else {}
                gate_scoring = details_for_veto.get("profile_feasibility_gate_scoring", {})
                if isinstance(gate_scoring, Mapping) and gate_scoring.get("hard_safety_vetoed", False):
                    profile_gate_veto_count += 1
            control = np.asarray(selected_control, dtype=np.float32)
            rule_weight = float(selected_rule_weight)
            self.last_rollout_selection = {
                "source": selected_name,
                "score": float(scored_candidates[0][0]),
                "rule_weight": float(rule_weight),
                "details": selected_details,
                "candidates": [
                    {
                        "name": item.get("name", ""),
                        "score": item.get("score"),
                        "rule_weight": item.get("rule_weight", 0.0),
                        "humidity_memory_rejected": item.get("humidity_memory_rejected", False),
                        "humidity_memory_reject_reason": item.get("humidity_memory_reject_reason", ""),
                    }
                    for item in rspc_action_scoring.get("candidates", [])
                    if isinstance(item, dict)
                ],
                "rspc_action_scoring": rspc_action_scoring,
                "expert_prediction": dict(getattr(self, "last_expert_prediction", {})),
                "humidity_memory_prediction": humidity_memory_prediction,
                "humidity_memory_horizon_filter": dict(best_memory_eval or {}),
                "mc_sero_shadow": mc_sero_shadow,
                "profile_feasibility_gate_hard_safety_veto_count": profile_gate_veto_count,
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
                "rspc_action_scoring": {
                    "enabled": False,
                    "shadow_only": True,
                    "reason": "rollout_candidate_sharing_disabled",
                    "candidate_count": 0,
                },
                "expert_prediction": dict(getattr(self, "last_expert_prediction", {})),
                "humidity_memory_prediction": dict(getattr(self, "last_humidity_memory_prediction", {})),
                "mc_sero_shadow": mc_sero_shadow,
            }

        try:
            profile_shadow = self._evaluate_profile_rspc_shadow(
                state,
                plan if isinstance(plan, Mapping) else {},
                np.asarray(control, dtype=np.float32),
            )
        except Exception as exc:
            profile_shadow = {
                "enabled": False,
                "shadow_only": True,
                "reason": "profile_rspc_shadow_error",
                "error": str(exc),
            }
        if isinstance(getattr(self, "last_rollout_selection", None), dict):
            self.last_rollout_selection["profile_rspc_shadow"] = profile_shadow
            if bool(getattr(self.config, "profile_action_candidate_shadow_enabled", False)) and bool(
                getattr(self.config, "profile_action_candidate_shadow_record_provenance", True)
            ):
                try:
                    profile_action_shadow = self._compose_profile_action_candidates_shadow(profile_shadow)
                except Exception as exc:
                    profile_action_shadow = {
                        "enabled": False,
                        "shadow_only": True,
                        "schema_version": "profile_action_candidate_shadow_v69",
                        "reason": "profile_action_candidate_shadow_error",
                        "error": str(exc),
                        "final_action_changed": False,
                        "candidate_count": 0,
                        "eligible_candidate_count": 0,
                        "candidates": [],
                    }
                self.last_rollout_selection["profile_action_candidate_shadow"] = profile_action_shadow
            if bool(getattr(self.config, "profile_action_envelope_shadow_enabled", False)) and bool(
                getattr(self.config, "profile_action_envelope_shadow_record_provenance", True)
            ):
                try:
                    profile_action_envelope_shadow = self._compose_profile_action_envelopes_shadow(profile_shadow)
                except Exception as exc:
                    profile_action_envelope_shadow = {
                        "enabled": False,
                        "shadow_only": True,
                        "schema_version": "profile_action_envelope_shadow_v74",
                        "reason": "profile_action_envelope_shadow_error",
                        "error": str(exc),
                        "final_action_changed": False,
                        "candidate_count": 0,
                        "eligible_candidate_count": 0,
                        "candidates": [],
                    }
                self.last_rollout_selection["profile_action_envelope_shadow"] = profile_action_envelope_shadow

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
            hot_dry_features = self._rspc_hot_dry_features(state)
            if temp_air >= 28.0:
                vent_cap = float(self.config.dry_hot_vent_cap)
            elif temp_air >= 24.0:
                vent_cap = float(self.config.dry_warm_vent_cap)
            else:
                vent_cap = float(self.config.dry_vent_cap)
            if hot_dry_features["active"]:
                vent_cap = max(vent_cap, self._rspc_hot_dry_vent_relief_floor(state, hot_dry_features))
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
                "hot_dry_vent_relief": bool(hot_dry_features["active"]),
                "vent_cap": float(vent_cap),
                "before": before.tolist(),
                "after": np.asarray(control, dtype=np.float32).tolist(),
            }
            self._record_post_guardrail_runtime_provenance(
                state,
                hook_id="dry_recovery_override",
                source_function="_plan_control_step",
                pre_rule_action=before,
                post_rule_action=np.asarray(control, dtype=np.float32),
                info=self.last_rollout_selection.get("dry_recovery_override", {}),
            )
        else:
            self.last_rollout_selection.setdefault("dry_recovery_override", {"applied": False})

        control = self._apply_hot_dry_proposer_control_access(
            state,
            np.asarray(control, dtype=np.float32),
            (
                self.last_rollout_selection.get("rspc_action_scoring", {})
                if isinstance(getattr(self, "last_rollout_selection", None), dict)
                else {}
            ),
        )

        if isinstance(getattr(self, "last_rollout_selection", None), dict):
            scoring_audit = self.last_rollout_selection.get("rspc_action_scoring", {})
            reference_control = None
            if isinstance(scoring_audit, Mapping):
                reference_control = self._control_from_terms(scoring_audit.get("post_shape_selected_action", {}))
                if reference_control is None:
                    reference_control = self._control_from_terms(scoring_audit.get("selected_action", {}))
            self._store_post_guardrail_final_risk_shadow(state, reference_control, np.asarray(control, dtype=np.float32))

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

    def _transition_gate_action_record(self, control: Sequence[float] | np.ndarray) -> Dict[str, float]:
        values = np.asarray(control, dtype=np.float32).reshape(-1)
        return {
            name: float(values[idx]) if idx < int(values.size) else 0.0
            for idx, name in enumerate(ACTION_NAMES)
        }

    def _transition_gate_load_soft_limits(self) -> Dict[str, Dict[str, float]]:
        if self._transition_gate_soft_limits is not None:
            return self._transition_gate_soft_limits

        default_limits = {
            "neutral_or_mild": {name: 0.20 for name in ("heating", "ventilation", "screen", "shading")}
        }
        path_text = str(getattr(self.config, "transition_gate_soft_limit_path", "") or "").strip()
        if not path_text:
            self._transition_gate_soft_limits = default_limits
            self._transition_gate_soft_limit_source = "default_no_path"
            self._transition_gate_soft_limit_error = ""
            return self._transition_gate_soft_limits

        try:
            path = Path(path_text)
            if not path.is_absolute():
                path = Path.cwd() / path
            payload = json.loads(path.read_text(encoding="utf-8"))
            limits: Dict[str, Dict[str, float]] = {}
            for item in payload.get("transition_stats_by_regime", []) or []:
                if str(item.get("label", "")) != "stable_positive":
                    continue
                regime = str(item.get("regime", "neutral_or_mild") or "neutral_or_mild")
                stats = item.get("action_delta_stats", {}) or {}
                limits.setdefault(regime, {})
                for name in ("heating", "ventilation", "screen", "shading"):
                    stat = stats.get(f"u_{name}", {}) or {}
                    raw_limit = _pg_float(stat.get("p95"), 0.20)
                    limits[regime][name] = float(max(0.02, min(raw_limit if raw_limit > 0 else 0.20, 0.20)))
            self._transition_gate_soft_limits = limits or default_limits
            self._transition_gate_soft_limit_source = str(path)
            self._transition_gate_soft_limit_error = ""
        except Exception as exc:
            self._transition_gate_soft_limits = default_limits
            self._transition_gate_soft_limit_source = path_text
            self._transition_gate_soft_limit_error = str(exc)
        return self._transition_gate_soft_limits

    def _transition_gate_limit_for(self, regime: str, action_name: str) -> float:
        limits = self._transition_gate_load_soft_limits()
        if regime in limits and action_name in limits[regime]:
            return float(limits[regime][action_name])
        if "neutral_or_mild" in limits and action_name in limits["neutral_or_mild"]:
            return float(limits["neutral_or_mild"][action_name])
        return 0.20

    def _transition_gate_regime(self, state: Any) -> str:
        temp_air = _pg_state_float(state, "temp_air", 20.0)
        rh_air = _pg_state_float(state, "rh_air", 70.0)
        vpd = float(calculate_vpd_kpa(temp_air, rh_air))
        dew_margin_air = _pg_state_float(state, "dew_margin_air", 3.0)
        canopy_margin = _pg_state_float(state, "canopy_dew_margin", dew_margin_air)
        if bool(getattr(state, "dew_risk", False)) or canopy_margin < 3.0:
            return "dew_or_canopy_risk"
        if bool(getattr(state, "dry_risk", False)) or (vpd >= 2.25 and rh_air <= 45.0):
            return "hot_dry_pressure"
        if rh_air >= 85.0:
            return "high_humidity"
        if temp_air >= 28.0:
            return "warm_or_hot"
        return "neutral_or_mild"

    def _transition_gate_bypass_reasons(self, state: Any) -> List[str]:
        if not bool(getattr(self.config, "transition_gate_hard_safety_bypass", True)):
            return []
        reasons: List[str] = []
        tomato = getattr(self, "last_tomato_safety_v2", {}) or {}
        if isinstance(tomato, Mapping) and bool(tomato.get("applied", False)):
            reasons.append("tomato_safety_v2_applied")
        if bool(getattr(state, "dew_risk", False)):
            reasons.append("dew_risk")
        dew_margin_air = _pg_state_float(state, "dew_margin_air", 3.0)
        canopy_margin = _pg_state_float(state, "canopy_dew_margin", dew_margin_air)
        if canopy_margin < 3.0:
            reasons.append("canopy_dew_margin_lt3")
        return reasons

    def _transition_gate_recent_reversal(self, action_name: str, sign: int) -> bool:
        if sign == 0:
            return False
        history = self._transition_gate_sign_history.get(action_name)
        if history is None:
            return False
        return any(int(previous) != 0 and int(previous) != sign for previous in history)

    def _transition_gate_update_sign_history(self, previous: np.ndarray, current: np.ndarray) -> None:
        for idx, name in enumerate(ACTION_NAMES):
            if idx >= len(previous) or idx >= len(current):
                continue
            delta = float(current[idx] - previous[idx])
            sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
            if sign:
                self._transition_gate_sign_history.setdefault(
                    name,
                    deque(maxlen=max(1, int(getattr(self.config, "transition_gate_reversal_window_steps", 6) or 6))),
                ).append(sign)

    def _apply_transition_gate(self, control: Sequence[float] | np.ndarray, state: Any) -> np.ndarray:
        original = np.asarray(control, dtype=np.float32).reshape(-1).copy()
        if not bool(getattr(self.config, "transition_gate_enabled", False)):
            self.last_transition_gate = {
                "enabled": False,
                "applied": False,
                "bypassed": False,
            }
            return original

        action_size = max(len(ACTION_NAMES), int(original.size))
        if int(original.size) < action_size:
            padded = np.zeros(action_size, dtype=np.float32)
            padded[: int(original.size)] = original
            original = padded

        if self._transition_gate_previous_action is None:
            self._transition_gate_previous_action = original.copy()
            self.last_transition_gate = {
                "enabled": True,
                "applied": False,
                "bypassed": False,
                "reason": "initial_action",
                "regime": self._transition_gate_regime(state),
                "before": self._transition_gate_action_record(original),
                "after": self._transition_gate_action_record(original),
                "soft_limit_source": self._transition_gate_soft_limit_source,
                "soft_limit_error": self._transition_gate_soft_limit_error,
            }
            return original

        previous = self._transition_gate_previous_action.copy()
        bypass_reasons = self._transition_gate_bypass_reasons(state)
        regime = self._transition_gate_regime(state)
        adjusted = original.copy()
        adjusted_fields: List[str] = []
        reversal_fields: List[str] = []
        delta_limited_count = 0
        reversal_projected_count = 0

        if bypass_reasons:
            self._transition_gate_update_sign_history(previous, original)
            self._transition_gate_previous_action = original.copy()
            self.last_transition_gate = {
                "enabled": True,
                "applied": False,
                "bypassed": True,
                "bypass_reasons": list(bypass_reasons),
                "regime": regime,
                "before": self._transition_gate_action_record(original),
                "after": self._transition_gate_action_record(original),
                "soft_limit_source": self._transition_gate_soft_limit_source,
                "soft_limit_error": self._transition_gate_soft_limit_error,
            }
            return original

        for action_name in ("heating", "ventilation", "screen", "shading"):
            idx = ACTION_NAMES.index(action_name)
            desired_delta = float(original[idx] - previous[idx])
            sign = 1 if desired_delta > 1e-6 else -1 if desired_delta < -1e-6 else 0
            limit = self._transition_gate_limit_for(regime, action_name)
            if self._transition_gate_recent_reversal(action_name, sign):
                clipped_delta = 0.0
                reversal_fields.append(action_name)
                reversal_projected_count += 1
            else:
                clipped_delta = float(max(-limit, min(limit, desired_delta)))
            if abs(clipped_delta - desired_delta) > 1e-9:
                adjusted_fields.append(action_name)
                delta_limited_count += 1
            adjusted[idx] = float(np.clip(previous[idx] + clipped_delta, 0.0, 1.0))

        for action_name in ("co2", "lighting"):
            idx = ACTION_NAMES.index(action_name)
            adjusted[idx] = original[idx]

        self._transition_gate_update_sign_history(previous, adjusted)
        self._transition_gate_previous_action = adjusted.copy()
        applied = bool(np.max(np.abs(adjusted - original)) > 1e-9)
        self.last_transition_gate = {
            "enabled": True,
            "applied": applied,
            "bypassed": False,
            "bypass_reasons": [],
            "regime": regime,
            "before": self._transition_gate_action_record(original),
            "after": self._transition_gate_action_record(adjusted),
            "adjusted_fields": adjusted_fields,
            "reversal_fields": reversal_fields,
            "delta_limited_count": int(delta_limited_count),
            "reversal_projected_count": int(reversal_projected_count),
            "reversal_window_steps": int(getattr(self.config, "transition_gate_reversal_window_steps", 6) or 6),
            "soft_limit_source": self._transition_gate_soft_limit_source,
            "soft_limit_error": self._transition_gate_soft_limit_error,
        }
        return adjusted.astype(np.float32)
    
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
        self.last_tomato_safety_v2_suppressed_replan = {"applied": False}
        self.last_strict_replay_suppressed_replan = {"applied": False}
        if not self._is_plan_active(int(state.timestep)):
            replan_reason = "init_plan" if self.current_plan is None else "plan_expired"
        elif self._should_follow_strict_replay_frozen_replan_step(state):
            replan_reason = "frozen_replan"
        elif self._should_emergency_replan(analysis, state):
            if (
                self._should_suppress_strict_replay_emergency_replan()
                and not self._has_strict_replay_plan_for_step(state)
            ):
                suppressed = {
                    "applied": True,
                    "step": int(state.timestep),
                    "reason": "emergency_replan",
                    "critical_level": str(analysis.get("critical_level", "normal")),
                    "critical_violations": list(analysis.get("critical_violations", [])),
                }
                self.last_strict_replay_suppressed_replan = dict(suppressed)
                if bool(getattr(self.config, "tomato_safety_v2_enabled", False)):
                    self.tomato_safety_v2_suppressed_replan_steps += 1
                    self.last_tomato_safety_v2_suppressed_replan = dict(suppressed)
                self.emergency_streak_steps = 0
                self.rh_emergency_streak_steps = 0
            else:
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
        final_control = self._apply_transition_gate(final_control, state)
        self._run_cstcc_shadow_runtime_hook(
            state,
            final_control,
            rule_control=rule_control,
            anchor_control=anchor_control,
        )
        
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
            current_target_temp, current_target_co2, current_target_rh, _gate_record = self._apply_profile_feasibility_gate_targets(
                state,
                np.asarray(final_control, dtype=np.float32),
                current_target_temp,
                current_target_co2,
                current_target_rh,
                source="plan_info_current_targets",
                update_last=True,
            )
            current_target_temp, current_target_co2, current_target_rh, _template_record = self._apply_profile_template_patch_targets(
                state,
                np.asarray(final_control, dtype=np.float32),
                current_target_temp,
                current_target_co2,
                current_target_rh,
                source="plan_info_current_targets",
                update_last=True,
            )
            if isinstance(getattr(self, "last_profile_feasibility_gate", None), dict):
                self.last_profile_feasibility_gate["hard_safety_veto_count"] = int(
                    self.last_rollout_selection.get("profile_feasibility_gate_hard_safety_veto_count", 0)
                    if isinstance(getattr(self, "last_rollout_selection", None), dict)
                    else 0
                )
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
                "intent_contract": self.current_plan.get("intent_contract", {}),
                "intent_contract_diagnostics": self.current_plan.get("intent_contract_diagnostics", {}),
                "profile_candidates": self.current_plan.get("profile_candidates", []),
                "profile_generator_diagnostics": self.current_plan.get("profile_generator_diagnostics", {}),
                **(
                    {
                        "structured_anchor_profile_bridge": self.current_plan.get(
                            "structured_anchor_profile_bridge",
                            dict(getattr(self, "last_structured_anchor_profile_bridge", {})),
                        ),
                        "structured_anchor_profile_bridge_candidates": self.current_plan.get(
                            "structured_anchor_profile_bridge_candidates",
                            [],
                        ),
                    }
                    if bool(getattr(self.config, "structured_anchor_profile_bridge_enabled", False))
                    and bool(getattr(self.config, "structured_anchor_profile_bridge_record_provenance", True))
                    else {}
                ),
                "rule_weight": rule_weight,
                "reason": self.current_plan.get("reason"),
                "anchor_source": self.current_plan.get("anchor_source", "unknown"),
                "fallback_selection": self.current_plan.get("fallback_selection", {}),
                "plan_cache_event": self.current_plan.get("plan_cache_event", dict(getattr(self, "last_plan_cache_event", {}))),
                "structured_anchor": self.current_plan.get("structured_anchor", dict(getattr(self, "last_structured_anchor", {}))),
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
            "transition_gate": dict(getattr(self, "last_transition_gate", {})),
            **(
                {"profile_feasibility_gate": dict(getattr(self, "last_profile_feasibility_gate", {}))}
                if bool(getattr(self.config, "profile_feasibility_gate_enabled", False))
                else {}
            ),
            **(
                {"profile_template_patch": dict(getattr(self, "last_profile_template_patch", {}))}
                if bool(getattr(self.config, "profile_template_patch_record_provenance", True))
                else {}
            ),
            "structured_anchor": dict(getattr(self, "last_structured_anchor", {})),
            **(
                {"structured_anchor_profile_bridge": dict(getattr(self, "last_structured_anchor_profile_bridge", {}))}
                if bool(getattr(self.config, "structured_anchor_profile_bridge_enabled", False))
                and bool(getattr(self.config, "structured_anchor_profile_bridge_record_provenance", True))
                else {}
            ),
            **(
                {"cstcc_shadow": dict(getattr(self, "last_cstcc_shadow", {}))}
                if bool(getattr(self.config, "cstcc_shadow_enabled", False))
                else {}
            ),
            "fallback_selection": dict(self.last_fallback_selection),
        }
