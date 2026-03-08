"""
LangChain Agent 核心模块

该模块提供了基于 LangChain 的 LLM 智能体实现，用于控制温室环境。
采用了【混合控制架构 (Hybrid Control Architecture)】：
- 规则控制器 (RuleBasedController)：负责处理常规的、明显的边界违规。
- LLM 智能体 (GreenhouseAgent)：负责处理复杂的多目标冲突和微调。

采用 LangChain v1.0+ (LangGraph 核心) 的 create_agent 接口。
固定使用百炼平台 qwen-plus 模型。
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import json
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

from gl_gym.environments.baseline import RuleBasedController
from gl_gym.common.utils import load_model_hyperparams, calculate_vpd_kpa
from gl_gym.agent.tools import create_langchain_tools, GreenhouseTools # 导入 Tools 类


@dataclass
class AgentConfig:
    """
    智能体配置类
    
    固定使用百炼平台 qwen-plus 模型，参数经过优化以适应温室控制任务。
    """
    
    # 模型相关配置
    model_name: str = "qwen-plus"               # 使用的模型名称
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼地址
    api_key: Optional[str] = None                # API 密钥
    temperature: float = 0.0                      # 随机性控制 (最小随机性，提升控制稳定性)
    max_tokens: int = 2000                        # 单次生成最大长度
    
    # Agent 执行参数
    max_iterations: int = 10                       # LangChain Agent 的最大思考轮数
    max_execution_time: Optional[float] = None     # 超时时间
    early_stopping_method: str = "force"           # 早停策略
    
    # 调试
    verbose: bool = True


@dataclass
class GreenhousePrompts:
    """温室控制智能体的提示词模板 (Prompt Engineering)"""
    
    SYSTEM_PROMPT = """你是温室智能控制系统的 AI 助手。你的任务是监控温室环境状态，并根据作物生长需求和经济效益做出最优的控制决策。

## 温室设备
- 加热锅炉 (heating): 0 表示停止供暖，1 表示最大供暖
- CO2供给系统 (co2): 0 表示关闭，1 表示最大供给
- 保温幕 (screen): 0 表示完全打开，1 表示完全关闭
- 通风窗 (ventilation): 0 表示关闭，1 表示完全打开
- 补光灯 (lamps): 0 表示关闭，1 表示最大亮度
- 遮阳网 (blindscreen): 0 表示收起，1 表示放下

## 控制目标
- 维持温度在状态报告给出的建议范围（白天通常 18-25°C，夜间通常 12-16°C）
- 保持适当的 CO2 浓度 (400-1000 ppm)
- 维持适当的饱和水汽压差 (VPD) 在 0.4-1.2 kPa (比单纯看 RH 更重要)
- 优化能源使用成本
- 最大化作物产量和收益

## 工作原则
- 每次决策前先查看当前温室状态
- 分析环境条件与目标的差距
- 做出控制决策并执行工具
- 观察执行结果，必要时调整
- 注意不要让温度、湿度、CO2 超出安全范围
- 重要：你的工具调用不会立即执行，而是会先缓冲。请一次性调用所有需要调整的工具，系统会在你决策完成后统一执行。
- 重要：同一决策轮内调用 get_status 不会看到刚设置的控制量生效，请不要据此判断“执行器故障”。
- 安全约束：当室内温度低于 12°C 时，通风应谨慎（通常不高于 0.15），优先提高加热与保温幕。
- 优化目标：优先减少温度与 VPD 违规，再考虑能耗；除非除湿(VPD<0.4)，否则禁止同时开启加热与通风；夜间与弱光下不施 CO2。
- 工具调用约束：每轮最多调用一次 get_status，优先使用一次 set_all_controls 完成设置。"""

    USER_PROMPT_TEMPLATE = """## 当前温室状态
{state}

请基于以上状态做出最优控制决策。如果需要调整设备，请使用对应的工具。"""


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
    is_night = hour_of_day < 6.0 or hour_of_day > 18.0
    
    vpd = calculate_vpd_kpa(temp_air, rh_air)
    print(f"[Guard] T={temp_air:.1f}, RH={rh_air:.1f}, VPD={vpd:.2f} kPa")

    # --- 基础安全约束 ---
    # 1. 低温保护
    if temp_air < 12.0:
        guarded[0] = max(guarded[0], 0.90)  # 强加热
        guarded[2] = max(guarded[2], 0.95)  # 强保温
        vent_cap = 0.15 if rh_air > 92.0 else 0.05 # 极低通风上限
        guarded[3] = min(guarded[3], vent_cap)
    
    # 2. 高湿/低VPD 处理 (防病害)
    # 如果 VPD < 0.4 (过湿)，需要除湿
    if vpd < 0.4:
        # 允许一定程度的加热+通风 (除湿模式)
        if temp_air < 18.0:
            guarded[0] = max(guarded[0], 0.3) # 至少有点加热来驱动除湿
            guarded[3] = max(guarded[3], 0.1) # 至少有点通风排湿
        else:
             # 温度够高，主要靠通风
             guarded[3] = max(guarded[3], 0.3)
    
    # 3. 正常/干燥 VPD 处理 (防萎蔫)
    # 如果 VPD > 1.2 (偏干)，减少通风，增加保湿
    elif vpd > 1.2:
        guarded[3] = min(guarded[3], 0.3) # 限制通风
        if glob_rad > 500: # 强光下
            guarded[5] = max(guarded[5], 0.5) # 用遮阳网降温而不是强通风

    # --- 互斥与效率约束 ---
    # 4. 避免能源浪费 (加热与通风互斥)
    # 除非是为了除湿 (VPD < 0.4)，否则禁止同时高加热和高通风
    if vpd >= 0.4:
        if guarded[0] > 0.1 and guarded[3] > 0.1:
            if temp_air > 22.0: # 热，优先关加热
                guarded[0] = 0.0
            elif temp_air < 16.0: # 冷，优先关通风
                guarded[3] = 0.0
            else: # 中间态，双重限制
                guarded[0] = min(guarded[0], 0.1)
                guarded[3] = min(guarded[3], 0.1)

    # 5. CO2 效率
    # 通风量大时关 CO2 (避免漏气)
    if guarded[3] > 0.2:
        guarded[1] = 0.0
    # 夜间或弱光不施 CO2 (植物不吸收)
    if is_night or glob_rad < 50.0:
        guarded[1] = 0.0

    # 6. 风速保护
    if wind_speed > 8.0:
        guarded[3] = min(guarded[3], 0.1) # 强风关窗
    elif wind_speed > 6.0 and temp_out < temp_air:
        guarded[3] = min(guarded[3], 0.20)

    # 7. 夜间保温
    if is_night:
        guarded[5] = 0.0 # 夜间不用遮阳网
        guarded[4] = 0.0 # 夜间不用补光 (假设非光周期补光)
        if temp_air < 15.0:
            guarded[2] = max(guarded[2], 0.80) # 夜间低温必拉保温幕

    return np.clip(guarded, 0.0, 1.0)


def log_control_tracking(prefix: str, target_control: np.ndarray, applied_control: np.ndarray, action_cmd: Optional[np.ndarray] = None) -> None:
    names = ["加热", "CO2", "保温幕", "通风", "补光", "遮阳"]
    target = np.asarray(target_control, dtype=np.float32).flatten()
    applied = np.asarray(applied_control, dtype=np.float32).flatten()
    cmd = None if action_cmd is None else np.asarray(action_cmd, dtype=np.float32).flatten()
    n = min(len(names), len(target), len(applied))
    records = []
    for i in range(n):
        diff = applied[i] - target[i]
        if cmd is None or i >= len(cmd):
            records.append(f"{names[i]}(目标:{target[i]:.2f},执行:{applied[i]:.2f},差值:{diff:+.2f})")
        else:
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
            system_prompt=self.system_prompt
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
        执行一步 LLM 决策
        
        关键逻辑：
        1. 重置工具缓冲区。
        2. 调用 LLM 进行推理，LLM 调用工具只会更新缓冲区。
        3. LLM 思考结束后，统一执行缓冲区中的最终动作 (env.step)。
        """
        # 1. 重置工具缓冲区
        if hasattr(create_langchain_tools, "instance"):
             create_langchain_tools.instance.reset_buffer()

        state = self.interface.get_state()
        state_desc = self.interface.get_state_description()
        user_input = self._build_user_prompt(state_desc)
        
        if context:
            user_input = f"{context}\n\n{user_input}"
        
        try:
            # 构造输入消息
            current_messages = self.history + [HumanMessage(content=user_input)]
            
            # 调用 LangGraph
            result_state = self.agent_graph.invoke({"messages": current_messages})
            
            messages = result_state.get("messages", [])
            output = ""
            if messages and isinstance(messages[-1], AIMessage):
                output = messages[-1].content
            
            # 2. 统一执行环境步进
            # 获取工具实例
            tools_instance = None
            if hasattr(create_langchain_tools, "instance"):
                tools_instance = create_langchain_tools.instance
            
            reward = 0.0
            done = False
            
            if tools_instance:
                 # 执行缓冲的动作
                 # 构造完整的动作日志
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
                 
                 target_control = tools_instance.buffered_action.to_array()
                 target_control = apply_safety_guardrails(state, target_control)
                 action_cmd = control_to_action_command(self.interface.env, target_control)
                 obs, reward, terminated, truncated, info = self.interface.env.step(action_cmd)
                 log_control_tracking("[Agent]", target_control, self.interface.env.u, action_cmd)
                 # 更新接口状态
                 self.interface.last_reward = reward
                 self.interface.last_terminated = terminated
                 self.interface.last_truncated = truncated
                 done = terminated or truncated
                 print(f"[Agent] 动作执行完毕 -> 奖励: {reward:.4f}, 是否结束: {done}")

            # 更新历史
            self.history.append(HumanMessage(content=user_input))
            self.history.append(AIMessage(content=output))
            
            if len(self.history) > 20:
                self.history = self.history[-20:]
            
            return {
                "success": True,
                "output": output,
                "state": state,
                "reward": reward,
                "done": done
            }
            
        except Exception as e:
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
    规则引导的 LLM 导演模式 (Hybrid Director)
    
    核心逻辑：
    1. 优先使用规则判断当前状态是否【极度危险】或【明显违规】。
    2. 如果是，直接使用规则控制器 (RuleBasedController) 的输出，快速响应。
    3. 如果状态复杂 (多指标冲突) 或需要精细调节，则委托给 LLM 决策。
    4. 如果状态完美，则维持现状 (无操作)。
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
        
        # 初始化规则阈值
        self.rules = self._init_rules()
        self.env_id = env_id or getattr(self.interface.env, "env_id", self.interface.env.__class__.__name__)
        # 加载规则控制器 (PID 或其他算法)
        self.rule_controller = self._init_rule_controller(rule_params)
        
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
            "temp_low": 18.0,
            "temp_high": 25.0,
            "temp_critical_low": 15.0,
            "temp_critical_high": 30.0,
            "co2_low": 400.0,
            "co2_high": 1000.0,
            "rh_low": 50.0,
            "rh_high": 85.0
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
    
    def _get_weather_vector(self) -> np.ndarray:
        env = self.interface.env
        if hasattr(env, "weather_data") and hasattr(env, "timestep"):
            if 0 <= env.timestep < len(env.weather_data):
                return env.weather_data[env.timestep]
        return np.zeros(getattr(env, "nd", 10))
    
    def _select_control_mode(self, state) -> str:
        """
        根据当前状态选择控制模式：
        - none: 状态良好，无需干预
        - rule: 严重违规或单一指标违规，规则可快速修复
        - llm: 多个指标同时违规，可能存在冲突，需要 LLM 权衡
        """
        critical = (
            state.temp_air < self.rules["temp_critical_low"]
            or state.temp_air > self.rules["temp_critical_high"]
            or state.co2_air < self.rules["co2_low"]
            or state.co2_air > self.rules["co2_high"]
            or state.rh_air < self.rules["rh_low"]
            or state.rh_air > self.rules["rh_high"]
        )
        violations = 0
        if state.temp_air < self.rules["temp_low"] or state.temp_air > self.rules["temp_high"]:
            violations += 1
        if state.co2_air < self.rules["co2_low"] or state.co2_air > self.rules["co2_high"]:
            violations += 1
        if state.rh_air < self.rules["rh_low"] or state.rh_air > self.rules["rh_high"]:
            violations += 1
        
        if violations == 0:
            return "none"
        return "llm"
    
    def analyze_state(self) -> Dict:
        """生成详细的状态分析报告"""
        state = self.interface.get_state()
        suggestions = []
        
        if state.temp_air < self.rules["temp_critical_low"]:
            suggestions.append("CRITICAL: 温度过低，需要立即加热!")
        elif state.temp_air < self.rules["temp_low"]:
            suggestions.append("温度偏低，建议增加供暖")
        elif state.temp_air > self.rules["temp_critical_high"]:
            suggestions.append("CRITICAL: 温度过高，需要通风降温")
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
        
        return {
            "state": state,
            "suggestions": suggestions,
            "needs_action": len(suggestions) > 0,
            "violation_count": violation_count
        }
    
    def step_with_rules(self) -> Dict:
        """
        执行一步混合控制决策
        """
        analysis = self.analyze_state()
        mode = self._select_control_mode(analysis["state"])
        
        print(f"[Director] 当前状态: T={analysis['state'].temp_air:.1f}, CO2={analysis['state'].co2_air:.0f}, RH={analysis['state'].rh_air:.1f}")
        print(f"[Director] 违规数: {analysis['violation_count']}, 决策模式: {mode}")
        
        # --- 模式 1: 无操作 ---
        if mode == "none":
            env = self.interface.env
            current_state = self.interface.get_state()
            control = np.array([
                current_state.u_boil, current_state.u_co2, current_state.u_th_scr,
                current_state.u_vent, current_state.u_lamp, current_state.u_bl_scr
            ], dtype=np.float32)
            
            if hasattr(env, "step"):
                zero_action = np.zeros(getattr(env, "nu", len(control)), dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(zero_action)
            elif hasattr(env, "step_raw_control"):
                obs, reward, terminated, truncated, info = env.step_raw_control(control)
            else:
                reward, terminated, truncated, info = 0.0, False, False, {}
            self.interface.last_reward = reward
            self.interface.last_terminated = terminated
            self.interface.last_truncated = truncated
            
            return {
                "action": "none",
                "message": "当前状态良好，维持当前控制",
                "analysis": analysis,
                "success": True,
                "reward": reward,
                "done": terminated or truncated
            }
        
        # --- 模式 2: 规则控制 ---
        if mode == "rule" and self.rule_controller is not None:
            env = self.interface.env
            weather = self._get_weather_vector()
            control = self.rule_controller.predict(env.x, weather, env)
            control = apply_safety_guardrails(analysis["state"], np.asarray(control, dtype=np.float32))
            
            # 补全规则控制的工具调用日志
            print(f"[Tools] set_all_controls(heating={control[0]:.2f}, co2={control[1]:.2f}, screen={control[2]:.2f}, "
                  f"ventilation={control[3]:.2f}, lamps={control[4]:.2f}, blindscreen={control[5]:.2f}) 被调用")
            
            if hasattr(env, "step"):
                action_cmd = control_to_action_command(env, np.asarray(control, dtype=np.float32))
                obs, reward, terminated, truncated, info = env.step(action_cmd)
                log_control_tracking("[Director-Rule]", np.asarray(control, dtype=np.float32), env.u, action_cmd)
                self.interface.last_reward = reward
                self.interface.last_terminated = terminated
                self.interface.last_truncated = truncated
                
                print(f"[Tools] 动作执行完毕 -> 奖励: {reward:.4f}, 是否结束: {terminated or truncated}")
                
                return {
                    "action": "rule_control",
                    "message": "规则控制已执行",
                    "analysis": analysis,
                    "control": np.asarray(control).tolist(),
                    "reward": reward,
                    "done": terminated or truncated,
                    "success": True
                }
            if hasattr(env, "step_raw_control"):
                obs, reward, terminated, truncated, info = env.step_raw_control(control)
                log_control_tracking("[Director-Rule]", np.asarray(control, dtype=np.float32), env.u)
                self.interface.last_reward = reward
                self.interface.last_terminated = terminated
                self.interface.last_truncated = truncated
                
                print(f"[Tools] 动作执行完毕 -> 奖励: {reward:.4f}, 是否结束: {terminated or truncated}")
                
                return {
                    "action": "rule_control",
                    "message": "规则控制已执行",
                    "analysis": analysis,
                    "control": np.asarray(control).tolist(),
                    "reward": reward,
                    "done": terminated or truncated,
                    "success": True
                }
            return {
                "action": "rule_control",
                "message": "规则控制已计算，但环境不支持 step_raw_control",
                "analysis": analysis,
                "control": np.asarray(control).tolist(),
                "success": False
            }
        
        # --- 模式 3: LLM 控制 ---
        # 使用 LLM 控制
        prompt_text = f"""
当前温室状态存在问题，需要你做出控制决策：

当前状态：
{self.interface.get_state_description()}

问题分析：
{chr(10).join(analysis['suggestions'])}

请使用工具调整温室控制设备。"""
        
        print(f"[Director] 调用 LLM 进行决策...")
        
        try:
            # 1. 重置工具缓冲区
            if hasattr(create_langchain_tools, "instance"):
                 create_langchain_tools.instance.reset_buffer()

            # 临时构建一个 Agent Graph
            agent_graph = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=GreenhousePrompts.SYSTEM_PROMPT
            )
            
            # 直接 invoke graph
            result_state = agent_graph.invoke({"messages": [HumanMessage(content=prompt_text)]})
            
            messages = result_state.get("messages", [])
            output = ""
            if messages and isinstance(messages[-1], AIMessage):
                output = messages[-1].content
            
            # 2. 统一执行环境步进
            tools_instance = None
            if hasattr(create_langchain_tools, "instance"):
                tools_instance = create_langchain_tools.instance
            
            reward = 0.0
            done = False
            
            if tools_instance:
                 # 执行缓冲的动作
                 # 构造完整的动作日志
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
                 
                 target_control = tools_instance.buffered_action.to_array()
                 target_control = apply_safety_guardrails(analysis["state"], target_control)
                 action_cmd = control_to_action_command(self.interface.env, target_control)
                 obs, reward, terminated, truncated, info = self.interface.env.step(action_cmd)
                 log_control_tracking("[Agent]", target_control, self.interface.env.u, action_cmd)
                 self.interface.last_reward = reward
                 self.interface.last_terminated = terminated
                 self.interface.last_truncated = truncated
                 done = terminated or truncated
                 print(f"[Agent] 动作执行完毕 -> 奖励: {reward:.4f}, 是否结束: {done}")

            return {
                "action": "llm_control",
                "message": output,
                "analysis": analysis,
                "success": True,
                "reward": reward,
                "done": done
            }
            
        except Exception as e:
            return {
                "action": "error",
                "message": str(e),
                "analysis": analysis,
                "success": False
            }
