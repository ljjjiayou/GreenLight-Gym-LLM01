"""
主控制循环模块

该模块实现了智能体与温室环境的主要交互循环。
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
import time
from datetime import datetime

from gl_gym.agent.interface import (
    GreenhouseAgentInterface,
    ControlAction,
    GreenhouseState
)
from gl_gym.agent.tools import (
    GreenhouseTools,
    create_langchain_tools
)
from gl_gym.agent.llm_agent import (
    GreenhouseAgent,
    AgentConfig,
    RuleBasedLLMDirector
)


@dataclass
class ControlLoopConfig:
    """控制循环配置"""
    max_steps: Optional[int] = None
    eval_freq: int = 100
    log_freq: int = 10
    save_freq: int = 1000
    
    verbose: bool = True
    render: bool = False
    
    track_history: bool = True
    save_trajectory: bool = False


@dataclass
class StepResult:
    """单步执行结果"""
    step: int
    state: GreenhouseState
    action: ControlAction
    reward: float
    done: bool
    elapsed_time: float
    agent_output: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    source: str = "unknown"  # 决策来源：rule, llm, none


class GreenhouseControlLoop:
    """
    温室控制主循环
    
    负责协调智能体与环境之间的交互，执行控制决策，
    并记录执行历史。
    """
    
    def __init__(
        self,
        agent: Any, # 支持 GreenhouseAgent 或 RuleBasedLLMDirector
        agent_interface: GreenhouseAgentInterface,
        config: Optional[ControlLoopConfig] = None
    ):
        """
        初始化控制循环
        
        Args:
            agent: 温室智能体 (需实现 step 方法)
            agent_interface: 智能体接口
            config: 控制循环配置
        """
        self.agent = agent
        self.interface = agent_interface
        self.config = config or ControlLoopConfig()
        
        self.step_results: List[StepResult] = []
        self.episode_rewards: List[float] = []
        
        self.current_step = 0
        self.current_episode = 0
        
        self.start_time: Optional[datetime] = None
        self.total_elapsed_time = 0.0
        
    def reset(self):
        """重置循环状态"""
        obs, info = self.interface.env.reset()
        if hasattr(self.agent, "reset"):
            self.agent.reset()
        
        self.step_results = []
        self.current_step = 0
        
    def step(self, context: Optional[str] = None) -> StepResult:
        """
        执行一步
        
        Args:
            context: 额外的上下文信息
            
        Returns:
            StepResult: 步骤执行结果
        """
        start_time = time.time()
        
        state_before = self.interface.get_state()
        
        # 兼容不同的 Agent 接口
        source = "unknown"
        if hasattr(self.agent, "step_with_rules"):
             # RuleBasedLLMDirector
            result = self.agent.step_with_rules()
            source = result.get("action", "unknown") # rule_control, llm_control, none
        else:
            # GreenhouseAgent
            result = self.agent.step(context)
            source = "llm_agent"
        
        # 获取当前控制状态（可能被 Agent 修改，也可能保持不变）
        # 注意：Agent 的 step 方法通常已经通过工具调用了 env.step
        # 这里我们需要获取最新的动作状态用于记录
        
        # 如果 Agent 没有返回显式的 action 对象，我们假设它通过副作用修改了环境
        # 我们从环境或接口中获取最新的动作
        state_after = self.interface.get_state()
        
        action = ControlAction(
            u_boil=state_after.u_boil,
            u_co2=state_after.u_co2,
            u_th_scr=state_after.u_th_scr,
            u_vent=state_after.u_vent,
            u_lamp=state_after.u_lamp,
            u_bl_scr=state_after.u_bl_scr
        )
        
        # 检查环境是否结束
        # 如果 Agent 内部调用了 env.step，我们不需要再次调用
        # 但我们需要知道 reward 和 done
        
        reward = self.interface.last_reward
        if "reward" in result:
             reward = result["reward"]
             
        # 获取 done 状态
        done = False
        # 优先使用 result 中的 done
        if "done" in result:
            done = result["done"]
        # 其次使用 interface 中的 last_terminated/last_truncated
        elif hasattr(self.interface, "last_terminated"):
            done = self.interface.last_terminated or self.interface.last_truncated
        
        elapsed = time.time() - start_time
        
        step_result = StepResult(
            step=self.current_step,
            state=state_before, # 记录执行前的状态
            action=action,
            reward=reward,
            done=done,
            elapsed_time=elapsed,
            agent_output=result.get("message") or result.get("output"),
            success=result.get("success", True),
            error=result.get("error"),
            source=source
        )
        
        self.step_results.append(step_result)
        self.current_step += 1
        
        if self.config.verbose:
            print(f"Step {self.current_step}: Source={source}, Reward={reward:.2f}, Done={done}")
            if source == "llm_control":
                print(f"  > LLM Output: {step_result.agent_output[:100]}...")
            elif source == "rule_control":
                print(f"  > Rule Action Executed")
        
        return step_result
    
    def run(
        self,
        max_steps: Optional[int] = None,
        callback: Optional[Callable] = None,
        early_stop: Optional[Callable[[StepResult], bool]] = None
    ) -> Dict:
        """
        运行完整的控制循环
        
        Args:
            max_steps: 最大步数
            callback: 每步执行后的回调函数
            early_stop: 提前停止条件
            
        Returns:
            Dict: 运行结果统计
        """
        self.start_time = datetime.now()
        
        max_steps = max_steps or self.config.max_steps or 1000
        
        self.reset()
        
        episode_reward = 0.0
        
        while self.current_step < max_steps:
            step_result = self.step()
            
            episode_reward += step_result.reward
            
            if callback:
                callback(step_result)
            
            if early_stop and early_stop(step_result):
                if self.config.verbose:
                    print(f"Early stopping at step {self.current_step}")
                break
            
            if step_result.done:
                break
        
        self.episode_rewards.append(episode_reward)
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "steps": self.current_step,
            "total_reward": episode_reward,
            "avg_reward": episode_reward / max(self.current_step, 1),
            "elapsed_time": elapsed,
            "success_rate": sum(1 for r in self.step_results if r.success) / max(len(self.step_results), 1)
        }
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.step_results:
            return {}
        
        rewards = [r.reward for r in self.step_results]
        elapsed_times = [r.elapsed_time for r in self.step_results]
        
        return {
            "total_steps": self.current_step,
            "total_reward": sum(rewards),
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "avg_step_time": np.mean(elapsed_times),
            "total_elapsed_time": sum(elapsed_times),
            "success_rate": sum(1 for r in self.step_results if r.success) / len(self.step_results)
        }
    
    def save_results(self, filepath: str):
        """保存运行结果"""
        import json
        
        results = {
            "config": {
                "max_steps": self.config.max_steps,
                "verbose": self.config.verbose
            },
            "statistics": self.get_statistics(),
            "episode_rewards": self.episode_rewards,
            "steps": [
                {
                    "step": r.step,
                    "source": r.source,
                    "reward": r.reward,
                    "done": r.done,
                    "elapsed_time": r.elapsed_time,
                    "success": r.success,
                    "error": r.error,
                    "state": {
                        "timestep": r.state.timestep,
                        "temp_air": r.state.temp_air,
                        "co2_air": r.state.co2_air,
                        "rh_air": r.state.rh_air
                    },
                    "action": {
                        "u_boil": r.action.u_boil,
                        "u_vent": r.action.u_vent
                    }
                }
                for r in self.step_results
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
