from langchain_core.tools import StructuredTool
from gl_gym.agent.interface import GreenhouseAgentInterface, ControlAction

class GreenhouseTools:
    """
    温室控制工具集 (Toolbox)
    
    该类封装了 LLM 可用的所有操作工具。
    为了解决 LLM 连续调用工具导致的环境步进问题，
    采用了【动作缓冲机制 (Action Buffering)】。
    
    工作原理：
    1. LLM 调用 set_heating, set_ventilation 等工具时，不立即执行环境步进。
    2. 工具仅更新内存中的 `buffered_action` 对象。
    3. 所有的动作配置完成后，由 Agent 主循环统一调用环境的 step 方法执行。
    """
    
    def __init__(self, interface: GreenhouseAgentInterface):
        self.interface = interface
        self.last_reward = 0.0
        self.last_done = False
        self.status_calls_this_round = 0
        # 初始化动作缓冲区，默认所有设备关闭 (0.0)
        self.buffered_action = ControlAction()
        
    def reset_buffer(self):
        """
        重置动作缓冲区
        
        在每一轮决策开始前调用。
        默认继承当前环境的执行器状态，而不是全零重置。
        这意味着如果 LLM 不调整某个设备，该设备将保持上一时刻的状态（惯性保持）。
        """
        current_state = self.interface.get_state()
        self.buffered_action = ControlAction(
            u_boil=current_state.u_boil,
            u_co2=current_state.u_co2,
            u_th_scr=current_state.u_th_scr,
            u_vent=current_state.u_vent,
            u_lamp=current_state.u_lamp,
            u_bl_scr=current_state.u_bl_scr
        )
        self.status_calls_this_round = 0
        print(f"[Tools] 缓冲区已重置: 加热={self.buffered_action.u_boil:.2f}, 通风={self.buffered_action.u_vent:.2f}")

    def _update_buffer(self, **kwargs) -> str:
        """
        内部方法：更新动作缓冲区中的特定字段
        
        Args:
            **kwargs: 键值对，如 u_boil=0.5
            
        Returns:
            str: 反馈给 LLM 的文本消息，告知其设置已记录。
        """
        for k, v in kwargs.items():
            if hasattr(self.buffered_action, k):
                # 确保控制量在 [0, 1] 范围内
                setattr(self.buffered_action, k, max(0, min(1, v)))
        
        return (f"已记录设置 -> 加热:{self.buffered_action.u_boil:.2f}, "
                f"CO2:{self.buffered_action.u_co2:.2f}, "
                f"通风:{self.buffered_action.u_vent:.2f}, "
                f"遮阳:{self.buffered_action.u_bl_scr:.2f}, "
                f"保温幕:{self.buffered_action.u_th_scr:.2f}。 "
                "请继续配置其他设备，或停止调用以执行生效。")
    
    def get_status(self) -> str:
        """
        工具：获取温室当前状态
        
        LLM 使用此工具来观察环境，获取温度、湿度等传感器数据。
        """
        print("[Tools] get_status 被调用")
        self.status_calls_this_round += 1
        if self.status_calls_this_round > 1:
            return (
                "同一决策轮内已获取过状态。当前再次查询不会看到新动作生效。"
                "请直接使用已设置的控制量结束本轮，让系统统一 step 执行。"
            )
        state_text = self.interface.get_state_description()
        pending_text = (
            f"\n\n【本轮待执行控制（缓冲区）】"
            f"\n- 加热: {self.buffered_action.u_boil:.2f}"
            f"\n- CO2: {self.buffered_action.u_co2:.2f}"
            f"\n- 保温幕: {self.buffered_action.u_th_scr:.2f}"
            f"\n- 通风: {self.buffered_action.u_vent:.2f}"
            f"\n- 补光: {self.buffered_action.u_lamp:.2f}"
            f"\n- 遮阳: {self.buffered_action.u_bl_scr:.2f}"
            f"\n\n提示：同一决策轮中，以上控制尚未执行生效，需等本轮结束统一 step 后才会反映到传感器。"
        )
        return state_text + pending_text

    def set_heating(self, level: float) -> str:
        """
        工具：设置加热锅炉功率
        
        Args:
            level: 0.0 (关闭) ~ 1.0 (全开)
        """
        print(f"[Tools] set_heating({level}) 被调用")
        return self._update_buffer(u_boil=level)
    
    def set_co2(self, level: float) -> str:
        """
        工具：设置 CO2 供给水平
        
        Args:
            level: 0.0 (关闭) ~ 1.0 (最大供给)
        """
        print(f"[Tools] set_co2({level}) 被调用")
        return self._update_buffer(u_co2=level)
    
    def set_screen(self, position: float) -> str:
        """
        工具：设置保温幕位置
        
        Args:
            position: 0.0 (完全打开) ~ 1.0 (完全闭合/保温)
        """
        print(f"[Tools] set_screen({position}) 被调用")
        return self._update_buffer(u_th_scr=position)
    
    def set_ventilation(self, level: float) -> str:
        """
        工具：设置通风窗开度
        
        Args:
            level: 0.0 (关闭) ~ 1.0 (全开)
        """
        print(f"[Tools] set_ventilation({level}) 被调用")
        return self._update_buffer(u_vent=level)
    
    def set_lamps(self, intensity: float) -> str:
        """
        工具：设置补光灯强度
        
        Args:
            intensity: 0.0 (关闭) ~ 1.0 (最大亮度)
        """
        print(f"[Tools] set_lamps({intensity}) 被调用")
        return self._update_buffer(u_lamp=intensity)
    
    def set_blindscreen(self, position: float) -> str:
        """
        工具：设置遮阳网位置
        
        Args:
            position: 0.0 (收起) ~ 1.0 (放下/遮阳)
        """
        print(f"[Tools] set_blindscreen({position}) 被调用")
        return self._update_buffer(u_bl_scr=position)
    
    def set_all_controls(self, heating: float, co2: float, screen: float, 
                        ventilation: float, lamps: float, blindscreen: float) -> str:
        """
        工具：一次性设置所有控制设备
        
        高效工具，允许 LLM 用一次调用完成所有配置。
        """
        print(f"[Tools] set_all_controls(heating={heating:.2f}, co2={co2:.2f}, ...) 被调用")
        return self._update_buffer(
            u_boil=heating, u_co2=co2, u_th_scr=screen,
            u_vent=ventilation, u_lamp=lamps, u_bl_scr=blindscreen
        )

def create_langchain_tools(interface: GreenhouseAgentInterface) -> list:
    """
    工厂函数：创建并注册 LangChain 工具列表
    
    Args:
        interface: 智能体接口实例
        
    Returns:
        list[StructuredTool]: LangChain 可用的工具列表
    """
    # 单例模式：确保全局只有一个工具实例，以便 Action Buffer 状态能够跨调用保持
    # 这对于 llm_agent.py 正确读取 buffer 至关重要
    if not hasattr(create_langchain_tools, "instance"):
        create_langchain_tools.instance = GreenhouseTools(interface)
    tools_instance = create_langchain_tools.instance
    
    return [
        StructuredTool.from_function(
            func=tools_instance.get_status,
            name="get_status",
            description="获取温室当前状态，包括温度、湿度、CO2、作物状态和设备状态"
        ),
        StructuredTool.from_function(
            func=tools_instance.set_heating,
            name="set_heating",
            description="设置加热锅炉功率 (0-1)"
        ),
        StructuredTool.from_function(
            func=tools_instance.set_co2,
            name="set_co2",
            description="设置CO2供给水平 (0-1)"
        ),
        StructuredTool.from_function(
            func=tools_instance.set_screen,
            name="set_screen",
            description="设置保温幕位置 (0=打开, 1=关闭)"
        ),
        StructuredTool.from_function(
            func=tools_instance.set_ventilation,
            name="set_ventilation",
            description="设置通风窗开度 (0-1)"
        ),
        StructuredTool.from_function(
            func=tools_instance.set_lamps,
            name="set_lamps",
            description="设置补光灯强度 (0-1)"
        ),
        StructuredTool.from_function(
            func=tools_instance.set_blindscreen,
            name="set_blindscreen",
            description="设置遮阳网位置 (0-1)"
        ),
        StructuredTool.from_function(
            func=tools_instance.set_all_controls,
            name="set_all_controls",
            description="一次性设置所有设备 (所有值范围 0-1)"
        ),
    ]
