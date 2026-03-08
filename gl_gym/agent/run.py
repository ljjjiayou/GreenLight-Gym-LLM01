"""
温室智能控制系统启动脚本 (EntryPoint)

该脚本经过修改，增加了自动路径识别和工作目录切换功能，
以解决 ModuleNotFoundError 和 FileNotFoundError 问题。
"""

import os
import sys
import yaml
import dotenv

# ==============================================================
# 第一步：路径与环境初始化 (核心修复)
# ==============================================================
# 1. 获取项目根目录 (GreenLight-Gym2-master)
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

# 2. 将根目录加入模块搜索路径，确保能 import gl_gym
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 3. 切换工作目录到根目录，确保程序能根据相对路径找到 weather 数据文件
os.chdir(project_root)
print(f"[*] 工作目录已切换至项目根目录: {project_root}")

# 现在可以安全地导入项目模块了
from gl_gym.environments.tomato_env import TomatoEnv
from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector
from gl_gym.agent.interface import GreenhouseAgentInterface
from gl_gym.agent.tools import create_langchain_tools
from gl_gym.agent.control_loop import GreenhouseControlLoop, ControlLoopConfig

# 加载环境变量 (如 BAILIAN_API_KEY)
dotenv.load_dotenv()

# ==============================================================
# 第二步：功能函数定义
# ==============================================================

def create_env(config_path: str = None) -> TomatoEnv:
    """
    工厂函数：从 YAML 配置文件创建温室环境实例
    """
    if config_path is None:
        # 统一使用绝对路径拼接，避免路径歧义
        config_path = os.path.join(project_root, "gl_gym", "configs", "envs", "TomatoEnv.yml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
    print(f"[*] 正在从配置加载环境: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 提取参数
    base_env_params = config['GreenLightEnv']
    tomato_cfg = config['TomatoEnv']
    
    # 实例化环境
    env = TomatoEnv(
        reward_function=tomato_cfg['reward_function'],
        observation_modules=tomato_cfg['observation_modules'],
        constraints=tomato_cfg['constraints'],
        eval_options=tomato_cfg['eval_options'],
        reward_params=tomato_cfg['reward_params'],
        base_env_params=base_env_params,
        uncertainty_scale=tomato_cfg.get('uncertainty_scale', 0.0)
    )
    return env

def main():
    # 1. 验证 API Key
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        print("\n[错误] 未在 .env 文件或环境变量中找到 BAILIAN_API_KEY")
        return

    # 2. 初始化环境
    try:
        env = create_env()
    except Exception as e:
        print(f"\n[环境初始化失败]: {e}")
        return
    
    # 3. 配置智能体
    agent_config = AgentConfig(
        model_name="qwen-plus", 
        verbose=True,           
        max_iterations=10,      
        api_key=api_key
    )
    
    # 4. 创建接口与工具
    interface = GreenhouseAgentInterface(env)
    tools = create_langchain_tools(interface)
    
    # 5. 创建混合智能体
    print("[*] 正在初始化混合智能体 (Rule-Based + LLM)...")
    agent = RuleBasedLLMDirector(
        agent_interface=interface,
        tools=tools,
        config=agent_config
    )
    
    # 6. 配置控制循环
    loop_config = ControlLoopConfig(
        max_steps=10,  # 快速验证设为10步
        log_freq=1,
        verbose=True
    )
    
    control_loop = GreenhouseControlLoop(
        agent=agent,
        agent_interface=interface,
        config=loop_config
    )
    
    # 7. 启动
    print(f"\n{'='*50}")
    print(f"开始运行温室控制系统")
    print(f"{'='*50}\n")
    
    try:
        results = control_loop.run()
        
        # 8. 统计报告
        print(f"\n{'='*50}")
        print(f"运行完成报告:")
        print(f"- 最终步数: {results.get('steps', 'N/A')}")
        print(f"- 总奖励值: {results.get('total_reward', 0.0):.2f}")
        print(f"- 仿真耗时: {results.get('elapsed_time', 0.0):.2f} 秒")
        print(f"{'='*50}")
        
    except KeyboardInterrupt:
        print("\n[!] 运行被用户手动中断。")
    except Exception as e:
        print(f"\n[!] 运行过程中发生崩溃: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()