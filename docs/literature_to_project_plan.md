# Literature-to-Project Plan

## 读文献的目的

这些论文不只用来写相关工作。它们主要用于三件事：

1. 找到温室控制领域认可的问题表述：约束、安全、经济收益、天气不确定性、模型误差、实时性。
2. 把已有方法转成项目里的可实现模块：setpoint 轨迹、约束收紧、fallback、鲁棒控制、PPO 强基线。
3. 设计论文实验：强 baseline、多 seed、多天气、多生长阶段、消融实验、运行时间与安全性指标。

## 已投喂论文的可用信息

### 1. RL-MPC for greenhouse climate control

本地文件：
`D:/桌面/温室环控论文/RL控制.pdf`
`D:/桌面/温室环控论文/生菜温室 RL+MPC.pdf`

外部链接：
https://www.sciencedirect.com/science/article/pii/S2772375524003551

核心思想：
该文提出参数化 MPC，并用 RL 在线学习 MPC 中的约束、预测模型和目标函数参数。它的重点不是让 RL 直接控制执行器，而是让 RL 改善 MPC 的结构参数，从而减少模型误差和天气预测不确定性导致的约束违规。

对本项目的启发：
当前 LLM Director 可以被表述成“语义层参数化 MPC”：LLM 不直接长期控制执行器，而是给出 setpoint profile、anchor action、约束风险偏好；底层规则/护栏负责闭环约束执行。后续可把 RH-debt、target_rh cap、fallback scoring 解释为 LLM 参数化的安全约束层。

可实现改进：
   - 记录每次 LLM 规划对 setpoint contract 的修正量。
   - 把 RH/Temp/CO2 的约束上限做成可学习或可调参数。
   - 加入消融：固定约束 vs 风险自适应约束。

### 2. Semi-closed greenhouse DRL + robust optimization

本地文件：
`D:/桌面/温室环控论文/基于人工智能的半封闭式温室节能控制：利用深度强化学习中的鲁棒优化.pdf`

外部链接：
https://doi.org/10.1016/j.adapen.2022.100119

核心思想：
将深度强化学习与鲁棒优化结合，面对天气、作物状态、外部扰动等不确定性时，生成更节能且不过度保守的控制。论文强调 energy efficiency、setpoint deviation 和 robustness。

对本项目的启发：
目前 LLM Director 的 240 步结果已经安全，但 LLM runtime 很高。后续可以让 LLM 只输出高层经济目标，底层用快速鲁棒优化或候选评分近似处理不确定性。

可实现改进：
   - 加入天气扰动场景测试：`uncertainty_scale = 0.0, 0.1, 0.2, 0.3`。
   - 增加 robust score：不是只看当前 state，还根据短期天气趋势惩罚高风险动作。
   - 把 energy / violation / yield 分开报告，不只报 total_reward。

### 3. RL versus MPC on greenhouse climate control

本地文件：
`D:/桌面/温室环控论文/生菜温室 RL对比MPC.pdf`

外部链接：
https://arxiv.org/abs/2303.06110

核心思想：
在统一框架下比较 MPC 与 RL。MPC 优点是显式约束和可解释，缺点是依赖模型和在线优化成本；RL 优点是决策快、可学习长期策略，缺点是安全约束不天然保证。

对本项目的启发：
论文写作时不要只说“LLM 打败 PPO”。更稳的叙事是：LLM 用于高层规划，规则/护栏提供 MPC 式约束保证，形成一个兼具语义规划和安全闭环的混合控制器。

可实现改进：
   - 强化 PPO baseline，重新训练并调参。
   - 加入 MPC/规则 baseline 的统一指标表。
   - 报告 runtime：PPO 快，LLM 慢，但 LLM 可解释和安全约束更强。

### 4. RL-SMPC under parametric uncertainty

本地文件：
`D:/桌面/温室环控论文/生菜温室 SMPC+RL.pdf`

外部链接：
https://www.sciencedirect.com/science/article/abs/pii/S0967066126000316

核心思想：
用 RL 学习 SMPC 的 terminal cost、terminal region constraints 和 nonlinear feedback policy，从而解决 SMPC 长预测时不确定性膨胀和短预测时性能不足的问题。

对本项目的启发：
当前 LLM 生成未来 11 步目标状态，本质上可以扩展为“terminal guidance”：LLM 负责给未来短期目标轨迹和终端安全区域，底层 rollout 在这个区域内跟踪。

可实现改进：
   - 在 `target_profile` 之外增加 terminal target：周期末温度/RH/CO2 目标区间。
   - fallback_plan 不只给一个动作，还给短期 terminal-safe plan。
   - 消融：有无 terminal target。

### 5. Grower-in-the-loop interactive RL

本地文件：
`D:/桌面/温室环控论文/生菜温室 交互式RL.pdf`

外部链接：
https://www.sciencedirect.com/science/article/pii/S0168169925014188

核心思想：
将种植者输入加入 RL。论文比较 reward shaping、policy shaping、control sharing，指出直接影响动作选择的方法更能处理不完美的人类输入。

对本项目的启发：
LLM 可以扮演“虚拟专家/种植者策略输入”，但不应该直接修改 reward。更合适的是 policy shaping / control sharing：LLM 给 anchor 和 setpoint，底层控制器保留最终执行权。

可实现改进：
   - 明确把 LLM anchor 作为 control sharing 权重的一部分。
   - 记录 rule_weight 随 plan progress 的变化。
   - 加入“专家建议错误/缺失”鲁棒性实验：LLM 输出异常时 fallback 能否守住安全。

### 6. Constrained temperature and RH predictive control

本地文件：
`D:/桌面/温室环控论文/受限温度和相对湿度预测控制：农业温室案例研究.pdf`

外部链接：
https://www.sciencedirect.com/science/article/pii/S2214317323000525

核心思想：
使用 N4SID 识别温度和相对湿度两个 MISO 模型，然后做带约束 MPC，使温度和 RH 保持在给定范围内。

对本项目的启发：
RH 不能只作为 reward 惩罚项，必须作为显式约束对象。当前新增的 RH-debt、risk cap 和风速-除湿协调，可以被写成“湿度约束管理模块”。

可实现改进：
   - 把 RH 约束分成硬约束、内部收紧约束和恢复区间。
   - 报告 RH violation area，而不是只报告最大值。
   - 加入 VPD/露点裕度作为 RH 约束的风险增强项。

### 7. Adaptive robust greenhouse climate control

本地文件：
`D:/桌面/温室环控论文/自适应稳健温室气候控制：结合深度强化学习和经济优化.pdf`

外部链接：
https://www.sciencedirect.com/science/article/pii/S2772375525005581

核心思想：
分层控制：上层经济优化生成参考状态轨迹，下层 DRL 控制器负责鲁棒实时跟踪。论文强调层级结构、动态能源价格、参考轨迹、鲁棒跟踪和迁移适应。

对本项目的启发：
这是最贴近当前项目的论文结构。我们的 LLM Director 可以作为上层经济/语义规划器，底层规则和 RH-debt guardrail 作为鲁棒跟踪器。

可实现改进：
   - 将 LLM 输出明确命名为 reference trajectory。
   - 加入动态电价或日内电价场景，凸显经济优化。
   - 增加 transfer test：不同年份、不同 day、不同天气扰动。

## 项目下一步建议

### A. 方法重命名

建议将方法命名为：

LLM-guided Robust Setpoint Planning and Safe Rollout Control

或：

LLM-RSPC: LLM-guided Robust Setpoint Planning for Greenhouse Climate Control

核心模块：
   - LLM high-level setpoint planner
   - Setpoint contract and target-profile completion
   - Fallback candidate scoring
   - RH-debt constraint tightening
   - Safe rollout controller

### B. 必做工程改进

1. PPO 强基线重新训练：
   - 多 seed。
   - 超参搜索。
   - validation set 选模型，test set 只汇报最终结果。

2. 960 步和全生长周期测试：
   - 240 步用于调试。
   - 960 步用于中期稳定性。
   - 40 天全周期用于论文主结果。

3. 消融实验：
   - Full method。
   - w/o RH-debt。
   - w/o target profile。
   - w/o fallback scoring。
   - w/o setpoint contract risk cap。
   - LLM anchor only。

4. 鲁棒性测试：
   - 不同年份。
   - 不同开始日期。
   - 不同 uncertainty_scale。
   - LLM 失败/缺失/异常输出场景。

### C. 论文主张

不要把论文主张写成“LLM 控制器比 PPO 更强”。更建议写成：

在温室气候控制中，LLM 适合承担高层目标规划和专家策略输入，而不适合直接承担每步低层控制。通过 setpoint contract、candidate fallback、RH-debt 约束收紧和安全 rollout，可以把 LLM 的语义推理能力转化为可验证、可解释、较安全的闭环控制策略。

