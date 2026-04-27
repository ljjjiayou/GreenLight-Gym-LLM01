# Related Work and Method Framework

## 拟定题目

建议题目方向：

**LLM-guided Robust Setpoint Planning for Safe and Economic Greenhouse Climate Control**

中文可表述为：

**面向安全与经济性的 LLM 引导鲁棒设定点规划温室气候控制方法**

核心不是证明 LLM 可以替代传统控制器，而是证明：

> LLM 适合承担高层策略理解、目标状态规划和专家知识注入；显式约束、候选 fallback 和底层安全 rollout 负责把 LLM 输出转化为可验证、可复现、可安全执行的控制动作。

## 1. Related Work 框架

### 1.1 约束预测控制与温室气候控制

温室气候控制通常需要同时优化作物生长、资源消耗和安全约束。传统 MPC 的优势是可以显式处理温度、湿度、CO2 和执行器约束，因此在农业温室控制中长期占有重要地位。

`受限温度和相对湿度预测控制：农业温室案例研究.pdf` 使用 N4SID 建立温度与相对湿度两个 MISO 状态空间模型，再使用 constrained MPC 控制风扇和加热器，使温度和相对湿度保持在约束范围内。该文对本项目最直接的启发是：RH 不能只作为 reward 违规惩罚，而应该作为显式约束管理对象。

本项目对应改进：

- 将环境 RH 硬约束外移为内部收紧约束。
- 增加 `RH-debt` 机制，记录连续高湿、低 VPD 和露点风险。
- 在 setpoint contract 中对 `target_rh` 做风险封顶。
- 在 safety guardrail 中协调风速保护和除湿需求，避免通风被后续护栏抵消。

论文中可写的差异：

传统 CMPC 通常依赖明确的线性或非线性预测模型，而本项目使用 LLM 生成高层参考轨迹，再由规则和安全护栏进行鲁棒执行。这样可以避免让 LLM 直接承担低层连续控制，同时保留显式约束思想。

### 1.2 强化学习与 MPC 的比较

`生菜温室 RL对比MPC.pdf` 系统比较了 RL 与 MPC 在温室气候控制中的数学形式和性能差异。MPC 的优势是约束清晰、可解释；RL 的优势是在线决策快、能从数据中学习长期策略；但 RL 难以天然保证安全约束。

对本项目的启发：

- PPO/SAC 等 RL 方法必须作为强 baseline，而不是随便拿一个旧模型对比。
- 对照实验必须报告 reward、profit、energy cost、constraint violation 和 runtime。
- LLM 方法的优势不应该只写成 reward 更高，而应强调可解释计划、安全过滤和对异常输出的可恢复性。

本项目对应改进：

- 需要重新训练 PPO 强基线：多 seed、超参搜索、验证集选模型、测试集报告。
- LLM Director 需要报告 LLM 调用次数、平均耗时、fallback rate。
- 需要消融验证每个安全模块的贡献。

### 1.3 RL-MPC：让学习方法改进预测控制

`RL控制.pdf` 与 `生菜温室 RL+MPC.pdf` 是同一篇 RL-MPC 论文。该文提出用 RL 在线学习参数化 MPC 中的约束、模型和目标函数参数，以减少模型误差和天气预测误差导致的约束违规。

对本项目的启发：

LLM 可以被解释为一种高层参数化规划器。它不直接优化底层连续动作，而是输出：

- anchor action；
- `target_temp`、`target_co2`、`target_rh`；
- 未来目标状态 profile；
- 对除湿、节能、CO2、补光等策略的语义偏好。

这些输出相当于为底层控制器提供短期 MPC-like reference trajectory 和约束偏好。

本项目对应改进：

- 将 `target_profile` 明确命名为 reference trajectory。
- 保存 `profile_contract`，记录 LLM 原始轨迹如何被裁剪、补全和修正。
- 增加 terminal-safe target，即规划周期末的目标安全区域。

### 1.4 鲁棒优化与 DRL 的结合

`基于人工智能的半封闭式温室节能控制：利用深度强化学习中的鲁棒优化.pdf` 将 DRL 与鲁棒优化结合，用于半封闭温室节能控制。该文强调不确定性处理、能耗下降和 setpoint deviation 下降。

对本项目的启发：

当前项目的 LLM 调用很慢，但底层候选控制和安全 rollout 很快。因此可以把 LLM 规划看成低频高层策略，而把 uncertainty handling 放在快速规则层或候选评分层。

本项目对应改进：

- 增加天气扰动下的鲁棒性评估。
- 候选 fallback scoring 不只看当前状态，还加入未来天气趋势、RH 趋势和执行器冲突。
- 比较 deterministic 和 uncertainty_scale 下的表现。

### 1.5 SMPC + RL：短视预测与终端约束

`生菜温室 SMPC+RL.pdf` 提出 RL-SMPC，用 RL 学习 terminal cost、terminal region constraints 和 nonlinear feedback policy。它解决的是 SMPC 短预测时性能不足、长预测时不确定性膨胀的问题。

对本项目的启发：

LLM 生成未来 11 步目标状态时，本质上就是在给底层控制器一个短期参考轨迹。下一步可以进一步加入 terminal target：

- 周期末 RH 应低于某阈值；
- 周期末温度不能跌入冷害区；
- 周期末 CO2 不应在高通风下继续提升；
- 规划结束时系统应落在安全集合内。

本项目对应改进：

- 在 plan 中加入 `terminal_target` 字段。
- fallback_plan 不只提供一个动作，还提供短期 terminal-safe plan。
- 做消融：有无 terminal target。

### 1.6 Grower-in-the-loop 与 LLM 作为专家输入

`生菜温室 交互式RL.pdf` 研究 grower-in-the-loop interactive RL，比较 reward shaping、policy shaping 和 control sharing。该文的重要结论是：在专家输入不完美时，直接影响动作选择的 policy shaping / control sharing 通常比修改 reward 更稳。

对本项目的启发：

LLM 可以视为虚拟 grower 或专家策略输入，但不应该直接改 reward。更合理的是让 LLM 参与 policy shaping / control sharing：

- LLM 给 anchor；
- LLM 给 setpoint；
- 底层规则控制器仍保留最终执行权；
- 安全护栏可以覆盖 LLM 输出。

本项目对应改进：

- 继续保留 anchor 与 rule action 的加权融合。
- 报告 `rule_weight` 随 plan progress 的变化。
- 设计 LLM 输出错误、缺失、延迟时的鲁棒性测试。

### 1.7 分层鲁棒控制与经济优化

`自适应稳健温室气候控制：结合深度强化学习和经济优化.pdf` 采用分层结构：上层经济优化生成参考轨迹，下层 DRL 控制器负责鲁棒实时跟踪。该文与本项目架构最接近。

对本项目的启发：

LLM Director 可作为上层经济/语义规划器，底层规则、fallback 和 RH-debt 模块作为鲁棒跟踪器。这样能形成清楚的层级控制表述：

- 上层：低频、语义、经济、多目标规划；
- 下层：高频、约束、安全、鲁棒执行。

本项目对应改进：

- 将 LLM 输出统一称为 reference plan。
- 增加动态电价或日内能耗权重，强化 economic optimization。
- 用不同年份、开始日期和不确定性水平做迁移测试。

## 2. 文献启发下的方法定义

### 2.1 方法名称

建议命名：

**LLM-RSPC: LLM-guided Robust Setpoint Planning and Safe Rollout Control**

中文：

**LLM 引导的鲁棒设定点规划与安全滚动控制**

### 2.2 控制问题

温室状态记为：

```text
s_t = [T_air, CO2_air, RH_air, radiation, outdoor_weather, crop_state, actuator_state, time]
```

控制动作记为：

```text
u_t = [heating, co2, screen, ventilation, lighting, shading]
```

控制目标：

- 最大化经济收益和长期 reward；
- 降低加热、CO2、补光成本；
- 保持温度、CO2、RH 在安全范围；
- 避免高湿、低 VPD 和结露风险；
- 在 LLM 失败时仍能安全执行。

### 2.3 模块 1：状态分析与重规划触发

状态分析器读取当前环境状态，判断是否需要重规划。

触发条件包括：

- 初始无计划；
- 当前计划到期；
- 温度、CO2、RH 出现紧急风险；
- RH 连续接近硬约束；
- 低 VPD 或露点裕度不足；
- LLM 计划已经不适配当前天气。

该模块对应文献中的 event-triggered MPC 或 receding horizon 思想。

### 2.4 模块 2：LLM 高层参考计划

LLM 不直接逐步控制温室，而是生成一个短期 reference plan：

```text
P_t = {
  anchor_action,
  target_temp,
  target_co2,
  target_rh,
  target_temp_profile,
  target_co2_profile,
  target_rh_profile,
  reasoning
}
```

其中：

- `anchor_action` 是本周期第一个动作锚点；
- `target_*` 是标量目标；
- `target_*_profile` 是未来若干步目标轨迹；
- `reasoning` 用于解释当前规划目的和约束。

这对应 RL-MPC 和分层控制文献中的高层参考轨迹。

### 2.5 模块 3：Setpoint Contract

LLM 输出必须经过合同机制：

- 缺失目标自动填充；
- 越界目标自动裁剪；
- 高湿风险时自动降低 `target_rh` 上限；
- profile 长度不足时自动扩展；
- profile 过长时裁剪；
- 保存修正记录用于可解释性和消融实验。

该模块是本项目区别于普通 LLM agent 的关键安全层。

### 2.6 模块 4：Fallback Candidate Scoring

若 LLM 没有生成有效 anchor，系统不直接失败，而是构造候选控制：

- hold current；
- rule controller；
- adaptive rule；
- recent anchor；
- emergency dehumidify；
- cold survival；
- heat relief；
- economy hold。

每个候选按以下风险评分：

```text
score = temp_risk + rh_risk + co2_risk + cost_risk + smoothness_risk + conflict_risk
```

选择 score 最低的候选作为 fallback anchor。

这对应文献中 robust backup policy 和 control sharing 的思想。

### 2.7 模块 5：RH-debt Constraint Tightening

为解决长周期 RH 轻微越界反复积累的问题，引入湿度债务：

```text
D_t = decay * D_{t-1}
      + near_constraint_risk
      + violation_risk
      + low_vpd_risk
      + dew_risk
```

当 `D_t` 升高时：

- 提高通风底线；
- 降低保温幕闭合上限；
- 关闭 CO2 和补光；
- 在低温时允许轻度 heating + ventilation 除湿。

该模块可写成“RH constraint tightening with memory”，对应 MPC 中的约束收紧和鲁棒余量思想。

### 2.8 模块 6：Safe Rollout Control

在两个 LLM 规划周期之间，底层控制器执行滚动控制：

```text
u_rollout = (1 - w) * u_anchor + w * u_rule + tracking_correction
```

其中 `w` 随计划进度增加，使系统逐步从 LLM anchor 过渡到规则闭环控制。

之后经过：

- 温度护栏；
- RH/VPD/露点护栏；
- CO2 效率护栏；
- 补光预算护栏；
- 风速保护；
- 执行器限幅。

最终输出可执行动作。

## 3. 正式 Method 章节建议结构

### 3.1 System Architecture

介绍整体结构：

```text
State Observer
  -> Risk Analyzer
  -> Event-triggered LLM Planner
  -> Setpoint Contract
  -> Fallback Candidate Scoring
  -> Safe Rollout Controller
  -> Guardrail-filtered Action
  -> Greenhouse Environment
```

建议配一张流程图，强调 LLM 只在低频重规划时调用。

### 3.2 LLM as High-level Setpoint Planner

说明为什么不让 LLM 直接控制每一步：

- LLM 输出不稳定；
- LLM 调用延迟高；
- 连续控制需要严格约束；
- 温室动力学有滞后和不确定性。

因此 LLM 只负责高层规划。

### 3.3 Setpoint Contract and Profile Completion

形式化写出合同机制：

```text
z_t = C(z_t^LLM, s_t)
```

其中 `C` 是合同函数，保证目标值合法、安全、完整。

### 3.4 Candidate-based Fallback Planning

给出候选集合：

```text
U_fallback = {u_hold, u_rule, u_adaptive, u_recent, u_dehumidify, ...}
```

选择：

```text
u_fallback = argmin score(u_i, s_t)
```

### 3.5 RH-debt Safe Rollout

解释 RH-debt 的必要性：

短期 RH 只超一点不一定触发强护栏，但长期累计会导致显著 violation area。RH-debt 让控制器记住历史湿度压力。

### 3.6 Runtime and Failure Handling

LLM 控制论文必须报告 runtime：

- PPO 推理极快；
- LLM 调用较慢；
- 但 LLM 调用是低频的；
- 需要 fallback 防止 API 异常影响安全。

## 4. 实验设计

### 4.1 Baselines

必须包含：

- Rule-based controller；
- PPO；
- SAC 或 DDPG；
- LLM anchor only；
- LLM + setpoint contract；
- Full LLM-RSPC。

若能实现，还可加入：

- MPC baseline；
- simplified economic MPC；
- oracle setpoint planner。

### 4.2 PPO 强基线训练

PPO 不能只用现有模型。论文级 PPO baseline 应满足：

- 多 seed 训练；
- 超参搜索；
- validation set 选择 best checkpoint；
- test set 只汇报最终模型；
- 与 LLM 使用同一环境、同一 reward、同一 constraints。

建议搜索：

- learning rate；
- batch size；
- n_steps；
- gamma；
- gae_lambda；
- clip_range；
- ent_coef；
- vf_coef；
- network architecture；
- total_timesteps。

### 4.3 Evaluation Scenarios

建议分三层：

1. Short horizon smoke test：240 steps。
2. Medium horizon stability test：960 steps。
3. Full growth cycle：40 days。

天气和日期：

- 不同年份；
- 不同开始日期；
- 多 seed；
- uncertainty_scale = 0.0, 0.1, 0.2, 0.3。

### 4.4 Metrics

主指标：

- total reward；
- total profit；
- revenue；
- heating cost；
- CO2 cost；
- electricity cost；
- temp violation；
- CO2 violation；
- RH violation；
- safety pass rate。

LLM 专属指标：

- LLM call count；
- mean LLM latency；
- fallback rate；
- setpoint contract correction count；
- RH-debt peak；
- target profile correction count。

### 4.5 Ablation Study

必须做：

- Full；
- w/o RH-debt；
- w/o setpoint contract；
- w/o target profile；
- w/o fallback candidate scoring；
- w/o dynamic interval；
- anchor only。

预期：

- 去掉 RH-debt 后，长周期 RH violation 显著增加；
- 去掉 setpoint contract 后，LLM 输出偶尔会导致目标过湿或不完整；
- 去掉 fallback scoring 后，LLM 失败时安全性下降；
- anchor only 在短期可能有效，但长期稳定性弱。

### 4.6 Robustness Study

测试：

- LLM timeout；
- LLM missing action；
- LLM malformed output；
- high humidity weather；
- cold night；
- high radiation day；
- actuator constraints。

重点证明：

即使 LLM 不完美，底层安全机制仍能保持温室状态可控。

## 5. 项目代码改进清单

### P0：论文主实验前必须完成

- 固定环境、reward、observation 和 constraints 版本。
- 重新训练 PPO 强基线。
- 增加统一实验 runner，支持 PPO、Rule、LLM、ablation。
- 输出每步日志，包括 state、target、control、violation、contract correction。
- 添加 960 步和 40 天实验配置。

### P1：增强方法创新性

- 增加 terminal target。
- 增加天气趋势风险评分。
- 增加 LLM call cache 或 direct JSON planner，降低 runtime。
- 增加 `LLM failure injection` 测试。

### P2：论文可视化

- RH 曲线与 RH hard limit；
- RH-debt 曲线；
- target_rh profile 与实际 RH；
- anchor/rule/applied control 对比；
- PPO vs LLM 的 profit-cost-violation tradeoff；
- LLM runtime and call count。

## 6. 论文贡献表述

建议贡献写成四点：

1. 提出 LLM-RSPC，一种将 LLM 作为高层设定点规划器、底层安全控制器负责约束执行的温室气候混合控制框架。
2. 提出 setpoint contract 和 target profile completion，使 LLM 输出变成完整、合法、可执行的短期参考轨迹。
3. 提出 fallback candidate scoring 与 RH-debt constraint tightening，提高 LLM 失败和长周期高湿场景下的安全性。
4. 在 GreenLight-Gym 风格的温室环境中，与 PPO、规则控制和消融变体进行对比，验证安全性、经济性和可解释性。

## 7. 下一步执行顺序

1. 锁定当前环境与 reward 版本。
2. 设计 PPO 强基线训练脚本和 sweep。
3. 将 LLM-RSPC 模块参数化，支持 ablation 开关。
4. 跑 240、960、40 天实验。
5. 根据实验结果反向优化 RH-debt、fallback score 和 setpoint contract。
6. 写论文 Related Work、Method、Experiment。

