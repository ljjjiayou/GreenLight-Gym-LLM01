# Greenhouse LLM Hybrid Control: Method Plan

## 核心问题

当前项目的目标不是单纯让 LLM 直接输出 6 维控制量，而是构建一个“高层语义规划 + 底层约束执行”的温室闭环控制器。论文可以聚焦在：在 GreenLight-Gym 类型的温室动力学环境中，LLM 如何作为可解释的短期目标规划器，与规则控制、fallback 候选评分、约束护栏共同形成稳定控制策略。

## 方法概述

1. 状态解析
   系统读取温室状态，包括温度、CO2、RH、光照、室外天气、作物状态、露点风险等，并给出是否需要重规划的判断。

2. 高层 LLM 规划
   LLM 不直接逐步控制所有执行器，而是在重规划时输出：
   - 本周期锚点动作：heating、co2、screen、ventilation、lighting、shading
   - 本周期目标状态：target_temp、target_co2、target_rh
   - 可选的未来目标轨迹：target_temp_profile、target_co2_profile、target_rh_profile

3. Setpoint Contract
   所有目标状态都经过合同机制校验。若 LLM 缺失目标，则由规则填充；若目标越界，则裁剪；若存在高湿、低 VPD 或露点风险，则自动收紧 target_rh，避免 LLM 把湿度目标设得过高。

4. Fallback Candidate Scoring
   当 LLM 未产生有效锚点动作时，系统不再只使用单一路径 fallback，而是构造多个候选动作，包括规则控制、保持当前、自适应控制、历史锚点、紧急除湿、冷害保护、热害缓解、省成本保持等。候选动作通过温度风险、RH 风险、成本、平滑性和控制冲突进行评分，选择风险最低的动作作为规划锚点。

5. RH-Debt Constraint Tightening
   为解决长周期中 RH 轻微越界反复积累的问题，引入 RH violation debt：
   - RH 接近硬约束时开始积累风险债务
   - RH 超过约束、VPD 过低、露点裕度过小时加速积累
   - 环境恢复干爽后债务快速衰减
   - 债务越高，底层 rollout 越倾向提高通风底线、打开保温幕、关闭补光和 CO2

6. Plan Rollout
   在 LLM 两次重规划之间，底层控制器读取当前计划，按计划进度融合：
   - LLM 锚点动作
   - 规则控制动作
   - 当前目标轨迹误差修正
   然后再经过安全护栏和执行器映射，输出实际控制动作。

## 本轮新增机制

### 1. 高湿内部约束收紧

环境硬约束是 RH <= 90%，但控制器内部不等到 90% 才反应，而是在 86% 左右开始预控。这样相当于使用 robust MPC 中常见的 constraint tightening 思路：把真实硬约束前移，给模型误差、天气扰动和执行器滞后留余量。

### 2. 风险相关 RH 目标封顶

当当前 RH、VPD 或露点裕度提示病害/结露风险时，合同机制会把 target_rh 的上限从常规 88% 降到更保守区间：
   - 预警高湿：target_rh <= 80%
   - 高湿/低 VPD：target_rh <= 76%
   - 极高湿/结露风险：target_rh <= 72%

这能避免 LLM 在短期收益目标下给出“看似合理但过湿”的未来目标。

### 3. RH Violation Debt

原来的控制只看当前 RH，长周期会出现“每次只超一点点，但累计惩罚很大”的问题。新增的 RH-debt 把这种历史风险记住：

```
debt_t = decay * debt_{t-1}
       + near_constraint_risk
       + violation_risk
       + low_vpd_risk
       + dew_risk
```

债务直接影响通风底线和保温幕上限，使系统在连续潮湿时越来越积极地排湿。

### 4. 风速护栏与除湿护栏协调

之前 RH 已经高于 88% 时，后面的风速保护可能又把通风压到 0.20，抵消除湿动作。现在风速保护根据 RH 风险分层设定通风上限：风大时仍保护结构，但在 RH 高风险时允许更高的最小排湿能力，避免湿度被锁死。

## 可发表实验设计

建议主实验：
   - PPO baseline
   - Rule-based baseline
   - LLM anchor only
   - LLM + setpoint contract
   - LLM + setpoint profile
   - Full method: LLM + profile + fallback scoring + RH-debt guardrail

建议消融：
   - w/o RH-debt
   - w/o fallback candidate scoring
   - w/o target profile
   - w/o setpoint contract risk cap
   - fixed interval vs dynamic interval

核心指标：
   - total_reward
   - economic profit
   - heating / CO2 / lighting cost
   - temp violation
   - CO2 violation
   - RH violation
   - safety pass rate
   - LLM call count and runtime

论文贡献可以表述为：
   - 将 LLM 从直接控制器转化为高层 setpoint planner，降低动作噪声和执行风险
   - 提出可解释 fallback candidate scoring，减少 LLM 缺失动作时的不确定性
   - 引入 RH-debt 约束收紧机制，解决长周期高湿尾部累计违规
   - 用 PPO、规则控制和多组消融实验验证收益、安全性和可解释性
