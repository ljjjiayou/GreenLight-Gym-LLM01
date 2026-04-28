# LLM-RSPC 优化日志

本文档用于记录方法层面的改动，便于后续复现实验和撰写论文。每条记录应说明改动动机、项目变更、预期实验以及剩余风险。

## 2026-04-28：Expert-Distilled Safe Rollout，第一阶段

### 动机

当前的 LLM-RSPC 控制器具有较强的安全结构，但底层 rollout 仍主要依赖人工编写的规则。已有 PPO 策略在部分短时域场景中表现更好，尤其是在收益和能耗权衡方面；但是直接使用 PPO 控制器存在解释性弱、约束难的问题。受温室 RL+MPC、鲁棒强化学习以及自动化温室决策系统相关论文启发，下一步优化思路是：将 PPO 的短期控制习惯蒸馏为一个小型专家模块，并且只把它作为现有安全 rollout 候选池中的候选动作来源之一。

### 代码改动

- 新增 `gl_gym/agent/expert_distillation.py`。
  - 定义了一个在轨迹导出和部署阶段共享的、具有物理含义的特征向量。
  - 特征包括温室气候状态、执行器状态、天气预报摘要、VPD/结露风险特征、周期时间编码、RH 违规债务、补光预算以及 setpoint 跟踪误差。
  - 定义 `DistilledExpertPolicy`，这是一个经过归一化处理的 ridge 专家模型，用于预测 `[0, 1]` 范围内的原始控制动作。

- 新增 `gl_gym/experiments/export_ppo_trajectories.py`。
  - 在可配置的 year/day/seed 窗口上运行已有 PPO 模型。
  - 保存用于审计的 JSONL 逐步轨迹，以及用于训练的压缩 NPZ 矩阵。
  - 记录状态特征、PPO 动作命令、规划原始控制、实际执行原始控制、奖励以及约束/成本指标。

- 新增 `gl_gym/experiments/train_distilled_expert.py`。
  - 基于导出的 PPO 轨迹训练一个紧凑的归一化 ridge 回归器。
  - 保存验证集 MAE/MSE，以及用于 OOD 过滤的特征距离统计量。

- 更新 `gl_gym/agent/llm_agent.py`。
  - 新增可选配置 `expert_rollout_enabled`。
  - 只有在显式启用时才加载蒸馏专家模型。
  - 将 `distilled_expert`、`expert_anchor_blend` 和 `expert_rule_blend` 加入 rollout 候选池。
  - 保持现有候选评分和安全护栏作为最终裁决机制。
  - 在 rollout 诊断中记录 `expert_prediction`。
  - 默认使用专家模型的特征距离统计量作为 OOD 门控，而不是无条件接受所有专家预测。

- 更新 `gl_gym/experiments/compare_ppo_rule_based.py`。
  - 新增 CLI 开关，用于在评估 LLM-RSPC 时启用蒸馏专家。

### 预期实验

1. 在多个训练窗口上导出 PPO 专家轨迹：

   ```powershell
   python gl_gym/experiments/export_ppo_trajectories.py --years 2018,2019,2020 --days 120,180,240 --max-steps 240
   ```

2. 训练蒸馏专家：

   ```powershell
   python gl_gym/experiments/train_distilled_expert.py
   ```

3. 进行消融实验：

   - PPO baseline。
   - 不使用专家的 LLM-RSPC。
   - LLM-RSPC + Expert-Distilled Safe Rollout。
   - Rule-based baseline。

### 剩余风险

- 当前 PPO 模型还不是适合论文级对照实验的最优 PPO baseline。它应被视为第一版专家来源，而不是最终公平对照基线。
- 如果训练天气窗口过少，蒸馏专家可能过拟合。下一阶段应在更多年份、日期、随机种子和不确定性尺度下导出轨迹。
- 由于专家只作为候选动作来源，当它与当前 setpoint 合同冲突时，可能会被安全评分拒绝。这是有意设计的安全机制，应在论文中作为安全设计选择进行说明。
- 在 2020/day240 上进行的第一次 240 步闭环测试没有优于之前的 LLM-RSPC 控制器。诊断显示，即使特征距离超出训练分布，专家预测仍被接受，因此 OOD 门控已改为使用模型元数据进行收紧。

### 论文角度

该改动支持如下方法主张：

> LLM-RSPC 将高层可解释 setpoint 规划与底层执行解耦。PPO 知识被蒸馏为轻量 rollout 候选，而规则合同、候选评分和安全护栏共同保持系统安全性与可解释性。

## 2026-04-28：PPO vs LLM-RSPC 诊断工具

### 动机

继续手动调整阈值已经不是最优下一步。控制器需要一个可重复的诊断工具，用来回答为什么 PPO 在收益上更好，而 LLM-RSPC 往往在湿度安全上更好。因此，该诊断工具记录匹配的 PPO 与 LLM-RSPC 轨迹，并按照温室状态区间分组统计动作、成本和安全差异。

### 代码改动

- 新增 `gl_gym/experiments/diagnose_ppo_vs_llm.py`。
  - 在相同 year/day/seed 窗口上运行 PPO 和 LLM-RSPC。
  - 保存逐步 JSON 和 CSV 轨迹。
  - 汇总 reward/profit/cost/violation 指标。
  - 计算每个动作维度上的 PPO 与 LLM-RSPC 差异。
  - 按 phase、RH band、temperature band 和 radiation band 分组统计结果。
  - 统计安全/成本模式，例如 heat+vent 冲突、通风时 CO2 泄漏、高风险补光使用以及 RH>=90 状态步数。

### 第一次 smoke 测试结果

场景：2020/day240，seed 42，24 步。

- PPO reward：15.063；profit：-0.00453；RH violation：1.467。
- LLM-RSPC reward：15.032；profit：-0.00712；RH violation：0.000。
- 在这个短窗口中，LLM-RSPC 的 RH 安全性更好，但加热成本更高。
- 平均绝对动作差异最大的两个维度是 screen 和 ventilation。
- Heat+vent 冲突步数：
  - PPO：1。
  - LLM-RSPC：8。

### 下一步假设

下一步控制器改进应集中在除湿成本门控上：LLM-RSPC 已经能较积极地控制 RH，但需要针对 heat+vent 脉冲加入 duty-cycle 或边际收益门控。一个可发表的表述是：

> 面向安全 rollout 控制的湿度风险感知经济脉冲门控。

## 2026-04-28：用于安全蒸馏的 PPO 策略标签审计

### 动机

只有当 PPO 的连续动作能够被翻译成稳定、可审计的策略倾向时，PPO 才适合作为专家来源。之前的轨迹导出和 ridge 蒸馏可以从数值上复制动作，但这对于论文级温室控制器还不够，因为一个混合动作，例如同时加热和通风，可能表示除湿、无效能耗冲突，也可能只是某种过渡动作；具体含义取决于 RH 风险以及作物-气候上下文。

因此，在进一步进行 PPO-to-LLM-RSPC 蒸馏之前，本步骤加入一个保守的策略标签层。PPO 被当作行为参考和诊断镜子，而不是被视作绝对正确的 ground-truth 控制器。

### 代码改动

- 新增 `gl_gym/agent/ppo_strategy_labeler.py`。
  - 将每个连续动作与一组可解释策略候选进行打分：economy hold、vent-only dehumidification、heat+vent dehumidification、screen-release dehumidification、heat preservation、free air exchange、cooling ventilation、CO2 enrichment、lighting assist 和 shade cooling。
  - 评分同时使用动作值和气候特征，例如 RH、温度、VPD、结露风险、辐射和一天中的时间。
  - 当最高分过低时，将样本标记为 `unknown`；当前两名策略分数过近时，将样本标记为 `ambiguous`。
  - 导出完整 soft score 向量和原因字符串，便于后续审计。

- 更新 `gl_gym/experiments/diagnose_ppo_vs_llm.py`。
  - 为 PPO 和 LLM-RSPC 的轨迹行都添加策略标签。
  - 增加标签审计模块，统计 confident/ambiguous/unknown 数量。
  - 增加 useful-PPO-tendency 过滤器：只有当高置信度 PPO 标签具有同一步 profit 优势，并且不会让同一步 RH 或温度违规增加超过小阈值时，才将其计为有价值倾向。

- 新增 `tests/test_ppo_strategy_labeler.py`。
  - 覆盖湿度风险感知除湿标签、heat preservation、free-air exchange，以及对弱混合动作的保守拒绝。

### 第一次审计结果

场景：2020/day240，seed 42，24 步。

- PPO reward：15.063；profit：-0.00453；RH violation：1.467。
- LLM-RSPC reward：15.032；profit：-0.00712；RH violation：0.000。
- Heat+vent 冲突步数：
  - PPO：1。
  - LLM-RSPC：8。
- 策略标签审计：
  - PPO：8 个高置信度 `free_air_exchange` 样本，1 个 ambiguous 样本，15 个 unknown 样本。
  - LLM-RSPC：14 个高置信度 `economy_hold` 样本，1 个高置信度 `free_air_exchange` 样本，9 个 unknown 样本。
- 有价值的 PPO 倾向候选：
  - 8 个高置信度、有价值的 PPO 样本。
  - 这 8 个样本全部是 `free_air_exchange`。
  - 在这个短窗口中，没有高置信度 PPO 样本被同一步安全过滤器拒绝。

### 解释

第一次标签审计支持当前优化方向。在该窗口中，LLM-RSPC 的 RH 安全性更好，但支付了额外加热成本，并触发了更多 heat+vent 冲突。PPO 中有价值的倾向并不是“更强除湿”，而是在中等湿度风险下的低成本空气交换：高通风、打开保温幕、低加热，并且不使用 CO2 和补光。

因此，下一步控制器改动应是除湿经济门控：

- 当 RH/结露风险严重时，保留现有安全除湿行为；
- 当 RH 风险中等且温度不太低时，在 heat+vent 脉冲之前优先尝试有界的 free-air-exchange 候选；
- 当动作含义模糊或超出已审计区域时，回退到现有规则和护栏栈。

相比直接克隆 PPO，这能形成更清晰的论文主张：

> LLM-RSPC 从 PPO 轨迹中挖掘可解释、带置信度门控的策略倾向，然后只将经过安全过滤的经济行为注入到规则约束的 rollout 控制器中。

## 2026-04-28：Plan-Conditioned Intent Residual Distillation

### 动机

上一阶段已经能把 PPO 连续动作翻译成可审计策略标签，但仍存在一个关键缺口：PPO 轨迹导出时没有显式包含 LLM-RSPC 的目标计划条件。因此，专家模型学到的是 `state -> action`，而不是 `state + target -> intent -> action`。这会削弱泛化能力，也不利于论文中解释“为什么此时采用这个动作”。

本阶段将 PPO 蒸馏增强为目标条件化意图蒸馏。核心思路是：先用 setpoint 合同为每个 PPO 状态补一个伪目标计划，再根据当前状态与目标状态差距推断控制意图，最后只把与目标意图一致的 PPO 策略样本用于 residual 专家训练。

### 代码改动

- 新增 `gl_gym/agent/plan_intent.py`。
  - 提供 `default_setpoint_contract`，为 PPO 轨迹生成状态驱动的伪目标计划。
  - 提供 `infer_plan_intent`，根据当前状态、目标温度、目标 RH、目标 CO2、VPD 和结露风险推断控制意图。
  - 当前意图包括 `economic_dehumidify`、`safe_dehumidify`、`heat_recovery`、`heat_preservation`、`cooling`、`co2_enrichment`、`lighting_assist`、`humidity_preservation` 和 `economy_hold`。
  - 提供 `strategy_intent_alignment`，判断 PPO 动作策略标签是否服务于当前目标意图。
  - 提供 `target_tracking_baseline_control`，作为 residual 蒸馏的可解释 baseline。

- 更新 `gl_gym/experiments/export_ppo_trajectories.py`。
  - 每条 PPO 轨迹现在同时记录 `target_plan`、`plan_contract`、`intent`、`strategy` 和 `intent_strategy_alignment`。
  - 特征提取时将伪目标计划传入 `extract_expert_features`，使 `target_temp_delta`、`target_co2_delta`、`target_rh_delta` 不再为空壳特征。
  - 同时保存 `base_action` 和 `residual_action = PPO action - base_action`。
  - NPZ 数据集新增 intent/strategy 标签、置信度和目标一致性 mask，便于后续训练过滤。

- 更新 `gl_gym/experiments/train_distilled_expert.py`。
  - 新增 `--require-intent-aligned`，只使用目标意图与 PPO 策略一致的样本。
  - 新增 `--target-mode action|residual`，支持直接动作蒸馏和 residual 蒸馏。
  - residual 模式下，模型学习的是 PPO 相对目标跟踪 baseline 的修正量，而不是完整动作硬复制。

- 更新 `gl_gym/agent/expert_distillation.py`。
  - `DistilledExpertPolicy` 现在可以读取模型元数据中的 `target_mode`。
  - 当 `target_mode=residual` 时，专家先预测 residual，再与 `target_tracking_baseline_control` 合成最终动作。

- 更新 `gl_gym/agent/llm_agent.py`。
  - rollout 调用专家时会推断当前 plan intent。
  - 专家预测会记录 `plan_intent_label`、`expert_strategy_label` 和 `intent_strategy_alignment`。
  - 默认启用轻量 intent gate：只有当目标意图和专家动作策略都很明确但互相冲突时，才拒绝专家候选。

- 新增 `tests/test_plan_intent_distillation.py`。
  - 覆盖伪 setpoint 合同、经济除湿意图、策略-意图一致性、以及 residual 专家动作合成。

### Smoke 测试

场景：2020/day240，seed 42，24 步 PPO 轨迹。

导出命令：

```powershell
python gl_gym\experiments\export_ppo_trajectories.py --years 2020 --days 240 --max-steps 24 --base-seed 42 --output-jsonl gl_gym\result\expert_distillation\ppo_plan_intent_smoke_s24.jsonl --output-npz gl_gym\result\expert_distillation\ppo_plan_intent_smoke_s24.npz
```

标签统计：

- 目标意图：
  - `safe_dehumidify`: 18。
  - `economic_dehumidify`: 6。
- PPO 策略标签：
  - `free_air_exchange`: 8。
  - `ambiguous`: 1。
  - `unknown`: 15。
- 目标意图与 PPO 策略一致样本：
  - aligned: 8。
  - not aligned: 16。

Residual 专家训练命令：

```powershell
python gl_gym\experiments\train_distilled_expert.py --input gl_gym\result\expert_distillation\ppo_plan_intent_smoke_s24.npz --output-model train_data\AgriControl\ppo\deterministic\distilled_expert\llm_rspc_intent_residual_smoke_s24.npz --output-report gl_gym\result\expert_distillation\llm_rspc_intent_residual_smoke_s24_metrics.json --target-mode residual --require-intent-aligned --min-intent-confidence 0.5 --min-strategy-confidence 0.55
```

训练结果：

- 源样本：24。
- 目标一致样本：8。
- residual validation MSE：0.000328。
- validation MAE：
  - heating: 0.0173。
  - ventilation: 0.0406。
  - screen: 0.0000。
  - CO2/lighting/shading: 0.0000。

### 解释

该结果说明新链路能把 PPO 中可解释、与目标一致的低成本换气行为筛选出来，而不是直接复制全部 PPO 动作。对于当前窗口，PPO 的可借鉴部分仍然集中在 `free_air_exchange`，并且这些样本与目标意图过滤后的数量一致。

下一步应将该流程扩展到更多年份、日期和随机种子，并进行四组消融实验：

- LLM-RSPC 原版。
- LLM-RSPC + action-only PPO distillation。
- LLM-RSPC + plan-conditioned action distillation。
- LLM-RSPC + plan-conditioned intent residual distillation。

该方向的论文表述可以是：

> 通过目标条件化意图推断，将 PPO 行为从黑箱动作模仿转化为“目标差距 -> 控制意图 -> 安全 residual 候选”的可解释蒸馏过程。
