# v39 Qwen-Max Long-Horizon Control Improvement DeepResearch Brief

> Date: 2026-05-29  
> Purpose: 给 GPT DeepResearch 的路线分析输入。本文只作为研究与工程规划建议，不自动变成控制逻辑、实验授权、performance claim 或默认控制器 promotion。

## 1. 当前项目状态

当前项目已经不再处于最早期的 protocol/readiness 补洞阶段。已有工作基本完成了以下基础：

- `llm_rspc_v2` 默认控制器保持不变。
- runtime provenance、joint prediction、metadata trace、action-diff 等审计链路已经建立。
- strict metadata replay、minimal controlled canary、triggered canary、Wave-1 expanded controlled canary 等安全验证已经走过。
- v38 已经开始从短窗口 strict-trigger 搜索转向长周期可评估场景池。

最新 v38 H240 结果：

```text
candidate_count = 32
stable_scenario_count = 23
baseline_runtime_error_count = 0
default_runtime_error_count = 9
cvodes_failure_count = 9
simulator_or_weather_numerical_instability_count = 0
default_controller_action_induced_instability_count = 9
next_action = execute_v38_h720_stable_pool_feasibility
```

解释：

- 23 个场景已经具备 H240 层面的稳定长周期评估潜力。
- 9 个场景里，PPO baseline 能跑，但默认 `llm_rspc_v2` 出现 runtime/CVODES 或 cache 不完整，说明问题更像是默认 LLM 控制路径或动作轨迹诱发的稳定性问题，而不是天气本身所有控制器都跑不动。
- 这些失败不能直接写成控制性能失败，但必须作为控制稳定性和仿真有效性的边界条件。

当前仍然禁止：

```text
controlled replay 扩展
默认控制器 promotion
performance claim
threshold tuning
直接把 shadow/canary evidence 写成控制效果提升
```

## 2. 本轮路线修正

用户当前判断是：

```text
暂不切换到 qwen-3.7；
先继续使用 qwen-max-latest，把当前资源包用完；
不再把 cache / strict-trigger 搜索作为主线；
改为通过长周期或全周期 rollout 暴露更全面的控制场景；
profile 是根据具体场景生成的，因此必须在真实长周期场景中评估；
rollout 过程中可以发现候选动作池的缺口，并逐步优化候选动作。
```

我认为这个方向总体合理，而且比继续短窗口 strict-trigger 搜索更接近控制效果优化。

原因：

1. 现在立即切换模型会增加归因混乱。  
   如果同时换成 qwen-3.7、改 profile、改候选动作、改控制稳定性，很难判断效果变化来自哪里。

2. cache 应退居为复现工具。  
   cache 对 replay、证据冻结、失败复现非常重要，但它不是发现长周期控制问题的最佳工具。下一阶段可以在线跑 qwen-max-latest，必要时只做 isolated passive record，不把 cache coverage 作为探索阶段的主 gate。

3. 长周期 rollout 更适合暴露真实控制问题。  
   短窗口更适合验证某个 safety gate 或 strict trigger，但无法充分暴露日夜切换、湿度积累、VPD 漂移、CO2 与通风冲突、补光与温度冲突、能耗积累等问题。

4. profile 质量必须在连续轨迹中判断。  
   单步 profile 看起来合理，不代表长期目标轨迹稳定。需要检查目标是否跳变、滞后、和天气不匹配、与候选动作或 safety guardrail 冲突。

5. 候选动作池应从 rollout 模式中归纳，不应按具体场景硬编码。  
   新候选动作应来自可泛化模式，例如 hot-dry retention、dawn dew relief、CO2 ventilation conflict，而不是来自某个 year/day/seed/step 的硬编码动作。

## 3. 建议的 v39 主线

建议下一步命名为：

```text
v39 Qwen-Max Long-Horizon Control Improvement Discovery
```

v39 不是 performance claim 实验，而是控制优化发现阶段。

### 3.1 执行范围

推荐先使用 v38 的 H240 stable pool：

```text
input = stable_long_horizon_evaluation_pool_20260529_v38_h240
controller = llm_rspc_v2
model = qwen-max-latest
mode = online long-horizon rollout discovery
cache = optional isolated passive record only
controlled controller = disabled
performance claim = disabled
promotion evidence = false
```

优先推进：

```text
H720 on selected stable scenarios
then H1440 on fewer stable scenarios
then full-cycle shadow benchmark design
```

不建议本轮做：

```text
qwen-3.7 migration
controlled replay expansion
strict threshold tuning
candidate hardcoding by scenario
selected cache overwrite
performance improvement claim
```

### 3.2 主要观测对象

v39 应该重点记录和审计以下内容。

#### A. Profile trajectory quality

需要回答：

```text
profile 是否随天气和作物状态合理变化？
温度、湿度/VPD、CO2、补光目标是否跳变？
profile 是否滞后于天气变化？
夜间防结露和白天控 VPD 是否衔接？
CO2 目标是否经常和高通风冲突？
```

建议指标：

```text
profile_target_jump_count
profile_target_reversal_count
profile_weather_mismatch_count
profile_guardrail_conflict_count
profile_co2_vent_conflict_count
profile_dew_vpd_conflict_count
```

#### B. Candidate action pool adequacy

需要回答：

```text
哪些风险窗口没有合适候选动作？
哪些候选动作经常出现但被 strict gate 拦下？
哪些候选动作经常被 guardrail 大幅改写？
哪些候选动作在短期改善一个指标但引发另一个指标恶化？
```

建议分类：

```text
missing_candidate_for_hot_dry
missing_candidate_for_dawn_dew
missing_candidate_for_high_humidity_low_temp
missing_candidate_for_co2_vent_conflict
missing_candidate_for_radiation_spike
candidate_filtered_by_safety
candidate_rewritten_by_guardrail
candidate_selected_but_effect_unclear
```

#### C. Control stability

需要回答：

```text
动作是否抖动？
通风、幕布、遮阳、加热是否频繁反向切换？
是否存在控制动作诱发 CVODES/runtime instability？
是否存在 LLM plan 与 fallback/guardrail 持续拉扯？
```

建议指标：

```text
action_oscillation_count
large_action_delta_count
guardrail_rewrite_count
fallback_dominance_steps
runtime_error_count
cvodes_failure_count
first_runtime_failure_step
action_before_runtime_failure
```

#### D. Control effect proxy

v39 可以观察控制效果，但不能立即做正式 performance claim。推荐先作为 discovery metrics：

```text
temperature_band_violation_area
RH_high_duration
RH_low_duration
VPD_high_duration
VPD_low_duration
dew_or_canopy_risk_duration
heating_energy_proxy
ventilation_intensity_proxy
lamp_energy_proxy
CO2_waste_proxy_under_high_vent
```

## 4. 推荐的数据流

建议 v39 的数据流如下：

```text
v38 stable pool
  -> qwen-max long-horizon online rollout
  -> trace with profile/candidate/guardrail/runtime metadata
  -> opportunity catalog
  -> profile improvement plan
  -> candidate pool improvement plan
  -> safety/stability admission
  -> later controlled canary if needed
```

其中 opportunity catalog 是关键中间产物，不应直接改控制器。它应把发现的问题归类：

```text
profile_issue
candidate_pool_gap
guardrail_conflict
fallback_overuse
runtime_instability
safety_boundary_pressure
energy_control_tradeoff
```

## 5. 推荐的 v39 artifacts

建议生成以下项目内 artifacts：

```text
qwen_max_long_horizon_rollout_manifest_20260529_v39.json/md
qwen_max_long_horizon_rollout_execution_record_20260529_v39.json/md
qwen_max_long_horizon_rollout_summary_audit_20260529_v39.json/md
profile_trajectory_quality_audit_20260529_v39.json/md
candidate_action_pool_gap_catalog_20260529_v39.json/md
control_stability_and_runtime_attribution_20260529_v39.json/md
metadata_replay_readiness_checklist_20260529_v39.json/md
```

v39 readiness 必须保持：

```text
controlled_replay_allowed = false
controlled_replay_execution_allowed = false
performance_claim_allowed = false
promotion_evidence = false
qwen_3_7_migration_allowed = false
```

## 6. Stop rules

应停止并诊断，而不是扩大实验的情况：

```text
runtime_error_count unexpectedly high
cvodes_failure repeats in stable pool
hard safety regression appears
trace missing profile/candidate/guardrail metadata
LLM response parse failure blocks rollout
action oscillation becomes dominant
guardrail rewrites most selected actions
```

这些情况不说明控制效果失败，但说明当前 rollout 不能作为优化依据，必须先修诊断链路或稳定性问题。

## 7. DeepResearch 需要重点分析的问题

请 DeepResearch 针对以下问题给出严谨建议：

1. 在温室控制论文或工程系统中，长周期 online rollout discovery 是否应优先于继续 strict-trigger/cached replay？
2. 当前将 cache 降级为复现工具，而非探索主线，是否科学？
3. qwen-max-latest 继续作为当前开发模型、qwen-3.7 延后迁移，是否更利于归因？
4. profile trajectory 应该用哪些指标判断好坏？
5. candidate action pool 应如何从 rollout 中归纳，而不是按场景硬编码？
6. CVODES/runtime instability 应如何在论文中定位：仿真有效性指标、控制稳定性指标，还是安全指标？
7. 下一阶段最小可发表路线应该优先做：
   - 事件触发 LLM planning
   - profile trajectory generator
   - candidate action pool expansion
   - action smoothing / rate guard
   - formal safety shield
   - semantic plan cache
8. 如果资源有限，v39 应该先跑 H720、H1440，还是直接选择少量 stable scenarios 做 full-cycle rollout？

## 8. 我当前的倾向

我的当前判断是：

```text
先不切 qwen-3.7；
先不继续以 cache/search strict trigger 为主线；
用 qwen-max-latest 在 v38 stable pool 上跑更长周期；
把 profile、candidate、guardrail、fallback、runtime 的问题系统整理出来；
先做 profile trajectory 和候选动作池的诊断型改进；
再考虑 controlled canary 或 qwen-3.7 migration。
```

最小下一步建议：

```text
v39: run qwen-max-latest H720 on v38 stable pool subset
```

但执行前应先确认：

```text
本轮是否允许在线 LLM；
是否允许 isolated passive cache record；
是否限制 H720 场景数量；
是否把 full-cycle rollout 留到 v40。
```

## 9. 边界声明

本文不是实验授权，不是控制逻辑，不是性能结论。

当前所有结论仍应遵守：

```text
hard safety > performance gain
shadow evidence != performance evidence
canary evidence != promotion evidence
historical cache != current model evidence
```

