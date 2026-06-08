# Greenhouse Project 上下文交接包 2026-05-05

## 使用目的

这份文档用于在新对话或新 agent 中同步项目关键上下文，避免继续携带当前长对话中的噪声。这里只保留与项目改进、实验设计和论文主线有关的信息；PPT 制作过程、临时排版、非技术对话不进入交接范围。

## 当前仓库状态

- 工作目录：`d:/My_Project/Greenhouse_Project/GreenLight-Gym2-LLM05`
- 当前分支：`codex/literature-method-backup`
- 最近技术提交：
  - `496993e Add sentinel gate diagnostics`
  - `4fc9229 Add tomato safety v2 replay diagnostics`
  - `73a152a Record frozen benchmark pilot results`
- 当前存在未提交内容：
  - `gl_gym/agent/expert_distillation.py` 有未提交修改，来源未确认；后续操作不要随意覆盖或回退。
  - `reports/` 下有 PPT 汇报文件，与控制器优化无关，后续代码优化时可忽略。
- GitHub 推送曾因网络失败，已生成本地 bundle 备份：
  - `d:/My_Project/Greenhouse_Project/git_bundles/greenlight_sentinel_gate_496993e.bundle`

## 项目主线

当前论文/项目主线应保持为：

**LLM-RSPC: LLM-guided Robust Setpoint Planning and Safe Rollout Control**

中文表述：

**LLM 引导的鲁棒设定点规划与安全滚动控制**

核心观点不是“让 LLM 直接控制温室”，而是：

- LLM 负责高层策略理解、目标状态规划和专家知识注入。
- RSPC、fallback、guardrail 负责把高层计划转化为每一步可执行、安全、可诊断的控制动作。
- 可复现评估体系负责隔离 LLM 输出随机性，公平比较底层控制逻辑。

## 当前整体控制流程

1. 状态抓取与风险分析
   - 读取温室状态：`temp_air`、`rh_air`、`vpd_kpa`、CO2、辐射、外界天气、作物状态、执行器状态。
   - 判断是否需要重规划：计划到期、状态风险升高、目标偏离、紧急风险等。

2. LLM 高层规划
   - 生成 anchor action。
   - 生成未来约 11 个时间步的目标温室状态或 reference profile。
   - 给出策略意图、重规划原因和 fallback 信息。
   - LLM 输出不能直接执行，必须经过后续合同和护栏。

3. Setpoint Contract
   - 对 LLM 输出的目标状态做补齐、裁剪、修正。
   - 缺失未来目标时使用合同机制填充。
   - 避免过激 RH、温度、CO2 目标直接进入底层控制。

4. RSPC Rollout
   - 读取当前状态和下一目标状态。
   - 将 anchor/rule/blend/fallback 候选进行加权和误差修正。
   - 生成本步候选控制动作。

5. Fallback 与 Guardrail
   - 如果动作生成失败，使用备用控制策略。
   - 最终动作经过安全护栏过滤，约束温度、RH、VPD、露点风险和执行器冲突。

6. 执行与记录
   - 执行动作。
   - 记录 reward、profit、cost、violation、source、plan cache、MC-SERO、Tomato Safety v2 等诊断字段。

## 模块边界

### LLM-RSPC

- 当前主控制器。
- LLM 负责高层 reference plan。
- RSPC 负责低层滚动执行和安全候选选择。
- 论文叙事的主方法，不应被 HEM、MC-SERO 或 PPO 抢主线。

### PPO

- 当前只作为行为参考和诊断镜子。
- 不是最终教师真值，也不是论文里可直接用来证明强 baseline 的最终 PPO。
- 当前 PPO 参数和权重仍可能不是最优。
- 后续正式对比前需要固定 reward 版本，并重新训练/调参 PPO 或 SAC 强基线。

### HEM: Humidity Experience Memory

- 已实现湿度经济经验库框架。
- 目标是从 PPO 与 LLM-RSPC paired rollouts 中挖掘“局部更经济且未来不更危险”的经验。
- 当前 HEM 有版本门控：`teacher_policy_id`、`baseline_controller_id`、`memory_schema_version`。
- 当前收益叙事暂停：
  - 在 dryfix/v2 相关验证中，HEM available 但 selected 不稳定或为 0。
  - 不应宣称 HEM 带来了闭环性能提升。
- 后续定位：Counterfactual Experience Learning，需要等强 PPO/SAC 和更稳定 RSPC 后重新挖掘经验。

### MC-SERO Shadow

- 当前只处于 `shadow` 诊断模式。
- 不接管动作，不改变 `selected_control`、reward、profit 或 baseline 行为。
- 作用是后台生成机制候选并评分，记录 would-select、best candidate、margin、score terms。
- 候选包括：`economy_hold`、`economic_dehumidify`、`safe_dehumidify`、`emergency_dehumidify`、`dry_recovery`、`heat_buffer` 以及已有 anchor/rule/blend。
- 当前价值：帮助定位当前 RSPC 哪些状态可能有更合理的机制候选。
- 后续只有在 shadow 跨场景稳定、候选符合机理后，才考虑 guarded select。

### Tomato Safety v2

- 默认关闭，通过 `llm_rspc_v2` opt-in controller 启用。
- 针对番茄机理问题补充保护：
  - RH 过低。
  - VPD 过高。
  - 低温恢复失败。
  - 目标 RH 与高通风动作不一致。
- 目前 pilot 有明显收益，但 sentinel gate 未通过，因此 v2 还不是最终稳定版本。

## 可复现评估体系

### LLM Plan Cache

支持三种模式：

- `record`：调用 LLM 并记录 raw response、parsed plan、contract 后计划、fallback 信息。
- `replay`：不再调用新 LLM，复用缓存计划。
- `refresh`：重新调用 LLM 并覆盖旧计划。

关键原则：

- 同一批 cached LLM plans 下比较 `llm`、`llm_rspc_v2`、`llm_sero_shadow`、HEM 等底层控制差异。
- `plan-cache-strict` 下 cache miss 不能被静默吞掉。
- 不能让 LLM 输出波动污染控制器对比。

### Frozen Benchmark

用于冻结高层计划，比较不同控制器的低层执行差异。

重要检查：

- cache hit steps 必须等于 episode steps。
- record/replay 指标应一致。
- `llm_sero_shadow` 与 `llm` 行为必须完全一致。
- unknown source、cache miss、shadow invariance 失败都应停止。

### Sentinel Gate

sentinel gate 是正式扩大实验前的泛化检查。

当前计划使用：

- `years=2010,2015,2020`
- `days=59,120,180,240`
- `seed=44`
- `max_steps=240`
- controllers：`llm,llm_rspc_v2,llm_sero_shadow,ppo`

设计目的：

- 避免只在 pilot 场景上调参。
- 使用未参与调参与诊断的 seed 检查跨年份、跨季节稳定性。
- 任一 shard 失败立即停止，不进入后续 shard。

## 已完成关键结果

### Pilot 1: `2015/day120/seed43`, 240 steps

使用 strict `scenario_timestep` replay。

| controller | reward | profit | RH low | VPD high | temp viol | cache hits | v2 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm` | 137.7791 | -0.0851 | 170.7133 | 21.3516 | 7.1855 | 240 | 0 |
| `llm_rspc_v2` | 148.9268 | -0.0859 | 4.5938 | 8.4842 | 5.4667 | 240 | 61 |
| `llm_sero_shadow` | 137.7791 | -0.0851 | 170.7133 | 21.3516 | 7.1855 | 240 | 0 |
| `ppo` | 152.0762 | -0.0147 | 0.0000 | 6.5975 | 8.3291 | n/a | n/a |

结论：

- Tomato Safety v2 在该 pilot 中有效降低 RH low 与 VPD high。
- RH low 下降约 97.3%。
- VPD high 下降约 60.3%。
- temp violation 没有变差，反而下降。
- 但这只是 pilot 结果，不可直接作为最终泛化结论。

### Pilot 2: `2020/day240/seed42`, 240 steps

| controller | reward | profit | RH low | VPD high | RH total | temp viol | cache hits | v2 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm` | 151.6291 | -0.0441 | 0.0000 | 0.6746 | 0.5029 | 0.0000 | 240 | 0 |
| `llm_rspc_v2` | 151.6899 | -0.0428 | 0.0000 | 0.4196 | 0.6025 | 0.0000 | 240 | 44 |
| `llm_sero_shadow` | 151.6291 | -0.0441 | 0.0000 | 0.6746 | 0.5029 | 0.0000 | 240 | 0 |
| `ppo` | 151.9194 | -0.0349 | 0.0000 | 0.0000 | 2.9681 | 0.0000 | n/a | n/a |

结论：

- v2 在安全场景中没有明显破坏 baseline。
- VPD high 降低。
- RH total 略升但仍很小。

### Sentinel Gate: `2010/day180/seed44`

执行 2010 year shard 后停止。

失败场景：

- scenario：`y2010_d180_s44_n240`
- controller：`llm_rspc_v2`
- replay rows：55
- cache enabled steps：54
- cache hit steps：54
- unknown source steps：1

机理判断：

- 该场景是高辐射、高温、偏干场景。
- v2 的 dry/VPD guard 在该场景中过度偏向“保湿”。
- 多步出现 `u_screen = 1.0`、`u_shading = 0.0`，配合高辐射可能造成积热。
- `temp_air` 上升到约 `35.6 C`，提前终止。
- 这不是单纯 cache 噪声，而是真实安全/泛化问题。

当前决策：

- Sentinel gate 未通过。
- 不继续运行 2015/2020 shard。
- 不进入正式 `36x240`。
- 先修 hot override。

## 当前首要问题

Tomato Safety v2 的 dry/VPD 保护逻辑从冷春/安全 pilot 迁移到夏季强辐射场景时，会出现温度优先级不足。

需要修复：

- 当 `temp_air >= 32 C` 或高辐射且温度快速上升时，温度安全必须优先于干侧保湿。
- hot override 下应释放或降低 screen。
- hot override 下应启用 shade。
- hot override 下允许足够通风降温。
- dry/VPD guard 只在温度不高，或温度仍处于番茄安全区间时主导动作。
- 新增单测避免高温强辐射 dry state 再输出 `screen=1.0, shade=0.0` 的积热组合。

## 下一步推荐执行顺序

1. 版本卫生检查
   - `git status --short`
   - 注意不要覆盖 `gl_gym/agent/expert_distillation.py` 的既有未提交修改。
   - 忽略 `reports/` PPT 文件，除非用户明确要求处理。

2. 修复 Tomato Safety v2 hot override
   - 重点文件预计是 `gl_gym/agent/llm_agent.py`。
   - 搜索 `apply_tomato_safety_v2`、`hot_dry_cooling_guard`、`tomato_safety_v2`。
   - 明确高温优先级：高温强辐射时不允许积热式保湿组合。

3. 新增或扩展单测
   - 高温强辐射 dry state 下：
     - 不应强制 `u_screen=1.0`。
     - 应启用或提高 `u_shading`。
     - 应允许足够 ventilation。
     - heat/CO2/lamp 应保持关闭或受限。
   - dry/VPD 非高温状态下，原低湿保护仍应有效。
   - 极端高湿/结露风险不能被 hot override 错误放过。

4. 基础检查
   - `python -m py_compile gl_gym\agent\llm_agent.py gl_gym\experiments\run_frozen_benchmark.py gl_gym\experiments\frozen_benchmark_protocol.py gl_gym\experiments\failure_window_extractor.py`
   - 运行相关 tests：
     - `tests/test_planning_extensions.py`
     - `tests/test_frozen_benchmark.py`
     - `tests/test_frozen_benchmark_protocol.py`
     - `tests/test_failure_window_extractor.py`
     - `tests/test_mc_sero_shadow.py`
     - `tests/test_plan_cache.py`

5. Canary 验证
   - 先跑 `2010/day180/seed44`，不要直接跑 12x240。
   - controllers 建议：`llm,llm_rspc_v2,llm_sero_shadow,ppo`
   - strict replay 使用 sentinel cache 或重新 record/replay，根据当前 cache 是否完整决定。
   - 验证：
     - `llm_rspc_v2` 不再提前终止。
     - cache hit steps 满步。
     - temp violation 不明显高于 baseline。
     - 不以牺牲 RH/VPD 安全为代价。

6. 恢复 12x240 sentinel gate
   - 只有 canary 通过后再跑。
   - 按 year shard 执行，任一 shard 失败立即停止。

7. 通过后再做正式 benchmark
   - `36x240`：`years=2010,2015,2020` x `days=59,120,180,240` x `seeds=42,43,44`
   - 再考虑 `12x960` 和全周期。

8. 强 RL baseline
   - 在 RSPC/v2 稳定后推进。
   - 固定 reward 版本。
   - 系统调 PPO/SAC。
   - 选模用 Pareto：reward、profit、RH violation、temp violation、heat cost、heat+vent conflict。

## 当前不建议做的事

- 不要继续堆 HEM 收益叙事。
- 不要从当前弱 PPO 继续直接蒸馏动作。
- 不要让 MC-SERO 从 shadow 直接接管闭环。
- 不要直接跑 36x240 或 960/full cycle。
- 不要把 pilot 上 v2 的收益写成最终泛化结论。
- 不要忽略 sentinel gate 发现的高温强辐射问题。

## 新对话建议目标

新对话第一阶段目标应设为：

**修复 Tomato Safety v2 在 `2010/day180/seed44` 高温强辐射场景下的 hot override 问题，并用 canary + sentinel gate 验证。**

具体第一条任务可以是：

```text
请基于项目上下文交接包继续工作。当前首要任务是修复 Tomato Safety v2 在 2010/day180/seed44 高温强辐射场景下的泛化失败。请先检查当前代码和 git 状态，不要覆盖未提交修改；然后改进 hot override，补充测试，跑 canary 验证，只有通过后再恢复 12x240 sentinel gate。
```

