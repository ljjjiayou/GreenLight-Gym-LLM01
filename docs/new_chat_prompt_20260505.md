# 新对话启动 Prompt 2026-05-05

请你接手我的温室控制项目，当前工作目录是：

`d:/My_Project/Greenhouse_Project/GreenLight-Gym2-LLM05`

请先阅读：

- `docs/project_context_handoff_20260505.md`
- `docs/tomato_safety_v2_diagnostic_20260430.md`
- `docs/sentinel_12x240_gate_20260430.md`
- 必要时再看 `docs/reproducible_evaluation_plan_20260429.md`、`docs/mc_sero_shadow_diagnostic_20260430.md`、`docs/humidity_experience_memory_mvp.md`

## 当前项目主线

项目主线是：

**LLM-RSPC: LLM-guided Robust Setpoint Planning and Safe Rollout Control**

即：LLM 负责高层设定点规划和未来目标状态，RSPC/fallback/guardrail 负责每一步安全滚动执行。

不要把 LLM 当作直接控制器，也不要把当前 PPO 当作最终教师。

## 当前模块边界

- `LLM-RSPC`：主控制器。
- `PPO`：诊断镜子，不是最终强 baseline，也不是最终教师。
- `HEM`：经验库框架已实现，但当前不作为收益主线。
- `MC-SERO`：仍是 shadow 诊断层，不接管动作。
- `Tomato Safety v2`：默认关闭，通过 `llm_rspc_v2` opt-in，用于番茄低 RH、高 VPD、低温恢复保护。
- `Frozen Benchmark / Plan Cache`：用于冻结 LLM 高层计划，公平比较底层控制逻辑。

## 当前关键结果

Pilot 中 `llm_rspc_v2` 有明显积极信号：

- `2015/day120/seed43`
  - RH-low：`170.71 -> 4.59`
  - VPD-high：`21.35 -> 8.48`
  - temp violation：`7.19 -> 5.47`
- `2020/day240/seed42`
  - VPD-high：`0.67 -> 0.42`
  - 安全场景没有明显被破坏。

但 sentinel gate 未通过：

- 失败场景：`2010/day180/seed44`
- controller：`llm_rspc_v2`
- replay 约 55 步提前终止
- `temp_air` 升到约 `35.6 C`
- 机理判断：高温强辐射下，v2 过度偏向干侧保湿，出现 `u_screen=1.0`、`u_shading=0.0` 的积热组合。

因此当前不能进入 36x240，也不能宣称 v2 已经泛化稳定。

## 当前首要任务

请修复 Tomato Safety v2 的 hot override 泛化问题。

要求：

1. 先检查 `git status --short`。
2. 不要覆盖或回退已有未提交修改，尤其是当前可能存在的 `gl_gym/agent/expert_distillation.py` 修改。
3. 忽略 `reports/` PPT 文件，除非我明确要求处理。
4. 检查 `gl_gym/agent/llm_agent.py` 中的 `apply_tomato_safety_v2`、`hot_dry_cooling_guard`、`tomato_safety_v2` 相关逻辑。
5. 实现高温优先级：
   - 当 `temp_air >= 32 C`，或高辐射且温度快速上升时，温度安全优先于干侧保湿。
   - hot override 下释放或降低 screen。
   - hot override 下启用 shade。
   - hot override 下允许足够通风降温。
   - heat、CO2、lamp 应保持关闭或受限。
   - dry/VPD guard 只在温度不高或仍处于安全温度区间时主导。
6. 补充测试：
   - 高温强辐射 dry state 不再输出 `screen=1.0, shade=0.0` 积热组合。
   - 非高温 dry/VPD 状态下原有低湿保护仍有效。
   - 极端高湿/结露风险不能被 hot override 错误放过。
7. 跑基础检查：
   - `python -m py_compile gl_gym\agent\llm_agent.py gl_gym\experiments\run_frozen_benchmark.py gl_gym\experiments\frozen_benchmark_protocol.py gl_gym\experiments\failure_window_extractor.py`
   - 相关 pytest：planning extensions、frozen benchmark、protocol、failure extractor、MC-SERO shadow、plan cache。
8. 先跑 `2010/day180/seed44` canary，不要直接跑完整 12x240。
9. Canary 通过后，再恢复 12x240 sentinel gate。

## 当前不要做

- 不要继续调 HEM。
- 不要让 MC-SERO 接管动作。
- 不要直接跑 36x240、960 步或全周期。
- 不要从当前弱 PPO 中继续蒸馏新经验。
- 不要把 pilot 结果写成最终性能结论。

请以“温室控制领域专家 + 严谨工程实现者”的标准继续推进：每一步改动都要说明目的、验证结果和是否进入下一阶段。
