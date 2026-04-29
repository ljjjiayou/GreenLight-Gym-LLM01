# RSPC Dry-Recovery Validation 2026-04-29

## 专家会议结论

本轮讨论把下一步方向从继续堆叠 HEM，转为先修复 RSPC 本体的湿热经济控制缺口。关键判断是：2015/day120/seed43 的大 RH violation 不是单纯高湿失控，而是包含大量低湿侧违规和高 VPD 风险。继续加强除湿会让问题更严重，必须先加入低湿恢复、冷夜除湿保护和统一候选评分。

专家组形成的共识：

- 温室控制专家：优先处理低湿和冷夜缓冲。高湿除湿动作在低温或已偏干时必须被二次保护，避免短期排湿导致后续过干和补热成本。
- RL 专家：当前 PPO 仍只能作为行为参考和诊断镜子，不能直接作为最终教师。后续应先固定 reward 版本，再训练强 PPO/SAC 基线。
- 实验设计专家：小规模 frozen replay 足够验证机制修复，但不足以做论文性能声明。通过 smoke 后再扩展到 36 场景 240-step。
- 方法创新专家：论文主线应保持为 LLM-guided robust setpoint planning and safe rollout control。HEM 目前更适合作为安全过滤和反事实经验学习的后续增强。

## 文献启发

本轮文献和检索给出的启发不是简单复制某个算法，而是把温室控制问题拆成约束安全、短期预测和经济优化三层：

- Greenhouse MPC/SMPC 工作强调温度、湿度和执行器约束要在预测窗口内显式处理，不能只看当前步动作。
- RL+MPC/robust RL 工作强调 RL 可以提供经济倾向或策略先验，但安全约束仍需要由 MPC/RSPC 层兜底。
- 针对温室 RH/VPD 的控制研究说明，湿度安全不是只压制高 RH。低 RH 和高 VPD 也会造成作物胁迫，因此候选评分要同时包含高湿、低湿和 VPD 风险。

参考入口：

- [Adaptive robust greenhouse climate control combining deep reinforcement learning and economic optimization](https://www.sciencedirect.com/science/article/pii/S2772375525005581)
- [Constrained temperature and relative humidity predictive control: Agricultural greenhouse case of study](https://www.sciencedirect.com/science/article/pii/S2214317323000525)
- [Trends in automated systems development for greenhouse horticulture](https://scialert.net/fulltext/?doi=ijar.2011.1.9)

## 本轮代码改动

修改文件：

- `gl_gym/agent/llm_agent.py`
- `tests/test_planning_extensions.py`

新增逻辑：

1. Dry-side recovery 参数
   - 增加 `dry_rh_on`、`dry_vpd_on`、`dry_vent_cap`、`dry_target_rh_floor` 等参数。
   - 目的：当 RH 偏低或 VPD 偏高时，不再允许 rollout 继续高通风、高加热、高 CO2 或补光。

2. Guardrail 末端保护
   - 在高湿、风速、夜间等已有规则之后，增加最终 dry-side 和 cold-buffer guard。
   - 目的：防止前面的除湿 pulse 在低温或偏干状态下重新抬高通风。

3. Setpoint contract dry 修正
   - 当 `RH < dry_rh_on` 或 `VPD > dry_vpd_on` 时，强制把 `target_rh` 提升到 dry recovery floor。
   - 同时限制过高 `target_temp` 和无效 CO2，减少加热加干、通风漏 CO2。

4. Fallback 候选评分
   - 在一阶响应估计中加入 `vpd_next`。
   - 评分中加入低 RH penalty 和高 VPD penalty。
   - 增加 dry recovery mitigation bonus，使低通风、关 CO2、关补光、必要保温的候选更容易胜出。

5. 新增 `dry_recovery` fallback 候选
   - 低湿或高 VPD 时生成低加热、低 CO2、低补光、低通风、必要遮阳或保温的候选动作。

6. Rollout 最终 dry override
   - 在 `_plan_control_step` 的末端再次检查 dry-side 风险。
   - 如果触发，则记录 `dry_recovery_override`，并限制通风、CO2、补光、加热和遮阳。

## 单元测试

已通过：

```bash
python -m py_compile gl_gym\agent\llm_agent.py gl_gym\experiments\run_frozen_benchmark.py
python -m pytest tests\test_planning_extensions.py tests\test_plan_cache.py tests\test_humidity_experience_memory.py tests\test_plan_intent_distillation.py tests\test_ppo_strategy_labeler.py -q
```

结果：

- `py_compile` 通过
- `42 passed`

新增测试覆盖：

- 低 RH / 高 VPD 下 guardrail 限制通风、CO2、补光和加热。
- 冷夜非极端高湿下限制过大通风，同时保留生存加热。
- dry setpoint contract 会提高 `target_rh`。
- dry fallback 会选择 `dry_recovery` 候选。

## Frozen Replay 结果

使用已有 cache，不调用新 LLM：

```bash
python gl_gym\experiments\run_frozen_benchmark.py --years 2015 --days 120 --seeds 43 --controllers llm,ppo --max-steps 240 --plan-cache-mode replay --plan-cache-strict --plan-cache-key-policy scenario_timestep --plan-cache-path gl_gym/result/plan_cache/smoke_2x240_qwen.json --output-json gl_gym/result/benchmarks/phase1_2015_d120_s43_dryfix.json

python gl_gym\experiments\run_frozen_benchmark.py --years 2020 --days 240 --seeds 42 --controllers llm,ppo --max-steps 240 --plan-cache-mode replay --plan-cache-strict --plan-cache-key-policy scenario_timestep --plan-cache-path gl_gym/result/plan_cache/smoke_2x240_qwen.json --output-json gl_gym/result/benchmarks/phase1_2020_d240_s42_dryfix.json
```

| Scenario | Controller | Reward | Profit | Heat | CO2 | Elec | Temp V | RH V | Heat+Vent | Cache Hits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2015/d120/s43 | LLM-RSPC before | 133.361 | -0.09108 | 0.10352 | 0.00357 | 0.00435 | 10.627 | 229.106 | 8 | 240 |
| 2015/d120/s43 | LLM-RSPC dryfix | 137.290 | -0.08542 | 0.10061 | 0.00469 | 0.00000 | 7.354 | 177.638 | 3 | 240 |
| 2015/d120/s43 | PPO | 152.076 | -0.01474 | 0.03370 | 0.00000 | 0.00000 | 8.329 | 7.242 | 3 | 0 |
| 2020/d240/s42 | LLM-RSPC before | 151.630 | -0.04358 | 0.05955 | 0.00000 | 0.00000 | 0.000 | 0.894 | 64 | 240 |
| 2020/d240/s42 | LLM-RSPC dryfix | 151.817 | -0.04021 | 0.05621 | 0.00000 | 0.00000 | 0.000 | 0.580 | 21 | 240 |
| 2020/d240/s42 | PPO | 151.919 | -0.03492 | 0.05408 | 0.00000 | 0.00000 | 0.000 | 2.968 | 49 | 0 |

HEM 补跑：

| Scenario | Controller | Reward | Profit | RH V | Heat+Vent | HEM Available | HEM Selected | HEM Rejected |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2015/d120/s43 | LLM-RSPC+HEM dryfix | 137.290 | -0.08542 | 177.638 | 3 | 23 | 0 | 23 |
| 2020/d240/s42 | LLM-RSPC+HEM dryfix | 151.817 | -0.04021 | 0.580 | 21 | 52 | 0 | 52 |

## 判断

本轮 dryfix 是有效机制修复：

- 两个 smoke 场景中 LLM-RSPC 的 reward 和 profit 均提升。
- 2015 冷春场景 RH violation 从 `229.106` 降到 `177.638`，temp violation 从 `10.627` 降到 `7.354`。
- 2020 场景 RH violation 从 `0.894` 降到 `0.580`，heat+vent conflict 从 `64` 降到 `21`。
- HEM 仍没有被选中，说明本轮收益来自 RSPC 本体，不应把 HEM 写成当前收益来源。

仍未解决的问题：

- 2015/d120/s43 与 PPO 仍存在明显差距，尤其 RH violation 仍偏高。
- 当前 PPO 不是最优教师，但在这个冷春场景仍提示：RSPC 的湿热协同还需要继续重构。

## 下一步建议

1. 把 dryfix 纳入当前主线并提交。
2. 扩展到 36 场景 240-step frozen benchmark，验证这个机制是否多场景稳健。
3. 同时开启 reward audit，确定 `reward_v0` 或 `reward_v1`，再训练强 PPO/SAC 基线。
4. 暂停 HEM 收益叙事，只保留 HEM 作为安全过滤与后续反事实经验学习模块。
