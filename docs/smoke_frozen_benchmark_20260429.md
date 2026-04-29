# Smoke Frozen Benchmark 2026-04-29

## 设置

- Scenarios: `2020/day240/seed42`, `2015/day120/seed43`
- Steps: `240`
- Controllers: `llm`, `llm_hem`, `ppo`
- Cache: `gl_gym/result/plan_cache/smoke_2x240_qwen.json`
- Key policy: `scenario_timestep`

本轮先用 `record` 生成 baseline LLM plans，再用 `replay --strict` 复现。`scenario_timestep` key policy 用场景与重规划 timestep 作为冻结计划身份，避免 HEM 改变闭环状态后生成自己的 LLM plan。

## 关键结果

| Scenario | Controller | Reward | Profit | Heat | CO2 | Elec | Temp V | RH V | Heat+Vent | Cache Hits | HEM Avail | HEM Selected |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020/d240/s42 | LLM-RSPC | 151.630 | -0.04358 | 0.05955 | 0.00000 | 0.00000 | 0.000 | 0.894 | 64 | 240 replay | 0 | 0 |
| 2020/d240/s42 | LLM-RSPC+HEM | 151.630 | -0.04358 | 0.05955 | 0.00000 | 0.00000 | 0.000 | 0.894 | 64 | 240 | 53 | 0 |
| 2020/d240/s42 | PPO | 151.919 | -0.03492 | 0.05408 | 0.00000 | 0.00000 | 0.000 | 2.968 | 49 | 0 | 0 | 0 |
| 2015/d120/s43 | LLM-RSPC | 133.361 | -0.09108 | 0.10352 | 0.00357 | 0.00435 | 10.627 | 229.106 | 8 | 240 replay | 0 | 0 |
| 2015/d120/s43 | LLM-RSPC+HEM | 133.428 | -0.08952 | 0.10410 | 0.00358 | 0.00217 | 10.628 | 229.259 | 8 | 240 | 19 | 1 |
| 2015/d120/s43 | PPO | 152.076 | -0.01474 | 0.03370 | 0.00000 | 0.00000 | 8.329 | 7.242 | 3 | 0 | 0 | 0 |

Record 和 replay 在两个场景中完全一致，说明 frozen benchmark 现在可复现。`replay --strict` 未出现 cache miss。

## 结论

- 评估系统通过 smoke 验证：cache record/replay 可用，且 HEM 与 baseline 确认共享同一高层 plan。
- HEM 不应作为当前主线收益点：2020 场景可用但未选中；2015 场景只选中 1 步，收益很小且 RH violation 没有改善。
- 2015 春季冷场景暴露了主问题：LLM-RSPC 的温湿协同明显弱于 PPO，尤其 RH violation 差距很大。
- PPO 仍不是最终教师，但在该 cold-start 场景中说明当前 RSPC 本体需要先修，而不是继续扩大 HEM。

## 下一步建议

1. 暂停 HEM 收益叙事，只保留为安全过滤与候选机制证据。
2. 优先改 RSPC 除湿/热缓冲本体，特别是低温高湿场景下避免通风造成冷冲击后长时间 RH debt。
3. 同时决定 `reward_v0` 或 `reward_v1`，再启动 PPO/SAC 强基线训练。
4. 在 RSPC 本体修完后，再跑 36 场景 frozen benchmark。

