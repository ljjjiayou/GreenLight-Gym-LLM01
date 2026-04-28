# HEM-RSPC 240 步验证记录 2026-04-29

## 目的

本次验证用于检查 HEM-RSPC 第一阶段经验记忆是否能在更长窗口中发现稳定信号。该阶段仍是离线流程，不改变主控制闭环。

流程：

```text
240 步 PPO vs LLM-RSPC 配对诊断
-> 湿度经济经验挖掘
-> 同版本经验 shadow 验证
```

## 配对诊断

轨迹：

```text
gl_gym/result/diagnostics/ppo_vs_llm_hem_s240_day240_20260429.json
```

场景：

```text
year=2020
day=240
seed=42
steps=240
LLM model=qwen-max-latest
LLM interval=12
```

聚合结果：

| 方法 | Reward | Profit | Heat cost | CO2 cost | Elec cost | Temp violation | RH violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM-RSPC | 151.542 | -0.04540 | 0.05758 | 0.00228 | 0.00174 | 0.000 | 0.866 |
| PPO | 151.919 | -0.03492 | 0.05408 | 0.00000 | 0.00000 | 0.000 | 2.968 |

诊断结论：

```text
PPO 的 profit 更好，成本更低；
LLM-RSPC 的 RH 安全性更好；
PPO 不是可直接替代控制器，但可作为经济行为候选来源。
```

安全模式：

| 方法 | heat+vent conflict | CO2 leak | lamp risk | RH>=90 steps |
|---|---:|---:|---:|---:|
| LLM-RSPC | 62 | 0 | 0 | 3 |
| PPO | 49 | 0 | 0 | 6 |

策略标签：

```text
LLM-RSPC confident labels: 133/240
PPO confident labels: 76/240
PPO useful high-confidence samples: 65
PPO useful free_air_exchange samples: 63
```

## 经验挖掘

输出：

```text
gl_gym/result/experience_memory/humidity_memory_s240_day240_20260429.jsonl
```

版本信息：

```text
teacher_policy_id=ppo_best_model_20260428
baseline_controller_id=llm_rspc_887a5ef
data_split=validation_s240
memory_schema_version=hem_rspc_v1
```

挖掘结果：

```text
paired steps: 240
accepted raw cases: 20
stored memory cases after merge: 2
accepted risk-intent override cases: 9
rejected cases: 220
```

主要拒绝原因：

```text
ppo_strategy_is_not_free_air_exchange: 171
ppo_strategy_confidence_too_low: 162
intent_is_not_economic_dehumidify: 158
target_rh_gap_outside_band: 145
vpd_outside_safe_economic_band: 104
not_moderate_rh: 88
outside_air_has_no_dehumidification_potential: 71
no_economic_advantage: 58
```

这说明筛选器仍然比较保守，大部分 PPO 动作没有被当作经验吸收。

## 记忆内容

合并后得到 2 条湿度经济经验。

经验 1：

```text
case_id=bb2b3fe60455641b
support_count=7
trust=0.718
target_rh=72
state: temp=16.76, RH=84.29, VPD=0.30
PPO tendency: low heating, low screen, ventilation near 1.0
```

经验 2：

```text
case_id=b8c5f14df90a5c8c
support_count=13
trust=0.765
target_rh=75
state: temp=18.59, RH=81.52, VPD=0.40
PPO tendency: near-zero heating, low screen, ventilation near 1.0
```

两条经验共同指向：

```text
在温度和结露风险可接受、外界空气有除湿潜力时，
减少 heating + screen 保守组合，
提高 free_air_exchange 候选通风，
用更低成本处理中等湿度风险。
```

## Shadow 验证

输出：

```text
gl_gym/result/experience_memory/humidity_memory_s240_day240_20260429_shadow.json
```

结果：

```text
evaluated LLM steps: 240
memory hit steps: 58
risk-intent override context steps: 74
hit rate: 0.242
mean proxy cost delta memory-LLM: -0.027904
mean heat+vent delta memory-LLM: -0.014582
```

解释：

```text
memory candidate 在约 24.2% 步骤中命中；
命中时代理成本下降；
heat+vent 冲突下降；
但这仍是 shadow 指标，不等价于闭环收益。
```

## 当前判断

240 步结果支持继续推进 HEM-RSPC，但还不能直接声称最终效果提升。合理下一步是：

```text
1. 将 memory candidate 接入 rollout 候选池，但默认关闭。
2. 加入安全硬门控和候选评分。
3. 跑 240 步闭环 A/B：
   - LLM-RSPC
   - LLM-RSPC + HEM shadow
   - LLM-RSPC + HEM candidate
4. 如果 240 步安全不变差，再跑 960 步。
```

论文表述应保持克制：

```text
HEM-RSPC 不是直接模仿 PPO，而是从 PPO 和 LLM-RSPC 的局部差异中挖掘可验证的经济湿度控制经验，
并将其作为安全滚动预测控制中的候选动作来源。
```
