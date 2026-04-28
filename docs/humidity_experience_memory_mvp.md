# HEM-RSPC 第一阶段 MVP 设计

## 目标

当前阶段不直接把 PPO 作为最终控制器，也不把 LLM API 当作可在线更新参数的学习器。我们新增一个外部经验层：

```text
从 PPO 与 LLM-RSPC 的配对轨迹中，筛选“中等湿度风险下更经济且不更危险”的 free_air_exchange 行为，
存成带安全证据和目标意图的经验候选，
先离线 shadow 验证，再决定是否接入闭环候选池。
```

该阶段聚焦一个窄问题：减少不必要的 heating + ventilation 除湿成本，同时不增加 RH violation 和 temperature violation。

## 新增模块

- `gl_gym/agent/humidity_experience_memory.py`
  - 定义湿度经济经验的数据结构。
  - 计算 VPD、露点、绝对湿度和外界空气除湿潜力。
  - 判断一对 PPO/LLM-RSPC 配对轨迹是否可以收录。
  - 存储 residual 动作：`ppo_action - target_tracking_baseline_action`。
  - 支持相似经验检索，并生成 memory candidate。

- `gl_gym/experiments/mine_humidity_experiences.py`
  - 读取 `diagnose_ppo_vs_llm.py` 产生的 JSON/CSV 配对轨迹。
  - 对每个相同步数的 PPO 与 LLM-RSPC 行为做筛选。
  - 输出经验库、拒绝样本和挖掘报告。

- `gl_gym/experiments/shadow_humidity_memory.py`
  - 在不改变控制动作的情况下，离线记录经验库会在哪些 LLM-RSPC 步骤命中。
  - 输出 memory candidate 与原 LLM 动作的代理成本差、heat+vent 冲突差和命中率。

## 收录逻辑

一条经验必须同时满足：

```text
plan intent = economic_dehumidify
PPO strategy = free_air_exchange
RH_air 位于中等湿度风险区间
RH_air - target_RH 位于合理除湿缺口区间
VPD 不过低也不过高
空气和冠层结露余量安全
外界空气具备除湿潜力
当前温度有安全余量
PPO 动作是低加热、低 CO2、低补光、高通风
PPO 相比 LLM-RSPC 有经济优势
PPO 不增加 RH/温度违规
未来短窗口内不把湿度债务转移给后续步骤
```

这意味着 PPO 只是“候选证据来源”。经验能否进入记忆库，取决于配对对照、安全门控和未来窗口验证。

## 风险型经济意图修正

已有诊断轨迹显示，当前 LLM-RSPC 有时会在 RH 仍处于中等风险、VPD 和结露余量仍安全时，把目标 RH 压到很低，例如 72%。这会把本来可以经济换气处理的状态判成 `safe_dehumidify`，从而诱发更高成本的 heating + ventilation 组合。

因此 MVP 增加了一个很窄的修正：

```text
如果实际气候风险仍是中等，
外界空气有除湿潜力，
温度和结露余量安全，
PPO 给出高置信 free_air_exchange，
且 PPO 不增加 RH/温度违规，
则允许把该步记录为 risk_based economic_dehumidify 候选。
```

这个修正不能绕过安全条件；它只能放松“目标 RH 过激导致的意图误判”。在经验证据中会记录 `plan_intent_overridden=1`，方便后续论文消融时单独验证。

## 经验库版本管理

经验库不是一次性训练产物。由于 PPO 权重、LLM-RSPC 规则、setpoint 合同和安全门控后续都可能更新，每条经验都必须记录来源版本：

```text
memory_schema_version
teacher_policy_id
baseline_controller_id
data_split
source_trace_id
mining_config_hash
```

后续如果 PPO 或 LLM-RSPC 改进，不应直接混用旧经验，而应按下面流程处理：

```text
1. 保留旧经验库，作为可复现实验快照。
2. 用新 PPO / 新 RSPC 重新生成配对轨迹。
3. 对旧经验做 revalidation：
   - 如果相对新 RSPC 不再省成本，降权或淘汰。
   - 如果安全风险变差，标记 deprecated。
   - 如果仍然有效，可以迁移到新版本经验库。
4. 从新轨迹中继续挖掘新经验。
5. 实验时只加载同一 teacher_policy_id / baseline_controller_id / data_split 的经验。
```

因此，HEM-RSPC 的经验学习不是“死记 PPO 当前动作”，而是一个可持续的离线经验生命周期：

```text
mine -> pending -> shadow validate -> validated -> revalidate -> migrate/deprecate
```

论文实验中应冻结某一版 PPO 和 RSPC，保证对照公平；工程优化中则可以随着控制器升级持续重挖和复验。

## 典型命令

先生成配对诊断轨迹：

```powershell
python gl_gym\experiments\diagnose_ppo_vs_llm.py --year 2020 --day 240 --seed 42 --max-steps 240 --output-json gl_gym\result\diagnostics\ppo_vs_llm_s240_day240.json --output-csv gl_gym\result\diagnostics\ppo_vs_llm_s240_day240.csv --output-report gl_gym\result\diagnostics\ppo_vs_llm_s240_day240.md
```

再挖掘经验：

```powershell
python gl_gym\experiments\mine_humidity_experiences.py --trace gl_gym\result\diagnostics\ppo_vs_llm_s240_day240.json --output-memory gl_gym\result\experience_memory\humidity_memory_s240.jsonl
```

最后做 shadow 验证：

```powershell
python gl_gym\experiments\shadow_humidity_memory.py --trace gl_gym\result\diagnostics\ppo_vs_llm_s240_day240.json --memory-jsonl gl_gym\result\experience_memory\humidity_memory_s240.jsonl
```

## 下一阶段

第一阶段只做离线经验挖掘和 shadow 验证。如果 240 步和 960 步验证显示：

```text
memory candidate 命中率不是零
代理加热成本下降
heat+vent 冲突下降
RH/temp 风险不增加
```

再把 memory candidate 接入 `llm_agent.py` 的 rollout 候选池，并保持默认关闭开关，做正式消融实验。
