# 可复现评估与强基线实施记录 2026-04-29

## 本轮目标

本轮不继续扩大 HEM，而是先建立可复现评估底座。核心原则是：LLM 只负责产生高层设定点规划，后续 RSPC、HEM、候选评分或消融实验必须能在同一批 LLM plans 下比较，减少大模型输出波动对实验结论的污染。

## 已实现模块

### 1. LLM Plan Cache

新增 `gl_gym/agent/plan_cache.py`，并接入 `RuleBasedLLMDirector._replan_with_llm`。

支持三种模式：

- `record`：缓存不存在时调用 LLM 并写入；缓存已存在时直接复用，避免重复覆盖。
- `replay`：优先从缓存恢复 LLM 工具动作和 setpoints；`--llm-plan-cache-strict` 开启时 cache miss 直接报错。
- `refresh`：始终重新调用 LLM，并覆盖同 key 的旧计划。

缓存内容包括：

- `model_name`
- `config_hash` 与公开规划配置指纹
- `prompt_hash`
- `env_id`
- `reason`
- `planning_horizon`
- `timestep`
- `state_summary`
- `status_brief`
- `raw_response`
- `buffered_action`
- `buffered_setpoints`
- setpoint contract 后的 `parsed_plan`
- fallback/anchor metadata

注意：cache key 不包含 HEM/expert rollout 开关，因为这些模块属于低层 rollout 增强，不应改变同一高层 LLM plan 的回放身份。

### 2. 诊断脚本扩展

`gl_gym/experiments/diagnose_ppo_vs_llm.py` 新增参数：

```bash
--llm-plan-cache-mode off|record|replay|refresh
--llm-plan-cache-path gl_gym/result/plan_cache/llm_plan_cache.json
--llm-plan-cache-strict
```

逐步诊断 CSV/JSON 中新增：

- `plan_cache_mode`
- `plan_cache_enabled`
- `plan_cache_hit`
- `plan_cache_status`
- `plan_cache_key`
- `plan_cache_attempt`

这些字段用于确认每次重规划到底来自 LLM、cache hit，还是 replay miss fallback。

### 3. Frozen Benchmark Runner

新增 `gl_gym/experiments/run_frozen_benchmark.py`。

默认矩阵：

- years: `2010,2015,2020`
- days: `59,120,180,240`
- seeds: `42,43,44`
- max steps: `240`
- controllers: `llm,llm_hem,ppo`

建议流程：

```bash
python gl_gym/experiments/run_frozen_benchmark.py \
  --controllers llm,llm_hem,ppo \
  --plan-cache-mode record \
  --plan-cache-path gl_gym/result/plan_cache/frozen_36x240_qwen.json \
  --output-json gl_gym/result/benchmarks/frozen_36x240_record.json
```

随后在不调用新 LLM 的情况下回放：

```bash
python gl_gym/experiments/run_frozen_benchmark.py \
  --controllers llm,llm_hem,ppo \
  --plan-cache-mode replay \
  --plan-cache-strict \
  --plan-cache-path gl_gym/result/plan_cache/frozen_36x240_qwen.json \
  --output-json gl_gym/result/benchmarks/frozen_36x240_replay.json
```

### 4. Reward 审计

新增 `gl_gym/experiments/audit_reward_contract.py`。

当前审计结论：

- `GreenhouseReward.compute_reward()` 会计算 `self.penalty = self.output_penalty_reward(violations)`。
- 但最终返回是 `scaled_profit - scaled_pen - self.control_pen`。
- 因此 `pen_weights` 没有直接进入训练 reward 返回值。
- 状态违规则通过未加权的归一化 `scaled_pen` 进入 reward。

论文实验前建议把当前版本冻结为 `reward_v0`。如果后续重训 PPO/SAC，应明确选择：

- 保持 `reward_v0`，用于和已有结果连续比较。
- 或定义 `reward_v1`，让 RH/temp penalty 权重显式进入训练目标，并重新训练全部 RL 基线。

## 下一步实验口径

先不跑 960 步，也不跑全周期。建议顺序：

1. 先用 `record` 生成 36 场景、240 步的 frozen LLM plan cache。
2. 用 `replay --strict` 对比 `llm` 和 `llm_hem`，确认 HEM/RSPC 改动是在同一高层计划下比较。
3. 跑 reward audit，确认本轮 RL baseline 用的是 `reward_v0` 还是准备切到 `reward_v1`。
4. 开始 PPO/SAC 搜参，选模标准使用 Pareto：
   - total reward
   - profit
   - RH violation
   - temp violation
   - heat cost
   - heat+vent conflict
5. RSPC 主线优先改除湿本体，而不是继续扩大 HEM：
   - `economic_dehumidify`
   - `safe_dehumidify`
   - `emergency_dehumidify`
   - 统一 horizon risk score

## 本轮验证

已通过：

```bash
python -m py_compile gl_gym/agent/plan_cache.py gl_gym/agent/llm_agent.py gl_gym/experiments/diagnose_ppo_vs_llm.py gl_gym/experiments/run_frozen_benchmark.py gl_gym/experiments/audit_reward_contract.py
python -m pytest tests/test_planning_extensions.py tests/test_humidity_experience_memory.py tests/test_plan_intent_distillation.py tests/test_ppo_strategy_labeler.py tests/test_plan_cache.py tests/test_frozen_benchmark.py -q
python gl_gym/experiments/run_frozen_benchmark.py --dry-run --years 2020 --days 240 --seeds 42 --controllers llm,llm_hem,ppo --max-steps 2 --plan-cache-mode replay
python gl_gym/experiments/audit_reward_contract.py
```

结果：

- `38 passed`
- benchmark dry-run 正确生成 3 个 job
- reward audit 输出当前 reward 合同问题

