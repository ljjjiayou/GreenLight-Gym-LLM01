# HEM-RSPC Horizon Filter Validation 2026-04-29

## Purpose

This run tested a HEM-only horizon-aware safety filter. The goal was to prevent
night free-air memory pulses from consuming thermal buffer and shifting RH risk
to later dawn steps, while keeping normal LLM-RSPC behavior unchanged when HEM is
disabled.

## Implementation Notes

- Added a HEM-only candidate filter after the existing rollout/fallback scoring.
- The filter estimates direct saving, short-horizon RH debt, terminal humidity
  risk, and temperature recovery cost against the best non-HEM candidate.
- HEM is rejected when night temperature buffer is low, dew margin is low,
  forecast humidity risk is high, the recent night-pulse gap is too short, or
  the adjusted HEM score does not beat the best non-HEM candidate by a margin.
- Night HEM post-guardrail shaping is now less aggressive: ventilation is capped
  lower, heat has a small floor, and screen is not forced as open as before.
- HEM remains disabled by default and is only enabled by
  `--llm-humidity-memory`.

## 240-Step Results

Scenario:

```text
year=2020
day=240
seed=42
steps=240
model=qwen-max-latest
llm_interval=12
teacher_policy_id=ppo_best_model_20260428
baseline_controller_id=llm_rspc_887a5ef
memory_schema_version=hem_rspc_v1
```

Diagnostic files:

```text
gl_gym/result/diagnostics/ppo_vs_llm_horizon_baseline_s240_day240_20260429.json
gl_gym/result/diagnostics/ppo_vs_llm_hem_horizon_s240_day240_20260429.json
```

| Controller | Reward | Profit | Revenue | Heat | CO2 | Elec | Temp V | RH V | Heat+Vent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLM-RSPC, HEM off | 151.613 | -0.04391 | 0.01604 | 0.05996 | 0.00000 | 0.00000 | 0.000 | 0.898 | 68 |
| LLM-RSPC + HEM horizon filter | 151.692 | -0.04233 | 0.01643 | 0.05876 | 0.00000 | 0.00000 | 0.000 | 0.894 | 67 |
| PPO reference | 151.919 | -0.03492 | 0.01916 | 0.05408 | 0.00000 | 0.00000 | 0.000 | 2.968 | 49 |

HEM horizon filter vs same-batch LLM-RSPC:

```text
reward: +0.078365
profit: +0.001581
revenue: +0.000384
heat cost: -0.001197
CO2 cost: +0.000000
electricity cost: +0.000000
temperature violation: +0.000000
RH violation: -0.003990
heat+vent conflict: -1 step
```

HEM activity:

```text
memory available steps: 53 / 240
memory accepted steps: 53 / 240
memory selected steps: 0 / 240
memory rejected steps: 53 / 240
reject reasons:
  - insufficient_score_margin: 31
  - night_temp_buffer: 22
```

## Decision

The horizon filter fixed the previous unsafe behavior: night low-temperature
free-air pulses were blocked, and RH violation did not increase. However, no HEM
candidate was selected in this 240-step scenario. Therefore, this result should
not be claimed as a closed-loop HEM performance improvement.

Do not run the 960-step experiment from this checkpoint. The next step should be
to re-mine or revalidate the humidity experience memory under the current
LLM-RSPC baseline and promote only cases that retain positive direct saving after
the horizon filter.

## Verification

```text
python -m py_compile gl_gym/agent/humidity_experience_memory.py gl_gym/agent/llm_agent.py gl_gym/experiments/diagnose_ppo_vs_llm.py
python -m pytest tests/test_planning_extensions.py tests/test_humidity_experience_memory.py tests/test_plan_intent_distillation.py tests/test_ppo_strategy_labeler.py -q
```

Result:

```text
33 passed
```
