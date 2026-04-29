# HEM-RSPC Closed-Loop Validation 2026-04-29

## Purpose

This validation checked whether the humidity experience memory (HEM) can be used
inside the LLM-RSPC closed loop as an optional rollout candidate. HEM is still
disabled by default and must be enabled with `--llm-humidity-memory`.

## Implementation Notes

- Added HEM rollout candidates to the LLM-RSPC candidate pool:
  `humidity_memory`, `humidity_memory_anchor_blend`, and
  `humidity_memory_rule_blend`.
- Kept metadata gating for `teacher_policy_id`, `baseline_controller_id`, and
  `memory_schema_version`.
- Shaped `free_air_exchange` memory candidates to keep low heating, low CO2,
  low lighting, low screen, and sufficient ventilation.
- Added a narrow post-guardrail HEM shaping gate. It only applies after a HEM
  candidate is selected and the current state remains in a safe economic band.
  At night below 18 C, the gate uses a moderated free-air pulse rather than full
  ventilation: low screen, ventilation capped near 0.65, and a small heat buffer.
- Added diagnostics fields for memory availability, selection, metadata, trust,
  distance, strategy label, and post-guardrail shaping.

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

Main diagnostic files:

```text
gl_gym/result/diagnostics/ppo_vs_llm_current_baseline_s240_day240_20260429.json
gl_gym/result/diagnostics/ppo_vs_llm_hem_moderated_s240_day240_20260429.json
```

| Controller | Reward | Profit | Revenue | Heat | CO2 | Elec | Temp V | RH V | Heat+Vent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current LLM-RSPC | 151.320 | -0.04995 | 0.01620 | 0.05866 | 0.00228 | 0.00522 | 0.000 | 0.822 | 69 |
| LLM-RSPC + HEM moderated | 151.531 | -0.04557 | 0.01605 | 0.06162 | 0.00000 | 0.00000 | 0.000 | 0.909 | 69 |
| PPO reference | 151.919 | -0.03492 | 0.01916 | 0.05408 | 0.00000 | 0.00000 | 0.000 | 2.968 | 49 |

HEM moderated vs current LLM-RSPC:

```text
reward: +0.210912
profit: +0.004387
heat cost: +0.002960
CO2 cost: -0.002278
electricity cost: -0.005220
temperature violation: +0.000000
RH violation: +0.087059
heat+vent conflict: +0 steps
```

HEM activity:

```text
memory accepted steps: 52 / 240
memory selected steps: 4 / 240
post-guardrail shaped steps: 4 / 240
```

Selected HEM steps were night-time moderate humidity free-air pulses. The
moderated gate kept low screen and bounded ventilation, instead of allowing the
earlier overly aggressive `screen=0.35, vent=1.0` behavior.

## Decision

The current HEM closed-loop version is promising but not strong enough for a
960-step run yet. It improves reward and profit against the same-day current
LLM-RSPC baseline and does not increase heat+vent conflict, but RH violation
increases slightly. The next optimization should improve the RH/economic tradeoff
before long-horizon evaluation.

Recommended next step:

```text
Keep HEM optional and default-off.
Use the moderated 240-step result as an ablation checkpoint.
Before 960-step testing, add a horizon-aware HEM selection penalty that rejects
night pulses when the predicted temperature recovery cost is higher than the
CO2/electricity savings.
```

## Verification

```text
python -m py_compile gl_gym/agent/humidity_experience_memory.py gl_gym/agent/llm_agent.py gl_gym/experiments/diagnose_ppo_vs_llm.py
python -m pytest tests/test_planning_extensions.py tests/test_humidity_experience_memory.py tests/test_plan_intent_distillation.py tests/test_ppo_strategy_labeler.py -q
```

Result:

```text
30 passed
```
