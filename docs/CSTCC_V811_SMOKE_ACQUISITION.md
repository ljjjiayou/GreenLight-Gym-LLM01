# C-STCC v81.1 Smoke Acquisition

## Purpose

Validate that C-STCC runtime shadow can run safely in a short real runtime trace, generate compact audit rows, and preserve the original runtime final action.

This stage is not a new controller, not a rollout stage, and not performance evidence.

## Boundaries

- Do not evaluate reward or controller quality.
- Do not change final action.
- Do not call online LLM.
- Do not execute predictive rollout.
- Do not use real Tomato Safety projection inside C-STCC.
- Do not make performance, safety, or promotion claims.

Required boundary values:

```text
online_llm_called = false
predictive_rollout_executed = false
real_tomato_safety_projection = false
selected_action_is_shadow_only = true
prediction_level = 0
```

## First Smoke Run

Use a very short trace first:

```text
max_steps = 30
sample_rate = 1.0
save_full_candidates = false
save_raw_sequences = false
save_projected_sequences = false
fail_closed = false
```

The first run only validates the side-channel audit path and compact JSONL output.

## Pass Criteria

- C-STCC shadow JSONL is generated.
- `audit_success_rate` is 100%, or failures are recorded without interrupting runtime.
- `final_action_invariant_rate` is 100%.
- `final_action_changed` is always false.
- `online_llm_called` is always false.
- `predictive_rollout_executed` is always false.
- `real_tomato_safety_projection` is always false.
- Compact rows can be flattened by the diagnostic path.
- Audit latency and JSONL row size are acceptable for a short trace.

## Review Order

1. Safety boundary metrics:
   `final_action_invariant_rate`, `final_action_changed`, `audit_success_rate`, `audit_failure_rate`, `online_llm_called`, `predictive_rollout_executed`, `real_tomato_safety_projection`.
2. Runtime overhead:
   `audit_latency_ms` mean/p95 and `json_size_kb` mean/max.
3. Candidate health:
   `candidate_count`, `feasible_candidate_count`, `infeasible_candidate_ratio`, `fallback_rate`, hard violation reason distribution.
4. Shadow/runtime difference:
   mean/p95/max absolute action difference, especially `u_ventilation`, `u_heating`, `u_co2`, and `u_screen`.

## Branching

If all boundary checks pass, proceed to a full-episode v81.1 trace while still preserving final action.

If audit failures occur but runtime continues and final action remains invariant, repair the failure cause and rerun smoke before expanding.

If final-action invariant fails, stop the C-STCC shadow integration and inspect the hook, mutable final-action references, normalization code, flattening code, and `env.step` action sharing.

Do not enter v82 until v81.5 shadow analytics has summarized smoke and full-episode traces.
