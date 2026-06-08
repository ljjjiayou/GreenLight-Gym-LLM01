---
name: greenhouse-architecture-innovation
description: "Greenhouse architecture innovation workflow for LLM/MPC/PPO-inspired control-chain redesign, action continuity, and evidence-backed experiments."
---

# Greenhouse Architecture Innovation

Use this skill when the user discusses control-chain redesign, PPO/MPC
comparison, LLM role design, action smoothness, long-horizon behavior, runtime
stability, or publishable method innovation.

This skill is intentionally pro-experiment. It should help move from insight to
evidence, not keep the project stuck in design-only work.

## Working View

- LLM should remain a high-level reasoner unless an explicit experiment tests a different role.
- Useful LLM roles include regime recognizer, risk forecaster, objective/prior generator, critic, cache indexer, constraint-priority estimator, and experiment explainer.
- Low-level continuous control should be handled by deterministic composers, learned policies, MPC-style optimizers, scorers, safety projection, or shields.
- PPO performance is a useful reference because PPO learns temporal action continuity into parameters; LLM-based control needs an explicit temporal mechanism instead of only step-local action selection.

## Architecture Directions To Explore

- State/history-aware regime and risk recognition.
- Trajectory-level objective or priority generation instead of single-step profile selection.
- Smooth action-envelope generation with rate, reversal, and total-variation penalties.
- Regime-conditioned policy priors or expert-distilled candidate families.
- MPC or rollout-style short-horizon evaluation around LLM semantic guidance.
- Tomato Safety projection inside scoring plus a final hard shield.
- Runtime stability features that distinguish solver diagnostics from controller-quality evidence.

## Experiment Rule

If online LLM evidence is useful, use it with a clear manifest:

```text
question:
model:
scenario_windows:
max_calls_or_budget:
cache_write_policy:
trace_outputs:
final_action_changed: true/false
default_controller_changed: true/false
success_condition:
stop_condition:
```

Do not avoid online LLM merely to reduce calls. Avoid it only when it cannot
answer the current question or would make the evidence unreproducible.

## Output Shape

For architecture work, prefer:

```text
observed_gap:
control_chain_layer:
new_architecture_hypothesis:
why_it_addresses_temporal_continuity:
how_llm_is_used:
how_actions_are_smoothed_or_optimized:
required_trace_fields:
minimal_experiment:
evidence_level_after_success:
```
