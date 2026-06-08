---
name: greenhouse-mainline-guard
description: "Minimal Greenhouse project guardrails for scientific integrity, safety boundaries, default-controller changes, and evidence claims."
---

# Greenhouse Mainline Guard

Use this skill when work may affect controller logic, default configuration,
experimental evidence, safety claims, secrets, or project architecture.

This is a minimal guardrail. It must not slow useful architecture work or online
LLM experiments when they are the right next step.

## Non-Negotiables

- Do not expose secrets, API keys, private cache payloads, or credentials.
- Do not fabricate experiments, traces, metrics, provenance, or benchmark wins.
- Do not claim performance or safety improvement beyond the evidence actually run.
- Do not silently change the default controller, final-action path, or promotion status.
- Do not promote a controller or default config without explicit user authorization.
- Do not offset dew, canopy, or hard-safety regressions with reward, energy, RH-low, VPD, or yield gains.

## Go Rules

- The current mainline is a working hypothesis, not doctrine.
- Architecture revision, controller redesign, hybrid MPC/LLM structures, and online LLM experiments are allowed.
- Use online LLM whenever it can produce useful project evidence; record scenario scope, cache policy, trace location, budget, and whether final actions can change.
- If action diff, hard-safety regression, runtime error, cache miss, or CVODES/CasADi failure appears, diagnose it before claiming higher evidence.
- Prefer the smallest experiment that answers the question, but do not avoid a larger experiment merely to reduce LLM calls.

## Evidence Labels

Keep these levels distinct:

```text
design-only
unit/synthetic test
existing-trace audit
strict cache replay
opt-in shadow trace
controlled canary
promotion evidence
```

## Hypothesis Revision Note

When changing the working direction, keep this compact note:

```text
old_hypothesis:
failure_evidence_or_missing_evidence:
new_hypothesis:
next_experiment:
default_controller_changed: true/false
online_llm_needed: true/false
hard_safety_impact:
success_condition:
stop_condition:
```
