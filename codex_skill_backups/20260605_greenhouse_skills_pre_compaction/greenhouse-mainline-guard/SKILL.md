---
name: greenhouse-mainline-guard
description: "Greenhouse project core guardrails for safety, evidence levels, default-controller changes, performance claims, and hypothesis revision."
---

# Greenhouse Mainline Guard

Use this skill when a task may affect controller logic, default configuration,
experimental evidence, safety claims, or project architecture.

This skill is a guardrail, not a freeze. It protects scientific rigor and
deployment safety while allowing architecture exploration.

## Hard Invariants

These are permanent constraints:

- Do not expose secrets, API keys, private cache payloads, or credentials.
- Do not fabricate experiments, traces, metrics, or provenance.
- Do not claim performance improvement from shadow-only evidence.
- Do not treat controlled-canary evidence as default-controller promotion.
- Do not change the default controller or promote a controller without explicit user authorization.
- Do not offset dew/canopy hard-safety regression with reward, energy, RH-low, VPD, or yield gains.
- If an action diff, hard-safety regression, runtime error, cache miss, or CVODES/CasADi failure appears, diagnose before escalating the evidence level.

## Mainline As Hypothesis

The mainline is the current best-supported working hypothesis, not a permanent
architecture constraint.

Current default assumptions may include `llm_rspc_v2`, Tomato Safety as a hard
shield, and normal-path profile/action candidates. These assumptions can be
revised when evidence shows they are insufficient.

Architecture exploration is allowed if it is separated from deployment:

- It may design alternate controllers, profile generators, composers, scorers, caches, or hybrid/MPC structures.
- It may write shadow-only instrumentation or offline audit code.
- It must not silently change the default controller, final action path, or promotion status.
- It must state which assumption is being challenged and what evidence would validate the revision.

## Hypothesis Revision Protocol

When proposing a mainline or architecture change, include:

```text
old_hypothesis:
failure_evidence_or_missing_evidence:
new_hypothesis:
smallest_testable_next_step:
default_controller_changed: true/false
online_llm_needed: true/false
hard_safety_impact:
success_condition:
stop_condition:
```

## Evidence Ladder

Keep these levels distinct:

```text
source hint -> design-only -> unit/synthetic test -> existing-trace audit
-> strict cache replay -> opt-in shadow trace -> controlled canary
-> promotion evidence
```

Moving up the ladder requires an explicit reason and a clear artifact. Moving
down the ladder for diagnosis is always allowed.

## Compact Alignment Note

Use this short note when needed:

```text
impact_level:
default_llm_rspc_v2_changed:
current_mode:
online_llm_needed:
key_gate:
controlled_or_promotion_allowed:
hypothesis_revision:
```
