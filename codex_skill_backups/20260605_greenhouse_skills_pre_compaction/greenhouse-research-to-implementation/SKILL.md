---
name: greenhouse-research-to-implementation
description: "Translate greenhouse research ideas, GPT reviews, papers, expert feedback, and architecture hypotheses into testable implementation plans."
---

# Greenhouse Research To Implementation

Use this skill when converting external advice, paper ideas, GPT reviews, or
architecture hypotheses into project work.

Research input is not automatically controller logic. It becomes a hypothesis
with a smallest testable next step.

## Research Translation Flow

1. Extract the core claim:
   method contribution, engineering claim, experimental claim, or paper narrative.
2. Map the layer:
   intent, profile, candidate composer, scorer/arbitrator, safety, cache, replay,
   evaluation, reproducibility, or deployment.
3. Classify the evidence:
   source hint, design-only, shadow evidence, cache/replay evidence, controlled
   canary evidence, promotion evidence, or missing evidence.
4. Decide the smallest useful next step:
   design, instrumentation, audit, cache acquisition, shadow trace, canary
   admission, or architecture revision.

## Architecture Exploration Is Allowed

The current project mainline is a working hypothesis, not a fixed doctrine.

The agent may propose alternate architectures when the current hypothesis looks
insufficient, including:

- replacing or reshaping `profile_generator`;
- using action envelopes instead of direct profile-to-action mapping;
- adding regime-conditioned policy priors;
- restructuring fallback into explicit baseline candidate families;
- redesigning scorer/arbitrator objectives;
- changing the role of LLM from setpoint/profile author to regime switch, critic, cache indexer, or constraint-priority estimator;
- introducing semantic cache, expert distillation, MPC wrappers, or formal shield designs.

Such exploration must stay separated from deployment. It may plan, prototype
shadow-only instrumentation, or define audits; it must not silently promote a
controller or make performance claims.

## Hypothesis Revision Output

When the recommendation changes the mainline, write:

```text
current_hypothesis:
observed_gap:
alternative_hypothesis:
why_this_may_be_better:
minimal_experiment:
required_trace_fields:
online_llm_needed:
expected_artifacts:
success_condition:
stop_condition:
```

## Evidence Discipline

- Write target values and expected gains as hypotheses or ranges until measured.
- Do not present paper motivation, GPT opinion, or source hints as experiment results.
- Separate architecture exploration from controlled-canary or promotion evidence.
- Prefer reusable audits and trace fields over one-off narrative claims.
- If the best next step requires online LLM, name it as cache acquisition or online shadow trace authorization rather than banning it.
