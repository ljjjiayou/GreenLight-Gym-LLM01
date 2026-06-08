---
name: greenhouse-safety-boundary
description: "Greenhouse safety guidance for canopy, dew, RH, VPD, temperature, Tomato Safety, guardrails, runtime provenance, and hard-safety failures."
---

# Greenhouse Safety Boundary

Use this skill for tasks involving dew/canopy risk, VPD/RH, temperature,
Tomato Safety, guardrails, post-guardrail rewrites, unsafe candidates, or
hard-safety failure diagnosis.

This skill protects hard safety. It does not prohibit architecture exploration.

## Hard Safety

- Dew/canopy hard-safety regression is a hard blocker; do not offset it with reward, energy, RH-low, VPD, or yield gains.
- Distinguish warning, near-miss, unsafe preferred, unsafe projected, and unsafe applied.
- Tomato Safety or post-guardrail rewrites need runtime provenance when they affect action interpretation.
- Unknown post-guardrail rewrites or missing reasons should trigger trace/audit repair before deployment claims.
- Safety-affecting runtime changes move through shadow-only, strict cache/replay, controlled canary, then promotion evidence.

## Safety Does Not Freeze Research

Allowed under safety guardrails:

- designing alternate safety layers or formal shields;
- auditing scorer/safety incompatibility;
- testing safety logic in shadow-only instrumentation;
- proposing architecture changes that may reduce safety conflicts.

Not allowed without explicit evidence and authorization:

- silently changing final safety enforcement;
- weakening Tomato Safety hard constraints;
- claiming a safety improvement from shadow-only evidence;
- promoting a controller after only warning/near-miss analysis.

## Diagnostic Categories

- `response_proxy_false_safe`
- `guardrail_introduced_risk`
- `scorer_selected_unsafe`
- `unsafe_projected_candidate`
- `unsafe_applied_action`
- `no_safe_candidate_available`
- `candidate_metadata_missing`
- `runtime_provenance_missing`
- `tomato_safety_projection_missing`
- `hard_safety_regression`

## Recommended Evidence

- canonical failure trace;
- false-safe / false-positive audit;
- runtime provenance audit;
- Tomato Safety projection audit;
- action-diff and post-guardrail rewrite audit;
- controlled trace safety audit when deployment behavior changes.
