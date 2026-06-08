---
name: greenhouse-shadow-audit-runner
description: "Greenhouse experiment navigation for audits, cache/replay, opt-in shadow traces, online LLM authorization, action-diff checks, and readiness decisions."
---

# Greenhouse Shadow Audit Runner

Use this skill when executing or planning audits, trace analysis, cache coverage,
strict replay, opt-in shadow traces, online cache acquisition, shadow sweeps, or
controlled canaries.

The goal is not to avoid experiments. The goal is to run the right evidence
level with explicit provenance.

## Experiment Matrix

Use the least risky level that can answer the question.

| experiment type | online LLM | default execution | purpose |
| --- | --- | --- | --- |
| unit or synthetic audit | no | allowed | verify code and schema |
| existing-trace audit | no | allowed | analyze current evidence |
| design/readiness checklist | no | allowed | define next gate |
| strict cache replay | no | allowed when cache coverage is known | reproduce cached LLM behavior |
| opt-in shadow trace with existing cache | no | requires explicit scenario/config | validate instrumentation with action invariance |
| cache acquisition | yes | requires authorization | obtain new LLM plans/anchors/cache |
| online LLM shadow rollout | yes | requires authorization | generate new shadow evidence |
| controlled canary | maybe | requires high-evidence admission | validate behavior changes |

Online LLM use is not forbidden. It is an experiment type that needs explicit
authorization, scope, provenance, and budget.

## Online LLM Authorization Protocol

When online LLM is needed, produce or request:

```text
why_existing_cache_or_trace_is_insufficient:
model:
scenario_windows:
max_calls_or_budget:
cache_write_policy:
default_controller_changed: true/false
final_action_changed: true/false
shadow_only: true/false
required_provenance_fields:
success_condition:
stop_condition:
```

Do not run online LLM as an incidental side effect of an audit or unit test.

## Required Checks

- cache: hit rate, strict miss, payload completeness, selected-cache misuse.
- runtime: runtime error, short trajectory, simulator collapse, CVODES/CasADi failure.
- action: metadata-only and shadow-only work must preserve final action unless explicitly authorized.
- safety: unsafe preferred/applied, canopy/dew hard regression, Tomato Safety and post-guardrail provenance.
- evidence: report only claims supported by the current evidence level.

## Advancement Rules

- If the audit can answer the question with existing traces, do that first.
- If existing traces cannot answer the question, propose cache acquisition or opt-in shadow trace authorization instead of pretending evidence exists.
- If action diff is nonzero in a metadata-only or shadow-only run, stop and diagnose.
- If a short-window source is identified, long-horizon/full-cycle shadow feasibility is useful but not mandatory before architecture diagnosis.
- New scripts are allowed when existing audit/status/readiness scripts cannot express the evidence; new scripts should have CLI, JSON/MD outputs, and tests.

## Failure Taxonomy

Every failed experiment should leave a useful category, such as:

- `cache_missing`
- `online_llm_authorization_missing`
- `runtime_error`
- `cvodes_or_casadi_failure`
- `action_invariance_violation`
- `contract_metadata_missing`
- `tomato_safety_provenance_missing`
- `hard_safety_regression`
- `insufficient_trace_fields`
