# MC-SERO Shadow Diagnostic 2026-04-30

## Version Context

- Base branch: `codex/literature-method-backup`
- Previous commit: `4e5d538 Add MC-SERO metric audit fields`
- This phase keeps the applied LLM-RSPC action unchanged.

## Implemented Scope

This phase adds MC-SERO as a shadow diagnostic layer only.

- `mc_sero_mode="off"` remains the default.
- `mc_sero_mode="shadow"` evaluates mechanism candidates in the rollout layer.
- The shadow evaluator records what it would select, its margin against the actual baseline rollout action, score terms, and reject reasons.
- It does not overwrite `control`, `selected_control`, source labels, reward, or applied environment actions.

## Candidate Logic

The shadow pool contains the existing rollout candidates plus mechanism candidates:

- existing anchor/rule/blend candidates
- `economy_hold`
- `economic_dehumidify`
- `safe_dehumidify`
- `emergency_dehumidify`
- `dry_recovery`
- `heat_buffer`

Each candidate is scored over a deterministic 6-step proxy horizon. The score includes:

- low/high temperature risk
- high RH risk
- low RH risk
- high/low VPD risk
- dew risk
- energy proxy
- heat+vent, CO2+vent, and lamp+vent conflict
- dry-state ventilation penalty
- cold-state ventilation penalty

## Diagnostics

Per-step diagnostics now include:

- `mc_sero_enabled`
- `mc_sero_available`
- `mc_sero_would_select`
- `mc_sero_best_candidate`
- `mc_sero_best_control`
- `mc_sero_margin`
- `mc_sero_reject_reason`
- `mc_sero_baseline_source`
- `mc_sero_score_terms`
- `mc_sero_candidate_count`
- `mc_sero_horizon_steps`

Aggregate diagnostics include enabled/available/would-select steps, mean margin, best-candidate counts, and reject-reason counts.

## Intended Next Gate

Run replay-only frozen benchmarks under the same cached LLM plans:

- `llm`
- `llm_sero_shadow`

The expected result is identical applied behavior, with additional MC-SERO diagnostics only. If the shadow best-candidate distribution is mechanistically reasonable across the 36-scenario benchmark, the next phase can consider a guarded `select` mode.
