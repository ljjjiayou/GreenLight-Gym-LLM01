# C-STCC v8173.4 Risk-Semantic Energy Proxy Formula Redesign

Existing-trace counterfactual audit only. No online LLM, no rollout, no runtime scorer/template change.

## Observed Gap

- v8173.3 no-op best: canopy_lt0=1.0|canopy_lt1=1.0|dew=1.0|dry_vent=1.0|dry=1.0
- Interpretation: risk-family energy-gap relief did not touch the core action-risk semantics
- Dry-vent gap: dry_vent=21/21 over-target under v8173.2/v8173.3
- Canopy/dew gap: canopy_dew_lt0+dew_or_humidity=0/6 under-target under v8173.2/v8173.3

## Boundary

- Audit success rate: 1.000
- Final-action invariant rate: 1.000
- Online LLM / rollout / real projection steps: 0 / 0 / 0

## Best Formula

- Policy / formula: canopy_dew_first_v1 / baseline_current_energy
- Conservative risk selected rate: 0.540
- Family / mixed combo / single spread: 0.641 / 1.000 / 0.375
- Dry-vent selected: 21 / 21
- Canopy-dew/dew selected: 0 / 6
- Dominant component after formula: raw_energy_proxy
- Raw/projected gap identical: True

## Decision

- Next action: v8173_combo_specific_energy_formula_iteration_plan
- Readiness reasons: specific_mixed_combo_extremes_remain
