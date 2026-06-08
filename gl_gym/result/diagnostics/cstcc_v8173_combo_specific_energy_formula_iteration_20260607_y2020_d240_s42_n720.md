# C-STCC v8173.5 Combo-Specific Risk-State Energy Semantics

Existing-trace counterfactual audit only. No online LLM, no rollout, no runtime scorer/template change.

## Hypothesis Revision

- Old hypothesis: action-direction energy formulas can resolve mixed-risk conservative selection imbalance
- Failure evidence: v8173.4 best remained baseline_current_energy; dry_vent stayed saturated; canopy_dew_lt0+dew_or_humidity stayed zero; raw_energy_proxy stayed dominant
- New hypothesis: combo-specific risk-state transition semantics are required before opt-in runtime tracing

## Boundary

- Audit success rate: 1.000
- Final-action invariant rate: 1.000
- Online LLM / rollout / real projection steps: 0 / 0 / 0

## Best Risk-State Formula

- Policy / formula: canopy_dew_first_v1 / baseline_current_energy
- Conservative risk selected rate: 0.540
- Family / mixed combo / single spread: 0.641 / 1.000 / 0.375
- Dry-vent selected: 21 / 21
- Canopy-dew/dew selected: 0 / 6
- Dominant component after formula: raw_energy_proxy

## Decision

- Next action: v8174_risk_specific_template_design_plan
- Readiness reasons: canopy_dew_dew_combo_still_under_target
