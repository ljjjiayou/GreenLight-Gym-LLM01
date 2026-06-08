# C-STCC v8174 Risk-Specific Conservative Template Design

Existing-trace offline template design audit only. Runtime template pool and final actions are unchanged.

## Boundary

- Audit success rate: 1.000
- Final-action invariant rate: 1.000
- Online LLM / rollout / real projection steps: 0 / 0 / 0

## Best Template Counterfactual

- Policy / formula: canopy_dew_first_v1 / formula_G_combo_specific_two_sided_semantics
- Pseudo-template risk selected rate: 0.722
- Canopy-dew/dew pseudo selected: 6 / 6
- Dry-vent pseudo selected: 21 / 21
- Nonrisk pseudo selected: 0

## Decision

- Next action: v8174_template_parameter_iteration_plan
- Readiness reasons: dry_vent_pseudo_template_saturated
