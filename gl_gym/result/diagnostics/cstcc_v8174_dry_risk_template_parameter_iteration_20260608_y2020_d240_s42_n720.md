# C-STCC v8174.1 Dry-Risk Template Parameter Iteration

Existing-trace offline template parameter audit only. Runtime template pool and final actions are unchanged.

## Boundary

- Audit success rate: 1.000
- Final-action invariant rate: 1.000
- Online LLM / rollout / real projection steps: 0 / 0 / 0

## Best Dry-Risk Counterfactual

- Policy / formula: canopy_dew_first_v1 / formula_G_combo_specific_two_sided_semantics
- Pseudo-template risk selected rate: 0.675
- Canopy-dew/dew pseudo selected: 6 / 6
- Dry-vent pseudo selected: 21 / 21
- Dry-vent limited guard selected: 17 / 21
- Mild dry limited-guard selections: 0
- Nonrisk pseudo selected: 0

## Decision

- Next action: v8174_multi_scenario_existing_trace_scan_or_acquisition_plan
- Readiness reasons: single_h720_trace_only
