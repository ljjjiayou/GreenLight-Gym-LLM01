# C-STCC v8173.1 Mixed-Risk Conflict Resolver Design

Existing-trace counterfactual audit only. No online LLM, no rollout, no runtime scorer/template change.

## Boundary

- Audit success rate: 1.000
- Final-action invariant rate: 1.000
- Online LLM / rollout / real projection steps: 0 / 0 / 0
- Final action changed steps: 0

## Mixed-Risk Diagnostics

- Risk rows: 126
- Mixed / single risk rows: 78 / 48
- Mixed combo distribution: {"canopy_dew_lt0+dew_or_humidity": 6, "canopy_dew_lt0+dew_or_humidity+dry_or_high_vpd": 8, "canopy_dew_lt1+dew_or_humidity": 32, "canopy_dew_lt1+dew_or_humidity+dry_or_high_vpd": 3, "dry_vent+canopy_dew_lt0+dew_or_humidity+dry_or_high_vpd": 4, "dry_vent+canopy_dew_lt1+dew_or_humidity+dry_or_high_vpd": 4, "dry_vent+dry_or_high_vpd": 21}
- Conflict type distribution: {"dew_vs_canopy_dew": 38, "multi_3plus": 19, "single_risk": 69}

## Resolver Counterfactual

- Best vector: dry=0.06|dry_vent=0.06|dew=0.10|canopy=0.12
- Conservative risk selected rate: 0.841
- Risk-type / mixed-combo spread: 0.450 / 0.000
- Min-spread vector: dry=0.00|dry_vent=0.00|dew=0.00|canopy=0.00
- Why min-spread not selected: min_spread_vector_conservative_risk_selected_rate_outside_viable_range
- Selected id vs zero-bonus mismatch count: 248

## Decision

- Next action: v8173_2_conflict_resolver_policy_iteration
- Readiness reasons: mixed_risk_combo_or_family_spread_still_high
