# C-STCC v8173 Risk-Type-Specific Bonus + Template Counterfactual

Existing-trace counterfactual scorer audit only. No online LLM, no rollout, no runtime scorer or template change.

## Boundary

- Audit success rate: 1.000
- Final-action invariant rate: 1.000
- Online LLM / rollout / real projection steps: 0 / 0 / 0
- Final action changed steps: 0

## Best Bonus Vector

- Vector: dry=0.06|dry_vent=0.06|dew=0.10|canopy=0.12
- Conservative risk selected rate: 0.841
- Risk-type switch spread: 0.450
- Nonrisk conservative selected count: 0
- Selected id vs zero-bonus mismatch count: 248

## Template Variant Counterfactual

- Best pseudo template: conservative_stabilize_dew for dew_or_humidity
- Mean score delta vs conservative_stabilize: 0.005550
- Would beat zero-bonus winner rate: 0.000

## Decision

- Next action: v8173_mixed_risk_conflict_resolver_design_plan
- Readiness reasons: risk_type_bonus_vector_not_balanced
