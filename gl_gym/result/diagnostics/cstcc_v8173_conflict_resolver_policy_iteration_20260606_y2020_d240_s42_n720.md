# C-STCC v8173.2 Conflict Resolver Policy Iteration

Existing-trace counterfactual audit only. No online LLM, no rollout, no runtime scorer/template change.

## Boundary

- Audit success rate: 1.000
- Final-action invariant rate: 1.000
- Online LLM / rollout / real projection steps: 0 / 0 / 0

## Best Policy

- Policy: canopy_dew_first_v1
- Vector: dry=0.06|dry_vent=0.06|dew=0.08|canopy=0.10
- Conservative risk selected rate: 0.540
- Family / mixed combo / single spread: 0.641 / 1.000 / 0.375
- Family target penalty: 0.200
- Family extreme rate count: 1
- Dominant non-final component: raw_energy_proxy

## Decision

- Next action: v8173_energy_proxy_reweight_design_plan
- Readiness reasons: family_spread_high_and_energy_proxy_dominant
