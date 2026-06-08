# C-STCC v8173.3 Risk-Conditioned Energy Proxy Reweight Design

Existing-trace counterfactual audit only. No online LLM, no rollout, no runtime scorer/template change.

## Observed Gap

- v8173.2 best compromise: canopy_dew_first_v1 / dry=0.06|dry_vent=0.06|dew=0.08|canopy=0.10
- This is a target-band/spread compromise, not a claim of control-semantic optimality.
- Dominant v8173.2 component: raw_energy_proxy
- dry_vent over-target: dry_vent=21/21 selected under v8173.2 best compromise
- canopy-dew/dew mixed under-target: canopy_dew_lt0+dew_or_humidity=0/6 selected under v8173.2 best compromise

## Boundary

- Audit success rate: 1.000
- Final-action invariant rate: 1.000
- Online LLM / rollout / real projection steps: 0 / 0 / 0

## Best Reweight Vector

- Factor vector: canopy_lt0=1.0|canopy_lt1=1.0|dew=1.0|dry_vent=1.0|dry=1.0
- Conservative risk selected rate: 0.540
- Family / mixed combo / single spread: 0.641 / 1.000 / 0.375
- Dominant component after reweight: raw_energy_proxy

## Decision

- Next action: v8173_energy_proxy_formula_redesign_plan
- Readiness reasons: spread_still_high_and_energy_proxy_still_dominant
