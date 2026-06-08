# C-STCC v8172 Risk-Aligned Bonus Calibration

Existing-trace counterfactual scorer audit only. No online LLM, no rollout, no final-action change.

## Boundary

- Audit success rate: 1.000
- Final-action invariant rate: 1.000
- Online LLM / rollout / real projection steps: 0 / 0 / 0
- Final action changed steps: 0

## Calibration

- Risk rows: 126
- Conservative risk candidate / feasible steps: 126 / 126
- Conservative risk margin p50/p75/p90/p95/max: 0.079750 / 0.103333 / 0.105208 / 0.107083 / 0.110458
- Dominant component gap: raw_energy_proxy
- Best 0.06-0.10 bonus selected rate: 0.611 at bonus 0.10
- Nonrisk conservative selected count max: 0
- Risk-type switch spread: 0.804 at bonus 0.06

## Decision

- Next action: v8173_risk_type_specific_bonus_or_template_plan
- Readiness reasons: risk_type_switch_divergence_sharp
