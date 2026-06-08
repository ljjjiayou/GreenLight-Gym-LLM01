# C-STCC v81.1 Smoke Acquisition

## Scope

Short opt-in runtime shadow trace. No online LLM, no predictive rollout, no final-action change, no reward claim.

## Scenario

- Stage label: scorer_diagnostic_trace
- Year/day/seed: 2020/240/42
- Max steps: 720
- Log root: `logs/cstcc_shadow/v8170_scorer_diagnostic_20260606_y2020_d240_s42_n720`

## Six Questions

1. JSONL generated: True
2. Audit success rate: 1.000
3. Final-action invariant rate: 1.000
4. Audit latency mean/p95 ms: 15.746 / 22.735
5. JSON row size mean/max KB: 35.820 / 38.783
6. Fallback rate: 0.000

## Boundary Checks

- Online LLM called steps: 0
- Predictive rollout steps: 0
- Real Tomato Safety projection steps: 0
- Final action changed steps: 0

## Candidate Health

- Mean candidate count: 8.072
- Mean feasible candidate count: 8.072
- Mean infeasible candidate ratio: 0.000
- Hard constraint violation count: 0
- Selected source prior distribution: `{"ppo_prior": 1, "previous_plan_prior": 719}`
- Selected template distribution: `{"hold_all": 89, "previous_shift_all": 622, "screen_dwell_hold": 9}`
- Hard constraint reason distribution: `{}`
- Hard constraint field distribution: `{}`

## Shadow/Runtime Action Difference

| Field | Mean | P95 | Max |
|---|---:|---:|---:|
| u_heating | 0.0568 | 0.2000 | 0.2000 |
| u_co2 | 0.0000 | 0.0000 | 0.0000 |
| u_screen | 0.0327 | 0.2000 | 0.2000 |
| u_ventilation | 0.0643 | 0.2000 | 0.2000 |
| u_lighting | 0.0000 | 0.0000 | 0.0000 |
| u_shading | 0.0000 | 0.0000 | 0.0000 |

## Decision

- Blockers: none
- Diagnostic notes: none
- Next action: v815_shadow_analytics

Reward and controller-quality comparisons are intentionally out of scope for v81.1.
