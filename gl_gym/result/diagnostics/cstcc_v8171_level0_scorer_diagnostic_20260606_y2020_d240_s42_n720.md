# C-STCC v817.0 Level-0 Scorer Diagnostic

Offline scorer diagnostic only. No online LLM, no predictive rollout, no final-action change, no reward claim.

## Boundary

- Audit success rate: 1.000
- Final-action invariant rate: 1.000
- Final action changed steps: 0
- Online LLM / rollout / real projection steps: 0 / 0 / 0
- JSON row max KB: 45.807

## Scorer Diagnostic

- Risk rows: 126
- Risk rule margin coverage: 1.000
- Risk rule margin mean/max: 0.036931 / 0.071708
- Conservative no-candidate count: 594
- Conservative risk candidate / feasible / no-candidate steps: 126 / 126 / 0
- Conservative normal candidate steps: 0
- Conservative floor applied count: 126
- Conservative risk margin coverage/mean: 1.000 / 0.081027
- Conservative no-candidate reasons: `{"confidence_below_candidate_threshold": 594}`
- Rule best templates: `{"co2_vent_conflict_reduce": 101, "hold_all": 594, "ventilation_ramp_limited": 25}`

## Decision

- Next action: v8172_risk_aligned_bonus_calibration_plan
- Readiness reasons: conservative_risk_margin_large
