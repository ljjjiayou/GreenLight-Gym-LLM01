# C-STCC v80 Contract And Shadow Audit

## Definition

C-STCC is a calibrated hybrid temporal control framework. The LLM does not
directly generate actuator actions. It provides semantic regime, risk-factor,
and priority suggestions; those suggestions must pass through semantic
hysteresis, evidence confidence checks, weight governance, prior credibility
estimation, finite sequence feasibility checks, level-0 scoring placeholders,
and Tomato Safety dual-score audit records before they can indirectly affect a
future control decision.

In v80, C-STCC does not replace the existing controller, does not execute online
control optimization, does not call an online LLM, and does not change the final
runtime action. v80 only fixes contracts, configuration, documentation, tests,
candidate sequence descriptions, and audit record shape for future v81 shadow
runtime instrumentation.

## Boundary

- `controller_changed=false`
- `default_controller_changed=false`
- `final_action_changed=false`
- `online_llm_enabled=false`
- `prediction_level=0`
- `score_validity=sequence_only_no_rollout`
- no performance or safety improvement claim is allowed from v80 evidence

## Contract Set

1. `SemanticSuggestion`
   - Allowed regimes are fixed to seven finite state-machine values.
   - LLM-reported confidence is recorded but not blindly trusted.
   - Invalid regimes must be rejected or retained as `invalid_regime`.

2. `CalibratedSemanticState`
   - Records active regime, age, effective confidence, calibrated weights,
     tightened constraints, and LLM influence alpha.

3. `TemporalContext`
   - Stores trends, normalized debt features, action smoothness, safety pressure,
     solver risk, and data-quality features.
   - Debt metrics are accumulated in real time units, then normalized and
     saturated.

4. `PriorCandidate`
   - Stores source, base action, confidence, confidence reasons, and candidate
     count.
   - Confidence affects the number of generated sequence candidates.

5. `SequenceCandidate`
   - Stores raw sequence, feasibility status, violation reasons, projected
     sequence, and soft penalties.

6. `SafetyProjectionReport`
   - Records rewrite magnitude, rewrite reasons, severity, severe count, and
     projected-safe status.

7. `ScoreBreakdown`
   - Separates `raw_score`, `projected_score`, `rewrite_penalty`, and
     `final_score`.
   - v80 score is explicitly sequence-only; it is not future-state rollout
     scoring.

8. `CSTCCAuditRecord`
   - The central v80 artifact for v81 runtime shadow instrumentation.
   - It records semantic suggestion, state-machine result, calibrated weights,
     prior confidence, sequence candidates, hard and soft checks, dual scores,
     selected shadow sequence, runtime final action, shadow difference, fallback,
     and boundary flags.

## Semantic State Machine

v80 fixes the regime set:

- `NORMAL_BALANCED`
- `HEAT_ACCUMULATION`
- `HIGH_VPD_DRY_STRESS`
- `HUMIDITY_EXCESS_DISEASE_RISK`
- `LOW_LIGHT_ENERGY_SAVING`
- `CO2_VENTILATION_CONFLICT`
- `SOLVER_SENSITIVE_EMERGENCY`

Each regime has entry conditions, exit conditions, minimum hold steps, allowed
transitions, default weights, tightened constraints, preferred priors, and
forbidden conflicts. v80 does not allow arbitrary new LLM-generated regime
strings.

## Confidence Governance

LLM confidence is split into two parts:

```text
llm_reported_confidence
evidence_confidence
```

v80 uses min-mode by default:

```text
effective_confidence = min(llm_reported_confidence, evidence_confidence)
```

The configured alternative is weighted mode:

```text
effective_confidence =
  0.4 * llm_reported_confidence
+ 0.6 * evidence_confidence
```

Weight influence is bounded:

```text
alpha = clip(alpha_max * effective_confidence * data_quality * regime_stability, 0, alpha_max)
```

v80.1 hardens this by making evidence confidence regime-specific. A high generic
risk score is not enough to support an unrelated LLM regime suggestion. For
example, `HIGH_VPD_DRY_STRESS` evidence checks VPD slope, RH decline, VPD debt,
and dryness debt, while `CO2_VENTILATION_CONFLICT` checks CO2 and ventilation
conflict evidence.

## Sequence Templates

v80 implements a minimal template library:

- `hold_all`
- `previous_shift_all`
- `ventilation_ramp_limited`
- `heating_smooth_recover`
- `co2_vent_conflict_reduce`
- `screen_dwell_hold`
- `ppo_follow_limited`
- `conservative_stabilize`

These templates are finite, deterministic, and actuator-aware enough for v80
audit tests without pretending to solve continuous horizon optimization.

## Level-0 Scoring

v80 uses level-0 sequence-only feasibility and audit scoring. It evaluates
smoothness, reversal count, energy proxy, hard violations, soft penalties, and
Tomato Safety dual-score shape.

v80 does not evaluate future environment trajectories. True rollout or surrogate
prediction belongs to later stages:

- Level 1: simulator shadow rollout
- Level 2: fast surrogate rollout
- Level 3: uncertainty-aware surrogate
- Level 4: MPC-like optimization

The v80 Tomato Safety projection record is a placeholder shape check unless the
record explicitly says `projection_level=runtime_tomato_safety_projection`.
`placeholder_no_real_projection` must not be read as a real safety guarantee.

All selected action fields are shadow-only in v80. Runtime integration must use
`shadow_selected_first_action` and must keep `final_action_changed=false`.

## Falsifiable Later Hypothesis

Compared with step-local LLM control and LLM plus a rate limiter, future C-STCC
stages should reduce action total variation, reversal count, Tomato Safety
rewrite pressure, and solver-sensitive events while keeping environment tracking
and cumulative reward within an acceptable degradation band. v80 does not test
or claim this result.
