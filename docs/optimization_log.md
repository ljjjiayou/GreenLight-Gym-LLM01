# LLM-RSPC Optimization Log

This file records method-level changes for reproducible experiments and paper
writing. Each entry should state the motivation, project change, expected
experiment, and residual risk.

## 2026-04-28: Expert-Distilled Safe Rollout, phase 1

### Motivation

The current LLM-RSPC controller has strong safety structure but still relies on
hand-written low-level rollout rules. The existing PPO policy performs better in
some short-horizon scenarios, especially in profit and energy tradeoffs, but a
direct PPO controller is hard to explain and hard to constrain. Inspired by
greenhouse RL+MPC, robust RL, and automated greenhouse decision-system papers,
the next optimization is to distill PPO's short-term control habit into a small
expert module and use it only as a candidate inside the existing safe rollout
pool.

### Code changes

- Added `gl_gym/agent/expert_distillation.py`.
  - Defines a shared physics-aware feature vector for both trajectory export and
    deployment.
  - Features include climate state, actuator state, weather forecast summary,
    VPD/dew-risk features, cyclic time encoding, RH violation debt, lamp budget,
    and setpoint-tracking errors.
  - Defines `DistilledExpertPolicy`, a normalized ridge expert that predicts raw
    control actions in `[0, 1]`.

- Added `gl_gym/experiments/export_ppo_trajectories.py`.
  - Runs the existing PPO model on configurable year/day/seed windows.
  - Saves JSONL rows for audit and compressed NPZ matrices for training.
  - Records state features, PPO action command, planned raw control, applied raw
    control, reward, and constraint/cost metrics.

- Added `gl_gym/experiments/train_distilled_expert.py`.
  - Trains a compact normalized ridge regressor from exported PPO trajectories.
  - Saves validation MAE/MSE and feature-distance statistics for out-of-distribution
    filtering.

- Updated `gl_gym/agent/llm_agent.py`.
  - Added optional `expert_rollout_enabled` config.
  - Loads the distilled expert only when explicitly enabled.
  - Adds `distilled_expert`, `expert_anchor_blend`, and `expert_rule_blend` to the
    rollout candidate pool.
  - Keeps the existing candidate scoring and safety guardrails as the final
    authority.
  - Records `expert_prediction` in rollout diagnostics.
  - Uses the expert model's feature-distance statistics as an out-of-distribution
    gate by default, instead of accepting all expert predictions.

- Updated `gl_gym/experiments/compare_ppo_rule_based.py`.
  - Added CLI switches to enable the distilled expert during LLM-RSPC evaluation.

### Expected experiment

1. Export PPO expert trajectories on several training windows:

   ```powershell
   python gl_gym/experiments/export_ppo_trajectories.py --years 2018,2019,2020 --days 120,180,240 --max-steps 240
   ```

2. Train the distilled expert:

   ```powershell
   python gl_gym/experiments/train_distilled_expert.py
   ```

3. Evaluate ablations:

   - PPO baseline.
   - LLM-RSPC without expert.
   - LLM-RSPC + Expert-Distilled Safe Rollout.
   - Rule-based baseline.

### Residual risk

- The current PPO model is not yet a paper-grade tuned PPO baseline. It should be
  treated as a first expert source, not a final fair baseline.
- The distilled expert may overfit if trained on too few weather windows. The
  next phase should export trajectories across more years, days, random seeds,
  and uncertainty scales.
- Because the expert is used only as a candidate, it may be rejected by the
  safety score when it conflicts with the current setpoint contract. This is
  intentional and should be reported as a safety design choice.
- A first 240-step closed-loop test on 2020/day240 did not improve over the
  previous LLM-RSPC controller. Diagnostics showed that expert predictions were
  accepted even when feature distance exceeded the training distribution, so the
  OOD gate was tightened to use model metadata.

### Paper angle

This change supports the method claim:

> LLM-RSPC separates high-level explainable setpoint planning from low-level
> execution. PPO knowledge is distilled into a lightweight rollout candidate,
> while rule contracts, candidate scoring, and guardrails preserve safety and
> interpretability.

## 2026-04-28: PPO vs LLM-RSPC diagnostic tooling

### Motivation

Manual threshold tuning is no longer the best next step. The controller needs a
repeatable way to answer why PPO has better profit while LLM-RSPC often has
better humidity safety. The diagnostic tool therefore records matched PPO and
LLM-RSPC trajectories and groups action/cost/safety differences by greenhouse
regime.

### Code changes

- Added `gl_gym/experiments/diagnose_ppo_vs_llm.py`.
  - Runs PPO and LLM-RSPC on the same year/day/seed window.
  - Saves per-step JSON and CSV trajectories.
  - Summarizes aggregate reward/profit/cost/violation metrics.
  - Computes per-action PPO-vs-LLM differences.
  - Groups results by phase, RH band, temperature band, and radiation band.
  - Counts safety/cost patterns such as heat+vent conflict, CO2 leakage during
    ventilation, risky lamp usage, and RH>=90 states.

### First smoke result

Scenario: 2020/day240, seed 42, 24 steps.

- PPO reward: 15.063; profit: -0.00453; RH violation: 1.467.
- LLM-RSPC reward: 15.032; profit: -0.00712; RH violation: 0.000.
- LLM-RSPC is safer on RH in this short window, but it spends more heat.
- Mean absolute action gaps are largest for screen and ventilation.
- Heat+vent conflict steps:
  - PPO: 1.
  - LLM-RSPC: 8.

### Next hypothesis

The next controller improvement should focus on dehumidification cost gating:
LLM-RSPC already controls RH aggressively, but it needs a duty-cycle or marginal
benefit gate for heat+vent pulses. A publishable framing is:

> humidity-risk-aware economic pulse gating for safe rollout control.

## 2026-04-28: PPO strategy-label audit for safe distillation

### Motivation

Using PPO as an expert source is only useful if its continuous actions can be
translated into stable, auditable strategy tendencies. The previous trajectory
export and ridge distillation can copy actions numerically, but that is not
enough for a paper-grade greenhouse controller because a mixed action such as
heating plus ventilation may mean dehumidification, wasteful conflict, or a
transition artifact depending on RH risk and crop-climate context.

This step therefore adds a conservative strategy-label layer before any further
PPO-to-LLM-RSPC distillation. PPO is treated as a behavior reference and
diagnostic mirror, not as a ground-truth controller.

### Code changes

- Added `gl_gym/agent/ppo_strategy_labeler.py`.
  - Scores each continuous action against interpretable strategy candidates:
    economy hold, vent-only dehumidification, heat+vent dehumidification,
    screen-release dehumidification, heat preservation, free air exchange,
    cooling ventilation, CO2 enrichment, lighting assist, and shade cooling.
  - Uses action values together with climate features such as RH, temperature,
    VPD, dew-margin risk, radiation, and time of day.
  - Marks a sample as `unknown` when the best score is too weak and
    `ambiguous` when the top two strategies are too close.
  - Exports the full soft score vector and reason string for later auditing.

- Updated `gl_gym/experiments/diagnose_ppo_vs_llm.py`.
  - Adds strategy labels to both PPO and LLM-RSPC trace rows.
  - Adds a label audit block with confident/ambiguous/unknown counts.
  - Adds a useful-PPO-tendency filter: high-confidence PPO labels are counted
    as useful only when PPO has same-step profit advantage without increasing
    same-step RH or temperature violation beyond a small tolerance.

- Added `tests/test_ppo_strategy_labeler.py`.
  - Covers humidity-risk-aware dehumidification labels, heat preservation,
    free-air exchange, and conservative rejection of weak mixed actions.

### First audit result

Scenario: 2020/day240, seed 42, 24 steps.

- PPO reward: 15.063; profit: -0.00453; RH violation: 1.467.
- LLM-RSPC reward: 15.032; profit: -0.00712; RH violation: 0.000.
- Heat+vent conflict steps:
  - PPO: 1.
  - LLM-RSPC: 8.
- Strategy label audit:
  - PPO: 8 confident `free_air_exchange` samples, 1 ambiguous sample, 15
    unknown samples.
  - LLM-RSPC: 14 confident `economy_hold` samples, 1 confident
    `free_air_exchange` sample, 9 unknown samples.
- Useful PPO tendency candidates:
  - 8 high-confidence useful PPO samples.
  - All 8 are `free_air_exchange`.
  - No high-confidence PPO samples were rejected by the same-step safety filter
    in this short window.

### Interpretation

The first label audit supports the current optimization direction. In this
window, LLM-RSPC is safer on RH but pays extra heat cost and triggers more
heat+vent conflict. PPO's useful tendency is not "stronger dehumidification";
it is low-cost air exchange under moderate humidity risk: high ventilation,
open screen, low heating, and no CO2 or lamp use.

The next controller change should therefore be a dehumidification economic gate:

- when RH/dew risk is severe, keep the existing safe dehumidification behavior;
- when RH risk is moderate and temperature is not too low, prefer a bounded
  free-air-exchange candidate before heat+vent pulses;
- when the action is ambiguous or out of the audited region, fall back to the
  existing rule and guardrail stack.

This gives a clearer paper claim than direct PPO cloning:

> LLM-RSPC mines PPO trajectories for interpretable, confidence-gated strategy
> tendencies, then injects only safety-filtered economic behaviors into a
> rule-constrained rollout controller.
