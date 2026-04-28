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
