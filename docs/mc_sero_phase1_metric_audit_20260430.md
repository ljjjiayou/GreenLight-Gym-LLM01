# MC-SERO Phase 1 Metric Audit 2026-04-30

## Current Version

Before this phase, the project was at:

- Branch: `codex/literature-method-backup`
- Commit: `686b138 Add RSPC dry recovery safeguards`
- GitHub backup branch: `codex/pre-mc-sero-backup-20260430`

This backup branch points to the pre-MC-SERO state and does not overwrite older backups.

## Expert Discussion

Two expert tracks reviewed the next step before code changes.

### Expert A: Greenhouse Mechanism and MPC

Recommendation:

- Do not start by adding a closed-loop optimizer.
- First add mechanism-level diagnostics so RH failures can be split into high RH, low RH, VPD, dew risk, dry ventilation risk, and cold ventilation risk.
- A short-horizon optimizer can improve generalization only if it uses mechanism variables, not scenario identifiers or hand-tuned day-specific rules.

Key constraints:

- Keep `T < 12` cold protection intact.
- Keep extreme RH/dew-risk dehumidification intact.
- Keep dry-side recovery guardrails intact.
- Do not put MC-SERO options into the plan cache key; cache should freeze only high-level LLM plans.

### Expert B: RL Generalization and Benchmarking

Recommendation:

- PPO/SAC should be proposal sources, not teachers.
- The first phase should be `off/shadow/select` compatible, with `shadow` before `select`.
- Before any PPO/SAC proposal can affect actions, every proposal needs metadata: policy id, reward version, split id, OOD/trust diagnostics, and rejection reasons.
- Strong PPO/SAC baselines require reward-version audit and validation/test separation.

## Implemented Slice

This phase intentionally does not change controller behavior.

Changed files:

- `gl_gym/experiments/diagnose_ppo_vs_llm.py`
- `tests/test_frozen_benchmark.py`

Added per-step derived metrics:

- `vpd_air`
- `vpd_kpa`
- `rh_low_violation`
- `rh_high_violation`
- `vpd_low_excess`
- `vpd_high_excess`
- `dew_margin_min`
- `dry_risk`
- `dew_risk`
- `cold_vent_risk`
- `dry_vent_risk`
- `dry_vent_cap`

Added aggregate metrics:

- `mean_vpd`
- `total_rh_low_violation`
- `total_rh_high_violation`
- `total_vpd_low_excess`
- `total_vpd_high_excess`
- `dry_risk_steps`
- `dew_risk_steps`
- `cold_vent_risk_steps`
- `dry_vent_risk_steps`
- `source_counts`

Added safety-pattern metrics:

- `state_rh_lt_50_steps`
- `state_vpd_gt_1p2_steps`
- `dry_risk_steps`
- `dew_risk_steps`
- `cold_vent_risk_steps`
- `dry_vent_risk_steps`
- `mean_vent_when_dry_risk`

## Verification

Static checks:

```powershell
python -m py_compile gl_gym\experiments\diagnose_ppo_vs_llm.py gl_gym\experiments\run_frozen_benchmark.py
```

Unit tests:

```powershell
python -m pytest tests\test_frozen_benchmark.py tests\test_planning_extensions.py tests\test_plan_cache.py tests\test_humidity_experience_memory.py tests\test_plan_intent_distillation.py tests\test_ppo_strategy_labeler.py -q
```

Result:

- `46 passed`

Frozen replay smoke:

```powershell
python gl_gym\experiments\run_frozen_benchmark.py --years 2020 --days 240 --seeds 42 --controllers llm --max-steps 24 --plan-cache-mode replay --plan-cache-strict --plan-cache-key-policy scenario_timestep --plan-cache-path gl_gym/result/plan_cache/smoke_2x240_qwen.json --output-json gl_gym/result/benchmarks/mc_sero_metric_smoke_20260430.json
```

Observed summary:

- `cacheHits=24`
- `sourceCounts={"anchor":17,"nominal_blend":1,"rule":5,"rule_lean_blend":1}`
- New RH/VPD/risk metrics were present in the benchmark summary.

## Next Step

The next code slice should add MC-SERO in `shadow` mode only:

- Build mechanism candidates.
- Score candidates over a deterministic short horizon.
- Record selected/shadow-selected source, risk terms, and reject reasons.
- Do not change applied actions until shadow diagnostics are stable across a 36-scenario frozen benchmark.
