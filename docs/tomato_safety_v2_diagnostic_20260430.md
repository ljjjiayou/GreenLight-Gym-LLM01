# Tomato Safety v2 Diagnostic Layer

Date: 2026-04-30

## Purpose

This phase responds to the pilot failure pattern in
`2015/day120/seed43`: low RH / high VPD stress plus local cold recovery
problems. The goal is not to add another reward-optimizing module immediately.
Instead, this change adds better trace visibility and an opt-in conservative
RSPC safety layer for tomato climate control.

## Scope

Implemented components:

- `--output-trace-dir` in the frozen benchmark runner.
- `failure_window_extractor.py` for dry stress, cold recovery, target-action
  mismatch, cold ventilation risk, and MC-SERO shadow opportunity windows.
- `llm_rspc_v2` benchmark controller label.
- `AgentConfig.tomato_safety_v2_enabled=False` as the default-off gate.
- Tomato Safety v2 diagnostics in per-step traces and aggregate summaries.

Preserved components:

- `llm` baseline behavior stays unchanged.
- `llm_sero_shadow` remains a diagnostic-only controller.
- HEM remains disabled unless explicitly requested.
- PPO remains a diagnostic mirror, not a teacher policy.

## Safety Logic

Tomato Safety v2 only runs when `tomato_safety_v2_enabled=True`.

Dry-side protection:

- If `RH < 55` or `VPD > 1.2`, cap ventilation, disable CO2 and lamp, and add
  screen/shade when appropriate.
- If `RH < 50` or `VPD > 1.6`, use stronger caps.
- If `target_rh - RH > 10`, treat high ventilation as target-action mismatch
  and cap it unless the crop is in high-temperature cooling.
- In hot-dry, high-radiation windows, use a separate cooling guard: keep heat,
  CO2, and lamp off; raise ventilation and screen; do not force shade. This
  follows the pilot trace diagnosis where PPO kept RH safer with high
  ventilation plus closed screen.
- Low canopy dew margin alone does not block hot-dry cooling when air RH is not
  high. The block is reserved for high-RH condensation risk.

Cold recovery:

- If `temp_air < 15.5` and there is no extreme dew risk, reduce ventilation,
  disable CO2/lamp, increase screen, and add light heating.
- If `temp_air < 12`, apply a stronger cold buffer.
- Extreme dew risk is exempted so that true condensation emergencies are not
  accidentally blocked.

## Diagnostics

Per-step traces include:

- climate state: `temp_air`, `rh_air`, `vpd_kpa`, target RH;
- control action and source;
- reward, safety violations, and safety pattern flags;
- plan-cache hit/miss information;
- MC-SERO shadow fields;
- Tomato Safety v2 enabled/applied flags, reasons, and before/after heat/vent.

The failure window extractor records each window with start/end step, duration,
min/max climate values, violation areas, action means, source counts, MC-SERO
candidate counts, v2 reason counts, and short context before/after the window.

## Evaluation Rule

This phase should be judged by frozen replay first:

- `llm` must remain bit-for-bit comparable with the existing replay baseline.
- `llm_rspc_v2` should reduce low RH / high VPD stress in
  `2015/day120/seed43`.
- `2020/day240/seed42` should not be harmed.
- If reward/profit worsens by more than 5%, the safety caps should be tuned
  before any larger benchmark.

## Pilot Validation

All runs used strict `scenario_timestep` plan-cache replay with the existing
`pilot_2x240_qwen.json` cache. Result JSON, traces, protocol checks, and failure
windows are under `gl_gym/result/` and are not committed.

Canary `2020/day240/seed42`, 24 steps:

- `llm`, `llm_rspc_v2`, and `llm_sero_shadow` all had identical reward/profit
  and 24/24 cache hits.
- Tomato Safety v2 did not trigger in this safe short window.

Pilot `2015/day120/seed43`, 240 steps:

| controller | reward | profit | RH low | VPD high | temp viol | cache hits | v2 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm` | 137.7791 | -0.0851 | 170.7133 | 21.3516 | 7.1855 | 240 | 0 |
| `llm_rspc_v2` | 148.9268 | -0.0859 | 4.5938 | 8.4842 | 5.4667 | 240 | 61 |
| `llm_sero_shadow` | 137.7791 | -0.0851 | 170.7133 | 21.3516 | 7.1855 | 240 | 0 |
| `ppo` | 152.0762 | -0.0147 | 0.0000 | 6.5975 | 8.3291 | n/a | n/a |

Pilot `2020/day240/seed42`, 240 steps:

| controller | reward | profit | RH low | VPD high | RH total | temp viol | cache hits | v2 steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm` | 151.6291 | -0.0441 | 0.0000 | 0.6746 | 0.5029 | 0.0000 | 240 | 0 |
| `llm_rspc_v2` | 151.6899 | -0.0428 | 0.0000 | 0.4196 | 0.6025 | 0.0000 | 240 | 44 |
| `llm_sero_shadow` | 151.6291 | -0.0441 | 0.0000 | 0.6746 | 0.5029 | 0.0000 | 240 | 0 |
| `ppo` | 151.9194 | -0.0349 | 0.0000 | 0.0000 | 2.9681 | 0.0000 | n/a | n/a |

Acceptance status:

- Strict replay fairness passed: `llm_rspc_v2` cache hits were 240/240 in both
  pilot scenarios.
- `2015/day120/seed43` passed the safety targets: RH low decreased by 97.3%,
  VPD high decreased by 60.3%, and temp violation decreased.
- `2020/day240/seed42` stayed safe: no temp violation, no RH-low violation,
  improved reward/profit, and VPD high decreased. RH total rose slightly but
  remained very small.
- PPO remains a diagnostic reference, not a teacher policy.
