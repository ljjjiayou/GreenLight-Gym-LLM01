# Frozen Benchmark Pilot Results

Date: 2026-04-30

## Purpose

This pilot validates the frozen benchmark workflow before expanding the
experiment grid. The goal is reproducible evaluation, not controller
improvement.

Protocol:

1. Run a 24-step canary with `record -> strict replay -> check-replay`.
2. Run two 240-step representative shards.
3. Require strict cache replay, record/replay equality, and MC-SERO shadow
   invariance before treating outputs as usable.

## Files

- Canary cache: `gl_gym/result/plan_cache/canary_qwen.json`
- Pilot cache: `gl_gym/result/plan_cache/pilot_2x240_qwen.json`
- Pilot outputs: `gl_gym/result/benchmarks/pilot_2x240/`

Result files are under `gl_gym/result/` and remain ignored by git.

## Canary: 2020 Day 240 Seed 42, 24 Steps

Outcome: passed.

- `strict_replay.ok = true`
- `record_replay.ok = true`
- `shadow_invariance.ok = true`
- Cache entries: 2
- MC-SERO shadow available steps: 24
- MC-SERO would-select steps: 11

Main aggregate:

| controller | reward | profit | heat cost | CO2 cost | elec cost | RH violation | temp violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm record | 15.0324 | -0.0071 | 0.0073 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| llm replay | 15.0324 | -0.0071 | 0.0073 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| llm_sero_shadow replay | 15.0324 | -0.0071 | 0.0073 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Pilot Shard 1: 2020 Day 240 Seed 42, 240 Steps

Outcome: passed.

- `strict_replay.ok = true`
- `record_replay.ok = true`
- `shadow_invariance.ok = true`
- Cache entries for this scenario: 24
- Replay cache hits: 240/240
- MC-SERO shadow available steps: 240
- MC-SERO would-select steps: 105

Main aggregate:

| controller | reward | profit | heat cost | CO2 cost | elec cost | RH violation | temp violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm record | 151.6291 | -0.0441 | 0.0564 | 0.0023 | 0.0017 | 0.5029 | 0.0000 |
| llm replay | 151.6291 | -0.0441 | 0.0564 | 0.0023 | 0.0017 | 0.5029 | 0.0000 |
| llm_sero_shadow replay | 151.6291 | -0.0441 | 0.0564 | 0.0023 | 0.0017 | 0.5029 | 0.0000 |

MC-SERO best-candidate counts:

| candidate | steps |
| --- | ---: |
| heat_buffer | 77 |
| dry_recovery | 58 |
| economy_hold | 46 |
| rule | 30 |
| safe_dehumidify | 7 |
| anchor | 6 |
| economic_dehumidify | 6 |
| emergency_dehumidify | 5 |
| rule_lean_blend | 5 |

## Pilot Shard 2: 2015 Day 120 Seed 43, 240 Steps

Outcome: passed as a benchmark shard, but it exposes weak LLM-RSPC control.

- `strict_replay.ok = true`
- `record_replay.ok = true`
- `shadow_invariance.ok = true`
- Cache entries for this scenario: 20
- Replay cache hits: 240/240
- MC-SERO shadow available steps: 240
- MC-SERO would-select steps: 102

Main aggregate:

| controller | reward | profit | heat cost | CO2 cost | elec cost | RH violation | temp violation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm record | 137.7791 | -0.0851 | 0.1019 | 0.0024 | 0.0000 | 170.7133 | 7.1855 |
| llm replay | 137.7791 | -0.0851 | 0.1019 | 0.0024 | 0.0000 | 170.7133 | 7.1855 |
| llm_sero_shadow replay | 137.7791 | -0.0851 | 0.1019 | 0.0024 | 0.0000 | 170.7133 | 7.1855 |

MC-SERO best-candidate counts:

| candidate | steps |
| --- | ---: |
| economy_hold | 119 |
| dry_recovery | 58 |
| heat_buffer | 29 |
| anchor | 22 |
| rule | 7 |
| economic_dehumidify | 3 |
| safe_dehumidify | 2 |

## Interpretation

The frozen benchmark protocol is now usable on small shards:

- Replay can be made fully deterministic under cached LLM plans.
- MC-SERO shadow does not change LLM-RSPC behavior.
- MC-SERO produces non-trivial diagnostic alternatives in both 240-step shards.

The second shard shows that the next control improvement should focus on the
RSPC humidity and low-temperature recovery logic itself. The observed weakness
is not caused by LLM randomness or HEM. It appears in a frozen replay with
identical high-level plans.

## Next Decision

Do not expand to 36 scenarios yet.

Next recommended step:

1. Run PPO on the same two pilot scenarios as a diagnostic baseline.
2. Add a paired failure extractor for frozen LLM-RSPC outputs:
   - high RH area;
   - low temperature area;
   - heat plus vent conflict;
   - dawn or night recovery failure.
3. Use those extracted failure windows to redesign the RSPC humidity and
   temperature recovery candidate logic.

