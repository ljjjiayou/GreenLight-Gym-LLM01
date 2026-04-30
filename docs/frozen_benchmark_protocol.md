# Frozen Benchmark Protocol

This protocol prevents long LLM record runs from producing partial caches that
look usable but fail strict replay later.

## Current Rule

Do not run the 36-scenario benchmark as one long `record` job.

Use single-scenario or small-batch shards:

1. Record one scenario with `controllers=llm`.
2. Immediately replay the same scenario with `controllers=llm,llm_sero_shadow`.
3. Accept the shard only if strict replay has no cache misses and shadow does
   not change reward, cost, safety, or action aggregates.
4. Only merge accepted shards into a formal benchmark.

## Tools

Audit a cache:

```powershell
python gl_gym\experiments\frozen_benchmark_protocol.py audit-cache `
  --years 2010,2015,2020 `
  --days 59,120,180,240 `
  --seeds 42,43,44 `
  --max-steps 240 `
  --cache-path gl_gym/result/plan_cache/frozen_36x240_qwen_mc_sero_shadow.json `
  --min-entries-per-env 20
```

Generate a manifest:

```powershell
python gl_gym\experiments\frozen_benchmark_protocol.py init-manifest `
  --years 2020 `
  --days 240 `
  --seeds 42 `
  --max-steps 240 `
  --cache-path gl_gym/result/plan_cache/canary_qwen.json `
  --commit b146719 `
  --output-json gl_gym/result/benchmarks/canary_manifest.json
```

Print canary commands:

```powershell
python gl_gym\experiments\frozen_benchmark_protocol.py commands `
  --years 2020 `
  --days 240 `
  --seeds 42 `
  --max-steps 240 `
  --cache-path gl_gym/result/plan_cache/canary_qwen.json `
  --output-dir gl_gym/result/benchmarks/canary
```

Check replay output:

```powershell
python gl_gym\experiments\frozen_benchmark_protocol.py check-replay `
  --record-json gl_gym/result/benchmarks/canary/y2020_d240_s42_n240_record.json `
  --replay-json gl_gym/result/benchmarks/canary/y2020_d240_s42_n240_replay.json `
  --fail-on-error
```

## Acceptance

A shard is acceptable only when:

- every LLM row has plan cache enabled and hit;
- no row uses an `unknown` source caused by failed replanning;
- record and strict replay aggregates match;
- `llm_sero_shadow` and `llm` aggregates match;
- MC-SERO fields are present only as diagnostics.

HEM remains outside the formal benchmark unless explicitly marked exploratory.
