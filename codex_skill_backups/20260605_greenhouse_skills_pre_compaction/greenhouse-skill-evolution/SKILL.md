---
name: greenhouse-skill-evolution
description: "Maintain greenhouse skills as compact research-navigation and engineering-guardrail documents, including repo/local sync and anti-stagnation rules."
---

# Greenhouse Skill Evolution

Use this skill when the user asks to optimize, delete, synchronize, or rewrite
greenhouse skills.

Skills are navigation aids and reusable judgment records. They should not become
long experiment recaps, frozen architecture doctrine, or repeated prohibition
lists.

## Maintenance Goals

- Keep hard safety, evidence integrity, and promotion boundaries clear.
- Allow architecture innovation through explicit hypothesis revision.
- Allow online LLM experiments through explicit authorization protocols.
- Keep skills compact enough to guide action instead of slowing every task.
- Record reusable workflows, failure categories, and evidence ladders.
- Do not store secrets, one-off authorization text, stale vXX details, or temporary metrics.

## Repo And Local Sync

1. The repo copy under `codex_skills/` is authoritative.
2. The local installed copy under `.codex/skills/` must match the repo copy.
3. After edits, sync repo to local and run:

```text
python scripts/check_greenhouse_skills.py
```

## Anti-Stagnation Rules

- If a skill mostly makes progress slower without adding judgment, compress it to hard invariants and go-rules.
- Do not turn a current mainline into a permanent architecture constraint.
- Every stop rule should have a recovery path, such as instrumentation repair, cache acquisition, hypothesis revision, or canary admission.
- When repeated failures occur, preserve the reusable pattern, not the whole historical audit.
- If online LLM is needed for new evidence, describe authorization requirements instead of banning the experiment.

## Required Skill Roles

- `greenhouse-mainline-guard`: hard invariants, evidence ladder, and hypothesis revision.
- `greenhouse-shadow-audit-runner`: experiment levels, cache/replay, online LLM authorization, action invariance.
- `greenhouse-safety-boundary`: hard safety and provenance boundaries.
- `greenhouse-research-to-implementation`: research ideas, architecture exploration, and testable plans.
- `greenhouse-skill-evolution`: this maintenance and synchronization guide.

## Quality Checks For Future Skill Changes

A good greenhouse skill should answer:

- What must never be faked or silently changed?
- What is merely the current working hypothesis?
- What exploration is allowed?
- What evidence level is required for the next step?
- If the task is blocked, what is the smallest recovery path?
