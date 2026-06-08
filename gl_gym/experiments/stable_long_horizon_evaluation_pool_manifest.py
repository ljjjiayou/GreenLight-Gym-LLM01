"""Build the v38 stable long-horizon evaluation pool manifest."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONTROLLED_USED_BASES = {
    (2015, 120, 42),
    (2015, 120, 43),
    (2015, 120, 44),
    (2020, 120, 44),
    (2010, 120, 44),
    (2010, 180, 44),
    (2015, 180, 44),
    (2020, 180, 44),
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def scenario_id(year: int, day: int, seed: int, max_steps: int = 240) -> str:
    return f"y{year}_d{day}_s{seed}_n{max_steps}"


def _parts_from_scenario_id(value: str) -> dict[str, int]:
    match = re.search(r"y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)(?:_n(?P<steps>\d+))?", str(value))
    if not match:
        return {"year": 0, "day": 0, "seed": 0, "max_steps": 240}
    return {
        "year": int(match.group("year")),
        "day": int(match.group("day")),
        "seed": int(match.group("seed")),
        "max_steps": int(match.group("steps") or 240),
    }


def _base_key(value: Mapping[str, Any] | str) -> tuple[int, int, int]:
    if isinstance(value, Mapping):
        return (int(value.get("year", 0)), int(value.get("day", 0)), int(value.get("seed", 0)))
    parts = _parts_from_scenario_id(str(value))
    return (parts["year"], parts["day"], parts["seed"])


def _scenario_from_base(
    base: tuple[int, int, int],
    *,
    source_tier: str,
    source: str,
    cache_strategy: str,
) -> dict[str, Any]:
    year, day, seed = base
    return {
        "scenario_id": scenario_id(year, day, seed, 240),
        "year": year,
        "day": day,
        "seed": seed,
        "max_steps": 240,
        "source_tier": source_tier,
        "source": source,
        "regime": f"{source_tier}_d{day}",
        "cache_strategy_h240": cache_strategy,
    }


def _v36_failed_bases(v36_diagnosis: Mapping[str, Any]) -> set[tuple[int, int, int]]:
    bases: set[tuple[int, int, int]] = set()
    for env_id, report in dict(v36_diagnosis.get("env_reports", {})).items():
        trace_path = str(report.get("trace_path", "") or "")
        bases.add(_base_key(trace_path or str(env_id).replace("TomatoEnv_", "")))
    return bases


def _v37_failed_bases(v37_manifest: Mapping[str, Any], v37_result: Mapping[str, Any]) -> set[tuple[int, int, int]]:
    bases = {_base_key(item) for item in v37_manifest.get("primary_scenarios", []) or [] if isinstance(item, Mapping)}
    for item in v37_result.get("scenario_reports", []) or []:
        if isinstance(item, Mapping) and not item.get("horizon_pass", False):
            bases.add(_base_key(item))
    return bases


def _backlog_bases(v30_manifest: Mapping[str, Any]) -> set[tuple[int, int, int]]:
    return {_base_key(item) for item in v30_manifest.get("blocked_cache_repair_backlog", []) or [] if isinstance(item, Mapping)}


def _excluded_bases(
    *,
    v30_manifest: Mapping[str, Any],
    v36_diagnosis: Mapping[str, Any],
    v37_manifest: Mapping[str, Any],
    v37_result: Mapping[str, Any],
) -> dict[tuple[int, int, int], str]:
    out: dict[tuple[int, int, int], str] = {}
    for base in CONTROLLED_USED_BASES:
        out[base] = "used_controlled_or_canary_scene"
    for base in _backlog_bases(v30_manifest):
        out[base] = "incomplete_selected_cache_backlog"
    for base in _v36_failed_bases(v36_diagnosis):
        out[base] = "v36_runtime_failed"
    for base in _v37_failed_bases(v37_manifest, v37_result):
        out[base] = "v37_runtime_or_coverage_failed"
    return out


def _tier_a_scenarios(v30_manifest: Mapping[str, Any], excluded: Mapping[tuple[int, int, int], str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in v30_manifest.get("selected_scenarios", []) or v30_manifest.get("scenarios", []) or []:
        if not isinstance(item, Mapping):
            continue
        base = _base_key(item)
        if base in excluded:
            continue
        out.append(
            _scenario_from_base(
                base,
                source_tier="tier_a_selected_cache",
                source="v30_coverage_qualified_selected_cache",
                cache_strategy="selected_cache_replay",
            )
        )
    return sorted(out, key=lambda item: (item["year"], item["day"], item["seed"]))


def _tier_b_scenarios(
    v35_manifest: Mapping[str, Any],
    excluded: Mapping[tuple[int, int, int], str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for item in v35_manifest.get("requested_scenarios", []) or []:
        if not isinstance(item, Mapping):
            continue
        base = _base_key(item)
        if base in excluded or base in seen:
            continue
        seen.add(base)
        out.append(
            _scenario_from_base(
                base,
                source_tier="tier_b_weather_source",
                source="v35_weather_candidate_after_runtime_exclusions",
                cache_strategy="isolated_record",
            )
        )
        if len(out) >= limit:
            break
    return out


def build_report(
    *,
    v30_manifest: Mapping[str, Any],
    v35_manifest: Mapping[str, Any],
    v36_diagnosis: Mapping[str, Any],
    v37_manifest: Mapping[str, Any],
    v37_result: Mapping[str, Any],
    tier_b_limit: int = 12,
) -> dict[str, Any]:
    excluded = _excluded_bases(
        v30_manifest=v30_manifest,
        v36_diagnosis=v36_diagnosis,
        v37_manifest=v37_manifest,
        v37_result=v37_result,
    )
    tier_a = _tier_a_scenarios(v30_manifest, excluded)
    tier_b = _tier_b_scenarios(v35_manifest, excluded, limit=int(tier_b_limit))
    scenarios = tier_a + tier_b
    manifest_ready = bool(
        v30_manifest.get("strict_targeted_shadow_sweep_v30_manifest_ready", False)
        and v35_manifest.get("shadow_scenario_acquisition_manifest_ready", False)
        and len(tier_a) > 0
        and len(tier_b) <= int(tier_b_limit)
    )
    return {
        "schema_version": "stable_long_horizon_evaluation_pool_manifest_20260529_v38",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "stable long-horizon pool construction",
        },
        "stable_long_horizon_manifest_ready": manifest_ready,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "controllers_allowed": ["ppo", "llm_rspc_v2"],
        "controlled_controllers_blocked": ["llm_rspc_v2_hot_dry_proposer_strict"],
        "excluded_scenarios": [
            {
                "scenario_id": scenario_id(year, day, seed, 240),
                "year": year,
                "day": day,
                "seed": seed,
                "max_steps": 240,
                "excluded_reason": reason,
            }
            for (year, day, seed), reason in sorted(excluded.items())
        ],
        "tier_a_scenarios": tier_a,
        "tier_b_scenarios": tier_b,
        "candidate_scenarios": scenarios,
        "candidate_scenario_ids": [item["scenario_id"] for item in scenarios],
        "metrics": {
            "excluded_count": len(excluded),
            "tier_a_count": len(tier_a),
            "tier_b_count": len(tier_b),
            "candidate_count": len(scenarios),
        },
        "horizon_ladder": [
            {"horizon_steps": 240, "selection": "all_candidates"},
            {"horizon_steps": 720, "selection": "previous_pass_max_2_per_regime"},
            {"horizon_steps": 1440, "selection": "previous_pass_max_6_total"},
        ],
        "next_action": "build_v38_h240_execution_record" if manifest_ready else "v38_manifest_blocked",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stable Long-Horizon Evaluation Pool Manifest v38",
        "",
        f"- Manifest ready: `{report.get('stable_long_horizon_manifest_ready', False)}`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in dict(report.get("metrics", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Candidate Scenarios", "", "| scenario | tier | cache strategy | regime |", "| --- | --- | --- | --- |"])
    for item in report.get("candidate_scenarios", []) or []:
        lines.append(
            f"| `{item.get('scenario_id', '')}` | `{item.get('source_tier', '')}` | "
            f"`{item.get('cache_strategy_h240', '')}` | `{item.get('regime', '')}` |"
        )
    lines.extend(["", "## Excluded Scenarios", "", "| scenario | reason |", "| --- | --- |"])
    for item in report.get("excluded_scenarios", []) or []:
        lines.append(f"| `{item.get('scenario_id', '')}` | `{item.get('excluded_reason', '')}` |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v30-manifest-json", required=True)
    parser.add_argument("--v35-manifest-json", required=True)
    parser.add_argument("--v36-diagnosis-json", required=True)
    parser.add_argument("--v37-manifest-json", required=True)
    parser.add_argument("--v37-result-json", required=True)
    parser.add_argument("--tier-b-limit", type=int, default=12)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        v30_manifest=_load(args.v30_manifest_json),
        v35_manifest=_load(args.v35_manifest_json),
        v36_diagnosis=_load(args.v36_diagnosis_json),
        v37_manifest=_load(args.v37_manifest_json),
        v37_result=_load(args.v37_result_json),
        tier_b_limit=args.tier_b_limit,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"stable_long_horizon_manifest_ready={report['stable_long_horizon_manifest_ready']}")
    print(f"candidate_count={report['metrics']['candidate_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["stable_long_horizon_manifest_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
