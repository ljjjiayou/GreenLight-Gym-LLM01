"""Build the v35 weather-space expansion target specification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAX_STEPS = 240
DEFAULT_SEEDS = [42, 43, 44]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _load_optional(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = _resolve(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _scenario_id(year: int, day: int, seed: int) -> str:
    return f"y{year}_d{day}_s{seed}_n{MAX_STEPS}"


def _v30_swept_scenarios() -> set[str]:
    scenarios: set[str] = set()
    for year in [2010, 2015]:
        for day in [59, 120, 180, 240]:
            for seed in DEFAULT_SEEDS:
                scenarios.add(_scenario_id(year, day, seed))
    for day in [59, 120, 180, 240]:
        scenarios.add(_scenario_id(2020, day, 44))
    return scenarios


def _incomplete_backlog() -> set[str]:
    return {_scenario_id(2020, day, seed) for day in [59, 120, 180, 240] for seed in [42, 43]}


def build_report(
    *,
    near_miss_catalog: Mapping[str, Any],
    v34_source_inventory: Mapping[str, Any],
    v34_readiness: Mapping[str, Any],
    weather_sources: Sequence[str] | None = None,
    candidate_seeds: Sequence[int] = tuple(DEFAULT_SEEDS),
    max_candidate_days: int = 12,
) -> dict[str, Any]:
    v33_ready = bool(near_miss_catalog.get("near_miss_catalog_ready", False))
    v34_exhausted = bool(
        v34_source_inventory.get("new_shadow_source_inventory_ready", False)
        and not v34_source_inventory.get("new_shadow_source_candidates_found", False)
        and v34_readiness.get("next_action") == "external_weather_or_scenario_space_expansion_design"
    )
    inherited_excluded = {str(item) for item in v34_source_inventory.get("excluded_scenario_ids", []) or [] if str(item or "")}
    inherited_excluded.update(str(item) for item in near_miss_catalog.get("excluded_scenario_ids", []) or [] if str(item or ""))
    excluded = sorted(inherited_excluded | _v30_swept_scenarios() | _incomplete_backlog())
    weather_sources = list(weather_sources or [
        "gl_gym/environments/weather/Amsterdam",
        "gl_gym/environments/weather/Bleiswijk",
    ])
    return {
        "schema_version": "controlled_canary_weather_space_expansion_target_spec_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only external weather/scenario space expansion design",
            "reopens_rejected_preset": False,
        },
        "target_spec_ready": bool(v33_ready and v34_exhausted),
        "shadow_only": True,
        "cache_fill_authorized": False,
        "online_llm_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "weather_proxy_source_hint_only": True,
        "weather_proxy_is_strict_applied_evidence": False,
        "strict_threshold_modified": False,
        "controller_logic_modified": False,
        "strict_weather_proxy_gate": {
            "candidate": "shadow_hot_dry_humidity_retention",
            "rh_max": 45.0,
            "vpd_min": 2.25,
            "temp_min": 20.0,
            "temp_max": 30.5,
            "dew_point_spread_min": 3.0,
            "global_radiation_min": 200.0,
            "unsafe_signal_allowed": False,
        },
        "candidate_seeds": [int(seed) for seed in candidate_seeds],
        "max_steps": MAX_STEPS,
        "max_candidate_days": int(max_candidate_days),
        "weather_sources": [str(item) for item in weather_sources],
        "excluded_scenario_ids": excluded,
        "excluded_groups": {
            "v31_or_wave1_or_prior_used": sorted(inherited_excluded),
            "v30_swept_subset": sorted(_v30_swept_scenarios()),
            "incomplete_cache_backlog": sorted(_incomplete_backlog()),
        },
        "v33_near_miss_scenario_count": int(near_miss_catalog.get("near_miss_scenario_count", 0) or 0),
        "v33_near_miss_row_count": int(near_miss_catalog.get("near_miss_row_count", 0) or 0),
        "v34_candidate_scenario_count": int(v34_source_inventory.get("candidate_scenario_count", 0) or 0),
        "next_action": "run_v35_read_only_weather_window_inventory",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Weather Space Expansion Target Spec v35",
        "",
        f"- Target spec ready: {report.get('target_spec_ready', False)}",
        "- Shadow only: true",
        "- Cache fill authorized: false",
        "- Online LLM allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "- Weather proxy is strict-applied evidence: false",
        f"- Excluded scenarios: {len(report.get('excluded_scenario_ids', []) or [])}",
        f"- Weather sources: {len(report.get('weather_sources', []) or [])}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Weather Proxy Gate",
        "",
        "| gate | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("strict_weather_proxy_gate", {})).items():
        lines.append(f"| {key} | `{value}` |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near-miss-catalog-json", required=True)
    parser.add_argument("--v34-source-inventory-json", required=True)
    parser.add_argument("--v34-readiness-json", required=True)
    parser.add_argument("--weather-source", action="append", default=[])
    parser.add_argument("--candidate-seed", action="append", type=int, default=[])
    parser.add_argument("--max-candidate-days", type=int, default=12)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        near_miss_catalog=_load(args.near_miss_catalog_json),
        v34_source_inventory=_load(args.v34_source_inventory_json),
        v34_readiness=_load(args.v34_readiness_json),
        weather_sources=args.weather_source or None,
        candidate_seeds=args.candidate_seed or DEFAULT_SEEDS,
        max_candidate_days=args.max_candidate_days,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"target_spec_ready={report['target_spec_ready']}")
    print(f"excluded_scenario_count={len(report['excluded_scenario_ids'])}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["target_spec_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
