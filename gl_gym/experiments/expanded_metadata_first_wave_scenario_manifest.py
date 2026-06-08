"""Build the first-wave expanded metadata replay scenario manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIRST_WAVE_SCENARIOS = [
    {
        "scenario_id": "y2010_d120_s44_n240",
        "env_id": "TomatoEnv_y2010_d120_s44",
        "year": 2010,
        "day": 120,
        "seed": 44,
        "max_steps": 240,
        "scenario_families": [
            "neutral_no_risk",
            "radiation_spike",
            "wind_stress",
            "dawn_dew_or_high_humidity",
        ],
        "source_regimes": [
            "neutral_no_risk",
            "radiation_spike",
            "wind_stress",
            "cold_humid",
        ],
    },
    {
        "scenario_id": "y2010_d180_s44_n240",
        "env_id": "TomatoEnv_y2010_d180_s44",
        "year": 2010,
        "day": 180,
        "seed": 44,
        "max_steps": 240,
        "scenario_families": ["dawn_dew_or_high_humidity"],
        "source_regimes": ["dawn_dew"],
    },
]

BLOCKED_FAMILIES = [
    {
        "family": "pure_hot_dry",
        "status": "blocked_missing_cache_or_scenario",
        "scenario_count": 0,
        "reason": "No qualified scenario/cache pair is available for first-wave metadata coverage.",
    },
    {
        "family": "mixed_dry_dew",
        "status": "blocked_missing_cache_or_scenario",
        "scenario_count": 0,
        "reason": "No qualified scenario/cache pair is available for first-wave metadata coverage.",
    },
]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _windows_for_regimes(counterexample_suite: Mapping[str, Any], regimes: Sequence[str]) -> list[dict[str, Any]]:
    by_regime = counterexample_suite.get("windows_by_regime", {}) if isinstance(counterexample_suite, Mapping) else {}
    windows: list[dict[str, Any]] = []
    for regime in regimes:
        items = by_regime.get(regime, []) if isinstance(by_regime, Mapping) else []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, Mapping):
                    windows.append(dict(item))
    return windows


def build_report(
    *,
    counterexample_suite: Mapping[str, Any],
    selected_cache_path: str,
) -> dict[str, Any]:
    scenarios: list[dict[str, Any]] = []
    for spec in FIRST_WAVE_SCENARIOS:
        windows = _windows_for_regimes(counterexample_suite, spec["source_regimes"])
        scenarios.append(
            {
                **spec,
                "selected_for_first_wave": True,
                "window_count": len(windows),
                "windows": windows,
            }
        )
    return {
        "schema_version": "expanded_metadata_first_wave_scenario_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "first-wave expanded strict metadata replay manifest",
            "reopens_rejected_preset": False,
        },
        "first_wave_manifest_ready": True,
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "counterexample_replay_allowed": False,
        "performance_claim_allowed": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "selected_cache_path": selected_cache_path,
        "scope": "first_wave_expanded_metadata_coverage",
        "controller": "llm_rspc_v2",
        "scenario_count": len(scenarios),
        "selected_scenarios": scenarios,
        "scenarios": scenarios,
        "blocked_families": BLOCKED_FAMILIES,
        "next_action": "run_no_fill_expanded_cache_coverage_check",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# First-Wave Expanded Metadata Scenario Manifest",
        "",
        "- Mode: first-wave expanded strict metadata replay manifest",
        "- Default `llm_rspc_v2` changed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Selected cache: `{report.get('selected_cache_path', '')}`",
        "",
        "## Selected Scenarios",
        "",
        "| scenario | families | windows |",
        "| --- | --- | ---: |",
    ]
    for item in report.get("selected_scenarios", []) or []:
        if not isinstance(item, Mapping):
            continue
        families = ", ".join(str(value) for value in item.get("scenario_families", []) or [])
        lines.append(f"| {item.get('scenario_id', '')} | {families} | {item.get('window_count', 0)} |")
    lines.extend(["", "## Blocked Families", "", "| family | status |", "| --- | --- |"])
    for item in report.get("blocked_families", []) or []:
        if isinstance(item, Mapping):
            lines.append(f"| {item.get('family', '')} | {item.get('status', '')} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterexample-suite-json", required=True)
    parser.add_argument("--selected-cache-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        counterexample_suite=_load(args.counterexample_suite_json),
        selected_cache_path=args.selected_cache_path,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"first_wave_manifest_ready={report['first_wave_manifest_ready']}")
    print(f"scenario_count={report['scenario_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
