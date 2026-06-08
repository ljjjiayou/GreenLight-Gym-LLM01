"""Build the v37 long-horizon shadow feasibility manifest."""

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

PRIMARY_SCENARIOS = [
    (2006, 182, 42),
    (2006, 182, 43),
    (2006, 182, 44),
    (2006, 162, 42),
    (2006, 162, 43),
    (2006, 162, 44),
    (2006, 161, 42),
    (2006, 161, 43),
    (2006, 161, 44),
]


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


def _v35_requested_bases(v35_manifest: Mapping[str, Any]) -> set[tuple[int, int, int]]:
    return {_base_key(item) for item in v35_manifest.get("requested_scenarios", []) or [] if isinstance(item, Mapping)}


def _v36_blocked_scenarios(v36_diagnosis: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for env_id, report in dict(v36_diagnosis.get("env_reports", {})).items():
        trace_path = str(report.get("trace_path", "") or "")
        parts = _parts_from_scenario_id(trace_path or str(env_id).replace("TomatoEnv_", ""))
        sid = scenario_id(parts["year"], parts["day"], parts["seed"], 240)
        out.append(
            {
                "scenario_id": sid,
                "year": parts["year"],
                "day": parts["day"],
                "seed": parts["seed"],
                "max_steps": 240,
                "blocked_reason": "v36_runtime_failed_short_trace",
                "last_step": report.get("last_step"),
                "runtime_error_count": int(report.get("runtime_error_count", 0) or 0),
                "cvodes_failure": any(
                    bool(item.get("error_contains_cvodes_convergence_failure"))
                    for item in report.get("runtime_errors", []) or []
                    if isinstance(item, Mapping)
                ),
            }
        )
    return sorted(out, key=lambda item: (item["year"], item["day"], item["seed"]))


def build_report(
    *,
    v35_manifest: Mapping[str, Any],
    v36_diagnosis: Mapping[str, Any],
) -> dict[str, Any]:
    v35_bases = _v35_requested_bases(v35_manifest)
    blocked = _v36_blocked_scenarios(v36_diagnosis)
    blocked_bases = {_base_key(item) for item in blocked}
    primary = [
        {
            "scenario_id": scenario_id(year, day, seed, 240),
            "year": year,
            "day": day,
            "seed": seed,
            "max_steps": 240,
            "source": "v35_weather_candidate_after_v36_failed_top9",
        }
        for year, day, seed in PRIMARY_SCENARIOS
    ]
    primary_bases = {_base_key(item) for item in primary}
    missing_from_v35 = sorted(primary_bases - v35_bases)
    primary_blocked_overlap = sorted(primary_bases & blocked_bases)
    manifest_ready = bool(
        len(primary) == 9
        and len(blocked) == 9
        and not missing_from_v35
        and not primary_blocked_overlap
        and v36_diagnosis.get("shadow_cache_coverage_pass") is False
    )
    return {
        "schema_version": "long_horizon_shadow_feasibility_manifest_20260529_v37",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only long-horizon feasibility",
        },
        "long_horizon_shadow_feasibility_manifest_ready": manifest_ready,
        "controller": "llm_rspc_v2",
        "blocked_controllers": ["llm_rspc_v2_hot_dry_proposer_strict"],
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "weather_proxy_source_hint_only": True,
        "blocked_runtime_failed_scenarios": blocked,
        "primary_scenarios": primary,
        "primary_scenario_ids": [item["scenario_id"] for item in primary],
        "horizon_ladder": [
            {"horizon_steps": 240, "max_scenarios": 9},
            {"horizon_steps": 720, "max_scenarios": 3},
            {"horizon_steps": 1440, "max_scenarios": 1},
        ],
        "missing_primary_candidates_from_v35": [
            scenario_id(year, day, seed, 240) for year, day, seed in missing_from_v35
        ],
        "primary_blocked_overlap": [
            scenario_id(year, day, seed, 240) for year, day, seed in primary_blocked_overlap
        ],
        "next_action": "build_v37_h240_execution_record" if manifest_ready else "v37_manifest_blocked",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Long-Horizon Shadow Feasibility Manifest v37",
        "",
        f"- Manifest ready: {report.get('long_horizon_shadow_feasibility_manifest_ready', False)}",
        "- Controller: `llm_rspc_v2`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Primary Candidates",
        "",
        "| scenario | source |",
        "| --- | --- |",
    ]
    for item in report.get("primary_scenarios", []) or []:
        lines.append(f"| `{item.get('scenario_id', '')}` | `{item.get('source', '')}` |")
    lines.extend(["", "## Blocked v36 Runtime-Failed Scenarios", "", "| scenario | reason | last_step |", "| --- | --- | ---: |"])
    for item in report.get("blocked_runtime_failed_scenarios", []) or []:
        lines.append(f"| `{item.get('scenario_id', '')}` | `{item.get('blocked_reason', '')}` | `{item.get('last_step')}` |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v35-manifest-json", required=True)
    parser.add_argument("--v36-diagnosis-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        v35_manifest=_load(args.v35_manifest_json),
        v36_diagnosis=_load(args.v36_diagnosis_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"long_horizon_shadow_feasibility_manifest_ready={report['long_horizon_shadow_feasibility_manifest_ready']}")
    print(f"primary_scenario_count={len(report['primary_scenarios'])}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["long_horizon_shadow_feasibility_manifest_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
