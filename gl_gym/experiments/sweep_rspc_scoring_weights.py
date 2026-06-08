"""Generate or execute frozen-replay sweeps for RSPC/fallback scoring weights.

Default mode is dry-run so the command matrix can be inspected before any
controller replay is launched. All weight changes are opt-in through
``--agent-config-overrides`` and leave the default ``llm_rspc_v2`` behaviour
unchanged.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.check_plan_cache_coverage import (  # noqa: E402
    _parse_int_list,
    audit_cache_coverage,
    build_markdown_report as build_cache_coverage_report,
    load_scenario_specs,
)


PRESETS: Dict[str, Dict[str, float]] = {
    "balanced": {},
    "conservative_dry": {
        "fallback_dry_penalty_weight": 1.45,
        "fallback_vpd_high_penalty_weight": 1.20,
        "rspc_hot_dry_score_weight": 1.45,
    },
    "dew_safe": {
        "fallback_dry_penalty_weight": 1.05,
        "fallback_vpd_high_penalty_weight": 1.00,
        "fallback_dew_penalty_weight": 1.05,
        "fallback_rh_penalty_weight": 1.90,
    },
    "hot_dry_relief": {
        "fallback_dry_penalty_weight": 1.80,
        "fallback_vpd_high_penalty_weight": 1.55,
        "rspc_hot_dry_score_weight": 1.90,
        "fallback_dew_penalty_weight": 0.70,
    },
}

METRIC_FIELDS = [
    "total_reward",
    "total_profit",
    "total_rh_low_violation",
    "total_rh_high_violation",
    "total_vpd_high_excess",
    "total_temp_violation",
    "dry_risk_steps",
    "dry_vent_risk_steps",
    "dew_risk_steps",
    "dew_margin_air_lt1_steps",
    "dew_margin_air_lt0_steps",
    "canopy_dew_margin_lt1_steps",
    "canopy_dew_margin_lt0_steps",
    "runtime_error_steps",
    "strict_cache_miss_runtime_error_steps",
    "plan_cache_enabled_steps",
    "plan_cache_hit_steps",
]


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _select_presets(names: Sequence[str] | None) -> Dict[str, Dict[str, float]]:
    if not names:
        return dict(PRESETS)
    selected: Dict[str, Dict[str, float]] = {}
    for name in names:
        key = str(name).strip()
        if key not in PRESETS:
            raise ValueError(f"Unknown preset {key!r}; allowed={sorted(PRESETS)}")
        selected[key] = dict(PRESETS[key])
    return selected


def build_run_command(args: argparse.Namespace, preset: str, overrides: Mapping[str, Any]) -> list[str]:
    output_dir = Path(args.benchmark_output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_json = output_dir / f"{preset}.json"
    command = [
        sys.executable,
        "gl_gym/experiments/run_frozen_benchmark.py",
        "--years",
        str(args.years),
        "--days",
        str(args.days),
        "--seeds",
        str(args.seeds),
        "--controllers",
        str(args.controllers),
        "--max-steps",
        str(args.max_steps),
        "--plan-cache-mode",
        str(args.plan_cache_mode),
        "--plan-cache-path",
        str(args.plan_cache_path),
        "--plan-cache-key-policy",
        str(args.plan_cache_key_policy),
        "--output-json",
        str(output_json),
        "--agent-config-overrides",
        _compact_json(dict(overrides)),
    ]
    if bool(args.plan_cache_strict):
        command.append("--plan-cache-strict")
    if args.output_trace_root:
        trace_root = Path(args.output_trace_root)
        if not trace_root.is_absolute():
            trace_root = PROJECT_ROOT / trace_root
        command.extend(["--output-trace-dir", str(trace_root / preset)])
    return command


def build_scenario_run_command(
    args: argparse.Namespace,
    preset: str,
    overrides: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> tuple[list[str], Path]:
    output_dir = Path(args.benchmark_output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    scenario_id = str(
        scenario.get(
            "scenario_id",
            f"y{int(scenario['year'])}_d{int(scenario['day'])}_s{int(scenario['seed'])}_n{int(scenario.get('max_steps', args.max_steps))}",
        )
    )
    output_json = output_dir / preset / f"{scenario_id}.json"
    command = [
        sys.executable,
        "gl_gym/experiments/run_frozen_benchmark.py",
        "--years",
        str(int(scenario["year"])),
        "--days",
        str(int(scenario["day"])),
        "--seeds",
        str(int(scenario["seed"])),
        "--controllers",
        str(args.controllers),
        "--max-steps",
        str(int(scenario.get("max_steps", args.max_steps))),
        "--plan-cache-mode",
        str(args.plan_cache_mode),
        "--plan-cache-path",
        str(args.plan_cache_path),
        "--plan-cache-key-policy",
        str(args.plan_cache_key_policy),
        "--output-json",
        str(output_json),
        "--agent-config-overrides",
        _compact_json(dict(overrides)),
    ]
    if bool(args.plan_cache_strict):
        command.append("--plan-cache-strict")
    if args.output_trace_root:
        trace_root = Path(args.output_trace_root)
        if not trace_root.is_absolute():
            trace_root = PROJECT_ROOT / trace_root
        command.extend(["--output-trace-dir", str(trace_root / preset)])
    return command, output_json


def _summarise_benchmark(path: Path, preset: str, overrides: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summaries = payload.get("summaries", []) if isinstance(payload, dict) else []
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        if not isinstance(summary, dict):
            continue
        aggregate = summary.get("aggregate", {})
        if not isinstance(aggregate, dict):
            aggregate = {}
        enabled = _num(aggregate.get("plan_cache_enabled_steps", 0.0))
        hits = _num(aggregate.get("plan_cache_hit_steps", 0.0))
        row = {
            "preset": preset,
            "controller": str(summary.get("controller", "")),
            "scenario_id": str(summary.get("scenario_id", "")),
            "overrides": dict(overrides),
            "plan_cache_hit_rate": float(hits / enabled) if enabled else 0.0,
        }
        for field in METRIC_FIELDS:
            row[field] = _num(aggregate.get(field, 0.0))
        rows.append(row)
    return rows


def build_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# RSPC Scoring Weight Sweep",
        "",
        f"- Mode: {'dry-run' if result.get('dry_run') else 'execute'}",
        f"- Presets: {', '.join(result.get('presets', []))}",
        f"- Command count: {len(result.get('commands', []))}",
        "",
        "## Commands",
        "",
    ]
    for item in result.get("commands", []):
        lines.append(f"### {item.get('preset')}")
        lines.append("")
        lines.append("```powershell")
        lines.append(" ".join(str(part) for part in item.get("command", [])))
        lines.append("```")
        lines.append("")

    rows = list(result.get("rows", []))
    if rows:
        lines.extend(["## Metrics", ""])
        headers = [
            "preset",
            "scenario",
            "controller",
            "reward",
            "profit",
            "RHlow",
            "VPDhi",
            "temp",
            "cache_hit_rate",
            "runtime",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("preset", "")),
                        str(row.get("scenario_id", "")),
                        str(row.get("controller", "")),
                        f"{_num(row.get('total_reward')):.3f}",
                        f"{_num(row.get('total_profit')):.3f}",
                        f"{_num(row.get('total_rh_low_violation')):.3f}",
                        f"{_num(row.get('total_vpd_high_excess')):.3f}",
                        f"{_num(row.get('total_temp_violation')):.3f}",
                        f"{_num(row.get('plan_cache_hit_rate')):.3f}",
                        f"{_num(row.get('runtime_error_steps')):.0f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", action="append", choices=sorted(PRESETS), help="Preset to include; repeatable.")
    parser.add_argument("--years", default="2015")
    parser.add_argument("--days", default="120,180,240")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--controllers", default="llm_rspc_v2")
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--scenario-list-json", default="", help="Run exactly these selected scenarios instead of years/days/seeds cross product.")
    parser.add_argument("--control-interval", type=int, default=12)
    parser.add_argument("--plan-cache-mode", default="replay", choices=["off", "record", "replay", "refresh"])
    parser.add_argument(
        "--plan-cache-path",
        default="gl_gym/result/plan_cache/holdout_seed42_43_hot_dry_qwen_20260517_merged.json",
    )
    parser.add_argument("--plan-cache-key-policy", default="scenario_timestep", choices=["prompt", "scenario_timestep"])
    parser.add_argument("--no-plan-cache-strict", dest="plan_cache_strict", action="store_false")
    parser.set_defaults(plan_cache_strict=True)
    parser.add_argument(
        "--benchmark-output-dir",
        default="gl_gym/result/benchmarks/rspc_scoring_weight_sweep_v1_20260518",
    )
    parser.add_argument("--output-trace-root", default="")
    parser.add_argument("--output-json", default="gl_gym/result/audits/rspc_scoring_weight_sweep_v1_20260518.json")
    parser.add_argument("--output-md", default="gl_gym/result/audits/rspc_scoring_weight_sweep_v1_20260518.md")
    parser.add_argument(
        "--skip-cache-coverage-check",
        action="store_true",
        help="Skip strict replay plan-cache coverage preflight.",
    )
    parser.add_argument(
        "--cache-coverage-output-dir",
        default="",
        help="Directory for coverage preflight JSON/Markdown. Defaults beside --output-json.",
    )
    parser.add_argument("--execute", dest="dry_run", action="store_false", help="Execute generated commands.")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Only emit commands.")
    parser.set_defaults(dry_run=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    presets = _select_presets(args.preset)
    scenario_specs = load_scenario_specs(args.scenario_list_json) if args.scenario_list_json else []
    commands: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    benchmark_dir = Path(args.benchmark_output_dir)
    if not benchmark_dir.is_absolute():
        benchmark_dir = PROJECT_ROOT / benchmark_dir
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    if (
        not args.dry_run
        and not args.skip_cache_coverage_check
        and args.plan_cache_mode == "replay"
        and bool(args.plan_cache_strict)
    ):
        coverage = audit_cache_coverage(
            plan_cache_path=args.plan_cache_path,
            years=[] if scenario_specs else _parse_int_list(str(args.years)),
            days=[] if scenario_specs else _parse_int_list(str(args.days)),
            seeds=[] if scenario_specs else _parse_int_list(str(args.seeds)),
            max_steps=int(args.max_steps),
            control_interval=int(args.control_interval),
            scenario_specs=scenario_specs,
        )
        coverage_dir = Path(args.cache_coverage_output_dir) if args.cache_coverage_output_dir else Path(args.output_json).parent
        if not coverage_dir.is_absolute():
            coverage_dir = PROJECT_ROOT / coverage_dir
        coverage_dir.mkdir(parents=True, exist_ok=True)
        coverage_json = coverage_dir / "plan_cache_coverage.json"
        coverage_md = coverage_dir / "plan_cache_coverage.md"
        coverage_json.write_text(json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        coverage_md.write_text(build_cache_coverage_report(coverage), encoding="utf-8")
        print(
            "[sweep] cache coverage "
            f"rate={_num(coverage.get('coverage_rate')):.3f} "
            f"can_strict_replay={coverage.get('can_strict_replay')} "
            f"report={coverage_json}",
            flush=True,
        )
        if not coverage.get("can_strict_replay"):
            print("[sweep] aborting strict replay because plan-cache coverage is incomplete", flush=True)
            return 1

    for preset, overrides in presets.items():
        if scenario_specs:
            for scenario in scenario_specs:
                command, output_json = build_scenario_run_command(args, preset, overrides, scenario)
                commands.append(
                    {
                        "preset": preset,
                        "scenario_id": str(scenario.get("scenario_id", "")),
                        "overrides": overrides,
                        "command": command,
                    }
                )
                if not args.dry_run:
                    print(f"[sweep] running preset={preset} scenario={scenario.get('scenario_id')}", flush=True)
                    completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
                    if completed.returncode != 0:
                        return int(completed.returncode)
                    rows.extend(_summarise_benchmark(output_json, preset, overrides))
        else:
            command = build_run_command(args, preset, overrides)
            commands.append({"preset": preset, "overrides": overrides, "command": command})
            if not args.dry_run:
                print(f"[sweep] running preset={preset}", flush=True)
                completed = subprocess.run(command, cwd=str(PROJECT_ROOT))
                if completed.returncode != 0:
                    return int(completed.returncode)
                rows.extend(_summarise_benchmark(benchmark_dir / f"{preset}.json", preset, overrides))

    result = {
        "schema_version": "rspc_scoring_weight_sweep_v1",
        "dry_run": bool(args.dry_run),
        "presets": list(presets),
        "commands": commands,
        "rows": rows,
        "scenario_list_json": str(args.scenario_list_json or ""),
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json
    if not output_md.is_absolute():
        output_md = PROJECT_ROOT / output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(build_report(result), encoding="utf-8")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    if args.dry_run:
        for item in commands:
            print(f"[dry-run:{item['preset']}] " + " ".join(str(part) for part in item["command"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
