"""Check frozen plan-cache coverage before strict replay sweeps.

The cache can contain irregular replanning timesteps because emergency replans
are recorded in addition to regular control-interval boundaries.  For strict
replay preflight we therefore verify scenario-level coverage rather than
requiring every multiple of the control interval to exist exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).split(","):
        token = part.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _load_entries(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", {}) if isinstance(payload, Mapping) else {}
    if isinstance(entries, Mapping):
        return [entry for entry in entries.values() if isinstance(entry, Mapping)]
    if isinstance(entries, list):
        return [entry for entry in entries if isinstance(entry, Mapping)]
    return []


def expected_env_ids(years: Sequence[int], days: Sequence[int], seeds: Sequence[int]) -> list[str]:
    return [
        f"TomatoEnv_y{year}_d{day}_s{seed}"
        for year in years
        for day in days
        for seed in seeds
    ]


def load_scenario_specs(path: str | Path) -> list[dict[str, Any]]:
    raw_path = Path(path)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, Mapping):
        raw_items = payload.get("selected_scenarios", payload.get("scenarios", []))
    else:
        raw_items = []
    specs: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        try:
            specs.append(
                {
                    "year": int(raw["year"]),
                    "day": int(raw["day"]),
                    "seed": int(raw["seed"]),
                    "max_steps": int(raw.get("max_steps", 240)),
                    "scenario_id": str(raw.get("scenario_id", "")),
                }
            )
        except Exception:
            continue
    return specs


def audit_cache_coverage(
    *,
    plan_cache_path: str | Path,
    years: Sequence[int],
    days: Sequence[int],
    seeds: Sequence[int],
    max_steps: int,
    control_interval: int = 12,
    min_entries_per_env: int | None = None,
    scenario_specs: Sequence[Mapping[str, Any]] | None = None,
    require_payloads: bool = False,
    plan_cache_key_policy: str = "scenario_timestep",
) -> dict[str, Any]:
    path = Path(plan_cache_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if scenario_specs:
        env_specs = []
        for raw in scenario_specs:
            year = int(raw.get("year", 0))
            day = int(raw.get("day", 0))
            seed = int(raw.get("seed", 0))
            steps = int(raw.get("max_steps", max_steps))
            env_specs.append(
                {
                    "env_id": f"TomatoEnv_y{year}_d{day}_s{seed}",
                    "year": year,
                    "day": day,
                    "seed": seed,
                    "max_steps": steps,
                    "scenario_id": str(raw.get("scenario_id", f"y{year}_d{day}_s{seed}_n{steps}")),
                }
            )
    else:
        env_specs = [
            {
                "env_id": env_id,
                "year": None,
                "day": None,
                "seed": None,
                "max_steps": int(max_steps),
                "scenario_id": env_id,
            }
            for env_id in expected_env_ids(years, days, seeds)
        ]
    env_ids = [str(spec["env_id"]) for spec in env_specs]

    warnings: list[str] = []
    if not path.exists():
        return {
            "schema_version": "plan_cache_coverage_v1",
            "plan_cache_path": str(path),
            "expected_env_count": len(env_ids),
            "found_env_count": 0,
            "coverage_rate": 0.0,
            "cache_coverage_pass": False,
            "can_strict_replay": False,
            "can_strict_metadata_replay": False,
            "cache_check_run": True,
            "cache_fill_run": False,
            "online_llm_called": False,
            "require_payloads": bool(require_payloads),
            "missing_key_count": len(env_ids),
            "missing_buffered_action_count": 0,
            "missing_parsed_plan_count": 0,
            "missing_envs": env_ids,
            "low_coverage_envs": env_ids,
            "envs": {},
            "warnings": [f"plan_cache_not_found:{path}"],
        }

    entries = _load_entries(path)
    by_env: dict[str, list[int]] = {env_id: [] for env_id in env_ids}
    missing_buffered_action_count = 0
    missing_parsed_plan_count = 0
    for entry in entries:
        env_id = str(entry.get("env_id", "") or "")
        if env_id not in by_env:
            continue
        timestep = int(round(_num(entry.get("timestep"), -1)))
        if timestep >= 0:
            by_env[env_id].append(timestep)
            if require_payloads:
                if not isinstance(entry.get("buffered_action"), Mapping):
                    missing_buffered_action_count += 1
                if not isinstance(entry.get("parsed_plan"), Mapping):
                    missing_parsed_plan_count += 1

    env_reports: dict[str, dict[str, Any]] = {}
    missing_envs: list[str] = []
    low_coverage_envs: list[str] = []
    spec_by_env = {str(spec["env_id"]): spec for spec in env_specs}
    for env_id in env_ids:
        timesteps = sorted(set(by_env.get(env_id, [])))
        env_max_steps = int(spec_by_env[env_id].get("max_steps", max_steps))
        required_min_entries = (
            int(min_entries_per_env)
            if min_entries_per_env is not None
            else max(1, (env_max_steps + int(control_interval) - 1) // int(control_interval))
        )
        min_last_timestep = max(0, env_max_steps - int(control_interval))
        has_zero = 0 in timesteps
        max_timestep = max(timesteps) if timesteps else None
        entry_count = len(timesteps)
        issues: list[str] = []
        if not timesteps:
            missing_envs.append(env_id)
            issues.append("missing_env")
        if timesteps and not has_zero:
            issues.append("missing_timestep_zero")
        if timesteps and max_timestep is not None and max_timestep < min_last_timestep:
            issues.append("last_timestep_too_early")
        if timesteps and entry_count < required_min_entries:
            issues.append("entry_count_below_expected")
        if issues:
            low_coverage_envs.append(env_id)
        env_reports[env_id] = {
            "entry_count": entry_count,
            "required_min_entries": required_min_entries,
            "has_timestep_zero": has_zero,
            "min_timestep": min(timesteps) if timesteps else None,
            "max_timestep": max_timestep,
            "min_required_last_timestep": min_last_timestep,
            "scenario_id": spec_by_env[env_id].get("scenario_id", ""),
            "max_steps": env_max_steps,
            "first_timesteps": timesteps[:8],
            "last_timesteps": timesteps[-8:],
            "issues": issues,
            "ok": not issues,
        }

    found_env_count = sum(1 for env_id in env_ids if env_reports[env_id]["entry_count"] > 0)
    coverage_rate = found_env_count / len(env_ids) if env_ids else 1.0
    missing_key_count = len(set(missing_envs + low_coverage_envs))
    payloads_ok = not require_payloads or (
        missing_buffered_action_count == 0 and missing_parsed_plan_count == 0
    )
    can_strict_replay = not missing_envs and not low_coverage_envs and payloads_ok
    if entries and not env_ids:
        warnings.append("no_expected_envs_requested")

    return {
        "schema_version": "plan_cache_coverage_v1",
        "plan_cache_path": str(path),
        "years": list(years),
        "days": list(days),
        "seeds": list(seeds),
        "max_steps": int(max_steps),
        "control_interval": int(control_interval),
        "plan_cache_key_policy": str(plan_cache_key_policy),
        "scenario_specs_enabled": bool(scenario_specs),
        "expected_env_count": len(env_ids),
        "found_env_count": found_env_count,
        "coverage_rate": coverage_rate,
        "cache_coverage_pass": can_strict_replay,
        "can_strict_replay": can_strict_replay,
        "can_strict_metadata_replay": can_strict_replay,
        "cache_check_run": True,
        "cache_fill_run": False,
        "online_llm_called": False,
        "require_payloads": bool(require_payloads),
        "missing_key_count": missing_key_count,
        "missing_buffered_action_count": missing_buffered_action_count,
        "missing_parsed_plan_count": missing_parsed_plan_count,
        "missing_envs": missing_envs,
        "low_coverage_envs": low_coverage_envs,
        "envs": env_reports,
        "warnings": warnings,
    }


def build_markdown_report(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Plan Cache Coverage Check",
        "",
        f"- Cache: `{audit.get('plan_cache_path', '')}`",
        f"- Expected envs: {audit.get('expected_env_count', 0)}",
        f"- Found envs: {audit.get('found_env_count', 0)}",
        f"- Coverage rate: {_num(audit.get('coverage_rate')):.3f}",
        f"- Can strict replay: **{str(audit.get('can_strict_replay')).upper()}**",
        f"- Plan cache key policy: `{audit.get('plan_cache_key_policy', '')}`",
        f"- Cache coverage pass: **{str(audit.get('cache_coverage_pass')).upper()}**",
        f"- Missing key count: {audit.get('missing_key_count', 0)}",
        f"- Missing buffered action count: {audit.get('missing_buffered_action_count', 0)}",
        f"- Missing parsed plan count: {audit.get('missing_parsed_plan_count', 0)}",
        "",
        "## Env Coverage",
        "",
        "| env_id | entries | zero | min_ts | max_ts | ok | issues |",
        "| --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    envs = audit.get("envs", {})
    if isinstance(envs, Mapping):
        for env_id, report in sorted(envs.items()):
            if not isinstance(report, Mapping):
                continue
            issues = ",".join(str(item) for item in report.get("issues", []))
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(env_id),
                        str(report.get("entry_count", 0)),
                        str(report.get("has_timestep_zero", False)),
                        str(report.get("min_timestep", "")),
                        str(report.get("max_timestep", "")),
                        str(report.get("ok", False)),
                        issues,
                    ]
                )
                + " |"
            )
    if audit.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit.get("warnings", []))
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-cache-path", required=True)
    parser.add_argument("--years", default="")
    parser.add_argument("--days", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--scenario-list-json", default="", help="JSON with scenarios/selected_scenarios entries.")
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--control-interval", type=int, default=12)
    parser.add_argument("--min-entries-per-env", type=int, default=None)
    parser.add_argument("--plan-cache-key-policy", default="scenario_timestep")
    parser.add_argument("--require-payloads", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = load_scenario_specs(args.scenario_list_json) if args.scenario_list_json else None
    audit = audit_cache_coverage(
        plan_cache_path=args.plan_cache_path,
        years=_parse_int_list(args.years) if args.years else [],
        days=_parse_int_list(args.days) if args.days else [],
        seeds=_parse_int_list(args.seeds) if args.seeds else [],
        max_steps=args.max_steps,
        control_interval=args.control_interval,
        min_entries_per_env=args.min_entries_per_env,
        scenario_specs=specs,
        require_payloads=args.require_payloads,
        plan_cache_key_policy=args.plan_cache_key_policy,
    )
    if args.output_json:
        output_json = Path(args.output_json)
        if not output_json.is_absolute():
            output_json = PROJECT_ROOT / output_json
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {output_json}")
    if args.output_md:
        output_md = Path(args.output_md)
        if not output_md.is_absolute():
            output_md = PROJECT_ROOT / output_md
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_markdown_report(audit), encoding="utf-8")
        print(f"wrote {output_md}")
    print(
        "coverage_rate="
        f"{_num(audit.get('coverage_rate')):.3f} "
        f"can_strict_replay={audit.get('can_strict_replay')}"
    )
    return 0 if audit.get("can_strict_replay") else 1


if __name__ == "__main__":
    raise SystemExit(main())
