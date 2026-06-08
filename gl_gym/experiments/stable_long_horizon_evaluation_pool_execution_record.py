"""Build a v38 stable long-horizon evaluation pool execution record."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SELECTED_CACHE_PATH = "gl_gym/result/plan_cache/holdout_seed42_43_hot_dry_qwen_20260517_merged.json"
OUTPUT_ROOT = "gl_gym/result/benchmarks/stable_long_horizon_evaluation_pool_v38_20260529"
ISOLATED_CACHE_PATHS = {
    240: "gl_gym/result/plan_cache/stable_long_horizon_eval_v38_h240_20260529.json",
    720: "gl_gym/result/plan_cache/stable_long_horizon_eval_v38_h720_20260529.json",
    1440: "gl_gym/result/plan_cache/stable_long_horizon_eval_v38_h1440_20260529.json",
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _load_optional(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = _resolve(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _manual_dotenv_has_key(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key and value.strip().strip('"').strip("'"):
            return True
    return False


def _api_key_present(dotenv_path: str | Path = ".env") -> bool:
    if bool(os.getenv("BAILIAN_API_KEY")):
        return True
    dotenv = _resolve(dotenv_path)
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=dotenv, override=False)
    except Exception:
        return _manual_dotenv_has_key(dotenv, "BAILIAN_API_KEY")
    return bool(os.getenv("BAILIAN_API_KEY")) or _manual_dotenv_has_key(dotenv, "BAILIAN_API_KEY")


def _scenario_id(year: int, day: int, seed: int, horizon_steps: int) -> str:
    return f"y{year}_d{day}_s{seed}_n{horizon_steps}"


def _base_key(item: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(item.get("year", 0)), int(item.get("day", 0)), int(item.get("seed", 0)))


def _scenario_for_horizon(item: Mapping[str, Any], horizon_steps: int) -> dict[str, Any]:
    year, day, seed = _base_key(item)
    return {
        "scenario_id": _scenario_id(year, day, seed, horizon_steps),
        "year": year,
        "day": day,
        "seed": seed,
        "max_steps": int(horizon_steps),
        "regime": str(item.get("regime", f"d{day}")),
        "source_tier": str(item.get("source_tier", "")),
        "cache_strategy_h240": str(item.get("cache_strategy_h240", "")),
    }


def _stable_previous_bases(previous_result: Mapping[str, Any] | None) -> set[tuple[int, int, int]]:
    if not previous_result:
        return set()
    out: set[tuple[int, int, int]] = set()
    for item in previous_result.get("stable_scenario_reports", []) or []:
        if isinstance(item, Mapping):
            out.add(_base_key(item))
    for item in previous_result.get("stable_base_keys", []) or []:
        if isinstance(item, list) and len(item) == 3:
            out.add((int(item[0]), int(item[1]), int(item[2])))
    return out


def _eligible_scenarios(
    *,
    manifest: Mapping[str, Any],
    previous_result: Mapping[str, Any] | None,
    horizon_steps: int,
) -> list[dict[str, Any]]:
    candidates = [item for item in manifest.get("candidate_scenarios", []) or [] if isinstance(item, Mapping)]
    if horizon_steps == 240:
        return [_scenario_for_horizon(item, horizon_steps) for item in candidates]
    previous_horizon = 240 if horizon_steps == 720 else 720
    if not previous_result or int(previous_result.get("horizon_steps", 0) or 0) != previous_horizon:
        return []
    stable_bases = _stable_previous_bases(previous_result)
    ordered = [item for item in candidates if _base_key(item) in stable_bases]
    if horizon_steps == 720:
        selected: list[Mapping[str, Any]] = []
        counts: dict[str, int] = {}
        for item in ordered:
            regime = str(item.get("regime", "unknown"))
            if counts.get(regime, 0) >= 2:
                continue
            counts[regime] = counts.get(regime, 0) + 1
            selected.append(item)
        return [_scenario_for_horizon(item, horizon_steps) for item in selected]
    return [_scenario_for_horizon(item, horizon_steps) for item in ordered[:6]]


def _overlay_ok(overlay_preflight: Mapping[str, Any]) -> bool:
    return bool(
        overlay_preflight.get("controlled_canary_overlay_validated", False)
        and overlay_preflight.get("controlled_canary_runner_surface_validated", False)
        and int(overlay_preflight.get("protocol_hash_mismatch_count", 0) or 0) == 0
    )


def _group_scenarios(scenarios: Sequence[Mapping[str, Any]], *, key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str], list[int]] = {}
    for item in scenarios:
        group_key = (int(item["year"]), int(item["day"]), str(item.get(key, "")))
        groups.setdefault(group_key, []).append(int(item["seed"]))
    out: list[dict[str, Any]] = []
    for (year, day, label), seeds in sorted(groups.items()):
        horizon = int(scenarios[0]["max_steps"]) if scenarios else 0
        out.append(
            {
                "group_id": f"h{horizon}_y{year}_d{day}_{label or 'all'}",
                "years": [year],
                "days": [day],
                "seeds": sorted(seeds),
                "max_steps": horizon,
                "scenario_ids": [_scenario_id(year, day, seed, horizon) for seed in sorted(seeds)],
                "cartesian_product_safe": True,
            }
        )
    return out


def _command_for_group(
    *,
    runner_path: str,
    group: Mapping[str, Any],
    controller: str,
    horizon_steps: int,
    output_root: str,
    plan_cache_mode: str = "off",
    plan_cache_path: str = "",
    plan_cache_strict: bool = False,
) -> list[str]:
    group_id = str(group.get("group_id", f"h{horizon_steps}"))
    command = [
        "python",
        runner_path,
        "--years",
        ",".join(str(item) for item in group.get("years", []) or []),
        "--days",
        ",".join(str(item) for item in group.get("days", []) or []),
        "--seeds",
        ",".join(str(item) for item in group.get("seeds", []) or []),
        "--controllers",
        controller,
        "--max-steps",
        str(int(horizon_steps)),
        "--output-json",
        str(Path(output_root) / f"stable_long_horizon_eval_v38_{controller}_{group_id}.json"),
        "--output-trace-dir",
        str(Path(output_root) / f"traces_h{horizon_steps}"),
    ]
    if controller.startswith("llm"):
        command.extend(["--plan-cache-mode", plan_cache_mode])
        if plan_cache_path:
            command.extend(["--plan-cache-path", plan_cache_path])
        if plan_cache_strict:
            command.append("--plan-cache-strict")
        command.extend(["--plan-cache-key-policy", "scenario_timestep"])
    return command


def _build_commands(
    *,
    runner_path: str,
    scenarios: Sequence[Mapping[str, Any]],
    horizon_steps: int,
    output_root: str,
) -> list[dict[str, Any]]:
    command_records: list[dict[str, Any]] = []
    baseline_groups = _group_scenarios(scenarios, key="source_tier")
    for group in baseline_groups:
        command_records.append(
            {
                "command_type": "baseline_feasibility",
                "controller": "ppo",
                "cache_strategy": "none",
                "scenario_ids": list(group.get("scenario_ids", []) or []),
                "command": _command_for_group(
                    runner_path=runner_path,
                    group=group,
                    controller="ppo",
                    horizon_steps=horizon_steps,
                    output_root=output_root,
                ),
            }
        )
    if horizon_steps == 240:
        tier_a = [item for item in scenarios if item.get("cache_strategy_h240") == "selected_cache_replay"]
        tier_b = [item for item in scenarios if item.get("cache_strategy_h240") != "selected_cache_replay"]
        for group in _group_scenarios(tier_a, key="source_tier"):
            command_records.append(
                {
                    "command_type": "default_llm_feasibility",
                    "controller": "llm_rspc_v2",
                    "cache_strategy": "selected_cache_replay",
                    "scenario_ids": list(group.get("scenario_ids", []) or []),
                    "command": _command_for_group(
                        runner_path=runner_path,
                        group=group,
                        controller="llm_rspc_v2",
                        horizon_steps=horizon_steps,
                        output_root=output_root,
                        plan_cache_mode="replay",
                        plan_cache_path=SELECTED_CACHE_PATH,
                        plan_cache_strict=True,
                    ),
                }
            )
        llm_record_scenarios = tier_b
    else:
        llm_record_scenarios = list(scenarios)
    isolated_cache = ISOLATED_CACHE_PATHS[horizon_steps]
    for group in _group_scenarios(llm_record_scenarios, key="source_tier"):
        command_records.append(
            {
                "command_type": "default_llm_feasibility",
                "controller": "llm_rspc_v2",
                "cache_strategy": "isolated_record",
                "scenario_ids": list(group.get("scenario_ids", []) or []),
                "command": _command_for_group(
                    runner_path=runner_path,
                    group=group,
                    controller="llm_rspc_v2",
                    horizon_steps=horizon_steps,
                    output_root=output_root,
                    plan_cache_mode="record",
                    plan_cache_path=isolated_cache,
                    plan_cache_strict=False,
                ),
            }
        )
    return command_records


def _commands_ok(records: Sequence[Mapping[str, Any]]) -> bool:
    if not records:
        return False
    for record in records:
        command = list(record.get("command", []) or [])
        controller = str(record.get("controller", ""))
        if controller == "llm_rspc_v2_hot_dry_proposer_strict":
            return False
        if controller.startswith("llm"):
            if "--plan-cache-key-policy" not in command or "scenario_timestep" not in command:
                return False
            cache_path = ""
            if "--plan-cache-path" in command:
                cache_path = str(command[command.index("--plan-cache-path") + 1])
            if record.get("cache_strategy") == "isolated_record":
                if "--plan-cache-mode" not in command or "record" not in command:
                    return False
                if not cache_path or _resolve(cache_path) == _resolve(SELECTED_CACHE_PATH):
                    return False
            if record.get("cache_strategy") == "selected_cache_replay":
                if "--plan-cache-mode" not in command or "replay" not in command or "--plan-cache-strict" not in command:
                    return False
    return True


def build_report(
    *,
    manifest: Mapping[str, Any],
    overlay_preflight: Mapping[str, Any],
    authorization_source: str,
    horizon_steps: int,
    previous_result: Mapping[str, Any] | None = None,
    controllers: Sequence[str] = ("ppo", "llm_rspc_v2"),
    api_key_present: bool | None = None,
    output_root: str = OUTPUT_ROOT,
    dotenv_path: str | Path = ".env",
) -> dict[str, Any]:
    horizon_steps = int(horizon_steps)
    api_key_present = _api_key_present(dotenv_path) if api_key_present is None else bool(api_key_present)
    manifest_ready = bool(manifest.get("stable_long_horizon_manifest_ready", False))
    eligible = _eligible_scenarios(manifest=manifest, previous_result=previous_result, horizon_steps=horizon_steps)
    overlay_valid = _overlay_ok(overlay_preflight)
    runner_path = str(overlay_preflight.get("overlay_runner_path", "") or "")
    runner_exists = bool(runner_path and _resolve(runner_path).exists())
    controller_scope_ok = list(controllers) == ["ppo", "llm_rspc_v2"]
    horizon_valid = horizon_steps in ISOLATED_CACHE_PATHS
    commands = _build_commands(
        runner_path=runner_path,
        scenarios=eligible,
        horizon_steps=horizon_steps,
        output_root=output_root,
    ) if horizon_valid else []
    commands_ok = _commands_ok(commands)
    authorized = bool(manifest_ready and horizon_valid and eligible and controller_scope_ok)
    executable = bool(authorized and overlay_valid and runner_exists and commands_ok and api_key_present)
    blockers: list[str] = []
    if not manifest_ready:
        blockers.append("manifest_not_ready")
    if not horizon_valid:
        blockers.append("invalid_horizon")
    if not eligible:
        blockers.append("no_eligible_scenarios_for_horizon")
    if not controller_scope_ok:
        blockers.append("controller_scope_mismatch")
    if not overlay_valid:
        blockers.append("protocol_overlay_mismatch")
    if not runner_exists:
        blockers.append("overlay_runner_missing")
    if not commands_ok:
        blockers.append("execution_command_scope_invalid")
    if not api_key_present:
        blockers.append("online_llm_credentials_missing")
    return {
        "schema_version": "stable_long_horizon_evaluation_pool_execution_record_20260529_v38",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "stable long-horizon feasibility execution",
        },
        "authorization_source": authorization_source,
        "horizon_steps": horizon_steps,
        "stable_long_horizon_execution_authorized": authorized,
        "stable_long_horizon_execution_executable": executable,
        "controllers": list(controllers),
        "blocked_controllers": ["llm_rspc_v2_hot_dry_proposer_strict"],
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "eligible_scenarios": eligible,
        "eligible_scenario_count": len(eligible),
        "isolated_cache_path": ISOLATED_CACHE_PATHS.get(horizon_steps, ""),
        "selected_cache_path": SELECTED_CACHE_PATH,
        "selected_cache_overwrite_allowed": False,
        "online_llm_credentials_present": api_key_present,
        "overlay_valid": overlay_valid,
        "overlay_runner_path": runner_path,
        "planned_command_records": commands,
        "planned_environment": {
            "PYTHONPATH": f"{overlay_preflight.get('protocol_v1_overlay_root') or overlay_preflight.get('overlay_root', '')};{PROJECT_ROOT}",
        },
        "failure_taxonomy": blockers,
        "next_action": f"execute_v38_h{horizon_steps}_stable_pool_feasibility" if executable else "v38_execution_precheck_blocked",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stable Long-Horizon Evaluation Pool Execution Record v38",
        "",
        f"- Horizon steps: `{report.get('horizon_steps')}`",
        f"- Executable: `{report.get('stable_long_horizon_execution_executable', False)}`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Commands",
        "",
    ]
    for record in report.get("planned_command_records", []) or []:
        lines.append(f"- `{record.get('command_type')}` / `{record.get('controller')}` / `{record.get('cache_strategy')}`: {len(record.get('scenario_ids', []) or [])} scenarios")
    failures = report.get("failure_taxonomy", []) or []
    lines.extend(["", "## Failure Taxonomy", ""])
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--overlay-preflight-json", required=True)
    parser.add_argument("--authorization-source", required=True)
    parser.add_argument("--horizon-steps", type=int, required=True)
    parser.add_argument("--previous-result-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        manifest=_load(args.manifest_json),
        overlay_preflight=_load(args.overlay_preflight_json),
        authorization_source=args.authorization_source,
        horizon_steps=args.horizon_steps,
        previous_result=_load_optional(args.previous_result_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"stable_long_horizon_execution_executable={report['stable_long_horizon_execution_executable']}")
    print(f"eligible_scenario_count={report['eligible_scenario_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["stable_long_horizon_execution_executable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
