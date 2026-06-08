"""Build a v37 long-horizon shadow feasibility execution record."""

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
DEFAULT_OUTPUT_ROOT = "gl_gym/result/benchmarks/long_horizon_shadow_feasibility_v37_20260529"
HORIZON_CACHE_PATHS = {
    240: "gl_gym/result/plan_cache/long_horizon_shadow_feasibility_v37_h240_20260529.json",
    720: "gl_gym/result/plan_cache/long_horizon_shadow_feasibility_v37_h720_20260529.json",
    1440: "gl_gym/result/plan_cache/long_horizon_shadow_feasibility_v37_h1440_20260529.json",
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
        if name.strip() != key:
            continue
        return bool(value.strip().strip('"').strip("'"))
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


def _overlay_ok(overlay_preflight: Mapping[str, Any]) -> bool:
    return bool(
        overlay_preflight.get("controlled_canary_overlay_validated", False)
        and overlay_preflight.get("controlled_canary_runner_surface_validated", False)
        and int(overlay_preflight.get("protocol_hash_mismatch_count", 0) or 0) == 0
    )


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
    }


def _eligible_from_previous(
    *,
    manifest: Mapping[str, Any],
    previous_result: Mapping[str, Any] | None,
    horizon_steps: int,
) -> list[dict[str, Any]]:
    primary = list(manifest.get("primary_scenarios", []) or [])
    if horizon_steps == 240:
        return [_scenario_for_horizon(item, horizon_steps) for item in primary if isinstance(item, Mapping)]
    required_previous_horizon = 240 if horizon_steps == 720 else 720 if horizon_steps == 1440 else None
    if not previous_result or previous_result.get("horizon_steps") != required_previous_horizon:
        return []
    pass_bases = {
        tuple(item)
        for item in previous_result.get("horizon_pass_base_keys", []) or []
        if isinstance(item, list) and len(item) == 3
    }
    ordered = [
        item
        for item in primary
        if isinstance(item, Mapping) and _base_key(item) in pass_bases
    ]
    limit = 3 if horizon_steps == 720 else 1
    return [_scenario_for_horizon(item, horizon_steps) for item in ordered[:limit]]


def _command_groups(scenarios: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for item in scenarios:
        groups.setdefault((int(item["year"]), int(item["day"])), []).append(int(item["seed"]))
    out: list[dict[str, Any]] = []
    for (year, day), seeds in sorted(groups.items()):
        horizon = int(scenarios[0]["max_steps"]) if scenarios else 0
        out.append(
            {
                "group_id": f"h{horizon}_y{year}_d{day}",
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
    cache_path: str,
    output_root: str,
    horizon_steps: int,
) -> list[str]:
    group_id = str(group.get("group_id", f"h{horizon_steps}"))
    return [
        "python",
        runner_path,
        "--years",
        ",".join(str(item) for item in group.get("years", []) or []),
        "--days",
        ",".join(str(item) for item in group.get("days", []) or []),
        "--seeds",
        ",".join(str(item) for item in group.get("seeds", []) or []),
        "--controllers",
        "llm_rspc_v2",
        "--max-steps",
        str(int(horizon_steps)),
        "--plan-cache-mode",
        "record",
        "--plan-cache-path",
        cache_path,
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--output-json",
        str(Path(output_root) / f"long_horizon_shadow_feasibility_v37_{group_id}.json"),
        "--output-trace-dir",
        str(Path(output_root) / f"traces_h{horizon_steps}"),
    ]


def _record_flags_ok(commands: Sequence[Sequence[str]]) -> bool:
    return bool(
        commands
        and all(
            "--plan-cache-mode" in command
            and "record" in command
            and "--plan-cache-key-policy" in command
            and "scenario_timestep" in command
            and "--plan-cache-strict" not in command
            for command in commands
        )
    )


def build_report(
    *,
    manifest: Mapping[str, Any],
    overlay_preflight: Mapping[str, Any],
    authorization_source: str,
    horizon_steps: int,
    previous_result: Mapping[str, Any] | None = None,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    controllers: Sequence[str] = ("llm_rspc_v2",),
    api_key_present: bool | None = None,
    dotenv_path: str | Path = ".env",
) -> dict[str, Any]:
    horizon_steps = int(horizon_steps)
    api_key_present = _api_key_present(dotenv_path) if api_key_present is None else bool(api_key_present)
    manifest_ready = bool(manifest.get("long_horizon_shadow_feasibility_manifest_ready", False))
    eligible_scenarios = _eligible_from_previous(
        manifest=manifest,
        previous_result=previous_result,
        horizon_steps=horizon_steps,
    )
    groups = _command_groups(eligible_scenarios)
    cache_path = HORIZON_CACHE_PATHS.get(horizon_steps, "")
    selected_cache_overwrite = bool(cache_path and _resolve(cache_path) == _resolve(SELECTED_CACHE_PATH))
    cache_path_ok = bool(cache_path and "long_horizon_shadow_feasibility_v37_" in cache_path and not selected_cache_overwrite)
    controller_scope_ok = list(controllers) == ["llm_rspc_v2"]
    overlay_valid = _overlay_ok(overlay_preflight)
    runner_path = str(overlay_preflight.get("overlay_runner_path", "") or "")
    runner_exists = bool(runner_path and _resolve(runner_path).exists())
    commands = [
        _command_for_group(
            runner_path=runner_path,
            group=group,
            cache_path=cache_path,
            output_root=output_root,
            horizon_steps=horizon_steps,
        )
        for group in groups
    ]
    flags_ok = _record_flags_ok(commands)
    cartesian_ok = bool(groups and all(group.get("cartesian_product_safe", False) for group in groups))
    horizon_valid = horizon_steps in HORIZON_CACHE_PATHS
    authorized = bool(manifest_ready and horizon_valid and eligible_scenarios and controller_scope_ok and cache_path_ok)
    executable = bool(authorized and api_key_present and overlay_valid and runner_exists and flags_ok and cartesian_ok)
    blockers: list[str] = []
    if not manifest_ready:
        blockers.append("manifest_not_ready")
    if not horizon_valid:
        blockers.append("invalid_horizon")
    if not eligible_scenarios:
        blockers.append("no_eligible_scenarios_for_horizon")
    if not controller_scope_ok:
        blockers.append("controller_scope_mismatch")
    if not cache_path_ok:
        blockers.append("isolated_cache_path_invalid")
    if selected_cache_overwrite:
        blockers.append("selected_cache_overwrite_blocked")
    if not api_key_present:
        blockers.append("online_llm_credentials_missing")
    if not overlay_valid:
        blockers.append("protocol_overlay_mismatch")
    if not runner_exists:
        blockers.append("overlay_runner_missing")
    if not flags_ok:
        blockers.append("record_mode_flags_missing")
    if not cartesian_ok:
        blockers.append("cartesian_product_risk")
    return {
        "schema_version": "long_horizon_shadow_feasibility_execution_record_20260529_v37",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only long-horizon feasibility execution",
        },
        "authorization_source": authorization_source,
        "horizon_steps": horizon_steps,
        "previous_horizon_steps": previous_result.get("horizon_steps") if previous_result else None,
        "long_horizon_shadow_feasibility_authorized": authorized,
        "long_horizon_shadow_feasibility_executable": executable,
        "cache_fill_authorized": authorized,
        "online_llm_allowed": authorized,
        "cache_fill_run": False,
        "online_llm_called": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "controllers": list(controllers),
        "blocked_controllers": ["llm_rspc_v2_hot_dry_proposer_strict"],
        "scenario_ids": [str(item["scenario_id"]) for item in eligible_scenarios],
        "eligible_scenarios": eligible_scenarios,
        "isolated_cache_path": cache_path,
        "selected_cache_path": SELECTED_CACHE_PATH,
        "selected_cache_overwrite_allowed": False,
        "actual_protocol_implementation_for_execution": "protocol_v1_controlled_canary_overlay_20260525",
        "protocol_v1_overlay_root": str(overlay_preflight.get("protocol_v1_overlay_root", "") or ""),
        "runner_path": runner_path,
        "planned_environment": {
            "PYTHONPATH": f"{overlay_preflight.get('protocol_v1_overlay_root', '')};{PROJECT_ROOT}",
        },
        "output_root": output_root,
        "trace_dir": str(Path(output_root) / f"traces_h{horizon_steps}"),
        "command_groups": groups,
        "planned_commands": commands,
        "precheck": {
            "manifest_ready": manifest_ready,
            "horizon_valid": horizon_valid,
            "eligible_scenario_count": len(eligible_scenarios),
            "controller_scope_ok": controller_scope_ok,
            "isolated_cache_path_ok": cache_path_ok,
            "selected_cache_overwrite_blocked": selected_cache_overwrite,
            "online_llm_credentials_present": api_key_present,
            "dotenv_checked": True,
            "overlay_validated": overlay_valid,
            "overlay_runner_path_exists": runner_exists,
            "record_mode_flags_present": flags_ok,
            "cartesian_product_safe": cartesian_ok,
        },
        "failure_taxonomy": sorted(set(blockers)),
        "next_action": (
            f"execute_v37_h{horizon_steps}_shadow_feasibility"
            if executable
            else "online_llm_credentials_missing"
            if "online_llm_credentials_missing" in blockers
            else "v37_long_horizon_execution_precheck_blocked"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Long-Horizon Shadow Feasibility Execution Record v37",
        "",
        f"- Horizon steps: `{report.get('horizon_steps')}`",
        f"- Authorized: {report.get('long_horizon_shadow_feasibility_authorized', False)}",
        f"- Executable: {report.get('long_horizon_shadow_feasibility_executable', False)}",
        "- Controller: `llm_rspc_v2`",
        f"- Isolated cache path: `{report.get('isolated_cache_path', '')}`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Precheck",
        "",
        "| gate | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("precheck", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Scenario IDs", ""])
    lines.extend(f"- `{item}`" for item in report.get("scenario_ids", []) or []) or lines.append("- none")
    lines.extend(["", "## Planned Commands", ""])
    for command in report.get("planned_commands", []) or []:
        lines.append("```powershell")
        lines.append(" ".join(str(part) for part in command))
        lines.append("```")
    lines.extend(["", "## Failure Taxonomy", ""])
    failures = report.get("failure_taxonomy", []) or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--overlay-preflight-json", required=True)
    parser.add_argument("--authorization-source", required=True)
    parser.add_argument("--horizon-steps", type=int, required=True)
    parser.add_argument("--previous-result-json", default="")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--controllers", default="llm_rspc_v2")
    parser.add_argument("--dotenv-path", default=".env")
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
        output_root=args.output_root,
        controllers=[item.strip() for item in str(args.controllers).split(",") if item.strip()],
        dotenv_path=args.dotenv_path,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"horizon_steps={report['horizon_steps']}")
    print(f"long_horizon_shadow_feasibility_executable={report['long_horizon_shadow_feasibility_executable']}")
    print(f"eligible_scenario_count={len(report['scenario_ids'])}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["long_horizon_shadow_feasibility_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
