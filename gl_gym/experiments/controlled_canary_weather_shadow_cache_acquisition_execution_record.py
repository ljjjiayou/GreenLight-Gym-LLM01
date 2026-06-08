"""Record v36 authorization/precheck for weather-sourced shadow cache acquisition."""

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

DEFAULT_OUTPUT_ROOT = "gl_gym/result/benchmarks/weather_shadow_cache_acquisition_v36_20260529"
SELECTED_CACHE_PATH = "gl_gym/result/plan_cache/holdout_seed42_43_hot_dry_qwen_20260517_merged.json"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


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
        cleaned = value.strip().strip('"').strip("'")
        return bool(cleaned)
    return False


def _api_key_present_from_env_or_dotenv(dotenv_path: str | Path = ".env") -> bool:
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
        and int(_num(overlay_preflight.get("protocol_hash_mismatch_count"))) == 0
    )


def _command_for_group(
    *,
    runner_path: str,
    group: Mapping[str, Any],
    isolated_cache_path: str,
    output_root: str,
) -> list[str]:
    group_id = str(group.get("group_id", "weather_v36"))
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
        str(int(group.get("max_steps", 240) or 240)),
        "--plan-cache-mode",
        "record",
        "--plan-cache-path",
        isolated_cache_path,
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--output-json",
        str(Path(output_root) / f"weather_shadow_cache_acquisition_v36_{group_id}.json"),
        "--output-trace-dir",
        str(Path(output_root) / "traces"),
    ]


def _record_flags_present(command: Sequence[str]) -> bool:
    return bool(
        "--plan-cache-mode" in command
        and "record" in command
        and "--plan-cache-key-policy" in command
        and "scenario_timestep" in command
        and "--plan-cache-strict" not in command
    )


def build_report(
    *,
    manifest: Mapping[str, Any],
    overlay_preflight: Mapping[str, Any],
    authorization_source: str,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    controllers: Sequence[str] = ("llm_rspc_v2",),
    api_key_present: bool | None = None,
    dotenv_path: str | Path = ".env",
) -> dict[str, Any]:
    api_key_present = _api_key_present_from_env_or_dotenv(dotenv_path) if api_key_present is None else bool(api_key_present)
    manifest_ready = bool(manifest.get("weather_shadow_cache_acquisition_manifest_ready", False))
    isolated_cache_path = str(manifest.get("isolated_cache_path", "") or "")
    selected_cache_overwrite = _resolve(isolated_cache_path) == _resolve(SELECTED_CACHE_PATH)
    cache_path_ok = bool(isolated_cache_path and not selected_cache_overwrite and "weather_expansion_shadow_cache_v36_20260529.json" in isolated_cache_path)
    controller_list = list(controllers)
    controller_scope_ok = controller_list == ["llm_rspc_v2"]
    runner_path = str(overlay_preflight.get("overlay_runner_path", "") or "")
    runner_exists = bool(runner_path and _resolve(runner_path).exists())
    overlay_valid = _overlay_ok(overlay_preflight)
    commands = [
        _command_for_group(
            runner_path=runner_path,
            group=group,
            isolated_cache_path=isolated_cache_path,
            output_root=output_root,
        )
        for group in manifest.get("command_groups", []) or []
        if isinstance(group, Mapping)
    ]
    record_flags_ok = bool(commands and all(_record_flags_present(command) for command in commands))
    cartesian_ok = bool(
        commands
        and all(bool(group.get("cartesian_product_safe", False)) for group in manifest.get("command_groups", []) or [] if isinstance(group, Mapping))
    )
    scope_ok = bool(int(_num(manifest.get("scenario_count"))) == 9 and len(manifest.get("command_groups", []) or []) == 3)
    authorized = bool(manifest_ready and scope_ok and cache_path_ok and controller_scope_ok)
    executable = bool(authorized and api_key_present and overlay_valid and runner_exists and record_flags_ok and cartesian_ok)
    blockers: list[str] = []
    if not manifest_ready:
        blockers.append("manifest_not_ready")
    if not scope_ok:
        blockers.append("top9_scope_mismatch")
    if not cache_path_ok:
        blockers.append("isolated_cache_path_invalid")
    if selected_cache_overwrite:
        blockers.append("selected_cache_overwrite_blocked")
    if not controller_scope_ok:
        blockers.append("controller_scope_mismatch")
    if not api_key_present:
        blockers.append("online_llm_credentials_missing")
    if not overlay_valid:
        blockers.append("protocol_overlay_mismatch")
    if not runner_exists:
        blockers.append("overlay_runner_missing")
    if not record_flags_ok:
        blockers.append("record_mode_flags_missing")
    if not cartesian_ok:
        blockers.append("cartesian_product_risk")
    return {
        "schema_version": "controlled_canary_weather_shadow_cache_acquisition_execution_record_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "weather-sourced shadow cache acquisition only",
            "reopens_rejected_preset": False,
        },
        "authorization_source": authorization_source,
        "weather_shadow_cache_acquisition_authorized": authorized,
        "weather_shadow_cache_acquisition_executable": executable,
        "scope": "top_3_weather_days_9_shadow_scenarios" if authorized else "blocked",
        "cache_fill_authorized": authorized,
        "online_llm_allowed": authorized,
        "cache_fill_run": False,
        "online_llm_called": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "controllers": controller_list,
        "blocked_controllers": ["llm_rspc_v2_hot_dry_proposer_strict"],
        "scenario_ids": list(manifest.get("scenario_ids", []) or []),
        "isolated_cache_path": isolated_cache_path,
        "selected_cache_path": SELECTED_CACHE_PATH,
        "selected_cache_overwrite_allowed": False,
        "actual_protocol_implementation_for_execution": "protocol_v1_controlled_canary_overlay_20260525",
        "protocol_v1_overlay_root": str(overlay_preflight.get("protocol_v1_overlay_root", "") or ""),
        "runner_path": runner_path,
        "planned_environment": {
            "PYTHONPATH": f"{overlay_preflight.get('protocol_v1_overlay_root', '')};{PROJECT_ROOT}",
        },
        "output_root": output_root,
        "planned_commands": commands,
        "precheck": {
            "manifest_ready": manifest_ready,
            "scope_top9_match": scope_ok,
            "scenario_count": int(_num(manifest.get("scenario_count"))),
            "isolated_cache_path_ok": cache_path_ok,
            "selected_cache_overwrite_blocked": selected_cache_overwrite,
            "controller_scope_ok": controller_scope_ok,
            "online_llm_credentials_present": api_key_present,
            "dotenv_checked": True,
            "overlay_validated": overlay_valid,
            "overlay_runner_path_exists": runner_exists,
            "record_mode_flags_present": record_flags_ok,
            "cartesian_product_safe": cartesian_ok,
        },
        "failure_taxonomy": sorted(set(blockers)),
        "next_action": (
            "execute_v36_weather_shadow_cache_acquisition"
            if executable
            else "online_llm_credentials_missing"
            if "online_llm_credentials_missing" in blockers
            else "v36_shadow_cache_acquisition_precheck_blocked"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Weather-Sourced Shadow Cache Acquisition Execution Record v36",
        "",
        f"- Authorized: {report.get('weather_shadow_cache_acquisition_authorized', False)}",
        f"- Executable: {report.get('weather_shadow_cache_acquisition_executable', False)}",
        f"- Scope: `{report.get('scope', '')}`",
        f"- Isolated cache path: `{report.get('isolated_cache_path', '')}`",
        "- Controller: `llm_rspc_v2`",
        "- Blocked controlled controller: `llm_rspc_v2_hot_dry_proposer_strict`",
        f"- Cache fill authorized: {report.get('cache_fill_authorized', False)}",
        f"- Online LLM allowed: {report.get('online_llm_allowed', False)}",
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
    lines.extend(["", "## Failure Taxonomy", ""])
    failures = report.get("failure_taxonomy", []) or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- none")
    lines.extend(["", "## Planned Commands", ""])
    for command in report.get("planned_commands", []) or []:
        lines.append("```powershell")
        lines.append(" ".join(str(part) for part in command))
        lines.append("```")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--overlay-preflight-json", required=True)
    parser.add_argument("--authorization-source", required=True)
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
    print(f"weather_shadow_cache_acquisition_authorized={report['weather_shadow_cache_acquisition_authorized']}")
    print(f"weather_shadow_cache_acquisition_executable={report['weather_shadow_cache_acquisition_executable']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
