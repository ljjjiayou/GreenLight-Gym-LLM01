"""Record authorization for the v33 strict-targeted shadow-only sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONTROLLER = "llm_rspc_v2"


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


def _coverage_pass(coverage: Mapping[str, Any]) -> bool:
    return bool(
        coverage.get("cache_coverage_pass", False)
        and _num(coverage.get("coverage_rate")) == 1.0
        and int(_num(coverage.get("missing_key_count"))) == 0
        and int(_num(coverage.get("missing_buffered_action_count"))) == 0
        and int(_num(coverage.get("missing_parsed_plan_count"))) == 0
    )


def _strict_flags_present(command: Sequence[str]) -> bool:
    return bool(
        "--plan-cache-mode" in command
        and "replay" in command
        and "--plan-cache-strict" in command
        and "--plan-cache-key-policy" in command
        and "scenario_timestep" in command
    )


def _command_for_group(*, runner_path: str, group: Mapping[str, Any], output_root: str) -> list[str]:
    group_id = str(group.get("group_id", "v33"))
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
        CONTROLLER,
        "--max-steps",
        "240",
        "--plan-cache-mode",
        "replay",
        "--plan-cache-path",
        str(group.get("cache_path", "")),
        "--plan-cache-strict",
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--output-json",
        str(Path(output_root) / f"strict_targeted_shadow_source_sweep_v33_{group_id}_20260528.json"),
        "--output-trace-dir",
        str(Path(output_root) / "traces"),
    ]


def build_report(
    *,
    manifest: Mapping[str, Any],
    cache_coverage: Mapping[str, Any],
    overlay_preflight: Mapping[str, Any],
    authorization_source: str,
    output_root: str,
    controllers: Sequence[str] = (CONTROLLER,),
) -> dict[str, Any]:
    manifest_ready = bool(manifest.get("strict_targeted_shadow_sweep_v33_manifest_ready", False))
    cache_ok = _coverage_pass(cache_coverage)
    overlay_ok = bool(
        overlay_preflight.get("controlled_canary_overlay_validated", False)
        and overlay_preflight.get("controlled_canary_runner_surface_validated", False)
        and int(_num(overlay_preflight.get("protocol_hash_mismatch_count"))) == 0
    )
    runner_path = str(overlay_preflight.get("overlay_runner_path", "") or "")
    runner_exists = bool(runner_path and _resolve(runner_path).exists())
    controller_list = list(controllers)
    controller_scope_ok = controller_list == [CONTROLLER]
    commands = [
        _command_for_group(runner_path=runner_path, group=group, output_root=output_root)
        for group in manifest.get("command_groups", []) or []
        if isinstance(group, Mapping)
    ]
    strict_flags_ok = bool(commands and all(_strict_flags_present(command) for command in commands))
    cartesian_ok = bool(commands and all(bool(group.get("cartesian_product_safe", False)) for group in manifest.get("command_groups", []) or [] if isinstance(group, Mapping)))
    authorized = bool(manifest_ready and cache_ok and overlay_ok and runner_exists and controller_scope_ok and strict_flags_ok and cartesian_ok)
    return {
        "schema_version": "strict_targeted_shadow_sweep_v33_execution_record_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "existing-cache strict-targeted shadow-only sweep v33",
            "reopens_rejected_preset": False,
        },
        "authorization_source": authorization_source,
        "strict_targeted_shadow_sweep_v33_authorized": authorized,
        "scope": "strict_targeted_shadow_only_sweep_v33" if authorized else "blocked",
        "metadata_replay_execution_allowed": authorized,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "controllers": controller_list,
        "blocked_controllers": ["llm_rspc_v2_hot_dry_proposer_strict"],
        "scenario_ids": list(manifest.get("scenario_ids", []) or []),
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
            "scenario_count": int(_num(manifest.get("scenario_count"))),
            "cache_coverage_pass": cache_ok,
            "cache_coverage_rate": _num(cache_coverage.get("coverage_rate")),
            "cache_missing_key_count": int(_num(cache_coverage.get("missing_key_count"))),
            "cache_missing_buffered_action_count": int(_num(cache_coverage.get("missing_buffered_action_count"))),
            "cache_missing_parsed_plan_count": int(_num(cache_coverage.get("missing_parsed_plan_count"))),
            "overlay_validated": overlay_ok,
            "overlay_runner_path_exists": runner_exists,
            "controller_scope_ok": controller_scope_ok,
            "strict_replay_flags_present": strict_flags_ok,
            "cartesian_product_safe": cartesian_ok,
        },
        "next_action": "execute_v33_shadow_only_sweep" if authorized else "v33_shadow_sweep_not_authorized",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strict-Targeted Shadow Sweep v33 Execution Record",
        "",
        f"- Authorized: {report.get('strict_targeted_shadow_sweep_v33_authorized', False)}",
        f"- Scope: `{report.get('scope', '')}`",
        f"- Controllers: `{','.join(str(item) for item in report.get('controllers', []) or [])}`",
        "- Blocked controlled controller: `llm_rspc_v2_hot_dry_proposer_strict`",
        "- Cache fill: false",
        "- Online LLM: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Precheck",
        "",
        "| gate | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("precheck", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Planned Commands", ""])
    for command in report.get("planned_commands", []) or []:
        lines.append("```powershell")
        lines.append(" ".join(str(part) for part in command))
        lines.append("```")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--cache-coverage-json", required=True)
    parser.add_argument("--overlay-preflight-json", required=True)
    parser.add_argument("--authorization-source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--controllers", default=CONTROLLER)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        manifest=_load(args.manifest_json),
        cache_coverage=_load(args.cache_coverage_json),
        overlay_preflight=_load(args.overlay_preflight_json),
        authorization_source=args.authorization_source,
        output_root=args.output_root,
        controllers=[item.strip() for item in str(args.controllers).split(",") if item.strip()],
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"strict_targeted_shadow_sweep_v33_authorized={report['strict_targeted_shadow_sweep_v33_authorized']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["strict_targeted_shadow_sweep_v33_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
