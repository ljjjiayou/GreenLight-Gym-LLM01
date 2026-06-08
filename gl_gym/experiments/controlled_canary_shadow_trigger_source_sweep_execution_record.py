"""Record authorization for the shadow-only trigger source sweep v26."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_SCENARIOS = [
    "y2015_d180_s42_n240",
    "y2015_d180_s43_n240",
    "y2015_d240_s42_n240",
    "y2015_d240_s43_n240",
    "y2015_d240_s44_n240",
]

COMMAND_GROUPS = [
    {"group_id": "y2015_d180_s42_s43", "years": [2015], "days": [180], "seeds": [42, 43]},
    {"group_id": "y2015_d240_s42_s43_s44", "years": [2015], "days": [240], "seeds": [42, 43, 44]},
]


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


def _command_for_group(
    *,
    runner_path: str,
    group: Mapping[str, Any],
    selected_cache_path: str,
    output_root: str,
) -> list[str]:
    group_id = str(group.get("group_id", "shadow_trigger_source"))
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
        "240",
        "--plan-cache-mode",
        "replay",
        "--plan-cache-path",
        selected_cache_path,
        "--plan-cache-strict",
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--output-json",
        str(Path(output_root) / f"shadow_trigger_source_sweep_{group_id}_20260526.json"),
        "--output-trace-dir",
        str(Path(output_root) / "traces"),
    ]


def _strict_flags_present(command: Sequence[str]) -> bool:
    return bool(
        "--plan-cache-mode" in command
        and "replay" in command
        and "--plan-cache-strict" in command
        and "--plan-cache-key-policy" in command
        and "scenario_timestep" in command
    )


def build_report(
    *,
    readiness_v25: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    cache_coverage: Mapping[str, Any],
    overlay_preflight: Mapping[str, Any],
    authorization_source: str,
    output_root: str,
    controllers: Sequence[str] = ("llm_rspc_v2",),
) -> dict[str, Any]:
    source_manifest_ready = bool(
        readiness_v25.get("shadow_trigger_source_manifest_ready", False)
        and source_manifest.get("shadow_trigger_source_manifest_ready", False)
        and source_manifest.get("shadow_only", False)
    )
    source_ids = list(source_manifest.get("scenario_ids", []) or [])
    expected_source_set = set(EXPECTED_SCENARIOS)
    source_scope_match = bool(set(source_ids) == expected_source_set and len(source_ids) == len(EXPECTED_SCENARIOS))
    cache_ok = bool(readiness_v25.get("shadow_trigger_source_cache_precheck_pass", False) and _coverage_pass(cache_coverage))
    overlay_ok = bool(
        overlay_preflight.get("controlled_canary_overlay_validated", False)
        and overlay_preflight.get("controlled_canary_runner_surface_validated", False)
        and int(_num(overlay_preflight.get("protocol_hash_mismatch_count"))) == 0
    )
    runner_path = str(overlay_preflight.get("overlay_runner_path", ""))
    runner_exists = bool(runner_path and _resolve(runner_path).exists())
    controller_list = list(controllers)
    controller_scope_ok = controller_list == ["llm_rspc_v2"]
    selected_cache_path = str(source_manifest.get("selected_cache_path") or cache_coverage.get("plan_cache_path") or "")
    commands = [
        _command_for_group(
            runner_path=runner_path,
            group=group,
            selected_cache_path=selected_cache_path,
            output_root=output_root,
        )
        for group in COMMAND_GROUPS
    ]
    strict_flags_ok = bool(commands and all(_strict_flags_present(command) for command in commands))
    authorized = bool(source_manifest_ready and source_scope_match and cache_ok and overlay_ok and runner_exists and controller_scope_ok and strict_flags_ok)
    return {
        "schema_version": "controlled_canary_shadow_trigger_source_sweep_execution_record_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only trigger source sweep",
            "reopens_rejected_preset": False,
        },
        "authorization_source": authorization_source,
        "shadow_trigger_source_sweep_authorized": authorized,
        "scope": "shadow_trigger_source_sweep_only" if authorized else "blocked",
        "metadata_replay_execution_allowed": authorized,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "controllers": controller_list,
        "controller_scope_exactly_llm_rspc_v2": controller_scope_ok,
        "blocked_controllers": ["llm_rspc_v2_hot_dry_proposer_strict"],
        "scenario_ids": EXPECTED_SCENARIOS,
        "source_manifest_scenario_ids": source_ids,
        "selected_cache_path": selected_cache_path,
        "actual_protocol_implementation_for_execution": "protocol_v1_controlled_canary_overlay_20260525",
        "protocol_v1_overlay_root": str(overlay_preflight.get("protocol_v1_overlay_root", "")),
        "runner_path": runner_path,
        "planned_environment": {
            "PYTHONPATH": f"{overlay_preflight.get('protocol_v1_overlay_root', '')};{PROJECT_ROOT}",
        },
        "output_root": output_root,
        "command_groups": COMMAND_GROUPS,
        "planned_commands": commands,
        "precheck": {
            "source_manifest_ready": source_manifest_ready,
            "source_scope_match": source_scope_match,
            "cache_coverage_pass": cache_ok,
            "cache_coverage_rate": _num(cache_coverage.get("coverage_rate")),
            "cache_missing_key_count": int(_num(cache_coverage.get("missing_key_count"))),
            "cache_missing_buffered_action_count": int(_num(cache_coverage.get("missing_buffered_action_count"))),
            "cache_missing_parsed_plan_count": int(_num(cache_coverage.get("missing_parsed_plan_count"))),
            "overlay_validated": overlay_ok,
            "overlay_runner_path_exists": runner_exists,
            "controller_scope_ok": controller_scope_ok,
            "strict_replay_flags_present": strict_flags_ok,
        },
        "next_action": "execute_shadow_trigger_source_sweep" if authorized else "shadow_trigger_source_sweep_not_authorized",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Shadow-Only Trigger Source Sweep Execution Record v26",
        "",
        f"- Authorized: {report.get('shadow_trigger_source_sweep_authorized', False)}",
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
    parser.add_argument("--readiness-v25-json", required=True)
    parser.add_argument("--source-manifest-json", required=True)
    parser.add_argument("--cache-coverage-json", required=True)
    parser.add_argument("--overlay-preflight-json", required=True)
    parser.add_argument("--authorization-source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--controllers", default="llm_rspc_v2")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        readiness_v25=_load(args.readiness_v25_json),
        source_manifest=_load(args.source_manifest_json),
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
    print(f"shadow_trigger_source_sweep_authorized={report['shadow_trigger_source_sweep_authorized']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["shadow_trigger_source_sweep_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
