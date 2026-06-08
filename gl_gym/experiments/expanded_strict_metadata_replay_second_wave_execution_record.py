"""Record authorization for second-wave expanded strict metadata replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _scenario_ids(manifest: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for item in manifest.get("selected_scenarios", []) or manifest.get("scenarios", []) or []:
        if isinstance(item, Mapping) and item.get("scenario_id"):
            out.append(str(item["scenario_id"]))
    return out


def _scenario_parts(manifest: Mapping[str, Any]) -> tuple[list[int], list[int], list[int]]:
    years: set[int] = set()
    days: set[int] = set()
    seeds: set[int] = set()
    for item in manifest.get("selected_scenarios", []) or manifest.get("scenarios", []) or []:
        if not isinstance(item, Mapping):
            continue
        years.add(int(item.get("year", 0)))
        days.add(int(item.get("day", 0)))
        seeds.add(int(item.get("seed", 0)))
    return sorted(years), sorted(days), sorted(seeds)


def _cartesian_safe(manifest: Mapping[str, Any]) -> bool:
    scenario_ids = set(_scenario_ids(manifest))
    years, days, seeds = _scenario_parts(manifest)
    expected = {f"y{year}_d{day}_s{seed}_n240" for year in years for day in days for seed in seeds}
    return expected == scenario_ids


def _command_has_flags(command: Sequence[Any], flags: Sequence[str]) -> bool:
    text = " ".join(str(part) for part in command)
    return all(flag in text for flag in flags)


def build_planned_command(
    *,
    overlay_runner_path: str,
    selected_cache_path: str,
    years: Sequence[int],
    days: Sequence[int],
    seeds: Sequence[int],
    output_json: str,
    output_trace_dir: str,
) -> list[str]:
    return [
        "python",
        overlay_runner_path,
        "--years",
        ",".join(str(item) for item in years),
        "--days",
        ",".join(str(item) for item in days),
        "--seeds",
        ",".join(str(item) for item in seeds),
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
        output_json,
        "--output-trace-dir",
        output_trace_dir,
    ]


def build_report(
    *,
    scenario_manifest: Mapping[str, Any],
    cache_coverage: Mapping[str, Any],
    protocol_v1_overlay_preflight: Mapping[str, Any],
    authorization_source: str,
    output_json_path: str,
    output_trace_dir: str,
    baseline_trace_dir: str,
) -> dict[str, Any]:
    overlay_root = str(protocol_v1_overlay_preflight.get("protocol_v1_overlay_root", "") or "")
    overlay_runner = str(Path(overlay_root) / "gl_gym" / "experiments" / "run_frozen_benchmark.py")
    selected_cache_path = str(scenario_manifest.get("selected_cache_path", ""))
    years, days, seeds = _scenario_parts(scenario_manifest)
    planned_command = build_planned_command(
        overlay_runner_path=overlay_runner,
        selected_cache_path=selected_cache_path,
        years=years,
        days=days,
        seeds=seeds,
        output_json=output_json_path,
        output_trace_dir=output_trace_dir,
    )
    scenario_ids = _scenario_ids(scenario_manifest)
    baseline_root = _resolve(baseline_trace_dir)
    baseline_paths = {scenario: baseline_root / f"{scenario}_llm_rspc_v2.csv" for scenario in scenario_ids}
    cartesian_safe = _cartesian_safe(scenario_manifest)
    precheck = {
        "scenario_manifest_ready": bool(scenario_manifest.get("second_wave_manifest_ready", False)),
        "scenario_scope_is_second_wave": scenario_manifest.get("scope") == "second_wave_expanded_metadata_coverage",
        "scenario_count_positive": len(scenario_ids) > 0,
        "covers_pure_hot_dry": "pure_hot_dry" in set(scenario_manifest.get("covered_families", []) or []),
        "covers_mixed_dry_dew": "mixed_dry_dew" in set(scenario_manifest.get("covered_families", []) or []),
        "runner_cartesian_product_matches_manifest": cartesian_safe,
        "cache_coverage_pass": bool(cache_coverage.get("cache_coverage_pass", False)),
        "cache_coverage_rate_is_one": float(cache_coverage.get("coverage_rate", 0.0) or 0.0) == 1.0,
        "missing_key_count_zero": int(cache_coverage.get("missing_key_count", 0) or 0) == 0,
        "missing_buffered_action_count_zero": int(cache_coverage.get("missing_buffered_action_count", 0) or 0) == 0,
        "missing_parsed_plan_count_zero": int(cache_coverage.get("missing_parsed_plan_count", 0) or 0) == 0,
        "cache_fill_blocked": not bool(cache_coverage.get("cache_fill_run", False)),
        "online_llm_blocked": not bool(cache_coverage.get("online_llm_called", False)),
        "protocol_v1_overlay_validated": bool(protocol_v1_overlay_preflight.get("protocol_v1_overlay_validated", False)),
        "overlay_runner_path_exists": _resolve(overlay_runner).exists(),
        "baseline_trace_dir_exists": baseline_root.exists(),
        "baseline_traces_exist": all(path.exists() for path in baseline_paths.values()),
        "planned_command_has_strict_replay_flags": _command_has_flags(
            planned_command,
            [
                "--plan-cache-mode replay",
                "--plan-cache-strict",
                "--plan-cache-key-policy scenario_timestep",
            ],
        ),
    }
    precheck_pass = all(precheck.values())
    return {
        "schema_version": "expanded_strict_metadata_replay_second_wave_execution_record_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "second-wave expanded strict metadata replay only",
            "reopens_rejected_preset": False,
        },
        "authorization_source": authorization_source,
        "second_wave_expanded_metadata_replay_authorized": bool(precheck_pass),
        "metadata_replay_execution_allowed": bool(precheck_pass),
        "metadata_replay_allowed": bool(precheck_pass),
        "controlled_replay_allowed": False,
        "counterexample_replay_allowed": False,
        "performance_claim_allowed": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "replay_run": False,
        "precheck_pass": bool(precheck_pass),
        "precheck": precheck,
        "scope": "second_wave_expanded_metadata_coverage",
        "scenario_ids": scenario_ids,
        "controller": "llm_rspc_v2",
        "max_steps": 240,
        "selected_cache_path": selected_cache_path,
        "actual_protocol_implementation_for_execution": "protocol_v1_snapshot_ephemeral_overlay",
        "protocol_v1_overlay_root": overlay_root,
        "overlay_runner_path": overlay_runner,
        "baseline_trace_dir": baseline_trace_dir,
        "baseline_trace_paths": {key: str(value) for key, value in baseline_paths.items()},
        "runner_cartesian_product_safe": cartesian_safe,
        "planned_environment": {
            "PYTHONPATH": f"{overlay_root};{PROJECT_ROOT}",
        },
        "planned_command": planned_command,
        "output_json": output_json_path,
        "output_trace_dir": output_trace_dir,
        "failure_taxonomy": [
            "action_changed",
            "metadata_missing",
            "runtime_provenance_missing",
            "joint_prediction_missing",
            "cache_mismatch",
            "protocol_implementation_mismatch",
            "overlay_path_or_hash_mismatch",
            "runtime_error",
            "strict_cache_miss",
            "unknown_post_guardrail_rewrite",
        ],
        "next_action": "execute_second_wave_expanded_strict_metadata_replay" if precheck_pass else "second_wave_precheck_failure",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Second-Wave Expanded Strict Metadata Replay Execution Record",
        "",
        f"- Authorized: {report.get('second_wave_expanded_metadata_replay_authorized', False)}",
        f"- Metadata replay execution allowed: {report.get('metadata_replay_execution_allowed', False)}",
        "- Controlled replay allowed: false",
        "- Counterexample replay allowed: false",
        "- Performance claim allowed: false",
        f"- Scope: `{report.get('scope', '')}`",
        f"- Scenarios: `{', '.join(str(item) for item in report.get('scenario_ids', []) or [])}`",
        f"- Runner Cartesian product safe: {report.get('runner_cartesian_product_safe', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Precheck",
        "",
        "| check | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("precheck", {})).items():
        lines.append(f"| {key} | `{value}` |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifest-json", required=True)
    parser.add_argument("--cache-coverage-json", required=True)
    parser.add_argument("--protocol-v1-overlay-preflight-json", required=True)
    parser.add_argument("--authorization-source", required=True)
    parser.add_argument("--replay-output-json", required=True)
    parser.add_argument("--replay-output-trace-dir", required=True)
    parser.add_argument("--baseline-trace-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        scenario_manifest=_load(args.scenario_manifest_json),
        cache_coverage=_load(args.cache_coverage_json),
        protocol_v1_overlay_preflight=_load(args.protocol_v1_overlay_preflight_json),
        authorization_source=args.authorization_source,
        output_json_path=args.replay_output_json,
        output_trace_dir=args.replay_output_trace_dir,
        baseline_trace_dir=args.baseline_trace_dir,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"second_wave_expanded_metadata_replay_authorized={report['second_wave_expanded_metadata_replay_authorized']}")
    print(f"metadata_replay_execution_allowed={report['metadata_replay_execution_allowed']}")
    print(f"precheck_pass={report['precheck_pass']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["precheck_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
