"""Run grouped no-fill cache coverage precheck for the v33 shadow-only sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.check_plan_cache_coverage import audit_cache_coverage


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


def build_report(*, manifest: Mapping[str, Any], require_payloads: bool = True) -> dict[str, Any]:
    group_results: list[dict[str, Any]] = []
    missing_key_count = 0
    missing_buffered_action_count = 0
    missing_parsed_plan_count = 0
    expected_env_count = 0
    found_env_count = 0
    for group in manifest.get("command_groups", []) or []:
        if not isinstance(group, Mapping):
            continue
        scenarios = group.get("selected_scenarios", []) or []
        audit = audit_cache_coverage(
            plan_cache_path=str(group.get("cache_path", "")),
            years=[],
            days=[],
            seeds=[],
            max_steps=240,
            scenario_specs=scenarios,
            require_payloads=require_payloads,
            plan_cache_key_policy="scenario_timestep",
        )
        group_result = {
            "group_id": str(group.get("group_id", "")),
            "cache_path": str(group.get("cache_path", "")),
            "scenario_ids": list(group.get("scenario_ids", []) or []),
            "cache_coverage_pass": bool(audit.get("cache_coverage_pass", False)),
            "coverage_rate": _num(audit.get("coverage_rate")),
            "expected_env_count": int(_num(audit.get("expected_env_count"))),
            "found_env_count": int(_num(audit.get("found_env_count"))),
            "missing_key_count": int(_num(audit.get("missing_key_count"))),
            "missing_buffered_action_count": int(_num(audit.get("missing_buffered_action_count"))),
            "missing_parsed_plan_count": int(_num(audit.get("missing_parsed_plan_count"))),
            "missing_envs": list(audit.get("missing_envs", []) or []),
            "low_coverage_envs": list(audit.get("low_coverage_envs", []) or []),
        }
        group_results.append(group_result)
        expected_env_count += int(group_result["expected_env_count"])
        found_env_count += int(group_result["found_env_count"])
        missing_key_count += int(group_result["missing_key_count"])
        missing_buffered_action_count += int(group_result["missing_buffered_action_count"])
        missing_parsed_plan_count += int(group_result["missing_parsed_plan_count"])
    coverage_rate = found_env_count / expected_env_count if expected_env_count else 1.0
    cache_coverage_pass = bool(
        manifest.get("strict_targeted_shadow_sweep_v33_manifest_ready", False)
        and group_results
        and all(group["cache_coverage_pass"] for group in group_results)
        and missing_key_count == 0
        and missing_buffered_action_count == 0
        and missing_parsed_plan_count == 0
    )
    return {
        "schema_version": "strict_targeted_shadow_sweep_v33_cache_coverage_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "grouped no-fill cache coverage precheck",
            "reopens_rejected_preset": False,
        },
        "cache_check_run": True,
        "cache_fill_run": False,
        "online_llm_called": False,
        "require_payloads": bool(require_payloads),
        "plan_cache_key_policy": "scenario_timestep",
        "cache_coverage_pass": cache_coverage_pass,
        "can_strict_replay": cache_coverage_pass,
        "can_strict_metadata_replay": cache_coverage_pass,
        "expected_env_count": expected_env_count,
        "found_env_count": found_env_count,
        "coverage_rate": coverage_rate,
        "missing_key_count": missing_key_count,
        "missing_buffered_action_count": missing_buffered_action_count,
        "missing_parsed_plan_count": missing_parsed_plan_count,
        "group_results": group_results,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": "v33_shadow_sweep_execution_record" if cache_coverage_pass else "v33_cache_coverage_blocked",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strict-Targeted Shadow Sweep v33 Cache Coverage",
        "",
        "- Cache fill run: false",
        "- Online LLM called: false",
        f"- Coverage pass: {report.get('cache_coverage_pass', False)}",
        f"- Coverage rate: {_num(report.get('coverage_rate')):.3f}",
        f"- Missing key count: {report.get('missing_key_count', 0)}",
        f"- Missing buffered action count: {report.get('missing_buffered_action_count', 0)}",
        f"- Missing parsed plan count: {report.get('missing_parsed_plan_count', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Groups",
        "",
        "| group | cache | scenarios | pass |",
        "| --- | --- | --- | --- |",
    ]
    for group in report.get("group_results", []) or []:
        if isinstance(group, Mapping):
            lines.append(f"| {group.get('group_id', '')} | {Path(str(group.get('cache_path', ''))).name} | {','.join(str(item) for item in group.get('scenario_ids', []) or [])} | {group.get('cache_coverage_pass', False)} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--require-payloads", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(manifest=_load(args.manifest_json), require_payloads=bool(args.require_payloads))
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"cache_coverage_pass={report['cache_coverage_pass']}")
    print(f"coverage_rate={_num(report.get('coverage_rate')):.3f}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["cache_coverage_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
