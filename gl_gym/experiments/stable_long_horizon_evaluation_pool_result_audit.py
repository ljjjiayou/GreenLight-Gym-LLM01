"""Audit v38 stable long-horizon evaluation pool traces."""

from __future__ import annotations

import argparse
import csv
import json
import re
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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


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


def _base_key(item: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(item.get("year", 0)), int(item.get("day", 0)), int(item.get("seed", 0)))


def _env_id(item: Mapping[str, Any]) -> str:
    return f"TomatoEnv_y{int(item.get('year', 0))}_d{int(item.get('day', 0))}_s{int(item.get('seed', 0))}"


def _cache_report(item: Mapping[str, Any], coverage_reports: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    if not coverage_reports:
        return {"cache_coverage_checked": False, "cache_coverage_pass": True, "coverage_issues": []}
    env_id = _env_id(item)
    matches: list[Mapping[str, Any]] = []
    for report in coverage_reports:
        envs = report.get("envs", {}) if isinstance(report.get("envs"), Mapping) else {}
        row = envs.get(env_id)
        if isinstance(row, Mapping):
            matches.append(row)
    if not matches:
        return {"cache_coverage_checked": True, "cache_coverage_pass": False, "coverage_issues": ["coverage_env_missing"]}
    issues: list[str] = []
    for row in matches:
        issues.extend(str(issue) for issue in row.get("issues", []) or [])
    return {
        "cache_coverage_checked": True,
        "cache_coverage_pass": any(bool(row.get("ok", False)) for row in matches),
        "coverage_issues": sorted(set(issues)),
    }


def _trace_path_for(trace_dir: Path, scenario_id: str, controller: str) -> Path:
    expected = trace_dir / f"{scenario_id}_{controller}.csv"
    if expected.exists():
        return expected
    parts = _parts_from_scenario_id(scenario_id)
    prefix = f"y{parts['year']}_d{parts['day']}_s{parts['seed']}_"
    matches = sorted(trace_dir.glob(f"{prefix}*_{controller}.csv"))
    return matches[0] if matches else expected


def _trace_report(path: Path, horizon_steps: int) -> dict[str, Any]:
    row_count = 0
    last_step: int | None = None
    runtime_error_count = 0
    cvodes_failure_count = 0
    if not path.exists():
        return {
            "trace_path": str(path),
            "trace_exists": False,
            "row_count": 0,
            "last_step": None,
            "runtime_error_count": 0,
            "cvodes_failure_count": 0,
            "short_trace_before_horizon": True,
        }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            step = int(_num(row.get("step"), row_count - 1))
            last_step = step
            error_text = " ".join(
                str(row.get(key, "") or "")
                for key in ("runtime_error", "runtime_error_type", "replan_reason")
            )
            if row.get("runtime_error") or row.get("runtime_error_type") or row.get("replan_reason") == "runtime_error":
                runtime_error_count += 1
                if "CV_CONV_FAILURE" in error_text or "CvodesInterface" in error_text or "CVODES" in error_text:
                    cvodes_failure_count += 1
    return {
        "trace_path": str(path),
        "trace_exists": True,
        "row_count": row_count,
        "last_step": last_step,
        "runtime_error_count": runtime_error_count,
        "cvodes_failure_count": cvodes_failure_count,
        "short_trace_before_horizon": last_step is None or last_step < horizon_steps - 1,
    }


def _controller_pass(report: Mapping[str, Any]) -> bool:
    return bool(
        report.get("trace_exists", False)
        and not report.get("short_trace_before_horizon", True)
        and int(report.get("runtime_error_count", 0) or 0) == 0
    )


def _attribution(baseline: Mapping[str, Any], default: Mapping[str, Any]) -> str:
    baseline_pass = _controller_pass(baseline)
    default_pass = _controller_pass(default)
    if baseline_pass and default_pass:
        return "stable_long_horizon_evaluation_candidate"
    if not baseline_pass and not default_pass:
        return "simulator_or_weather_numerical_instability"
    if baseline_pass and not default_pass:
        return "default_controller_action_induced_instability"
    return "baseline_specific_instability_default_stable"


def build_report(
    *,
    execution_record: Mapping[str, Any],
    trace_dir: str | Path,
    coverage_reports: Sequence[Mapping[str, Any]] | None = None,
    horizon_steps: int | None = None,
) -> dict[str, Any]:
    horizon = int(horizon_steps or execution_record.get("horizon_steps", 0) or 0)
    trace_root = _resolve(trace_dir)
    scenario_reports: list[dict[str, Any]] = []
    for raw in execution_record.get("eligible_scenarios", []) or []:
        if not isinstance(raw, Mapping):
            continue
        sid = str(raw.get("scenario_id", ""))
        ppo = _trace_report(_trace_path_for(trace_root, sid, "ppo"), horizon)
        llm = _trace_report(_trace_path_for(trace_root, sid, "llm_rspc_v2"), horizon)
        cache = _cache_report(raw, coverage_reports)
        attribution = _attribution(ppo, llm)
        stable = attribution == "stable_long_horizon_evaluation_candidate" and bool(cache["cache_coverage_pass"])
        scenario_reports.append(
            {
                "scenario_id": sid,
                "year": int(raw.get("year", 0)),
                "day": int(raw.get("day", 0)),
                "seed": int(raw.get("seed", 0)),
                "regime": str(raw.get("regime", "")),
                "source_tier": str(raw.get("source_tier", "")),
                "horizon_steps": horizon,
                "baseline_controller": "ppo",
                "default_controller": "llm_rspc_v2",
                "baseline_report": ppo,
                "default_report": llm,
                "default_cache_report": cache,
                "attribution": attribution,
                "stable_pool_candidate": stable,
            }
        )
    stable_reports = [item for item in scenario_reports if item["stable_pool_candidate"]]
    failure_taxonomy: list[str] = []
    if any(item["attribution"] == "simulator_or_weather_numerical_instability" for item in scenario_reports):
        failure_taxonomy.append("simulator_or_weather_numerical_instability")
    if any(item["attribution"] == "default_controller_action_induced_instability" for item in scenario_reports):
        failure_taxonomy.append("default_controller_action_induced_instability")
    if any(item["attribution"] == "baseline_specific_instability_default_stable" for item in scenario_reports):
        failure_taxonomy.append("baseline_specific_instability_default_stable")
    if any(not item["default_cache_report"].get("cache_coverage_pass", False) for item in scenario_reports):
        failure_taxonomy.append("cache_coverage_incomplete")
    if not stable_reports:
        failure_taxonomy.append("no_stable_long_horizon_candidate")
    return {
        "schema_version": "stable_long_horizon_evaluation_pool_result_audit_20260529_v38",
        "horizon_steps": horizon,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "scenario_reports": scenario_reports,
        "stable_scenario_reports": stable_reports,
        "stable_scenario_ids": [item["scenario_id"] for item in stable_reports],
        "stable_base_keys": [[item["year"], item["day"], item["seed"]] for item in stable_reports],
        "metrics": {
            "scenario_count": len(scenario_reports),
            "stable_scenario_count": len(stable_reports),
            "simulator_or_weather_numerical_instability_count": sum(
                1 for item in scenario_reports if item["attribution"] == "simulator_or_weather_numerical_instability"
            ),
            "default_controller_action_induced_instability_count": sum(
                1 for item in scenario_reports if item["attribution"] == "default_controller_action_induced_instability"
            ),
            "baseline_specific_instability_default_stable_count": sum(
                1 for item in scenario_reports if item["attribution"] == "baseline_specific_instability_default_stable"
            ),
            "baseline_runtime_error_count": sum(
                int(item["baseline_report"].get("runtime_error_count", 0) or 0) for item in scenario_reports
            ),
            "default_runtime_error_count": sum(
                int(item["default_report"].get("runtime_error_count", 0) or 0) for item in scenario_reports
            ),
            "cache_coverage_incomplete_count": sum(
                1 for item in scenario_reports if not item["default_cache_report"].get("cache_coverage_pass", False)
            ),
            "cvodes_failure_count": sum(
                int(item["baseline_report"].get("cvodes_failure_count", 0) or 0)
                + int(item["default_report"].get("cvodes_failure_count", 0) or 0)
                for item in scenario_reports
            ),
        },
        "failure_taxonomy": sorted(set(failure_taxonomy)),
        "next_action": (
            "build_v38_h720_execution_record"
            if horizon == 240 and stable_reports
            else "build_v38_h1440_execution_record"
            if horizon == 720 and stable_reports
            else "full_cycle_shadow_benchmark_design"
            if horizon == 1440 and stable_reports
            else "simulator_stability_diagnosis_or_weather_space_redesign"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stable Long-Horizon Evaluation Pool Result Audit v38",
        "",
        f"- Horizon steps: `{report.get('horizon_steps')}`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in dict(report.get("metrics", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Scenario Attribution", "", "| scenario | attribution | stable | cache | ppo_last | llm_last | ppo_err | llm_err |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for item in report.get("scenario_reports", []) or []:
        ppo = item.get("baseline_report", {}) or {}
        llm = item.get("default_report", {}) or {}
        cache = item.get("default_cache_report", {}) or {}
        lines.append(
            f"| `{item.get('scenario_id', '')}` | `{item.get('attribution', '')}` | "
            f"`{item.get('stable_pool_candidate', False)}` | `{cache.get('cache_coverage_pass')}` | "
            f"`{ppo.get('last_step')}` | `{llm.get('last_step')}` | "
            f"`{ppo.get('runtime_error_count')}` | `{llm.get('runtime_error_count')}` |"
        )
    lines.extend(["", "## Failure Taxonomy", ""])
    failures = report.get("failure_taxonomy", []) or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- none")
    return "\n".join(lines) + "\n"


def build_stable_pool(report: Mapping[str, Any]) -> dict[str, Any]:
    stable = list(report.get("stable_scenario_reports", []) or [])
    return {
        "schema_version": "stable_long_horizon_evaluation_pool_20260529_v38",
        "horizon_steps": int(report.get("horizon_steps", 0) or 0),
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "stable_scenario_count": len(stable),
        "stable_scenario_ids": [str(item.get("scenario_id", "")) for item in stable],
        "stable_scenarios": [
            {
                "scenario_id": item.get("scenario_id"),
                "year": item.get("year"),
                "day": item.get("day"),
                "seed": item.get("seed"),
                "max_steps": item.get("horizon_steps"),
                "regime": item.get("regime"),
                "source_tier": item.get("source_tier"),
            }
            for item in stable
        ],
        "next_action": report.get("next_action"),
    }


def build_stable_pool_markdown(pool: Mapping[str, Any]) -> str:
    lines = [
        "# Stable Long-Horizon Evaluation Pool v38",
        "",
        f"- Horizon steps: `{pool.get('horizon_steps')}`",
        f"- Stable scenario count: `{pool.get('stable_scenario_count')}`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "",
        "| scenario | regime | source |",
        "| --- | --- | --- |",
    ]
    for item in pool.get("stable_scenarios", []) or []:
        lines.append(f"| `{item.get('scenario_id', '')}` | `{item.get('regime', '')}` | `{item.get('source_tier', '')}` |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-record-json", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--coverage-json", action="append", default=[])
    parser.add_argument("--horizon-steps", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--stable-pool-json", default="")
    parser.add_argument("--stable-pool-md", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        execution_record=_load(args.execution_record_json),
        trace_dir=args.trace_dir,
        coverage_reports=[_load(path) for path in args.coverage_json],
        horizon_steps=args.horizon_steps or None,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    if args.stable_pool_json:
        stable_pool_json = _resolve(args.stable_pool_json)
        stable_pool_json.parent.mkdir(parents=True, exist_ok=True)
        stable_pool_json.write_text(json.dumps(build_stable_pool(report), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {stable_pool_json}")
    if args.stable_pool_md:
        stable_pool_md = _resolve(args.stable_pool_md)
        stable_pool_md.parent.mkdir(parents=True, exist_ok=True)
        stable_pool_md.write_text(build_stable_pool_markdown(build_stable_pool(report)), encoding="utf-8")
        print(f"wrote {stable_pool_md}")
    print(f"horizon_steps={report['horizon_steps']}")
    print(f"stable_scenario_count={report['metrics']['stable_scenario_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
