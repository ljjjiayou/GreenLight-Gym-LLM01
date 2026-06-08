"""Audit v37 long-horizon shadow feasibility traces and cache coverage."""

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


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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


def _env_id(item: Mapping[str, Any]) -> str:
    return f"TomatoEnv_y{int(item.get('year', 0))}_d{int(item.get('day', 0))}_s{int(item.get('seed', 0))}"


def _trace_path_for(trace_dir: Path, scenario_id: str) -> Path:
    expected = f"{scenario_id}_llm_rspc_v2.csv"
    path = trace_dir / expected
    if path.exists():
        return path
    parts = _parts_from_scenario_id(scenario_id)
    fallback_prefix = f"y{parts['year']}_d{parts['day']}_s{parts['seed']}_"
    matches = sorted(trace_dir.glob(f"{fallback_prefix}*_llm_rspc_v2.csv"))
    return matches[0] if matches else path


def _strict_source_row(row: Mapping[str, Any]) -> bool:
    candidate = str(
        row.get("rspc_hot_dry_replay_best_candidate_name")
        or row.get("rspc_hot_dry_proposer_control_candidate")
        or ""
    )
    safe = _truthy(row.get("rspc_hot_dry_replay_safe_hot_dry")) or _truthy(row.get("rspc_hot_dry_proposer_control_safe_hot_dry"))
    unsafe = _truthy(row.get("rspc_hot_dry_replay_unsafe_preferred")) or _truthy(row.get("rspc_hot_dry_replay_unsafe_conflict"))
    margin = _num(row.get("rspc_hot_dry_replay_best_margin") or row.get("rspc_hot_dry_proposer_control_margin"))
    rh = _num(row.get("rh_air") or row.get("rspc_hot_dry_proposer_control_rh_air"), 100.0)
    vpd = _num(row.get("vpd_air") or row.get("rspc_hot_dry_proposer_control_vpd_air"))
    temp = _num(row.get("temp_air") or row.get("rspc_hot_dry_proposer_control_temp_air"), 99.0)
    canopy = _num(row.get("canopy_dew_margin") or row.get("rspc_hot_dry_proposer_control_canopy_dew_margin"), -99.0)
    return bool(
        candidate == "shadow_hot_dry_humidity_retention"
        and safe
        and not unsafe
        and margin >= 0.20
        and rh <= 45.0
        and vpd >= 2.25
        and temp <= 30.5
        and canopy >= 3.0
    )


def _trace_report(path: Path, horizon_steps: int) -> dict[str, Any]:
    row_count = 0
    last_step: int | None = None
    runtime_error_count = 0
    cvodes_failure_count = 0
    shadow_signal_count = 0
    strict_source_candidate_count = 0
    if not path.exists():
        return {
            "trace_path": str(path),
            "trace_exists": False,
            "row_count": 0,
            "last_step": None,
            "runtime_error_count": 0,
            "cvodes_failure_count": 0,
            "short_trace_before_horizon": True,
            "shadow_signal_count": 0,
            "strict_source_candidate_count": 0,
        }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            step = int(_num(row.get("step"), row_count - 1))
            last_step = step
            error_text = str(row.get("runtime_error", "") or "")
            if row.get("runtime_error") or row.get("runtime_error_type") or row.get("replan_reason") == "runtime_error":
                runtime_error_count += 1
                if "CV_CONV_FAILURE" in error_text or "CvodesInterface" in error_text:
                    cvodes_failure_count += 1
            if (
                _truthy(row.get("rspc_hot_dry_replay_would_apply"))
                or _truthy(row.get("rspc_hot_dry_proposer_control_enabled"))
                or _truthy(row.get("rspc_action_hot_dry_shadow_active"))
            ):
                shadow_signal_count += 1
            if _strict_source_row(row):
                strict_source_candidate_count += 1
    return {
        "trace_path": str(path),
        "trace_exists": True,
        "row_count": row_count,
        "last_step": last_step,
        "runtime_error_count": runtime_error_count,
        "cvodes_failure_count": cvodes_failure_count,
        "short_trace_before_horizon": last_step is None or last_step < horizon_steps - 1,
        "shadow_signal_count": shadow_signal_count,
        "strict_source_candidate_count": strict_source_candidate_count,
    }


def build_report(
    *,
    execution_record: Mapping[str, Any],
    coverage: Mapping[str, Any],
    trace_dir: str | Path,
    horizon_steps: int | None = None,
) -> dict[str, Any]:
    horizon = int(horizon_steps or execution_record.get("horizon_steps", 0) or coverage.get("max_steps", 0))
    trace_root = _resolve(trace_dir)
    coverage_envs = coverage.get("envs", {}) if isinstance(coverage.get("envs"), Mapping) else {}
    scenario_reports: list[dict[str, Any]] = []
    for raw in execution_record.get("eligible_scenarios", []) or []:
        if not isinstance(raw, Mapping):
            continue
        scenario_id = str(raw.get("scenario_id", ""))
        trace = _trace_report(_trace_path_for(trace_root, scenario_id), horizon)
        env_report = coverage_envs.get(_env_id(raw), {}) if isinstance(coverage_envs, Mapping) else {}
        cache_ok = bool(env_report.get("ok", False))
        scenario_pass = bool(
            cache_ok
            and trace["trace_exists"]
            and not trace["short_trace_before_horizon"]
            and int(trace["runtime_error_count"]) == 0
        )
        scenario_reports.append(
            {
                "scenario_id": scenario_id,
                "year": int(raw.get("year", 0)),
                "day": int(raw.get("day", 0)),
                "seed": int(raw.get("seed", 0)),
                "horizon_steps": horizon,
                "cache_coverage_pass": cache_ok,
                "coverage_issues": list(env_report.get("issues", []) or []),
                **trace,
                "horizon_pass": scenario_pass,
            }
        )
    pass_reports = [item for item in scenario_reports if item.get("horizon_pass")]
    failure_taxonomy: list[str] = []
    if not coverage.get("cache_coverage_pass", False):
        failure_taxonomy.append("cache_coverage_incomplete")
    if any(not item["trace_exists"] for item in scenario_reports):
        failure_taxonomy.append("trace_missing")
    if any(item["short_trace_before_horizon"] for item in scenario_reports):
        failure_taxonomy.append("short_trace_before_horizon")
    if any(int(item["runtime_error_count"]) > 0 for item in scenario_reports):
        failure_taxonomy.append("runtime_error")
    if any(int(item["cvodes_failure_count"]) > 0 for item in scenario_reports):
        failure_taxonomy.append("cvodes_or_casadi_failure")
    if not pass_reports:
        failure_taxonomy.append("no_horizon_pass_scenario")
    return {
        "schema_version": "long_horizon_shadow_feasibility_result_audit_20260529_v37",
        "horizon_steps": horizon,
        "controller": "llm_rspc_v2",
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_coverage_pass": bool(coverage.get("cache_coverage_pass", False)),
        "scenario_reports": scenario_reports,
        "horizon_pass_scenario_ids": [str(item["scenario_id"]) for item in pass_reports],
        "horizon_pass_base_keys": [[item["year"], item["day"], item["seed"]] for item in pass_reports],
        "metrics": {
            "scenario_count": len(scenario_reports),
            "horizon_pass_count": len(pass_reports),
            "runtime_error_count": sum(int(item["runtime_error_count"]) for item in scenario_reports),
            "cvodes_failure_count": sum(int(item["cvodes_failure_count"]) for item in scenario_reports),
            "short_trace_count": sum(1 for item in scenario_reports if item["short_trace_before_horizon"]),
            "shadow_signal_count": sum(int(item["shadow_signal_count"]) for item in scenario_reports),
            "strict_source_candidate_count": sum(int(item["strict_source_candidate_count"]) for item in scenario_reports),
        },
        "failure_taxonomy": sorted(set(failure_taxonomy)),
        "next_action": (
            "build_v37_h720_execution_record"
            if horizon == 240 and pass_reports
            else "build_v37_h1440_execution_record"
            if horizon == 720 and pass_reports
            else "full_cycle_shadow_benchmark_design"
            if horizon == 1440 and pass_reports
            else "long_horizon_shadow_feasibility_blocked_by_runtime_or_coverage"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Long-Horizon Shadow Feasibility Result Audit v37",
        "",
        f"- Horizon steps: `{report.get('horizon_steps')}`",
        "- Controller: `llm_rspc_v2`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("metrics", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Scenario Reports", "", "| scenario | pass | last_step | runtime_errors | cvodes | cache_ok | shadow_signals | strict_sources |", "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |"])
    for item in report.get("scenario_reports", []) or []:
        lines.append(
            f"| `{item.get('scenario_id', '')}` | `{item.get('horizon_pass', False)}` | "
            f"`{item.get('last_step')}` | `{item.get('runtime_error_count')}` | "
            f"`{item.get('cvodes_failure_count')}` | `{item.get('cache_coverage_pass')}` | "
            f"`{item.get('shadow_signal_count')}` | `{item.get('strict_source_candidate_count')}` |"
        )
    lines.extend(["", "## Failure Taxonomy", ""])
    failures = report.get("failure_taxonomy", []) or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-record-json", required=True)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--horizon-steps", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        execution_record=_load(args.execution_record_json),
        coverage=_load(args.coverage_json),
        trace_dir=args.trace_dir,
        horizon_steps=args.horizon_steps or None,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"horizon_steps={report['horizon_steps']}")
    print(f"horizon_pass_count={report['metrics']['horizon_pass_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
