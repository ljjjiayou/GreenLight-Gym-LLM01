"""Diagnose failed v36 weather-sourced shadow cache acquisition coverage."""

from __future__ import annotations

import argparse
import csv
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


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _trace_env_id(path: Path) -> str:
    stem = path.stem
    scenario = stem.rsplit("_llm_rspc_v2", 1)[0]
    scenario = scenario.rsplit("_n", 1)[0]
    return f"TomatoEnv_{scenario}"


def _trace_report(path: Path, max_steps: int) -> dict[str, Any]:
    rows = 0
    last_step: int | None = None
    runtime_errors: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            step = int(_num(row.get("step"), rows - 1))
            last_step = step
            if row.get("runtime_error") or row.get("runtime_error_type") or row.get("replan_reason") == "runtime_error":
                error_text = str(row.get("runtime_error", ""))
                runtime_errors.append(
                    {
                        "step": step,
                        "timestep": _num(row.get("timestep"), step),
                        "runtime_error_type": str(row.get("runtime_error_type", "")),
                        "rollout_source_before_error": str(row.get("rollout_source_before_error", "")),
                        "error_contains_cvodes_convergence_failure": "CV_CONV_FAILURE" in error_text,
                        "error_excerpt": error_text[:300],
                    }
                )
    return {
        "trace_path": str(path),
        "row_count": rows,
        "last_step": last_step,
        "short_trace_before_max_steps": last_step is not None and last_step < max_steps - 1,
        "runtime_error_count": len(runtime_errors),
        "runtime_errors": runtime_errors,
    }


def build_report(
    *,
    coverage: Mapping[str, Any],
    trace_dir: str | Path,
    max_steps: int = 240,
) -> dict[str, Any]:
    trace_root = _resolve(trace_dir)
    coverage_envs = coverage.get("envs", {}) if isinstance(coverage.get("envs"), Mapping) else {}
    env_reports: dict[str, dict[str, Any]] = {}
    for trace_path in sorted(trace_root.glob("*.csv")):
        env_id = _trace_env_id(trace_path)
        trace = _trace_report(trace_path, max_steps=max_steps)
        coverage_report = coverage_envs.get(env_id, {}) if isinstance(coverage_envs, Mapping) else {}
        env_reports[env_id] = {
            **trace,
            "coverage_entry_count": int(_num(coverage_report.get("entry_count"))),
            "coverage_required_min_entries": int(_num(coverage_report.get("required_min_entries"))),
            "coverage_max_timestep": coverage_report.get("max_timestep"),
            "coverage_issues": list(coverage_report.get("issues", []) or []),
        }

    runtime_error_count = sum(int(item["runtime_error_count"]) for item in env_reports.values())
    cvodes_failure_count = sum(
        1
        for item in env_reports.values()
        for error in item.get("runtime_errors", [])
        if error.get("error_contains_cvodes_convergence_failure")
    )
    short_trace_count = sum(1 for item in env_reports.values() if item.get("short_trace_before_max_steps"))
    failure_taxonomy: list[str] = []
    if not coverage.get("cache_coverage_pass", False):
        failure_taxonomy.append("shadow_cache_coverage_incomplete")
    if int(_num(coverage.get("missing_key_count"))) > 0:
        failure_taxonomy.append("cache_low_coverage")
    if runtime_error_count:
        failure_taxonomy.append("runtime_error_during_shadow_cache_acquisition")
    if cvodes_failure_count:
        failure_taxonomy.append("cvodes_convergence_failure")
    if short_trace_count:
        failure_taxonomy.append("short_trace_before_max_steps")

    return {
        "schema_version": "controlled_canary_weather_shadow_cache_acquisition_failure_diagnosis_20260529_v36",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "v36 shadow cache acquisition failure diagnosis",
        },
        "shadow_cache_coverage_pass": bool(coverage.get("cache_coverage_pass", False)),
        "strict_metadata_replay_admission_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "metrics": {
            "coverage_rate": _num(coverage.get("coverage_rate")),
            "missing_key_count": int(_num(coverage.get("missing_key_count"))),
            "missing_buffered_action_count": int(_num(coverage.get("missing_buffered_action_count"))),
            "missing_parsed_plan_count": int(_num(coverage.get("missing_parsed_plan_count"))),
            "runtime_error_trace_count": runtime_error_count,
            "cvodes_convergence_failure_count": cvodes_failure_count,
            "short_trace_count": short_trace_count,
            "env_count": len(env_reports),
        },
        "env_reports": env_reports,
        "failure_taxonomy": sorted(set(failure_taxonomy)),
        "next_action": "weather_source_screening_or_simulator_stability_diagnosis_before_new_cache_acquisition",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    metrics = dict(report.get("metrics", {}))
    lines = [
        "# v36 Weather Shadow Cache Acquisition Failure Diagnosis",
        "",
        "- Strict metadata replay admission allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Failure Taxonomy", ""])
    failures = report.get("failure_taxonomy", []) or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- none")
    lines.extend(["", "## Env Reports", "", "| env_id | last_step | cache_entries | coverage_issues | runtime_error |", "| --- | ---: | ---: | --- | --- |"])
    for env_id, item in dict(report.get("env_reports", {})).items():
        issues = ",".join(str(x) for x in item.get("coverage_issues", []))
        runtime = "yes" if int(item.get("runtime_error_count", 0)) else "no"
        lines.append(
            f"| `{env_id}` | `{item.get('last_step')}` | `{item.get('coverage_entry_count')}` | `{issues}` | `{runtime}` |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        coverage=_load_json(args.coverage_json),
        trace_dir=args.trace_dir,
        max_steps=args.max_steps,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"shadow_cache_coverage_pass={report['shadow_cache_coverage_pass']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
