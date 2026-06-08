"""Build v36 readiness for weather-sourced shadow cache acquisition."""

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


def _load_optional(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = _resolve(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _coverage_pass(coverage: Mapping[str, Any] | None) -> bool:
    if not coverage:
        return False
    return bool(
        coverage.get("cache_coverage_pass", False)
        and _num(coverage.get("coverage_rate")) == 1.0
        and int(_num(coverage.get("missing_key_count"))) == 0
        and int(_num(coverage.get("missing_buffered_action_count"))) == 0
        and int(_num(coverage.get("missing_parsed_plan_count"))) == 0
    )


def build_report(
    *,
    manifest: Mapping[str, Any],
    execution_record: Mapping[str, Any] | None = None,
    cache_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_ready = bool(manifest.get("weather_shadow_cache_acquisition_manifest_ready", False))
    execution_authorized = bool(execution_record and execution_record.get("weather_shadow_cache_acquisition_authorized", False))
    execution_executable = bool(execution_record and execution_record.get("weather_shadow_cache_acquisition_executable", False))
    coverage_ok = _coverage_pass(cache_coverage)
    failure_taxonomy: list[str] = []
    if not manifest_ready:
        failure_taxonomy.append("manifest_not_ready")
    if execution_record:
        failure_taxonomy.extend(str(item) for item in execution_record.get("failure_taxonomy", []) or [])
    if cache_coverage and not coverage_ok:
        failure_taxonomy.append("shadow_cache_coverage_incomplete")
    if coverage_ok:
        next_action = "shadow_only_strict_metadata_replay_admission_for_v36_cache"
    elif execution_record and not execution_executable:
        next_action = str(execution_record.get("next_action", "v36_shadow_cache_acquisition_precheck_blocked"))
    elif execution_executable and not cache_coverage:
        next_action = "execute_v36_weather_shadow_cache_acquisition"
    elif cache_coverage and not coverage_ok:
        next_action = "shadow_cache_acquisition_failure_diagnosis"
    else:
        next_action = "v36_shadow_cache_acquisition_readiness_blocked"
    return {
        "schema_version": "metadata_replay_readiness_checklist_20260529_v36",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "v36 weather-sourced shadow cache acquisition readiness",
            "reopens_rejected_preset": False,
        },
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "weather_proxy_source_hint_only": True,
        "weather_proxy_is_strict_applied_evidence": False,
        "cache_fill_authorized": bool(execution_record and execution_record.get("cache_fill_authorized", False)),
        "online_llm_allowed": bool(execution_record and execution_record.get("online_llm_allowed", False)),
        "cache_fill_run": bool(cache_coverage is not None and coverage_ok),
        "online_llm_called": bool(cache_coverage is not None and coverage_ok),
        "weather_shadow_cache_acquisition_manifest_ready": manifest_ready,
        "weather_shadow_cache_acquisition_authorized": execution_authorized,
        "weather_shadow_cache_acquisition_executable": execution_executable,
        "shadow_cache_coverage_pass": coverage_ok,
        "isolated_cache_path": str(manifest.get("isolated_cache_path", "")),
        "scenario_count": int(_num(manifest.get("scenario_count"))),
        "metrics": {
            "requested_scenario_count": int(_num(manifest.get("scenario_count"))),
            "coverage_rate": _num(cache_coverage.get("coverage_rate") if cache_coverage else 0.0),
            "missing_key_count": int(_num(cache_coverage.get("missing_key_count") if cache_coverage else 0)),
            "missing_buffered_action_count": int(_num(cache_coverage.get("missing_buffered_action_count") if cache_coverage else 0)),
            "missing_parsed_plan_count": int(_num(cache_coverage.get("missing_parsed_plan_count") if cache_coverage else 0)),
        },
        "failure_taxonomy": sorted(set(item for item in failure_taxonomy if item)),
        "next_action": next_action,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Metadata Replay Readiness Checklist v36",
        "",
        "- Mode: weather-sourced shadow cache acquisition readiness",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Controlled replay execution allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Cache fill authorized: {report.get('cache_fill_authorized', False)}",
        f"- Online LLM allowed: {report.get('online_llm_allowed', False)}",
        f"- Shadow cache coverage pass: {report.get('shadow_cache_coverage_pass', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("metrics", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Failure Taxonomy", ""])
    failures = report.get("failure_taxonomy", []) or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--execution-record-json", default="")
    parser.add_argument("--cache-coverage-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        manifest=_load_optional(args.manifest_json) or {},
        execution_record=_load_optional(args.execution_record_json),
        cache_coverage=_load_optional(args.cache_coverage_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"weather_shadow_cache_acquisition_authorized={report['weather_shadow_cache_acquisition_authorized']}")
    print(f"shadow_cache_coverage_pass={report['shadow_cache_coverage_pass']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
