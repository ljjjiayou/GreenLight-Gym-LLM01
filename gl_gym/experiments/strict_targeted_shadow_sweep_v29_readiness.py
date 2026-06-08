"""Build readiness v29 for the cache-covered strict-targeted shadow-only sweep."""

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
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _coverage_pass(coverage: Mapping[str, Any] | None) -> bool:
    return bool(
        coverage
        and coverage.get("cache_coverage_pass", False)
        and _num(coverage.get("coverage_rate")) == 1.0
        and int(_num(coverage.get("missing_key_count"))) == 0
        and int(_num(coverage.get("missing_buffered_action_count"))) == 0
        and int(_num(coverage.get("missing_parsed_plan_count"))) == 0
    )


def _aggregate(report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(report, Mapping):
        return {}
    value = report.get("aggregate", {})
    return value if isinstance(value, Mapping) else {}


def build_report(
    *,
    manifest: Mapping[str, Any],
    cache_coverage: Mapping[str, Any] | None = None,
    execution_record: Mapping[str, Any] | None = None,
    strict_sourcing: Mapping[str, Any] | None = None,
    summary_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_ready = bool(manifest.get("strict_targeted_shadow_sweep_manifest_ready", False))
    cache_ok = _coverage_pass(cache_coverage)
    execution_authorized = bool(execution_record and execution_record.get("strict_targeted_shadow_sweep_v29_authorized", False))
    sourcing_run = isinstance(strict_sourcing, Mapping)
    source_found = bool(strict_sourcing and strict_sourcing.get("strict_targeted_source_found", False))
    expected_steps = int(_num(strict_sourcing.get("expected_strict_eligible_applied_steps") if strict_sourcing else 0))
    summary_agg = _aggregate(summary_audit)
    summary_pass = summary_agg.get("decision") == "pass" if summary_audit else False

    failure_taxonomy: list[str] = []
    if not manifest_ready:
        failure_taxonomy.append("manifest_missing_or_incomplete")
    if cache_coverage and not cache_ok:
        failure_taxonomy.append("cache_mismatch")
    if sourcing_run and not source_found:
        failure_taxonomy.append("strict_targeted_source_exhausted_in_selected_cache")
        failure_taxonomy.extend(list(strict_sourcing.get("blocker_taxonomy", []) or []))
    if summary_audit and not summary_pass:
        failure_taxonomy.append("summary_gate_failed")

    if not manifest_ready:
        next_action = "v29_manifest_required"
    elif not cache_coverage:
        next_action = "run_v29_no_fill_cache_coverage_precheck"
    elif not cache_ok:
        next_action = "repair_or_expand_selected_cache_before_v29_sweep"
    elif not execution_authorized:
        next_action = "v29_shadow_sweep_execution_record_required"
    elif not sourcing_run:
        next_action = "execute_v29_shadow_sweep_and_strict_targeted_sourcing"
    elif source_found:
        next_action = "strict_targeted_source_manifest_and_no_fill_cache_precheck"
    else:
        next_action = "new_scenario_cache_discovery_for_strict_targeted_source"

    return {
        "schema_version": "metadata_replay_readiness_checklist_v29",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "cache-covered strict-targeted shadow-only sweep readiness",
            "reopens_rejected_preset": False,
        },
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "v29_manifest_ready": manifest_ready,
        "v29_cache_coverage_pass": cache_ok,
        "v29_shadow_sweep_authorized": execution_authorized,
        "v29_shadow_sweep_executed": bool(execution_authorized and sourcing_run),
        "strict_targeted_source_found": source_found,
        "strict_targeted_source_manifest_ready": source_found,
        "post_run_metrics": {
            "scenario_count": int(_num(manifest.get("scenario_count"))),
            "expected_env_count": int(_num(cache_coverage.get("expected_env_count") if cache_coverage else 0)),
            "coverage_rate": _num(cache_coverage.get("coverage_rate") if cache_coverage else 0.0),
            "missing_key_count": int(_num(cache_coverage.get("missing_key_count") if cache_coverage else 0)),
            "missing_buffered_action_count": int(_num(cache_coverage.get("missing_buffered_action_count") if cache_coverage else 0)),
            "missing_parsed_plan_count": int(_num(cache_coverage.get("missing_parsed_plan_count") if cache_coverage else 0)),
            "expected_strict_eligible_applied_steps": expected_steps,
            "would_apply_steps": int(_num(strict_sourcing.get("would_apply_steps") if strict_sourcing else 0)),
            "strict_filtered_steps": int(_num(strict_sourcing.get("strict_filtered_steps") if strict_sourcing else 0)),
        },
        "failure_taxonomy": sorted(set(failure_taxonomy)),
        "next_action": next_action,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Metadata Replay Readiness Checklist v29",
        "",
        "- Mode: cache-covered strict-targeted shadow-only sweep readiness",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Controlled replay execution allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Cache coverage pass: {report.get('v29_cache_coverage_pass', False)}",
        f"- Strict-targeted source found: {report.get('strict_targeted_source_found', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Metrics",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("post_run_metrics", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Failure Taxonomy", ""])
    failures = report.get("failure_taxonomy", []) or []
    lines.extend(f"- `{item}`" for item in failures) if failures else lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--cache-coverage-json", default="")
    parser.add_argument("--execution-record-json", default="")
    parser.add_argument("--strict-sourcing-json", default="")
    parser.add_argument("--summary-audit-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        manifest=_load_optional(args.manifest_json) or {},
        cache_coverage=_load_optional(args.cache_coverage_json),
        execution_record=_load_optional(args.execution_record_json),
        strict_sourcing=_load_optional(args.strict_sourcing_json),
        summary_audit=_load_optional(args.summary_audit_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"v29_cache_coverage_pass={report['v29_cache_coverage_pass']}")
    print(f"strict_targeted_source_found={report['strict_targeted_source_found']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
