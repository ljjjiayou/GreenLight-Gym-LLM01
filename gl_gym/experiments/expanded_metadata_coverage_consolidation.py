"""Consolidate strict metadata replay evidence before controlled canary admission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_PHASES = (
    "canonical_stage_b_rerun",
    "first_wave_expanded_metadata",
    "second_wave_expanded_metadata",
)


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


def _phase(report: Mapping[str, Any]) -> str:
    schema = str(report.get("schema_version", ""))
    checks = report.get("checks", {}) if isinstance(report.get("checks", {}), Mapping) else {}
    if schema.endswith("_v19") or bool(checks.get("second_wave_expanded_metadata_replay_authorized", False)):
        return "second_wave_expanded_metadata"
    if schema.endswith("_v18") or bool(checks.get("first_wave_expanded_metadata_replay_authorized", False)):
        return "first_wave_expanded_metadata"
    if schema.endswith("_v17") or bool(checks.get("stage_b_rerun_authorized", False)):
        return "canonical_stage_b_rerun"
    return schema or "unknown"


def _metrics_pass(metrics: Mapping[str, Any]) -> bool:
    return bool(
        _num(metrics.get("cache_hit_rate")) == 1.0
        and _num(metrics.get("runtime_error_steps")) == 0.0
        and _num(metrics.get("strict_cache_miss_runtime_error_steps")) == 0.0
        and _num(metrics.get("action_diff_steps")) == 0.0
        and _num(metrics.get("max_abs_delta")) == 0.0
        and _num(metrics.get("runtime_provenance_record_count")) > 0.0
        and _num(metrics.get("runtime_reason_missing_count")) == 0.0
        and _num(metrics.get("unknown_post_guardrail_rewrite_count")) == 0.0
        and _num(metrics.get("joint_prediction_missing_fields")) == 0.0
    )


def _evidence_item(report: Mapping[str, Any], source_path: str = "") -> dict[str, Any]:
    metrics = report.get("post_run_metrics", {}) if isinstance(report.get("post_run_metrics", {}), Mapping) else {}
    failures = list(report.get("failure_taxonomy", []) or [])
    passed = bool(report.get("canonical_strict_metadata_replay_pass", False)) and not failures and _metrics_pass(metrics)
    checks = report.get("checks", {}) if isinstance(report.get("checks", {}), Mapping) else {}
    return {
        "source_path": source_path,
        "schema_version": str(report.get("schema_version", "")),
        "phase": _phase(report),
        "next_action": str(report.get("next_action", "")),
        "evidence_pass": passed,
        "failure_taxonomy": failures,
        "scenario_count": int(
            checks.get("second_wave_expanded_scenario_count")
            or checks.get("first_wave_expanded_scenario_count")
            or 1
        ),
        "metrics": {
            "cache_hit_rate": _num(metrics.get("cache_hit_rate")),
            "runtime_error_steps": _num(metrics.get("runtime_error_steps")),
            "strict_cache_miss_runtime_error_steps": _num(metrics.get("strict_cache_miss_runtime_error_steps")),
            "action_diff_steps": _num(metrics.get("action_diff_steps")),
            "max_abs_delta": _num(metrics.get("max_abs_delta")),
            "runtime_provenance_record_count": _num(metrics.get("runtime_provenance_record_count")),
            "runtime_reason_missing_count": _num(metrics.get("runtime_reason_missing_count")),
            "unknown_post_guardrail_rewrite_count": _num(metrics.get("unknown_post_guardrail_rewrite_count")),
            "joint_prediction_missing_fields": _num(metrics.get("joint_prediction_missing_fields")),
            "joint_prediction_row_count": _num(metrics.get("joint_prediction_row_count")),
        },
    }


def build_report(readiness_reports: Sequence[Mapping[str, Any]], source_paths: Sequence[str] | None = None) -> dict[str, Any]:
    source_paths = list(source_paths or [])
    evidence: list[dict[str, Any]] = []
    for index, report in enumerate(readiness_reports):
        path = source_paths[index] if index < len(source_paths) else ""
        evidence.append(_evidence_item(report, path))
    by_phase = {item["phase"]: item for item in evidence}
    missing_phases = [phase for phase in REQUIRED_PHASES if phase not in by_phase]
    failed_phases = [item["phase"] for item in evidence if not item["evidence_pass"]]
    consolidation_pass = not missing_phases and not failed_phases
    return {
        "schema_version": "expanded_metadata_coverage_consolidation_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "expanded metadata evidence consolidation",
            "reopens_rejected_preset": False,
        },
        "expanded_metadata_consolidation_pass": consolidation_pass,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "metadata_replay_execution_allowed": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "required_phases": list(REQUIRED_PHASES),
        "covered_phases": sorted(by_phase),
        "missing_phases": missing_phases,
        "failed_phases": failed_phases,
        "evidence": evidence,
        "aggregate": {
            "evidence_count": len(evidence),
            "scenario_count": int(sum(int(item.get("scenario_count", 0) or 0) for item in evidence)),
            "cache_hit_rate_min": min((item["metrics"]["cache_hit_rate"] for item in evidence), default=0.0),
            "runtime_provenance_record_count": int(
                sum(int(item["metrics"]["runtime_provenance_record_count"]) for item in evidence)
            ),
            "joint_prediction_row_count": int(
                sum(int(item["metrics"]["joint_prediction_row_count"]) for item in evidence)
            ),
        },
        "next_action": (
            "controlled_replay_admission_review"
            if consolidation_pass
            else "expanded_metadata_evidence_gap_diagnosis"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Expanded Metadata Coverage Consolidation",
        "",
        "- Mode: expanded strict metadata replay evidence consolidation",
        "- Default `llm_rspc_v2` changed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Consolidation pass: {report.get('expanded_metadata_consolidation_pass', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Evidence",
        "",
        "| phase | pass | scenarios | cache hit | action diff | provenance records | joint missing |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.get("evidence", []) or []:
        if not isinstance(item, Mapping):
            continue
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics", {}), Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("phase", "")),
                    str(item.get("evidence_pass", False)),
                    str(item.get("scenario_count", 0)),
                    str(metrics.get("cache_hit_rate", 0.0)),
                    str(metrics.get("action_diff_steps", 0.0)),
                    str(metrics.get("runtime_provenance_record_count", 0)),
                    str(metrics.get("joint_prediction_missing_fields", 0)),
                ]
            )
            + " |"
        )
    if report.get("missing_phases"):
        lines.extend(["", "## Missing Phases", ""])
        lines.extend(f"- {phase}" for phase in report.get("missing_phases", []) or [])
    if report.get("failed_phases"):
        lines.extend(["", "## Failed Phases", ""])
        lines.extend(f"- {phase}" for phase in report.get("failed_phases", []) or [])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-json", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reports = [_load(path) for path in args.readiness_json]
    report = build_report(reports, args.readiness_json)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"expanded_metadata_consolidation_pass={report['expanded_metadata_consolidation_pass']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["expanded_metadata_consolidation_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
