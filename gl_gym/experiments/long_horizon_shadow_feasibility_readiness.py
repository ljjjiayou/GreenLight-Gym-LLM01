"""Build v37 long-horizon shadow feasibility readiness."""

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


def build_report(
    *,
    manifest: Mapping[str, Any],
    execution_record: Mapping[str, Any] | None = None,
    result_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_ready = bool(manifest.get("long_horizon_shadow_feasibility_manifest_ready", False))
    executable = bool(execution_record and execution_record.get("long_horizon_shadow_feasibility_executable", False))
    horizon = int((result_audit or execution_record or {}).get("horizon_steps", 0) or 0)
    pass_count = int(((result_audit or {}).get("metrics", {}) or {}).get("horizon_pass_count", 0) or 0)
    failures: list[str] = []
    if not manifest_ready:
        failures.append("manifest_not_ready")
    if execution_record:
        failures.extend(str(item) for item in execution_record.get("failure_taxonomy", []) or [])
    if result_audit:
        failures.extend(str(item) for item in result_audit.get("failure_taxonomy", []) or [])
    if result_audit:
        next_action = str(result_audit.get("next_action", "long_horizon_shadow_feasibility_result_review"))
    elif execution_record and executable:
        next_action = str(execution_record.get("next_action", "execute_v37_shadow_feasibility"))
    elif execution_record:
        next_action = str(execution_record.get("next_action", "v37_long_horizon_execution_precheck_blocked"))
    else:
        next_action = "build_v37_h240_execution_record" if manifest_ready else "v37_manifest_blocked"
    return {
        "schema_version": "metadata_replay_readiness_checklist_20260529_v37",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only long-horizon feasibility readiness",
        },
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "long_horizon_shadow_feasibility_manifest_ready": manifest_ready,
        "long_horizon_shadow_feasibility_executable": executable,
        "horizon_steps": horizon,
        "horizon_pass_count": pass_count,
        "metrics": {
            "primary_scenario_count": len(manifest.get("primary_scenarios", []) or []),
            "blocked_v36_runtime_failed_count": len(manifest.get("blocked_runtime_failed_scenarios", []) or []),
            "horizon_steps": horizon,
            "horizon_pass_count": pass_count,
            "runtime_error_count": int(((result_audit or {}).get("metrics", {}) or {}).get("runtime_error_count", 0) or 0),
            "short_trace_count": int(((result_audit or {}).get("metrics", {}) or {}).get("short_trace_count", 0) or 0),
            "shadow_signal_count": int(((result_audit or {}).get("metrics", {}) or {}).get("shadow_signal_count", 0) or 0),
            "strict_source_candidate_count": int(((result_audit or {}).get("metrics", {}) or {}).get("strict_source_candidate_count", 0) or 0),
        },
        "failure_taxonomy": sorted(set(item for item in failures if item)),
        "next_action": next_action,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Metadata Replay Readiness Checklist v37",
        "",
        "- Mode: shadow-only long-horizon feasibility",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Controlled replay execution allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
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
    parser.add_argument("--result-audit-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        manifest=_load_optional(args.manifest_json) or {},
        execution_record=_load_optional(args.execution_record_json),
        result_audit=_load_optional(args.result_audit_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"horizon_steps={report['horizon_steps']}")
    print(f"horizon_pass_count={report['horizon_pass_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
