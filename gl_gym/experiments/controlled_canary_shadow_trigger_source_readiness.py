"""Build readiness v25 for shadow-only trigger source diagnosis."""

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


def build_report(
    *,
    readiness_v24: Mapping[str, Any],
    blocker_diagnosis: Mapping[str, Any],
    sourcing: Mapping[str, Any],
    source_manifest: Mapping[str, Any] | None = None,
    cache_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_found = bool(sourcing.get("shadow_trigger_sources_found", False))
    manifest_ready = bool(source_manifest and source_manifest.get("shadow_trigger_source_manifest_ready", False))
    coverage_pass = _coverage_pass(cache_coverage)
    if not blocker_diagnosis.get("blocker_diagnosis_complete", False):
        next_action = "independent_trigger_blocker_diagnosis_required"
    elif not source_found:
        next_action = "independent_trigger_source_exhausted_or_needs_new_shadow_sweep"
    elif not manifest_ready:
        next_action = "shadow_trigger_source_manifest_required"
    elif not coverage_pass:
        next_action = "shadow_trigger_source_cache_precheck_required"
    else:
        next_action = "shadow_only_trigger_source_sweep_design_required"
    return {
        "schema_version": "metadata_replay_readiness_checklist_v25",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only trigger source readiness",
            "reopens_rejected_preset": False,
        },
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "previous_wave2_controlled_canary_pass": bool(readiness_v24.get("wave2_controlled_canary_pass", False)),
        "blocker_diagnosis_complete": bool(blocker_diagnosis.get("blocker_diagnosis_complete", False)),
        "shadow_trigger_sources_found": source_found,
        "shadow_trigger_source_manifest_ready": manifest_ready,
        "shadow_trigger_source_cache_precheck_pass": coverage_pass,
        "source_scenario_count": len(sourcing.get("source_scenarios", []) or []),
        "source_scenario_ids": list((source_manifest or {}).get("scenario_ids", []) or []),
        "blocker_taxonomy": list(sourcing.get("blocker_taxonomy", []) or []),
        "next_action": next_action,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Metadata Replay Readiness Checklist v25",
        "",
        "- Mode: shadow-only trigger source readiness",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Controlled replay execution allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Source scenarios found: {report.get('shadow_trigger_sources_found', False)}",
        f"- Cache precheck pass: {report.get('shadow_trigger_source_cache_precheck_pass', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-v24-json", required=True)
    parser.add_argument("--blocker-diagnosis-json", required=True)
    parser.add_argument("--sourcing-json", required=True)
    parser.add_argument("--source-manifest-json", default="")
    parser.add_argument("--cache-coverage-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        readiness_v24=_load_optional(args.readiness_v24_json) or {},
        blocker_diagnosis=_load_optional(args.blocker_diagnosis_json) or {},
        sourcing=_load_optional(args.sourcing_json) or {},
        source_manifest=_load_optional(args.source_manifest_json),
        cache_coverage=_load_optional(args.cache_coverage_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"shadow_trigger_sources_found={report['shadow_trigger_sources_found']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
