"""Build v34 readiness for shadow-only scenario/cache acquisition design."""

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


def build_report(
    *,
    target_spec: Mapping[str, Any],
    source_inventory: Mapping[str, Any] | None = None,
    cache_acquisition_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target_ready = bool(target_spec.get("target_spec_ready", False))
    inventory_ready = bool(source_inventory and source_inventory.get("new_shadow_source_inventory_ready", False))
    source_found = bool(source_inventory and source_inventory.get("new_shadow_source_candidates_found", False))
    request_ready = bool(cache_acquisition_request and cache_acquisition_request.get("shadow_cache_acquisition_request_ready", False))
    failure_taxonomy: list[str] = []
    if not target_ready:
        failure_taxonomy.append("target_spec_missing")
    if source_inventory and not source_found:
        failure_taxonomy.append("new_shadow_source_not_found")
    if request_ready:
        next_action = "await_explicit_shadow_cache_acquisition_authorization"
    elif inventory_ready and not source_found:
        next_action = "external_weather_or_scenario_space_expansion_design"
    elif target_ready and not source_inventory:
        next_action = "run_v34_read_only_trace_source_inventory"
    elif source_found and not request_ready:
        next_action = "shadow_cache_acquisition_request_required"
    else:
        next_action = "v34_shadow_acquisition_design_blocked"
    return {
        "schema_version": "metadata_replay_readiness_checklist_20260528_v34",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "v34 shadow-only scenario/cache acquisition design readiness",
            "reopens_rejected_preset": False,
        },
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "cache_fill_authorized": False,
        "online_llm_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "target_spec_ready": target_ready,
        "new_shadow_source_inventory_ready": inventory_ready,
        "new_shadow_source_candidates_found": source_found,
        "shadow_cache_acquisition_request_ready": request_ready,
        "metrics": {
            "v33_near_miss_scenario_count": int(_num(target_spec.get("v33_near_miss_scenario_count"))),
            "v33_near_miss_row_count": int(_num(target_spec.get("v33_near_miss_row_count"))),
            "trace_count": int(_num(source_inventory.get("trace_count") if source_inventory else 0)),
            "candidate_scenario_count": int(_num(source_inventory.get("candidate_scenario_count") if source_inventory else 0)),
            "requested_scenario_count": int(_num(cache_acquisition_request.get("requested_scenario_count") if cache_acquisition_request else 0)),
        },
        "failure_taxonomy": sorted(set(failure_taxonomy)),
        "next_action": next_action,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Metadata Replay Readiness Checklist v34",
        "",
        "- Mode: shadow-only scenario/cache acquisition design",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Controlled replay execution allowed: false",
        "- Cache fill authorized: false",
        "- Online LLM allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- New shadow source candidates found: {report.get('new_shadow_source_candidates_found', False)}",
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
    parser.add_argument("--target-spec-json", required=True)
    parser.add_argument("--source-inventory-json", default="")
    parser.add_argument("--cache-acquisition-request-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        target_spec=_load_optional(args.target_spec_json) or {},
        source_inventory=_load_optional(args.source_inventory_json),
        cache_acquisition_request=_load_optional(args.cache_acquisition_request_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"new_shadow_source_candidates_found={report['new_shadow_source_candidates_found']}")
    print(f"shadow_cache_acquisition_request_ready={report['shadow_cache_acquisition_request_ready']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
