"""Build v35 readiness for weather-space shadow source acquisition design."""

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


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def build_report(
    *,
    target_spec: Mapping[str, Any],
    weather_inventory: Mapping[str, Any] | None = None,
    acquisition_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target_ready = bool(target_spec.get("target_spec_ready", False))
    inventory_ready = bool(weather_inventory and weather_inventory.get("weather_window_inventory_ready", False))
    source_found = bool(weather_inventory and weather_inventory.get("new_weather_source_candidates_found", False))
    manifest_ready = bool(acquisition_manifest and acquisition_manifest.get("shadow_scenario_acquisition_manifest_ready", False))
    failures: list[str] = []
    if not target_ready:
        failures.append("target_spec_missing")
    if inventory_ready and not source_found:
        failures.append("weather_strict_proxy_source_not_found")
    if source_found and not manifest_ready:
        failures.append("shadow_scenario_acquisition_manifest_missing")
    if manifest_ready:
        next_action = "await_explicit_shadow_cache_acquisition_authorization_for_weather_expansion"
    elif inventory_ready and not source_found:
        next_action = "external_weather_dataset_or_relaxed_shadow_source_design_required"
    elif target_ready and not weather_inventory:
        next_action = "run_v35_read_only_weather_window_inventory"
    else:
        next_action = "v35_weather_space_expansion_design_blocked"
    return {
        "schema_version": "metadata_replay_readiness_checklist_20260528_v35",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "v35 weather-space shadow source acquisition readiness",
            "reopens_rejected_preset": False,
        },
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "cache_fill_authorized": False,
        "online_llm_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "weather_proxy_source_hint_only": True,
        "weather_proxy_is_strict_applied_evidence": False,
        "target_spec_ready": target_ready,
        "weather_window_inventory_ready": inventory_ready,
        "new_weather_source_candidates_found": source_found,
        "shadow_scenario_acquisition_manifest_ready": manifest_ready,
        "metrics": {
            "v33_near_miss_scenario_count": _int(target_spec.get("v33_near_miss_scenario_count")),
            "v33_near_miss_row_count": _int(target_spec.get("v33_near_miss_row_count")),
            "weather_file_count": _int(weather_inventory.get("weather_file_count") if weather_inventory else 0),
            "candidate_weather_day_count": _int(weather_inventory.get("candidate_weather_day_count") if weather_inventory else 0),
            "selected_candidate_weather_day_count": _int(
                weather_inventory.get("selected_candidate_weather_day_count") if weather_inventory else 0
            ),
            "requested_scenario_count": _int(acquisition_manifest.get("requested_scenario_count") if acquisition_manifest else 0),
        },
        "failure_taxonomy": sorted(set(failures)),
        "next_action": next_action,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Metadata Replay Readiness Checklist v35",
        "",
        "- Mode: weather-space shadow source acquisition design",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Controlled replay execution allowed: false",
        "- Cache fill authorized: false",
        "- Online LLM allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        "- Weather proxy is strict-applied evidence: false",
        f"- New weather source candidates found: {report.get('new_weather_source_candidates_found', False)}",
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
    parser.add_argument("--weather-inventory-json", default="")
    parser.add_argument("--acquisition-manifest-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        target_spec=_load_optional(args.target_spec_json) or {},
        weather_inventory=_load_optional(args.weather_inventory_json),
        acquisition_manifest=_load_optional(args.acquisition_manifest_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"new_weather_source_candidates_found={report['new_weather_source_candidates_found']}")
    print(f"shadow_scenario_acquisition_manifest_ready={report['shadow_scenario_acquisition_manifest_ready']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
