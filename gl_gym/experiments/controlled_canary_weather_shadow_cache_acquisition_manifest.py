"""Build the v36 weather-sourced shadow cache acquisition batch manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CACHE_PATH = "gl_gym/result/plan_cache/weather_expansion_shadow_cache_v36_20260529.json"
DEFAULT_TOP_WEATHER_DAYS = 3


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None and value != "" else default)
    except Exception:
        return int(default)


def _scenario_parts(scenario_id: str) -> dict[str, int]:
    parts = {"year": 0, "day": 0, "seed": 0, "max_steps": 240}
    for token in str(scenario_id).split("_"):
        try:
            if token.startswith("y"):
                parts["year"] = int(token[1:])
            elif token.startswith("d"):
                parts["day"] = int(token[1:])
            elif token.startswith("s"):
                parts["seed"] = int(token[1:])
            elif token.startswith("n"):
                parts["max_steps"] = int(token[1:])
        except ValueError:
            continue
    return parts


def _selected_weather_windows(weather_inventory: Mapping[str, Any], top_weather_days: int) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for item in weather_inventory.get("candidate_windows", []) or []:
        if not isinstance(item, Mapping):
            continue
        scenarios = [str(s) for s in item.get("scenario_ids_for_cache_acquisition", []) or [] if str(s or "")]
        if not scenarios:
            continue
        if not bool(item.get("acquisition_ready_under_current_runner", False)):
            continue
        windows.append(dict(item))
        if len(windows) >= int(top_weather_days):
            break
    return windows


def build_report(
    *,
    weather_inventory: Mapping[str, Any],
    acquisition_manifest: Mapping[str, Any],
    isolated_cache_path: str = DEFAULT_CACHE_PATH,
    top_weather_days: int = DEFAULT_TOP_WEATHER_DAYS,
) -> dict[str, Any]:
    windows = _selected_weather_windows(weather_inventory, top_weather_days)
    scenarios: list[dict[str, Any]] = []
    for window in windows:
        for scenario_id in window.get("scenario_ids_for_cache_acquisition", []) or []:
            parts = _scenario_parts(str(scenario_id))
            scenarios.append(
                {
                    "scenario_id": str(scenario_id),
                    **parts,
                    "weather_site": str(window.get("weather_site", "")),
                    "weather_label": str(window.get("weather_label", "")),
                    "strict_proxy_row_count": _as_int(window.get("strict_proxy_row_count")),
                    "weather_proxy_score": float(window.get("weather_proxy_score", 0.0) or 0.0),
                    "source_hint_only": True,
                    "strict_applied_evidence": False,
                }
            )
    command_groups = [
        {
            "group_id": f"weather_v36_y{window.get('year')}_d{window.get('day')}",
            "years": [_as_int(window.get("year"))],
            "days": [_as_int(window.get("day"))],
            "seeds": sorted(_scenario_parts(str(s))["seed"] for s in window.get("scenario_ids_for_cache_acquisition", []) or []),
            "max_steps": _as_int(window.get("max_steps"), 240),
            "scenario_ids": [str(s) for s in window.get("scenario_ids_for_cache_acquisition", []) or []],
            "cache_path": isolated_cache_path,
            "cartesian_product_safe": True,
        }
        for window in windows
    ]
    manifest_ready = bool(
        weather_inventory.get("new_weather_source_candidates_found", False)
        and acquisition_manifest.get("shadow_scenario_acquisition_manifest_ready", False)
        and len(windows) == int(top_weather_days)
        and len(scenarios) == int(top_weather_days) * 3
    )
    return {
        "schema_version": "controlled_canary_weather_shadow_cache_acquisition_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "weather-sourced shadow cache acquisition admission",
            "reopens_rejected_preset": False,
        },
        "weather_shadow_cache_acquisition_manifest_ready": manifest_ready,
        "scope": "top_3_weather_days_9_shadow_scenarios" if manifest_ready else "blocked",
        "weather_proxy_source_hint_only": True,
        "weather_proxy_is_strict_applied_evidence": False,
        "isolated_cache_path": isolated_cache_path,
        "selected_cache_overwrite_allowed": False,
        "cache_fill_authorized": False,
        "online_llm_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "controller": "llm_rspc_v2",
        "blocked_controllers": ["llm_rspc_v2_hot_dry_proposer_strict"],
        "top_weather_day_count": len(windows),
        "scenario_count": len(scenarios),
        "selected_weather_windows": windows,
        "selected_scenarios": scenarios,
        "scenario_ids": [item["scenario_id"] for item in scenarios],
        "command_groups": command_groups,
        "next_action": "v36_shadow_cache_acquisition_execution_record" if manifest_ready else "v36_weather_shadow_manifest_blocked",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Weather-Sourced Shadow Cache Acquisition Manifest v36",
        "",
        f"- Manifest ready: {report.get('weather_shadow_cache_acquisition_manifest_ready', False)}",
        f"- Scope: `{report.get('scope', '')}`",
        "- Controller: `llm_rspc_v2`",
        "- Blocked controlled controller: `llm_rspc_v2_hot_dry_proposer_strict`",
        "- Cache fill authorized: false",
        "- Online LLM allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Isolated cache path: `{report.get('isolated_cache_path', '')}`",
        f"- Scenario count: {report.get('scenario_count', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Selected Scenarios",
        "",
        "| scenario | weather source | proxy rows | score |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in report.get("selected_scenarios", []) or []:
        if isinstance(item, Mapping):
            source = f"{item.get('weather_site', '')}/{item.get('weather_label', '')}"
            lines.append(
                f"| {item.get('scenario_id', '')} | {source} | "
                f"{item.get('strict_proxy_row_count', 0)} | {item.get('weather_proxy_score', 0)} |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-inventory-json", required=True)
    parser.add_argument("--acquisition-manifest-json", required=True)
    parser.add_argument("--isolated-cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--top-weather-days", type=int, default=DEFAULT_TOP_WEATHER_DAYS)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        weather_inventory=_load(args.weather_inventory_json),
        acquisition_manifest=_load(args.acquisition_manifest_json),
        isolated_cache_path=args.isolated_cache_path,
        top_weather_days=int(args.top_weather_days),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"weather_shadow_cache_acquisition_manifest_ready={report['weather_shadow_cache_acquisition_manifest_ready']}")
    print(f"scenario_count={report['scenario_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["weather_shadow_cache_acquisition_manifest_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
