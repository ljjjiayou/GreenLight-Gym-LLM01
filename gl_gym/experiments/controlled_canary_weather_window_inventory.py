"""Scan local weather files for v35 shadow-only source acquisition windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def _scenario_id(year: int, day: int, seed: int, max_steps: int) -> str:
    return f"y{year}_d{day}_s{seed}_n{max_steps}"


def _infer_year(path: Path) -> int:
    match = re.search(r"(20\d{2})", path.stem)
    return int(match.group(1)) if match else 0


def _saturation_vapor_pressure(temp_c: float) -> float:
    return 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))


def _vpd(temp_c: float, rh: float) -> float:
    return _saturation_vapor_pressure(temp_c) * (1.0 - max(0.0, min(100.0, rh)) / 100.0)


def _dew_point(temp_c: float, rh: float) -> float:
    rh = max(0.001, min(100.0, rh))
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(rh / 100.0)
    return (b * alpha) / (a - alpha)


def _iter_weather_files(weather_sources: Sequence[str | Path]) -> list[Path]:
    files: list[Path] = []
    for source in weather_sources:
        p = _resolve(source)
        if p.is_file() and p.suffix.lower() == ".csv":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(item for item in p.rglob("*.csv") if item.is_file()))
    return sorted(set(files))


def _row_record(path: Path, row: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any] | None:
    year = _infer_year(path)
    day = int(math.floor(_num(row.get("day number"), -1.0)))
    if year <= 0 or day < 0:
        return None
    day_number = _num(row.get("day number"), float(day))
    step_in_day = int(round((day_number - day) * 288.0))
    temp = _num(row.get("air temperature"))
    rh = _num(row.get("RH"))
    radiation = _num(row.get("global radiation"))
    wind = _num(row.get("wind speed"))
    vpd = _vpd(temp, rh)
    dew_spread = temp - _dew_point(temp, rh)
    strict_proxy = bool(
        rh <= _num(gate.get("rh_max"), 45.0)
        and vpd >= _num(gate.get("vpd_min"), 2.25)
        and temp >= _num(gate.get("temp_min"), 20.0)
        and temp <= _num(gate.get("temp_max"), 30.5)
        and dew_spread >= _num(gate.get("dew_point_spread_min"), 3.0)
        and radiation >= _num(gate.get("global_radiation_min"), 200.0)
    )
    if not strict_proxy:
        return None
    return {
        "weather_path": str(path),
        "weather_site": path.parent.name,
        "weather_label": path.stem,
        "year": year,
        "day": day,
        "step_in_day": max(0, step_in_day),
        "hour": round(max(0, step_in_day) * 5.0 / 60.0, 3),
        "temp_air": round(temp, 6),
        "rh_air": round(rh, 6),
        "vpd_air": round(vpd, 6),
        "global_radiation": round(radiation, 6),
        "wind_speed": round(wind, 6),
        "dew_point_spread": round(dew_spread, 6),
        "candidate": gate.get("candidate", "shadow_hot_dry_humidity_retention"),
        "weather_proxy_source_hint_only": True,
        "strict_applied_evidence": False,
    }


def _summarize_day(rows: Sequence[Mapping[str, Any]], *, target_spec: Mapping[str, Any]) -> dict[str, Any]:
    first = rows[0]
    max_steps = int(target_spec.get("max_steps", 240) or 240)
    seeds = [int(seed) for seed in target_spec.get("candidate_seeds", []) or [42, 43, 44]]
    excluded = {str(item) for item in target_spec.get("excluded_scenario_ids", []) or [] if str(item or "")}
    scenario_ids = [
        _scenario_id(int(first["year"]), int(first["day"]), seed, max_steps)
        for seed in seeds
        if _scenario_id(int(first["year"]), int(first["day"]), seed, max_steps) not in excluded
    ]
    source_rows = sorted(rows, key=lambda item: int(item.get("step_in_day", 0)))
    score = (
        len(source_rows) * 10.0
        + max(float(item.get("vpd_air", 0.0)) for item in source_rows)
        + max(float(item.get("global_radiation", 0.0)) for item in source_rows) / 1000.0
        + max(0.0, 45.0 - min(float(item.get("rh_air", 100.0)) for item in source_rows)) / 10.0
    )
    return {
        "weather_site": first.get("weather_site", ""),
        "weather_label": first.get("weather_label", ""),
        "weather_path": first.get("weather_path", ""),
        "year": int(first.get("year", 0)),
        "day": int(first.get("day", 0)),
        "max_steps": max_steps,
        "seed_candidates": seeds,
        "scenario_ids_for_cache_acquisition": scenario_ids,
        "runner_weather_binding_required": first.get("weather_site") != "Amsterdam",
        "strict_proxy_row_count": len(source_rows),
        "first_step": int(source_rows[0].get("step_in_day", 0)),
        "last_step": int(source_rows[-1].get("step_in_day", 0)),
        "min_rh_air": min(float(item.get("rh_air", 0.0)) for item in source_rows),
        "max_vpd_air": max(float(item.get("vpd_air", 0.0)) for item in source_rows),
        "max_temp_air": max(float(item.get("temp_air", 0.0)) for item in source_rows),
        "min_dew_point_spread": min(float(item.get("dew_point_spread", 0.0)) for item in source_rows),
        "max_global_radiation": max(float(item.get("global_radiation", 0.0)) for item in source_rows),
        "acquisition_ready_under_current_runner": bool(scenario_ids and first.get("weather_site") == "Amsterdam"),
        "weather_proxy_score": round(score, 6),
        "sample_rows": list(source_rows[:12]),
        "weather_proxy_source_hint_only": True,
        "strict_applied_evidence": False,
    }


def build_report(*, target_spec: Mapping[str, Any], weather_sources: Sequence[str | Path] | None = None) -> dict[str, Any]:
    gate = dict(target_spec.get("strict_weather_proxy_gate", {}) or {})
    sources = list(weather_sources or target_spec.get("weather_sources", []) or [])
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    files = _iter_weather_files(sources)
    for path in files:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                record = _row_record(path, row, gate)
                if record:
                    key = (
                        str(record["weather_site"]),
                        str(record["weather_label"]),
                        int(record["year"]),
                        int(record["day"]),
                    )
                    grouped.setdefault(key, []).append(record)
    candidates = [_summarize_day(rows, target_spec=target_spec) for rows in grouped.values()]
    candidates.sort(
        key=lambda item: (
            bool(item.get("acquisition_ready_under_current_runner", False)),
            float(item.get("weather_proxy_score", 0.0)),
            int(item.get("strict_proxy_row_count", 0)),
        ),
        reverse=True,
    )
    max_days = int(target_spec.get("max_candidate_days", 12) or 12)
    selected = candidates[:max_days]
    requested: list[dict[str, Any]] = []
    for item in selected:
        if not item.get("acquisition_ready_under_current_runner", False):
            continue
        for scenario_id in item.get("scenario_ids_for_cache_acquisition", []) or []:
            requested.append(
                {
                    "scenario_id": scenario_id,
                    "year": int(item.get("year", 0)),
                    "day": int(item.get("day", 0)),
                    "seed": int(str(scenario_id).split("_s")[-1].split("_")[0]),
                    "max_steps": int(item.get("max_steps", 240)),
                    "weather_site": item.get("weather_site", ""),
                    "weather_label": item.get("weather_label", ""),
                    "expected_strict_source_rationale": "weather_proxy_hot_dry_humidity_retention_source_hint",
                    "source_hint_only": True,
                }
            )
    return {
        "schema_version": "controlled_canary_weather_window_inventory_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only weather window inventory",
            "reopens_rejected_preset": False,
        },
        "weather_window_inventory_ready": bool(target_spec.get("target_spec_ready", False)),
        "read_only_weather_inventory": True,
        "weather_proxy_source_hint_only": True,
        "weather_proxy_is_strict_applied_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "cache_fill_authorized": False,
        "online_llm_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "weather_file_count": len(files),
        "candidate_weather_day_count": len(candidates),
        "selected_candidate_weather_day_count": len(selected),
        "new_weather_source_candidates_found": bool(requested),
        "candidate_windows": selected,
        "candidate_scenarios_for_cache_acquisition": requested,
        "requested_scenario_count": len(requested),
        "next_action": (
            "shadow_scenario_acquisition_manifest"
            if requested
            else "external_weather_dataset_or_relaxed_shadow_source_design_required"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Weather Window Inventory v35",
        "",
        f"- Inventory ready: {report.get('weather_window_inventory_ready', False)}",
        "- Read-only weather inventory: true",
        "- Cache fill authorized: false",
        "- Online LLM allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "- Weather proxy is strict-applied evidence: false",
        f"- Weather files scanned: {report.get('weather_file_count', 0)}",
        f"- Candidate weather days: {report.get('candidate_weather_day_count', 0)}",
        f"- Requested scenarios: {report.get('requested_scenario_count', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Selected Candidate Windows",
        "",
        "| source | year | day | rows | score | acquisition ready | scenario count |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for item in report.get("candidate_windows", []) or []:
        if isinstance(item, Mapping):
            source = f"{item.get('weather_site', '')}/{item.get('weather_label', '')}"
            lines.append(
                f"| {source} | {item.get('year', 0)} | {item.get('day', 0)} | "
                f"{item.get('strict_proxy_row_count', 0)} | {item.get('weather_proxy_score', 0)} | "
                f"{item.get('acquisition_ready_under_current_runner', False)} | "
                f"{len(item.get('scenario_ids_for_cache_acquisition', []) or [])} |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-spec-json", required=True)
    parser.add_argument("--weather-source", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target_spec = _load(args.target_spec_json)
    report = build_report(target_spec=target_spec, weather_sources=args.weather_source or None)
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"new_weather_source_candidates_found={report['new_weather_source_candidates_found']}")
    print(f"requested_scenario_count={report['requested_scenario_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
