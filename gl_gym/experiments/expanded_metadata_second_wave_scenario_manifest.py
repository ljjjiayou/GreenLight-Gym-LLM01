"""Build the second-wave expanded metadata replay scenario manifest."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CANDIDATES = ("y2015_d180_s44_n240", "y2020_d180_s44_n240")
REQUIRED_FAMILIES = ("pure_hot_dry", "mixed_dry_dew")
SCENARIO_RE = re.compile(r"^y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)_n(?P<steps>\d+)$")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_scenario_id(scenario_id: str) -> dict[str, int]:
    match = SCENARIO_RE.match(scenario_id)
    if not match:
        raise ValueError(f"invalid scenario_id: {scenario_id}")
    return {
        "year": int(match.group("year")),
        "day": int(match.group("day")),
        "seed": int(match.group("seed")),
        "max_steps": int(match.group("steps")),
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _window(row: Mapping[str, Any], *, regime: str, scenario_id: str) -> dict[str, Any]:
    step = int(round(_num(row.get("step", row.get("timestep", 0)))))
    return {
        "scenario_id": scenario_id,
        "regime": regime,
        "selection_source": "existing_baseline_trace",
        "center_step": step,
        "start_step": max(0, step - 2),
        "end_step": step + 2,
        "state_summary": {
            "temp_air": _num(row.get("temp_air")),
            "rh_air": _num(row.get("rh_air")),
            "vpd_air": _num(row.get("vpd_air")),
            "dew_margin_air": _num(row.get("dew_margin_air")),
            "canopy_dew_margin": _num(row.get("canopy_dew_margin")),
            "glob_rad": _num(row.get("glob_rad")),
            "wind_speed": _num(row.get("wind_speed")),
            "dry_risk": _bool(row.get("dry_risk")),
            "dew_risk": _bool(row.get("dew_risk")),
        },
    }


def _pure_hot_dry(row: Mapping[str, Any]) -> bool:
    temp = _num(row.get("temp_air"))
    rh = _num(row.get("rh_air"))
    vpd = _num(row.get("vpd_air"))
    dew_margin = _num(row.get("dew_margin_air"))
    canopy_margin = _num(row.get("canopy_dew_margin"))
    dry_pressure = vpd >= 1.2 or rh <= 55.0 or _bool(row.get("dry_risk"))
    return bool(
        dry_pressure
        and temp >= 20.0
        and dew_margin >= 2.0
        and canopy_margin >= 2.0
        and not _bool(row.get("dew_risk"))
    )


def _dew_or_humidity_pressure(row: Mapping[str, Any]) -> bool:
    rh = _num(row.get("rh_air"))
    dew_margin = _num(row.get("dew_margin_air"))
    canopy_margin = _num(row.get("canopy_dew_margin"))
    return bool(rh >= 88.0 or dew_margin <= 2.0 or canopy_margin <= 2.0 or _bool(row.get("dew_risk")))


def _scenario_report(
    *,
    scenario_id: str,
    baseline_trace_dir: Path,
    max_windows_per_family: int,
) -> dict[str, Any]:
    parts = _parse_scenario_id(scenario_id)
    csv_path = baseline_trace_dir / f"{scenario_id}_llm_rspc_v2.csv"
    jsonl_path = baseline_trace_dir / f"{scenario_id}_llm_rspc_v2.jsonl"
    baseline_trace_exists = csv_path.exists()
    rows = _read_rows(csv_path) if baseline_trace_exists else []
    pure_windows = [
        _window(row, regime="pure_hot_dry", scenario_id=scenario_id)
        for row in rows
        if _pure_hot_dry(row)
    ]
    dew_windows = [
        _window(row, regime="dew_or_high_humidity_pressure", scenario_id=scenario_id)
        for row in rows
        if _dew_or_humidity_pressure(row)
    ]
    scenario_families: list[str] = []
    if pure_windows:
        scenario_families.append("pure_hot_dry")
    if pure_windows and dew_windows:
        scenario_families.append("mixed_dry_dew")
    selected = baseline_trace_exists and all(family in scenario_families for family in REQUIRED_FAMILIES)
    return {
        **parts,
        "scenario_id": scenario_id,
        "env_id": f"TomatoEnv_y{parts['year']}_d{parts['day']}_s{parts['seed']}",
        "baseline_trace_csv": str(csv_path),
        "baseline_trace_jsonl": str(jsonl_path),
        "baseline_trace_exists": baseline_trace_exists,
        "baseline_trace_jsonl_exists": jsonl_path.exists(),
        "selected_for_second_wave": selected,
        "scenario_families": scenario_families,
        "pure_hot_dry_window_count": len(pure_windows),
        "dew_or_high_humidity_window_count": len(dew_windows),
        "mixed_dry_dew_window_count": min(len(pure_windows), len(dew_windows)) if pure_windows and dew_windows else 0,
        "windows": {
            "pure_hot_dry": pure_windows[:max_windows_per_family],
            "dew_or_high_humidity_pressure": dew_windows[:max_windows_per_family],
        },
    }


def build_report(
    *,
    baseline_trace_dir: str | Path,
    selected_cache_path: str,
    candidate_scenario_ids: Sequence[str] = DEFAULT_CANDIDATES,
    max_windows_per_family: int = 25,
) -> dict[str, Any]:
    baseline_root = _resolve(baseline_trace_dir)
    candidates = [
        _scenario_report(
            scenario_id=scenario_id,
            baseline_trace_dir=baseline_root,
            max_windows_per_family=max_windows_per_family,
        )
        for scenario_id in candidate_scenario_ids
    ]
    selected = [item for item in candidates if item.get("selected_for_second_wave")]
    covered = sorted({family for item in selected for family in item.get("scenario_families", [])})
    missing = [family for family in REQUIRED_FAMILIES if family not in covered]
    manifest_ready = not missing and bool(selected)
    return {
        "schema_version": "expanded_metadata_second_wave_scenario_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "second-wave expanded strict metadata replay manifest",
            "reopens_rejected_preset": False,
        },
        "second_wave_manifest_ready": manifest_ready,
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "counterexample_replay_allowed": False,
        "performance_claim_allowed": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "selected_cache_path": selected_cache_path,
        "baseline_trace_dir": str(baseline_root),
        "scope": "second_wave_expanded_metadata_coverage",
        "controller": "llm_rspc_v2",
        "required_families": list(REQUIRED_FAMILIES),
        "covered_families": covered,
        "missing_families": missing,
        "candidate_scenarios": candidates,
        "selected_scenarios": selected,
        "scenarios": selected,
        "scenario_count": len(selected),
        "next_action": "run_no_fill_second_wave_cache_coverage_check" if manifest_ready else "second_wave_scenario_discovery_required",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Second-Wave Expanded Metadata Scenario Manifest",
        "",
        "- Mode: second-wave expanded strict metadata replay manifest",
        "- Default `llm_rspc_v2` changed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Manifest ready: {report.get('second_wave_manifest_ready', False)}",
        f"- Selected cache: `{report.get('selected_cache_path', '')}`",
        f"- Baseline trace dir: `{report.get('baseline_trace_dir', '')}`",
        "",
        "## Selected Scenarios",
        "",
        "| scenario | families | pure windows | dew/high-humidity windows |",
        "| --- | --- | ---: | ---: |",
    ]
    for item in report.get("selected_scenarios", []) or []:
        if not isinstance(item, Mapping):
            continue
        families = ", ".join(str(value) for value in item.get("scenario_families", []) or [])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("scenario_id", "")),
                    families,
                    str(item.get("pure_hot_dry_window_count", 0)),
                    str(item.get("dew_or_high_humidity_window_count", 0)),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Missing Families", ""])
    for family in report.get("missing_families", []) or []:
        lines.append(f"- {family}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-trace-dir", required=True)
    parser.add_argument("--selected-cache-path", required=True)
    parser.add_argument("--candidate-scenario-ids", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--max-windows-per-family", type=int, default=25)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    candidates = [token.strip() for token in str(args.candidate_scenario_ids).split(",") if token.strip()]
    report = build_report(
        baseline_trace_dir=args.baseline_trace_dir,
        selected_cache_path=args.selected_cache_path,
        candidate_scenario_ids=candidates,
        max_windows_per_family=args.max_windows_per_family,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"second_wave_manifest_ready={report['second_wave_manifest_ready']}")
    print(f"scenario_count={report['scenario_count']}")
    print(f"covered_families={','.join(report['covered_families'])}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["second_wave_manifest_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
