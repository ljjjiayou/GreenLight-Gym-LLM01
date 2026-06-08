"""Build a reusable extreme-regime stress suite from benchmark traces.

The builder is intentionally read-only with respect to controllers: it mines
existing per-step traces, tags greenhouse regimes, merges risk windows, and
writes a manifest that later runners can execute with frozen replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml


DEFAULT_CONFIG_PATH = project_root / "gl_gym" / "configs" / "stress" / "regime_stress_v1.yaml"
DEFAULT_OUTPUT_JSON = project_root / "gl_gym" / "result" / "stress_suites" / "regime_stress_v1.json"
DEFAULT_OUTPUT_REPORT = project_root / "gl_gym" / "result" / "stress_suites" / "regime_stress_v1_report.md"

TRACE_NAME_RE = re.compile(
    r"^(?P<scenario_id>y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)_n(?P<max_steps>\d+))_(?P<controller>.+)$"
)

REGIME_ORDER = (
    "hot_dry",
    "hot_humid",
    "cold_humid",
    "cold_dry",
    "dawn_dew",
    "radiation_spike",
    "wind_stress",
)

CRITICAL_TRACE_FIELDS = (
    "step",
    "hour_of_day",
    "temp_air",
    "rh_air",
    "vpd_air",
    "wind_speed",
)


@dataclass(frozen=True)
class TraceIdentity:
    path: str
    scenario_id: str
    controller: str
    year: int
    day: int
    seed: int
    max_steps: int
    suffix: str


def _coerce(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Regime stress config must be a mapping: {path}")
    return data


def read_trace(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    item = json.loads(text)
                    if isinstance(item, dict):
                        rows.append(item)
        return rows
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{key: _coerce(value) for key, value in row.items()} for row in reader]


def parse_trace_identity(path: str | Path) -> TraceIdentity | None:
    path = Path(path)
    match = TRACE_NAME_RE.match(path.stem)
    if not match:
        return None
    data = match.groupdict()
    return TraceIdentity(
        path=str(path),
        scenario_id=str(data["scenario_id"]),
        controller=str(data["controller"]),
        year=int(data["year"]),
        day=int(data["day"]),
        seed=int(data["seed"]),
        max_steps=int(data["max_steps"]),
        suffix=path.suffix.lower(),
    )


def discover_trace_files(inputs: Sequence[str | Path]) -> List[Path]:
    files: List[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            for suffix in ("*.csv", "*.jsonl"):
                files.extend(sorted(path.rglob(suffix)))
        elif path.is_file() and path.suffix.lower() in {".csv", ".jsonl"}:
            files.append(path)
    return sorted(set(files), key=lambda item: str(item).lower())


def _controller_rank(controller: str, preference: Sequence[str]) -> int:
    try:
        return list(preference).index(controller)
    except ValueError:
        return len(preference)


def _suffix_rank(suffix: str, preference: Sequence[str]) -> int:
    try:
        return list(preference).index(suffix.lower())
    except ValueError:
        return len(preference)


def select_source_traces(identities: Sequence[TraceIdentity], config: Mapping[str, Any]) -> Dict[str, TraceIdentity]:
    trace_selection = config.get("trace_selection", {}) if isinstance(config.get("trace_selection"), Mapping) else {}
    controller_preference = [str(x) for x in trace_selection.get("controller_preference", [])]
    extension_preference = [str(x).lower() for x in trace_selection.get("extension_preference", [])]
    grouped: Dict[str, List[TraceIdentity]] = {}
    for identity in identities:
        grouped.setdefault(identity.scenario_id, []).append(identity)
    selected: Dict[str, TraceIdentity] = {}
    for scenario_id, items in grouped.items():
        ranked = sorted(
            items,
            key=lambda item: (
                _controller_rank(item.controller, controller_preference),
                _suffix_rank(item.suffix, extension_preference),
                str(item.path).lower(),
            ),
        )
        selected[scenario_id] = ranked[0]
    return dict(sorted(selected.items()))


def _rad_heat_load(row: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    derived = config.get("derived_features", {}) if isinstance(config.get("derived_features"), Mapping) else {}
    fields = derived.get("rad_heat_load_fields", ["rad_heat_load", "glob_rad", "forecast_rad_peak_2h"])
    return max(_num(row, str(field), 0.0) for field in fields)


def _dew_margin_min(row: Mapping[str, Any]) -> float:
    if row.get("dew_margin_min") is not None:
        return _num(row, "dew_margin_min", 99.0)
    return min(_num(row, "dew_margin_air", 99.0), _num(row, "canopy_dew_margin", 99.0))


def _temp_rise(rows: Sequence[Mapping[str, Any]], index: int, config: Mapping[str, Any]) -> float:
    derived = config.get("derived_features", {}) if isinstance(config.get("derived_features"), Mapping) else {}
    lookback = int(derived.get("radiation_spike_lookback_steps", 4) or 4)
    base_index = max(0, int(index) - max(lookback, 1))
    return _num(rows[index], "temp_air", 0.0) - _num(rows[base_index], "temp_air", 0.0)


def tag_row(row: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], index: int, config: Mapping[str, Any]) -> List[str]:
    regimes = config.get("regimes", {}) if isinstance(config.get("regimes"), Mapping) else {}
    tags: List[str] = []
    temp = _num(row, "temp_air", 20.0)
    rh = _num(row, "rh_air", 70.0)
    vpd = _num(row, "vpd_air", _num(row, "vpd_kpa", 0.0))
    hour = _num(row, "hour_of_day", 0.0)
    wind = _num(row, "wind_speed", 0.0)
    rad = _rad_heat_load(row, config)
    dew_margin = _dew_margin_min(row)
    canopy_dew = _num(row, "canopy_dew_margin", 99.0)

    hot_dry = regimes.get("hot_dry", {})
    if (
        temp >= float(hot_dry.get("temp_air_min", 30.0))
        and rad >= float(hot_dry.get("rad_heat_load_min", 600.0))
        and (rh <= float(hot_dry.get("rh_air_max", 55.0)) or vpd >= float(hot_dry.get("vpd_air_min", 1.6)))
    ):
        tags.append("hot_dry")

    hot_humid = regimes.get("hot_humid", {})
    if temp >= float(hot_humid.get("temp_air_min", 28.0)) and (
        rh >= float(hot_humid.get("rh_air_min", 85.0))
        or dew_margin <= float(hot_humid.get("dew_margin_min_max", 1.0))
    ):
        tags.append("hot_humid")

    cold_humid = regimes.get("cold_humid", {})
    if temp <= float(cold_humid.get("temp_air_max", 16.5)) and (
        rh >= float(cold_humid.get("rh_air_min", 88.0))
        or vpd <= float(cold_humid.get("vpd_air_max", 0.35))
        or dew_margin <= float(cold_humid.get("dew_margin_min_max", 1.0))
    ):
        tags.append("cold_humid")

    cold_dry = regimes.get("cold_dry", {})
    if temp <= float(cold_dry.get("temp_air_max", 16.5)) and rh <= float(cold_dry.get("rh_air_max", 55.0)):
        tags.append("cold_dry")

    dawn_dew = regimes.get("dawn_dew", {})
    if float(dawn_dew.get("hour_start", 4.0)) <= hour <= float(dawn_dew.get("hour_end", 8.0)) and (
        rh >= float(dawn_dew.get("rh_air_min", 88.0))
        or canopy_dew <= float(dawn_dew.get("canopy_dew_margin_max", 1.2))
    ):
        tags.append("dawn_dew")

    radiation_spike = regimes.get("radiation_spike", {})
    rise_threshold = float(
        (config.get("derived_features", {}) if isinstance(config.get("derived_features"), Mapping) else {}).get(
            "radiation_spike_temp_rise_c",
            1.0,
        )
    )
    if rad >= float(radiation_spike.get("rad_heat_load_min", 650.0)) and _temp_rise(rows, index, config) >= rise_threshold:
        tags.append("radiation_spike")

    wind_stress = regimes.get("wind_stress", {})
    if wind >= float(wind_stress.get("wind_speed_min", 6.0)) and (
        temp >= float(wind_stress.get("temp_air_min", 28.0))
        or rh >= float(wind_stress.get("rh_air_min", 85.0))
        or dew_margin <= float(wind_stress.get("dew_margin_min_max", 1.0))
        or canopy_dew <= float(wind_stress.get("canopy_dew_margin_max", 1.2))
    ):
        tags.append("wind_stress")

    return [tag for tag in REGIME_ORDER if tag in set(tags)]


def tag_rows(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> List[List[str]]:
    return [tag_row(row, rows, index, config) for index, row in enumerate(rows)]


def _segments(flags: Sequence[bool], *, merge_gap: int = 1) -> List[tuple[int, int]]:
    raw: List[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            raw.append((start, index - 1))
            start = None
    if start is not None:
        raw.append((start, len(flags) - 1))
    if not raw:
        return []
    merged = [raw[0]]
    for start, end in raw[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= merge_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _window_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    regime: str,
    start_index: int,
    end_index: int,
) -> Dict[str, Any]:
    selected = list(rows[start_index : end_index + 1])
    start_row = selected[0]
    end_row = selected[-1]
    duration = len(selected)
    risk_score = (
        duration
        + sum(max(_num(row, "temp_air", 0.0) - 30.0, 0.0) for row in selected)
        + sum(max(55.0 - _num(row, "rh_air", 100.0), 0.0) * 0.05 for row in selected)
        + sum(max(_num(row, "vpd_air", _num(row, "vpd_kpa", 0.0)) - 1.6, 0.0) for row in selected)
    )
    return {
        "regime": regime,
        "start_step": int(_num(start_row, "step", start_index)),
        "end_step": int(_num(end_row, "step", end_index)),
        "duration_steps": int(duration),
        "start_hour": float(_num(start_row, "hour_of_day", 0.0)),
        "end_hour": float(_num(end_row, "hour_of_day", 0.0)),
        "min_rh": float(min(_num(row, "rh_air", 100.0) for row in selected)),
        "max_rh": float(max(_num(row, "rh_air", 0.0) for row in selected)),
        "min_temp": float(min(_num(row, "temp_air", 100.0) for row in selected)),
        "max_temp": float(max(_num(row, "temp_air", 0.0) for row in selected)),
        "max_vpd": float(max(_num(row, "vpd_air", _num(row, "vpd_kpa", 0.0)) for row in selected)),
        "max_rad_heat_load": float(max(_num(row, "rad_heat_load", _num(row, "glob_rad", 0.0)) for row in selected)),
        "max_wind_speed": float(max(_num(row, "wind_speed", 0.0) for row in selected)),
        "min_dew_margin": float(min(_dew_margin_min(row) for row in selected)),
        "min_canopy_dew_margin": float(min(_num(row, "canopy_dew_margin", 99.0) for row in selected)),
        "risk_score": float(risk_score),
    }


def merge_regime_windows(
    rows: Sequence[Mapping[str, Any]],
    row_tags: Sequence[Sequence[str]],
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    windowing = config.get("windowing", {}) if isinstance(config.get("windowing"), Mapping) else {}
    merge_gap = int(windowing.get("merge_gap_steps", 1) or 1)
    min_window_steps = int(windowing.get("min_window_steps", 1) or 1)
    windows: List[Dict[str, Any]] = []
    for regime in REGIME_ORDER:
        flags = [regime in set(tags) for tags in row_tags]
        for start_index, end_index in _segments(flags, merge_gap=merge_gap):
            if end_index - start_index + 1 < min_window_steps:
                continue
            windows.append(_window_summary(rows, regime=regime, start_index=start_index, end_index=end_index))
    return sorted(
        windows,
        key=lambda item: (
            str(item["regime"]),
            int(item["start_step"]),
            int(item["end_step"]),
        ),
    )


def _sum_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(sum(_num(row, key, 0.0) for row in rows))


def _scenario_warnings(rows: Sequence[Mapping[str, Any]], identity: TraceIdentity) -> List[str]:
    warnings: List[str] = []
    if not rows:
        warnings.append("empty_trace")
        return warnings
    keys = set().union(*(row.keys() for row in rows[: min(len(rows), 5)]))
    missing = [key for key in CRITICAL_TRACE_FIELDS if key not in keys]
    if missing:
        warnings.append(f"missing_fields:{','.join(missing)}")
    if len(rows) < identity.max_steps:
        warnings.append(f"short_trace:{len(rows)}<{identity.max_steps}")
    return warnings


def _risk_summary(rows: Sequence[Mapping[str, Any]], row_tags: Sequence[Sequence[str]], windows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tag_counts = {regime: int(sum(regime in set(tags) for tags in row_tags)) for regime in REGIME_ORDER}
    if not rows:
        return {
            "steps": 0,
            "regime_step_counts": tag_counts,
            "window_count": int(len(windows)),
            "runtime_error_steps": 0,
        }
    return {
        "steps": int(len(rows)),
        "regime_step_counts": tag_counts,
        "window_count": int(len(windows)),
        "runtime_error_steps": int(
            sum(_truthy(row.get("runtime_error", False)) or str(row.get("source", "")).lower() == "unknown" for row in rows)
        ),
        "max_temp_air": float(max(_num(row, "temp_air", 0.0) for row in rows)),
        "min_temp_air": float(min(_num(row, "temp_air", 100.0) for row in rows)),
        "min_rh_air": float(min(_num(row, "rh_air", 100.0) for row in rows)),
        "max_rh_air": float(max(_num(row, "rh_air", 0.0) for row in rows)),
        "max_vpd_air": float(max(_num(row, "vpd_air", _num(row, "vpd_kpa", 0.0)) for row in rows)),
        "max_rad_heat_load": float(max(_num(row, "rad_heat_load", _num(row, "glob_rad", 0.0)) for row in rows)),
        "max_wind_speed": float(max(_num(row, "wind_speed", 0.0) for row in rows)),
        "min_dew_margin": float(min(_dew_margin_min(row) for row in rows)),
        "min_canopy_dew_margin": float(min(_num(row, "canopy_dew_margin", 99.0) for row in rows)),
        "rh_low_area": _sum_numeric(rows, "rh_low_violation"),
        "rh_high_area": _sum_numeric(rows, "rh_high_violation"),
        "vpd_high_area": _sum_numeric(rows, "vpd_high_excess"),
        "temp_violation_area": _sum_numeric(rows, "temp_violation"),
        "mean_temp_air": float(mean(_num(row, "temp_air", 0.0) for row in rows)),
        "mean_rh_air": float(mean(_num(row, "rh_air", 0.0) for row in rows)),
    }


def assign_split(identity: TraceIdentity, config: Mapping[str, Any]) -> str:
    splits = config.get("splits", {}) if isinstance(config.get("splits"), Mapping) else {}
    for item in splits.get("regression_scenarios", []) or []:
        if not isinstance(item, Mapping):
            continue
        if (
            int(item.get("year", -1)) == identity.year
            and int(item.get("day", -1)) == identity.day
            and int(item.get("seed", -1)) == identity.seed
            and int(item.get("max_steps", identity.max_steps)) == identity.max_steps
        ):
            return "regression"

    validation = splits.get("validation", {}) if isinstance(splits.get("validation"), Mapping) else {}
    if (
        identity.year in [int(x) for x in validation.get("years", [])]
        and identity.day in [int(x) for x in validation.get("days", [])]
        and identity.seed in [int(x) for x in validation.get("seeds", [])]
        and identity.max_steps == int(validation.get("max_steps", identity.max_steps))
    ):
        return "validation"

    if identity.seed in [int(x) for x in splits.get("holdout_seeds", []) or []]:
        return "holdout"
    if identity.seed in [int(x) for x in splits.get("tuning_seeds", []) or []]:
        return "tuning"
    return str(splits.get("default_split", "tuning"))


def build_manifest(
    trace_inputs: Sequence[str | Path],
    *,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    warnings: List[str] = []
    files = discover_trace_files(trace_inputs)
    identities: List[TraceIdentity] = []
    for path in files:
        identity = parse_trace_identity(path)
        if identity is None:
            warnings.append(f"ignored_unrecognized_trace:{path}")
            continue
        identities.append(identity)

    selected = select_source_traces(identities, config)
    by_scenario: Dict[str, List[TraceIdentity]] = {}
    for identity in identities:
        by_scenario.setdefault(identity.scenario_id, []).append(identity)

    entries: List[Dict[str, Any]] = []
    top_window_count = int((config.get("windowing", {}) if isinstance(config.get("windowing"), Mapping) else {}).get("top_window_count", 5) or 5)
    for scenario_id, identity in selected.items():
        rows = sorted(read_trace(identity.path), key=lambda row: _num(row, "step", 0.0))
        scenario_warnings = _scenario_warnings(rows, identity)
        warnings.extend(f"{scenario_id}:{warning}" for warning in scenario_warnings)
        row_tags = tag_rows(rows, config)
        windows = merge_regime_windows(rows, row_tags, config)
        regime_tags = [regime for regime in REGIME_ORDER if any(regime in set(tags) for tags in row_tags)]
        top_windows = sorted(windows, key=lambda item: (-float(item["risk_score"]), -int(item["duration_steps"]), str(item["regime"])))[:top_window_count]
        entries.append(
            {
                "scenario_id": scenario_id,
                "year": identity.year,
                "day": identity.day,
                "seed": identity.seed,
                "max_steps": identity.max_steps,
                "regime_tags": regime_tags,
                "split": assign_split(identity, config),
                "risk_summary": _risk_summary(rows, row_tags, windows),
                "top_windows": top_windows,
                "source_trace": identity.path,
                "source_controller": identity.controller,
                "available_traces": [asdict(item) for item in sorted(by_scenario.get(scenario_id, []), key=lambda item: (item.controller, item.suffix, item.path))],
                "warnings": scenario_warnings,
            }
        )

    entries = sorted(entries, key=lambda item: (str(item["split"]), int(item["year"]), int(item["day"]), int(item["seed"]), int(item["max_steps"])))
    return {
        "schema_version": "regime_stress_manifest_v1",
        "suite_name": str(config.get("suite_name", "regime_stress_v1")),
        "config": {
            "path": str(DEFAULT_CONFIG_PATH),
            "schema_version": str(config.get("schema_version", "")),
        },
        "trace_inputs": [str(Path(path)) for path in trace_inputs],
        "trace_file_count": int(len(files)),
        "scenario_count": int(len(entries)),
        "regime_order": list(REGIME_ORDER),
        "scenarios": entries,
        "warnings": warnings,
    }


def _regime_aggregate(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {
        regime: {"scenario_count": 0, "window_count": 0, "steps": 0, "splits": {}, "worst_scenario": ""}
        for regime in REGIME_ORDER
    }
    for entry in entries:
        summary = entry.get("risk_summary", {}) if isinstance(entry.get("risk_summary"), Mapping) else {}
        counts = summary.get("regime_step_counts", {}) if isinstance(summary.get("regime_step_counts"), Mapping) else {}
        split = str(entry.get("split", "unknown"))
        for regime in REGIME_ORDER:
            steps = int(counts.get(regime, 0) or 0)
            if steps <= 0:
                continue
            item = out[regime]
            item["scenario_count"] += 1
            item["steps"] += steps
            item["splits"][split] = int(item["splits"].get(split, 0)) + 1
            windows = [w for w in entry.get("top_windows", []) if isinstance(w, Mapping) and w.get("regime") == regime]
            item["window_count"] += len(windows)
            if not item["worst_scenario"] or steps > int(item.get("worst_steps", 0)):
                item["worst_scenario"] = str(entry.get("scenario_id", ""))
                item["worst_steps"] = steps
    return out


def build_report(manifest: Mapping[str, Any]) -> str:
    entries = [item for item in manifest.get("scenarios", []) if isinstance(item, Mapping)]
    regime_summary = _regime_aggregate(entries)
    split_counts: Dict[str, int] = {}
    for entry in entries:
        split = str(entry.get("split", "unknown"))
        split_counts[split] = split_counts.get(split, 0) + 1

    lines = [
        "# Regime Stress Suite v1",
        "",
        f"- Scenarios: {manifest.get('scenario_count', 0)}",
        f"- Trace files scanned: {manifest.get('trace_file_count', 0)}",
        f"- Warnings: {len(manifest.get('warnings', []) or [])}",
        "",
        "## Regime Summary",
        "",
        "| regime | scenarios | tagged steps | top-window count | splits | worst scenario |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for regime in REGIME_ORDER:
        item = regime_summary[regime]
        splits = ", ".join(f"{key}:{value}" for key, value in sorted(item["splits"].items()))
        lines.append(
            "| {regime} | {scenarios} | {steps} | {windows} | {splits} | {worst} |".format(
                regime=regime,
                scenarios=int(item["scenario_count"]),
                steps=int(item["steps"]),
                windows=int(item["window_count"]),
                splits=splits or "-",
                worst=item.get("worst_scenario") or "-",
            )
        )

    lines.extend(["", "## Split Summary", "", "| split | scenarios |", "| --- | ---: |"])
    for split, count in sorted(split_counts.items()):
        lines.append(f"| {split} | {count} |")

    lines.extend(
        [
            "",
            "## Scenario Map",
            "",
            "| scenario | split | regimes | source | max temp | min RH | max VPD | max wind |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for entry in entries:
        summary = entry.get("risk_summary", {}) if isinstance(entry.get("risk_summary"), Mapping) else {}
        lines.append(
            "| {scenario} | {split} | {regimes} | {source} | {temp:.2f} | {rh:.2f} | {vpd:.2f} | {wind:.2f} |".format(
                scenario=entry.get("scenario_id", ""),
                split=entry.get("split", ""),
                regimes=", ".join(entry.get("regime_tags", []) or []) or "-",
                source=entry.get("source_controller", ""),
                temp=float(summary.get("max_temp_air", 0.0) or 0.0),
                rh=float(summary.get("min_rh_air", 0.0) or 0.0),
                vpd=float(summary.get("max_vpd_air", 0.0) or 0.0),
                wind=float(summary.get("max_wind_speed", 0.0) or 0.0),
            )
        )

    warnings = manifest.get("warnings", []) or []
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings[:100]:
            lines.append(f"- {warning}")
        if len(warnings) > 100:
            lines.append(f"- ... {len(warnings) - 100} more")
    return "\n".join(lines) + "\n"


def write_outputs(manifest: Mapping[str, Any], output_json: str | Path, output_report: str | Path) -> None:
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_report = Path(output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(build_report(manifest), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an extreme-regime stress-suite manifest from benchmark traces.")
    parser.add_argument("--input", nargs="+", required=True, help="Trace CSV/JSONL files or directories.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-json", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-report", type=str, default=str(DEFAULT_OUTPUT_REPORT))
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = build_manifest(args.input, config=config)
    manifest["config"]["path"] = str(args.config)
    write_outputs(manifest, args.output_json, args.output_report)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_report}")
    if manifest.get("warnings"):
        print(f"warnings: {len(manifest.get('warnings', []))}")


if __name__ == "__main__":
    main()
