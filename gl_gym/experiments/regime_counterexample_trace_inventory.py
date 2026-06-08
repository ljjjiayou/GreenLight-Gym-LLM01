"""Inventory existing trace coverage for non-hot-dry counterexample regimes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _scenario_from_path(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_llm_rspc_v2")] if stem.endswith("_llm_rspc_v2") else stem


def _iter_paths(trace_dirs: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for raw in trace_dirs:
        root = Path(raw)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        paths.extend([root] if root.is_file() else sorted(root.rglob("*.csv")))
    return paths


def _regime_tags(row: Mapping[str, Any]) -> list[str]:
    tags: list[str] = []
    temp = _num(row.get("temp_air"), 20.0)
    rh = _num(row.get("rh_air"), 70.0)
    vpd = _num(row.get("vpd_air"), 0.8)
    dew = _num(row.get("dew_margin_air"), 3.0)
    canopy = _num(row.get("canopy_dew_margin"), 3.0)
    rad = max(_num(row.get("glob_rad")), _num(row.get("forecast_rad_peak_2h")), _num(row.get("forecast_rad_mean_1h")))
    wind = max(_num(row.get("wind_speed")), _num(row.get("forecast_wind_peak_1h")))
    hour = _num(row.get("hour_of_day"), -1.0)
    if temp <= 18.0 and rh >= 82.0:
        tags.append("cold_humid")
    if 4.0 <= hour <= 8.0 and (dew < 1.5 or canopy < 1.5):
        tags.append("dawn_dew")
    if rad >= 500.0:
        tags.append("radiation_spike")
    if wind >= 5.0:
        tags.append("wind_stress")
    hot_dry = bool(
        _truthy(row.get("rspc_action_hot_dry_active"))
        or _truthy(row.get("rspc_action_hot_dry_semantic_active"))
        or _num(row.get("rh_low_violation")) > 0.0
        or _num(row.get("vpd_high_excess")) > 0.0
    )
    if (
        not hot_dry
        and 18.0 <= temp <= 28.0
        and 55.0 <= rh <= 80.0
        and 0.35 <= vpd <= 1.25
        and dew >= 2.0
        and canopy >= 2.0
    ):
        tags.append("neutral_no_risk")
    return tags


def build_report(*, trace_dirs: Sequence[str | Path], max_examples_per_regime: int = 10) -> dict[str, Any]:
    regime_counts: Counter[str] = Counter()
    scenario_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trace_count = 0
    scenario_ids: set[str] = set()
    for path in _iter_paths(trace_dirs):
        trace_count += 1
        scenario = _scenario_from_path(path)
        scenario_ids.add(scenario)
        preset = path.parent.name or "trace"
        for row in _read_trace(path):
            step = int(round(_num(row.get("step", row.get("timestep", 0)))))
            for tag in _regime_tags(row):
                regime_counts[tag] += 1
                scenario_counts[scenario][tag] += 1
                if len(examples[tag]) < max_examples_per_regime:
                    examples[tag].append(
                        {
                            "scenario_id": scenario,
                            "preset": preset,
                            "step": step,
                            "temp_air": _num(row.get("temp_air")),
                            "rh_air": _num(row.get("rh_air")),
                            "vpd_air": _num(row.get("vpd_air")),
                            "dew_margin_air": _num(row.get("dew_margin_air")),
                            "canopy_dew_margin": _num(row.get("canopy_dew_margin")),
                            "glob_rad": _num(row.get("glob_rad")),
                            "wind_speed": _num(row.get("wind_speed")),
                        }
                    )
    available = {tag: int(regime_counts.get(tag, 0)) > 0 for tag in ["cold_humid", "dawn_dew", "radiation_spike", "wind_stress", "neutral_no_risk"]}
    return {
        "schema_version": "regime_counterexample_trace_inventory_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only trace inventory",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "trace_count": trace_count,
        "scenario_count": len(scenario_ids),
        "regime_available": available,
        "regime_step_counts": dict(sorted(regime_counts.items())),
        "scenario_regime_counts": {key: dict(sorted(value.items())) for key, value in sorted(scenario_counts.items())},
        "examples": {key: value for key, value in sorted(examples.items())},
        "notes": ["Inventory only; no replay or cache fill was run."],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Regime Counterexample Trace Inventory",
        "",
        "- Mode: read-only trace inventory",
        "- Controlled replay allowed: false",
        "- No replay/cache fill/online LLM was run.",
        "",
        "## Regime Availability",
        "",
        "| regime | available | steps |",
        "| --- | --- | ---: |",
    ]
    counts = dict(report.get("regime_step_counts", {}))
    for regime, available in dict(report.get("regime_available", {})).items():
        lines.append(f"| {regime} | {available} | {counts.get(regime, 0)} |")
    lines.extend(["", "## Example Windows", ""])
    for regime, rows in dict(report.get("examples", {})).items():
        lines.append(f"### {regime}")
        lines.append("")
        lines.append("| scenario | preset | step | temp | RH | VPD | dew | canopy | rad | wind |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("scenario_id", "")),
                        str(row.get("preset", "")),
                        str(row.get("step", "")),
                        f"{_num(row.get('temp_air')):.2f}",
                        f"{_num(row.get('rh_air')):.2f}",
                        f"{_num(row.get('vpd_air')):.3f}",
                        f"{_num(row.get('dew_margin_air')):.2f}",
                        f"{_num(row.get('canopy_dew_margin')):.2f}",
                        f"{_num(row.get('glob_rad')):.1f}",
                        f"{_num(row.get('wind_speed')):.1f}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(trace_dirs=args.trace_dir)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json
    if not output_md.is_absolute():
        output_md = PROJECT_ROOT / output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"regimes={report['regime_available']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
