"""Create a fixed counterexample-suite manifest from existing trace inventory."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REGIME_ORDER = ("cold_humid", "dawn_dew", "neutral_no_risk", "radiation_spike", "wind_stress")


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _window_from_example(example: Mapping[str, Any], radius: int) -> dict[str, Any]:
    step = int(round(float(example.get("step", 0) or 0)))
    return {
        "scenario_id": example.get("scenario_id", ""),
        "preset": example.get("preset", ""),
        "center_step": step,
        "start_step": max(0, step - radius),
        "end_step": min(239, step + radius),
        "selection_source": "existing_trace_inventory",
        "state_summary": {
            "temp_air": example.get("temp_air"),
            "rh_air": example.get("rh_air"),
            "vpd_air": example.get("vpd_air"),
            "dew_margin_air": example.get("dew_margin_air"),
            "canopy_dew_margin": example.get("canopy_dew_margin"),
            "glob_rad": example.get("glob_rad"),
            "wind_speed": example.get("wind_speed"),
        },
    }


def build_report(inventory_report: Mapping[str, Any], *, max_windows_per_regime: int = 5, radius: int = 2) -> dict[str, Any]:
    windows_by_regime: dict[str, list[dict[str, Any]]] = {}
    flat: list[dict[str, Any]] = []
    examples = inventory_report.get("examples", {}) if isinstance(inventory_report.get("examples"), Mapping) else {}
    for regime in REGIME_ORDER:
        selected: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for example in examples.get(regime, []) or []:
            if not isinstance(example, Mapping):
                continue
            key = (str(example.get("scenario_id", "")), int(round(float(example.get("step", 0) or 0))))
            if key in seen:
                continue
            seen.add(key)
            item = _window_from_example(example, radius)
            item["regime"] = regime
            selected.append(item)
            flat.append(item)
            if len(selected) >= max_windows_per_regime:
                break
        windows_by_regime[regime] = selected
    return {
        "schema_version": "regime_counterexample_suite_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary"],
            "default_llm_rspc_v2_changed": False,
            "mode": "manifest-only / no replay",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "suite_name": "regime_counterexample_suite_v1",
        "notes": [
            "Manifest only; no replay, cache fill, or online LLM call is performed.",
            "These windows are inputs for a future shadow audit and are not safety-pass evidence.",
        ],
        "regime_counts": {regime: len(items) for regime, items in windows_by_regime.items()},
        "windows_by_regime": windows_by_regime,
        "windows": flat,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Regime Counterexample Suite Manifest v1",
        "",
        "- Mode: manifest-only / no replay",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "",
        "## Regime Counts",
        "",
        "| regime | windows |",
        "| --- | ---: |",
    ]
    for regime, count in dict(report.get("regime_counts", {})).items():
        lines.append(f"| {regime} | {count} |")
    lines.extend(["", "## Windows", "", "| regime | scenario | preset | start | center | end |", "| --- | --- | --- | ---: | ---: | ---: |"])
    for item in report.get("windows", []) or []:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("regime", "")),
                    str(item.get("scenario_id", "")),
                    str(item.get("preset", "")),
                    str(item.get("start_step", "")),
                    str(item.get("center_step", "")),
                    str(item.get("end_step", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-json", required=True)
    parser.add_argument("--max-windows-per-regime", type=int, default=5)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.inventory_json), max_windows_per_regime=args.max_windows_per_regime, radius=args.radius)
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
    print(f"regime_counts={report['regime_counts']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
