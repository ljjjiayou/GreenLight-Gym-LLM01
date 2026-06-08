"""Convert a counterexample manifest into a metadata-only shadow audit plan."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _state(window: Mapping[str, Any]) -> Mapping[str, Any]:
    state = window.get("state_summary", {})
    return state if isinstance(state, Mapping) else {}


def _risk_flags(window: Mapping[str, Any]) -> list[str]:
    regime = str(window.get("regime", ""))
    state = _state(window)
    temp = _num(state.get("temp_air"), 20.0)
    rh = _num(state.get("rh_air"), 70.0)
    vpd = _num(state.get("vpd_air"), 1.0)
    canopy = _num(state.get("canopy_dew_margin"), 9.0)
    dew = _num(state.get("dew_margin_air"), 9.0)
    wind = _num(state.get("wind_speed"), 0.0)
    flags: list[str] = []
    if regime == "neutral_no_risk" and canopy > 1.0 and dew > 1.0:
        flags.append("neutral_should_not_trigger_blocker")
    if regime == "cold_humid" and temp < 16.0 and rh > 82.0:
        flags.append("cold_humid_overvent_risk")
    if regime == "dawn_dew" or min(canopy, dew) < 1.0:
        flags.append("dew_boundary_sensitive")
    if regime == "radiation_spike":
        flags.append("radiation_temp_shade_conflict_sensitive")
    if regime == "wind_stress" or wind > 6.0:
        flags.append("wind_vent_conflict_sensitive")
    if vpd > 1.6 and rh < 60.0:
        flags.append("dry_relief_preservation_sensitive")
    return flags or ["metadata_insufficient"]


def build_report(manifest: Mapping[str, Any]) -> dict[str, Any]:
    windows = list(manifest.get("windows", []) or [])
    rows: list[dict[str, Any]] = []
    regime_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    for window in windows:
        if not isinstance(window, Mapping):
            continue
        regime = str(window.get("regime", "unknown"))
        flags = _risk_flags(window)
        regime_counts[regime] += 1
        for flag in flags:
            flag_counts[flag] += 1
        rows.append(
            {
                "scenario_id": window.get("scenario_id", ""),
                "preset": window.get("preset", ""),
                "regime": regime,
                "start_step": window.get("start_step"),
                "end_step": window.get("end_step"),
                "center_step": window.get("center_step"),
                "shadow_audit_flags": flags,
                "replay_run": False,
                "cache_fill_run": False,
                "online_llm_called": False,
            }
        )
    return {
        "schema_version": "regime_counterexample_shadow_audit_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary"],
            "default_llm_rspc_v2_changed": False,
            "mode": "manifest-to-shadow-audit-plan / no replay",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "manifest_only": True,
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "regime_counts": dict(sorted(regime_counts.items())),
        "shadow_audit_flag_counts": dict(sorted(flag_counts.items())),
        "rows": rows,
        "notes": [
            "This is not safety-pass evidence.",
            "It converts the manifest into a fixed input set for future shadow audits only.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Regime Counterexample Shadow Audit Plan",
        "",
        "- Mode: manifest-to-shadow-audit-plan / no replay",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "- Replay run: false",
        "",
        "## Flag Counts",
        "",
        "| flag | count |",
        "| --- | ---: |",
    ]
    for flag, count in dict(report.get("shadow_audit_flag_counts", {})).items():
        lines.append(f"| {flag} | {count} |")
    lines.extend(["", "## Regime Counts", "", "| regime | count |", "| --- | ---: |"])
    for regime, count in dict(report.get("regime_counts", {})).items():
        lines.append(f"| {regime} | {count} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.manifest_json))
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
    print(f"window_count={len(report['rows'])}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
