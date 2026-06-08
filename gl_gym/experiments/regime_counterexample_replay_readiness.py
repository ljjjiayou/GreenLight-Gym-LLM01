"""Check readiness for future counterexample shadow replay without running it."""

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

REPLAY_ORDER = ["neutral_no_risk", "cold_humid", "dawn_dew", "radiation_spike", "wind_stress"]
REQUIRED_WINDOW_FIELDS = ["scenario_id", "preset", "start_step", "end_step", "center_step", "state_summary"]


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def build_report(manifest: Mapping[str, Any]) -> dict[str, Any]:
    windows = list(manifest.get("windows", []) or [])
    rows: list[dict[str, Any]] = []
    regime_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    for window in windows:
        if not isinstance(window, Mapping):
            continue
        regime = str(window.get("regime", "unknown"))
        missing = [field for field in REQUIRED_WINDOW_FIELDS if field not in window]
        for field in missing:
            missing_counts[field] += 1
        regime_counts[regime] += 1
        rows.append(
            {
                "scenario_id": window.get("scenario_id", ""),
                "preset": window.get("preset", ""),
                "regime": regime,
                "start_step": window.get("start_step"),
                "end_step": window.get("end_step"),
                "missing_fields": missing,
                "ready_for_future_shadow_replay_input": not missing,
            }
        )
    return {
        "schema_version": "regime_counterexample_replay_readiness_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary"],
            "default_llm_rspc_v2_changed": False,
            "mode": "readiness-only / no replay",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "recommended_replay_order": REPLAY_ORDER,
        "regime_counts": dict(sorted(regime_counts.items())),
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "ready_for_future_shadow_replay_input": bool(rows) and not missing_counts,
        "rows": rows,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Regime Counterexample Replay Readiness",
        "",
        "- Mode: readiness-only / no replay",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Ready for future shadow replay input: {report.get('ready_for_future_shadow_replay_input', False)}",
        "",
        "## Recommended Replay Order",
        "",
    ]
    lines.extend(f"{idx}. `{name}`" for idx, name in enumerate(report.get("recommended_replay_order", []), start=1))
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
    print(f"ready_for_future_shadow_replay_input={report['ready_for_future_shadow_replay_input']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
