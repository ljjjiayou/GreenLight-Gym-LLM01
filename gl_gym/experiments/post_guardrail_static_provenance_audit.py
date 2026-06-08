"""Infer likely post-guardrail rewrite provenance from source and recorded rows.

This audit is intentionally read-only. It does not claim complete provenance:
when runtime rule reasons are missing, it only maps harmful rewrite patterns to
candidate source families and marks runtime instrumentation as required.
"""

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


SOURCE_PATTERNS = {
    "apply_safety_guardrails_heat_vent_conflict": [
        "if vpd >= 0.4",
        "guarded[3] = 0.0",
        "guarded[3] = min(guarded[3], 0.1)",
    ],
    "apply_safety_guardrails_dry_vpd_vent_cap": [
        "elif vpd > 1.2",
        "guarded[3] = min(guarded[3], 0.3)",
    ],
    "apply_safety_guardrails_wind_cap": [
        "wind_cap",
        "guarded[3] = min(guarded[3], wind_cap)",
    ],
    "rspc_post_score_dry_recovery_vent_cap": [
        "dry_recovery_applied",
        "shaped[3] = min(shaped[3], vent_cap)",
    ],
    "humidity_memory_night_vent_cap": [
        "humidity_memory_night_vent_cap",
        "final[3] = min(float(final[3])",
    ],
}


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _read_source(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.read_text(encoding="utf-8", errors="replace")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _action_value(action: Mapping[str, Any] | Any, key: str) -> float:
    if not isinstance(action, Mapping):
        return 0.0
    aliases = {
        "heat": ("heat", "u_heating"),
        "screen": ("screen", "u_screen"),
        "vent": ("vent", "u_ventilation"),
        "shade": ("shade", "u_shading"),
    }
    for candidate in aliases.get(key, (key,)):
        if candidate in action:
            return _num(action.get(candidate))
    return 0.0


def _source_hits(source_text: str) -> dict[str, dict[str, Any]]:
    lines = source_text.splitlines()
    hits: dict[str, dict[str, Any]] = {}
    for family, patterns in SOURCE_PATTERNS.items():
        pattern_hits: list[dict[str, Any]] = []
        for pattern in patterns:
            for idx, line in enumerate(lines, start=1):
                if pattern in line:
                    pattern_hits.append({"pattern": pattern, "line": idx})
                    break
        hits[family] = {
            "all_patterns_found": len(pattern_hits) == len(patterns),
            "hits": pattern_hits,
        }
    return hits


def _infer_source_families(row: Mapping[str, Any]) -> list[str]:
    pre = row.get("pre_score_action", {})
    post = row.get("post_guardrail_action", {})
    delta_vent = _num(row.get("delta_vent"))
    post_vent = _action_value(post, "vent")
    pre_vent = _action_value(pre, "vent")
    post_heat = _action_value(post, "heat")
    post_screen = _action_value(post, "screen")
    families: list[str] = []
    if delta_vent >= -1e-6 and post_vent >= pre_vent - 1e-6:
        return ["no_vent_decrease_pattern"]
    if post_vent <= 0.11 and post_heat > 0.10:
        families.append("apply_safety_guardrails_heat_vent_conflict")
    if 0.18 <= post_vent <= 0.22 and post_screen >= 0.90:
        families.append("apply_safety_guardrails_wind_cap")
    if post_vent <= 0.31 and pre_vent > post_vent:
        families.append("apply_safety_guardrails_dry_vpd_vent_cap")
    if post_vent < pre_vent and not families:
        families.append("unknown_vent_cap_or_smoothing_path")
    if not str(row.get("tomato_safety_v2_reasons", "") or "").strip():
        families.append("runtime_rule_reason_missing")
    return families


def build_report(rewrite_report: Mapping[str, Any], source_text: str) -> dict[str, Any]:
    static_source_hits = _source_hits(source_text)
    rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    runtime_reason_missing = 0
    for row in rewrite_report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        families = _infer_source_families(row)
        for family in families:
            family_counts[family] += 1
        if "runtime_rule_reason_missing" in families:
            runtime_reason_missing += 1
        rows.append(
            {
                "scenario_id": row.get("scenario_id", ""),
                "preset": row.get("preset", ""),
                "step": row.get("step"),
                "delta_vent": _num(row.get("delta_vent")),
                "pre_score_vent": _action_value(row.get("pre_score_action", {}), "vent"),
                "post_guardrail_vent": _action_value(row.get("post_guardrail_action", {}), "vent"),
                "post_guardrail_heat": _action_value(row.get("post_guardrail_action", {}), "heat"),
                "post_guardrail_screen": _action_value(row.get("post_guardrail_action", {}), "screen"),
                "likely_source_families": families,
                "provenance_confidence": "pattern_only_not_runtime_provenance",
            }
        )
    return {
        "schema_version": "post_guardrail_static_provenance_audit_v1",
        "mainline_alignment": {
            "affected_layers": ["Safety Boundary", "Evaluation"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only static provenance audit",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "runtime_provenance_instrumentation_required": runtime_reason_missing > 0,
        "runtime_reason_missing_count": runtime_reason_missing,
        "likely_source_family_counts": dict(sorted(family_counts.items())),
        "static_source_hits": static_source_hits,
        "rows": rows,
        "notes": [
            "This report maps harmful rewrite patterns to likely source families only.",
            "It is not sufficient for promotion until runtime rule provenance is recorded.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Static Provenance Audit",
        "",
        "- Mode: read-only static provenance audit",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Runtime provenance instrumentation required: {report.get('runtime_provenance_instrumentation_required', False)}",
        f"- Runtime reason missing count: {report.get('runtime_reason_missing_count', 0)}",
        "",
        "## Likely Source Family Counts",
        "",
        "| family | count |",
        "| --- | ---: |",
    ]
    for family, count in dict(report.get("likely_source_family_counts", {})).items():
        lines.append(f"| {family} | {count} |")
    lines.extend(["", "## Rows", "", "| preset | step | d_vent | likely families |", "| --- | ---: | ---: | --- |"])
    for row in report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"| {row.get('preset', '')} | {row.get('step', '')} | {_num(row.get('delta_vent')):.3f} | "
            f"{', '.join(str(x) for x in row.get('likely_source_families', []))} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrite-json", required=True)
    parser.add_argument("--source-file", default="gl_gym/agent/llm_agent.py")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.rewrite_json), _read_source(args.source_file))
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
    print(f"runtime_reason_missing_count={report['runtime_reason_missing_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
