"""Build the v33 near-miss source catalog after independent trigger discovery."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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


def _bool(value: Any) -> bool:
    return bool(value)


def _near_miss_rows(discovery: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in discovery.get("sample_rows", []) or []:
        if not isinstance(raw, Mapping):
            continue
        if not _bool(raw.get("strict_filtered", False)):
            continue
        if _bool(raw.get("strict_eligible_applied", False)):
            continue
        item = dict(raw)
        item["strict_applied_evidence"] = False
        item["source_hint_only"] = True
        rows.append(item)
    return rows


def _distribution(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "unknown") or "unknown") for row in rows).items()))


def _scenario_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        scenario_id = str(row.get("scenario_id", "") or "")
        if scenario_id:
            grouped[scenario_id].append(row)
    summaries: list[dict[str, Any]] = []
    for scenario_id, items in sorted(grouped.items()):
        steps = sorted({int(_num(item.get("step"), -1)) for item in items if _num(item.get("step"), -1) >= 0})
        summaries.append(
            {
                "scenario_id": scenario_id,
                "near_miss_rows": len(items),
                "strict_applied_steps": 0,
                "source_hint_only": True,
                "steps": steps,
                "reason_counts": _distribution(items, "reason"),
                "candidate_counts": _distribution(items, "candidate"),
                "variant_counts": _distribution(items, "variant"),
                "min_margin": min((_num(item.get("margin")) for item in items), default=0.0),
                "max_margin": max((_num(item.get("margin")) for item in items), default=0.0),
            }
        )
    return summaries


def _margin_gap_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gaps_by_reason: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        margin = _num(row.get("margin"))
        min_margin = _num(row.get("min_margin"), 0.2)
        gaps_by_reason[str(row.get("reason", "unknown") or "unknown")].append(min_margin - margin)
    summary: dict[str, Any] = {}
    for reason, gaps in sorted(gaps_by_reason.items()):
        summary[reason] = {
            "count": len(gaps),
            "avg_gap": sum(gaps) / len(gaps) if gaps else 0.0,
            "min_gap": min(gaps) if gaps else 0.0,
            "max_gap": max(gaps) if gaps else 0.0,
        }
    return summary


def build_report(*, blocker_diagnosis: Mapping[str, Any], discovery: Mapping[str, Any]) -> dict[str, Any]:
    rows = _near_miss_rows(discovery)
    scenario_summaries = _scenario_summary(rows)
    blocker_ready = bool(blocker_diagnosis.get("blocker_diagnosis_complete", False))
    strict_filtered_steps = int(_num(discovery.get("strict_filtered_steps", blocker_diagnosis.get("strict_filtered_steps", len(rows)))))
    catalog_ready = bool(blocker_ready and strict_filtered_steps == len(rows))
    return {
        "schema_version": "controlled_canary_near_miss_source_catalog_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "near-miss consolidation and shadow-only source hints",
            "reopens_rejected_preset": False,
        },
        "near_miss_catalog_ready": catalog_ready,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "strict_applied_evidence": False,
        "source_hints_only": True,
        "v32_blocker_taxonomy": list(blocker_diagnosis.get("blocker_taxonomy", []) or []),
        "excluded_scenario_ids": sorted(set(str(item) for item in discovery.get("excluded_scenario_ids", []) or [])),
        "near_miss_scenario_count": len(scenario_summaries),
        "near_miss_row_count": len(rows),
        "strict_filtered_steps": strict_filtered_steps,
        "strict_filter_explained_steps": int(_num(blocker_diagnosis.get("strict_filter_explained_steps", len(rows)))),
        "strict_filter_reason_distribution": dict(blocker_diagnosis.get("strict_filter_reason_distribution", {}) or _distribution(rows, "reason")),
        "candidate_distribution": dict(blocker_diagnosis.get("candidate_distribution", {}) or _distribution(rows, "candidate")),
        "variant_distribution": dict(blocker_diagnosis.get("variant_distribution", {}) or _distribution(rows, "variant")),
        "margin_gap_summary": dict(blocker_diagnosis.get("margin_gap_summary", {}) or _margin_gap_summary(rows)),
        "source_scenarios": scenario_summaries,
        "near_miss_rows": rows,
        "failure_taxonomy": sorted(
            set(
                list(blocker_diagnosis.get("blocker_taxonomy", []) or [])
                + ["near_miss_not_strict_applied_evidence"]
            )
        ),
        "next_action": "existing_cache_source_inventory" if catalog_ready else "near_miss_catalog_blocked",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Controlled Canary Near-Miss Source Catalog v33",
        "",
        f"- Catalog ready: {report.get('near_miss_catalog_ready', False)}",
        "- Source hints only: true",
        "- Strict-applied evidence: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Near-miss scenarios: {report.get('near_miss_scenario_count', 0)}",
        f"- Near-miss rows: {report.get('near_miss_row_count', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Reason Distribution",
        "",
        "| reason | rows |",
        "| --- | ---: |",
    ]
    for reason, count in dict(report.get("strict_filter_reason_distribution", {})).items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Source Hint Scenarios", "", "| scenario | rows | steps | candidates |", "| --- | ---: | --- | --- |"])
    for item in report.get("source_scenarios", []) or []:
        if not isinstance(item, Mapping):
            continue
        steps = ",".join(str(step) for step in item.get("steps", []) or [])
        candidates = ",".join(dict(item.get("candidate_counts", {}) or {}).keys())
        lines.append(f"| {item.get('scenario_id', '')} | {item.get('near_miss_rows', 0)} | {steps} | {candidates} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocker-diagnosis-json", required=True)
    parser.add_argument("--discovery-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        blocker_diagnosis=_load(args.blocker_diagnosis_json),
        discovery=_load(args.discovery_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"near_miss_catalog_ready={report['near_miss_catalog_ready']}")
    print(f"near_miss_scenario_count={report['near_miss_scenario_count']}")
    print(f"near_miss_row_count={report['near_miss_row_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["near_miss_catalog_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
