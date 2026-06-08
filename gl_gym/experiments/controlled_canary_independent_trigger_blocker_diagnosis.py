"""Diagnose why independent triggered holdout discovery is blocked."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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


def _add_counts(target: dict[str, int], values: Mapping[str, Any] | None) -> None:
    if not isinstance(values, Mapping):
        return
    for key, value in values.items():
        if not str(key):
            continue
        target[str(key)] = target.get(str(key), 0) + int(_num(value))


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _strict_filtered_rows(discovery: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in discovery.get("sample_rows", []) or []
        if isinstance(row, Mapping) and bool(row.get("strict_filtered", False))
    ]


def _margin_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_reason: dict[str, list[float]] = {}
    for row in rows:
        reason = str(row.get("reason", "") or "unknown")
        gap = _num(row.get("min_margin")) - _num(row.get("margin"))
        by_reason.setdefault(reason, []).append(gap)
    return {
        reason: {
            "count": len(values),
            "min_gap": round(min(values), 6) if values else None,
            "max_gap": round(max(values), 6) if values else None,
            "avg_gap": round(sum(values) / len(values), 6) if values else None,
        }
        for reason, values in sorted(by_reason.items())
    }


def build_report(*, readiness_v24: Mapping[str, Any], discovery: Mapping[str, Any]) -> dict[str, Any]:
    filtered_rows = _strict_filtered_rows(discovery)
    strict_filter_reason_counts: dict[str, int] = {}
    would_apply_scenarios: list[dict[str, Any]] = []
    strict_filtered_scenarios: list[dict[str, Any]] = []
    for trace in discovery.get("traces", []) or []:
        if not isinstance(trace, Mapping):
            continue
        would_apply = int(_num(trace.get("would_apply_rows")))
        strict_filtered = int(_num(trace.get("strict_filtered_rows")))
        if would_apply > 0:
            would_apply_scenarios.append(
                {
                    "scenario_id": trace.get("scenario_id", ""),
                    "controller": trace.get("controller", ""),
                    "would_apply_rows": would_apply,
                    "strict_filtered_rows": strict_filtered,
                    "expected_strict_eligible_applied_rows": int(
                        _num(trace.get("expected_strict_eligible_applied_rows"))
                    ),
                }
            )
        if strict_filtered > 0:
            strict_filtered_scenarios.append(
                {
                    "scenario_id": trace.get("scenario_id", ""),
                    "controller": trace.get("controller", ""),
                    "strict_filtered_rows": strict_filtered,
                    "reason_counts": trace.get("reason_counts", {}),
                    "candidate_counts": trace.get("candidate_counts", {}),
                }
            )
            _add_counts(strict_filter_reason_counts, trace.get("reason_counts", {}))
    strict_filtered_steps = int(_num(discovery.get("strict_filtered_steps")))
    explained_steps = sum(strict_filter_reason_counts.values())
    blockers = list(discovery.get("blocker_taxonomy", []) or [])
    readiness_next_action = str(readiness_v24.get("next_action", "") or "")
    diagnosis_pass = bool(
        readiness_next_action
        in {
            "independent_trigger_blocker_diagnosis",
            "new_shadow_only_scenario_cache_discovery_after_independent_holdout_blocked",
        }
        and not discovery.get("independent_trigger_candidates_found", False)
        and strict_filtered_steps == explained_steps
    )
    return {
        "schema_version": "controlled_canary_independent_trigger_blocker_diagnosis_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "independent trigger blocker diagnosis",
            "reopens_rejected_preset": False,
        },
        "blocker_diagnosis_complete": diagnosis_pass,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "discovery_decision": discovery.get("discovery_decision", ""),
        "excluded_scenario_ids": list(discovery.get("excluded_scenario_ids", []) or []),
        "blocker_taxonomy": blockers,
        "strict_filtered_steps": strict_filtered_steps,
        "strict_filter_explained_steps": explained_steps,
        "strict_filter_reason_distribution": dict(sorted(strict_filter_reason_counts.items())),
        "candidate_distribution": _counts(str(row.get("candidate", "")) for row in filtered_rows),
        "variant_distribution": _counts(str(row.get("variant", "")) for row in filtered_rows),
        "margin_gap_summary": _margin_summary(filtered_rows),
        "would_apply_scenarios": sorted(would_apply_scenarios, key=lambda item: (item["scenario_id"], item["controller"])),
        "strict_filtered_scenarios": sorted(
            strict_filtered_scenarios,
            key=lambda item: (item["scenario_id"], item["controller"]),
        ),
        "next_action": (
            "near_miss_shadow_trigger_sourcing"
            if diagnosis_pass
            else "independent_trigger_blocker_diagnosis_incomplete"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Independent Trigger Blocker Diagnosis",
        "",
        f"- Complete: {report.get('blocker_diagnosis_complete', False)}",
        f"- Decision: `{report.get('discovery_decision', '')}`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Strict filtered steps: {report.get('strict_filtered_steps', 0)}",
        f"- Explained strict filtered steps: {report.get('strict_filter_explained_steps', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Strict Filter Reasons",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ]
    for reason, count in dict(report.get("strict_filter_reason_distribution", {})).items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Blockers", ""])
    blockers = report.get("blocker_taxonomy", []) or []
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-v24-json", required=True)
    parser.add_argument("--discovery-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        readiness_v24=_load(args.readiness_v24_json),
        discovery=_load(args.discovery_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"blocker_diagnosis_complete={report['blocker_diagnosis_complete']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["blocker_diagnosis_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
