"""Controlled shadow audit for hot-dry action proposer candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gl_gym.experiments.profile_scorer_decision_audit import (  # noqa: E402
    discover_traces,
    read_trace,
    trace_identity,
)
from gl_gym.experiments.rspc_action_scoring_audit import (  # noqa: E402
    _candidate_name,
    _is_proposer_candidate,
    _is_safe_dry_candidate,
    _parse_candidates,
    _safe_hot_dry,
    _selected_candidate,
)
from gl_gym.experiments.rspc_score_weight_sensitivity_audit import (  # noqa: E402
    SCORE_VARIANTS,
    _dry_benefit,
    _shadow_score,
    _unsafe_candidate,
)


CONTROLLED_SCORE_VARIANTS = {
    "balanced_hot_dry": SCORE_VARIANTS["balanced_hot_dry"],
    "dry_vpd_x2": SCORE_VARIANTS["dry_vpd_x2"],
}
CLASSIFICATIONS = (
    "controlled_shadow_safe_proposer_gain",
    "controlled_shadow_no_gain",
    "controlled_shadow_no_eligible_proposer",
    "controlled_shadow_unsafe_proposer",
    "controlled_shadow_no_proposer",
    "not_safe_hot_dry",
    "missing_metadata",
)


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def classify_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = _parse_candidates(row)
    enabled = str(row.get("rspc_action_audit_enabled", "")).strip().lower() in {"1", "true", "yes"}
    if not candidates or not enabled:
        return {"classification": "missing_metadata", "safe_hot_dry": False, "variant_results": []}

    selected = _selected_candidate(row, candidates)
    selected_name = _candidate_name(selected)
    safe_hot_dry = _safe_hot_dry(row)
    if not safe_hot_dry:
        return {
            "classification": "not_safe_hot_dry",
            "safe_hot_dry": False,
            "selected_name": selected_name,
            "variant_results": [],
        }

    proposers = [candidate for candidate in candidates if _is_proposer_candidate(candidate)]
    if not proposers:
        return {
            "classification": "controlled_shadow_no_proposer",
            "safe_hot_dry": True,
            "selected_name": selected_name,
            "proposer_count": 0,
            "eligible_proposer_count": 0,
            "variant_results": [],
        }

    eligible = [
        candidate
        for candidate in proposers
        if _is_safe_dry_candidate(row, candidate, selected) and not _unsafe_candidate(row, candidate, selected)
    ]
    variant_results: List[Dict[str, Any]] = []
    safe_gain_count = 0
    unsafe_preferred_count = 0
    best_gain_margin = 0.0
    best_gain_name = ""
    best_gain_variant = ""
    for variant_name, weights in CONTROLLED_SCORE_VARIANTS.items():
        selected_score = _shadow_score(selected, weights)
        raw_best = min(proposers, key=lambda candidate: _shadow_score(candidate, weights))
        raw_margin = selected_score - _shadow_score(raw_best, weights)
        raw_unsafe = bool(raw_margin > 0.05 and _unsafe_candidate(row, raw_best, selected))
        best = min(eligible, key=lambda candidate: _shadow_score(candidate, weights)) if eligible else {}
        margin = selected_score - _shadow_score(best, weights) if best else 0.0
        dry_benefit = bool(best and margin > 0.05 and _dry_benefit(row, best, selected))
        if raw_unsafe:
            unsafe_preferred_count += 1
        if dry_benefit:
            safe_gain_count += 1
            if margin > best_gain_margin:
                best_gain_margin = float(margin)
                best_gain_name = _candidate_name(best)
                best_gain_variant = variant_name
        variant_results.append(
            {
                "variant": variant_name,
                "selected_score": round(float(selected_score), 6),
                "raw_best_name": _candidate_name(raw_best),
                "raw_margin": round(float(raw_margin), 6),
                "raw_unsafe_preferred": raw_unsafe,
                "eligible_best_name": _candidate_name(best),
                "eligible_margin": round(float(margin), 6),
                "dry_benefit": dry_benefit,
            }
        )

    if unsafe_preferred_count > 0:
        classification = "controlled_shadow_unsafe_proposer"
    elif safe_gain_count > 0:
        classification = "controlled_shadow_safe_proposer_gain"
    elif not eligible:
        classification = "controlled_shadow_no_eligible_proposer"
    else:
        classification = "controlled_shadow_no_gain"

    return {
        "classification": classification,
        "safe_hot_dry": True,
        "selected_name": selected_name,
        "proposer_count": int(len(proposers)),
        "eligible_proposer_count": int(len(eligible)),
        "safe_gain_variant_count": int(safe_gain_count),
        "unsafe_preferred_variant_count": int(unsafe_preferred_count),
        "best_gain_name": best_gain_name,
        "best_gain_variant": best_gain_variant,
        "best_gain_margin": round(float(best_gain_margin), 6),
        "variant_results": variant_results,
    }


def _recommendation(summary: Mapping[str, Any]) -> Dict[str, str]:
    if int(summary.get("metadata_steps", 0) or 0) <= 0:
        return {"decision": "needs_trace_metadata", "reason": "No RSPC action scoring metadata found."}
    if int(summary.get("unsafe_preferred_steps", 0) or 0) > 0:
        return {
            "decision": "unsafe_to_connect",
            "reason": "A calibrated profile would prefer an unsafe hot-dry proposer.",
        }
    gains = int(summary.get("safe_gain_steps", 0) or 0)
    if gains > 0:
        return {
            "decision": "controlled_shadow_signal_detected",
            "reason": "Hot-dry proposers produce safe dry-benefit signal under controlled shadow scoring.",
        }
    if int(summary.get("no_eligible_proposer_steps", 0) or 0) > 0:
        return {
            "decision": "proposer_eligibility_gap",
            "reason": "Hot-dry proposers exist but often fail safe dry eligibility.",
        }
    return {"decision": "no_controlled_shadow_signal", "reason": "Hot-dry proposers do not improve controlled shadow score."}


def audit_trace(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    classified = [classify_row(row) for row in rows]
    metadata_rows = [item for item in classified if item["classification"] != "missing_metadata"]
    safe_rows = [item for item in metadata_rows if bool(item.get("safe_hot_dry", False))]
    gain_rows = [item for item in safe_rows if item["classification"] == "controlled_shadow_safe_proposer_gain"]
    margins = [float(item.get("best_gain_margin", 0.0) or 0.0) for item in gain_rows]
    summary: Dict[str, Any] = {
        **trace_identity(path),
        "rows": int(len(rows)),
        "metadata_steps": int(len(metadata_rows)),
        "safe_hot_dry_steps": int(len(safe_rows)),
        "proposer_present_steps": int(sum(int(item.get("proposer_count", 0) or 0) > 0 for item in safe_rows)),
        "eligible_proposer_steps": int(
            sum(int(item.get("eligible_proposer_count", 0) or 0) > 0 for item in safe_rows)
        ),
        "safe_gain_steps": int(len(gain_rows)),
        "unsafe_preferred_steps": int(
            sum(item["classification"] == "controlled_shadow_unsafe_proposer" for item in safe_rows)
        ),
        "no_eligible_proposer_steps": int(
            sum(item["classification"] == "controlled_shadow_no_eligible_proposer" for item in safe_rows)
        ),
        "classification_counts": _counts(item["classification"] for item in metadata_rows),
        "selected_counts": _counts(item.get("selected_name") for item in safe_rows),
        "best_gain_counts": _counts(item.get("best_gain_name") for item in gain_rows),
        "best_gain_variant_counts": _counts(item.get("best_gain_variant") for item in gain_rows),
        "mean_best_gain_margin": round(float(mean(margins)) if margins else 0.0, 6),
        "warnings": [],
    }
    if not metadata_rows:
        summary["warnings"].append("missing_rspc_action_scoring_metadata")
    summary["recommendation"] = _recommendation(summary)
    return summary


def audit_traces(inputs: Sequence[str | Path]) -> Dict[str, Any]:
    traces = [audit_trace(path) for path in discover_traces(inputs)]
    aggregate: Dict[str, Any] = {
        "trace_count": int(len(traces)),
        "rows": int(sum(int(trace.get("rows", 0) or 0) for trace in traces)),
        "metadata_steps": int(sum(int(trace.get("metadata_steps", 0) or 0) for trace in traces)),
        "safe_hot_dry_steps": int(sum(int(trace.get("safe_hot_dry_steps", 0) or 0) for trace in traces)),
        "proposer_present_steps": int(sum(int(trace.get("proposer_present_steps", 0) or 0) for trace in traces)),
        "eligible_proposer_steps": int(sum(int(trace.get("eligible_proposer_steps", 0) or 0) for trace in traces)),
        "safe_gain_steps": int(sum(int(trace.get("safe_gain_steps", 0) or 0) for trace in traces)),
        "unsafe_preferred_steps": int(sum(int(trace.get("unsafe_preferred_steps", 0) or 0) for trace in traces)),
        "no_eligible_proposer_steps": int(
            sum(int(trace.get("no_eligible_proposer_steps", 0) or 0) for trace in traces)
        ),
        "classification_counts": {},
        "selected_counts": {},
        "best_gain_counts": {},
        "best_gain_variant_counts": {},
        "warnings": sorted({warning for trace in traces for warning in trace.get("warnings", [])}),
    }
    for key in ("classification_counts", "selected_counts", "best_gain_counts", "best_gain_variant_counts"):
        counts: Dict[str, int] = {}
        for trace in traces:
            for name, value in trace.get(key, {}).items():
                counts[str(name)] = counts.get(str(name), 0) + int(value)
        aggregate[key] = dict(sorted(counts.items()))
    margins = [
        float(trace.get("mean_best_gain_margin", 0.0) or 0.0)
        for trace in traces
        if float(trace.get("mean_best_gain_margin", 0.0) or 0.0) > 0.0
    ]
    aggregate["mean_best_gain_margin"] = round(float(mean(margins)) if margins else 0.0, 6)
    aggregate["recommendation"] = _recommendation(aggregate)
    return {"aggregate": aggregate, "traces": traces}


def _fmt_counts(counts: Mapping[str, Any], limit: int = 8) -> str:
    if not counts:
        return "-"
    items = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items)


def build_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {}) if isinstance(audit.get("aggregate", {}), Mapping) else {}
    rec = aggregate.get("recommendation", {}) if isinstance(aggregate.get("recommendation", {}), Mapping) else {}
    lines = [
        "# Hot-Dry Action Proposer Controlled Shadow Audit v1",
        "",
        "## Aggregate",
        "",
        f"- decision: `{rec.get('decision', 'unknown')}`",
        f"- reason: {rec.get('reason', '')}",
        f"- traces: {aggregate.get('trace_count', 0)}",
        f"- metadata steps: {aggregate.get('metadata_steps', 0)}",
        f"- safe hot-dry steps: {aggregate.get('safe_hot_dry_steps', 0)}",
        f"- proposer present steps: {aggregate.get('proposer_present_steps', 0)}",
        f"- eligible proposer steps: {aggregate.get('eligible_proposer_steps', 0)}",
        f"- safe gain steps: {aggregate.get('safe_gain_steps', 0)}",
        f"- unsafe preferred steps: {aggregate.get('unsafe_preferred_steps', 0)}",
        f"- no eligible proposer steps: {aggregate.get('no_eligible_proposer_steps', 0)}",
        f"- classification counts: {_fmt_counts(aggregate.get('classification_counts', {}))}",
        f"- best gain counts: {_fmt_counts(aggregate.get('best_gain_counts', {}))}",
        f"- best gain variant counts: {_fmt_counts(aggregate.get('best_gain_variant_counts', {}))}",
        f"- selected counts: {_fmt_counts(aggregate.get('selected_counts', {}))}",
        f"- mean best gain margin: {aggregate.get('mean_best_gain_margin', 0.0)}",
        "",
        "## Traces",
        "",
        "| trace | controller | decision | metadata | safe hot-dry | proposer | eligible | safe gain | unsafe | variants |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for trace in audit.get("traces", []):
        rec = trace.get("recommendation", {}) if isinstance(trace.get("recommendation", {}), Mapping) else {}
        lines.append(
            "| {trace_id} | {controller} | `{decision}` | {metadata} | {safe} | {proposer} | {eligible} | {gain} | {unsafe} | {variants} |".format(
                trace_id=trace.get("trace_id", ""),
                controller=trace.get("controller", ""),
                decision=rec.get("decision", "unknown"),
                metadata=trace.get("metadata_steps", 0),
                safe=trace.get("safe_hot_dry_steps", 0),
                proposer=trace.get("proposer_present_steps", 0),
                eligible=trace.get("eligible_proposer_steps", 0),
                gain=trace.get("safe_gain_steps", 0),
                unsafe=trace.get("unsafe_preferred_steps", 0),
                variants=_fmt_counts(trace.get("best_gain_variant_counts", {}), limit=4),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit controlled shadow hot-dry proposer selection.")
    parser.add_argument("--input-trace", action="append", required=True, help="Trace CSV/JSONL file or directory.")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-report", type=str, default="")
    args = parser.parse_args()

    audit = audit_traces(args.input_trace)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(audit)
    if args.output_report:
        path = Path(args.output_report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
