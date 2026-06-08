"""Audit shadow sensitivity of RSPC/fallback scoring weights."""

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
    _action,
    _candidate_name,
    _candidate_score,
    _is_safe_dry_candidate,
    _parse_candidates,
    _safe_hot_dry,
    _score_terms,
    _selected_candidate,
)


SCORE_VARIANTS: Dict[str, Dict[str, float]] = {
    "dry_vpd_x1_5": {"dry": 1.80, "vpd": 1.50, "hot_dry": 1.35},
    "dry_vpd_x2": {"dry": 2.40, "vpd": 2.00, "hot_dry": 1.35},
    "hot_dry_x2": {"dry": 1.20, "vpd": 1.00, "hot_dry": 2.70},
    "balanced_hot_dry": {"dry": 1.80, "vpd": 1.60, "hot_dry": 2.00},
}
CLASSIFICATIONS = (
    "score_weight_switch_safe_dry_benefit",
    "score_weight_switch_unsafe_conflict",
    "score_weight_switch_non_dry",
    "score_weight_no_switch",
    "no_safe_dry_candidate",
    "not_safe_hot_dry",
    "missing_metadata",
)


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _term(terms: Mapping[str, Any], key: str) -> float:
    try:
        return float(terms.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _shadow_score(candidate: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    terms = _score_terms(candidate)
    return float(
        1.20 * _term(terms, "temp_penalty")
        + 1.60 * _term(terms, "rh_penalty")
        + float(weights.get("dry", 1.20)) * _term(terms, "dry_penalty")
        + float(weights.get("vpd", 1.00)) * _term(terms, "vpd_penalty")
        + 0.45 * _term(terms, "energy_penalty")
        + 0.20 * _term(terms, "smooth_penalty")
        + 0.75 * _term(terms, "conflict_penalty")
        + 0.65 * _term(terms, "dew_penalty")
        + float(weights.get("hot_dry", 1.35)) * _term(terms, "hot_dry_penalty")
        - _term(terms, "mitigation_bonus")
    )


def _dry_benefit(row: Mapping[str, Any], candidate: Mapping[str, Any], selected: Mapping[str, Any]) -> bool:
    terms = _score_terms(candidate)
    selected_terms = _score_terms(selected)
    rh_next = _term(terms, "rh_next")
    selected_rh_next = _term(selected_terms, "rh_next")
    vpd_next = _term(terms, "vpd_next")
    selected_vpd_next = _term(selected_terms, "vpd_next")
    if rh_next >= selected_rh_next + 0.5:
        return True
    if vpd_next <= selected_vpd_next - 0.03:
        return True
    if _term(terms, "dry_penalty") <= _term(selected_terms, "dry_penalty") - 0.25:
        return True
    if _term(terms, "vpd_penalty") <= _term(selected_terms, "vpd_penalty") - 0.05:
        return True
    return False


def _unsafe_candidate(row: Mapping[str, Any], candidate: Mapping[str, Any], selected: Mapping[str, Any]) -> bool:
    terms = _score_terms(candidate)
    selected_terms = _score_terms(selected)
    action = _action(candidate)
    temp_next = _term(terms, "temp_next") or _num(row, "temp_air", 20.0)
    if temp_next >= 32.0:
        return True
    if _term(terms, "temp_penalty") > _term(selected_terms, "temp_penalty") + 1.0:
        return True
    if _term(terms, "dew_penalty") > _term(selected_terms, "dew_penalty") + 0.25:
        return True
    if float(action.get("heat", 0.0) or 0.0) > 0.05:
        return True
    if float(action.get("co2", 0.0) or 0.0) > 0.05:
        return True
    if float(action.get("lamp", 0.0) or 0.0) > 0.05:
        return True
    return False


def classify_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = _parse_candidates(row)
    if not candidates or str(row.get("rspc_action_audit_enabled", "")).strip().lower() not in {"1", "true", "yes"}:
        return {"classification": "missing_metadata", "variant_results": [], "safe_hot_dry": False}

    selected = _selected_candidate(row, candidates)
    selected_name = _candidate_name(selected)
    safe_hot_dry = _safe_hot_dry(row)
    if not safe_hot_dry:
        return {
            "classification": "not_safe_hot_dry",
            "variant_results": [],
            "safe_hot_dry": False,
            "selected_name": selected_name,
        }

    safe_dry_candidates = [candidate for candidate in candidates if _is_safe_dry_candidate(row, candidate, selected)]
    variant_results: List[Dict[str, Any]] = []
    safe_switches = 0
    unsafe_switches = 0
    non_dry_switches = 0
    for variant_name, weights in SCORE_VARIANTS.items():
        selected_shadow_score = _shadow_score(selected, weights)
        best = min(candidates, key=lambda candidate: _shadow_score(candidate, weights))
        best_name = _candidate_name(best)
        best_shadow_score = _shadow_score(best, weights)
        margin = selected_shadow_score - best_shadow_score
        switched = bool(best_name and best_name != selected_name and margin > 0.05)
        unsafe = bool(switched and _unsafe_candidate(row, best, selected))
        dry = bool(switched and _dry_benefit(row, best, selected) and not unsafe)
        if dry:
            safe_switches += 1
        elif unsafe:
            unsafe_switches += 1
        elif switched:
            non_dry_switches += 1
        variant_results.append(
            {
                "variant": variant_name,
                "selected_shadow_score": round(float(selected_shadow_score), 6),
                "best_name": best_name,
                "best_shadow_score": round(float(best_shadow_score), 6),
                "margin": round(float(margin), 6),
                "switched": switched,
                "dry_benefit": dry,
                "unsafe_conflict": unsafe,
                "best_original_score": round(_candidate_score(best), 6),
                "selected_original_score": round(_candidate_score(selected), 6),
            }
        )

    if unsafe_switches > 0:
        classification = "score_weight_switch_unsafe_conflict"
    elif safe_switches > 0:
        classification = "score_weight_switch_safe_dry_benefit"
    elif non_dry_switches > 0:
        classification = "score_weight_switch_non_dry"
    elif not safe_dry_candidates:
        classification = "no_safe_dry_candidate"
    else:
        classification = "score_weight_no_switch"

    best_safe_dry = min(safe_dry_candidates, key=_candidate_score) if safe_dry_candidates else {}
    return {
        "classification": classification,
        "safe_hot_dry": True,
        "selected_name": selected_name,
        "selected_score": round(_candidate_score(selected), 6),
        "safe_dry_candidate_count": int(len(safe_dry_candidates)),
        "best_safe_dry_name": _candidate_name(best_safe_dry),
        "best_safe_dry_score": round(_candidate_score(best_safe_dry), 6),
        "safe_switch_variant_count": int(safe_switches),
        "unsafe_switch_variant_count": int(unsafe_switches),
        "non_dry_switch_variant_count": int(non_dry_switches),
        "variant_results": variant_results,
    }


def _recommendation(summary: Mapping[str, Any]) -> Dict[str, str]:
    if int(summary.get("metadata_steps", 0) or 0) <= 0:
        return {"decision": "needs_trace_metadata", "reason": "No RSPC action scoring metadata found."}
    if int(summary.get("unsafe_switch_steps", 0) or 0) > 0:
        return {
            "decision": "unsafe_weight_profile",
            "reason": "At least one alternate scoring profile would select an unsafe candidate.",
        }
    safe_switches = int(summary.get("safe_dry_switch_steps", 0) or 0)
    if safe_switches > 0:
        return {
            "decision": "score_weight_signal_detected",
            "reason": "Dry/VPD weighting can select safe humidity-retention candidates in shadow.",
        }
    if int(summary.get("no_safe_dry_candidate_steps", 0) or 0) > 0:
        return {
            "decision": "candidate_pool_gap",
            "reason": "Safe hot-dry windows often lack a safe humidity-retention candidate.",
        }
    return {"decision": "no_score_weight_signal", "reason": "Alternate scoring does not change safe hot-dry selection."}


def audit_trace(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    classified = [classify_row(row) for row in rows]
    metadata_rows = [item for item in classified if item["classification"] != "missing_metadata"]
    safe_rows = [item for item in metadata_rows if bool(item.get("safe_hot_dry", False))]
    safe_switch_rows = [
        item for item in safe_rows if item["classification"] == "score_weight_switch_safe_dry_benefit"
    ]
    unsafe_rows = [item for item in safe_rows if item["classification"] == "score_weight_switch_unsafe_conflict"]
    summary: Dict[str, Any] = {
        **trace_identity(path),
        "rows": int(len(rows)),
        "metadata_steps": int(len(metadata_rows)),
        "safe_hot_dry_steps": int(len(safe_rows)),
        "safe_dry_switch_steps": int(len(safe_switch_rows)),
        "unsafe_switch_steps": int(len(unsafe_rows)),
        "no_safe_dry_candidate_steps": int(
            sum(item["classification"] == "no_safe_dry_candidate" for item in safe_rows)
        ),
        "classification_counts": _counts(item["classification"] for item in metadata_rows),
        "selected_counts": _counts(item.get("selected_name") for item in safe_rows),
        "best_safe_dry_counts": _counts(item.get("best_safe_dry_name") for item in safe_rows),
        "variant_safe_switch_counts": {},
        "variant_unsafe_switch_counts": {},
        "mean_safe_switch_margin": 0.0,
        "warnings": [],
    }
    margins: List[float] = []
    for item in safe_rows:
        for result in item.get("variant_results", []):
            if not isinstance(result, Mapping):
                continue
            name = str(result.get("variant", "") or "")
            if bool(result.get("dry_benefit", False)):
                counts = summary["variant_safe_switch_counts"]
                counts[name] = counts.get(name, 0) + 1
                margins.append(float(result.get("margin", 0.0) or 0.0))
            if bool(result.get("unsafe_conflict", False)):
                counts = summary["variant_unsafe_switch_counts"]
                counts[name] = counts.get(name, 0) + 1
    summary["variant_safe_switch_counts"] = dict(sorted(summary["variant_safe_switch_counts"].items()))
    summary["variant_unsafe_switch_counts"] = dict(sorted(summary["variant_unsafe_switch_counts"].items()))
    summary["mean_safe_switch_margin"] = round(float(mean(margins)) if margins else 0.0, 6)
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
        "safe_dry_switch_steps": int(sum(int(trace.get("safe_dry_switch_steps", 0) or 0) for trace in traces)),
        "unsafe_switch_steps": int(sum(int(trace.get("unsafe_switch_steps", 0) or 0) for trace in traces)),
        "no_safe_dry_candidate_steps": int(
            sum(int(trace.get("no_safe_dry_candidate_steps", 0) or 0) for trace in traces)
        ),
        "classification_counts": {},
        "selected_counts": {},
        "best_safe_dry_counts": {},
        "variant_safe_switch_counts": {},
        "variant_unsafe_switch_counts": {},
        "warnings": sorted({warning for trace in traces for warning in trace.get("warnings", [])}),
    }
    for key in (
        "classification_counts",
        "selected_counts",
        "best_safe_dry_counts",
        "variant_safe_switch_counts",
        "variant_unsafe_switch_counts",
    ):
        counts: Dict[str, int] = {}
        for trace in traces:
            for name, value in trace.get(key, {}).items():
                counts[str(name)] = counts.get(str(name), 0) + int(value)
        aggregate[key] = dict(sorted(counts.items()))
    margins = [
        float(trace.get("mean_safe_switch_margin", 0.0) or 0.0)
        for trace in traces
        if float(trace.get("mean_safe_switch_margin", 0.0) or 0.0) > 0.0
    ]
    aggregate["mean_safe_switch_margin"] = round(float(mean(margins)) if margins else 0.0, 6)
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
        "# RSPC Score Weight Sensitivity Audit v1",
        "",
        "## Aggregate",
        "",
        f"- decision: `{rec.get('decision', 'unknown')}`",
        f"- reason: {rec.get('reason', '')}",
        f"- traces: {aggregate.get('trace_count', 0)}",
        f"- metadata steps: {aggregate.get('metadata_steps', 0)}",
        f"- safe hot-dry steps: {aggregate.get('safe_hot_dry_steps', 0)}",
        f"- safe dry switch steps: {aggregate.get('safe_dry_switch_steps', 0)}",
        f"- unsafe switch steps: {aggregate.get('unsafe_switch_steps', 0)}",
        f"- no safe dry candidate steps: {aggregate.get('no_safe_dry_candidate_steps', 0)}",
        f"- classification counts: {_fmt_counts(aggregate.get('classification_counts', {}))}",
        f"- variant safe switch counts: {_fmt_counts(aggregate.get('variant_safe_switch_counts', {}))}",
        f"- variant unsafe switch counts: {_fmt_counts(aggregate.get('variant_unsafe_switch_counts', {}))}",
        f"- selected counts: {_fmt_counts(aggregate.get('selected_counts', {}))}",
        f"- best safe dry counts: {_fmt_counts(aggregate.get('best_safe_dry_counts', {}))}",
        f"- mean safe switch margin: {aggregate.get('mean_safe_switch_margin', 0.0)}",
        "",
        "## Traces",
        "",
        "| trace | controller | decision | metadata | safe hot-dry | safe switch | unsafe switch | no safe dry | variants |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for trace in audit.get("traces", []):
        rec = trace.get("recommendation", {}) if isinstance(trace.get("recommendation", {}), Mapping) else {}
        lines.append(
            "| {trace_id} | {controller} | `{decision}` | {metadata} | {safe} | {switch} | {unsafe} | {no_safe} | {variants} |".format(
                trace_id=trace.get("trace_id", ""),
                controller=trace.get("controller", ""),
                decision=rec.get("decision", "unknown"),
                metadata=trace.get("metadata_steps", 0),
                safe=trace.get("safe_hot_dry_steps", 0),
                switch=trace.get("safe_dry_switch_steps", 0),
                unsafe=trace.get("unsafe_switch_steps", 0),
                no_safe=trace.get("no_safe_dry_candidate_steps", 0),
                variants=_fmt_counts(trace.get("variant_safe_switch_counts", {}), limit=4),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RSPC score-weight sensitivity from trace metadata.")
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
