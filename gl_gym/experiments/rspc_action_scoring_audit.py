"""Audit RSPC/fallback action candidate scoring from benchmark traces."""

from __future__ import annotations

import argparse
import ast
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
    risk_flags,
    trace_identity,
)


ACTION_KEYS = ("heat", "co2", "screen", "vent", "lamp", "shade")
FINAL_ACTION_KEYS = {
    "heat": "u_boil",
    "co2": "u_co2",
    "screen": "u_th_scr",
    "vent": "u_ventilation",
    "lamp": "u_lamp",
    "shade": "u_bl_scr",
}
CLASSIFICATIONS = (
    "post_shape_unsafe_conflict",
    "candidate_gap",
    "scoring_gap",
    "post_shape_selection_gap",
    "post_score_override_gap",
    "safety_guard_gap",
    "no_action_layer_signal",
)


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _truthy(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    return float(mean(data)) if data else 0.0


def _parse_candidates(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = row.get("rspc_action_candidates_json", [])
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, str) and raw.strip():
        for loader in (json.loads, ast.literal_eval):
            try:
                data = loader(raw)
                if isinstance(data, list):
                    return [item for item in data if isinstance(item, dict)]
            except Exception:
                continue
    return []


def _score_terms(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    terms = candidate.get("score_terms", {})
    return terms if isinstance(terms, Mapping) else {}


def _action(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    action = candidate.get("action", {})
    return action if isinstance(action, Mapping) else {}


def _candidate_name(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("name", "") or "").strip()


def _is_proposer_name(name: str) -> bool:
    return str(name or "").strip().startswith("shadow_hot_dry_")


def _is_proposer_candidate(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("shadow_proposer", False)) or _is_proposer_name(_candidate_name(candidate))


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    try:
        value = candidate.get("score", 0.0)
        return float(value) if value is not None else 0.0
    except Exception:
        return 0.0


def _selected_candidate(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    selected_name = str(row.get("rspc_action_selected_name", "") or "").strip()
    for candidate in candidates:
        if bool(candidate.get("selected", False)) or _candidate_name(candidate) == selected_name:
            return candidate
    return candidates[0] if candidates else {}


def _dry_vent_cap(temp_air: float) -> float:
    if temp_air >= 28.0:
        return 0.35
    if temp_air >= 24.0:
        return 0.18
    return 0.12


def _safe_hot_dry(row: Mapping[str, Any]) -> bool:
    flags = risk_flags(row)
    hot_dry_active = (
        _truthy(row, "rspc_action_hot_dry_active")
        or _truthy(row, "rspc_action_hot_dry_semantic_active")
        or _truthy(row, "rspc_action_hot_dry_proposer_active")
    )
    return bool((flags["dry_performance"] or hot_dry_active) and not flags["safety"])


def _final_action(row: Mapping[str, Any]) -> Dict[str, float]:
    return {name: _num(row, key, 0.0) for name, key in FINAL_ACTION_KEYS.items()}


def _action_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return float(sum(abs(float(left.get(key, 0.0) or 0.0) - float(right.get(key, 0.0) or 0.0)) for key in ACTION_KEYS))


def _is_safe_dry_candidate(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> bool:
    if candidate is selected or bool(candidate.get("selected", False)):
        return False
    terms = _score_terms(candidate)
    selected_terms = _score_terms(selected)
    action = _action(candidate)
    temp_next = float(terms.get("temp_next", _num(row, "temp_air", 20.0)) or 20.0)
    selected_rh_next = float(selected_terms.get("rh_next", _num(row, "rh_air", 70.0)) or 70.0)
    selected_vpd_next = float(selected_terms.get("vpd_next", _num(row, "vpd_air", 0.0)) or 0.0)
    rh_next = float(terms.get("rh_next", selected_rh_next) or selected_rh_next)
    vpd_next = float(terms.get("vpd_next", selected_vpd_next) or selected_vpd_next)
    dry_penalty = float(terms.get("dry_penalty", 0.0) or 0.0)
    selected_dry = float(selected_terms.get("dry_penalty", 0.0) or 0.0)
    vpd_penalty = float(terms.get("vpd_penalty", 0.0) or 0.0)
    selected_vpd = float(selected_terms.get("vpd_penalty", 0.0) or 0.0)
    temp_penalty = float(terms.get("temp_penalty", 0.0) or 0.0)
    selected_temp = float(selected_terms.get("temp_penalty", 0.0) or 0.0)
    dew_penalty = float(terms.get("dew_penalty", 0.0) or 0.0)
    selected_dew = float(selected_terms.get("dew_penalty", 0.0) or 0.0)

    dry_better = (
        rh_next >= selected_rh_next + 0.5
        or vpd_next <= selected_vpd_next - 0.03
        or dry_penalty <= selected_dry - 0.25
        or vpd_penalty <= selected_vpd - 0.05
    )
    actuator_safe = (
        float(action.get("heat", 0.0) or 0.0) <= 0.05
        and float(action.get("co2", 0.0) or 0.0) <= 0.05
        and float(action.get("lamp", 0.0) or 0.0) <= 0.05
    )
    temp_safe = temp_next < 32.0 and temp_penalty <= selected_temp + 1.0
    dew_safe = dew_penalty <= selected_dew + 0.25
    return bool(dry_better and actuator_safe and temp_safe and dew_safe)


def _post_score_override(row: Mapping[str, Any], selected: Mapping[str, Any]) -> bool:
    selected_action = _action(selected)
    if not selected_action:
        selected_action = {key: _num(row, f"rspc_action_selected_{key}", 0.0) for key in ACTION_KEYS}
    final_action = _final_action(row)
    if _action_distance(selected_action, final_action) > 0.15:
        return True
    return bool(_truthy(row, "rspc_action_dry_recovery_applied") or _truthy(row, "tomato_safety_v2_applied"))


def classify_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    candidates = _parse_candidates(row)
    if not candidates or not _truthy(row, "rspc_action_audit_enabled"):
        return {"classification": "missing_metadata", "safe_hot_dry": False, "candidate_count": 0}

    flags = risk_flags(row)
    selected = _selected_candidate(row, candidates)
    selected_score = _candidate_score(selected)
    dry_candidates = [
        candidate for candidate in candidates if _is_safe_dry_candidate(row, candidate, selected)
    ]
    best_dry = min(dry_candidates, key=_candidate_score) if dry_candidates else {}
    safe_hot_dry = _safe_hot_dry(row)
    post_override = _post_score_override(row, selected)
    post_shape_switch = bool(_truthy(row, "rspc_action_post_shape_would_switch"))
    post_shape_margin = _num(row, "rspc_action_post_shape_margin", 0.0)
    post_shape_best_name = str(row.get("rspc_action_post_shape_best_name", "") or "").strip()
    post_shape_best_is_proposer = bool(
        _truthy(row, "rspc_action_post_shape_best_is_proposer") or _is_proposer_name(post_shape_best_name)
    )
    post_shape_raw_best_name = str(row.get("rspc_action_post_shape_raw_best_name", "") or "").strip()
    post_shape_raw_best_is_proposer = bool(
        _truthy(row, "rspc_action_post_shape_raw_best_is_proposer") or _is_proposer_name(post_shape_raw_best_name)
    )
    post_shape_alignment = str(row.get("rspc_action_post_shape_alignment", "neutral_hold") or "neutral_hold").strip()
    post_shape_unsafe = bool(
        _truthy(row, "rspc_action_post_shape_unsafe_conflict")
        or (post_shape_switch and post_shape_alignment == "unsafe_conflict")
    )
    post_shape_raw_unsafe = bool(_truthy(row, "rspc_action_post_shape_raw_unsafe_conflict"))
    post_shape_dry_benefit = bool(
        _truthy(row, "rspc_action_post_shape_dry_benefit")
        or (post_shape_switch and post_shape_alignment == "dry_benefit")
    )

    if post_shape_unsafe:
        classification = "post_shape_unsafe_conflict"
    elif flags["safety"]:
        classification = "safety_guard_gap" if _truthy(row, "tomato_safety_v2_applied") else "no_action_layer_signal"
    elif safe_hot_dry and post_shape_switch and post_shape_margin > 0.05 and post_shape_dry_benefit:
        classification = "post_shape_selection_gap"
    elif safe_hot_dry and post_override:
        classification = "post_score_override_gap"
    elif safe_hot_dry and not dry_candidates:
        classification = "candidate_gap"
    elif safe_hot_dry and best_dry and _candidate_score(best_dry) > selected_score + 0.05:
        classification = "scoring_gap"
    else:
        classification = "no_action_layer_signal"

    return {
        "classification": classification,
        "safe_hot_dry": bool(safe_hot_dry),
        "candidate_count": int(len(candidates)),
        "actual_candidate_count": int(_num(row, "rspc_action_actual_candidate_count", 0.0)),
        "hot_dry_candidate_count": int(_num(row, "rspc_action_hot_dry_candidate_count", 0.0)),
        "hot_dry_proposer_active": bool(_truthy(row, "rspc_action_hot_dry_proposer_active")),
        "hot_dry_proposer_candidate_count": int(
            _num(
                row,
                "rspc_action_hot_dry_proposer_candidate_count",
                sum(1 for candidate in candidates if _is_proposer_candidate(candidate)),
            )
        ),
        "hot_dry_proposer_best": bool(post_shape_best_is_proposer),
        "hot_dry_proposer_raw_best": bool(post_shape_raw_best_is_proposer),
        "selected_name": _candidate_name(selected),
        "selected_score": round(selected_score, 6),
        "best_dry_name": _candidate_name(best_dry),
        "best_dry_score": round(_candidate_score(best_dry), 6) if best_dry else None,
        "best_dry_score_delta": round(_candidate_score(best_dry) - selected_score, 6) if best_dry else None,
        "post_score_override": bool(post_override),
        "post_shape_enabled": bool(_truthy(row, "rspc_action_post_shape_enabled")),
        "post_shape_would_switch": bool(post_shape_switch),
        "post_shape_best_name": post_shape_best_name,
        "post_shape_best_is_proposer": bool(post_shape_best_is_proposer),
        "post_shape_margin": round(float(post_shape_margin), 6),
        "post_shape_alignment": post_shape_alignment,
        "post_shape_unsafe_conflict": bool(post_shape_unsafe),
        "post_shape_raw_unsafe_conflict": bool(post_shape_raw_unsafe),
        "post_shape_raw_best_name": post_shape_raw_best_name,
        "post_shape_raw_best_is_proposer": bool(post_shape_raw_best_is_proposer),
        "post_shape_raw_best_alignment": str(
            row.get("rspc_action_post_shape_raw_best_alignment", "neutral_hold") or "neutral_hold"
        ).strip(),
        "post_shape_dry_benefit": bool(post_shape_dry_benefit),
        "post_shape_proposer_dry_benefit": bool(
            post_shape_best_is_proposer and post_shape_dry_benefit
        ),
        "post_shape_safe_relief": bool(
            _truthy(row, "rspc_action_post_shape_safe_relief")
            or (post_shape_switch and post_shape_alignment == "safe_relief")
        ),
        "dry_candidate_count": int(len(dry_candidates)),
    }


def _recommendation(summary: Mapping[str, Any]) -> Dict[str, Any]:
    rows = int(summary.get("action_audit_steps", 0) or 0)
    if rows <= 0:
        return {"decision": "needs_trace_metadata", "reason": "No RSPC action scoring metadata found."}
    if int(summary.get("post_shape_unsafe_conflict_steps", 0) or 0) > 0:
        return {
            "decision": "post_shape_unsafe_conflict",
            "reason": "Post-shape best has positive switch signal with safety-conflict alignment.",
        }
    counts = summary.get("classification_counts", {})
    if not isinstance(counts, Mapping):
        counts = {}
    ranked = sorted(
        ((name, int(counts.get(name, 0) or 0)) for name in CLASSIFICATIONS if name != "no_action_layer_signal"),
        key=lambda item: (-item[1], item[0]),
    )
    top, value = ranked[0] if ranked else ("", 0)
    if value > 0:
        return {"decision": top, "reason": f"Dominant action-layer bottleneck is {top}."}
    return {"decision": "no_action_layer_signal", "reason": "No clear action-layer bottleneck detected."}


def audit_trace(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    classified = [classify_row(row) for row in rows]
    action_rows = [item for item in classified if item["classification"] != "missing_metadata"]
    safe_hot_dry = [item for item in action_rows if item.get("safe_hot_dry")]
    summary: Dict[str, Any] = {
        **trace_identity(path),
        "rows": int(len(rows)),
        "action_audit_steps": int(len(action_rows)),
        "safe_hot_dry_steps": int(len(safe_hot_dry)),
        "classification_counts": _counts(item["classification"] for item in action_rows),
        "selected_counts": _counts(item.get("selected_name") for item in action_rows),
        "best_dry_counts": _counts(item.get("best_dry_name") for item in action_rows),
        "post_shape_best_counts": _counts(item.get("post_shape_best_name") for item in action_rows),
        "post_shape_alignment_counts": _counts(item.get("post_shape_alignment") for item in action_rows),
        "hot_dry_candidate_steps": int(sum(int(item.get("hot_dry_candidate_count", 0) or 0) > 0 for item in action_rows)),
        "hot_dry_proposer_active_steps": int(sum(bool(item.get("hot_dry_proposer_active", False)) for item in action_rows)),
        "hot_dry_proposer_candidate_steps": int(
            sum(int(item.get("hot_dry_proposer_candidate_count", 0) or 0) > 0 for item in action_rows)
        ),
        "hot_dry_proposer_best_steps": int(sum(bool(item.get("hot_dry_proposer_best", False)) for item in action_rows)),
        "hot_dry_proposer_raw_best_steps": int(sum(bool(item.get("hot_dry_proposer_raw_best", False)) for item in action_rows)),
        "post_shape_enabled_steps": int(sum(bool(item.get("post_shape_enabled", False)) for item in action_rows)),
        "post_shape_switch_steps": int(sum(bool(item.get("post_shape_would_switch", False)) for item in action_rows)),
        "post_shape_unsafe_conflict_steps": int(sum(bool(item.get("post_shape_unsafe_conflict", False)) for item in action_rows)),
        "post_shape_dry_benefit_steps": int(sum(bool(item.get("post_shape_dry_benefit", False)) for item in action_rows)),
        "post_shape_proposer_dry_benefit_steps": int(sum(bool(item.get("post_shape_proposer_dry_benefit", False)) for item in action_rows)),
        "post_shape_safe_relief_steps": int(sum(bool(item.get("post_shape_safe_relief", False)) for item in action_rows)),
        "safe_hot_dry_post_shape_dry_benefit_steps": int(
            sum(bool(item.get("safe_hot_dry", False)) and bool(item.get("post_shape_dry_benefit", False)) for item in action_rows)
        ),
        "safe_hot_dry_proposer_dry_benefit_steps": int(
            sum(bool(item.get("safe_hot_dry", False)) and bool(item.get("post_shape_proposer_dry_benefit", False)) for item in action_rows)
        ),
        "post_shape_raw_unsafe_conflict_steps": int(sum(bool(item.get("post_shape_raw_unsafe_conflict", False)) for item in action_rows)),
        "mean_candidate_count": round(_mean(float(item.get("candidate_count", 0) or 0) for item in action_rows), 6),
        "mean_actual_candidate_count": round(_mean(float(item.get("actual_candidate_count", 0) or 0) for item in action_rows), 6),
        "mean_hot_dry_proposer_candidate_count": round(
            _mean(float(item.get("hot_dry_proposer_candidate_count", 0) or 0) for item in action_rows),
            6,
        ),
        "mean_dry_candidate_count": round(_mean(float(item.get("dry_candidate_count", 0) or 0) for item in action_rows), 6),
        "mean_post_shape_margin": round(_mean(float(item.get("post_shape_margin", 0.0) or 0.0) for item in action_rows), 6),
        "warnings": [],
    }
    if not action_rows:
        summary["warnings"].append("missing_rspc_action_scoring_metadata")
    summary["recommendation"] = _recommendation(summary)
    return summary


def audit_traces(inputs: Sequence[str | Path]) -> Dict[str, Any]:
    traces = [audit_trace(path) for path in discover_traces(inputs)]
    aggregate = {
        "trace_count": int(len(traces)),
        "rows": int(sum(int(trace.get("rows", 0) or 0) for trace in traces)),
        "action_audit_steps": int(sum(int(trace.get("action_audit_steps", 0) or 0) for trace in traces)),
        "safe_hot_dry_steps": int(sum(int(trace.get("safe_hot_dry_steps", 0) or 0) for trace in traces)),
        "hot_dry_candidate_steps": int(sum(int(trace.get("hot_dry_candidate_steps", 0) or 0) for trace in traces)),
        "hot_dry_proposer_active_steps": int(sum(int(trace.get("hot_dry_proposer_active_steps", 0) or 0) for trace in traces)),
        "hot_dry_proposer_candidate_steps": int(sum(int(trace.get("hot_dry_proposer_candidate_steps", 0) or 0) for trace in traces)),
        "hot_dry_proposer_best_steps": int(sum(int(trace.get("hot_dry_proposer_best_steps", 0) or 0) for trace in traces)),
        "hot_dry_proposer_raw_best_steps": int(sum(int(trace.get("hot_dry_proposer_raw_best_steps", 0) or 0) for trace in traces)),
        "classification_counts": {},
        "selected_counts": {},
        "best_dry_counts": {},
        "post_shape_best_counts": {},
        "post_shape_alignment_counts": {},
        "warnings": sorted({warning for trace in traces for warning in trace.get("warnings", [])}),
    }
    aggregate["post_shape_enabled_steps"] = int(sum(int(trace.get("post_shape_enabled_steps", 0) or 0) for trace in traces))
    aggregate["post_shape_switch_steps"] = int(sum(int(trace.get("post_shape_switch_steps", 0) or 0) for trace in traces))
    aggregate["post_shape_unsafe_conflict_steps"] = int(
        sum(int(trace.get("post_shape_unsafe_conflict_steps", 0) or 0) for trace in traces)
    )
    aggregate["post_shape_dry_benefit_steps"] = int(
        sum(int(trace.get("post_shape_dry_benefit_steps", 0) or 0) for trace in traces)
    )
    aggregate["post_shape_proposer_dry_benefit_steps"] = int(
        sum(int(trace.get("post_shape_proposer_dry_benefit_steps", 0) or 0) for trace in traces)
    )
    aggregate["post_shape_safe_relief_steps"] = int(
        sum(int(trace.get("post_shape_safe_relief_steps", 0) or 0) for trace in traces)
    )
    aggregate["post_shape_raw_unsafe_conflict_steps"] = int(
        sum(int(trace.get("post_shape_raw_unsafe_conflict_steps", 0) or 0) for trace in traces)
    )
    aggregate["safe_hot_dry_post_shape_dry_benefit_steps"] = int(
        sum(int(trace.get("safe_hot_dry_post_shape_dry_benefit_steps", 0) or 0) for trace in traces)
    )
    aggregate["safe_hot_dry_proposer_dry_benefit_steps"] = int(
        sum(int(trace.get("safe_hot_dry_proposer_dry_benefit_steps", 0) or 0) for trace in traces)
    )
    post_margins = [
        float(trace.get("mean_post_shape_margin", 0.0) or 0.0)
        for trace in traces
        if int(trace.get("action_audit_steps", 0) or 0) > 0
    ]
    aggregate["mean_post_shape_margin"] = round(_mean(post_margins), 6)
    for key in ("classification_counts", "selected_counts", "best_dry_counts", "post_shape_best_counts", "post_shape_alignment_counts"):
        counts: Dict[str, int] = {}
        for trace in traces:
            for name, value in trace.get(key, {}).items():
                counts[str(name)] = counts.get(str(name), 0) + int(value)
        aggregate[key] = dict(sorted(counts.items()))
    aggregate["recommendation"] = _recommendation(aggregate)
    return {"aggregate": aggregate, "traces": traces}


def _fmt_counts(counts: Mapping[str, Any]) -> str:
    if not counts:
        return "-"
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0]))))


def build_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {}) if isinstance(audit.get("aggregate", {}), Mapping) else {}
    rec = aggregate.get("recommendation", {}) if isinstance(aggregate.get("recommendation", {}), Mapping) else {}
    lines = [
        "# RSPC/Fallback Action Scoring Audit v1",
        "",
        "## Aggregate",
        "",
        f"- decision: `{rec.get('decision', 'unknown')}`",
        f"- reason: {rec.get('reason', '')}",
        f"- traces: {aggregate.get('trace_count', 0)}",
        f"- action audit steps: {aggregate.get('action_audit_steps', 0)}",
        f"- safe hot-dry steps: {aggregate.get('safe_hot_dry_steps', 0)}",
        f"- hot-dry candidate steps: {aggregate.get('hot_dry_candidate_steps', 0)}",
        f"- hot-dry proposer active steps: {aggregate.get('hot_dry_proposer_active_steps', 0)}",
        f"- hot-dry proposer candidate steps: {aggregate.get('hot_dry_proposer_candidate_steps', 0)}",
        f"- hot-dry proposer best steps: {aggregate.get('hot_dry_proposer_best_steps', 0)}",
        f"- hot-dry proposer raw-best steps: {aggregate.get('hot_dry_proposer_raw_best_steps', 0)}",
        f"- post-shape enabled steps: {aggregate.get('post_shape_enabled_steps', 0)}",
        f"- post-shape switch steps: {aggregate.get('post_shape_switch_steps', 0)}",
        f"- post-shape dry-benefit steps: {aggregate.get('post_shape_dry_benefit_steps', 0)}",
        f"- post-shape proposer dry-benefit steps: {aggregate.get('post_shape_proposer_dry_benefit_steps', 0)}",
        f"- safe hot-dry post-shape dry benefit: {aggregate.get('safe_hot_dry_post_shape_dry_benefit_steps', 0)} / {aggregate.get('safe_hot_dry_steps', 0)}",
        f"- safe hot-dry proposer dry benefit: {aggregate.get('safe_hot_dry_proposer_dry_benefit_steps', 0)} / {aggregate.get('safe_hot_dry_steps', 0)}",
        f"- post-shape safe-relief steps: {aggregate.get('post_shape_safe_relief_steps', 0)}",
        f"- post-shape unsafe-conflict steps: {aggregate.get('post_shape_unsafe_conflict_steps', 0)}",
        f"- raw post-shape unsafe-conflict steps: {aggregate.get('post_shape_raw_unsafe_conflict_steps', 0)}",
        f"- mean post-shape margin: {aggregate.get('mean_post_shape_margin', 0.0)}",
        f"- classification counts: {_fmt_counts(aggregate.get('classification_counts', {}))}",
        f"- selected counts: {_fmt_counts(aggregate.get('selected_counts', {}))}",
        f"- best dry candidate counts: {_fmt_counts(aggregate.get('best_dry_counts', {}))}",
        f"- post-shape best counts: {_fmt_counts(aggregate.get('post_shape_best_counts', {}))}",
        f"- post-shape alignment counts: {_fmt_counts(aggregate.get('post_shape_alignment_counts', {}))}",
        "",
        "## Traces",
        "",
        "| trace | controller | decision | action steps | safe hot-dry | classes | hot-dry candidate steps | proposer best | post-shape switches | dry benefit | unsafe |",
        "|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for trace in audit.get("traces", []):
        rec = trace.get("recommendation", {}) if isinstance(trace.get("recommendation", {}), Mapping) else {}
        lines.append(
            "| {trace_id} | {controller} | `{decision}` | {steps} | {safe} | {classes} | {hot} | {proposer} | {switches} | {dry} | {unsafe} |".format(
                trace_id=trace.get("trace_id", ""),
                controller=trace.get("controller", ""),
                decision=rec.get("decision", "unknown"),
                steps=trace.get("action_audit_steps", 0),
                safe=trace.get("safe_hot_dry_steps", 0),
                classes=_fmt_counts(trace.get("classification_counts", {})),
                hot=trace.get("hot_dry_candidate_steps", 0),
                proposer=trace.get("hot_dry_proposer_best_steps", 0),
                switches=trace.get("post_shape_switch_steps", 0),
                dry=trace.get("safe_hot_dry_post_shape_dry_benefit_steps", 0),
                unsafe=trace.get("post_shape_unsafe_conflict_steps", 0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RSPC/fallback action candidate scoring traces.")
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
