"""Audit root causes of post-score action overrides in RSPC/fallback traces."""

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
from gl_gym.experiments.rspc_action_scoring_audit import classify_row as classify_action_scoring_row  # noqa: E402


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
    "dry_recovery_overrides_safe_hot_dry",
    "tomato_safety_required_safety_relief",
    "tomato_safety_overrides_safe_hot_dry",
    "target_tracking_override_gap",
    "final_action_matches_shadow_benefit",
    "no_material_override",
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


def _loads(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str) and value.strip():
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(value)
            except Exception:
                continue
    return None


def _vector_to_action(value: Any) -> Dict[str, float]:
    data = _loads(value)
    if not isinstance(data, list):
        return {}
    return {name: float(data[i]) if i < len(data) else 0.0 for i, name in enumerate(ACTION_KEYS)}


def _row_action(row: Mapping[str, Any], prefix: str) -> Dict[str, float]:
    return {name: _num(row, f"{prefix}_{name}", 0.0) for name in ACTION_KEYS}


def _final_action(row: Mapping[str, Any]) -> Dict[str, float]:
    return {name: _num(row, key, 0.0) for name, key in FINAL_ACTION_KEYS.items()}


def _tomato_action(row: Mapping[str, Any], suffix: str) -> Dict[str, float]:
    from_vector = _vector_to_action(row.get(f"tomato_safety_v2_{suffix}", []))
    if from_vector:
        return from_vector
    return {
        "heat": _num(row, f"tomato_safety_v2_heat_{suffix}", 0.0),
        "co2": _num(row, f"tomato_safety_v2_co2_{suffix}", 0.0),
        "screen": _num(row, f"tomato_safety_v2_screen_{suffix}", 0.0),
        "vent": _num(row, f"tomato_safety_v2_vent_{suffix}", 0.0),
        "lamp": _num(row, f"tomato_safety_v2_lamp_{suffix}", 0.0),
        "shade": _num(row, f"tomato_safety_v2_shade_{suffix}", 0.0),
    }


def _dry_recovery_action(row: Mapping[str, Any], suffix: str) -> Dict[str, float]:
    return _vector_to_action(row.get(f"rspc_action_dry_recovery_{suffix}", []))


def _action_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return float(sum(abs(float(left.get(key, 0.0) or 0.0) - float(right.get(key, 0.0) or 0.0)) for key in ACTION_KEYS))


def _action_delta(after: Mapping[str, Any], before: Mapping[str, Any]) -> Dict[str, float]:
    return {key: float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0) for key in ACTION_KEYS}


def _dry_worsening(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    if not before or not after:
        return False
    delta = _action_delta(after, before)
    vent_push = float(delta.get("vent", 0.0) or 0.0) > 0.10
    screen_release = float(delta.get("screen", 0.0) or 0.0) < -0.10
    shade_release = float(delta.get("shade", 0.0) or 0.0) < -0.20
    buffered_relief = (
        float(delta.get("screen", 0.0) or 0.0) >= 0.10
        or float(delta.get("shade", 0.0) or 0.0) >= 0.10
    )
    if screen_release or shade_release:
        return True
    if vent_push and not buffered_relief:
        return True
    return False


def _safe_hot_dry(row: Mapping[str, Any]) -> bool:
    flags = risk_flags(row)
    hot_dry_active = (
        _truthy(row, "rspc_action_hot_dry_active")
        or _truthy(row, "rspc_action_hot_dry_semantic_active")
        or _truthy(row, "rspc_action_hot_dry_proposer_active")
    )
    return bool((flags["dry_performance"] or hot_dry_active) and not flags["safety"])


def _score_delta(row: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "temp": _num(row, "rspc_action_post_shape_delta_score_temp", 0.0),
        "rh": _num(row, "rspc_action_post_shape_delta_score_rh", 0.0),
        "dry": _num(row, "rspc_action_post_shape_delta_score_dry", 0.0),
        "vpd": _num(row, "rspc_action_post_shape_delta_score_vpd", 0.0),
        "dew": _num(row, "rspc_action_post_shape_delta_score_dew", 0.0),
        "hot_dry": _num(row, "rspc_action_post_shape_delta_score_hot_dry", 0.0),
        "energy": _num(row, "rspc_action_post_shape_delta_score_energy", 0.0),
    }


def classify_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    if not _truthy(row, "rspc_action_audit_enabled"):
        return {"classification": "missing_metadata", "safe_hot_dry": False, "material_override": False}

    action_classification = str(
        row.get("rspc_action_layer_classification")
        or row.get("rspc_action_classification")
        or classify_action_scoring_row(row).get("classification", "")
        or ""
    )
    in_post_score_override_gap = action_classification == "post_score_override_gap"
    flags = risk_flags(row)
    safety_gate = str(row.get("rspc_action_post_shape_safety_gate_reason", "none") or "none")
    safe_hot_dry = _safe_hot_dry(row)
    selected = _row_action(row, "rspc_action_selected")
    post_selected = _row_action(row, "rspc_action_post_shape_selected")
    post_best = _row_action(row, "rspc_action_post_shape_best")
    final = _final_action(row)
    dry_before = _dry_recovery_action(row, "before")
    dry_after = _dry_recovery_action(row, "after")
    tomato_before = _tomato_action(row, "before")
    tomato_after = _tomato_action(row, "after")
    selected_to_post = _action_distance(selected, post_selected)
    post_to_final = _action_distance(post_selected, final)
    selected_to_final = _action_distance(selected, final)
    final_to_best = _action_distance(final, post_best)
    dry_applied = _truthy(row, "rspc_action_dry_recovery_applied")
    tomato_applied = _truthy(row, "tomato_safety_v2_applied")
    shadow_dry_benefit = _truthy(row, "rspc_action_post_shape_dry_benefit")
    shadow_best_is_proposer = _truthy(row, "rspc_action_post_shape_best_is_proposer")
    final_matches_shadow = bool(
        shadow_dry_benefit
        and (shadow_best_is_proposer or str(row.get("rspc_action_post_shape_best_name", "")).startswith("shadow_hot_dry_"))
        and final_to_best <= 0.20
    )
    material_override = bool(
        selected_to_post > 0.15
        or post_to_final > 0.15
        or selected_to_final > 0.15
        or dry_applied
        or tomato_applied
    )

    if not in_post_score_override_gap:
        classification = "no_material_override"
    elif tomato_applied and (safety_gate != "none" or flags["safety"]):
        classification = "tomato_safety_required_safety_relief"
    elif final_matches_shadow:
        classification = "final_action_matches_shadow_benefit"
    elif safe_hot_dry and dry_applied and _dry_worsening(dry_before or selected, dry_after or final):
        classification = "dry_recovery_overrides_safe_hot_dry"
    elif safe_hot_dry and tomato_applied and _dry_worsening(tomato_before or post_selected, tomato_after or final):
        classification = "tomato_safety_overrides_safe_hot_dry"
    elif selected_to_post > 0.15 and not dry_applied and not tomato_applied:
        classification = "target_tracking_override_gap"
    else:
        classification = "no_material_override"

    return {
        "classification": classification,
        "action_layer_classification": action_classification,
        "post_score_override_gap": bool(in_post_score_override_gap),
        "safe_hot_dry": bool(safe_hot_dry),
        "safety_gate_reason": safety_gate,
        "material_override": bool(material_override),
        "dry_recovery_applied": bool(dry_applied),
        "tomato_safety_applied": bool(tomato_applied),
        "shadow_dry_benefit": bool(shadow_dry_benefit),
        "shadow_best_is_proposer": bool(shadow_best_is_proposer),
        "final_matches_shadow_benefit": bool(final_matches_shadow),
        "selected_to_post_distance": round(selected_to_post, 6),
        "post_to_final_distance": round(post_to_final, 6),
        "selected_to_final_distance": round(selected_to_final, 6),
        "final_to_shadow_best_distance": round(final_to_best, 6),
        "dry_recovery_delta": _action_delta(dry_after, dry_before) if dry_before and dry_after else {},
        "tomato_safety_delta": _action_delta(tomato_after, tomato_before) if tomato_before and tomato_after else {},
        "selected_to_post_delta": _action_delta(post_selected, selected),
        "post_to_final_delta": _action_delta(final, post_selected),
        "score_delta": _score_delta(row),
    }


def _mean_delta(items: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for action_key in ACTION_KEYS:
        values = [
            float(item.get(key, {}).get(action_key, 0.0) or 0.0)
            for item in items
            if isinstance(item.get(key, {}), Mapping)
        ]
        out[action_key] = round(_mean(values), 6)
    return out


def _mean_score_delta(items: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    keys = ("temp", "rh", "dry", "vpd", "dew", "hot_dry", "energy")
    return {
        key: round(
            _mean(
                float(item.get("score_delta", {}).get(key, 0.0) or 0.0)
                for item in items
                if isinstance(item.get("score_delta", {}), Mapping)
            ),
            6,
        )
        for key in keys
    }


def _recommendation(summary: Mapping[str, Any]) -> Dict[str, Any]:
    counts = summary.get("classification_counts", {})
    if not isinstance(counts, Mapping) or int(summary.get("post_score_override_gap_steps", 0) or 0) <= 0:
        return {"decision": "needs_trace_metadata", "reason": "No override root-cause metadata found."}
    ranked = sorted(
        ((name, int(counts.get(name, 0) or 0)) for name in CLASSIFICATIONS if name != "no_material_override"),
        key=lambda item: (-item[1], item[0]),
    )
    top, value = ranked[0] if ranked else ("", 0)
    if value > 0:
        return {"decision": top, "reason": f"Dominant post-score override root cause is {top}."}
    return {"decision": "no_material_override", "reason": "No material post-score override root cause detected."}


def audit_trace(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    classified = [classify_row(row) for row in rows]
    action_rows = [item for item in classified if item["classification"] != "missing_metadata"]
    gap_rows = [item for item in action_rows if bool(item.get("post_score_override_gap", False))]
    material_rows = [item for item in gap_rows if bool(item.get("material_override", False))]
    safe_hot_dry_rows = [item for item in action_rows if bool(item.get("safe_hot_dry", False))]
    summary: Dict[str, Any] = {
        **trace_identity(path),
        "rows": int(len(rows)),
        "override_audit_steps": int(len(action_rows)),
        "post_score_override_gap_steps": int(len(gap_rows)),
        "material_override_steps": int(len(material_rows)),
        "safe_hot_dry_steps": int(len(safe_hot_dry_rows)),
        "classification_counts": _counts(item["classification"] for item in gap_rows),
        "safety_gate_counts": _counts(item.get("safety_gate_reason") for item in gap_rows),
        "mean_selected_to_post_distance": round(_mean(float(item.get("selected_to_post_distance", 0.0) or 0.0) for item in gap_rows), 6),
        "mean_post_to_final_distance": round(_mean(float(item.get("post_to_final_distance", 0.0) or 0.0) for item in gap_rows), 6),
        "mean_selected_to_final_distance": round(_mean(float(item.get("selected_to_final_distance", 0.0) or 0.0) for item in gap_rows), 6),
        "mean_selected_to_post_delta": _mean_delta(gap_rows, "selected_to_post_delta"),
        "mean_post_to_final_delta": _mean_delta(gap_rows, "post_to_final_delta"),
        "mean_score_delta": _mean_score_delta(gap_rows),
        "warnings": [],
    }
    if not action_rows:
        summary["warnings"].append("missing_override_root_cause_metadata")
    summary["recommendation"] = _recommendation(summary)
    return summary


def audit_traces(inputs: Sequence[str | Path]) -> Dict[str, Any]:
    traces = [audit_trace(path) for path in discover_traces(inputs)]
    aggregate: Dict[str, Any] = {
        "trace_count": int(len(traces)),
        "rows": int(sum(int(trace.get("rows", 0) or 0) for trace in traces)),
        "override_audit_steps": int(sum(int(trace.get("override_audit_steps", 0) or 0) for trace in traces)),
        "post_score_override_gap_steps": int(sum(int(trace.get("post_score_override_gap_steps", 0) or 0) for trace in traces)),
        "material_override_steps": int(sum(int(trace.get("material_override_steps", 0) or 0) for trace in traces)),
        "safe_hot_dry_steps": int(sum(int(trace.get("safe_hot_dry_steps", 0) or 0) for trace in traces)),
        "classification_counts": {},
        "safety_gate_counts": {},
        "warnings": sorted({warning for trace in traces for warning in trace.get("warnings", [])}),
    }
    for key in ("classification_counts", "safety_gate_counts"):
        counts: Dict[str, int] = {}
        for trace in traces:
            for name, value in trace.get(key, {}).items():
                counts[str(name)] = counts.get(str(name), 0) + int(value)
        aggregate[key] = dict(sorted(counts.items()))
    aggregate["mean_selected_to_post_distance"] = round(
        _mean(float(trace.get("mean_selected_to_post_distance", 0.0) or 0.0) for trace in traces),
        6,
    )
    aggregate["mean_post_to_final_distance"] = round(
        _mean(float(trace.get("mean_post_to_final_distance", 0.0) or 0.0) for trace in traces),
        6,
    )
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
        "# Post-Score Override Root Cause Audit v1",
        "",
        "## Aggregate",
        "",
        f"- decision: `{rec.get('decision', 'unknown')}`",
        f"- reason: {rec.get('reason', '')}",
        f"- traces: {aggregate.get('trace_count', 0)}",
        f"- override audit steps: {aggregate.get('override_audit_steps', 0)}",
        f"- post-score override gap steps: {aggregate.get('post_score_override_gap_steps', 0)}",
        f"- material override steps: {aggregate.get('material_override_steps', 0)}",
        f"- safe hot-dry steps: {aggregate.get('safe_hot_dry_steps', 0)}",
        f"- classification counts: {_fmt_counts(aggregate.get('classification_counts', {}))}",
        f"- safety gate counts: {_fmt_counts(aggregate.get('safety_gate_counts', {}))}",
        f"- mean selected-to-post distance: {aggregate.get('mean_selected_to_post_distance', 0.0)}",
        f"- mean post-to-final distance: {aggregate.get('mean_post_to_final_distance', 0.0)}",
        "",
        "## Traces",
        "",
        "| trace | controller | decision | audit steps | post-score gaps | material | safe hot-dry | classes | gates |",
        "|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for trace in audit.get("traces", []):
        rec = trace.get("recommendation", {}) if isinstance(trace.get("recommendation", {}), Mapping) else {}
        lines.append(
            "| {trace_id} | {controller} | `{decision}` | {steps} | {gaps} | {material} | {safe} | {classes} | {gates} |".format(
                trace_id=trace.get("trace_id", ""),
                controller=trace.get("controller", ""),
                decision=rec.get("decision", "unknown"),
                steps=trace.get("override_audit_steps", 0),
                gaps=trace.get("post_score_override_gap_steps", 0),
                material=trace.get("material_override_steps", 0),
                safe=trace.get("safe_hot_dry_steps", 0),
                classes=_fmt_counts(trace.get("classification_counts", {})),
                gates=_fmt_counts(trace.get("safety_gate_counts", {})),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit post-score override root causes in RSPC traces.")
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
