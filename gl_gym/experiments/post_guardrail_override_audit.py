"""Audit post-score guardrail and Tomato Safety v2 action rewrites.

This read-only audit focuses on whether post-processing protects the system,
does nothing material, or introduces/amplifies canopy-dew risk in a selected
window.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from gl_gym.experiments.candidate_pool_audit import classify_step as _classify_candidate_pool_step
except Exception:  # pragma: no cover - import fallback for standalone partial environments.
    _classify_candidate_pool_step = None

ACTION_FIELDS = ("heat", "co2", "screen", "vent", "lamp", "shade")
TRACE_ACTION_FIELDS = {
    "heat": "u_heating",
    "co2": "u_co2",
    "screen": "u_screen",
    "vent": "u_ventilation",
    "lamp": "u_lighting",
    "shade": "u_shading",
}
VECTOR_ACTION_FIELDS = ("heat", "co2", "screen", "vent", "lamp", "shade")
PROXY_V2_1_BUFFER_SCALE = 0.75


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_jsonish(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return default
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            continue
    return default


def _read_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_trace(root: str | Path, scenario_id: str) -> Path:
    path = Path(root)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file():
        return path
    matches = sorted(path.rglob(f"{scenario_id}*.csv"))
    if not matches:
        raise FileNotFoundError(f"trace not found for {scenario_id} under {path}")
    return matches[0]


def _iter_trace_paths(root: str | Path, scenario_id: str) -> list[Path]:
    path = Path(root)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file():
        return [path]
    matches = sorted(path.rglob(f"{scenario_id}*.csv"))
    if not matches:
        raise FileNotFoundError(f"trace not found for {scenario_id} under {path}")
    return matches


def _rows_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(round(_num(row.get("step", row.get("timestep", 0))))): row for row in rows}


def _parse_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _parse_jsonish(row.get("rspc_action_candidates_json"), default=[])
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _selected_candidate(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    selected_name = str(row.get("rspc_action_selected_name", "") or row.get("selected_fallback_candidate", "") or "").strip()
    for candidate in candidates:
        if _truthy(candidate.get("selected")) or str(candidate.get("name", "") or "").strip() == selected_name:
            return candidate
    return candidates[0] if candidates else {}


def _candidate_action(candidate: Mapping[str, Any], key: str = "post_shape_action") -> dict[str, float]:
    raw = candidate.get(key, {})
    if not isinstance(raw, Mapping):
        return {}
    return {field: _num(raw.get(field)) for field in ACTION_FIELDS}


def _candidate_score_terms(candidate: Mapping[str, Any]) -> dict[str, Any]:
    terms: dict[str, Any] = {}
    for key in ("score_terms", "post_shape_score_terms"):
        raw = candidate.get(key, {})
        if isinstance(raw, Mapping):
            terms.update(dict(raw))
    return terms


def _prediction_from_terms(terms: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prediction_model": terms.get("prediction_model", ""),
        "prediction_schema_version": terms.get("prediction_schema_version", ""),
        "predicted_temp_next": terms.get("predicted_temp_next", terms.get("temp_next")),
        "predicted_rh_next": terms.get("predicted_rh_next", terms.get("rh_next")),
        "predicted_vpd_next": terms.get("predicted_vpd_next", terms.get("vpd_next")),
        "predicted_dew_margin_air_next": terms.get("predicted_dew_margin_air_next", terms.get("dew_margin_next")),
        "predicted_canopy_dew_margin_next": terms.get(
            "predicted_canopy_dew_margin_next", terms.get("canopy_dew_margin_next")
        ),
        "predicted_canopy_dew_margin_next_v2": terms.get("predicted_canopy_dew_margin_next_v2"),
        "predicted_canopy_warning_v2": terms.get("predicted_canopy_warning_v2"),
        "predicted_canopy_lt0": terms.get("predicted_canopy_lt0"),
        "predicted_canopy_lt0_v2": terms.get("predicted_canopy_lt0_v2"),
    }


def _final_proxy_prediction(row: Mapping[str, Any], final_pred_v21: float) -> dict[str, Any]:
    return {
        "prediction_model": "final_action_proxy_shadow",
        "predicted_canopy_dew_margin_next_v2": row.get("final_action_predicted_canopy_dew_margin_next_v2"),
        "predicted_canopy_warning_v2": row.get("final_action_predicted_canopy_warning_v2"),
        "predicted_canopy_lt0_v2": row.get("final_action_predicted_canopy_lt0_v2"),
        "predicted_canopy_dew_margin_next_v2_1": final_pred_v21,
        "predicted_canopy_warning_v2_1": bool(final_pred_v21 < 0.25),
        "canopy_proxy_v2_risk_buffer": row.get("final_action_canopy_proxy_v2_risk_buffer"),
    }


def _vector_action(value: Any) -> dict[str, float]:
    raw = _parse_jsonish(value, default=[])
    if not isinstance(raw, list):
        return {}
    return {field: _num(raw[idx]) for idx, field in enumerate(VECTOR_ACTION_FIELDS) if idx < len(raw)}


def _final_action(row: Mapping[str, Any]) -> dict[str, float]:
    return {field: _num(row.get(trace_key)) for field, trace_key in TRACE_ACTION_FIELDS.items()}


def _action_delta(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {field: _num(left.get(field)) - _num(right.get(field)) for field in sorted(set(left) | set(right))}


def _distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    return float(sum(abs(_num(left.get(field)) - _num(right.get(field))) for field in set(left) | set(right)))


def _screen_vent_risk(action: Mapping[str, float]) -> bool:
    return bool(_num(action.get("screen")) >= 0.55 and _num(action.get("vent")) <= 0.25)


def _selected_pre_action(row: Mapping[str, Any]) -> dict[str, float]:
    candidates = _parse_candidates(row)
    selected = _selected_candidate(row, candidates)
    return _candidate_action(selected, "post_shape_action") or _candidate_action(selected, "action")


def _selected_candidate_context(row: Mapping[str, Any]) -> dict[str, Any]:
    candidates = _parse_candidates(row)
    selected = _selected_candidate(row, candidates)
    terms = _candidate_score_terms(selected)
    return {
        "candidate_name": str(selected.get("name", "") or row.get("rspc_action_selected_name", "") or ""),
        "selected_candidate": selected,
        "pre_score_selected_action": _candidate_action(selected, "post_shape_action")
        or _candidate_action(selected, "action"),
        "pre_score_proxy_prediction": _prediction_from_terms(terms),
    }


def classify_override(
    row: Mapping[str, Any],
    next_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_context = _selected_candidate_context(row)
    pre = selected_context["pre_score_selected_action"]
    final = _final_action(row)
    tomato_before = _vector_action(row.get("tomato_safety_v2_before"))
    tomato_after = _vector_action(row.get("tomato_safety_v2_after"))
    dry_before = _vector_action(row.get("rspc_action_dry_recovery_before"))
    dry_after = _vector_action(row.get("rspc_action_dry_recovery_after"))
    effective_pre = pre or tomato_before or dry_before or final
    delta = _action_delta(final, effective_pre)
    tomato_delta = _action_delta(tomato_after, tomato_before) if tomato_before and tomato_after else {}
    dry_delta = _action_delta(dry_after, dry_before) if dry_before and dry_after else {}
    override_magnitude = _distance(final, effective_pre)
    increased_screen = _num(delta.get("screen")) > 0.05
    decreased_vent = _num(delta.get("vent")) < -0.05
    increased_heat_vent_conflict = _num(final.get("heat")) > 0.05 and _num(final.get("vent")) > 0.2
    current_canopy = _num(row.get("canopy_dew_margin"), 99.0)
    next_canopy = _num(next_row.get("canopy_dew_margin"), current_canopy) if next_row else current_canopy
    canopy_delta_next = next_canopy - current_canopy
    vpd_delta_next = (_num(next_row.get("vpd_air")) - _num(row.get("vpd_air"))) if next_row else 0.0
    rh_delta_next = (_num(next_row.get("rh_air")) - _num(row.get("rh_air"))) if next_row else 0.0
    temp_delta_next = (_num(next_row.get("temp_air")) - _num(row.get("temp_air"))) if next_row else 0.0
    tomato_applied = _truthy(row.get("tomato_safety_v2_applied"))
    dry_applied = _truthy(row.get("rspc_action_dry_recovery_applied"))
    material_override = override_magnitude > 0.05 or tomato_applied or dry_applied
    final_checker_would_fail = _truthy(row.get("final_action_would_fail_canopy_boundary"))
    final_checker_predicted_canopy_lt0 = _truthy(row.get("final_action_predicted_canopy_lt0"))
    final_checker_screen_vent_conflict = _truthy(row.get("final_action_screen_vent_conflict"))
    final_checker_high_screen = _truthy(row.get("final_action_high_screen_under_canopy_risk"))
    final_checker_low_ventilation = _truthy(row.get("final_action_low_ventilation_under_canopy_risk"))
    final_checker_reason = str(row.get("final_action_risk_reason", "") or "")
    final_checker_would_fail_v2 = _truthy(row.get("final_action_would_fail_canopy_boundary_v2"))
    final_checker_warning_v2 = _truthy(row.get("final_action_predicted_canopy_warning_v2"))
    final_checker_screen_vent_conflict_v2 = _truthy(row.get("final_action_screen_vent_conflict_v2"))
    final_checker_reason_v2 = str(row.get("final_action_risk_reason_v2", "") or "")
    final_pred_v2 = _num(
        row.get("final_action_predicted_canopy_dew_margin_next_v2"),
        _num(row.get("canopy_boundary_shadow_predicted_canopy_dew_margin_next_v2"), 99.0),
    )
    final_risk_buffer_v2 = _num(
        row.get("final_action_canopy_proxy_v2_risk_buffer"),
        _num(row.get("canopy_boundary_shadow_v2_risk_buffer"), 0.0),
    )
    final_pred_v1_proxy = final_pred_v2 + max(final_risk_buffer_v2, 0.0)
    final_pred_v21 = final_pred_v1_proxy - PROXY_V2_1_BUFFER_SCALE * max(final_risk_buffer_v2, 0.0)
    final_checker_warning_v21 = bool(final_pred_v21 < 0.25)
    post_guardrail_proxy_prediction = _final_proxy_prediction(row, final_pred_v21)

    protective_direction = _num(delta.get("screen")) < -0.05 or _num(delta.get("vent")) > 0.05 or _num(delta.get("shade")) > 0.05
    harmful_direction = increased_screen or decreased_vent or increased_heat_vent_conflict
    if harmful_direction and (next_canopy < 0.0 or canopy_delta_next < -1.0):
        outcome = "harmful_override"
    elif next_canopy < 0.0 and not protective_direction:
        outcome = "insufficient_override"
    elif protective_direction and (next_canopy >= 0.0 or canopy_delta_next >= -0.25):
        outcome = "protective_override"
    elif material_override:
        outcome = "neutral_override"
    else:
        outcome = "no_material_override"
    candidate_pool_label = ""
    if _classify_candidate_pool_step is not None:
        try:
            candidate_pool_label = str(_classify_candidate_pool_step(row, next_row).get("label", "") or "")
        except Exception:
            candidate_pool_label = ""
    response_proxy_false_safe = bool(
        next_canopy < 0.0 and not final_checker_predicted_canopy_lt0 and not final_checker_would_fail
    )
    guardrail_introduced_risk = bool(outcome == "harmful_override" and material_override and harmful_direction)
    scorer_selected_unsafe = bool(
        next_canopy < 0.0
        and not guardrail_introduced_risk
        and (_num(effective_pre.get("screen")) >= 0.55 and _num(effective_pre.get("vent")) <= 0.25)
    )
    no_safe_candidate_available = bool(candidate_pool_label == "no_safe_candidate_available")
    preventable_v2 = bool(outcome == "harmful_override" and (final_checker_warning_v2 or final_checker_would_fail_v2))
    preventable_v21 = bool(outcome == "harmful_override" and final_checker_warning_v21)
    explainable_by_proxy_or_pool = bool(outcome == "harmful_override" and (preventable_v2 or no_safe_candidate_available))
    root_causes: list[str] = []
    if response_proxy_false_safe:
        root_causes.append("response_proxy_false_safe")
    if guardrail_introduced_risk:
        root_causes.append("guardrail_introduced_risk")
    if scorer_selected_unsafe:
        root_causes.append("scorer_selected_unsafe")
    if no_safe_candidate_available:
        root_causes.append("no_safe_candidate_available")

    pre_score_candidate_safe_but_post_guardrail_unsafe = bool(
        outcome == "harmful_override"
        and not _screen_vent_risk(effective_pre)
        and (_screen_vent_risk(final) or harmful_direction)
    )
    pre_score_candidate_unsafe_and_guardrail_failed = bool(
        next_canopy < 0.0 and _screen_vent_risk(effective_pre) and not protective_direction
    )
    no_candidate_can_satisfy_canopy_boundary = bool(no_safe_candidate_available)
    candidate_exists_but_ranked_low = bool(candidate_pool_label == "good_candidate_not_selected")
    proxy_warning_exists_but_not_actionable = bool(
        outcome == "harmful_override" and (final_checker_warning_v2 or final_checker_warning_v21)
    )
    screen_increase_dominant = bool(
        increased_screen and abs(_num(delta.get("screen"))) >= abs(_num(delta.get("vent")))
    )
    ventilation_decrease_dominant = bool(
        decreased_vent and abs(_num(delta.get("vent"))) >= abs(_num(delta.get("screen")))
    )
    screen_vent_coupled_risk = bool((increased_screen and decreased_vent) or _screen_vent_risk(final))
    fine_grained_root_causes: list[str] = []
    fine_flags = {
        "pre_score_candidate_safe_but_post_guardrail_unsafe": pre_score_candidate_safe_but_post_guardrail_unsafe,
        "pre_score_candidate_unsafe_and_guardrail_failed": pre_score_candidate_unsafe_and_guardrail_failed,
        "no_candidate_can_satisfy_canopy_boundary": no_candidate_can_satisfy_canopy_boundary,
        "candidate_exists_but_ranked_low": candidate_exists_but_ranked_low,
        "proxy_warning_exists_but_not_actionable": proxy_warning_exists_but_not_actionable,
        "screen_increase_dominant": screen_increase_dominant,
        "ventilation_decrease_dominant": ventilation_decrease_dominant,
        "screen_vent_coupled_risk": screen_vent_coupled_risk,
    }
    for label, active in fine_flags.items():
        if active:
            fine_grained_root_causes.append(label)
    if guardrail_introduced_risk:
        root_cause_layer = "post_guardrail"
    elif scorer_selected_unsafe:
        root_cause_layer = "pre_score_scorer"
    elif no_safe_candidate_available:
        root_cause_layer = "candidate_pool"
    elif response_proxy_false_safe:
        root_cause_layer = "response_proxy"
    elif outcome == "harmful_override":
        root_cause_layer = "unclassified_harmful_override"
    else:
        root_cause_layer = "no_harmful_override"

    return {
        "outcome_label": outcome,
        "selected_candidate_name": selected_context["candidate_name"],
        "pre_score_selected_action": pre,
        "post_guardrail_action": final,
        "post_guardrail_action_delta": delta,
        "pre_score_proxy_prediction": selected_context["pre_score_proxy_prediction"],
        "post_guardrail_proxy_prediction": post_guardrail_proxy_prediction,
        "root_cause_layer": root_cause_layer,
        "pre_guardrail_action": effective_pre,
        "final_action": final,
        "action_delta": delta,
        "tomato_safety_delta": tomato_delta,
        "dry_recovery_delta": dry_delta,
        "override_magnitude": override_magnitude,
        "tomato_safety_v2_applied": tomato_applied,
        "tomato_safety_v2_reasons": str(row.get("tomato_safety_v2_reasons", "") or ""),
        "dry_recovery_applied": dry_applied,
        "override_increased_screen": increased_screen,
        "override_decreased_ventilation": decreased_vent,
        "override_increased_heat_vent_conflict": increased_heat_vent_conflict,
        "final_checker_would_fail_canopy_boundary": final_checker_would_fail,
        "final_checker_predicted_canopy_lt0": final_checker_predicted_canopy_lt0,
        "final_checker_screen_vent_conflict": final_checker_screen_vent_conflict,
        "final_checker_would_fail_canopy_boundary_v2": final_checker_would_fail_v2,
        "final_checker_warning_v2": final_checker_warning_v2,
        "final_checker_screen_vent_conflict_v2": final_checker_screen_vent_conflict_v2,
        "final_checker_high_screen_under_canopy_risk": final_checker_high_screen,
        "final_checker_low_ventilation_under_canopy_risk": final_checker_low_ventilation,
        "final_checker_risk_reason": final_checker_reason,
        "final_checker_risk_reason_v2": final_checker_reason_v2,
        "harmful_override_caught_by_final_checker": bool(outcome == "harmful_override" and final_checker_would_fail),
        "harmful_override_preventable_by_proxy_v2": preventable_v2,
        "harmful_override_preventable_by_proxy_v2_1": preventable_v21,
        "harmful_override_explainable_by_proxy_v2_or_candidate_repair": explainable_by_proxy_or_pool,
        "override_created_screen_vent_conflict": bool(harmful_direction and material_override),
        "override_risk_lag_steps": 1 if response_proxy_false_safe else 0,
        "final_action_predicted_canopy_dew_margin_next_v2_1": final_pred_v21,
        "final_action_predicted_canopy_warning_v2_1": final_checker_warning_v21,
        "final_action_proxy_v2_1_buffer_scale": PROXY_V2_1_BUFFER_SCALE,
        "response_proxy_false_safe": response_proxy_false_safe,
        "guardrail_introduced_risk": guardrail_introduced_risk,
        "scorer_selected_unsafe": scorer_selected_unsafe,
        "no_safe_candidate_available": no_safe_candidate_available,
        "candidate_pool_label": candidate_pool_label,
        "root_causes": root_causes,
        "fine_grained_root_causes": fine_grained_root_causes,
        "pre_score_candidate_safe_but_post_guardrail_unsafe": pre_score_candidate_safe_but_post_guardrail_unsafe,
        "pre_score_candidate_unsafe_and_guardrail_failed": pre_score_candidate_unsafe_and_guardrail_failed,
        "no_candidate_can_satisfy_canopy_boundary": no_candidate_can_satisfy_canopy_boundary,
        "candidate_exists_but_ranked_low": candidate_exists_but_ranked_low,
        "proxy_warning_exists_but_not_actionable": proxy_warning_exists_but_not_actionable,
        "screen_increase_dominant": screen_increase_dominant,
        "ventilation_decrease_dominant": ventilation_decrease_dominant,
        "screen_vent_coupled_risk": screen_vent_coupled_risk,
        "next_response": {
            "d_RH": rh_delta_next,
            "d_VPD": vpd_delta_next,
            "d_temp": temp_delta_next,
            "d_canopy_dew_margin": canopy_delta_next,
            "next_canopy_dew_margin": next_canopy,
            "next_dew_margin_air": _num(next_row.get("dew_margin_air"), _num(row.get("dew_margin_air")))
            if next_row
            else _num(row.get("dew_margin_air")),
            "next_rh_air": _num(next_row.get("rh_air"), _num(row.get("rh_air"))) if next_row else _num(row.get("rh_air")),
            "next_vpd_air": _num(next_row.get("vpd_air"), _num(row.get("vpd_air"))) if next_row else _num(row.get("vpd_air")),
            "next_temp_air": _num(next_row.get("temp_air"), _num(row.get("temp_air")))
            if next_row
            else _num(row.get("temp_air")),
        },
        "actual_next_metrics": {
            "canopy_dew_margin": next_canopy,
            "dew_margin_air": _num(next_row.get("dew_margin_air"), _num(row.get("dew_margin_air")))
            if next_row
            else _num(row.get("dew_margin_air")),
            "rh_air": _num(next_row.get("rh_air"), _num(row.get("rh_air"))) if next_row else _num(row.get("rh_air")),
            "vpd_air": _num(next_row.get("vpd_air"), _num(row.get("vpd_air"))) if next_row else _num(row.get("vpd_air")),
            "temp_air": _num(next_row.get("temp_air"), _num(row.get("temp_air"))) if next_row else _num(row.get("temp_air")),
        },
    }


def audit_trace(
    *,
    trace_path: str | Path,
    preset: str,
    scenario_id: str,
    start_step: int,
    end_step: int,
) -> dict[str, Any]:
    rows_by_step = _rows_by_step(_read_trace(trace_path))
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    root_cause_counts: Counter[str] = Counter()
    fine_root_cause_counts: Counter[str] = Counter()
    for step in range(int(start_step), int(end_step) + 1):
        row = rows_by_step.get(step)
        if row is None:
            continue
        item = classify_override(row, rows_by_step.get(step + 1))
        counts[str(item.get("outcome_label", ""))] += 1
        for cause in item.get("root_causes", []) or []:
            root_cause_counts[str(cause)] += 1
        for cause in item.get("fine_grained_root_causes", []) or []:
            fine_root_cause_counts[str(cause)] += 1
        item.update(
            {
                "preset": preset,
                "scenario_id": scenario_id,
                "step": step,
                "state": {
                    "temp_air": _num(row.get("temp_air")),
                    "rh_air": _num(row.get("rh_air")),
                    "vpd_air": _num(row.get("vpd_air")),
                    "dew_margin_air": _num(row.get("dew_margin_air")),
                    "canopy_dew_margin": _num(row.get("canopy_dew_margin")),
                },
            }
        )
        rows.append(item)
    return {
        "preset": preset,
        "scenario_id": scenario_id,
        "trace_path": str(trace_path),
        "outcome_counts": dict(sorted(counts.items())),
        "root_cause_counts": dict(sorted(root_cause_counts.items())),
        "fine_grained_root_cause_counts": dict(sorted(fine_root_cause_counts.items())),
        "rows": rows,
    }


def build_report(
    *,
    trace_dirs: Sequence[str | Path],
    scenario_id: str,
    start_step: int,
    end_step: int,
) -> dict[str, Any]:
    traces = []
    aggregate: Counter[str] = Counter()
    root_aggregate: Counter[str] = Counter()
    fine_root_aggregate: Counter[str] = Counter()
    harmful_override_count = 0
    harmful_override_preventable_by_proxy_v2_count = 0
    harmful_override_preventable_by_proxy_v2_1_count = 0
    harmful_override_explainable_by_proxy_or_pool_count = 0
    harmful_root_cause_combos: Counter[str] = Counter()
    harmful_fine_root_cause_counts: Counter[str] = Counter()
    seen: set[Path] = set()
    for raw in trace_dirs:
        for trace_path in _iter_trace_paths(raw, scenario_id):
            resolved = trace_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            preset = trace_path.parent.name or "trace"
            if preset.lower() == "traces":
                preset = trace_path.stem
            trace = audit_trace(
                trace_path=trace_path,
                preset=preset,
                scenario_id=scenario_id,
                start_step=start_step,
                end_step=end_step,
            )
            traces.append(trace)
            aggregate.update(trace.get("outcome_counts", {}))
            root_aggregate.update(trace.get("root_cause_counts", {}))
            fine_root_aggregate.update(trace.get("fine_grained_root_cause_counts", {}))
            for row in trace.get("rows", []) or []:
                if not isinstance(row, Mapping):
                    continue
                if str(row.get("outcome_label", "")) == "harmful_override":
                    harmful_override_count += 1
                    if bool(row.get("harmful_override_preventable_by_proxy_v2")):
                        harmful_override_preventable_by_proxy_v2_count += 1
                    if bool(row.get("harmful_override_preventable_by_proxy_v2_1")):
                        harmful_override_preventable_by_proxy_v2_1_count += 1
                    if bool(row.get("harmful_override_explainable_by_proxy_v2_or_candidate_repair")):
                        harmful_override_explainable_by_proxy_or_pool_count += 1
                    combo = ",".join(sorted(str(item) for item in (row.get("root_causes", []) or []))) or "unclassified"
                    harmful_root_cause_combos[combo] += 1
                    for cause in row.get("fine_grained_root_causes", []) or []:
                        harmful_fine_root_cause_counts[str(cause)] += 1
    return {
        "schema_version": "post_guardrail_override_audit_v1",
        "scenario_id": scenario_id,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "trace_count": len(traces),
        "aggregate_outcome_counts": dict(sorted(aggregate.items())),
        "root_cause_summary": dict(sorted(root_aggregate.items())),
        "fine_grained_root_cause_summary": dict(sorted(fine_root_aggregate.items())),
        "harmful_fine_grained_root_cause_summary": dict(sorted(harmful_fine_root_cause_counts.items())),
        "harmful_override_count": int(harmful_override_count),
        "harmful_override_preventable_by_proxy_v2_count": int(harmful_override_preventable_by_proxy_v2_count),
        "harmful_override_preventable_by_proxy_v2_rate": (
            float(harmful_override_preventable_by_proxy_v2_count / harmful_override_count)
            if harmful_override_count
            else None
        ),
        "harmful_override_preventable_by_proxy_v2_1_count": int(harmful_override_preventable_by_proxy_v2_1_count),
        "harmful_override_preventable_by_proxy_v2_1_rate": (
            float(harmful_override_preventable_by_proxy_v2_1_count / harmful_override_count)
            if harmful_override_count
            else None
        ),
        "harmful_override_explainable_by_proxy_v2_or_candidate_repair_count": int(
            harmful_override_explainable_by_proxy_or_pool_count
        ),
        "harmful_override_explainable_by_proxy_v2_or_candidate_repair_rate": (
            float(harmful_override_explainable_by_proxy_or_pool_count / harmful_override_count)
            if harmful_override_count
            else None
        ),
        "harmful_root_cause_combinations": dict(sorted(harmful_root_cause_combos.items())),
        "traces": traces,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Override Audit",
        "",
        f"- Scenario: `{report.get('scenario_id', '')}`",
        f"- Steps: {report.get('start_step')}..{report.get('end_step')}",
        "",
        "## Aggregate Outcomes",
        "",
        "| outcome | count |",
        "| --- | ---: |",
    ]
    for label, count in dict(report.get("aggregate_outcome_counts", {})).items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Root Cause Summary", "", "| root_cause | count |", "| --- | ---: |"])
    for label, count in dict(report.get("root_cause_summary", {})).items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Fine-Grained Root Cause Summary", "", "| fine_root_cause | count |", "| --- | ---: |"])
    for label, count in dict(report.get("fine_grained_root_cause_summary", {})).items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Harmful Override Fine-Grained Root Causes", "", "| fine_root_cause | count |", "| --- | ---: |"])
    for label, count in dict(report.get("harmful_fine_grained_root_cause_summary", {})).items():
        lines.append(f"| {label} | {count} |")
    rate = report.get("harmful_override_preventable_by_proxy_v2_rate")
    rate_v21 = report.get("harmful_override_preventable_by_proxy_v2_1_rate")
    joint_rate = report.get("harmful_override_explainable_by_proxy_v2_or_candidate_repair_rate")
    lines.extend(
        [
            "",
            "## Proxy v2 Preventability",
            "",
            f"- Harmful override count: {report.get('harmful_override_count', 0)}",
            f"- Preventable by proxy v2: {report.get('harmful_override_preventable_by_proxy_v2_count', 0)}",
            f"- Preventable rate: {'-' if rate is None else f'{_num(rate):.3f}'}",
            f"- Preventable by proxy v2.1 candidate: {report.get('harmful_override_preventable_by_proxy_v2_1_count', 0)}",
            f"- Proxy v2.1 preventable rate: {'-' if rate_v21 is None else f'{_num(rate_v21):.3f}'}",
            f"- Explainable by proxy v2 or candidate repair: {report.get('harmful_override_explainable_by_proxy_v2_or_candidate_repair_count', 0)}",
            f"- Explainable rate: {'-' if joint_rate is None else f'{_num(joint_rate):.3f}'}",
            f"- Root-cause combinations: {json.dumps(report.get('harmful_root_cause_combinations', {}), sort_keys=True)}",
        ]
    )
    lines.extend(["", "## Step Summary", ""])
    lines.append("| preset | step | outcome | root_layer | selected | d_screen | d_vent | checker_v1 | checker_v2 | checker_v2_1 | root_causes | fine_root_causes | tomato | next_canopy | next_dew | next_rh | next_vpd | d_canopy |")
    lines.append("| --- | ---: | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for trace in report.get("traces", []) or []:
        for row in trace.get("rows", []) or []:
            delta = row.get("action_delta", {}) if isinstance(row.get("action_delta"), Mapping) else {}
            nxt = row.get("next_response", {}) if isinstance(row.get("next_response"), Mapping) else {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(trace.get("preset", "")),
                        str(row.get("step", "")),
                        str(row.get("outcome_label", "")),
                        str(row.get("root_cause_layer", "")),
                        str(row.get("selected_candidate_name", "")),
                        f"{_num(delta.get('screen')):.3f}",
                        f"{_num(delta.get('vent')):.3f}",
                        str(row.get("final_checker_would_fail_canopy_boundary", False)),
                        str(row.get("final_checker_warning_v2", False)),
                        str(row.get("final_action_predicted_canopy_warning_v2_1", False)),
                        ",".join(str(item) for item in row.get("root_causes", []) or []) or "-",
                        ",".join(str(item) for item in row.get("fine_grained_root_causes", []) or []) or "-",
                        str(row.get("tomato_safety_v2_reasons", "")) or "-",
                        f"{_num(nxt.get('next_canopy_dew_margin')):.3f}",
                        f"{_num(nxt.get('next_dew_margin_air')):.3f}",
                        f"{_num(nxt.get('next_rh_air')):.3f}",
                        f"{_num(nxt.get('next_vpd_air')):.3f}",
                        f"{_num(nxt.get('d_canopy_dew_margin')):.3f}",
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", action="append", required=True)
    parser.add_argument("--scenario-id", default="y2020_d120_s44_n240_llm_rspc_v2")
    parser.add_argument("--start-step", type=int, default=220)
    parser.add_argument("--end-step", type=int, default=239)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        trace_dirs=args.trace_dir,
        scenario_id=args.scenario_id,
        start_step=args.start_step,
        end_step=args.end_step,
    )
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
    print(f"outcomes={report['aggregate_outcome_counts']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
