"""Build v78 stateful heat-dry accumulation attribution artifacts.

This stage is offline-only. It reads existing v75/v76 short traces and v77
runtime-stability artifacts to determine whether hot/dry/high-VPD CVODES
failures are driven by weather forcing, accumulated control history, Tomato
Safety rewrite coupling, or insufficient history. It does not call an online
LLM, run rollout or replay, change final actions, enable a supervisor, execute
solver fallback, or make performance/safety claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments import qwen37plus_runtime_stability_architecture_diagnosis_v77 as v77  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import AUDIT_DIR  # noqa: E402


MODEL_NAME = "qwen3.7-plus"
ARTIFACT_DATE = "20260604"
VERSION = "v78"
WINDOW_STEPS = (60, 120, 240)

V75_TRACE_DIR = v77.V75_TRACE_DIR
V77_SNAPSHOT_JSON = v77.SNAPSHOT_JSON
V77_ATTRIBUTION_JSON = v77.ATTRIBUTION_JSON
V77_DESIGN_JSON = v77.DESIGN_JSON

ATTRIBUTION_JSON = AUDIT_DIR / "qwen37plus_stateful_heat_dry_accumulation_attribution_20260604_v78.json"
ATTRIBUTION_MD = AUDIT_DIR / "qwen37plus_stateful_heat_dry_accumulation_attribution_20260604_v78.md"
DESIGN_JSON = AUDIT_DIR / "stateful_runtime_stability_supervisor_design_20260604_v78.json"
DESIGN_MD = AUDIT_DIR / "stateful_runtime_stability_supervisor_design_20260604_v78.md"
READINESS_JSON = AUDIT_DIR / "stateful_heat_dry_accumulation_readiness_20260604_v78.json"
READINESS_MD = AUDIT_DIR / "stateful_heat_dry_accumulation_readiness_20260604_v78.md"

CORE_HISTORY_FIELDS = (
    "temp_air",
    "rh_air",
    "vpd_air",
    "u_screen",
    "u_ventilation",
    "u_shading",
)
OPTIONAL_WEATHER_FIELDS = (
    "temp_out",
    "rh_out",
    "glob_rad",
    "forecast_temp_out_delta_1h",
    "forecast_rh_out_mean_1h",
    "forecast_rad_mean_1h",
    "forecast_rad_peak_2h",
)
ACTION_FIELDS = (
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
)
TOMATO_CHANNELS = ("heat", "screen", "shade", "vent")
ACCUMULATION_CATEGORIES = (
    "weather_forcing_dominant",
    "control_history_accumulation_dominant",
    "mixed_weather_control_accumulation",
    "tomato_safety_rewrite_coupled_instability",
    "insufficient_history_for_attribution",
)
BOUNDARY_FALSE_FIELDS = (
    "online_llm_called",
    "new_rollout_run",
    "default_llm_rspc_v2_changed",
    "fallback_enhanced",
    "controlled_replay_allowed",
    "controlled_replay_execution_allowed",
    "metadata_replay_execution_allowed",
    "strict_replay_allowed",
    "performance_claim_allowed",
    "promotion_evidence",
    "final_action_changed",
    "solver_fallback_executed",
    "runtime_policy_enabled",
    "runtime_supervisor_enabled",
    "rollout_command_generated",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _rel(path: str | Path) -> str:
    p = _resolve(path)
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _with_boundaries(payload: dict[str, Any]) -> dict[str, Any]:
    for field in BOUNDARY_FALSE_FIELDS:
        payload[field] = False
    return payload


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        result = float(value)
        if not math.isfinite(result):
            return float(default)
        return result
    except Exception:
        return float(default)


def _maybe_num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except Exception:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "applied"}


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    p = _resolve(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _discover_trace_paths(trace_dir: str | Path) -> list[Path]:
    root = _resolve(trace_dir)
    if not root.exists():
        return []
    return sorted(root.glob("*_llm_rspc_v2.csv"))


def _trace_identity(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    stem = p.stem
    controller = ""
    scenario_id = stem
    for suffix in ("_llm_rspc_v2", "_ppo", "_rule_based"):
        if stem.endswith(suffix):
            controller = suffix[1:]
            scenario_id = stem[: -len(suffix)]
            break
    match = re.search(r"y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)(?:_n(?P<max_steps>\d+))?", scenario_id)
    return {
        "trace_id": stem,
        "scenario_id": scenario_id,
        "controller": controller,
        "year": int(match.group("year")) if match else 0,
        "day": int(match.group("day")) if match else 0,
        "seed": int(match.group("seed")) if match else 0,
        "max_steps": int(match.group("max_steps") or 720) if match else 720,
    }


def _runtime_failure_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for row in rows:
        if str(row.get("runtime_error", "") or "").strip():
            return row
        if str(row.get("runtime_error_type", "") or "").strip():
            return row
    return None


def _step(row: Mapping[str, Any] | None, default: int = -1) -> int:
    if row is None:
        return int(default)
    return int(_num(row.get("step"), default))


def _window(rows: Sequence[Mapping[str, Any]], *, center_step: int, window_steps: int) -> list[Mapping[str, Any]]:
    start = max(0, int(center_step) - int(window_steps) + 1)
    return [row for row in rows if start <= _step(row) <= int(center_step)]


def _field_stats(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, float | None]:
    values = [_maybe_num(row.get(field)) for row in rows if row.get(field) not in (None, "")]
    clean = [value for value in values if value is not None]
    return {
        "start": clean[0] if clean else None,
        "end": clean[-1] if clean else None,
        "min": min(clean) if clean else None,
        "max": max(clean) if clean else None,
        "delta": clean[-1] - clean[0] if len(clean) >= 2 else 0.0,
        "mean": sum(clean) / len(clean) if clean else None,
    }


def _sum_excess(rows: Sequence[Mapping[str, Any]], field: str, threshold: float, *, above: bool) -> float:
    total = 0.0
    for row in rows:
        value = _maybe_num(row.get(field))
        if value is None:
            continue
        total += max(0.0, value - threshold) if above else max(0.0, threshold - value)
    return float(total)


def _history_field_status(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    columns = set().union(*(row.keys() for row in rows)) if rows else set()
    missing_core = [field for field in CORE_HISTORY_FIELDS if field not in columns]
    missing_core_values = [
        field for field in CORE_HISTORY_FIELDS if field in columns and not any(row.get(field) not in (None, "") for row in rows)
    ]
    present_weather = [
        field for field in OPTIONAL_WEATHER_FIELDS if field in columns and any(row.get(field) not in (None, "") for row in rows)
    ]
    return {
        "core_required_fields": list(CORE_HISTORY_FIELDS),
        "missing_core_columns": missing_core,
        "missing_core_values": missing_core_values,
        "present_weather_fields": present_weather,
        "weather_context_available": bool(present_weather),
        "complete": not missing_core and not missing_core_values and bool(rows),
    }


def _action_history_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    previous: dict[str, float] = {}
    previous_signs: dict[str, int] = {}
    large_delta_counts: Counter[str] = Counter()
    reversal_counts: Counter[str] = Counter()
    max_abs_delta: dict[str, float] = {field: 0.0 for field in ACTION_FIELDS}
    high_vent_steps = 0
    vent_drying_pressure = 0.0
    for row in rows:
        vent = _num(row.get("u_ventilation"))
        rh_air = _num(row.get("rh_air"), 100.0)
        rh_out = _num(row.get("rh_out"), rh_air)
        vpd = _num(row.get("vpd_air"))
        if vent >= 0.75:
            high_vent_steps += 1
        vent_drying_pressure += vent * (max(0.0, 55.0 - rh_air) / 10.0 + max(0.0, 65.0 - rh_out) / 20.0 + max(0.0, vpd - 1.8))
        for field in ACTION_FIELDS:
            value = _maybe_num(row.get(field))
            if value is None:
                continue
            if field in previous:
                delta = value - previous[field]
                abs_delta = abs(delta)
                max_abs_delta[field] = max(max_abs_delta[field], abs_delta)
                if abs_delta > 0.20:
                    large_delta_counts[field] += 1
                sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
                if sign and previous_signs.get(field, 0) and sign != previous_signs[field]:
                    reversal_counts[field] += 1
                if sign:
                    previous_signs[field] = sign
            previous[field] = value
    return {
        "high_vent_steps": int(high_vent_steps),
        "vent_drying_pressure": float(vent_drying_pressure),
        "large_delta_count_by_field": dict(sorted(large_delta_counts.items())),
        "reversal_count_by_field": dict(sorted(reversal_counts.items())),
        "large_action_delta_count": int(sum(large_delta_counts.values())),
        "action_reversal_count": int(sum(reversal_counts.values())),
        "max_abs_delta_by_field": {key: float(value) for key, value in sorted(max_abs_delta.items())},
    }


def _tomato_rewrite_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewrite_steps = 0
    rewrite_fields: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for row in rows:
        changed = False
        for channel in TOMATO_CHANNELS:
            before = _maybe_num(row.get(f"tomato_safety_v2_{channel}_before"))
            after = _maybe_num(row.get(f"tomato_safety_v2_{channel}_after"))
            if before is not None and after is not None and abs(after - before) > 1e-6:
                changed = True
                rewrite_fields[channel] += 1
        if _truthy(row.get("tomato_safety_v2_applied")):
            changed = True
        if changed:
            rewrite_steps += 1
        reasons = str(row.get("tomato_safety_v2_reasons", "") or row.get("final_action_risk_reason_v2", "") or "")
        for reason in re.split(r"[,;|]", reasons):
            reason = reason.strip()
            if reason:
                reason_counts[reason] += 1
    return {
        "rewrite_steps": int(rewrite_steps),
        "rewrite_rate": float(rewrite_steps / max(len(rows), 1)),
        "rewrite_field_counts": dict(sorted(rewrite_fields.items())),
        "rewrite_reason_counts": dict(sorted(reason_counts.items())),
    }


def _weather_forcing_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    temp_out = _field_stats(rows, "temp_out")
    rh_out = _field_stats(rows, "rh_out")
    glob_rad = _field_stats(rows, "glob_rad")
    forecast_temp = _field_stats(rows, "forecast_temp_out_delta_1h")
    forecast_rh = _field_stats(rows, "forecast_rh_out_mean_1h")
    rad_delta = float(glob_rad.get("delta") or 0.0)
    temp_delta = float(temp_out.get("delta") or 0.0)
    rh_drop = -float(rh_out.get("delta") or 0.0)
    forecast_rh_drop = -float(forecast_rh.get("delta") or 0.0)
    signals = {
        "temp_out_ramp_ge_5": temp_delta >= 5.0,
        "rh_out_drop_ge_20": rh_drop >= 20.0,
        "glob_rad_ramp_ge_300": rad_delta >= 300.0,
        "glob_rad_peak_ge_700": _num(glob_rad.get("max")) >= 700.0,
        "forecast_rh_drop_ge_20": forecast_rh_drop >= 20.0,
        "forecast_temp_delta_positive": _num(forecast_temp.get("max")) > 0.0,
    }
    return {
        "temp_out": temp_out,
        "rh_out": rh_out,
        "glob_rad": glob_rad,
        "forecast_temp_out_delta_1h": forecast_temp,
        "forecast_rh_out_mean_1h": forecast_rh,
        "signal_flags": signals,
        "signal_count": int(sum(bool(value) for value in signals.values())),
    }


def _accumulation_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    heat_debt = _sum_excess(rows, "temp_air", 28.0, above=True)
    dryness_debt = _sum_excess(rows, "rh_air", 55.0, above=False)
    vpd_debt = _sum_excess(rows, "vpd_air", 2.0, above=True)
    return {
        "heat_debt": float(heat_debt),
        "dryness_debt": float(dryness_debt),
        "vpd_debt": float(vpd_debt),
        "temp_air": _field_stats(rows, "temp_air"),
        "rh_air": _field_stats(rows, "rh_air"),
        "vpd_air": _field_stats(rows, "vpd_air"),
        "canopy_dew_margin": _field_stats(rows, "canopy_dew_margin"),
        "dew_margin_air": _field_stats(rows, "dew_margin_air"),
        "heat_dry_vpd_accumulation_present": bool(heat_debt > 10.0 and vpd_debt > 2.0 and dryness_debt > 10.0),
    }


def _window_attribution(window: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    history_status = window.get("history_field_status", {}) or {}
    if not bool(history_status.get("complete", False)):
        return "insufficient_history_for_attribution", {
            "weather_signal": False,
            "control_signal": False,
            "tomato_signal": False,
            "reason": "missing_core_history_fields",
        }
    weather = window.get("weather_forcing", {}) or {}
    action = window.get("action_history", {}) or {}
    tomato = window.get("tomato_rewrite", {}) or {}
    accumulation = window.get("accumulation", {}) or {}
    weather_signal = int(weather.get("signal_count", 0)) >= 2
    control_signal = (
        float(action.get("vent_drying_pressure", 0.0)) >= 12.0
        or int(action.get("large_action_delta_count", 0)) >= 5
        or int(action.get("action_reversal_count", 0)) >= 5
        or int(action.get("high_vent_steps", 0)) >= max(6, int(window.get("row_count", 0)) // 4)
    )
    tomato_signal = int(tomato.get("rewrite_steps", 0)) >= max(5, int(window.get("row_count", 0)) // 4)
    accumulation_signal = bool(accumulation.get("heat_dry_vpd_accumulation_present", False))
    if weather_signal and control_signal:
        category = "mixed_weather_control_accumulation"
    elif control_signal:
        category = "control_history_accumulation_dominant"
    elif weather_signal:
        category = "weather_forcing_dominant"
    elif tomato_signal:
        category = "tomato_safety_rewrite_coupled_instability"
    else:
        category = "insufficient_history_for_attribution"
    if tomato_signal and category == "weather_forcing_dominant":
        category = "tomato_safety_rewrite_coupled_instability"
    return category, {
        "weather_signal": bool(weather_signal),
        "control_signal": bool(control_signal),
        "tomato_signal": bool(tomato_signal),
        "heat_dry_vpd_accumulation_signal": bool(accumulation_signal),
    }


def _trace_report(path: str | Path, *, window_steps: Sequence[int]) -> dict[str, Any]:
    rows = _read_rows(path)
    identity = _trace_identity(path)
    failure_row = _runtime_failure_row(rows)
    failure_step = _step(failure_row, len(rows) - 1 if rows else -1)
    windows: list[dict[str, Any]] = []
    for size in window_steps:
        selected = _window(rows, center_step=failure_step, window_steps=int(size)) if rows else []
        history_status = _history_field_status(selected)
        window_payload = {
            "window_steps_requested": int(size),
            "row_count": int(len(selected)),
            "window_start_step": _step(selected[0], 0) if selected else None,
            "window_end_step": _step(selected[-1], 0) if selected else None,
            "short_window": bool(len(selected) < int(size)),
            "history_field_status": history_status,
            "accumulation": _accumulation_metrics(selected),
            "weather_forcing": _weather_forcing_metrics(selected),
            "action_history": _action_history_metrics(selected),
            "tomato_rewrite": _tomato_rewrite_metrics(selected),
        }
        category, signals = _window_attribution(window_payload)
        window_payload["window_attribution"] = category
        window_payload["signal_summary"] = signals
        windows.append(window_payload)
    category_counts = Counter(str(window.get("window_attribution")) for window in windows)
    if category_counts.get("insufficient_history_for_attribution", 0) == len(windows):
        dominant = "insufficient_history_for_attribution"
    elif category_counts.get("mixed_weather_control_accumulation", 0):
        dominant = "mixed_weather_control_accumulation"
    else:
        dominant = category_counts.most_common(1)[0][0] if category_counts else "insufficient_history_for_attribution"
    return {
        **identity,
        "path": _rel(path),
        "trace_exists": bool(rows),
        "row_count": int(len(rows)),
        "short_trajectory_before_max_steps": bool(rows and len(rows) < int(identity.get("max_steps", 720))),
        "failure_detected": bool(failure_row is not None),
        "failure_step": int(failure_step),
        "window_reports": windows,
        "dominant_accumulation_attribution": dominant,
        "window_attribution_counts": dict(sorted(category_counts.items())),
    }


def _next_action(dominant: str, *, insufficient_count: int, scenario_count: int) -> str:
    if scenario_count <= 0:
        return "stateful_heat_dry_trace_acquisition_plan"
    if insufficient_count == scenario_count:
        return "stateful_history_instrumentation_plan"
    if dominant == "weather_forcing_dominant":
        return "solver_sensitive_regime_guard_design_plan"
    if dominant == "control_history_accumulation_dominant":
        return "stateful_runtime_stability_supervisor_shadow_instrumentation_plan"
    if dominant == "mixed_weather_control_accumulation":
        return "stateful_runtime_stability_supervisor_shadow_instrumentation_plan"
    if dominant == "tomato_safety_rewrite_coupled_instability":
        return "tomato_safety_transition_supervisor_shadow_design_plan"
    return "stateful_history_instrumentation_plan"


def build_accumulation_attribution(
    *,
    trace_dir: str | Path = V75_TRACE_DIR,
    v77_snapshot_json: str | Path = V77_SNAPSHOT_JSON,
    v77_attribution_json: str | Path = V77_ATTRIBUTION_JSON,
    window_steps: Sequence[int] = WINDOW_STEPS,
) -> dict[str, Any]:
    reports = [_trace_report(path, window_steps=window_steps) for path in _discover_trace_paths(trace_dir)]
    scenario_counts = Counter(str(report.get("dominant_accumulation_attribution")) for report in reports)
    insufficient_count = int(scenario_counts.get("insufficient_history_for_attribution", 0))
    if scenario_counts.get("mixed_weather_control_accumulation", 0):
        dominant = "mixed_weather_control_accumulation"
    else:
        dominant = scenario_counts.most_common(1)[0][0] if scenario_counts else "insufficient_history_for_attribution"
    v77_snapshot = _load_json(v77_snapshot_json)
    v77_attribution = _load_json(v77_attribution_json)
    payload = {
        "artifact": "qwen37plus_stateful_heat_dry_accumulation_attribution_20260604_v78",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "scope": "offline_stateful_heat_dry_vpd_accumulation_attribution",
        "trace_dir": _rel(trace_dir),
        "window_steps": [int(item) for item in window_steps],
        "source_v77_snapshot_artifact": v77_snapshot.get("artifact", ""),
        "source_v77_dominant_primary_attribution": v77_attribution.get("dominant_primary_attribution", ""),
        "accumulation_categories": list(ACCUMULATION_CATEGORIES),
        "trace_count": int(len(reports)),
        "failure_trace_count": int(sum(bool(report.get("failure_detected")) for report in reports)),
        "short_trajectory_count": int(sum(bool(report.get("short_trajectory_before_max_steps")) for report in reports)),
        "scenario_attribution_counts": dict(sorted(scenario_counts.items())),
        "dominant_accumulation_attribution": dominant,
        "scenario_reports": reports,
        "next_action": _next_action(dominant, insufficient_count=insufficient_count, scenario_count=len(reports)),
    }
    return _with_boundaries(payload)


def build_supervisor_design(attribution: Mapping[str, Any]) -> dict[str, Any]:
    design = {
        "artifact": "stateful_runtime_stability_supervisor_design_20260604_v78",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "scope": "design_only_stateful_runtime_stability_supervisor",
        "current_hypothesis": "Step-local controller response is not enough for accumulated hot/dry/high-VPD risk.",
        "observed_gap": (
            "v77 identified hot-dry high-VPD solver-sensitive failures; v78 attributes whether that state is "
            "accumulated through weather forcing, control history, Tomato Safety rewrites, or a mixture."
        ),
        "source_attribution_artifact": attribution.get("artifact", ""),
        "source_dominant_accumulation_attribution": attribution.get("dominant_accumulation_attribution", ""),
        "supervisor_contract": {
            "name": "stateful_runtime_stability_supervisor",
            "runtime_supervisor_enabled": False,
            "mode": "shadow_design_only",
            "inputs": [
                "recent_state_history",
                "recent_action_history",
                "recent_weather_history",
                "tomato_safety_rewrite_history",
                "runtime_failure_snapshot_reference",
            ],
            "shadow_outputs": [
                "heat_debt",
                "dryness_debt",
                "vpd_debt",
                "weather_forcing_signal",
                "control_history_accumulation_signal",
                "tomato_rewrite_coupling_signal",
                "risk_state",
                "recommended_shadow_plan",
            ],
            "risk_states": ["watch", "warning", "solver_sensitive", "abort_for_diagnosis"],
            "future_design_options": [
                "hot_dry_prevention_mode_shadow_gate",
                "vpd_ramp_guard",
                "screen_vent_shade_action_rate_envelope",
                "tomato_safety_rewrite_rate_monitor",
                "solver_sensitive_regime_guard",
            ],
            "final_action_changes_authorized": False,
            "future_behavior_changes_require_later_opt_in_stage": True,
        },
        "next_action": attribution.get("next_action", "stateful_history_instrumentation_plan"),
    }
    return _with_boundaries(design)


def build_readiness(attribution: Mapping[str, Any], design: Mapping[str, Any]) -> dict[str, Any]:
    dominant = str(attribution.get("dominant_accumulation_attribution", "insufficient_history_for_attribution"))
    trace_count = int(attribution.get("trace_count", 0) or 0)
    if trace_count <= 0:
        next_action = "stateful_heat_dry_trace_acquisition_plan"
        ready = False
    elif dominant == "insufficient_history_for_attribution":
        next_action = "stateful_history_instrumentation_plan"
        ready = False
    else:
        next_action = str(design.get("next_action") or "stateful_runtime_stability_supervisor_shadow_instrumentation_plan")
        ready = True
    payload = {
        "artifact": "stateful_heat_dry_accumulation_readiness_20260604_v78",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "trace_count": trace_count,
        "failure_trace_count": int(attribution.get("failure_trace_count", 0) or 0),
        "dominant_accumulation_attribution": dominant,
        "history_attribution_complete": bool(ready),
        "runtime_supervisor_contract_defined": bool("supervisor_contract" in design),
        "runtime_supervisor_enabled": False,
        "recommended_followup": next_action,
        "next_action": next_action,
    }
    return _with_boundaries(payload)


def build_attribution_report(attribution: Mapping[str, Any]) -> str:
    lines = [
        "# qwen3.7-plus v78 Stateful Heat-Dry Accumulation Attribution",
        "",
        f"- trace_count={int(attribution.get('trace_count', 0) or 0)}",
        f"- failure_trace_count={int(attribution.get('failure_trace_count', 0) or 0)}",
        f"- dominant_accumulation_attribution={attribution.get('dominant_accumulation_attribution', '')}",
        f"- next_action={attribution.get('next_action', '')}",
        f"- final_action_changed={bool(attribution.get('final_action_changed', False))}",
        f"- performance_claim_allowed={bool(attribution.get('performance_claim_allowed', False))}",
        f"- promotion_evidence={bool(attribution.get('promotion_evidence', False))}",
        "",
        "## Scenario Attribution",
    ]
    for report in attribution.get("scenario_reports", []) or []:
        lines.append(
            f"- {report.get('trace_id')}: {report.get('dominant_accumulation_attribution')} "
            f"at step {report.get('failure_step')}"
        )
    return "\n".join(lines) + "\n"


def build_design_report(design: Mapping[str, Any]) -> str:
    contract = design.get("supervisor_contract", {}) or {}
    lines = [
        "# v78 Stateful Runtime Stability Supervisor Design",
        "",
        f"- source_dominant_accumulation_attribution={design.get('source_dominant_accumulation_attribution', '')}",
        f"- next_action={design.get('next_action', '')}",
        f"- runtime_supervisor_enabled={bool(design.get('runtime_supervisor_enabled', False))}",
        f"- runtime_policy_enabled={bool(design.get('runtime_policy_enabled', False))}",
        f"- solver_fallback_executed={bool(design.get('solver_fallback_executed', False))}",
        "",
        "Future behavior changes require a later opt-in stage.",
        "",
        f"- contract_name={contract.get('name', '')}",
        f"- risk_states={','.join(contract.get('risk_states', []) or [])}",
    ]
    return "\n".join(lines) + "\n"


def build_readiness_report(readiness: Mapping[str, Any]) -> str:
    lines = [
        "# v78 Stateful Heat-Dry Accumulation Readiness",
        "",
        f"- trace_count={int(readiness.get('trace_count', 0) or 0)}",
        f"- failure_trace_count={int(readiness.get('failure_trace_count', 0) or 0)}",
        f"- dominant_accumulation_attribution={readiness.get('dominant_accumulation_attribution', '')}",
        f"- history_attribution_complete={bool(readiness.get('history_attribution_complete', False))}",
        f"- runtime_supervisor_contract_defined={bool(readiness.get('runtime_supervisor_contract_defined', False))}",
        f"- runtime_supervisor_enabled={bool(readiness.get('runtime_supervisor_enabled', False))}",
        f"- performance_claim_allowed={bool(readiness.get('performance_claim_allowed', False))}",
        f"- promotion_evidence={bool(readiness.get('promotion_evidence', False))}",
        f"- next_action={readiness.get('next_action', '')}",
    ]
    return "\n".join(lines) + "\n"


def write_all(
    *,
    trace_dir: str | Path = V75_TRACE_DIR,
    v77_snapshot_json: str | Path = V77_SNAPSHOT_JSON,
    v77_attribution_json: str | Path = V77_ATTRIBUTION_JSON,
    window_steps: Sequence[int] = WINDOW_STEPS,
    attribution_json: str | Path = ATTRIBUTION_JSON,
    attribution_md: str | Path = ATTRIBUTION_MD,
    design_json: str | Path = DESIGN_JSON,
    design_md: str | Path = DESIGN_MD,
    readiness_json: str | Path = READINESS_JSON,
    readiness_md: str | Path = READINESS_MD,
) -> dict[str, str]:
    attribution = build_accumulation_attribution(
        trace_dir=trace_dir,
        v77_snapshot_json=v77_snapshot_json,
        v77_attribution_json=v77_attribution_json,
        window_steps=window_steps,
    )
    design = build_supervisor_design(attribution)
    readiness = build_readiness(attribution, design)
    outputs = {
        "attribution_json": Path(attribution_json),
        "attribution_md": Path(attribution_md),
        "design_json": Path(design_json),
        "design_md": Path(design_md),
        "readiness_json": Path(readiness_json),
        "readiness_md": Path(readiness_md),
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs["attribution_json"].write_text(
        json.dumps(attribution, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    outputs["attribution_md"].write_text(build_attribution_report(attribution), encoding="utf-8")
    outputs["design_json"].write_text(
        json.dumps(design, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    outputs["design_md"].write_text(build_design_report(design), encoding="utf-8")
    outputs["readiness_json"].write_text(
        json.dumps(readiness, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    outputs["readiness_md"].write_text(build_readiness_report(readiness), encoding="utf-8")
    return {key: str(path) for key, path in outputs.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default=str(V75_TRACE_DIR))
    parser.add_argument("--v77-snapshot-json", default=str(V77_SNAPSHOT_JSON))
    parser.add_argument("--v77-attribution-json", default=str(V77_ATTRIBUTION_JSON))
    parser.add_argument("--window-step", type=int, action="append", default=[])
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    windows = tuple(args.window_step or WINDOW_STEPS)
    if args.no_write:
        attribution = build_accumulation_attribution(
            trace_dir=args.trace_dir,
            v77_snapshot_json=args.v77_snapshot_json,
            v77_attribution_json=args.v77_attribution_json,
            window_steps=windows,
        )
        design = build_supervisor_design(attribution)
        readiness = build_readiness(attribution, design)
        print(json.dumps({"attribution": attribution, "design": design, "readiness": readiness}, indent=2, ensure_ascii=False))
        return 0
    print(
        json.dumps(
            write_all(
                trace_dir=args.trace_dir,
                v77_snapshot_json=args.v77_snapshot_json,
                v77_attribution_json=args.v77_attribution_json,
                window_steps=windows,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
