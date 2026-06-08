"""Build v77 runtime stability diagnosis and architecture design artifacts.

This stage is offline-only. It reads existing short traces and prior CVODES
attribution artifacts, then produces failure snapshots, root-cause attribution,
and a design contract for a future runtime stability layer. It does not call an
online LLM, run rollout or replay, execute solver fallback, change final
actions, or make performance/safety claims.
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

from gl_gym.experiments import cvodes_failure_causal_attribution_v44 as v44  # noqa: E402
from gl_gym.experiments import qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_execution_v76 as v76  # noqa: E402
from gl_gym.experiments import qwen37plus_structured_bridge_runtime_failure_attribution_v65 as v65  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import AUDIT_DIR  # noqa: E402


MODEL_NAME = "qwen3.7-plus"
ARTIFACT_DATE = "20260604"
VERSION = "v77"
DEFAULT_WINDOW_STEPS = 30
DEFAULT_MAX_STEPS = 720

V75_TRACE_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "v75_profile_action_envelope_opt_in_shadow_traces"
    / "traces"
)
V76_EXECUTION_JSON = v76.EXECUTION_RECORD_JSON
V76_READINESS_JSON = v76.READINESS_JSON
V44_CAUSAL_JSON = v44.CAUSAL_JSON
V65_ATTRIBUTION_JSON = v65.ATTRIBUTION_JSON
V65_REPAIR_JSON = v65.REPAIR_JSON

SNAPSHOT_JSON = AUDIT_DIR / "qwen37plus_runtime_failure_snapshot_catalog_20260604_v77.json"
SNAPSHOT_MD = AUDIT_DIR / "qwen37plus_runtime_failure_snapshot_catalog_20260604_v77.md"
ATTRIBUTION_JSON = AUDIT_DIR / "qwen37plus_runtime_failure_root_cause_attribution_20260604_v77.json"
ATTRIBUTION_MD = AUDIT_DIR / "qwen37plus_runtime_failure_root_cause_attribution_20260604_v77.md"
DESIGN_JSON = AUDIT_DIR / "runtime_stability_architecture_design_20260604_v77.json"
DESIGN_MD = AUDIT_DIR / "runtime_stability_architecture_design_20260604_v77.md"

STATE_FIELDS = (
    "temp_air",
    "rh_air",
    "vpd_air",
    "co2_air",
    "pipe_temp",
    "canopy_dew_margin",
    "dew_margin_air",
)
ACTION_FIELDS = (
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
)
EXOGENOUS_FIELDS = (
    "temp_out",
    "rh_out",
    "glob_rad",
    "radiation",
    "forecast_temp_out_delta_1h",
    "forecast_rh_out_mean_1h",
)
TOMATO_CHANNELS = ("heat", "screen", "shade", "vent")
ENVELOPE_INVARIANCE_FIELDS = (
    "profile_action_envelope_shadow_enabled",
    "profile_action_envelope_shadow_candidate_count",
    "profile_action_envelope_shadow_eligible_candidate_count",
    "profile_action_envelope_shadow_best_name",
    "profile_action_envelope_shadow_final_action_changed",
    "profile_action_candidate_shadow_final_action_changed",
)
ROOT_CAUSE_CATEGORIES = (
    "cvodes_or_casadi_failure",
    "state_domain_or_nan_risk",
    "hot_dry_high_vpd_solver_sensitive_regime",
    "action_transition_or_rate_sensitive",
    "tomato_safety_rewrite_transition_sensitive",
    "weather_or_exogenous_boundary_sensitive",
    "insufficient_reproducer_state",
    "unknown_runtime_numerical_failure",
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
        if math.isnan(result) or math.isinf(result):
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
        "max_steps": int(match.group("max_steps") or DEFAULT_MAX_STEPS) if match else DEFAULT_MAX_STEPS,
    }


def _error_text(row: Mapping[str, Any]) -> str:
    return " ".join(
        str(row.get(key, "") or "")
        for key in ("runtime_error", "runtime_error_type", "error", "replan_reason")
    ).strip()


def classify_runtime_error(text: str) -> str:
    lowered = str(text or "").lower()
    if not lowered:
        return ""
    if "cvodes" in lowered or "cvodesinterface" in lowered or "cv_conv_failure" in lowered or "casadi" in lowered:
        return "cvodes_or_casadi_failure"
    if "invalid argument" in lowered or "[errno 22]" in lowered:
        return "simulator_io_invalid_argument"
    return "runtime_error_other"


def _runtime_failure_row(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for row in rows:
        if str(row.get("runtime_error", "") or "").strip():
            return row
        if str(row.get("runtime_error_type", "") or "").strip():
            return row
        if "runtime" in str(row.get("replan_reason", "") or "").lower():
            return row
    return None


def _step(row: Mapping[str, Any] | None, default: int = -1) -> int:
    if row is None:
        return int(default)
    return int(_num(row.get("step"), default))


def _window(rows: Sequence[Mapping[str, Any]], *, center_step: int | None, window_steps: int) -> list[Mapping[str, Any]]:
    if not rows:
        return []
    if center_step is None:
        center_step = _step(rows[-1], len(rows) - 1)
    start = max(0, int(center_step) - int(window_steps) + 1)
    return [row for row in rows if start <= _step(row) <= int(center_step)]


def _field_values(row: Mapping[str, Any], fields: Sequence[str]) -> dict[str, float | str | None]:
    result: dict[str, float | str | None] = {}
    for field in fields:
        if field not in row:
            result[field] = None
            continue
        value = row.get(field)
        number = _maybe_num(value)
        result[field] = number if number is not None else (str(value) if value not in (None, "") else None)
    return result


def _field_stats(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[str, dict[str, float | None]]:
    stats: dict[str, dict[str, float | None]] = {}
    for field in fields:
        values = [_maybe_num(row.get(field)) for row in rows if row.get(field) not in (None, "")]
        clean = [value for value in values if value is not None]
        stats[field] = {
            "start": clean[0] if clean else None,
            "end": clean[-1] if clean else None,
            "min": min(clean) if clean else None,
            "max": max(clean) if clean else None,
            "delta": clean[-1] - clean[0] if len(clean) >= 2 else 0.0,
        }
    return stats


def _required_field_status(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = list(STATE_FIELDS + ACTION_FIELDS)
    columns = set().union(*(row.keys() for row in rows)) if rows else set()
    missing_columns = [field for field in fields if field not in columns]
    missing_values = [
        field
        for field in fields
        if field in columns and not any(row.get(field) not in (None, "") for row in rows)
    ]
    return {
        "required_fields": fields,
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "complete": not missing_columns and not missing_values,
    }


def _state_domain_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    nonfinite_counts: Counter[str] = Counter()
    illegal_examples: list[dict[str, Any]] = []
    illegal_count = 0
    for row in rows:
        step = _step(row)
        for field in STATE_FIELDS:
            raw = row.get(field)
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except Exception:
                nonfinite_counts[field] += 1
                continue
            if not math.isfinite(value):
                nonfinite_counts[field] += 1
                continue
            illegal = False
            if field == "rh_air" and not 0.0 <= value <= 100.0:
                illegal = True
            elif field == "co2_air" and not 0.0 < value <= 5000.0:
                illegal = True
            elif field == "temp_air" and not -20.0 <= value <= 60.0:
                illegal = True
            elif field == "pipe_temp" and not -20.0 <= value <= 90.0:
                illegal = True
            elif field == "vpd_air" and not 0.0 <= value <= 10.0:
                illegal = True
            elif field in {"canopy_dew_margin", "dew_margin_air"} and not -30.0 <= value <= 60.0:
                illegal = True
            if illegal:
                illegal_count += 1
                if len(illegal_examples) < 10:
                    illegal_examples.append({"step": step, "field": field, "value": value})
    return {
        "nonfinite_count_by_field": dict(sorted(nonfinite_counts.items())),
        "nonfinite_total": int(sum(nonfinite_counts.values())),
        "illegal_range_count": int(illegal_count),
        "illegal_examples": illegal_examples,
        "state_domain_or_nan_risk": bool(sum(nonfinite_counts.values()) or illegal_count),
    }


def _action_transition_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    previous: dict[str, float] = {}
    previous_signs: dict[str, int] = {}
    max_abs_delta: dict[str, float] = {field: 0.0 for field in ACTION_FIELDS}
    large_delta_counts: Counter[str] = Counter()
    reversal_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for row in rows:
        step = _step(row)
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
                    if len(examples) < 12:
                        examples.append({"step": step, "field": field, "delta": delta})
                sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
                if sign and previous_signs.get(field, 0) and sign != previous_signs[field]:
                    reversal_counts[field] += 1
                if sign:
                    previous_signs[field] = sign
            previous[field] = value
    large_total = int(sum(large_delta_counts.values()))
    reversal_total = int(sum(reversal_counts.values()))
    return {
        "max_abs_delta_by_field": {field: float(value) for field, value in sorted(max_abs_delta.items())},
        "large_delta_count_by_field": dict(sorted(large_delta_counts.items())),
        "reversal_count_by_field": dict(sorted(reversal_counts.items())),
        "large_action_delta_count": large_total,
        "action_reversal_count": reversal_total,
        "examples": examples,
        "action_transition_or_rate_sensitive": bool(large_total or reversal_total),
    }


def _tomato_safety_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewrite_steps = 0
    rewrite_fields: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for row in rows:
        changed_fields: list[str] = []
        for channel in TOMATO_CHANNELS:
            before = _maybe_num(row.get(f"tomato_safety_v2_{channel}_before"))
            after = _maybe_num(row.get(f"tomato_safety_v2_{channel}_after"))
            if before is not None and after is not None and abs(after - before) > 1e-6:
                changed_fields.append(channel)
                rewrite_fields[channel] += 1
        reasons_text = str(
            row.get("tomato_safety_v2_reasons", "")
            or row.get("final_action_risk_reason_v2", "")
            or row.get("final_action_risk_reason", "")
            or ""
        )
        for reason in re.split(r"[,;|]", reasons_text):
            reason = reason.strip()
            if reason:
                reason_counts[reason] += 1
        if changed_fields or _truthy(row.get("tomato_safety_v2_applied")):
            rewrite_steps += 1
            if len(examples) < 10:
                examples.append({"step": _step(row), "changed_fields": changed_fields, "reasons": reasons_text})
    return {
        "rewrite_steps": int(rewrite_steps),
        "rewrite_field_counts": dict(sorted(rewrite_fields.items())),
        "rewrite_reason_counts": dict(sorted(reason_counts.items())),
        "examples": examples,
        "tomato_safety_rewrite_transition_sensitive": bool(rewrite_steps),
    }


def _shadow_invariance_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    envelope_changed = sum(_truthy(row.get("profile_action_envelope_shadow_final_action_changed")) for row in rows)
    candidate_changed = sum(_truthy(row.get("profile_action_candidate_shadow_final_action_changed")) for row in rows)
    return {
        "profile_action_envelope_shadow_enabled_steps": int(
            sum(_truthy(row.get("profile_action_envelope_shadow_enabled")) for row in rows)
        ),
        "profile_action_envelope_candidate_steps": int(
            sum(_num(row.get("profile_action_envelope_shadow_candidate_count"), 0) > 0 for row in rows)
        ),
        "profile_action_envelope_shadow_final_action_changed_steps": int(envelope_changed),
        "profile_action_candidate_shadow_final_action_changed_steps": int(candidate_changed),
        "envelope_action_invariance_preserved": bool(envelope_changed == 0),
        "candidate_action_invariance_preserved": bool(candidate_changed == 0),
    }


def _weather_boundary_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stats = _field_stats(rows, EXOGENOUS_FIELDS)
    sensitive_examples: list[dict[str, Any]] = []
    for field, item in stats.items():
        delta = item.get("delta")
        if delta is None:
            continue
        threshold = 300.0 if field in {"glob_rad", "radiation"} else 20.0 if "rh" in field else 5.0
        if abs(float(delta)) > threshold:
            sensitive_examples.append({"field": field, "delta": float(delta), "threshold": threshold})
    return {
        "stats": stats,
        "weather_or_exogenous_boundary_sensitive": bool(sensitive_examples),
        "examples": sensitive_examples,
    }


def _window_rows_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for row in rows:
        payload.append(
            {
                "step": _step(row),
                "state": _field_values(row, STATE_FIELDS),
                "action": _field_values(row, ACTION_FIELDS),
                "exogenous": _field_values(row, EXOGENOUS_FIELDS),
                "tomato_safety": {
                    "applied": _truthy(row.get("tomato_safety_v2_applied")),
                    "reasons": str(
                        row.get("tomato_safety_v2_reasons", "")
                        or row.get("final_action_risk_reason_v2", "")
                        or row.get("final_action_risk_reason", "")
                        or ""
                    ),
                    "before_after": {
                        channel: {
                            "before": _maybe_num(row.get(f"tomato_safety_v2_{channel}_before")),
                            "after": _maybe_num(row.get(f"tomato_safety_v2_{channel}_after")),
                        }
                        for channel in TOMATO_CHANNELS
                    },
                },
                "shadow_invariance": _field_values(row, ENVELOPE_INVARIANCE_FIELDS),
            }
        )
    return payload


def _snapshot_for_trace(path: str | Path, *, window_steps: int) -> dict[str, Any]:
    rows = _read_rows(path)
    identity = _trace_identity(path)
    failure_row = _runtime_failure_row(rows)
    failure_step = _step(failure_row, len(rows) - 1 if rows else -1)
    window_rows = _window(rows, center_step=failure_step if failure_row else None, window_steps=window_steps)
    error_text = _error_text(failure_row or {})
    max_steps = int(identity.get("max_steps") or DEFAULT_MAX_STEPS)
    return {
        **identity,
        "path": _rel(path),
        "trace_exists": bool(rows),
        "row_count": int(len(rows)),
        "max_steps": max_steps,
        "short_trajectory_before_max_steps": bool(rows and len(rows) < max_steps),
        "failure_detected": bool(failure_row is not None),
        "failure_step": int(failure_step),
        "runtime_error_kind": classify_runtime_error(error_text),
        "runtime_error_type": str((failure_row or {}).get("runtime_error_type", "") or ""),
        "runtime_error_message": error_text[:1000],
        "pre_failure_window": {
            "window_steps_requested": int(window_steps),
            "row_count": int(len(window_rows)),
            "window_start_step": _step(window_rows[0], 0) if window_rows else None,
            "window_end_step": _step(window_rows[-1], 0) if window_rows else None,
            "short_window": bool(len(window_rows) < window_steps),
            "rows": _window_rows_payload(window_rows),
        },
        "state_stats": _field_stats(window_rows, STATE_FIELDS),
        "action_stats": _field_stats(window_rows, ACTION_FIELDS),
        "required_reproducer_field_status": _required_field_status(window_rows),
        "state_domain_metrics": _state_domain_metrics(window_rows),
        "action_transition_metrics": _action_transition_metrics(window_rows),
        "tomato_safety_metrics": _tomato_safety_metrics(window_rows),
        "shadow_invariance_metrics": _shadow_invariance_metrics(window_rows),
        "weather_boundary_metrics": _weather_boundary_metrics(window_rows),
    }


def build_snapshot_catalog(
    *,
    trace_dir: str | Path = V75_TRACE_DIR,
    v76_execution_json: str | Path = V76_EXECUTION_JSON,
    v76_readiness_json: str | Path = V76_READINESS_JSON,
    v44_causal_json: str | Path = V44_CAUSAL_JSON,
    v65_attribution_json: str | Path = V65_ATTRIBUTION_JSON,
    window_steps: int = DEFAULT_WINDOW_STEPS,
) -> dict[str, Any]:
    paths = _discover_trace_paths(trace_dir)
    snapshots = [_snapshot_for_trace(path, window_steps=window_steps) for path in paths]
    failures = [item for item in snapshots if item.get("failure_detected")]
    v76_execution = _load_json(v76_execution_json)
    v76_readiness = _load_json(v76_readiness_json)
    prior_v44 = _load_json(v44_causal_json)
    prior_v65 = _load_json(v65_attribution_json)
    catalog = {
        "artifact": "qwen37plus_runtime_failure_snapshot_catalog_20260604_v77",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "scope": "offline_runtime_failure_snapshot_catalog_from_v75_v76_short_traces",
        "trace_dir": _rel(trace_dir),
        "window_steps": int(window_steps),
        "trace_count": int(len(snapshots)),
        "failure_trace_count": int(len(failures)),
        "short_trajectory_count": int(sum(bool(item.get("short_trajectory_before_max_steps")) for item in snapshots)),
        "source_v76_execution_artifact": v76_execution.get("artifact", ""),
        "source_v76_readiness_next_action": v76_readiness.get("next_action", ""),
        "source_v76_final_action_changed": bool(v76_readiness.get("final_action_changed", False)),
        "source_v76_profile_action_envelope_candidate_rows_total": int(
            v76_readiness.get("profile_action_envelope_candidate_rows_total", 0) or 0
        ),
        "source_v44_dominant_attribution": prior_v44.get("dominant_attribution", ""),
        "source_v65_dominant_attribution": prior_v65.get("dominant_attribution", ""),
        "snapshots": snapshots,
    }
    return _with_boundaries(catalog)


def _categories_for_snapshot(snapshot: Mapping[str, Any]) -> list[str]:
    categories: list[str] = []
    if snapshot.get("runtime_error_kind") == "cvodes_or_casadi_failure":
        categories.append("cvodes_or_casadi_failure")
    required = snapshot.get("required_reproducer_field_status", {}) or {}
    if not bool(required.get("complete", False)):
        categories.append("insufficient_reproducer_state")
    state_domain = snapshot.get("state_domain_metrics", {}) or {}
    if bool(state_domain.get("state_domain_or_nan_risk", False)):
        categories.append("state_domain_or_nan_risk")
    state_stats = snapshot.get("state_stats", {}) or {}
    temp_max = _num((state_stats.get("temp_air", {}) or {}).get("max"), 0.0)
    vpd_max = _num((state_stats.get("vpd_air", {}) or {}).get("max"), 0.0)
    rh_min = _num((state_stats.get("rh_air", {}) or {}).get("min"), 100.0)
    if (temp_max >= 32.0 and vpd_max >= 2.5) or (vpd_max >= 3.0 and rh_min <= 55.0):
        categories.append("hot_dry_high_vpd_solver_sensitive_regime")
    action = snapshot.get("action_transition_metrics", {}) or {}
    if bool(action.get("action_transition_or_rate_sensitive", False)):
        categories.append("action_transition_or_rate_sensitive")
    tomato = snapshot.get("tomato_safety_metrics", {}) or {}
    if bool(tomato.get("tomato_safety_rewrite_transition_sensitive", False)):
        categories.append("tomato_safety_rewrite_transition_sensitive")
    weather = snapshot.get("weather_boundary_metrics", {}) or {}
    if bool(weather.get("weather_or_exogenous_boundary_sensitive", False)):
        categories.append("weather_or_exogenous_boundary_sensitive")
    if not categories:
        categories.append("unknown_runtime_numerical_failure")
    return [item for item in ROOT_CAUSE_CATEGORIES if item in set(categories)]


def _primary_category(categories: Sequence[str]) -> str:
    priority = (
        "insufficient_reproducer_state",
        "state_domain_or_nan_risk",
        "hot_dry_high_vpd_solver_sensitive_regime",
        "action_transition_or_rate_sensitive",
        "tomato_safety_rewrite_transition_sensitive",
        "weather_or_exogenous_boundary_sensitive",
        "cvodes_or_casadi_failure",
        "unknown_runtime_numerical_failure",
    )
    category_set = set(categories)
    for item in priority:
        if item in category_set:
            return item
    return "unknown_runtime_numerical_failure"


def _attribution_next_action(primary_counts: Mapping[str, int], failure_count: int) -> tuple[str, bool]:
    if failure_count <= 0:
        return "runtime_failure_trace_acquisition_or_trace_selection_plan", False
    if primary_counts.get("insufficient_reproducer_state", 0):
        return "one_step_reproducer_instrumentation_plan", False
    dominant, count = max(primary_counts.items(), key=lambda item: (int(item[1]), item[0]))
    clear = bool(count / max(failure_count, 1) >= 0.67)
    if not clear:
        return "one_step_reproducer_instrumentation_plan", False
    if dominant == "hot_dry_high_vpd_solver_sensitive_regime":
        return "runtime_stability_architecture_design_review_plan", True
    if dominant == "action_transition_or_rate_sensitive":
        return "action_rate_stability_envelope_design_plan", True
    if dominant == "state_domain_or_nan_risk":
        return "pre_step_domain_guard_design_plan", True
    if dominant == "tomato_safety_rewrite_transition_sensitive":
        return "tomato_safety_runtime_transition_design_plan", True
    return "one_step_reproducer_instrumentation_plan", False


def build_root_cause_attribution(catalog: Mapping[str, Any]) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    failures = [item for item in catalog.get("snapshots", []) or [] if item.get("failure_detected")]
    for snapshot in failures:
        categories = _categories_for_snapshot(snapshot)
        primary = _primary_category(categories)
        category_counts.update(categories)
        primary_counts[primary] += 1
        shadow = snapshot.get("shadow_invariance_metrics", {}) or {}
        reports.append(
            {
                "trace_id": snapshot.get("trace_id"),
                "scenario_id": snapshot.get("scenario_id"),
                "failure_step": snapshot.get("failure_step"),
                "runtime_error_kind": snapshot.get("runtime_error_kind"),
                "categories": categories,
                "primary_attribution": primary,
                "required_reproducer_field_status": snapshot.get("required_reproducer_field_status", {}),
                "envelope_action_invariance_preserved": bool(
                    shadow.get("envelope_action_invariance_preserved", False)
                ),
                "profile_action_envelope_shadow_final_action_changed_steps": int(
                    shadow.get("profile_action_envelope_shadow_final_action_changed_steps", 0)
                ),
                "state_domain_metrics": snapshot.get("state_domain_metrics", {}),
                "action_transition_metrics": snapshot.get("action_transition_metrics", {}),
                "tomato_safety_metrics": snapshot.get("tomato_safety_metrics", {}),
                "weather_boundary_metrics": snapshot.get("weather_boundary_metrics", {}),
            }
        )
    next_action, clear = _attribution_next_action(primary_counts, len(failures))
    attribution = {
        "artifact": "qwen37plus_runtime_failure_root_cause_attribution_20260604_v77",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "scope": "offline_root_cause_attribution_for_v75_v76_short_traces",
        "source_snapshot_artifact": catalog.get("artifact", ""),
        "trace_count": int(catalog.get("trace_count", 0) or 0),
        "failure_trace_count": int(len(failures)),
        "root_cause_categories": list(ROOT_CAUSE_CATEGORIES),
        "category_counts": dict(sorted(category_counts.items())),
        "primary_attribution_counts": dict(sorted(primary_counts.items())),
        "dominant_primary_attribution": max(primary_counts.items(), key=lambda item: (int(item[1]), item[0]))[0]
        if primary_counts
        else "unknown_runtime_numerical_failure",
        "dominant_attribution_clear": bool(clear),
        "scenario_reports": reports,
        "next_action": next_action,
    }
    return _with_boundaries(attribution)


def build_architecture_design(
    attribution: Mapping[str, Any],
    *,
    v65_repair_json: str | Path = V65_REPAIR_JSON,
) -> dict[str, Any]:
    prior_repair = _load_json(v65_repair_json)
    design = {
        "artifact": "runtime_stability_architecture_design_20260604_v77",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "scope": "design_only_runtime_stability_architecture_contract",
        "current_hypothesis": "Profile/action-envelope repairs alone will not guarantee long-horizon numerical stability.",
        "observed_gap": (
            "v76 envelope provenance is complete and final-action invariant, but v75/v76 traces still end early "
            "with CVODES/CasADi runtime failures."
        ),
        "alternative_hypothesis": (
            "Runtime numerical stability needs an explicit architecture layer with failure snapshots, "
            "one-step reproducer requests, pre-step domain checks, and opt-in stability policy design."
        ),
        "source_attribution_artifact": attribution.get("artifact", ""),
        "source_dominant_primary_attribution": attribution.get("dominant_primary_attribution", ""),
        "source_v65_repair_next_action": prior_repair.get("next_action", ""),
        "contracts": {
            "failure_snapshot": {
                "purpose": "Durable pre-failure capture for every simulator/runtime failure.",
                "required_fields": [
                    "scenario_id",
                    "trace_id",
                    "failure_step",
                    "runtime_error_kind",
                    "runtime_error_message",
                    *STATE_FIELDS,
                    *ACTION_FIELDS,
                    "tomato_safety_before_after",
                    "tomato_safety_reasons",
                    "profile_action_envelope_shadow_final_action_changed",
                ],
                "status": "design_only_not_runtime_enabled",
            },
            "one_step_reproducer_request": {
                "purpose": "Define the minimum payload for a later deterministic failing-step reproducer.",
                "required_fields": [
                    "scenario_id",
                    "controller",
                    "failure_step",
                    "previous_state_vector_or_observation",
                    "attempted_action",
                    "weather_or_exogenous_inputs",
                    "tomato_safety_projection_or_rewrite",
                    "plan_cache_key_or_llm_provenance_reference",
                    "simulator_solver_config",
                ],
                "execution_authorized": False,
                "next_action_when_missing": "one_step_reproducer_instrumentation_plan",
            },
            "pre_step_domain_guard": {
                "purpose": "Offline/spec-only validation rules for illegal numerical domains before simulator step.",
                "validation_rules": [
                    "reject_or_diagnose_nonfinite_state_or_action",
                    "diagnose_rh_outside_0_100",
                    "diagnose_negative_or_extreme_vpd",
                    "diagnose_extreme_temperature_or_pipe_temperature",
                    "diagnose_missing_required_state_action_fields",
                    "diagnose_hot_dry_high_vpd_solver_sensitive_regime",
                ],
                "runtime_policy_enabled": False,
            },
            "runtime_stability_policy": {
                "purpose": "Design-only menu for a later opt-in stability stage.",
                "design_only_options": [
                    "substep_retry",
                    "tolerance_retry",
                    "safe_diagnostic_abort",
                    "action_rate_envelope",
                    "solver_sensitive_regime_specific_shadow_gate",
                ],
                "future_behavior_changes_require_later_opt_in_stage": True,
                "solver_fallback_executed": False,
                "final_action_changes_authorized": False,
            },
        },
        "success_condition_for_future_stage": (
            "A later opt-in stage can reproduce or deterministically classify failing steps without weakening "
            "Tomato Safety or changing default llm_rspc_v2 behavior."
        ),
        "stop_condition_for_future_stage": (
            "If required reproducer state is missing or any final-action-affecting behavior is proposed without "
            "authorization, stop at instrumentation repair."
        ),
        "next_action": attribution.get("next_action", "one_step_reproducer_instrumentation_plan"),
    }
    return _with_boundaries(design)


def build_snapshot_report(catalog: Mapping[str, Any]) -> str:
    lines = [
        "# qwen3.7-plus v77 Runtime Failure Snapshot Catalog",
        "",
        f"- trace_count={int(catalog.get('trace_count', 0) or 0)}",
        f"- failure_trace_count={int(catalog.get('failure_trace_count', 0) or 0)}",
        f"- short_trajectory_count={int(catalog.get('short_trajectory_count', 0) or 0)}",
        f"- online_llm_called={bool(catalog.get('online_llm_called', False))}",
        f"- performance_claim_allowed={bool(catalog.get('performance_claim_allowed', False))}",
        f"- promotion_evidence={bool(catalog.get('promotion_evidence', False))}",
        "",
        "## Failure Traces",
    ]
    for item in catalog.get("snapshots", []) or []:
        if not item.get("failure_detected"):
            continue
        lines.append(
            f"- {item.get('trace_id')}: step={item.get('failure_step')} rows={item.get('row_count')}/"
            f"{item.get('max_steps')} kind={item.get('runtime_error_kind')}"
        )
    return "\n".join(lines) + "\n"


def build_attribution_report(attribution: Mapping[str, Any]) -> str:
    lines = [
        "# qwen3.7-plus v77 Runtime Failure Root-Cause Attribution",
        "",
        f"- failure_trace_count={int(attribution.get('failure_trace_count', 0) or 0)}",
        f"- dominant_primary_attribution={attribution.get('dominant_primary_attribution', '')}",
        f"- dominant_attribution_clear={bool(attribution.get('dominant_attribution_clear', False))}",
        f"- next_action={attribution.get('next_action', '')}",
        f"- final_action_changed={bool(attribution.get('final_action_changed', False))}",
        f"- performance_claim_allowed={bool(attribution.get('performance_claim_allowed', False))}",
        f"- promotion_evidence={bool(attribution.get('promotion_evidence', False))}",
    ]
    return "\n".join(lines) + "\n"


def build_design_report(design: Mapping[str, Any]) -> str:
    contracts = design.get("contracts", {}) or {}
    lines = [
        "# v77 Runtime Stability Architecture Design",
        "",
        f"- source_dominant_primary_attribution={design.get('source_dominant_primary_attribution', '')}",
        f"- next_action={design.get('next_action', '')}",
        f"- runtime_policy_enabled={bool(design.get('runtime_policy_enabled', False))}",
        f"- solver_fallback_executed={bool(design.get('solver_fallback_executed', False))}",
        f"- performance_claim_allowed={bool(design.get('performance_claim_allowed', False))}",
        "",
        "Future behavior changes require a later opt-in stage.",
        "",
        "## Contracts",
    ]
    for name in ("failure_snapshot", "one_step_reproducer_request", "pre_step_domain_guard", "runtime_stability_policy"):
        lines.append(f"- {name}: {bool(name in contracts)}")
    return "\n".join(lines) + "\n"


def write_all(
    *,
    trace_dir: str | Path = V75_TRACE_DIR,
    v76_execution_json: str | Path = V76_EXECUTION_JSON,
    v76_readiness_json: str | Path = V76_READINESS_JSON,
    v44_causal_json: str | Path = V44_CAUSAL_JSON,
    v65_attribution_json: str | Path = V65_ATTRIBUTION_JSON,
    v65_repair_json: str | Path = V65_REPAIR_JSON,
    window_steps: int = DEFAULT_WINDOW_STEPS,
    snapshot_json: str | Path = SNAPSHOT_JSON,
    snapshot_md: str | Path = SNAPSHOT_MD,
    attribution_json: str | Path = ATTRIBUTION_JSON,
    attribution_md: str | Path = ATTRIBUTION_MD,
    design_json: str | Path = DESIGN_JSON,
    design_md: str | Path = DESIGN_MD,
) -> dict[str, str]:
    catalog = build_snapshot_catalog(
        trace_dir=trace_dir,
        v76_execution_json=v76_execution_json,
        v76_readiness_json=v76_readiness_json,
        v44_causal_json=v44_causal_json,
        v65_attribution_json=v65_attribution_json,
        window_steps=window_steps,
    )
    attribution = build_root_cause_attribution(catalog)
    design = build_architecture_design(attribution, v65_repair_json=v65_repair_json)
    outputs = {
        "snapshot_json": Path(snapshot_json),
        "snapshot_md": Path(snapshot_md),
        "attribution_json": Path(attribution_json),
        "attribution_md": Path(attribution_md),
        "design_json": Path(design_json),
        "design_md": Path(design_md),
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs["snapshot_json"].write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    outputs["snapshot_md"].write_text(build_snapshot_report(catalog), encoding="utf-8")
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
    return {name: str(path) for name, path in outputs.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default=str(V75_TRACE_DIR))
    parser.add_argument("--v76-execution-json", default=str(V76_EXECUTION_JSON))
    parser.add_argument("--v76-readiness-json", default=str(V76_READINESS_JSON))
    parser.add_argument("--v44-causal-json", default=str(V44_CAUSAL_JSON))
    parser.add_argument("--v65-attribution-json", default=str(V65_ATTRIBUTION_JSON))
    parser.add_argument("--v65-repair-json", default=str(V65_REPAIR_JSON))
    parser.add_argument("--window-steps", type=int, default=DEFAULT_WINDOW_STEPS)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    if args.no_write:
        catalog = build_snapshot_catalog(
            trace_dir=args.trace_dir,
            v76_execution_json=args.v76_execution_json,
            v76_readiness_json=args.v76_readiness_json,
            v44_causal_json=args.v44_causal_json,
            v65_attribution_json=args.v65_attribution_json,
            window_steps=args.window_steps,
        )
        attribution = build_root_cause_attribution(catalog)
        design = build_architecture_design(attribution, v65_repair_json=args.v65_repair_json)
        print(json.dumps({"catalog": catalog, "attribution": attribution, "design": design}, indent=2, ensure_ascii=False))
        return 0
    print(
        json.dumps(
            write_all(
                trace_dir=args.trace_dir,
                v76_execution_json=args.v76_execution_json,
                v76_readiness_json=args.v76_readiness_json,
                v44_causal_json=args.v44_causal_json,
                v65_attribution_json=args.v65_attribution_json,
                v65_repair_json=args.v65_repair_json,
                window_steps=args.window_steps,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
