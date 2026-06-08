"""v8173 risk-type-specific C-STCC scorer counterfactuals.

This stage is an existing-trace offline audit. It does not run runtime
acquisition, call online LLM, execute predictive rollout, add runtime
templates, or change final actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.cstcc.constraints import DEFAULT_MAX_DELTA, check_hard_constraints, soft_penalties
from gl_gym.cstcc.contracts import ACTION_FIELDS, clamp, normalize_action
from gl_gym.cstcc.dual_scorer import level0_sequence_score


DEFAULT_TRACE_CSV = Path(
    "gl_gym/result/diagnostics/cstcc_v8171_conservative_allocation_trace_20260606_y2020_d240_s42_n720.csv"
)
DEFAULT_SUMMARY_JSON = Path(
    "gl_gym/result/diagnostics/cstcc_v8171_conservative_allocation_trace_20260606_y2020_d240_s42_n720.json"
)
DEFAULT_JSONL_ROOT = Path("logs/cstcc_shadow/v8171_conservative_allocation_20260606_y2020_d240_s42_n720")
DEFAULT_V8172_JSON = Path(
    "gl_gym/result/diagnostics/cstcc_v8172_risk_aligned_bonus_calibration_20260606_y2020_d240_s42_n720.json"
)
DEFAULT_OUTPUT_PREFIX = Path(
    "gl_gym/result/diagnostics/cstcc_v8173_risk_type_specific_bonus_template_counterfactual_20260606_y2020_d240_s42_n720"
)

DRY_OR_HIGH_VPD_GRID = (0.00, 0.02, 0.04, 0.06)
DRY_VENT_GRID = (0.00, 0.02, 0.04, 0.06)
DEW_OR_HUMIDITY_GRID = (0.00, 0.04, 0.06, 0.08, 0.10, 0.12)
CANOPY_DEW_GRID = (0.00, 0.04, 0.06, 0.08, 0.10, 0.12)

RISK_FAMILIES = (
    "dry_vent",
    "canopy_dew_lt0",
    "canopy_dew_lt1",
    "dew_or_humidity",
    "dry_or_high_vpd",
    "none",
)
RISK_TYPE_KEYS = (
    "dew_risk",
    "humidity_or_dew_risk",
    "dry_risk",
    "high_vpd",
    "dry_or_high_vpd_risk",
    "dry_vent_risk",
    "canopy_dew_margin_lt1",
    "canopy_dew_margin_lt0",
)
PSEUDO_TEMPLATE_NAMES = (
    "conservative_stabilize_dry",
    "conservative_stabilize_high_vpd",
    "conservative_stabilize_dew",
    "conservative_stabilize_canopy_dew",
    "conservative_stabilize_dry_vent_guard",
)
BOUNDARY_FALSE_COUNTS = (
    "online_llm_called_steps",
    "predictive_rollout_executed_steps",
    "real_tomato_safety_projection_steps",
    "final_action_changed_steps",
)
COMPONENT_KEYS = (
    "base_final_score",
    "raw_smoothness",
    "raw_low_reversal",
    "raw_energy_proxy",
    "projected_smoothness",
    "projected_low_reversal",
    "projected_energy_proxy",
    "prior_confidence_bonus",
    "risk_static_hold_penalty",
    "rewrite_penalty",
    "uncertainty_penalty",
    "final_score",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except Exception:
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}
    return {}


def _load_csv(path: str | Path) -> list[dict[str, Any]]:
    input_path = _resolve(path)
    if not input_path.exists():
        return []
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json(path: str | Path) -> dict[str, Any]:
    input_path = _resolve(path)
    if not input_path.exists():
        return {}
    return json.loads(input_path.read_text(encoding="utf-8"))


def _load_jsonl_rows(root: str | Path) -> list[dict[str, Any]]:
    input_root = _resolve(root)
    rows: list[dict[str, Any]] = []
    if not input_root.exists():
        return rows
    for path in sorted(input_root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _compact_json(value) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            )


def _row_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    output: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        try:
            output[int(row.get("step", len(output)))] = row
        except Exception:
            continue
    return output


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _quantile_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean": float(mean(values)) if values else 0.0,
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
        "p95": _quantile(values, 0.95),
        "max": max(values) if values else 0.0,
    }


def _parse_candidate_source(candidate_id: str) -> tuple[str, str]:
    parts = str(candidate_id or "").split(":")
    return (parts[0] if parts else "", parts[1] if len(parts) > 1 else "")


def risk_flags_from_row(row: Mapping[str, Any]) -> dict[str, bool]:
    flags = _json_obj(row.get("risk_flags") or row.get("cstcc_shadow_risk_flags_json"))
    normalized = {str(key): _bool(value) for key, value in flags.items()} if flags else {}
    rh = _num(row.get("rh_air"), 70.0)
    vpd = _num(row.get("vpd_air"), _num(row.get("vpd_kpa")))
    canopy_dew = _num(row.get("canopy_dew_margin"), 999.0)
    dew_air = _num(row.get("dew_margin_air"), 999.0)
    normalized.setdefault("rh_ge_90", rh >= 90.0)
    normalized.setdefault("dew_risk", canopy_dew < 1.0 or dew_air < 1.0)
    normalized.setdefault("dry_risk", rh < 55.0)
    normalized.setdefault("high_vpd", vpd > 1.20)
    normalized["humidity_or_dew_risk"] = bool(
        normalized.get("humidity_or_dew_risk", False)
        or normalized.get("rh_ge_90", False)
        or normalized.get("dew_risk", False)
    )
    normalized["dry_or_high_vpd_risk"] = bool(
        normalized.get("dry_or_high_vpd_risk", False)
        or normalized.get("dry_risk", False)
        or normalized.get("high_vpd", False)
    )
    normalized["dry_vent_risk"] = bool(
        normalized.get("dry_vent_risk", False)
        or _bool(row.get("dry_vent_risk"))
        or (_num(row.get("u_ventilation"), 0.0) > 0.70 and normalized.get("dry_risk", False))
    )
    normalized["canopy_dew_margin_lt1"] = bool(canopy_dew < 1.0)
    normalized["canopy_dew_margin_lt0"] = bool(canopy_dew < 0.0)
    normalized["any_risk"] = bool(
        normalized.get("any_risk", False)
        or any(value for key, value in normalized.items() if key != "any_risk")
    )
    return normalized


def primary_risk_family(flags: Mapping[str, Any]) -> str:
    """Resolve mixed risk into one non-additive bonus family."""

    if _bool(flags.get("dry_vent_risk")):
        return "dry_vent"
    if _bool(flags.get("canopy_dew_margin_lt0")):
        return "canopy_dew_lt0"
    if _bool(flags.get("canopy_dew_margin_lt1")):
        return "canopy_dew_lt1"
    if _bool(flags.get("humidity_or_dew_risk")) or _bool(flags.get("dew_risk")):
        return "dew_or_humidity"
    if _bool(flags.get("dry_or_high_vpd_risk")) or _bool(flags.get("dry_risk")) or _bool(flags.get("high_vpd")):
        return "dry_or_high_vpd"
    return "none"


def active_risk_flags(flags: Mapping[str, Any]) -> list[str]:
    return [key for key in RISK_TYPE_KEYS if _bool(flags.get(key))]


def bonus_vectors() -> list[dict[str, float]]:
    vectors: list[dict[str, float]] = []
    for dry, dry_vent, dew, canopy in product(
        DRY_OR_HIGH_VPD_GRID,
        DRY_VENT_GRID,
        DEW_OR_HUMIDITY_GRID,
        CANOPY_DEW_GRID,
    ):
        vectors.append(
            {
                "dry_or_high_vpd_bonus": float(dry),
                "dry_vent_bonus": float(dry_vent),
                "dew_or_humidity_bonus": float(dew),
                "canopy_dew_bonus": float(canopy),
            }
        )
    return vectors


def vector_id(vector: Mapping[str, Any]) -> str:
    return (
        f"dry={_num(vector.get('dry_or_high_vpd_bonus')):.2f}|"
        f"dry_vent={_num(vector.get('dry_vent_bonus')):.2f}|"
        f"dew={_num(vector.get('dew_or_humidity_bonus')):.2f}|"
        f"canopy={_num(vector.get('canopy_dew_bonus')):.2f}"
    )


def bonus_for_family(vector: Mapping[str, Any], family: str) -> float:
    if family == "dry_vent":
        return _num(vector.get("dry_vent_bonus"))
    if family in {"canopy_dew_lt0", "canopy_dew_lt1"}:
        return _num(vector.get("canopy_dew_bonus"))
    if family == "dew_or_humidity":
        return _num(vector.get("dew_or_humidity_bonus"))
    if family == "dry_or_high_vpd":
        return _num(vector.get("dry_or_high_vpd_bonus"))
    return 0.0


def _candidate_info(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_candidates = row.get("candidate_violation_provenance")
    if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
        raw_candidates = _json_obj(row.get("cstcc_shadow_candidate_violation_provenance_json")).get("candidates", [])
    output: dict[str, dict[str, Any]] = {}
    for candidate in raw_candidates or []:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            continue
        source, template = _parse_candidate_source(candidate_id)
        output[candidate_id] = {
            "candidate_id": candidate_id,
            "source_prior": str(candidate.get("source_prior") or source),
            "template_name": str(candidate.get("template_name") or template),
            "feasible": bool(candidate.get("feasible", False)),
            "violation_reason_distribution": dict(candidate.get("violation_reason_distribution") or {}),
            "violation_field_distribution": dict(candidate.get("violation_field_distribution") or {}),
        }
    return output


def _score_components(row: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    raw = row.get("score_component_by_candidate")
    if not isinstance(raw, Mapping):
        raw = _json_obj(row.get("cstcc_shadow_score_component_by_candidate_json"))
    output: dict[str, dict[str, float]] = {}
    for candidate_id, components in (raw or {}).items():
        if not isinstance(components, Mapping):
            continue
        output[str(candidate_id)] = {
            key: _num(components.get(key), 0.0)
            for key in COMPONENT_KEYS
            if key in components or key == "final_score"
        }
    return output


def _current_action(row: Mapping[str, Any]) -> dict[str, float]:
    for key in (
        "current_runtime_final_action",
        "constraint_reference_action",
        "cstcc_shadow_current_runtime_final_action_json",
        "cstcc_shadow_constraint_reference_action_json",
    ):
        raw = row.get(key)
        action = _json_obj(raw)
        if action:
            return normalize_action(action)
    return normalize_action({field: row.get(field, 0.0) for field in ACTION_FIELDS})


def _eligible_conservative(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("feasible", False)
        and candidate.get("source_prior") == "conservative_prior"
        and candidate.get("template_name") == "conservative_stabilize"
    )


def _conservative_candidate_id(candidates: Mapping[str, Mapping[str, Any]]) -> str:
    for candidate_id, candidate in candidates.items():
        if _eligible_conservative(candidate):
            return candidate_id
    return ""


def _winner(
    *,
    components: Mapping[str, Mapping[str, float]],
    candidates: Mapping[str, Mapping[str, Any]],
    vector: Mapping[str, Any] | None = None,
    family: str = "none",
) -> tuple[str, float]:
    assigned_bonus = bonus_for_family(vector or {}, family)
    best_id = ""
    best_score = -1e18
    for candidate_id, component in components.items():
        candidate = candidates.get(candidate_id)
        if candidate is None:
            source, template = _parse_candidate_source(candidate_id)
            candidate = {"source_prior": source, "template_name": template, "feasible": False}
        if not candidate.get("feasible", False):
            continue
        score = _num(component.get("final_score"), -1e18)
        if _eligible_conservative(candidate) and family != "none":
            score += assigned_bonus
        if score > best_score:
            best_id = candidate_id
            best_score = score
    return best_id, best_score


def _source(candidate_id: str) -> str:
    return _parse_candidate_source(candidate_id)[0]


def _template(candidate_id: str) -> str:
    return _parse_candidate_source(candidate_id)[1]


def _component_delta(
    lhs: Mapping[str, float] | None,
    rhs: Mapping[str, float] | None,
) -> dict[str, float]:
    left = lhs or {}
    right = rhs or {}
    return {key: _num(left.get(key)) - _num(right.get(key)) for key in COMPONENT_KEYS}


def _boundary_summary(summary_json: Mapping[str, Any], jsonl_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trace = summary_json.get("trace_summary", {}) if isinstance(summary_json.get("trace_summary", {}), Mapping) else {}
    answers = summary_json.get("answers", {}) if isinstance(summary_json.get("answers", {}), Mapping) else {}
    return {
        "jsonl_row_count": len(jsonl_rows),
        "audit_success_rate": _num(trace.get("cstcc_shadow_audit_success_rate"), _num(answers.get("audit_success_rate"))),
        "final_action_invariant_rate": _num(
            trace.get("cstcc_shadow_final_action_invariant_rate"),
            _num(answers.get("final_action_invariant_rate")),
        ),
        "final_action_changed_steps": int(_num(trace.get("cstcc_shadow_final_action_changed_steps"))),
        "online_llm_called_steps": int(_num(trace.get("cstcc_shadow_online_llm_called_steps"))),
        "predictive_rollout_executed_steps": int(_num(trace.get("cstcc_shadow_predictive_rollout_executed_steps"))),
        "real_tomato_safety_projection_steps": int(_num(trace.get("cstcc_shadow_real_tomato_safety_projection_steps"))),
        "fallback_rate": _num(trace.get("cstcc_shadow_fallback_rate"), _num(answers.get("fallback_rate"))),
        "mean_infeasible_candidate_ratio": _num(trace.get("cstcc_shadow_mean_infeasible_candidate_ratio")),
        "previous_shift_all_max_delta_count": int(_num(trace.get("cstcc_shadow_previous_shift_all_max_delta_count"))),
        "selected_candidate_max_delta_count": int(_num(trace.get("cstcc_shadow_selected_candidate_violation_count"))),
        "json_row_size_kb_max": _num(answers.get("json_size_kb_max")),
    }


def _missing_schema_fields(jsonl_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not jsonl_rows:
        return ["jsonl_rows_missing"]
    first_success = next((row for row in jsonl_rows if row.get("audit_success", False)), jsonl_rows[0])
    required = (
        "candidate_violation_provenance",
        "score_component_by_candidate",
        "risk_flags",
        "shadow_selected_sequence_id",
        "current_runtime_final_action",
    )
    missing = [f"missing_field:{field}" for field in required if field not in first_success]
    if not _score_components(first_success):
        missing.append("missing_or_empty_field:score_component_by_candidate")
    if not _candidate_info(first_success):
        missing.append("missing_or_empty_field:candidate_violation_provenance")
    return missing


def _target_for_pseudo_template(template_name: str, last_action: Mapping[str, Any]) -> dict[str, float]:
    last = normalize_action(last_action)
    target = dict(last)
    if template_name == "conservative_stabilize_dry":
        target["u_heating"] = min(last["u_heating"], 0.25)
        target["u_co2"] = min(last["u_co2"], 0.20)
        target["u_ventilation"] = min(max(last["u_ventilation"], 0.20), 0.45)
        target["u_shading"] = min(max(last["u_shading"], 0.25), 0.70)
    elif template_name == "conservative_stabilize_high_vpd":
        target["u_heating"] = min(last["u_heating"], 0.25)
        target["u_co2"] = min(last["u_co2"], 0.20)
        target["u_ventilation"] = min(max(last["u_ventilation"], 0.25), 0.50)
        target["u_shading"] = min(max(last["u_shading"], 0.25), 0.75)
    elif template_name == "conservative_stabilize_dew":
        target["u_heating"] = min(max(last["u_heating"], 0.20), 0.45)
        target["u_co2"] = min(last["u_co2"], 0.20)
        target["u_screen"] = min(last["u_screen"], 0.55)
        target["u_ventilation"] = min(max(last["u_ventilation"], 0.45), 0.70)
        target["u_shading"] = min(last["u_shading"], 0.55)
    elif template_name == "conservative_stabilize_canopy_dew":
        target["u_heating"] = min(max(last["u_heating"], 0.25), 0.50)
        target["u_co2"] = min(last["u_co2"], 0.20)
        target["u_screen"] = min(last["u_screen"], 0.45)
        target["u_ventilation"] = min(max(last["u_ventilation"], 0.35), 0.65)
        target["u_shading"] = min(last["u_shading"], 0.50)
    elif template_name == "conservative_stabilize_dry_vent_guard":
        target["u_heating"] = min(last["u_heating"], 0.30)
        target["u_co2"] = min(last["u_co2"], 0.20)
        target["u_ventilation"] = min(max(last["u_ventilation"], 0.15), 0.45)
        target["u_lighting"] = min(last["u_lighting"], 0.40)
        target["u_shading"] = min(max(last["u_shading"], 0.20), 0.70)
    else:
        raise ValueError(f"unknown pseudo template: {template_name}")
    return normalize_action(target)


def _ramp_sequence(
    last_action: Mapping[str, Any],
    target_action: Mapping[str, Any],
    *,
    horizon: int = 12,
    step_delta: float = 0.06,
) -> list[dict[str, float]]:
    current = normalize_action(last_action)
    target = normalize_action(target_action)
    sequence: list[dict[str, float]] = []
    for _ in range(max(1, int(horizon))):
        next_action: dict[str, float] = {}
        for field in ACTION_FIELDS:
            current_value = current[field]
            target_value = target[field]
            if target_value > current_value:
                next_action[field] = clamp(min(target_value, current_value + step_delta))
            else:
                next_action[field] = clamp(max(target_value, current_value - step_delta))
        sequence.append(next_action)
        current = next_action
    return sequence


def generate_pseudo_template_sequence(
    template_name: str,
    *,
    current_runtime_final_action: Mapping[str, Any],
    horizon: int = 12,
) -> list[dict[str, float]]:
    target = _target_for_pseudo_template(template_name, current_runtime_final_action)
    return _ramp_sequence(current_runtime_final_action, target, horizon=horizon, step_delta=0.06)


def score_pseudo_template(
    template_name: str,
    *,
    current_runtime_final_action: Mapping[str, Any],
    conservative_prior_bonus: float = 0.0,
    horizon: int = 12,
) -> dict[str, Any]:
    sequence = generate_pseudo_template_sequence(
        template_name,
        current_runtime_final_action=current_runtime_final_action,
        horizon=horizon,
    )
    violations = check_hard_constraints(sequence, previous_action=current_runtime_final_action, max_delta=DEFAULT_MAX_DELTA)
    base_score, components = level0_sequence_score(sequence, soft_penalties(sequence))
    score_components = {
        "base_final_score": base_score,
        "raw_smoothness": components.get("smoothness", 0.0),
        "raw_low_reversal": components.get("low_reversal", 0.0),
        "raw_energy_proxy": components.get("energy_proxy", 0.0),
        "projected_smoothness": components.get("smoothness", 0.0),
        "projected_low_reversal": components.get("low_reversal", 0.0),
        "projected_energy_proxy": components.get("energy_proxy", 0.0),
        "prior_confidence_bonus": conservative_prior_bonus,
        "risk_static_hold_penalty": 0.0,
        "rewrite_penalty": 0.0,
        "uncertainty_penalty": 0.0,
        "final_score": base_score + conservative_prior_bonus,
    }
    return {
        "template_name": template_name,
        "feasible": not violations,
        "violation_count": len(violations),
        "violation_reason_distribution": _distribution(item.get("reason") for item in violations),
        "violation_field_distribution": _distribution(item.get("field") for item in violations),
        "score_components": score_components,
        "first_action_delta_from_reference": {
            field: sequence[0][field] - normalize_action(current_runtime_final_action)[field]
            for field in ACTION_FIELDS
        },
    }


def _component_quantiles(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in COMPONENT_KEYS:
        values = [_num((record.get("zero_minus_conservative") or {}).get(key)) for record in records]
        if values:
            rows.append({"component": key, **_quantile_summary(values)})
    return rows


def _build_step_records(
    *,
    trace_csv_rows: Sequence[Mapping[str, Any]],
    jsonl_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    csv_by_step = _row_by_step(trace_csv_rows)
    records: list[dict[str, Any]] = []
    schema_issues: list[str] = []
    for index, row in enumerate(row for row in jsonl_rows if row.get("audit_success", False)):
        step = int(_num(row.get("step"), index))
        merged = {**dict(csv_by_step.get(step, {})), **dict(row)}
        flags = risk_flags_from_row(merged)
        family = primary_risk_family(flags)
        candidates = _candidate_info(row)
        components = _score_components(row)
        runtime_id = str(row.get("shadow_selected_sequence_id") or merged.get("cstcc_shadow_selected_sequence_id") or "")
        zero_id, zero_score = _winner(components=components, candidates=candidates, family="none", vector={})
        conservative_id = _conservative_candidate_id(candidates)
        if not zero_id:
            schema_issues.append(f"zero_bonus_winner_missing:step={step}")
        if not runtime_id:
            schema_issues.append(f"runtime_selected_id_missing:step={step}")
        if flags.get("any_risk", False) and not conservative_id:
            schema_issues.append(f"conservative_candidate_missing:step={step}")
        conservative_components = components.get(conservative_id, {})
        zero_components = components.get(zero_id, {})
        records.append(
            {
                "step": step,
                "merged": merged,
                "risk_flags": flags,
                "active_risk_flags": active_risk_flags(flags),
                "primary_risk_family": family,
                "candidates": candidates,
                "components": components,
                "runtime_selected_id": runtime_id,
                "runtime_selected_source": _source(runtime_id),
                "runtime_selected_template": _template(runtime_id),
                "zero_bonus_winner_id": zero_id,
                "zero_bonus_winner_score": zero_score,
                "zero_bonus_winner_source": _source(zero_id),
                "zero_bonus_winner_template": _template(zero_id),
                "conservative_candidate_id": conservative_id,
                "conservative_score": _num(conservative_components.get("final_score"), -1e18),
                "conservative_prior_bonus": _num(conservative_components.get("prior_confidence_bonus")),
                "zero_minus_conservative": _component_delta(zero_components, conservative_components),
                "current_runtime_final_action": _current_action(merged),
                "hour_of_day": _num(merged.get("hour_of_day")),
                "active_regime": row.get("active_regime_after") or merged.get("cstcc_shadow_active_regime_after"),
                "temp_air": _num(merged.get("temp_air")),
                "rh_air": _num(merged.get("rh_air")),
                "vpd_air": _num(merged.get("vpd_air")),
                "canopy_dew_margin": _num(merged.get("canopy_dew_margin"), 999.0),
                "dew_margin_air": _num(merged.get("dew_margin_air"), 999.0),
            }
        )
    return records, sorted(set(schema_issues))


def evaluate_bonus_vectors(
    records: Sequence[Mapping[str, Any]],
    vectors: Sequence[Mapping[str, float]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    vectors = list(vectors or bonus_vectors())
    runtime_zero_mismatch_count = sum(
        1 for record in records if record.get("runtime_selected_id") != record.get("zero_bonus_winner_id")
    )
    risk_records = [record for record in records if (record.get("risk_flags") or {}).get("any_risk", False)]
    curve_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    switch_rows: list[dict[str, Any]] = []
    best_vector_row: dict[str, Any] = {}

    for vector in vectors:
        vid = vector_id(vector)
        row_count = len(records)
        risk_count = len(risk_records)
        conservative_selected = 0
        conservative_risk_selected = 0
        nonrisk_conservative_selected = 0
        switch_runtime = 0
        switch_zero = 0
        switch_to_conservative = 0
        switch_from_previous = 0
        switch_from_rule = 0
        switch_from_ppo = 0
        switch_nonconservative_to_nonconservative = 0
        by_family = {
            family: {"risk_row_count": 0, "conservative_selected_count": 0, "switch_to_conservative_count": 0}
            for family in RISK_FAMILIES
        }

        for record in records:
            family = str(record.get("primary_risk_family") or "none")
            winner_id, winner_score = _winner(
                components=record.get("components") or {},
                candidates=record.get("candidates") or {},
                vector=vector,
                family=family,
            )
            winner_source = _source(winner_id)
            zero_id = str(record.get("zero_bonus_winner_id") or "")
            zero_source = _source(zero_id)
            runtime_id = str(record.get("runtime_selected_id") or "")
            is_risk = bool((record.get("risk_flags") or {}).get("any_risk", False))
            is_conservative = winner_source == "conservative_prior" and _template(winner_id) == "conservative_stabilize"
            if is_conservative:
                conservative_selected += 1
                if is_risk:
                    conservative_risk_selected += 1
                else:
                    nonrisk_conservative_selected += 1
            if winner_id != runtime_id:
                switch_runtime += 1
            if winner_id != zero_id:
                switch_zero += 1
                if is_conservative and zero_source != "conservative_prior":
                    switch_to_conservative += 1
                    if zero_source == "previous_plan_prior":
                        switch_from_previous += 1
                    elif zero_source == "rule_prior":
                        switch_from_rule += 1
                    elif zero_source == "ppo_prior":
                        switch_from_ppo += 1
                elif winner_source != "conservative_prior" and zero_source != "conservative_prior":
                    switch_nonconservative_to_nonconservative += 1
                if len(switch_rows) < 5000:
                    zero_components = (record.get("components") or {}).get(zero_id, {})
                    winner_components = (record.get("components") or {}).get(winner_id, {})
                    conservative_components = (record.get("components") or {}).get(
                        record.get("conservative_candidate_id", ""),
                        {},
                    )
                    zero_minus_winner = _component_delta(zero_components, winner_components)
                    zero_minus_conservative = _component_delta(zero_components, conservative_components)
                    switch_rows.append(
                        {
                            "vector_id": vid,
                            **dict(vector),
                            "assigned_bonus": bonus_for_family(vector, family),
                            "step": record.get("step"),
                            "hour_of_day": record.get("hour_of_day"),
                            "active_regime": record.get("active_regime"),
                            "primary_risk_family": family,
                            "active_risk_flags": record.get("active_risk_flags"),
                            "runtime_selected_id": runtime_id,
                            "zero_bonus_winner_id": zero_id,
                            "counterfactual_winner_id": winner_id,
                            "counterfactual_winner_score": winner_score,
                            "switch_relative_to_runtime": winner_id != runtime_id,
                            "switch_relative_to_zero_bonus": winner_id != zero_id,
                            "switch_to_conservative": is_conservative and zero_source != "conservative_prior",
                            "zero_source_prior": zero_source,
                            "winner_source_prior": winner_source,
                            "energy_proxy_delta_on_switch": zero_minus_conservative.get("projected_energy_proxy", 0.0),
                            "smoothness_delta_on_switch": zero_minus_conservative.get("projected_smoothness", 0.0),
                            "prior_confidence_delta_on_switch": zero_minus_conservative.get("prior_confidence_bonus", 0.0),
                            "zero_minus_winner_components": zero_minus_winner,
                            "zero_minus_conservative_components": zero_minus_conservative,
                            "temp_air": record.get("temp_air"),
                            "rh_air": record.get("rh_air"),
                            "vpd_air": record.get("vpd_air"),
                            "canopy_dew_margin": record.get("canopy_dew_margin"),
                            "dew_margin_air": record.get("dew_margin_air"),
                        }
                    )
            family_item = by_family.setdefault(
                family,
                {"risk_row_count": 0, "conservative_selected_count": 0, "switch_to_conservative_count": 0},
            )
            if is_risk:
                family_item["risk_row_count"] += 1
                if is_conservative:
                    family_item["conservative_selected_count"] += 1
                if is_conservative and zero_source != "conservative_prior":
                    family_item["switch_to_conservative_count"] += 1

        curve = {
            "vector_id": vid,
            **dict(vector),
            "row_count": row_count,
            "risk_row_count": risk_count,
            "conservative_selected_count": conservative_selected,
            "conservative_risk_selected_count": conservative_risk_selected,
            "conservative_risk_selected_rate": conservative_risk_selected / max(risk_count, 1),
            "nonrisk_conservative_selected_count": nonrisk_conservative_selected,
            "switch_relative_to_runtime_count": switch_runtime,
            "switch_relative_to_zero_bonus_count": switch_zero,
            "switch_to_conservative_count": switch_to_conservative,
            "switch_from_previous_to_conservative_count": switch_from_previous,
            "switch_from_rule_to_conservative_count": switch_from_rule,
            "switch_from_ppo_to_conservative_count": switch_from_ppo,
            "switch_nonconservative_to_nonconservative_count": switch_nonconservative_to_nonconservative,
            "selected_id_vs_zero_bonus_winner_mismatch_count": runtime_zero_mismatch_count,
        }
        rates = []
        for family, item in by_family.items():
            family_count = int(item.get("risk_row_count", 0) or 0)
            selected_count = int(item.get("conservative_selected_count", 0) or 0)
            selected_rate = selected_count / max(family_count, 1)
            if family != "none" and family_count > 0:
                rates.append(selected_rate)
            family_rows.append(
                {
                    "vector_id": vid,
                    **dict(vector),
                    "primary_risk_family": family,
                    "risk_row_count": family_count,
                    "conservative_selected_count": selected_count,
                    "conservative_selected_rate": selected_rate,
                    "switch_to_conservative_count": int(item.get("switch_to_conservative_count", 0) or 0),
                }
            )
        spread = max(rates) - min(rates) if len(rates) >= 2 else 0.0
        curve["risk_type_switch_spread"] = spread
        curve["rule_suppression_detected"] = switch_from_rule > 0
        curve["previous_rule_ppo_suppression_detected"] = conservative_risk_selected / max(risk_count, 1) > 0.85
        curve_rows.append(curve)
        if _is_better_vector(curve, best_vector_row):
            best_vector_row = dict(curve)

    diagnostics = {
        "row_count": len(records),
        "risk_row_count": len(risk_records),
        "bonus_vector_count": len(vectors),
        "selected_id_vs_zero_bonus_winner_mismatch_count": runtime_zero_mismatch_count,
        "best_balanced_vector": best_vector_row,
        "minimum_risk_type_switch_spread": min(
            (float(row.get("risk_type_switch_spread", 0.0)) for row in curve_rows if row.get("conservative_risk_selected_count", 0)),
            default=0.0,
        ),
        "zero_minus_conservative_component_quantiles": _component_quantiles(records),
    }
    return curve_rows, family_rows, switch_rows, diagnostics


def _is_better_vector(candidate: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    if not current:
        return True
    candidate_rate = _num(candidate.get("conservative_risk_selected_rate"))
    current_rate = _num(current.get("conservative_risk_selected_rate"))
    candidate_spread = _num(candidate.get("risk_type_switch_spread"), 999.0)
    current_spread = _num(current.get("risk_type_switch_spread"), 999.0)
    candidate_viable = 0.25 <= candidate_rate <= 0.85 and int(candidate.get("nonrisk_conservative_selected_count", 0) or 0) == 0
    current_viable = 0.25 <= current_rate <= 0.85 and int(current.get("nonrisk_conservative_selected_count", 0) or 0) == 0
    if candidate_viable != current_viable:
        return candidate_viable
    return (candidate_spread, abs(candidate_rate - 0.55), -candidate_rate) < (
        current_spread,
        abs(current_rate - 0.55),
        -current_rate,
    )


def evaluate_template_variants(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if not (record.get("risk_flags") or {}).get("any_risk", False):
            continue
        baseline_score = _num(record.get("conservative_score"), -1e18)
        zero_score = _num(record.get("zero_bonus_winner_score"), -1e18)
        prior_bonus = _num(record.get("conservative_prior_bonus"))
        for template_name in PSEUDO_TEMPLATE_NAMES:
            scored = score_pseudo_template(
                template_name,
                current_runtime_final_action=record.get("current_runtime_final_action") or {},
                conservative_prior_bonus=prior_bonus,
            )
            final_score = _num((scored.get("score_components") or {}).get("final_score"), -1e18)
            rows.append(
                {
                    "step": record.get("step"),
                    "primary_risk_family": record.get("primary_risk_family"),
                    "active_risk_flags": record.get("active_risk_flags"),
                    "template_name": template_name,
                    "feasible": bool(scored.get("feasible", False)),
                    "violation_count": int(scored.get("violation_count", 0) or 0),
                    "score_delta_vs_conservative_stabilize": final_score - baseline_score,
                    "score_delta_vs_zero_bonus_winner": final_score - zero_score,
                    "would_beat_zero_bonus_winner": bool(scored.get("feasible", False)) and final_score > zero_score,
                    "first_action_delta_from_reference": scored.get("first_action_delta_from_reference", {}),
                    "violation_reason_distribution": scored.get("violation_reason_distribution", {}),
                    "violation_field_distribution": scored.get("violation_field_distribution", {}),
                }
            )
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("primary_risk_family")), str(row.get("template_name"))), []).append(row)
    for (family, template_name), items in sorted(grouped.items()):
        feasible_items = [item for item in items if item.get("feasible", False)]
        deltas = [_num(item.get("score_delta_vs_conservative_stabilize")) for item in feasible_items]
        winner_count = sum(1 for item in feasible_items if item.get("would_beat_zero_bonus_winner", False))
        summary_rows.append(
            {
                "primary_risk_family": family,
                "template_name": template_name,
                "row_count": len(items),
                "feasible_count": len(feasible_items),
                "feasible_rate": len(feasible_items) / max(len(items), 1),
                "mean_score_delta_vs_conservative_stabilize": float(mean(deltas)) if deltas else 0.0,
                "positive_delta_rate": sum(1 for value in deltas if value > 0.0) / max(len(deltas), 1),
                "would_beat_zero_bonus_winner_count": winner_count,
                "would_beat_zero_bonus_winner_rate": winner_count / max(len(items), 1),
            }
        )
    best_template = max(
        summary_rows,
        key=lambda row: (
            _num(row.get("would_beat_zero_bonus_winner_rate")),
            _num(row.get("mean_score_delta_vs_conservative_stabilize")),
        ),
        default={},
    )
    diagnostics = {
        "template_variant_summary": summary_rows,
        "best_template_variant": best_template,
        "template_variant_material_improvement": bool(
            _num(best_template.get("mean_score_delta_vs_conservative_stabilize")) >= 0.015
            and _num(best_template.get("positive_delta_rate")) >= 0.50
        ),
    }
    return rows, diagnostics


def _boundary_clean(boundary: Mapping[str, Any]) -> bool:
    return bool(
        _num(boundary.get("audit_success_rate")) >= 1.0
        and _num(boundary.get("final_action_invariant_rate")) >= 1.0
        and not any(int(boundary.get(field, 0) or 0) for field in BOUNDARY_FALSE_COUNTS)
        and _num(boundary.get("fallback_rate")) <= 0.0
        and _num(boundary.get("mean_infeasible_candidate_ratio")) <= 0.0
        and int(boundary.get("previous_shift_all_max_delta_count", 0) or 0) == 0
        and int(boundary.get("selected_candidate_max_delta_count", 0) or 0) == 0
    )


def _dominant_component(component_rows: Sequence[Mapping[str, Any]]) -> str:
    if not component_rows:
        return ""
    return str(max(component_rows, key=lambda row: abs(_num(row.get("mean")))).get("component") or "")


def _mixed_risk_unstable(records: Sequence[Mapping[str, Any]], best_vector: Mapping[str, Any]) -> bool:
    mixed = [
        record
        for record in records
        if (record.get("risk_flags") or {}).get("any_risk", False) and len(record.get("active_risk_flags") or []) >= 2
    ]
    if len(mixed) < 10:
        return False
    return _num(best_vector.get("risk_type_switch_spread")) > 0.45


def _route_next_action(
    *,
    boundary: Mapping[str, Any],
    schema_issues: Sequence[str],
    diagnostics: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> tuple[str, list[str]]:
    if not _boundary_clean(boundary):
        return "stop_and_repair_cstcc_shadow_boundary", ["boundary_or_candidate_health_failure"]
    if schema_issues:
        return "v8173_schema_or_zero_bonus_reconstruction_repair_plan", list(schema_issues)
    best = diagnostics.get("best_balanced_vector") or {}
    if (diagnostics.get("template_variant") or {}).get("template_variant_material_improvement", False):
        return "v8174_risk_specific_conservative_template_shadow_design_plan", ["risk_specific_template_variant_material_improvement"]
    if _mixed_risk_unstable(records, best):
        return "v8173_mixed_risk_conflict_resolver_design_plan", ["mixed_risk_response_unstable"]
    dominant = _dominant_component(diagnostics.get("zero_minus_conservative_component_quantiles") or [])
    if (
        _num(best.get("risk_type_switch_spread"), 999.0) <= 0.30
        and int(best.get("nonrisk_conservative_selected_count", 0) or 0) == 0
        and not bool(best.get("rule_suppression_detected", False))
        and not bool(best.get("previous_rule_ppo_suppression_detected", False))
        and _num(best.get("conservative_risk_selected_rate")) > 0.0
    ):
        return "v8173_1_opt_in_risk_bonus_shadow_trace_plan", ["risk_type_specific_bonus_vector_balanced"]
    if _num(best.get("conservative_risk_selected_rate")) <= 0.0 or dominant in {"projected_energy_proxy", "raw_energy_proxy"}:
        return "v8173_energy_proxy_reweight_design_plan", ["bonus_response_still_energy_proxy_dominated"]
    return "v8173_mixed_risk_conflict_resolver_design_plan", ["risk_type_bonus_vector_not_balanced"]


def build_report(
    *,
    trace_csv_rows: list[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    jsonl_rows: list[Mapping[str, Any]],
    v8172_json: Mapping[str, Any] | None = None,
    vectors: Sequence[Mapping[str, float]] | None = None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    boundary = _boundary_summary(summary_json, jsonl_rows)
    schema_issues = _missing_schema_fields(jsonl_rows)
    records, reconstruction_issues = _build_step_records(trace_csv_rows=trace_csv_rows, jsonl_rows=jsonl_rows)
    schema_issues = sorted(set(schema_issues + reconstruction_issues))
    curve_rows, family_rows, switch_rows, bonus_diagnostics = evaluate_bonus_vectors(records, vectors=vectors)
    template_rows, template_diagnostics = evaluate_template_variants(records)
    diagnostics = {
        **bonus_diagnostics,
        "template_variant": template_diagnostics,
        "v8172_reference_next_action": (v8172_json or {}).get("next_action"),
        "v8172_reference_risk_type_switch_spread": ((v8172_json or {}).get("diagnostics") or {})
        .get("risk_type_switch_divergence", {})
        .get("max_spread"),
    }
    next_action, readiness_reasons = _route_next_action(
        boundary=boundary,
        schema_issues=schema_issues,
        diagnostics=diagnostics,
        records=records,
    )
    best = diagnostics.get("best_balanced_vector") or {}
    report = {
        "schema_version": "cstcc_v8173_risk_type_specific_bonus_template_counterfactual_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_level": "existing-trace counterfactual scorer audit",
        "boundaries": {
            "online_llm_called": False,
            "predictive_rollout_executed": False,
            "real_tomato_safety_projection": False,
            "final_action_changed": False,
            "performance_claim_allowed": False,
            "promotion_evidence": False,
            "runtime_scoring_changed": False,
            "runtime_template_pool_changed": False,
            "default_controller_changed": False,
        },
        "boundary_runtime_safety": boundary,
        "schema_issues": schema_issues,
        "counterfactual_definition": {
            "global_bonus_used": False,
            "bonus_is_non_additive": True,
            "primary_family_priority": list(RISK_FAMILIES),
            "bonus_grids": {
                "dry_or_high_vpd_bonus": list(DRY_OR_HIGH_VPD_GRID),
                "dry_vent_bonus": list(DRY_VENT_GRID),
                "dew_or_humidity_bonus": list(DEW_OR_HUMIDITY_GRID),
                "canopy_dew_bonus": list(CANOPY_DEW_GRID),
            },
            "pseudo_templates_are_runtime_templates": False,
            "runtime_final_action_changed": False,
            "baseline_score_dual_changed": False,
            "prior_confidence_changed": False,
            "rule_prior_changed": False,
        },
        "diagnostics": diagnostics,
        "acceptance": {
            "boundary_clean": _boundary_clean(boundary),
            "nonrisk_conservative_selected_count_eq_0": int(best.get("nonrisk_conservative_selected_count", 0) or 0) == 0,
            "risk_type_switch_spread_le_0_30": _num(best.get("risk_type_switch_spread"), 999.0) <= 0.30,
            "spread_reduced_vs_v8172": _num(best.get("risk_type_switch_spread"), 999.0)
            < _num(diagnostics.get("v8172_reference_risk_type_switch_spread"), 999.0),
            "selected_id_vs_zero_bonus_winner_mismatch_count_present": "selected_id_vs_zero_bonus_winner_mismatch_count"
            in best,
        },
        "next_action": next_action,
        "readiness_reasons": readiness_reasons,
        "notes": [
            "v8173 is offline counterfactual design only.",
            "Pseudo templates are evaluator-only and are not added to runtime TEMPLATE_NAMES.",
            "It does not validate reward, profit, safety improvement, or closed-loop controller quality.",
        ],
    }
    return report, {
        "bonus_vector_curve": curve_rows,
        "risk_family_curve": family_rows,
        "template_variant_counterfactual": template_rows,
        "switch_explainability": switch_rows,
    }


def build_markdown(report: Mapping[str, Any]) -> str:
    boundary = report.get("boundary_runtime_safety", {})
    diagnostics = report.get("diagnostics", {})
    best = diagnostics.get("best_balanced_vector", {})
    template = (diagnostics.get("template_variant") or {}).get("best_template_variant", {})
    lines = [
        "# C-STCC v8173 Risk-Type-Specific Bonus + Template Counterfactual",
        "",
        "Existing-trace counterfactual scorer audit only. No online LLM, no rollout, no runtime scorer or template change.",
        "",
        "## Boundary",
        "",
        f"- Audit success rate: {_num(boundary.get('audit_success_rate')):.3f}",
        f"- Final-action invariant rate: {_num(boundary.get('final_action_invariant_rate')):.3f}",
        f"- Online LLM / rollout / real projection steps: {boundary.get('online_llm_called_steps', 0)} / {boundary.get('predictive_rollout_executed_steps', 0)} / {boundary.get('real_tomato_safety_projection_steps', 0)}",
        f"- Final action changed steps: {boundary.get('final_action_changed_steps', 0)}",
        "",
        "## Best Bonus Vector",
        "",
        f"- Vector: {best.get('vector_id', 'none')}",
        f"- Conservative risk selected rate: {_num(best.get('conservative_risk_selected_rate')):.3f}",
        f"- Risk-type switch spread: {_num(best.get('risk_type_switch_spread')):.3f}",
        f"- Nonrisk conservative selected count: {best.get('nonrisk_conservative_selected_count', 0)}",
        f"- Selected id vs zero-bonus mismatch count: {best.get('selected_id_vs_zero_bonus_winner_mismatch_count', 0)}",
        "",
        "## Template Variant Counterfactual",
        "",
        f"- Best pseudo template: {template.get('template_name', 'none')} for {template.get('primary_risk_family', 'none')}",
        f"- Mean score delta vs conservative_stabilize: {_num(template.get('mean_score_delta_vs_conservative_stabilize')):.6f}",
        f"- Would beat zero-bonus winner rate: {_num(template.get('would_beat_zero_bonus_winner_rate')):.3f}",
        "",
        "## Decision",
        "",
        f"- Next action: {report.get('next_action')}",
        f"- Readiness reasons: {', '.join(report.get('readiness_reasons', [])) or 'none'}",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    report: Mapping[str, Any],
    tables: Mapping[str, list[Mapping[str, Any]]],
    output_prefix: str | Path,
) -> dict[str, str]:
    prefix = _resolve(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stem = prefix.name
    parent = prefix.parent
    base = "risk_type_specific_bonus_template_counterfactual"
    paths = {
        "json": prefix.with_suffix(".json"),
        "md": prefix.with_suffix(".md"),
        "bonus_vector_curve_csv": (parent / stem.replace(base, "bonus_vector_curve")).with_suffix(".csv"),
        "risk_family_curve_csv": (parent / stem.replace(base, "risk_family_curve")).with_suffix(".csv"),
        "template_variant_counterfactual_csv": (
            parent / stem.replace(base, "template_variant_counterfactual")
        ).with_suffix(".csv"),
        "switch_explainability_csv": (parent / stem.replace(base, "switch_explainability")).with_suffix(".csv"),
    }
    paths["json"].write_text(_compact_json(report) + "\n", encoding="utf-8")
    paths["md"].write_text(build_markdown(report), encoding="utf-8")
    _write_csv(paths["bonus_vector_curve_csv"], tables.get("bonus_vector_curve", []))
    _write_csv(paths["risk_family_curve_csv"], tables.get("risk_family_curve", []))
    _write_csv(paths["template_variant_counterfactual_csv"], tables.get("template_variant_counterfactual", []))
    _write_csv(paths["switch_explainability_csv"], tables.get("switch_explainability", []))
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v8173 C-STCC risk-type-specific counterfactuals.")
    parser.add_argument("--trace-csv", type=str, default=str(DEFAULT_TRACE_CSV))
    parser.add_argument("--summary-json", type=str, default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--jsonl-root", type=str, default=str(DEFAULT_JSONL_ROOT))
    parser.add_argument("--v8172-json", type=str, default=str(DEFAULT_V8172_JSON))
    parser.add_argument("--output-prefix", type=str, default=str(DEFAULT_OUTPUT_PREFIX))
    args = parser.parse_args()

    report, tables = build_report(
        trace_csv_rows=_load_csv(args.trace_csv),
        summary_json=_load_json(args.summary_json),
        jsonl_rows=_load_jsonl_rows(args.jsonl_root),
        v8172_json=_load_json(args.v8172_json),
    )
    outputs = write_outputs(report=report, tables=tables, output_prefix=args.output_prefix)
    print(
        json.dumps(
            {"outputs": outputs, "next_action": report["next_action"], "readiness_reasons": report["readiness_reasons"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
