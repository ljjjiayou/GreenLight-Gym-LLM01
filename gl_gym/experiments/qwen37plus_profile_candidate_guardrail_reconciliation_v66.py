"""Build v66 profile/candidate/guardrail reconciliation shadow artifacts.

This stage is offline-only. It reads v64/v65 failure traces and asks whether
the existing final-control path had a safer, profile-compatible alternative in
the already-recorded candidates. It does not run rollout, call online LLMs,
change llm_rspc_v2, or authorize controlled replay.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gl_gym.experiments.cvodes_failure_causal_attribution_v44 as v44  # noqa: E402
import gl_gym.experiments.qwen37plus_structured_bridge_runtime_failure_attribution_v65 as v65  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import AUDIT_DIR  # noqa: E402


MODEL_NAME = "qwen3.7-plus"
FAILURE_SCENARIOS = tuple(v65.FAILURE_SCENARIOS)
WINDOW_STEPS = 120
V64_TRACE_DIR = v65.V64_TRACE_DIR
V65_ATTRIBUTION_JSON = v65.ATTRIBUTION_JSON

RECONCILIATION_JSON = AUDIT_DIR / "qwen37plus_profile_candidate_guardrail_reconciliation_audit_20260603_v66.json"
RECONCILIATION_MD = AUDIT_DIR / "qwen37plus_profile_candidate_guardrail_reconciliation_audit_20260603_v66.md"
SHADOW_SELECTION_JSON = AUDIT_DIR / "qwen37plus_reconciled_candidate_shadow_selection_20260603_v66.json"
SHADOW_SELECTION_MD = AUDIT_DIR / "qwen37plus_reconciled_candidate_shadow_selection_20260603_v66.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260603_v66.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260603_v66.md"

ACTION_FIELDS = {
    "heating": "u_heating",
    "co2": "u_co2",
    "screen": "u_screen",
    "ventilation": "u_ventilation",
    "lighting": "u_lighting",
    "shading": "u_shading",
}
ACTION_ALIASES = {
    "heating": ("heating", "heat", "u_heating"),
    "co2": ("co2", "u_co2"),
    "screen": ("screen", "u_screen"),
    "ventilation": ("ventilation", "vent", "u_ventilation"),
    "lighting": ("lighting", "lamp", "u_lighting"),
    "shading": ("shading", "shade", "u_shading"),
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _rel(path: str | Path) -> str:
    p = _resolve(path)
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    p = _resolve(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _trace_path(trace_dir: str | Path, scenario_id: str, controller: str = "llm_rspc_v2") -> Path:
    root = _resolve(trace_dir)
    expected = root / f"{scenario_id}_{controller}.csv"
    if expected.exists():
        return expected
    match = re.search(r"y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)", scenario_id)
    if not match:
        return expected
    prefix = f"y{match.group('year')}_d{match.group('day')}_s{match.group('seed')}_"
    matches = sorted(root.glob(f"{prefix}*_{controller}.csv"))
    return matches[0] if matches else expected


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
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "applied"}


def _step(row: Mapping[str, Any] | None, default: int = -1) -> int:
    if row is None:
        return int(default)
    return int(_num(row.get("step"), default))


def _parse_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


def _action_from_mapping(mapping: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for canonical, aliases in ACTION_ALIASES.items():
        value = 0.0
        for key in aliases:
            if key in mapping:
                value = _num(mapping.get(key))
                break
        out[canonical] = float(value)
    return out


def _action_from_row(row: Mapping[str, Any]) -> dict[str, float]:
    return {canonical: float(_num(row.get(field))) for canonical, field in ACTION_FIELDS.items()}


def _tomato_projected_action(row: Mapping[str, Any]) -> dict[str, float]:
    action = _action_from_row(row)
    after_map = {
        "heating": "tomato_safety_v2_heat_after",
        "ventilation": "tomato_safety_v2_vent_after",
        "screen": "tomato_safety_v2_screen_after",
        "shading": "tomato_safety_v2_shade_after",
    }
    for canonical, field in after_map.items():
        if str(row.get(field, "") or "").strip():
            action[canonical] = float(_num(row.get(field)))
    return action


def _runtime_failure_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    failure = v65._runtime_failure_row(rows)
    return _step(failure) if failure is not None else None


def _window(rows: Sequence[Mapping[str, Any]], *, center_step: int | None, window_steps: int) -> list[Mapping[str, Any]]:
    if not rows:
        return []
    if center_step is None:
        center_step = _step(rows[-1], len(rows) - 1)
    start = max(0, int(center_step) - int(window_steps) + 1)
    return [row for row in rows if start <= _step(row) <= int(center_step)]


def _candidate_json_items(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    decoded = _parse_json(row.get("rspc_action_candidates_json"))
    if not isinstance(decoded, list):
        return []
    return [item for item in decoded if isinstance(item, dict)]


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    if "post_shape_score" in candidate:
        return _num(candidate.get("post_shape_score"))
    if "score" in candidate:
        return _num(candidate.get("score"))
    terms = candidate.get("score_terms", {})
    if isinstance(terms, Mapping):
        return _num(terms.get("total_score", terms.get("score", 0.0)))
    return 0.0


def _candidate_prediction(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    terms = candidate.get("post_shape_score_terms")
    if isinstance(terms, Mapping):
        return terms
    terms = candidate.get("score_terms")
    return terms if isinstance(terms, Mapping) else {}


def _executable_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    candidates.append(
        {
            "name": "executed_final_action",
            "source": "final_control",
            "action": _action_from_row(row),
            "original_selected": True,
            "base_score": 0.0,
            "prediction": {},
        }
    )
    tomato_action = _tomato_projected_action(row)
    if tomato_action != _action_from_row(row):
        candidates.append(
            {
                "name": "tomato_safety_projected_action",
                "source": "tomato_safety_projection",
                "action": tomato_action,
                "original_selected": False,
                "base_score": 0.0,
                "prediction": {},
            }
        )
    recovery = _parse_json(row.get("recovery_anchor_action"))
    if isinstance(recovery, Mapping) and recovery:
        candidates.append(
            {
                "name": "recovery_anchor_action",
                "source": "recovery_anchor",
                "action": _action_from_mapping(recovery),
                "original_selected": False,
                "base_score": 0.0,
                "prediction": {},
            }
        )
    for item in _candidate_json_items(row):
        post_shape = item.get("post_shape_action")
        action_source = "post_shape_action" if isinstance(post_shape, Mapping) else "raw_action"
        action_map = post_shape if isinstance(post_shape, Mapping) else item.get("action", {})
        if not isinstance(action_map, Mapping):
            continue
        candidates.append(
            {
                "name": str(item.get("name") or "candidate"),
                "source": f"rspc_candidate:{action_source}",
                "action": _action_from_mapping(action_map),
                "original_selected": bool(item.get("selected", False)),
                "base_score": float(_candidate_score(item)),
                "prediction": dict(_candidate_prediction(item)),
            }
        )
    return candidates


def _profile_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = [part.strip() for part in str(row.get("profile_candidate_names", "") or "").split(",") if part.strip()]
    return [
        {
            "name": name,
            "source": "profile_generator_shadow_candidate",
            "executable_low_level_action": False,
        }
        for name in names
    ]


def _safety_required_vent(row: Mapping[str, Any]) -> bool:
    reasons = str(row.get("tomato_safety_v2_reasons", "") or row.get("final_action_risk_reason_v2", "") or "").lower()
    return any(token in reasons for token in ("hot_temperature", "hard", "canopy", "dew", "rh_hard"))


def _row_dry_risk(row: Mapping[str, Any]) -> bool:
    return (
        _truthy(row.get("dry_risk"))
        or _num(row.get("vpd_air"), 0.0) >= 2.25
        or _num(row.get("rh_air"), 70.0) <= 45.0
    )


def evaluate_candidate(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    previous_action: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    action = candidate.get("action", {})
    if not isinstance(action, Mapping):
        action = {}
    target_temp = _num(row.get("target_temp"))
    target_rh = _num(row.get("target_rh"))
    target_co2 = _num(row.get("target_co2"))
    temp_air = _num(row.get("temp_air"))
    safety_required = _safety_required_vent(row)
    profile_conflicts: list[str] = []
    if target_rh >= 75.0 and _num(action.get("ventilation")) >= 0.70:
        profile_conflicts.append("target_rh_vs_high_vent")
    if target_co2 >= 800.0 and _num(action.get("ventilation")) >= 0.30:
        profile_conflicts.append("co2_target_vs_vent")
    if target_temp <= temp_air - 1.0 and _num(action.get("heating")) >= 0.10:
        profile_conflicts.append("cooling_target_vs_heat")
    if target_temp >= temp_air + 1.0 and _num(action.get("ventilation")) >= 0.50:
        profile_conflicts.append("heating_target_vs_vent")
    dry_high_vent = bool(_row_dry_risk(row) and _num(action.get("ventilation")) >= 0.70 and not safety_required)
    if dry_high_vent:
        profile_conflicts.append("dry_risk_vs_high_vent")

    actuator_conflicts: list[str] = []
    if _num(action.get("heating")) >= 0.10 and _num(action.get("ventilation")) >= 0.40:
        actuator_conflicts.append("heat_vent_conflict")
    if _num(action.get("co2")) >= 0.05 and _num(action.get("ventilation")) >= 0.20:
        actuator_conflicts.append("co2_vent_leak")
    if _num(action.get("screen")) >= 0.30 and _num(action.get("ventilation")) >= 0.85:
        actuator_conflicts.append("screen_vent_conflict")
    if _num(action.get("shading")) >= 0.80 and _num(action.get("screen")) >= 0.30:
        actuator_conflicts.append("shade_screen_conflict")

    prediction = candidate.get("prediction", {})
    if not isinstance(prediction, Mapping):
        prediction = {}
    hard_safety = bool(
        _truthy(prediction.get("predicted_canopy_lt0_v2"))
        or _truthy(prediction.get("predicted_canopy_warning_v2"))
        or _truthy(prediction.get("predicted_dew_lt0"))
        or _truthy(row.get("final_action_would_fail_canopy_boundary_v2"))
        and str(candidate.get("source")) == "final_control"
    )
    expected_rewrite_fields: set[str] = set()
    if profile_conflicts:
        expected_rewrite_fields.add("ventilation")
    if "screen_vent_conflict" in actuator_conflicts:
        expected_rewrite_fields.update({"screen", "ventilation"})
    if "shade_screen_conflict" in actuator_conflicts:
        expected_rewrite_fields.update({"shading", "screen"})
    if "heat_vent_conflict" in actuator_conflicts:
        expected_rewrite_fields.update({"heating", "ventilation"})
    if hard_safety:
        expected_rewrite_fields.add("hard_safety")

    large_delta = 0
    delta_abs_sum = 0.0
    if previous_action:
        for field in ACTION_FIELDS:
            delta = abs(_num(action.get(field)) - _num(previous_action.get(field)))
            delta_abs_sum += delta
            if delta > 0.20:
                large_delta += 1

    base_score = _num(candidate.get("base_score"))
    compatibility_penalty = (
        100.0 * int(hard_safety)
        + 6.0 * len(profile_conflicts)
        + 4.0 * len(actuator_conflicts)
        + 2.0 * len(expected_rewrite_fields)
        + 5.0 * large_delta
        + 2.0 * delta_abs_sum
    )
    compatible = bool(not hard_safety and not profile_conflicts and len(expected_rewrite_fields) == 0)
    return {
        "name": str(candidate.get("name") or "candidate"),
        "source": str(candidate.get("source") or ""),
        "action": dict(action),
        "base_score": float(base_score),
        "reconciled_score": float(base_score + compatibility_penalty),
        "compatible": compatible,
        "hard_safety_rewrite_predicted": bool(hard_safety),
        "profile_action_conflict": bool(profile_conflicts),
        "profile_action_conflicts": profile_conflicts,
        "actuator_conflicts": actuator_conflicts,
        "screen_vent_conflict": "screen_vent_conflict" in actuator_conflicts,
        "major_or_hard_rewrite_predicted": bool(hard_safety or len(expected_rewrite_fields) >= 2),
        "expected_rewrite_fields": sorted(expected_rewrite_fields),
        "large_action_delta": bool(large_delta),
        "large_action_delta_count": int(large_delta),
        "delta_abs_sum": float(delta_abs_sum),
        "safety_required_vent": bool(safety_required),
    }


def reconcile_row(row: Mapping[str, Any], previous_action: Mapping[str, float] | None = None) -> dict[str, Any]:
    executable = _executable_candidates(row)
    profile_only = _profile_candidates(row)
    evaluated = [evaluate_candidate(row, candidate, previous_action=previous_action) for candidate in executable]
    original = next((item for item in evaluated if item["name"] == "executed_final_action"), evaluated[0] if evaluated else None)
    compatible = [item for item in evaluated if item["compatible"]]
    selected = min(compatible or evaluated, key=lambda item: item["reconciled_score"]) if evaluated else None
    return {
        "step": _step(row),
        "status": "scored" if evaluated else "candidate_metadata_missing",
        "profile_candidate_count": len(profile_only),
        "profile_candidates_shadow_only": profile_only,
        "executable_candidate_count": len(evaluated),
        "compatible_candidate_count": len(compatible),
        "original": original,
        "shadow_selected": selected,
        "shadow_changed": bool(original and selected and original["name"] != selected["name"]),
        "top_candidates": sorted(evaluated, key=lambda item: item["reconciled_score"])[:5],
    }


def _candidate_issue(item: Mapping[str, Any] | None) -> dict[str, bool]:
    if not item:
        return {
            "hard_safety_rewrite_predicted": False,
            "profile_action_conflict": False,
            "candidate_guardrail_issue": False,
            "screen_vent_conflict": False,
            "large_action_delta": False,
            "major_or_hard_rewrite_predicted": False,
        }
    return {
        "hard_safety_rewrite_predicted": bool(item.get("hard_safety_rewrite_predicted", False)),
        "profile_action_conflict": bool(item.get("profile_action_conflict", False)),
        "candidate_guardrail_issue": bool(
            item.get("hard_safety_rewrite_predicted")
            or item.get("profile_action_conflict")
            or item.get("actuator_conflicts")
            or item.get("expected_rewrite_fields")
        ),
        "screen_vent_conflict": bool(item.get("screen_vent_conflict", False)),
        "large_action_delta": bool(item.get("large_action_delta", False)),
        "major_or_hard_rewrite_predicted": bool(item.get("major_or_hard_rewrite_predicted", False)),
    }


def summarize_reconciliation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    results = []
    previous_action: dict[str, float] | None = None
    original_counts: Counter[str] = Counter()
    shadow_counts: Counter[str] = Counter()
    compatible_found = 0
    changed = 0
    missing = 0
    for row in rows:
        result = reconcile_row(row, previous_action=previous_action)
        results.append(result)
        previous_action = _action_from_row(row)
        if result["status"] != "scored":
            missing += 1
            continue
        if int(result["compatible_candidate_count"]) > 0:
            compatible_found += 1
        if result["shadow_changed"]:
            changed += 1
        for key, value in _candidate_issue(result.get("original")).items():
            if value:
                original_counts[key] += 1
        for key, value in _candidate_issue(result.get("shadow_selected")).items():
            if value:
                shadow_counts[key] += 1
    return {
        "row_count": len(rows),
        "scored_row_count": len(rows) - missing,
        "candidate_metadata_missing_steps": int(missing),
        "compatible_candidate_found_steps": int(compatible_found),
        "shadow_changed_steps": int(changed),
        "original": {
            "hard_safety_rewrite_predicted_steps": int(original_counts["hard_safety_rewrite_predicted"]),
            "profile_action_conflict_steps": int(original_counts["profile_action_conflict"]),
            "candidate_guardrail_issue_steps": int(original_counts["candidate_guardrail_issue"]),
            "screen_vent_conflict_steps": int(original_counts["screen_vent_conflict"]),
            "large_action_delta_steps": int(original_counts["large_action_delta"]),
            "major_or_hard_rewrite_predicted_steps": int(original_counts["major_or_hard_rewrite_predicted"]),
        },
        "shadow": {
            "hard_safety_rewrite_predicted_steps": int(shadow_counts["hard_safety_rewrite_predicted"]),
            "profile_action_conflict_steps": int(shadow_counts["profile_action_conflict"]),
            "candidate_guardrail_issue_steps": int(shadow_counts["candidate_guardrail_issue"]),
            "screen_vent_conflict_steps": int(shadow_counts["screen_vent_conflict"]),
            "large_action_delta_steps": int(shadow_counts["large_action_delta"]),
            "major_or_hard_rewrite_predicted_steps": int(shadow_counts["major_or_hard_rewrite_predicted"]),
        },
        "sample_rows": results[:20],
    }


def _scenario_rows(trace_dir: str | Path, scenario_id: str, window_steps: int) -> list[Mapping[str, Any]]:
    rows = v65._read_rows(_trace_path(trace_dir, scenario_id))
    failure_step = v65._runtime_failure_row(rows)
    center = v65._step(failure_step, len(rows) - 1 if rows else -1) if failure_step is not None else None
    return v65._window(rows, center_step=center, window_steps=window_steps)


def build_reconciliation_audit(
    *,
    trace_dir: str | Path = V64_TRACE_DIR,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    window_steps: int = WINDOW_STEPS,
    v65_attribution_json: str | Path = V65_ATTRIBUTION_JSON,
) -> dict[str, Any]:
    scenario_reports = []
    aggregate_original: Counter[str] = Counter()
    aggregate_shadow: Counter[str] = Counter()
    compatible_found_total = 0
    row_total = 0
    for scenario in failure_scenarios:
        rows = _scenario_rows(trace_dir, scenario, window_steps)
        summary = summarize_reconciliation(rows)
        row_total += int(summary["row_count"])
        compatible_found_total += int(summary["compatible_candidate_found_steps"])
        for key, value in summary["original"].items():
            aggregate_original[key] += int(value)
        for key, value in summary["shadow"].items():
            aggregate_shadow[key] += int(value)
        scenario_reports.append(
            {
                "scenario_id": scenario,
                "window_row_count": int(summary["row_count"]),
                "candidate_metadata_missing_steps": int(summary["candidate_metadata_missing_steps"]),
                "compatible_candidate_found_steps": int(summary["compatible_candidate_found_steps"]),
                "shadow_changed_steps": int(summary["shadow_changed_steps"]),
                "original": summary["original"],
                "shadow": summary["shadow"],
                "sample_rows": summary["sample_rows"],
            }
        )
    v65_attr = _load_json(v65_attribution_json)
    acceptance = {
        "hard_safety_rewrite_predicted_steps_shadow_lt_original": aggregate_shadow["hard_safety_rewrite_predicted_steps"]
        < aggregate_original["hard_safety_rewrite_predicted_steps"],
        "profile_action_conflict_steps_shadow_lt_original": aggregate_shadow["profile_action_conflict_steps"]
        < aggregate_original["profile_action_conflict_steps"],
        "candidate_guardrail_issue_steps_shadow_lt_original": aggregate_shadow["candidate_guardrail_issue_steps"]
        < aggregate_original["candidate_guardrail_issue_steps"],
        "screen_vent_conflict_shadow_lte_original": aggregate_shadow["screen_vent_conflict_steps"]
        <= aggregate_original["screen_vent_conflict_steps"],
        "large_action_delta_shadow_lte_original": aggregate_shadow["large_action_delta_steps"]
        <= aggregate_original["large_action_delta_steps"],
        "compatible_candidate_exists": compatible_found_total > 0,
    }
    pass_all = bool(all(acceptance.values()))
    no_compatible = compatible_found_total == 0
    return {
        "artifact": "qwen37plus_profile_candidate_guardrail_reconciliation_audit_20260603_v66",
        "current_stage": "v66_qwen37plus_profile_candidate_guardrail_reconciliation_shadow",
        "model_name": MODEL_NAME,
        "scope": "offline_shadow_candidate_reconciliation_only",
        "source_v65_artifact": v65_attr.get("artifact", ""),
        "source_v65_dominant_attribution": v65_attr.get("dominant_attribution", ""),
        "failure_scenarios": list(failure_scenarios),
        "window_steps": int(window_steps),
        "row_count": int(row_total),
        "compatible_candidate_found_steps": int(compatible_found_total),
        "no_safe_compatible_candidate_available": bool(no_compatible),
        "aggregate": {
            "original": dict(sorted(aggregate_original.items())),
            "shadow": dict(sorted(aggregate_shadow.items())),
        },
        "acceptance": acceptance,
        "reconciliation_shadow_pass": pass_all,
        "scenario_reports": scenario_reports,
        "online_llm_called": False,
        "new_rollout_run": False,
        "default_llm_rspc_v2_changed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": "minimal_profile_candidate_guardrail_reconciliation_shadow_rollout_plan"
        if pass_all
        else (
            "profile_generator_tomato_safety_boundary_reconciliation_plan"
            if no_compatible
            else "profile_candidate_guardrail_reconciliation_design_refinement"
        ),
    }


def build_shadow_selection(audit: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for report in audit.get("scenario_reports", []) or []:
        for row in report.get("sample_rows", []) or []:
            rows.append(
                {
                    "scenario_id": report.get("scenario_id"),
                    "step": row.get("step"),
                    "status": row.get("status"),
                    "original": row.get("original"),
                    "shadow_selected": row.get("shadow_selected"),
                    "shadow_changed": row.get("shadow_changed"),
                    "compatible_candidate_count": row.get("compatible_candidate_count"),
                    "profile_candidates_shadow_only": row.get("profile_candidates_shadow_only"),
                }
            )
    return {
        "artifact": "qwen37plus_reconciled_candidate_shadow_selection_20260603_v66",
        "current_stage": "v66_qwen37plus_profile_candidate_guardrail_reconciliation_shadow",
        "selection_rows": rows,
        "selection_row_count": len(rows),
        "shadow_selection_changes_final_control": False,
        "rollout_execution_authorized": False,
        "online_llm_called": False,
        "new_rollout_run": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": audit.get("next_action", "profile_candidate_guardrail_reconciliation_design_refinement"),
    }


def build_readiness(audit: Mapping[str, Any], selection: Mapping[str, Any]) -> dict[str, Any]:
    taxonomy = []
    if bool(audit.get("no_safe_compatible_candidate_available", False)):
        taxonomy.append("no_safe_compatible_candidate_available")
    if not bool(audit.get("acceptance", {}).get("candidate_guardrail_issue_steps_shadow_lt_original", False)):
        taxonomy.append("candidate_guardrail_pressure_persists")
    if not bool(audit.get("acceptance", {}).get("profile_action_conflict_steps_shadow_lt_original", False)):
        taxonomy.append("profile_action_conflict_persists")
    if int(sum(int(r.get("candidate_metadata_missing_steps", 0) or 0) for r in audit.get("scenario_reports", []) or [])) > 0:
        taxonomy.append("candidate_metadata_missing")
    return {
        "artifact": "metadata_replay_readiness_checklist_20260603_v66",
        "current_stage": "v66_qwen37plus_profile_candidate_guardrail_reconciliation_shadow",
        "model_name": MODEL_NAME,
        "reconciliation_shadow_pass": bool(audit.get("reconciliation_shadow_pass", False)),
        "compatible_candidate_found_steps": int(audit.get("compatible_candidate_found_steps", 0) or 0),
        "no_safe_compatible_candidate_available": bool(audit.get("no_safe_compatible_candidate_available", False)),
        "rollout_execution_authorized": False,
        "online_llm_called": False,
        "new_rollout_run": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": audit.get("next_action", "profile_candidate_guardrail_reconciliation_design_refinement"),
        "failure_taxonomy": sorted(set(taxonomy)),
    }


def _write_json_md(path_json: Path, path_md: Path, payload: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    lines = [
        f"# {title}",
        "",
        f"- Controlled replay allowed: `{payload.get('controlled_replay_allowed', False)}`",
        f"- Performance claim allowed: `{payload.get('performance_claim_allowed', False)}`",
        f"- Promotion evidence: `{payload.get('promotion_evidence', False)}`",
        f"- Next action: `{payload.get('next_action', '')}`",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        "```",
        "",
    ]
    path_md.write_text("\n".join(lines), encoding="utf-8")


def write_all(
    *,
    trace_dir: str | Path = V64_TRACE_DIR,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    window_steps: int = WINDOW_STEPS,
) -> dict[str, Any]:
    audit = build_reconciliation_audit(
        trace_dir=trace_dir,
        failure_scenarios=failure_scenarios,
        window_steps=window_steps,
    )
    selection = build_shadow_selection(audit)
    readiness = build_readiness(audit, selection)
    _write_json_md(RECONCILIATION_JSON, RECONCILIATION_MD, audit, "v66 qwen3.7-plus Profile/Candidate/Guardrail Reconciliation Audit")
    _write_json_md(SHADOW_SELECTION_JSON, SHADOW_SELECTION_MD, selection, "v66 qwen3.7-plus Reconciled Candidate Shadow Selection")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v66 Metadata Replay Readiness")
    return {"audit": audit, "selection": selection, "readiness": readiness}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default=_rel(V64_TRACE_DIR))
    parser.add_argument("--window-steps", type=int, default=WINDOW_STEPS)
    parser.add_argument("--failure-scenario", action="append", default=[])
    parser.add_argument("--write-all", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = write_all(
        trace_dir=args.trace_dir,
        failure_scenarios=args.failure_scenario or list(FAILURE_SCENARIOS),
        window_steps=int(args.window_steps),
    )
    if not args.write_all:
        print(json.dumps(result["readiness"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
