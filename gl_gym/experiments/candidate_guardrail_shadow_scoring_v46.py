"""Build v46 shadow candidate/guardrail compatibility scoring artifacts.

This audit is offline-only. It re-ranks existing per-step candidate JSON from
v39/v43 traces with a guardrail-compatibility penalty. It does not run rollout,
call online LLMs, mutate the default controller, authorize controlled replay,
or make performance claims.
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

from gl_gym.experiments.diagnose_ppo_vs_llm import derive_candidate_guardrail_compatibility


FAILURE_SCENARIOS = (
    "y2010_d180_s43_n720",
    "y2018_d181_s42_n720",
    "y2018_d181_s43_n720",
)
ACTION_ALIASES = {
    "heating": ("heat", "heating", "u_heating"),
    "co2": ("co2", "u_co2"),
    "screen": ("screen", "u_screen"),
    "ventilation": ("vent", "ventilation", "u_ventilation"),
    "lighting": ("lamp", "lighting", "u_lighting"),
    "shading": ("shade", "shading", "u_shading"),
}
ACTION_NAMES = tuple(ACTION_ALIASES.keys())
WEIGHTS = {
    "profile_action_conflict": 1.25,
    "dry_risk_high_vent_conflict": 1.50,
    "actuator_inconsistency": 1.00,
    "expected_rewrite_field": 0.75,
    "candidate_filter_reason": 0.40,
    "anchor_or_previous_compatible_bonus": -0.25,
}

AUDIT_DIR = PROJECT_ROOT / "gl_gym" / "result" / "audits"
ORIGINAL_TRACE_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "stable_long_horizon_evaluation_pool_v38_20260529"
    / "traces_h720"
)
V43_TRACE_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "transition_gate_shadow_rollout_v43_20260530"
    / "traces"
)
V45_AUDIT_JSON = AUDIT_DIR / "profile_candidate_guardrail_compatibility_audit_20260531_v45.json"
V45_DESIGN_JSON = AUDIT_DIR / "candidate_guardrail_scoring_patch_design_20260531_v45.json"

PATCH_JSON = AUDIT_DIR / "candidate_guardrail_shadow_scoring_patch_20260531_v46.json"
PATCH_MD = AUDIT_DIR / "candidate_guardrail_shadow_scoring_patch_20260531_v46.md"
COMPARISON_JSON = AUDIT_DIR / "candidate_guardrail_shadow_scoring_comparison_20260531_v46.json"
COMPARISON_MD = AUDIT_DIR / "candidate_guardrail_shadow_scoring_comparison_20260531_v46.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260531_v46.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260531_v46.md"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


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
    return json.loads(p.read_text(encoding="utf-8"))


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
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "applied"}


def _parts_from_scenario_id(value: str) -> dict[str, int]:
    match = re.search(r"y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)(?:_n(?P<steps>\d+))?", str(value))
    if not match:
        return {"year": 0, "day": 0, "seed": 0, "max_steps": 720}
    return {
        "year": int(match.group("year")),
        "day": int(match.group("day")),
        "seed": int(match.group("seed")),
        "max_steps": int(match.group("steps") or 720),
    }


def _scenario_from_path(path: Path) -> str:
    match = re.match(r"(y\d+_d\d+_s\d+_n\d+)_llm_rspc_v2\.csv$", path.name)
    return match.group(1) if match else path.stem.replace("_llm_rspc_v2", "")


def _trace_path(trace_dir: str | Path, scenario_id: str, controller: str = "llm_rspc_v2") -> Path:
    root = _resolve(trace_dir)
    expected = root / f"{scenario_id}_{controller}.csv"
    if expected.exists():
        return expected
    parts = _parts_from_scenario_id(scenario_id)
    prefix = f"y{parts['year']}_d{parts['day']}_s{parts['seed']}_"
    matches = sorted(root.glob(f"{prefix}*_{controller}.csv"))
    return matches[0] if matches else expected


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    p = _resolve(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _runtime_failure_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    for row in rows:
        text = " ".join(str(row.get(key, "") or "") for key in ("runtime_error", "runtime_error_type", "replan_reason"))
        if text.strip() and (
            "CV_CONV_FAILURE" in text
            or "CVODES" in text
            or "CvodesInterface" in text
            or "runtime_error" in text
            or str(row.get("runtime_error", "") or "").strip()
            or str(row.get("runtime_error_type", "") or "").strip()
        ):
            return int(_num(row.get("step"), -1))
    return None


def _window(rows: Sequence[Mapping[str, Any]], *, center_step: int | None, window_steps: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    if center_step is None:
        center_step = int(_num(rows[-1].get("step"), len(rows) - 1))
    start = max(0, int(center_step) - int(window_steps) + 1)
    return [
        dict(row)
        for row in rows
        if start <= int(_num(row.get("step"), -1)) <= int(center_step)
    ]


def _parse_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = str(row.get("rspc_action_candidates_json", "") or "").strip()
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except Exception:
        return []
    return [item for item in decoded if isinstance(item, dict)] if isinstance(decoded, list) else []


def _candidate_name(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("name") or "candidate")


def _candidate_score(candidate: Mapping[str, Any]) -> float:
    if "score" in candidate:
        return _num(candidate.get("score"))
    terms = candidate.get("score_terms", {})
    if isinstance(terms, Mapping):
        return _num(terms.get("total_score", terms.get("score", 0.0)))
    return 0.0


def _candidate_action(candidate: Mapping[str, Any]) -> dict[str, float]:
    action = candidate.get("action", {})
    if not isinstance(action, Mapping):
        action = {}
    out: dict[str, float] = {}
    for canonical, aliases in ACTION_ALIASES.items():
        value = 0.0
        for key in aliases:
            if key in action:
                value = _num(action.get(key))
                break
        out[canonical] = float(value)
    return out


def _split_values(text: Any) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip() and part.strip() != "none"]


def _reason_count(row: Mapping[str, Any]) -> int:
    derived = derive_candidate_guardrail_compatibility(dict(row))
    text = str(derived.get("candidate_filter_reason_summary") or "")
    return len([part for part in text.split(";") if part.strip()])


def _row_dry_risk(row: Mapping[str, Any]) -> bool:
    return _truthy(row.get("dry_risk")) or _num(row.get("vpd_air")) >= 1.20 or _num(row.get("rh_air"), 70.0) <= 55.0


def _candidate_penalty_terms(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    action = _candidate_action(candidate)
    target_temp = _num(row.get("target_temp"))
    target_rh = _num(row.get("target_rh"))
    target_co2 = _num(row.get("target_co2"))
    temp_air = _num(row.get("temp_air"))
    profile_conflicts: list[str] = []
    if target_rh >= 75.0 and action["ventilation"] >= 0.70:
        profile_conflicts.append("target_rh_vs_high_vent")
    if target_co2 >= 800.0 and action["ventilation"] >= 0.30:
        profile_conflicts.append("co2_target_vs_vent")
    if target_temp <= temp_air - 1.0 and action["heating"] >= 0.10:
        profile_conflicts.append("cooling_target_vs_heat")
    if target_temp >= temp_air + 1.0 and action["ventilation"] >= 0.50:
        profile_conflicts.append("heating_target_vs_vent")

    dry_cap = _num(row.get("dry_vent_cap"), 0.60)
    dry_threshold = max(0.55, min(0.85, dry_cap if dry_cap > 0 else 0.60))
    dry_conflict = bool(_row_dry_risk(row) and action["ventilation"] >= dry_threshold)
    if dry_conflict and "dry_risk_vs_high_vent" not in profile_conflicts:
        profile_conflicts.append("dry_risk_vs_high_vent")

    actuator_conflicts: list[str] = []
    if action["heating"] >= 0.10 and action["ventilation"] >= 0.40:
        actuator_conflicts.append("heat_vent_conflict")
    if action["co2"] >= 0.05 and action["ventilation"] >= 0.20:
        actuator_conflicts.append("co2_vent_leak")
    if action["screen"] >= 0.30 and action["ventilation"] >= 0.85:
        actuator_conflicts.append("screen_vent_latch")
    if action["shading"] >= 0.80 and action["screen"] >= 0.30:
        actuator_conflicts.append("shade_screen_latch")

    row_rewrites = _split_values(derive_candidate_guardrail_compatibility(dict(row)).get("candidate_guardrail_rewrite_fields"))
    expected_rewrite_fields: set[str] = set()
    if dry_conflict or "target_rh_vs_high_vent" in profile_conflicts:
        expected_rewrite_fields.add("ventilation")
    if "heat_vent_conflict" in actuator_conflicts:
        expected_rewrite_fields.update({"heating", "ventilation"})
    if "screen_vent_latch" in actuator_conflicts:
        expected_rewrite_fields.update({"screen", "ventilation"})
    if "shade_screen_latch" in actuator_conflicts:
        expected_rewrite_fields.update({"shading", "screen"})
    for field in row_rewrites:
        if field in expected_rewrite_fields:
            continue
        if field == "ventilation" and action["ventilation"] >= 0.70:
            expected_rewrite_fields.add(field)
        if field == "screen" and action["screen"] >= 0.30:
            expected_rewrite_fields.add(field)
        if field == "shading" and action["shading"] >= 0.50:
            expected_rewrite_fields.add(field)
        if field == "heating" and action["heating"] >= 0.10:
            expected_rewrite_fields.add(field)

    name = _candidate_name(candidate).lower()
    anchor_bonus = 1 if any(token in name for token in ("anchor", "hold", "previous")) else 0
    filter_reasons = _reason_count(row)
    hard_predicted = bool(
        expected_rewrite_fields
        and any(
            token in str(row.get("tomato_safety_v2_reasons", "") or "").lower()
            for token in ("hard", "dew", "canopy", "extreme", "hot_temperature")
        )
    )
    return {
        "action": action,
        "profile_action_conflict_count": len(profile_conflicts),
        "profile_action_conflicts": profile_conflicts,
        "dry_risk_high_vent_conflict": int(dry_conflict),
        "actuator_inconsistency_count": len(actuator_conflicts),
        "actuator_inconsistencies": actuator_conflicts,
        "expected_rewrite_field_count": len(expected_rewrite_fields),
        "expected_rewrite_fields": sorted(expected_rewrite_fields),
        "candidate_filter_reason_count": filter_reasons,
        "anchor_or_previous_compatible_bonus": anchor_bonus,
        "hard_safety_rewrite_predicted": hard_predicted,
    }


def score_candidate(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    terms = _candidate_penalty_terms(row, candidate)
    original_score = _candidate_score(candidate)
    adjustment = (
        WEIGHTS["profile_action_conflict"] * terms["profile_action_conflict_count"]
        + WEIGHTS["dry_risk_high_vent_conflict"] * terms["dry_risk_high_vent_conflict"]
        + WEIGHTS["actuator_inconsistency"] * terms["actuator_inconsistency_count"]
        + WEIGHTS["expected_rewrite_field"] * terms["expected_rewrite_field_count"]
        + WEIGHTS["candidate_filter_reason"] * terms["candidate_filter_reason_count"]
        + WEIGHTS["anchor_or_previous_compatible_bonus"] * terms["anchor_or_previous_compatible_bonus"]
    )
    return {
        "name": _candidate_name(candidate),
        "original_score": float(original_score),
        "adjustment": float(adjustment),
        "adjusted_score": float(original_score + adjustment),
        "selected_original": bool(candidate.get("selected", False)),
        **terms,
    }


def shadow_score_row(row: Mapping[str, Any]) -> dict[str, Any]:
    candidates = _parse_candidates(row)
    if not candidates:
        return {
            "status": "candidate_metadata_missing",
            "step": int(_num(row.get("step"), -1)),
            "candidate_count": 0,
            "original_selected_name": "",
            "shadow_selected_name": "",
            "shadow_changed": False,
            "hard_safety_rewrite_preferred_new": False,
            "original_selected": None,
            "shadow_selected": None,
        }
    scored = [score_candidate(row, candidate) for candidate in candidates]
    original_selected = next((item for item in scored if item["selected_original"]), None)
    if original_selected is None:
        original_selected = min(scored, key=lambda item: item["original_score"])
    shadow_selected = min(scored, key=lambda item: item["adjusted_score"])
    hard_new = bool(
        shadow_selected["hard_safety_rewrite_predicted"]
        and not bool(original_selected.get("hard_safety_rewrite_predicted", False))
    )
    return {
        "status": "scored",
        "step": int(_num(row.get("step"), -1)),
        "candidate_count": len(scored),
        "original_selected_name": original_selected["name"],
        "shadow_selected_name": shadow_selected["name"],
        "shadow_changed": original_selected["name"] != shadow_selected["name"],
        "hard_safety_rewrite_preferred_new": hard_new,
        "original_selected": original_selected,
        "shadow_selected": shadow_selected,
        "top_candidates": sorted(scored, key=lambda item: item["adjusted_score"])[:5],
    }


def _row_original_issue(row: Mapping[str, Any]) -> dict[str, Any]:
    derived = derive_candidate_guardrail_compatibility(dict(row))
    label = str(derived.get("candidate_guardrail_compatibility_label") or "compatible")
    conflicts = _split_values(derived.get("profile_action_conflict_label"))
    return {
        "candidate_guardrail_issue": bool(
            label in {"major_rewrite", "hard_safety_rewrite"}
            or conflicts
            or str(derived.get("candidate_filter_reason_summary") or "").strip()
        ),
        "major_or_hard_rewrite": label in {"major_rewrite", "hard_safety_rewrite"},
        "profile_action_conflict": bool(conflicts),
        "target_rh_vs_high_vent": "target_rh_vs_high_vent" in conflicts,
        "dry_risk_vs_high_vent": "dry_risk_vs_high_vent" in conflicts,
        "hard_safety_rewrite": label == "hard_safety_rewrite",
    }


def _candidate_issue(selected: Mapping[str, Any] | None) -> dict[str, Any]:
    if not selected:
        return {
            "candidate_guardrail_issue": False,
            "major_or_hard_rewrite": False,
            "profile_action_conflict": False,
            "target_rh_vs_high_vent": False,
            "dry_risk_vs_high_vent": False,
            "hard_safety_rewrite": False,
        }
    conflicts = set(selected.get("profile_action_conflicts", []) or [])
    return {
        "candidate_guardrail_issue": bool(
            selected.get("profile_action_conflict_count", 0)
            or selected.get("expected_rewrite_field_count", 0)
            or selected.get("candidate_filter_reason_count", 0)
        ),
        "major_or_hard_rewrite": bool(selected.get("hard_safety_rewrite_predicted") or selected.get("expected_rewrite_field_count", 0) >= 2),
        "profile_action_conflict": bool(selected.get("profile_action_conflict_count", 0)),
        "target_rh_vs_high_vent": "target_rh_vs_high_vent" in conflicts,
        "dry_risk_vs_high_vent": "dry_risk_vs_high_vent" in conflicts,
        "hard_safety_rewrite": bool(selected.get("hard_safety_rewrite_predicted", False)),
    }


def summarize_scored_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored_rows = [shadow_score_row(row) for row in rows]
    valid = [item for item in scored_rows if item["status"] == "scored"]
    missing = [item for item in scored_rows if item["status"] == "candidate_metadata_missing"]
    original_counts: Counter[str] = Counter()
    shadow_counts: Counter[str] = Counter()
    original_selected_counts: Counter[str] = Counter()
    shadow_selected_counts: Counter[str] = Counter()
    hard_new = 0
    for row, scored in zip(rows, scored_rows):
        original_issue = _row_original_issue(row)
        shadow_issue = _candidate_issue(scored.get("shadow_selected") if scored["status"] == "scored" else None)
        for key, value in original_issue.items():
            if value:
                original_counts[key] += 1
        for key, value in shadow_issue.items():
            if value:
                shadow_counts[key] += 1
        if scored["status"] == "scored":
            original_selected_counts[scored["original_selected_name"]] += 1
            shadow_selected_counts[scored["shadow_selected_name"]] += 1
            if scored["hard_safety_rewrite_preferred_new"]:
                hard_new += 1
    changed = sum(1 for item in valid if item["shadow_changed"])
    return {
        "row_count": len(rows),
        "scored_row_count": len(valid),
        "candidate_metadata_missing_steps": len(missing),
        "shadow_changed_steps": int(changed),
        "shadow_change_rate": float(changed / len(valid)) if valid else 0.0,
        "hard_safety_rewrite_preferred_new_steps": int(hard_new),
        "original": {
            "candidate_guardrail_issue_steps": int(original_counts["candidate_guardrail_issue"]),
            "major_or_hard_rewrite_predicted_steps": int(original_counts["major_or_hard_rewrite"]),
            "profile_action_conflict_steps": int(original_counts["profile_action_conflict"]),
            "target_rh_vs_high_vent_steps": int(original_counts["target_rh_vs_high_vent"]),
            "dry_risk_vs_high_vent_steps": int(original_counts["dry_risk_vs_high_vent"]),
            "hard_safety_rewrite_preferred_steps": int(original_counts["hard_safety_rewrite"]),
            "selected_counts": dict(sorted(original_selected_counts.items())),
        },
        "shadow": {
            "candidate_guardrail_issue_steps": int(shadow_counts["candidate_guardrail_issue"]),
            "major_or_hard_rewrite_predicted_steps": int(shadow_counts["major_or_hard_rewrite"]),
            "profile_action_conflict_steps": int(shadow_counts["profile_action_conflict"]),
            "target_rh_vs_high_vent_steps": int(shadow_counts["target_rh_vs_high_vent"]),
            "dry_risk_vs_high_vent_steps": int(shadow_counts["dry_risk_vs_high_vent"]),
            "hard_safety_rewrite_preferred_steps": int(shadow_counts["hard_safety_rewrite"]),
            "selected_counts": dict(sorted(shadow_selected_counts.items())),
        },
        "top_shadow_changes": [
            {
                "step": item["step"],
                "original_selected_name": item["original_selected_name"],
                "shadow_selected_name": item["shadow_selected_name"],
                "original_selected": item.get("original_selected"),
                "shadow_selected": item.get("shadow_selected"),
            }
            for item in valid
            if item["shadow_changed"]
        ][:25],
    }


def _scenario_report(
    scenario_id: str,
    *,
    trace_dir: str | Path,
    window_steps: int,
    mode: str,
) -> dict[str, Any]:
    trace = _trace_path(trace_dir, scenario_id)
    rows = _read_rows(trace)
    failure_step = _runtime_failure_step(rows)
    window = _window(rows, center_step=failure_step, window_steps=window_steps)
    summary = summarize_scored_rows(window)
    return {
        "scenario_id": scenario_id,
        "mode": mode,
        "trace": _rel(trace),
        "trace_exists": bool(rows),
        "failure_step": failure_step,
        "window_steps": window_steps,
        "window_row_count": len(window),
        **summary,
    }


def _discover_stable_scenarios(trace_dir: str | Path, failure_scenarios: Sequence[str]) -> list[str]:
    root = _resolve(trace_dir)
    scenarios = []
    for path in sorted(root.glob("*_llm_rspc_v2.csv")):
        scenario = _scenario_from_path(path)
        if scenario not in failure_scenarios:
            scenarios.append(scenario)
    return scenarios


def _aggregate_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    row_count = sum(int(report.get("row_count", 0) or 0) for report in reports)
    scored = sum(int(report.get("scored_row_count", 0) or 0) for report in reports)
    changed = sum(int(report.get("shadow_changed_steps", 0) or 0) for report in reports)
    missing = sum(int(report.get("candidate_metadata_missing_steps", 0) or 0) for report in reports)
    hard_new = sum(int(report.get("hard_safety_rewrite_preferred_new_steps", 0) or 0) for report in reports)
    original = Counter()
    shadow = Counter()
    for report in reports:
        for key, value in (report.get("original", {}) or {}).items():
            if key != "selected_counts":
                original[key] += int(value or 0)
        for key, value in (report.get("shadow", {}) or {}).items():
            if key != "selected_counts":
                shadow[key] += int(value or 0)
    return {
        "scenario_count": len(reports),
        "row_count": int(row_count),
        "scored_row_count": int(scored),
        "candidate_metadata_missing_steps": int(missing),
        "shadow_changed_steps": int(changed),
        "shadow_change_rate": float(changed / scored) if scored else 0.0,
        "hard_safety_rewrite_preferred_new_steps": int(hard_new),
        "original": dict(sorted(original.items())),
        "shadow": dict(sorted(shadow.items())),
    }


def _accepted(main: Mapping[str, Any], stable: Mapping[str, Any]) -> bool:
    original = main.get("original", {}) or {}
    shadow = main.get("shadow", {}) or {}
    return bool(
        shadow.get("candidate_guardrail_issue_steps", 0) < original.get("candidate_guardrail_issue_steps", 0)
        and shadow.get("profile_action_conflict_steps", 0) < original.get("profile_action_conflict_steps", 0)
        and shadow.get("major_or_hard_rewrite_predicted_steps", 0) <= original.get("major_or_hard_rewrite_predicted_steps", 0)
        and shadow.get("target_rh_vs_high_vent_steps", 0) < original.get("target_rh_vs_high_vent_steps", 0)
        and shadow.get("dry_risk_vs_high_vent_steps", 0) < original.get("dry_risk_vs_high_vent_steps", 0)
        and stable.get("shadow_change_rate", 1.0) <= 0.35
        and stable.get("hard_safety_rewrite_preferred_new_steps", 0) == 0
    )


def build_patch_design(v45_design_json: str | Path = V45_DESIGN_JSON) -> dict[str, Any]:
    v45_design = _load_json(v45_design_json)
    return {
        "schema_version": "candidate_guardrail_shadow_scoring_patch_20260531_v46",
        "design_status": "shadow_scoring_only",
        "source_v45_design_schema": v45_design.get("schema_version", ""),
        "default_llm_rspc_v2_changed": False,
        "online_llm_called": False,
        "new_rollout_run": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "scoring_direction": "lower_score_is_better",
        "weights": dict(WEIGHTS),
        "formula": (
            "adjusted_score = original_score + 1.25*profile_action_conflict_count "
            "+ 1.50*dry_risk_high_vent_conflict + 1.00*actuator_inconsistency_count "
            "+ 0.75*expected_rewrite_field_count + 0.40*candidate_filter_reason_count "
            "- 0.25*anchor_or_previous_compatible_bonus"
        ),
    }


def build_comparison(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v43_trace_dir: str | Path = V43_TRACE_DIR,
    v45_audit_json: str | Path = V45_AUDIT_JSON,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    window_steps: int = 120,
) -> dict[str, Any]:
    v45_audit = _load_json(v45_audit_json)
    stable_scenarios = _discover_stable_scenarios(original_trace_dir, failure_scenarios)
    main_reports = [
        _scenario_report(scenario, trace_dir=original_trace_dir, window_steps=window_steps, mode="main_failure_pre_window")
        for scenario in failure_scenarios
    ]
    stable_reports = [
        _scenario_report(scenario, trace_dir=original_trace_dir, window_steps=window_steps, mode="stable_control_pre_window")
        for scenario in stable_scenarios
    ]
    v43_reports = []
    pending = []
    for scenario in failure_scenarios:
        trace = _trace_path(v43_trace_dir, scenario)
        if trace.exists():
            v43_reports.append(_scenario_report(scenario, trace_dir=v43_trace_dir, window_steps=window_steps, mode="v43_existing_trace"))
        else:
            pending.append(scenario)
    main_aggregate = _aggregate_reports(main_reports)
    stable_aggregate = _aggregate_reports(stable_reports)
    v43_aggregate = _aggregate_reports(v43_reports)
    accepted = _accepted(main_aggregate, stable_aggregate)
    return {
        "schema_version": "candidate_guardrail_shadow_scoring_comparison_20260531_v46",
        "mainline_alignment": {
            "impact_layer": "Evaluation / Safety Boundary / Metadata Trace",
            "default_llm_rspc_v2_changed": False,
            "mode": "offline_shadow_candidate_scoring",
        },
        "source_v45_schema": v45_audit.get("schema_version", ""),
        "source_v45_top_issue_type": v45_audit.get("top_issue_type", ""),
        "online_llm_called": False,
        "new_rollout_run": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "failure_scenarios": list(failure_scenarios),
        "stable_control_scenarios": stable_scenarios,
        "pending_v43_trace_scenarios": pending,
        "main_failure_reports": main_reports,
        "stable_control_reports": stable_reports,
        "v43_existing_trace_reports": v43_reports,
        "main_failure_aggregate": main_aggregate,
        "stable_control_aggregate": stable_aggregate,
        "v43_existing_trace_aggregate": v43_aggregate,
        "acceptance": {
            "shadow_scoring_acceptance_pass": accepted,
            "candidate_guardrail_issue_steps_shadow_lt_original": main_aggregate.get("shadow", {}).get("candidate_guardrail_issue_steps", 0)
            < main_aggregate.get("original", {}).get("candidate_guardrail_issue_steps", 0),
            "profile_action_conflict_steps_shadow_lt_original": main_aggregate.get("shadow", {}).get("profile_action_conflict_steps", 0)
            < main_aggregate.get("original", {}).get("profile_action_conflict_steps", 0),
            "major_or_hard_rewrite_predicted_steps_shadow_lte_original": main_aggregate.get("shadow", {}).get("major_or_hard_rewrite_predicted_steps", 0)
            <= main_aggregate.get("original", {}).get("major_or_hard_rewrite_predicted_steps", 0),
            "target_rh_vs_high_vent_shadow_lt_original": main_aggregate.get("shadow", {}).get("target_rh_vs_high_vent_steps", 0)
            < main_aggregate.get("original", {}).get("target_rh_vs_high_vent_steps", 0),
            "dry_risk_vs_high_vent_shadow_lt_original": main_aggregate.get("shadow", {}).get("dry_risk_vs_high_vent_steps", 0)
            < main_aggregate.get("original", {}).get("dry_risk_vs_high_vent_steps", 0),
            "stable_shadow_change_rate_lte_0_35": stable_aggregate.get("shadow_change_rate", 1.0) <= 0.35,
            "stable_hard_safety_rewrite_preferred_new_zero": stable_aggregate.get("hard_safety_rewrite_preferred_new_steps", 0) == 0,
        },
        "next_action": "minimal_shadow_rollout_with_candidate_guardrail_scoring_plan"
        if accepted
        else "profile_trajectory_feasibility_fix_plan",
    }


def build_readiness(comparison: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "metadata_replay_readiness_checklist_20260531_v46",
        "stage": "v46_shadow_candidate_guardrail_scoring",
        "shadow_scoring_complete": True,
        "shadow_scoring_acceptance_pass": bool((comparison.get("acceptance", {}) or {}).get("shadow_scoring_acceptance_pass", False)),
        "default_llm_rspc_v2_changed": False,
        "online_llm_called": False,
        "new_rollout_run": False,
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "stable_shadow_change_rate": float((comparison.get("stable_control_aggregate", {}) or {}).get("shadow_change_rate", 0.0)),
        "stable_hard_safety_rewrite_preferred_new_steps": int(
            (comparison.get("stable_control_aggregate", {}) or {}).get("hard_safety_rewrite_preferred_new_steps", 0)
        ),
        "pending_v43_trace_scenarios": list(comparison.get("pending_v43_trace_scenarios", [])),
        "patch_execution_authorized": False,
        "next_action": comparison.get("next_action", "profile_trajectory_feasibility_fix_plan"),
    }


def _write_json_md(path_json: Path, path_md: Path, data: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    lines = [f"# {title}", ""]
    for key in (
        "schema_version",
        "stage",
        "design_status",
        "source_v45_top_issue_type",
        "scoring_direction",
        "next_action",
    ):
        if key in data:
            lines.append(f"- `{key}`: `{data[key]}`")
    if "acceptance" in data:
        lines.append(f"- `shadow_scoring_acceptance_pass`: `{data['acceptance'].get('shadow_scoring_acceptance_pass')}`")
    for key in (
        "online_llm_called",
        "new_rollout_run",
        "controlled_replay_allowed",
        "controlled_replay_execution_allowed",
        "metadata_replay_execution_allowed",
        "performance_claim_allowed",
        "promotion_evidence",
    ):
        if key in data:
            lines.append(f"- `{key}`: `{str(data[key]).lower()}`")
    for key in ("main_failure_aggregate", "stable_control_aggregate", "v43_existing_trace_aggregate", "acceptance"):
        if key in data:
            lines.extend(["", f"## {key}", "```json", json.dumps(data[key], indent=2, sort_keys=True), "```"])
    if "weights" in data:
        lines.extend(["", "## weights", "```json", json.dumps(data["weights"], indent=2, sort_keys=True), "```"])
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v43_trace_dir: str | Path = V43_TRACE_DIR,
    v45_audit_json: str | Path = V45_AUDIT_JSON,
    v45_design_json: str | Path = V45_DESIGN_JSON,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    window_steps: int = 120,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    patch = build_patch_design(v45_design_json=v45_design_json)
    comparison = build_comparison(
        original_trace_dir=original_trace_dir,
        v43_trace_dir=v43_trace_dir,
        v45_audit_json=v45_audit_json,
        failure_scenarios=failure_scenarios,
        window_steps=window_steps,
    )
    readiness = build_readiness(comparison, patch)
    _write_json_md(PATCH_JSON, PATCH_MD, patch, "v46 Shadow Candidate-Guardrail Scoring Patch")
    _write_json_md(COMPARISON_JSON, COMPARISON_MD, comparison, "v46 Shadow Candidate-Guardrail Scoring Comparison")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v46 Metadata Replay Readiness")
    return patch, comparison, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-trace-dir", type=str, default=str(ORIGINAL_TRACE_DIR))
    parser.add_argument("--v43-trace-dir", type=str, default=str(V43_TRACE_DIR))
    parser.add_argument("--v45-audit-json", type=str, default=str(V45_AUDIT_JSON))
    parser.add_argument("--v45-design-json", type=str, default=str(V45_DESIGN_JSON))
    parser.add_argument("--failure-scenario", action="append", default=[])
    parser.add_argument("--window-steps", type=int, default=120)
    parser.add_argument("--write-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios = args.failure_scenario or list(FAILURE_SCENARIOS)
    patch, comparison, readiness = write_all(
        original_trace_dir=args.original_trace_dir,
        v43_trace_dir=args.v43_trace_dir,
        v45_audit_json=args.v45_audit_json,
        v45_design_json=args.v45_design_json,
        failure_scenarios=scenarios,
        window_steps=int(args.window_steps),
    )
    if not args.write_all:
        print(json.dumps({"patch": patch, "comparison": comparison, "readiness": readiness}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
