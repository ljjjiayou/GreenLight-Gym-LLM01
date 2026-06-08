"""Build v67 normal-path profile-driven candidate arbitration artifacts.

This stage is offline-only. It reads existing v64/v65/v66 traces and audits to
separate three questions that were previously mixed together:

* did qwen3.7-plus structured anchors reach the profile generator?
* did the profile generator produce feasible profile trajectories?
* did those profiles become executable action candidates for final arbitration?

It does not run rollout, call online LLMs, change llm_rspc_v2, authorize
metadata replay, authorize controlled replay, or make performance claims.
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

import gl_gym.experiments.qwen37plus_profile_candidate_guardrail_reconciliation_v66 as v66  # noqa: E402
import gl_gym.experiments.qwen37plus_structured_bridge_runtime_failure_attribution_v65 as v65  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import AUDIT_DIR  # noqa: E402


MODEL_NAME = "qwen3.7-plus"
ARTIFACT_DATE = "20260603"
FAILURE_SCENARIOS = tuple(v65.FAILURE_SCENARIOS)
WINDOW_STEPS = 120
V64_TRACE_DIR = v65.V64_TRACE_DIR
V65_ATTRIBUTION_JSON = v65.ATTRIBUTION_JSON
V66_RECONCILIATION_JSON = v66.RECONCILIATION_JSON

ARBITRATION_JSON = AUDIT_DIR / "qwen37plus_normal_path_profile_candidate_arbitration_audit_20260603_v67.json"
ARBITRATION_MD = AUDIT_DIR / "qwen37plus_normal_path_profile_candidate_arbitration_audit_20260603_v67.md"
GAP_JSON = AUDIT_DIR / "qwen37plus_profile_to_action_candidate_gap_audit_20260603_v67.json"
GAP_MD = AUDIT_DIR / "qwen37plus_profile_to_action_candidate_gap_audit_20260603_v67.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260603_v67.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260603_v67.md"

REQUIRED_METADATA_FIELDS = (
    "step",
    "runtime_error_type",
    "structured_anchor_valid",
    "structured_anchor_profile_bridge_attempted",
    "structured_anchor_profile_bridge_bridgeable",
    "structured_anchor_profile_bridge_profile_candidate_count",
    "structured_anchor_profile_bridge_selected_shadow_profile",
    "structured_anchor_profile_bridge_low_level_action_generated",
    "profile_generator_candidate_count",
    "profile_candidate_names",
    "profile_rspc_shadow_candidate_count",
    "profile_rspc_shadow_eligible_candidate_count",
    "profile_rspc_shadow_best_eligible",
    "profile_rspc_shadow_best_profile",
    "candidate_selection_source",
    "selected_fallback_candidate",
    "rspc_action_candidates_json",
    "rspc_action_candidate_count",
    "rspc_action_actual_candidate_count",
    "rspc_action_selected_name",
    "tomato_safety_v2_applied",
    "final_action_would_fail_canopy_boundary_v2",
    "target_temp",
    "target_rh",
    "target_co2",
    "temp_air",
    "rh_air",
    "vpd_air",
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
)

CATEGORY_PRIORITY = (
    "metadata_missing",
    "structured_anchor_missing",
    "profile_candidate_missing",
    "profile_to_action_candidate_missing",
    "candidate_composer_missing",
    "profile_candidate_not_executable",
    "no_safe_compatible_candidate_available",
    "safety_shield_only_repair",
    "normal_path_candidate_available",
    "fallback_should_not_have_been_primary",
)

PROFILE_HINT_KEYS = (
    "name",
    "source",
    "family",
    "origin",
    "candidate_source",
    "profile_name",
    "profile",
    "generated_by",
    "composer",
)


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


def _split_names(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


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


def _candidate_json_items(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    decoded = _parse_json(row.get("rspc_action_candidates_json"))
    if not isinstance(decoded, list):
        return []
    return [item for item in decoded if isinstance(item, dict)]


def _missing_metadata(row: Mapping[str, Any]) -> list[str]:
    return [field for field in REQUIRED_METADATA_FIELDS if field not in row]


def _profile_names(row: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for field in (
        "profile_candidate_names",
        "profile_rspc_shadow_top_candidates",
        "profile_rspc_shadow_raw_top_candidates",
        "profile_scorer_ranked_candidates",
        "structured_anchor_profile_bridge_requested_shapes",
    ):
        names.extend(_split_names(row.get(field)))
    for field in (
        "structured_anchor_profile_bridge_selected_shadow_profile",
        "profile_generator_selected_shadow_profile",
        "profile_rspc_shadow_best_profile",
        "profile_rspc_shadow_raw_best_profile",
        "profile_scorer_selected_shadow_profile",
    ):
        value = str(row.get(field, "") or "").strip()
        if value:
            names.append(value)
    return sorted({name for name in names if name})


def _profile_candidate_count(row: Mapping[str, Any]) -> int:
    counts = [
        int(_num(row.get("structured_anchor_profile_bridge_profile_candidate_count"), 0)),
        int(_num(row.get("profile_generator_candidate_count"), 0)),
        len(_split_names(row.get("profile_candidate_names"))),
    ]
    return max(counts)


def _structured_anchor_valid(row: Mapping[str, Any]) -> bool:
    return bool(
        _truthy(row.get("structured_anchor_valid"))
        or _truthy(row.get("structured_anchor_profile_bridge_valid_structured_anchor"))
    )


def _profile_bridge_attempted(row: Mapping[str, Any]) -> bool:
    return bool(
        _truthy(row.get("structured_anchor_profile_bridge_attempted"))
        or _truthy(row.get("profile_generator_shadow_only"))
    )


def _profile_bridgeable(row: Mapping[str, Any]) -> bool:
    return bool(
        _truthy(row.get("structured_anchor_profile_bridge_bridgeable"))
        or _profile_candidate_count(row) > 0
    )


def _profile_shadow_candidate_count(row: Mapping[str, Any]) -> int:
    return max(
        int(_num(row.get("profile_rspc_shadow_candidate_count"), 0)),
        len(_split_names(row.get("profile_rspc_shadow_top_candidates"))),
        len(_split_names(row.get("profile_rspc_shadow_raw_top_candidates"))),
    )


def _profile_shadow_eligible_count(row: Mapping[str, Any]) -> int:
    return int(_num(row.get("profile_rspc_shadow_eligible_candidate_count"), 0))


def _low_level_action_generated(row: Mapping[str, Any]) -> bool:
    return bool(_truthy(row.get("structured_anchor_profile_bridge_low_level_action_generated")))


def _candidate_action(candidate: Mapping[str, Any]) -> dict[str, float] | None:
    post_shape = candidate.get("post_shape_action")
    action_map = post_shape if isinstance(post_shape, Mapping) else candidate.get("action", {})
    if not isinstance(action_map, Mapping):
        return None
    return v66._action_from_mapping(action_map)


def _candidate_is_profile_derived(candidate: Mapping[str, Any], profile_names: Sequence[str]) -> bool:
    profile_name_set = {str(name).strip().lower() for name in profile_names if str(name).strip()}
    values = [str(candidate.get(key, "") or "") for key in PROFILE_HINT_KEYS]
    lowered = " ".join(values).lower()
    if "profile" in lowered or "structured_anchor_profile" in lowered or "profile_generator" in lowered:
        return True
    name = str(candidate.get("name", "") or "").strip().lower()
    if name and name in profile_name_set:
        return True
    for key in ("profile_name", "profile", "source_profile", "selected_shadow_profile"):
        value = str(candidate.get(key, "") or "").strip().lower()
        if value and value in profile_name_set:
            return True
    return False


def _profile_derived_action_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = _profile_names(row)
    candidates: list[dict[str, Any]] = []
    for item in _candidate_json_items(row):
        if not _candidate_is_profile_derived(item, names):
            continue
        action = _candidate_action(item)
        if action is None:
            continue
        candidates.append(
            {
                "name": str(item.get("name") or "profile_candidate"),
                "source": str(item.get("source") or item.get("candidate_source") or "profile_derived_candidate"),
                "action": action,
                "original_selected": bool(item.get("selected", False)),
                "base_score": float(v66._candidate_score(item)),
                "prediction": dict(v66._candidate_prediction(item)),
            }
        )
    return candidates


def _fallback_primary(row: Mapping[str, Any]) -> bool:
    selection_source = str(row.get("candidate_selection_source", "") or "").strip().lower()
    selected_fallback = str(row.get("selected_fallback_candidate", "") or "").strip()
    source = str(row.get("source", "") or "").strip().lower()
    return bool(selection_source == "fallback_candidate" or (selected_fallback and source in {"fallback", "anchor", "runtime_error"}))


def _safety_shield_applied(row: Mapping[str, Any]) -> bool:
    return bool(
        _truthy(row.get("tomato_safety_v2_applied"))
        or _truthy(row.get("final_action_would_fail_canopy_boundary_v2"))
        or _num(row.get("unknown_post_guardrail_rewrite_count"), 0) > 0
    )


def _primary_category(categories: Sequence[str]) -> str:
    category_set = set(categories)
    for category in CATEGORY_PRIORITY:
        if category in category_set:
            return category
    return categories[0] if categories else "metadata_missing"


def classify_step(row: Mapping[str, Any], *, previous_action: Mapping[str, float] | None = None) -> dict[str, Any]:
    missing_fields = _missing_metadata(row)
    structured_valid = _structured_anchor_valid(row)
    profile_count = _profile_candidate_count(row)
    profile_names = _profile_names(row)
    bridge_attempted = _profile_bridge_attempted(row)
    bridgeable = _profile_bridgeable(row)
    profile_shadow_count = _profile_shadow_candidate_count(row)
    profile_shadow_eligible = _profile_shadow_eligible_count(row)
    low_level_action = _low_level_action_generated(row)
    profile_action_candidates = _profile_derived_action_candidates(row)
    profile_action_evaluations = [
        v66.evaluate_candidate(row, candidate, previous_action=previous_action)
        for candidate in profile_action_candidates
    ]
    profile_compatible = [candidate for candidate in profile_action_evaluations if candidate.get("compatible")]
    reconciliation = v66.reconcile_row(row, previous_action=previous_action)
    existing_compatible = int(reconciliation.get("compatible_candidate_count", 0) or 0)
    fallback_primary = _fallback_primary(row)
    safety_shield = _safety_shield_applied(row)

    categories: list[str] = []
    if missing_fields:
        categories.append("metadata_missing")
    if not structured_valid:
        categories.append("structured_anchor_missing")
    elif profile_count <= 0:
        categories.append("profile_candidate_missing")
    else:
        if len(profile_action_candidates) > 0:
            categories.append("normal_path_candidate_available")
        if not low_level_action and len(profile_action_candidates) == 0:
            categories.extend(
                [
                    "profile_to_action_candidate_missing",
                    "candidate_composer_missing",
                    "profile_candidate_not_executable",
                ]
            )
    if fallback_primary and (profile_count > 0 or len(profile_action_candidates) > 0):
        categories.append("fallback_should_not_have_been_primary")
    if safety_shield and len(profile_compatible) == 0:
        categories.append("safety_shield_only_repair")
    if existing_compatible <= 0 and len(profile_compatible) == 0:
        categories.append("no_safe_compatible_candidate_available")
    if not categories:
        categories.append("metadata_missing")

    return {
        "step": _step(row),
        "primary_category": _primary_category(categories),
        "categories": sorted(set(categories), key=categories.index),
        "metadata_missing_fields": missing_fields,
        "structured_anchor_valid": bool(structured_valid),
        "profile_bridge_attempted": bool(bridge_attempted),
        "profile_bridgeable": bool(bridgeable),
        "profile_candidate_count": int(profile_count),
        "profile_names": profile_names,
        "profile_rspc_shadow_candidate_count": int(profile_shadow_count),
        "profile_rspc_shadow_eligible_candidate_count": int(profile_shadow_eligible),
        "profile_rspc_shadow_best_eligible": bool(_truthy(row.get("profile_rspc_shadow_best_eligible"))),
        "profile_rspc_shadow_best_profile": str(row.get("profile_rspc_shadow_best_profile", "") or ""),
        "profile_low_level_action_generated": bool(low_level_action),
        "normal_path_profile_action_candidate_count": int(len(profile_action_candidates)),
        "normal_path_profile_compatible_candidate_count": int(len(profile_compatible)),
        "normal_path_profile_top_candidates": sorted(
            profile_action_evaluations,
            key=lambda item: float(item.get("reconciled_score", 0.0)),
        )[:5],
        "existing_executable_candidate_count": int(reconciliation.get("executable_candidate_count", 0) or 0),
        "existing_compatible_candidate_count": int(existing_compatible),
        "fallback_primary": bool(fallback_primary),
        "candidate_selection_source": str(row.get("candidate_selection_source", "") or ""),
        "selected_fallback_candidate": str(row.get("selected_fallback_candidate", "") or ""),
        "source": str(row.get("source", "") or ""),
        "rspc_action_selected_name": str(row.get("rspc_action_selected_name", "") or ""),
        "rspc_action_post_shape_best_name": str(row.get("rspc_action_post_shape_best_name", "") or ""),
        "tomato_safety_v2_applied": bool(_truthy(row.get("tomato_safety_v2_applied"))),
        "final_action_would_fail_canopy_boundary_v2": bool(_truthy(row.get("final_action_would_fail_canopy_boundary_v2"))),
    }


def _failure_window(trace_dir: str | Path, scenario_id: str, window_steps: int) -> tuple[Path, list[dict[str, str]]]:
    trace = _trace_path(trace_dir, scenario_id)
    rows = _read_rows(trace)
    failure = v65._runtime_failure_row(rows)
    center = v65._step(failure, len(rows) - 1 if rows else -1) if failure is not None else None
    return trace, v65._window(rows, center_step=center, window_steps=window_steps)


def _count_step_categories(step_reports: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for report in step_reports:
        for category in report.get("categories", []) or []:
            counts[str(category)] += 1
    return counts


def _counter_field(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(field, "") or "").strip()
        counts[value or "missing"] += 1
    return dict(sorted(counts.items()))


def _scenario_report(scenario_id: str, *, trace_dir: str | Path, window_steps: int) -> dict[str, Any]:
    trace, rows = _failure_window(trace_dir, scenario_id, window_steps)
    previous_action: dict[str, float] | None = None
    step_reports: list[dict[str, Any]] = []
    for row in rows:
        report = classify_step(row, previous_action=previous_action)
        step_reports.append(report)
        previous_action = v66._action_from_row(row)
    category_counts = _count_step_categories(step_reports)
    return {
        "scenario_id": scenario_id,
        "trace_path": _rel(trace),
        "trace_exists": bool(trace.exists()),
        "window_row_count": int(len(rows)),
        "window_start_step": _step(rows[0], 0) if rows else None,
        "window_end_step": _step(rows[-1], 0) if rows else None,
        "category_counts": dict(sorted(category_counts.items())),
        "primary_category_counts": dict(sorted(Counter(row["primary_category"] for row in step_reports).items())),
        "structured_anchor_valid_steps": int(sum(bool(row["structured_anchor_valid"]) for row in step_reports)),
        "profile_candidate_available_steps": int(sum(int(row["profile_candidate_count"]) > 0 for row in step_reports)),
        "profile_rspc_shadow_eligible_steps": int(
            sum(int(row["profile_rspc_shadow_eligible_candidate_count"]) > 0 for row in step_reports)
        ),
        "profile_low_level_action_generated_steps": int(
            sum(bool(row["profile_low_level_action_generated"]) for row in step_reports)
        ),
        "normal_path_profile_action_candidate_steps": int(
            sum(int(row["normal_path_profile_action_candidate_count"]) > 0 for row in step_reports)
        ),
        "normal_path_profile_compatible_candidate_steps": int(
            sum(int(row["normal_path_profile_compatible_candidate_count"]) > 0 for row in step_reports)
        ),
        "existing_compatible_candidate_steps": int(
            sum(int(row["existing_compatible_candidate_count"]) > 0 for row in step_reports)
        ),
        "fallback_primary_steps": int(sum(bool(row["fallback_primary"]) for row in step_reports)),
        "candidate_selection_source_counts": _counter_field(rows, "candidate_selection_source"),
        "selected_fallback_candidate_counts": _counter_field(rows, "selected_fallback_candidate"),
        "rspc_action_selected_name_counts": _counter_field(rows, "rspc_action_selected_name"),
        "step_classifications": step_reports,
    }


def _sum_reports(reports: Sequence[Mapping[str, Any]], key: str) -> int:
    return int(sum(int(report.get(key, 0) or 0) for report in reports))


def _aggregate_category_counts(reports: Sequence[Mapping[str, Any]], *, field: str = "category_counts") -> dict[str, int]:
    counts: Counter[str] = Counter()
    for report in reports:
        for key, value in (report.get(field, {}) or {}).items():
            counts[str(key)] += int(value)
    return dict(sorted(counts.items()))


def _merge_counts(reports: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for report in reports:
        for item, value in (report.get(key, {}) or {}).items():
            counts[str(item)] += int(value)
    return dict(sorted(counts.items()))


def _next_action(row_count: int, aggregate: Mapping[str, int]) -> str:
    metadata_missing = int(aggregate.get("metadata_missing", 0) or 0)
    if metadata_missing:
        return "normal_path_metadata_instrumentation_patch_plan"
    normal_compatible = int(aggregate.get("normal_path_profile_compatible_candidate_steps", 0) or 0)
    if row_count and normal_compatible > row_count // 2:
        return "minimal_normal_path_profile_arbitration_shadow_patch_plan"
    profile_available = int(aggregate.get("profile_candidate_available_steps", 0) or 0)
    profile_low_level = int(aggregate.get("profile_low_level_action_generated_steps", 0) or 0)
    normal_profile_action = int(aggregate.get("normal_path_profile_action_candidate_steps", 0) or 0)
    profile_eligible = int(aggregate.get("profile_rspc_shadow_eligible_steps", 0) or 0)
    if row_count and profile_available > row_count // 2 and profile_low_level == 0 and normal_profile_action == 0:
        return "profile_to_action_candidate_composer_design_plan"
    if row_count and profile_available > row_count // 2 and profile_eligible <= row_count // 2:
        return "profile_generator_feasibility_template_repair_plan"
    return "normal_path_profile_arbitration_instrumentation_patch_plan"


def build_normal_path_arbitration_audit(
    *,
    trace_dir: str | Path = V64_TRACE_DIR,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    window_steps: int = WINDOW_STEPS,
    v65_attribution_json: str | Path = V65_ATTRIBUTION_JSON,
    v66_reconciliation_json: str | Path = V66_RECONCILIATION_JSON,
) -> dict[str, Any]:
    scenario_reports = [
        _scenario_report(scenario, trace_dir=trace_dir, window_steps=window_steps)
        for scenario in failure_scenarios
    ]
    row_count = _sum_reports(scenario_reports, "window_row_count")
    category_counts = _aggregate_category_counts(scenario_reports)
    primary_counts = _aggregate_category_counts(scenario_reports, field="primary_category_counts")
    aggregate = {
        "row_count": int(row_count),
        "structured_anchor_valid_steps": _sum_reports(scenario_reports, "structured_anchor_valid_steps"),
        "profile_candidate_available_steps": _sum_reports(scenario_reports, "profile_candidate_available_steps"),
        "profile_rspc_shadow_eligible_steps": _sum_reports(scenario_reports, "profile_rspc_shadow_eligible_steps"),
        "profile_low_level_action_generated_steps": _sum_reports(scenario_reports, "profile_low_level_action_generated_steps"),
        "normal_path_profile_action_candidate_steps": _sum_reports(scenario_reports, "normal_path_profile_action_candidate_steps"),
        "normal_path_profile_compatible_candidate_steps": _sum_reports(
            scenario_reports,
            "normal_path_profile_compatible_candidate_steps",
        ),
        "existing_compatible_candidate_steps": _sum_reports(scenario_reports, "existing_compatible_candidate_steps"),
        "fallback_primary_steps": _sum_reports(scenario_reports, "fallback_primary_steps"),
    }
    next_action = _next_action(row_count, {**category_counts, **aggregate})
    v65_attribution = _load_json(v65_attribution_json)
    v66_reconciliation = _load_json(v66_reconciliation_json)
    acceptance = {
        "metadata_sufficient_for_offline_audit": bool(category_counts.get("metadata_missing", 0) == 0),
        "structured_anchor_reaches_profile_generator_majority": bool(
            row_count and aggregate["profile_candidate_available_steps"] > row_count // 2
        ),
        "profile_generator_has_shadow_eligible_profiles_majority": bool(
            row_count and aggregate["profile_rspc_shadow_eligible_steps"] > row_count // 2
        ),
        "profile_to_action_candidate_composer_present": bool(
            aggregate["profile_low_level_action_generated_steps"] > 0
            or aggregate["normal_path_profile_action_candidate_steps"] > 0
        ),
        "majority_normal_path_profile_compatible_candidate_available": bool(
            row_count and aggregate["normal_path_profile_compatible_candidate_steps"] > row_count // 2
        ),
        "fallback_primary_despite_profile_path_majority": bool(
            row_count and category_counts.get("fallback_should_not_have_been_primary", 0) > row_count // 2
        ),
    }
    return {
        "artifact": "qwen37plus_normal_path_profile_candidate_arbitration_audit_20260603_v67",
        "current_stage": "v67_normal_path_profile_candidate_arbitration_audit",
        "model_name": MODEL_NAME,
        "scope": "offline_audit_only_existing_v64_v65_v66_artifacts",
        "controller": "llm_rspc_v2",
        "failure_scenarios": list(failure_scenarios),
        "window_steps": int(window_steps),
        "source_v65_artifact": v65_attribution.get("artifact", ""),
        "source_v65_dominant_attribution": v65_attribution.get("dominant_attribution", ""),
        "source_v66_artifact": v66_reconciliation.get("artifact", ""),
        "source_v66_compatible_candidate_found_steps": int(v66_reconciliation.get("compatible_candidate_found_steps", 0) or 0),
        "category_counts": category_counts,
        "primary_category_counts": primary_counts,
        "aggregate": aggregate,
        "candidate_selection_source_counts": _merge_counts(scenario_reports, "candidate_selection_source_counts"),
        "selected_fallback_candidate_counts": _merge_counts(scenario_reports, "selected_fallback_candidate_counts"),
        "rspc_action_selected_name_counts": _merge_counts(scenario_reports, "rspc_action_selected_name_counts"),
        "question_results": {
            "structured_anchor_enters_profile_generator": {
                "answer": "yes" if acceptance["structured_anchor_reaches_profile_generator_majority"] else "no_or_insufficient_metadata",
                "structured_anchor_valid_steps": aggregate["structured_anchor_valid_steps"],
                "profile_candidate_available_steps": aggregate["profile_candidate_available_steps"],
            },
            "profile_generator_produces_profile_trajectory": {
                "answer": "yes_shadow_profile_candidates"
                if acceptance["profile_generator_has_shadow_eligible_profiles_majority"]
                else "no_or_not_feasible",
                "profile_rspc_shadow_eligible_steps": aggregate["profile_rspc_shadow_eligible_steps"],
            },
            "profile_trajectory_generates_low_level_action_candidate": {
                "answer": "yes"
                if acceptance["profile_to_action_candidate_composer_present"]
                else "no_profile_to_action_candidate_missing",
                "profile_low_level_action_generated_steps": aggregate["profile_low_level_action_generated_steps"],
                "normal_path_profile_action_candidate_steps": aggregate["normal_path_profile_action_candidate_steps"],
            },
            "normal_path_candidates_can_replace_fallback_primary": {
                "answer": "not_yet_profile_candidates_are_not_executable"
                if not acceptance["profile_to_action_candidate_composer_present"]
                else (
                    "yes_majority_compatible_profile_action_candidates"
                    if acceptance["majority_normal_path_profile_compatible_candidate_available"]
                    else "requires_profile_action_compatibility_scoring"
                ),
                "fallback_primary_steps": aggregate["fallback_primary_steps"],
                "existing_compatible_candidate_steps": aggregate["existing_compatible_candidate_steps"],
                "normal_path_profile_compatible_candidate_steps": aggregate["normal_path_profile_compatible_candidate_steps"],
            },
        },
        "acceptance": acceptance,
        "normal_path_profile_arbitration_audit_complete": True,
        "online_llm_called": False,
        "new_rollout_run": False,
        "default_llm_rspc_v2_changed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": next_action,
        "scenario_reports": scenario_reports,
    }


def build_profile_to_action_gap_audit(arbitration: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = dict(arbitration.get("aggregate", {}) or {})
    category_counts = dict(arbitration.get("category_counts", {}) or {})
    row_count = int(aggregate.get("row_count", 0) or 0)
    profile_available = int(aggregate.get("profile_candidate_available_steps", 0) or 0)
    profile_eligible = int(aggregate.get("profile_rspc_shadow_eligible_steps", 0) or 0)
    low_level_steps = int(aggregate.get("profile_low_level_action_generated_steps", 0) or 0)
    profile_action_steps = int(aggregate.get("normal_path_profile_action_candidate_steps", 0) or 0)
    composer_missing = int(category_counts.get("candidate_composer_missing", 0) or 0)
    gap_confirmed = bool(profile_available > 0 and profile_eligible > 0 and low_level_steps == 0 and profile_action_steps == 0)
    dominant_gap = (
        "profile_to_action_candidate_composer_missing"
        if gap_confirmed
        else (
            "metadata_missing"
            if int(category_counts.get("metadata_missing", 0) or 0)
            else "profile_generator_feasibility_or_metadata_gap"
        )
    )
    scenario_gap_reports = []
    for report in arbitration.get("scenario_reports", []) or []:
        scenario_gap_reports.append(
            {
                "scenario_id": report.get("scenario_id"),
                "window_row_count": int(report.get("window_row_count", 0) or 0),
                "profile_candidate_available_steps": int(report.get("profile_candidate_available_steps", 0) or 0),
                "profile_rspc_shadow_eligible_steps": int(report.get("profile_rspc_shadow_eligible_steps", 0) or 0),
                "profile_low_level_action_generated_steps": int(report.get("profile_low_level_action_generated_steps", 0) or 0),
                "normal_path_profile_action_candidate_steps": int(
                    report.get("normal_path_profile_action_candidate_steps", 0) or 0
                ),
                "fallback_primary_steps": int(report.get("fallback_primary_steps", 0) or 0),
                "candidate_selection_source_counts": report.get("candidate_selection_source_counts", {}),
                "selected_fallback_candidate_counts": report.get("selected_fallback_candidate_counts", {}),
                "sample_gap_steps": [
                    {
                        "step": row.get("step"),
                        "categories": row.get("categories", []),
                        "profile_names": row.get("profile_names", []),
                        "profile_rspc_shadow_best_profile": row.get("profile_rspc_shadow_best_profile", ""),
                        "selected_fallback_candidate": row.get("selected_fallback_candidate", ""),
                        "rspc_action_selected_name": row.get("rspc_action_selected_name", ""),
                    }
                    for row in (report.get("step_classifications", []) or [])
                    if "profile_to_action_candidate_missing" in (row.get("categories", []) or [])
                ][:10],
            }
        )
    return {
        "artifact": "qwen37plus_profile_to_action_candidate_gap_audit_20260603_v67",
        "current_stage": "v67_profile_to_action_candidate_gap_audit",
        "model_name": MODEL_NAME,
        "scope": "offline_profile_generator_to_executable_action_candidate_gap_only",
        "row_count": int(row_count),
        "profile_candidate_available_steps": int(profile_available),
        "profile_rspc_shadow_eligible_steps": int(profile_eligible),
        "profile_low_level_action_generated_steps": int(low_level_steps),
        "normal_path_profile_action_candidate_steps": int(profile_action_steps),
        "candidate_composer_missing_steps": int(composer_missing),
        "profile_to_action_candidate_missing_steps": int(category_counts.get("profile_to_action_candidate_missing", 0) or 0),
        "fallback_should_not_have_been_primary_steps": int(
            category_counts.get("fallback_should_not_have_been_primary", 0) or 0
        ),
        "profile_to_action_gap_confirmed": bool(gap_confirmed),
        "dominant_gap": dominant_gap,
        "gap_taxonomy": sorted(
            {
                key
                for key in (
                    "structured_anchor_missing" if category_counts.get("structured_anchor_missing", 0) else "",
                    "profile_candidate_missing" if category_counts.get("profile_candidate_missing", 0) else "",
                    "profile_candidate_not_executable" if category_counts.get("profile_candidate_not_executable", 0) else "",
                    "profile_to_action_candidate_missing" if category_counts.get("profile_to_action_candidate_missing", 0) else "",
                    "candidate_composer_missing" if category_counts.get("candidate_composer_missing", 0) else "",
                    "fallback_should_not_have_been_primary" if category_counts.get("fallback_should_not_have_been_primary", 0) else "",
                    "metadata_missing" if category_counts.get("metadata_missing", 0) else "",
                )
                if key
            }
        ),
        "candidate_selection_source_counts": arbitration.get("candidate_selection_source_counts", {}),
        "selected_fallback_candidate_counts": arbitration.get("selected_fallback_candidate_counts", {}),
        "scenario_gap_reports": scenario_gap_reports,
        "candidate_composer_design_requirements": [
            "consume structured anchor profile bridge provenance without changing final action in v67",
            "convert selected/eligible profile trajectories into bounded low-level action candidates",
            "score profile-derived candidates against Tomato Safety compatibility, profile targets, and action continuity",
            "record candidate source, profile name, pre-shield action, post-shield action, score terms, and rejection reason",
            "keep Tomato Safety as the final hard shield and keep fallback only as exceptional fallback",
        ],
        "online_llm_called": False,
        "new_rollout_run": False,
        "default_llm_rspc_v2_changed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": arbitration.get("next_action", "profile_to_action_candidate_composer_design_plan"),
    }


def build_readiness(arbitration: Mapping[str, Any], gap: Mapping[str, Any]) -> dict[str, Any]:
    category_counts = dict(arbitration.get("category_counts", {}) or {})
    next_action = str(arbitration.get("next_action") or gap.get("next_action") or "profile_to_action_candidate_composer_design_plan")
    metadata_sufficient = bool((arbitration.get("acceptance", {}) or {}).get("metadata_sufficient_for_offline_audit", False))
    failure_taxonomy = sorted(
        {
            str(gap.get("dominant_gap") or ""),
            *[str(item) for item in gap.get("gap_taxonomy", []) or []],
        }
        - {""}
    )
    return {
        "artifact": "metadata_replay_readiness_checklist_20260603_v67",
        "current_stage": "v67_normal_path_profile_candidate_arbitration_audit",
        "model_name": MODEL_NAME,
        "normal_path_profile_arbitration_audit_complete": bool(
            arbitration.get("normal_path_profile_arbitration_audit_complete", False)
        ),
        "profile_to_action_gap_audit_complete": True,
        "metadata_sufficient_for_offline_audit": bool(metadata_sufficient),
        "profile_to_action_gap_confirmed": bool(gap.get("profile_to_action_gap_confirmed", False)),
        "dominant_gap": gap.get("dominant_gap", ""),
        "failure_taxonomy": failure_taxonomy,
        "category_counts": category_counts,
        "rollout_execution_authorized": False,
        "online_llm_called": False,
        "new_rollout_run": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": next_action,
        "blocked_until": [
            "profile-to-action candidate composer design"
            if next_action == "profile_to_action_candidate_composer_design_plan"
            else "normal-path metadata instrumentation"
            if next_action == "normal_path_metadata_instrumentation_patch_plan"
            else "minimal normal-path profile arbitration shadow patch design"
            if next_action == "minimal_normal_path_profile_arbitration_shadow_patch_plan"
            else "profile generator feasibility/template repair design"
        ],
    }


def _write_json_md(path_json: Path, path_md: Path, payload: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    lines = [
        f"# {title}",
        "",
        f"- Controlled replay allowed: `{payload.get('controlled_replay_allowed', False)}`",
        f"- Metadata replay execution allowed: `{payload.get('metadata_replay_execution_allowed', False)}`",
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
    arbitration = build_normal_path_arbitration_audit(
        trace_dir=trace_dir,
        failure_scenarios=failure_scenarios,
        window_steps=window_steps,
    )
    gap = build_profile_to_action_gap_audit(arbitration)
    readiness = build_readiness(arbitration, gap)
    _write_json_md(ARBITRATION_JSON, ARBITRATION_MD, arbitration, "v67 qwen3.7-plus Normal-Path Profile Candidate Arbitration Audit")
    _write_json_md(GAP_JSON, GAP_MD, gap, "v67 qwen3.7-plus Profile-to-Action Candidate Gap Audit")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v67 Metadata Replay Readiness")
    return {"arbitration": arbitration, "gap": gap, "readiness": readiness}


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
