"""Build v53 qwen3.7 profile-template failure attribution artifacts.

This pass is offline-only. It reads the v52 qwen3.7-max shadow traces and
previous v39/v48/v49/v50 artifacts, then explains why the profile-template
opt-in patch still produced hard-safety rewrite-preferred rows and persistent
candidate/guardrail pressure. It does not run rollout, call online LLMs,
mutate the default controller, authorize controlled replay, or make
performance claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (  # noqa: E402
    AUDIT_DIR,
    FAILURE_SCENARIOS,
    ORIGINAL_TRACE_DIR,
    _candidate_name,
    _parse_candidates,
    _read_rows,
    _runtime_failure_step,
    _trace_path,
    _truthy,
)
from gl_gym.experiments.diagnose_ppo_vs_llm import derive_candidate_guardrail_compatibility  # noqa: E402
from gl_gym.experiments.profile_feasibility_gate_shadow_rollout_v48 import V48_TRACE_DIR  # noqa: E402
from gl_gym.experiments.profile_template_opt_in_shadow_rollout_v52 import (  # noqa: E402
    MODEL_NAME,
    RESULT_AUDIT_JSON as V52_RESULT_AUDIT_JSON,
    V52_TRACE_DIR,
)


V49_RUNTIME_AUDIT_JSON = AUDIT_DIR / "profile_candidate_runtime_consistency_audit_20260601_v49.json"
V50_SHADOW_AUDIT_JSON = AUDIT_DIR / "profile_template_shadow_patch_audit_20260601_v50.json"

ATTRIBUTION_JSON = AUDIT_DIR / "qwen37_profile_template_patch_failure_attribution_20260601_v53.json"
ATTRIBUTION_MD = AUDIT_DIR / "qwen37_profile_template_patch_failure_attribution_20260601_v53.md"
LEAK_AUDIT_JSON = AUDIT_DIR / "qwen37_hard_safety_rewrite_preferred_leak_audit_20260601_v53.json"
LEAK_AUDIT_MD = AUDIT_DIR / "qwen37_hard_safety_rewrite_preferred_leak_audit_20260601_v53.md"
PRESSURE_DELTA_JSON = AUDIT_DIR / "qwen37_candidate_guardrail_pressure_delta_audit_20260601_v53.json"
PRESSURE_DELTA_MD = AUDIT_DIR / "qwen37_candidate_guardrail_pressure_delta_audit_20260601_v53.md"
REPAIR_DESIGN_JSON = AUDIT_DIR / "qwen37_profile_template_repair_design_20260601_v53.json"
REPAIR_DESIGN_MD = AUDIT_DIR / "qwen37_profile_template_repair_design_20260601_v53.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v53.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v53.md"

LEAK_CLASSIFICATIONS = (
    "fallback_veto_not_invoked_on_non_fallback_candidate",
    "veto_no_compatible_alternative",
    "template_repair_created_new_conflict",
    "safety_required_vent_misclassified",
    "candidate_metadata_missing",
    "planner_anchor_empty_contract_filled",
    "unknown_requires_instrumentation",
)


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
        return str(p).replace("\\", "/")


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


def _bool(value: Any) -> bool:
    return _truthy(value)


def _split_csv(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip() and part.strip() != "none"]


def _rows_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[int(_num(row.get("step"), -1))] = dict(row)
    return out


def _compat(row: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    merged.update(derive_candidate_guardrail_compatibility(dict(row)))
    return merged


def _compatibility_label(row: Mapping[str, Any]) -> str:
    label = str(row.get("candidate_guardrail_compatibility_label") or "").strip()
    if label:
        return label
    return str(_compat(row).get("candidate_guardrail_compatibility_label") or "compatible")


def _profile_conflict_label(row: Mapping[str, Any]) -> str:
    label = str(row.get("profile_action_conflict_label") or "").strip()
    if label:
        return label
    return str(_compat(row).get("profile_action_conflict_label") or "none")


def _is_hard_safety(row: Mapping[str, Any] | None) -> bool:
    if not row:
        return False
    return _compatibility_label(row) == "hard_safety_rewrite"


def _candidate_metadata_status(row: Mapping[str, Any]) -> str:
    candidates = _parse_candidates(row)
    actual_count = int(_num(row.get("rspc_action_actual_candidate_count") or row.get("rspc_action_candidate_count")))
    return "present" if candidates or actual_count > 0 else "candidate_metadata_missing"


def _selected_candidate_name(row: Mapping[str, Any]) -> str:
    for candidate in _parse_candidates(row):
        if _bool(candidate.get("selected")):
            return _candidate_name(candidate)
    return str(
        row.get("rspc_action_selected_name")
        or row.get("selected_fallback_candidate")
        or row.get("source")
        or "unknown"
    )


def _candidate_selection_source(row: Mapping[str, Any]) -> str:
    return str(row.get("candidate_selection_source") or _compat(row).get("candidate_selection_source") or row.get("source") or "unknown")


def _anchor_contract_proxy(row: Mapping[str, Any]) -> bool:
    """Detect steps where planning anchor/contract looks filled by anchor/rule fallback.

    The current traces do not expose a literal "empty planning anchor" field.
    The best available non-invasive proxy is a fallback/anchor-selected row with
    an anchor source such as hold_current/recent_anchor and no fallback veto.
    """

    source = str(row.get("source") or "").lower()
    anchor = str(row.get("anchor_source") or "").lower()
    selected = str(row.get("selected_fallback_candidate") or "").lower()
    candidate_source = _candidate_selection_source(row).lower()
    return bool(
        anchor in {"hold_current", "recent_anchor"}
        or selected in {"hold_current", "recent_anchor"}
        or "anchor" in source
        or "rule" in source
        or "fallback" in candidate_source
    )


def _fallback_veto_applied(row: Mapping[str, Any]) -> bool:
    return _bool(row.get("profile_template_patch_fallback_veto_applied"))


def _fallback_veto_no_alternative(row: Mapping[str, Any]) -> bool:
    return _bool(row.get("profile_template_patch_fallback_veto_no_alternative"))


def _safety_required_vent_misclassified(row: Mapping[str, Any]) -> bool:
    reasons = str(row.get("tomato_safety_v2_reasons") or row.get("rspc_action_post_shape_safety_gate_reason") or "").lower()
    conflict = _profile_conflict_label(row).lower()
    vent_required = _bool(row.get("profile_template_patch_vent_required_by_safety"))
    has_safety_vent_signal = any(
        token in reasons
        for token in (
            "hard_dry_vpd",
            "dry_vpd_guard",
            "hot_dry",
            "canopy_dew",
            "high_rad",
            "cooling_guard",
        )
    )
    return bool("dry_risk_vs_high_vent" in conflict and has_safety_vent_signal and not vent_required)


def _template_repair_created_new_conflict(original_row: Mapping[str, Any], v52_row: Mapping[str, Any]) -> bool:
    patch_applied = _bool(v52_row.get("profile_template_patch_applied"))
    corrections = _split_csv(v52_row.get("profile_template_patch_corrections"))
    original_conflict = _profile_conflict_label(original_row)
    v52_conflict = _profile_conflict_label(v52_row)
    return bool(patch_applied and (corrections or v52_conflict != original_conflict))


def _classify_leak(original_row: Mapping[str, Any], v52_row: Mapping[str, Any]) -> tuple[str, list[str]]:
    secondary: list[str] = []
    if _candidate_metadata_status(v52_row) == "candidate_metadata_missing":
        return "candidate_metadata_missing", secondary
    if _fallback_veto_no_alternative(v52_row):
        return "veto_no_compatible_alternative", secondary

    candidate_source = _candidate_selection_source(v52_row).lower()
    if _safety_required_vent_misclassified(v52_row):
        secondary.append("safety_required_vent_misclassified")
    if _template_repair_created_new_conflict(original_row, v52_row):
        secondary.append("template_repair_created_new_conflict")
    if _anchor_contract_proxy(v52_row):
        secondary.append("planner_anchor_empty_contract_filled")

    if not _fallback_veto_applied(v52_row) and "fallback" not in candidate_source:
        return "fallback_veto_not_invoked_on_non_fallback_candidate", secondary
    if _anchor_contract_proxy(v52_row):
        return "planner_anchor_empty_contract_filled", secondary
    if _safety_required_vent_misclassified(v52_row):
        return "safety_required_vent_misclassified", secondary
    if _template_repair_created_new_conflict(original_row, v52_row):
        return "template_repair_created_new_conflict", secondary
    return "unknown_requires_instrumentation", secondary


def _target_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "target_temp": _num(row.get("target_temp")),
        "target_rh": _num(row.get("target_rh")),
        "target_co2": _num(row.get("target_co2")),
        "repaired_target_temp": _num(row.get("profile_template_patch_repaired_target_temp"), None)
        if str(row.get("profile_template_patch_repaired_target_temp", "") or "").strip()
        else None,
        "repaired_target_rh": _num(row.get("profile_template_patch_repaired_target_rh"), None)
        if str(row.get("profile_template_patch_repaired_target_rh", "") or "").strip()
        else None,
        "repaired_target_co2": _num(row.get("profile_template_patch_repaired_target_co2"), None)
        if str(row.get("profile_template_patch_repaired_target_co2", "") or "").strip()
        else None,
    }


def _leak_entry(scenario: str, step: int, original_row: Mapping[str, Any], v52_row: Mapping[str, Any]) -> dict[str, Any]:
    classification, secondary = _classify_leak(original_row, v52_row)
    comp = _compat(v52_row)
    return {
        "scenario_id": scenario,
        "step": int(step),
        "model_name": str(v52_row.get("model_name") or ""),
        "source": str(v52_row.get("source") or ""),
        "anchor_source": str(v52_row.get("anchor_source") or ""),
        "anchor_contract_proxy": _anchor_contract_proxy(v52_row),
        "candidate_selection_source": _candidate_selection_source(v52_row),
        "selected_candidate": _selected_candidate_name(v52_row),
        "selected_fallback_candidate": str(v52_row.get("selected_fallback_candidate") or ""),
        "candidate_metadata_status": _candidate_metadata_status(v52_row),
        "candidate_count": int(_num(v52_row.get("rspc_action_actual_candidate_count") or v52_row.get("rspc_action_candidate_count"))),
        "targets": _target_snapshot(v52_row),
        "profile_template_patch": {
            "enabled": _bool(v52_row.get("profile_template_patch_enabled")),
            "applied": _bool(v52_row.get("profile_template_patch_applied")),
            "corrections": _split_csv(v52_row.get("profile_template_patch_corrections")),
            "forbidden_combinations": _split_csv(v52_row.get("profile_template_patch_forbidden_combinations")),
            "vent_required_by_safety": _bool(v52_row.get("profile_template_patch_vent_required_by_safety")),
            "humidity_retention_allowed": _bool(v52_row.get("profile_template_patch_humidity_retention_allowed")),
            "co2_enrichment_allowed": _bool(v52_row.get("profile_template_patch_co2_enrichment_allowed")),
            "fallback_veto_applied": _fallback_veto_applied(v52_row),
            "fallback_veto_no_alternative": _fallback_veto_no_alternative(v52_row),
            "fallback_veto_reason": str(v52_row.get("profile_template_patch_fallback_veto_reason") or ""),
        },
        "tomato_safety_reasons": str(v52_row.get("tomato_safety_v2_reasons") or ""),
        "rewrite_fields": _split_csv(comp.get("candidate_guardrail_rewrite_fields")),
        "profile_action_conflict_label": _profile_conflict_label(v52_row),
        "v39_profile_action_conflict_label": _profile_conflict_label(original_row),
        "v39_hard_safety_rewrite": _is_hard_safety(original_row),
        "v52_hard_safety_rewrite": _is_hard_safety(v52_row),
        "classification": classification,
        "secondary_signals": sorted(set(secondary)),
    }


def _scenario_report_from_result(result: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    for report in result.get("scenario_reports", []) or []:
        if report.get("scenario_id") == scenario:
            return dict(report)
    return {}


def build_hard_safety_leak_audit(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v52_trace_dir: str | Path = V52_TRACE_DIR,
    v52_result_json: str | Path = V52_RESULT_AUDIT_JSON,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    result = _load_json(v52_result_json)
    leak_entries: list[dict[str, Any]] = []
    resolved_entries: list[dict[str, Any]] = []
    scenario_reports: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    secondary_counts: Counter[str] = Counter()

    for scenario in scenarios:
        original_rows = _read_rows(_trace_path(original_trace_dir, scenario))
        v52_rows = _read_rows(_trace_path(v52_trace_dir, scenario))
        original_by_step = _rows_by_step(original_rows)
        v52_by_step = _rows_by_step(v52_rows)
        aligned_steps = sorted(set(original_by_step) & set(v52_by_step))
        exact_new_steps = []
        resolved_steps = []
        for step in aligned_steps:
            original_hard = _is_hard_safety(original_by_step[step])
            v52_hard = _is_hard_safety(v52_by_step[step])
            if v52_hard and not original_hard:
                entry = _leak_entry(scenario, step, original_by_step[step], v52_by_step[step])
                leak_entries.append(entry)
                exact_new_steps.append(step)
                classification_counts[entry["classification"]] += 1
                for item in entry.get("secondary_signals", []) or []:
                    secondary_counts[str(item)] += 1
            elif original_hard and not v52_hard:
                resolved_entries.append(
                    {
                        "scenario_id": scenario,
                        "step": int(step),
                        "v39_profile_action_conflict_label": _profile_conflict_label(original_by_step[step]),
                        "v52_profile_action_conflict_label": _profile_conflict_label(v52_by_step[step]),
                        "v52_selected_candidate": _selected_candidate_name(v52_by_step[step]),
                    }
                )
                resolved_steps.append(step)
        v52_report = _scenario_report_from_result(result, scenario)
        original_hard_count = int((v52_report.get("original", {}) or {}).get("hard_safety_rewrite_steps", 0) or 0)
        v52_hard_count = int((v52_report.get("v52", {}) or {}).get("hard_safety_rewrite_steps", 0) or 0)
        scenario_reports.append(
            {
                "scenario_id": scenario,
                "original_trace_exists": bool(original_rows),
                "v52_trace_exists": bool(v52_rows),
                "aligned_step_count": len(aligned_steps),
                "v39_hard_safety_rewrite_steps": original_hard_count,
                "v52_hard_safety_rewrite_steps": v52_hard_count,
                "aggregate_net_new_hard_safety_steps": int(v52_hard_count - original_hard_count),
                "step_aligned_new_hard_safety_steps": len(exact_new_steps),
                "step_aligned_resolved_original_hard_safety_steps": len(resolved_steps),
                "exact_new_steps": exact_new_steps,
                "resolved_original_hard_steps_sample": resolved_steps[:25],
            }
        )
    reported_net = int((result.get("totals", {}) or {}).get("hard_safety_rewrite_preferred_new_steps", 0) or 0)
    missing_categories = {name: 0 for name in LEAK_CLASSIFICATIONS if name not in classification_counts}
    classification_payload = {**missing_categories, **dict(sorted(classification_counts.items()))}
    return {
        "artifact": "qwen37_hard_safety_rewrite_preferred_leak_audit_20260601_v53",
        "scope": "offline_qwen37_v52_hard_safety_leak_attribution_only",
        "source_artifacts": {
            "v52_result_json": _rel(v52_result_json),
            "original_trace_dir": _rel(original_trace_dir),
            "v52_trace_dir": _rel(v52_trace_dir),
        },
        "model_name": MODEL_NAME,
        "scenarios": list(scenarios),
        "online_llm_called": False,
        "new_rollout_run": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "reported_aggregate_net_new_hard_safety_count_from_v52": reported_net,
        "step_aligned_new_hard_safety_leak_count": len(leak_entries),
        "step_aligned_resolved_original_hard_safety_count": len(resolved_entries),
        "counting_method_note": (
            "The v52 reported value is an aggregate net increase. The v53 leak entries "
            "are exact scenario/step rows where v52 is hard_safety_rewrite and v39 is not; "
            "therefore the exact row count can exceed the aggregate net count when v52 also "
            "resolves some original hard-safety rows."
        ),
        "classification_counts": classification_payload,
        "secondary_signal_counts": dict(sorted(secondary_counts.items())),
        "candidate_metadata_missing_count": int(classification_counts.get("candidate_metadata_missing", 0)),
        "scenario_reports": scenario_reports,
        "leak_entries": leak_entries,
        "resolved_original_hard_entries_sample": resolved_entries[:50],
    }


def _count_pressure(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "row_count": len(rows),
        "runtime_error_steps": sum(bool(str(row.get("runtime_error") or "").strip()) for row in rows),
        "hard_safety_rewrite_steps": sum(_is_hard_safety(row) for row in rows),
        "major_or_hard_rewrite_predicted_steps": sum(
            _compatibility_label(row) in {"major_rewrite", "hard_safety_rewrite"} for row in rows
        ),
        "profile_action_conflict_steps": sum(_profile_conflict_label(row) != "none" for row in rows),
        "candidate_guardrail_issue_steps": sum(
            _compatibility_label(row) in {"major_rewrite", "hard_safety_rewrite"}
            or _profile_conflict_label(row) != "none"
            for row in rows
        ),
        "profile_template_patch_applied_steps": sum(_bool(row.get("profile_template_patch_applied")) for row in rows),
        "profile_template_patch_applied_but_conflict_persisted_steps": sum(
            _bool(row.get("profile_template_patch_applied")) and _profile_conflict_label(row) != "none" for row in rows
        ),
        "fallback_veto_applied_steps": sum(_fallback_veto_applied(row) for row in rows),
        "fallback_veto_no_alternative_steps": sum(_fallback_veto_no_alternative(row) for row in rows),
        "candidate_metadata_missing_steps": sum(_candidate_metadata_status(row) != "present" for row in rows),
        "anchor_contract_proxy_steps": sum(_anchor_contract_proxy(row) for row in rows),
    }


def _counter(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "missing") for row in rows).items()))


def _window_by_result(rows: Sequence[Mapping[str, Any]], comparison_steps: int | None) -> list[dict[str, Any]]:
    if comparison_steps is None:
        return [dict(row) for row in rows]
    return [dict(row) for row in rows if int(_num(row.get("step"), -1)) < int(comparison_steps)]


def build_candidate_guardrail_pressure_delta_audit(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v48_trace_dir: str | Path = V48_TRACE_DIR,
    v52_trace_dir: str | Path = V52_TRACE_DIR,
    v52_result_json: str | Path = V52_RESULT_AUDIT_JSON,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    result = _load_json(v52_result_json)
    scenario_reports = []
    totals = Counter()
    for scenario in scenarios:
        report = _scenario_report_from_result(result, scenario)
        comparison_steps = int(report.get("comparison_window_steps") or 0) or None
        original_rows = _window_by_result(_read_rows(_trace_path(original_trace_dir, scenario)), comparison_steps)
        v48_rows = _window_by_result(_read_rows(_trace_path(v48_trace_dir, scenario)), comparison_steps)
        v52_rows = _window_by_result(_read_rows(_trace_path(v52_trace_dir, scenario)), comparison_steps)
        original_pressure = _count_pressure(original_rows)
        v48_pressure = _count_pressure(v48_rows)
        v52_pressure = _count_pressure(v52_rows)
        for key in ("profile_action_conflict_steps", "candidate_guardrail_issue_steps", "major_or_hard_rewrite_predicted_steps"):
            totals[f"v52_delta_vs_v39_{key}"] += v52_pressure[key] - original_pressure[key]
            totals[f"v52_delta_vs_v48_{key}"] += v52_pressure[key] - v48_pressure[key]
        scenario_reports.append(
            {
                "scenario_id": scenario,
                "comparison_window_steps": comparison_steps,
                "original": original_pressure,
                "v48": v48_pressure,
                "v52": v52_pressure,
                "v52_delta_vs_v39": {
                    key: v52_pressure[key] - original_pressure[key]
                    for key in (
                        "profile_action_conflict_steps",
                        "candidate_guardrail_issue_steps",
                        "major_or_hard_rewrite_predicted_steps",
                        "hard_safety_rewrite_steps",
                    )
                },
                "v52_delta_vs_v48": {
                    key: v52_pressure[key] - v48_pressure[key]
                    for key in (
                        "profile_action_conflict_steps",
                        "candidate_guardrail_issue_steps",
                        "major_or_hard_rewrite_predicted_steps",
                        "hard_safety_rewrite_steps",
                    )
                },
                "qwen37_runtime_first_error_step": _runtime_failure_step(v52_rows),
                "anchor_source_counts": _counter(v52_rows, "anchor_source"),
                "source_counts": _counter(v52_rows, "source"),
                "candidate_selection_source_counts": _counter(v52_rows, "candidate_selection_source"),
                "profile_conflict_counts": _counter(v52_rows, "profile_action_conflict_label"),
                "compatibility_label_counts": _counter(v52_rows, "candidate_guardrail_compatibility_label"),
            }
        )
    result_totals = result.get("totals", {}) or {}
    reported_artifact_deltas = {
        "v52_delta_vs_v48_candidate_guardrail_issue_steps": int(
            result_totals.get("v52_candidate_guardrail_issue_steps", 0) or 0
        )
        - int(result_totals.get("v48_candidate_guardrail_issue_steps", 0) or 0),
        "v52_delta_vs_v48_major_or_hard_rewrite_predicted_steps": int(
            result_totals.get("v52_major_or_hard_rewrite_predicted_steps", 0) or 0
        )
        - int(result_totals.get("v48_major_or_hard_rewrite_predicted_steps", 0) or 0),
        "v52_delta_vs_v48_profile_action_conflict_steps": int(
            result_totals.get("v52_profile_action_conflict_steps", 0) or 0
        )
        - int(result_totals.get("v48_profile_action_conflict_steps", 0) or 0),
    }
    return {
        "artifact": "qwen37_candidate_guardrail_pressure_delta_audit_20260601_v53",
        "scope": "offline_qwen37_candidate_guardrail_pressure_delta_only",
        "source_artifacts": {
            "v52_result_json": _rel(v52_result_json),
            "original_trace_dir": _rel(original_trace_dir),
            "v48_trace_dir": _rel(v48_trace_dir),
            "v52_trace_dir": _rel(v52_trace_dir),
        },
        "model_name": MODEL_NAME,
        "scenarios": list(scenarios),
        "online_llm_called": False,
        "new_rollout_run": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "scenario_reports": scenario_reports,
        "aggregate_deltas": dict(sorted(totals.items())),
        "reported_artifact_deltas": reported_artifact_deltas,
        "delta_method_note": (
            "aggregate_deltas are recomputed from trace rows with the shared compatibility helper. "
            "reported_artifact_deltas preserve the v52 result-audit summary counters."
        ),
    }


def _dominant_root_cause(leak_audit: Mapping[str, Any]) -> str:
    counts = Counter(leak_audit.get("classification_counts", {}) or {})
    counts.pop("unknown_requires_instrumentation", None)
    counts.pop("candidate_metadata_missing", None)
    if not counts:
        if int((leak_audit.get("classification_counts", {}) or {}).get("candidate_metadata_missing", 0)) > 0:
            return "candidate_metadata_missing"
        return "unknown_requires_instrumentation"
    return str(counts.most_common(1)[0][0])


def _next_action_for_root(root: str, leak_audit: Mapping[str, Any]) -> str:
    unknown = int((leak_audit.get("classification_counts", {}) or {}).get("unknown_requires_instrumentation", 0) or 0)
    missing = int(leak_audit.get("candidate_metadata_missing_count", 0) or 0)
    if unknown or missing:
        return "additional_instrumentation_required_before_fix"
    if root == "fallback_veto_not_invoked_on_non_fallback_candidate":
        return "post_selection_hard_safety_veto_for_all_sources"
    if root == "planner_anchor_empty_contract_filled":
        return "qwen37_planning_anchor_contract_fix"
    if root == "template_repair_created_new_conflict":
        return "profile_template_patch_v2_shadow_scoring"
    if root == "safety_required_vent_misclassified":
        return "profile_template_patch_v2_shadow_scoring"
    if root == "veto_no_compatible_alternative":
        return "candidate_guardrail_compatible_alternative_design"
    return "additional_instrumentation_required_before_fix"


def build_failure_attribution(
    leak_audit: Mapping[str, Any],
    pressure_delta: Mapping[str, Any],
    *,
    v52_result_json: str | Path = V52_RESULT_AUDIT_JSON,
    v49_runtime_audit_json: str | Path = V49_RUNTIME_AUDIT_JSON,
    v50_shadow_audit_json: str | Path = V50_SHADOW_AUDIT_JSON,
) -> dict[str, Any]:
    v52 = _load_json(v52_result_json)
    root = _dominant_root_cause(leak_audit)
    next_action = _next_action_for_root(root, leak_audit)
    acceptance = v52.get("acceptance", {}) or {}
    totals = v52.get("totals", {}) or {}
    reported_deltas = pressure_delta.get("reported_artifact_deltas", {}) or {}
    return {
        "artifact": "qwen37_profile_template_patch_failure_attribution_20260601_v53",
        "scope": "offline_qwen37_v52_failure_attribution_only",
        "source_artifacts": {
            "v52_result_json": _rel(v52_result_json),
            "v49_runtime_audit_json": _rel(v49_runtime_audit_json),
            "v50_shadow_audit_json": _rel(v50_shadow_audit_json),
            "leak_audit_json": _rel(LEAK_AUDIT_JSON),
            "pressure_delta_json": _rel(PRESSURE_DELTA_JSON),
        },
        "model_name": MODEL_NAME,
        "v52_qwen37_model_consistency_pass": bool(acceptance.get("qwen37_model_consistency_pass", False)),
        "v52_acceptance_pass": bool(acceptance.get("v52_acceptance_pass", False)),
        "online_llm_called": False,
        "new_rollout_run": False,
        "default_llm_rspc_v2_changed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "observed_v52_failure_summary": {
            "runtime_error_steps": int(totals.get("v52_runtime_error_steps", 0) or 0),
            "reported_aggregate_net_new_hard_safety_count": int(
                totals.get("hard_safety_rewrite_preferred_new_steps", 0) or 0
            ),
            "step_aligned_new_hard_safety_leak_count": int(
                leak_audit.get("step_aligned_new_hard_safety_leak_count", 0) or 0
            ),
            "candidate_guardrail_issue_delta_vs_v48": int(
                reported_deltas.get(
                    "v52_delta_vs_v48_candidate_guardrail_issue_steps", 0
                )
            ),
            "profile_action_conflict_delta_vs_v48": int(
                reported_deltas.get(
                    "v52_delta_vs_v48_profile_action_conflict_steps", 0
                )
            ),
        },
        "dominant_root_cause": root,
        "classification_counts": leak_audit.get("classification_counts", {}),
        "secondary_signal_counts": leak_audit.get("secondary_signal_counts", {}),
        "root_cause_interpretation": {
            "planner_anchor_empty_contract_filled": (
                "Qwen3.7 rows often fall back to hold_current/recent_anchor anchors; "
                "the profile patch repairs targets but does not guarantee a "
                "hard-safety-compatible post-selection anchor."
            ),
            "fallback_veto_not_invoked_on_non_fallback_candidate": (
                "A selected candidate can bypass the fallback-specific veto, so the "
                "veto boundary must move to all selected sources."
            ),
            "template_repair_created_new_conflict": (
                "Target repair may change the profile context without making the "
                "selected action compatible with Tomato Safety."
            ),
            "safety_required_vent_misclassified": (
                "Dry-risk high-vent penalties or template modes are still misclassifying "
                "safety-required ventilation in some rows."
            ),
        }.get(root, "Current metadata is insufficient for a clean root-cause patch."),
        "next_action": next_action,
    }


def build_repair_design(attribution: Mapping[str, Any], leak_audit: Mapping[str, Any]) -> dict[str, Any]:
    root = str(attribution.get("dominant_root_cause") or "unknown_requires_instrumentation")
    next_action = str(attribution.get("next_action") or "additional_instrumentation_required_before_fix")
    patch_options = {
        "post_selection_hard_safety_veto_for_all_sources": {
            "trigger": "selected action from any source is predicted to introduce hard_safety_rewrite",
            "repair_intent": "apply the hard-safety veto after final selection, not only inside fallback-specific paths",
            "must_not_change": ["Tomato Safety hard boundary", "default llm_rspc_v2 unless opt-in"],
        },
        "qwen37_planning_anchor_contract_fix": {
            "trigger": "qwen3.7 selected anchor/hold_current/recent_anchor rows dominate leak windows",
            "repair_intent": "make anchor contract require a hard-safety-compatible anchor or explicit no-compatible-alternative provenance",
            "must_not_change": ["controller promotion state", "controlled replay gate"],
        },
        "profile_template_patch_v2_shadow_scoring": {
            "trigger": "template repair or safety-vent mode causes persistent profile/candidate conflicts",
            "repair_intent": "score repaired profile against candidate compatibility before choosing the patch",
            "must_not_change": ["hard-safety boundary", "qwen model"],
        },
        "candidate_guardrail_compatible_alternative_design": {
            "trigger": "veto detects hard-safety issue but no compatible alternative exists",
            "repair_intent": "add a safe anchor/rule alternative with explicit provenance before any rollout",
            "must_not_change": ["controlled replay gate", "performance claims"],
        },
        "additional_instrumentation_required_before_fix": {
            "trigger": "unknown or metadata-missing leak rows remain",
            "repair_intent": "write missing planning-anchor/candidate-selection/veto provenance before another patch",
            "must_not_change": ["default controller", "rollout scope"],
        },
    }
    selected = patch_options.get(next_action, patch_options["additional_instrumentation_required_before_fix"])
    return {
        "artifact": "qwen37_profile_template_repair_design_20260601_v53",
        "scope": "offline_repair_design_only",
        "model_name": MODEL_NAME,
        "dominant_root_cause": root,
        "recommended_next_action": next_action,
        "design_ready": bool(next_action != "additional_instrumentation_required_before_fix"),
        "selected_repair_design": selected,
        "required_guardrails": [
            "no rollout in v53",
            "no online LLM call in v53",
            "default llm_rspc_v2 unchanged",
            "controlled replay remains disabled",
            "no performance or promotion claim",
        ],
        "leak_counting_note": leak_audit.get("counting_method_note"),
        "implementation_boundary": {
            "modify_default_controller_now": False,
            "run_rollout_now": False,
            "online_llm_called": False,
            "controlled_replay_allowed": False,
            "controlled_replay_execution_allowed": False,
            "metadata_replay_execution_allowed": False,
            "performance_claim_allowed": False,
            "promotion_evidence": False,
        },
    }


def build_readiness(
    attribution: Mapping[str, Any],
    leak_audit: Mapping[str, Any],
    repair_design: Mapping[str, Any],
) -> dict[str, Any]:
    root = str(attribution.get("dominant_root_cause") or "unknown_requires_instrumentation")
    next_action = str(attribution.get("next_action") or "additional_instrumentation_required_before_fix")
    unknown = int((leak_audit.get("classification_counts", {}) or {}).get("unknown_requires_instrumentation", 0) or 0)
    missing = int(leak_audit.get("candidate_metadata_missing_count", 0) or 0)
    attribution_complete = bool(root != "unknown_requires_instrumentation" and unknown == 0 and missing == 0)
    return {
        "artifact": "metadata_replay_readiness_checklist_20260601_v53",
        "current_stage": "v53_qwen37_profile_template_patch_failure_attribution",
        "model_name": MODEL_NAME,
        "v53_failure_attribution_done": True,
        "v53_failure_attribution_complete": attribution_complete,
        "v52_qwen37_model_consistency_pass": bool(attribution.get("v52_qwen37_model_consistency_pass", False)),
        "v52_acceptance_pass": bool(attribution.get("v52_acceptance_pass", False)),
        "dominant_root_cause": root,
        "reported_aggregate_net_new_hard_safety_count_from_v52": int(
            leak_audit.get("reported_aggregate_net_new_hard_safety_count_from_v52", 0) or 0
        ),
        "step_aligned_new_hard_safety_leak_count": int(
            leak_audit.get("step_aligned_new_hard_safety_leak_count", 0) or 0
        ),
        "candidate_metadata_missing_count": missing,
        "unknown_requires_instrumentation_count": unknown,
        "repair_design_ready": bool(repair_design.get("design_ready", False)),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": next_action,
        "stop_taxonomy": [
            item
            for item in (
                "metadata_missing" if missing else "",
                "unknown_requires_instrumentation" if unknown else "",
                "qwen37_profile_template_patch_failed" if not attribution.get("v52_acceptance_pass", False) else "",
            )
            if item
        ],
    }


def _write_json_md(path_json: Path, path_md: Path, data: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    lines = [f"# {title}", ""]
    for key in (
        "artifact",
        "scope",
        "current_stage",
        "model_name",
        "dominant_root_cause",
        "recommended_next_action",
        "next_action",
    ):
        if key in data:
            lines.append(f"- {key}: `{data.get(key)}`")
    for key in (
        "controlled_replay_allowed",
        "controlled_replay_execution_allowed",
        "metadata_replay_execution_allowed",
        "performance_claim_allowed",
        "promotion_evidence",
    ):
        if key in data:
            lines.append(f"- {key}: `{str(data.get(key)).lower()}`")
    for key in (
        "observed_v52_failure_summary",
        "classification_counts",
        "secondary_signal_counts",
        "scenario_reports",
        "aggregate_deltas",
        "stop_taxonomy",
    ):
        if key in data:
            lines.extend(["", f"## {key}", "```json", json.dumps(data[key], indent=2, sort_keys=True), "```"])
    if "leak_entries" in data:
        lines.extend(
            [
                "",
                "## leak_entries_sample",
                "```json",
                json.dumps(list(data.get("leak_entries", []))[:20], indent=2, sort_keys=True),
                "```",
            ]
        )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v48_trace_dir: str | Path = V48_TRACE_DIR,
    v52_trace_dir: str | Path = V52_TRACE_DIR,
    v52_result_json: str | Path = V52_RESULT_AUDIT_JSON,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    leak_audit = build_hard_safety_leak_audit(
        original_trace_dir=original_trace_dir,
        v52_trace_dir=v52_trace_dir,
        v52_result_json=v52_result_json,
        scenarios=scenarios,
    )
    pressure_delta = build_candidate_guardrail_pressure_delta_audit(
        original_trace_dir=original_trace_dir,
        v48_trace_dir=v48_trace_dir,
        v52_trace_dir=v52_trace_dir,
        v52_result_json=v52_result_json,
        scenarios=scenarios,
    )
    attribution = build_failure_attribution(leak_audit, pressure_delta, v52_result_json=v52_result_json)
    repair_design = build_repair_design(attribution, leak_audit)
    readiness = build_readiness(attribution, leak_audit, repair_design)
    _write_json_md(LEAK_AUDIT_JSON, LEAK_AUDIT_MD, leak_audit, "v53 Qwen3.7 Hard-Safety Rewrite Preferred Leak Audit")
    _write_json_md(PRESSURE_DELTA_JSON, PRESSURE_DELTA_MD, pressure_delta, "v53 Qwen3.7 Candidate/Guardrail Pressure Delta Audit")
    _write_json_md(ATTRIBUTION_JSON, ATTRIBUTION_MD, attribution, "v53 Qwen3.7 Profile Template Patch Failure Attribution")
    _write_json_md(REPAIR_DESIGN_JSON, REPAIR_DESIGN_MD, repair_design, "v53 Qwen3.7 Profile Template Repair Design")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v53 Metadata Replay Readiness")
    return leak_audit, pressure_delta, attribution, repair_design, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-trace-dir", default=str(ORIGINAL_TRACE_DIR))
    parser.add_argument("--v48-trace-dir", default=str(V48_TRACE_DIR))
    parser.add_argument("--v52-trace-dir", default=str(V52_TRACE_DIR))
    parser.add_argument("--v52-result-json", default=str(V52_RESULT_AUDIT_JSON))
    parser.add_argument("--failure-scenario", action="append")
    parser.add_argument("--write-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios = args.failure_scenario or list(FAILURE_SCENARIOS)
    leak_audit, pressure_delta, attribution, repair_design, readiness = write_all(
        original_trace_dir=args.original_trace_dir,
        v48_trace_dir=args.v48_trace_dir,
        v52_trace_dir=args.v52_trace_dir,
        v52_result_json=args.v52_result_json,
        scenarios=scenarios,
    )
    if not args.write_all:
        print(
            json.dumps(
                {
                    "step_aligned_new_hard_safety_leak_count": leak_audit.get(
                        "step_aligned_new_hard_safety_leak_count"
                    ),
                    "reported_aggregate_net_new_hard_safety_count_from_v52": leak_audit.get(
                        "reported_aggregate_net_new_hard_safety_count_from_v52"
                    ),
                    "dominant_root_cause": attribution.get("dominant_root_cause"),
                    "next_action": readiness.get("next_action"),
                    "repair_design_ready": repair_design.get("design_ready"),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
