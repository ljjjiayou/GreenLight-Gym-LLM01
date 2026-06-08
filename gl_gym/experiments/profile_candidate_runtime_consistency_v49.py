"""Build v49 profile/candidate runtime consistency artifacts.

This audit is offline-only. It compares v47 offline profile feasibility repair
against v48 opt-in runtime traces, separates fallback/provider-access evidence,
and writes a template-fix design. It does not run rollout, call online LLMs,
mutate the default controller, authorize controlled replay, or make performance
claims.
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
from gl_gym.experiments.profile_feasibility_gate_shadow_rollout_v48 import (  # noqa: E402
    EXECUTION_RECORD_JSON as V48_EXECUTION_RECORD_JSON,
    RESULT_AUDIT_JSON as V48_RESULT_AUDIT_JSON,
    V48_TRACE_DIR,
)
from gl_gym.experiments.profile_feasibility_repair_v47 import (  # noqa: E402
    COMPARISON_JSON as V47_COMPARISON_JSON,
    repair_profile_targets,
    shadow_score_row_with_repair,
)


V45_AUDIT_JSON = AUDIT_DIR / "profile_candidate_guardrail_compatibility_audit_20260531_v45.json"
V46_COMPARISON_JSON = AUDIT_DIR / "candidate_guardrail_shadow_scoring_comparison_20260531_v46.json"

RUNTIME_AUDIT_JSON = AUDIT_DIR / "profile_candidate_runtime_consistency_audit_20260601_v49.json"
RUNTIME_AUDIT_MD = AUDIT_DIR / "profile_candidate_runtime_consistency_audit_20260601_v49.md"
ONLINE_DESIGN_JSON = AUDIT_DIR / "online_llm_accessibility_precheck_design_20260601_v49.json"
ONLINE_DESIGN_MD = AUDIT_DIR / "online_llm_accessibility_precheck_design_20260601_v49.md"
TEMPLATE_FIX_JSON = AUDIT_DIR / "profile_template_feasibility_fix_design_20260601_v49.json"
TEMPLATE_FIX_MD = AUDIT_DIR / "profile_template_feasibility_fix_design_20260601_v49.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v49.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v49.md"

PROVIDER_ERROR_PATTERNS = (
    "403",
    "Access denied",
    "Forbidden",
    "InvalidApiKey",
    "Invalid API key",
    "ModelStudio",
    "Bailian",
    "DashScope",
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


def _csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = _resolve(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _scenario_report_from_v48(v48_audit: Mapping[str, Any], scenario: str) -> dict[str, Any]:
    for report in v48_audit.get("scenario_reports", []) or []:
        if report.get("scenario_id") == scenario:
            return dict(report)
    return {}


def _rows_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        by_step[int(_num(row.get("step"), -1))] = dict(row)
    return by_step


def _compat(row: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    merged.update(derive_candidate_guardrail_compatibility(dict(row)))
    return merged


def _is_hard_safety(row: Mapping[str, Any]) -> bool:
    return str(_compat(row).get("candidate_guardrail_compatibility_label") or "") == "hard_safety_rewrite"


def _selected_candidate_name(row: Mapping[str, Any]) -> str:
    for candidate in _parse_candidates(row):
        if _bool(candidate.get("selected")):
            return _candidate_name(candidate)
    return str(row.get("rspc_action_selected_name") or row.get("source") or "unknown")


def _candidate_metadata_status(row: Mapping[str, Any]) -> str:
    return "present" if _parse_candidates(row) else "candidate_metadata_missing"


def _fallback_like(row: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(row.get(field, "") or "")
        for field in (
            "source",
            "candidate_selection_source",
            "replan_reason",
            "anchor_source",
            "profile_feasibility_gate_source",
        )
    ).lower()
    return any(token in text for token in ("fallback", "anchor", "rule", "conservative"))


def _repair_targets_from_v48(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "target_temp": _num(row.get("profile_feasibility_gate_repaired_target_temp")),
        "target_co2": _num(row.get("profile_feasibility_gate_repaired_target_co2")),
        "target_rh": _num(row.get("profile_feasibility_gate_repaired_target_rh")),
    }


def _offline_repair_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    repaired, repair = repair_profile_targets(row)
    scored = shadow_score_row_with_repair(row)
    return {
        "status": scored.get("status"),
        "corrections": list(repair.get("corrections", []) or []),
        "mode": repair.get("mode", {}),
        "repaired_targets": dict(repair.get("repaired", {})),
        "original_targets": dict(repair.get("original", {})),
        "shadow_selected_name": scored.get("shadow_selected_name"),
        "original_selected_name": scored.get("original_selected_name"),
        "hard_safety_rewrite_preferred_new": bool(scored.get("hard_safety_rewrite_preferred_new", False)),
        "candidate_count": int(scored.get("candidate_count", 0) or 0),
        "repaired_row_preview": {
            "target_temp": repaired.get("target_temp"),
            "target_co2": repaired.get("target_co2"),
            "target_rh": repaired.get("target_rh"),
        },
    }


def _runtime_gate_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": _bool(row.get("profile_feasibility_gate_enabled")),
        "applied": _bool(row.get("profile_feasibility_gate_applied")),
        "corrections": _split_csv(row.get("profile_feasibility_gate_corrections")),
        "source": str(row.get("profile_feasibility_gate_source") or ""),
        "repaired_targets": _repair_targets_from_v48(row),
        "hard_safety_veto_count": int(_num(row.get("profile_feasibility_gate_hard_safety_veto_count"))),
        "vent_required_by_safety": _bool(row.get("profile_feasibility_gate_vent_required_by_safety")),
        "humidity_retention_allowed": _bool(row.get("profile_feasibility_gate_humidity_retention_allowed")),
        "co2_enrichment_allowed": _bool(row.get("profile_feasibility_gate_co2_enrichment_allowed")),
    }


def _targets_mismatch(offline: Mapping[str, Any], runtime: Mapping[str, Any], *, tol: float = 1e-6) -> bool:
    offline_targets = offline.get("repaired_targets", {}) or {}
    runtime_targets = runtime.get("repaired_targets", {}) or {}
    for key in ("target_temp", "target_co2", "target_rh"):
        if abs(_num(offline_targets.get(key)) - _num(runtime_targets.get(key))) > tol:
            return True
    return False


def _alignment_issue(offline: Mapping[str, Any], runtime: Mapping[str, Any], runtime_row: Mapping[str, Any]) -> str:
    offline_has_corrections = bool(offline.get("corrections"))
    runtime_has_corrections = bool(runtime.get("corrections"))
    if offline_has_corrections and not runtime.get("applied"):
        return "offline_repair_not_applied_runtime"
    if offline_has_corrections and not runtime_has_corrections:
        return "offline_repair_runtime_no_correction"
    if runtime_has_corrections and _targets_mismatch(offline, runtime):
        return "offline_runtime_repair_target_mismatch"
    if _fallback_like(runtime_row):
        return "runtime_fallback_or_rule_path"
    return "aligned_or_no_repair_needed"


def _row_pressure(row: Mapping[str, Any]) -> dict[str, Any]:
    comp = _compat(row)
    label = str(comp.get("candidate_guardrail_compatibility_label") or "compatible")
    conflict_label = str(comp.get("profile_action_conflict_label") or "none")
    return {
        "compatibility_label": label,
        "profile_action_conflict_label": conflict_label,
        "candidate_guardrail_issue": label in {"major_rewrite", "hard_safety_rewrite"} or conflict_label != "none",
        "major_or_hard_rewrite": label in {"major_rewrite", "hard_safety_rewrite"},
        "hard_safety_rewrite": label == "hard_safety_rewrite",
        "rewrite_fields": _split_csv(comp.get("candidate_guardrail_rewrite_fields")),
        "candidate_selection_source": str(comp.get("candidate_selection_source") or row.get("source") or "unknown"),
        "candidate_filter_reason_summary": str(comp.get("candidate_filter_reason_summary") or ""),
    }


def _row_consistency_entry(
    *,
    scenario: str,
    step: int,
    original_row: Mapping[str, Any],
    runtime_row: Mapping[str, Any],
) -> dict[str, Any]:
    offline = _offline_repair_summary(original_row)
    runtime = _runtime_gate_summary(runtime_row)
    original_pressure = _row_pressure(original_row)
    runtime_pressure = _row_pressure(runtime_row)
    issue = _alignment_issue(offline, runtime, runtime_row)
    return {
        "scenario_id": scenario,
        "step": int(step),
        "offline_status": offline.get("status"),
        "offline_corrections": offline.get("corrections", []),
        "runtime_corrections": runtime.get("corrections", []),
        "runtime_gate_applied": bool(runtime.get("applied", False)),
        "runtime_gate_source": runtime.get("source", ""),
        "offline_shadow_selected_name": offline.get("shadow_selected_name"),
        "runtime_selected_candidate_name": _selected_candidate_name(runtime_row),
        "candidate_metadata_status": _candidate_metadata_status(runtime_row),
        "alignment_issue": issue,
        "original_pressure": original_pressure,
        "runtime_pressure": runtime_pressure,
        "runtime_candidate_guardrail_pressure_persists": bool(runtime_pressure["candidate_guardrail_issue"]),
    }


def _hard_leak_classification(row: Mapping[str, Any]) -> str:
    if _candidate_metadata_status(row) == "candidate_metadata_missing":
        return "candidate_metadata_missing"
    if _fallback_like(row):
        return "fallback_path_bypassed_veto"
    if int(_num(row.get("profile_feasibility_gate_hard_safety_veto_count"))) <= 0:
        return "veto_not_evaluated"
    return "veto_score_penalty_insufficient"


def _hard_leaks_for_scenario(
    *,
    scenario: str,
    original_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    target_count: int,
) -> list[dict[str, Any]]:
    original_by_step = _rows_by_step(original_rows)
    runtime_hard = [dict(row) for row in runtime_rows if _is_hard_safety(row)]
    primary = [row for row in runtime_hard if not _is_hard_safety(original_by_step.get(int(_num(row.get("step"), -1)), {}))]
    if len(primary) < target_count:
        primary_steps = {int(_num(row.get("step"), -1)) for row in primary}
        primary.extend(row for row in runtime_hard if int(_num(row.get("step"), -1)) not in primary_steps)
    entries: list[dict[str, Any]] = []
    for row in primary[:target_count]:
        step = int(_num(row.get("step"), -1))
        comp = _compat(row)
        entries.append(
            {
                "scenario_id": scenario,
                "step": step,
                "candidate": _selected_candidate_name(row),
                "selected_source": str(comp.get("candidate_selection_source") or row.get("source") or "unknown"),
                "rewrite_fields": _split_csv(comp.get("candidate_guardrail_rewrite_fields")),
                "safety_reason": str(row.get("tomato_safety_v2_reasons") or row.get("rspc_action_post_shape_safety_gate_reason") or ""),
                "classification": _hard_leak_classification(row),
                "candidate_metadata_status": _candidate_metadata_status(row),
                "veto_count": int(_num(row.get("profile_feasibility_gate_hard_safety_veto_count"))),
                "aggregate_target_count_source": "v48_hard_safety_rewrite_preferred_new_steps",
            }
        )
    return entries


def build_runtime_consistency_audit(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v48_trace_dir: str | Path = V48_TRACE_DIR,
    v47_comparison_json: str | Path = V47_COMPARISON_JSON,
    v48_result_json: str | Path = V48_RESULT_AUDIT_JSON,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    v47 = _load_json(v47_comparison_json)
    v48 = _load_json(v48_result_json)
    scenario_reports: list[dict[str, Any]] = []
    all_leaks: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    for scenario in failure_scenarios:
        original_rows = _read_rows(_trace_path(original_trace_dir, scenario))
        runtime_rows = _read_rows(_trace_path(v48_trace_dir, scenario))
        original_by_step = _rows_by_step(original_rows)
        runtime_by_step = _rows_by_step(runtime_rows)
        v48_report = _scenario_report_from_v48(v48, scenario)
        comparison_steps = int(v48_report.get("comparison_window_steps") or min(len(original_rows), len(runtime_rows)))
        aligned_steps = [step for step in sorted(original_by_step) if step in runtime_by_step and step < comparison_steps]
        entries = [
            _row_consistency_entry(
                scenario=scenario,
                step=step,
                original_row=original_by_step[step],
                runtime_row=runtime_by_step[step],
            )
            for step in aligned_steps
        ]
        for entry in entries:
            issue_counts[str(entry["alignment_issue"])] += 1
        target_leak_count = int(v48_report.get("hard_safety_rewrite_preferred_new_steps", 0) or 0)
        leaks = _hard_leaks_for_scenario(
            scenario=scenario,
            original_rows=[original_by_step[step] for step in aligned_steps],
            runtime_rows=[runtime_by_step[step] for step in aligned_steps],
            target_count=target_leak_count,
        )
        all_leaks.extend(leaks)
        scenario_reports.append(
            {
                "scenario_id": scenario,
                "original_trace_exists": bool(original_rows),
                "v48_trace_exists": bool(runtime_rows),
                "comparison_window_steps": comparison_steps,
                "aligned_step_count": len(aligned_steps),
                "original_first_runtime_error_step": _runtime_failure_step(original_rows),
                "v48_first_runtime_error_step": _runtime_failure_step(runtime_rows),
                "alignment_issue_counts": dict(Counter(entry["alignment_issue"] for entry in entries)),
                "runtime_candidate_guardrail_pressure_persist_steps": int(
                    sum(bool(entry["runtime_candidate_guardrail_pressure_persists"]) for entry in entries)
                ),
                "hard_safety_rewrite_preferred_new_steps_expected": target_leak_count,
                "hard_safety_leak_entries": leaks,
                "top_runtime_mismatch_examples": [
                    entry
                    for entry in entries
                    if entry["alignment_issue"] != "aligned_or_no_repair_needed"
                    or entry["runtime_candidate_guardrail_pressure_persists"]
                ][:20],
            }
        )
    leak_counts = Counter(entry["classification"] for entry in all_leaks)
    unclassified = int(leak_counts.get("candidate_metadata_missing", 0))
    return {
        "artifact": "profile_candidate_runtime_consistency_audit_20260601_v49",
        "scope": "offline_runtime_consistency_audit_only",
        "source_artifacts": {
            "v47_comparison_json": _rel(v47_comparison_json),
            "v48_result_json": _rel(v48_result_json),
            "original_trace_dir": _rel(original_trace_dir),
            "v48_trace_dir": _rel(v48_trace_dir),
        },
        "scenarios": list(failure_scenarios),
        "v47_profile_feasibility_acceptance_pass": bool((v47.get("acceptance", {}) or {}).get("profile_feasibility_acceptance_pass", False)),
        "v48_profile_feasibility_gate_shadow_rollout_pass": bool((v48.get("acceptance", {}) or {}).get("v48_acceptance_pass", False)),
        "online_llm_called": False,
        "new_rollout_run": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "scenario_reports": scenario_reports,
        "alignment_issue_counts": dict(sorted(issue_counts.items())),
        "hard_safety_veto_leak_audit": {
            "expected_leak_count_from_v48": int(sum(int(report.get("hard_safety_rewrite_preferred_new_steps_expected", 0)) for report in scenario_reports)),
            "located_leak_count": len(all_leaks),
            "classification_counts": dict(sorted(leak_counts.items())),
            "candidate_metadata_missing_count": unclassified,
            "leak_entries": all_leaks,
        },
    }


def _scan_text_file(path: Path) -> tuple[bool, list[str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False, []
    hits = [pattern for pattern in PROVIDER_ERROR_PATTERNS if pattern.lower() in text.lower()]
    return bool(hits), hits


def scan_provider_error_evidence(paths: Sequence[str | Path]) -> dict[str, Any]:
    scanned = 0
    error_files: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for item in paths:
        p = _resolve(item)
        candidates = sorted(p.rglob("*")) if p.is_dir() else [p]
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if not candidate.is_file() or candidate.suffix.lower() not in {".json", ".jsonl", ".csv", ".txt", ".log", ".md"}:
                continue
            scanned += 1
            found, patterns = _scan_text_file(candidate)
            if found:
                error_files.append({"path": _rel(candidate), "patterns": patterns})
    return {
        "scanned_file_count": scanned,
        "provider_error_evidence_found": bool(error_files),
        "provider_error_files": error_files[:20],
    }


def build_online_llm_accessibility_design(
    *,
    v48_execution_record_json: str | Path = V48_EXECUTION_RECORD_JSON,
    scan_paths: Sequence[str | Path] = (V48_TRACE_DIR, PROJECT_ROOT / "gl_gym" / "result" / "benchmarks" / "profile_feasibility_gate_shadow_rollout_v48_20260531"),
) -> dict[str, Any]:
    execution = _load_json(v48_execution_record_json)
    scan = scan_provider_error_evidence(scan_paths)
    credentials_present = bool(execution.get("online_llm_credentials_present", False))
    cache_path = _resolve(str(execution.get("cache_path") or ""))
    cache_exists = bool(str(execution.get("cache_path") or "").strip() and cache_path.exists())
    accessible = bool(credentials_present and cache_exists and not scan["provider_error_evidence_found"])
    status = "accessible" if accessible else "not_proven_accessible"
    if scan["provider_error_evidence_found"]:
        status = "provider_error_detected"
    elif credentials_present and not cache_exists:
        status = "credentials_present_but_no_recorded_cache"
    elif not credentials_present:
        status = "credentials_missing"
    return {
        "artifact": "online_llm_accessibility_precheck_design_20260601_v49",
        "scope": "design_only_no_online_call",
        "v48_execution_record_json": _rel(v48_execution_record_json),
        "online_llm_credentials_present": credentials_present,
        "online_llm_accessible": accessible,
        "online_llm_accessibility_status": status,
        "isolated_cache_path": _rel(cache_path) if str(execution.get("cache_path") or "").strip() else "",
        "isolated_cache_exists": cache_exists,
        "provider_error_scan": scan,
        "required_future_execution_record_fields": [
            "online_llm_credentials_present",
            "online_llm_accessible",
            "online_llm_accessibility_status",
            "provider_model_access_precheck_timestamp",
        ],
        "future_rollout_rule": (
            "If online_llm_accessible is false, the run may only be labeled fallback-path evidence, "
            "not qwen-max planning evidence."
        ),
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def build_profile_template_fix_design(runtime_audit: Mapping[str, Any], online_design: Mapping[str, Any]) -> dict[str, Any]:
    leak_counts = ((runtime_audit.get("hard_safety_veto_leak_audit", {}) or {}).get("classification_counts", {}) or {})
    issue_counts = runtime_audit.get("alignment_issue_counts", {}) or {}
    fix_ready = bool(
        runtime_audit.get("scenario_reports")
        and int((runtime_audit.get("hard_safety_veto_leak_audit", {}) or {}).get("located_leak_count", 0) or 0) > 0
    )
    return {
        "artifact": "profile_template_feasibility_fix_design_20260601_v49",
        "scope": "design_only_no_controller_mutation",
        "default_llm_rspc_v2_changed": False,
        "template_fix_design_ready": fix_ready,
        "primary_runtime_findings": {
            "alignment_issue_counts": issue_counts,
            "hard_safety_leak_classification_counts": leak_counts,
            "online_llm_accessibility_status": online_design.get("online_llm_accessibility_status"),
        },
        "template_level_forbidden_combinations": [
            {
                "name": "safety_vent_blocks_high_rh_high_co2",
                "condition": "vent_required_by_safety=true",
                "forbid": "target_rh high together with target_co2 enrichment",
                "repair_intent": "cap target_rh and target_co2 before candidate scoring",
            },
            {
                "name": "rh_hard_pressure_blocks_humidity_retention",
                "condition": "rh_hard_pressure=true or Tomato Safety RH/dew/canopy pressure active",
                "forbid": "humidity_retention_allowed=true",
                "repair_intent": "select dehumidification-compatible profile template",
            },
            {
                "name": "dry_high_vent_penalty_only_without_safety_vent",
                "condition": "dry_side=true and vent_required_by_safety=false",
                "forbid": "penalizing safety-required high ventilation as dry-risk conflict",
                "repair_intent": "move dry-risk high-vent conflict behind safety-vent mode check",
            },
            {
                "name": "fallback_path_requires_post_selection_veto",
                "condition": "candidate_selection_source is fallback/rule/anchor path",
                "forbid": "assuming score-only veto was evaluated",
                "repair_intent": "apply hard-safety veto semantics after fallback candidate selection metadata is available",
            },
        ],
        "implementation_boundary": {
            "modify_default_controller_now": False,
            "run_rollout_now": False,
            "controlled_replay_allowed": False,
            "metadata_replay_execution_allowed": False,
            "performance_claim_allowed": False,
            "promotion_evidence": False,
        },
        "next_patch_plan": "minimal_profile_template_shadow_patch_plan"
        if fix_ready
        else "profile_candidate_metadata_instrumentation_patch_plan",
    }


def build_readiness(
    runtime_audit: Mapping[str, Any],
    online_design: Mapping[str, Any],
    template_design: Mapping[str, Any],
) -> dict[str, Any]:
    leak_audit = runtime_audit.get("hard_safety_veto_leak_audit", {}) or {}
    missing = int(leak_audit.get("candidate_metadata_missing_count", 0) or 0)
    fix_ready = bool(template_design.get("template_fix_design_ready", False))
    next_action = (
        "minimal_profile_template_shadow_patch_plan"
        if fix_ready and missing == 0
        else "profile_candidate_metadata_instrumentation_patch_plan"
    )
    return {
        "artifact": "metadata_replay_readiness_checklist_20260601_v49",
        "current_stage": "v49_profile_candidate_runtime_consistency_audit",
        "v49_runtime_consistency_audit_done": bool(runtime_audit.get("scenario_reports")),
        "hard_safety_veto_leak_count": int(leak_audit.get("located_leak_count", 0) or 0),
        "hard_safety_veto_leak_classification_counts": leak_audit.get("classification_counts", {}),
        "online_llm_accessible": bool(online_design.get("online_llm_accessible", False)),
        "online_llm_accessibility_status": online_design.get("online_llm_accessibility_status"),
        "template_fix_design_ready": fix_ready,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": next_action,
        "stop_taxonomy": [
            item
            for item in (
                "fallback_path_pollutes_qwen_evidence" if not online_design.get("online_llm_accessible", False) else "",
                "hard_safety_veto_failed" if int(leak_audit.get("located_leak_count", 0) or 0) > 0 else "",
                "candidate_metadata_missing" if missing else "",
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
        "online_llm_accessibility_status",
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
        "alignment_issue_counts",
        "hard_safety_veto_leak_audit",
        "primary_runtime_findings",
        "provider_error_scan",
        "stop_taxonomy",
    ):
        if key in data:
            lines.extend(["", f"## {key}", "```json", json.dumps(data[key], indent=2, sort_keys=True), "```"])
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v48_trace_dir: str | Path = V48_TRACE_DIR,
    v47_comparison_json: str | Path = V47_COMPARISON_JSON,
    v48_result_json: str | Path = V48_RESULT_AUDIT_JSON,
    v48_execution_record_json: str | Path = V48_EXECUTION_RECORD_JSON,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    runtime_audit = build_runtime_consistency_audit(
        original_trace_dir=original_trace_dir,
        v48_trace_dir=v48_trace_dir,
        v47_comparison_json=v47_comparison_json,
        v48_result_json=v48_result_json,
        failure_scenarios=failure_scenarios,
    )
    online_design = build_online_llm_accessibility_design(v48_execution_record_json=v48_execution_record_json)
    template_design = build_profile_template_fix_design(runtime_audit, online_design)
    readiness = build_readiness(runtime_audit, online_design, template_design)
    _write_json_md(RUNTIME_AUDIT_JSON, RUNTIME_AUDIT_MD, runtime_audit, "v49 Profile-Candidate Runtime Consistency Audit")
    _write_json_md(ONLINE_DESIGN_JSON, ONLINE_DESIGN_MD, online_design, "v49 Online LLM Accessibility Precheck Design")
    _write_json_md(TEMPLATE_FIX_JSON, TEMPLATE_FIX_MD, template_design, "v49 Profile Template Feasibility Fix Design")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v49 Metadata Replay Readiness")
    return runtime_audit, online_design, template_design, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-trace-dir", default=str(ORIGINAL_TRACE_DIR))
    parser.add_argument("--v48-trace-dir", default=str(V48_TRACE_DIR))
    parser.add_argument("--v47-comparison-json", default=str(V47_COMPARISON_JSON))
    parser.add_argument("--v48-result-json", default=str(V48_RESULT_AUDIT_JSON))
    parser.add_argument("--v48-execution-record-json", default=str(V48_EXECUTION_RECORD_JSON))
    parser.add_argument("--failure-scenario", action="append")
    parser.add_argument("--write-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios = args.failure_scenario or list(FAILURE_SCENARIOS)
    runtime_audit, online_design, template_design, readiness = write_all(
        original_trace_dir=args.original_trace_dir,
        v48_trace_dir=args.v48_trace_dir,
        v47_comparison_json=args.v47_comparison_json,
        v48_result_json=args.v48_result_json,
        v48_execution_record_json=args.v48_execution_record_json,
        failure_scenarios=scenarios,
    )
    if not args.write_all:
        print(
            json.dumps(
                {
                    "runtime_audit": runtime_audit,
                    "online_design": online_design,
                    "template_design": template_design,
                    "readiness": readiness,
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
