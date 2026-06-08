"""Build v50 profile-template shadow patch artifacts.

This audit is offline-only. It checks whether template-level feasibility
repairs plus fallback post-selection hard-safety veto semantics would explain
and block the v49 hard-safety leak steps. It does not run rollout, call online
LLMs, mutate the default controller, authorize controlled replay, or make
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
    _parse_candidates,
    _read_rows,
    _trace_path,
    _truthy,
)
from gl_gym.experiments.diagnose_ppo_vs_llm import derive_candidate_guardrail_compatibility  # noqa: E402
from gl_gym.experiments.profile_candidate_runtime_consistency_v49 import (  # noqa: E402
    RUNTIME_AUDIT_JSON as V49_RUNTIME_AUDIT_JSON,
    _candidate_metadata_status,
    _fallback_like,
    _load_json,
    _num,
    _rel,
    _rows_by_step,
    _selected_candidate_name,
    _split_csv,
)
from gl_gym.experiments.profile_feasibility_gate_shadow_rollout_v48 import V48_TRACE_DIR  # noqa: E402
from gl_gym.experiments.profile_feasibility_repair_v47 import repair_profile_targets  # noqa: E402


AUDIT_JSON = AUDIT_DIR / "profile_template_shadow_patch_audit_20260601_v50.json"
AUDIT_MD = AUDIT_DIR / "profile_template_shadow_patch_audit_20260601_v50.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v50.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v50.md"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = _resolve(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _compat(row: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    merged.update(derive_candidate_guardrail_compatibility(dict(row)))
    return merged


def _conflict_list(row: Mapping[str, Any]) -> list[str]:
    label = str(_compat(row).get("profile_action_conflict_label") or "none")
    return _split_csv(label)


def _with_shadow_targets(row: Mapping[str, Any], targets: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["target_temp"] = targets.get("target_temp", row.get("target_temp"))
    out["target_co2"] = targets.get("target_co2", row.get("target_co2"))
    out["target_rh"] = targets.get("target_rh", row.get("target_rh"))
    return out


def profile_template_shadow_targets(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return template-level shadow target repair without changing row/action."""

    repaired, repair = repair_profile_targets(row)
    mode = repair.get("mode", {}) or {}
    original = repair.get("original", {}) or {}
    shadow_targets = repair.get("repaired", {}) or {}
    forbidden: list[str] = []
    high_rh = _num(original.get("target_rh")) >= 75.0
    high_co2 = _num(original.get("target_co2")) >= 800.0
    high_vent = _num(mode.get("max_vent")) >= 0.70
    if mode.get("vent_required_by_safety") and high_rh and high_co2:
        forbidden.append("safety_vent_blocks_high_rh_high_co2")
    if (
        mode.get("rh_hard_pressure")
        or mode.get("dew_canopy_pressure")
        or mode.get("vent_required_by_safety")
    ) and not mode.get("humidity_retention_allowed", True):
        forbidden.append("rh_dew_canopy_pressure_blocks_humidity_retention")
    if mode.get("dry_side") and high_vent and mode.get("vent_required_by_safety"):
        forbidden.append("dry_high_vent_penalty_suppressed_by_safety_vent")
    if mode.get("dry_side") and high_vent and not mode.get("vent_required_by_safety"):
        forbidden.append("dry_high_vent_penalty_active")
    patched = _with_shadow_targets(row, shadow_targets)
    original_conflicts = _conflict_list(row)
    patched_conflicts = _conflict_list(patched)
    return {
        "mode": mode,
        "original_targets": original,
        "template_shadow_targets": shadow_targets,
        "corrections": list(repair.get("corrections", []) or []),
        "forbidden_combinations": sorted(set(forbidden)),
        "original_profile_conflicts": original_conflicts,
        "shadow_profile_conflicts": patched_conflicts,
        "would_reduce_profile_conflict": len(patched_conflicts) < len(original_conflicts),
        "repaired_row_preview": {
            "target_temp": repaired.get("target_temp"),
            "target_co2": repaired.get("target_co2"),
            "target_rh": repaired.get("target_rh"),
        },
    }


def fallback_post_selection_veto_shadow(row: Mapping[str, Any]) -> dict[str, Any]:
    """Shadow-only veto semantics for already selected fallback/rule/anchor rows."""

    comp = _compat(row)
    label = str(comp.get("candidate_guardrail_compatibility_label") or "compatible")
    source = str(comp.get("candidate_selection_source") or row.get("source") or "unknown")
    fallback_path = _fallback_like({**row, **comp})
    hard_safety = label == "hard_safety_rewrite"
    metadata_status = _candidate_metadata_status(row)
    would_block = bool(fallback_path and hard_safety)
    if metadata_status == "candidate_metadata_missing":
        reason = "candidate_metadata_missing"
    elif would_block:
        reason = "fallback_post_selection_hard_safety_rewrite"
    elif fallback_path:
        reason = "fallback_path_no_hard_safety_rewrite"
    else:
        reason = "not_fallback_path"
    return {
        "fallback_path": bool(fallback_path),
        "candidate_metadata_status": metadata_status,
        "selected_candidate": _selected_candidate_name(row),
        "selected_source": source,
        "compatibility_label": label,
        "rewrite_fields": _split_csv(comp.get("candidate_guardrail_rewrite_fields")),
        "safety_reason": str(row.get("tomato_safety_v2_reasons") or row.get("rspc_action_post_shape_safety_gate_reason") or ""),
        "fallback_veto_shadow_would_block": would_block,
        "veto_reason": reason,
        "candidate_count": len(_parse_candidates(row)),
    }


def shadow_patch_for_row(row: Mapping[str, Any]) -> dict[str, Any]:
    template = profile_template_shadow_targets(row)
    fallback_veto = fallback_post_selection_veto_shadow(row)
    would_block = bool(fallback_veto["fallback_veto_shadow_would_block"])
    return {
        "template_shadow": template,
        "fallback_post_selection_veto_shadow": fallback_veto,
        "would_reduce_profile_conflict": bool(template["would_reduce_profile_conflict"]),
        "would_block_hard_safety_leak": would_block,
        "metadata_gap": fallback_veto["candidate_metadata_status"] != "present",
    }


def _load_leak_entries(v49_runtime_audit_json: str | Path) -> list[dict[str, Any]]:
    audit = _load_json(v49_runtime_audit_json)
    leak_audit = audit.get("hard_safety_veto_leak_audit", {}) or {}
    return [dict(item) for item in leak_audit.get("leak_entries", []) or []]


def build_shadow_patch_audit(
    *,
    v49_runtime_audit_json: str | Path = V49_RUNTIME_AUDIT_JSON,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v48_trace_dir: str | Path = V48_TRACE_DIR,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    leak_entries = _load_leak_entries(v49_runtime_audit_json)
    scenario_set = set(failure_scenarios)
    rows_by_scenario = {
        scenario: _rows_by_step(_read_rows(_trace_path(v48_trace_dir, scenario)))
        for scenario in failure_scenarios
    }
    original_rows_by_scenario = {
        scenario: _rows_by_step(_read_rows(_trace_path(original_trace_dir, scenario)))
        for scenario in failure_scenarios
    }
    patched_entries: list[dict[str, Any]] = []
    missing_entries: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    forbidden_counts: Counter[str] = Counter()
    for leak in leak_entries:
        scenario = str(leak.get("scenario_id") or "")
        step = int(_num(leak.get("step"), -1))
        if scenario not in scenario_set:
            missing_entries.append({**leak, "missing_reason": "scenario_out_of_scope"})
            continue
        row = rows_by_scenario.get(scenario, {}).get(step)
        original_row = original_rows_by_scenario.get(scenario, {}).get(step, {})
        if not row:
            missing_entries.append({**leak, "missing_reason": "v48_row_missing"})
            continue
        patch = shadow_patch_for_row(row)
        veto = patch["fallback_post_selection_veto_shadow"]
        template = patch["template_shadow"]
        reason_counts[str(veto.get("veto_reason") or "unknown")] += 1
        for item in template.get("forbidden_combinations", []) or []:
            forbidden_counts[str(item)] += 1
        patched_entries.append(
            {
                "scenario_id": scenario,
                "step": step,
                "v49_classification": leak.get("classification"),
                "v49_candidate": leak.get("candidate"),
                "v49_selected_source": leak.get("selected_source"),
                "v49_rewrite_fields": leak.get("rewrite_fields", []),
                "v49_safety_reason": leak.get("safety_reason", ""),
                "original_trace_present": bool(original_row),
                "template_shadow_targets": template.get("template_shadow_targets", {}),
                "template_mode": template.get("mode", {}),
                "template_forbidden_combinations": template.get("forbidden_combinations", []),
                "would_reduce_profile_conflict": patch["would_reduce_profile_conflict"],
                "fallback_post_selection_veto_shadow": veto,
                "would_block_hard_safety_leak": patch["would_block_hard_safety_leak"],
                "metadata_gap": patch["metadata_gap"],
            }
        )
    expected = len(leak_entries)
    blocked = sum(bool(item["would_block_hard_safety_leak"]) for item in patched_entries)
    metadata_gaps = sum(bool(item["metadata_gap"]) for item in patched_entries) + len(missing_entries)
    all_explained = bool(expected > 0 and len(patched_entries) == expected and not missing_entries)
    all_blocked = bool(all_explained and blocked == expected and metadata_gaps == 0)
    return {
        "artifact": "profile_template_shadow_patch_audit_20260601_v50",
        "scope": "offline_profile_template_shadow_patch_only",
        "source_artifacts": {
            "v49_runtime_audit_json": _rel(v49_runtime_audit_json),
            "original_trace_dir": _rel(original_trace_dir),
            "v48_trace_dir": _rel(v48_trace_dir),
        },
        "scenarios": list(failure_scenarios),
        "online_llm_called": False,
        "new_rollout_run": False,
        "default_llm_rspc_v2_changed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "expected_leak_step_count": expected,
        "evaluated_leak_step_count": len(patched_entries),
        "missing_leak_step_count": len(missing_entries),
        "would_block_hard_safety_leak_count": blocked,
        "metadata_gap_count": metadata_gaps,
        "would_reduce_profile_conflict_count": sum(
            bool(item["would_reduce_profile_conflict"]) for item in patched_entries
        ),
        "fallback_veto_reason_counts": dict(sorted(reason_counts.items())),
        "template_forbidden_combination_counts": dict(sorted(forbidden_counts.items())),
        "all_leak_steps_explained": all_explained,
        "all_leak_steps_shadow_blocked": all_blocked,
        "patched_leak_entries": patched_entries,
        "missing_leak_entries": missing_entries,
    }


def build_readiness(audit: Mapping[str, Any]) -> dict[str, Any]:
    passed = bool(audit.get("all_leak_steps_explained") and audit.get("all_leak_steps_shadow_blocked"))
    return {
        "artifact": "metadata_replay_readiness_checklist_20260601_v50",
        "current_stage": "v50_minimal_profile_template_shadow_patch",
        "profile_template_shadow_patch_done": bool(audit.get("evaluated_leak_step_count", 0)),
        "profile_template_shadow_patch_pass": passed,
        "expected_leak_step_count": int(audit.get("expected_leak_step_count", 0) or 0),
        "evaluated_leak_step_count": int(audit.get("evaluated_leak_step_count", 0) or 0),
        "would_block_hard_safety_leak_count": int(audit.get("would_block_hard_safety_leak_count", 0) or 0),
        "metadata_gap_count": int(audit.get("metadata_gap_count", 0) or 0),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": "minimal_profile_template_opt_in_shadow_rollout_plan"
        if passed
        else "profile_candidate_metadata_instrumentation_patch_plan",
        "stop_taxonomy": [] if passed else _stop_taxonomy(audit),
    }


def _stop_taxonomy(audit: Mapping[str, Any]) -> list[str]:
    taxonomy: list[str] = []
    if int(audit.get("missing_leak_step_count", 0) or 0):
        taxonomy.append("leak_step_trace_missing")
    if int(audit.get("metadata_gap_count", 0) or 0):
        taxonomy.append("candidate_metadata_missing")
    if int(audit.get("would_block_hard_safety_leak_count", 0) or 0) < int(audit.get("expected_leak_step_count", 0) or 0):
        taxonomy.append("fallback_veto_shadow_incomplete")
    return sorted(set(taxonomy or ["template_shadow_patch_inconclusive"]))


def _write_json_md(path_json: Path, path_md: Path, data: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    lines = [f"# {title}", ""]
    for key in (
        "artifact",
        "scope",
        "current_stage",
        "expected_leak_step_count",
        "evaluated_leak_step_count",
        "would_block_hard_safety_leak_count",
        "metadata_gap_count",
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
        "fallback_veto_reason_counts",
        "template_forbidden_combination_counts",
        "stop_taxonomy",
    ):
        if key in data:
            lines.extend(["", f"## {key}", "```json", json.dumps(data[key], indent=2, sort_keys=True), "```"])
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(
    *,
    v49_runtime_audit_json: str | Path = V49_RUNTIME_AUDIT_JSON,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v48_trace_dir: str | Path = V48_TRACE_DIR,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit = build_shadow_patch_audit(
        v49_runtime_audit_json=v49_runtime_audit_json,
        original_trace_dir=original_trace_dir,
        v48_trace_dir=v48_trace_dir,
        failure_scenarios=failure_scenarios,
    )
    readiness = build_readiness(audit)
    _write_json_md(AUDIT_JSON, AUDIT_MD, audit, "v50 Profile Template Shadow Patch Audit")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v50 Metadata Replay Readiness")
    return audit, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v49-runtime-audit-json", default=str(V49_RUNTIME_AUDIT_JSON))
    parser.add_argument("--original-trace-dir", default=str(ORIGINAL_TRACE_DIR))
    parser.add_argument("--v48-trace-dir", default=str(V48_TRACE_DIR))
    parser.add_argument("--failure-scenario", action="append")
    parser.add_argument("--write-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios = args.failure_scenario or list(FAILURE_SCENARIOS)
    audit, readiness = write_all(
        v49_runtime_audit_json=args.v49_runtime_audit_json,
        original_trace_dir=args.original_trace_dir,
        v48_trace_dir=args.v48_trace_dir,
        failure_scenarios=scenarios,
    )
    if not args.write_all:
        print(json.dumps({"audit": audit, "readiness": readiness}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
