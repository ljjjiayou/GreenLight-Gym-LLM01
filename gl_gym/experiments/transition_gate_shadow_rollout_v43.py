"""v43 opt-in transition-gate shadow rollout records and audits.

This module keeps v43 scoped as shadow feasibility evidence. It never enables a
controlled controller, never promotes the default controller, and writes only
isolated benchmark/cache artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ACTION_FIELDS = ("u_heating", "u_co2", "u_screen", "u_ventilation", "u_lighting", "u_shading")
TARGET_FIELDS = ("u_heating", "u_ventilation", "u_screen", "u_shading")
FAILURE_SCENARIOS = (
    "y2010_d180_s43_n720",
    "y2018_d181_s42_n720",
    "y2018_d181_s43_n720",
)
CONTROLLED_CONTROLLER = "llm_rspc_v2_hot_dry_proposer_strict"

AUDIT_DIR = PROJECT_ROOT / "gl_gym" / "result" / "audits"
BENCHMARK_DIR = PROJECT_ROOT / "gl_gym" / "result" / "benchmarks" / "transition_gate_shadow_rollout_v43_20260530"
TRACE_DIR = BENCHMARK_DIR / "traces"
ISOLATED_CACHE_PATH = PROJECT_ROOT / "gl_gym" / "result" / "plan_cache" / "transition_gate_shadow_v43_20260530.json"
ENVELOPE_PATH = PROJECT_ROOT / "gl_gym" / "result" / "audits" / "stable_action_transition_envelope_20260530_v41.json"
ORIGINAL_TRACE_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "stable_long_horizon_evaluation_pool_v38_20260529"
    / "traces_h720"
)
EXECUTION_RECORD_JSON = AUDIT_DIR / "transition_gate_shadow_rollout_execution_record_20260530_v43.json"
EXECUTION_RECORD_MD = AUDIT_DIR / "transition_gate_shadow_rollout_execution_record_20260530_v43.md"
AUDIT_JSON = AUDIT_DIR / "transition_gate_shadow_rollout_result_audit_20260530_v43.json"
AUDIT_MD = AUDIT_DIR / "transition_gate_shadow_rollout_result_audit_20260530_v43.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260530_v43.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260530_v43.md"


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _write_json_md(path_json: Path, path_md: Path, payload: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_md.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [f"# {title}", "", "```json", json.dumps(payload, ensure_ascii=False, indent=2, default=str), "```", ""]
    path_md.write_text("\n".join(lines), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _trace_path(trace_dir: Path, scenario_id: str, controller: str = "llm_rspc_v2") -> Path:
    return trace_dir / f"{scenario_id}_{controller}.csv"


def _runtime_failure_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    for row in rows:
        text = " ".join(str(row.get(key, "") or "") for key in ("runtime_error", "runtime_error_type", "replan_reason"))
        if "CV_CONV_FAILURE" in text or "CVODES" in text or "CvodesInterface" in text:
            return int(_num(row.get("step"), -1))
    if rows and bool(rows[-1].get("runtime_error")):
        return int(_num(rows[-1].get("step"), len(rows) - 1))
    return None


def _action(row: Mapping[str, Any]) -> dict[str, float]:
    return {field: _num(row.get(field)) for field in ACTION_FIELDS}


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    previous: dict[str, float] = {}
    signs: dict[str, int] = {}
    large_delta = 0
    oscillations = 0
    screen_vent = 0
    max_abs_delta = 0.0
    for row in rows:
        action = _action(row)
        if _num(action.get("u_ventilation")) >= 0.85 and _num(action.get("u_screen")) >= 0.30:
            screen_vent += 1
        for field in ACTION_FIELDS:
            value = _num(action.get(field))
            if field in previous:
                delta = value - previous[field]
                max_abs_delta = max(max_abs_delta, abs(delta))
                if abs(delta) > 0.20:
                    large_delta += 1
                sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
                if sign and signs.get(field, 0) and sign != signs[field]:
                    oscillations += 1
                if sign:
                    signs[field] = sign
            previous[field] = value
    return {
        "steps": len(rows),
        "large_action_delta_count": int(large_delta),
        "action_oscillation_count": int(oscillations),
        "screen_vent_conflict_count": int(screen_vent),
        "max_abs_delta": float(max_abs_delta),
        "transition_gate_applied_steps": int(sum(_truthy(row.get("transition_gate_applied")) for row in rows)),
        "transition_gate_bypassed_steps": int(sum(_truthy(row.get("transition_gate_bypassed")) for row in rows)),
        "transition_gate_delta_limited_count": int(sum(int(_num(row.get("transition_gate_delta_limited_count"))) for row in rows)),
        "transition_gate_reversal_projected_count": int(
            sum(int(_num(row.get("transition_gate_reversal_projected_count"))) for row in rows)
        ),
        "hard_safety_risk_rows": int(
            sum(
                _truthy(row.get("final_action_would_fail_canopy_boundary_v2"))
                or _truthy(row.get("final_action_predicted_canopy_lt0_v2"))
                for row in rows
            )
        ),
    }


def _failure_taxonomy(report: Mapping[str, Any]) -> list[str]:
    taxonomy: list[str] = []
    if report.get("new_runtime_failure_step") is not None:
        if report.get("new_runtime_failure_not_earlier_than_original", False):
            taxonomy.append("gate_insufficient")
        else:
            taxonomy.append("simulator_boundary_pressure")
    new_metrics = report.get("new_metrics", {}) or {}
    original_metrics = report.get("original_metrics", {}) or {}
    if int(new_metrics.get("action_oscillation_count", 0)) >= int(original_metrics.get("action_oscillation_count", 0)):
        taxonomy.append("profile_candidate_conflict")
    if int(new_metrics.get("transition_gate_bypassed_steps", 0)) > 0:
        taxonomy.append("guardrail_pressure")
    return sorted(set(taxonomy))


def build_execution_record() -> dict[str, Any]:
    overrides = {
        "transition_gate_enabled": True,
        "transition_gate_soft_limit_path": _rel(ENVELOPE_PATH),
        "transition_gate_reversal_window_steps": 6,
        "transition_gate_hard_safety_bypass": True,
    }
    overrides_json = json.dumps(overrides, separators=(",", ":"))
    common = [
        "python",
        "gl_gym/experiments/run_frozen_benchmark.py",
        "--controllers",
        "llm_rspc_v2",
        "--max-steps",
        "720",
        "--plan-cache-mode",
        "record",
        "--plan-cache-path",
        _rel(ISOLATED_CACHE_PATH),
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--agent-config-overrides",
        overrides_json,
        "--output-trace-dir",
        _rel(TRACE_DIR),
    ]
    group_a = common + [
        "--years",
        "2010",
        "--days",
        "180",
        "--seeds",
        "43",
        "--output-json",
        _rel(BENCHMARK_DIR / "transition_gate_shadow_rollout_v43_group_a.json"),
    ]
    group_b = common + [
        "--years",
        "2018",
        "--days",
        "181",
        "--seeds",
        "42,43",
        "--output-json",
        _rel(BENCHMARK_DIR / "transition_gate_shadow_rollout_v43_group_b.json"),
    ]
    return {
        "schema_version": "transition_gate_shadow_rollout_execution_record_v43",
        "authorization_source": "user_delegated_v43_minimal_shadow_rollout_authorization_20260530",
        "scope": "minimal_transition_gate_shadow_rollout_only",
        "scenario_ids": list(FAILURE_SCENARIOS),
        "controller": "llm_rspc_v2",
        "controlled_controller_present": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "default_llm_rspc_v2_changed": False,
        "qwen_3_7_migration_allowed": False,
        "online_llm_allowed": True,
        "online_llm_scope": "qwen_max_latest_for_three_fixed_v43_shadow_rollouts_only",
        "plan_cache_mode": "record",
        "plan_cache_key_policy": "scenario_timestep",
        "isolated_cache_path": _rel(ISOLATED_CACHE_PATH),
        "selected_cache_overwrite_allowed": False,
        "output_dir": _rel(BENCHMARK_DIR),
        "trace_dir": _rel(TRACE_DIR),
        "agent_config_overrides": overrides,
        "soft_limit_path_exists": ENVELOPE_PATH.exists(),
        "command_groups": [
            {"name": "group_a", "scenario_ids": ["y2010_d180_s43_n720"], "argv": group_a},
            {"name": "group_b", "scenario_ids": ["y2018_d181_s42_n720", "y2018_d181_s43_n720"], "argv": group_b},
        ],
        "precheck_pass": bool(ENVELOPE_PATH.exists() and CONTROLLED_CONTROLLER not in "llm_rspc_v2"),
    }


def build_result_audit(trace_dir: Path = TRACE_DIR, original_trace_dir: Path = ORIGINAL_TRACE_DIR) -> dict[str, Any]:
    scenario_reports: list[dict[str, Any]] = []
    for scenario_id in FAILURE_SCENARIOS:
        original_rows = _read_rows(_trace_path(original_trace_dir, scenario_id))
        new_rows = _read_rows(_trace_path(trace_dir, scenario_id))
        original_failure_step = _runtime_failure_step(original_rows)
        new_failure_step = _runtime_failure_step(new_rows)
        report = {
            "scenario_id": scenario_id,
            "original_trace_path": _rel(_trace_path(original_trace_dir, scenario_id)),
            "new_trace_path": _rel(_trace_path(trace_dir, scenario_id)),
            "original_trace_exists": bool(original_rows),
            "new_trace_exists": bool(new_rows),
            "original_runtime_failure_step": original_failure_step,
            "new_runtime_failure_step": new_failure_step,
            "new_runtime_failure_not_earlier_than_original": bool(
                new_failure_step is None
                or original_failure_step is None
                or int(new_failure_step) >= int(original_failure_step)
            ),
            "original_metrics": _metrics(original_rows),
            "new_metrics": _metrics(new_rows),
        }
        report["acceptance"] = {
            "trace_exists": bool(new_rows),
            "runtime_error_steps_zero_or_not_earlier": bool(new_rows)
            and report["new_runtime_failure_not_earlier_than_original"],
            "large_action_delta_reduced": bool(new_rows)
            and int(report["new_metrics"]["large_action_delta_count"])
            < int(report["original_metrics"]["large_action_delta_count"]),
            "action_oscillation_reduced": bool(new_rows)
            and int(report["new_metrics"]["action_oscillation_count"])
            < int(report["original_metrics"]["action_oscillation_count"]),
            "screen_vent_conflict_not_increased": bool(new_rows)
            and int(report["new_metrics"]["screen_vent_conflict_count"])
            <= int(report["original_metrics"]["screen_vent_conflict_count"]),
            "no_new_canopy_dew_hard_violation": bool(new_rows)
            and int(report["new_metrics"]["hard_safety_risk_rows"])
            <= int(report["original_metrics"]["hard_safety_risk_rows"]),
        }
        report["failure_taxonomy"] = _failure_taxonomy(report)
        scenario_reports.append(report)

    all_acceptance = {
        "trace_exists": all(item["acceptance"]["trace_exists"] for item in scenario_reports),
        "runtime_error_steps_zero_or_not_earlier": all(
            item["acceptance"]["runtime_error_steps_zero_or_not_earlier"] for item in scenario_reports
        ),
        "large_action_delta_reduced": all(item["acceptance"]["large_action_delta_reduced"] for item in scenario_reports),
        "action_oscillation_reduced": all(item["acceptance"]["action_oscillation_reduced"] for item in scenario_reports),
        "screen_vent_conflict_not_increased": all(
            item["acceptance"]["screen_vent_conflict_not_increased"] for item in scenario_reports
        ),
        "no_new_canopy_dew_hard_violation": all(
            item["acceptance"]["no_new_canopy_dew_hard_violation"] for item in scenario_reports
        ),
    }
    return {
        "schema_version": "transition_gate_shadow_rollout_result_audit_v43",
        "scenario_reports": scenario_reports,
        "acceptance": all_acceptance,
        "transition_gate_shadow_rollout_pass": bool(all(all_acceptance.values())),
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": "transition_gate_h720_shadow_expansion_plan"
        if bool(all(all_acceptance.values()))
        else "profile_candidate_guardrail_compatibility_audit",
    }


def build_readiness(execution_record: Mapping[str, Any] | None = None, result_audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    execution_record = dict(execution_record or {})
    result_audit = dict(result_audit or {})
    runnable = bool(execution_record.get("precheck_pass", False))
    result_pass = bool(result_audit.get("transition_gate_shadow_rollout_pass", False))
    if result_audit:
        next_action = result_audit.get("next_action")
    else:
        next_action = "execute_v43_minimal_transition_gate_shadow_rollout" if runnable else "v43_precheck_failure"
    return {
        "schema_version": "metadata_replay_readiness_checklist_20260530_v43",
        "stage": "v43_minimal_shadow_rollout_with_transition_gate",
        "transition_gate_shadow_rollout_authorized": True,
        "transition_gate_shadow_rollout_executable": runnable,
        "transition_gate_shadow_rollout_pass": result_pass,
        "scenario_ids": list(FAILURE_SCENARIOS),
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "default_llm_rspc_v2_changed": False,
        "selected_cache_overwrite_allowed": False,
        "next_action": next_action,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/audit v43 transition-gate shadow rollout artifacts.")
    parser.add_argument("--write-execution-record", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--write-readiness", action="store_true")
    parser.add_argument("--trace-dir", type=str, default=_rel(TRACE_DIR))
    parser.add_argument("--original-trace-dir", type=str, default=_rel(ORIGINAL_TRACE_DIR))
    args = parser.parse_args()

    execution_record = build_execution_record()
    result_audit: dict[str, Any] = {}

    if args.write_execution_record:
        _write_json_md(EXECUTION_RECORD_JSON, EXECUTION_RECORD_MD, execution_record, "v43 Transition Gate Shadow Rollout Execution Record")
    if args.audit:
        result_audit = build_result_audit(PROJECT_ROOT / args.trace_dir, PROJECT_ROOT / args.original_trace_dir)
        _write_json_md(AUDIT_JSON, AUDIT_MD, result_audit, "v43 Transition Gate Shadow Rollout Result Audit")
    elif AUDIT_JSON.exists():
        result_audit = _load_json(AUDIT_JSON)
    if args.write_readiness:
        readiness = build_readiness(execution_record, result_audit)
        _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v43 Metadata Replay Readiness Checklist")
    if not (args.write_execution_record or args.audit or args.write_readiness):
        print(json.dumps(build_readiness(execution_record, result_audit), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
