"""v48 minimal profile-feasibility-gate shadow rollout artifacts.

This module writes the v48 execution record before rollout and audits the
fixed three-scenario shadow rollout afterwards. It does not authorize
controlled replay, promotion, or performance claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (
    AUDIT_DIR,
    FAILURE_SCENARIOS,
    ORIGINAL_TRACE_DIR,
    _read_rows,
    _runtime_failure_step,
    _trace_path,
)
from gl_gym.experiments.diagnose_ppo_vs_llm import derive_candidate_guardrail_compatibility


BENCHMARK_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "profile_feasibility_gate_shadow_rollout_v48_20260531"
)
V48_TRACE_DIR = BENCHMARK_DIR / "traces"
V48_CACHE_PATH = PROJECT_ROOT / "gl_gym" / "result" / "plan_cache" / "profile_feasibility_gate_shadow_v48_20260531.json"
EXECUTION_RECORD_JSON = AUDIT_DIR / "profile_feasibility_gate_shadow_rollout_v48_execution_record_20260531.json"
EXECUTION_RECORD_MD = AUDIT_DIR / "profile_feasibility_gate_shadow_rollout_v48_execution_record_20260531.md"
RESULT_AUDIT_JSON = AUDIT_DIR / "profile_feasibility_gate_shadow_rollout_result_audit_20260531_v48.json"
RESULT_AUDIT_MD = AUDIT_DIR / "profile_feasibility_gate_shadow_rollout_result_audit_20260531_v48.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260531_v48.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260531_v48.md"

AGENT_OVERRIDES = {
    "profile_feasibility_gate_enabled": True,
    "profile_feasibility_gate_hard_safety_veto": True,
    "transition_gate_enabled": False,
}
SCENARIO_GROUPS = [
    {"name": "group_a", "years": "2010", "days": "180", "seeds": "43"},
    {"name": "group_b", "years": "2018", "days": "181", "seeds": "42,43"},
]


def _rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except Exception:
        return str(path)


def _has_dotenv_key(key: str) -> bool:
    if os.getenv(key):
        return True
    dotenv = PROJECT_ROOT / ".env"
    if not dotenv.exists():
        return False
    try:
        for raw in dotenv.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key and value.strip().strip('"').strip("'"):
                return True
    except Exception:
        return False
    return False


def _command_for_group(group: Mapping[str, str]) -> list[str]:
    output_json = BENCHMARK_DIR / f"profile_feasibility_gate_shadow_rollout_v48_{group['name']}.json"
    return [
        "python",
        "gl_gym\\experiments\\run_frozen_benchmark.py",
        "--years",
        str(group["years"]),
        "--days",
        str(group["days"]),
        "--seeds",
        str(group["seeds"]),
        "--controllers",
        "llm_rspc_v2",
        "--max-steps",
        "720",
        "--plan-cache-mode",
        "record",
        "--plan-cache-path",
        _rel(V48_CACHE_PATH),
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--agent-config-overrides",
        json.dumps(AGENT_OVERRIDES, sort_keys=True),
        "--output-json",
        _rel(output_json),
        "--output-trace-dir",
        _rel(V48_TRACE_DIR),
    ]


def build_execution_record() -> dict[str, Any]:
    online_llm_credentials_present = _has_dotenv_key("BAILIAN_API_KEY")
    commands = [
        {
            "group": group["name"],
            "years": group["years"],
            "days": group["days"],
            "seeds": group["seeds"],
            "argv": _command_for_group(group),
            "command": " ".join(_command_for_group(group)),
        }
        for group in SCENARIO_GROUPS
    ]
    return {
        "artifact": "profile_feasibility_gate_shadow_rollout_v48_execution_record_20260531",
        "authorization_source": "user_delegated_profile_feasibility_gate_shadow_rollout_authorization_20260531",
        "scope": "minimal_profile_feasibility_gate_shadow_rollout_only",
        "scenarios": list(FAILURE_SCENARIOS),
        "controller": "llm_rspc_v2",
        "controlled_controller_included": False,
        "max_steps": 720,
        "transition_gate_enabled": False,
        "profile_feasibility_gate_enabled": True,
        "profile_feasibility_gate_hard_safety_veto": True,
        "plan_cache_mode": "record",
        "cache_path": _rel(V48_CACHE_PATH),
        "output_dir": _rel(BENCHMARK_DIR),
        "trace_dir": _rel(V48_TRACE_DIR),
        "online_llm_allowed": True,
        "online_llm_scope": "only these three v48 shadow rollout scenarios",
        "online_llm_credentials_present": bool(online_llm_credentials_present),
        "weather_model_change_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "executable": bool(online_llm_credentials_present),
        "blocked_reason": "" if online_llm_credentials_present else "online_llm_credentials_missing",
        "commands": commands,
    }


def write_execution_record(record: Mapping[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    V48_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    EXECUTION_RECORD_JSON.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# v48 Profile Feasibility Gate Shadow Rollout Execution Record",
        "",
        f"- executable: `{str(record.get('executable')).lower()}`",
        f"- scope: `{record.get('scope')}`",
        f"- scenarios: `{', '.join(record.get('scenarios', []))}`",
        f"- controller: `{record.get('controller')}`",
        f"- cache_path: `{record.get('cache_path')}`",
        f"- online_llm_allowed: `{str(record.get('online_llm_allowed')).lower()}`",
        f"- credentials_present: `{str(record.get('online_llm_credentials_present')).lower()}`",
        f"- controlled_replay_allowed: `{str(record.get('controlled_replay_allowed')).lower()}`",
        f"- performance_claim_allowed: `{str(record.get('performance_claim_allowed')).lower()}`",
        "",
        "## Commands",
    ]
    for item in record.get("commands", []):
        lines.extend(["", f"### {item.get('group')}", "", "```powershell", str(item.get("command", "")), "```"])
    EXECUTION_RECORD_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_trace_rows(trace_dir: Path, scenario: str) -> list[dict[str, Any]]:
    path = _trace_path(trace_dir, scenario)
    if not path.exists():
        return []
    return _read_rows(path)


def _compat(row: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    merged.update(derive_candidate_guardrail_compatibility(dict(row)))
    return merged


def _count_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged = [_compat(row) for row in rows]
    profile_conflicts = [
        row for row in merged if str(row.get("profile_action_conflict_label", "none") or "none") != "none"
    ]
    issues = [
        row
        for row in merged
        if str(row.get("candidate_guardrail_compatibility_label", "compatible") or "compatible") != "compatible"
        or str(row.get("profile_action_conflict_label", "none") or "none") != "none"
        or bool(str(row.get("candidate_filter_reason_summary", "") or "").strip())
    ]
    major_or_hard = [
        row
        for row in merged
        if str(row.get("candidate_guardrail_compatibility_label", "") or "") in {"major_rewrite", "hard_safety_rewrite"}
    ]
    hard = [
        row
        for row in merged
        if str(row.get("candidate_guardrail_compatibility_label", "") or "") == "hard_safety_rewrite"
    ]
    runtime = [row for row in merged if str(row.get("runtime_error", "") or "").strip()]
    return {
        "row_count": int(len(merged)),
        "runtime_error_steps": int(len(runtime)),
        "first_runtime_error_step": _runtime_failure_step(list(merged)),
        "profile_action_conflict_steps": int(len(profile_conflicts)),
        "candidate_guardrail_issue_steps": int(len(issues)),
        "major_or_hard_rewrite_predicted_steps": int(len(major_or_hard)),
        "hard_safety_rewrite_steps": int(len(hard)),
        "profile_feasibility_gate_enabled_steps": int(sum(bool(row.get("profile_feasibility_gate_enabled", False)) for row in merged)),
        "profile_feasibility_gate_applied_steps": int(sum(bool(row.get("profile_feasibility_gate_applied", False)) for row in merged)),
        "profile_feasibility_gate_hard_safety_veto_count": int(
            sum(int(float(row.get("profile_feasibility_gate_hard_safety_veto_count", 0) or 0)) for row in merged)
        ),
    }


def build_result_audit(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v48_trace_dir: str | Path = V48_TRACE_DIR,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    original_root = Path(original_trace_dir)
    v48_root = Path(v48_trace_dir)
    scenario_reports = []
    for scenario in scenarios:
        original_rows = _read_trace_rows(original_root, scenario)
        v48_rows = _read_trace_rows(v48_root, scenario)
        original_full = _count_metrics(original_rows)
        v48_full = _count_metrics(v48_rows)
        original_failure = original_full.get("first_runtime_error_step")
        shadow_failure = v48_full.get("first_runtime_error_step")
        compare_len = int(original_failure) + 1 if original_failure is not None else len(original_rows)
        compare_len = max(0, compare_len)
        original = _count_metrics(original_rows[:compare_len])
        shadow = _count_metrics(v48_rows[:compare_len])
        not_earlier = (
            shadow_failure is None
            or original_failure is None
            or int(shadow_failure) >= int(original_failure)
        )
        report = {
            "scenario_id": scenario,
            "original_trace_exists": bool(original_rows),
            "v48_trace_exists": bool(v48_rows),
            "comparison_window_steps": int(compare_len),
            "original_full": original_full,
            "v48_full": v48_full,
            "original": original,
            "v48": shadow,
            "failure_not_earlier_than_v39": bool(not_earlier),
            "profile_action_conflict_reduced": bool(
                shadow.get("profile_action_conflict_steps", 10**9)
                < original.get("profile_action_conflict_steps", -1)
            ),
            "candidate_guardrail_issue_reduced": bool(
                shadow.get("candidate_guardrail_issue_steps", 10**9)
                < original.get("candidate_guardrail_issue_steps", -1)
            ),
            "major_or_hard_rewrite_not_increased": bool(
                shadow.get("major_or_hard_rewrite_predicted_steps", 10**9)
                <= original.get("major_or_hard_rewrite_predicted_steps", -1)
            ),
            "hard_safety_rewrite_preferred_new_steps": int(
                max(0, shadow.get("hard_safety_rewrite_steps", 0) - original.get("hard_safety_rewrite_steps", 0))
            ),
        }
        scenario_reports.append(report)

    totals = {
        "original_runtime_error_steps": sum(int(r["original"].get("runtime_error_steps", 0)) for r in scenario_reports),
        "v48_runtime_error_steps": sum(int(r["v48_full"].get("runtime_error_steps", 0)) for r in scenario_reports),
        "original_profile_action_conflict_steps": sum(int(r["original"].get("profile_action_conflict_steps", 0)) for r in scenario_reports),
        "v48_profile_action_conflict_steps": sum(int(r["v48"].get("profile_action_conflict_steps", 0)) for r in scenario_reports),
        "original_candidate_guardrail_issue_steps": sum(int(r["original"].get("candidate_guardrail_issue_steps", 0)) for r in scenario_reports),
        "v48_candidate_guardrail_issue_steps": sum(int(r["v48"].get("candidate_guardrail_issue_steps", 0)) for r in scenario_reports),
        "original_major_or_hard_rewrite_predicted_steps": sum(int(r["original"].get("major_or_hard_rewrite_predicted_steps", 0)) for r in scenario_reports),
        "v48_major_or_hard_rewrite_predicted_steps": sum(int(r["v48"].get("major_or_hard_rewrite_predicted_steps", 0)) for r in scenario_reports),
        "hard_safety_rewrite_preferred_new_steps": sum(int(r.get("hard_safety_rewrite_preferred_new_steps", 0)) for r in scenario_reports),
        "profile_feasibility_gate_applied_steps": sum(int(r["v48"].get("profile_feasibility_gate_applied_steps", 0)) for r in scenario_reports),
        "profile_feasibility_gate_hard_safety_veto_count": sum(
            int(r["v48"].get("profile_feasibility_gate_hard_safety_veto_count", 0)) for r in scenario_reports
        ),
    }
    acceptance = {
        "runtime_error_steps_zero": bool(totals["v48_runtime_error_steps"] == 0),
        "failure_not_earlier_all_scenarios": bool(all(r["failure_not_earlier_than_v39"] for r in scenario_reports)),
        "profile_action_conflict_reduced": bool(
            totals["v48_profile_action_conflict_steps"] < totals["original_profile_action_conflict_steps"]
        ),
        "candidate_guardrail_issue_reduced": bool(
            totals["v48_candidate_guardrail_issue_steps"] < totals["original_candidate_guardrail_issue_steps"]
        ),
        "major_or_hard_rewrite_not_increased": bool(
            totals["v48_major_or_hard_rewrite_predicted_steps"]
            <= totals["original_major_or_hard_rewrite_predicted_steps"]
        ),
        "hard_safety_rewrite_preferred_new_steps_zero": bool(totals["hard_safety_rewrite_preferred_new_steps"] == 0),
    }
    acceptance["v48_acceptance_pass"] = bool(
        (acceptance["runtime_error_steps_zero"] or acceptance["failure_not_earlier_all_scenarios"])
        and acceptance["profile_action_conflict_reduced"]
        and acceptance["candidate_guardrail_issue_reduced"]
        and acceptance["major_or_hard_rewrite_not_increased"]
        and acceptance["hard_safety_rewrite_preferred_new_steps_zero"]
    )
    return {
        "artifact": "profile_feasibility_gate_shadow_rollout_result_audit_20260531_v48",
        "scope": "minimal_profile_feasibility_gate_shadow_rollout_only",
        "scenarios": list(scenarios),
        "original_trace_dir": _rel(original_root),
        "v48_trace_dir": _rel(v48_root),
        "scenario_reports": scenario_reports,
        "totals": totals,
        "acceptance": acceptance,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def build_readiness(result_audit: Mapping[str, Any], execution_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    passed = bool(acceptance.get("v48_acceptance_pass", False))
    return {
        "artifact": "metadata_replay_readiness_checklist_20260531_v48",
        "current_stage": "v48_minimal_profile_feasibility_gate_shadow_rollout",
        "profile_feasibility_gate_shadow_rollout_done": bool(result_audit.get("scenario_reports")),
        "profile_feasibility_gate_shadow_rollout_pass": bool(passed),
        "execution_record_executable": bool((execution_record or {}).get("executable", False)),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": (
            "profile_feasibility_gate_h720_shadow_expansion_plan"
            if passed
            else "profile_candidate_instrumentation_or_template_fix"
        ),
        "stop_taxonomy": [] if passed else _stop_taxonomy(result_audit),
    }


def _stop_taxonomy(result_audit: Mapping[str, Any]) -> list[str]:
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    taxonomy: list[str] = []
    if not acceptance.get("runtime_error_steps_zero", False) and not acceptance.get("failure_not_earlier_all_scenarios", False):
        taxonomy.append("cvodes_earlier_failure")
    if not acceptance.get("profile_action_conflict_reduced", False):
        taxonomy.append("profile_repair_not_applied")
    if not acceptance.get("candidate_guardrail_issue_reduced", False):
        taxonomy.append("candidate_guardrail_pressure_persists")
    if not acceptance.get("major_or_hard_rewrite_not_increased", False):
        taxonomy.append("hard_safety_veto_failed")
    if not acceptance.get("hard_safety_rewrite_preferred_new_steps_zero", False):
        taxonomy.append("hard_safety_veto_failed")
    return sorted(set(taxonomy or ["gate_insufficient"]))


def write_result_artifacts(result_audit: Mapping[str, Any], readiness: Mapping[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_AUDIT_JSON.write_text(json.dumps(result_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    READINESS_JSON.write_text(json.dumps(readiness, ensure_ascii=False, indent=2), encoding="utf-8")
    totals = result_audit.get("totals", {}) if isinstance(result_audit.get("totals", {}), Mapping) else {}
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    lines = [
        "# v48 Profile Feasibility Gate Shadow Rollout Result Audit",
        "",
        f"- acceptance_pass: `{str(acceptance.get('v48_acceptance_pass', False)).lower()}`",
        f"- v48_runtime_error_steps: `{totals.get('v48_runtime_error_steps', 0)}`",
        f"- profile_action_conflict_steps: `{totals.get('original_profile_action_conflict_steps', 0)} -> {totals.get('v48_profile_action_conflict_steps', 0)}`",
        f"- candidate_guardrail_issue_steps: `{totals.get('original_candidate_guardrail_issue_steps', 0)} -> {totals.get('v48_candidate_guardrail_issue_steps', 0)}`",
        f"- major_or_hard_rewrite_predicted_steps: `{totals.get('original_major_or_hard_rewrite_predicted_steps', 0)} -> {totals.get('v48_major_or_hard_rewrite_predicted_steps', 0)}`",
        f"- hard_safety_rewrite_preferred_new_steps: `{totals.get('hard_safety_rewrite_preferred_new_steps', 0)}`",
        f"- controlled_replay_allowed: `{str(result_audit.get('controlled_replay_allowed')).lower()}`",
        f"- performance_claim_allowed: `{str(result_audit.get('performance_claim_allowed')).lower()}`",
        "",
        "## Scenario Reports",
    ]
    for report in result_audit.get("scenario_reports", []):
        lines.append(
            f"- `{report.get('scenario_id')}`: runtime `{report.get('original_full', {}).get('first_runtime_error_step')} -> "
            f"{report.get('v48_full', {}).get('first_runtime_error_step')}`, "
            f"profile conflicts `{report.get('original', {}).get('profile_action_conflict_steps')} -> "
            f"{report.get('v48', {}).get('profile_action_conflict_steps')}`"
        )
    RESULT_AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    readiness_lines = [
        "# v48 Metadata Replay Readiness",
        "",
        f"- profile_feasibility_gate_shadow_rollout_pass: `{str(readiness.get('profile_feasibility_gate_shadow_rollout_pass')).lower()}`",
        f"- next_action: `{readiness.get('next_action')}`",
        f"- controlled_replay_allowed: `{str(readiness.get('controlled_replay_allowed')).lower()}`",
        f"- performance_claim_allowed: `{str(readiness.get('performance_claim_allowed')).lower()}`",
        f"- promotion_evidence: `{str(readiness.get('promotion_evidence')).lower()}`",
    ]
    if readiness.get("stop_taxonomy"):
        readiness_lines.append(f"- stop_taxonomy: `{', '.join(readiness.get('stop_taxonomy', []))}`")
    READINESS_MD.write_text("\n".join(readiness_lines) + "\n", encoding="utf-8")


def write_all() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    execution = build_execution_record()
    write_execution_record(execution)
    result = build_result_audit()
    readiness = build_readiness(result, execution_record=execution)
    write_result_artifacts(result, readiness)
    return execution, result, readiness


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-record-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)

    execution = build_execution_record()
    if not args.audit_only:
        write_execution_record(execution)
        print(f"Wrote {EXECUTION_RECORD_JSON}")
    if not args.execution_record_only:
        result = build_result_audit()
        readiness = build_readiness(result, execution_record=execution)
        write_result_artifacts(result, readiness)
        print(f"Wrote {RESULT_AUDIT_JSON}")
        print(f"Wrote {READINESS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
