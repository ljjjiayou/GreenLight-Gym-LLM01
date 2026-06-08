"""v56 qwen3.7 recovery-anchor opt-in shadow rollout artifacts.

This is a narrow shadow validation for the v55 recovery-anchor design. It keeps
the default llm_rspc_v2 path unchanged, runs only the three fixed H720 failure
scenarios when online precheck passes, writes an isolated cache, and never
authorizes controlled replay, promotion, or performance claims.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gl_gym.experiments.profile_template_opt_in_shadow_rollout_v51 as v51  # noqa: E402
import gl_gym.experiments.profile_template_opt_in_shadow_rollout_v52 as v52  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (  # noqa: E402
    AUDIT_DIR,
    FAILURE_SCENARIOS,
    ORIGINAL_TRACE_DIR,
)
from gl_gym.experiments.profile_feasibility_gate_shadow_rollout_v48 import V48_TRACE_DIR  # noqa: E402


MODEL_NAME = "qwen3.7-max"
BENCHMARK_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "qwen37_recovery_anchor_opt_in_shadow_rollout_v56_20260601"
)
V56_TRACE_DIR = BENCHMARK_DIR / "traces"
V56_CACHE_PATH = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "plan_cache"
    / "qwen37_recovery_anchor_opt_in_shadow_v56_20260601.json"
)
V55_READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v55.json"
V52_RESULT_AUDIT_JSON = AUDIT_DIR / "profile_template_opt_in_shadow_rollout_v52_qwen37_result_audit_20260601.json"

ONLINE_PRECHECK_JSON = AUDIT_DIR / "online_llm_accessibility_precheck_result_v56_qwen37_20260601.json"
ONLINE_PRECHECK_MD = AUDIT_DIR / "online_llm_accessibility_precheck_result_v56_qwen37_20260601.md"
EXECUTION_RECORD_JSON = AUDIT_DIR / "qwen37_recovery_anchor_opt_in_shadow_rollout_v56_execution_record_20260601.json"
EXECUTION_RECORD_MD = AUDIT_DIR / "qwen37_recovery_anchor_opt_in_shadow_rollout_v56_execution_record_20260601.md"
RESULT_AUDIT_JSON = AUDIT_DIR / "qwen37_recovery_anchor_opt_in_shadow_rollout_result_audit_20260601_v56.json"
RESULT_AUDIT_MD = AUDIT_DIR / "qwen37_recovery_anchor_opt_in_shadow_rollout_result_audit_20260601_v56.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v56.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v56.md"

AGENT_OVERRIDES = dict(v52.AGENT_OVERRIDES)
AGENT_OVERRIDES.update(
    {
        "profile_template_patch_enabled": True,
        "fallback_post_selection_veto_enabled": True,
        "recovery_anchor_enabled": True,
        "recovery_anchor_source": "tomato_safety_projected_anchor",
        "recovery_anchor_record_provenance": True,
        "transition_gate_enabled": False,
        "profile_feasibility_gate_enabled": False,
    }
)
SCENARIO_GROUPS = list(v52.SCENARIO_GROUPS)


def _rel(path: str | Path) -> str:
    return v51._rel(path)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json_md(path_json: Path, path_md: Path, data: Mapping[str, Any], title: str) -> None:
    v51._write_json_md(path_json, path_md, data, title)


def build_online_precheck(*, probe_online: bool = True, model_name: str = MODEL_NAME) -> dict[str, Any]:
    result = v52.build_online_precheck(probe_online=probe_online, model_name=model_name)
    result["artifact"] = "online_llm_accessibility_precheck_result_v56_qwen37_20260601"
    result["model_name"] = model_name
    return result


def _v55_passed(v55_readiness_json: str | Path = V55_READINESS_JSON) -> bool:
    readiness = _load_json(v55_readiness_json)
    no_recovery = readiness.get("no_recovery_anchor_available_count", 1)
    return bool(
        readiness.get("v55_recovery_anchor_design_complete", False)
        and float(readiness.get("recovery_anchor_coverage_rate", 0.0) or 0.0) >= 1.0
        and int(no_recovery if no_recovery is not None else 1) == 0
    )


def _command_for_group(group: Mapping[str, str]) -> list[str]:
    output_json = BENCHMARK_DIR / f"qwen37_recovery_anchor_opt_in_shadow_rollout_v56_{group['name']}.json"
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
        "--llm-model",
        MODEL_NAME,
        "--plan-cache-mode",
        "record",
        "--plan-cache-path",
        _rel(V56_CACHE_PATH),
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--agent-config-overrides",
        json.dumps(AGENT_OVERRIDES, sort_keys=True),
        "--output-json",
        _rel(output_json),
        "--output-trace-dir",
        _rel(V56_TRACE_DIR),
    ]


def build_execution_record(
    precheck: Mapping[str, Any] | None = None,
    *,
    v55_readiness_json: str | Path = V55_READINESS_JSON,
) -> dict[str, Any]:
    precheck = dict(precheck or build_online_precheck(probe_online=True))
    v55_pass = _v55_passed(v55_readiness_json)
    model_matches = str(precheck.get("model_name", "")) == MODEL_NAME
    executable = bool(v55_pass and model_matches and precheck.get("online_llm_accessible", False))
    blocked_reason = ""
    if not v55_pass:
        blocked_reason = "v55_recovery_anchor_design_not_passed"
    elif not model_matches:
        blocked_reason = "model_name_mismatch"
    elif not precheck.get("online_llm_accessible", False):
        blocked_reason = str(precheck.get("blocked_reason") or "online_llm_accessibility_blocked")
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
        "artifact": "qwen37_recovery_anchor_opt_in_shadow_rollout_v56_execution_record_20260601",
        "authorization_source": "user_delegated_qwen37_recovery_anchor_opt_in_shadow_rollout_authorization_20260601",
        "scope": "minimal_qwen37_recovery_anchor_opt_in_shadow_rollout_only",
        "model_name": MODEL_NAME,
        "scenarios": list(FAILURE_SCENARIOS),
        "controller": "llm_rspc_v2",
        "controlled_controller_included": False,
        "max_steps": 720,
        "agent_config_overrides": dict(AGENT_OVERRIDES),
        "profile_template_patch_enabled": True,
        "fallback_post_selection_veto_enabled": True,
        "recovery_anchor_enabled": True,
        "recovery_anchor_source": "tomato_safety_projected_anchor",
        "transition_gate_enabled": False,
        "profile_feasibility_gate_enabled": False,
        "plan_cache_mode": "record",
        "cache_path": _rel(V56_CACHE_PATH),
        "output_dir": _rel(BENCHMARK_DIR),
        "trace_dir": _rel(V56_TRACE_DIR),
        "online_llm_allowed": True,
        "online_llm_scope": "only these three v56 qwen3.7-max shadow rollout scenarios",
        "online_llm_credentials_present": bool(precheck.get("online_llm_credentials_present", False)),
        "online_llm_accessible": bool(precheck.get("online_llm_accessible", False)),
        "provider_error_detected": bool(precheck.get("provider_error_detected", False)),
        "v55_recovery_anchor_design_pass": bool(v55_pass),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "executable": bool(executable),
        "blocked_reason": blocked_reason,
        "commands": commands,
    }


def write_online_precheck(precheck: Mapping[str, Any]) -> None:
    _write_json_md(ONLINE_PRECHECK_JSON, ONLINE_PRECHECK_MD, precheck, "v56 Qwen3.7 Online LLM Accessibility Precheck")


def write_execution_record(record: Mapping[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    V56_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    _write_json_md(EXECUTION_RECORD_JSON, EXECUTION_RECORD_MD, record, "v56 Qwen3.7 Recovery Anchor Opt-In Shadow Rollout Execution Record")


def _read_trace_rows(trace_dir: str | Path, scenario: str) -> list[dict[str, Any]]:
    return v51._read_trace_rows(trace_dir, scenario)


def _truthy(value: Any) -> bool:
    return v51._truthy(value)


def _canopy_dew_hard_violation_steps(rows: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        reason = " ".join(
            str(row.get(key, "") or "")
            for key in ("final_action_risk_reason", "final_action_risk_reason_v2", "tomato_safety_v2_reasons")
        ).lower()
        if _truthy(row.get("final_action_would_fail_canopy_boundary_v2")):
            count += 1
        elif _truthy(row.get("final_action_predicted_canopy_lt0_v2")):
            count += 1
        elif any(token in reason for token in ("canopy", "dew")) and _truthy(row.get("tomato_safety_v2_applied")):
            count += 1
    return int(count)


def _recovery_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def _selected_recovery(row: Mapping[str, Any]) -> bool:
        return any(
            str(row.get(key, "") or "") == "tomato_safety_projected_anchor"
            for key in (
                "recovery_anchor_selected_after",
                "profile_template_patch_fallback_veto_selected_after",
                "selected_fallback_candidate",
            )
        )

    return {
        "recovery_anchor_enabled_steps": int(sum(_truthy(row.get("recovery_anchor_enabled")) for row in rows)),
        "recovery_anchor_applied_steps": int(
            sum(_truthy(row.get("recovery_anchor_applied")) or _selected_recovery(row) for row in rows)
        ),
        "recovery_anchor_projected_source_steps": int(
            sum(str(row.get("recovery_anchor_source", "") or "") == "tomato_safety_projected_anchor" for row in rows)
        ),
        "recovery_anchor_still_hard_safety_rewrite_steps": int(
            sum(_truthy(row.get("recovery_anchor_hard_safety_rewrite")) for row in rows)
        ),
        "recovery_anchor_no_compatible_existing_candidate_steps": int(
            sum(_truthy(row.get("recovery_anchor_no_compatible_existing_candidate")) for row in rows)
        ),
        "canopy_dew_hard_violation_steps": _canopy_dew_hard_violation_steps(rows),
    }


def _v52_baseline_totals(v52_result_json: str | Path = V52_RESULT_AUDIT_JSON) -> dict[str, Any]:
    data = _load_json(v52_result_json)
    totals = data.get("totals", {}) if isinstance(data.get("totals", {}), Mapping) else {}
    return dict(totals)


def build_result_audit(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v48_trace_dir: str | Path = V48_TRACE_DIR,
    v56_trace_dir: str | Path = V56_TRACE_DIR,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
    v52_result_json: str | Path = V52_RESULT_AUDIT_JSON,
) -> dict[str, Any]:
    base = v52.build_result_audit(
        original_trace_dir=original_trace_dir,
        v48_trace_dir=v48_trace_dir,
        v52_trace_dir=v56_trace_dir,
        scenarios=scenarios,
    )
    base["artifact"] = "qwen37_recovery_anchor_opt_in_shadow_rollout_result_audit_20260601_v56"
    base["scope"] = "minimal_qwen37_recovery_anchor_opt_in_shadow_rollout_only"
    base["model_name"] = MODEL_NAME
    base["v56_trace_dir"] = base.pop("v52_trace_dir", _rel(v56_trace_dir))

    totals = {
        (str(key).replace("v52_", "v56_")): value
        for key, value in dict(base.get("totals", {}) or {}).items()
    }
    recovery_total = {
        "recovery_anchor_enabled_steps": 0,
        "recovery_anchor_applied_steps": 0,
        "recovery_anchor_projected_source_steps": 0,
        "recovery_anchor_still_hard_safety_rewrite_steps": 0,
        "recovery_anchor_no_compatible_existing_candidate_steps": 0,
        "v56_canopy_dew_hard_violation_steps": 0,
        "original_canopy_dew_hard_violation_steps": 0,
    }
    for report in base.get("scenario_reports", []):
        scenario = str(report.get("scenario_id", ""))
        v56_rows = _read_trace_rows(v56_trace_dir, scenario)
        original_rows = _read_trace_rows(original_trace_dir, scenario)
        original_failure = (report.get("original_full", {}) or {}).get("first_runtime_error_step")
        compare_len = int(original_failure) + 1 if original_failure is not None else len(original_rows)
        v56_counts = _recovery_counts(v56_rows[:compare_len])
        original_counts = _recovery_counts(original_rows[:compare_len])
        report["v56_recovery_anchor"] = v56_counts
        report["original_canopy_dew_hard_violation_steps"] = original_counts["canopy_dew_hard_violation_steps"]
        report["v56_trace_exists"] = report.pop("v52_trace_exists", False)
        report["v56_full"] = report.pop("v52_full", {})
        report["v56"] = report.pop("v52", {})
        recovery_total["recovery_anchor_enabled_steps"] += int(v56_counts["recovery_anchor_enabled_steps"])
        recovery_total["recovery_anchor_applied_steps"] += int(v56_counts["recovery_anchor_applied_steps"])
        recovery_total["recovery_anchor_projected_source_steps"] += int(v56_counts["recovery_anchor_projected_source_steps"])
        recovery_total["recovery_anchor_still_hard_safety_rewrite_steps"] += int(
            v56_counts["recovery_anchor_still_hard_safety_rewrite_steps"]
        )
        recovery_total["recovery_anchor_no_compatible_existing_candidate_steps"] += int(
            v56_counts["recovery_anchor_no_compatible_existing_candidate_steps"]
        )
        recovery_total["v56_canopy_dew_hard_violation_steps"] += int(v56_counts["canopy_dew_hard_violation_steps"])
        recovery_total["original_canopy_dew_hard_violation_steps"] += int(
            original_counts["canopy_dew_hard_violation_steps"]
        )

    v52_totals = _v52_baseline_totals(v52_result_json)
    v52_hard_new = v52_totals.get("hard_safety_rewrite_preferred_new_steps")
    v52_candidate_issues = v52_totals.get("v52_candidate_guardrail_issue_steps")
    base["totals"] = {**totals, **recovery_total}
    base["v52_baseline_totals"] = v52_totals

    acceptance = dict(base.get("acceptance", {}) or {})
    model_consistency_pass = bool(acceptance.get("qwen37_model_consistency_pass", False))
    hard_less = bool(
        v52_hard_new is not None
        and int(totals.get("hard_safety_rewrite_preferred_new_steps", 0)) < int(v52_hard_new)
    )
    candidate_not_worse = bool(
        v52_candidate_issues is not None
        and int(totals.get("v56_candidate_guardrail_issue_steps", 0)) <= int(v52_candidate_issues)
    )
    no_new_canopy_dew = bool(
        recovery_total["v56_canopy_dew_hard_violation_steps"]
        <= recovery_total["original_canopy_dew_hard_violation_steps"]
    )
    acceptance["all_v56_traces_present"] = bool(acceptance.get("all_v52_traces_present", False))
    acceptance["recovery_anchor_applied_steps_positive"] = bool(recovery_total["recovery_anchor_applied_steps"] > 0)
    acceptance["recovery_anchor_still_hard_safety_rewrite_zero"] = bool(
        recovery_total["recovery_anchor_still_hard_safety_rewrite_steps"] == 0
    )
    acceptance["hard_safety_rewrite_preferred_new_steps_less_than_v52"] = hard_less
    acceptance["candidate_guardrail_issue_not_worse_than_v52"] = candidate_not_worse
    acceptance["no_new_canopy_dew_hard_violation_vs_v39"] = no_new_canopy_dew
    acceptance["v56_acceptance_pass"] = bool(
        acceptance["all_v56_traces_present"]
        and model_consistency_pass
        and (acceptance.get("runtime_error_steps_zero", False) or acceptance.get("failure_not_earlier_all_scenarios", False))
        and acceptance["recovery_anchor_applied_steps_positive"]
        and acceptance["hard_safety_rewrite_preferred_new_steps_less_than_v52"]
        and acceptance.get("fallback_post_selection_hard_safety_leak_steps_zero", False)
        and acceptance["candidate_guardrail_issue_not_worse_than_v52"]
        and acceptance["no_new_canopy_dew_hard_violation_vs_v39"]
        and acceptance["recovery_anchor_still_hard_safety_rewrite_zero"]
    )
    base["acceptance"] = acceptance
    return base


def _stop_taxonomy(result_audit: Mapping[str, Any], execution_record: Mapping[str, Any] | None = None) -> list[str]:
    taxonomy: list[str] = []
    record = execution_record or {}
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    if not bool(record.get("executable", True)):
        blocked = str(record.get("blocked_reason") or "")
        if blocked:
            return [blocked]
    if not acceptance.get("all_v56_traces_present", False):
        taxonomy.append("metadata_missing")
    if not acceptance.get("qwen37_model_consistency_pass", False):
        taxonomy.append("model_consistency_failed")
    if not acceptance.get("runtime_error_steps_zero", False) and not acceptance.get("failure_not_earlier_all_scenarios", False):
        taxonomy.append("cvodes_earlier_failure")
    if not acceptance.get("recovery_anchor_applied_steps_positive", False):
        taxonomy.append("recovery_anchor_not_applied")
    if not acceptance.get("recovery_anchor_still_hard_safety_rewrite_zero", False):
        taxonomy.append("recovery_anchor_still_hard_safety_rewrite")
    if not acceptance.get("hard_safety_rewrite_preferred_new_steps_less_than_v52", False):
        taxonomy.append("fallback_veto_failed")
    if not acceptance.get("fallback_post_selection_hard_safety_leak_steps_zero", False):
        taxonomy.append("fallback_veto_no_alternative")
    if not acceptance.get("candidate_guardrail_issue_not_worse_than_v52", False):
        taxonomy.append("candidate_guardrail_pressure_persists")
    if not acceptance.get("no_new_canopy_dew_hard_violation_vs_v39", False):
        taxonomy.append("hard_safety_regression")
    return sorted(set(taxonomy or ["gate_insufficient"]))


def build_readiness(result_audit: Mapping[str, Any], execution_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    executable = bool((execution_record or {}).get("executable", False))
    passed = bool(acceptance.get("v56_acceptance_pass", False))
    if not executable:
        next_action = str((execution_record or {}).get("blocked_reason") or "online_llm_accessibility_blocked")
    elif passed:
        next_action = "recovery_anchor_h720_shadow_expansion_plan"
    else:
        next_action = "qwen37_recovery_anchor_failure_attribution"
    return {
        "artifact": "metadata_replay_readiness_checklist_20260601_v56",
        "current_stage": "v56_qwen37_recovery_anchor_opt_in_shadow_rollout",
        "model_name": MODEL_NAME,
        "recovery_anchor_opt_in_shadow_rollout_done": bool(acceptance.get("all_v56_traces_present", False)),
        "recovery_anchor_opt_in_shadow_rollout_pass": bool(passed),
        "recovery_anchor_applied_steps": int((result_audit.get("totals", {}) or {}).get("recovery_anchor_applied_steps", 0)),
        "execution_record_executable": bool(executable),
        "online_llm_accessible": bool((execution_record or {}).get("online_llm_accessible", False)),
        "qwen37_model_consistency_pass": bool(acceptance.get("qwen37_model_consistency_pass", False)),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": next_action,
        "stop_taxonomy": [] if passed else _stop_taxonomy(result_audit, execution_record),
    }


def write_result_artifacts(result_audit: Mapping[str, Any], readiness: Mapping[str, Any]) -> None:
    _write_json_md(RESULT_AUDIT_JSON, RESULT_AUDIT_MD, result_audit, "v56 Qwen3.7 Recovery Anchor Opt-In Shadow Rollout Result Audit")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v56 Metadata Replay Readiness")


def run_execution_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = []
    if not bool(record.get("executable", False)):
        return [{"skipped": True, "blocked_reason": str(record.get("blocked_reason", ""))}]
    for item in record.get("commands", []) or []:
        argv = [str(part) for part in item.get("argv", [])]
        completed = subprocess.run(argv, cwd=PROJECT_ROOT, check=False)
        results.append(
            {
                "group": item.get("group"),
                "returncode": int(completed.returncode),
                "command": item.get("command"),
            }
        )
        if completed.returncode != 0:
            break
    return results


def write_all(*, probe_online: bool = True, run_rollout: bool = False) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    precheck = build_online_precheck(probe_online=probe_online)
    write_online_precheck(precheck)
    execution = build_execution_record(precheck)
    write_execution_record(execution)
    run_results = []
    if run_rollout:
        run_results = run_execution_record(execution)
    result = build_result_audit()
    if run_results:
        result["execution_run_results"] = run_results
    readiness = build_readiness(result, execution)
    write_result_artifacts(result, readiness)
    return precheck, execution, result, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-all", action="store_true")
    parser.add_argument("--run-rollout", action="store_true")
    parser.add_argument("--skip-online-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    precheck, execution, result, readiness = write_all(
        probe_online=not args.skip_online_probe,
        run_rollout=bool(args.run_rollout),
    )
    if not args.write_all:
        print(
            json.dumps(
                {
                    "model_name": MODEL_NAME,
                    "online_llm_accessible": precheck.get("online_llm_accessible"),
                    "executable": execution.get("executable"),
                    "v56_acceptance_pass": (result.get("acceptance", {}) or {}).get("v56_acceptance_pass"),
                    "recovery_anchor_applied_steps": (result.get("totals", {}) or {}).get("recovery_anchor_applied_steps"),
                    "next_action": readiness.get("next_action"),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
