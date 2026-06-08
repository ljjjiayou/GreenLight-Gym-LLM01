"""v52 qwen3.7-max profile-template opt-in shadow rollout artifacts.

This module reuses the v51 three-scenario shadow validation but isolates the
cache/output/artifacts for the qwen3.7-max migration check. It does not enable
controlled replay, default-controller promotion, or performance claims.
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
from gl_gym.agent.llm_agent import AgentConfig  # noqa: E402
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
    / "profile_template_opt_in_shadow_rollout_v52_qwen37_20260601"
)
V52_TRACE_DIR = BENCHMARK_DIR / "traces"
V52_CACHE_PATH = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "plan_cache"
    / "profile_template_opt_in_shadow_v52_qwen37_20260601.json"
)
V50_READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v50.json"

ONLINE_PRECHECK_JSON = AUDIT_DIR / "online_llm_accessibility_precheck_result_v52_qwen37_20260601.json"
ONLINE_PRECHECK_MD = AUDIT_DIR / "online_llm_accessibility_precheck_result_v52_qwen37_20260601.md"
EXECUTION_RECORD_JSON = AUDIT_DIR / "profile_template_opt_in_shadow_rollout_v52_qwen37_execution_record_20260601.json"
EXECUTION_RECORD_MD = AUDIT_DIR / "profile_template_opt_in_shadow_rollout_v52_qwen37_execution_record_20260601.md"
RESULT_AUDIT_JSON = AUDIT_DIR / "profile_template_opt_in_shadow_rollout_v52_qwen37_result_audit_20260601.json"
RESULT_AUDIT_MD = AUDIT_DIR / "profile_template_opt_in_shadow_rollout_v52_qwen37_result_audit_20260601.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_v52_qwen37_20260601.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_v52_qwen37_20260601.md"

AGENT_OVERRIDES = dict(v51.AGENT_OVERRIDES)
SCENARIO_GROUPS = list(v51.SCENARIO_GROUPS)


def _rel(path: str | Path) -> str:
    return v51._rel(path)


def _write_json_md(path_json: Path, path_md: Path, data: Mapping[str, Any], title: str) -> None:
    v51._write_json_md(path_json, path_md, data, title)


def build_online_precheck(*, probe_online: bool = True, model_name: str = MODEL_NAME) -> dict[str, Any]:
    result = v51.build_online_precheck(probe_online=probe_online, model_name=model_name)
    result["artifact"] = "online_llm_accessibility_precheck_result_v52_qwen37_20260601"
    result["model_name"] = model_name
    return result


def _command_for_group(group: Mapping[str, str]) -> list[str]:
    output_json = BENCHMARK_DIR / f"profile_template_opt_in_shadow_rollout_v52_qwen37_{group['name']}.json"
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
        _rel(V52_CACHE_PATH),
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--agent-config-overrides",
        json.dumps(AGENT_OVERRIDES, sort_keys=True),
        "--output-json",
        _rel(output_json),
        "--output-trace-dir",
        _rel(V52_TRACE_DIR),
    ]


def build_execution_record(
    precheck: Mapping[str, Any] | None = None,
    *,
    v50_readiness_json: str | Path = V50_READINESS_JSON,
) -> dict[str, Any]:
    precheck = dict(precheck or build_online_precheck(probe_online=True))
    v50_pass = v51._v50_passed(v50_readiness_json)
    model_matches = str(precheck.get("model_name", "")) == MODEL_NAME
    executable = bool(v50_pass and model_matches and precheck.get("online_llm_accessible", False))
    blocked_reason = ""
    if not v50_pass:
        blocked_reason = "v50_profile_template_shadow_patch_not_passed"
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
        "artifact": "profile_template_opt_in_shadow_rollout_v52_qwen37_execution_record_20260601",
        "authorization_source": "user_delegated_qwen37_profile_template_opt_in_shadow_rollout_authorization_20260601",
        "scope": "minimal_qwen37_profile_template_opt_in_shadow_rollout_only",
        "model_name": MODEL_NAME,
        "scenarios": list(FAILURE_SCENARIOS),
        "controller": "llm_rspc_v2",
        "controlled_controller_included": False,
        "max_steps": 720,
        "agent_config_overrides": dict(AGENT_OVERRIDES),
        "transition_gate_enabled": False,
        "profile_feasibility_gate_enabled": False,
        "profile_template_patch_enabled": True,
        "fallback_post_selection_veto_enabled": True,
        "plan_cache_mode": "record",
        "cache_path": _rel(V52_CACHE_PATH),
        "output_dir": _rel(BENCHMARK_DIR),
        "trace_dir": _rel(V52_TRACE_DIR),
        "online_llm_allowed": True,
        "online_llm_scope": "only these three v52 qwen3.7-max shadow rollout scenarios",
        "online_llm_credentials_present": bool(precheck.get("online_llm_credentials_present", False)),
        "online_llm_accessible": bool(precheck.get("online_llm_accessible", False)),
        "provider_error_detected": bool(precheck.get("provider_error_detected", False)),
        "v50_profile_template_shadow_patch_pass": bool(v50_pass),
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
    _write_json_md(ONLINE_PRECHECK_JSON, ONLINE_PRECHECK_MD, precheck, "v52 Qwen3.7 Online LLM Accessibility Precheck")


def write_execution_record(record: Mapping[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    V52_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    _write_json_md(EXECUTION_RECORD_JSON, EXECUTION_RECORD_MD, record, "v52 Qwen3.7 Profile Template Opt-In Shadow Rollout Execution Record")


def _read_trace_rows(trace_dir: str | Path, scenario: str) -> list[dict[str, Any]]:
    return v51._read_trace_rows(trace_dir, scenario)


def _model_consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted({str(row.get("model_name", "") or "") for row in rows})
    missing_count = sum(1 for row in rows if not str(row.get("model_name", "") or "").strip())
    provider_error_rows = [
        row
        for row in rows
        if any(
            token in str(row.get("runtime_error", "") or "").lower()
            for token in ("provider", "access denied", "permission", "api", "dashscope", "bailian")
        )
    ]
    non_runtime_rows = [row for row in rows if not str(row.get("runtime_error", "") or "").strip()]
    fallback_only_path = bool(
        non_runtime_rows
        and all("fallback" in str(row.get("source", "") or "").lower() for row in non_runtime_rows)
    )
    return {
        "model_name_values": values,
        "model_name_missing_count": int(missing_count),
        "provider_error_row_count": int(len(provider_error_rows)),
        "fallback_only_path": bool(fallback_only_path),
        "model_consistency_pass": bool(
            rows
            and missing_count == 0
            and values == [MODEL_NAME]
            and not provider_error_rows
            and not fallback_only_path
        ),
    }


def build_result_audit(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v48_trace_dir: str | Path = V48_TRACE_DIR,
    v52_trace_dir: str | Path = V52_TRACE_DIR,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    base = v51.build_result_audit(
        original_trace_dir=original_trace_dir,
        v48_trace_dir=v48_trace_dir,
        v51_trace_dir=v52_trace_dir,
        scenarios=scenarios,
    )
    base["artifact"] = "profile_template_opt_in_shadow_rollout_v52_qwen37_result_audit_20260601"
    base["scope"] = "minimal_qwen37_profile_template_opt_in_shadow_rollout_only"
    base["model_name"] = MODEL_NAME
    base["v52_trace_dir"] = base.pop("v51_trace_dir", _rel(v52_trace_dir))
    model_reports = {}
    for report in base.get("scenario_reports", []):
        scenario = str(report.get("scenario_id", ""))
        rows = _read_trace_rows(v52_trace_dir, scenario)
        model_report = _model_consistency(rows)
        report["v52_model_consistency"] = model_report
        report["v52_trace_exists"] = report.pop("v51_trace_exists", False)
        report["v52_full"] = report.pop("v51_full", {})
        report["v52"] = report.pop("v51", {})
        model_reports[scenario] = model_report
    acceptance = dict(base.get("acceptance", {}) or {})
    model_consistency_pass = bool(
        model_reports
        and all(item.get("model_consistency_pass", False) for item in model_reports.values())
    )
    totals = {
        (str(key).replace("v51_", "v52_")): value
        for key, value in dict(base.get("totals", {}) or {}).items()
    }
    base["totals"] = totals
    acceptance["all_v52_traces_present"] = bool(acceptance.get("all_v51_traces_present", False))
    acceptance["qwen37_model_consistency_pass"] = model_consistency_pass
    acceptance["v52_acceptance_pass"] = bool(acceptance.get("v51_acceptance_pass", False) and model_consistency_pass)
    acceptance.pop("v51_acceptance_pass", None)
    base["acceptance"] = acceptance
    base["model_consistency"] = {
        "expected_model_name": MODEL_NAME,
        "scenario_reports": model_reports,
        "qwen37_model_consistency_pass": model_consistency_pass,
    }
    return base


def _stop_taxonomy(result_audit: Mapping[str, Any], execution_record: Mapping[str, Any] | None = None) -> list[str]:
    taxonomy: list[str] = []
    record = execution_record or {}
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    if not bool(record.get("executable", True)):
        blocked = str(record.get("blocked_reason") or "")
        if blocked:
            return [blocked]
    if not acceptance.get("all_v52_traces_present", acceptance.get("all_v51_traces_present", False)):
        taxonomy.append("metadata_missing")
    if not acceptance.get("runtime_error_steps_zero", False) and not acceptance.get("failure_not_earlier_all_scenarios", False):
        taxonomy.append("cvodes_earlier_failure")
    if not acceptance.get("fallback_post_selection_hard_safety_leak_steps_zero", False):
        taxonomy.append("fallback_veto_no_alternative")
    if not acceptance.get("hard_safety_rewrite_preferred_new_steps_zero", False):
        taxonomy.append("fallback_veto_failed")
    if not acceptance.get("profile_action_conflict_not_worse_than_v48", False):
        taxonomy.append("template_patch_not_applied")
    if not acceptance.get("candidate_guardrail_issue_not_worse_than_v48", False):
        taxonomy.append("candidate_guardrail_pressure_persists")
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    if not acceptance.get("qwen37_model_consistency_pass", False):
        taxonomy.append("model_consistency_failed")
    return sorted(set(taxonomy or ["gate_insufficient"]))


def build_readiness(result_audit: Mapping[str, Any], execution_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    executable = bool((execution_record or {}).get("executable", False))
    passed = bool(acceptance.get("v52_acceptance_pass", False))
    if not executable:
        next_action = "online_llm_accessibility_blocked"
    elif passed:
        next_action = "profile_template_patch_h720_shadow_expansion_plan"
    else:
        next_action = "qwen37_profile_template_patch_failure_attribution"
    return {
        "artifact": "metadata_replay_readiness_checklist_v52_qwen37_20260601",
        "current_stage": "v52_qwen37_minimal_profile_template_opt_in_shadow_rollout",
        "model_name": MODEL_NAME,
        "profile_template_opt_in_shadow_rollout_done": bool(
            acceptance.get("all_v52_traces_present", acceptance.get("all_v51_traces_present", False))
        ),
        "profile_template_opt_in_shadow_rollout_pass": bool(passed),
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
    _write_json_md(RESULT_AUDIT_JSON, RESULT_AUDIT_MD, result_audit, "v52 Qwen3.7 Profile Template Opt-In Shadow Rollout Result Audit")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v52 Qwen3.7 Metadata Replay Readiness")


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
                    "v52_acceptance_pass": (result.get("acceptance", {}) or {}).get("v52_acceptance_pass"),
                    "qwen37_model_consistency_pass": (result.get("acceptance", {}) or {}).get("qwen37_model_consistency_pass"),
                    "next_action": readiness.get("next_action"),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
