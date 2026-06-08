"""v51 minimal profile-template opt-in shadow rollout artifacts.

This module prepares and audits the fixed three-scenario v51 shadow rollout.
The opt-in patch is default-off in the controller and is only enabled through
agent-config overrides in the execution record. Controlled replay, promotion,
and performance claims remain blocked.
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

from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from gl_gym.agent.llm_agent import AgentConfig  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (  # noqa: E402
    AUDIT_DIR,
    FAILURE_SCENARIOS,
    ORIGINAL_TRACE_DIR,
    _read_rows,
    _runtime_failure_step,
    _trace_path,
)
from gl_gym.experiments.diagnose_ppo_vs_llm import derive_candidate_guardrail_compatibility  # noqa: E402
from gl_gym.experiments.profile_feasibility_gate_shadow_rollout_v48 import V48_TRACE_DIR  # noqa: E402


BENCHMARK_DIR = PROJECT_ROOT / "gl_gym" / "result" / "benchmarks" / "profile_template_opt_in_shadow_rollout_v51_20260601"
V51_TRACE_DIR = BENCHMARK_DIR / "traces"
V51_CACHE_PATH = PROJECT_ROOT / "gl_gym" / "result" / "plan_cache" / "profile_template_opt_in_shadow_v51_20260601.json"
V50_READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v50.json"

ONLINE_PRECHECK_JSON = AUDIT_DIR / "online_llm_accessibility_precheck_result_20260601_v51.json"
ONLINE_PRECHECK_MD = AUDIT_DIR / "online_llm_accessibility_precheck_result_20260601_v51.md"
EXECUTION_RECORD_JSON = AUDIT_DIR / "profile_template_opt_in_shadow_rollout_v51_execution_record_20260601.json"
EXECUTION_RECORD_MD = AUDIT_DIR / "profile_template_opt_in_shadow_rollout_v51_execution_record_20260601.md"
RESULT_AUDIT_JSON = AUDIT_DIR / "profile_template_opt_in_shadow_rollout_result_audit_20260601_v51.json"
RESULT_AUDIT_MD = AUDIT_DIR / "profile_template_opt_in_shadow_rollout_result_audit_20260601_v51.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v51.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v51.md"

AGENT_OVERRIDES = {
    "profile_template_patch_enabled": True,
    "fallback_post_selection_veto_enabled": True,
    "profile_template_patch_record_provenance": True,
    "profile_feasibility_gate_enabled": False,
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


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _dotenv_key_present(key: str) -> bool:
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


def build_online_precheck(*, probe_online: bool = True, model_name: str = AgentConfig.model_name) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    credentials_present = _dotenv_key_present("BAILIAN_API_KEY")
    result: dict[str, Any] = {
        "artifact": "online_llm_accessibility_precheck_result_20260601_v51",
        "model_name": model_name,
        "online_llm_credentials_present": bool(credentials_present),
        "online_llm_accessible": False,
        "provider_error_detected": False,
        "probe_online": bool(probe_online),
        "secret_value_recorded": False,
        "blocked_reason": "",
    }
    if not credentials_present:
        result["blocked_reason"] = "online_llm_credentials_missing"
        return result
    if not probe_online:
        result["blocked_reason"] = "online_llm_probe_not_run"
        return result
    try:
        llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
            max_tokens=4,
            api_key=os.getenv("BAILIAN_API_KEY"),
            base_url=AgentConfig.base_url,
        )
        response = llm.invoke([HumanMessage(content="Return OK.")])
        content = str(getattr(response, "content", "") or "")
        result.update(
            {
                "online_llm_accessible": True,
                "provider_error_detected": False,
                "probe_response_present": bool(content.strip()),
            }
        )
    except Exception as exc:  # pragma: no cover - provider behavior is environment-specific.
        message = str(exc)
        result.update(
            {
                "online_llm_accessible": False,
                "provider_error_detected": True,
                "provider_error_type": type(exc).__name__,
                "provider_error_summary": message[:240],
                "blocked_reason": "online_llm_accessibility_blocked",
            }
        )
    return result


def _command_for_group(group: Mapping[str, str]) -> list[str]:
    output_json = BENCHMARK_DIR / f"profile_template_opt_in_shadow_rollout_v51_{group['name']}.json"
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
        _rel(V51_CACHE_PATH),
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--agent-config-overrides",
        json.dumps(AGENT_OVERRIDES, sort_keys=True),
        "--output-json",
        _rel(output_json),
        "--output-trace-dir",
        _rel(V51_TRACE_DIR),
    ]


def _v50_passed(v50_readiness_json: str | Path = V50_READINESS_JSON) -> bool:
    readiness = _load_json(v50_readiness_json)
    return bool(readiness.get("profile_template_shadow_patch_pass", False))


def build_execution_record(
    precheck: Mapping[str, Any] | None = None,
    *,
    v50_readiness_json: str | Path = V50_READINESS_JSON,
) -> dict[str, Any]:
    precheck = dict(precheck or build_online_precheck(probe_online=True))
    v50_pass = _v50_passed(v50_readiness_json)
    executable = bool(v50_pass and precheck.get("online_llm_accessible", False))
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
    blocked_reason = ""
    if not v50_pass:
        blocked_reason = "v50_profile_template_shadow_patch_not_passed"
    elif not precheck.get("online_llm_accessible", False):
        blocked_reason = str(precheck.get("blocked_reason") or "online_llm_accessibility_blocked")
    return {
        "artifact": "profile_template_opt_in_shadow_rollout_v51_execution_record_20260601",
        "authorization_source": "user_delegated_profile_template_opt_in_shadow_rollout_authorization_20260601",
        "scope": "minimal_profile_template_opt_in_shadow_rollout_only",
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
        "cache_path": _rel(V51_CACHE_PATH),
        "output_dir": _rel(BENCHMARK_DIR),
        "trace_dir": _rel(V51_TRACE_DIR),
        "online_llm_allowed": True,
        "online_llm_scope": "only these three v51 shadow rollout scenarios",
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


def _write_json_md(path_json: Path, path_md: Path, data: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    lines = [f"# {title}", ""]
    for key in (
        "artifact",
        "scope",
        "current_stage",
        "executable",
        "blocked_reason",
        "online_llm_credentials_present",
        "online_llm_accessible",
        "provider_error_detected",
        "v51_acceptance_pass",
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
    for key in ("totals", "acceptance", "stop_taxonomy"):
        if key in data:
            lines.extend(["", f"## {key}", "```json", json.dumps(data[key], indent=2, sort_keys=True), "```"])
    if "commands" in data:
        lines.extend(["", "## Commands"])
        for item in data.get("commands", []):
            lines.extend(["", f"### {item.get('group')}", "```powershell", str(item.get("command", "")), "```"])
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_online_precheck(precheck: Mapping[str, Any]) -> None:
    _write_json_md(ONLINE_PRECHECK_JSON, ONLINE_PRECHECK_MD, precheck, "v51 Online LLM Accessibility Precheck")


def write_execution_record(record: Mapping[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    V51_TRACE_DIR.mkdir(parents=True, exist_ok=True)
    _write_json_md(EXECUTION_RECORD_JSON, EXECUTION_RECORD_MD, record, "v51 Profile Template Opt-In Shadow Rollout Execution Record")


def _read_trace_rows(trace_dir: str | Path, scenario: str) -> list[dict[str, Any]]:
    path = _trace_path(Path(trace_dir), scenario)
    if not path.exists():
        return []
    return _read_rows(path)


def _compat(row: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(row)
    merged.update(derive_candidate_guardrail_compatibility(dict(row)))
    return merged


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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
    hard = [
        row
        for row in merged
        if str(row.get("candidate_guardrail_compatibility_label", "") or "") == "hard_safety_rewrite"
    ]
    major_or_hard = [
        row
        for row in merged
        if str(row.get("candidate_guardrail_compatibility_label", "") or "") in {"major_rewrite", "hard_safety_rewrite"}
    ]
    runtime = [row for row in merged if str(row.get("runtime_error", "") or "").strip()]
    fallback_leak = [
        row for row in merged if _truthy(row.get("profile_template_patch_fallback_veto_no_alternative"))
    ]
    return {
        "row_count": int(len(merged)),
        "runtime_error_steps": int(len(runtime)),
        "first_runtime_error_step": _runtime_failure_step(list(merged)),
        "profile_action_conflict_steps": int(len(profile_conflicts)),
        "candidate_guardrail_issue_steps": int(len(issues)),
        "major_or_hard_rewrite_predicted_steps": int(len(major_or_hard)),
        "hard_safety_rewrite_steps": int(len(hard)),
        "fallback_post_selection_hard_safety_leak_steps": int(len(fallback_leak)),
        "profile_template_patch_enabled_steps": int(sum(_truthy(row.get("profile_template_patch_enabled")) for row in merged)),
        "profile_template_patch_applied_steps": int(sum(_truthy(row.get("profile_template_patch_applied")) for row in merged)),
        "fallback_veto_applied_steps": int(sum(_truthy(row.get("profile_template_patch_fallback_veto_applied")) for row in merged)),
        "fallback_veto_no_alternative_steps": int(
            sum(_truthy(row.get("profile_template_patch_fallback_veto_no_alternative")) for row in merged)
        ),
    }


def build_result_audit(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v48_trace_dir: str | Path = V48_TRACE_DIR,
    v51_trace_dir: str | Path = V51_TRACE_DIR,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    scenario_reports = []
    for scenario in scenarios:
        original_rows = _read_trace_rows(original_trace_dir, scenario)
        v48_rows = _read_trace_rows(v48_trace_dir, scenario)
        v51_rows = _read_trace_rows(v51_trace_dir, scenario)
        original_full = _count_metrics(original_rows)
        v48_full = _count_metrics(v48_rows)
        v51_full = _count_metrics(v51_rows)
        original_failure = original_full.get("first_runtime_error_step")
        v51_failure = v51_full.get("first_runtime_error_step")
        compare_len = int(original_failure) + 1 if original_failure is not None else len(original_rows)
        compare_len = max(0, compare_len)
        original = _count_metrics(original_rows[:compare_len])
        v48 = _count_metrics(v48_rows[:compare_len])
        v51 = _count_metrics(v51_rows[:compare_len])
        failure_not_earlier = bool(v51_failure is None or original_failure is None or int(v51_failure) >= int(original_failure))
        scenario_reports.append(
            {
                "scenario_id": scenario,
                "original_trace_exists": bool(original_rows),
                "v48_trace_exists": bool(v48_rows),
                "v51_trace_exists": bool(v51_rows),
                "comparison_window_steps": int(compare_len),
                "original_full": original_full,
                "v48_full": v48_full,
                "v51_full": v51_full,
                "original": original,
                "v48": v48,
                "v51": v51,
                "failure_not_earlier_than_v39": failure_not_earlier,
            }
        )

    totals = {
        "original_runtime_error_steps": sum(int(r["original"].get("runtime_error_steps", 0)) for r in scenario_reports),
        "v48_runtime_error_steps": sum(int(r["v48_full"].get("runtime_error_steps", 0)) for r in scenario_reports),
        "v51_runtime_error_steps": sum(int(r["v51_full"].get("runtime_error_steps", 0)) for r in scenario_reports),
        "v48_profile_action_conflict_steps": sum(int(r["v48"].get("profile_action_conflict_steps", 0)) for r in scenario_reports),
        "v51_profile_action_conflict_steps": sum(int(r["v51"].get("profile_action_conflict_steps", 0)) for r in scenario_reports),
        "v48_candidate_guardrail_issue_steps": sum(int(r["v48"].get("candidate_guardrail_issue_steps", 0)) for r in scenario_reports),
        "v51_candidate_guardrail_issue_steps": sum(int(r["v51"].get("candidate_guardrail_issue_steps", 0)) for r in scenario_reports),
        "v48_major_or_hard_rewrite_predicted_steps": sum(int(r["v48"].get("major_or_hard_rewrite_predicted_steps", 0)) for r in scenario_reports),
        "v51_major_or_hard_rewrite_predicted_steps": sum(int(r["v51"].get("major_or_hard_rewrite_predicted_steps", 0)) for r in scenario_reports),
        "fallback_post_selection_hard_safety_leak_steps": sum(
            int(r["v51"].get("fallback_post_selection_hard_safety_leak_steps", 0)) for r in scenario_reports
        ),
        "hard_safety_rewrite_preferred_new_steps": sum(
            max(0, int(r["v51"].get("hard_safety_rewrite_steps", 0)) - int(r["original"].get("hard_safety_rewrite_steps", 0)))
            for r in scenario_reports
        ),
        "profile_template_patch_applied_steps": sum(
            int(r["v51"].get("profile_template_patch_applied_steps", 0)) for r in scenario_reports
        ),
        "fallback_veto_applied_steps": sum(int(r["v51"].get("fallback_veto_applied_steps", 0)) for r in scenario_reports),
        "fallback_veto_no_alternative_steps": sum(
            int(r["v51"].get("fallback_veto_no_alternative_steps", 0)) for r in scenario_reports
        ),
    }
    acceptance = {
        "all_v51_traces_present": all(bool(r["v51_trace_exists"]) for r in scenario_reports),
        "runtime_error_steps_zero": bool(totals["v51_runtime_error_steps"] == 0),
        "failure_not_earlier_all_scenarios": bool(all(r["failure_not_earlier_than_v39"] for r in scenario_reports)),
        "fallback_post_selection_hard_safety_leak_steps_zero": bool(
            totals["fallback_post_selection_hard_safety_leak_steps"] == 0
        ),
        "hard_safety_rewrite_preferred_new_steps_zero": bool(
            totals["hard_safety_rewrite_preferred_new_steps"] == 0
        ),
        "profile_action_conflict_not_worse_than_v48": bool(
            totals["v51_profile_action_conflict_steps"] <= totals["v48_profile_action_conflict_steps"]
        ),
        "candidate_guardrail_issue_not_worse_than_v48": bool(
            totals["v51_candidate_guardrail_issue_steps"] <= totals["v48_candidate_guardrail_issue_steps"]
        ),
        "major_or_hard_rewrite_not_worse_than_v48": bool(
            totals["v51_major_or_hard_rewrite_predicted_steps"] <= totals["v48_major_or_hard_rewrite_predicted_steps"]
        ),
    }
    acceptance["v51_acceptance_pass"] = bool(
        acceptance["all_v51_traces_present"]
        and (acceptance["runtime_error_steps_zero"] or acceptance["failure_not_earlier_all_scenarios"])
        and acceptance["fallback_post_selection_hard_safety_leak_steps_zero"]
        and acceptance["hard_safety_rewrite_preferred_new_steps_zero"]
        and acceptance["profile_action_conflict_not_worse_than_v48"]
        and acceptance["candidate_guardrail_issue_not_worse_than_v48"]
        and acceptance["major_or_hard_rewrite_not_worse_than_v48"]
    )
    return {
        "artifact": "profile_template_opt_in_shadow_rollout_result_audit_20260601_v51",
        "scope": "minimal_profile_template_opt_in_shadow_rollout_only",
        "scenarios": list(scenarios),
        "original_trace_dir": _rel(original_trace_dir),
        "v48_trace_dir": _rel(v48_trace_dir),
        "v51_trace_dir": _rel(v51_trace_dir),
        "scenario_reports": scenario_reports,
        "totals": totals,
        "acceptance": acceptance,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def _stop_taxonomy(result_audit: Mapping[str, Any], execution_record: Mapping[str, Any] | None = None) -> list[str]:
    taxonomy: list[str] = []
    record = execution_record or {}
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    if not bool(record.get("executable", True)):
        blocked = str(record.get("blocked_reason") or "")
        if blocked:
            return [blocked]
    if not acceptance.get("all_v51_traces_present", False):
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
    return sorted(set(taxonomy or ["gate_insufficient"]))


def build_readiness(result_audit: Mapping[str, Any], execution_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    acceptance = result_audit.get("acceptance", {}) if isinstance(result_audit.get("acceptance", {}), Mapping) else {}
    executable = bool((execution_record or {}).get("executable", False))
    passed = bool(acceptance.get("v51_acceptance_pass", False))
    if not executable:
        next_action = "online_llm_accessibility_blocked"
    elif passed:
        next_action = "profile_template_patch_h720_shadow_expansion_plan"
    else:
        next_action = "profile_template_patch_failure_attribution"
    return {
        "artifact": "metadata_replay_readiness_checklist_20260601_v51",
        "current_stage": "v51_minimal_profile_template_opt_in_shadow_rollout",
        "profile_template_opt_in_shadow_rollout_done": bool(acceptance.get("all_v51_traces_present", False)),
        "profile_template_opt_in_shadow_rollout_pass": bool(passed),
        "execution_record_executable": bool(executable),
        "online_llm_accessible": bool((execution_record or {}).get("online_llm_accessible", False)),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": next_action,
        "stop_taxonomy": [] if passed else _stop_taxonomy(result_audit, execution_record),
    }


def write_result_artifacts(result_audit: Mapping[str, Any], readiness: Mapping[str, Any]) -> None:
    _write_json_md(RESULT_AUDIT_JSON, RESULT_AUDIT_MD, result_audit, "v51 Profile Template Opt-In Shadow Rollout Result Audit")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v51 Metadata Replay Readiness")


def write_all(*, probe_online: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    precheck = build_online_precheck(probe_online=probe_online)
    write_online_precheck(precheck)
    execution = build_execution_record(precheck)
    write_execution_record(execution)
    result = build_result_audit()
    readiness = build_readiness(result, execution)
    write_result_artifacts(result, readiness)
    return precheck, execution, result, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-all", action="store_true")
    parser.add_argument("--skip-online-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.write_all:
        write_all(probe_online=not args.skip_online_probe)
    else:
        precheck, execution, result, readiness = write_all(probe_online=not args.skip_online_probe)
        print(
            json.dumps(
                {
                    "online_llm_accessible": precheck.get("online_llm_accessible"),
                    "executable": execution.get("executable"),
                    "v51_acceptance_pass": (result.get("acceptance", {}) or {}).get("v51_acceptance_pass"),
                    "next_action": readiness.get("next_action"),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
