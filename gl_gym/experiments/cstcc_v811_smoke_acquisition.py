"""v81.1 C-STCC smoke acquisition runner.

This is an offline, opt-in runtime trace. It exercises the real
RuleBasedLLMDirector runtime path and C-STCC shadow hook, but it seeds an
offline fallback plan and blocks online LLM replanning. It does not compare
reward, promote a controller, execute rollout, or change final action.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import yaml

from gl_gym.agent.interface import GreenhouseAgentInterface
from gl_gym.agent.llm_agent import (
    AgentConfig,
    RuleBasedLLMDirector,
    apply_safety_guardrails,
    create_langchain_tools,
)
from gl_gym.cstcc.audit_writer import read_jsonl, summarize_shadow_rows
from gl_gym.experiments.diagnose_ppo_vs_llm import (
    DEFAULT_RULE_PARAMS,
    build_env,
    control_to_record,
    cstcc_shadow_to_record,
    finalize_trace_row,
    info_to_metrics,
    state_to_record,
    sum_metrics,
    write_rows_csv,
)


ACTION_FIELDS = (
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
)

FULL_EPISODE_MIN_STEPS = 720
LATENCY_P95_ACCEPTABLE_MS = 250.0
JSON_ROW_SIZE_MAX_ACCEPTABLE_KB = 64.0
FALLBACK_RATE_DIAGNOSTIC_THRESHOLD = 0.05
INFEASIBLE_RATIO_DIAGNOSTIC_THRESHOLD = 0.50
ACTION_DIFF_DIAGNOSTIC_THRESHOLD = 0.75


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(100.0, float(percentile))) / 100.0
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return ordered[lower]
    frac = position - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def build_offline_smoke_plan(
    agent: RuleBasedLLMDirector,
    state: Any,
    *,
    horizon_steps: int,
) -> Dict[str, Any]:
    """Build a no-LLM fallback plan so the runtime hook can be smoke-tested."""

    analysis = agent.analyze_state()
    fallback_control = apply_safety_guardrails(
        state,
        agent._select_fallback_control(state=state, analysis=analysis),
    )
    target_temp, target_co2, target_rh, missing_fields, corrected_fields = agent._enforce_setpoint_contract(
        state,
        np.asarray(fallback_control, dtype=np.float32),
        None,
        None,
        None,
    )
    target_profile, profile_contract = agent._build_target_profiles(
        {},
        target_temp,
        target_co2,
        target_rh,
        horizon_steps,
    )
    return {
        "anchor_control": np.asarray(fallback_control, dtype=np.float32),
        "anchor_source": "offline_smoke_fallback_no_online_llm",
        "target_temp": target_temp,
        "target_co2": target_co2,
        "target_rh": target_rh,
        "target_profile": target_profile,
        "profile_contract": profile_contract,
        "setpoint_contract": {
            "filled": list(missing_fields),
            "corrected": list(corrected_fields),
        },
        "fallback_selection": dict(getattr(agent, "last_fallback_selection", {}) or {}),
        "created_timestep": int(getattr(state, "timestep", 0)),
        "expires_timestep": int(getattr(state, "timestep", 0)) + int(horizon_steps),
        "reason": "offline_smoke_seed_plan",
        "llm_action_found": False,
        "plan_interval": int(horizon_steps),
        "plan_cache_event": {
            "enabled": False,
            "mode": "off",
            "hit": False,
            "status": "offline_smoke_no_online_llm",
        },
    }


def _install_offline_replan_blocker(agent: RuleBasedLLMDirector) -> Dict[str, int]:
    counter = {"blocked_online_llm_replans": 0}

    def _offline_replan(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        counter["blocked_online_llm_replans"] += 1
        agent.last_plan_cache_event = {
            "enabled": False,
            "mode": "off",
            "hit": False,
            "status": "offline_smoke_blocked_online_llm",
        }
        return {
            "success": False,
            "error": "offline_smoke_online_llm_disabled",
            "llm_attempts": 0,
            "llm_action_found": False,
        }

    agent._replan_with_llm = _offline_replan  # type: ignore[method-assign]
    return counter


def make_agent(
    *,
    interface: GreenhouseAgentInterface,
    year: int,
    day: int,
    seed: int,
    max_steps: int,
    log_root: str,
    cstcc_config_path: str = "",
) -> RuleBasedLLMDirector:
    if hasattr(create_langchain_tools, "instance"):
        delattr(create_langchain_tools, "instance")
    tools = create_langchain_tools(interface)
    cfg = AgentConfig(
        api_key="offline-cstcc-smoke-no-online-llm",
        verbose=False,
        control_interval=max(2, int(max_steps) + 2),
        max_iterations=1,
        max_tokens=1,
        plan_cache_mode="off",
        cstcc_shadow_enabled=True,
        cstcc_shadow_audit_log_root=str(log_root),
        cstcc_shadow_sample_rate=1.0,
        cstcc_shadow_save_full_candidates=False,
        cstcc_shadow_save_raw_sequences=False,
        cstcc_shadow_save_projected_sequences=False,
        cstcc_shadow_fail_closed=False,
        cstcc_shadow_assert_final_action_invariant=True,
        cstcc_shadow_config_path=str(cstcc_config_path or "") or None,
    )
    return RuleBasedLLMDirector(
        agent_interface=interface,
        tools=tools,
        config=cfg,
        env_id=f"TomatoEnv_y{int(year)}_d{int(day)}_s{int(seed)}_v811_smoke",
        rule_params=dict(DEFAULT_RULE_PARAMS),
    )


def run_smoke_trace(
    *,
    config: Mapping[str, Any],
    year: int,
    day: int,
    seed: int,
    max_steps: int = 30,
    uncertainty_scale: float = 0.0,
    log_root: str = "logs/cstcc_shadow/v811_smoke",
    cstcc_config_path: str = "",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_env = build_env(dict(config), year, day, seed, uncertainty_scale)
    interface = GreenhouseAgentInterface(raw_env)
    agent = make_agent(
        interface=interface,
        year=year,
        day=day,
        seed=seed,
        max_steps=max_steps,
        log_root=log_root,
        cstcc_config_path=cstcc_config_path,
    )
    offline_replan_counter = _install_offline_replan_blocker(agent)
    state = interface.get_state()
    agent.current_plan = build_offline_smoke_plan(agent, state, horizon_steps=max_steps + 4)

    rows: List[Dict[str, Any]] = []
    done = False
    step = 0
    while (not done) and step < int(max_steps):
        state = interface.get_state()
        if not agent._is_plan_active(int(getattr(state, "timestep", step))):
            agent.current_plan = build_offline_smoke_plan(agent, state, horizon_steps=max_steps + 4)
        try:
            result = agent.step_with_rules()
        except Exception as exc:
            rollout_selection = dict(getattr(agent, "last_rollout_selection", {}) or {})
            applied_control = getattr(raw_env, "u", np.zeros(6, dtype=np.float32))
            try:
                info = raw_env._get_info()
            except Exception:
                info = {}
            row = {
                "algo": "llm_director_offline_cstcc_smoke",
                "year": int(year),
                "day": int(day),
                "seed": int(seed),
                "step": int(step),
                "reward": 0.0,
                "done": True,
                "source": "runtime_error",
                "runtime_error": str(exc),
                "runtime_error_type": "simulator_or_controller_error",
                **cstcc_shadow_to_record({"rollout_selection": rollout_selection}),
                **state_to_record(state),
                **control_to_record(applied_control),
                **info_to_metrics(info),
            }
            rows.append(finalize_trace_row(row))
            break

        plan = result.get("plan", {}) if isinstance(result, dict) else {}
        rollout = plan.get("rollout_selection", {}) if isinstance(plan, dict) else {}
        rollout_source = str(rollout.get("source", result.get("action", "unknown"))) if isinstance(rollout, dict) else "unknown"
        applied_control = result.get("applied_control", getattr(raw_env, "u", np.zeros(6, dtype=np.float32)))
        info = raw_env._get_info()
        row = {
            "algo": "llm_director_offline_cstcc_smoke",
            "year": int(year),
            "day": int(day),
            "seed": int(seed),
            "step": int(step),
            "reward": float(result.get("reward", 0.0)),
            "done": bool(result.get("done", False)),
            "source": rollout_source,
            "replan_reason": result.get("replan_reason"),
            "llm_attempts": int(result.get("llm_attempts", 0) or 0),
            "llm_action_found": bool(result.get("llm_action_found", False)),
            "anchor_source": plan.get("anchor_source") if isinstance(plan, dict) else None,
            "target_temp": plan.get("current_target_temp") if isinstance(plan, dict) else None,
            "target_co2": plan.get("current_target_co2") if isinstance(plan, dict) else None,
            "target_rh": plan.get("current_target_rh") if isinstance(plan, dict) else None,
            **cstcc_shadow_to_record(result),
            **cstcc_shadow_to_record(plan),
            **state_to_record(state),
            **control_to_record(applied_control),
            **control_to_record(result.get("anchor_control", np.zeros(6)), prefix="anchor"),
            **control_to_record(result.get("rule_control", np.zeros(6)), prefix="rule"),
            **info_to_metrics(info),
        }
        rows.append(finalize_trace_row(row))
        done = bool(result.get("done", False))
        step += 1

    metadata = {
        "online_llm_called": False,
        "offline_replan_blocked_count": int(offline_replan_counter["blocked_online_llm_replans"]),
        "log_root": str(log_root),
        "cstcc_config_path": str(cstcc_config_path or ""),
    }
    return rows, metadata


def _jsonl_rows(log_root: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(Path(log_root).glob("v81_episode_*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def _action_diff_stats(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    values = {field: [] for field in ACTION_FIELDS}
    for row in rows:
        raw = row.get("cstcc_shadow_action_diff_json") or row.get("shadow_action_difference_from_runtime") or "{}"
        try:
            diff = json.loads(str(raw)) if isinstance(raw, str) else dict(raw)
        except Exception:
            diff = {}
        if not isinstance(diff, dict):
            continue
        for field in ACTION_FIELDS:
            values[field].append(abs(float(diff.get(field, 0.0) or 0.0)))
    return {
        field: {
            "mean": float(mean(items)) if items else 0.0,
            "p95": _percentile(items, 95.0),
            "max": max(items) if items else 0.0,
        }
        for field, items in values.items()
    }


def build_smoke_summary(
    *,
    rows: List[Dict[str, Any]],
    metadata: Mapping[str, Any],
    year: int,
    day: int,
    seed: int,
    max_steps: int,
    log_root: str,
    stage_label: str = "smoke_acquisition",
) -> Dict[str, Any]:
    trace_summary = sum_metrics(rows)
    jsonl_rows = _jsonl_rows(log_root)
    jsonl_summary = summarize_shadow_rows(jsonl_rows)
    is_full_episode = str(stage_label) == "full_episode_trace" or int(max_steps) >= FULL_EPISODE_MIN_STEPS
    enabled_rows = [row for row in rows if bool(row.get("cstcc_shadow_enabled", False))]
    latency_values = [float(row.get("cstcc_shadow_audit_latency_ms", 0.0) or 0.0) for row in enabled_rows]
    size_values = [float(row.get("cstcc_shadow_audit_json_size_kb", 0.0) or 0.0) for row in enabled_rows]
    audit_success_rate = float(trace_summary.get("cstcc_shadow_audit_success_rate", 0.0) or 0.0)
    invariant_rate = float(trace_summary.get("cstcc_shadow_final_action_invariant_rate", 0.0) or 0.0)
    final_action_changed_steps = int(trace_summary.get("cstcc_shadow_final_action_changed_steps", 0) or 0)
    latency_p95 = _percentile(latency_values, 95.0)
    json_size_kb_max = max(size_values) if size_values else 0.0
    fallback_rate = float(trace_summary.get("cstcc_shadow_fallback_rate", 0.0) or 0.0)
    mean_infeasible_candidate_ratio = float(
        trace_summary.get("cstcc_shadow_mean_infeasible_candidate_ratio", 0.0) or 0.0
    )
    action_diff_stats = _action_diff_stats(rows)
    max_action_diff = max((float(stats.get("max", 0.0) or 0.0) for stats in action_diff_stats.values()), default=0.0)
    answers = {
        "jsonl_generated": bool(jsonl_rows),
        "audit_success_rate": audit_success_rate,
        "final_action_invariant_rate": invariant_rate,
        "final_action_invariant_is_100_percent": bool(invariant_rate == 1.0 and final_action_changed_steps == 0),
        "audit_latency_ms_mean": float(mean(latency_values)) if latency_values else 0.0,
        "audit_latency_ms_p95": latency_p95,
        "audit_latency_p95_acceptable": bool(latency_p95 <= LATENCY_P95_ACCEPTABLE_MS),
        "json_size_kb_mean": float(mean(size_values)) if size_values else 0.0,
        "json_size_kb_max": json_size_kb_max,
        "json_size_kb_max_acceptable": bool(json_size_kb_max <= JSON_ROW_SIZE_MAX_ACCEPTABLE_KB),
        "fallback_rate": fallback_rate,
        "mean_infeasible_candidate_ratio": mean_infeasible_candidate_ratio,
        "max_action_diff": max_action_diff,
        "action_diff_stats_by_field": action_diff_stats,
        "hard_constraint_violation_reason_distribution": trace_summary.get(
            "cstcc_shadow_hard_constraint_violation_reason_distribution", {}
        ),
        "hard_constraint_violation_field_distribution": trace_summary.get(
            "cstcc_shadow_hard_constraint_violation_field_distribution", {}
        ),
        "selected_source_prior_distribution": trace_summary.get(
            "cstcc_shadow_selected_source_prior_distribution", {}
        ),
        "selected_template_name_distribution": trace_summary.get(
            "cstcc_shadow_selected_template_name_distribution", {}
        ),
    }
    blockers: List[str] = []
    if is_full_episode and len(rows) < int(max_steps):
        blockers.append("full_episode_trace_short")
    if not answers["jsonl_generated"]:
        blockers.append("jsonl_not_generated")
    if invariant_rate != 1.0 or final_action_changed_steps:
        blockers.append("final_action_invariant_failure")
    if int(trace_summary.get("cstcc_shadow_online_llm_called_steps", 0) or 0):
        blockers.append("online_llm_called")
    if int(trace_summary.get("cstcc_shadow_predictive_rollout_executed_steps", 0) or 0):
        blockers.append("predictive_rollout_executed")
    if int(trace_summary.get("cstcc_shadow_real_tomato_safety_projection_steps", 0) or 0):
        blockers.append("real_tomato_safety_projection_used")
    if not enabled_rows:
        blockers.append("cstcc_shadow_rows_missing")
    if not answers["audit_latency_p95_acceptable"]:
        blockers.append("audit_latency_unacceptable")
    if not answers["json_size_kb_max_acceptable"]:
        blockers.append("audit_json_size_unacceptable")

    diagnostic_notes: List[str] = []
    if fallback_rate > FALLBACK_RATE_DIAGNOSTIC_THRESHOLD:
        diagnostic_notes.append("fallback_rate_above_0_05")
    if mean_infeasible_candidate_ratio > INFEASIBLE_RATIO_DIAGNOSTIC_THRESHOLD:
        diagnostic_notes.append("high_infeasible_candidate_ratio")
    if max_action_diff > ACTION_DIFF_DIAGNOSTIC_THRESHOLD:
        diagnostic_notes.append("large_shadow_runtime_action_diff")

    if "final_action_invariant_failure" in blockers:
        next_action = "stop_and_repair_cstcc_runtime_hook"
    elif blockers:
        next_action = "repair_v811_full_episode_instrumentation" if is_full_episode else "repair_v811_smoke_instrumentation"
    elif audit_success_rate < 0.99:
        next_action = "v811_audit_failure_repair_then_rerun_full_episode" if is_full_episode else "repair_v811_audit_failures_then_rerun_smoke"
    elif is_full_episode and diagnostic_notes:
        next_action = "v815_shadow_analytics_diagnostic_first"
    elif is_full_episode:
        next_action = "v815_shadow_analytics"
    elif diagnostic_notes:
        next_action = "v811_full_episode_trace_diagnostic_first"
    else:
        next_action = "v811_full_episode_trace"

    return {
        "schema_version": "cstcc_v811_runtime_trace_result_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stage_label": str(stage_label),
        "scenario": {
            "year": int(year),
            "day": int(day),
            "seed": int(seed),
            "max_steps": int(max_steps),
            "log_root": str(log_root),
        },
        "boundaries": {
            "online_llm_called": False,
            "predictive_rollout_executed": False,
            "real_tomato_safety_projection": False,
            "final_action_changed_allowed": False,
            "performance_claim_allowed": False,
            "promotion_evidence": False,
        },
        "metadata": dict(metadata),
        "trace_summary": trace_summary,
        "jsonl_summary": jsonl_summary,
        "answers": answers,
        "blockers": blockers,
        "diagnostic_notes": diagnostic_notes,
        "next_action": next_action,
    }


def build_report(summary: Mapping[str, Any]) -> str:
    answers = summary.get("answers", {})
    trace = summary.get("trace_summary", {})
    scenario = summary.get("scenario", {})
    stage_label = str(summary.get("stage_label", "smoke_acquisition"))
    title = "Full-Episode Trace" if stage_label == "full_episode_trace" else "Smoke Acquisition"
    scope = (
        "Full opt-in runtime shadow trace under the H720 convention."
        if stage_label == "full_episode_trace"
        else "Short opt-in runtime shadow trace."
    )
    diff = answers.get("action_diff_stats_by_field", {}) if isinstance(answers, Mapping) else {}
    reason_dist = answers.get("hard_constraint_violation_reason_distribution", {})
    field_dist = answers.get("hard_constraint_violation_field_distribution", {})
    source_prior_dist = answers.get("selected_source_prior_distribution", {})
    template_dist = answers.get("selected_template_name_distribution", {})
    lines = [
        f"# C-STCC v81.1 {title}",
        "",
        "## Scope",
        "",
        f"{scope} No online LLM, no predictive rollout, no final-action change, no reward claim.",
        "",
        "## Scenario",
        "",
        f"- Stage label: {stage_label}",
        f"- Year/day/seed: {scenario.get('year')}/{scenario.get('day')}/{scenario.get('seed')}",
        f"- Max steps: {scenario.get('max_steps')}",
        f"- Log root: `{scenario.get('log_root')}`",
        "",
        "## Six Questions",
        "",
        f"1. JSONL generated: {answers.get('jsonl_generated')}",
        f"2. Audit success rate: {float(answers.get('audit_success_rate', 0.0)):.3f}",
        f"3. Final-action invariant rate: {float(answers.get('final_action_invariant_rate', 0.0)):.3f}",
        f"4. Audit latency mean/p95 ms: {float(answers.get('audit_latency_ms_mean', 0.0)):.3f} / {float(answers.get('audit_latency_ms_p95', 0.0)):.3f}",
        f"5. JSON row size mean/max KB: {float(answers.get('json_size_kb_mean', 0.0)):.3f} / {float(answers.get('json_size_kb_max', 0.0)):.3f}",
        f"6. Fallback rate: {float(answers.get('fallback_rate', 0.0)):.3f}",
        "",
        "## Boundary Checks",
        "",
        f"- Online LLM called steps: {trace.get('cstcc_shadow_online_llm_called_steps', 0)}",
        f"- Predictive rollout steps: {trace.get('cstcc_shadow_predictive_rollout_executed_steps', 0)}",
        f"- Real Tomato Safety projection steps: {trace.get('cstcc_shadow_real_tomato_safety_projection_steps', 0)}",
        f"- Final action changed steps: {trace.get('cstcc_shadow_final_action_changed_steps', 0)}",
        "",
        "## Candidate Health",
        "",
        f"- Mean candidate count: {float(trace.get('cstcc_shadow_mean_candidate_count', 0.0)):.3f}",
        f"- Mean feasible candidate count: {float(trace.get('cstcc_shadow_mean_feasible_candidate_count', 0.0)):.3f}",
        f"- Mean infeasible candidate ratio: {float(trace.get('cstcc_shadow_mean_infeasible_candidate_ratio', 0.0)):.3f}",
        f"- Hard constraint violation count: {trace.get('cstcc_shadow_hard_constraint_violation_count', 0)}",
        f"- Selected source prior distribution: `{_compact_json(source_prior_dist)}`",
        f"- Selected template distribution: `{_compact_json(template_dist)}`",
        f"- Hard constraint reason distribution: `{_compact_json(reason_dist)}`",
        f"- Hard constraint field distribution: `{_compact_json(field_dist)}`",
        "",
        "## Shadow/Runtime Action Difference",
        "",
        "| Field | Mean | P95 | Max |",
        "|---|---:|---:|---:|",
    ]
    for field in ACTION_FIELDS:
        stats = diff.get(field, {}) if isinstance(diff, Mapping) else {}
        lines.append(
            f"| {field} | {float(stats.get('mean', 0.0)):.4f} | "
            f"{float(stats.get('p95', 0.0)):.4f} | {float(stats.get('max', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Blockers: {', '.join(summary.get('blockers', [])) if summary.get('blockers') else 'none'}",
            f"- Diagnostic notes: {', '.join(summary.get('diagnostic_notes', [])) if summary.get('diagnostic_notes') else 'none'}",
            f"- Next action: {summary.get('next_action')}",
            "",
            "Reward and controller-quality comparisons are intentionally out of scope for v81.1.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    rows: List[Dict[str, Any]],
    summary: Mapping[str, Any],
    output_prefix: Path,
) -> Dict[str, str]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(_compact_json(summary) + "\n", encoding="utf-8")
    write_rows_csv(csv_path, rows)
    md_path.write_text(build_report(summary), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "md": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v81.1 C-STCC smoke acquisition.")
    parser.add_argument("--config", type=str, default="gl_gym/configs/envs/TomatoEnv.yml")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--day", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--uncertainty-scale", type=float, default=0.0)
    parser.add_argument("--log-root", type=str, default="")
    parser.add_argument("--output-prefix", type=str, default="")
    parser.add_argument("--stage-label", type=str, default="")
    parser.add_argument("--cstcc-config", type=str, default="")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = args.log_root or f"logs/cstcc_shadow/v811_smoke_{stamp}"
    output_prefix = Path(args.output_prefix or f"gl_gym/result/diagnostics/cstcc_v811_smoke_acquisition_{stamp}")
    stage_label = args.stage_label or ("full_episode_trace" if int(args.max_steps) >= FULL_EPISODE_MIN_STEPS else "smoke_acquisition")
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    started = time.perf_counter()
    rows, metadata = run_smoke_trace(
        config=config,
        year=args.year,
        day=args.day,
        seed=args.seed,
        max_steps=args.max_steps,
        uncertainty_scale=args.uncertainty_scale,
        log_root=log_root,
        cstcc_config_path=args.cstcc_config,
    )
    metadata = {**metadata, "elapsed_seconds": float(time.perf_counter() - started)}
    summary = build_smoke_summary(
        rows=rows,
        metadata=metadata,
        year=args.year,
        day=args.day,
        seed=args.seed,
        max_steps=args.max_steps,
        log_root=log_root,
        stage_label=stage_label,
    )
    outputs = write_outputs(rows=rows, summary=summary, output_prefix=output_prefix)
    print(json.dumps({"outputs": outputs, "next_action": summary["next_action"], "answers": summary["answers"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
