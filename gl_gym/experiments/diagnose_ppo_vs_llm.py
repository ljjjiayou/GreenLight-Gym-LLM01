"""Diagnose where PPO and LLM-RSPC differ in greenhouse control.

The goal is not to produce a final benchmark table. It records matched
per-step trajectories, groups them by meaningful greenhouse regimes, and
highlights action/cost/safety gaps that should drive the next controller change.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import yaml
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gl_gym.agent.expert_distillation import ACTION_NAMES
from gl_gym.agent.interface import GreenhouseAgentInterface
from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector, create_langchain_tools
from gl_gym.environments.baseline import RuleBasedController as GreenhouseRuleController
from gl_gym.environments.tomato_env import TomatoEnv

load_dotenv()


METRIC_MAP = {
    "profit": "EPI",
    "revenue": "revenue",
    "heat_cost": "heat_cost",
    "co2_cost": "co2_cost",
    "elec_cost": "elec_cost",
    "fixed_cost": "fixed_costs",
    "temp_violation": "temp_violation",
    "co2_violation": "co2_violation",
    "rh_violation": "rh_violation",
    "lamp_violation": "lamp_violation",
}


DEFAULT_RULE_PARAMS = {
    "lamps_on": 6.0,
    "lamps_off": 20.0,
    "lamps_day_start": 1,
    "lamps_day_stop": 365,
    "lamps_off_sun": 400.0,
    "lamp_rad_sum_limit": 1000.0,
    "temp_setpoint_day": 20.0,
    "temp_setpoint_night": 16.0,
    "heat_correction": 1.0,
    "heat_deadzone": 0.5,
    "co2_day": 850.0,
    "vent_heat_Pband": 4.0,
    "rh_max": 83.0,
    "mech_dehumid_Pband": 7.0,
    "vent_rh_Pband": 7.0,
    "t_vent_off": 10.0,
    "vent_cold_Pband": 5.0,
    "thScrSpDay": 25.0,
    "thScrSpNight": 10.0,
    "thScrPband": 5.0,
    "thScrDeadZone": 2.0,
    "thScrRh": 90.0,
    "thScrRhPband": 5.0,
    "lampExtraHeat": 2.0,
    "blScrExtraRh": 5.0,
    "rhMax": 90.0,
    "tHeatBand": 2.0,
    "co2Band": 180.0,
    "useBlScr": False,
}


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_env(config: Dict[str, Any], year: int, day: int, seed: int, uncertainty_scale: float) -> TomatoEnv:
    base = dict(config["GreenLightEnv"])
    base["training"] = False
    tomato = dict(config["TomatoEnv"])
    eval_options = dict(tomato["eval_options"])
    eval_options["eval_years"] = [int(year)]
    eval_options["eval_days"] = [int(day)]
    tomato["eval_options"] = eval_options
    env = TomatoEnv(
        reward_function=tomato["reward_function"],
        observation_modules=tomato["observation_modules"],
        constraints=tomato["constraints"],
        eval_options=tomato["eval_options"],
        reward_params=tomato["reward_params"],
        base_env_params=base,
        uncertainty_scale=float(uncertainty_scale),
    )
    env.reset(seed=seed)
    return env


def state_to_record(state: Any) -> Dict[str, float]:
    names = (
        "timestep",
        "day_of_year",
        "hour_of_day",
        "temp_air",
        "rh_air",
        "co2_air",
        "pipe_temp",
        "fruit_weight",
        "canopy_temp_24h",
        "temperature_sum",
        "glob_rad",
        "temp_out",
        "rh_out",
        "wind_speed",
        "dli",
        "dew_margin_air",
        "canopy_dew_margin",
        "forecast_rad_mean_1h",
        "forecast_rad_peak_2h",
        "forecast_temp_out_delta_1h",
        "forecast_rh_out_mean_1h",
        "forecast_wind_peak_1h",
        "forecast_humidity_risk",
    )
    record: Dict[str, float] = {}
    for name in names:
        try:
            record[name] = float(getattr(state, name))
        except Exception:
            record[name] = 0.0
    return record


def info_to_metrics(info: Dict[str, Any]) -> Dict[str, float]:
    return {name: float(info.get(source, 0.0)) for name, source in METRIC_MAP.items()}


def control_to_record(control: Iterable[float], prefix: str = "u") -> Dict[str, float]:
    values = np.asarray(control, dtype=np.float32).reshape(-1)
    return {f"{prefix}_{name}": float(values[i]) if i < len(values) else 0.0 for i, name in enumerate(ACTION_NAMES)}


def regime_labels(row: Dict[str, Any]) -> Dict[str, str]:
    hour = float(row.get("hour_of_day", 0.0))
    rh = float(row.get("rh_air", 70.0))
    temp = float(row.get("temp_air", 20.0))
    rad = float(row.get("glob_rad", 0.0))
    if 4.0 <= hour < 8.0:
        phase = "dawn"
    elif 8.0 <= hour < 18.0:
        phase = "day"
    elif 18.0 <= hour < 21.0:
        phase = "dusk"
    else:
        phase = "night"
    if rh >= 90.0:
        rh_band = "rh_ge_90"
    elif rh >= 85.0:
        rh_band = "rh_85_90"
    elif rh >= 80.0:
        rh_band = "rh_80_85"
    else:
        rh_band = "rh_lt_80"
    if temp < 15.0:
        temp_band = "temp_lt_15"
    elif temp < 18.0:
        temp_band = "temp_15_18"
    elif temp < 22.0:
        temp_band = "temp_18_22"
    else:
        temp_band = "temp_ge_22"
    if rad < 20.0:
        rad_band = "dark"
    elif rad < 120.0:
        rad_band = "low_rad"
    else:
        rad_band = "useful_rad"
    return {"phase": phase, "rh_band": rh_band, "temp_band": temp_band, "rad_band": rad_band}


def append_regime_labels(row: Dict[str, Any]) -> Dict[str, Any]:
    row.update(regime_labels(row))
    return row


def run_ppo_trace(
    model: PPO,
    vecnorm_path: Path,
    raw_env: TomatoEnv,
    year: int,
    day: int,
    seed: int,
    max_steps: int,
) -> List[Dict[str, Any]]:
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(vecnorm_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    obs = vec_env.reset()
    interface = GreenhouseAgentInterface(raw_env)
    rows: List[Dict[str, Any]] = []
    done = False
    step = 0
    while (not done) and step < max_steps:
        state = interface.get_state()
        action_cmd, _ = model.predict(obs, deterministic=True)
        action_cmd_1d = np.asarray(action_cmd, dtype=np.float32).reshape(-1)
        planned_control = raw_env.action_to_control(action_cmd_1d)
        obs, rewards, dones, infos = vec_env.step(action_cmd)
        applied_control = np.asarray(getattr(raw_env, "u", planned_control), dtype=np.float32)
        row: Dict[str, Any] = {
            "algo": "ppo",
            "year": int(year),
            "day": int(day),
            "seed": int(seed),
            "step": int(step),
            "reward": float(rewards[0]),
            "done": bool(dones[0]),
            "source": "ppo",
            **state_to_record(state),
            **control_to_record(applied_control),
            **control_to_record(action_cmd_1d, prefix="action_cmd"),
            **info_to_metrics(dict(infos[0])),
        }
        rows.append(append_regime_labels(row))
        done = bool(dones[0])
        step += 1
    vec_env.close()
    return rows


def run_llm_trace(
    config: Dict[str, Any],
    api_key: str,
    year: int,
    day: int,
    seed: int,
    max_steps: int,
    llm_model: str,
    llm_interval: int,
    llm_max_iterations: int,
    llm_max_tokens: int,
    expert_rollout: bool,
    expert_policy_path: str,
    expert_max_distance: float,
    uncertainty_scale: float,
) -> List[Dict[str, Any]]:
    raw_env = build_env(config, year, day, seed, uncertainty_scale)
    interface = GreenhouseAgentInterface(raw_env)
    if hasattr(create_langchain_tools, "instance"):
        delattr(create_langchain_tools, "instance")
    tools = create_langchain_tools(interface)
    agent_cfg = AgentConfig(
        model_name=llm_model,
        api_key=api_key,
        verbose=False,
        control_interval=int(llm_interval),
        max_iterations=int(llm_max_iterations),
        max_tokens=int(llm_max_tokens),
        expert_rollout_enabled=bool(expert_rollout),
        expert_policy_path=expert_policy_path or AgentConfig.expert_policy_path,
        expert_candidate_max_distance=float(expert_max_distance),
    )
    agent = RuleBasedLLMDirector(
        agent_interface=interface,
        tools=tools,
        config=agent_cfg,
        rule_params=dict(DEFAULT_RULE_PARAMS),
    )
    raw_env.reset(seed=seed)
    rows: List[Dict[str, Any]] = []
    done = False
    step = 0
    while (not done) and step < max_steps:
        state = interface.get_state()
        try:
            result = agent.step_with_rules()
        except Exception as exc:
            row = {
                "algo": "llm_director",
                "year": int(year),
                "day": int(day),
                "seed": int(seed),
                "step": int(step),
                "reward": 0.0,
                "done": True,
                "runtime_error": str(exc),
                **state_to_record(state),
            }
            rows.append(append_regime_labels(row))
            break
        info = raw_env._get_info()
        plan = result.get("plan", {}) if isinstance(result, dict) else {}
        rollout = plan.get("rollout_selection", {}) if isinstance(plan, dict) else {}
        expert_info = rollout.get("expert_prediction", {}) if isinstance(rollout, dict) else {}
        applied_control = result.get("applied_control", getattr(raw_env, "u", np.zeros(6)))
        row = {
            "algo": "llm_director",
            "year": int(year),
            "day": int(day),
            "seed": int(seed),
            "step": int(step),
            "reward": float(result.get("reward", 0.0)),
            "done": bool(result.get("done", False)),
            "source": str(rollout.get("source", result.get("action", "unknown"))),
            "replan_reason": result.get("replan_reason"),
            "anchor_source": plan.get("anchor_source") if isinstance(plan, dict) else None,
            "target_temp": plan.get("current_target_temp") if isinstance(plan, dict) else None,
            "target_co2": plan.get("current_target_co2") if isinstance(plan, dict) else None,
            "target_rh": plan.get("current_target_rh") if isinstance(plan, dict) else None,
            "rh_violation_debt": plan.get("rh_violation_debt", 0.0) if isinstance(plan, dict) else 0.0,
            "expert_available": bool(expert_info.get("available", False)) if isinstance(expert_info, dict) else False,
            "expert_accepted": bool(expert_info.get("accepted", False)) if isinstance(expert_info, dict) else False,
            "expert_feature_distance": float(expert_info.get("feature_distance", 0.0)) if isinstance(expert_info, dict) else 0.0,
            **state_to_record(state),
            **control_to_record(applied_control),
            **control_to_record(result.get("anchor_control", np.zeros(6)), prefix="anchor"),
            **control_to_record(result.get("rule_control", np.zeros(6)), prefix="rule"),
            **info_to_metrics(info),
        }
        rows.append(append_regime_labels(row))
        done = bool(result.get("done", False))
        step += 1
    return rows


def sum_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "steps": len(rows),
        "total_reward": float(sum(float(r.get("reward", 0.0)) for r in rows)),
    }
    for metric in METRIC_MAP:
        summary[f"total_{metric}"] = float(sum(float(r.get(metric, 0.0)) for r in rows))
    if rows:
        for action in ACTION_NAMES:
            summary[f"mean_u_{action}"] = float(mean(float(r.get(f"u_{action}", 0.0)) for r in rows))
        summary["mean_rh"] = float(mean(float(r.get("rh_air", 0.0)) for r in rows))
        summary["mean_temp"] = float(mean(float(r.get("temp_air", 0.0)) for r in rows))
    return summary


def summarize_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(row)
    return {name: sum_metrics(group_rows) for name, group_rows in sorted(grouped.items())}


def pairwise_step_diffs(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_algo: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for row in rows:
        by_algo.setdefault(str(row["algo"]), {})[int(row["step"])] = row
    ppo = by_algo.get("ppo", {})
    llm = by_algo.get("llm_director", {})
    common_steps = sorted(set(ppo) & set(llm))
    diffs: List[Dict[str, Any]] = []
    for step in common_steps:
        p = ppo[step]
        l = llm[step]
        row: Dict[str, Any] = {
            "step": step,
            "phase": p.get("phase"),
            "rh_band": p.get("rh_band"),
            "temp_band": p.get("temp_band"),
            "ppo_reward": float(p.get("reward", 0.0)),
            "llm_reward": float(l.get("reward", 0.0)),
            "reward_diff_llm_minus_ppo": float(l.get("reward", 0.0)) - float(p.get("reward", 0.0)),
            "ppo_profit": float(p.get("profit", 0.0)),
            "llm_profit": float(l.get("profit", 0.0)),
            "profit_diff_llm_minus_ppo": float(l.get("profit", 0.0)) - float(p.get("profit", 0.0)),
        }
        for action in ACTION_NAMES:
            p_val = float(p.get(f"u_{action}", 0.0))
            l_val = float(l.get(f"u_{action}", 0.0))
            row[f"diff_{action}_llm_minus_ppo"] = l_val - p_val
            row[f"absdiff_{action}"] = abs(l_val - p_val)
        diffs.append(row)
    if not diffs:
        return {"common_steps": 0, "mean_abs_action_diff": {}, "largest_action_gap_steps": []}
    mean_abs = {
        action: float(mean(float(r[f"absdiff_{action}"]) for r in diffs))
        for action in ACTION_NAMES
    }
    largest = sorted(
        diffs,
        key=lambda r: sum(float(r[f"absdiff_{action}"]) for action in ACTION_NAMES),
        reverse=True,
    )[:10]
    return {
        "common_steps": len(common_steps),
        "mean_abs_action_diff": mean_abs,
        "sum_reward_diff_llm_minus_ppo": float(sum(r["reward_diff_llm_minus_ppo"] for r in diffs)),
        "sum_profit_diff_llm_minus_ppo": float(sum(r["profit_diff_llm_minus_ppo"] for r in diffs)),
        "largest_action_gap_steps": largest,
        "by_phase": summarize_pairwise_by(diffs, "phase"),
        "by_rh_band": summarize_pairwise_by(diffs, "rh_band"),
    }


def summarize_pairwise_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "unknown")), []).append(row)
    out: Dict[str, Any] = {}
    for name, group_rows in sorted(grouped.items()):
        out[name] = {
            "steps": len(group_rows),
            "reward_diff_llm_minus_ppo": float(sum(r["reward_diff_llm_minus_ppo"] for r in group_rows)),
            "profit_diff_llm_minus_ppo": float(sum(r["profit_diff_llm_minus_ppo"] for r in group_rows)),
            "mean_abs_action_diff": {
                action: float(mean(float(r[f"absdiff_{action}"]) for r in group_rows))
                for action in ACTION_NAMES
            },
        }
    return out


def safety_patterns(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for algo in sorted({str(r["algo"]) for r in rows}):
        selected = [r for r in rows if r["algo"] == algo]
        heat_vent = [r for r in selected if float(r.get("u_heating", 0.0)) > 0.15 and float(r.get("u_ventilation", 0.0)) > 0.30]
        co2_leak = [r for r in selected if float(r.get("u_co2", 0.0)) > 0.05 and float(r.get("u_ventilation", 0.0)) > 0.20]
        lamp_risk = [
            r
            for r in selected
            if float(r.get("u_lighting", 0.0)) > 0.05
            and (float(r.get("rh_air", 0.0)) >= 85.0 or float(r.get("u_ventilation", 0.0)) >= 0.30)
        ]
        high_rh = [r for r in selected if float(r.get("rh_air", 0.0)) >= 90.0]
        out[algo] = {
            "heat_vent_conflict_steps": len(heat_vent),
            "co2_leak_steps": len(co2_leak),
            "lamp_risk_steps": len(lamp_risk),
            "state_rh_ge_90_steps": len(high_rh),
            "mean_vent_when_rh_ge_90": float(mean(float(r.get("u_ventilation", 0.0)) for r in high_rh)) if high_rh else 0.0,
            "mean_heat_when_rh_ge_90": float(mean(float(r.get("u_heating", 0.0)) for r in high_rh)) if high_rh else 0.0,
        }
    return out


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# PPO vs LLM-RSPC Diagnostic Report",
        "",
        f"- Scenario: year={summary['scenario']['year']}, day={summary['scenario']['day']}, seed={summary['scenario']['seed']}",
        f"- Steps: {summary['scenario']['max_steps']}",
        f"- Expert rollout: {summary['scenario']['expert_rollout']}",
        "",
        "## Aggregate",
        "| Algorithm | Reward | Profit | Revenue | Heat | CO2 | Elec | Temp V | RH V |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for algo, item in summary["aggregate"].items():
        lines.append(
            f"| {algo} | {item['total_reward']:.3f} | {item['total_profit']:.5f} | "
            f"{item['total_revenue']:.5f} | {item['total_heat_cost']:.5f} | "
            f"{item['total_co2_cost']:.5f} | {item['total_elec_cost']:.5f} | "
            f"{item['total_temp_violation']:.3f} | {item['total_rh_violation']:.3f} |"
        )
    pair = summary.get("pairwise", {})
    lines.extend(
        [
            "",
            "## PPO - LLM Gap",
            f"- Common steps: {pair.get('common_steps', 0)}",
            f"- Sum reward diff (LLM minus PPO): {pair.get('sum_reward_diff_llm_minus_ppo', 0.0):.3f}",
            f"- Sum profit diff (LLM minus PPO): {pair.get('sum_profit_diff_llm_minus_ppo', 0.0):.5f}",
            "- Mean absolute action difference:",
        ]
    )
    for action, value in pair.get("mean_abs_action_diff", {}).items():
        lines.append(f"  - {action}: {value:.3f}")
    lines.extend(["", "## Safety Patterns"])
    for algo, item in summary.get("safety_patterns", {}).items():
        lines.append(
            f"- {algo}: heat+vent conflict={item['heat_vent_conflict_steps']}, "
            f"CO2 leak={item['co2_leak_steps']}, lamp risk={item['lamp_risk_steps']}, "
            f"RH>=90 state steps={item['state_rh_ge_90_steps']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose PPO vs LLM-RSPC trajectories.")
    parser.add_argument("--config", type=str, default="gl_gym/configs/envs/TomatoEnv.yml")
    parser.add_argument("--model-path", type=str, default="train_data/AgriControl/ppo/deterministic/models/None/best_model.zip")
    parser.add_argument("--vecnorm-path", type=str, default="train_data/AgriControl/ppo/deterministic/envs/None/best_vecnormalize.pkl")
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--day", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--uncertainty-scale", type=float, default=0.0)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--llm-model", type=str, default="qwen-max-latest")
    parser.add_argument("--llm-interval", type=int, default=12)
    parser.add_argument("--llm-max-iterations", type=int, default=1)
    parser.add_argument("--llm-max-tokens", type=int, default=260)
    parser.add_argument("--llm-expert-rollout", action="store_true")
    parser.add_argument("--llm-expert-policy-path", type=str, default=AgentConfig.expert_policy_path)
    parser.add_argument("--llm-expert-max-distance", type=float, default=AgentConfig.expert_candidate_max_distance)
    parser.add_argument("--output-json", type=str, default="gl_gym/result/diagnostics/ppo_vs_llm_diag.json")
    parser.add_argument("--output-csv", type=str, default="gl_gym/result/diagnostics/ppo_vs_llm_trace.csv")
    parser.add_argument("--output-report", type=str, default="gl_gym/result/diagnostics/ppo_vs_llm_diag.md")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_path = Path(args.model_path)
    vecnorm_path = Path(args.vecnorm_path)
    model = PPO.load(model_path)

    rows: List[Dict[str, Any]] = []
    ppo_env = build_env(config, args.year, args.day, args.seed, args.uncertainty_scale)
    t0 = time.perf_counter()
    rows.extend(run_ppo_trace(model, vecnorm_path, ppo_env, args.year, args.day, args.seed, args.max_steps))
    ppo_elapsed = time.perf_counter() - t0

    llm_elapsed = 0.0
    api_key = os.getenv("BAILIAN_API_KEY")
    if not args.skip_llm and api_key:
        t1 = time.perf_counter()
        rows.extend(
            run_llm_trace(
                config=config,
                api_key=api_key,
                year=args.year,
                day=args.day,
                seed=args.seed,
                max_steps=args.max_steps,
                llm_model=args.llm_model,
                llm_interval=args.llm_interval,
                llm_max_iterations=args.llm_max_iterations,
                llm_max_tokens=args.llm_max_tokens,
                expert_rollout=args.llm_expert_rollout,
                expert_policy_path=args.llm_expert_policy_path,
                expert_max_distance=args.llm_expert_max_distance,
                uncertainty_scale=args.uncertainty_scale,
            )
        )
        llm_elapsed = time.perf_counter() - t1
    elif not args.skip_llm:
        print("BAILIAN_API_KEY is missing; PPO trace only.")

    aggregate = {}
    for algo in sorted({str(r["algo"]) for r in rows}):
        aggregate[algo] = sum_metrics([r for r in rows if r["algo"] == algo])

    summary = {
        "scenario": {
            "year": int(args.year),
            "day": int(args.day),
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "uncertainty_scale": float(args.uncertainty_scale),
            "expert_rollout": bool(args.llm_expert_rollout),
        },
        "timing": {"ppo_seconds": float(ppo_elapsed), "llm_seconds": float(llm_elapsed)},
        "aggregate": aggregate,
        "by_phase": {algo: summarize_by([r for r in rows if r["algo"] == algo], "phase") for algo in aggregate},
        "by_rh_band": {algo: summarize_by([r for r in rows if r["algo"] == algo], "rh_band") for algo in aggregate},
        "by_temp_band": {algo: summarize_by([r for r in rows if r["algo"] == algo], "temp_band") for algo in aggregate},
        "pairwise": pairwise_step_diffs(rows),
        "safety_patterns": safety_patterns(rows),
    }

    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    report_path = Path(args.output_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    write_rows_csv(csv_path, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(summary), encoding="utf-8")
    print(f"Saved diagnostic JSON to {json_path}")
    print(f"Saved trace CSV to {csv_path}")
    print(f"Saved report to {report_path}")
    print(build_report(summary))


if __name__ == "__main__":
    main()
