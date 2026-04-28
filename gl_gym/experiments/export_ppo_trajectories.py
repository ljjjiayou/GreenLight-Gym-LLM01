"""Export PPO expert trajectories for LLM-RSPC rollout distillation.

This script runs the existing PPO policy on multiple greenhouse scenarios and
records state features together with the raw control selected by PPO. The output
is intentionally audit-friendly JSONL plus a compact NPZ matrix for training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gl_gym.agent.expert_distillation import ACTION_NAMES, FEATURE_NAMES, extract_expert_features
from gl_gym.agent.interface import GreenhouseAgentInterface
from gl_gym.agent.plan_intent import (
    action_record_for_labeling,
    default_setpoint_contract,
    infer_plan_intent,
    intent_to_dehumidify_mode,
    strategy_intent_alignment,
    target_tracking_baseline_control,
)
from gl_gym.agent.ppo_strategy_labeler import label_strategy
from gl_gym.environments.tomato_env import TomatoEnv


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_tomato_env(config: Dict[str, Any], year: int, day: int, seed: int, uncertainty_scale: float) -> TomatoEnv:
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
        uncertainty_scale=uncertainty_scale,
    )
    env.reset(seed=seed)
    return env


def run_ppo_case(
    model: PPO,
    vecnorm_path: Path,
    raw_env: TomatoEnv,
    year: int,
    day: int,
    seed: int,
    repeat: int,
    max_steps: int,
    plan_horizon: int,
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
        plan, plan_contract = default_setpoint_contract(state, horizon=plan_horizon)
        intent = infer_plan_intent(state, plan, horizon=plan_horizon)
        dehumidify_mode = intent_to_dehumidify_mode(intent.label)
        feature, feature_dict = extract_expert_features(
            state,
            plan=plan,
            dehumidify_mode=dehumidify_mode,
        )
        base_action = target_tracking_baseline_control(state, plan, intent)
        action_cmd, _ = model.predict(obs, deterministic=True)
        action_cmd_1d = np.asarray(action_cmd, dtype=np.float32).reshape(-1)
        try:
            planned_control = raw_env.action_to_control(action_cmd_1d)
        except Exception:
            planned_control = np.asarray(getattr(raw_env, "u", np.zeros(6)), dtype=np.float32)

        obs, rewards, dones, infos = vec_env.step(action_cmd)
        reward = float(rewards[0])
        done = bool(dones[0])
        info = dict(infos[0])
        applied_control = np.asarray(getattr(raw_env, "u", planned_control), dtype=np.float32)
        strategy = label_strategy(action_record_for_labeling(state, applied_control), action_prefix="u")
        alignment = strategy_intent_alignment(strategy, intent)

        rows.append(
            {
                "year": int(year),
                "day": int(day),
                "seed": int(seed),
                "repeat": int(repeat),
                "step": int(step),
                "timestep": int(getattr(state, "timestep", step)),
                "hour_of_day": float(getattr(state, "hour_of_day", 0.0)),
                "day_of_year": float(getattr(state, "day_of_year", 0.0)),
                "features": {name: float(feature_dict[name]) for name in FEATURE_NAMES},
                "feature": [float(x) for x in feature],
                "action": [float(x) for x in applied_control],
                "base_action": [float(x) for x in base_action],
                "residual_action": [float(x) for x in (applied_control - base_action)],
                "planned_control": [float(x) for x in planned_control],
                "ppo_action_command": [float(x) for x in action_cmd_1d],
                "target_plan": {
                    "target_temp": float(plan["target_temp"]),
                    "target_co2": float(plan["target_co2"]),
                    "target_rh": float(plan["target_rh"]),
                    "target_profile": plan.get("target_profile", {}),
                    "created_timestep": int(plan["created_timestep"]),
                    "expires_timestep": int(plan["expires_timestep"]),
                },
                "plan_contract": plan_contract,
                "intent": intent.to_record(prefix="intent"),
                "strategy": strategy.to_record(prefix="strategy"),
                "intent_strategy_alignment": alignment,
                "reward": reward,
                "done": done,
                "info": {
                    "profit": float(info.get("EPI", 0.0)),
                    "revenue": float(info.get("revenue", 0.0)),
                    "heat_cost": float(info.get("heat_cost", 0.0)),
                    "co2_cost": float(info.get("co2_cost", 0.0)),
                    "elec_cost": float(info.get("elec_cost", 0.0)),
                    "temp_violation": float(info.get("temp_violation", 0.0)),
                    "co2_violation": float(info.get("co2_violation", 0.0)),
                    "rh_violation": float(info.get("rh_violation", 0.0)),
                    "lamp_violation": float(info.get("lamp_violation", 0.0)),
                },
            }
        )
        step += 1
    vec_env.close()
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_npz(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y = np.asarray([row["action"] for row in rows], dtype=np.float32)
    base_action = np.asarray([row.get("base_action", np.zeros(len(ACTION_NAMES))) for row in rows], dtype=np.float32)
    residual_action = np.asarray([row.get("residual_action", np.zeros(len(ACTION_NAMES))) for row in rows], dtype=np.float32)
    scenario = np.asarray(
        [[row["year"], row["day"], row["seed"], row["repeat"], row["step"]] for row in rows],
        dtype=np.int32,
    )
    intent_label = np.asarray([row.get("intent", {}).get("intent_label", "unknown") for row in rows])
    intent_confidence = np.asarray(
        [float(row.get("intent", {}).get("intent_confidence", 0.0)) for row in rows],
        dtype=np.float32,
    )
    strategy_label = np.asarray([row.get("strategy", {}).get("strategy_label", "unknown") for row in rows])
    strategy_confidence = np.asarray(
        [float(row.get("strategy", {}).get("strategy_confidence", 0.0)) for row in rows],
        dtype=np.float32,
    )
    intent_strategy_aligned = np.asarray(
        [bool(row.get("intent_strategy_alignment", {}).get("aligned", False)) for row in rows],
        dtype=np.bool_,
    )
    np.savez_compressed(
        path,
        x=x,
        y=y,
        base_action=base_action,
        residual_action=residual_action,
        scenario=scenario,
        intent_label=intent_label,
        intent_confidence=intent_confidence,
        strategy_label=strategy_label,
        strategy_confidence=strategy_confidence,
        intent_strategy_aligned=intent_strategy_aligned,
        feature_names=np.asarray(FEATURE_NAMES),
        action_names=np.asarray(ACTION_NAMES),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PPO trajectories for expert distillation.")
    parser.add_argument("--config", type=str, default="gl_gym/configs/envs/TomatoEnv.yml")
    parser.add_argument("--model-path", type=str, default="train_data/AgriControl/ppo/deterministic/models/None/best_model.zip")
    parser.add_argument("--vecnorm-path", type=str, default="train_data/AgriControl/ppo/deterministic/envs/None/best_vecnormalize.pkl")
    parser.add_argument("--years", type=str, default="2018,2019,2020")
    parser.add_argument("--days", type=str, default="120,180,240")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--plan-horizon", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--uncertainty-scale", type=float, default=0.0)
    parser.add_argument("--output-jsonl", type=str, default="gl_gym/result/expert_distillation/ppo_expert_s240.jsonl")
    parser.add_argument("--output-npz", type=str, default="gl_gym/result/expert_distillation/ppo_expert_s240.npz")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = Path(args.model_path)
    vecnorm_path = Path(args.vecnorm_path)
    if not model_path.exists():
        raise FileNotFoundError(f"PPO model not found: {model_path}")
    if not vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found: {vecnorm_path}")

    model = PPO.load(model_path)
    years = parse_int_list(args.years)
    days = parse_int_list(args.days)
    all_rows: List[Dict[str, Any]] = []

    for repeat in range(int(args.repeats)):
        seed = int(args.base_seed) + repeat
        for year in years:
            for day in days:
                print(f"Exporting PPO trajectory year={year} day={day} seed={seed} max_steps={args.max_steps}")
                env = build_tomato_env(config, year, day, seed, float(args.uncertainty_scale))
                rows = run_ppo_case(
                    model,
                    vecnorm_path,
                    env,
                    year,
                    day,
                    seed,
                    repeat,
                    int(args.max_steps),
                    int(args.plan_horizon),
                )
                all_rows.extend(rows)

    jsonl_path = Path(args.output_jsonl)
    npz_path = Path(args.output_npz)
    write_jsonl(jsonl_path, all_rows)
    write_npz(npz_path, all_rows)
    print(f"Saved {len(all_rows)} rows to {jsonl_path}")
    print(f"Saved matrix dataset to {npz_path}")


if __name__ == "__main__":
    main()
