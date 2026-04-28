"""Train a lightweight PPO-distilled rollout expert.

The model is a normalized ridge regressor. This keeps the expert compact,
deterministic, easy to inspect, and safe to use only as a rollout candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from gl_gym.agent.expert_distillation import ACTION_NAMES, FEATURE_NAMES


def _nested_float(row: Dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    value = row.get(section, {})
    if isinstance(value, dict):
        value = value.get(key, default)
    else:
        value = row.get(key, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _target_from_action(row: Dict[str, Any], target_mode: str) -> np.ndarray:
    action = np.asarray(row.get("action", row.get("control")), dtype=np.float32)
    if target_mode == "residual":
        if "residual_action" in row:
            return np.asarray(row["residual_action"], dtype=np.float32)
        if "base_action" not in row:
            raise ValueError("Residual target requested, but row does not contain base_action")
        return action - np.asarray(row["base_action"], dtype=np.float32)
    return action


def load_dataset(
    path: Path,
    target_mode: str = "action",
    require_intent_aligned: bool = False,
    min_intent_confidence: float = 0.0,
    min_strategy_confidence: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        x_all = np.asarray(data["x"], dtype=np.float32)
        action_all = np.asarray(data["y"], dtype=np.float32)
        mask = np.ones(action_all.shape[0], dtype=bool)
        if require_intent_aligned:
            if "intent_strategy_aligned" not in data:
                raise ValueError("Dataset has no intent_strategy_aligned mask")
            mask &= np.asarray(data["intent_strategy_aligned"], dtype=bool)
        if min_intent_confidence > 0.0:
            if "intent_confidence" not in data:
                raise ValueError("Dataset has no intent_confidence values")
            mask &= np.asarray(data["intent_confidence"], dtype=np.float32) >= float(min_intent_confidence)
        if min_strategy_confidence > 0.0:
            if "strategy_confidence" not in data:
                raise ValueError("Dataset has no strategy_confidence values")
            mask &= np.asarray(data["strategy_confidence"], dtype=np.float32) >= float(min_strategy_confidence)
        if target_mode == "residual":
            if "residual_action" in data:
                y_all = np.asarray(data["residual_action"], dtype=np.float32)
            elif "base_action" in data:
                y_all = action_all - np.asarray(data["base_action"], dtype=np.float32)
            else:
                raise ValueError("Residual target requested, but dataset has no residual_action/base_action")
        else:
            y_all = action_all
        x = x_all[mask]
        y = y_all[mask]
        if x.shape[0] < 2:
            raise ValueError(f"Need at least 2 selected rows, got {x.shape[0]}")
        stats = {
            "source_rows": int(action_all.shape[0]),
            "selected_rows": int(x.shape[0]),
            "target_mode": target_mode,
            "require_intent_aligned": bool(require_intent_aligned),
            "min_intent_confidence": float(min_intent_confidence),
            "min_strategy_confidence": float(min_strategy_confidence),
        }
        return x, y, stats

    features = []
    actions = []
    source_rows = 0
    rejected_rows = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            source_rows += 1
            feature_map = row.get("features", {})
            action = row.get("action", row.get("control"))
            if action is None:
                rejected_rows += 1
                continue
            aligned = bool(row.get("intent_strategy_alignment", {}).get("aligned", False))
            intent_confidence = _nested_float(row, "intent", "intent_confidence", 0.0)
            strategy_confidence = _nested_float(row, "strategy", "strategy_confidence", 0.0)
            if require_intent_aligned and not aligned:
                rejected_rows += 1
                continue
            if intent_confidence < float(min_intent_confidence):
                rejected_rows += 1
                continue
            if strategy_confidence < float(min_strategy_confidence):
                rejected_rows += 1
                continue
            features.append([float(feature_map[name]) for name in FEATURE_NAMES])
            actions.append([float(x) for x in _target_from_action(row, target_mode)])
    if not features:
        raise ValueError(f"No usable trajectory rows in {path}")
    if len(features) < 2:
        raise ValueError(f"Need at least 2 selected rows, got {len(features)}")
    stats = {
        "source_rows": int(source_rows),
        "selected_rows": int(len(features)),
        "rejected_rows": int(rejected_rows),
        "target_mode": target_mode,
        "require_intent_aligned": bool(require_intent_aligned),
        "min_intent_confidence": float(min_intent_confidence),
        "min_strategy_confidence": float(min_strategy_confidence),
    }
    return np.asarray(features, dtype=np.float32), np.asarray(actions, dtype=np.float32), stats


def _clip_prediction(values: np.ndarray, target_mode: str) -> np.ndarray:
    if target_mode == "residual":
        return np.clip(values, -1.0, 1.0)
    return np.clip(values, 0.0, 1.0)


def train_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    validation_fraction: float,
    seed: int,
    target_mode: str = "action",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    val_size = int(np.clip(round(n * validation_fraction), 1, max(n - 1, 1))) if n > 1 else 0
    val_idx = indices[:val_size]
    train_idx = indices[val_size:] if val_size > 0 else indices

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_val = x[val_idx] if val_size > 0 else x_train
    y_val = y[val_idx] if val_size > 0 else y_train

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(np.abs(std) < 1e-6, 1.0, std)
    xz = (x_train - mean) / std
    xz_val = (x_val - mean) / std

    x_aug = np.concatenate([xz, np.ones((xz.shape[0], 1), dtype=np.float32)], axis=1)
    reg = np.eye(x_aug.shape[1], dtype=np.float32) * float(alpha)
    reg[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + reg
    rhs = x_aug.T @ y_train
    coef_aug = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    coef = coef_aug[:-1]
    intercept = coef_aug[-1]

    train_pred = _clip_prediction(xz @ coef + intercept, target_mode)
    val_pred = _clip_prediction(xz_val @ coef + intercept, target_mode)
    train_mae = np.mean(np.abs(train_pred - y_train), axis=0)
    val_mae = np.mean(np.abs(val_pred - y_val), axis=0)
    distance = np.mean(np.abs(xz), axis=1)

    metrics = {
        "rows": int(n),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "alpha": float(alpha),
        "target_mode": target_mode,
        "train_mse": float(np.mean((train_pred - y_train) ** 2)),
        "validation_mse": float(np.mean((val_pred - y_val) ** 2)),
        "train_mae": {name: float(value) for name, value in zip(ACTION_NAMES, train_mae)},
        "validation_mae": {name: float(value) for name, value in zip(ACTION_NAMES, val_mae)},
        "feature_distance_p95": float(np.percentile(distance, 95.0)),
        "feature_distance_p99": float(np.percentile(distance, 99.0)),
    }
    model = {
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "coef": coef.astype(np.float32),
        "intercept": intercept.astype(np.float32),
        "metrics": metrics,
    }
    return model, metrics


def save_model(path: Path, model: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        feature_names=np.asarray(FEATURE_NAMES),
        action_names=np.asarray(ACTION_NAMES),
        feature_mean=model["feature_mean"],
        feature_std=model["feature_std"],
        coef=model["coef"],
        intercept=model["intercept"],
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO-distilled rollout expert.")
    parser.add_argument("--input", type=str, default="gl_gym/result/expert_distillation/ppo_expert_s240.npz")
    parser.add_argument("--output-model", type=str, default="train_data/AgriControl/ppo/deterministic/distilled_expert/llm_rspc_expert_ridge.npz")
    parser.add_argument("--output-report", type=str, default="gl_gym/result/expert_distillation/llm_rspc_expert_ridge_metrics.json")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-mode", type=str, choices=["action", "residual"], default="action")
    parser.add_argument("--require-intent-aligned", action="store_true")
    parser.add_argument("--min-intent-confidence", type=float, default=0.0)
    parser.add_argument("--min-strategy-confidence", type=float, default=0.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Trajectory dataset not found: {input_path}")

    x, y, selection = load_dataset(
        input_path,
        target_mode=str(args.target_mode),
        require_intent_aligned=bool(args.require_intent_aligned),
        min_intent_confidence=float(args.min_intent_confidence),
        min_strategy_confidence=float(args.min_strategy_confidence),
    )
    model, metrics = train_ridge(
        x,
        y,
        alpha=float(args.alpha),
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
        target_mode=str(args.target_mode),
    )
    metadata = {
        "model": "normalized_ridge",
        "source_dataset": str(input_path),
        "target_mode": str(args.target_mode),
        "feature_names": list(FEATURE_NAMES),
        "action_names": list(ACTION_NAMES),
        "train_cases": int(metrics["rows"]),
        "selection": selection,
        **metrics,
    }

    save_model(Path(args.output_model), model, metadata)
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved distilled expert model to {args.output_model}")
    print(f"Saved metrics report to {args.output_report}")


if __name__ == "__main__":
    main()
