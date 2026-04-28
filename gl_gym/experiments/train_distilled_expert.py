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


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        return np.asarray(data["x"], dtype=np.float32), np.asarray(data["y"], dtype=np.float32)

    features = []
    actions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            feature_map = row.get("features", {})
            action = row.get("action", row.get("control"))
            if action is None:
                continue
            features.append([float(feature_map[name]) for name in FEATURE_NAMES])
            actions.append([float(x) for x in action])
    if not features:
        raise ValueError(f"No usable trajectory rows in {path}")
    return np.asarray(features, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def train_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    validation_fraction: float,
    seed: int,
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

    train_pred = np.clip(xz @ coef + intercept, 0.0, 1.0)
    val_pred = np.clip(xz_val @ coef + intercept, 0.0, 1.0)
    train_mae = np.mean(np.abs(train_pred - y_train), axis=0)
    val_mae = np.mean(np.abs(val_pred - y_val), axis=0)
    distance = np.mean(np.abs(xz), axis=1)

    metrics = {
        "rows": int(n),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "alpha": float(alpha),
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
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Trajectory dataset not found: {input_path}")

    x, y = load_dataset(input_path)
    model, metrics = train_ridge(
        x,
        y,
        alpha=float(args.alpha),
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
    )
    metadata = {
        "model": "normalized_ridge",
        "source_dataset": str(input_path),
        "feature_names": list(FEATURE_NAMES),
        "action_names": list(ACTION_NAMES),
        "train_cases": int(metrics["rows"]),
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
