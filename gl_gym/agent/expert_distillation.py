"""Utilities for PPO expert trajectory distillation.

The distilled expert is intentionally small: a normalized ridge model that maps
greenhouse state features to raw control values in [0, 1]. It is not a safety
controller by itself. The LLM director treats its prediction as one candidate in
the safe rollout pool, then applies the existing score-and-guardrail layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from gl_gym.common.utils import calculate_vpd_kpa


ACTION_NAMES = (
    "heating",
    "co2",
    "screen",
    "ventilation",
    "lighting",
    "shading",
)


FEATURE_NAMES = (
    "temp_air",
    "rh_air",
    "co2_air",
    "pipe_temp",
    "fruit_weight",
    "canopy_temp_24h",
    "temperature_sum",
    "u_boil",
    "u_co2",
    "u_th_scr",
    "u_vent",
    "u_lamp",
    "u_bl_scr",
    "glob_rad",
    "temp_out",
    "rh_out",
    "co2_out",
    "wind_speed",
    "dli",
    "vpd_kpa",
    "dew_margin_air",
    "canopy_dew_margin",
    "forecast_rad_mean_1h",
    "forecast_rad_peak_2h",
    "forecast_temp_out_delta_1h",
    "forecast_rh_out_mean_1h",
    "forecast_wind_peak_1h",
    "forecast_humidity_risk",
    "time_to_sunrise_steps",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "rh_violation_debt",
    "lamp_budget_remaining",
    "target_temp_delta",
    "target_co2_delta",
    "target_rh_delta",
    "dehumidify_mild",
    "dehumidify_strong",
)


def _float_attr(obj: Any, name: str, default: float) -> float:
    try:
        return float(getattr(obj, name, default))
    except Exception:
        return float(default)


def _plan_target(plan: Optional[Dict[str, Any]], key: str, state: Any) -> Optional[float]:
    if not isinstance(plan, dict):
        return None
    profile = plan.get("target_profile", {})
    values = profile.get(key) if isinstance(profile, dict) else None
    if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 0:
        created = int(plan.get("created_timestep", _float_attr(state, "timestep", 0.0)))
        idx = int(np.clip(int(_float_attr(state, "timestep", 0.0)) - created, 0, len(values) - 1))
        try:
            return float(values[idx])
        except Exception:
            pass
    value = plan.get(key)
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def extract_expert_features(
    state: Any,
    plan: Optional[Dict[str, Any]] = None,
    rh_violation_debt: float = 0.0,
    lamp_budget_remaining: Optional[float] = None,
    dehumidify_mode: str = "normal",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Build a physics-aware feature vector for expert distillation.

    The feature vector avoids absolute timestep indices so the model learns
    climate regimes rather than memorizing a fixed trajectory.
    """

    temp_air = _float_attr(state, "temp_air", 20.0)
    rh_air = _float_attr(state, "rh_air", 70.0)
    co2_air = _float_attr(state, "co2_air", 430.0)
    hour = _float_attr(state, "hour_of_day", 12.0)
    day = _float_attr(state, "day_of_year", 180.0)
    vpd = float(calculate_vpd_kpa(temp_air, rh_air))
    target_temp = _plan_target(plan, "target_temp", state)
    target_co2 = _plan_target(plan, "target_co2", state)
    target_rh = _plan_target(plan, "target_rh", state)
    if lamp_budget_remaining is None:
        lamp_budget_remaining = 0.0

    values: Dict[str, float] = {
        "temp_air": temp_air,
        "rh_air": rh_air,
        "co2_air": co2_air,
        "pipe_temp": _float_attr(state, "pipe_temp", 30.0),
        "fruit_weight": _float_attr(state, "fruit_weight", 0.0),
        "canopy_temp_24h": _float_attr(state, "canopy_temp_24h", temp_air),
        "temperature_sum": _float_attr(state, "temperature_sum", 0.0),
        "u_boil": _float_attr(state, "u_boil", 0.0),
        "u_co2": _float_attr(state, "u_co2", 0.0),
        "u_th_scr": _float_attr(state, "u_th_scr", 0.0),
        "u_vent": _float_attr(state, "u_vent", 0.0),
        "u_lamp": _float_attr(state, "u_lamp", 0.0),
        "u_bl_scr": _float_attr(state, "u_bl_scr", 0.0),
        "glob_rad": _float_attr(state, "glob_rad", 0.0),
        "temp_out": _float_attr(state, "temp_out", temp_air),
        "rh_out": _float_attr(state, "rh_out", 70.0),
        "co2_out": _float_attr(state, "co2_out", 430.0),
        "wind_speed": _float_attr(state, "wind_speed", 0.0),
        "dli": _float_attr(state, "dli", 0.0),
        "vpd_kpa": vpd,
        "dew_margin_air": _float_attr(state, "dew_margin_air", 3.0),
        "canopy_dew_margin": _float_attr(state, "canopy_dew_margin", 3.0),
        "forecast_rad_mean_1h": _float_attr(state, "forecast_rad_mean_1h", _float_attr(state, "glob_rad", 0.0)),
        "forecast_rad_peak_2h": _float_attr(state, "forecast_rad_peak_2h", _float_attr(state, "glob_rad", 0.0)),
        "forecast_temp_out_delta_1h": _float_attr(state, "forecast_temp_out_delta_1h", 0.0),
        "forecast_rh_out_mean_1h": _float_attr(state, "forecast_rh_out_mean_1h", _float_attr(state, "rh_out", 70.0)),
        "forecast_wind_peak_1h": _float_attr(state, "forecast_wind_peak_1h", _float_attr(state, "wind_speed", 0.0)),
        "forecast_humidity_risk": _float_attr(state, "forecast_humidity_risk", 0.0),
        "time_to_sunrise_steps": _float_attr(state, "time_to_sunrise_steps", 4.0),
        "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
        "day_sin": float(np.sin(2.0 * np.pi * day / 365.0)),
        "day_cos": float(np.cos(2.0 * np.pi * day / 365.0)),
        "rh_violation_debt": float(rh_violation_debt),
        "lamp_budget_remaining": float(lamp_budget_remaining),
        "target_temp_delta": 0.0 if target_temp is None else float(target_temp - temp_air),
        "target_co2_delta": 0.0 if target_co2 is None else float((target_co2 - co2_air) / 100.0),
        "target_rh_delta": 0.0 if target_rh is None else float(target_rh - rh_air),
        "dehumidify_mild": 1.0 if dehumidify_mode == "mild" else 0.0,
        "dehumidify_strong": 1.0 if dehumidify_mode == "strong" else 0.0,
    }
    feature = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    return feature, values


@dataclass
class DistilledExpertPolicy:
    feature_names: Tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    action_names: Tuple[str, ...] = ACTION_NAMES
    metadata: Dict[str, Any] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "DistilledExpertPolicy":
        data = np.load(Path(path), allow_pickle=False)
        metadata: Dict[str, Any] = {}
        if "metadata_json" in data:
            try:
                metadata = json.loads(str(data["metadata_json"].item()))
            except Exception:
                metadata = {}
        return cls(
            feature_names=tuple(str(x) for x in data["feature_names"].tolist()),
            feature_mean=np.asarray(data["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(data["feature_std"], dtype=np.float32),
            coef=np.asarray(data["coef"], dtype=np.float32),
            intercept=np.asarray(data["intercept"], dtype=np.float32),
            action_names=tuple(str(x) for x in data["action_names"].tolist()) if "action_names" in data else ACTION_NAMES,
            metadata=metadata,
        )

    def _standardize(self, feature: np.ndarray) -> np.ndarray:
        std = np.where(np.abs(self.feature_std) < 1e-6, 1.0, self.feature_std)
        return (np.asarray(feature, dtype=np.float32) - self.feature_mean) / std

    def predict_from_features(self, feature: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        if len(feature) != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {len(feature)}")
        z = self._standardize(feature)
        raw = z @ self.coef + self.intercept
        action = np.clip(raw, 0.0, 1.0).astype(np.float32)
        distance = float(np.mean(np.abs(z)))
        max_abs_z = float(np.max(np.abs(z))) if len(z) else 0.0
        threshold = None
        threshold_p99 = None
        if self.metadata:
            threshold = self.metadata.get("feature_distance_p95")
            threshold_p99 = self.metadata.get("feature_distance_p99")
        return action, {
            "feature_distance": distance,
            "max_abs_z": max_abs_z,
            "distance_threshold": threshold,
            "distance_threshold_p99": threshold_p99,
            "model": (self.metadata or {}).get("model", "ridge"),
            "train_cases": (self.metadata or {}).get("train_cases"),
        }

    def predict(
        self,
        state: Any,
        plan: Optional[Dict[str, Any]] = None,
        rh_violation_debt: float = 0.0,
        lamp_budget_remaining: Optional[float] = None,
        dehumidify_mode: str = "normal",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        feature, feature_dict = extract_expert_features(
            state,
            plan=plan,
            rh_violation_debt=rh_violation_debt,
            lamp_budget_remaining=lamp_budget_remaining,
            dehumidify_mode=dehumidify_mode,
        )
        action, info = self.predict_from_features(feature)
        info["features"] = feature_dict
        return action, info


def rows_to_arrays(rows: Iterable[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    actions = []
    for row in rows:
        feature_map = row.get("features", {})
        action = row.get("action", row.get("control"))
        if action is None:
            continue
        features.append([float(feature_map[name]) for name in FEATURE_NAMES])
        actions.append([float(x) for x in action])
    if not features:
        raise ValueError("No valid rows found for expert distillation")
    return np.asarray(features, dtype=np.float32), np.asarray(actions, dtype=np.float32)
