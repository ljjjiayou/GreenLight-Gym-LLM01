"""Plan cache utilities for frozen LLM-RSPC benchmarks.

The cache stores the LLM planning boundary, not rollout controls. This keeps
replay tied to the same prompt/config/state while still letting downstream RSPC
changes compete against identical high-level plans.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


SCHEMA_VERSION = "llm_plan_cache_v1"
VALID_MODES = {"off", "record", "replay", "refresh"}


def to_jsonable(value: Any) -> Any:
    """Convert common numpy/dataclass values into stable JSON values."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return [to_jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def rounded_float(value: Any, digits: int = 4) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def state_summary_from_state(state: Any) -> Dict[str, float]:
    """Small state fingerprint used for cache metadata and benchmark audits."""
    names = (
        "timestep",
        "day_of_year",
        "hour_of_day",
        "temp_air",
        "rh_air",
        "co2_air",
        "glob_rad",
        "temp_out",
        "rh_out",
        "wind_speed",
        "dew_margin_air",
        "canopy_dew_margin",
        "forecast_humidity_risk",
        "forecast_rad_mean_1h",
        "forecast_temp_out_delta_1h",
        "forecast_rh_out_mean_1h",
    )
    return {name: rounded_float(getattr(state, name, 0.0)) for name in names}


def config_fingerprint(config: Any) -> Dict[str, Any]:
    """Only include public planning knobs; never store API keys."""
    return {
        "model_name": str(getattr(config, "model_name", "")),
        "base_url": str(getattr(config, "base_url", "")),
        "temperature": rounded_float(getattr(config, "temperature", 0.0), digits=6),
        "max_tokens": int(getattr(config, "max_tokens", 0) or 0),
        "max_iterations": int(getattr(config, "max_iterations", 0) or 0),
        "control_interval": int(getattr(config, "control_interval", 0) or 0),
        "fallback_strategy": str(getattr(config, "fallback_strategy", "")),
        "plan_cache_key_policy": str(getattr(config, "plan_cache_key_policy", "prompt")),
    }


class PlanCache:
    """JSON-backed plan cache with record/replay/refresh semantics."""

    def __init__(self, path: str | Path, mode: str = "off", strict: bool = False):
        mode = str(mode or "off").lower()
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported plan cache mode: {mode}")
        self.path = Path(path)
        self.mode = mode
        self.strict = bool(strict)
        self.data: Dict[str, Any] = {"schema_version": SCHEMA_VERSION, "entries": {}}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                self.data = loaded
            self.data.setdefault("schema_version", SCHEMA_VERSION)
            self.data.setdefault("entries", {})

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def make_key(
        self,
        *,
        env_id: str,
        state_summary: Dict[str, Any],
        reason: str,
        planning_horizon: int,
        config_hash: str,
        prompt_hash: str,
        attempt: int,
    ) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "env_id": str(env_id),
            "state_summary": state_summary,
            "reason": str(reason),
            "planning_horizon": int(planning_horizon),
            "config_hash": str(config_hash),
            "prompt_hash": str(prompt_hash),
            "attempt": int(attempt),
        }
        return stable_hash(payload)[:24]

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self.data.get("entries", {}).get(str(key))
        return entry if isinstance(entry, dict) else None

    def put(self, key: str, entry: Dict[str, Any]) -> None:
        entries = self.data.setdefault("entries", {})
        entries[str(key)] = to_jsonable(entry)
        self.save()

    def update(self, key: str, fields: Dict[str, Any]) -> None:
        entries = self.data.setdefault("entries", {})
        entry = entries.setdefault(str(key), {})
        if isinstance(entry, dict):
            entry.update(to_jsonable(fields))
        else:
            entries[str(key)] = to_jsonable(fields)
        self.save()

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(self.data), f, ensure_ascii=False, indent=2, sort_keys=True)
        tmp_path.replace(self.path)
