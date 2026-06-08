"""Configuration loading for C-STCC v80."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "cstcc_v80.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if payload.get("version") != "v80":
        raise ValueError("C-STCC config must declare version: v80")
    if payload.get("online_llm_enabled", False):
        raise ValueError("v80 config must not enable online LLM")
    if payload.get("controller_changed", False) or payload.get("final_action_changed", False):
        raise ValueError("v80 config must not change controller or final action")
    return payload

