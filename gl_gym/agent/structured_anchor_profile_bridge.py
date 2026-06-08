"""Runtime shadow bridge from structured anchors to profile candidates.

The bridge is intentionally metadata-only: it converts a valid structured
planning anchor into an IntentContract and profile-generator candidates, but it
never creates low-level controls or mutates the active plan.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from gl_gym.agent.intent_contract import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_PROFILE_SHAPE,
    IntentContract,
    TARGET_LIMITS,
    TARGET_WIDTHS,
)
from gl_gym.agent.profile_generator import build_profile_generator_shadow_payload

HORIZON_CAP_STEPS = 24


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(number) if math.isfinite(number) else float(default)


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _tokens(*values: Any) -> set[str]:
    out: set[str] = set()
    for value in values:
        if isinstance(value, str):
            text = value
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            text = " ".join(str(item) for item in value)
        else:
            text = str(value or "")
        normalized = (
            text.lower()
            .replace("-", "_")
            .replace("/", "_")
            .replace(",", " ")
            .replace(";", " ")
            .replace(":", " ")
        )
        for piece in normalized.split():
            if piece:
                out.add(piece)
            for subpiece in piece.split("_"):
                if subpiece:
                    out.add(subpiece)
    return out


def _range_from_target(value: Any, key: str) -> tuple[tuple[float, float], dict[str, Any]]:
    low, high = TARGET_LIMITS[key]
    width = TARGET_WIDTHS[key]
    raw = _as_float(value, (low + high) / 2.0)
    center = float(min(max(raw, low), high))
    return (
        (
            float(min(max(center - width, low), high)),
            float(min(max(center + width, low), high)),
        ),
        {
            "key": key,
            "raw_center": float(raw),
            "clipped_center": float(center),
            "was_clipped": bool(raw != center),
        },
    )


def _cap_horizon(value: Any) -> tuple[int, dict[str, Any]]:
    raw = _as_int(value, 12)
    clipped = int(min(max(raw, 1), HORIZON_CAP_STEPS))
    return clipped, {
        "raw_planning_horizon_steps": raw,
        "planning_horizon_steps": clipped,
        "horizon_cap_steps": HORIZON_CAP_STEPS,
        "was_clipped": bool(raw != clipped),
    }


def _regime_from_anchor(anchor: Mapping[str, Any], state: Any) -> tuple[str, tuple[str, ...], dict[str, str], dict[str, float], list[str]]:
    intent = str(anchor.get("profile_intent") or "")
    risk_flags = anchor.get("risk_flags", [])
    forbidden = anchor.get("forbidden_intents", [])
    words = _tokens(intent, risk_flags, forbidden)
    diagnostics: list[str] = []
    constraints = dict(DEFAULT_CONSTRAINTS)
    profile_shape = dict(DEFAULT_PROFILE_SHAPE)
    hour = _as_float(getattr(state, "hour_of_day", 12.0), 12.0)

    dew_words = {"dew", "canopy", "humidity", "humid", "rh", "high_rh", "low_vpd", "condensation"}
    dry_words = {"heat", "hot", "vpd", "dry", "radiation", "cooling", "thermal", "surge", "spike"}
    co2_words = {"co2", "carbon", "enrichment"}
    night_words = {"night", "predawn", "pre_dawn", "dawn", "stabilization", "stabilise", "stabilize"}

    has_dew = bool(words & dew_words) or "high_humidity" in intent.lower()
    has_dry = bool(words & dry_words) or "hot_dry" in intent.lower()
    has_radiation = "radiation" in words or "spike" in words
    has_co2 = bool(words & co2_words)
    co2_vent_conflict = "co2_vent_conflict" in intent.lower() or "co2_vent_conflict" in "_".join(words)
    co2_forbidden = any("co2" in str(item).lower() and "vent" in str(item).lower() for item in forbidden or [])
    is_nightish = hour < 6.0 or hour > 18.0 or bool(words & night_words)

    if has_dew:
        if 4.0 <= hour <= 8.0 or "dawn" in words or "predawn" in words or "pre_dawn" in intent.lower():
            regime = "dawn_predehumidify"
            profile_shape = {"temp": "dawn_predehumidify", "co2": "constant_hold", "rh": "strict_dehumidify_then_relax"}
        else:
            regime = "high_humidity_recovery"
            profile_shape = {"temp": "constant_hold", "co2": "constant_hold", "rh": "strict_dehumidify_then_relax"}
        priority = ("safety", "humidity", "energy", "growth")
        constraints.update({"rh_hard_max": 88.0, "dew_margin_min": 1.2, "canopy_dew_margin_min": 1.2, "forbid_co2_when_vent_gt": 0.18})
    elif has_dry:
        regime = "radiation_spike_relief" if has_radiation else "hot_dry_relief"
        profile_shape = {"temp": "shade_cooling", "co2": "constant_hold", "rh": "hot_dry_protect"}
        priority = ("safety", "humidity", "temperature", "energy", "growth")
        constraints.update({"vpd_max": 1.55, "temp_max": 30.8, "rh_hard_max": 88.0})
    elif has_co2 and not co2_vent_conflict and not co2_forbidden:
        regime = "co2_day_boost"
        profile_shape = {"temp": "constant_hold", "co2": "co2_day_boost", "rh": "constant_hold"}
        priority = ("safety", "growth", "energy")
        constraints.update({"forbid_co2_when_vent_gt": 0.18})
    elif is_nightish:
        regime = "night_heat_hold"
        profile_shape = {"temp": "night_heat_hold", "co2": "constant_hold", "rh": "constant_hold"}
        priority = ("safety", "temperature", "energy", "growth")
    else:
        regime = "economy_hold"
        priority = ("safety", "energy", "growth")
        diagnostics.append("unknown_profile_intent_mapped")

    if has_co2 and (co2_vent_conflict or co2_forbidden):
        diagnostics.append("co2_day_boost_blocked_by_co2_vent_conflict")
    return regime, priority, profile_shape, constraints, diagnostics


def structured_anchor_to_intent_contract(
    anchor: Mapping[str, Any],
    state: Any,
    *,
    model_name: str = "qwen3.7-plus",
) -> tuple[IntentContract, dict[str, Any]]:
    target_range: dict[str, tuple[float, float]] = {}
    target_diagnostics = []
    for anchor_key, short_key in (("target_temp", "temp"), ("target_co2", "co2"), ("target_rh", "rh")):
        target_range[short_key], item = _range_from_target(anchor.get(anchor_key), short_key)
        target_diagnostics.append(item)

    horizon, horizon_diagnostics = _cap_horizon(anchor.get("planning_horizon_steps"))
    regime, priority, profile_shape, constraints, regime_diagnostics = _regime_from_anchor(anchor, state)
    confidence_raw = _as_float(anchor.get("confidence"), 0.5)
    confidence = float(min(max(confidence_raw, 0.0), 1.0))
    contract = IntentContract(
        regime=regime,
        target_range=target_range,
        priority=priority,
        constraints=constraints,
        profile_shape=profile_shape,
        confidence=confidence,
    )
    diagnostics = {
        "source": "structured_anchor_profile_bridge_v64",
        "model_name": str(model_name),
        "horizon": horizon_diagnostics,
        "planning_horizon_steps": int(horizon),
        "target_range_diagnostics": target_diagnostics,
        "regime_mapping_diagnostics": regime_diagnostics,
        "unknown_profile_intent_mapped": "unknown_profile_intent_mapped" in regime_diagnostics,
        "confidence_raw": float(confidence_raw),
        "confidence": float(confidence),
        "confidence_was_clipped": bool(confidence != confidence_raw),
        "final_control_generation_allowed": False,
    }
    return contract, diagnostics


def build_structured_anchor_profile_bridge_shadow_payload(
    structured_anchor: Mapping[str, Any],
    state: Any,
    *,
    current_plan: Mapping[str, Any] | None = None,
    model_name: str = "qwen3.7-plus",
    enabled: bool = True,
    shadow_only: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    anchor_block = structured_anchor if isinstance(structured_anchor, Mapping) else {}
    shadow_plan = anchor_block.get("shadow_plan", {}) if isinstance(anchor_block.get("shadow_plan", {}), Mapping) else {}
    base = {
        "enabled": bool(enabled),
        "shadow_only": bool(shadow_only),
        "source": "structured_anchor_profile_bridge_v64",
        "attempted": bool(anchor_block.get("attempted", False)),
        "valid_structured_anchor": bool(anchor_block.get("valid", False)),
        "bridgeable": False,
        "applied": False,
        "final_control_change": False,
        "current_plan_modified": False,
        "low_level_action_generated": False,
    }
    if not bool(enabled):
        return base, []
    if not bool(anchor_block.get("valid", False)) or not shadow_plan:
        base["failure_reason"] = "valid_structured_anchor_missing"
        return base, []

    try:
        contract, bridge_diagnostics = structured_anchor_to_intent_contract(shadow_plan, state, model_name=model_name)
        horizon = int(bridge_diagnostics["planning_horizon_steps"])
        candidates, profile_diagnostics = build_profile_generator_shadow_payload(
            contract,
            state,
            current_plan=current_plan,
            horizon_steps=horizon,
        )
    except Exception as exc:
        base.update(
            {
                "failure_reason": "profile_candidate_generation_error",
                "error": str(exc),
            }
        )
        return base, []

    selected_name = str(
        profile_diagnostics.get("score_selected_shadow_profile_name")
        or profile_diagnostics.get("selected_shadow_profile_name")
        or ""
    )
    safety_gate_reason = str(profile_diagnostics.get("score_safety_gate_reason") or "none")
    hard_safety_profile_violation = bool(
        safety_gate_reason in {"dew_gate", "canopy_gate", "rh_high_gate", "temp_high_gate"}
        and selected_name in {"hot_dry_protect", "co2_day_boost", "lighting_assist"}
    )
    base.update(
        {
            "bridgeable": True,
            "applied": True,
            "failure_reason": "",
            "model_name": str(model_name),
            "target_contract": contract.to_dict(),
            "bridge_diagnostics": bridge_diagnostics,
            "profile_generator_diagnostics": profile_diagnostics,
            "profile_candidate_count": int(len(candidates)),
            "selected_shadow_profile_name": selected_name,
            "score_safety_gate_reason": safety_gate_reason,
            "hard_safety_profile_violation": bool(hard_safety_profile_violation),
        }
    )
    return base, candidates
