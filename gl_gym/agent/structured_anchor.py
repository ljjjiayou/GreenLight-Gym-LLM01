"""Structured planning-anchor parser for the greenhouse LLM planner.

The parser validates high-level profile/intent/target/risk anchors. It rejects
low-level actuator controls so the LLM cannot directly command final actions
through this migration path.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

STRUCTURED_ANCHOR_FIELDS = (
    "profile_intent",
    "target_temp",
    "target_co2",
    "target_rh",
    "risk_flags",
    "forbidden_intents",
    "planning_horizon_steps",
    "confidence",
)

FINAL_CONTROL_FIELDS = (
    "u_boil",
    "u_co2",
    "u_th_scr",
    "u_vent",
    "u_lamp",
    "u_bl_scr",
    "heating",
    "co2",
    "screen",
    "ventilation",
    "lighting",
    "shading",
    "blindscreen",
    "final_control",
)

FINAL_CONTROL_FIELD_SET = set(FINAL_CONTROL_FIELDS)
REQUIRED_FIELD_SET = set(STRUCTURED_ANCHOR_FIELDS)
ALLOWED_FIELD_SET = set(STRUCTURED_ANCHOR_FIELDS)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        out.append(item.strip())
    return out


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped


def _recursive_final_control_fields(value: Any, *, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            here = f"{path}.{key_text}" if path else key_text
            if key_text in FINAL_CONTROL_FIELD_SET:
                hits.append(here)
            hits.extend(_recursive_final_control_fields(item, path=here))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(_recursive_final_control_fields(item, path=f"{path}[{index}]"))
    return hits


def _normalize_anchor_payload(payload: Any) -> tuple[dict[str, Any] | None, list[str], bool]:
    errors: list[str] = []
    wrapper_used = False
    if payload is None:
        return None, ["empty_anchor"], wrapper_used
    if isinstance(payload, str):
        if not payload.strip():
            return None, ["empty_anchor"], wrapper_used
        try:
            payload = json.loads(_extract_json_object(payload))
        except json.JSONDecodeError:
            return None, ["invalid_json_anchor"], wrapper_used
    if not isinstance(payload, Mapping):
        return None, ["anchor_not_object"], wrapper_used
    if "structured_planning_anchor" in payload and isinstance(payload.get("structured_planning_anchor"), Mapping):
        payload = payload["structured_planning_anchor"]
        wrapper_used = True
    anchor = dict(payload)
    if not anchor:
        errors.append("empty_anchor")
    return anchor, errors, wrapper_used


def parse_structured_anchor(payload: Any) -> dict[str, Any]:
    anchor, errors, wrapper_used = _normalize_anchor_payload(payload)
    if anchor is None:
        return {
            "valid": False,
            "clean_planning_evidence": False,
            "errors": errors,
            "shadow_plan": {},
            "wrapper_used": wrapper_used,
        }

    final_hits = _recursive_final_control_fields(anchor)
    if final_hits:
        errors.append("final_control_fields_present")

    fields = set(anchor)
    missing = sorted(REQUIRED_FIELD_SET - fields)
    unexpected = sorted(fields - ALLOWED_FIELD_SET)
    if missing:
        errors.append("missing_required_fields")
    if unexpected:
        errors.append("unexpected_fields")

    profile_intent = anchor.get("profile_intent")
    if not isinstance(profile_intent, str) or not profile_intent.strip():
        errors.append("profile_intent_invalid")

    target_temp = _numeric(anchor.get("target_temp"))
    target_co2 = _numeric(anchor.get("target_co2"))
    target_rh = _numeric(anchor.get("target_rh"))
    if target_temp is None:
        errors.append("target_temp_invalid")
    if target_co2 is None:
        errors.append("target_co2_invalid")
    if target_rh is None:
        errors.append("target_rh_invalid")

    risk_flags = _string_list(anchor.get("risk_flags"))
    forbidden_intents = _string_list(anchor.get("forbidden_intents"))
    if risk_flags is None:
        errors.append("risk_flags_invalid")
    if forbidden_intents is None:
        errors.append("forbidden_intents_invalid")

    horizon_raw = anchor.get("planning_horizon_steps")
    horizon = None
    if isinstance(horizon_raw, bool):
        horizon = None
    elif isinstance(horizon_raw, int):
        horizon = horizon_raw
    elif isinstance(horizon_raw, float) and horizon_raw.is_integer():
        horizon = int(horizon_raw)
    elif isinstance(horizon_raw, str) and horizon_raw.strip().isdigit():
        horizon = int(horizon_raw.strip())
    if horizon is None or horizon < 1:
        errors.append("planning_horizon_steps_invalid")

    confidence = _numeric(anchor.get("confidence"))
    if confidence is None or confidence < 0.0 or confidence > 1.0:
        errors.append("confidence_invalid")

    unique_errors = sorted(set(errors))
    valid = not unique_errors
    shadow_plan = {}
    if valid:
        shadow_plan = {
            "source": "structured_anchor_shadow_parser",
            "profile_intent": str(profile_intent).strip(),
            "target_temp": target_temp,
            "target_co2": target_co2,
            "target_rh": target_rh,
            "risk_flags": risk_flags,
            "forbidden_intents": forbidden_intents,
            "planning_horizon_steps": horizon,
            "confidence": confidence,
            "final_control_generation_allowed": False,
        }

    return {
        "valid": bool(valid),
        "clean_planning_evidence": bool(valid),
        "errors": unique_errors,
        "missing_fields": missing,
        "unexpected_fields": unexpected,
        "final_control_field_hits": final_hits,
        "wrapper_used": wrapper_used,
        "shadow_plan": shadow_plan,
        "raw_anchor_keys": sorted(str(key) for key in anchor),
    }
