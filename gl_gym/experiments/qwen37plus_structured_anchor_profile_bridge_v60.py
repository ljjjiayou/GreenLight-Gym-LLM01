"""v60 qwen3.7-plus model switch and structured-anchor profile bridge.

This stage does not run rollout and does not change the default controller. It
checks qwen3.7-plus accessibility and converts existing v59 structured anchors
into shadow IntentContract/profile-generator candidates for bridge validation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.agent.intent_contract import (  # noqa: E402
    DEFAULT_CONSTRAINTS,
    DEFAULT_PROFILE_SHAPE,
    IntentContract,
    TARGET_LIMITS,
    TARGET_WIDTHS,
)
from gl_gym.agent.profile_generator import build_profile_generator_shadow_payload  # noqa: E402
import gl_gym.experiments.profile_template_opt_in_shadow_rollout_v51 as v51  # noqa: E402
import gl_gym.experiments.profile_template_opt_in_shadow_rollout_v52 as v52  # noqa: E402
import gl_gym.experiments.qwen37_structured_anchor_opt_in_shadow_rollout_v59 as v59  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (  # noqa: E402
    AUDIT_DIR,
    FAILURE_SCENARIOS,
)


MODEL_NAME = "qwen3.7-plus"
SOURCE_MODEL_NAME = "qwen3.7-max"
ARTIFACT_DATE = "20260602"

V59_CACHE_PATH = v59.V59_CACHE_PATH
V59_TRACE_DIR = v59.V59_TRACE_DIR
V59_READINESS_JSON = v59.READINESS_JSON

ONLINE_PRECHECK_JSON = AUDIT_DIR / "online_llm_accessibility_precheck_result_v60_qwen37plus_20260602.json"
ONLINE_PRECHECK_MD = AUDIT_DIR / "online_llm_accessibility_precheck_result_v60_qwen37plus_20260602.md"
MODEL_SWITCH_READINESS_JSON = AUDIT_DIR / "qwen37plus_model_switch_readiness_20260602_v60.json"
MODEL_SWITCH_READINESS_MD = AUDIT_DIR / "qwen37plus_model_switch_readiness_20260602_v60.md"
BRIDGE_AUDIT_JSON = AUDIT_DIR / "qwen37plus_structured_anchor_profile_bridge_audit_20260602_v60.json"
BRIDGE_AUDIT_MD = AUDIT_DIR / "qwen37plus_structured_anchor_profile_bridge_audit_20260602_v60.md"
BRIDGE_COMPARISON_JSON = AUDIT_DIR / "qwen37plus_structured_anchor_profile_bridge_candidate_comparison_20260602_v60.json"
BRIDGE_COMPARISON_MD = AUDIT_DIR / "qwen37plus_structured_anchor_profile_bridge_candidate_comparison_20260602_v60.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260602_v60.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260602_v60.md"

FUTURE_QWEN37PLUS_CACHE_PATH = (
    PROJECT_ROOT / "gl_gym" / "result" / "plan_cache" / "qwen37plus_structured_anchor_shadow_v61_20260602.json"
)
FUTURE_QWEN37PLUS_OUTPUT_DIR = (
    PROJECT_ROOT / "gl_gym" / "result" / "benchmarks" / "qwen37plus_structured_anchor_shadow_rollout_v61_20260602"
)

HORIZON_CAP_STEPS = 24


def _rel(path: str | Path) -> str:
    return v51._rel(path)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json_md(path_json: Path, path_md: Path, data: Mapping[str, Any], title: str) -> None:
    v51._write_json_md(path_json, path_md, data, title)


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


def _env_id_to_scenario(env_id: str, *, max_steps: int = 720) -> str:
    text = str(env_id or "")
    if text.startswith("TomatoEnv_"):
        text = text[len("TomatoEnv_") :]
    return f"{text}_n{max_steps}" if text and "_n" not in text else text


def _cache_entries(cache_path: str | Path = V59_CACHE_PATH) -> list[tuple[str, dict[str, Any]]]:
    data = _load_json(cache_path)
    entries = data.get("entries", data if isinstance(data, dict) else {})
    if not isinstance(entries, Mapping):
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for key, value in entries.items():
        if isinstance(value, Mapping):
            out.append((str(key), dict(value)))
    return out


def _trace_rows_by_scenario(trace_dir: str | Path = V59_TRACE_DIR, scenarios: Sequence[str] = FAILURE_SCENARIOS) -> dict[str, dict[int, dict[str, Any]]]:
    rows_by_scenario: dict[str, dict[int, dict[str, Any]]] = {}
    for scenario in scenarios:
        rows = v51._read_trace_rows(trace_dir, scenario)
        rows_by_scenario[scenario] = {_as_int(row.get("timestep", row.get("step", 0)), 0): dict(row) for row in rows}
    return rows_by_scenario


def _range_from_target(value: Any, key: str) -> tuple[tuple[float, float], dict[str, Any]]:
    low, high = TARGET_LIMITS[key]
    width = TARGET_WIDTHS[key]
    raw = _as_float(value, (low + high) / 2.0)
    center = float(min(max(raw, low), high))
    clipped = raw != center
    return (
        (
            float(min(max(center - width, low), high)),
            float(min(max(center + width, low), high)),
        ),
        {
            "key": key,
            "raw_center": float(raw),
            "clipped_center": float(center),
            "was_clipped": bool(clipped),
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
        profile_shape = dict(DEFAULT_PROFILE_SHAPE)
        priority = ("safety", "energy", "growth")
        diagnostics.append("unknown_profile_intent_mapped")

    if has_co2 and (co2_vent_conflict or co2_forbidden):
        diagnostics.append("co2_day_boost_blocked_by_co2_vent_conflict")
    return regime, priority, profile_shape, constraints, diagnostics


def structured_anchor_to_intent_contract(anchor: Mapping[str, Any], state: Any) -> tuple[IntentContract, dict[str, Any]]:
    target_range: dict[str, tuple[float, float]] = {}
    target_diagnostics = []
    for anchor_key, short_key in (("target_temp", "temp"), ("target_co2", "co2"), ("target_rh", "rh")):
        target_range[short_key], item = _range_from_target(anchor.get(anchor_key), short_key)
        target_diagnostics.append(item)

    horizon, horizon_diagnostics = _cap_horizon(anchor.get("planning_horizon_steps"))
    regime, priority, profile_shape, constraints, regime_diagnostics = _regime_from_anchor(anchor, state)
    confidence_raw = _as_float(anchor.get("confidence"), 0.5)
    confidence = float(min(max(confidence_raw, 0.0), 1.0))
    confidence_clipped = confidence != confidence_raw
    contract = IntentContract(
        regime=regime,
        target_range=target_range,
        priority=priority,
        constraints=constraints,
        profile_shape=profile_shape,
        confidence=confidence,
    )
    diagnostics = {
        "source": "structured_anchor_profile_bridge_v60",
        "model_name_for_next_online_stage": MODEL_NAME,
        "historical_anchor_source_model_name": SOURCE_MODEL_NAME,
        "horizon": horizon_diagnostics,
        "planning_horizon_steps": horizon,
        "target_range_diagnostics": target_diagnostics,
        "regime_mapping_diagnostics": regime_diagnostics,
        "unknown_profile_intent_mapped": "unknown_profile_intent_mapped" in regime_diagnostics,
        "confidence_raw": float(confidence_raw),
        "confidence": float(confidence),
        "confidence_was_clipped": bool(confidence_clipped),
        "final_control_generation_allowed": False,
    }
    return contract, diagnostics


def _current_plan_from_entry(entry: Mapping[str, Any]) -> Mapping[str, Any] | None:
    plan = entry.get("parsed_plan")
    if isinstance(plan, Mapping):
        return plan
    plan = entry.get("setpoint_contract_plan")
    return plan if isinstance(plan, Mapping) else None


def _target_midpoints(contract: IntentContract) -> dict[str, float]:
    return {key: float((bounds[0] + bounds[1]) / 2.0) for key, bounds in contract.target_range.items()}


def _plan_targets(plan: Mapping[str, Any] | None) -> dict[str, float | None]:
    if not isinstance(plan, Mapping):
        return {"temp": None, "co2": None, "rh": None}
    return {
        "temp": _as_float(plan.get("target_temp"), math.nan),
        "co2": _as_float(plan.get("target_co2"), math.nan),
        "rh": _as_float(plan.get("target_rh"), math.nan),
    }


def _target_delta_abs(contract: IntentContract, plan: Mapping[str, Any] | None) -> dict[str, float | None]:
    mid = _target_midpoints(contract)
    old = _plan_targets(plan)
    out: dict[str, float | None] = {}
    for key in ("temp", "co2", "rh"):
        value = old.get(key)
        out[key] = None if value is None or math.isnan(float(value)) else float(abs(mid[key] - float(value)))
    return out


def _bridge_one_entry(
    *,
    cache_key: str,
    entry: Mapping[str, Any],
    row: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    env_id = str(entry.get("env_id") or "")
    scenario = _env_id_to_scenario(env_id)
    timestep = _as_int(entry.get("timestep"), 0)
    anchor_block = entry.get("structured_anchor", {}) if isinstance(entry.get("structured_anchor"), Mapping) else {}
    shadow_plan = anchor_block.get("shadow_plan", {}) if isinstance(anchor_block, Mapping) else {}
    plan = _current_plan_from_entry(entry)
    if not row:
        return (
            {
                "cache_key": cache_key,
                "scenario_id": scenario,
                "timestep": timestep,
                "bridgeable": False,
                "failure_reason": "metadata_alignment_missing",
                "valid_structured_anchor": bool(anchor_block.get("valid", False)),
            },
            {},
        )
    if not bool(anchor_block.get("valid", False)) or not isinstance(shadow_plan, Mapping) or not shadow_plan:
        return (
            {
                "cache_key": cache_key,
                "scenario_id": scenario,
                "timestep": timestep,
                "bridgeable": False,
                "failure_reason": "valid_structured_anchor_missing",
                "valid_structured_anchor": bool(anchor_block.get("valid", False)),
            },
            {},
        )

    state = SimpleNamespace(**dict(row))
    contract, contract_diagnostics = structured_anchor_to_intent_contract(shadow_plan, state)
    horizon = int(contract_diagnostics["planning_horizon_steps"])
    profile_candidates, profile_diagnostics = build_profile_generator_shadow_payload(
        contract,
        state,
        current_plan=plan,
        horizon_steps=horizon,
    )
    selected_name = str(profile_diagnostics.get("score_selected_shadow_profile_name") or profile_diagnostics.get("selected_shadow_profile_name") or "")
    safety_gate_reason = str(profile_diagnostics.get("score_safety_gate_reason") or "none")
    hard_safety_profile_violation = bool(
        safety_gate_reason in {"dew_gate", "canopy_gate", "rh_high_gate", "temp_high_gate"}
        and selected_name in {"hot_dry_protect", "co2_day_boost", "lighting_assist"}
    )
    comparison = {
        "cache_key": cache_key,
        "scenario_id": scenario,
        "timestep": timestep,
        "profile_intent": str(shadow_plan.get("profile_intent") or ""),
        "mapped_regime": contract.regime,
        "selected_profile_name": selected_name,
        "profile_candidate_count": int(len(profile_candidates)),
        "score_safety_gate_reason": safety_gate_reason,
        "target_delta_abs_vs_v59_plan": _target_delta_abs(contract, plan),
        "unknown_profile_intent_mapped": bool(contract_diagnostics.get("unknown_profile_intent_mapped", False)),
        "horizon_was_clipped": bool(contract_diagnostics["horizon"]["was_clipped"]),
    }
    row_result = {
        "cache_key": cache_key,
        "scenario_id": scenario,
        "env_id": env_id,
        "timestep": timestep,
        "bridgeable": True,
        "valid_structured_anchor": True,
        "source_model_name": str(entry.get("model_name") or SOURCE_MODEL_NAME),
        "target_contract": contract.to_dict(),
        "bridge_diagnostics": contract_diagnostics,
        "profile_generator_diagnostics": profile_diagnostics,
        "profile_candidate_count": int(len(profile_candidates)),
        "selected_shadow_profile_name": selected_name,
        "score_safety_gate_reason": safety_gate_reason,
        "hard_safety_profile_violation": hard_safety_profile_violation,
        "final_control_change": False,
        "current_plan_modified": False,
        "low_level_action_generated": False,
    }
    return row_result, comparison


def build_online_precheck(*, probe_online: bool = True, model_name: str = MODEL_NAME) -> dict[str, Any]:
    result = v52.build_online_precheck(probe_online=probe_online, model_name=model_name)
    result["artifact"] = "online_llm_accessibility_precheck_result_v60_qwen37plus_20260602"
    result["model_name"] = model_name
    result["model_switch_target"] = MODEL_NAME
    result["previous_model_name"] = SOURCE_MODEL_NAME
    return result


def build_bridge_audit(
    *,
    cache_path: str | Path = V59_CACHE_PATH,
    trace_dir: str | Path = V59_TRACE_DIR,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace_rows = _trace_rows_by_scenario(trace_dir, scenarios)
    entries = _cache_entries(cache_path)
    row_results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for cache_key, entry in entries:
        scenario = _env_id_to_scenario(str(entry.get("env_id") or ""))
        if scenario not in trace_rows:
            continue
        timestep = _as_int(entry.get("timestep"), 0)
        row = trace_rows.get(scenario, {}).get(timestep)
        result, comparison = _bridge_one_entry(cache_key=cache_key, entry=entry, row=row)
        row_results.append(result)
        if comparison:
            comparisons.append(comparison)

    source_attempt_count = len(row_results)
    bridgeable = [row for row in row_results if bool(row.get("bridgeable", False))]
    bridgeable_count = len(bridgeable)
    bridge_input_coverage_rate = float(bridgeable_count / source_attempt_count) if source_attempt_count else 0.0
    candidate_counts = [int(row.get("profile_candidate_count", 0) or 0) for row in bridgeable]
    unknown_count = sum(bool(row.get("bridge_diagnostics", {}).get("unknown_profile_intent_mapped", False)) for row in bridgeable)
    horizon_clip_count = sum(bool(row.get("bridge_diagnostics", {}).get("horizon", {}).get("was_clipped", False)) for row in bridgeable)
    target_clip_count = sum(
        any(bool(item.get("was_clipped", False)) for item in row.get("bridge_diagnostics", {}).get("target_range_diagnostics", []) or [])
        for row in bridgeable
    )
    hard_safety_violations = sum(bool(row.get("hard_safety_profile_violation", False)) for row in bridgeable)
    error_count = sum(not bool(row.get("bridgeable", False)) for row in row_results)
    regime_counts = Counter(str(row.get("target_contract", {}).get("regime") or "") for row in bridgeable)
    selected_profile_counts = Counter(str(row.get("selected_shadow_profile_name") or "") for row in bridgeable)
    failure_reasons = Counter(str(row.get("failure_reason") or "") for row in row_results if not bool(row.get("bridgeable", False)))
    unknown_rate = float(unknown_count / bridgeable_count) if bridgeable_count else 0.0
    acceptance = {
        "source_v59_structured_anchor_available": bool(source_attempt_count > 0),
        "bridge_input_coverage_ge_0_90": bool(bridge_input_coverage_rate >= 0.90),
        "profile_candidate_generation_error_count_zero": bool(error_count == 0),
        "profile_candidate_count_positive": bool(candidate_counts and min(candidate_counts) > 0),
        "hard_safety_profile_violation_count_zero": bool(hard_safety_violations == 0),
        "final_control_change_count_zero": True,
        "unknown_profile_intent_mapped_rate_le_0_40": bool(unknown_rate <= 0.40),
    }
    acceptance["v60_bridge_pass"] = bool(all(acceptance.values()))
    audit = {
        "artifact": "qwen37plus_structured_anchor_profile_bridge_audit_20260602_v60",
        "current_stage": "v60_qwen37plus_structured_anchor_profile_bridge",
        "scope": "offline_structured_anchor_to_profile_generator_bridge_only",
        "model_name_for_next_online_stage": MODEL_NAME,
        "historical_source_model_name": SOURCE_MODEL_NAME,
        "source_cache_path": _rel(cache_path),
        "source_trace_dir": _rel(trace_dir),
        "scenarios": list(scenarios),
        "source_attempt_count": int(source_attempt_count),
        "bridgeable_count": int(bridgeable_count),
        "bridge_input_coverage_rate": float(bridge_input_coverage_rate),
        "metadata_alignment_missing_count": int(failure_reasons.get("metadata_alignment_missing", 0)),
        "valid_structured_anchor_missing_count": int(failure_reasons.get("valid_structured_anchor_missing", 0)),
        "profile_candidate_generation_error_count": int(error_count),
        "profile_candidate_count_min": int(min(candidate_counts) if candidate_counts else 0),
        "profile_candidate_count_max": int(max(candidate_counts) if candidate_counts else 0),
        "unknown_profile_intent_mapped_count": int(unknown_count),
        "unknown_profile_intent_mapped_rate": float(unknown_rate),
        "horizon_clipped_count": int(horizon_clip_count),
        "target_clipped_count": int(target_clip_count),
        "hard_safety_profile_violation_count": int(hard_safety_violations),
        "final_control_change_count": 0,
        "current_plan_modified_count": 0,
        "low_level_action_generated_count": 0,
        "mapped_regime_counts": dict(sorted(regime_counts.items())),
        "selected_profile_counts": dict(sorted(selected_profile_counts.items())),
        "failure_reason_counts": dict(sorted((key, value) for key, value in failure_reasons.items() if key)),
        "sample_rows": row_results[:12],
        "acceptance": acceptance,
        "bridge_pass": bool(acceptance["v60_bridge_pass"]),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in comparisons:
        by_scenario[str(item.get("scenario_id") or "")].append(item)
    comparison = {
        "artifact": "qwen37plus_structured_anchor_profile_bridge_candidate_comparison_20260602_v60",
        "current_stage": "v60_qwen37plus_structured_anchor_profile_bridge",
        "scope": "offline_profile_candidate_comparison_only",
        "model_name_for_next_online_stage": MODEL_NAME,
        "historical_source_model_name": SOURCE_MODEL_NAME,
        "comparison_count": int(len(comparisons)),
        "by_scenario_counts": {key: len(value) for key, value in sorted(by_scenario.items())},
        "selected_profile_counts": dict(sorted(selected_profile_counts.items())),
        "mapped_regime_counts": dict(sorted(regime_counts.items())),
        "target_delta_abs_summary": _target_delta_summary(comparisons),
        "sample_comparisons": comparisons[:20],
        "final_control_change_count": 0,
        "current_plan_modified_count": 0,
        "low_level_action_generated_count": 0,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }
    return audit, comparison


def _target_delta_summary(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for key in ("temp", "co2", "rh"):
        values: list[float] = []
        for row in comparisons:
            raw = row.get("target_delta_abs_vs_v59_plan", {})
            if not isinstance(raw, Mapping):
                continue
            value = raw.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                values.append(number)
        out[key] = {
            "count": int(len(values)),
            "mean": float(sum(values) / len(values)) if values else 0.0,
            "max": float(max(values)) if values else 0.0,
        }
    return out


def build_model_switch_readiness(precheck: Mapping[str, Any], bridge_audit: Mapping[str, Any]) -> dict[str, Any]:
    precheck_pass = bool(precheck.get("online_llm_credentials_present", False) and precheck.get("online_llm_accessible", False))
    bridge_pass = bool(bridge_audit.get("bridge_pass", False))
    return {
        "artifact": "qwen37plus_model_switch_readiness_20260602_v60",
        "current_stage": "v60_qwen37plus_model_switch_and_bridge",
        "model_name": MODEL_NAME,
        "previous_model_name": SOURCE_MODEL_NAME,
        "future_online_evidence_model_name": MODEL_NAME,
        "historical_v59_evidence_model_name": SOURCE_MODEL_NAME,
        "online_llm_credentials_present": bool(precheck.get("online_llm_credentials_present", False)),
        "online_llm_accessible": bool(precheck.get("online_llm_accessible", False)),
        "provider_error_detected": bool(precheck.get("provider_error_detected", False)),
        "qwen37plus_precheck_pass": bool(precheck_pass),
        "bridge_offline_done": True,
        "bridge_pass": bool(bridge_pass),
        "v60_no_rollout": True,
        "planned_future_model_name": MODEL_NAME,
        "planned_future_cache_path": _rel(FUTURE_QWEN37PLUS_CACHE_PATH),
        "planned_future_output_dir": _rel(FUTURE_QWEN37PLUS_OUTPUT_DIR),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def build_readiness(precheck: Mapping[str, Any], bridge_audit: Mapping[str, Any]) -> dict[str, Any]:
    precheck_pass = bool(precheck.get("online_llm_credentials_present", False) and precheck.get("online_llm_accessible", False))
    bridge_pass = bool(bridge_audit.get("bridge_pass", False))
    unknown_rate = float(bridge_audit.get("unknown_profile_intent_mapped_rate", 0.0) or 0.0)
    if not precheck_pass:
        next_action = "online_llm_accessibility_blocked"
    elif bridge_pass:
        next_action = "qwen37plus_structured_anchor_opt_in_shadow_rollout_plan"
    elif unknown_rate > 0.40 or int(bridge_audit.get("hard_safety_profile_violation_count", 0) or 0) > 0:
        next_action = "structured_anchor_prompt_or_bridge_mapping_repair_plan"
    else:
        next_action = "structured_anchor_prompt_or_bridge_mapping_repair_plan"
    stop_taxonomy: list[str] = []
    if not precheck_pass:
        stop_taxonomy.append(str(precheck.get("blocked_reason") or "online_llm_accessibility_blocked"))
    if not bridge_pass:
        acceptance = bridge_audit.get("acceptance", {}) if isinstance(bridge_audit.get("acceptance"), Mapping) else {}
        if not acceptance.get("bridge_input_coverage_ge_0_90", False):
            stop_taxonomy.append("bridge_input_coverage_low")
        if not acceptance.get("profile_candidate_generation_error_count_zero", False):
            stop_taxonomy.append("profile_candidate_generation_error")
        if not acceptance.get("hard_safety_profile_violation_count_zero", False):
            stop_taxonomy.append("hard_safety_profile_violation")
        if not acceptance.get("unknown_profile_intent_mapped_rate_le_0_40", False):
            stop_taxonomy.append("unknown_profile_intent_mapping_high")
    return {
        "artifact": "metadata_replay_readiness_checklist_20260602_v60",
        "current_stage": "v60_qwen37plus_model_switch_and_structured_anchor_profile_bridge",
        "model_name": MODEL_NAME,
        "previous_model_name": SOURCE_MODEL_NAME,
        "qwen37plus_precheck_pass": bool(precheck_pass),
        "online_llm_credentials_present": bool(precheck.get("online_llm_credentials_present", False)),
        "online_llm_accessible": bool(precheck.get("online_llm_accessible", False)),
        "bridge_offline_done": True,
        "bridge_pass": bool(bridge_pass),
        "bridge_input_coverage_rate": float(bridge_audit.get("bridge_input_coverage_rate", 0.0) or 0.0),
        "unknown_profile_intent_mapped_rate": unknown_rate,
        "hard_safety_profile_violation_count": int(bridge_audit.get("hard_safety_profile_violation_count", 0) or 0),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": next_action,
        "stop_taxonomy": sorted(set(stop_taxonomy)),
    }


def write_all(*, probe_online: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    precheck = build_online_precheck(probe_online=probe_online)
    bridge_audit, comparison = build_bridge_audit()
    model_switch = build_model_switch_readiness(precheck, bridge_audit)
    readiness = build_readiness(precheck, bridge_audit)
    _write_json_md(ONLINE_PRECHECK_JSON, ONLINE_PRECHECK_MD, precheck, "v60 qwen3.7-plus Online LLM Accessibility Precheck")
    _write_json_md(MODEL_SWITCH_READINESS_JSON, MODEL_SWITCH_READINESS_MD, model_switch, "v60 qwen3.7-plus Model Switch Readiness")
    _write_json_md(BRIDGE_AUDIT_JSON, BRIDGE_AUDIT_MD, bridge_audit, "v60 qwen3.7-plus Structured Anchor Profile Bridge Audit")
    _write_json_md(BRIDGE_COMPARISON_JSON, BRIDGE_COMPARISON_MD, comparison, "v60 qwen3.7-plus Structured Anchor Profile Bridge Candidate Comparison")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v60 Metadata Replay Readiness")
    return precheck, model_switch, bridge_audit, comparison, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-all", action="store_true")
    parser.add_argument("--skip-online-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    precheck, model_switch, bridge_audit, _comparison, readiness = write_all(probe_online=not args.skip_online_probe)
    if not args.write_all:
        print(
            json.dumps(
                {
                    "model_name": MODEL_NAME,
                    "online_llm_accessible": precheck.get("online_llm_accessible"),
                    "qwen37plus_precheck_pass": model_switch.get("qwen37plus_precheck_pass"),
                    "bridge_pass": bridge_audit.get("bridge_pass"),
                    "bridge_input_coverage_rate": bridge_audit.get("bridge_input_coverage_rate"),
                    "next_action": readiness.get("next_action"),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
