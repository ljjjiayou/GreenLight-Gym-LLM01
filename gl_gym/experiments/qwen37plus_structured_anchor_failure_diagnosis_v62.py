"""v62 qwen3.7-plus structured-anchor failure diagnosis.

This stage is offline-only. It reads v61 qwen3.7-plus structured-anchor
shadow evidence and explains why the output contract failed. It does not run
rollout, call online LLMs, mutate the default controller, authorize replay, or
make performance claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gl_gym.experiments.profile_template_opt_in_shadow_rollout_v51 as v51  # noqa: E402
import gl_gym.experiments.qwen37plus_structured_anchor_opt_in_shadow_rollout_v61 as v61  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (  # noqa: E402
    AUDIT_DIR,
    FAILURE_SCENARIOS,
)


MODEL_NAME = "qwen3.7-plus"
ARTIFACT_DATE = "20260602"

V61_CACHE_PATH = v61.V61_CACHE_PATH
V61_RESULT_AUDIT_JSON = v61.RESULT_AUDIT_JSON
V61_BRIDGE_AUDIT_JSON = v61.BRIDGE_AUDIT_JSON
V61_READINESS_JSON = v61.READINESS_JSON

DIAGNOSIS_JSON = AUDIT_DIR / "qwen37plus_structured_anchor_failure_diagnosis_20260602_v62.json"
DIAGNOSIS_MD = AUDIT_DIR / "qwen37plus_structured_anchor_failure_diagnosis_20260602_v62.md"
CATALOG_JSON = AUDIT_DIR / "qwen37plus_structured_anchor_invalid_response_catalog_20260602_v62.json"
CATALOG_MD = AUDIT_DIR / "qwen37plus_structured_anchor_invalid_response_catalog_20260602_v62.md"
REPAIR_DESIGN_JSON = AUDIT_DIR / "qwen37plus_structured_anchor_prompt_retry_repair_design_20260602_v62.json"
REPAIR_DESIGN_MD = AUDIT_DIR / "qwen37plus_structured_anchor_prompt_retry_repair_design_20260602_v62.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260602_v62.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260602_v62.md"

FAILURE_TAXONOMY = (
    "invalid_json_truncated",
    "empty_raw_response",
    "retry_not_invoked_or_not_effective",
    "provider_returned_empty",
    "parser_salvage_possible_not_clean_evidence",
    "unknown_requires_instrumentation",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _rel(path: str | Path) -> str:
    p = _resolve(path)
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _load_json(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json_md(path_json: Path, path_md: Path, data: Mapping[str, Any], title: str) -> None:
    v51._write_json_md(path_json, path_md, data, title)


def _plan_cache_entries(cache_path: str | Path = V61_CACHE_PATH) -> list[tuple[str, dict[str, Any]]]:
    data = _load_json(cache_path)
    entries = data.get("entries", data if isinstance(data, Mapping) else {})
    if not isinstance(entries, Mapping):
        return []
    out: list[tuple[str, dict[str, Any]]] = []
    for key, value in entries.items():
        if isinstance(value, Mapping):
            item = dict(value)
            item.setdefault("key", str(key))
            out.append((str(key), item))
    return out


def _env_id_to_scenario(env_id: Any, *, max_steps: int = 720) -> str:
    text = str(env_id or "")
    if text.startswith("TomatoEnv_"):
        text = text[len("TomatoEnv_") :]
    if not text:
        return ""
    return text if "_n" in text else f"{text}_n{max_steps}"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _raw_preview(raw: str, *, limit: int = 220) -> str:
    text = str(raw or "").replace("\r", "\\r").replace("\n", "\\n")
    return text[:limit]


def _json_truncation_signals(raw: str) -> dict[str, Any]:
    stripped = str(raw or "").strip()
    open_braces = stripped.count("{")
    close_braces = stripped.count("}")
    open_brackets = stripped.count("[")
    close_brackets = stripped.count("]")
    try:
        json.loads(stripped)
        json_loads_ok = True
    except Exception:
        json_loads_ok = False
    trailing = stripped[-1:] if stripped else ""
    brace_balance_positive = open_braces > close_braces or open_brackets > close_brackets
    suspicious_trailing = trailing in {"", ",", ":", ".", "[", "{", '"'} or not stripped.endswith("}")
    return {
        "raw_length": int(len(stripped)),
        "starts_with_json_object": bool(stripped.startswith("{")),
        "ends_with_json_object": bool(stripped.endswith("}")),
        "json_loads_ok": bool(json_loads_ok),
        "open_brace_count": int(open_braces),
        "close_brace_count": int(close_braces),
        "open_bracket_count": int(open_brackets),
        "close_bracket_count": int(close_brackets),
        "brace_balance_positive": bool(brace_balance_positive),
        "suspicious_trailing_fragment": bool(suspicious_trailing),
        "looks_truncated": bool(stripped and not json_loads_ok and (brace_balance_positive or suspicious_trailing)),
    }


def _known_field_prefix_count(raw: str) -> int:
    fields = (
        "profile_intent",
        "target_temp",
        "target_co2",
        "target_rh",
        "risk_flags",
        "forbidden_intents",
        "planning_horizon_steps",
        "confidence",
    )
    return sum(1 for field in fields if f'"{field}"' in str(raw or ""))


def classify_invalid_entry(entry: Mapping[str, Any]) -> tuple[str, list[str], dict[str, Any]]:
    raw = str(entry.get("raw_response") or "")
    structured = entry.get("structured_anchor", {})
    structured = structured if isinstance(structured, Mapping) else {}
    errors = [str(item) for item in structured.get("errors", []) or []]
    empty = bool(structured.get("empty", False) or not raw.strip() or "empty_anchor" in errors)
    signals = _json_truncation_signals(raw)
    attempt = _safe_int(structured.get("attempt", entry.get("attempt", 0)), 0)

    secondary: list[str] = []
    primary = "unknown_requires_instrumentation"
    if empty:
        primary = "empty_raw_response"
        secondary.append("provider_returned_empty")
    elif "invalid_json_anchor" in errors and signals["looks_truncated"]:
        primary = "invalid_json_truncated"
    elif "invalid_json_anchor" in errors:
        primary = "unknown_requires_instrumentation"
        secondary.append("parser_salvage_possible_not_clean_evidence")

    if attempt <= 1:
        secondary.append("retry_not_invoked_or_not_effective")

    known_field_count = _known_field_prefix_count(raw)
    if primary == "invalid_json_truncated" and known_field_count >= 4:
        secondary.append("parser_salvage_possible_not_clean_evidence")

    secondary = sorted(set(item for item in secondary if item != primary))
    diagnostics = {
        **signals,
        "structured_anchor_errors": errors,
        "attempt": int(attempt),
        "llm_duration_seconds": _safe_float(entry.get("llm_duration_seconds"), 0.0),
        "known_structured_anchor_field_prefix_count": int(known_field_count),
        "salvage_must_not_count_as_clean_evidence": True,
    }
    return primary, secondary, diagnostics


def build_invalid_response_catalog(
    *,
    cache_path: str | Path = V61_CACHE_PATH,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    scenario_set = set(scenarios)
    entries = _plan_cache_entries(cache_path)
    invalid_entries: list[dict[str, Any]] = []
    primary_counts: Counter[str] = Counter()
    secondary_counts: Counter[str] = Counter()
    scenario_counts: Counter[str] = Counter()
    model_names: Counter[str] = Counter()

    for key, entry in entries:
        scenario = _env_id_to_scenario(entry.get("env_id"))
        if scenario not in scenario_set:
            continue
        structured = entry.get("structured_anchor", {})
        structured = structured if isinstance(structured, Mapping) else {}
        if not bool(structured.get("attempted", False)):
            continue
        if bool(structured.get("valid", False)):
            continue

        primary, secondary, diagnostics = classify_invalid_entry(entry)
        primary_counts[primary] += 1
        for item in secondary:
            secondary_counts[item] += 1
        scenario_counts[scenario] += 1
        model_names[str(entry.get("model_name") or "")] += 1
        raw = str(entry.get("raw_response") or "")
        invalid_entries.append(
            {
                "key": str(key),
                "scenario_id": scenario,
                "env_id": str(entry.get("env_id") or ""),
                "timestep": _safe_int(entry.get("timestep"), -1),
                "model_name": str(entry.get("model_name") or ""),
                "prompt_hash": str(structured.get("prompt_hash") or entry.get("prompt_hash") or ""),
                "primary_failure_type": primary,
                "secondary_failure_types": secondary,
                "raw_response_preview": _raw_preview(raw),
                "raw_response_tail": _raw_preview(raw[-180:]),
                "clean_planning_evidence": False,
                "diagnostics": diagnostics,
            }
        )

    invalid_entries.sort(key=lambda item: (str(item["scenario_id"]), int(item["timestep"]), str(item["key"])))
    return {
        "artifact": "qwen37plus_structured_anchor_invalid_response_catalog_20260602_v62",
        "current_stage": "v62_qwen37plus_structured_anchor_failure_diagnosis",
        "scope": "offline_invalid_structured_anchor_catalog_only",
        "model_name": MODEL_NAME,
        "cache_path": _rel(cache_path),
        "scenarios": list(scenarios),
        "invalid_or_empty_entry_count": int(len(invalid_entries)),
        "expected_invalid_or_empty_entry_count": 13,
        "catalog_count_matches_v61_failure": bool(len(invalid_entries) == 13),
        "primary_failure_type_counts": dict(sorted(primary_counts.items())),
        "secondary_failure_type_counts": dict(sorted(secondary_counts.items())),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "model_name_counts": dict(sorted(model_names.items())),
        "invalid_entries": invalid_entries,
        "salvaged_or_truncated_clean_evidence_count": 0,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def build_failure_diagnosis(
    catalog: Mapping[str, Any],
    *,
    result_audit_path: str | Path = V61_RESULT_AUDIT_JSON,
    bridge_audit_path: str | Path = V61_BRIDGE_AUDIT_JSON,
    readiness_path: str | Path = V61_READINESS_JSON,
) -> dict[str, Any]:
    result = _load_json(result_audit_path)
    bridge = _load_json(bridge_audit_path)
    readiness = _load_json(readiness_path)
    primary_counts = Counter(catalog.get("primary_failure_type_counts", {}) or {})
    secondary_counts = Counter(catalog.get("secondary_failure_type_counts", {}) or {})
    unknown_count = int(primary_counts.get("unknown_requires_instrumentation", 0))
    format_or_empty_count = int(
        primary_counts.get("invalid_json_truncated", 0) + primary_counts.get("empty_raw_response", 0)
    )
    invalid_count = int(catalog.get("invalid_or_empty_entry_count", 0) or 0)
    dominant_root_cause = (
        "structured_anchor_format_truncation_and_empty_response_with_missing_compact_retry"
        if invalid_count > 0 and unknown_count == 0 and format_or_empty_count == invalid_count
        else "additional_instrumentation_required_before_fix"
    )
    repair_direction = (
        "compact_json_prompt_and_invalid_or_empty_anchor_retry"
        if dominant_root_cause != "additional_instrumentation_required_before_fix"
        else "structured_anchor_provider_instrumentation_patch"
    )
    clean_valid_count = int((result.get("cache_audit", {}) or {}).get("valid_structured_anchor_count", 0) or 0)
    attempt_count = int(readiness.get("structured_anchor_attempt_count", result.get("structured_anchor_attempt_count", 0)) or 0)
    return {
        "artifact": "qwen37plus_structured_anchor_failure_diagnosis_20260602_v62",
        "current_stage": "v62_qwen37plus_structured_anchor_failure_diagnosis",
        "scope": "offline_failure_diagnosis_only",
        "model_name": MODEL_NAME,
        "inputs": {
            "cache_path": str(catalog.get("cache_path", "")),
            "v61_result_audit": _rel(result_audit_path),
            "v61_bridge_audit": _rel(bridge_audit_path),
            "v61_readiness": _rel(readiness_path),
        },
        "v61_fact_source": {
            "online_llm_accessible": bool(readiness.get("online_llm_accessible", False)),
            "model_consistency_pass": bool(readiness.get("model_consistency_pass", False)),
            "provider_error_steps": int(readiness.get("provider_error_steps", 0) or 0),
            "structured_anchor_attempt_count": attempt_count,
            "valid_structured_anchor_count": clean_valid_count,
            "valid_structured_anchor_rate": float(readiness.get("valid_structured_anchor_rate", 0.0) or 0.0),
            "empty_structured_anchor_rate": float(readiness.get("empty_structured_anchor_rate", 0.0) or 0.0),
            "bridge_input_coverage_rate": float(readiness.get("bridge_input_coverage_rate", 0.0) or 0.0),
            "bridge_failure_reason_counts": bridge.get("failure_reason_counts", {}),
        },
        "invalid_or_empty_entry_count": invalid_count,
        "invalid_or_empty_entry_expected_count": 13,
        "catalog_count_matches_v61_failure": bool(catalog.get("catalog_count_matches_v61_failure", False)),
        "primary_failure_type_counts": dict(primary_counts),
        "secondary_failure_type_counts": dict(secondary_counts),
        "unknown_requires_instrumentation_count": unknown_count,
        "dominant_root_cause": dominant_root_cause,
        "repair_direction": repair_direction,
        "clean_evidence_policy": {
            "invalid_anchor_clean_planning_evidence_allowed": False,
            "truncated_or_salvaged_anchor_clean_planning_evidence_allowed": False,
            "invalid_anchor_not_clean_evidence": True,
        },
        "next_action": (
            "qwen37plus_structured_anchor_prompt_retry_opt_in_shadow_rollout_plan"
            if dominant_root_cause != "additional_instrumentation_required_before_fix"
            else "structured_anchor_provider_instrumentation_patch_plan"
        ),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def build_repair_design(diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    repair_ready = str(diagnosis.get("dominant_root_cause", "")) != "additional_instrumentation_required_before_fix"
    return {
        "artifact": "qwen37plus_structured_anchor_prompt_retry_repair_design_20260602_v62",
        "current_stage": "v62_qwen37plus_structured_anchor_prompt_retry_repair_design",
        "scope": "design_only_no_online_rollout",
        "model_name": MODEL_NAME,
        "repair_design_ready": bool(repair_ready),
        "prompt_contract_design": {
            "response_shape": "single compact JSON object",
            "wrapper_key": "structured_planning_anchor",
            "fixed_key_order": [
                "profile_intent",
                "target_temp",
                "target_co2",
                "target_rh",
                "risk_flags",
                "forbidden_intents",
                "planning_horizon_steps",
                "confidence",
            ],
            "forbid_markdown": True,
            "forbid_prose": True,
            "forbid_tool_calls": True,
            "forbid_low_level_final_control_fields": True,
            "risk_flags_max_items": 3,
            "forbidden_intents_max_items": 3,
        },
        "retry_policy_design": {
            "retry_on": ["invalid_json_anchor", "empty_anchor"],
            "max_structured_anchor_retries": 1,
            "retry_prompt_mode": "compact_json_only_repair",
            "retry_must_not_use_tools_or_final_controls": True,
            "retry_failure_not_clean_evidence": True,
        },
        "parser_salvage_policy": {
            "shadow_salvage_diagnostics_allowed": True,
            "salvaged_truncated_anchor_clean_evidence_allowed": False,
            "salvage_use": "diagnosis_only_until_exact_required_fields_parse_cleanly",
        },
        "max_tokens_policy": {
            "current_v61_max_tokens": 2048,
            "first_fix": "compact_prompt_and_retry_before_token_increase",
            "increase_tokens_only_if_compact_retry_still_truncates": True,
        },
        "default_controller_changed": False,
        "rollout_execution_record_generated": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": diagnosis.get("next_action", "structured_anchor_provider_instrumentation_patch_plan"),
    }


def build_readiness(
    diagnosis: Mapping[str, Any],
    catalog: Mapping[str, Any],
    repair_design: Mapping[str, Any],
) -> dict[str, Any]:
    diagnosis_ready = bool(
        catalog.get("catalog_count_matches_v61_failure", False)
        and int(diagnosis.get("unknown_requires_instrumentation_count", 0) or 0) == 0
        and repair_design.get("repair_design_ready", False)
    )
    return {
        "artifact": "metadata_replay_readiness_checklist_20260602_v62",
        "current_stage": "v62_qwen37plus_structured_anchor_failure_diagnosis",
        "model_name": MODEL_NAME,
        "structured_anchor_failure_diagnosis_done": True,
        "invalid_response_catalog_done": True,
        "prompt_retry_repair_design_done": True,
        "invalid_or_empty_entry_count": int(catalog.get("invalid_or_empty_entry_count", 0) or 0),
        "catalog_count_matches_v61_failure": bool(catalog.get("catalog_count_matches_v61_failure", False)),
        "dominant_root_cause": str(diagnosis.get("dominant_root_cause", "")),
        "unknown_requires_instrumentation_count": int(diagnosis.get("unknown_requires_instrumentation_count", 0) or 0),
        "repair_design_ready": bool(repair_design.get("repair_design_ready", False)),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": (
            "qwen37plus_structured_anchor_prompt_retry_opt_in_shadow_rollout_plan"
            if diagnosis_ready
            else "structured_anchor_provider_instrumentation_patch_plan"
        ),
        "stop_taxonomy": [] if diagnosis_ready else ["additional_instrumentation_required_before_fix"],
    }


def write_artifacts(
    diagnosis: Mapping[str, Any],
    catalog: Mapping[str, Any],
    repair_design: Mapping[str, Any],
    readiness: Mapping[str, Any],
) -> None:
    _write_json_md(DIAGNOSIS_JSON, DIAGNOSIS_MD, diagnosis, "v62 qwen3.7-plus Structured Anchor Failure Diagnosis")
    _write_json_md(CATALOG_JSON, CATALOG_MD, catalog, "v62 qwen3.7-plus Structured Anchor Invalid Response Catalog")
    _write_json_md(REPAIR_DESIGN_JSON, REPAIR_DESIGN_MD, repair_design, "v62 qwen3.7-plus Structured Anchor Prompt/Retry Repair Design")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v62 Metadata Replay Readiness")


def build_all(*, cache_path: str | Path = V61_CACHE_PATH) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    catalog = build_invalid_response_catalog(cache_path=cache_path)
    diagnosis = build_failure_diagnosis(catalog)
    repair_design = build_repair_design(diagnosis)
    readiness = build_readiness(diagnosis, catalog, repair_design)
    return diagnosis, catalog, repair_design, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-all", action="store_true")
    parser.add_argument("--cache-path", default=str(V61_CACHE_PATH))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    diagnosis, catalog, repair_design, readiness = build_all(cache_path=args.cache_path)
    if args.write_all:
        write_artifacts(diagnosis, catalog, repair_design, readiness)
    print(
        json.dumps(
            {
                "invalid_or_empty_entry_count": catalog["invalid_or_empty_entry_count"],
                "primary_failure_type_counts": catalog["primary_failure_type_counts"],
                "dominant_root_cause": diagnosis["dominant_root_cause"],
                "repair_design_ready": repair_design["repair_design_ready"],
                "next_action": readiness["next_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
