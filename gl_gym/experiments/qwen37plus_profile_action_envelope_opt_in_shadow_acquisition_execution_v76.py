"""Build v76 execution records for profile-action envelope shadow acquisition.

This module summarizes an already-authorized v75 opt-in online shadow
acquisition and its v74 audit result. It does not call online LLMs, run rollout,
run replay, change the default controller, alter final actions, or make
performance claims.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments import qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_v75 as v75  # noqa: E402
from gl_gym.experiments import qwen37plus_profile_action_envelope_shadow_patch_v74 as v74  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import AUDIT_DIR  # noqa: E402
from gl_gym.experiments.profile_scorer_decision_audit import (  # noqa: E402
    discover_traces,
    read_trace,
    trace_identity,
)


MODEL_NAME = "qwen3.7-plus"
ARTIFACT_DATE = "20260604"
VERSION = "v76"

AUTHORIZATION_JSON = v75.AUTHORIZATION_JSON
MANIFEST_JSON = v75.MANIFEST_JSON
V75_READINESS_JSON = v75.READINESS_JSON
V74_AUDIT_JSON = v74.AUDIT_JSON
V74_READINESS_JSON = v74.READINESS_JSON

EXECUTION_RECORD_JSON = (
    AUDIT_DIR / "qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_execution_record_20260604_v76.json"
)
EXECUTION_RECORD_MD = (
    AUDIT_DIR / "qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_execution_record_20260604_v76.md"
)
READINESS_JSON = AUDIT_DIR / "profile_action_envelope_opt_in_shadow_acquisition_result_readiness_20260604_v76.json"
READINESS_MD = AUDIT_DIR / "profile_action_envelope_opt_in_shadow_acquisition_result_readiness_20260604_v76.md"

BOUNDARY_FALSE_FIELDS = (
    "default_llm_rspc_v2_changed",
    "fallback_enhanced",
    "controlled_replay_allowed",
    "controlled_replay_execution_allowed",
    "metadata_replay_execution_allowed",
    "performance_claim_allowed",
    "promotion_evidence",
    "final_action_changed",
    "strict_replay_allowed",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _rel(path: str | Path) -> str:
    try:
        return str(_resolve(path).relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(path)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _with_boundaries(payload: dict[str, Any]) -> dict[str, Any]:
    for field in BOUNDARY_FALSE_FIELDS:
        payload[field] = False
    return payload


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _parse_candidates_json(value: Any) -> tuple[list[dict[str, Any]], bool]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)], False
    if isinstance(value, str) and value.strip():
        for loader in (json.loads, ast.literal_eval):
            try:
                decoded = loader(value)
            except Exception:
                continue
            if isinstance(decoded, list):
                return [item for item in decoded if isinstance(item, dict)], False
    return [], bool(str(value or "").strip())


def _summarize_trace(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    candidate_rows = 0
    parse_fail_rows = 0
    tomato_projection_rows = 0
    for row in rows:
        count = _int(row.get("profile_action_envelope_shadow_candidate_count", 0))
        candidates, failed = _parse_candidates_json(row.get("profile_action_envelope_shadow_candidates_json", ""))
        if count > 0 or candidates:
            candidate_rows += 1
        if failed:
            parse_fail_rows += 1
        if candidates and all(isinstance(item.get("tomato_safety_projection", {}), Mapping) for item in candidates):
            tomato_projection_rows += 1
    return {
        **trace_identity(path),
        "path": _rel(path),
        "rows": int(len(rows)),
        "runtime_error_steps": int(sum(bool(row.get("runtime_error")) for row in rows)),
        "strict_cache_miss_rows": int(
            sum(str(row.get("runtime_error_type", "")) == "strict_plan_cache_miss" for row in rows)
        ),
        "profile_action_envelope_shadow_enabled_rows": int(
            sum(_truthy(row.get("profile_action_envelope_shadow_enabled", False)) for row in rows)
        ),
        "profile_action_envelope_candidate_rows": int(candidate_rows),
        "profile_action_envelope_eligible_rows": int(
            sum(_int(row.get("profile_action_envelope_shadow_eligible_candidate_count", 0)) > 0 for row in rows)
        ),
        "tomato_safety_projection_rows": int(tomato_projection_rows),
        "candidate_json_parse_fail_rows": int(parse_fail_rows),
        "final_action_changed_rows": int(
            sum(_truthy(row.get("profile_action_envelope_shadow_final_action_changed", False)) for row in rows)
        ),
    }


def summarize_traces(trace_dir: str | Path) -> dict[str, Any]:
    paths = discover_traces([trace_dir]) if _resolve(trace_dir).exists() else []
    traces = [_summarize_trace(path) for path in paths]
    return {
        "trace_dir": _rel(trace_dir),
        "trace_count": int(len(traces)),
        "rows_total": int(sum(_int(item.get("rows")) for item in traces)),
        "runtime_error_steps_total": int(sum(_int(item.get("runtime_error_steps")) for item in traces)),
        "strict_cache_miss_rows_total": int(sum(_int(item.get("strict_cache_miss_rows")) for item in traces)),
        "profile_action_envelope_shadow_enabled_rows_total": int(
            sum(_int(item.get("profile_action_envelope_shadow_enabled_rows")) for item in traces)
        ),
        "profile_action_envelope_candidate_rows_total": int(
            sum(_int(item.get("profile_action_envelope_candidate_rows")) for item in traces)
        ),
        "profile_action_envelope_eligible_rows_total": int(
            sum(_int(item.get("profile_action_envelope_eligible_rows")) for item in traces)
        ),
        "tomato_safety_projection_rows_total": int(sum(_int(item.get("tomato_safety_projection_rows")) for item in traces)),
        "candidate_json_parse_fail_rows_total": int(sum(_int(item.get("candidate_json_parse_fail_rows")) for item in traces)),
        "final_action_changed_rows_total": int(sum(_int(item.get("final_action_changed_rows")) for item in traces)),
        "traces": traces,
    }


def summarize_cache(cache_path: str | Path) -> dict[str, Any]:
    p = _resolve(cache_path)
    if not p.exists():
        return {"cache_path": _rel(cache_path), "exists": False, "size_bytes": 0, "entry_count": 0}
    data = _load_json(p)
    entries = data.get("entries", data if isinstance(data, Mapping) else {})
    entry_count = len(entries) if isinstance(entries, Mapping) else 0
    return {
        "cache_path": _rel(cache_path),
        "exists": True,
        "size_bytes": int(p.stat().st_size),
        "entry_count": int(entry_count),
    }


def _v74_next_action(v74_audit: Mapping[str, Any]) -> str:
    readiness = v74_audit.get("readiness", {})
    if isinstance(readiness, Mapping) and readiness.get("next_action"):
        return str(readiness.get("next_action"))
    aggregate = v74_audit.get("aggregate", {})
    if isinstance(aggregate, Mapping):
        recommendation = aggregate.get("recommendation", {})
        if isinstance(recommendation, Mapping) and recommendation.get("next_action"):
            return str(recommendation.get("next_action"))
    return "v74_audit_missing"


def build_execution_record(
    *,
    authorization: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
    v75_readiness: Mapping[str, Any] | None = None,
    v74_audit: Mapping[str, Any] | None = None,
    trace_dir: str | Path | None = None,
    cache_path: str | Path | None = None,
) -> dict[str, Any]:
    authorization = dict(authorization or _load_json(AUTHORIZATION_JSON))
    manifest = dict(manifest or _load_json(MANIFEST_JSON))
    v75_readiness = dict(v75_readiness or _load_json(V75_READINESS_JSON))
    v74_audit = dict(v74_audit or _load_json(V74_AUDIT_JSON))
    trace_dir = trace_dir or manifest.get("trace_dir") or v75.TRACE_DIR
    cache_path = cache_path or manifest.get("cache_path") or v75.CACHE_PATH
    trace_summary = summarize_traces(trace_dir)
    cache_summary = summarize_cache(cache_path)
    v74_readiness = v74_audit.get("readiness", {})
    if not isinstance(v74_readiness, Mapping):
        v74_readiness = {}
    aggregate = v74_audit.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        aggregate = {}

    record = {
        "artifact": "qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_execution_record_20260604_v76",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "purpose": "Summarize authorized v75 opt-in online shadow acquisition and v74 envelope audit closure.",
        "authorization": {
            "authorization_source": str(authorization.get("authorization_source", "")),
            "user_authorized_online_llm_acquisition": bool(
                authorization.get("user_authorized_online_llm_acquisition", False)
            ),
            "online_llm_allowed_for_acquisition": bool(
                authorization.get("online_llm_allowed_for_acquisition", False)
            ),
            "scenario_windows": list(authorization.get("scenario_windows", []) or []),
            "cache_write_policy": str(authorization.get("cache_write_policy", "")),
        },
        "boundary": {
            "default_controller": "llm_rspc_v2",
            "default_llm_rspc_v2_changed": False,
            "shadow_only": True,
            "final_action_changed": False,
            "online_llm_called_for_acquisition": bool(trace_summary["trace_count"] > 0),
            "online_llm_called_for_audit_or_tests": False,
            "controlled_replay_allowed": False,
            "controlled_replay_execution_allowed": False,
            "metadata_replay_execution_allowed": False,
            "strict_replay_allowed": False,
            "performance_claim_allowed": False,
            "promotion_evidence": False,
            "fallback_enhanced": False,
        },
        "inputs": {
            "controller": "llm_rspc_v2",
            "llm_model": MODEL_NAME,
            "agent_config_overrides": dict(manifest.get("agent_config_overrides", {}) or {}),
            "plan_cache_mode": str(manifest.get("plan_cache_mode", "")),
            "plan_cache_key_policy": str(manifest.get("plan_cache_key_policy", "")),
            "v75_readiness_next_action": str(v75_readiness.get("next_action", "")),
            "manifest_command_count": int(len(manifest.get("commands", []) or [])),
        },
        "outputs": {
            "trace_dir": _rel(trace_dir),
            "cache_path": _rel(cache_path),
            "v74_audit_json": _rel(V74_AUDIT_JSON),
            "v74_readiness_json": _rel(V74_READINESS_JSON),
            "execution_record_md": _rel(EXECUTION_RECORD_MD),
        },
        "acquisition_summary": trace_summary,
        "cache_summary": cache_summary,
        "v74_audit_result": {
            "trace_count": _int(v74_audit.get("trace_count", 0)),
            "rows": _int(aggregate.get("rows")),
            "profile_action_envelope_shadow_steps": _int(aggregate.get("profile_action_envelope_shadow_steps")),
            "normal_path_profile_action_envelope_steps": _int(
                aggregate.get("normal_path_profile_action_envelope_steps")
            ),
            "contract_missing_steps": _int(aggregate.get("contract_missing_steps")),
            "tomato_safety_projection_missing_steps": _int(
                aggregate.get("tomato_safety_projection_missing_steps")
            ),
            "action_diff_steps": _int(aggregate.get("action_diff_steps")),
            "fallback_candidate_source_steps": _int(aggregate.get("fallback_candidate_source_steps")),
            "next_action": _v74_next_action(v74_audit),
            "reason": str(v74_readiness.get("reason", "")),
        },
        "runtime_notes": {
            "runtime_error_steps_total": int(trace_summary["runtime_error_steps_total"]),
            "strict_cache_miss_rows_total": int(trace_summary["strict_cache_miss_rows_total"]),
            "short_trajectory_observed": bool(
                any(_int(item.get("rows")) < _int(manifest.get("max_steps", 720), 720) for item in trace_summary["traces"])
            ),
            "casadi_or_cvodes_warning_possible": bool(trace_summary["runtime_error_steps_total"] > 0),
            "performance_claim_allowed": False,
        },
    }
    return _with_boundaries(record)


def build_result_readiness(record: Mapping[str, Any]) -> dict[str, Any]:
    acquisition = record.get("acquisition_summary", {})
    if not isinstance(acquisition, Mapping):
        acquisition = {}
    v74_result = record.get("v74_audit_result", {})
    if not isinstance(v74_result, Mapping):
        v74_result = {}
    trace_count = _int(acquisition.get("trace_count"))
    if trace_count <= 0:
        next_action = "v75_acquisition_trace_missing_repair_plan"
        blocked_reason = "v75_trace_outputs_missing"
    elif _int(v74_result.get("action_diff_steps")) > 0:
        next_action = "stop_profile_action_envelope_shadow_action_invariance_violation"
        blocked_reason = "v74_audit_detected_action_diff"
    elif _int(v74_result.get("fallback_candidate_source_steps")) > 0:
        next_action = "profile_action_envelope_candidate_source_repair_plan"
        blocked_reason = "v74_audit_detected_fallback_source"
    elif _int(v74_result.get("contract_missing_steps")) > 0 or _int(
        v74_result.get("tomato_safety_projection_missing_steps")
    ) > 0:
        next_action = "profile_action_envelope_contract_or_projection_repair_plan"
        blocked_reason = "v74_audit_detected_contract_or_projection_gap"
    elif _int(v74_result.get("normal_path_profile_action_envelope_steps")) <= 0:
        next_action = "profile_action_envelope_shadow_instrumentation_repair_plan"
        blocked_reason = "v74_audit_missing_envelope_candidates"
    else:
        next_action = str(v74_result.get("next_action") or "profile_action_envelope_compatibility_shadow_audit_plan")
        blocked_reason = ""

    readiness = {
        "artifact": "profile_action_envelope_opt_in_shadow_acquisition_result_readiness_20260604_v76",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "trace_count": trace_count,
        "rows_total": _int(acquisition.get("rows_total")),
        "cache_entry_count": _int(record.get("cache_summary", {}).get("entry_count") if isinstance(record.get("cache_summary", {}), Mapping) else 0),
        "profile_action_envelope_candidate_rows_total": _int(
            acquisition.get("profile_action_envelope_candidate_rows_total")
        ),
        "contract_complete": bool(
            _int(v74_result.get("normal_path_profile_action_envelope_steps")) > 0
            and _int(v74_result.get("contract_missing_steps")) == 0
        ),
        "tomato_safety_projection_present": bool(
            _int(v74_result.get("normal_path_profile_action_envelope_steps")) > 0
            and _int(v74_result.get("tomato_safety_projection_missing_steps")) == 0
        ),
        "final_action_invariant": bool(_int(v74_result.get("action_diff_steps")) == 0),
        "fallback_not_enhanced": bool(_int(v74_result.get("fallback_candidate_source_steps")) == 0),
        "online_llm_called_for_acquisition": bool(record.get("boundary", {}).get("online_llm_called_for_acquisition", False))
        if isinstance(record.get("boundary", {}), Mapping)
        else False,
        "online_llm_called_for_audit_or_tests": False,
        "strict_replay_executed": False,
        "recommended_followup": (
            "profile_action_envelope_scorer_shadow_design_plan"
            if next_action == "profile_action_envelope_compatibility_shadow_audit_plan"
            else next_action
        ),
        "blocked_reason": blocked_reason,
        "next_action": next_action,
    }
    return _with_boundaries(readiness)


def build_execution_report(record: Mapping[str, Any]) -> str:
    acquisition = record.get("acquisition_summary", {})
    v74_result = record.get("v74_audit_result", {})
    boundary = record.get("boundary", {})
    if not isinstance(acquisition, Mapping):
        acquisition = {}
    if not isinstance(v74_result, Mapping):
        v74_result = {}
    if not isinstance(boundary, Mapping):
        boundary = {}
    lines = [
        "# qwen3.7-plus v76 Profile Action Envelope Acquisition Execution Record",
        "",
        "## Boundary",
        "",
        f"- default_llm_rspc_v2_changed={bool(boundary.get('default_llm_rspc_v2_changed', False))}",
        f"- shadow_only={bool(boundary.get('shadow_only', False))}",
        f"- final_action_changed={bool(boundary.get('final_action_changed', False))}",
        f"- online_llm_called_for_acquisition={bool(boundary.get('online_llm_called_for_acquisition', False))}",
        f"- online_llm_called_for_audit_or_tests={bool(boundary.get('online_llm_called_for_audit_or_tests', False))}",
        f"- strict_replay_allowed={bool(boundary.get('strict_replay_allowed', False))}",
        f"- performance_claim_allowed={bool(boundary.get('performance_claim_allowed', False))}",
        f"- promotion_evidence={bool(boundary.get('promotion_evidence', False))}",
        "",
        "## Acquisition Summary",
        "",
        f"- trace_count={_int(acquisition.get('trace_count'))}",
        f"- rows_total={_int(acquisition.get('rows_total'))}",
        f"- envelope_candidate_rows={_int(acquisition.get('profile_action_envelope_candidate_rows_total'))}",
        f"- final_action_changed_rows={_int(acquisition.get('final_action_changed_rows_total'))}",
        "",
        "## v74 Audit Result",
        "",
        f"- normal_path_profile_action_envelope_steps={_int(v74_result.get('normal_path_profile_action_envelope_steps'))}",
        f"- contract_missing_steps={_int(v74_result.get('contract_missing_steps'))}",
        f"- tomato_safety_projection_missing_steps={_int(v74_result.get('tomato_safety_projection_missing_steps'))}",
        f"- action_diff_steps={_int(v74_result.get('action_diff_steps'))}",
        f"- fallback_candidate_source_steps={_int(v74_result.get('fallback_candidate_source_steps'))}",
        f"- next_action={v74_result.get('next_action', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_readiness_report(readiness: Mapping[str, Any]) -> str:
    lines = [
        "# qwen3.7-plus v76 Profile Action Envelope Acquisition Readiness",
        "",
        f"- trace_count={_int(readiness.get('trace_count'))}",
        f"- rows_total={_int(readiness.get('rows_total'))}",
        f"- contract_complete={bool(readiness.get('contract_complete', False))}",
        f"- tomato_safety_projection_present={bool(readiness.get('tomato_safety_projection_present', False))}",
        f"- final_action_invariant={bool(readiness.get('final_action_invariant', False))}",
        f"- online_llm_called_for_acquisition={bool(readiness.get('online_llm_called_for_acquisition', False))}",
        f"- online_llm_called_for_audit_or_tests={bool(readiness.get('online_llm_called_for_audit_or_tests', False))}",
        f"- strict_replay_executed={bool(readiness.get('strict_replay_executed', False))}",
        f"- performance_claim_allowed={bool(readiness.get('performance_claim_allowed', False))}",
        f"- promotion_evidence={bool(readiness.get('promotion_evidence', False))}",
        f"- recommended_followup={readiness.get('recommended_followup', '')}",
        f"- blocked_reason={readiness.get('blocked_reason', '')}",
        f"- next_action={readiness.get('next_action', '')}",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(
    record: Mapping[str, Any],
    *,
    execution_json: str | Path = EXECUTION_RECORD_JSON,
    execution_md: str | Path = EXECUTION_RECORD_MD,
    readiness_json: str | Path = READINESS_JSON,
    readiness_md: str | Path = READINESS_MD,
) -> dict[str, str]:
    readiness = build_result_readiness(record)
    execution_json = Path(execution_json)
    execution_md = Path(execution_md)
    readiness_json = Path(readiness_json)
    readiness_md = Path(readiness_md)
    for path in (execution_json, execution_md, readiness_json, readiness_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    execution_json.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    execution_md.write_text(build_execution_report(record), encoding="utf-8")
    readiness_json.write_text(json.dumps(readiness, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    readiness_md.write_text(build_readiness_report(readiness), encoding="utf-8")
    return {
        "execution_json": str(execution_json),
        "execution_md": str(execution_md),
        "readiness_json": str(readiness_json),
        "readiness_md": str(readiness_md),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-json", default=str(AUTHORIZATION_JSON))
    parser.add_argument("--manifest-json", default=str(MANIFEST_JSON))
    parser.add_argument("--v75-readiness-json", default=str(V75_READINESS_JSON))
    parser.add_argument("--v74-audit-json", default=str(V74_AUDIT_JSON))
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--cache-path", default="")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    manifest = _load_json(args.manifest_json)
    record = build_execution_record(
        authorization=_load_json(args.authorization_json),
        manifest=manifest,
        v75_readiness=_load_json(args.v75_readiness_json),
        v74_audit=_load_json(args.v74_audit_json),
        trace_dir=args.trace_dir or manifest.get("trace_dir") or v75.TRACE_DIR,
        cache_path=args.cache_path or manifest.get("cache_path") or v75.CACHE_PATH,
    )
    if args.no_write:
        print(json.dumps({"record": record, "readiness": build_result_readiness(record)}, indent=2, ensure_ascii=False))
        return 0
    print(json.dumps(write_artifacts(record), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
