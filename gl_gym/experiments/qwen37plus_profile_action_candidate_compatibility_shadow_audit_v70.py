"""Audit v70 profile-action candidate compatibility and hypothesis readiness.

This stage consumes v69 opt-in shadow trace provenance. It does not run rollout,
call online LLMs, change the default controller, alter final actions, or make
performance claims. If no v69-compatible trace is present, it produces an
opt-in shadow trace authorization plan instead of pretending evidence exists.
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

from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import AUDIT_DIR  # noqa: E402
from gl_gym.experiments.profile_scorer_decision_audit import (  # noqa: E402
    discover_traces,
    read_trace,
    trace_identity,
)


MODEL_NAME = "qwen3.7-plus"
ARTIFACT_DATE = "20260604"
VERSION = "v70"

AUDIT_JSON = AUDIT_DIR / "qwen37plus_profile_action_candidate_compatibility_shadow_audit_20260604_v70.json"
AUDIT_MD = AUDIT_DIR / "qwen37plus_profile_action_candidate_compatibility_shadow_audit_20260604_v70.md"
HYPOTHESIS_JSON = AUDIT_DIR / "profile_action_candidate_hypothesis_check_20260604_v70.json"
HYPOTHESIS_MD = AUDIT_DIR / "profile_action_candidate_hypothesis_check_20260604_v70.md"
AUTHORIZATION_JSON = AUDIT_DIR / "opt_in_shadow_trace_authorization_plan_20260604_v70.json"
AUTHORIZATION_MD = AUDIT_DIR / "opt_in_shadow_trace_authorization_plan_20260604_v70.md"

ACTION_FIELDS = ("heat", "co2", "screen", "vent", "lamp", "shade")
CONTRACT_REQUIRED_FIELDS = (
    "name",
    "profile_name",
    "candidate_source",
    "raw_action",
    "post_tomato_action",
    "score_terms",
    "tomato_safety_v2_applied",
    "eligible",
    "rejection_reason",
)
SCORE_TERM_FIELDS = (
    "raw_score",
    "post_tomato_score",
    "selection_score",
    "profile_target_error",
    "action_delta_penalty",
    "tomato_safety_penalty",
    "compatibility_penalty",
    "hard_safety_rewrite_predicted",
)
BOUNDARY_FALSE_FIELDS = (
    "online_llm_called",
    "new_rollout_run",
    "default_llm_rspc_v2_changed",
    "fallback_enhanced",
    "controlled_replay_allowed",
    "controlled_replay_execution_allowed",
    "metadata_replay_execution_allowed",
    "performance_claim_allowed",
    "promotion_evidence",
)

TOMATO_REWRITE_MAX_DELTA = 0.35
PROFILE_TARGET_ERROR_MAX = 2.0
ACTION_DELTA_PENALTY_MAX = 1.20

FAILURE_SCENARIOS = (
    "y2010_d180_s43_n720",
    "y2018_d181_s42_n720",
    "y2018_d181_s43_n720",
)
REQUIRED_V70_TRACE_FIELDS = (
    "profile_action_candidate_shadow_enabled",
    "profile_action_candidate_shadow_candidate_count",
    "profile_action_candidate_shadow_eligible_candidate_count",
    "profile_action_candidate_shadow_final_action_changed",
    "profile_action_candidate_shadow_candidates_json",
)


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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _parse_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("profile_action_candidate_shadow_candidates_json", [])
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, str) and raw.strip():
        for loader in (json.loads, ast.literal_eval):
            try:
                decoded = loader(raw)
            except Exception:
                continue
            if isinstance(decoded, list):
                return [item for item in decoded if isinstance(item, dict)]
    return []


def _action_complete(action: Any) -> bool:
    if not isinstance(action, Mapping):
        return False
    for field in ACTION_FIELDS:
        if field not in action:
            return False
        try:
            float(action.get(field))
        except Exception:
            return False
    return True


def _candidate_missing_fields(candidate: Mapping[str, Any]) -> list[str]:
    missing = [field for field in CONTRACT_REQUIRED_FIELDS if field not in candidate]
    if not str(candidate.get("name", "")).startswith("profile_action:"):
        missing.append("name:profile_action_prefix")
    if str(candidate.get("candidate_source", "")) != "normal_path_profile_candidate":
        missing.append("candidate_source:normal_path_profile_candidate")
    if not _action_complete(candidate.get("raw_action", {})):
        missing.append("raw_action:action_fields")
    if not _action_complete(candidate.get("post_tomato_action", {})):
        missing.append("post_tomato_action:action_fields")
    score_terms = candidate.get("score_terms", {})
    if not isinstance(score_terms, Mapping):
        missing.append("score_terms:mapping")
    else:
        missing.extend(f"score_terms:{field}" for field in SCORE_TERM_FIELDS if field not in score_terms)
    return sorted(dict.fromkeys(missing))


def _action_max_abs_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    if not _action_complete(a) or not _action_complete(b):
        return 0.0
    return max(abs(_num(b.get(field)) - _num(a.get(field))) for field in ACTION_FIELDS)


def _row_action_changed(row: Mapping[str, Any]) -> bool:
    return any(
        _truthy(row.get(key, False))
        for key in (
            "profile_action_candidate_shadow_final_action_changed",
            "profile_action_candidate_shadow_action_diff_nonzero",
            "v69_action_invariant_violation",
            "v70_action_invariant_violation",
        )
    )


def _candidate_flags(candidate: Mapping[str, Any]) -> dict[str, Any]:
    missing = _candidate_missing_fields(candidate)
    score_terms = candidate.get("score_terms", {})
    if not isinstance(score_terms, Mapping):
        score_terms = {}
    raw_action = candidate.get("raw_action", {})
    post_action = candidate.get("post_tomato_action", {})
    rewrite_delta = _action_max_abs_delta(raw_action, post_action) if isinstance(raw_action, Mapping) and isinstance(post_action, Mapping) else 0.0
    rejection = str(candidate.get("rejection_reason", "") or "")
    source = str(candidate.get("candidate_source", "") or "")
    fallback_source = source == "fallback_candidate"
    tomato_projection_missing = "tomato_safety_v2_applied" not in candidate or not _action_complete(post_action)
    tomato_safety_incompatible = bool(
        rewrite_delta > TOMATO_REWRITE_MAX_DELTA
        or "hard_safety" in rejection
        or "unsafe" in rejection
    )
    profile_target_incompatible = bool(
        _num(score_terms.get("profile_target_error")) > PROFILE_TARGET_ERROR_MAX
        or _truthy(candidate.get("profile_target_incompatible", False))
    )
    action_continuity_incompatible = bool(
        _num(score_terms.get("action_delta_penalty")) > ACTION_DELTA_PENALTY_MAX
        or _truthy(candidate.get("action_continuity_incompatible", False))
    )
    compatible = bool(
        not missing
        and not fallback_source
        and not tomato_projection_missing
        and not tomato_safety_incompatible
        and not profile_target_incompatible
        and not action_continuity_incompatible
        and bool(candidate.get("eligible", False))
    )
    return {
        "contract_missing": bool(missing),
        "missing_fields": missing,
        "fallback_source": bool(fallback_source),
        "tomato_projection_missing": bool(tomato_projection_missing),
        "tomato_rewrite_max_abs_delta": float(rewrite_delta),
        "tomato_safety_incompatible": bool(tomato_safety_incompatible),
        "profile_target_incompatible": bool(profile_target_incompatible),
        "action_continuity_incompatible": bool(action_continuity_incompatible),
        "compatible": bool(compatible),
    }


def _classify_row(row: Mapping[str, Any]) -> dict[str, Any]:
    candidates = _parse_candidates(row)
    candidate_flags = [_candidate_flags(candidate) for candidate in candidates]
    candidate_available = bool(_int(row.get("profile_action_candidate_shadow_candidate_count", 0)) > 0 or candidates)
    contract_missing = bool(not candidates or any(flags["contract_missing"] for flags in candidate_flags))
    tomato_projection_missing = bool(not candidates or any(flags["tomato_projection_missing"] for flags in candidate_flags))
    fallback_source = bool(any(flags["fallback_source"] for flags in candidate_flags))
    tomato_safety_incompatible = bool(any(flags["tomato_safety_incompatible"] for flags in candidate_flags))
    profile_target_incompatible = bool(any(flags["profile_target_incompatible"] for flags in candidate_flags))
    action_continuity_incompatible = bool(any(flags["action_continuity_incompatible"] for flags in candidate_flags))
    action_diff = _row_action_changed(row)
    compatible_candidates = [flags for flags in candidate_flags if bool(flags["compatible"])]
    ready = bool(
        candidate_available
        and compatible_candidates
        and not contract_missing
        and not tomato_projection_missing
        and not fallback_source
        and not action_diff
    )
    categories: list[str] = []
    if not candidate_available:
        categories.append("profile_action_candidate_missing")
    if contract_missing:
        categories.append("contract_missing")
    if tomato_projection_missing:
        categories.append("tomato_projection_missing")
    if action_diff:
        categories.append("action_invariance_violation")
    if fallback_source:
        categories.append("fallback_candidate_source_violation")
    if tomato_safety_incompatible:
        categories.append("tomato_safety_incompatible")
    if profile_target_incompatible:
        categories.append("profile_target_incompatible")
    if action_continuity_incompatible:
        categories.append("action_continuity_incompatible")
    if ready:
        categories.append("compatibility_shadow_ready")
    return {
        "candidate_available": bool(candidate_available),
        "candidate_count": int(len(candidates)),
        "compatible_candidate_count": int(len(compatible_candidates)),
        "contract_missing": bool(contract_missing),
        "tomato_projection_missing": bool(tomato_projection_missing),
        "action_diff": bool(action_diff),
        "fallback_source": bool(fallback_source),
        "tomato_safety_incompatible": bool(tomato_safety_incompatible),
        "profile_target_incompatible": bool(profile_target_incompatible),
        "action_continuity_incompatible": bool(action_continuity_incompatible),
        "ready": bool(ready),
        "categories": categories,
    }


def _recommendation(summary: Mapping[str, Any]) -> dict[str, str]:
    if _int(summary.get("action_invariance_violation_steps")) > 0:
        return {
            "next_action": "stop_runtime_shadow_patch_action_invariance_violation",
            "reason": "v70 detected final-action change in shadow provenance.",
        }
    if _int(summary.get("trace_count")) <= 0 or _int(summary.get("v69_shadow_metadata_steps")) <= 0:
        return {
            "next_action": "needs_opt_in_shadow_trace_authorization",
            "reason": "No v69 opt-in profile action candidate shadow trace is available.",
        }
    if _int(summary.get("profile_action_candidate_available_steps")) <= 0:
        return {
            "next_action": "profile_action_candidate_shadow_instrumentation_repair_plan",
            "reason": "v69 shadow metadata exists but no profile_action candidates were recorded.",
        }
    if _int(summary.get("fallback_candidate_source_violation_steps")) > 0:
        return {
            "next_action": "profile_action_candidate_source_repair_plan",
            "reason": "Profile action shadow contains fallback candidate source.",
        }
    if _int(summary.get("contract_missing_steps")) > 0 or _int(summary.get("tomato_projection_missing_steps")) > 0:
        return {
            "next_action": "profile_action_candidate_shadow_instrumentation_repair_plan",
            "reason": "Profile action candidate contract or Tomato Safety projection provenance is incomplete.",
        }
    if _int(summary.get("tomato_safety_incompatible_steps")) > 0:
        return {
            "next_action": "profile_to_action_mapping_repair_or_action_envelope_design",
            "reason": "Profile actions are often substantially rewritten or rejected by safety logic.",
        }
    if _int(summary.get("profile_target_incompatible_steps")) > 0:
        return {
            "next_action": "profile_generator_or_composer_hypothesis_revision_plan",
            "reason": "Profile action candidates do not reliably align with profile targets.",
        }
    if _int(summary.get("action_continuity_incompatible_steps")) > 0:
        return {
            "next_action": "profile_action_candidate_continuity_repair_plan",
            "reason": "Profile action candidates exceed the shadow continuity threshold.",
        }
    if _int(summary.get("compatibility_shadow_ready_steps")) > 0:
        return {
            "next_action": "minimal_profile_candidate_compatibility_scorer_shadow_plan",
            "reason": "Compatible profile action candidates are present and action invariant is preserved.",
        }
    return {
        "next_action": "profile_action_candidate_compatibility_metadata_repair_plan",
        "reason": "Compatibility audit could not identify ready candidates.",
    }


def audit_trace(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    classified = [_classify_row(row) for row in rows]
    shadow_rows = [row for row in rows if _truthy(row.get("profile_action_candidate_shadow_enabled", False))]
    category_counts: dict[str, int] = {}
    for result in classified:
        for category in result["categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1
    summary = {
        **trace_identity(path),
        "rows": int(len(rows)),
        "v69_shadow_metadata_steps": int(len(shadow_rows)),
        "profile_action_candidate_available_steps": int(sum(item["candidate_available"] for item in classified)),
        "compatible_profile_action_candidate_steps": int(sum(item["compatible_candidate_count"] > 0 for item in classified)),
        "compatibility_shadow_ready_steps": int(sum(item["ready"] for item in classified)),
        "contract_missing_steps": int(sum(item["contract_missing"] for item in classified)),
        "tomato_projection_missing_steps": int(sum(item["tomato_projection_missing"] for item in classified)),
        "action_invariance_violation_steps": int(sum(item["action_diff"] for item in classified)),
        "fallback_candidate_source_violation_steps": int(sum(item["fallback_source"] for item in classified)),
        "tomato_safety_incompatible_steps": int(sum(item["tomato_safety_incompatible"] for item in classified)),
        "profile_target_incompatible_steps": int(sum(item["profile_target_incompatible"] for item in classified)),
        "action_continuity_incompatible_steps": int(sum(item["action_continuity_incompatible"] for item in classified)),
        "category_counts": dict(sorted(category_counts.items())),
        "warnings": [],
    }
    if not shadow_rows:
        summary["warnings"].append("missing_v69_profile_action_candidate_shadow_metadata")
    summary["recommendation"] = _recommendation({"trace_count": 1, **summary})
    return summary


def _aggregate(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "rows",
        "v69_shadow_metadata_steps",
        "profile_action_candidate_available_steps",
        "compatible_profile_action_candidate_steps",
        "compatibility_shadow_ready_steps",
        "contract_missing_steps",
        "tomato_projection_missing_steps",
        "action_invariance_violation_steps",
        "fallback_candidate_source_violation_steps",
        "tomato_safety_incompatible_steps",
        "profile_target_incompatible_steps",
        "action_continuity_incompatible_steps",
    )
    aggregate = {"trace_count": int(len(traces))}
    for key in keys:
        aggregate[key] = int(sum(_int(trace.get(key)) for trace in traces))
    category_counts: dict[str, int] = {}
    for trace in traces:
        counts = trace.get("category_counts", {})
        if not isinstance(counts, Mapping):
            continue
        for category, count in counts.items():
            category_counts[str(category)] = category_counts.get(str(category), 0) + _int(count)
    aggregate["category_counts"] = dict(sorted(category_counts.items()))
    aggregate["recommendation"] = _recommendation(aggregate)
    return aggregate


def build_authorization_plan(audit: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = audit.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        aggregate = {}
    reason = "No v69 opt-in profile action candidate trace is available."
    if _int(aggregate.get("trace_count")) > 0 and _int(aggregate.get("v69_shadow_metadata_steps")) <= 0:
        reason = "Trace files exist, but they lack v69 profile action candidate shadow metadata."
    plan = {
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "authorization_type": "opt_in_shadow_trace_or_cache_acquisition",
        "why_existing_cache_or_trace_is_insufficient": reason,
        "scenario_windows": list(FAILURE_SCENARIOS),
        "max_calls_or_budget": "user_authorized_small_failure_window_only",
        "cache_write_policy": "write_only_explicit_qwen37plus_shadow_cache_if_authorized",
        "default_controller_changed": False,
        "final_action_changed": False,
        "shadow_only": True,
        "required_provenance_fields": list(REQUIRED_V70_TRACE_FIELDS),
        "success_condition": "v69 opt-in trace records profile_action candidates with final action invariant.",
        "stop_condition": "action diff, hard-safety regression, runtime collapse, cache miss without authorization, or missing required provenance.",
    }
    return _with_boundaries(plan)


def build_hypothesis_check(audit: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = audit.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        aggregate = {}
    recommendation = _recommendation(aggregate)
    if recommendation["next_action"] == "minimal_profile_candidate_compatibility_scorer_shadow_plan":
        status = "current_profile_driven_hypothesis_supported_for_shadow_scorer"
        alternative = ""
    elif recommendation["next_action"] == "profile_to_action_mapping_repair_or_action_envelope_design":
        status = "profile_to_action_direct_mapping_hypothesis_needs_revision"
        alternative = "action_envelope_or_safety_projected_mapping"
    elif recommendation["next_action"] == "profile_generator_or_composer_hypothesis_revision_plan":
        status = "profile_target_to_action_alignment_hypothesis_needs_revision"
        alternative = "profile_generator_or_composer_target_alignment_repair"
    elif recommendation["next_action"] == "needs_opt_in_shadow_trace_authorization":
        status = "insufficient_evidence_for_profile_action_hypothesis"
        alternative = "opt_in_shadow_trace_or_cache_acquisition"
    else:
        status = "profile_action_candidate_hypothesis_blocked_by_metadata_or_invariance"
        alternative = recommendation["next_action"]
    check = {
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "current_hypothesis": "profile_generator trajectory can be composed into normal-path profile_action candidates before scorer/arbitrator.",
        "observed_gap": recommendation["reason"],
        "hypothesis_status": status,
        "alternative_hypothesis": alternative,
        "minimal_experiment": recommendation["next_action"],
        "required_trace_fields": list(REQUIRED_V70_TRACE_FIELDS),
        "online_llm_needed": bool(recommendation["next_action"] == "needs_opt_in_shadow_trace_authorization"),
        "expected_artifacts": [
            str(AUDIT_JSON.name),
            str(HYPOTHESIS_JSON.name),
            str(AUTHORIZATION_JSON.name) if recommendation["next_action"] == "needs_opt_in_shadow_trace_authorization" else "",
        ],
        "success_condition": "Profile action candidates are compatible in shadow and action invariant is preserved.",
        "stop_condition": "Final action diff, missing contract, missing Tomato Safety projection, or hard-safety conflict.",
        "next_action": recommendation["next_action"],
    }
    check["expected_artifacts"] = [item for item in check["expected_artifacts"] if item]
    return _with_boundaries(check)


def audit_traces(inputs: Sequence[str | Path]) -> dict[str, Any]:
    paths = discover_traces(inputs) if inputs else []
    traces = [audit_trace(path) for path in paths]
    aggregate = _aggregate(traces)
    audit = {
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "trace_count": int(len(traces)),
        "compatibility_thresholds": {
            "tomato_rewrite_max_delta": TOMATO_REWRITE_MAX_DELTA,
            "profile_target_error_max": PROFILE_TARGET_ERROR_MAX,
            "action_delta_penalty_max": ACTION_DELTA_PENALTY_MAX,
        },
        "traces": traces,
        "aggregate": aggregate,
    }
    audit["hypothesis_check"] = build_hypothesis_check(audit)
    if aggregate["recommendation"]["next_action"] == "needs_opt_in_shadow_trace_authorization":
        audit["authorization_plan"] = build_authorization_plan(audit)
    return _with_boundaries(audit)


def build_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {})
    hypothesis = audit.get("hypothesis_check", {})
    if not isinstance(aggregate, Mapping):
        aggregate = {}
    if not isinstance(hypothesis, Mapping):
        hypothesis = {}
    lines = [
        "# qwen3.7-plus v70 Profile Action Candidate Compatibility Shadow Audit",
        "",
        "## Boundary",
        "",
        "- online_llm_called=false",
        "- new_rollout_run=false",
        "- default_llm_rspc_v2_changed=false",
        "- performance_claim_allowed=false",
        "- promotion_evidence=false",
        "",
        "## Aggregate",
        "",
        f"- trace_count={_int(audit.get('trace_count'))}",
        f"- rows={_int(aggregate.get('rows'))}",
        f"- v69_shadow_metadata_steps={_int(aggregate.get('v69_shadow_metadata_steps'))}",
        f"- profile_action_candidate_available_steps={_int(aggregate.get('profile_action_candidate_available_steps'))}",
        f"- compatibility_shadow_ready_steps={_int(aggregate.get('compatibility_shadow_ready_steps'))}",
        f"- action_invariance_violation_steps={_int(aggregate.get('action_invariance_violation_steps'))}",
        f"- tomato_safety_incompatible_steps={_int(aggregate.get('tomato_safety_incompatible_steps'))}",
        f"- profile_target_incompatible_steps={_int(aggregate.get('profile_target_incompatible_steps'))}",
        f"- action_continuity_incompatible_steps={_int(aggregate.get('action_continuity_incompatible_steps'))}",
        "",
        "## Hypothesis Check",
        "",
        f"- hypothesis_status={hypothesis.get('hypothesis_status', '')}",
        f"- next_action={hypothesis.get('next_action', '')}",
    ]
    counts = aggregate.get("category_counts", {})
    if isinstance(counts, Mapping) and counts:
        lines.extend(["", "## Category Counts", ""])
        lines.extend(f"- {key}: {value}" for key, value in sorted(counts.items()))
    return "\n".join(lines) + "\n"


def build_hypothesis_report(hypothesis: Mapping[str, Any]) -> str:
    lines = [
        "# qwen3.7-plus v70 Profile Action Candidate Hypothesis Check",
        "",
        f"- current_hypothesis={hypothesis.get('current_hypothesis', '')}",
        f"- observed_gap={hypothesis.get('observed_gap', '')}",
        f"- hypothesis_status={hypothesis.get('hypothesis_status', '')}",
        f"- alternative_hypothesis={hypothesis.get('alternative_hypothesis', '')}",
        f"- minimal_experiment={hypothesis.get('minimal_experiment', '')}",
        f"- online_llm_needed={bool(hypothesis.get('online_llm_needed', False))}",
        f"- next_action={hypothesis.get('next_action', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_authorization_report(plan: Mapping[str, Any]) -> str:
    lines = [
        "# qwen3.7-plus v70 Opt-In Shadow Trace Authorization Plan",
        "",
        f"- authorization_type={plan.get('authorization_type', '')}",
        f"- why_existing_cache_or_trace_is_insufficient={plan.get('why_existing_cache_or_trace_is_insufficient', '')}",
        f"- model={plan.get('model', '')}",
        f"- scenario_windows={','.join(str(item) for item in plan.get('scenario_windows', []) or [])}",
        f"- default_controller_changed={bool(plan.get('default_controller_changed', False))}",
        f"- final_action_changed={bool(plan.get('final_action_changed', False))}",
        f"- shadow_only={bool(plan.get('shadow_only', False))}",
        f"- success_condition={plan.get('success_condition', '')}",
        f"- stop_condition={plan.get('stop_condition', '')}",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(
    audit: Mapping[str, Any],
    *,
    audit_json: str | Path = AUDIT_JSON,
    audit_md: str | Path = AUDIT_MD,
    hypothesis_json: str | Path = HYPOTHESIS_JSON,
    hypothesis_md: str | Path = HYPOTHESIS_MD,
    authorization_json: str | Path = AUTHORIZATION_JSON,
    authorization_md: str | Path = AUTHORIZATION_MD,
) -> dict[str, str]:
    audit_json = Path(audit_json)
    audit_md = Path(audit_md)
    hypothesis_json = Path(hypothesis_json)
    hypothesis_md = Path(hypothesis_md)
    for path in (audit_json, audit_md, hypothesis_json, hypothesis_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    audit_md.write_text(build_report(audit), encoding="utf-8")
    hypothesis = audit.get("hypothesis_check", {})
    hypothesis_json.write_text(json.dumps(hypothesis, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    hypothesis_md.write_text(build_hypothesis_report(hypothesis if isinstance(hypothesis, Mapping) else {}), encoding="utf-8")
    written = {
        "audit_json": str(audit_json),
        "audit_md": str(audit_md),
        "hypothesis_json": str(hypothesis_json),
        "hypothesis_md": str(hypothesis_md),
    }
    authorization = audit.get("authorization_plan", {})
    if isinstance(authorization, Mapping) and authorization:
        authorization_json = Path(authorization_json)
        authorization_md = Path(authorization_md)
        authorization_json.parent.mkdir(parents=True, exist_ok=True)
        authorization_json.write_text(
            json.dumps(authorization, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        authorization_md.write_text(build_authorization_report(authorization), encoding="utf-8")
        written["authorization_json"] = str(authorization_json)
        written["authorization_md"] = str(authorization_md)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="v69 opt-in shadow trace CSV/JSONL files or directories.")
    parser.add_argument("--no-write", action="store_true", help="Print audit JSON without writing artifacts.")
    args = parser.parse_args(argv)

    audit = audit_traces(args.inputs)
    if args.no_write:
        print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
        return 0
    print(json.dumps(write_artifacts(audit), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
