"""Audit v74 profile-action envelope shadow instrumentation.

This audit is shadow-only and trace-only. It verifies whether opt-in runtime
traces contain v73-shaped profile action envelope provenance while preserving
final-action invariance. It does not run rollout, call online LLMs, authorize
controlled replay, enhance fallback, promote behavior, or make performance
claims.
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
VERSION = "v74"

AUDIT_JSON = AUDIT_DIR / "qwen37plus_profile_action_envelope_shadow_patch_audit_20260604_v74.json"
AUDIT_MD = AUDIT_DIR / "qwen37plus_profile_action_envelope_shadow_patch_audit_20260604_v74.md"
READINESS_JSON = AUDIT_DIR / "profile_action_envelope_shadow_readiness_20260604_v74.json"
READINESS_MD = AUDIT_DIR / "profile_action_envelope_shadow_readiness_20260604_v74.md"

ACTION_FIELDS = ("heat", "co2", "screen", "vent", "lamp", "shade")
CONTRACT_REQUIRED_OUTPUT_FIELDS = (
    "name",
    "profile_name",
    "candidate_source",
    "intent",
    "target_direction",
    "action_bounds",
    "preferred_direction",
    "priority_terms",
    "continuity_constraints",
    "tomato_safety_projection",
    "projected_action",
    "score_terms",
    "eligible",
    "rejection_reason",
    "compatibility_category",
)
SCORE_TERM_FIELDS = (
    "profile_target_alignment",
    "tomato_projection_delta",
    "continuity_penalty",
    "envelope_width_penalty",
    "compatibility_penalty",
    "selection_score",
)
TOMATO_PROJECTION_FIELDS = (
    "projection_required",
    "projected_action",
    "projection_applied",
    "projection_reasons",
    "rewrite_delta_by_field",
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
    "final_action_changed",
)


def _with_boundaries(payload: dict[str, Any]) -> dict[str, Any]:
    for field in BOUNDARY_FALSE_FIELDS:
        payload[field] = False
    payload["rollout_command_generated"] = False
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


def _parse_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = row.get("profile_action_envelope_shadow_candidates_json", [])
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


def _action_bounds_complete(bounds: Any) -> bool:
    if not isinstance(bounds, Mapping):
        return False
    for field in ACTION_FIELDS:
        bound = bounds.get(field)
        if not isinstance(bound, Mapping):
            return False
        for key in ("min", "max"):
            if key not in bound:
                return False
            try:
                float(bound.get(key))
            except Exception:
                return False
    return True


def _preferred_direction_complete(directions: Any) -> bool:
    if not isinstance(directions, Mapping):
        return False
    allowed = {"increase", "decrease", "hold", "any"}
    return all(str(directions.get(field, "")) in allowed for field in ACTION_FIELDS)


def _tomato_projection_missing_fields(candidate: Mapping[str, Any]) -> list[str]:
    projection = candidate.get("tomato_safety_projection", {})
    if not isinstance(projection, Mapping):
        return ["tomato_safety_projection:mapping"]
    missing = [f"tomato_safety_projection:{field}" for field in TOMATO_PROJECTION_FIELDS if field not in projection]
    if not _action_complete(projection.get("projected_action", {})):
        missing.append("tomato_safety_projection:projected_action_fields")
    rewrite_delta = projection.get("rewrite_delta_by_field", {})
    if not isinstance(rewrite_delta, Mapping):
        missing.append("tomato_safety_projection:rewrite_delta_mapping")
    else:
        missing.extend(f"tomato_safety_projection:rewrite_delta:{field}" for field in ACTION_FIELDS if field not in rewrite_delta)
    return sorted(dict.fromkeys(missing))


def _candidate_missing_fields(candidate: Mapping[str, Any]) -> list[str]:
    missing = [field for field in CONTRACT_REQUIRED_OUTPUT_FIELDS if field not in candidate]
    if not str(candidate.get("name", "")).startswith("profile_action_envelope:"):
        missing.append("name:profile_action_envelope_prefix")
    if str(candidate.get("candidate_source", "")) != "normal_path_profile_action_envelope":
        missing.append("candidate_source:normal_path_profile_action_envelope")
    if not _action_complete(candidate.get("projected_action", {})):
        missing.append("projected_action:action_fields")
    if not _action_bounds_complete(candidate.get("action_bounds", {})):
        missing.append("action_bounds:min_max_action_fields")
    if not _preferred_direction_complete(candidate.get("preferred_direction", {})):
        missing.append("preferred_direction:action_fields")
    if not isinstance(candidate.get("target_direction", {}), Mapping):
        missing.append("target_direction:mapping")
    if not isinstance(candidate.get("priority_terms", {}), Mapping):
        missing.append("priority_terms:mapping")
    if not isinstance(candidate.get("continuity_constraints", {}), Mapping):
        missing.append("continuity_constraints:mapping")
    score_terms = candidate.get("score_terms", {})
    if not isinstance(score_terms, Mapping):
        missing.append("score_terms:mapping")
    else:
        missing.extend(f"score_terms:{field}" for field in SCORE_TERM_FIELDS if field not in score_terms)
    missing.extend(_tomato_projection_missing_fields(candidate))
    return sorted(dict.fromkeys(missing))


def _row_action_changed(row: Mapping[str, Any]) -> bool:
    return any(
        _truthy(row.get(key, False))
        for key in (
            "profile_action_envelope_shadow_final_action_changed",
            "profile_action_envelope_shadow_action_diff_nonzero",
            "v74_action_invariant_violation",
        )
    )


def _row_has_candidate(row: Mapping[str, Any]) -> bool:
    return _int(row.get("profile_action_envelope_shadow_candidate_count", 0)) > 0 or bool(_parse_candidates(row))


def _row_contract_status(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    candidates = _parse_candidates(row)
    if not candidates:
        return False, ["missing_candidates_json"]
    missing: list[str] = []
    for candidate in candidates:
        missing.extend(_candidate_missing_fields(candidate))
    return len(missing) == 0, sorted(dict.fromkeys(missing))


def _row_tomato_projection_present(row: Mapping[str, Any]) -> bool:
    candidates = _parse_candidates(row)
    if not candidates:
        return False
    return all(not _tomato_projection_missing_fields(candidate) for candidate in candidates)


def _recommendation(summary: Mapping[str, Any]) -> dict[str, str]:
    if _int(summary.get("action_diff_steps")) > 0:
        return {
            "next_action": "stop_profile_action_envelope_shadow_action_invariance_violation",
            "reason": "v74 envelope shadow provenance indicates final action changed.",
        }
    if _int(summary.get("trace_count")) <= 0 or _int(summary.get("profile_action_envelope_shadow_steps")) <= 0:
        return {
            "next_action": "needs_profile_action_envelope_opt_in_shadow_trace_acquisition",
            "reason": "No v74 opt-in envelope shadow trace metadata was found.",
        }
    candidate_steps = _int(summary.get("normal_path_profile_action_envelope_steps"))
    if candidate_steps <= 0:
        return {
            "next_action": "profile_action_envelope_shadow_instrumentation_repair_plan",
            "reason": "Envelope shadow metadata exists but no candidates were recorded.",
        }
    if _int(summary.get("fallback_candidate_source_steps")) > 0:
        return {
            "next_action": "profile_action_envelope_candidate_source_repair_plan",
            "reason": "Envelope shadow candidates must not come from fallback_candidate.",
        }
    if _int(summary.get("contract_missing_steps")) > 0 or _int(summary.get("tomato_safety_projection_missing_steps")) > 0:
        return {
            "next_action": "profile_action_envelope_contract_or_projection_repair_plan",
            "reason": "Envelope contract or Tomato Safety projection provenance is incomplete.",
        }
    return {
        "next_action": "profile_action_envelope_compatibility_shadow_audit_plan",
        "reason": "Contract-shaped envelope candidates are present and final action invariance is preserved.",
    }


def audit_trace(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    shadow_rows = [row for row in rows if _truthy(row.get("profile_action_envelope_shadow_enabled", False))]
    candidate_rows = [row for row in rows if _row_has_candidate(row)]
    contract_missing_rows = []
    missing_field_counts: dict[str, int] = {}
    for row in candidate_rows:
        complete, missing = _row_contract_status(row)
        if complete:
            continue
        contract_missing_rows.append(row)
        for field in missing:
            missing_field_counts[field] = missing_field_counts.get(field, 0) + 1
    tomato_missing_rows = [row for row in candidate_rows if not _row_tomato_projection_present(row)]
    fallback_source_rows = []
    for row in candidate_rows:
        for candidate in _parse_candidates(row):
            if str(candidate.get("candidate_source", "")) == "fallback_candidate":
                fallback_source_rows.append(row)
                break
    summary: dict[str, Any] = {
        **trace_identity(path),
        "rows": int(len(rows)),
        "profile_action_envelope_shadow_steps": int(len(shadow_rows)),
        "normal_path_profile_action_envelope_steps": int(len(candidate_rows)),
        "eligible_profile_action_envelope_steps": int(
            sum(_int(row.get("profile_action_envelope_shadow_eligible_candidate_count", 0)) > 0 for row in candidate_rows)
        ),
        "contract_complete_steps": int(len(candidate_rows) - len(contract_missing_rows)),
        "contract_missing_steps": int(len(contract_missing_rows)),
        "contract_missing_field_counts": dict(sorted(missing_field_counts.items())),
        "tomato_safety_projection_present_steps": int(len(candidate_rows) - len(tomato_missing_rows)),
        "tomato_safety_projection_missing_steps": int(len(tomato_missing_rows)),
        "action_diff_steps": int(sum(_row_action_changed(row) for row in rows)),
        "fallback_candidate_source_steps": int(len(fallback_source_rows)),
        "warnings": [],
    }
    if not shadow_rows:
        summary["warnings"].append("missing_profile_action_envelope_shadow_metadata")
    if candidate_rows and fallback_source_rows:
        summary["warnings"].append("fallback_candidate_source_inside_profile_action_envelope_shadow")
    summary["recommendation"] = _recommendation({**summary, "trace_count": 1})
    return summary


def _aggregate_trace_summaries(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = {
        "trace_count": int(len(traces)),
        "rows": int(sum(_int(trace.get("rows")) for trace in traces)),
        "profile_action_envelope_shadow_steps": int(
            sum(_int(trace.get("profile_action_envelope_shadow_steps")) for trace in traces)
        ),
        "normal_path_profile_action_envelope_steps": int(
            sum(_int(trace.get("normal_path_profile_action_envelope_steps")) for trace in traces)
        ),
        "eligible_profile_action_envelope_steps": int(
            sum(_int(trace.get("eligible_profile_action_envelope_steps")) for trace in traces)
        ),
        "contract_complete_steps": int(sum(_int(trace.get("contract_complete_steps")) for trace in traces)),
        "contract_missing_steps": int(sum(_int(trace.get("contract_missing_steps")) for trace in traces)),
        "tomato_safety_projection_present_steps": int(
            sum(_int(trace.get("tomato_safety_projection_present_steps")) for trace in traces)
        ),
        "tomato_safety_projection_missing_steps": int(
            sum(_int(trace.get("tomato_safety_projection_missing_steps")) for trace in traces)
        ),
        "action_diff_steps": int(sum(_int(trace.get("action_diff_steps")) for trace in traces)),
        "fallback_candidate_source_steps": int(sum(_int(trace.get("fallback_candidate_source_steps")) for trace in traces)),
        "contract_missing_field_counts": {},
    }
    field_counts: dict[str, int] = {}
    for trace in traces:
        counts = trace.get("contract_missing_field_counts", {})
        if not isinstance(counts, Mapping):
            continue
        for field, count in counts.items():
            field_counts[str(field)] = field_counts.get(str(field), 0) + _int(count)
    aggregate["contract_missing_field_counts"] = dict(sorted(field_counts.items()))
    aggregate["recommendation"] = _recommendation(aggregate)
    return aggregate


def build_readiness(audit: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = audit.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        aggregate = {}
    recommendation = _recommendation(aggregate)
    readiness = {
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "contract_required_output_fields": list(CONTRACT_REQUIRED_OUTPUT_FIELDS),
        "score_term_fields": list(SCORE_TERM_FIELDS),
        "action_fields": list(ACTION_FIELDS),
        "normal_path_profile_action_envelope_steps": _int(
            aggregate.get("normal_path_profile_action_envelope_steps")
        ),
        "contract_complete": bool(
            _int(aggregate.get("normal_path_profile_action_envelope_steps")) > 0
            and _int(aggregate.get("contract_missing_steps")) == 0
        ),
        "final_action_invariant": bool(_int(aggregate.get("action_diff_steps")) == 0),
        "tomato_safety_projection_present": bool(
            _int(aggregate.get("normal_path_profile_action_envelope_steps")) > 0
            and _int(aggregate.get("tomato_safety_projection_missing_steps")) == 0
        ),
        "fallback_not_enhanced": bool(_int(aggregate.get("fallback_candidate_source_steps")) == 0),
        "next_action": recommendation["next_action"],
        "reason": recommendation["reason"],
    }
    return _with_boundaries(readiness)


def audit_traces(inputs: Sequence[str | Path]) -> dict[str, Any]:
    paths = discover_traces(inputs) if inputs else []
    traces = [audit_trace(path) for path in paths]
    aggregate = _aggregate_trace_summaries(traces)
    audit = {
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "trace_count": int(len(traces)),
        "traces": traces,
        "aggregate": aggregate,
    }
    audit["readiness"] = build_readiness(audit)
    return _with_boundaries(audit)


def build_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {})
    readiness = audit.get("readiness", {})
    if not isinstance(aggregate, Mapping):
        aggregate = {}
    if not isinstance(readiness, Mapping):
        readiness = {}
    lines = [
        "# qwen3.7-plus v74 Profile Action Envelope Shadow Patch Audit",
        "",
        "## Boundary",
        "",
        "- controlled_replay_allowed=false",
        "- controlled_replay_execution_allowed=false",
        "- metadata_replay_execution_allowed=false",
        "- performance_claim_allowed=false",
        "- promotion_evidence=false",
        "- online_llm_called=false",
        "- rollout_command_generated=false",
        "",
        "## Aggregate",
        "",
        f"- trace_count={_int(audit.get('trace_count'))}",
        f"- rows={_int(aggregate.get('rows'))}",
        f"- normal_path_profile_action_envelope_steps={_int(aggregate.get('normal_path_profile_action_envelope_steps'))}",
        f"- contract_missing_steps={_int(aggregate.get('contract_missing_steps'))}",
        f"- tomato_safety_projection_missing_steps={_int(aggregate.get('tomato_safety_projection_missing_steps'))}",
        f"- action_diff_steps={_int(aggregate.get('action_diff_steps'))}",
        f"- fallback_candidate_source_steps={_int(aggregate.get('fallback_candidate_source_steps'))}",
        "",
        "## Readiness",
        "",
        f"- next_action={readiness.get('next_action', '')}",
        f"- reason={readiness.get('reason', '')}",
    ]
    missing_counts = aggregate.get("contract_missing_field_counts", {})
    if isinstance(missing_counts, Mapping) and missing_counts:
        lines.extend(["", "## Missing Contract Fields", ""])
        lines.extend(f"- {field}: {count}" for field, count in sorted(missing_counts.items()))
    return "\n".join(lines) + "\n"


def build_readiness_report(readiness: Mapping[str, Any]) -> str:
    lines = [
        "# qwen3.7-plus v74 Profile Action Envelope Shadow Readiness",
        "",
        "- controlled_replay_allowed=false",
        "- controlled_replay_execution_allowed=false",
        "- metadata_replay_execution_allowed=false",
        "- performance_claim_allowed=false",
        "- promotion_evidence=false",
        "- online_llm_called=false",
        "- rollout_command_generated=false",
        "",
        f"- normal_path_profile_action_envelope_steps={_int(readiness.get('normal_path_profile_action_envelope_steps'))}",
        f"- contract_complete={bool(readiness.get('contract_complete', False))}",
        f"- final_action_invariant={bool(readiness.get('final_action_invariant', False))}",
        f"- tomato_safety_projection_present={bool(readiness.get('tomato_safety_projection_present', False))}",
        f"- fallback_not_enhanced={bool(readiness.get('fallback_not_enhanced', False))}",
        f"- next_action={readiness.get('next_action', '')}",
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(
    audit: Mapping[str, Any],
    *,
    audit_json: str | Path = AUDIT_JSON,
    audit_md: str | Path = AUDIT_MD,
    readiness_json: str | Path = READINESS_JSON,
    readiness_md: str | Path = READINESS_MD,
) -> dict[str, str]:
    audit_json = Path(audit_json)
    audit_md = Path(audit_md)
    readiness_json = Path(readiness_json)
    readiness_md = Path(readiness_md)
    readiness = audit.get("readiness", {})
    for path in (audit_json, audit_md, readiness_json, readiness_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    audit_md.write_text(build_report(audit), encoding="utf-8")
    readiness_json.write_text(json.dumps(readiness, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    readiness_md.write_text(build_readiness_report(readiness if isinstance(readiness, Mapping) else {}), encoding="utf-8")
    return {
        "audit_json": str(audit_json),
        "audit_md": str(audit_md),
        "readiness_json": str(readiness_json),
        "readiness_md": str(readiness_md),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="Opt-in v74 shadow trace CSV files or directories.")
    parser.add_argument("--no-write", action="store_true", help="Print audit JSON without writing artifacts.")
    args = parser.parse_args(argv)

    audit = audit_traces(args.inputs)
    if args.no_write:
        print(json.dumps(audit, indent=2, ensure_ascii=False, default=str))
        return 0
    written = write_artifacts(audit)
    print(json.dumps(written, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
