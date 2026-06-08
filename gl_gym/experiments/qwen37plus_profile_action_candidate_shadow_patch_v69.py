"""Audit v69 profile-action candidate shadow instrumentation.

This audit is instrumentation-only. It checks whether opt-in runtime traces
contain contract-shaped normal-path profile action candidates while preserving
final-action invariance. It does not run rollout, call online LLMs, change the
default controller, authorize replay, or make performance claims.
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
ARTIFACT_DATE = "20260603"
VERSION = "v69"

AUDIT_JSON = AUDIT_DIR / "qwen37plus_profile_action_candidate_shadow_patch_audit_20260603_v69.json"
AUDIT_MD = AUDIT_DIR / "qwen37plus_profile_action_candidate_shadow_patch_audit_20260603_v69.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260603_v69.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260603_v69.md"

ACTION_FIELDS = ("heat", "co2", "screen", "vent", "lamp", "shade")
CONTRACT_REQUIRED_OUTPUT_FIELDS = (
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
    missing = [field for field in CONTRACT_REQUIRED_OUTPUT_FIELDS if field not in candidate]
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


def _row_action_changed(row: Mapping[str, Any]) -> bool:
    return any(
        _truthy(row.get(key, False))
        for key in (
            "profile_action_candidate_shadow_final_action_changed",
            "profile_action_candidate_shadow_action_diff_nonzero",
            "v69_action_invariant_violation",
        )
    )


def _row_has_candidate(row: Mapping[str, Any]) -> bool:
    return _int(row.get("profile_action_candidate_shadow_candidate_count", 0)) > 0 or bool(_parse_candidates(row))


def _row_contract_status(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    candidates = _parse_candidates(row)
    if not candidates:
        return False, ["missing_candidates_json"]
    missing: list[str] = []
    for candidate in candidates:
        missing.extend(_candidate_missing_fields(candidate))
    return len(missing) == 0, sorted(dict.fromkeys(missing))


def _row_tomato_provenance_present(row: Mapping[str, Any]) -> bool:
    candidates = _parse_candidates(row)
    if not candidates:
        return False
    return all(
        "tomato_safety_v2_applied" in candidate
        and "post_tomato_action" in candidate
        and _action_complete(candidate.get("post_tomato_action", {}))
        for candidate in candidates
    )


def _recommendation(summary: Mapping[str, Any]) -> dict[str, str]:
    if _int(summary.get("action_diff_steps")) > 0:
        return {
            "next_action": "stop_runtime_shadow_patch_action_invariance_violation",
            "reason": "v69 shadow provenance indicates final action changed.",
        }
    candidate_steps = _int(summary.get("normal_path_profile_action_candidate_steps"))
    if candidate_steps <= 0:
        return {
            "next_action": "profile_action_candidate_shadow_instrumentation_repair_plan",
            "reason": "No normal-path profile action candidates were recorded.",
        }
    if _int(summary.get("contract_missing_steps")) > 0 or _int(summary.get("tomato_safety_provenance_missing_steps")) > 0:
        return {
            "next_action": "profile_action_candidate_shadow_instrumentation_repair_plan",
            "reason": "Candidate contract or Tomato Safety provenance is incomplete.",
        }
    return {
        "next_action": "normal_path_profile_candidate_compatibility_shadow_audit_plan",
        "reason": "Contract-shaped profile action candidates are present and action invariant is preserved.",
    }


def audit_trace(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    rows = read_trace(path)
    shadow_rows = [row for row in rows if _truthy(row.get("profile_action_candidate_shadow_enabled", False))]
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
    tomato_missing_rows = [row for row in candidate_rows if not _row_tomato_provenance_present(row)]
    fallback_source_rows = []
    for row in candidate_rows:
        for candidate in _parse_candidates(row):
            if str(candidate.get("candidate_source", "")) == "fallback_candidate":
                fallback_source_rows.append(row)
                break
    summary: dict[str, Any] = {
        **trace_identity(path),
        "rows": int(len(rows)),
        "profile_action_candidate_shadow_steps": int(len(shadow_rows)),
        "normal_path_profile_action_candidate_steps": int(len(candidate_rows)),
        "eligible_profile_action_candidate_steps": int(
            sum(_int(row.get("profile_action_candidate_shadow_eligible_candidate_count", 0)) > 0 for row in candidate_rows)
        ),
        "contract_complete_steps": int(len(candidate_rows) - len(contract_missing_rows)),
        "contract_missing_steps": int(len(contract_missing_rows)),
        "contract_missing_field_counts": dict(sorted(missing_field_counts.items())),
        "tomato_safety_provenance_present_steps": int(len(candidate_rows) - len(tomato_missing_rows)),
        "tomato_safety_provenance_missing_steps": int(len(tomato_missing_rows)),
        "action_diff_steps": int(sum(_row_action_changed(row) for row in rows)),
        "fallback_candidate_source_steps": int(len(fallback_source_rows)),
        "warnings": [],
    }
    if not shadow_rows:
        summary["warnings"].append("missing_profile_action_candidate_shadow_metadata")
    if candidate_rows and fallback_source_rows:
        summary["warnings"].append("fallback_candidate_source_inside_profile_action_shadow")
    summary["recommendation"] = _recommendation(summary)
    return summary


def _aggregate_trace_summaries(traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = {
        "trace_count": int(len(traces)),
        "rows": int(sum(_int(trace.get("rows")) for trace in traces)),
        "profile_action_candidate_shadow_steps": int(
            sum(_int(trace.get("profile_action_candidate_shadow_steps")) for trace in traces)
        ),
        "normal_path_profile_action_candidate_steps": int(
            sum(_int(trace.get("normal_path_profile_action_candidate_steps")) for trace in traces)
        ),
        "eligible_profile_action_candidate_steps": int(
            sum(_int(trace.get("eligible_profile_action_candidate_steps")) for trace in traces)
        ),
        "contract_complete_steps": int(sum(_int(trace.get("contract_complete_steps")) for trace in traces)),
        "contract_missing_steps": int(sum(_int(trace.get("contract_missing_steps")) for trace in traces)),
        "tomato_safety_provenance_present_steps": int(
            sum(_int(trace.get("tomato_safety_provenance_present_steps")) for trace in traces)
        ),
        "tomato_safety_provenance_missing_steps": int(
            sum(_int(trace.get("tomato_safety_provenance_missing_steps")) for trace in traces)
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
        "normal_path_profile_action_candidate_steps": _int(
            aggregate.get("normal_path_profile_action_candidate_steps")
        ),
        "contract_complete": bool(
            _int(aggregate.get("normal_path_profile_action_candidate_steps")) > 0
            and _int(aggregate.get("contract_missing_steps")) == 0
        ),
        "final_action_invariant": bool(_int(aggregate.get("action_diff_steps")) == 0),
        "tomato_safety_provenance_present": bool(
            _int(aggregate.get("normal_path_profile_action_candidate_steps")) > 0
            and _int(aggregate.get("tomato_safety_provenance_missing_steps")) == 0
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
        "# qwen3.7-plus v69 Profile Action Candidate Shadow Patch Audit",
        "",
        "## Boundary",
        "",
        "- controlled_replay_allowed=false",
        "- controlled_replay_execution_allowed=false",
        "- metadata_replay_execution_allowed=false",
        "- performance_claim_allowed=false",
        "- promotion_evidence=false",
        "- online_llm_called=false",
        "",
        "## Aggregate",
        "",
        f"- trace_count={_int(audit.get('trace_count'))}",
        f"- rows={_int(aggregate.get('rows'))}",
        f"- normal_path_profile_action_candidate_steps={_int(aggregate.get('normal_path_profile_action_candidate_steps'))}",
        f"- contract_missing_steps={_int(aggregate.get('contract_missing_steps'))}",
        f"- tomato_safety_provenance_missing_steps={_int(aggregate.get('tomato_safety_provenance_missing_steps'))}",
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
        "# qwen3.7-plus v69 Metadata Replay Readiness Checklist",
        "",
        "- controlled_replay_allowed=false",
        "- controlled_replay_execution_allowed=false",
        "- metadata_replay_execution_allowed=false",
        "- performance_claim_allowed=false",
        "- promotion_evidence=false",
        "- online_llm_called=false",
        "",
        f"- normal_path_profile_action_candidate_steps={_int(readiness.get('normal_path_profile_action_candidate_steps'))}",
        f"- contract_complete={bool(readiness.get('contract_complete', False))}",
        f"- final_action_invariant={bool(readiness.get('final_action_invariant', False))}",
        f"- tomato_safety_provenance_present={bool(readiness.get('tomato_safety_provenance_present', False))}",
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
    parser.add_argument("inputs", nargs="*", help="Opt-in v69 shadow trace CSV files or directories.")
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
