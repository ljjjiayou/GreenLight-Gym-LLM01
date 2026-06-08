"""Build v73 profile-to-action envelope design artifacts.

This stage is design-only. It consumes v70/v72 profile-action evidence and
defines an action-envelope contract for a future shadow patch. It does not run
rollout, call online LLMs, change llm_rspc_v2, modify runtime control logic,
authorize replay, enhance fallback, or make performance claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments import qwen37plus_profile_action_candidate_compatibility_shadow_audit_v70 as v70  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import AUDIT_DIR  # noqa: E402


MODEL_NAME = "qwen3.7-plus"
ARTIFACT_DATE = "20260604"
VERSION = "v73"

V70_AUDIT_JSON = v70.AUDIT_JSON
V70_HYPOTHESIS_JSON = v70.HYPOTHESIS_JSON
V72_EXECUTION_MD = (
    AUDIT_DIR / "qwen37plus_profile_action_candidate_online_shadow_acquisition_execution_record_20260604_v72.md"
)

DESIGN_JSON = AUDIT_DIR / "qwen37plus_profile_to_action_envelope_design_20260604_v73.json"
DESIGN_MD = AUDIT_DIR / "qwen37plus_profile_to_action_envelope_design_20260604_v73.md"
CONTRACT_JSON = AUDIT_DIR / "profile_action_envelope_contract_20260604_v73.json"
CONTRACT_MD = AUDIT_DIR / "profile_action_envelope_contract_20260604_v73.md"
READINESS_JSON = AUDIT_DIR / "profile_to_action_mapping_repair_readiness_20260604_v73.json"
READINESS_MD = AUDIT_DIR / "profile_to_action_mapping_repair_readiness_20260604_v73.md"

ACTION_FIELDS = ("heat", "co2", "screen", "vent", "lamp", "shade")
ACTION_TRACE_FIELD_MAP = {
    "heat": "u_heating",
    "co2": "u_co2",
    "screen": "u_screen",
    "vent": "u_ventilation",
    "lamp": "u_lighting",
    "shade": "u_shading",
}
PROFILE_TARGET_FIELDS = ("target_temp", "target_co2", "target_rh")
DIRECTION_VALUES = ("increase", "decrease", "hold", "any")
COMPATIBILITY_CATEGORIES = (
    "compatibility_shadow_ready",
    "tomato_safety_incompatible",
    "profile_target_incompatible",
    "action_continuity_incompatible",
    "contract_missing",
    "tomato_projection_missing",
    "action_invariance_violation",
)
ENVELOPE_REQUIRED_OUTPUT_FIELDS = (
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


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _rel(path: str | Path) -> str:
    p = _resolve(path)
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_text(path: str | Path) -> str:
    p = _resolve(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""


def _with_boundaries(payload: dict[str, Any]) -> dict[str, Any]:
    for field in BOUNDARY_FALSE_FIELDS:
        payload[field] = False
    payload["rollout_command_generated"] = False
    return payload


def _int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _parse_v72_execution_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- ") or "=" not in line:
            continue
        key, value = line[2:].split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = int(float(value))
        except Exception:
            metrics[key] = value
    metrics["cvodes_or_casadi_warning_limited"] = bool(
        "CasADi" in str(text or "") or "CVODES" in str(text or "") or "runtime-error" in str(text or "")
    )
    return metrics


def build_envelope_contract(
    *,
    include_tomato_projection: bool = True,
    include_target_direction: bool = True,
    contract_complete: bool = True,
) -> dict[str, Any]:
    output_schema: dict[str, Any] = {
        "name": {"type": "string", "format": "profile_action_envelope:<profile_name>"},
        "profile_name": {"type": "string", "source": "profile_candidate.name"},
        "candidate_source": {"type": "string", "constant": "normal_path_profile_action_envelope"},
        "intent": {"type": "string", "source": "profile candidate or structured anchor intent"},
        "action_bounds": {
            "type": "object",
            "fields": list(ACTION_FIELDS),
            "per_field_schema": {"min": "float", "max": "float"},
        },
        "preferred_direction": {
            "type": "object",
            "fields": list(ACTION_FIELDS),
            "allowed_values": list(DIRECTION_VALUES),
        },
        "priority_terms": {
            "type": "object",
            "required_terms": [
                "profile_target_alignment",
                "tomato_safety_compatibility",
                "action_continuity",
                "hard_safety_precedence",
            ],
        },
        "continuity_constraints": {
            "type": "object",
            "fields": list(ACTION_FIELDS),
            "required_terms": ["max_delta_from_previous_action", "previous_action_source"],
        },
        "projected_action": {
            "type": "object",
            "fields": list(ACTION_FIELDS),
            "purpose": "shadow projection candidate, never final control in v73",
        },
        "score_terms": {"type": "object", "fields": list(SCORE_TERM_FIELDS)},
        "eligible": {"type": "boolean"},
        "rejection_reason": {"type": "string"},
        "compatibility_category": {"type": "string", "allowed_values": list(COMPATIBILITY_CATEGORIES)},
    }
    required_fields = list(ENVELOPE_REQUIRED_OUTPUT_FIELDS)
    if include_target_direction:
        output_schema["target_direction"] = {
            "type": "object",
            "fields": list(PROFILE_TARGET_FIELDS),
            "allowed_values": list(DIRECTION_VALUES),
        }
    else:
        required_fields.remove("target_direction")
    if include_tomato_projection:
        output_schema["tomato_safety_projection"] = {
            "type": "object",
            "required_fields": [
                "projection_required",
                "projected_action",
                "projection_applied",
                "projection_reasons",
                "rewrite_delta_by_field",
            ],
            "source": "future shadow call to apply_tomato_safety_v2",
            "not_final_action_control": True,
        }
    else:
        required_fields.remove("tomato_safety_projection")

    contract = {
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "artifact": "profile_action_envelope_contract_20260604_v73",
        "current_stage": "v73_profile_to_action_envelope_design",
        "model_name": MODEL_NAME,
        "scope": "design_only_action_envelope_contract_no_runtime_control_change",
        "input_contract": {
            "profile_candidate_payload": {
                "required_fields": [
                    "name",
                    "regime",
                    "target_profile",
                    "priority",
                    "constraints",
                    "reason",
                    "score",
                    "score_breakdown",
                ],
                "target_profile_required_keys": list(PROFILE_TARGET_FIELDS),
            },
            "structured_anchor_provenance": {
                "required_fields": [
                    "structured_anchor_profile_intent",
                    "structured_anchor_target_temp",
                    "structured_anchor_target_co2",
                    "structured_anchor_target_rh",
                    "structured_anchor_risk_flags",
                ],
            },
            "profile_bridge_provenance": {
                "required_fields": [
                    "structured_anchor_profile_bridge_attempted",
                    "structured_anchor_profile_bridge_selected_shadow_profile",
                    "structured_anchor_profile_bridge_profile_candidate_count",
                ],
            },
            "state_snapshot": {
                "required_fields": [
                    "temp_air",
                    "rh_air",
                    "vpd_air",
                    "co2_air",
                    "glob_rad",
                    "hour_of_day",
                    "dew_margin_air",
                    "canopy_dew_margin",
                    "dry_risk",
                    "dew_risk",
                ],
            },
            "current_control": {
                "fields": list(ACTION_FIELDS),
                "trace_mapping": dict(ACTION_TRACE_FIELD_MAP),
            },
            "previous_action": {"fields": list(ACTION_FIELDS), "source": "previous trace row or runtime action memory"},
            "baseline_control": {"fields": list(ACTION_FIELDS), "source": "current llm_rspc_v2 baseline candidate"},
            "tomato_safety_config": {"required_family": "AgentConfig tomato_safety_v2_*"},
            "v70_incompatibility_categories": {"allowed_values": list(COMPATIBILITY_CATEGORIES)},
        },
        "output_contract": {
            "required_fields": required_fields,
            "field_schema": output_schema,
            "candidate_name_pattern": "profile_action_envelope:<profile_name>",
            "candidate_source_value": "normal_path_profile_action_envelope",
            "action_fields": list(ACTION_FIELDS),
            "action_trace_field_map": dict(ACTION_TRACE_FIELD_MAP),
        },
        "envelope_algorithm": [
            "derive intent and target_direction from profile trajectory and structured anchor provenance",
            "produce action_bounds instead of a single low-level direct action",
            "assign preferred_direction and priority_terms for scorer/arbitrator search",
            "apply continuity_constraints relative to previous_action before projection",
            "require tomato_safety_projection provenance and keep Tomato Safety as final hard shield",
            "emit projected_action only as shadow provenance for future scoring",
        ],
        "safety_contract": {
            "tomato_safety_remains_final_hard_shield": True,
            "dew_canopy_hard_safety_not_tradeable": True,
            "projected_action_is_shadow_provenance_only": True,
            "llm_direct_actuator_control_allowed": False,
            "fallback_exception_only_not_primary": True,
        },
        "contract_complete": bool(contract_complete and include_tomato_projection and include_target_direction),
    }
    return _with_boundaries(contract)


def _contract_has_tomato_projection(contract: Mapping[str, Any]) -> bool:
    output = contract.get("output_contract", {})
    if not isinstance(output, Mapping):
        return False
    required = set(output.get("required_fields", []) or [])
    schema = output.get("field_schema", {})
    if not isinstance(schema, Mapping):
        return False
    tomato = schema.get("tomato_safety_projection", {})
    return bool("tomato_safety_projection" in required and isinstance(tomato, Mapping) and tomato.get("required_fields"))


def _contract_has_target_direction(contract: Mapping[str, Any]) -> bool:
    output = contract.get("output_contract", {})
    if not isinstance(output, Mapping):
        return False
    required = set(output.get("required_fields", []) or [])
    schema = output.get("field_schema", {})
    if not isinstance(schema, Mapping):
        return False
    target = schema.get("target_direction", {})
    fields = set(target.get("fields", []) or []) if isinstance(target, Mapping) else set()
    return bool("target_direction" in required and set(PROFILE_TARGET_FIELDS) <= fields)


def _contract_has_envelope_fields(contract: Mapping[str, Any]) -> bool:
    output = contract.get("output_contract", {})
    if not isinstance(output, Mapping):
        return False
    required = set(output.get("required_fields", []) or [])
    source_value = str(output.get("candidate_source_value", "") or "")
    return bool(
        set(ENVELOPE_REQUIRED_OUTPUT_FIELDS) <= required
        and source_value == "normal_path_profile_action_envelope"
        and _contract_has_tomato_projection(contract)
        and _contract_has_target_direction(contract)
        and bool(contract.get("contract_complete", False))
    )


def _aggregate(v70_audit: Mapping[str, Any]) -> Mapping[str, Any]:
    aggregate = v70_audit.get("aggregate", {}) if isinstance(v70_audit, Mapping) else {}
    return aggregate if isinstance(aggregate, Mapping) else {}


def _v70_hypothesis(v70_audit: Mapping[str, Any], v70_hypothesis: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(v70_hypothesis, Mapping) and v70_hypothesis:
        return v70_hypothesis
    hypothesis = v70_audit.get("hypothesis_check", {}) if isinstance(v70_audit, Mapping) else {}
    return hypothesis if isinstance(hypothesis, Mapping) else {}


def _source_has_diagnostic_blocker(v70_audit: Mapping[str, Any]) -> bool:
    aggregate = _aggregate(v70_audit)
    return bool(
        _int(aggregate.get("action_invariance_violation_steps")) > 0
        or _int(aggregate.get("contract_missing_steps")) > 0
        or _int(aggregate.get("tomato_projection_missing_steps")) > 0
    )


def _direct_mapping_gap_confirmed(v70_audit: Mapping[str, Any], v70_hypothesis: Mapping[str, Any]) -> bool:
    aggregate = _aggregate(v70_audit)
    hypothesis = _v70_hypothesis(v70_audit, v70_hypothesis)
    return bool(
        str(hypothesis.get("hypothesis_status", "")) == "profile_to_action_direct_mapping_hypothesis_needs_revision"
        or str(hypothesis.get("alternative_hypothesis", "")) == "action_envelope_or_safety_projected_mapping"
        or _int(aggregate.get("tomato_safety_incompatible_steps")) > 0
        or _int(aggregate.get("profile_target_incompatible_steps")) > 0
    )


def build_design(
    *,
    v70_audit: Mapping[str, Any],
    v70_hypothesis: Mapping[str, Any],
    v72_execution_text: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = dict(_aggregate(v70_audit))
    hypothesis = dict(_v70_hypothesis(v70_audit, v70_hypothesis))
    v72_metrics = _parse_v72_execution_metrics(v72_execution_text)
    design = {
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "artifact": "qwen37plus_profile_to_action_envelope_design_20260604_v73",
        "current_stage": "v73_profile_to_action_envelope_design",
        "model_name": MODEL_NAME,
        "controller": "llm_rspc_v2",
        "scope": "design_only_no_runtime_control_change",
        "source_artifacts": {
            "v70_compatibility_audit": str(v70_audit.get("artifact", "")) or _rel(V70_AUDIT_JSON),
            "v70_hypothesis_check": str(hypothesis.get("artifact", "")) or _rel(V70_HYPOTHESIS_JSON),
            "v72_execution_record": _rel(V72_EXECUTION_MD),
        },
        "evidence_summary": {
            "v70_rows": _int(aggregate.get("rows")),
            "profile_action_candidate_available_steps": _int(
                aggregate.get("profile_action_candidate_available_steps")
            ),
            "compatibility_shadow_ready_steps": _int(aggregate.get("compatibility_shadow_ready_steps")),
            "tomato_safety_incompatible_steps": _int(aggregate.get("tomato_safety_incompatible_steps")),
            "profile_target_incompatible_steps": _int(aggregate.get("profile_target_incompatible_steps")),
            "action_continuity_incompatible_steps": _int(aggregate.get("action_continuity_incompatible_steps")),
            "action_invariance_violation_steps": _int(aggregate.get("action_invariance_violation_steps")),
            "fallback_candidate_source_violation_steps": _int(
                aggregate.get("fallback_candidate_source_violation_steps")
            ),
            "v72_rows": _int(v72_metrics.get("rows")),
            "v72_profile_action_candidate_available_steps": _int(
                v72_metrics.get("profile_action_candidate_available_steps")
            ),
            "v72_final_action_changed_rows": _int(v72_metrics.get("final_action_changed_rows")),
            "v72_strict_replay_cache_hit_rows": _int(v72_metrics.get("strict_replay_cache_hit_rows")),
            "v72_strict_cache_miss_rows": _int(v72_metrics.get("strict_cache_miss_rows")),
            "cvodes_or_casadi_warning_limited": bool(v72_metrics.get("cvodes_or_casadi_warning_limited", False)),
        },
        "current_hypothesis": "profile trajectory can be directly mapped into low-level profile_action candidates.",
        "observed_gap": str(hypothesis.get("observed_gap", "")),
        "hypothesis_status": "profile_to_action_direct_mapping_hypothesis_needs_revision",
        "alternative_hypothesis": "action_envelope_or_safety_projected_mapping",
        "new_hypothesis": "profile trajectory should define a Tomato Safety-compatible action envelope before scoring.",
        "design_decision": "define_profile_action_envelope_contract_not_runtime_shadow_patch",
        "normal_path_data_flow": [
            "structured_anchor",
            "profile_generator_candidates",
            "profile_action_envelope",
            "candidate_scorer_or_arbitrator",
            "Tomato_Safety_final_shield",
        ],
        "envelope_design_principles": [
            "profile candidate expresses intent, target direction, bounds, and priorities",
            "scorer searches within action_bounds rather than receiving one direct low-level action",
            "Tomato Safety projection is required provenance before a candidate can be shadow-scored",
            "continuity constraints bound deltas from previous_action",
            "projected_action is design/shadow provenance and not final runtime control in v73",
        ],
        "contract_artifact": contract.get("artifact", ""),
        "contract_complete": bool(contract.get("contract_complete", False)),
        "online_llm_needed": False,
        "runtime_control_change": False,
        "runtime_trace_written": False,
        "expected_artifacts": [
            str(DESIGN_JSON.name),
            str(CONTRACT_JSON.name),
            str(READINESS_JSON.name),
        ],
        "success_condition": "A complete design-only action-envelope contract is ready for future shadow instrumentation.",
        "stop_condition": "Stop before escalation if source action diff, missing Tomato Safety projection, or hard-safety provenance gaps appear.",
    }
    return _with_boundaries(design)


def _next_action(
    *,
    v70_audit: Mapping[str, Any],
    v70_hypothesis: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> str:
    if _source_has_diagnostic_blocker(v70_audit):
        return "stop_source_action_diff_or_safety_provenance_diagnostic"
    if not _contract_has_tomato_projection(contract):
        return "tomato_safety_projection_provenance_repair_plan"
    if not _contract_has_target_direction(contract):
        return "profile_target_direction_schema_repair_plan"
    if _contract_has_envelope_fields(contract) and _direct_mapping_gap_confirmed(v70_audit, v70_hypothesis):
        return "minimal_profile_action_envelope_shadow_instrumentation_plan"
    if _direct_mapping_gap_confirmed(v70_audit, v70_hypothesis):
        return "action_envelope_or_safety_projected_mapping_shadow_patch_plan"
    return "profile_to_action_mapping_repair_evidence_refresh_plan"


def build_readiness(
    *,
    design: Mapping[str, Any],
    contract: Mapping[str, Any],
    v70_audit: Mapping[str, Any],
    v70_hypothesis: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = _aggregate(v70_audit)
    next_action = _next_action(v70_audit=v70_audit, v70_hypothesis=v70_hypothesis, contract=contract)
    readiness = {
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "artifact": "profile_to_action_mapping_repair_readiness_20260604_v73",
        "current_stage": "v73_profile_to_action_envelope_design",
        "model_name": MODEL_NAME,
        "design_artifact": design.get("artifact", ""),
        "contract_artifact": contract.get("artifact", ""),
        "direct_mapping_gap_confirmed": _direct_mapping_gap_confirmed(v70_audit, v70_hypothesis),
        "source_action_invariant": bool(_int(aggregate.get("action_invariance_violation_steps")) == 0),
        "source_tomato_projection_present": bool(_int(aggregate.get("tomato_projection_missing_steps")) == 0),
        "source_contract_complete": bool(_int(aggregate.get("contract_missing_steps")) == 0),
        "action_envelope_contract_complete": _contract_has_envelope_fields(contract),
        "tomato_safety_projection_contract_present": _contract_has_tomato_projection(contract),
        "target_direction_schema_present": _contract_has_target_direction(contract),
        "design_only_ready": bool(next_action == "minimal_profile_action_envelope_shadow_instrumentation_plan"),
        "runtime_control_change": False,
        "rollout_execution_authorized": False,
        "online_llm_needed": False,
        "next_action": next_action,
        "blocked_until": [
            "future opt-in shadow instrumentation implements profile_action_envelope provenance"
            if next_action == "minimal_profile_action_envelope_shadow_instrumentation_plan"
            else "diagnose source action diff or Tomato Safety projection provenance"
            if next_action == "stop_source_action_diff_or_safety_provenance_diagnostic"
            else "repair the design contract field named by next_action"
        ],
    }
    return _with_boundaries(readiness)


def build_report(payload: Mapping[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "## Boundary",
        "",
        "- impact_level=design-only",
        "- default_llm_rspc_v2_changed=false",
        "- online_llm_called=false",
        "- new_rollout_run=false",
        "- fallback_enhanced=false",
        "- final_action_changed=false",
        "- performance_claim_allowed=false",
        "- promotion_evidence=false",
        "- rollout_command_generated=false",
        "",
        "## Hypothesis Revision",
        "",
        "- old_hypothesis=profile trajectory maps directly into low-level profile_action candidates",
        "- failure_evidence_or_missing_evidence=v70 profile_action compatibility gaps",
        "- new_hypothesis=profile_action_envelope mediates target trajectory into bounded action space",
        "- hypothesis_status=profile_to_action_direct_mapping_hypothesis_needs_revision",
        "- alternative_hypothesis=action_envelope_or_safety_projected_mapping",
        f"- smallest_testable_next_step={payload.get('next_action', '')}",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_artifacts(
    *,
    design: Mapping[str, Any],
    contract: Mapping[str, Any],
    readiness: Mapping[str, Any],
    design_json: str | Path = DESIGN_JSON,
    design_md: str | Path = DESIGN_MD,
    contract_json: str | Path = CONTRACT_JSON,
    contract_md: str | Path = CONTRACT_MD,
    readiness_json: str | Path = READINESS_JSON,
    readiness_md: str | Path = READINESS_MD,
) -> dict[str, str]:
    payloads = (
        (Path(design_json), Path(design_md), design, "v73 qwen3.7-plus Profile-to-Action Envelope Design"),
        (Path(contract_json), Path(contract_md), contract, "v73 Profile Action Envelope Contract"),
        (Path(readiness_json), Path(readiness_md), readiness, "v73 Profile-to-Action Mapping Repair Readiness"),
    )
    written: dict[str, str] = {}
    for json_path, md_path, payload, title in payloads:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(build_report(payload, title), encoding="utf-8")
        written[json_path.stem + "_json"] = str(json_path)
        written[md_path.stem + "_md"] = str(md_path)
    return written


def write_all(
    *,
    v70_audit_json: str | Path = V70_AUDIT_JSON,
    v70_hypothesis_json: str | Path = V70_HYPOTHESIS_JSON,
    v72_execution_md: str | Path = V72_EXECUTION_MD,
) -> dict[str, Any]:
    v70_audit = _load_json(v70_audit_json)
    v70_hypothesis = _load_json(v70_hypothesis_json)
    v72_execution_text = _load_text(v72_execution_md)
    contract = build_envelope_contract()
    design = build_design(
        v70_audit=v70_audit,
        v70_hypothesis=v70_hypothesis,
        v72_execution_text=v72_execution_text,
        contract=contract,
    )
    readiness = build_readiness(
        design=design,
        contract=contract,
        v70_audit=v70_audit,
        v70_hypothesis=v70_hypothesis,
    )
    design["next_action"] = readiness["next_action"]
    written = write_artifacts(design=design, contract=contract, readiness=readiness)
    return {
        "design": design,
        "contract": contract,
        "readiness": readiness,
        "written": written,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v70-audit-json", default=_rel(V70_AUDIT_JSON))
    parser.add_argument("--v70-hypothesis-json", default=_rel(V70_HYPOTHESIS_JSON))
    parser.add_argument("--v72-execution-md", default=_rel(V72_EXECUTION_MD))
    parser.add_argument("--no-write", action="store_true", help="Print readiness JSON without writing artifacts.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_write:
        v70_audit = _load_json(args.v70_audit_json)
        v70_hypothesis = _load_json(args.v70_hypothesis_json)
        contract = build_envelope_contract()
        design = build_design(
            v70_audit=v70_audit,
            v70_hypothesis=v70_hypothesis,
            v72_execution_text=_load_text(args.v72_execution_md),
            contract=contract,
        )
        readiness = build_readiness(
            design=design,
            contract=contract,
            v70_audit=v70_audit,
            v70_hypothesis=v70_hypothesis,
        )
        print(json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0
    result = write_all(
        v70_audit_json=args.v70_audit_json,
        v70_hypothesis_json=args.v70_hypothesis_json,
        v72_execution_md=args.v72_execution_md,
    )
    print(json.dumps(result["written"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
