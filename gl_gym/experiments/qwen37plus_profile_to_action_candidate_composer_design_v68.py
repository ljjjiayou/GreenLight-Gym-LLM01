"""Build v68 profile-to-action candidate composer design artifacts.

This stage is design-only. It consumes the v67 normal-path gap audit and
existing v64/v65/v66 provenance to define the future contract for composing
profile-generator trajectories into executable action candidates. It does not
run rollout, call online LLMs, change llm_rspc_v2, modify runtime control
logic, authorize replay, or make performance claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gl_gym.experiments.qwen37plus_normal_path_profile_candidate_arbitration_v67 as v67  # noqa: E402
import gl_gym.experiments.qwen37plus_profile_candidate_guardrail_reconciliation_v66 as v66  # noqa: E402
import gl_gym.experiments.qwen37plus_structured_bridge_runtime_failure_attribution_v65 as v65  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import AUDIT_DIR  # noqa: E402


MODEL_NAME = "qwen3.7-plus"
ARTIFACT_DATE = "20260603"
FAILURE_SCENARIOS = tuple(v67.FAILURE_SCENARIOS)
V64_TRACE_DIR = v67.V64_TRACE_DIR
V65_ATTRIBUTION_JSON = v65.ATTRIBUTION_JSON
V66_RECONCILIATION_JSON = v66.RECONCILIATION_JSON
V67_GAP_JSON = v67.GAP_JSON
V67_READINESS_JSON = v67.READINESS_JSON

DESIGN_JSON = AUDIT_DIR / "qwen37plus_profile_to_action_candidate_composer_design_20260603_v68.json"
DESIGN_MD = AUDIT_DIR / "qwen37plus_profile_to_action_candidate_composer_design_20260603_v68.md"
CONTRACT_JSON = AUDIT_DIR / "qwen37plus_profile_action_candidate_contract_20260603_v68.json"
CONTRACT_MD = AUDIT_DIR / "qwen37plus_profile_action_candidate_contract_20260603_v68.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260603_v68.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260603_v68.md"

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
PROFILE_CANDIDATE_REQUIRED_FIELDS = (
    "name",
    "regime",
    "target_profile",
    "priority",
    "constraints",
    "reason",
    "score",
    "score_breakdown",
)
BRIDGE_PROVENANCE_FIELDS = (
    "structured_anchor_profile_bridge_attempted",
    "structured_anchor_profile_bridge_bridgeable",
    "structured_anchor_profile_bridge_selected_shadow_profile",
    "structured_anchor_profile_bridge_profile_candidate_count",
    "structured_anchor_profile_bridge_requested_shapes",
    "structured_anchor_profile_bridge_valid_structured_anchor",
)
STATE_SNAPSHOT_FIELDS = (
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
)
PROFILE_SCORE_TRACE_FIELDS = (
    "profile_rspc_shadow_candidate_count",
    "profile_rspc_shadow_eligible_candidate_count",
    "profile_rspc_shadow_best_eligible",
    "profile_rspc_shadow_best_profile",
    "profile_rspc_shadow_best_score",
    "profile_rspc_shadow_best_raw_score",
    "profile_rspc_shadow_best_safety_score",
    "profile_rspc_shadow_safety_gate_reason",
    "profile_rspc_shadow_top_candidates",
)
REQUIRED_TRACE_METADATA_FIELDS = tuple(
    dict.fromkeys(
        tuple(v67.REQUIRED_METADATA_FIELDS)
        + BRIDGE_PROVENANCE_FIELDS
        + PROFILE_SCORE_TRACE_FIELDS
        + tuple(ACTION_TRACE_FIELD_MAP.values())
    )
)
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
    "controlled_replay_allowed",
    "controlled_replay_execution_allowed",
    "metadata_replay_execution_allowed",
    "performance_claim_allowed",
    "promotion_evidence",
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


def _trace_path(trace_dir: str | Path, scenario_id: str, controller: str = "llm_rspc_v2") -> Path:
    return v67._trace_path(trace_dir, scenario_id, controller=controller)


def _read_csv_header(path: str | Path) -> list[str]:
    p = _resolve(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            return [str(item) for item in next(reader)]
        except StopIteration:
            return []


def _with_boundaries(payload: dict[str, Any]) -> dict[str, Any]:
    for field in BOUNDARY_FALSE_FIELDS:
        payload[field] = False
    return payload


def _count(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def build_candidate_contract() -> dict[str, Any]:
    output_field_schema = {
        "name": {"type": "string", "format": "profile_action:<profile_name>"},
        "profile_name": {"type": "string", "source": "profile_candidate.name"},
        "candidate_source": {"type": "string", "constant": "normal_path_profile_candidate"},
        "raw_action": {"type": "object", "fields": list(ACTION_FIELDS)},
        "post_tomato_action": {"type": "object", "fields": list(ACTION_FIELDS)},
        "score_terms": {"type": "object", "fields": list(SCORE_TERM_FIELDS)},
        "tomato_safety_v2_applied": {"type": "boolean"},
        "eligible": {"type": "boolean"},
        "rejection_reason": {"type": "string"},
    }
    contract = {
        "artifact": "qwen37plus_profile_action_candidate_contract_20260603_v68",
        "current_stage": "v68_profile_to_action_candidate_composer_design",
        "model_name": MODEL_NAME,
        "scope": "future_contract_design_only_no_runtime_control_change",
        "input_contract": {
            "profile_candidate_payload": {
                "required_fields": list(PROFILE_CANDIDATE_REQUIRED_FIELDS),
                "target_profile_required_keys": list(PROFILE_TARGET_FIELDS),
                "source": "profile_generator.build_profile_generator_shadow_payload",
            },
            "structured_anchor_profile_bridge_provenance": {
                "required_fields": list(BRIDGE_PROVENANCE_FIELDS),
                "source": "structured_anchor_profile_bridge.build_structured_anchor_profile_bridge_shadow_payload",
            },
            "state_snapshot": {
                "required_fields": list(STATE_SNAPSHOT_FIELDS),
                "source": "llm_rspc_v2 trace/state at current step",
            },
            "baseline_control": {
                "fields": list(ACTION_FIELDS),
                "trace_mapping": dict(ACTION_TRACE_FIELD_MAP),
                "source": "current llm_rspc_v2 pre-composer action",
            },
            "previous_action": {
                "fields": list(ACTION_FIELDS),
                "source": "previous trace row or runtime action memory",
            },
            "tomato_safety_config": {
                "required_family": "AgentConfig tomato_safety_v2_*",
                "source": "apply_tomato_safety_v2",
            },
        },
        "output_contract": {
            "required_fields": list(CONTRACT_REQUIRED_OUTPUT_FIELDS),
            "field_schema": output_field_schema,
            "candidate_name_pattern": "profile_action:<profile_name>",
            "candidate_source_value": "normal_path_profile_candidate",
            "action_fields": list(ACTION_FIELDS),
        },
        "composition_algorithm": [
            "select eligible profile candidate from profile scorer output",
            "extract current target_temp/target_co2/target_rh from profile trajectory",
            "generate raw_action with the existing profile target tracking proxy",
            "project raw_action through Tomato Safety v2 as shadow post_tomato_action",
            "score raw and post_tomato actions with existing candidate scoring terms",
            "mark eligible=false with rejection_reason on safety, profile, or continuity conflict",
        ],
        "code_reuse_sources": [
            "profile_generator.build_profile_generator_shadow_payload",
            "profile_generator.score_profile_candidate_payloads",
            "LLMAgent._profile_shadow_target_value",
            "LLMAgent._apply_profile_target_tracking_proxy",
            "apply_tomato_safety_v2",
            "LLMAgent._score_fallback_candidate",
            "LLMAgent._profile_action_delta_terms",
            "LLMAgent._profile_score_delta_terms",
            "v66.evaluate_candidate",
        ],
        "safety_contract": {
            "tomato_safety_remains_final_hard_shield": True,
            "dew_canopy_hard_safety_not_tradeable": True,
            "fallback_exception_only_not_primary": True,
            "llm_direct_actuator_control_allowed": False,
        },
        "contract_complete": True,
        "rollout_command_generated": False,
    }
    return _with_boundaries(contract)


def build_metadata_inventory(
    *,
    trace_dir: str | Path = V64_TRACE_DIR,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    required_fields: Sequence[str] = REQUIRED_TRACE_METADATA_FIELDS,
) -> dict[str, Any]:
    scenario_reports = []
    missing_union: set[str] = set()
    trace_missing_count = 0
    for scenario in failure_scenarios:
        trace = _trace_path(trace_dir, scenario)
        header = _read_csv_header(trace)
        missing = [field for field in required_fields if field not in header]
        missing_union.update(missing)
        if not header:
            trace_missing_count += 1
        scenario_reports.append(
            {
                "scenario_id": scenario,
                "trace_path": _rel(trace),
                "trace_exists": bool(header),
                "field_count": int(len(header)),
                "required_field_count": int(len(required_fields)),
                "missing_required_fields": missing,
                "metadata_sufficient": bool(header and not missing),
            }
        )
    return {
        "required_fields": list(required_fields),
        "scenario_reports": scenario_reports,
        "trace_missing_count": int(trace_missing_count),
        "missing_required_fields": sorted(missing_union),
        "metadata_sufficient": bool(not missing_union and trace_missing_count == 0 and scenario_reports),
    }


def _evidence_summary(gap: Mapping[str, Any], readiness: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_v67_gap_artifact": gap.get("artifact", ""),
        "source_v67_readiness_artifact": readiness.get("artifact", ""),
        "row_count": _count(gap.get("row_count")),
        "profile_candidate_available_steps": _count(gap.get("profile_candidate_available_steps")),
        "profile_rspc_shadow_eligible_steps": _count(gap.get("profile_rspc_shadow_eligible_steps")),
        "profile_low_level_action_generated_steps": _count(gap.get("profile_low_level_action_generated_steps")),
        "normal_path_profile_action_candidate_steps": _count(gap.get("normal_path_profile_action_candidate_steps")),
        "candidate_composer_missing_steps": _count(gap.get("candidate_composer_missing_steps")),
        "fallback_should_not_have_been_primary_steps": _count(gap.get("fallback_should_not_have_been_primary_steps")),
        "profile_to_action_gap_confirmed": bool(gap.get("profile_to_action_gap_confirmed", False)),
        "dominant_gap": str(gap.get("dominant_gap") or readiness.get("dominant_gap") or ""),
        "v67_next_action": str(gap.get("next_action") or readiness.get("next_action") or ""),
    }


def build_composer_design(
    *,
    gap: Mapping[str, Any],
    v67_readiness: Mapping[str, Any],
    contract: Mapping[str, Any],
    metadata_inventory: Mapping[str, Any],
    v65_attribution: Mapping[str, Any] | None = None,
    v66_reconciliation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    v65_attribution = dict(v65_attribution or {})
    v66_reconciliation = dict(v66_reconciliation or {})
    evidence = _evidence_summary(gap, v67_readiness)
    design = {
        "artifact": "qwen37plus_profile_to_action_candidate_composer_design_20260603_v68",
        "current_stage": "v68_profile_to_action_candidate_composer_design",
        "model_name": MODEL_NAME,
        "scope": "design_only_profile_to_action_candidate_composer",
        "controller": "llm_rspc_v2",
        "source_artifacts": {
            "v67_gap": gap.get("artifact", ""),
            "v67_readiness": v67_readiness.get("artifact", ""),
            "v65_attribution": v65_attribution.get("artifact", ""),
            "v66_reconciliation": v66_reconciliation.get("artifact", ""),
        },
        "evidence_summary": evidence,
        "metadata_inventory_summary": {
            "metadata_sufficient": bool(metadata_inventory.get("metadata_sufficient", False)),
            "trace_missing_count": _count(metadata_inventory.get("trace_missing_count")),
            "missing_required_fields": list(metadata_inventory.get("missing_required_fields", []) or []),
        },
        "design_decision": "define_profile_to_action_candidate_composer_contract_not_fallback_patch",
        "normal_path_data_flow": [
            "structured_anchor",
            "IntentContract",
            "profile_generator_candidates",
            "profile_scorer",
            "profile_to_action_candidate_composer",
            "compatibility_scorer",
            "Tomato_Safety_final_shield",
        ],
        "composer_steps": [
            {
                "step": "select_profile",
                "input": "profile candidates and profile scorer diagnostics",
                "output": "profile candidate payload with name and target_profile",
            },
            {
                "step": "extract_targets",
                "input": "target_profile and current timestep",
                "output": "current target_temp, target_co2, target_rh",
            },
            {
                "step": "compose_raw_action",
                "input": "baseline_control, state snapshot, current profile targets",
                "output": "bounded raw_action over heat/co2/screen/vent/lamp/shade",
            },
            {
                "step": "project_tomato_safety",
                "input": "raw_action and Tomato Safety v2 config",
                "output": "post_tomato_action and tomato_safety_v2 provenance",
            },
            {
                "step": "score_and_filter",
                "input": "raw/post_tomato actions, profile targets, previous action, safety provenance",
                "output": "eligible candidate with score_terms or rejection_reason",
            },
        ],
        "future_shadow_patch_requirements": [
            "composer must be opt-in and shadow-only in the first runtime patch",
            "composer candidate must enter the same candidate scoring surface as RSPC candidates only after metadata proves action invariance is not required",
            "fallback_candidate must remain exception-only and must not become the primary normal-path selector",
            "Tomato Safety must still be the final hard shield after any composer-selected action",
        ],
        "stop_rules": [
            "stop if any hard safety candidate is preferred over a safe candidate",
            "stop if composer candidate provenance is missing candidate_source, profile_name, raw_action, post_tomato_action, or rejection_reason",
            "stop if action deltas cannot be attributed to profile target tracking versus Tomato Safety projection",
            "stop if fallback remains primary in a profile-available window after composer instrumentation",
        ],
        "contract_artifact": contract.get("artifact", ""),
        "contract_complete": bool(contract.get("contract_complete", False)),
        "runtime_control_change": False,
        "runtime_trace_written": False,
        "rollout_command_generated": False,
        "implementation_patch_allowed_in_v68": False,
        "next_action": "minimal_normal_path_profile_arbitration_shadow_patch_plan",
    }
    return _with_boundaries(design)


def _profile_payload_available(gap: Mapping[str, Any]) -> bool:
    return bool(
        _count(gap.get("profile_candidate_available_steps")) > 0
        and _count(gap.get("profile_rspc_shadow_eligible_steps")) > 0
    )


def _contract_has_action_and_score_fields(contract: Mapping[str, Any]) -> tuple[bool, bool]:
    output = contract.get("output_contract", {}) if isinstance(contract.get("output_contract", {}), Mapping) else {}
    required = set(output.get("required_fields", []) or [])
    schema = output.get("field_schema", {}) if isinstance(output.get("field_schema", {}), Mapping) else {}
    has_action = bool({"raw_action", "post_tomato_action", "profile_name", "candidate_source"} <= required)
    score_schema = schema.get("score_terms", {}) if isinstance(schema.get("score_terms", {}), Mapping) else {}
    score_fields = set(score_schema.get("fields", []) or [])
    has_score = bool("score_terms" in required and set(SCORE_TERM_FIELDS) <= score_fields)
    return has_action, has_score


def _next_action(
    *,
    gap: Mapping[str, Any],
    contract: Mapping[str, Any],
    metadata_inventory: Mapping[str, Any],
) -> str:
    if not _profile_payload_available(gap):
        return "profile_generator_payload_schema_repair_plan"
    action_fields_defined, score_fields_defined = _contract_has_action_and_score_fields(contract)
    if not bool(metadata_inventory.get("metadata_sufficient", False)) or not action_fields_defined or not score_fields_defined:
        return "profile_action_composer_instrumentation_plan"
    if bool(contract.get("contract_complete", False)):
        return "minimal_normal_path_profile_arbitration_shadow_patch_plan"
    return "profile_action_composer_instrumentation_plan"


def build_readiness(
    *,
    design: Mapping[str, Any],
    contract: Mapping[str, Any],
    gap: Mapping[str, Any],
    v67_readiness: Mapping[str, Any],
    metadata_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    action_fields_defined, score_fields_defined = _contract_has_action_and_score_fields(contract)
    next_action = _next_action(gap=gap, contract=contract, metadata_inventory=metadata_inventory)
    readiness = {
        "artifact": "metadata_replay_readiness_checklist_20260603_v68",
        "current_stage": "v68_profile_to_action_candidate_composer_design",
        "model_name": MODEL_NAME,
        "source_v67_next_action": str(gap.get("next_action") or v67_readiness.get("next_action") or ""),
        "profile_to_action_gap_confirmed": bool(gap.get("profile_to_action_gap_confirmed", False)),
        "composer_design_complete": True,
        "contract_complete": bool(contract.get("contract_complete", False)),
        "profile_payload_available": _profile_payload_available(gap),
        "metadata_sufficient_for_composer_contract": bool(metadata_inventory.get("metadata_sufficient", False)),
        "action_provenance_fields_defined": bool(action_fields_defined),
        "score_fields_defined": bool(score_fields_defined),
        "missing_required_metadata_fields": list(metadata_inventory.get("missing_required_fields", []) or []),
        "failure_taxonomy": sorted(
            {
                str(gap.get("dominant_gap") or ""),
                "composer_contract_defined" if bool(contract.get("contract_complete", False)) else "",
                "metadata_missing" if not bool(metadata_inventory.get("metadata_sufficient", False)) else "",
                "profile_payload_missing" if not _profile_payload_available(gap) else "",
            }
            - {""}
        ),
        "rollout_execution_authorized": False,
        "runtime_control_change": False,
        "rollout_command_generated": False,
        "next_action": next_action,
        "blocked_until": [
            "minimal normal-path profile arbitration shadow patch design"
            if next_action == "minimal_normal_path_profile_arbitration_shadow_patch_plan"
            else "profile-action composer instrumentation"
            if next_action == "profile_action_composer_instrumentation_plan"
            else "profile generator payload schema repair"
        ],
        "design_artifact": design.get("artifact", ""),
        "contract_artifact": contract.get("artifact", ""),
    }
    return _with_boundaries(readiness)


def _write_json_md(path_json: Path, path_md: Path, payload: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    lines = [
        f"# {title}",
        "",
        f"- Controlled replay allowed: `{payload.get('controlled_replay_allowed', False)}`",
        f"- Metadata replay execution allowed: `{payload.get('metadata_replay_execution_allowed', False)}`",
        f"- Performance claim allowed: `{payload.get('performance_claim_allowed', False)}`",
        f"- Promotion evidence: `{payload.get('promotion_evidence', False)}`",
        f"- Rollout command generated: `{payload.get('rollout_command_generated', False)}`",
        f"- Next action: `{payload.get('next_action', '')}`",
        "",
        "```json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        "```",
        "",
    ]
    path_md.write_text("\n".join(lines), encoding="utf-8")


def write_all(
    *,
    v67_gap_json: str | Path = V67_GAP_JSON,
    v67_readiness_json: str | Path = V67_READINESS_JSON,
    v65_attribution_json: str | Path = V65_ATTRIBUTION_JSON,
    v66_reconciliation_json: str | Path = V66_RECONCILIATION_JSON,
    trace_dir: str | Path = V64_TRACE_DIR,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    gap = _load_json(v67_gap_json)
    v67_readiness = _load_json(v67_readiness_json)
    v65_attribution = _load_json(v65_attribution_json)
    v66_reconciliation = _load_json(v66_reconciliation_json)
    contract = build_candidate_contract()
    metadata_inventory = build_metadata_inventory(trace_dir=trace_dir, failure_scenarios=failure_scenarios)
    design = build_composer_design(
        gap=gap,
        v67_readiness=v67_readiness,
        contract=contract,
        metadata_inventory=metadata_inventory,
        v65_attribution=v65_attribution,
        v66_reconciliation=v66_reconciliation,
    )
    readiness = build_readiness(
        design=design,
        contract=contract,
        gap=gap,
        v67_readiness=v67_readiness,
        metadata_inventory=metadata_inventory,
    )
    design["next_action"] = readiness["next_action"]
    _write_json_md(DESIGN_JSON, DESIGN_MD, design, "v68 qwen3.7-plus Profile-to-Action Candidate Composer Design")
    _write_json_md(CONTRACT_JSON, CONTRACT_MD, contract, "v68 qwen3.7-plus Profile Action Candidate Contract")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v68 Metadata Replay Readiness")
    return {
        "design": design,
        "contract": contract,
        "readiness": readiness,
        "metadata_inventory": metadata_inventory,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v67-gap-json", default=_rel(V67_GAP_JSON))
    parser.add_argument("--v67-readiness-json", default=_rel(V67_READINESS_JSON))
    parser.add_argument("--v65-attribution-json", default=_rel(V65_ATTRIBUTION_JSON))
    parser.add_argument("--v66-reconciliation-json", default=_rel(V66_RECONCILIATION_JSON))
    parser.add_argument("--trace-dir", default=_rel(V64_TRACE_DIR))
    parser.add_argument("--failure-scenario", action="append", default=[])
    parser.add_argument("--write-all", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = write_all(
        v67_gap_json=args.v67_gap_json,
        v67_readiness_json=args.v67_readiness_json,
        v65_attribution_json=args.v65_attribution_json,
        v66_reconciliation_json=args.v66_reconciliation_json,
        trace_dir=args.trace_dir,
        failure_scenarios=args.failure_scenario or list(FAILURE_SCENARIOS),
    )
    if not args.write_all:
        print(json.dumps(result["readiness"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
