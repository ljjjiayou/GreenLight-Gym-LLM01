"""v58 structured-anchor opt-in shadow parser and offline audit.

The parser is intentionally default-off and shadow-only. It validates a
high-level planning anchor that describes profile/intent/target/risk fields,
rejects low-level actuator commands, and never changes final control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.agent.llm_agent import AgentConfig  # noqa: E402
from gl_gym.agent.structured_anchor import (  # noqa: E402
    FINAL_CONTROL_FIELDS,
    STRUCTURED_ANCHOR_FIELDS,
    parse_structured_anchor,
)
from gl_gym.experiments.qwen37_tool_layer_usage_audit_v57 import (  # noqa: E402
    AUDIT_DIR,
    MODEL_NAME,
    READINESS_JSON as V57_READINESS_JSON,
    V56_CACHE_PATH,
    _load_json,
    _plan_cache_entries,
    _rel,
    _write_json_md,
)

PARSER_AUDIT_JSON = AUDIT_DIR / "qwen37_structured_anchor_parser_shadow_audit_20260601_v58.json"
PARSER_AUDIT_MD = AUDIT_DIR / "qwen37_structured_anchor_parser_shadow_audit_20260601_v58.md"
FIXTURES_JSON = AUDIT_DIR / "qwen37_structured_anchor_parser_contract_fixtures_20260601_v58.json"
FIXTURES_MD = AUDIT_DIR / "qwen37_structured_anchor_parser_contract_fixtures_20260601_v58.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v58.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v58.md"

def build_contract_fixtures() -> dict[str, Any]:
    fixtures = [
        {
            "name": "valid_structured_anchor",
            "expected_valid": True,
            "payload": {
                "profile_intent": "hot_dry_relief",
                "target_temp": 19.0,
                "target_co2": 430.0,
                "target_rh": 70.0,
                "risk_flags": ["dry_side", "high_radiation"],
                "forbidden_intents": ["co2_enrichment_high_vent"],
                "planning_horizon_steps": 12,
                "confidence": 0.78,
            },
        },
        {
            "name": "missing_target_rh",
            "expected_valid": False,
            "payload": {
                "profile_intent": "hot_dry_relief",
                "target_temp": 19.0,
                "target_co2": 430.0,
                "risk_flags": ["dry_side"],
                "forbidden_intents": [],
                "planning_horizon_steps": 12,
                "confidence": 0.78,
            },
        },
        {"name": "empty_anchor", "expected_valid": False, "payload": {}},
        {
            "name": "contains_final_control_field",
            "expected_valid": False,
            "payload": {
                "profile_intent": "hot_dry_relief",
                "target_temp": 19.0,
                "target_co2": 430.0,
                "target_rh": 70.0,
                "risk_flags": [],
                "forbidden_intents": [],
                "planning_horizon_steps": 12,
                "confidence": 0.78,
                "ventilation": 0.7,
            },
        },
        {
            "name": "confidence_out_of_range",
            "expected_valid": False,
            "payload": {
                "profile_intent": "hot_dry_relief",
                "target_temp": 19.0,
                "target_co2": 430.0,
                "target_rh": 70.0,
                "risk_flags": [],
                "forbidden_intents": [],
                "planning_horizon_steps": 12,
                "confidence": 1.5,
            },
        },
        {
            "name": "horizon_invalid",
            "expected_valid": False,
            "payload": {
                "profile_intent": "hot_dry_relief",
                "target_temp": 19.0,
                "target_co2": 430.0,
                "target_rh": 70.0,
                "risk_flags": [],
                "forbidden_intents": [],
                "planning_horizon_steps": 0,
                "confidence": 0.5,
            },
        },
    ]
    reports = []
    for fixture in fixtures:
        parsed = parse_structured_anchor(fixture["payload"])
        reports.append(
            {
                "name": fixture["name"],
                "expected_valid": fixture["expected_valid"],
                "actual_valid": parsed["valid"],
                "fixture_pass": bool(parsed["valid"] == fixture["expected_valid"]),
                "errors": parsed["errors"],
                "clean_planning_evidence": parsed["clean_planning_evidence"],
            }
        )
    return {
        "artifact": "qwen37_structured_anchor_parser_contract_fixtures_20260601_v58",
        "current_stage": "v58_structured_anchor_contract_fixtures",
        "model_name": MODEL_NAME,
        "required_fields": list(STRUCTURED_ANCHOR_FIELDS),
        "final_control_fields_rejected": list(FINAL_CONTROL_FIELDS),
        "fixture_reports": reports,
        "fixture_count": len(reports),
        "fixture_pass": all(report["fixture_pass"] for report in reports),
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def audit_existing_v56_empty_anchors(cache_path: str | Path = V56_CACHE_PATH) -> dict[str, Any]:
    entries = _plan_cache_entries(cache_path)
    empty_checked = 0
    clean_evidence = 0
    sample = []
    for key, entry in entries:
        action = entry.get("buffered_action", {}) if isinstance(entry, Mapping) else {}
        setpoints = entry.get("buffered_setpoints", {}) if isinstance(entry, Mapping) else {}
        if isinstance(action, Mapping) and isinstance(setpoints, Mapping):
            is_empty = not bool(action.get("action_set", False)) and not setpoints
        else:
            is_empty = True
        if not is_empty:
            continue
        parsed = parse_structured_anchor({})
        empty_checked += 1
        if parsed["clean_planning_evidence"]:
            clean_evidence += 1
        if len(sample) < 5:
            sample.append(
                {
                    "key": key,
                    "anchor_source": entry.get("anchor_source", "") if isinstance(entry, Mapping) else "",
                    "parsed_valid": parsed["valid"],
                    "errors": parsed["errors"],
                }
            )
    return {
        "cache_path": _rel(cache_path),
        "empty_anchor_entries_checked": empty_checked,
        "empty_anchor_clean_planning_evidence_count": clean_evidence,
        "empty_anchors_rejected": bool(empty_checked > 0 and clean_evidence == 0),
        "sample_rejected_empty_anchors": sample,
    }


def agent_config_structured_anchor_defaults() -> dict[str, Any]:
    cfg = AgentConfig()
    return {
        "structured_anchor_parser_enabled": bool(getattr(cfg, "structured_anchor_parser_enabled", True)),
        "structured_anchor_shadow_only": bool(getattr(cfg, "structured_anchor_shadow_only", False)),
        "structured_anchor_required_fields": str(getattr(cfg, "structured_anchor_required_fields", "")),
        "default_off": not bool(getattr(cfg, "structured_anchor_parser_enabled", True)),
        "shadow_only_default": bool(getattr(cfg, "structured_anchor_shadow_only", False)),
    }


def build_shadow_parser_audit(
    *,
    cache_path: str | Path = V56_CACHE_PATH,
    v57_readiness_json: str | Path = V57_READINESS_JSON,
) -> dict[str, Any]:
    fixtures = build_contract_fixtures()
    empty_anchor_audit = audit_existing_v56_empty_anchors(cache_path)
    defaults = agent_config_structured_anchor_defaults()
    v57_readiness = _load_json(v57_readiness_json)
    parser_ready = bool(
        fixtures["fixture_pass"]
        and empty_anchor_audit["empty_anchors_rejected"]
        and defaults["default_off"]
        and defaults["shadow_only_default"]
    )
    return {
        "artifact": "qwen37_structured_anchor_parser_shadow_audit_20260601_v58",
        "current_stage": "v58_structured_anchor_opt_in_shadow_parser",
        "model_name": MODEL_NAME,
        "scope": "offline_parser_and_shadow_audit_only",
        "inputs": {
            "cache_path": _rel(cache_path),
            "v57_readiness_json": _rel(v57_readiness_json),
        },
        "v57_next_action": str(v57_readiness.get("next_action", "")),
        "parser_ready_for_opt_in_shadow_rollout_plan": parser_ready,
        "fixture_pass": bool(fixtures["fixture_pass"]),
        "empty_anchors_rejected": bool(empty_anchor_audit["empty_anchors_rejected"]),
        "agent_config_defaults": defaults,
        "empty_anchor_audit": empty_anchor_audit,
        "structured_anchor_result_semantics": {
            "valid_anchor_outputs_shadow_plan_only": True,
            "invalid_anchor_is_not_clean_planning_evidence": True,
            "final_control_generation_allowed": False,
            "default_llm_rspc_v2_path_changed": False,
        },
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def build_readiness(parser_audit: Mapping[str, Any], fixture_artifact: Mapping[str, Any]) -> dict[str, Any]:
    passed = bool(
        parser_audit.get("parser_ready_for_opt_in_shadow_rollout_plan", False)
        and fixture_artifact.get("fixture_pass", False)
    )
    return {
        "artifact": "metadata_replay_readiness_checklist_20260601_v58",
        "current_stage": "v58_structured_anchor_opt_in_shadow_parser",
        "model_name": MODEL_NAME,
        "structured_anchor_parser_helper_done": True,
        "structured_anchor_contract_fixtures_done": True,
        "structured_anchor_parser_shadow_audit_done": True,
        "structured_anchor_parser_ready": passed,
        "default_controller_changed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": (
            "structured_anchor_opt_in_qwen37_shadow_rollout_plan"
            if passed
            else "structured_anchor_parser_fix_plan"
        ),
        "stop_taxonomy": [] if passed else ["structured_anchor_parser_not_ready"],
    }


def write_all() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fixtures = build_contract_fixtures()
    audit = build_shadow_parser_audit()
    readiness = build_readiness(audit, fixtures)
    _write_json_md(FIXTURES_JSON, FIXTURES_MD, fixtures, "v58 Structured Anchor Parser Contract Fixtures")
    _write_json_md(PARSER_AUDIT_JSON, PARSER_AUDIT_MD, audit, "v58 Qwen3.7 Structured Anchor Parser Shadow Audit")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v58 Metadata Replay Readiness")
    return fixtures, audit, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-all", action="store_true", help="Write all v58 JSON/MD artifacts.")
    return parser.parse_args()


def main() -> None:
    _parse_args()
    fixtures, audit, readiness = write_all()
    print(
        json.dumps(
            {
                "artifact": audit["artifact"],
                "fixture_pass": fixtures["fixture_pass"],
                "empty_anchors_rejected": audit["empty_anchors_rejected"],
                "structured_anchor_parser_ready": readiness["structured_anchor_parser_ready"],
                "next_action": readiness["next_action"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
