"""Build v79 control-chain trajectory architecture redesign artifacts.

This stage is offline and design-only. It turns the v77/v78 runtime-stability
evidence into a control-chain architecture contract: state/history and weather
context should produce smooth action-envelope trajectories, scored with
stability and Tomato Safety projection in the loop. It does not call an online
LLM, run rollout or replay, change final actions, enable a runtime supervisor,
execute solver fallback, or make performance/safety claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments import qwen37plus_stateful_heat_dry_accumulation_attribution_v78 as v78  # noqa: E402
from gl_gym.experiments import qwen37plus_runtime_stability_architecture_diagnosis_v77 as v77  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import AUDIT_DIR  # noqa: E402


MODEL_NAME = "qwen3.7-plus"
ARTIFACT_DATE = "20260605"
VERSION = "v79"

V75_TRACE_DIR = v78.V75_TRACE_DIR
V78_ATTRIBUTION_JSON = v78.ATTRIBUTION_JSON
V78_READINESS_JSON = v78.READINESS_JSON
V77_DESIGN_JSON = v77.DESIGN_JSON

GAP_JSON = AUDIT_DIR / "qwen37plus_control_chain_gap_audit_20260605_v79.json"
GAP_MD = AUDIT_DIR / "qwen37plus_control_chain_gap_audit_20260605_v79.md"
CONTRACT_JSON = AUDIT_DIR / "trajectory_action_envelope_contract_20260605_v79.json"
CONTRACT_MD = AUDIT_DIR / "trajectory_action_envelope_contract_20260605_v79.md"
READINESS_JSON = AUDIT_DIR / "control_chain_trajectory_architecture_readiness_20260605_v79.json"
READINESS_MD = AUDIT_DIR / "control_chain_trajectory_architecture_readiness_20260605_v79.md"

ACTION_FIELDS = (
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
)
STABILITY_METRIC_FIELDS = (
    "heat_debt",
    "dryness_debt",
    "vpd_debt",
    "vpd_ramp",
    "action_total_variation",
    "action_reversal_count",
    "tomato_rewrite_pressure",
    "solver_sensitive_risk",
)
BOUNDARY_FALSE_FIELDS = (
    "online_llm_called",
    "new_rollout_run",
    "default_llm_rspc_v2_changed",
    "fallback_enhanced",
    "controlled_replay_allowed",
    "controlled_replay_execution_allowed",
    "metadata_replay_execution_allowed",
    "strict_replay_allowed",
    "performance_claim_allowed",
    "promotion_evidence",
    "final_action_changed",
    "solver_fallback_executed",
    "runtime_policy_enabled",
    "runtime_supervisor_enabled",
    "trajectory_controller_enabled",
    "rollout_command_generated",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _rel(path: str | Path) -> str:
    p = _resolve(path)
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
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


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    p = _resolve(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _discover_trace_paths(trace_dir: str | Path) -> list[Path]:
    root = _resolve(trace_dir)
    if not root.exists():
        return []
    return sorted(root.glob("*_llm_rspc_v2.csv"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        result = float(value)
        return result if math.isfinite(result) else float(default)
    except Exception:
        return float(default)


def _maybe_num(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except Exception:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "applied"}


def _trace_identity(path: str | Path) -> dict[str, Any]:
    stem = Path(path).stem
    scenario_id = stem
    controller = ""
    for suffix in ("_llm_rspc_v2", "_ppo", "_rule_based"):
        if stem.endswith(suffix):
            scenario_id = stem[: -len(suffix)]
            controller = suffix[1:]
            break
    match = re.search(r"y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)(?:_n(?P<max_steps>\d+))?", scenario_id)
    return {
        "trace_id": stem,
        "scenario_id": scenario_id,
        "controller": controller,
        "year": int(match.group("year")) if match else 0,
        "day": int(match.group("day")) if match else 0,
        "seed": int(match.group("seed")) if match else 0,
        "max_steps": int(match.group("max_steps") or 720) if match else 720,
    }


def _runtime_failure_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    for row in rows:
        if str(row.get("runtime_error", "") or "").strip() or str(row.get("runtime_error_type", "") or "").strip():
            return int(_num(row.get("step"), -1))
    return None


def _action_sequence_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    previous: dict[str, float] = {}
    previous_signs: dict[str, int] = {}
    total_variation: dict[str, float] = {field: 0.0 for field in ACTION_FIELDS}
    max_delta: dict[str, float] = {field: 0.0 for field in ACTION_FIELDS}
    reversal_counts: Counter[str] = Counter()
    large_delta_counts: Counter[str] = Counter()
    for row in rows:
        for field in ACTION_FIELDS:
            value = _maybe_num(row.get(field))
            if value is None:
                continue
            if field in previous:
                delta = value - previous[field]
                abs_delta = abs(delta)
                total_variation[field] += abs_delta
                max_delta[field] = max(max_delta[field], abs_delta)
                if abs_delta > 0.20:
                    large_delta_counts[field] += 1
                sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
                if sign and previous_signs.get(field, 0) and sign != previous_signs[field]:
                    reversal_counts[field] += 1
                if sign:
                    previous_signs[field] = sign
            previous[field] = value
    return {
        "action_total_variation_by_field": {field: float(value) for field, value in sorted(total_variation.items())},
        "action_total_variation": float(sum(total_variation.values())),
        "max_delta_by_field": {field: float(value) for field, value in sorted(max_delta.items())},
        "max_delta": float(max(max_delta.values()) if max_delta else 0.0),
        "action_reversal_count_by_field": dict(sorted(reversal_counts.items())),
        "action_reversal_count": int(sum(reversal_counts.values())),
        "large_action_delta_count_by_field": dict(sorted(large_delta_counts.items())),
        "large_action_delta_count": int(sum(large_delta_counts.values())),
    }


def _tomato_rewrite_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    channels = ("heat", "screen", "shade", "vent")
    rewrite_steps = 0
    rewrite_delta_sum = 0.0
    reason_counts: Counter[str] = Counter()
    for row in rows:
        changed = False
        for channel in channels:
            before = _maybe_num(row.get(f"tomato_safety_v2_{channel}_before"))
            after = _maybe_num(row.get(f"tomato_safety_v2_{channel}_after"))
            if before is not None and after is not None:
                delta = abs(after - before)
                rewrite_delta_sum += delta
                if delta > 1e-6:
                    changed = True
        if _truthy(row.get("tomato_safety_v2_applied")):
            changed = True
        if changed:
            rewrite_steps += 1
        for reason in re.split(r"[,;|]", str(row.get("tomato_safety_v2_reasons", "") or row.get("final_action_risk_reason_v2", "") or "")):
            reason = reason.strip()
            if reason:
                reason_counts[reason] += 1
    return {
        "tomato_rewrite_steps": int(rewrite_steps),
        "tomato_rewrite_rate": float(rewrite_steps / max(len(rows), 1)),
        "tomato_rewrite_delta_sum": float(rewrite_delta_sum),
        "tomato_rewrite_reason_counts": dict(sorted(reason_counts.items())),
    }


def _debt_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    heat_debt = sum(max(0.0, _num(row.get("temp_air")) - 28.0) for row in rows)
    dryness_debt = sum(max(0.0, 55.0 - _num(row.get("rh_air"), 100.0)) for row in rows)
    vpd_debt = sum(max(0.0, _num(row.get("vpd_air")) - 2.0) for row in rows)
    values = [_num(row.get("vpd_air")) for row in rows if row.get("vpd_air") not in (None, "")]
    vpd_ramp = values[-1] - values[0] if len(values) >= 2 else 0.0
    return {
        "heat_debt": float(heat_debt),
        "dryness_debt": float(dryness_debt),
        "vpd_debt": float(vpd_debt),
        "vpd_ramp": float(vpd_ramp),
    }


def _trace_gap_report(path: str | Path) -> dict[str, Any]:
    rows = _read_rows(path)
    identity = _trace_identity(path)
    failure_step = _runtime_failure_step(rows)
    action = _action_sequence_metrics(rows)
    tomato = _tomato_rewrite_metrics(rows)
    debt = _debt_metrics(rows)
    return {
        **identity,
        "path": _rel(path),
        "row_count": int(len(rows)),
        "failure_step": failure_step,
        "short_trajectory_before_max_steps": bool(rows and len(rows) < int(identity.get("max_steps", 720))),
        "action_sequence_metrics": action,
        "tomato_projection_metrics": tomato,
        "debt_metrics": debt,
        "step_local_chain_gap_signals": {
            "short_trajectory": bool(rows and len(rows) < int(identity.get("max_steps", 720))),
            "high_action_total_variation": bool(action["action_total_variation"] > 20.0),
            "high_reversal_count": bool(action["action_reversal_count"] > 40),
            "large_tomato_rewrite_pressure": bool(tomato["tomato_rewrite_steps"] > max(10, len(rows) // 5)),
            "heat_dry_vpd_debt_present": bool(debt["heat_debt"] > 50.0 and debt["dryness_debt"] > 50.0 and debt["vpd_debt"] > 5.0),
        },
    }


def _aggregate_gap_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not reports:
        return {
            "trace_count": 0,
            "short_trajectory_count": 0,
            "mean_action_total_variation": 0.0,
            "mean_action_reversal_count": 0.0,
            "mean_tomato_rewrite_steps": 0.0,
            "mean_heat_debt": 0.0,
            "mean_dryness_debt": 0.0,
            "mean_vpd_debt": 0.0,
        }
    count = len(reports)
    return {
        "trace_count": int(count),
        "short_trajectory_count": int(sum(bool(report.get("short_trajectory_before_max_steps")) for report in reports)),
        "mean_action_total_variation": float(
            sum(float((report.get("action_sequence_metrics", {}) or {}).get("action_total_variation", 0.0)) for report in reports) / count
        ),
        "mean_action_reversal_count": float(
            sum(float((report.get("action_sequence_metrics", {}) or {}).get("action_reversal_count", 0.0)) for report in reports) / count
        ),
        "mean_tomato_rewrite_steps": float(
            sum(float((report.get("tomato_projection_metrics", {}) or {}).get("tomato_rewrite_steps", 0.0)) for report in reports) / count
        ),
        "mean_heat_debt": float(sum(float((report.get("debt_metrics", {}) or {}).get("heat_debt", 0.0)) for report in reports) / count),
        "mean_dryness_debt": float(sum(float((report.get("debt_metrics", {}) or {}).get("dryness_debt", 0.0)) for report in reports) / count),
        "mean_vpd_debt": float(sum(float((report.get("debt_metrics", {}) or {}).get("vpd_debt", 0.0)) for report in reports) / count),
    }


def build_control_chain_gap_audit(
    *,
    trace_dir: str | Path = V75_TRACE_DIR,
    v78_attribution_json: str | Path = V78_ATTRIBUTION_JSON,
    v78_readiness_json: str | Path = V78_READINESS_JSON,
) -> dict[str, Any]:
    reports = [_trace_gap_report(path) for path in _discover_trace_paths(trace_dir)]
    aggregate = _aggregate_gap_reports(reports)
    v78_attribution = _load_json(v78_attribution_json)
    v78_readiness = _load_json(v78_readiness_json)
    gap = {
        "artifact": "qwen37plus_control_chain_gap_audit_20260605_v79",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "scope": "offline_gap_audit_for_step_local_control_chain_vs_trajectory_control_chain",
        "trace_dir": _rel(trace_dir),
        "source_v78_attribution_artifact": v78_attribution.get("artifact", ""),
        "source_v78_dominant_accumulation_attribution": v78_attribution.get("dominant_accumulation_attribution", ""),
        "source_v78_readiness_next_action": v78_readiness.get("next_action", ""),
        "old_hypothesis": "Step-local candidate selection plus post-hoc Tomato Safety rewrite is sufficient for stable long-horizon control.",
        "observed_gap": (
            "v78 found mixed weather/control accumulation in all failure traces; action variation, reversals, "
            "Tomato Safety rewrite pressure, and heat/dry/VPD debt must be evaluated as trajectory-level effects."
        ),
        "step_local_chain_insufficient": bool(v78_attribution.get("dominant_accumulation_attribution") == "mixed_weather_control_accumulation"),
        "scenario_reports": reports,
        "aggregate_metrics": aggregate,
        "required_architecture_shift": "state_history_to_trajectory_action_envelope_chain",
        "next_action": "trajectory_action_envelope_shadow_design_plan",
    }
    return _with_boundaries(gap)


def build_trajectory_action_envelope_contract(gap_audit: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "artifact": "trajectory_action_envelope_contract_20260605_v79",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "scope": "design_only_contract_for_state_history_to_trajectory_action_envelope",
        "candidate_source": "normal_path_trajectory_action_envelope_shadow",
        "public_output_name_template": "trajectory_action_envelope:<regime_name>:h<horizon_steps>",
        "default_horizon_steps": 24,
        "input_contract": {
            "required_inputs": [
                "recent_state_history",
                "weather_forecast",
                "profile_intent_or_regime_priority",
                "previous_action",
                "tomato_safety_projection_model",
                "current_action_envelope_bounds",
            ],
            "history_metrics": list(STABILITY_METRIC_FIELDS),
        },
        "trajectory_output_contract": {
            "required_fields": [
                "name",
                "candidate_source",
                "horizon_steps",
                "regime_name",
                "trajectory_objective",
                "action_envelope_sequence",
                "preferred_direction_sequence",
                "smoothness_constraints",
                "tomato_safety_projection_sequence",
                "projected_action_sequence",
                "score_terms",
                "eligible",
                "rejection_reason",
                "compatibility_category",
            ],
            "action_fields": list(ACTION_FIELDS),
            "per_step_envelope_shape": {"min": "float", "max": "float", "preferred_direction": "increase/decrease/hold/any"},
        },
        "smoothness_constraints": {
            "max_delta_per_step_required": True,
            "direction_reversal_penalty_required": True,
            "total_variation_penalty_required": True,
            "screen_vent_shade_joint_smoothness_required": True,
        },
        "stability_aware_score_terms": [
            "target_tracking_proxy",
            "heat_debt_reduction_proxy",
            "dryness_debt_reduction_proxy",
            "vpd_ramp_penalty",
            "action_total_variation_penalty",
            "action_reversal_penalty",
            "tomato_rewrite_pressure_penalty",
            "solver_sensitive_risk_penalty",
            "tomato_projection_validity",
            "selection_score",
        ],
        "tomato_safety_in_loop": {
            "projection_before_scoring_required": True,
            "post_projection_score_required": True,
            "final_shield_still_required": True,
            "rewrite_magnitude_penalized_before_arbitration": True,
        },
        "architecture_flow": [
            "state_history_and_weather_forecast",
            "regime_risk_forecaster",
            "trajectory_objective_composer",
            "smooth_action_envelope_sequence_composer",
            "tomato_safety_projection_in_loop",
            "stability_aware_scorer",
            "final_safety_shield",
        ],
        "paper_innovation_candidates": [
            "LLM-guided regime and priority generation without direct actuator control",
            "State-history-to-trajectory action envelope generation for greenhouse climate control",
            "Tomato Safety projection-in-the-loop scoring with smoothness and solver-sensitive risk penalties",
        ],
        "source_gap_artifact": gap_audit.get("artifact", ""),
        "next_action": "trajectory_action_envelope_shadow_instrumentation_plan",
    }
    return _with_boundaries(contract)


def build_readiness(gap_audit: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    required = set((contract.get("trajectory_output_contract", {}) or {}).get("required_fields", []) or [])
    contract_complete = {
        "trajectory_output_required_fields_present": bool(
            {
                "action_envelope_sequence",
                "smoothness_constraints",
                "tomato_safety_projection_sequence",
                "projected_action_sequence",
                "score_terms",
            }.issubset(required)
        ),
        "stability_score_terms_present": bool(len(contract.get("stability_aware_score_terms", []) or []) >= 8),
        "tomato_projection_in_loop_present": bool((contract.get("tomato_safety_in_loop", {}) or {}).get("projection_before_scoring_required", False)),
        "paper_innovation_candidates_present": bool(len(contract.get("paper_innovation_candidates", []) or []) >= 3),
    }
    ready = bool(all(contract_complete.values()) and gap_audit.get("step_local_chain_insufficient"))
    readiness = {
        "artifact": "control_chain_trajectory_architecture_readiness_20260605_v79",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "source_gap_artifact": gap_audit.get("artifact", ""),
        "source_contract_artifact": contract.get("artifact", ""),
        "step_local_chain_insufficient": bool(gap_audit.get("step_local_chain_insufficient", False)),
        "contract_complete": contract_complete,
        "architecture_hypothesis_ready_for_shadow_instrumentation": ready,
        "default_controller_changed": False,
        "online_llm_needed_next": False,
        "minimum_next_evidence_level": "unit/synthetic test -> existing-trace audit -> opt-in shadow trace",
        "recommended_followup": (
            "trajectory_action_envelope_shadow_instrumentation_plan"
            if ready
            else "trajectory_action_envelope_contract_repair_plan"
        ),
        "next_action": (
            "trajectory_action_envelope_shadow_instrumentation_plan"
            if ready
            else "trajectory_action_envelope_contract_repair_plan"
        ),
    }
    return _with_boundaries(readiness)


def build_gap_report(gap: Mapping[str, Any]) -> str:
    aggregate = gap.get("aggregate_metrics", {}) or {}
    lines = [
        "# qwen3.7-plus v79 Control-Chain Gap Audit",
        "",
        f"- source_v78_dominant_accumulation_attribution={gap.get('source_v78_dominant_accumulation_attribution', '')}",
        f"- step_local_chain_insufficient={bool(gap.get('step_local_chain_insufficient', False))}",
        f"- trace_count={int(aggregate.get('trace_count', 0) or 0)}",
        f"- short_trajectory_count={int(aggregate.get('short_trajectory_count', 0) or 0)}",
        f"- mean_action_total_variation={float(aggregate.get('mean_action_total_variation', 0.0) or 0.0):.3f}",
        f"- mean_action_reversal_count={float(aggregate.get('mean_action_reversal_count', 0.0) or 0.0):.3f}",
        f"- mean_tomato_rewrite_steps={float(aggregate.get('mean_tomato_rewrite_steps', 0.0) or 0.0):.3f}",
        f"- next_action={gap.get('next_action', '')}",
        f"- performance_claim_allowed={bool(gap.get('performance_claim_allowed', False))}",
        f"- promotion_evidence={bool(gap.get('promotion_evidence', False))}",
    ]
    return "\n".join(lines) + "\n"


def build_contract_report(contract: Mapping[str, Any]) -> str:
    lines = [
        "# v79 Trajectory Action Envelope Contract",
        "",
        f"- candidate_source={contract.get('candidate_source', '')}",
        f"- public_output_name_template={contract.get('public_output_name_template', '')}",
        f"- default_horizon_steps={int(contract.get('default_horizon_steps', 0) or 0)}",
        f"- tomato_projection_before_scoring={bool((contract.get('tomato_safety_in_loop', {}) or {}).get('projection_before_scoring_required', False))}",
        f"- final_shield_still_required={bool((contract.get('tomato_safety_in_loop', {}) or {}).get('final_shield_still_required', False))}",
        f"- paper_innovation_candidate_count={len(contract.get('paper_innovation_candidates', []) or [])}",
        f"- trajectory_controller_enabled={bool(contract.get('trajectory_controller_enabled', False))}",
        f"- next_action={contract.get('next_action', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_readiness_report(readiness: Mapping[str, Any]) -> str:
    lines = [
        "# v79 Control-Chain Trajectory Architecture Readiness",
        "",
        f"- step_local_chain_insufficient={bool(readiness.get('step_local_chain_insufficient', False))}",
        f"- architecture_hypothesis_ready_for_shadow_instrumentation={bool(readiness.get('architecture_hypothesis_ready_for_shadow_instrumentation', False))}",
        f"- default_controller_changed={bool(readiness.get('default_controller_changed', False))}",
        f"- online_llm_needed_next={bool(readiness.get('online_llm_needed_next', False))}",
        f"- final_action_changed={bool(readiness.get('final_action_changed', False))}",
        f"- performance_claim_allowed={bool(readiness.get('performance_claim_allowed', False))}",
        f"- promotion_evidence={bool(readiness.get('promotion_evidence', False))}",
        f"- recommended_followup={readiness.get('recommended_followup', '')}",
        f"- next_action={readiness.get('next_action', '')}",
    ]
    return "\n".join(lines) + "\n"


def write_all(
    *,
    trace_dir: str | Path = V75_TRACE_DIR,
    v78_attribution_json: str | Path = V78_ATTRIBUTION_JSON,
    v78_readiness_json: str | Path = V78_READINESS_JSON,
    gap_json: str | Path = GAP_JSON,
    gap_md: str | Path = GAP_MD,
    contract_json: str | Path = CONTRACT_JSON,
    contract_md: str | Path = CONTRACT_MD,
    readiness_json: str | Path = READINESS_JSON,
    readiness_md: str | Path = READINESS_MD,
) -> dict[str, str]:
    gap = build_control_chain_gap_audit(
        trace_dir=trace_dir,
        v78_attribution_json=v78_attribution_json,
        v78_readiness_json=v78_readiness_json,
    )
    contract = build_trajectory_action_envelope_contract(gap)
    readiness = build_readiness(gap, contract)
    outputs = {
        "gap_json": Path(gap_json),
        "gap_md": Path(gap_md),
        "contract_json": Path(contract_json),
        "contract_md": Path(contract_md),
        "readiness_json": Path(readiness_json),
        "readiness_md": Path(readiness_md),
    }
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    outputs["gap_json"].write_text(json.dumps(gap, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    outputs["gap_md"].write_text(build_gap_report(gap), encoding="utf-8")
    outputs["contract_json"].write_text(
        json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    outputs["contract_md"].write_text(build_contract_report(contract), encoding="utf-8")
    outputs["readiness_json"].write_text(
        json.dumps(readiness, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    outputs["readiness_md"].write_text(build_readiness_report(readiness), encoding="utf-8")
    return {key: str(path) for key, path in outputs.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", default=str(V75_TRACE_DIR))
    parser.add_argument("--v78-attribution-json", default=str(V78_ATTRIBUTION_JSON))
    parser.add_argument("--v78-readiness-json", default=str(V78_READINESS_JSON))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    gap = build_control_chain_gap_audit(
        trace_dir=args.trace_dir,
        v78_attribution_json=args.v78_attribution_json,
        v78_readiness_json=args.v78_readiness_json,
    )
    contract = build_trajectory_action_envelope_contract(gap)
    readiness = build_readiness(gap, contract)
    if args.no_write:
        print(json.dumps({"gap": gap, "contract": contract, "readiness": readiness}, indent=2, ensure_ascii=False))
        return 0
    print(
        json.dumps(
            write_all(
                trace_dir=args.trace_dir,
                v78_attribution_json=args.v78_attribution_json,
                v78_readiness_json=args.v78_readiness_json,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
