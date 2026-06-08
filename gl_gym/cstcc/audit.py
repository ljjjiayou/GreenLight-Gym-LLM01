"""C-STCC v80 audit record builder.

This module assembles the contract-level pieces into a deterministic shadow
audit record. It does not execute predictive rollout, online LLM, MPC, or
runtime action changes.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import load_config
from .constraints import DEFAULT_MAX_DELTA, check_hard_constraints, soft_penalties
from .contracts import (
    ACTION_FIELDS,
    CSTCCAuditRecord,
    PlanMemory,
    PriorCandidate,
    SafetyProjectionReport,
    SemanticSuggestion,
    SequenceCandidate,
    asdict_clean,
    normalize_action,
)
from .dual_scorer import score_dual
from .prior_confidence import (
    build_prior_candidate,
    conservative_prior_confidence,
    candidate_count_from_confidence,
    ppo_prior_confidence,
    previous_plan_confidence,
)
from .regime_evidence import regime_evidence_score
from .semantic_state_machine import evaluate_transition
from .sequence_templates import generate_sequence, generate_sequence_with_metadata
from .temporal_context import action_smoothness_summary, build_temporal_context


def _last_action(action_history: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not action_history:
        return normalize_action({})
    return normalize_action(action_history[-1])


def _violation_distributions(violations: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    reasons: dict[str, int] = {}
    fields: dict[str, int] = {}
    for violation in violations:
        reason = str(
            violation.get("reason")
            or violation.get("violation")
            or violation.get("constraint")
            or violation.get("type")
            or "unknown"
        )
        field = str(violation.get("field") or violation.get("fields") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
        fields[field] = fields.get(field, 0) + 1
    return dict(sorted(reasons.items())), dict(sorted(fields.items()))


def _first_action_delta(sequence: Sequence[Mapping[str, Any]], reference_action: Mapping[str, Any]) -> dict[str, float]:
    first = normalize_action(sequence[0]) if sequence else normalize_action({})
    reference = normalize_action(reference_action)
    return {field: first[field] - reference[field] for field in ACTION_FIELDS}


def _max_action_delta(lhs: Mapping[str, Any] | None, rhs: Mapping[str, Any] | None) -> float:
    if lhs is None or rhs is None:
        return 0.0
    left = normalize_action(lhs)
    right = normalize_action(rhs)
    return max(abs(left[field] - right[field]) for field in ACTION_FIELDS)


def _risk_flags(state: Mapping[str, Any], context_dict: Mapping[str, Any]) -> dict[str, bool]:
    debts = context_dict.get("debt_features", {}) if isinstance(context_dict.get("debt_features", {}), Mapping) else {}
    try:
        rh_air = float(state.get("rh_air", 70.0) or 70.0)
    except Exception:
        rh_air = 70.0
    try:
        vpd_air = float(state.get("vpd_air", state.get("vpd_kpa", 0.0)) or 0.0)
    except Exception:
        vpd_air = 0.0
    try:
        canopy_dew_margin = float(state.get("canopy_dew_margin", 999.0) or 999.0)
    except Exception:
        canopy_dew_margin = 999.0
    try:
        dew_margin_air = float(state.get("dew_margin_air", 999.0) or 999.0)
    except Exception:
        dew_margin_air = 999.0
    vpd_debt = float(debts.get("vpd_debt_norm", 0.0) or 0.0)
    dryness_debt = float(debts.get("dryness_debt_norm", 0.0) or 0.0)
    flags = {
        "rh_ge_90": rh_air >= 90.0,
        "dew_risk": canopy_dew_margin < 1.0 or dew_margin_air < 1.0,
        "dry_risk": rh_air < 55.0 or dryness_debt > 0.50,
        "high_vpd": vpd_air > 1.20 or vpd_debt > 0.50,
    }
    flags["humidity_or_dew_risk"] = flags["rh_ge_90"] or flags["dew_risk"]
    flags["dry_or_high_vpd_risk"] = flags["dry_risk"] or flags["high_vpd"]
    flags["any_risk"] = any(flags.values())
    return flags


def _risk_category_flags(
    risk_flags: Mapping[str, Any],
    context_dict: Mapping[str, Any],
    reference_action: Mapping[str, Any],
) -> dict[str, bool]:
    solver = context_dict.get("solver_risk_features", {})
    if not isinstance(solver, Mapping):
        solver = {}
    action = normalize_action(reference_action)
    solver_sensitive = bool(
        float(solver.get("solver_warning_score", 0.0) or 0.0) > 0.0
        or float(solver.get("hot_dry_solver_sensitive_score", 0.0) or 0.0) > 0.0
    )
    co2_vent_conflict = bool(action["u_co2"] > 0.50 and action["u_ventilation"] > 0.75)
    categories = {
        "humidity_or_dew_risk": bool(risk_flags.get("humidity_or_dew_risk", False)),
        "dry_or_high_vpd_risk": bool(risk_flags.get("dry_or_high_vpd_risk", False)),
        "solver_sensitive_risk": solver_sensitive,
        "co2_ventilation_conflict": co2_vent_conflict,
    }
    categories["any_risk"] = bool(risk_flags.get("any_risk", False) or any(categories.values()))
    return categories


def _state_snapshot(state_history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return dict(state_history[-1]) if state_history else {}


def _weather_summary(weather_history: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    if not weather_history:
        return {"available": False}
    return {"available": True, "row_count": len(weather_history), "last": dict(weather_history[-1])}


def _validate_inline_config(cfg: Mapping[str, Any]) -> None:
    if cfg.get("version") != "v80":
        raise ValueError("C-STCC config must declare version: v80")
    if cfg.get("online_llm_enabled", False):
        raise ValueError("v80 config must not enable online LLM")
    if cfg.get("controller_changed", False) or cfg.get("final_action_changed", False) or cfg.get("default_controller_changed", False):
        raise ValueError("v80 config must not change controller, default controller, or final action")
    if int(cfg.get("prediction_level", 0)) != 0:
        raise ValueError("v80 config must keep prediction_level=0")


def _candidate_templates_for_prior(prior: PriorCandidate) -> list[str]:
    if prior.source == "ppo_prior":
        return ["ppo_follow_limited", "hold_all", "ventilation_ramp_limited", "heating_smooth_recover"]
    if prior.source == "previous_plan_prior":
        return ["previous_shift_all", "hold_all", "screen_dwell_hold", "ventilation_ramp_limited"]
    if prior.source == "conservative_prior":
        return ["conservative_stabilize", "hold_all", "co2_vent_conflict_reduce", "heating_smooth_recover"]
    if prior.source == "rule_prior":
        return ["hold_all", "ventilation_ramp_limited", "co2_vent_conflict_reduce", "screen_dwell_hold"]
    return ["hold_all"]


def _candidate_count_threshold(count: int, bins: Sequence[Sequence[Any]]) -> float:
    parsed: list[tuple[float, int]] = []
    for item in bins:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            parsed.append((float(item[0]), int(item[1])))
        except Exception:
            continue
    if not parsed:
        parsed = [(0.0, 0), (0.10, 1), (0.30, 2), (0.60, 3), (0.85, 4)]
    parsed.sort(key=lambda pair: pair[0])
    for threshold, candidate_count in parsed:
        if candidate_count == int(count):
            return threshold
    if int(count) <= 0:
        positive = [threshold for threshold, candidate_count in parsed if candidate_count > 0]
        return min(positive) if positive else 0.10
    candidates = [threshold for threshold, candidate_count in parsed if candidate_count <= int(count)]
    return max(candidates) if candidates else 0.0


def _first_candidate_threshold(bins: Sequence[Sequence[Any]]) -> float:
    parsed: list[float] = []
    for item in bins:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            if int(item[1]) > 0:
                parsed.append(float(item[0]))
        except Exception:
            continue
    return min(parsed) if parsed else 0.10


def _prior_no_candidate_reason(
    prior: PriorCandidate,
    *,
    effective_candidate_count: int,
    available_template_count: int,
    generated_template_count: int,
    max_candidates_exhausted: bool,
    risk_flags: Mapping[str, Any],
    candidate_count_threshold: float,
    first_candidate_threshold: float,
) -> tuple[str | None, list[str]]:
    if generated_template_count > 0:
        return None, []
    reasons: list[str] = []
    if available_template_count <= 0:
        reasons.append("no_template_available")
    if max_candidates_exhausted:
        reasons.append("max_candidates_exhausted")
    if effective_candidate_count <= 0 and float(prior.confidence) < float(first_candidate_threshold):
        reasons.append("confidence_below_candidate_threshold")
    if (
        prior.source == "conservative_prior"
        and effective_candidate_count <= 0
        and not bool(risk_flags.get("any_risk", False))
    ):
        reasons.append("risk_trigger_missing_or_low")
    if effective_candidate_count <= 0 and not reasons:
        reasons.append("confidence_below_candidate_threshold")
    if not reasons and effective_candidate_count > 0 and generated_template_count <= 0:
        reasons.append("max_candidates_exhausted" if max_candidates_exhausted else "not_generated_unknown")
    if "confidence_below_candidate_threshold" in reasons:
        primary = "confidence_below_candidate_threshold"
    elif reasons:
        primary = reasons[0]
    else:
        primary = None
    return primary, reasons


def _allocation_flag_active(flag: str, risk_flags: Mapping[str, Any], category_flags: Mapping[str, Any]) -> bool:
    name = str(flag or "")
    if not name:
        return False
    if name == "any_risk":
        return bool(risk_flags.get("any_risk", False) or category_flags.get("any_risk", False))
    if name in risk_flags:
        return bool(risk_flags.get(name, False))
    return bool(category_flags.get(name, False))


def _candidate_allocation_diagnostic(
    prior: PriorCandidate,
    *,
    risk_flags: Mapping[str, Any],
    category_flags: Mapping[str, Any],
    allocation_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    before = int(prior.candidate_count)
    after = before
    override_applied = False
    override_reason = None
    override_flags: list[str] = []

    conservative_cfg = allocation_cfg.get("conservative_risk_candidate_floor", {})
    if not isinstance(conservative_cfg, Mapping):
        conservative_cfg = {}
    enabled = bool(conservative_cfg.get("enabled", False))
    floor_count = int(conservative_cfg.get("floor_count", 1) or 1)
    trigger_flags = conservative_cfg.get("trigger_flags", ["any_risk"])
    if not isinstance(trigger_flags, Sequence) or isinstance(trigger_flags, (str, bytes)):
        trigger_flags = ["any_risk"]
    active_flags = [
        str(flag)
        for flag in trigger_flags
        if _allocation_flag_active(str(flag), risk_flags, category_flags)
    ]
    if prior.source == "conservative_prior" and enabled and active_flags:
        after = max(before, max(1, floor_count))
        override_applied = after > before
        if override_applied:
            override_reason = "risk_triggered_conservative_floor"
            override_flags = active_flags
    elif prior.source == "conservative_prior" and not enabled and bool(category_flags.get("any_risk", False)):
        override_reason = "risk_triggered_candidate_floor_disabled_by_config"

    return {
        "candidate_count_before_override": before,
        "candidate_count_after_override": after,
        "effective_candidate_count": after,
        "candidate_count_override_applied": override_applied,
        "candidate_count_override_reason": override_reason,
        "candidate_count_override_flags": override_flags,
        "risk_category_flags": dict(category_flags),
    }


SCORE_COMPONENT_KEYS = (
    "base_final_score",
    "raw_smoothness",
    "raw_low_reversal",
    "raw_energy_proxy",
    "projected_smoothness",
    "projected_low_reversal",
    "projected_energy_proxy",
    "rewrite_penalty",
    "uncertainty_penalty",
    "prior_confidence_bonus",
    "risk_static_hold_penalty",
    "final_score",
)


def _score_components(score_breakdown: Mapping[str, Any]) -> dict[str, float]:
    normalized = score_breakdown.get("normalized_components") or {}
    if not isinstance(normalized, Mapping):
        normalized = {}
    return {
        "base_final_score": float(score_breakdown.get("base_final_score", 0.0) or 0.0),
        "raw_smoothness": float(normalized.get("raw_smoothness", 0.0) or 0.0),
        "raw_low_reversal": float(normalized.get("raw_low_reversal", 0.0) or 0.0),
        "raw_energy_proxy": float(normalized.get("raw_energy_proxy", 0.0) or 0.0),
        "projected_smoothness": float(normalized.get("projected_smoothness", 0.0) or 0.0),
        "projected_low_reversal": float(normalized.get("projected_low_reversal", 0.0) or 0.0),
        "projected_energy_proxy": float(normalized.get("projected_energy_proxy", 0.0) or 0.0),
        "rewrite_penalty": float(score_breakdown.get("rewrite_penalty", 0.0) or 0.0),
        "uncertainty_penalty": float(score_breakdown.get("uncertainty_penalty", 0.0) or 0.0),
        "prior_confidence_bonus": float(score_breakdown.get("prior_confidence_bonus", 0.0) or 0.0),
        "risk_static_hold_penalty": float(score_breakdown.get("risk_static_hold_penalty", 0.0) or 0.0),
        "final_score": float(score_breakdown.get("final_score", 0.0) or 0.0),
    }


def _selected_minus_competitor_components(
    selected_components: Mapping[str, float],
    competitor_components: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: float(selected_components.get(key, 0.0) - competitor_components.get(key, 0.0))
        for key in SCORE_COMPONENT_KEYS
    }


def _build_priors(
    last_action: Mapping[str, Any],
    *,
    ppo_action: Mapping[str, Any] | None,
    rule_action: Mapping[str, Any] | None,
    context_dict: Mapping[str, Any],
    active_regime: str,
    min_conf: float,
    max_conf: float,
) -> list[PriorCandidate]:
    debts = context_dict.get("debt_features", {})
    smooth = context_dict.get("action_smoothness_features", {})
    safety = context_dict.get("safety_pressure_features", {})
    solver = context_dict.get("solver_risk_features", {})
    solver_emergency = active_regime == "SOLVER_SENSITIVE_EMERGENCY"

    ppo_conf, ppo_reasons = ppo_prior_confidence(
        ood_score=float(solver.get("hot_dry_solver_sensitive_score", 0.0)),
        rewrite_pressure=float(safety.get("rewrite_pressure_norm", 0.0)),
        reversal_risk=float(smooth.get("reversal_count_norm", 0.0)),
        min_conf=min_conf,
        max_conf=max_conf,
        solver_sensitive_emergency=solver_emergency,
    )
    prev_conf, prev_reasons = previous_plan_confidence(
        forecast_error_norm=0.0,
        rewrite_pressure=float(safety.get("rewrite_pressure_norm", 0.0)),
        min_conf=min_conf,
        max_conf=max_conf,
        emergency_regime_shift=solver_emergency,
    )
    conservative_conf, conservative_reasons = conservative_prior_confidence(
        vpd_risk=float(debts.get("vpd_debt_norm", 0.0)),
        solver_risk=float(solver.get("hot_dry_solver_sensitive_score", 0.0)),
        crash_warning_score=float(solver.get("solver_warning_score", 0.0)),
        min_conf=min_conf,
        max_conf=max_conf,
    )
    rule_conf = 0.60
    return [
        build_prior_candidate("ppo_prior", ppo_action or last_action, ppo_conf, ppo_reasons),
        build_prior_candidate("previous_plan_prior", last_action, prev_conf, prev_reasons),
        build_prior_candidate("conservative_prior", last_action, conservative_conf, conservative_reasons),
        build_prior_candidate("rule_prior", rule_action or last_action, rule_conf, {"fixed_rule_confidence": rule_conf}),
    ]


def _project_sequence_placeholder(raw_sequence: list[dict[str, float]]) -> tuple[list[dict[str, float]], SafetyProjectionReport]:
    # v80 keeps this as shape-only projection provenance; real Tomato Safety
    # projection belongs to later runtime shadow stages.
    return [normalize_action(action) for action in raw_sequence], SafetyProjectionReport(
        rewrite_magnitude=0.0,
        rewrite_reasons=[],
        rewrite_reason_severity={},
        severe_rewrite_count=0,
        projected_safe=True,
        projection_level="placeholder_no_real_projection",
        projection_validity="placeholder_no_real_projection",
    )


def build_audit_record(
    *,
    state_history: Sequence[Mapping[str, Any]],
    action_history: Sequence[Mapping[str, Any]],
    weather_history: Sequence[Mapping[str, Any]] | None = None,
    semantic_suggestion: SemanticSuggestion | Mapping[str, Any] | None = None,
    current_runtime_final_action: Mapping[str, Any] | None = None,
    active_regime: str = "NORMAL_BALANCED",
    regime_age_steps: int = 0,
    previous_plan_sequence: Sequence[Mapping[str, Any]] | None = None,
    ppo_action: Mapping[str, Any] | None = None,
    rule_action: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> CSTCCAuditRecord:
    cfg = dict(config or load_config())
    _validate_inline_config(cfg)
    temporal_cfg = cfg.get("temporal_context", {})
    context = build_temporal_context(
        state_history,
        action_history,
        weather_history,
        delta_t_hours=float(temporal_cfg.get("delta_t_hours", 1.0 / 12.0)),
        gamma=float(temporal_cfg.get("gamma", 0.94)),
        debt_refs=temporal_cfg.get("debt_refs", {}),
    )
    context_dict = asdict_clean(context)
    data_quality = float(context_dict["data_quality_features"].get("data_quality_score", 0.0))

    if semantic_suggestion is None:
        suggestion = SemanticSuggestion(
            regime="NORMAL_BALANCED",
            regime_confidence=0.0,
            priority_weight_suggestions={},
            control_intent="mock_default_no_online_llm",
            risk_factors=[],
            uncertain_fields=["semantic_suggestion_missing"],
        )
    elif isinstance(semantic_suggestion, SemanticSuggestion):
        suggestion = semantic_suggestion
    else:
        suggestion = SemanticSuggestion(**dict(semantic_suggestion))

    runtime_final = normalize_action(current_runtime_final_action or _last_action(action_history))
    last = runtime_final
    evidence_conf, evidence_reasons = regime_evidence_score(suggestion.regime, context_dict, last)
    state_snapshot = _state_snapshot(state_history)
    risk_flags = _risk_flags(state_snapshot, context_dict)
    risk_category_flags = _risk_category_flags(risk_flags, context_dict, last)

    weight_cfg = cfg.get("weight_governor", {})
    semantic_state = evaluate_transition(
        active_regime,
        regime_age_steps,
        suggestion,
        evidence_confidence=evidence_conf,
        data_quality_score=data_quality,
        default_weights=weight_cfg.get("default_weights", {}),
        weight_bounds=weight_cfg.get("weight_bounds", {}),
        alpha_max=float(weight_cfg.get("alpha_max", 0.30)),
        confidence_mode=str(weight_cfg.get("confidence_mode", "min")),
    )

    prior_cfg = cfg.get("prior_confidence", {})
    candidate_count_bins = prior_cfg.get("candidate_count_bins", [])
    first_candidate_threshold = _first_candidate_threshold(candidate_count_bins)
    priors = _build_priors(
        last,
        ppo_action=ppo_action,
        rule_action=rule_action,
        context_dict=context_dict,
        active_regime=semantic_state.active_regime,
        min_conf=float(prior_cfg.get("min_conf", 0.05)),
        max_conf=float(prior_cfg.get("max_conf", 1.0)),
    )
    diagnostics_cfg = cfg.get("diagnostics", {})
    scorer_diagnostic_enabled = bool(
        diagnostics_cfg.get("scorer_diagnostic_enabled", diagnostics_cfg.get("scorer_component_audit_enabled", False))
    )
    scorer_diagnostic_log_mode = str(diagnostics_cfg.get("scorer_diagnostic_log_mode", "diagnostic_full") or "diagnostic_full")
    if scorer_diagnostic_log_mode not in {"diagnostic_full", "compact_scan"}:
        scorer_diagnostic_log_mode = "diagnostic_full"

    sequence_cfg = cfg.get("sequence_generation", {})
    horizon = int(sequence_cfg.get("horizon", 12))
    max_candidates = int(sequence_cfg.get("max_candidates", 20))
    allocation_cfg = cfg.get("candidate_allocation", {})
    if not isinstance(allocation_cfg, Mapping):
        allocation_cfg = {}
    constraint_cfg = cfg.get("constraints", {})
    configured_max_delta = constraint_cfg.get("max_delta", {}) if isinstance(constraint_cfg, Mapping) else {}
    max_delta_by_field = {
        field: float(configured_max_delta.get(field, DEFAULT_MAX_DELTA.get(field, 1.0)))
        for field in ACTION_FIELDS
    }
    scoring_cfg = cfg.get("scoring", {})
    prior_confidence_weight = float(scoring_cfg.get("prior_confidence_score_weight", 0.08))
    risk_static_hold_penalty = float(scoring_cfg.get("risk_static_hold_penalty", 0.25))
    risk_static_hold_divergence_threshold = float(
        scoring_cfg.get("risk_static_hold_divergence_threshold", 0.10)
    )
    rule_divergence = _max_action_delta(rule_action, last)
    ppo_divergence = _max_action_delta(ppo_action, last)
    scoring_metadata = {
        "prediction_level": int(cfg.get("prediction_level", 0)),
        "score_validity": "sequence_only_no_rollout",
        "predictive_rollout_executed": False,
        "surrogate_model_used": False,
        "mpc_optimizer_executed": False,
        "real_tomato_safety_projection": False,
        "tomato_safety_projection_level": "placeholder_no_real_projection",
        "final_action_changed": False,
        "selected_action_is_shadow_only": True,
        "regime_evidence_score": evidence_conf,
        "regime_evidence_reasons": evidence_reasons,
        "candidate_attribution_mode": "candidate_level_attribution",
        "constraint_reference_kind": "current_runtime_final_action",
        "constraint_reference_action": last,
        "hold_all_reference_kind": "current_runtime_final_action",
        "hold_all_reference": last,
        "risk_flags": risk_flags,
        "risk_category_flags": risk_category_flags,
        "prior_confidence_score_weight": prior_confidence_weight,
        "risk_static_hold_penalty": risk_static_hold_penalty,
        "risk_static_hold_divergence_threshold": risk_static_hold_divergence_threshold,
        "reference_action_delta_max_to_rule": rule_divergence,
        "reference_action_delta_max_to_ppo": ppo_divergence,
    }
    candidates: list[SequenceCandidate] = []
    hard_violations: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    prior_generation_diagnostics: dict[str, dict[str, Any]] = {}
    for prior in priors:
        templates_available = _candidate_templates_for_prior(prior)
        allocation_diagnostic = _candidate_allocation_diagnostic(
            prior,
            risk_flags=risk_flags,
            category_flags=risk_category_flags,
            allocation_cfg=allocation_cfg,
        )
        effective_candidate_count = int(allocation_diagnostic["effective_candidate_count"])
        threshold = _candidate_count_threshold(effective_candidate_count, candidate_count_bins)
        prior_generation_diagnostics[prior.source] = {
            "confidence": float(prior.confidence),
            "candidate_count": effective_candidate_count,
            "candidate_count_threshold": float(threshold),
            "first_candidate_threshold": float(first_candidate_threshold),
            "candidate_count_from_confidence": int(candidate_count_from_confidence(prior.confidence)),
            "available_template_count": len(templates_available),
            "available_templates": templates_available,
            "generated_template_count": 0,
            "generated_templates": [],
            "max_candidates_exhausted": False,
            "no_candidate_reason": None,
            "no_candidate_reasons": [],
            "confidence_reasons": dict(prior.confidence_reasons),
            **allocation_diagnostic,
        }
    for prior in priors:
        effective_candidate_count = int(
            prior_generation_diagnostics[prior.source].get("effective_candidate_count", prior.candidate_count)
        )
        templates = _candidate_templates_for_prior(prior)[: effective_candidate_count]
        for template in templates:
            if len(candidates) >= max_candidates:
                prior_generation_diagnostics[prior.source]["max_candidates_exhausted"] = True
                break
            candidate_id = f"{prior.source}:{template}:{len(candidates)}"
            raw, template_metadata = generate_sequence_with_metadata(
                template,
                last_action=last,
                horizon=horizon,
                prior_action=prior.base_action,
                previous_sequence=previous_plan_sequence,
                max_delta_by_field=max_delta_by_field,
            )
            violations = [
                {
                    **dict(violation),
                    "candidate_id": candidate_id,
                    "source_prior": prior.source,
                    "template_name": template,
                }
                for violation in check_hard_constraints(
                    raw,
                    previous_action=last,
                    max_delta=max_delta_by_field,
                )
            ]
            violation_reason_distribution, violation_field_distribution = _violation_distributions(violations)
            projected, projection_report = _project_sequence_placeholder(raw)
            penalties = soft_penalties(projected)
            candidate = SequenceCandidate(
                candidate_id=candidate_id,
                source_prior=prior.source,
                template_name=template,
                raw_sequence=raw,
                feasible=not violations,
                feasibility_violations=violations,
                violation_reason_distribution=violation_reason_distribution,
                violation_field_distribution=violation_field_distribution,
                first_action_delta_from_reference=_first_action_delta(raw, last),
                template_metadata=template_metadata,
                projected_sequence=projected,
                soft_penalty_report=penalties,
            )
            candidates.append(candidate)
            prior_generation_diagnostics[prior.source]["generated_template_count"] = int(
                prior_generation_diagnostics[prior.source]["generated_template_count"]
            ) + 1
            prior_generation_diagnostics[prior.source]["generated_templates"].append(template)
            hard_violations.extend(violations)
            score = score_dual(candidate.raw_sequence, candidate.projected_sequence, projection_report)
            risk_static_hold_applied = bool(
                template == "hold_all"
                and risk_flags.get("any_risk", False)
                and max(rule_divergence, ppo_divergence) >= risk_static_hold_divergence_threshold
            )
            prior_bonus = prior_confidence_weight * prior.confidence
            risk_penalty = risk_static_hold_penalty if risk_static_hold_applied else 0.0
            score_breakdown = asdict_clean(score)
            score_breakdown["base_final_score"] = score.final_score
            score_breakdown["prior_confidence"] = prior.confidence
            score_breakdown["prior_confidence_bonus"] = prior_bonus
            score_breakdown["risk_static_hold_penalty"] = risk_penalty
            score_breakdown["risk_static_hold_penalty_applied"] = risk_static_hold_applied
            score_breakdown["risk_flags"] = risk_flags
            score_breakdown["reference_action_delta_max_to_rule"] = rule_divergence
            score_breakdown["reference_action_delta_max_to_ppo"] = ppo_divergence
            score_breakdown["final_score"] = score.final_score + prior_bonus - risk_penalty
            score_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "source_prior": prior.source,
                    "template_name": template,
                    "projection_report": asdict_clean(projection_report),
                    "score_breakdown": score_breakdown,
                }
            )

    for prior in priors:
        diagnostic = prior_generation_diagnostics[prior.source]
        no_candidate_reason, no_candidate_reasons = _prior_no_candidate_reason(
            prior,
            effective_candidate_count=int(diagnostic.get("effective_candidate_count", prior.candidate_count) or 0),
            available_template_count=int(diagnostic.get("available_template_count", 0) or 0),
            generated_template_count=int(diagnostic.get("generated_template_count", 0) or 0),
            max_candidates_exhausted=bool(diagnostic.get("max_candidates_exhausted", False)),
            risk_flags=risk_flags,
            candidate_count_threshold=float(diagnostic.get("candidate_count_threshold", 0.0) or 0.0),
            first_candidate_threshold=first_candidate_threshold,
        )
        diagnostic["no_candidate_reason"] = no_candidate_reason
        diagnostic["no_candidate_reasons"] = no_candidate_reasons

    feasible_candidates = [candidate for candidate in candidates if candidate.feasible]
    fallback_triggered = False
    fallback_reason = None
    if not feasible_candidates:
        fallback_triggered = True
        fallback_reason = "no_feasible_candidate"
        raw = generate_sequence("conservative_stabilize", last_action=last, horizon=horizon, prior_action=last)
        projected, projection_report = _project_sequence_placeholder(raw)
        fallback_candidate = SequenceCandidate(
            candidate_id="fallback:conservative_stabilize:0",
            source_prior="emergency_conservative_sequence",
            template_name="conservative_stabilize",
            raw_sequence=raw,
            feasible=True,
            feasibility_violations=[],
            violation_reason_distribution={},
            violation_field_distribution={},
            first_action_delta_from_reference=_first_action_delta(raw, last),
            template_metadata={},
            projected_sequence=projected,
            soft_penalty_report=soft_penalties(projected),
        )
        candidates.append(fallback_candidate)
        feasible_candidates = [fallback_candidate]
        fallback_score = score_dual(raw, projected, projection_report)
        score_rows.append(
            {
                "candidate_id": fallback_candidate.candidate_id,
                "source_prior": fallback_candidate.source_prior,
                "template_name": fallback_candidate.template_name,
                "projection_report": asdict_clean(projection_report),
                "score_breakdown": {
                    **asdict_clean(fallback_score),
                    "base_final_score": fallback_score.final_score,
                    "prior_confidence": 0.0,
                    "prior_confidence_bonus": 0.0,
                    "risk_static_hold_penalty": 0.0,
                    "risk_static_hold_penalty_applied": False,
                    "risk_flags": risk_flags,
                    "reference_action_delta_max_to_rule": rule_divergence,
                    "reference_action_delta_max_to_ppo": ppo_divergence,
                },
            }
        )

    score_by_id = {
        row["candidate_id"]: row["score_breakdown"]["final_score"]
        for row in score_rows
    }
    selected = max(feasible_candidates, key=lambda candidate: score_by_id.get(candidate.candidate_id, -1e9))
    selected_first = selected.projected_sequence[0] if selected.projected_sequence else selected.raw_sequence[0]
    action_diff = {field: selected_first[field] - runtime_final[field] for field in selected_first}
    selected_reason_dist, selected_field_dist = _violation_distributions(selected.feasibility_violations)
    selected_score = score_by_id.get(selected.candidate_id, -1e9)
    score_breakdown_by_id = {
        str(row.get("candidate_id")): dict(row.get("score_breakdown") or {})
        for row in score_rows
        if isinstance(row, Mapping)
    }
    score_component_by_candidate = {
        candidate_id: _score_components(score_breakdown)
        for candidate_id, score_breakdown in score_breakdown_by_id.items()
    }
    selected_components = score_component_by_candidate.get(selected.candidate_id, {})
    competitiveness: dict[str, dict[str, Any]] = {}
    score_margin_to_selected_by_prior: dict[str, dict[str, Any]] = {}
    for source in ("ppo_prior", "previous_plan_prior", "conservative_prior", "rule_prior"):
        source_candidates = [candidate for candidate in candidates if candidate.source_prior == source]
        source_feasible = [candidate for candidate in source_candidates if candidate.feasible]
        if source_feasible:
            best = max(source_feasible, key=lambda candidate: score_by_id.get(candidate.candidate_id, -1e9))
            best_score = score_by_id.get(best.candidate_id, -1e9)
            status = "selected" if best.candidate_id == selected.candidate_id else "lower_score"
            best_components = score_component_by_candidate.get(best.candidate_id, {})
            component_margin = _selected_minus_competitor_components(selected_components, best_components)
            competitiveness[source] = {
                "candidate_count": len(source_candidates),
                "feasible_candidate_count": len(source_feasible),
                "best_candidate_id": best.candidate_id,
                "best_template_name": best.template_name,
                "best_score": best_score,
                "selected_score_margin": selected_score - best_score,
                "status": status,
            }
            score_margin_to_selected_by_prior[source] = {
                "status": status,
                "total_margin": selected_score - best_score,
                "selected_candidate_id": selected.candidate_id,
                "selected_template_name": selected.template_name,
                "selected_source_prior": selected.source_prior,
                "selected_score": selected_score,
                "best_candidate_id": best.candidate_id,
                "best_template_name": best.template_name,
                "best_score": best_score,
                "selected_minus_competitor": component_margin,
            }
        else:
            prior_diagnostic = prior_generation_diagnostics.get(source, {})
            competitiveness[source] = {
                "candidate_count": len(source_candidates),
                "feasible_candidate_count": 0,
                "best_candidate_id": None,
                "best_template_name": None,
                "best_score": None,
                "selected_score_margin": None,
                "status": "no_feasible_candidate" if source_candidates else "no_candidate",
                "no_candidate_reason": prior_diagnostic.get("no_candidate_reason"),
            }
            score_margin_to_selected_by_prior[source] = {
                "status": "no_feasible_candidate" if source_candidates else "no_candidate",
                "total_margin": None,
                "selected_candidate_id": selected.candidate_id,
                "selected_template_name": selected.template_name,
                "selected_source_prior": selected.source_prior,
                "selected_score": selected_score,
                "best_candidate_id": None,
                "best_template_name": None,
                "best_score": None,
                "selected_minus_competitor": {},
                "no_candidate_reason": prior_diagnostic.get("no_candidate_reason"),
            }
    scoring_metadata.update(
        {
            "selected_candidate_violation_count": len(selected.feasibility_violations),
            "selected_candidate_violation_reason_distribution": selected_reason_dist,
            "selected_candidate_violation_field_distribution": selected_field_dist,
            "candidate_violation_provenance": [
                {
                    "candidate_id": candidate.candidate_id,
                    "source_prior": candidate.source_prior,
                    "template_name": candidate.template_name,
                    "feasible": candidate.feasible,
                    "violation_reason_distribution": candidate.violation_reason_distribution,
                    "violation_field_distribution": candidate.violation_field_distribution,
                    "first_action_delta_from_reference": candidate.first_action_delta_from_reference,
                    "template_metadata": candidate.template_metadata,
                }
                for candidate in candidates
            ],
            "rule_conservative_competitiveness": competitiveness,
        }
    )
    if scorer_diagnostic_enabled:
        diagnostic_payload: dict[str, Any] = {
            "scorer_diagnostic_enabled": True,
            "scorer_diagnostic_log_mode": scorer_diagnostic_log_mode,
            "prior_generation_diagnostics": prior_generation_diagnostics,
            "score_margin_to_selected_by_prior": score_margin_to_selected_by_prior,
        }
        if scorer_diagnostic_log_mode == "diagnostic_full":
            diagnostic_payload["score_component_by_candidate"] = score_component_by_candidate
        else:
            diagnostic_payload["score_component_by_candidate"] = {}
        scoring_metadata.update(
            diagnostic_payload
        )
    plan_memory = PlanMemory(
        selected_sequence=selected.projected_sequence,
        remaining_sequence=selected.projected_sequence[1:],
        invalidation_flags=[],
        confidence=max(score_by_id.get(selected.candidate_id, 0.0), 0.0),
    )

    return CSTCCAuditRecord(
        prediction_level=int(cfg.get("prediction_level", 0)),
        state_snapshot=_state_snapshot(state_history),
        action_history_summary=action_smoothness_summary(action_history),
        weather_history_summary=_weather_summary(weather_history),
        temporal_context=context_dict,
        llm_semantic_suggestion=asdict_clean(suggestion),
        evidence_confidence=evidence_conf,
        effective_confidence=semantic_state.effective_confidence,
        active_regime_before=active_regime,
        active_regime_after=semantic_state.active_regime,
        regime_transition_allowed=semantic_state.transition_allowed,
        regime_transition_reason=semantic_state.transition_reason,
        calibrated_weights=semantic_state.calibrated_weights,
        weight_governor_report={
            "llm_influence_alpha": semantic_state.llm_influence_alpha,
            "data_quality_score": semantic_state.data_quality_score,
            "regime_stability": semantic_state.regime_stability,
            "invalid_regime": semantic_state.invalid_regime,
        },
        prior_confidence_report={prior.source: prior.confidence_reasons | {"confidence": prior.confidence} for prior in priors},
        generated_prior_candidates=[asdict_clean(prior) for prior in priors],
        sequence_candidates=[asdict_clean(candidate) for candidate in candidates],
        hard_constraint_violations=hard_violations,
        soft_penalty_report={candidate.candidate_id: candidate.soft_penalty_report for candidate in candidates},
        tomato_safety_dual_scores=score_rows,
        scoring_metadata=scoring_metadata,
        selected_sequence_id=selected.candidate_id,
        selected_first_action=selected_first,
        shadow_selected_sequence_id=selected.candidate_id,
        shadow_selected_first_action=selected_first,
        selected_action_is_shadow_only=True,
        current_runtime_final_action=runtime_final,
        action_difference_from_runtime=action_diff,
        shadow_action_difference_from_runtime=action_diff,
        plan_memory_report=asdict_clean(plan_memory),
        invalidation_flags=[],
        fallback_triggered=fallback_triggered,
        fallback_reason=fallback_reason,
    )
