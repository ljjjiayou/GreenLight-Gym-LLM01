"""Dataclass contracts for C-STCC v80.

These contracts are intentionally plain Python dataclasses. They are easy to
serialize, audit, and test without adding runtime dependencies or connecting to
the active controller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Mapping


VERSION = "v80"
PREDICTION_LEVEL = 0
SCORE_VALIDITY = "sequence_only_no_rollout"
PLACEHOLDER_PROJECTION_LEVEL = "placeholder_no_real_projection"

ACTION_FIELDS = (
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
)

REGIMES = (
    "NORMAL_BALANCED",
    "HEAT_ACCUMULATION",
    "HIGH_VPD_DRY_STRESS",
    "HUMIDITY_EXCESS_DISEASE_RISK",
    "LOW_LIGHT_ENERGY_SAVING",
    "CO2_VENTILATION_CONFLICT",
    "SOLVER_SENSITIVE_EMERGENCY",
)

BOUNDARY_FALSE_FIELDS = (
    "online_llm_called",
    "controller_changed",
    "default_controller_changed",
    "final_action_changed",
    "runtime_policy_enabled",
    "predictive_rollout_executed",
    "surrogate_model_used",
    "mpc_optimizer_executed",
    "performance_claim_allowed",
    "promotion_evidence",
)

DEFAULT_WEIGHT_KEYS = (
    "temperature_tracking",
    "humidity_recovery",
    "vpd_risk",
    "energy",
    "action_smoothness",
    "rewrite_pressure",
    "solver_risk",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clamp(value: Any, lower: float = 0.0, upper: float = 1.0, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if number != number:
        number = float(default)
    return max(lower, min(upper, number))


def normalize_action(action: Mapping[str, Any] | None) -> dict[str, float]:
    action = action or {}
    return {field: clamp(action.get(field, 0.0)) for field in ACTION_FIELDS}


def asdict_clean(value: Any) -> Any:
    if is_dataclass(value):
        return {key: asdict_clean(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): asdict_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [asdict_clean(item) for item in value]
    return value


@dataclass(slots=True)
class SemanticSuggestion:
    regime: str
    regime_confidence: float
    priority_weight_suggestions: dict[str, float] = field(default_factory=dict)
    control_intent: str = ""
    risk_factors: list[str] = field(default_factory=list)
    uncertain_fields: list[str] = field(default_factory=list)
    suggested_refresh_steps: int = 12
    llm_reported_confidence: float | None = None
    source: str = "mock_semantic_suggestion"

    def __post_init__(self) -> None:
        self.regime_confidence = clamp(self.regime_confidence)
        if self.llm_reported_confidence is None:
            self.llm_reported_confidence = self.regime_confidence
        else:
            self.llm_reported_confidence = clamp(self.llm_reported_confidence)
        self.priority_weight_suggestions = {
            str(key): clamp(value) for key, value in self.priority_weight_suggestions.items()
        }
        self.suggested_refresh_steps = max(1, int(self.suggested_refresh_steps))


@dataclass(slots=True)
class CalibratedSemanticState:
    active_regime: str
    regime_age_steps: int
    regime_confidence: float
    effective_confidence: float
    calibrated_weights: dict[str, float]
    tightened_constraints: list[str] = field(default_factory=list)
    llm_influence_alpha: float = 0.0
    transition_allowed: bool = False
    transition_reason: str | None = None
    invalid_regime: str | None = None
    data_quality_score: float = 1.0
    regime_stability: float = 1.0

    def __post_init__(self) -> None:
        if self.active_regime not in REGIMES:
            self.invalid_regime = self.active_regime
            self.active_regime = "NORMAL_BALANCED"
        self.regime_age_steps = max(0, int(self.regime_age_steps))
        self.regime_confidence = clamp(self.regime_confidence)
        self.effective_confidence = clamp(self.effective_confidence)
        self.llm_influence_alpha = clamp(self.llm_influence_alpha)
        self.data_quality_score = clamp(self.data_quality_score)
        self.regime_stability = clamp(self.regime_stability)


@dataclass(slots=True)
class TemporalContext:
    trend_features: dict[str, float] = field(default_factory=dict)
    debt_features: dict[str, float] = field(default_factory=dict)
    action_smoothness_features: dict[str, Any] = field(default_factory=dict)
    safety_pressure_features: dict[str, float] = field(default_factory=dict)
    solver_risk_features: dict[str, float] = field(default_factory=dict)
    data_quality_features: dict[str, float] = field(default_factory=dict)
    delta_t_hours: float = 1.0 / 12.0
    window_hours: float = 2.0


@dataclass(slots=True)
class PriorCandidate:
    source: str
    base_action: dict[str, float]
    base_sequence: list[dict[str, float]] = field(default_factory=list)
    confidence: float = 0.0
    confidence_reasons: dict[str, float | str] = field(default_factory=dict)
    candidate_count: int = 0

    def __post_init__(self) -> None:
        self.base_action = normalize_action(self.base_action)
        self.base_sequence = [normalize_action(action) for action in self.base_sequence]
        self.confidence = clamp(self.confidence)
        self.candidate_count = max(0, int(self.candidate_count))


@dataclass(slots=True)
class SequenceCandidate:
    candidate_id: str
    source_prior: str
    template_name: str
    raw_sequence: list[dict[str, float]]
    feasible: bool = True
    feasibility_violations: list[dict[str, Any]] = field(default_factory=list)
    violation_reason_distribution: dict[str, int] = field(default_factory=dict)
    violation_field_distribution: dict[str, int] = field(default_factory=dict)
    first_action_delta_from_reference: dict[str, float] = field(default_factory=dict)
    template_metadata: dict[str, Any] = field(default_factory=dict)
    projected_sequence: list[dict[str, float]] = field(default_factory=list)
    soft_penalty_report: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.raw_sequence = [normalize_action(action) for action in self.raw_sequence]
        if self.projected_sequence:
            self.projected_sequence = [normalize_action(action) for action in self.projected_sequence]
        self.violation_reason_distribution = {
            str(key): max(0, int(value))
            for key, value in dict(self.violation_reason_distribution).items()
        }
        self.violation_field_distribution = {
            str(key): max(0, int(value))
            for key, value in dict(self.violation_field_distribution).items()
        }
        self.first_action_delta_from_reference = {
            field: float(self.first_action_delta_from_reference.get(field, 0.0))
            for field in ACTION_FIELDS
        }
        self.template_metadata = dict(self.template_metadata)


@dataclass(slots=True)
class SafetyProjectionReport:
    rewrite_magnitude: float = 0.0
    rewrite_reasons: list[str] = field(default_factory=list)
    rewrite_reason_severity: dict[str, str] = field(default_factory=dict)
    severe_rewrite_count: int = 0
    projected_safe: bool = True
    projection_level: str = PLACEHOLDER_PROJECTION_LEVEL
    projection_validity: str = PLACEHOLDER_PROJECTION_LEVEL

    def __post_init__(self) -> None:
        self.rewrite_magnitude = max(0.0, float(self.rewrite_magnitude))
        self.severe_rewrite_count = max(0, int(self.severe_rewrite_count))
        allowed = {PLACEHOLDER_PROJECTION_LEVEL, "runtime_tomato_safety_projection"}
        if self.projection_level not in allowed:
            raise ValueError(f"unsupported projection_level: {self.projection_level}")
        if self.projection_validity not in allowed:
            raise ValueError(f"unsupported projection_validity: {self.projection_validity}")


@dataclass(slots=True)
class ScoreBreakdown:
    raw_score: float
    projected_score: float
    normalized_components: dict[str, float]
    final_score: float
    uncertainty_penalty: float = 0.0
    rewrite_penalty: float = 0.0
    score_validity: str = SCORE_VALIDITY


@dataclass(slots=True)
class PlanMemory:
    selected_sequence: list[dict[str, float]] = field(default_factory=list)
    remaining_sequence: list[dict[str, float]] = field(default_factory=list)
    invalidation_flags: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        self.selected_sequence = [normalize_action(action) for action in self.selected_sequence]
        self.remaining_sequence = [normalize_action(action) for action in self.remaining_sequence]
        self.confidence = clamp(self.confidence)


@dataclass(slots=True)
class CSTCCAuditRecord:
    version: str = VERSION
    timestamp: str = field(default_factory=utc_now_iso)
    prediction_level: int = PREDICTION_LEVEL
    controller_changed: bool = False
    online_llm_called: bool = False
    final_action_changed: bool = False
    final_action_invariant_verified: bool = True
    default_controller_changed: bool = False
    runtime_policy_enabled: bool = False
    predictive_rollout_executed: bool = False
    surrogate_model_used: bool = False
    mpc_optimizer_executed: bool = False
    performance_claim_allowed: bool = False
    promotion_evidence: bool = False

    state_snapshot: dict[str, Any] = field(default_factory=dict)
    action_history_summary: dict[str, Any] = field(default_factory=dict)
    weather_history_summary: dict[str, Any] = field(default_factory=dict)
    temporal_context: dict[str, Any] = field(default_factory=dict)

    llm_semantic_suggestion: dict[str, Any] | None = None
    evidence_confidence: float = 0.0
    effective_confidence: float = 0.0

    active_regime_before: str = "NORMAL_BALANCED"
    active_regime_after: str = "NORMAL_BALANCED"
    regime_transition_allowed: bool = False
    regime_transition_reason: str | None = None

    calibrated_weights: dict[str, float] = field(default_factory=dict)
    weight_governor_report: dict[str, Any] = field(default_factory=dict)

    prior_confidence_report: dict[str, Any] = field(default_factory=dict)
    generated_prior_candidates: list[dict[str, Any]] = field(default_factory=list)

    sequence_candidates: list[dict[str, Any]] = field(default_factory=list)
    hard_constraint_violations: list[dict[str, Any]] = field(default_factory=list)
    soft_penalty_report: dict[str, Any] = field(default_factory=dict)

    tomato_safety_dual_scores: list[dict[str, Any]] = field(default_factory=list)
    scoring_metadata: dict[str, Any] = field(default_factory=dict)
    selected_sequence_id: str | None = None
    selected_first_action: dict[str, float] | None = None
    shadow_selected_sequence_id: str | None = None
    shadow_selected_first_action: dict[str, float] | None = None
    selected_action_is_shadow_only: bool = True

    current_runtime_final_action: dict[str, float] | None = None
    action_difference_from_runtime: dict[str, float] | None = None
    shadow_action_difference_from_runtime: dict[str, float] | None = None

    plan_memory_report: dict[str, Any] = field(default_factory=dict)
    invalidation_flags: list[str] = field(default_factory=list)

    fallback_triggered: bool = False
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        self.prediction_level = int(self.prediction_level)
        self.evidence_confidence = clamp(self.evidence_confidence)
        self.effective_confidence = clamp(self.effective_confidence)
        if self.current_runtime_final_action is not None:
            self.current_runtime_final_action = normalize_action(self.current_runtime_final_action)
        if self.selected_first_action is not None:
            self.selected_first_action = normalize_action(self.selected_first_action)
        if self.shadow_selected_first_action is not None:
            self.shadow_selected_first_action = normalize_action(self.shadow_selected_first_action)
        if not self.selected_action_is_shadow_only:
            raise ValueError("v80 selected action must remain shadow-only")
        if not self.final_action_invariant_verified:
            raise ValueError("v81 shadow audit must verify final-action invariant")
        for field_name in BOUNDARY_FALSE_FIELDS:
            if getattr(self, field_name):
                raise ValueError(f"v80 boundary field must remain false: {field_name}")
