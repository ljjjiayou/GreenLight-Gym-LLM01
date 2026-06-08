"""C-STCC v80 contract and shadow-audit helpers.

The v80 package is contract-first. It defines structured data, deterministic
helpers, and audit records for the Calibrated Semantic Temporal Control Chain.
It does not connect to runtime control, call online LLMs, or change final
actions.
"""

from .contracts import (
    ACTION_FIELDS,
    REGIMES,
    BOUNDARY_FALSE_FIELDS,
    CalibratedSemanticState,
    CSTCCAuditRecord,
    PlanMemory,
    PriorCandidate,
    SafetyProjectionReport,
    ScoreBreakdown,
    SemanticSuggestion,
    SequenceCandidate,
    TemporalContext,
    asdict_clean,
)
from .audit import build_audit_record
from .runtime_shadow import run_cstcc_shadow_step, safe_run_cstcc_shadow_step, verify_final_action_invariant
from .regime_evidence import regime_evidence_score

__all__ = [
    "ACTION_FIELDS",
    "REGIMES",
    "BOUNDARY_FALSE_FIELDS",
    "CalibratedSemanticState",
    "CSTCCAuditRecord",
    "PlanMemory",
    "PriorCandidate",
    "SafetyProjectionReport",
    "ScoreBreakdown",
    "SemanticSuggestion",
    "SequenceCandidate",
    "TemporalContext",
    "asdict_clean",
    "build_audit_record",
    "run_cstcc_shadow_step",
    "safe_run_cstcc_shadow_step",
    "verify_final_action_invariant",
    "regime_evidence_score",
]
