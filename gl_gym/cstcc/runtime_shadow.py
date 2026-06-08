"""Thin runtime shadow adapter for C-STCC v81.

This adapter intentionally does not return or apply a new control action. It
collects runtime-facing inputs, builds a C-STCC audit record, verifies that the
runtime final action did not change, and returns only the audit record.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .audit import build_audit_record
from .contracts import CSTCCAuditRecord, SemanticSuggestion, asdict_clean, normalize_action, utc_now_iso


def verify_final_action_invariant(
    final_action_before: Mapping[str, Any],
    final_action_after: Mapping[str, Any],
) -> tuple[bool, dict[str, float]]:
    """Compare normalized actions and return invariant status plus diff."""

    before = normalize_action(final_action_before)
    after = normalize_action(final_action_after)
    diff = {field: after[field] - before[field] for field in before}
    invariant_ok = all(abs(value) <= 1e-12 for value in diff.values())
    return invariant_ok, diff


def run_cstcc_shadow_step(
    *,
    state_history: Sequence[Mapping[str, Any]],
    action_history: Sequence[Mapping[str, Any]],
    weather_history: Sequence[Mapping[str, Any]] | None = None,
    current_runtime_final_action: Mapping[str, Any],
    active_regime: str,
    regime_age_steps: int,
    previous_plan_sequence: Sequence[Mapping[str, Any]] | None = None,
    ppo_action: Mapping[str, Any] | None = None,
    rule_action: Mapping[str, Any] | None = None,
    semantic_suggestion: SemanticSuggestion | Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> CSTCCAuditRecord:
    """Run one C-STCC shadow step and return only an audit record."""

    final_action_before = normalize_action(current_runtime_final_action)
    audit = build_audit_record(
        state_history=state_history,
        action_history=action_history,
        weather_history=weather_history,
        semantic_suggestion=semantic_suggestion,
        current_runtime_final_action=current_runtime_final_action,
        active_regime=active_regime,
        regime_age_steps=regime_age_steps,
        previous_plan_sequence=previous_plan_sequence,
        ppo_action=ppo_action,
        rule_action=rule_action,
        config=config,
    )
    final_action_after = normalize_action(current_runtime_final_action)
    invariant_ok, invariant_diff = verify_final_action_invariant(final_action_before, final_action_after)
    if not invariant_ok:
        raise RuntimeError("C-STCC v81 violated final-action invariant")
    audit.final_action_invariant_verified = True
    audit.scoring_metadata = {
        **audit.scoring_metadata,
        "runtime_shadow_version": "v81",
        "final_action_invariant_verified": True,
        "final_action_invariant_diff": invariant_diff,
    }
    return audit


def safe_run_cstcc_shadow_step(**kwargs: Any) -> tuple[CSTCCAuditRecord | None, dict[str, Any] | None]:
    """Failure-isolated wrapper for future runtime integration.

    A failed C-STCC shadow audit must not interrupt the original controller. The
    caller gets either an audit record or a compact failure payload to log.
    """

    try:
        return run_cstcc_shadow_step(**kwargs), None
    except Exception as exc:
        return None, {
            "version": "v81",
            "timestamp": utc_now_iso(),
            "audit_success": False,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "final_action_changed": False,
            "final_action_invariant_verified": False,
            "online_llm_called": False,
            "predictive_rollout_executed": False,
            "selected_action_is_shadow_only": True,
            "inputs_summary": {
                "state_history_len": len(kwargs.get("state_history") or []),
                "action_history_len": len(kwargs.get("action_history") or []),
                "weather_history_len": len(kwargs.get("weather_history") or []),
                "active_regime": kwargs.get("active_regime"),
                "regime_age_steps": kwargs.get("regime_age_steps"),
            },
        }


def audit_record_to_dict(record: CSTCCAuditRecord) -> dict[str, Any]:
    return asdict_clean(record)
