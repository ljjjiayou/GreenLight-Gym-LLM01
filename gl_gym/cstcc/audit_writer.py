"""JSONL audit writer and compact metrics helpers for C-STCC v81."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import ACTION_FIELDS, CSTCCAuditRecord, asdict_clean


DEFAULT_LOG_ROOT = Path("logs") / "cstcc_shadow"


def default_audit_jsonl_path(episode_id: str, *, root: str | Path = DEFAULT_LOG_ROOT) -> Path:
    safe_episode = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(episode_id))
    return Path(root) / f"v81_episode_{safe_episode}.jsonl"


def should_sample_step(step: int, sample_rate: float = 1.0) -> bool:
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False
    interval = max(1, round(1.0 / sample_rate))
    return int(step) % interval == 0


def _strip_sequences(candidate: Mapping[str, Any], *, save_raw_sequences: bool, save_projected_sequences: bool) -> dict[str, Any]:
    result = dict(candidate)
    if not save_raw_sequences:
        result.pop("raw_sequence", None)
    if not save_projected_sequences:
        result.pop("projected_sequence", None)
    return result


def parse_selected_sequence_id(sequence_id: Any) -> tuple[str, str]:
    """Return the prior and template components from a compact candidate id."""

    text = str(sequence_id or "")
    if not text:
        return "", ""
    parts = text.split(":")
    source_prior = parts[0] if parts else ""
    template_name = parts[1] if len(parts) > 1 else ""
    return source_prior, template_name


def _distribution_key(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "/".join(str(item) for item in value)
    return str(value or "unknown")


def _previous_shift_summary(
    candidates: Iterable[Mapping[str, Any]],
    *,
    selected_sequence_id: Any,
) -> dict[str, Any]:
    candidate_count = 0
    selected_count = 0
    max_delta_count = 0
    projection_applied_count = 0
    first_delta_after_projection_max = 0.0
    projection_abs_sums = {field: 0.0 for field in ACTION_FIELDS}
    internal_max_after_projection = {field: 0.0 for field in ACTION_FIELDS}
    violation_fields: dict[str, int] = {}
    selected_text = str(selected_sequence_id or "")
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or candidate.get("template_name") != "previous_shift_all":
            continue
        candidate_count += 1
        if str(candidate.get("candidate_id") or "") == selected_text:
            selected_count += 1
        reasons = candidate.get("violation_reason_distribution") or {}
        if isinstance(reasons, Mapping):
            max_delta_count += int(reasons.get("max_delta", 0) or 0)
        fields = candidate.get("violation_field_distribution") or {}
        if isinstance(fields, Mapping):
            for field, count in fields.items():
                key = str(field)
                violation_fields[key] = violation_fields.get(key, 0) + int(count or 0)
        metadata = candidate.get("template_metadata") or {}
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("previous_shift_projection_applied", False):
            projection_applied_count += 1
        first_delta_after = metadata.get("previous_shift_first_delta_after_projection_by_field") or {}
        if isinstance(first_delta_after, Mapping):
            for field in ACTION_FIELDS:
                first_delta_after_projection_max = max(
                    first_delta_after_projection_max,
                    abs(float(first_delta_after.get(field, 0.0) or 0.0)),
                )
        projection_delta = metadata.get("previous_shift_projection_delta_by_field") or {}
        if isinstance(projection_delta, Mapping):
            for field in ACTION_FIELDS:
                projection_abs_sums[field] += abs(float(projection_delta.get(field, 0.0) or 0.0))
        internal_delta = metadata.get("previous_shift_internal_max_delta_after_projection_by_field") or {}
        if isinstance(internal_delta, Mapping):
            for field in ACTION_FIELDS:
                internal_max_after_projection[field] = max(
                    internal_max_after_projection[field],
                    abs(float(internal_delta.get(field, 0.0) or 0.0)),
                )
    return {
        "previous_shift_all_candidate_count": candidate_count,
        "previous_shift_all_max_delta_count": max_delta_count,
        "previous_shift_all_selected_count": selected_count,
        "previous_shift_all_violation_field_distribution": dict(sorted(violation_fields.items())),
        "previous_shift_projection_applied_count": projection_applied_count,
        "previous_shift_first_delta_after_projection_max": first_delta_after_projection_max,
        "previous_shift_projection_mean_abs_delta_by_field": {
            field: projection_abs_sums[field] / max(projection_applied_count, 1)
            for field in ACTION_FIELDS
        },
        "previous_shift_internal_max_delta_after_projection_by_field": internal_max_after_projection,
    }


def compact_audit_record(
    record: CSTCCAuditRecord | Mapping[str, Any],
    *,
    step: int | None = None,
    episode_id: str | None = None,
    save_full_candidates: bool = False,
    save_raw_sequences: bool = False,
    save_projected_sequences: bool = False,
    max_sequence_dump: int = 20,
) -> dict[str, Any]:
    payload = asdict_clean(record)
    candidates = list(payload.get("sequence_candidates") or [])
    feasible_count = sum(1 for candidate in candidates if candidate.get("feasible", False))
    score_rows = list(payload.get("tomato_safety_dual_scores") or [])
    scoring = dict(payload.get("scoring_metadata") or {})
    hard_violations = list(payload.get("hard_constraint_violations") or [])
    violation_reasons: dict[str, int] = {}
    violation_fields: dict[str, int] = {}
    for violation in hard_violations:
        if not isinstance(violation, Mapping):
            continue
        reason = str(
            violation.get("reason")
            or violation.get("violation")
            or violation.get("constraint")
            or violation.get("type")
            or "unknown"
        )
        violation_reasons[reason] = violation_reasons.get(reason, 0) + 1
        field = _distribution_key(violation.get("field") or violation.get("fields"))
        violation_fields[field] = violation_fields.get(field, 0) + 1
    projection_level = scoring.get("tomato_safety_projection_level")
    if not projection_level and score_rows:
        projection_level = (
            score_rows[0].get("projection_report", {}).get("projection_level")
            or score_rows[0].get("projection_report", {}).get("projection_validity")
        )
    selected_sequence_id = payload.get("shadow_selected_sequence_id") or payload.get("selected_sequence_id")
    selected_source_prior, selected_template_name = parse_selected_sequence_id(selected_sequence_id)
    candidate_violation_provenance = list(scoring.get("candidate_violation_provenance") or [])
    previous_shift = _previous_shift_summary(
        candidate_violation_provenance,
        selected_sequence_id=selected_sequence_id,
    )
    prior_generation = scoring.get("prior_generation_diagnostics") or {}
    if not isinstance(prior_generation, Mapping):
        prior_generation = {}
    score_margin = scoring.get("score_margin_to_selected_by_prior") or {}
    if not isinstance(score_margin, Mapping):
        score_margin = {}
    score_components = scoring.get("score_component_by_candidate") or {}
    if not isinstance(score_components, Mapping):
        score_components = {}
    log_mode = str(scoring.get("scorer_diagnostic_log_mode") or "diagnostic_full")
    if log_mode == "compact_scan":
        score_components = {}
    rule_margin = score_margin.get("rule_prior") or {}
    if not isinstance(rule_margin, Mapping):
        rule_margin = {}
    conservative_diag = prior_generation.get("conservative_prior") or {}
    if not isinstance(conservative_diag, Mapping):
        conservative_diag = {}
    conservative_margin = score_margin.get("conservative_prior") or {}
    if not isinstance(conservative_margin, Mapping):
        conservative_margin = {}
    row = {
        "timestamp": payload.get("timestamp"),
        "version": payload.get("version"),
        "step": step,
        "episode_id": episode_id,
        "enabled": True,
        "shadow_only": True,
        "audit_success": True,
        "prediction_level": payload.get("prediction_level"),
        "online_llm_called": payload.get("online_llm_called", False),
        "final_action_changed": payload.get("final_action_changed", False),
        "final_action_invariant_verified": payload.get("final_action_invariant_verified", False),
        "selected_action_is_shadow_only": payload.get("selected_action_is_shadow_only", True),
        "active_regime_before": payload.get("active_regime_before"),
        "active_regime_after": payload.get("active_regime_after"),
        "evidence_confidence": payload.get("evidence_confidence"),
        "effective_confidence": payload.get("effective_confidence"),
        "llm_influence_alpha": payload.get("weight_governor_report", {}).get("llm_influence_alpha"),
        "shadow_selected_sequence_id": selected_sequence_id,
        "shadow_selected_source_prior": selected_source_prior,
        "shadow_selected_template_name": selected_template_name,
        "shadow_selected_first_action": payload.get("shadow_selected_first_action"),
        "current_runtime_final_action": payload.get("current_runtime_final_action"),
        "shadow_action_difference_from_runtime": payload.get("shadow_action_difference_from_runtime"),
        "fallback_triggered": payload.get("fallback_triggered"),
        "fallback_reason": payload.get("fallback_reason"),
        "candidate_count": len(candidates),
        "feasible_candidate_count": feasible_count,
        "infeasible_candidate_ratio": (len(candidates) - feasible_count) / max(len(candidates), 1),
        "hard_constraint_violation_count": len(hard_violations),
        "hard_constraint_violation_reason_distribution": violation_reasons,
        "hard_constraint_violation_field_distribution": violation_fields,
        "candidate_attribution_mode": scoring.get("candidate_attribution_mode", "selected_step_coincidence_attribution"),
        "candidate_violation_provenance": candidate_violation_provenance,
        "selected_candidate_violation_count": int(scoring.get("selected_candidate_violation_count", 0) or 0),
        "selected_candidate_violation_reason_distribution": scoring.get(
            "selected_candidate_violation_reason_distribution", {}
        ),
        "selected_candidate_violation_field_distribution": scoring.get(
            "selected_candidate_violation_field_distribution", {}
        ),
        "constraint_reference_kind": scoring.get("constraint_reference_kind"),
        "constraint_reference_action": scoring.get("constraint_reference_action"),
        "hold_all_reference_kind": scoring.get("hold_all_reference_kind"),
        "hold_all_reference": scoring.get("hold_all_reference"),
        "risk_flags": scoring.get("risk_flags", {}),
        "risk_category_flags": scoring.get("risk_category_flags", {}),
        "rule_conservative_competitiveness": scoring.get("rule_conservative_competitiveness", {}),
        "scorer_diagnostic_log_mode": log_mode,
        "prior_generation_diagnostics": prior_generation,
        "score_margin_to_selected_by_prior": score_margin,
        "score_component_by_candidate": score_components,
        "rule_prior_selected_score_margin": rule_margin.get("total_margin"),
        "rule_prior_best_candidate_id": rule_margin.get("best_candidate_id"),
        "rule_prior_best_template_name": rule_margin.get("best_template_name"),
        "rule_prior_margin_status": rule_margin.get("status"),
        "conservative_prior_confidence": conservative_diag.get("confidence"),
        "conservative_prior_candidate_count": conservative_diag.get("candidate_count"),
        "conservative_prior_candidate_count_before_override": conservative_diag.get(
            "candidate_count_before_override"
        ),
        "conservative_prior_candidate_count_after_override": conservative_diag.get(
            "candidate_count_after_override"
        ),
        "conservative_prior_effective_candidate_count": conservative_diag.get("effective_candidate_count"),
        "conservative_prior_candidate_count_override_applied": conservative_diag.get(
            "candidate_count_override_applied", False
        ),
        "conservative_prior_candidate_count_override_reason": conservative_diag.get(
            "candidate_count_override_reason"
        ),
        "conservative_prior_candidate_count_override_flags": conservative_diag.get(
            "candidate_count_override_flags", []
        ),
        "conservative_prior_risk_category_flags": conservative_diag.get(
            "risk_category_flags", scoring.get("risk_category_flags", {})
        ),
        "conservative_prior_generated_template_count": conservative_diag.get("generated_template_count"),
        "conservative_prior_no_candidate_reason": (
            conservative_diag.get("no_candidate_reason") or conservative_margin.get("no_candidate_reason")
        ),
        "conservative_prior_selected_score_margin": conservative_margin.get("total_margin"),
        "conservative_prior_best_candidate_id": conservative_margin.get("best_candidate_id"),
        "conservative_prior_best_template_name": conservative_margin.get("best_template_name"),
        "conservative_prior_margin_status": conservative_margin.get("status"),
        "previous_shift_projection_summary": previous_shift,
        **previous_shift,
        "score_validity": scoring.get("score_validity", "sequence_only_no_rollout"),
        "projection_level": projection_level,
        "real_tomato_safety_projection": scoring.get("real_tomato_safety_projection", False),
        "prior_confidence_report": payload.get("prior_confidence_report", {}),
        "scoring_metadata": scoring,
    }
    if save_full_candidates:
        limited = candidates[: max(0, int(max_sequence_dump))]
        row["sequence_candidates"] = [
            _strip_sequences(
                candidate,
                save_raw_sequences=save_raw_sequences,
                save_projected_sequences=save_projected_sequences,
            )
            for candidate in limited
        ]
    return row


def compact_failure_record(
    failure: Mapping[str, Any],
    *,
    step: int | None = None,
    episode_id: str | None = None,
) -> dict[str, Any]:
    payload = dict(failure)
    payload.setdefault("audit_success", False)
    payload.setdefault("final_action_changed", False)
    payload.setdefault("online_llm_called", False)
    payload.setdefault("predictive_rollout_executed", False)
    payload["step"] = step
    payload["episode_id"] = episode_id
    return payload


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return output


def append_audit_record(
    path: str | Path,
    record: CSTCCAuditRecord | Mapping[str, Any],
    *,
    step: int | None = None,
    episode_id: str | None = None,
    save_full_candidates: bool = False,
    save_raw_sequences: bool = False,
    save_projected_sequences: bool = False,
    max_sequence_dump: int = 20,
) -> Path:
    row = compact_audit_record(
        record,
        step=step,
        episode_id=episode_id,
        save_full_candidates=save_full_candidates,
        save_raw_sequences=save_raw_sequences,
        save_projected_sequences=save_projected_sequences,
        max_sequence_dump=max_sequence_dump,
    )
    return append_jsonl(path, row)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_shadow_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    total = len(rows)
    successes = [row for row in rows if row.get("audit_success", False)]
    action_abs_sums = {field: 0.0 for field in ACTION_FIELDS}
    action_abs_max = {field: 0.0 for field in ACTION_FIELDS}
    diff_count = 0
    for row in successes:
        diff = row.get("shadow_action_difference_from_runtime") or {}
        if isinstance(diff, Mapping):
            diff_count += 1
            for field in ACTION_FIELDS:
                value = abs(float(diff.get(field, 0.0)))
                action_abs_sums[field] += value
                action_abs_max[field] = max(action_abs_max[field], value)
    prior_keys = ("ppo_prior", "previous_plan_prior", "conservative_prior", "rule_prior")
    prior_confidence = {key: [] for key in prior_keys}
    for row in successes:
        report = row.get("prior_confidence_report") or {}
        for key in prior_keys:
            if key in report and "confidence" in report[key]:
                prior_confidence[key].append(float(report[key]["confidence"]))
    violation_reason_distribution: dict[str, int] = {}
    violation_field_distribution: dict[str, int] = {}
    selected_violation_reason_distribution: dict[str, int] = {}
    selected_violation_field_distribution: dict[str, int] = {}
    attribution_mode_distribution: dict[str, int] = {}
    selected_source_prior_distribution: dict[str, int] = {}
    selected_template_name_distribution: dict[str, int] = {}
    conservative_no_candidate_reason_distribution: dict[str, int] = {}
    rule_prior_margin_values: list[float] = []
    conservative_prior_margin_values: list[float] = []
    conservative_risk_candidate_steps = 0
    conservative_risk_feasible_steps = 0
    conservative_risk_no_candidate_count = 0
    conservative_normal_candidate_steps = 0
    conservative_floor_applied_count = 0
    conservative_floor_reason_distribution: dict[str, int] = {}
    selected_candidate_violation_count = 0
    previous_shift_candidate_count = 0
    previous_shift_max_delta_count = 0
    previous_shift_selected_count = 0
    previous_shift_projection_applied_count = 0
    previous_shift_first_delta_after_projection_max = 0.0
    previous_shift_projection_abs_sums = {field: 0.0 for field in ACTION_FIELDS}
    previous_shift_internal_max_after_projection = {field: 0.0 for field in ACTION_FIELDS}
    for row in successes:
        attribution_mode = str(row.get("candidate_attribution_mode") or "selected_step_coincidence_attribution")
        attribution_mode_distribution[attribution_mode] = attribution_mode_distribution.get(attribution_mode, 0) + 1
        conservative_reason = str(row.get("conservative_prior_no_candidate_reason") or "")
        if conservative_reason:
            conservative_no_candidate_reason_distribution[conservative_reason] = (
                conservative_no_candidate_reason_distribution.get(conservative_reason, 0) + 1
            )
        risk_flags = row.get("risk_flags") or {}
        risk_active = bool(isinstance(risk_flags, Mapping) and risk_flags.get("any_risk", False))
        conservative_generated = int(row.get("conservative_prior_generated_template_count", 0) or 0)
        conservative_status = str(row.get("conservative_prior_margin_status") or "")
        if risk_active and conservative_generated > 0:
            conservative_risk_candidate_steps += 1
        if risk_active and conservative_status in {"selected", "lower_score"}:
            conservative_risk_feasible_steps += 1
        if risk_active and conservative_reason:
            conservative_risk_no_candidate_count += 1
        if (not risk_active) and conservative_generated > 0:
            conservative_normal_candidate_steps += 1
        if bool(row.get("conservative_prior_candidate_count_override_applied", False)):
            conservative_floor_applied_count += 1
            floor_reason = str(row.get("conservative_prior_candidate_count_override_reason") or "unknown")
            conservative_floor_reason_distribution[floor_reason] = (
                conservative_floor_reason_distribution.get(floor_reason, 0) + 1
            )
        rule_margin = row.get("rule_prior_selected_score_margin")
        try:
            if rule_margin is not None:
                rule_prior_margin_values.append(float(rule_margin))
        except Exception:
            pass
        conservative_margin = row.get("conservative_prior_selected_score_margin")
        try:
            if conservative_margin is not None:
                conservative_prior_margin_values.append(float(conservative_margin))
        except Exception:
            pass
        selected_candidate_violation_count += int(row.get("selected_candidate_violation_count", 0) or 0)
        previous_shift_candidate_count += int(row.get("previous_shift_all_candidate_count", 0) or 0)
        previous_shift_max_delta_count += int(row.get("previous_shift_all_max_delta_count", 0) or 0)
        previous_shift_selected_count += int(row.get("previous_shift_all_selected_count", 0) or 0)
        applied = int(row.get("previous_shift_projection_applied_count", 0) or 0)
        previous_shift_projection_applied_count += applied
        previous_shift_first_delta_after_projection_max = max(
            previous_shift_first_delta_after_projection_max,
            abs(float(row.get("previous_shift_first_delta_after_projection_max", 0.0) or 0.0)),
        )
        projection_delta = row.get("previous_shift_projection_mean_abs_delta_by_field") or {}
        if isinstance(projection_delta, Mapping):
            for field in ACTION_FIELDS:
                previous_shift_projection_abs_sums[field] += (
                    abs(float(projection_delta.get(field, 0.0) or 0.0)) * max(applied, 1)
                )
        internal_delta = row.get("previous_shift_internal_max_delta_after_projection_by_field") or {}
        if isinstance(internal_delta, Mapping):
            for field in ACTION_FIELDS:
                previous_shift_internal_max_after_projection[field] = max(
                    previous_shift_internal_max_after_projection[field],
                    abs(float(internal_delta.get(field, 0.0) or 0.0)),
                )
        source_prior = str(row.get("shadow_selected_source_prior") or "")
        template_name = str(row.get("shadow_selected_template_name") or "")
        if (not source_prior) or (not template_name):
            source_prior, template_name = parse_selected_sequence_id(row.get("shadow_selected_sequence_id"))
        if source_prior:
            selected_source_prior_distribution[source_prior] = selected_source_prior_distribution.get(source_prior, 0) + 1
        if template_name:
            selected_template_name_distribution[template_name] = selected_template_name_distribution.get(template_name, 0) + 1
        reasons = row.get("hard_constraint_violation_reason_distribution") or {}
        if isinstance(reasons, Mapping):
            for reason, count in reasons.items():
                key = str(reason)
                violation_reason_distribution[key] = violation_reason_distribution.get(key, 0) + int(count or 0)
        fields = row.get("hard_constraint_violation_field_distribution") or {}
        if isinstance(fields, Mapping):
            for field, count in fields.items():
                key = str(field)
                violation_field_distribution[key] = violation_field_distribution.get(key, 0) + int(count or 0)
        selected_reasons = row.get("selected_candidate_violation_reason_distribution") or {}
        if isinstance(selected_reasons, Mapping):
            for reason, count in selected_reasons.items():
                key = str(reason)
                selected_violation_reason_distribution[key] = selected_violation_reason_distribution.get(key, 0) + int(count or 0)
        selected_fields = row.get("selected_candidate_violation_field_distribution") or {}
        if isinstance(selected_fields, Mapping):
            for field, count in selected_fields.items():
                key = str(field)
                selected_violation_field_distribution[key] = selected_violation_field_distribution.get(key, 0) + int(count or 0)
    return {
        "row_count": total,
        "audit_success_rate": len(successes) / max(total, 1),
        "audit_failure_rate": (total - len(successes)) / max(total, 1),
        "final_action_invariant_rate": sum(1 for row in successes if row.get("final_action_invariant_verified")) / max(len(successes), 1),
        "fallback_rate": sum(1 for row in successes if row.get("fallback_triggered")) / max(len(successes), 1),
        "mean_candidate_count": sum(float(row.get("candidate_count", 0.0)) for row in successes) / max(len(successes), 1),
        "mean_feasible_candidate_count": sum(float(row.get("feasible_candidate_count", 0.0)) for row in successes) / max(len(successes), 1),
        "mean_infeasible_candidate_ratio": sum(float(row.get("infeasible_candidate_ratio", 0.0)) for row in successes) / max(len(successes), 1),
        "hard_constraint_violation_count": sum(int(row.get("hard_constraint_violation_count", 0) or 0) for row in successes),
        "hard_constraint_violation_reason_distribution": dict(sorted(violation_reason_distribution.items())),
        "hard_constraint_violation_field_distribution": dict(sorted(violation_field_distribution.items())),
        "candidate_attribution_mode_distribution": dict(sorted(attribution_mode_distribution.items())),
        "selected_candidate_violation_count": selected_candidate_violation_count,
        "selected_candidate_violation_reason_distribution": dict(sorted(selected_violation_reason_distribution.items())),
        "selected_candidate_violation_field_distribution": dict(sorted(selected_violation_field_distribution.items())),
        "previous_shift_all_candidate_count": previous_shift_candidate_count,
        "previous_shift_all_max_delta_count": previous_shift_max_delta_count,
        "previous_shift_all_selected_count": previous_shift_selected_count,
        "previous_shift_projection_applied_count": previous_shift_projection_applied_count,
        "previous_shift_first_delta_after_projection_max": previous_shift_first_delta_after_projection_max,
        "previous_shift_projection_mean_abs_delta_by_field": {
            field: previous_shift_projection_abs_sums[field] / max(previous_shift_projection_applied_count, 1)
            for field in ACTION_FIELDS
        },
        "previous_shift_internal_max_delta_after_projection_by_field": previous_shift_internal_max_after_projection,
        "selected_source_prior_distribution": dict(sorted(selected_source_prior_distribution.items())),
        "selected_template_name_distribution": dict(sorted(selected_template_name_distribution.items())),
        "conservative_no_candidate_reason_distribution": dict(
            sorted(conservative_no_candidate_reason_distribution.items())
        ),
        "conservative_risk_candidate_steps": conservative_risk_candidate_steps,
        "conservative_risk_feasible_steps": conservative_risk_feasible_steps,
        "conservative_risk_no_candidate_count": conservative_risk_no_candidate_count,
        "conservative_normal_candidate_steps": conservative_normal_candidate_steps,
        "conservative_floor_applied_count": conservative_floor_applied_count,
        "conservative_floor_reason_distribution": dict(sorted(conservative_floor_reason_distribution.items())),
        "conservative_margin_coverage_count": len(conservative_prior_margin_values),
        "conservative_selected_score_margin_mean": (
            sum(conservative_prior_margin_values) / max(len(conservative_prior_margin_values), 1)
        ),
        "rule_prior_margin_coverage_count": len(rule_prior_margin_values),
        "rule_prior_selected_score_margin_mean": (
            sum(rule_prior_margin_values) / max(len(rule_prior_margin_values), 1)
        ),
        "mean_abs_action_diff_by_field": {
            field: action_abs_sums[field] / max(diff_count, 1)
            for field in ACTION_FIELDS
        },
        "max_action_diff_by_field": action_abs_max,
        "active_regime_distribution": _distribution(row.get("active_regime_after") for row in successes),
        "projection_level_distribution": _distribution(row.get("projection_level") for row in successes),
        "score_validity_distribution": _distribution(row.get("score_validity") for row in successes),
        "prior_confidence_mean": {
            key: sum(values) / max(len(values), 1)
            for key, values in prior_confidence.items()
        },
    }


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts
