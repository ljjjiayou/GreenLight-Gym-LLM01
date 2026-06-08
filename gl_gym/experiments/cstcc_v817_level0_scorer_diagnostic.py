"""v817.0 offline Level-0 scorer diagnostic for C-STCC shadow traces.

This stage explains scorer competition structure from an existing opt-in
shadow trace. It does not call online LLM, run simulator rollout, change final
actions, or make controller-quality claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_TRACE_CSV = Path(
    "gl_gym/result/diagnostics/cstcc_v8170_scorer_diagnostic_trace_20260606_y2020_d240_s42_n720.csv"
)
DEFAULT_SUMMARY_JSON = Path(
    "gl_gym/result/diagnostics/cstcc_v8170_scorer_diagnostic_trace_20260606_y2020_d240_s42_n720.json"
)
DEFAULT_JSONL_ROOT = Path("logs/cstcc_shadow/v8170_scorer_diagnostic_20260606_y2020_d240_s42_n720")
DEFAULT_OUTPUT_PREFIX = Path(
    "gl_gym/result/diagnostics/cstcc_v8170_level0_scorer_diagnostic_20260606_y2020_d240_s42_n720"
)

BOUNDARY_FALSE_COUNTS = (
    "online_llm_called_steps",
    "predictive_rollout_executed_steps",
    "real_tomato_safety_projection_steps",
    "final_action_changed_steps",
)
RISK_MARGIN_SMALL = 0.02
CONSERVATIVE_MARGIN_SMALL = 0.02


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except Exception:
            return {}
        return dict(decoded) if isinstance(decoded, Mapping) else {}
    return {}


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _load_csv(path: str | Path) -> list[dict[str, Any]]:
    input_path = _resolve(path)
    if not input_path.exists():
        return []
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_json(path: str | Path) -> dict[str, Any]:
    input_path = _resolve(path)
    if not input_path.exists():
        return {}
    return json.loads(input_path.read_text(encoding="utf-8"))


def _load_jsonl_rows(root: str | Path) -> list[dict[str, Any]]:
    input_root = _resolve(root)
    rows: list[dict[str, Any]] = []
    if not input_root.exists():
        return rows
    for path in sorted(input_root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    rows.append(json.loads(text))
    return rows


def _risk_flags(row: Mapping[str, Any]) -> dict[str, bool]:
    flags = _json_obj(row.get("risk_flags") or row.get("cstcc_shadow_risk_flags_json"))
    if flags:
        normalized = {str(key): _bool(value) for key, value in flags.items()}
        normalized["any_risk"] = bool(
            normalized.get("any_risk", False)
            or any(value for key, value in normalized.items() if key != "any_risk")
        )
        return normalized
    rh = _num(row.get("rh_air"), 70.0)
    vpd = _num(row.get("vpd_air"), _num(row.get("vpd_kpa")))
    canopy_dew = _num(row.get("canopy_dew_margin"), 999.0)
    dew_air = _num(row.get("dew_margin_air"), 999.0)
    normalized = {
        "rh_ge_90": rh >= 90.0,
        "dew_risk": canopy_dew < 1.0 or dew_air < 1.0,
        "dry_risk": rh < 55.0,
        "high_vpd": vpd > 1.20,
    }
    normalized["humidity_or_dew_risk"] = normalized["rh_ge_90"] or normalized["dew_risk"]
    normalized["dry_or_high_vpd_risk"] = normalized["dry_risk"] or normalized["high_vpd"]
    normalized["any_risk"] = any(normalized.values())
    return normalized


def _row_by_step(rows: list[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        try:
            out[int(row.get("step", len(out)))] = row
        except Exception:
            continue
    return out


def _boundary_summary(csv_rows: list[Mapping[str, Any]], summary_json: Mapping[str, Any], jsonl_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    trace = summary_json.get("trace_summary", {}) if isinstance(summary_json.get("trace_summary", {}), Mapping) else {}
    answers = summary_json.get("answers", {}) if isinstance(summary_json.get("answers", {}), Mapping) else {}
    enabled = [row for row in csv_rows if _bool(row.get("cstcc_shadow_enabled"))]
    return {
        "csv_row_count": len(csv_rows),
        "jsonl_row_count": len(jsonl_rows),
        "audit_success_rate": _num(trace.get("cstcc_shadow_audit_success_rate"), _num(answers.get("audit_success_rate"))),
        "final_action_invariant_rate": _num(
            trace.get("cstcc_shadow_final_action_invariant_rate"),
            _num(answers.get("final_action_invariant_rate")),
        ),
        "final_action_changed_steps": int(_num(trace.get("cstcc_shadow_final_action_changed_steps"))),
        "online_llm_called_steps": int(_num(trace.get("cstcc_shadow_online_llm_called_steps"))),
        "predictive_rollout_executed_steps": int(_num(trace.get("cstcc_shadow_predictive_rollout_executed_steps"))),
        "real_tomato_safety_projection_steps": int(_num(trace.get("cstcc_shadow_real_tomato_safety_projection_steps"))),
        "fallback_rate": _num(trace.get("cstcc_shadow_fallback_rate"), _num(answers.get("fallback_rate"))),
        "mean_infeasible_candidate_ratio": _num(trace.get("cstcc_shadow_mean_infeasible_candidate_ratio")),
        "previous_shift_all_max_delta_count": int(_num(trace.get("cstcc_shadow_previous_shift_all_max_delta_count"))),
        "selected_candidate_max_delta_count": int(_num(trace.get("cstcc_shadow_selected_candidate_violation_count"))),
        "json_row_size_kb_max": _num(answers.get("json_size_kb_max")),
        "enabled_steps": len(enabled),
    }


def _missing_schema_fields(jsonl_rows: list[Mapping[str, Any]]) -> list[str]:
    required = (
        "prior_generation_diagnostics",
        "score_margin_to_selected_by_prior",
        "score_component_by_candidate",
        "rule_conservative_competitiveness",
        "risk_flags",
    )
    if not jsonl_rows:
        return ["jsonl_rows_missing"]
    missing: list[str] = []
    first_success = next((row for row in jsonl_rows if row.get("audit_success", False)), jsonl_rows[0])
    for field in required:
        if field not in first_success:
            missing.append(f"missing_field:{field}")
    return missing


def _component_margin_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    keys: set[str] = set()
    for record in records:
        components = record.get("selected_minus_competitor") or {}
        if isinstance(components, Mapping):
            keys.update(str(key) for key in components)
    means: dict[str, float] = {}
    abs_means: dict[str, float] = {}
    for key in sorted(keys):
        values = [
            _num((record.get("selected_minus_competitor") or {}).get(key))
            for record in records
            if isinstance(record.get("selected_minus_competitor") or {}, Mapping)
        ]
        means[key] = float(mean(values)) if values else 0.0
        abs_means[key] = float(mean(abs(value) for value in values)) if values else 0.0
    dominant = max(abs_means, key=abs_means.get) if abs_means else ""
    return {
        "mean_selected_minus_competitor": means,
        "mean_abs_selected_minus_competitor": abs_means,
        "dominant_component": dominant,
        "dominant_component_mean_abs": abs_means.get(dominant, 0.0) if dominant else 0.0,
    }


def _score_diagnostics(csv_rows: list[Mapping[str, Any]], jsonl_rows: list[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    csv_by_step = _row_by_step(csv_rows)
    rule_records: list[dict[str, Any]] = []
    conservative_rows: list[dict[str, Any]] = []
    by_risk_flag: dict[str, list[float]] = {}
    by_template: dict[str, list[float]] = {}
    risk_rows = 0
    risk_rule_margin_rows = 0
    conservative_margin_values: list[float] = []
    conservative_risk_margin_values: list[float] = []
    conservative_risk_candidate_steps = 0
    conservative_risk_feasible_steps = 0
    conservative_risk_selected_steps = 0
    conservative_normal_candidate_steps = 0
    conservative_floor_applied_count = 0
    for jsonl_row in jsonl_rows:
        if not jsonl_row.get("audit_success", False):
            continue
        step = int(_num(jsonl_row.get("step"), len(rule_records)))
        csv_row = csv_by_step.get(step, {})
        merged = {**dict(csv_row), **dict(jsonl_row)}
        flags = _risk_flags(merged)
        if flags.get("any_risk", False):
            risk_rows += 1
        margins = _json_obj(jsonl_row.get("score_margin_to_selected_by_prior"))
        prior_diag = _json_obj(jsonl_row.get("prior_generation_diagnostics"))
        rule_margin = margins.get("rule_prior") if isinstance(margins.get("rule_prior"), Mapping) else {}
        conservative_margin = margins.get("conservative_prior") if isinstance(margins.get("conservative_prior"), Mapping) else {}
        conservative_diag = prior_diag.get("conservative_prior") if isinstance(prior_diag.get("conservative_prior"), Mapping) else {}
        conservative_generated = int(conservative_diag.get("generated_template_count", 0) or 0)
        conservative_status = str(conservative_margin.get("status") or "")
        conservative_total_margin = conservative_margin.get("total_margin")
        conservative_margin_value = None
        if conservative_total_margin is not None:
            conservative_margin_value = _num(conservative_total_margin)
            conservative_margin_values.append(conservative_margin_value)
            if flags.get("any_risk", False):
                conservative_risk_margin_values.append(conservative_margin_value)
        if flags.get("any_risk", False) and conservative_generated > 0:
            conservative_risk_candidate_steps += 1
        if flags.get("any_risk", False) and conservative_status in {"selected", "lower_score"}:
            conservative_risk_feasible_steps += 1
        if flags.get("any_risk", False) and conservative_status == "selected":
            conservative_risk_selected_steps += 1
        if (not flags.get("any_risk", False)) and conservative_generated > 0:
            conservative_normal_candidate_steps += 1
        if bool(conservative_diag.get("candidate_count_override_applied", False)):
            conservative_floor_applied_count += 1
        conservative_rows.append(
            {
                "step": step,
                "risk": flags.get("any_risk", False),
                "confidence": conservative_diag.get("confidence"),
                "candidate_count": conservative_diag.get("candidate_count"),
                "candidate_count_before_override": conservative_diag.get("candidate_count_before_override"),
                "candidate_count_after_override": conservative_diag.get("candidate_count_after_override"),
                "candidate_count_override_applied": conservative_diag.get("candidate_count_override_applied"),
                "candidate_count_override_reason": conservative_diag.get("candidate_count_override_reason"),
                "candidate_count_override_flags": _compact_json(conservative_diag.get("candidate_count_override_flags", [])),
                "risk_category_flags": _compact_json(conservative_diag.get("risk_category_flags", {})),
                "generated_template_count": conservative_diag.get("generated_template_count"),
                "no_candidate_reason": conservative_diag.get("no_candidate_reason"),
                "no_candidate_reasons": _compact_json(conservative_diag.get("no_candidate_reasons", [])),
                "confidence_reasons": _compact_json(conservative_diag.get("confidence_reasons", {})),
                "margin_status": conservative_status,
                "selected_score_margin": "" if conservative_margin_value is None else conservative_margin_value,
                "best_candidate_id": conservative_margin.get("best_candidate_id"),
                "best_template_name": conservative_margin.get("best_template_name"),
            }
        )
        if rule_margin:
            total_margin = rule_margin.get("total_margin")
            if total_margin is not None:
                risk_rule_margin_rows += 1 if flags.get("any_risk", False) else 0
                record = {
                    "step": step,
                    "hour_of_day": _num(merged.get("hour_of_day")),
                    "active_regime": jsonl_row.get("active_regime_after") or merged.get("cstcc_shadow_active_regime_after"),
                    "risk_flags": flags,
                    "selected_candidate_id": rule_margin.get("selected_candidate_id"),
                    "selected_source_prior": rule_margin.get("selected_source_prior"),
                    "selected_template_name": rule_margin.get("selected_template_name"),
                    "rule_best_candidate_id": rule_margin.get("best_candidate_id"),
                    "rule_best_template_name": rule_margin.get("best_template_name"),
                    "selected_score": rule_margin.get("selected_score"),
                    "rule_best_score": rule_margin.get("best_score"),
                    "total_margin": _num(total_margin),
                    "status": rule_margin.get("status"),
                    "selected_minus_competitor": rule_margin.get("selected_minus_competitor", {}),
                }
                rule_records.append(record)
                template = str(record["rule_best_template_name"] or "unknown")
                by_template.setdefault(template, []).append(float(record["total_margin"]))
                for flag, active in flags.items():
                    if active:
                        by_risk_flag.setdefault(str(flag), []).append(float(record["total_margin"]))
    top_loss = sorted(rule_records, key=lambda item: float(item.get("total_margin", 0.0)), reverse=True)[:50]
    risk_flag_rows = [
        {
            "risk_flag": flag,
            "count": len(values),
            "mean_rule_margin": float(mean(values)) if values else 0.0,
            "max_rule_margin": max(values) if values else 0.0,
        }
        for flag, values in sorted(by_risk_flag.items())
    ]
    template_rows = [
        {
            "rule_best_template_name": template,
            "count": len(values),
            "mean_rule_margin": float(mean(values)) if values else 0.0,
            "max_rule_margin": max(values) if values else 0.0,
        }
        for template, values in sorted(by_template.items())
    ]
    conservative_reason_distribution = _distribution(
        row.get("no_candidate_reason") for row in conservative_rows if row.get("no_candidate_reason")
    )
    risk_margins = [float(row["total_margin"]) for row in rule_records if row.get("risk_flags", {}).get("any_risk", False)]
    diagnostics = {
        "risk_row_count": risk_rows,
        "rule_margin_row_count": len(rule_records),
        "risk_rule_margin_row_count": risk_rule_margin_rows,
        "risk_rule_margin_coverage": risk_rule_margin_rows / max(risk_rows, 1),
        "rule_margin_mean": float(mean([float(row["total_margin"]) for row in rule_records])) if rule_records else 0.0,
        "risk_rule_margin_mean": float(mean(risk_margins)) if risk_margins else 0.0,
        "risk_rule_margin_max": max(risk_margins) if risk_margins else 0.0,
        "rule_best_template_distribution": _distribution(row.get("rule_best_template_name") for row in rule_records),
        "conservative_no_candidate_reason_distribution": conservative_reason_distribution,
        "conservative_no_candidate_count": sum(1 for row in conservative_rows if row.get("no_candidate_reason")),
        "conservative_risk_no_candidate_count": sum(
            1 for row in conservative_rows if row.get("risk") and row.get("no_candidate_reason")
        ),
        "conservative_risk_candidate_steps": conservative_risk_candidate_steps,
        "conservative_risk_feasible_steps": conservative_risk_feasible_steps,
        "conservative_risk_selected_steps": conservative_risk_selected_steps,
        "conservative_normal_candidate_steps": conservative_normal_candidate_steps,
        "conservative_floor_applied_count": conservative_floor_applied_count,
        "conservative_margin_coverage": len(conservative_margin_values) / max(len(conservative_rows), 1),
        "conservative_risk_margin_coverage": len(conservative_risk_margin_values) / max(risk_rows, 1),
        "conservative_mean_margin": float(mean(conservative_margin_values)) if conservative_margin_values else 0.0,
        "conservative_risk_mean_margin": (
            float(mean(conservative_risk_margin_values)) if conservative_risk_margin_values else 0.0
        ),
        "rule_component_margin_summary": _component_margin_summary(rule_records),
    }
    return diagnostics, top_loss, risk_flag_rows, template_rows, conservative_rows


def _route_next_action(boundary: Mapping[str, Any], missing_fields: list[str], diagnostics: Mapping[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if (
        float(boundary.get("audit_success_rate", 0.0)) < 1.0
        or float(boundary.get("final_action_invariant_rate", 0.0)) < 1.0
        or any(int(boundary.get(field, 0) or 0) for field in BOUNDARY_FALSE_COUNTS)
    ):
        return "stop_and_repair_cstcc_shadow_boundary", ["boundary_or_invariant_failure"]
    if missing_fields:
        return "v8170_trace_schema_repair_or_rerun_plan", missing_fields
    health_issue = (
        float(boundary.get("fallback_rate", 0.0) or 0.0) > 0.0
        or float(boundary.get("mean_infeasible_candidate_ratio", 0.0) or 0.0) > 0.03
        or int(boundary.get("previous_shift_all_max_delta_count", 0) or 0) > 0
        or int(boundary.get("selected_candidate_max_delta_count", 0) or 0) > 0
        or float(boundary.get("json_row_size_kb_max", 0.0) or 0.0) >= 64.0
    )
    if health_issue:
        return "v8171_candidate_allocation_repair", ["candidate_health_or_log_pressure_issue"]
    risk_rows = int(diagnostics.get("risk_row_count", 0) or 0)
    risk_no_candidate = int(diagnostics.get("conservative_risk_no_candidate_count", 0) or 0)
    risk_candidates = int(diagnostics.get("conservative_risk_candidate_steps", 0) or 0)
    risk_feasible = int(diagnostics.get("conservative_risk_feasible_steps", 0) or 0)
    if risk_rows > 0 and risk_no_candidate > 0:
        reasons.append("conservative_prior_has_no_candidate")
        if int(diagnostics.get("conservative_floor_applied_count", 0) or 0) > 0:
            return "v8171_candidate_allocation_repair", reasons
        return "v8171_conservative_risk_candidate_allocation_plan", reasons
    if risk_rows > 0 and risk_candidates <= 0:
        return "v8171_conservative_risk_candidate_allocation_plan", ["conservative_risk_candidate_steps_zero"]
    if risk_rows > 0 and risk_feasible <= 0:
        return "v8171_candidate_allocation_repair", ["conservative_risk_feasible_steps_zero"]
    if float(diagnostics.get("risk_rule_margin_coverage", 0.0) or 0.0) < 0.99:
        return "v8170_score_margin_schema_repair_plan", ["risk_rule_margin_coverage_below_0_99"]
    conservative_risk_margin = float(diagnostics.get("conservative_risk_mean_margin", 0.0) or 0.0)
    conservative_risk_selected = int(diagnostics.get("conservative_risk_selected_steps", 0) or 0)
    if conservative_risk_selected <= 0 and conservative_risk_margin <= CONSERVATIVE_MARGIN_SMALL:
        return "v8172_risk_window_tiebreaker_plan", ["conservative_risk_margin_small"]
    if conservative_risk_selected <= 0:
        return "v8172_risk_aligned_bonus_calibration_plan", ["conservative_risk_margin_large"]
    if float(diagnostics.get("risk_rule_margin_mean", 0.0) or 0.0) <= RISK_MARGIN_SMALL:
        return "v8172_risk_window_tiebreaker_plan", ["risk_rule_margin_small"]
    return "multi_scenario_v817_scan", ["single_scenario_allocation_structure_clean"]


def build_report(
    *,
    trace_csv_rows: list[Mapping[str, Any]],
    summary_json: Mapping[str, Any],
    jsonl_rows: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    boundary = _boundary_summary(trace_csv_rows, summary_json, jsonl_rows)
    missing_fields = _missing_schema_fields(jsonl_rows)
    diagnostics, top_loss, risk_flag_rows, template_rows, conservative_rows = _score_diagnostics(trace_csv_rows, jsonl_rows)
    next_action, readiness_reasons = _route_next_action(boundary, missing_fields, diagnostics)
    report = {
        "schema_version": "cstcc_v8170_level0_scorer_diagnostic_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_level": "opt-in shadow trace diagnostic",
        "boundaries": {
            "online_llm_called": False,
            "predictive_rollout_executed": False,
            "real_tomato_safety_projection": False,
            "final_action_changed": False,
            "performance_claim_allowed": False,
            "promotion_evidence": False,
        },
        "boundary_runtime_safety": boundary,
        "schema_issues": missing_fields,
        "scorer_diagnostics": diagnostics,
        "acceptance": {
            "risk_window_rule_margin_coverage_ge_0_99": diagnostics.get("risk_rule_margin_coverage", 0.0) >= 0.99,
            "conservative_risk_candidate_steps_gt_0": diagnostics.get("conservative_risk_candidate_steps", 0) > 0,
            "conservative_risk_feasible_steps_gt_0": diagnostics.get("conservative_risk_feasible_steps", 0) > 0,
            "conservative_risk_no_candidate_count_eq_0": diagnostics.get("conservative_risk_no_candidate_count", 0) == 0,
            "json_row_size_kb_max_le_64": boundary.get("json_row_size_kb_max", 0.0) <= 64.0,
        },
        "next_action": next_action,
        "readiness_reasons": readiness_reasons,
        "notes": [
            "v817.0 is diagnostic-only.",
            "Reward and controller-quality claims remain out of scope.",
        ],
    }
    return report, {
        "top_loss": top_loss,
        "risk_flag": risk_flag_rows,
        "template": template_rows,
        "conservative": conservative_rows,
    }


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _compact_json(value) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            )


def build_markdown(report: Mapping[str, Any]) -> str:
    boundary = report.get("boundary_runtime_safety", {})
    diagnostics = report.get("scorer_diagnostics", {})
    lines = [
        "# C-STCC v817.0 Level-0 Scorer Diagnostic",
        "",
        "Offline scorer diagnostic only. No online LLM, no predictive rollout, no final-action change, no reward claim.",
        "",
        "## Boundary",
        "",
        f"- Audit success rate: {float(boundary.get('audit_success_rate', 0.0)):.3f}",
        f"- Final-action invariant rate: {float(boundary.get('final_action_invariant_rate', 0.0)):.3f}",
        f"- Final action changed steps: {boundary.get('final_action_changed_steps', 0)}",
        f"- Online LLM / rollout / real projection steps: {boundary.get('online_llm_called_steps', 0)} / {boundary.get('predictive_rollout_executed_steps', 0)} / {boundary.get('real_tomato_safety_projection_steps', 0)}",
        f"- JSON row max KB: {float(boundary.get('json_row_size_kb_max', 0.0)):.3f}",
        "",
        "## Scorer Diagnostic",
        "",
        f"- Risk rows: {diagnostics.get('risk_row_count', 0)}",
        f"- Risk rule margin coverage: {float(diagnostics.get('risk_rule_margin_coverage', 0.0)):.3f}",
        f"- Risk rule margin mean/max: {float(diagnostics.get('risk_rule_margin_mean', 0.0)):.6f} / {float(diagnostics.get('risk_rule_margin_max', 0.0)):.6f}",
        f"- Conservative no-candidate count: {diagnostics.get('conservative_no_candidate_count', 0)}",
        f"- Conservative risk candidate / feasible / no-candidate steps: {diagnostics.get('conservative_risk_candidate_steps', 0)} / {diagnostics.get('conservative_risk_feasible_steps', 0)} / {diagnostics.get('conservative_risk_no_candidate_count', 0)}",
        f"- Conservative normal candidate steps: {diagnostics.get('conservative_normal_candidate_steps', 0)}",
        f"- Conservative floor applied count: {diagnostics.get('conservative_floor_applied_count', 0)}",
        f"- Conservative risk margin coverage/mean: {float(diagnostics.get('conservative_risk_margin_coverage', 0.0)):.3f} / {float(diagnostics.get('conservative_risk_mean_margin', 0.0)):.6f}",
        f"- Conservative no-candidate reasons: `{_compact_json(diagnostics.get('conservative_no_candidate_reason_distribution', {}))}`",
        f"- Rule best templates: `{_compact_json(diagnostics.get('rule_best_template_distribution', {}))}`",
        "",
        "## Decision",
        "",
        f"- Next action: {report.get('next_action')}",
        f"- Readiness reasons: {', '.join(report.get('readiness_reasons', [])) or 'none'}",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    report: Mapping[str, Any],
    tables: Mapping[str, list[Mapping[str, Any]]],
    output_prefix: str | Path,
) -> dict[str, str]:
    prefix = _resolve(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    stem = prefix.name
    parent = prefix.parent
    top_loss_path = parent / stem.replace("level0_scorer_diagnostic", "rule_margin_top_loss")
    risk_flag_path = parent / stem.replace("level0_scorer_diagnostic", "rule_margin_by_risk_flag")
    template_path = parent / stem.replace("level0_scorer_diagnostic", "rule_margin_by_template")
    conservative_path = parent / stem.replace("level0_scorer_diagnostic", "conservative_no_candidate_reasons")
    paths = {
        "json": json_path,
        "md": md_path,
        "rule_margin_top_loss_csv": top_loss_path.with_suffix(".csv"),
        "rule_margin_by_risk_flag_csv": risk_flag_path.with_suffix(".csv"),
        "rule_margin_by_template_csv": template_path.with_suffix(".csv"),
        "conservative_no_candidate_reasons_csv": conservative_path.with_suffix(".csv"),
    }
    json_path.write_text(_compact_json(report) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    _write_csv(paths["rule_margin_top_loss_csv"], tables.get("top_loss", []))
    _write_csv(paths["rule_margin_by_risk_flag_csv"], tables.get("risk_flag", []))
    _write_csv(paths["rule_margin_by_template_csv"], tables.get("template", []))
    _write_csv(paths["conservative_no_candidate_reasons_csv"], tables.get("conservative", []))
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v817.0 C-STCC Level-0 scorer diagnostic.")
    parser.add_argument("--trace-csv", type=str, default=str(DEFAULT_TRACE_CSV))
    parser.add_argument("--summary-json", type=str, default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--jsonl-root", type=str, default=str(DEFAULT_JSONL_ROOT))
    parser.add_argument("--output-prefix", type=str, default=str(DEFAULT_OUTPUT_PREFIX))
    args = parser.parse_args()

    report, tables = build_report(
        trace_csv_rows=_load_csv(args.trace_csv),
        summary_json=_load_json(args.summary_json),
        jsonl_rows=_load_jsonl_rows(args.jsonl_root),
    )
    outputs = write_outputs(report=report, tables=tables, output_prefix=args.output_prefix)
    print(json.dumps({"outputs": outputs, "next_action": report["next_action"], "readiness_reasons": report["readiness_reasons"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
