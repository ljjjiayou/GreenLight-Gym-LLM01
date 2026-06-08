"""v81.5 offline analytics for C-STCC shadow traces.

This stage explains shadow selection structure from an existing v81.1
full-episode trace. It does not call online LLM, execute rollout, change final
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


ACTION_FIELDS = (
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
)

DEFAULT_CSV = Path("gl_gym/result/diagnostics/cstcc_v811_full_episode_trace_20260605_y2020_d240_s42_n720.csv")
DEFAULT_SUMMARY_JSON = Path("gl_gym/result/diagnostics/cstcc_v811_full_episode_trace_20260605_y2020_d240_s42_n720.json")
DEFAULT_JSONL_ROOT = Path("logs/cstcc_shadow/v811_full_episode_20260605_y2020_d240_s42_n720")
DEFAULT_OUTPUT_PREFIX = Path("gl_gym/result/diagnostics/cstcc_v815_shadow_analytics_20260605_y2020_d240_s42_n720")

FALLBACK_RATE_MAX = 0.05
MEAN_INFEASIBLE_RATIO_MAX = 0.10
LATENCY_P95_MAX_MS = 250.0
JSON_ROW_MAX_KB = 64.0
RISK_HOLD_ALL_MAX = 0.70
PRIOR_DOMINANCE_MAX = 0.95
MAX_DELTA_CONCENTRATION_MAX = 0.60
TOP_DIFF_L1_MAX = 0.75
REGIME_ALIGNMENT_MIN = 0.50


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


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


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except Exception:
            return []
        return list(decoded) if isinstance(decoded, list) else []
    return []


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(100.0, float(percentile))) / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    frac = position - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _add_count(counts: dict[str, int], key: Any, count: int = 1) -> None:
    text = str(key or "unknown")
    counts[text] = counts.get(text, 0) + int(count or 0)


def _risk_flags(row: Mapping[str, Any]) -> dict[str, bool]:
    shadow_flags = _json_obj(row.get("cstcc_shadow_risk_flags_json"))
    if shadow_flags:
        flags = {str(key): _bool(value) for key, value in shadow_flags.items()}
        flags["any_risk"] = any(value for key, value in flags.items() if key != "any_risk") or flags.get("any_risk", False)
        return flags
    rh = _num(row.get("rh_air"))
    vpd = _num(row.get("vpd_air"), _num(row.get("vpd_kpa")))
    canopy_dew = _num(row.get("canopy_dew_margin"), 999.0)
    dew_margin_air = _num(row.get("dew_margin_air"), 999.0)
    flags = {
        "rh_ge_90": rh >= 90.0,
        "dew_risk": _bool(row.get("dew_risk")),
        "canopy_dew_margin_lt1": canopy_dew < 1.0,
        "canopy_dew_margin_lt0": canopy_dew < 0.0,
        "dew_margin_air_lt1": dew_margin_air < 1.0,
        "dry_risk": _bool(row.get("dry_risk")),
        "dry_vent_risk": _bool(row.get("dry_vent_risk")),
        "high_vpd": vpd > 1.20,
    }
    flags["humidity_or_dew_risk"] = bool(
        flags["rh_ge_90"] or flags["dew_risk"] or flags["canopy_dew_margin_lt1"] or flags["canopy_dew_margin_lt0"]
    )
    flags["any_risk"] = any(flags.values())
    return flags


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
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def _required_field_issues(rows: list[Mapping[str, Any]]) -> list[str]:
    required = (
        "cstcc_shadow_enabled",
        "cstcc_shadow_audit_success",
        "cstcc_shadow_final_action_invariant_verified",
        "cstcc_shadow_final_action_changed",
        "cstcc_shadow_online_llm_called",
        "cstcc_shadow_predictive_rollout_executed",
        "cstcc_shadow_real_tomato_safety_projection",
        "cstcc_shadow_selected_source_prior",
        "cstcc_shadow_selected_template_name",
        "cstcc_shadow_action_diff_json",
        "cstcc_shadow_selected_first_action_json",
        "cstcc_shadow_current_runtime_final_action_json",
        "cstcc_shadow_hard_constraint_violation_reason_distribution_json",
        "cstcc_shadow_hard_constraint_violation_field_distribution_json",
    )
    if not rows:
        return ["csv_rows_missing"]
    missing: list[str] = []
    for field in required:
        if field not in rows[0]:
            missing.append(f"missing_field:{field}")
    return missing


def _boundary_summary(rows: list[Mapping[str, Any]], v811_summary: Mapping[str, Any], jsonl_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    enabled = [row for row in rows if _bool(row.get("cstcc_shadow_enabled"))]
    success = [row for row in enabled if _bool(row.get("cstcc_shadow_audit_success"))]
    latencies = [_num(row.get("cstcc_shadow_audit_latency_ms")) for row in enabled]
    json_sizes = [_num(row.get("cstcc_shadow_audit_json_size_kb")) for row in enabled]
    trace_summary = v811_summary.get("trace_summary", {}) if isinstance(v811_summary.get("trace_summary", {}), Mapping) else {}
    return {
        "csv_row_count": len(rows),
        "jsonl_row_count": len(jsonl_rows),
        "enabled_steps": len(enabled),
        "audit_success_rate": len(success) / max(len(enabled), 1),
        "final_action_invariant_rate": (
            sum(_bool(row.get("cstcc_shadow_final_action_invariant_verified")) for row in enabled)
            / max(len(enabled), 1)
        ),
        "final_action_changed_steps": sum(_bool(row.get("cstcc_shadow_final_action_changed")) for row in enabled),
        "online_llm_called_steps": sum(_bool(row.get("cstcc_shadow_online_llm_called")) for row in enabled),
        "predictive_rollout_executed_steps": sum(
            _bool(row.get("cstcc_shadow_predictive_rollout_executed")) for row in enabled
        ),
        "real_tomato_safety_projection_steps": sum(
            _bool(row.get("cstcc_shadow_real_tomato_safety_projection")) for row in enabled
        ),
        "runtime_error_steps": int(_num(trace_summary.get("runtime_error_steps"))),
        "audit_latency_ms_mean": mean(latencies) if latencies else 0.0,
        "audit_latency_ms_p95": _percentile(latencies, 95.0),
        "json_row_size_kb_mean": mean(json_sizes) if json_sizes else 0.0,
        "json_row_size_kb_max": max(json_sizes) if json_sizes else 0.0,
    }


def _candidate_health(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    success = [row for row in rows if _bool(row.get("cstcc_shadow_audit_success"))]
    candidate_counts = [_num(row.get("cstcc_shadow_candidate_count")) for row in success]
    feasible_counts = [_num(row.get("cstcc_shadow_feasible_candidate_count")) for row in success]
    infeasible_ratios = [_num(row.get("cstcc_shadow_infeasible_candidate_ratio")) for row in success]
    reason_counts: dict[str, int] = {}
    field_counts: dict[str, int] = {}
    selected_reason_counts: dict[str, int] = {}
    selected_field_counts: dict[str, int] = {}
    attribution_modes: dict[str, int] = {}
    selected_candidate_violation_count = 0
    for row in success:
        attribution_mode = str(row.get("cstcc_shadow_candidate_attribution_mode") or "selected_step_coincidence_attribution")
        _add_count(attribution_modes, attribution_mode)
        selected_candidate_violation_count += int(_num(row.get("cstcc_shadow_selected_candidate_violation_count")))
        for reason, count in _json_obj(row.get("cstcc_shadow_hard_constraint_violation_reason_distribution_json")).items():
            _add_count(reason_counts, reason, int(_num(count)))
        for field, count in _json_obj(row.get("cstcc_shadow_hard_constraint_violation_field_distribution_json")).items():
            _add_count(field_counts, field, int(_num(count)))
        for reason, count in _json_obj(row.get("cstcc_shadow_selected_candidate_violation_reason_distribution_json")).items():
            _add_count(selected_reason_counts, reason, int(_num(count)))
        for field, count in _json_obj(row.get("cstcc_shadow_selected_candidate_violation_field_distribution_json")).items():
            _add_count(selected_field_counts, field, int(_num(count)))
    return {
        "mean_candidate_count": mean(candidate_counts) if candidate_counts else 0.0,
        "p95_candidate_count": _percentile(candidate_counts, 95.0),
        "mean_feasible_candidate_count": mean(feasible_counts) if feasible_counts else 0.0,
        "mean_infeasible_candidate_ratio": mean(infeasible_ratios) if infeasible_ratios else 0.0,
        "p95_infeasible_candidate_ratio": _percentile(infeasible_ratios, 95.0),
        "fallback_rate": (
            sum(_bool(row.get("cstcc_shadow_fallback_triggered")) for row in success)
            / max(len(success), 1)
        ),
        "hard_constraint_violation_count": sum(int(_num(row.get("cstcc_shadow_hard_constraint_violation_count"))) for row in success),
        "hard_constraint_violation_reason_distribution": dict(sorted(reason_counts.items())),
        "hard_constraint_violation_field_distribution": dict(sorted(field_counts.items())),
        "candidate_attribution_mode_distribution": dict(sorted(attribution_modes.items())),
        "selected_candidate_violation_count": selected_candidate_violation_count,
        "selected_candidate_violation_reason_distribution": dict(sorted(selected_reason_counts.items())),
        "selected_candidate_violation_field_distribution": dict(sorted(selected_field_counts.items())),
    }


def _risk_label_counts(row: Mapping[str, Any]) -> list[str]:
    return [name for name, value in _risk_flags(row).items() if value and name != "any_risk"]


def _max_delta_attribution(rows: list[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_template: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_regime: dict[str, int] = {}
    by_phase: dict[str, int] = {}
    by_risk: dict[str, int] = {}
    field_by_template: dict[tuple[str, str], int] = {}
    cross_rows: list[dict[str, Any]] = []
    total = 0
    selected_candidate_max_delta_count = 0
    non_selected_candidate_max_delta_count = 0
    candidate_level_rows = 0
    selected_step_rows = 0
    for row in rows:
        candidates = [
            candidate
            for candidate in _json_list(row.get("cstcc_shadow_candidate_violation_provenance_json"))
            if isinstance(candidate, Mapping)
        ]
        regime = str(row.get("cstcc_shadow_active_regime_after") or "unknown")
        phase = str(row.get("phase") or "unknown")
        risk_labels = _risk_label_counts(row) or ["no_risk_flag"]
        selected_id = str(row.get("cstcc_shadow_selected_sequence_id") or "")
        if candidates:
            candidate_level_rows += 1
            for candidate in candidates:
                reasons = _json_obj(candidate.get("violation_reason_distribution"))
                max_delta = int(_num(reasons.get("max_delta")))
                if max_delta <= 0:
                    continue
                total += max_delta
                candidate_id = str(candidate.get("candidate_id") or "")
                template = str(candidate.get("template_name") or "unknown")
                source = str(candidate.get("source_prior") or "unknown")
                if candidate_id == selected_id:
                    selected_candidate_max_delta_count += max_delta
                else:
                    non_selected_candidate_max_delta_count += max_delta
                _add_count(by_template, template, max_delta)
                _add_count(by_source, source, max_delta)
                _add_count(by_regime, regime, max_delta)
                _add_count(by_phase, phase, max_delta)
                for risk in risk_labels:
                    _add_count(by_risk, risk, max_delta)
                for field, count in _json_obj(candidate.get("violation_field_distribution")).items():
                    key = (str(field), template)
                    field_by_template[key] = field_by_template.get(key, 0) + int(_num(count))
            continue

        reasons = _json_obj(row.get("cstcc_shadow_hard_constraint_violation_reason_distribution_json"))
        max_delta = int(_num(reasons.get("max_delta")))
        if max_delta <= 0:
            continue
        selected_step_rows += 1
        total += max_delta
        template = str(row.get("cstcc_shadow_selected_template_name") or "unknown")
        source = str(row.get("cstcc_shadow_selected_source_prior") or "unknown")
        selected_candidate_max_delta_count += int(
            _num(_json_obj(row.get("cstcc_shadow_selected_candidate_violation_reason_distribution_json")).get("max_delta"))
        )
        _add_count(by_template, template, max_delta)
        _add_count(by_source, source, max_delta)
        _add_count(by_regime, regime, max_delta)
        _add_count(by_phase, phase, max_delta)
        for risk in risk_labels:
            _add_count(by_risk, risk, max_delta)
        for field, count in _json_obj(row.get("cstcc_shadow_hard_constraint_violation_field_distribution_json")).items():
            key = (str(field), template)
            field_by_template[key] = field_by_template.get(key, 0) + int(_num(count))
    tables = {
        "max_delta_by_template": by_template,
        "max_delta_by_source_prior": by_source,
        "max_delta_by_regime": by_regime,
        "max_delta_by_phase": by_phase,
        "max_delta_by_risk_flag": by_risk,
    }
    for table, values in tables.items():
        for key, count in sorted(values.items()):
            cross_rows.append({"table": table, "key_1": key, "key_2": "", "key_3": "", "count": count})
    for (field, template), count in sorted(field_by_template.items()):
        cross_rows.append(
            {
                "table": "max_delta_field_by_template",
                "key_1": field,
                "key_2": template,
                "key_3": "",
                "count": count,
            }
        )
    top_template_ratio = max(by_template.values(), default=0) / max(total, 1)
    top_field_count = 0
    by_field: dict[str, int] = {}
    for (field, _template), count in field_by_template.items():
        by_field[field] = by_field.get(field, 0) + count
        top_field_count = max(top_field_count, by_field[field])
    return (
        {
            "max_delta_count": total,
            "by_template": dict(sorted(by_template.items())),
            "by_source_prior": dict(sorted(by_source.items())),
            "by_regime": dict(sorted(by_regime.items())),
            "by_phase": dict(sorted(by_phase.items())),
            "by_risk_flag": dict(sorted(by_risk.items())),
            "by_field": dict(sorted(by_field.items())),
            "attribution_mode": (
                "candidate_level_attribution"
                if candidate_level_rows > 0
                else "selected_step_coincidence_attribution"
            ),
            "candidate_level_rows": candidate_level_rows,
            "selected_step_coincidence_rows": selected_step_rows,
            "selected_candidate_max_delta_count": selected_candidate_max_delta_count,
            "non_selected_candidate_max_delta_count": non_selected_candidate_max_delta_count,
            "top_template_share": top_template_ratio,
            "top_field_share": top_field_count / max(total, 1),
            "field_by_template": {
                f"{field}|{template}": count
                for (field, template), count in sorted(field_by_template.items())
            },
        },
        cross_rows,
    )


def _prior_template_analysis(rows: list[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_distribution = _distribution(row.get("cstcc_shadow_selected_source_prior") for row in rows)
    template_distribution = _distribution(row.get("cstcc_shadow_selected_template_name") for row in rows)
    cross_rows: list[dict[str, Any]] = []

    def cross(table: str, dim_a: str, dim_b: str) -> None:
        counts: dict[tuple[str, str], int] = {}
        for row in rows:
            key = (str(row.get(dim_a) or "unknown"), str(row.get(dim_b) or "unknown"))
            counts[key] = counts.get(key, 0) + 1
        for (a, b), count in sorted(counts.items()):
            cross_rows.append({"table": table, "key_1": a, "key_2": b, "key_3": "", "count": count})

    cross("selected_source_prior_by_regime", "cstcc_shadow_selected_source_prior", "cstcc_shadow_active_regime_after")
    cross("selected_template_by_regime", "cstcc_shadow_selected_template_name", "cstcc_shadow_active_regime_after")
    cross("selected_source_prior_by_runtime_source", "cstcc_shadow_selected_source_prior", "source")
    cross("selected_template_by_runtime_source", "cstcc_shadow_selected_template_name", "source")

    risk_template_counts: dict[tuple[str, str], int] = {}
    risk_source_counts: dict[tuple[str, str], int] = {}
    for row in rows:
        for risk in _risk_label_counts(row):
            template = str(row.get("cstcc_shadow_selected_template_name") or "unknown")
            source = str(row.get("cstcc_shadow_selected_source_prior") or "unknown")
            risk_template_counts[(risk, template)] = risk_template_counts.get((risk, template), 0) + 1
            risk_source_counts[(risk, source)] = risk_source_counts.get((risk, source), 0) + 1
    for (risk, template), count in sorted(risk_template_counts.items()):
        cross_rows.append({"table": "selected_template_by_risk_flag", "key_1": risk, "key_2": template, "key_3": "", "count": count})
    for (risk, source), count in sorted(risk_source_counts.items()):
        cross_rows.append({"table": "selected_source_prior_by_risk_flag", "key_1": risk, "key_2": source, "key_3": "", "count": count})

    ppo_previous = source_distribution.get("ppo_prior", 0) + source_distribution.get("previous_plan_prior", 0)
    conservative_rule = source_distribution.get("conservative_prior", 0) + source_distribution.get("rule_prior", 0)
    return (
        {
            "selected_source_prior_distribution": source_distribution,
            "selected_template_name_distribution": template_distribution,
            "ppo_previous_prior_share": ppo_previous / max(len(rows), 1),
            "conservative_rule_selected_count": conservative_rule,
        },
        cross_rows,
    )


def _regime_behavior(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    regime_rows: dict[str, list[Mapping[str, Any]]] = {}
    transition_count = 0
    previous_regime = None
    for row in rows:
        regime = str(row.get("cstcc_shadow_active_regime_after") or "unknown")
        regime_rows.setdefault(regime, []).append(row)
        if previous_regime is not None and regime != previous_regime:
            transition_count += 1
        previous_regime = regime
    confidence_by_regime: dict[str, dict[str, float]] = {}
    for regime, items in regime_rows.items():
        confidence = [_num(row.get("cstcc_shadow_effective_confidence")) for row in items]
        alpha = [_num(row.get("cstcc_shadow_llm_influence_alpha")) for row in items]
        confidence_by_regime[regime] = {
            "count": len(items),
            "effective_confidence_mean": mean(confidence) if confidence else 0.0,
            "llm_influence_alpha_mean": mean(alpha) if alpha else 0.0,
        }
    humidity_rows = regime_rows.get("HUMIDITY_EXCESS_DISEASE_RISK", [])
    aligned = sum(1 for row in humidity_rows if _risk_flags(row)["humidity_or_dew_risk"])
    return {
        "active_regime_distribution": {key: len(value) for key, value in sorted(regime_rows.items())},
        "regime_transition_count": transition_count,
        "confidence_by_regime": confidence_by_regime,
        "humidity_excess_disease_risk_steps": len(humidity_rows),
        "humidity_risk_alignment_steps": aligned,
        "humidity_risk_alignment_rate": aligned / max(len(humidity_rows), 1),
    }


def _risk_aligned_behavior(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    risk_summary: dict[str, dict[str, Any]] = {}
    all_flags = (
        "rh_ge_90",
        "dew_risk",
        "canopy_dew_margin_lt1",
        "canopy_dew_margin_lt0",
        "dry_risk",
        "dry_vent_risk",
        "high_vpd",
        "humidity_or_dew_risk",
        "any_risk",
    )
    for flag in all_flags:
        risk_rows = [row for row in rows if _risk_flags(row).get(flag, False)]
        template_dist = _distribution(row.get("cstcc_shadow_selected_template_name") for row in risk_rows)
        source_dist = _distribution(row.get("cstcc_shadow_selected_source_prior") for row in risk_rows)
        hold_all = template_dist.get("hold_all", 0)
        risk_summary[flag] = {
            "step_count": len(risk_rows),
            "selected_template_distribution": template_dist,
            "selected_source_prior_distribution": source_dist,
            "hold_all_share": hold_all / max(len(risk_rows), 1),
        }
    return risk_summary


def _action_diff_topk(rows: list[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        diff = {field: _num(_json_obj(row.get("cstcc_shadow_action_diff_json")).get(field)) for field in ACTION_FIELDS}
        runtime_action = _json_obj(row.get("cstcc_shadow_current_runtime_final_action_json"))
        shadow_action = _json_obj(row.get("cstcc_shadow_selected_first_action_json"))
        abs_by_field = {field: abs(value) for field, value in diff.items()}
        l1 = sum(abs_by_field.values())
        flags = _risk_flags(row)
        diagnostics.append(
            {
                "step": int(_num(row.get("step"))),
                "hour_of_day": _num(row.get("hour_of_day")),
                "phase": str(row.get("phase") or ""),
                "runtime_source": str(row.get("source") or ""),
                "active_regime": str(row.get("cstcc_shadow_active_regime_after") or ""),
                "selected_source_prior": str(row.get("cstcc_shadow_selected_source_prior") or ""),
                "selected_template": str(row.get("cstcc_shadow_selected_template_name") or ""),
                "hard_constraint_violation_count": int(_num(row.get("cstcc_shadow_hard_constraint_violation_count"))),
                "hard_constraint_violation_reasons_json": row.get("cstcc_shadow_hard_constraint_violation_reason_distribution_json") or "{}",
                "hard_constraint_violation_fields_json": row.get("cstcc_shadow_hard_constraint_violation_field_distribution_json") or "{}",
                "temp_air": _num(row.get("temp_air")),
                "rh_air": _num(row.get("rh_air")),
                "vpd_air": _num(row.get("vpd_air"), _num(row.get("vpd_kpa"))),
                "canopy_dew_margin": _num(row.get("canopy_dew_margin")),
                "dew_margin_air": _num(row.get("dew_margin_air")),
                "risk_flags_json": _compact_json({key: value for key, value in flags.items() if value}),
                "runtime_action_json": _compact_json(runtime_action),
                "shadow_action_json": _compact_json(shadow_action),
                "diff_json": _compact_json(diff),
                "total_l1_diff": l1,
                **{f"abs_diff_{field}": abs_by_field[field] for field in ACTION_FIELDS},
            }
        )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    def add_rows(kind: str, ordered: list[dict[str, Any]], limit: int) -> None:
        for item in ordered[:limit]:
            key = (kind, int(item["step"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append({"rank_group": kind, **item})

    add_rows("total_l1_top20", sorted(diagnostics, key=lambda item: item["total_l1_diff"], reverse=True), 20)
    for field in ("u_heating", "u_screen", "u_ventilation"):
        add_rows(
            f"{field}_top10",
            sorted(diagnostics, key=lambda item, field=field: item[f"abs_diff_{field}"], reverse=True),
            10,
        )
    max_l1 = max((item["total_l1_diff"] for item in diagnostics), default=0.0)
    return (
        {
            "max_total_l1_diff": max_l1,
            "mean_total_l1_diff": mean([item["total_l1_diff"] for item in diagnostics]) if diagnostics else 0.0,
            "topk_row_count": len(selected),
            "top_action_diff_explainable": max_l1 <= TOP_DIFF_L1_MAX,
        },
        selected,
    )


def _route_next_action(
    *,
    missing_issues: list[str],
    boundary: Mapping[str, Any],
    candidate: Mapping[str, Any],
    max_delta: Mapping[str, Any],
    prior_template: Mapping[str, Any],
    regime: Mapping[str, Any],
    risk: Mapping[str, Mapping[str, Any]],
    action_diff: Mapping[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if missing_issues:
        return "v811_trace_schema_repair_or_rerun_plan", list(missing_issues)
    forbidden = {
        "final_action_changed_steps": int(_num(boundary.get("final_action_changed_steps"))),
        "online_llm_called_steps": int(_num(boundary.get("online_llm_called_steps"))),
        "predictive_rollout_executed_steps": int(_num(boundary.get("predictive_rollout_executed_steps"))),
        "real_tomato_safety_projection_steps": int(_num(boundary.get("real_tomato_safety_projection_steps"))),
        "runtime_error_steps": int(_num(boundary.get("runtime_error_steps"))),
    }
    for name, value in forbidden.items():
        if value:
            reasons.append(f"forbidden_boundary:{name}={value}")
    if _num(boundary.get("audit_success_rate")) < 1.0:
        reasons.append("audit_success_rate_below_1")
    if _num(boundary.get("final_action_invariant_rate")) < 1.0:
        reasons.append("final_action_invariant_rate_below_1")
    if _num(boundary.get("audit_latency_ms_p95")) > LATENCY_P95_MAX_MS:
        reasons.append("audit_latency_p95_above_250ms")
    if _num(boundary.get("json_row_size_kb_max")) > JSON_ROW_MAX_KB:
        reasons.append("json_row_size_above_64kb")
    if reasons:
        return "v811_trace_schema_repair_or_rerun_plan", reasons

    template_reasons: list[str] = []
    if _num(candidate.get("fallback_rate")) > FALLBACK_RATE_MAX:
        template_reasons.append("fallback_rate_above_0_05")
    if _num(candidate.get("mean_infeasible_candidate_ratio")) > MEAN_INFEASIBLE_RATIO_MAX:
        template_reasons.append("mean_infeasible_ratio_above_0_10")
    if (
        int(_num(max_delta.get("max_delta_count"))) > 0
        and str(max_delta.get("attribution_mode") or "") == "selected_step_coincidence_attribution"
    ):
        template_reasons.append("max_delta_attribution_selected_step_coincidence_only")
    if _num(max_delta.get("top_template_share")) > MAX_DELTA_CONCENTRATION_MAX:
        template_reasons.append("max_delta_concentrated_in_single_template")
    if _num(max_delta.get("top_field_share")) > MAX_DELTA_CONCENTRATION_MAX:
        template_reasons.append("max_delta_concentrated_in_single_field")

    scorer_reasons: list[str] = []
    any_risk = risk.get("any_risk", {})
    if int(_num(any_risk.get("step_count"))) > 0 and _num(any_risk.get("hold_all_share")) > RISK_HOLD_ALL_MAX:
        scorer_reasons.append("risk_steps_hold_all_share_above_0_70")
    humidity_risk = risk.get("humidity_or_dew_risk", {})
    if int(_num(humidity_risk.get("step_count"))) > 0 and _num(humidity_risk.get("hold_all_share")) > RISK_HOLD_ALL_MAX:
        scorer_reasons.append("humidity_or_dew_risk_hold_all_share_above_0_70")
    if (
        _num(prior_template.get("ppo_previous_prior_share")) >= PRIOR_DOMINANCE_MAX
        and int(_num(prior_template.get("conservative_rule_selected_count"))) == 0
    ):
        scorer_reasons.append("ppo_previous_prior_dominance_without_conservative_or_rule_selection")
    if int(_num(regime.get("humidity_excess_disease_risk_steps"))) > 0 and _num(regime.get("humidity_risk_alignment_rate")) < REGIME_ALIGNMENT_MIN:
        scorer_reasons.append("humidity_regime_low_alignment_with_humidity_or_dew_risk")
    if not bool(action_diff.get("top_action_diff_explainable", False)):
        scorer_reasons.append("top_action_diff_exceeds_explainability_threshold")

    if template_reasons:
        return "v816_template_constraint_calibration_plan", template_reasons + scorer_reasons
    if scorer_reasons:
        return "v816_level0_scorer_calibration_plan", scorer_reasons
    return "v82_simulator_shadow_rollout_plan", []


def build_shadow_analytics(
    *,
    trace_csv: str | Path = DEFAULT_CSV,
    summary_json: str | Path = DEFAULT_SUMMARY_JSON,
    jsonl_root: str | Path = DEFAULT_JSONL_ROOT,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_csv(trace_csv)
    v811_summary = _load_json(summary_json)
    jsonl_rows = _load_jsonl_rows(jsonl_root)
    missing_issues = _required_field_issues(rows)

    boundary = _boundary_summary(rows, v811_summary, jsonl_rows)
    candidate = _candidate_health(rows)
    max_delta, violation_cross_rows = _max_delta_attribution(rows)
    prior_template, prior_cross_rows = _prior_template_analysis(rows)
    regime = _regime_behavior(rows)
    risk = _risk_aligned_behavior(rows)
    action_diff, topk_rows = _action_diff_topk(rows)
    next_action, readiness_reasons = _route_next_action(
        missing_issues=missing_issues,
        boundary=boundary,
        candidate=candidate,
        max_delta=max_delta,
        prior_template=prior_template,
        regime=regime,
        risk=risk,
        action_diff=action_diff,
    )
    cross_rows = violation_cross_rows + prior_cross_rows
    return (
        {
            "schema_version": "cstcc_v815_shadow_analytics_v1",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "evidence_level": "opt_in_shadow_trace_analytics",
            "inputs": {
                "trace_csv": str(trace_csv),
                "summary_json": str(summary_json),
                "jsonl_root": str(jsonl_root),
            },
            "boundaries": {
                "online_llm_called": False,
                "new_rollout_run": False,
                "simulator_rollout_executed": False,
                "runtime_changed": False,
                "final_action_changed": False,
                "performance_claim_allowed": False,
                "promotion_evidence": False,
            },
            "boundary_runtime_safety": boundary,
            "candidate_health": candidate,
            "max_delta_attribution": max_delta,
            "prior_template_dominance": prior_template,
            "regime_behavior": regime,
            "risk_aligned_behavior": risk,
            "action_diff_topk_summary": action_diff,
            "missing_or_invalid_inputs": missing_issues,
            "readiness_reasons": readiness_reasons,
            "next_action": next_action,
        },
        topk_rows,
        cross_rows,
    )


def build_markdown(report: Mapping[str, Any]) -> str:
    boundary = report.get("boundary_runtime_safety", {})
    candidate = report.get("candidate_health", {})
    max_delta = report.get("max_delta_attribution", {})
    prior = report.get("prior_template_dominance", {})
    regime = report.get("regime_behavior", {})
    risk = report.get("risk_aligned_behavior", {})
    action_diff = report.get("action_diff_topk_summary", {})
    lines = [
        "# C-STCC v81.5 Shadow Analytics",
        "",
        "Offline analysis of the v81.1 full-episode shadow trace. No online LLM, no simulator rollout, no final-action change, no performance claim.",
        "",
        "## Boundary & Runtime Safety",
        "",
        f"- CSV/JSONL rows: {boundary.get('csv_row_count', 0)} / {boundary.get('jsonl_row_count', 0)}",
        f"- Audit success rate: {float(boundary.get('audit_success_rate', 0.0)):.3f}",
        f"- Final-action invariant rate: {float(boundary.get('final_action_invariant_rate', 0.0)):.3f}",
        f"- Forbidden boundary counts: online={boundary.get('online_llm_called_steps', 0)}, rollout={boundary.get('predictive_rollout_executed_steps', 0)}, real_projection={boundary.get('real_tomato_safety_projection_steps', 0)}, final_action_changed={boundary.get('final_action_changed_steps', 0)}",
        f"- Latency p95 ms / JSON row max KB: {float(boundary.get('audit_latency_ms_p95', 0.0)):.3f} / {float(boundary.get('json_row_size_kb_max', 0.0)):.3f}",
        "",
        "## Candidate Health",
        "",
        f"- Fallback rate: {float(candidate.get('fallback_rate', 0.0)):.3f}",
        f"- Mean infeasible candidate ratio: {float(candidate.get('mean_infeasible_candidate_ratio', 0.0)):.3f}",
        f"- Hard constraint reasons: `{_compact_json(candidate.get('hard_constraint_violation_reason_distribution', {}))}`",
        f"- Hard constraint fields: `{_compact_json(candidate.get('hard_constraint_violation_field_distribution', {}))}`",
        "",
        "## Selection Structure",
        "",
        f"- Selected source prior: `{_compact_json(prior.get('selected_source_prior_distribution', {}))}`",
        f"- Selected template: `{_compact_json(prior.get('selected_template_name_distribution', {}))}`",
        f"- Max-delta attribution mode: {max_delta.get('attribution_mode', 'unknown')}",
        f"- Selected/non-selected candidate max-delta: {max_delta.get('selected_candidate_max_delta_count', 0)} / {max_delta.get('non_selected_candidate_max_delta_count', 0)}",
        f"- Max-delta by template: `{_compact_json(max_delta.get('by_template', {}))}`",
        f"- Max-delta by field: `{_compact_json(max_delta.get('by_field', {}))}`",
        "",
        "## Regime & Risk",
        "",
        f"- Active regimes: `{_compact_json(regime.get('active_regime_distribution', {}))}`",
        f"- Regime transitions: {regime.get('regime_transition_count', 0)}",
        f"- Humidity regime alignment rate: {float(regime.get('humidity_risk_alignment_rate', 0.0)):.3f}",
        f"- Any-risk hold_all share: {float(risk.get('any_risk', {}).get('hold_all_share', 0.0)):.3f}",
        f"- Humidity/dew-risk hold_all share: {float(risk.get('humidity_or_dew_risk', {}).get('hold_all_share', 0.0)):.3f}",
        "",
        "## Action Diff",
        "",
        f"- Max total L1 diff: {float(action_diff.get('max_total_l1_diff', 0.0)):.3f}",
        f"- Top-K row count: {action_diff.get('topk_row_count', 0)}",
        "",
        "## Decision",
        "",
        f"- Next action: {report.get('next_action')}",
        f"- Readiness reasons: {', '.join(report.get('readiness_reasons', [])) if report.get('readiness_reasons') else 'none'}",
        "",
        "v81.5 explains shadow structure only. v82, if selected later, must remain shadow rollout unless separately authorized.",
    ]
    return "\n".join(lines) + "\n"


def _write_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> Path:
    output = _resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _compact_json(value) if isinstance(value, (dict, list)) else value
                    for key, value in dict(row).items()
                }
            )
    return output


def output_paths(prefix: str | Path) -> dict[str, Path]:
    base = _resolve(prefix)
    name = base.name
    if "shadow_analytics" in name:
        topk_name = name.replace("shadow_analytics", "shadow_action_diff_topk")
        cross_name = name.replace("shadow_analytics", "shadow_violation_cross_tabs")
    else:
        topk_name = f"{name}_action_diff_topk"
        cross_name = f"{name}_violation_cross_tabs"
    return {
        "json": base.with_suffix(".json"),
        "md": base.with_suffix(".md"),
        "topk_csv": base.with_name(topk_name).with_suffix(".csv"),
        "cross_tabs_csv": base.with_name(cross_name).with_suffix(".csv"),
    }


def write_outputs(
    *,
    report: Mapping[str, Any],
    topk_rows: list[Mapping[str, Any]],
    cross_rows: list[Mapping[str, Any]],
    output_prefix: str | Path = DEFAULT_OUTPUT_PREFIX,
) -> dict[str, str]:
    paths = output_paths(output_prefix)
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(_compact_json(report) + "\n", encoding="utf-8")
    paths["md"].write_text(build_markdown(report), encoding="utf-8")
    _write_csv(paths["topk_csv"], topk_rows)
    _write_csv(paths["cross_tabs_csv"], cross_rows)
    return {key: str(path) for key, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v81.5 C-STCC shadow analytics report.")
    parser.add_argument("--trace-csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--summary-json", type=str, default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--jsonl-root", type=str, default=str(DEFAULT_JSONL_ROOT))
    parser.add_argument("--output-prefix", type=str, default=str(DEFAULT_OUTPUT_PREFIX))
    args = parser.parse_args()

    report, topk_rows, cross_rows = build_shadow_analytics(
        trace_csv=args.trace_csv,
        summary_json=args.summary_json,
        jsonl_root=args.jsonl_root,
    )
    outputs = write_outputs(
        report=report,
        topk_rows=topk_rows,
        cross_rows=cross_rows,
        output_prefix=args.output_prefix,
    )
    print(
        json.dumps(
            {
                "outputs": outputs,
                "next_action": report.get("next_action"),
                "readiness_reasons": report.get("readiness_reasons", []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
