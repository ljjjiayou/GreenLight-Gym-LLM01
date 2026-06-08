"""v81.6 C-STCC template and constraint calibration report.

This offline report compares the v81.5 pre-calibration diagnosis with a
candidate-level post-calibration trace. It does not run the simulator, call
online LLM, change final actions, or make controller-quality claims.
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

DEFAULT_PRE_ANALYTICS = Path("gl_gym/result/diagnostics/cstcc_v815_shadow_analytics_20260605_y2020_d240_s42_n720.json")
DEFAULT_POST_ANALYTICS = Path(
    "gl_gym/result/diagnostics/cstcc_v816_post_calibration_shadow_analytics_20260605_y2020_d240_s42_n720.json"
)
DEFAULT_POST_JSONL_ROOT = Path("logs/cstcc_shadow/v816_calibrated_full_episode_20260605_y2020_d240_s42_n720")
DEFAULT_OUTPUT_PREFIX = Path(
    "gl_gym/result/diagnostics/cstcc_v816_template_constraint_calibration_20260605_y2020_d240_s42_n720"
)

MAX_DELTA_CONCENTRATION_MAX = 0.60


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _load_json(path: str | Path) -> dict[str, Any]:
    input_path = _resolve(path)
    if not input_path.exists():
        return {}
    return json.loads(input_path.read_text(encoding="utf-8"))


def _load_jsonl_rows(root: str | Path) -> list[dict[str, Any]]:
    input_root = _resolve(root)
    if not input_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(input_root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


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


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _add_count(counts: dict[str, int], key: Any, count: int = 1) -> None:
    text = str(key or "unknown")
    counts[text] = counts.get(text, 0) + int(count or 0)


def _normalized_action(value: Any) -> dict[str, float]:
    action = _mapping(value)
    return {field: _num(action.get(field)) for field in ACTION_FIELDS}


def _action_equal(lhs: Any, rhs: Any) -> bool:
    left = _normalized_action(lhs)
    right = _normalized_action(rhs)
    return all(abs(left[field] - right[field]) <= 1e-9 for field in ACTION_FIELDS)


def _risk_present(row: Mapping[str, Any]) -> bool:
    flags = _mapping(row.get("risk_flags"))
    if flags:
        return any(_bool(value) for key, value in flags.items() if str(key) != "any_risk") or _bool(flags.get("any_risk"))
    return False


def summarize_hold_references(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    hold_candidates = []
    hold_zero_delta = 0
    hold_max_delta = 0.0
    reference_matches_runtime = 0
    for row in rows:
        if _action_equal(row.get("hold_all_reference"), row.get("current_runtime_final_action")):
            reference_matches_runtime += 1
        for candidate in row.get("candidate_violation_provenance") or []:
            if not isinstance(candidate, Mapping) or candidate.get("template_name") != "hold_all":
                continue
            hold_candidates.append(candidate)
            first_delta = _mapping(candidate.get("first_action_delta_from_reference"))
            max_delta = max((abs(_num(first_delta.get(field))) for field in ACTION_FIELDS), default=0.0)
            hold_max_delta = max(hold_max_delta, max_delta)
            if max_delta <= 1e-9:
                hold_zero_delta += 1
    return {
        "row_count": len(rows),
        "hold_all_reference_kind_distribution": _distribution(row.get("hold_all_reference_kind") for row in rows),
        "constraint_reference_kind_distribution": _distribution(row.get("constraint_reference_kind") for row in rows),
        "hold_all_reference_matches_runtime_rate": reference_matches_runtime / max(len(rows), 1),
        "hold_all_candidate_count": len(hold_candidates),
        "hold_all_zero_first_delta_rate": hold_zero_delta / max(len(hold_candidates), 1),
        "hold_all_max_abs_first_delta_from_reference": hold_max_delta,
        "answer": "hold_all holds current_runtime_final_action when reference kind is current_runtime_final_action and first delta is zero.",
    }


def summarize_candidate_attribution(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    mode_distribution = _distribution(row.get("candidate_attribution_mode") for row in rows)
    by_template: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_field: dict[str, int] = {}
    selected_max_delta = 0
    non_selected_max_delta = 0
    selected_violation_count = 0
    candidate_rows = 0
    for row in rows:
        selected_id = str(row.get("shadow_selected_sequence_id") or "")
        selected_violation_count += int(_num(row.get("selected_candidate_violation_count")))
        candidates = row.get("candidate_violation_provenance") or []
        if candidates:
            candidate_rows += 1
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            reasons = _mapping(candidate.get("violation_reason_distribution"))
            max_delta = int(_num(reasons.get("max_delta")))
            if max_delta <= 0:
                continue
            if str(candidate.get("candidate_id") or "") == selected_id:
                selected_max_delta += max_delta
            else:
                non_selected_max_delta += max_delta
            _add_count(by_template, candidate.get("template_name"), max_delta)
            _add_count(by_source, candidate.get("source_prior"), max_delta)
            for field, count in _mapping(candidate.get("violation_field_distribution")).items():
                _add_count(by_field, field, int(_num(count)))
    total = selected_max_delta + non_selected_max_delta
    return {
        "candidate_attribution_mode_distribution": mode_distribution,
        "candidate_level_row_count": candidate_rows,
        "max_delta_count": total,
        "selected_candidate_max_delta_count": selected_max_delta,
        "non_selected_candidate_max_delta_count": non_selected_max_delta,
        "selected_candidate_violation_count": selected_violation_count,
        "max_delta_by_template": dict(sorted(by_template.items())),
        "max_delta_by_source_prior": dict(sorted(by_source.items())),
        "max_delta_by_field": dict(sorted(by_field.items())),
        "top_template_share": max(by_template.values(), default=0) / max(total, 1),
        "answer": "New traces attribute max_delta to individual candidates; old selected-step coincidence is no longer used when candidate provenance exists.",
    }


def summarize_previous_shift_projection(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_count = 0
    selected_count = 0
    max_delta_count = 0
    projection_applied_count = 0
    first_delta_after_projection_max = 0.0
    projection_abs_sums = {field: 0.0 for field in ACTION_FIELDS}
    internal_max_after_projection = {field: 0.0 for field in ACTION_FIELDS}
    violation_fields: dict[str, int] = {}
    projection_metadata_rows = 0
    for row in rows:
        selected_id = str(row.get("shadow_selected_sequence_id") or "")
        candidates = row.get("candidate_violation_provenance") or []
        for candidate in candidates:
            if not isinstance(candidate, Mapping) or candidate.get("template_name") != "previous_shift_all":
                continue
            candidate_count += 1
            if str(candidate.get("candidate_id") or "") == selected_id:
                selected_count += 1
            reasons = _mapping(candidate.get("violation_reason_distribution"))
            max_delta_count += int(_num(reasons.get("max_delta")))
            for field, count in _mapping(candidate.get("violation_field_distribution")).items():
                _add_count(violation_fields, field, int(_num(count)))
            metadata = _mapping(candidate.get("template_metadata"))
            if metadata:
                projection_metadata_rows += 1
            if metadata.get("previous_shift_projection_applied"):
                projection_applied_count += 1
            first_delta_after = _mapping(metadata.get("previous_shift_first_delta_after_projection_by_field"))
            for field in ACTION_FIELDS:
                first_delta_after_projection_max = max(
                    first_delta_after_projection_max,
                    abs(_num(first_delta_after.get(field))),
                )
            projection_delta = _mapping(metadata.get("previous_shift_projection_delta_by_field"))
            for field in ACTION_FIELDS:
                projection_abs_sums[field] += abs(_num(projection_delta.get(field)))
            internal_delta = _mapping(metadata.get("previous_shift_internal_max_delta_after_projection_by_field"))
            for field in ACTION_FIELDS:
                internal_max_after_projection[field] = max(
                    internal_max_after_projection[field],
                    abs(_num(internal_delta.get(field))),
                )
    return {
        "previous_shift_all_candidate_count": candidate_count,
        "previous_shift_all_max_delta_count": max_delta_count,
        "previous_shift_all_selected_count": selected_count,
        "previous_shift_projection_metadata_rows": projection_metadata_rows,
        "previous_shift_projection_applied_count": projection_applied_count,
        "previous_shift_projection_applied_rate": projection_applied_count / max(candidate_count, 1),
        "previous_shift_first_delta_after_projection_max": first_delta_after_projection_max,
        "previous_shift_projection_mean_abs_delta_by_field": {
            field: projection_abs_sums[field] / max(projection_applied_count, 1)
            for field in ACTION_FIELDS
        },
        "previous_shift_internal_max_delta_after_projection_by_field": internal_max_after_projection,
        "previous_shift_all_violation_field_distribution": dict(sorted(violation_fields.items())),
        "answer": "previous_shift_all is admissible for scorer calibration only after its candidate-level max_delta count reaches zero.",
    }


def summarize_competitiveness(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, dict[str, Any]] = {}
    risk_rows = [row for row in rows if _risk_present(row)]
    for source in ("rule_prior", "conservative_prior"):
        candidate_steps = 0
        feasible_steps = 0
        selected_steps = 0
        lower_score_steps = 0
        no_candidate_steps = 0
        no_feasible_steps = 0
        risk_candidate_steps = 0
        risk_feasible_steps = 0
        risk_selected_steps = 0
        margins: list[float] = []
        for row in rows:
            info = _mapping(_mapping(row.get("rule_conservative_competitiveness")).get(source))
            status = str(info.get("status") or "missing")
            if int(_num(info.get("candidate_count"))) > 0:
                candidate_steps += 1
            if int(_num(info.get("feasible_candidate_count"))) > 0:
                feasible_steps += 1
            if status == "selected":
                selected_steps += 1
            if status == "lower_score":
                lower_score_steps += 1
            if status == "no_candidate":
                no_candidate_steps += 1
            if status == "no_feasible_candidate":
                no_feasible_steps += 1
            if info.get("selected_score_margin") is not None:
                margins.append(_num(info.get("selected_score_margin")))
            if _risk_present(row):
                if int(_num(info.get("candidate_count"))) > 0:
                    risk_candidate_steps += 1
                if int(_num(info.get("feasible_candidate_count"))) > 0:
                    risk_feasible_steps += 1
                if status == "selected":
                    risk_selected_steps += 1
        result[source] = {
            "candidate_steps": candidate_steps,
            "feasible_steps": feasible_steps,
            "selected_steps": selected_steps,
            "lower_score_steps": lower_score_steps,
            "no_candidate_steps": no_candidate_steps,
            "no_feasible_steps": no_feasible_steps,
            "risk_candidate_steps": risk_candidate_steps,
            "risk_feasible_steps": risk_feasible_steps,
            "risk_selected_steps": risk_selected_steps,
            "mean_selected_score_margin_when_not_selected": mean(margins) if margins else None,
        }
    return {
        "risk_row_count": len(risk_rows),
        "sources": result,
        "answer": "Competitiveness separates generation, feasibility, and score-margin loss for rule/conservative priors.",
    }


def _boundary_reasons(analytics: Mapping[str, Any]) -> list[str]:
    boundary = _mapping(analytics.get("boundary_runtime_safety"))
    reasons: list[str] = []
    forbidden = (
        "final_action_changed_steps",
        "online_llm_called_steps",
        "predictive_rollout_executed_steps",
        "real_tomato_safety_projection_steps",
        "runtime_error_steps",
    )
    for key in forbidden:
        value = int(_num(boundary.get(key)))
        if value:
            reasons.append(f"forbidden_boundary:{key}={value}")
    if _num(boundary.get("audit_success_rate")) < 0.99:
        reasons.append("audit_success_rate_below_0_99")
    if _num(boundary.get("final_action_invariant_rate")) < 1.0:
        reasons.append("final_action_invariant_rate_below_1")
    return reasons


def route_next_action(
    *,
    post_analytics: Mapping[str, Any],
    post_rows: list[Mapping[str, Any]],
    candidate_attribution: Mapping[str, Any],
    previous_shift: Mapping[str, Any],
    competitiveness: Mapping[str, Any],
) -> tuple[str, list[str]]:
    if not post_analytics or not post_rows:
        return "v816_calibrated_trace_acquisition_required", ["post_calibration_trace_or_analytics_missing"]
    boundary_reasons = _boundary_reasons(post_analytics)
    if boundary_reasons:
        return "stop_and_repair_v816_shadow_boundary", boundary_reasons
    mode_distribution = _mapping(candidate_attribution.get("candidate_attribution_mode_distribution"))
    if not mode_distribution.get("candidate_level_attribution"):
        return "repair_candidate_level_attribution", ["candidate_level_attribution_missing"]
    max_delta_count = int(_num(candidate_attribution.get("max_delta_count")))
    selected_max_delta = int(_num(candidate_attribution.get("selected_candidate_max_delta_count")))
    if selected_max_delta > 0:
        return "v816_template_constraint_calibration_iteration", ["selected_candidate_has_max_delta_violation"]
    previous_shift_max_delta = int(_num(previous_shift.get("previous_shift_all_max_delta_count")))
    if previous_shift_max_delta > 0:
        return "v816_previous_shift_projection_iteration", ["previous_shift_all_candidate_has_max_delta_violation"]
    if max_delta_count > 0 and _num(candidate_attribution.get("top_template_share")) > MAX_DELTA_CONCENTRATION_MAX:
        return "v816_template_constraint_calibration_iteration", ["non_selected_max_delta_concentrated_in_template"]
    sources = _mapping(competitiveness.get("sources"))
    risk_rows = int(_num(competitiveness.get("risk_row_count")))
    rule = _mapping(sources.get("rule_prior"))
    conservative = _mapping(sources.get("conservative_prior"))
    risk_feasible = int(_num(rule.get("risk_feasible_steps"))) + int(_num(conservative.get("risk_feasible_steps")))
    risk_selected = int(_num(rule.get("risk_selected_steps"))) + int(_num(conservative.get("risk_selected_steps")))
    if risk_rows > 0 and risk_feasible == 0:
        return "v816_template_constraint_calibration_iteration", ["rule_conservative_no_feasible_candidate_in_risk_window"]
    if risk_rows > 0 and risk_feasible > 0 and risk_selected == 0:
        return "v817_level0_scorer_calibration_plan", ["rule_conservative_feasible_but_never_selected_in_risk_window"]
    if post_analytics.get("next_action") == "v82_simulator_shadow_rollout_plan":
        return "v82_simulator_shadow_rollout_plan", []
    return str(post_analytics.get("next_action") or "v816_template_constraint_calibration_iteration"), list(
        post_analytics.get("readiness_reasons") or []
    )


def _comparison_summary(pre: Mapping[str, Any], post: Mapping[str, Any]) -> dict[str, Any]:
    pre_max = _mapping(pre.get("max_delta_attribution"))
    post_max = _mapping(post.get("max_delta_attribution"))
    pre_risk = _mapping(_mapping(pre.get("risk_aligned_behavior")).get("any_risk"))
    post_risk = _mapping(_mapping(post.get("risk_aligned_behavior")).get("any_risk"))
    return {
        "pre_next_action": pre.get("next_action"),
        "post_next_action": post.get("next_action"),
        "pre_max_delta_attribution_mode": pre_max.get("attribution_mode", "selected_step_coincidence_attribution"),
        "post_max_delta_attribution_mode": post_max.get("attribution_mode"),
        "pre_max_delta_count": int(_num(pre_max.get("max_delta_count"))),
        "post_max_delta_count": int(_num(post_max.get("max_delta_count"))),
        "pre_any_risk_hold_all_share": _num(pre_risk.get("hold_all_share")),
        "post_any_risk_hold_all_share": _num(post_risk.get("hold_all_share")),
    }


def build_calibration_report(
    *,
    pre_analytics: Mapping[str, Any] | None = None,
    post_analytics: Mapping[str, Any] | None = None,
    post_jsonl_rows: list[Mapping[str, Any]] | None = None,
    pre_analytics_path: str | Path = DEFAULT_PRE_ANALYTICS,
    post_analytics_path: str | Path = DEFAULT_POST_ANALYTICS,
    post_jsonl_root: str | Path = DEFAULT_POST_JSONL_ROOT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pre = dict(pre_analytics or _load_json(pre_analytics_path))
    post = dict(post_analytics or _load_json(post_analytics_path))
    rows = [dict(row) for row in (post_jsonl_rows if post_jsonl_rows is not None else _load_jsonl_rows(post_jsonl_root))]

    hold_reference = summarize_hold_references(rows)
    candidate_attribution = summarize_candidate_attribution(rows)
    previous_shift = summarize_previous_shift_projection(rows)
    competitiveness = summarize_competitiveness(rows)
    next_action, reasons = route_next_action(
        post_analytics=post,
        post_rows=rows,
        candidate_attribution=candidate_attribution,
        previous_shift=previous_shift,
        competitiveness=competitiveness,
    )
    report = {
        "schema_version": "cstcc_v816_template_constraint_calibration_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "evidence_level": "offline_shadow_trace_calibration",
        "inputs": {
            "pre_analytics": str(pre_analytics_path),
            "post_analytics": str(post_analytics_path),
            "post_jsonl_root": str(post_jsonl_root),
        },
        "boundaries": {
            "online_llm_called": False,
            "new_rollout_run": False,
            "simulator_rollout_executed": False,
            "runtime_changed": False,
            "final_action_changed": False,
            "performance_claim_allowed": False,
            "promotion_evidence": False,
            "real_tomato_safety_projection": False,
        },
        "comparison_summary": _comparison_summary(pre, post),
        "hold_all_reference_answer": hold_reference,
        "candidate_level_max_delta_attribution": candidate_attribution,
        "previous_shift_projection_calibration": previous_shift,
        "rule_conservative_competitiveness": competitiveness,
        "readiness_reasons": reasons,
        "next_action": next_action,
    }
    csv_rows = build_csv_rows(report)
    return report, csv_rows


def build_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparison = _mapping(report.get("comparison_summary"))
    for key, value in comparison.items():
        rows.append({"table": "comparison", "key_1": key, "key_2": "", "key_3": "", "count": "", "value": value})
    attribution = _mapping(report.get("candidate_level_max_delta_attribution"))
    for table_name, field_name in (
        ("max_delta_by_template", "max_delta_by_template"),
        ("max_delta_by_source_prior", "max_delta_by_source_prior"),
        ("max_delta_by_field", "max_delta_by_field"),
    ):
        for key, count in _mapping(attribution.get(field_name)).items():
            rows.append({"table": table_name, "key_1": key, "key_2": "", "key_3": "", "count": count, "value": ""})
    competitiveness = _mapping(_mapping(report.get("rule_conservative_competitiveness")).get("sources"))
    for source, metrics in competitiveness.items():
        for key, value in _mapping(metrics).items():
            rows.append({"table": "competitiveness", "key_1": source, "key_2": key, "key_3": "", "count": "", "value": value})
    previous_shift = _mapping(report.get("previous_shift_projection_calibration"))
    for key, value in previous_shift.items():
        if isinstance(value, Mapping):
            for subkey, subvalue in value.items():
                rows.append({"table": "previous_shift_projection", "key_1": key, "key_2": subkey, "key_3": "", "count": "", "value": subvalue})
        else:
            rows.append({"table": "previous_shift_projection", "key_1": key, "key_2": "", "key_3": "", "count": "", "value": value})
    return rows


def build_markdown(report: Mapping[str, Any]) -> str:
    comparison = _mapping(report.get("comparison_summary"))
    hold = _mapping(report.get("hold_all_reference_answer"))
    attribution = _mapping(report.get("candidate_level_max_delta_attribution"))
    previous_shift = _mapping(report.get("previous_shift_projection_calibration"))
    competitiveness = _mapping(report.get("rule_conservative_competitiveness"))
    lines = [
        "# C-STCC v81.6 Template & Constraint Calibration",
        "",
        "Offline calibration report. No online LLM, no simulator rollout, no final-action change, and no reward/control-effect claim.",
        "",
        "## What Changed",
        "",
        f"- Pre/post attribution mode: {comparison.get('pre_max_delta_attribution_mode')} -> {comparison.get('post_max_delta_attribution_mode')}",
        f"- Pre/post max_delta count: {comparison.get('pre_max_delta_count')} -> {comparison.get('post_max_delta_count')}",
        f"- Pre/post any-risk hold_all share: {float(comparison.get('pre_any_risk_hold_all_share', 0.0)):.3f} -> {float(comparison.get('post_any_risk_hold_all_share', 0.0)):.3f}",
        "",
        "## Three Answers",
        "",
        f"- hold_all reference: {hold.get('answer')}",
        f"- hold_all matches runtime rate: {float(hold.get('hold_all_reference_matches_runtime_rate', 0.0)):.3f}",
        f"- candidate-level max_delta selected/non-selected: {attribution.get('selected_candidate_max_delta_count', 0)} / {attribution.get('non_selected_candidate_max_delta_count', 0)}",
        f"- max_delta by template: `{_compact_json(attribution.get('max_delta_by_template', {}))}`",
        f"- max_delta by field: `{_compact_json(attribution.get('max_delta_by_field', {}))}`",
        f"- previous_shift_all max_delta: {previous_shift.get('previous_shift_all_max_delta_count', 0)}",
        f"- previous_shift projection applied: {previous_shift.get('previous_shift_projection_applied_count', 0)} / {previous_shift.get('previous_shift_all_candidate_count', 0)}",
        f"- previous_shift first-delta max after projection: {float(previous_shift.get('previous_shift_first_delta_after_projection_max', 0.0)):.3f}",
        f"- rule/conservative competitiveness: `{_compact_json(competitiveness.get('sources', {}))}`",
        "",
        "## Decision",
        "",
        f"- Next action: {report.get('next_action')}",
        f"- Readiness reasons: {', '.join(report.get('readiness_reasons', [])) if report.get('readiness_reasons') else 'none'}",
        "",
        "v81.6 remains shadow-only calibration. Any v82 work must be simulator shadow rollout, not final-action promotion.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    report: Mapping[str, Any],
    csv_rows: list[Mapping[str, Any]],
    output_prefix: str | Path = DEFAULT_OUTPUT_PREFIX,
) -> dict[str, str]:
    base = _resolve(output_prefix)
    paths = {
        "json": base.with_suffix(".json"),
        "md": base.with_suffix(".md"),
        "csv": base.with_suffix(".csv"),
    }
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(_compact_json(report) + "\n", encoding="utf-8")
    paths["md"].write_text(build_markdown(report), encoding="utf-8")
    fieldnames = ["table", "key_1", "key_2", "key_3", "count", "value"]
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return {key: str(path) for key, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v81.6 C-STCC calibration report.")
    parser.add_argument("--pre-analytics", type=str, default=str(DEFAULT_PRE_ANALYTICS))
    parser.add_argument("--post-analytics", type=str, default=str(DEFAULT_POST_ANALYTICS))
    parser.add_argument("--post-jsonl-root", type=str, default=str(DEFAULT_POST_JSONL_ROOT))
    parser.add_argument("--output-prefix", type=str, default=str(DEFAULT_OUTPUT_PREFIX))
    args = parser.parse_args()

    report, csv_rows = build_calibration_report(
        pre_analytics_path=args.pre_analytics,
        post_analytics_path=args.post_analytics,
        post_jsonl_root=args.post_jsonl_root,
    )
    outputs = write_outputs(report=report, csv_rows=csv_rows, output_prefix=args.output_prefix)
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
