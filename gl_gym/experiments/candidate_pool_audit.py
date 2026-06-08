"""Audit whether RSPC/fallback candidate pools contain safe dry-relief actions.

The audit is read-only and works from benchmark traces.  It is intended to
separate candidate-pool gaps from scorer/ranking gaps and post-guardrail
rewrite issues in selected dry-pressure/dew-conflict windows.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ACTION_MAP = {
    "heat": "u_heating",
    "co2": "u_co2",
    "screen": "u_screen",
    "vent": "u_ventilation",
    "lamp": "u_lighting",
    "shade": "u_shading",
}
STATE_FIELDS = ("temp_air", "rh_air", "vpd_air", "dew_margin_air", "canopy_dew_margin", "glob_rad", "wind_speed")
SCORE_FIELDS = (
    "temp_penalty",
    "rh_penalty",
    "dry_penalty",
    "vpd_high_penalty",
    "vpd_low_penalty",
    "dew_penalty",
    "canopy_dew_penalty",
    "energy_penalty",
    "conflict_penalty",
    "total_score",
)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_jsonish(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return default
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            continue
    return default


def _read_trace(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_trace(root: str | Path, scenario_id: str) -> Path:
    path = Path(root)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if path.is_file():
        return path
    matches = sorted(path.rglob(f"{scenario_id}*.csv"))
    if not matches:
        raise FileNotFoundError(f"trace not found for {scenario_id} under {path}")
    return matches[0]


def _rows_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    return {int(round(_num(row.get("step", row.get("timestep", 0))))): row for row in rows}


def _parse_candidates(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _parse_jsonish(row.get("rspc_action_candidates_json"), default=[])
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _candidate_name(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("name", "") or "").strip()


def _candidate_action(candidate: Mapping[str, Any], *, post_shape: bool = False) -> dict[str, float]:
    key = "post_shape_action" if post_shape else "action"
    raw = candidate.get(key, {})
    if not isinstance(raw, Mapping):
        return {}
    return {name: _num(raw.get(name)) for name in ACTION_MAP}


def _score_terms(candidate: Mapping[str, Any], *, post_shape: bool = False) -> dict[str, float]:
    key = "post_shape_score_terms" if post_shape else "score_terms"
    raw = candidate.get(key, {})
    if not isinstance(raw, Mapping):
        return {}
    terms = {field: _num(raw.get(field)) for field in SCORE_FIELDS}
    terms["score"] = _num(raw.get("score", candidate.get("score")))
    terms["temp_next"] = _num(raw.get("temp_next"))
    terms["rh_next"] = _num(raw.get("rh_next"))
    terms["vpd_next"] = _num(raw.get("vpd_next"))
    terms["predicted_dew_margin_air_next"] = _num(raw.get("predicted_dew_margin_air_next"), 99.0)
    terms["predicted_canopy_dew_margin_next"] = _num(raw.get("predicted_canopy_dew_margin_next"), 99.0)
    terms["predicted_dew_lt0"] = _num(raw.get("predicted_dew_lt0"), 0.0)
    terms["predicted_canopy_lt0"] = _num(raw.get("predicted_canopy_lt0"), 0.0)
    terms["dew_margin_next"] = _num(raw.get("dew_margin_next", raw.get("predicted_dew_margin_air_next")), 99.0)
    terms["canopy_dew_margin_next"] = _num(
        raw.get("canopy_dew_margin_next", raw.get("predicted_canopy_dew_margin_next")),
        99.0,
    )
    terms["prediction_model"] = str(raw.get("prediction_model", "") or "")
    return terms


def _selected_candidate(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    selected_name = str(row.get("rspc_action_selected_name", "") or row.get("selected_fallback_candidate", "") or "").strip()
    for candidate in candidates:
        if _truthy(candidate.get("selected")) or _candidate_name(candidate) == selected_name:
            return candidate
    return candidates[0] if candidates else {}


def _final_action(row: Mapping[str, Any]) -> dict[str, float]:
    return {name: _num(row.get(field)) for name, field in ACTION_MAP.items()}


def _action_delta(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {key: _num(left.get(key)) - _num(right.get(key)) for key in sorted(set(left) | set(right))}


def _state(row: Mapping[str, Any]) -> dict[str, float]:
    return {field: _num(row.get(field)) for field in STATE_FIELDS if field in row}


def _candidate_summary(candidate: Mapping[str, Any], *, selected: bool) -> dict[str, Any]:
    return {
        "name": _candidate_name(candidate),
        "selected": bool(selected),
        "score": _num(candidate.get("score")),
        "action": _candidate_action(candidate),
        "post_shape_action": _candidate_action(candidate, post_shape=True),
        "score_terms": _score_terms(candidate),
        "post_shape_score": _num(candidate.get("post_shape_score")),
        "post_shape_score_terms": _score_terms(candidate, post_shape=True),
        "post_shape_alignment": str(candidate.get("post_shape_alignment", "") or ""),
        "post_shape_eligible": _truthy(candidate.get("post_shape_eligible")),
    }


def _is_safe_dry_candidate(row: Mapping[str, Any], candidate: Mapping[str, Any], selected: Mapping[str, Any]) -> bool:
    if candidate is selected or _truthy(candidate.get("selected")):
        return False
    terms = _score_terms(candidate)
    selected_terms = _score_terms(selected)
    action = _candidate_action(candidate)
    selected_rh_next = _num(selected_terms.get("rh_next"), _num(row.get("rh_air")))
    selected_vpd_next = _num(selected_terms.get("vpd_next"), _num(row.get("vpd_air")))
    dry_better = (
        _num(terms.get("rh_next"), selected_rh_next) >= selected_rh_next + 0.5
        or _num(terms.get("vpd_next"), selected_vpd_next) <= selected_vpd_next - 0.03
        or _num(terms.get("dry_penalty")) <= _num(selected_terms.get("dry_penalty")) - 0.25
        or _num(terms.get("vpd_high_penalty")) <= _num(selected_terms.get("vpd_high_penalty")) - 0.05
    )
    actuator_safe = (
        _num(action.get("heat")) <= 0.05
        and _num(action.get("co2")) <= 0.05
        and _num(action.get("lamp")) <= 0.05
    )
    temp_safe = _num(terms.get("temp_next"), _num(row.get("temp_air"))) < 32.0 and _num(terms.get("temp_penalty")) <= _num(selected_terms.get("temp_penalty")) + 1.0
    dew_safe = (
        _num(terms.get("dew_penalty")) <= _num(selected_terms.get("dew_penalty")) + 0.25
        and _num(terms.get("canopy_dew_penalty")) <= _num(selected_terms.get("canopy_dew_penalty")) + 0.25
    )
    return bool(dry_better and actuator_safe and temp_safe and dew_safe)


def classify_step(
    row: Mapping[str, Any],
    next_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = _parse_candidates(row)
    if not candidates:
        return {"label": "missing_candidate_metadata", "candidate_count": 0}
    selected = _selected_candidate(row, candidates)
    selected_action = _candidate_action(selected, post_shape=True) or _candidate_action(selected)
    final_action = _final_action(row)
    post_guardrail_delta = _action_delta(final_action, selected_action)
    safe_dry_candidates = [c for c in candidates if _is_safe_dry_candidate(row, c, selected)]
    next_canopy = _num(next_row.get("canopy_dew_margin"), _num(row.get("canopy_dew_margin"), 99.0)) if next_row else _num(row.get("canopy_dew_margin"), 99.0)
    canopy_now = _num(row.get("canopy_dew_margin"), 99.0)
    canopy_drop = next_canopy - canopy_now
    dry_pressure = _num(row.get("rh_low_violation")) > 0.0 or _num(row.get("vpd_high_excess")) > 0.0 or _num(row.get("rh_air"), 70.0) < 62.0 or _num(row.get("vpd_air")) > 1.6
    guardrail_screen_up = _num(post_guardrail_delta.get("screen")) > 0.05
    guardrail_vent_down = _num(post_guardrail_delta.get("vent")) < -0.05
    guardrail_risk_pattern = guardrail_screen_up or guardrail_vent_down
    selected_terms = _score_terms(selected)
    selected_canopy_penalty = _num(selected_terms.get("canopy_dew_penalty"))
    selected_dew_penalty = _num(selected_terms.get("dew_penalty"))

    if guardrail_risk_pattern and (next_canopy < 0.0 or canopy_drop < -1.0):
        label = "guardrail_introduced_risk"
    elif (next_canopy < 0.0 or canopy_now < 1.0) and (selected_canopy_penalty > 0.0 or selected_dew_penalty > 0.0):
        label = "selected_candidate_unsafe_after_guardrail"
    elif safe_dry_candidates:
        best = min(safe_dry_candidates, key=lambda c: _num(c.get("score")))
        if _num(best.get("score")) > _num(selected.get("score")) + 0.05:
            label = "good_candidate_not_selected"
        else:
            label = "no_issue"
    elif dry_pressure or canopy_now < 1.0 or next_canopy < 0.0:
        label = "no_safe_candidate_available"
    else:
        label = "no_issue"

    return {
        "label": label,
        "candidate_count": len(candidates),
        "safe_dry_candidate_count": len(safe_dry_candidates),
        "best_safe_dry_candidate": _candidate_name(min(safe_dry_candidates, key=lambda c: _num(c.get("score")))) if safe_dry_candidates else "",
        "selected_candidate_name": _candidate_name(selected),
        "selected_score": _num(selected.get("score")),
        "post_guardrail_delta": post_guardrail_delta,
        "guardrail_screen_up": guardrail_screen_up,
        "guardrail_vent_down": guardrail_vent_down,
        "canopy_next": next_canopy,
        "canopy_delta_next": canopy_drop,
        "tomato_safety_v2_applied": _truthy(row.get("tomato_safety_v2_applied")),
        "tomato_safety_v2_reasons": str(row.get("tomato_safety_v2_reasons", "") or ""),
    }


def audit_trace(
    *,
    trace_path: str | Path,
    preset: str,
    scenario_id: str,
    start_step: int,
    end_step: int,
) -> dict[str, Any]:
    rows_by_step = _rows_by_step(_read_trace(trace_path))
    steps: list[dict[str, Any]] = []
    labels: Counter[str] = Counter()
    for step in range(int(start_step), int(end_step) + 1):
        row = rows_by_step.get(step)
        if row is None:
            continue
        next_row = rows_by_step.get(step + 1)
        classification = classify_step(row, next_row)
        labels[str(classification.get("label", ""))] += 1
        candidates = _parse_candidates(row)
        selected = _selected_candidate(row, candidates)
        steps.append(
            {
                "preset": preset,
                "scenario_id": scenario_id,
                "step": step,
                "state": _state(row),
                "final_action": _final_action(row),
                "selected_candidate": _candidate_summary(selected, selected=True) if selected else {},
                "candidates": [
                    _candidate_summary(candidate, selected=(candidate is selected or _candidate_name(candidate) == _candidate_name(selected)))
                    for candidate in candidates
                ],
                **classification,
            }
        )
    return {
        "preset": preset,
        "scenario_id": scenario_id,
        "trace_path": str(trace_path),
        "start_step": int(start_step),
        "end_step": int(end_step),
        "label_counts": dict(sorted(labels.items())),
        "steps": steps,
    }


def build_report(
    *,
    trace_dirs: Sequence[str | Path],
    scenario_id: str,
    start_step: int,
    end_step: int,
) -> dict[str, Any]:
    traces = []
    aggregate: Counter[str] = Counter()
    for trace_dir in trace_dirs:
        path = Path(trace_dir)
        preset = path.name if path.name else "trace"
        trace_path = _resolve_trace(path, scenario_id)
        item = audit_trace(
            trace_path=trace_path,
            preset=preset,
            scenario_id=scenario_id,
            start_step=start_step,
            end_step=end_step,
        )
        traces.append(item)
        aggregate.update(item.get("label_counts", {}))
    return {
        "schema_version": "candidate_pool_audit_v1",
        "scenario_id": scenario_id,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "trace_count": len(traces),
        "aggregate_label_counts": dict(sorted(aggregate.items())),
        "traces": traces,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Candidate Pool Audit",
        "",
        f"- Scenario: `{report.get('scenario_id', '')}`",
        f"- Steps: {report.get('start_step')}..{report.get('end_step')}",
        f"- Trace count: {report.get('trace_count', 0)}",
        "",
        "## Aggregate Labels",
        "",
        "| label | count |",
        "| --- | ---: |",
    ]
    for label, count in dict(report.get("aggregate_label_counts", {})).items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Step Summary", ""])
    lines.append("| preset | step | label | selected | safe_dry | screen | vent | d_screen_post | d_vent_post | tomato |")
    lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for trace in report.get("traces", []) or []:
        for step in trace.get("steps", []) or []:
            action = step.get("final_action", {}) if isinstance(step.get("final_action"), Mapping) else {}
            delta = step.get("post_guardrail_delta", {}) if isinstance(step.get("post_guardrail_delta"), Mapping) else {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(trace.get("preset", "")),
                        str(step.get("step", "")),
                        str(step.get("label", "")),
                        str(step.get("selected_candidate", {}).get("name", "")) if isinstance(step.get("selected_candidate"), Mapping) else "",
                        str(step.get("safe_dry_candidate_count", 0)),
                        f"{_num(action.get('screen')):.3f}",
                        f"{_num(action.get('vent')):.3f}",
                        f"{_num(delta.get('screen')):.3f}",
                        f"{_num(delta.get('vent')):.3f}",
                        str(step.get("tomato_safety_v2_reasons", "")) or "-",
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", action="append", required=True)
    parser.add_argument("--scenario-id", default="y2020_d120_s44_n240_llm_rspc_v2")
    parser.add_argument("--start-step", type=int, default=220)
    parser.add_argument("--end-step", type=int, default=239)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        trace_dirs=args.trace_dir,
        scenario_id=args.scenario_id,
        start_step=args.start_step,
        end_step=args.end_step,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json
    if not output_md.is_absolute():
        output_md = PROJECT_ROOT / output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"labels={report['aggregate_label_counts']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
