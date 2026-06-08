"""Detailed read-only table for post-guardrail action rewrites.

The script reuses the post-guardrail override audit and reshapes harmful rows
into an action-diff table focused on why a safe pre-score candidate can become
unsafe after post-processing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.post_guardrail_override_audit import build_report as build_override_report


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _layer(row: Mapping[str, Any]) -> str:
    if bool(row.get("pre_score_candidate_safe_but_post_guardrail_unsafe")):
        return "post_guardrail_destruction"
    if bool(row.get("no_candidate_can_satisfy_canopy_boundary")):
        return "candidate_space_insufficiency"
    if bool(row.get("candidate_exists_but_ranked_low")) or bool(row.get("scorer_selected_unsafe")):
        return "candidate_scoring_failure"
    if bool(row.get("response_proxy_false_safe")):
        return "response_proxy_false_safe"
    if str(row.get("outcome_label", "")) == "harmful_override":
        return "unclassified_harmful_override"
    return "no_harmful_override"


def _pred_canopy(pred: Mapping[str, Any], *, fallback: float = 0.0) -> float:
    for key in (
        "predicted_canopy_dew_margin_next_v2_1",
        "predicted_canopy_dew_margin_next_v2",
        "predicted_canopy_dew_margin_next",
        "canopy_dew_margin_next",
    ):
        if key in pred:
            return _num(pred.get(key), fallback)
    return fallback


def _flatten_row(trace: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    delta = row.get("post_guardrail_action_delta") or row.get("action_delta") or {}
    if not isinstance(delta, Mapping):
        delta = {}
    next_metrics = row.get("actual_next_metrics") or row.get("next_response") or {}
    if not isinstance(next_metrics, Mapping):
        next_metrics = {}
    pre_pred = row.get("pre_score_proxy_prediction", {})
    post_pred = row.get("post_guardrail_proxy_prediction", {})
    if not isinstance(pre_pred, Mapping):
        pre_pred = {}
    if not isinstance(post_pred, Mapping):
        post_pred = {}
    return {
        "preset": trace.get("preset", ""),
        "scenario_id": trace.get("scenario_id", ""),
        "step": row.get("step"),
        "outcome_label": row.get("outcome_label", ""),
        "root_cause_layer": _layer(row),
        "selected_candidate_name": row.get("selected_candidate_name", ""),
        "pre_score_action": row.get("pre_score_selected_action", row.get("pre_guardrail_action", {})),
        "post_guardrail_action": row.get("post_guardrail_action", row.get("final_action", {})),
        "delta_heat": _num(delta.get("heat")),
        "delta_screen": _num(delta.get("screen")),
        "delta_vent": _num(delta.get("vent")),
        "delta_shade": _num(delta.get("shade")),
        "pre_score_pred_canopy_margin": _pred_canopy(pre_pred, fallback=99.0),
        "post_guardrail_pred_canopy_margin": _pred_canopy(post_pred, fallback=99.0),
        "actual_next_canopy_margin": _num(
            next_metrics.get("canopy_dew_margin", next_metrics.get("next_canopy_dew_margin")), 99.0
        ),
        "actual_next_dew_margin": _num(next_metrics.get("dew_margin_air", next_metrics.get("next_dew_margin_air")), 99.0),
        "actual_next_rh": _num(next_metrics.get("rh_air", next_metrics.get("next_rh_air")), 0.0),
        "actual_next_vpd": _num(next_metrics.get("vpd_air", next_metrics.get("next_vpd_air")), 0.0),
        "tomato_safety_v2_reasons": row.get("tomato_safety_v2_reasons", ""),
        "root_causes": list(row.get("root_causes", []) or []),
        "fine_grained_root_causes": list(row.get("fine_grained_root_causes", []) or []),
    }


def build_report(
    *,
    trace_dirs: Sequence[str | Path],
    scenario_id: str,
    start_step: int,
    end_step: int,
    include_all: bool = False,
) -> dict[str, Any]:
    base = build_override_report(
        trace_dirs=trace_dirs,
        scenario_id=scenario_id,
        start_step=start_step,
        end_step=end_step,
    )
    rows: list[dict[str, Any]] = []
    layers: Counter[str] = Counter()
    for trace in base.get("traces", []) or []:
        if not isinstance(trace, Mapping):
            continue
        for row in trace.get("rows", []) or []:
            if not isinstance(row, Mapping):
                continue
            if not include_all and str(row.get("outcome_label", "")) != "harmful_override":
                continue
            flat = _flatten_row(trace, row)
            rows.append(flat)
            layers[str(flat.get("root_cause_layer", ""))] += 1
    return {
        "schema_version": "post_guardrail_action_rewrite_diff_audit_v1",
        "mainline_alignment": {
            "affected_layers": ["Safety Boundary", "Candidate Pool"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only root-cause audit",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "scenario_id": scenario_id,
        "start_step": int(start_step),
        "end_step": int(end_step),
        "harmful_override_count": int(base.get("harmful_override_count", 0)),
        "root_cause_layer_counts": dict(sorted(layers.items())),
        "source_root_cause_summary": dict(base.get("root_cause_summary", {})),
        "source_harmful_fine_grained_root_cause_summary": dict(
            base.get("harmful_fine_grained_root_cause_summary", {})
        ),
        "rows": rows,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Action Rewrite Diff Audit",
        "",
        "## Mainline Alignment",
        "",
        "- Mode: read-only root-cause audit",
        "- Default `llm_rspc_v2` changed: false",
        "- Reopens rejected preset: false",
        "",
        "## Summary",
        "",
        f"- Scenario: `{report.get('scenario_id', '')}`",
        f"- Steps: {report.get('start_step')}..{report.get('end_step')}",
        f"- Harmful override count: {report.get('harmful_override_count', 0)}",
        f"- Root-cause layer counts: {json.dumps(report.get('root_cause_layer_counts', {}), sort_keys=True)}",
        "",
        "## Harmful Rewrite Rows",
        "",
        "| preset | step | layer | candidate | d_heat | d_screen | d_vent | d_shade | pre_pred_canopy | post_pred_canopy | actual_next_canopy | actual_next_dew | actual_next_rh | actual_next_vpd | root_causes |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("preset", "")),
                    str(row.get("step", "")),
                    str(row.get("root_cause_layer", "")),
                    str(row.get("selected_candidate_name", "")),
                    f"{_num(row.get('delta_heat')):.3f}",
                    f"{_num(row.get('delta_screen')):.3f}",
                    f"{_num(row.get('delta_vent')):.3f}",
                    f"{_num(row.get('delta_shade')):.3f}",
                    f"{_num(row.get('pre_score_pred_canopy_margin')):.3f}",
                    f"{_num(row.get('post_guardrail_pred_canopy_margin')):.3f}",
                    f"{_num(row.get('actual_next_canopy_margin')):.3f}",
                    f"{_num(row.get('actual_next_dew_margin')):.3f}",
                    f"{_num(row.get('actual_next_rh')):.3f}",
                    f"{_num(row.get('actual_next_vpd')):.3f}",
                    ",".join(str(item) for item in row.get("root_causes", []) or []) or "-",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", action="append", required=True)
    parser.add_argument("--scenario-id", default="y2020_d120_s44_n240")
    parser.add_argument("--start-step", type=int, default=220)
    parser.add_argument("--end-step", type=int, default=239)
    parser.add_argument("--include-all", action="store_true")
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
        include_all=args.include_all,
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
    print(f"harmful_override_count={report['harmful_override_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
