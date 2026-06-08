"""Read-only counterfactual audit for post-guardrail ventilation rewrites.

The counterfactuals are metadata-only hypotheses over already recorded harmful
rows.  They do not replay the simulator and do not change any controller code.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ACTION_FIELDS = ("heat", "co2", "screen", "vent", "lamp", "shade")
VENT_FLOOR = 0.30
RISK_SCREEN = 0.55
RISK_VENT = 0.25


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _action(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {field: 0.0 for field in ACTION_FIELDS}
    return {field: _num(value.get(field)) for field in ACTION_FIELDS}


def _copy_action(action: Mapping[str, float], **updates: float) -> dict[str, float]:
    out = {field: _num(action.get(field)) for field in ACTION_FIELDS}
    for key, value in updates.items():
        out[key] = max(0.0, min(1.0, float(value)))
    return out


def _screen_vent_risk(action: Mapping[str, float]) -> bool:
    return bool(_num(action.get("screen")) >= RISK_SCREEN and _num(action.get("vent")) <= RISK_VENT)


def _proxy_counterfactual_prediction(row: Mapping[str, Any], action: Mapping[str, float]) -> dict[str, Any]:
    post = _action(row.get("post_guardrail_action", {}))
    post_pred = _num(row.get("post_guardrail_pred_canopy_margin"), 99.0)
    pre_pred = _num(row.get("pre_score_pred_canopy_margin"), post_pred)
    pre = _action(row.get("pre_score_action", {}))
    observed_vent_span = max(_num(pre.get("vent")) - _num(post.get("vent")), 0.0)
    observed_pred_span = pre_pred - post_pred
    vent_gain_per_unit = observed_pred_span / observed_vent_span if observed_vent_span > 1e-9 else 0.0
    vent_gain = max(_num(action.get("vent")) - _num(post.get("vent")), 0.0) * vent_gain_per_unit
    screen_penalty = max(_num(action.get("screen")) - _num(post.get("screen")), 0.0) * 0.25
    predicted = post_pred + vent_gain - screen_penalty
    return {
        "predicted_canopy_margin": predicted,
        "screen_vent_risk": _screen_vent_risk(action),
        "vent_gain_per_unit": vent_gain_per_unit,
        "prediction_model": "linearized_from_pre_post_proxy_delta",
    }


def _variants(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    pre = _action(row.get("pre_score_action", {}))
    post = _action(row.get("post_guardrail_action", {}))
    keep_pre_vent = _copy_action(post, vent=max(_num(post.get("vent")), _num(pre.get("vent"))))
    vent_floor = _copy_action(post, vent=max(_num(post.get("vent")), VENT_FLOOR))
    no_combo = dict(post)
    if _screen_vent_risk(no_combo):
        no_combo = _copy_action(
            no_combo,
            vent=max(_num(no_combo.get("vent")), VENT_FLOOR),
            screen=min(_num(no_combo.get("screen")), max(_num(pre.get("screen")), RISK_SCREEN)),
        )
    variants = [
        ("pre_score_action", pre),
        ("post_guardrail_action", post),
        ("post_guardrail_keep_pre_score_vent", keep_pre_vent),
        ("post_guardrail_canopy_aware_vent_floor", vent_floor),
        ("post_guardrail_no_screen_vent_risk_combo", no_combo),
    ]
    return [
        {
            "variant": name,
            "action": action,
            **_proxy_counterfactual_prediction(row, action),
        }
        for name, action in variants
    ]


def _conclusion(row: Mapping[str, Any]) -> str:
    d_vent = _num(row.get("delta_vent"))
    other_delta = max(abs(_num(row.get("delta_heat"))), abs(_num(row.get("delta_screen"))), abs(_num(row.get("delta_shade"))))
    fine = set(str(item) for item in row.get("fine_grained_root_causes", []) or [])
    if "proxy_warning_exists_but_not_actionable" in fine or _num(row.get("pre_score_pred_canopy_margin"), 99.0) < 0.25:
        return "proxy_warning_not_actionable"
    if d_vent < -0.05 and other_delta <= 0.05:
        return "vent_floor_likely_sufficient"
    if "screen_vent_coupled_risk" in fine:
        return "screen_vent_coupling_required"
    return "counterfactual_inconclusive"


def build_report(rewrite_report: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    conclusions: Counter[str] = Counter()
    for row in rewrite_report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        label = _conclusion(row)
        conclusions[label] += 1
        rows.append(
            {
                "scenario_id": row.get("scenario_id", ""),
                "preset": row.get("preset", ""),
                "step": row.get("step"),
                "selected_candidate_name": row.get("selected_candidate_name", ""),
                "delta_vent": _num(row.get("delta_vent")),
                "delta_screen": _num(row.get("delta_screen")),
                "actual_next_canopy_margin": _num(row.get("actual_next_canopy_margin"), 99.0),
                "pre_score_pred_canopy_margin": _num(row.get("pre_score_pred_canopy_margin"), 99.0),
                "post_guardrail_pred_canopy_margin": _num(row.get("post_guardrail_pred_canopy_margin"), 99.0),
                "counterfactual_conclusion": label,
                "variants": _variants(row),
            }
        )
    return {
        "schema_version": "post_guardrail_vent_rewrite_counterfactual_audit_v1",
        "mainline_alignment": {
            "affected_layers": ["Safety Boundary", "Response Estimate"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only counterfactual audit",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "vent_floor_likely_sufficient": bool(conclusions.get("vent_floor_likely_sufficient", 0) > 0),
        "screen_vent_coupling_required": bool(conclusions.get("screen_vent_coupling_required", 0) > 0),
        "proxy_warning_not_actionable": bool(conclusions.get("proxy_warning_not_actionable", 0) > 0),
        "counterfactual_inconclusive": bool(conclusions.get("counterfactual_inconclusive", 0) > 0),
        "conclusion_counts": dict(sorted(conclusions.items())),
        "rows": rows,
        "notes": [
            "Counterfactuals are proxy estimates over recorded rows, not simulator rollouts.",
            "A positive vent-floor signal supports shadow design only; it is not a control change.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Vent Rewrite Counterfactual Audit",
        "",
        "- Mode: read-only counterfactual audit",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "",
        "## Conclusions",
        "",
        f"- vent_floor_likely_sufficient: {report.get('vent_floor_likely_sufficient', False)}",
        f"- screen_vent_coupling_required: {report.get('screen_vent_coupling_required', False)}",
        f"- proxy_warning_not_actionable: {report.get('proxy_warning_not_actionable', False)}",
        f"- counterfactual_inconclusive: {report.get('counterfactual_inconclusive', False)}",
        f"- counts: {json.dumps(report.get('conclusion_counts', {}), sort_keys=True)}",
        "",
        "## Harmful Rows",
        "",
        "| preset | step | d_vent | pre_pred | post_pred | actual_next_canopy | conclusion | variants |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        variant_summary = ",".join(
            f"{item.get('variant')}:{_num(item.get('action', {}).get('vent')):.3f}/{_num(item.get('predicted_canopy_margin')):.3f}"
            for item in row.get("variants", []) or []
            if isinstance(item, Mapping)
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("preset", "")),
                    str(row.get("step", "")),
                    f"{_num(row.get('delta_vent')):.3f}",
                    f"{_num(row.get('pre_score_pred_canopy_margin')):.3f}",
                    f"{_num(row.get('post_guardrail_pred_canopy_margin')):.3f}",
                    f"{_num(row.get('actual_next_canopy_margin')):.3f}",
                    str(row.get("counterfactual_conclusion", "")),
                    variant_summary,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrite-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.rewrite_json))
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
    print(f"conclusion_counts={report['conclusion_counts']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
