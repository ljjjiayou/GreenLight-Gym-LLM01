"""Evaluate a metadata-only post-guardrail vent-floor shadow policy."""

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


def _find_variant(row: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for variant in row.get("variants", []) or []:
        if isinstance(variant, Mapping) and str(variant.get("variant", "")) == name:
            return variant
    return {}


def _action_vent(variant: Mapping[str, Any]) -> float:
    action = variant.get("action", {})
    if not isinstance(action, Mapping):
        return 0.0
    return _num(action.get("vent"))


def _label_row(row: Mapping[str, Any], *, vent_floor: float, warning_threshold: float) -> str:
    conclusion = str(row.get("counterfactual_conclusion", ""))
    pre_pred = _num(row.get("pre_score_pred_canopy_margin"), 9.0)
    post_pred = _num(row.get("post_guardrail_pred_canopy_margin"), 9.0)
    actual_next = _num(row.get("actual_next_canopy_margin"), 9.0)
    if conclusion == "proxy_warning_not_actionable" or min(pre_pred, post_pred, actual_next) < warning_threshold:
        return "hard_warning_to_action_shadow_needed"
    floor_variant = _find_variant(row, "post_guardrail_canopy_aware_vent_floor")
    if _action_vent(floor_variant) >= vent_floor - 1e-6 and _num(row.get("delta_vent")) < -1e-6:
        return "vent_floor_shadow_block"
    return "no_shadow_block"


def build_report(
    counterfactual_report: Mapping[str, Any],
    *,
    vent_floor: float = 0.30,
    warning_threshold: float = 0.25,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    for row in counterfactual_report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        post_variant = _find_variant(row, "post_guardrail_action")
        floor_variant = _find_variant(row, "post_guardrail_canopy_aware_vent_floor")
        pre_variant = _find_variant(row, "pre_score_action")
        label = _label_row(row, vent_floor=vent_floor, warning_threshold=warning_threshold)
        label_counts[label] += 1
        rows.append(
            {
                "scenario_id": row.get("scenario_id", ""),
                "preset": row.get("preset", ""),
                "step": row.get("step"),
                "counterfactual_conclusion": row.get("counterfactual_conclusion", ""),
                "policy_label": label,
                "would_block_rewrite": label in {"vent_floor_shadow_block", "hard_warning_to_action_shadow_needed"},
                "would_apply_vent_floor": _action_vent(post_variant) < vent_floor <= max(_action_vent(floor_variant), vent_floor),
                "pre_score_vent": _action_vent(pre_variant),
                "post_guardrail_vent": _action_vent(post_variant),
                "shadow_floor_vent": max(_action_vent(post_variant), vent_floor),
                "predicted_canopy_margin_after_floor": _num(floor_variant.get("predicted_canopy_margin"), 0.0),
                "post_guardrail_pred_canopy_margin": _num(row.get("post_guardrail_pred_canopy_margin"), 0.0),
                "actual_next_canopy_margin": _num(row.get("actual_next_canopy_margin"), 0.0),
                "vpd_prediction_available": False,
                "notes": "metadata-only; not applied to final action",
            }
        )
    return {
        "schema_version": "post_guardrail_vent_floor_shadow_policy_audit_v1",
        "mainline_alignment": {
            "affected_layers": ["Safety Boundary", "Response Estimate"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only policy audit",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "policy_name": "post_guardrail_vent_floor_shadow_policy_v1",
        "vent_floor": vent_floor,
        "warning_threshold": warning_threshold,
        "label_counts": dict(sorted(label_counts.items())),
        "would_block_count": sum(1 for row in rows if row["would_block_rewrite"]),
        "hard_warning_to_action_rows": label_counts.get("hard_warning_to_action_shadow_needed", 0),
        "vent_floor_shadow_block_rows": label_counts.get("vent_floor_shadow_block", 0),
        "ready_for_controlled_replay": False,
        "rows": rows,
        "notes": [
            "This policy is evaluated as metadata only.",
            "Rows labelled hard_warning_to_action_shadow_needed require a blocker design beyond a simple vent floor.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Vent-Floor Shadow Policy Audit",
        "",
        "- Mode: shadow-only policy audit",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Policy: `{report.get('policy_name', '')}`",
        f"- Would block count: {report.get('would_block_count', 0)}",
        f"- Ready for controlled replay: {report.get('ready_for_controlled_replay', False)}",
        "",
        "## Label Counts",
        "",
        "| label | count |",
        "| --- | ---: |",
    ]
    for label, count in dict(report.get("label_counts", {})).items():
        lines.append(f"| {label} | {count} |")
    lines.extend(["", "## Rows", "", "| preset | step | label | post vent | floor vent | pred canopy floor |", "| --- | ---: | --- | ---: | ---: | ---: |"])
    for row in report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"| {row.get('preset', '')} | {row.get('step', '')} | {row.get('policy_label', '')} | "
            f"{_num(row.get('post_guardrail_vent')):.3f} | {_num(row.get('shadow_floor_vent')):.3f} | "
            f"{_num(row.get('predicted_canopy_margin_after_floor')):.3f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual-json", required=True)
    parser.add_argument("--vent-floor", type=float, default=0.30)
    parser.add_argument("--warning-threshold", type=float, default=0.25)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        _load(args.counterfactual_json),
        vent_floor=args.vent_floor,
        warning_threshold=args.warning_threshold,
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
    print(f"would_block_count={report['would_block_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
