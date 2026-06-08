"""Design vent-floor shadow policy v2 from v1 audit rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

JOINT_PREDICTION_FIELDS = [
    "predicted_canopy_margin_after_floor",
    "predicted_dew_margin_after_floor",
    "predicted_rh_after_floor",
    "predicted_vpd_after_floor",
    "predicted_temp_after_floor",
    "predicted_dry_risk_after_floor",
]


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def build_report(vent_policy_v1: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    excluded = []
    for row in vent_policy_v1.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        label = str(row.get("policy_label", ""))
        if label == "vent_floor_shadow_block":
            output_row = {
                "scenario_id": row.get("scenario_id", ""),
                "preset": row.get("preset", ""),
                "step": row.get("step"),
                "trigger_condition": "post_guardrail_delta_vent_lt0_and_canopy_or_dry_risk_active",
                "shadow_constraint": "final_vent_not_below_shadow_floor",
                "shadow_floor_vent": row.get("shadow_floor_vent"),
                "required_joint_prediction_fields": JOINT_PREDICTION_FIELDS,
                "missing_joint_prediction_fields": [
                    field for field in JOINT_PREDICTION_FIELDS if field not in row
                ],
                "control_action_changed": False,
            }
            for field in JOINT_PREDICTION_FIELDS:
                if field in row:
                    output_row[field] = row.get(field)
            rows.append(output_row)
        elif label == "hard_warning_to_action_shadow_needed":
            excluded.append(
                {
                    "scenario_id": row.get("scenario_id", ""),
                    "preset": row.get("preset", ""),
                    "step": row.get("step"),
                    "reason": "belongs_to_warning_to_action_blocker_not_vent_floor_v2",
                }
            )
    return {
        "schema_version": "post_guardrail_vent_floor_shadow_policy_v2_design_v1",
        "mainline_alignment": {
            "affected_layers": ["Safety Boundary", "Response Estimate"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow policy design-only",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "policy_name": "post_guardrail_vent_floor_shadow_policy_v2",
        "design_row_count": len(rows),
        "excluded_hard_warning_rows": len(excluded),
        "requires_joint_prediction_before_policy_judgment": True,
        "rows": rows,
        "excluded_rows": excluded,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Post-Guardrail Vent-Floor Shadow Policy v2 Design",
            "",
            "- Mode: shadow policy design-only",
            "- Controlled replay allowed: false",
            "- Performance claim allowed: false",
            f"- Design rows: {report.get('design_row_count', 0)}",
            f"- Excluded hard-warning rows: {report.get('excluded_hard_warning_rows', 0)}",
            "- Requires joint prediction before policy judgment: true",
            "",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vent-policy-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.vent_policy_json))
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
    print(f"design_row_count={report['design_row_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
