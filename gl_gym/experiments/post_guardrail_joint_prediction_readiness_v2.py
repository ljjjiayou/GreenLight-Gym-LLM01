"""Audit joint prediction readiness for vent-floor and warning-to-action rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CORE_JOINT_FIELDS = [
    "predicted_canopy_margin_after_floor",
    "predicted_dew_margin_after_floor",
    "predicted_rh_after_floor",
    "predicted_vpd_after_floor",
    "predicted_temp_after_floor",
    "predicted_dry_risk_after_floor",
    "regime_conflict_flag",
]


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _read_trace_csv(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_trace_dir(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    rows: list[dict[str, Any]] = []
    for trace in sorted(p.glob("*.csv")):
        rows.extend(_read_trace_csv(trace))
    return rows


def _trace_row_to_joint_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": row.get("scenario_id", "y2020_d120_s44_n240"),
        "preset": row.get("preset", row.get("algo", "")),
        "step": row.get("step"),
        "policy_name": "stage_b_final_action_trace",
        "policy_label": "stage_b_final_action_trace",
        "predicted_canopy_margin_after_floor": row.get("rspc_action_selected_predicted_canopy_dew_margin_next"),
        "predicted_dew_margin_after_floor": row.get("rspc_action_selected_predicted_dew_margin_air_next"),
        "predicted_rh_after_floor": row.get("rspc_action_selected_predicted_rh_next"),
        "predicted_vpd_after_floor": row.get("rspc_action_selected_predicted_vpd_next"),
        "predicted_temp_after_floor": row.get("rspc_action_selected_predicted_temp_next"),
        "predicted_dry_risk_after_floor": row.get("dry_risk"),
        "regime_conflict_flag": row.get("final_action_screen_vent_conflict"),
    }


def build_trace_report(*, trace_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return build_report(
        vent_v2={"rows": [_trace_row_to_joint_row(row) for row in trace_rows]},
        warning_design={"rows": []},
    )


def _joint_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in CORE_JOINT_FIELDS:
        available = field in row and row.get(field) is not None and row.get(field) != ""
        payload[field] = {
            "available": bool(available),
            "value": row.get(field) if available else None,
            "prediction_source": "trace_metadata" if available else "missing_in_trace",
        }
    return payload


def _row_report(row: Mapping[str, Any], *, row_family: str) -> dict[str, Any]:
    joint = _joint_payload(row)
    missing = [field for field, spec in joint.items() if not bool(spec.get("available"))]
    return {
        "scenario_id": row.get("scenario_id", ""),
        "preset": row.get("preset", ""),
        "step": row.get("step"),
        "row_family": row_family,
        "policy_name": row.get("policy_name", ""),
        "policy_label": row.get("policy_label", row_family),
        "joint_prediction": joint,
        "missing_joint_prediction_fields": missing,
        "ready_for_policy_judgment": not bool(missing),
    }


def build_report(
    *,
    vent_v2: Mapping[str, Any],
    warning_design: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_counts: Counter[str] = Counter()
    for row in vent_v2.get("rows", []) or []:
        if isinstance(row, Mapping):
            report_row = _row_report(row, row_family="vent_floor_v2")
            rows.append(report_row)
            missing_counts.update(report_row["missing_joint_prediction_fields"])
    for row in warning_design.get("rows", []) or []:
        if isinstance(row, Mapping):
            report_row = _row_report(row, row_family="warning_to_action")
            rows.append(report_row)
            missing_counts.update(report_row["missing_joint_prediction_fields"])
    ready = bool(rows) and all(bool(row["ready_for_policy_judgment"]) for row in rows)
    return {
        "schema_version": "post_guardrail_joint_prediction_readiness_v2",
        "mainline_alignment": {
            "affected_layers": ["Response Estimate", "Safety Boundary"],
            "default_llm_rspc_v2_changed": False,
            "mode": "metadata readiness audit",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "ready_for_policy_judgment": ready,
        "ready_for_shadow_audit": ready,
        "required_fields": CORE_JOINT_FIELDS,
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "row_count": len(rows),
        "rows": rows,
        "notes": [
            "Fields are reported even when missing; missing values block policy judgment.",
            "This report defines readiness only. It does not change action selection.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Joint Prediction Readiness v2",
        "",
        "- Mode: metadata readiness audit",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Ready for policy judgment: {report.get('ready_for_policy_judgment', False)}",
        f"- Row count: {report.get('row_count', 0)}",
        "",
        "## Missing Field Counts",
        "",
        "| field | count |",
        "| --- | ---: |",
    ]
    for field, count in dict(report.get("missing_field_counts", {})).items():
        lines.append(f"| {field} | {count} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vent-v2-json", default="")
    parser.add_argument("--warning-design-json", default="")
    parser.add_argument("--trace-csv", default="")
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.trace_csv and args.trace_dir:
        raise SystemExit("provide only one of --trace-csv or --trace-dir")
    if args.trace_csv:
        report = build_trace_report(trace_rows=_read_trace_csv(args.trace_csv))
    elif args.trace_dir:
        report = build_trace_report(trace_rows=_read_trace_dir(args.trace_dir))
    else:
        if not args.vent_v2_json or not args.warning_design_json:
            raise SystemExit("--vent-v2-json/--warning-design-json or --trace-csv is required")
        report = build_report(
            vent_v2=_load(args.vent_v2_json),
            warning_design=_load(args.warning_design_json),
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
    print(f"ready_for_policy_judgment={report['ready_for_policy_judgment']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
