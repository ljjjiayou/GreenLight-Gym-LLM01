"""Audit whether post-guardrail shadow rows have joint prediction fields."""

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

REQUIRED_FIELDS = [
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


def build_report(vent_policy: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing_counts: Counter[str] = Counter()
    for row in vent_policy.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if not bool(row.get("vpd_prediction_available", False)):
            if "predicted_vpd_after_floor" not in missing:
                missing.append("predicted_vpd_after_floor")
        for field in missing:
            missing_counts[field] += 1
        rows.append(
            {
                "scenario_id": row.get("scenario_id", ""),
                "preset": row.get("preset", ""),
                "step": row.get("step"),
                "policy_label": row.get("policy_label", ""),
                "missing_joint_prediction_fields": missing,
                "ready_for_policy_judgment": not missing,
            }
        )
    ready = bool(rows) and all(row["ready_for_policy_judgment"] for row in rows)
    return {
        "schema_version": "post_guardrail_joint_prediction_readiness_v1",
        "mainline_alignment": {
            "affected_layers": ["Response Estimate", "Safety Boundary"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only metadata readiness audit",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "ready_for_policy_judgment": ready,
        "required_fields": REQUIRED_FIELDS,
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "rows": rows,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Joint Prediction Readiness",
        "",
        "- Mode: read-only metadata readiness audit",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Ready for policy judgment: {report.get('ready_for_policy_judgment', False)}",
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
    print(f"ready_for_policy_judgment={report['ready_for_policy_judgment']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
