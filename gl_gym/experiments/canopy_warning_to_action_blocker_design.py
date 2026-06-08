"""Design a shadow blocker for canopy warnings that are not actionable yet."""

from __future__ import annotations

import argparse
import json
import sys
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


def build_report(warning_audit: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for row in warning_audit.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "scenario_id": row.get("scenario_id", ""),
                "preset": row.get("preset", ""),
                "step": row.get("step"),
                "trigger_condition": "pre_or_post_guardrail_predicted_canopy_margin_below_threshold",
                "blocked_action_type": "post_guardrail_action_with_known_canopy_warning",
                "requires_safe_alternative_check": True,
                "fallback_needed_if_no_safe_alternative": True,
                "actual_hard_crossing_recall_target": True,
                "lead_time_steps_required": 1,
                "control_action_changed": False,
            }
        )
    return {
        "schema_version": "canopy_warning_to_action_blocker_design_v1",
        "mainline_alignment": {
            "affected_layers": ["Safety Boundary", "Response Estimate"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow blocker design-only",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "policy_name": "canopy_warning_to_action_blocker_v1",
        "design_row_count": len(rows),
        "actual_hard_crossing_rows_covered": sum(1 for row in warning_audit.get("rows", []) or [] if isinstance(row, Mapping) and row.get("actual_hard_crossing")),
        "rows": rows,
        "acceptance_criteria": {
            "actual_hard_crossing_recall": "100% on canonical failure warning rows",
            "false_positive_count_reported_by_regime": True,
            "safe_alternative_or_fallback_recorded": True,
        },
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Canopy Warning-to-Action Blocker v1 Design",
            "",
            "- Mode: shadow blocker design-only",
            "- Controlled replay allowed: false",
            "- Performance claim allowed: false",
            f"- Design rows: {report.get('design_row_count', 0)}",
            f"- Actual hard crossing rows covered: {report.get('actual_hard_crossing_rows_covered', 0)}",
            "",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warning-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.warning_json))
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
