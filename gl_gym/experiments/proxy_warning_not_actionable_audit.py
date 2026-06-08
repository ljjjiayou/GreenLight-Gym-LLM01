"""Audit proxy warnings that did not become actionable safety blockers."""

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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def build_report(counterfactual_report: Mapping[str, Any], *, warning_threshold: float = 0.25) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in counterfactual_report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        pre_pred = _num(row.get("pre_score_pred_canopy_margin"), 9.0)
        post_pred = _num(row.get("post_guardrail_pred_canopy_margin"), 9.0)
        actual_next = _num(row.get("actual_next_canopy_margin"), 9.0)
        warning = (
            str(row.get("counterfactual_conclusion", "")) == "proxy_warning_not_actionable"
            or pre_pred < warning_threshold
            or post_pred < warning_threshold
            or actual_next < 0.0
        )
        if not warning:
            continue
        rows.append(
            {
                "scenario_id": row.get("scenario_id", ""),
                "preset": row.get("preset", ""),
                "step": row.get("step"),
                "pre_score_pred_canopy_margin": pre_pred,
                "post_guardrail_pred_canopy_margin": post_pred,
                "actual_next_canopy_margin": actual_next,
                "warning_known_before_post_guardrail": pre_pred < warning_threshold,
                "warning_remained_after_post_guardrail": post_pred < warning_threshold,
                "actual_hard_crossing": actual_next < 0.0,
                "requires_warning_to_action_blocker": True,
                "suggested_shadow_blocker": "canopy_warning_to_action_blocker_v1",
            }
        )
    return {
        "schema_version": "proxy_warning_not_actionable_audit_v1",
        "mainline_alignment": {
            "affected_layers": ["Safety Boundary", "Response Estimate"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only warning audit",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "warning_threshold": warning_threshold,
        "proxy_warning_not_actionable_count": len(rows),
        "actual_hard_crossing_count": sum(1 for row in rows if row["actual_hard_crossing"]),
        "requires_warning_to_action_blocker_shadow_design": bool(rows),
        "rows": rows,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Proxy Warning Not Actionable Audit",
        "",
        "- Mode: read-only warning audit",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Proxy warning not actionable count: {report.get('proxy_warning_not_actionable_count', 0)}",
        f"- Requires warning-to-action blocker shadow design: {report.get('requires_warning_to_action_blocker_shadow_design', False)}",
        "",
        "| preset | step | pre pred | post pred | actual next | hard crossing |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"| {row.get('preset', '')} | {row.get('step', '')} | "
            f"{_num(row.get('pre_score_pred_canopy_margin')):.3f} | "
            f"{_num(row.get('post_guardrail_pred_canopy_margin')):.3f} | "
            f"{_num(row.get('actual_next_canopy_margin')):.3f} | {row.get('actual_hard_crossing', False)} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual-json", required=True)
    parser.add_argument("--warning-threshold", type=float, default=0.25)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.counterfactual_json), warning_threshold=args.warning_threshold)
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
    print(f"proxy_warning_not_actionable_count={report['proxy_warning_not_actionable_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
