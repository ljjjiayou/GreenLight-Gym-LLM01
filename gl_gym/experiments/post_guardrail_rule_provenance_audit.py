"""Audit whether post-guardrail action rewrites have explainable provenance."""

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


def _rewrite_magnitude(row: Mapping[str, Any]) -> float:
    return sum(abs(_num(row.get(key))) for key in ("delta_heat", "delta_screen", "delta_vent", "delta_shade"))


def _which_rule(row: Mapping[str, Any]) -> str:
    reasons = str(row.get("tomato_safety_v2_reasons", "") or "").strip()
    if reasons:
        return "tomato_safety_v2"
    if "guardrail_introduced_risk" in set(str(item) for item in row.get("root_causes", []) or []):
        return "unknown_post_guardrail_rewrite"
    return "none"


def build_report(rewrite_report: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    rule_counts: Counter[str] = Counter()
    missing_count = 0
    for row in rewrite_report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        rule = _which_rule(row)
        reasons = str(row.get("tomato_safety_v2_reasons", "") or "").strip()
        magnitude = _rewrite_magnitude(row)
        reason_missing = bool(magnitude > 0.05 and not reasons)
        if reason_missing:
            missing_count += 1
        rule_counts[rule] += 1
        rows.append(
            {
                "scenario_id": row.get("scenario_id", ""),
                "preset": row.get("preset", ""),
                "step": row.get("step"),
                "which_rule_fired": rule,
                "pre_rule_action": row.get("pre_score_action", {}),
                "post_rule_action": row.get("post_guardrail_action", {}),
                "rule_reason": reasons,
                "rule_priority": "unknown" if rule == "unknown_post_guardrail_rewrite" else "not_available",
                "reason_missing": reason_missing,
                "rewrite_magnitude": magnitude,
                "delta_vent": _num(row.get("delta_vent")),
                "unsafe_for_promotion_due_to_missing_provenance": reason_missing,
            }
        )
    return {
        "schema_version": "post_guardrail_rule_provenance_audit_v1",
        "mainline_alignment": {
            "affected_layers": ["Safety Boundary", "Evaluation"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only provenance audit",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "unsafe_for_promotion_due_to_missing_provenance": bool(missing_count > 0),
        "reason_missing_count": missing_count,
        "rule_counts": dict(sorted(rule_counts.items())),
        "rows": rows,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Rule Provenance Audit",
        "",
        "- Mode: read-only provenance audit",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Unsafe for promotion due to missing provenance: {report.get('unsafe_for_promotion_due_to_missing_provenance', False)}",
        f"- Missing reason count: {report.get('reason_missing_count', 0)}",
        "",
        "## Rule Counts",
        "",
        "| rule | count |",
        "| --- | ---: |",
    ]
    for rule, count in dict(report.get("rule_counts", {})).items():
        lines.append(f"| {rule} | {count} |")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| preset | step | rule | reason_missing | d_vent | reason |",
            "| --- | ---: | --- | --- | ---: | --- |",
        ]
    )
    for row in report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("preset", "")),
                    str(row.get("step", "")),
                    str(row.get("which_rule_fired", "")),
                    str(row.get("reason_missing", False)),
                    f"{_num(row.get('delta_vent')):.3f}",
                    str(row.get("rule_reason", "")) or "-",
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
    print(f"reason_missing_count={report['reason_missing_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
