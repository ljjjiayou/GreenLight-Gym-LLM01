"""Consolidate mainline cleanup and post-guardrail diagnostic readiness."""

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


def build_report(
    *,
    cleanup_plan: Mapping[str, Any],
    rewrite_audit: Mapping[str, Any],
    taxonomy_report: Mapping[str, Any],
    inventory_report: Mapping[str, Any],
    action_diff_report: Mapping[str, Any],
) -> dict[str, Any]:
    cleanup_blocked = bool(cleanup_plan.get("blocking_paths"))
    harmful = int(_num(rewrite_audit.get("harmful_override_count")))
    rewrite_layers = dict(rewrite_audit.get("root_cause_layer_counts", {}))
    post_guardrail_destruction = int(_num(rewrite_layers.get("post_guardrail_destruction")))
    taxonomy_counts = dict(taxonomy_report.get("taxonomy_counts", {}))
    candidate_failures = int(
        _num(taxonomy_counts.get("candidate_space_insufficiency"))
        + _num(taxonomy_counts.get("dry_relief_lost"))
        + _num(taxonomy_counts.get("no_safe_repair_found"))
    )
    action_diff_ok = bool(
        int(_num(action_diff_report.get("trace_count"))) > 0
        and int(_num(action_diff_report.get("action_diff_steps"))) == 0
        and int(_num(action_diff_report.get("missing_step_count"))) == 0
    )
    if cleanup_blocked:
        next_action = "mainline_diff_cleanup_required"
    elif post_guardrail_destruction or harmful:
        next_action = "post_guardrail_policy_shadow_design"
    elif candidate_failures:
        next_action = "candidate_space_redesign_shadow"
    else:
        next_action = "response_model_recalibration"
    return {
        "schema_version": "mainline_and_post_guardrail_status_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Candidate Pool"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only / audit-only",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "rejected_presets_remain_rejected": True,
        "next_action": next_action,
        "gates": {
            "mainline_diff_cleanup_complete": not cleanup_blocked,
            "metadata_only_action_diff_zero": action_diff_ok,
            "post_guardrail_harmful_override_resolved": harmful == 0,
            "candidate_failure_taxonomy_stable": bool(taxonomy_counts),
            "counterexample_trace_inventory_available": bool(inventory_report.get("regime_available")),
        },
        "mainline_cleanup": {
            "blocking_paths": list(cleanup_plan.get("blocking_paths", [])),
            "cleanup_action_counts": dict(cleanup_plan.get("cleanup_action_counts", {})),
            "boundary_summary": dict(cleanup_plan.get("boundary_summary", {})),
        },
        "post_guardrail": {
            "harmful_override_count": harmful,
            "root_cause_layer_counts": rewrite_layers,
            "source_root_cause_summary": dict(rewrite_audit.get("source_root_cause_summary", {})),
        },
        "candidate_failure_taxonomy": {
            "taxonomy_counts": taxonomy_counts,
            "next_action": taxonomy_report.get("next_action", ""),
        },
        "regime_inventory": {
            "regime_available": dict(inventory_report.get("regime_available", {})),
            "regime_step_counts": dict(inventory_report.get("regime_step_counts", {})),
        },
        "action_diff": {
            "trace_count": int(_num(action_diff_report.get("trace_count"))),
            "action_diff_steps": int(_num(action_diff_report.get("action_diff_steps"))),
            "missing_step_count": int(_num(action_diff_report.get("missing_step_count"))),
            "max_abs_delta": _num(action_diff_report.get("max_abs_delta")),
        },
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Mainline and Post-Guardrail Status",
        "",
        "## Mainline Alignment",
        "",
        "- Mode: shadow-only / audit-only",
        "- Default `llm_rspc_v2` changed: false",
        "- Reopens rejected preset: false",
        "",
        "## Decision",
        "",
        f"- Controlled replay allowed: {report.get('controlled_replay_allowed', False)}",
        f"- Performance claim allowed: {report.get('performance_claim_allowed', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Gates",
        "",
        "| gate | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("gates", {})).items():
        lines.append(f"| {key} | {value} |")
    for title, key in (
        ("Mainline Cleanup", "mainline_cleanup"),
        ("Post-Guardrail", "post_guardrail"),
        ("Candidate Failure Taxonomy", "candidate_failure_taxonomy"),
        ("Regime Inventory", "regime_inventory"),
        ("Action Diff", "action_diff"),
    ):
        lines.extend(["", f"## {title}", "", "```json"])
        lines.append(json.dumps(report.get(key, {}), ensure_ascii=False, indent=2, sort_keys=True))
        lines.append("```")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup-json", required=True)
    parser.add_argument("--rewrite-json", required=True)
    parser.add_argument("--taxonomy-json", required=True)
    parser.add_argument("--inventory-json", required=True)
    parser.add_argument("--action-diff-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        cleanup_plan=_load(args.cleanup_json),
        rewrite_audit=_load(args.rewrite_json),
        taxonomy_report=_load(args.taxonomy_json),
        inventory_report=_load(args.inventory_json),
        action_diff_report=_load(args.action_diff_json),
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
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
