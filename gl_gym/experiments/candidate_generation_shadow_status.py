"""Consolidate candidate-generation shadow readiness evidence.

The status is intentionally conservative.  It never allows controlled replay by
itself; it only reports the next diagnostic direction after combining mainline
diff boundaries, post-guardrail root cause, action-diff, and candidate shadow
suite evidence.
"""

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
    mainline_diff_report: Mapping[str, Any],
    post_guardrail_report: Mapping[str, Any],
    candidate_suite_report: Mapping[str, Any],
    action_diff_report: Mapping[str, Any],
    proxy_status_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    proxy_status_report = proxy_status_report or {}
    mainline_blocked = bool(mainline_diff_report.get("controlled_replay_blocked"))
    action_diff_ok = bool(
        int(_num(action_diff_report.get("trace_count"))) > 0
        and int(_num(action_diff_report.get("action_diff_steps"))) == 0
        and int(_num(action_diff_report.get("missing_step_count"))) == 0
    )
    harmful_count = int(_num(post_guardrail_report.get("harmful_override_count")))
    harmful_v21 = int(_num(post_guardrail_report.get("harmful_override_preventable_by_proxy_v2_1_count")))
    harmful_unexplained = bool(harmful_count > harmful_v21)
    candidate_conclusion = str(candidate_suite_report.get("candidate_generation_shadow_conclusion", "") or "")
    blocker_count = int(_num(candidate_suite_report.get("pure_hot_dry_false_positive_blocker_count")))
    safe_repair_count = int(_num(candidate_suite_report.get("post_guardrail_safe_useful_repair_count")))
    gap_steps = int(_num(candidate_suite_report.get("candidate_pool_gap_steps")))

    if mainline_blocked:
        next_action = "mainline_diff_cleanup_required"
    elif harmful_unexplained:
        next_action = "post_guardrail_policy_shadow_design"
    elif candidate_conclusion == "candidate_generation_shadow_expand" and gap_steps > 0 and safe_repair_count >= gap_steps:
        next_action = "candidate_generation_shadow_expand"
    else:
        next_action = "response_model_recalibration"

    return {
        "schema_version": "candidate_generation_shadow_status_v1",
        "mainline_alignment": {
            "affected_layers": ["Candidate Pool", "Safety Boundary", "Evaluation"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only / metadata-only",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "action_diff_required": True,
        "rejected_presets_remain_rejected": True,
        "next_action": next_action,
        "gates": {
            "mainline_diff_clean_enough_for_controlled_replay": not mainline_blocked,
            "metadata_only_action_diff_zero": action_diff_ok,
            "post_guardrail_harmful_override_fully_explained": not harmful_unexplained,
            "candidate_generation_shadow_has_safe_repair": bool(gap_steps > 0 and safe_repair_count >= gap_steps),
            "pure_hot_dry_false_positive_blocker_absent": bool(blocker_count == 0),
            "proxy_status_still_blocks_controlled_replay": bool(
                proxy_status_report.get("controlled_replay_allowed") is False
            ),
        },
        "mainline_diff": {
            "controlled_replay_blocked": mainline_blocked,
            "block_reasons": list(mainline_diff_report.get("controlled_replay_block_reasons", [])),
            "expert_distillation_status": mainline_diff_report.get("expert_distillation_status", ""),
            "evaluation_diff_status": mainline_diff_report.get("evaluation_diff_status", ""),
            "controller_diff_status": mainline_diff_report.get("controller_diff_status", ""),
        },
        "post_guardrail": {
            "harmful_override_count": harmful_count,
            "harmful_override_preventable_by_proxy_v2_1_count": harmful_v21,
            "harmful_override_unexplained_count": max(0, harmful_count - harmful_v21),
            "root_cause_summary": dict(post_guardrail_report.get("root_cause_summary", {})),
            "fine_grained_root_cause_summary": dict(post_guardrail_report.get("fine_grained_root_cause_summary", {})),
            "harmful_fine_grained_root_cause_summary": dict(
                post_guardrail_report.get("harmful_fine_grained_root_cause_summary", {})
            ),
        },
        "candidate_generation_shadow": {
            "candidate_pool_gap_steps": gap_steps,
            "post_guardrail_safe_useful_repair_count": safe_repair_count,
            "pure_hot_dry_false_positive_blocker_count": blocker_count,
            "pure_hot_dry_false_positive_blocker_rate": candidate_suite_report.get(
                "pure_hot_dry_false_positive_blocker_rate"
            ),
            "conclusion": candidate_conclusion,
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
        "# Candidate Generation Shadow Status",
        "",
        "## Mainline Alignment",
        "",
        "- Mode: shadow-only / metadata-only",
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
    lines.extend(["", "## Mainline Diff", ""])
    lines.append("```json")
    lines.append(json.dumps(report.get("mainline_diff", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.extend(["", "## Post-Guardrail", ""])
    lines.append("```json")
    lines.append(json.dumps(report.get("post_guardrail", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.extend(["", "## Candidate Generation Shadow", ""])
    lines.append("```json")
    lines.append(json.dumps(report.get("candidate_generation_shadow", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mainline-diff-json", required=True)
    parser.add_argument("--post-guardrail-json", required=True)
    parser.add_argument("--candidate-suite-json", required=True)
    parser.add_argument("--action-diff-json", required=True)
    parser.add_argument("--proxy-status-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        mainline_diff_report=_load(args.mainline_diff_json),
        post_guardrail_report=_load(args.post_guardrail_json),
        candidate_suite_report=_load(args.candidate_suite_json),
        action_diff_report=_load(args.action_diff_json),
        proxy_status_report=_load(args.proxy_status_json) if args.proxy_status_json else None,
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
