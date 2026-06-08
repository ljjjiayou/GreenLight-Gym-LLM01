"""Consolidate post-guardrail policy shadow readiness signals."""

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
    decision_packet: Mapping[str, Any],
    counterfactual: Mapping[str, Any],
    provenance: Mapping[str, Any],
    family_ablation: Mapping[str, Any],
    counterexample_manifest: Mapping[str, Any],
    action_diff: Mapping[str, Any],
) -> dict[str, Any]:
    cleanup_blocked = bool(decision_packet.get("blocking_paths"))
    reason_missing = bool(provenance.get("unsafe_for_promotion_due_to_missing_provenance"))
    hard_regression_family = bool(family_ablation.get("hard_safety_regression_present"))
    action_diff_ok = (
        int(_num(action_diff.get("trace_count"))) > 0
        and int(_num(action_diff.get("action_diff_steps"))) == 0
        and int(_num(action_diff.get("missing_step_count"))) == 0
    )
    if cleanup_blocked:
        next_action = "mainline_diff_cleanup_required"
    elif reason_missing or bool(counterfactual.get("vent_floor_likely_sufficient")):
        next_action = "post_guardrail_policy_shadow_design"
    elif str(family_ablation.get("candidate_family_ablation_conclusion", "")) == "candidate_space_redesign_shadow":
        next_action = "candidate_space_redesign_shadow"
    else:
        next_action = "response_model_recalibration"
    return {
        "schema_version": "post_guardrail_policy_shadow_status_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Candidate Pool", "Response Estimate"],
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
            "post_guardrail_rule_provenance_complete": not reason_missing,
            "counterfactual_supports_vent_floor_shadow_design": bool(counterfactual.get("vent_floor_likely_sufficient")),
            "candidate_family_has_hard_safety_regression": hard_regression_family,
            "counterexample_manifest_available": bool(counterexample_manifest.get("windows")),
        },
        "mainline_diff": {
            "blocking_paths": list(decision_packet.get("blocking_paths", [])),
            "decision_counts": dict(decision_packet.get("decision_counts", {})),
        },
        "vent_counterfactual": {
            "conclusion_counts": dict(counterfactual.get("conclusion_counts", {})),
            "vent_floor_likely_sufficient": bool(counterfactual.get("vent_floor_likely_sufficient")),
            "proxy_warning_not_actionable": bool(counterfactual.get("proxy_warning_not_actionable")),
        },
        "rule_provenance": {
            "reason_missing_count": int(_num(provenance.get("reason_missing_count"))),
            "unsafe_for_promotion_due_to_missing_provenance": reason_missing,
            "rule_counts": dict(provenance.get("rule_counts", {})),
        },
        "candidate_family_ablation": {
            "conclusion": family_ablation.get("candidate_family_ablation_conclusion", ""),
            "useful_shadow_families": list(family_ablation.get("useful_shadow_families", [])),
            "hard_safety_regression_present": hard_regression_family,
        },
        "counterexample_suite_manifest": {
            "suite_name": counterexample_manifest.get("suite_name", ""),
            "regime_counts": dict(counterexample_manifest.get("regime_counts", {})),
        },
        "action_diff": {
            "trace_count": int(_num(action_diff.get("trace_count"))),
            "action_diff_steps": int(_num(action_diff.get("action_diff_steps"))),
            "missing_step_count": int(_num(action_diff.get("missing_step_count"))),
            "max_abs_delta": _num(action_diff.get("max_abs_delta")),
        },
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Policy Shadow Status",
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
        ("Mainline Diff", "mainline_diff"),
        ("Vent Counterfactual", "vent_counterfactual"),
        ("Rule Provenance", "rule_provenance"),
        ("Candidate Family Ablation", "candidate_family_ablation"),
        ("Counterexample Suite Manifest", "counterexample_suite_manifest"),
        ("Action Diff", "action_diff"),
    ):
        lines.extend(["", f"## {title}", "", "```json"])
        lines.append(json.dumps(report.get(key, {}), ensure_ascii=False, indent=2, sort_keys=True))
        lines.append("```")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-json", required=True)
    parser.add_argument("--counterfactual-json", required=True)
    parser.add_argument("--provenance-json", required=True)
    parser.add_argument("--family-json", required=True)
    parser.add_argument("--counterexample-json", required=True)
    parser.add_argument("--action-diff-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        decision_packet=_load(args.decision_json),
        counterfactual=_load(args.counterfactual_json),
        provenance=_load(args.provenance_json),
        family_ablation=_load(args.family_json),
        counterexample_manifest=_load(args.counterexample_json),
        action_diff=_load(args.action_diff_json),
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
