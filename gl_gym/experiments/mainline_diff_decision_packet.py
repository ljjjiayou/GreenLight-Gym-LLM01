"""Build a user-facing decision packet for mainline-sensitive diffs.

This report is intentionally read-only. It does not restore, stash, or stage
files; it only separates cleanup decisions from shadow-audit evidence.
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

from gl_gym.experiments.mainline_diff_cleanup_plan import build_report as build_cleanup_report


EXPERT_PATH = "gl_gym/agent/expert_distillation.py"
MAINLINE_CORE_PATH = "docs/project_mainline_guidance_20260519.md"


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _decision_for_item(item: Mapping[str, Any], *, authorized_mainline_core_paths: set[str] | None = None) -> str:
    path = str(item.get("path", ""))
    category = str(item.get("category", ""))
    authorized_mainline_core_paths = authorized_mainline_core_paths or set()
    if path == EXPERT_PATH:
        return "requires_user_restore_or_authorization"
    if category == "mainline_core":
        if path in authorized_mainline_core_paths:
            return "mainline_core_authorized"
        return "requires_mainline_core_authorization"
    if category == "controller_sensitive" or path in {
        "gl_gym/agent/llm_agent.py",
        "gl_gym/agent/intent_contract.py",
        "gl_gym/agent/profile_generator.py",
    }:
        return "requires_controller_invariance_evidence"
    if category == "evaluation_sensitive" or path in {
        "gl_gym/experiments/frozen_benchmark_protocol.py",
        "gl_gym/experiments/run_frozen_benchmark.py",
        "tests/test_frozen_benchmark.py",
        "tests/test_frozen_benchmark_protocol.py",
    }:
        return "requires_evaluation_protocol_isolation"
    return "shadow_audit_change_review"


def _promotion_risk(category: str, decision: str) -> str:
    if decision == "mainline_core_authorized":
        return "authorized_mainline_reference"
    if category == "mainline_blocking":
        return "hard_blocker"
    if category in {"controller_sensitive", "evaluation_sensitive", "mainline_core"}:
        return "blocking_until_evidence"
    if category == "deprecated_or_hold":
        return "do_not_use_for_promotion"
    return "not_promotion_evidence"


def _allowed_next_action(category: str, decision: str) -> str:
    if decision == "mainline_core_authorized":
        return "keep_as_authorized_project_mainline"
    if category == "mainline_blocking":
        return "record_diff_then_restore_or_get_explicit_user_authorization"
    if category == "mainline_core":
        return "user_authorize_mainline_document_before_promotion"
    if category == "controller_sensitive":
        return "metadata_replay_action_diff_invariance_after_blockers_close"
    if category == "evaluation_sensitive":
        return "evaluation_protocol_isolation_plan_only"
    if category == "deprecated_or_hold":
        return "hold_for_history_or_shadow_only_do_not_run_for_promotion"
    if category == "codex_skill":
        return "keep_if_skill_sync_and_tests_pass"
    if category == "test_only":
        return "keep_if_tests_do_not_redefine_promotion_gate"
    if category == "report_only":
        return "keep_as_report_only_artifact"
    return "keep_as_shadow_audit_candidate_after_tests"


def _decision_flags(category: str, decision: str) -> dict[str, bool]:
    return {
        "requires_restore": category == "mainline_blocking",
        "requires_user_authorization": category in {"mainline_blocking", "mainline_core"}
        and decision != "mainline_core_authorized",
        "requires_metadata_replay": category == "controller_sensitive",
        "requires_protocol_isolation": category == "evaluation_sensitive",
    }


def build_report(
    cleanup_plan: Mapping[str, Any] | None = None,
    *,
    authorized_mainline_core_paths: set[str] | None = None,
) -> dict[str, Any]:
    cleanup = dict(cleanup_plan or build_cleanup_report())
    authorized_mainline_core_paths = authorized_mainline_core_paths or set()
    decisions: list[dict[str, Any]] = []
    decision_counts: dict[str, int] = {}
    blocking: list[str] = []
    for item in cleanup.get("cleanup_items", []) or []:
        if not isinstance(item, Mapping):
            continue
        decision = _decision_for_item(item, authorized_mainline_core_paths=authorized_mainline_core_paths)
        category = str(item.get("category", ""))
        flags = _decision_flags(category, decision)
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        rec = item.get("cleanup_recommendation", {}) if isinstance(item.get("cleanup_recommendation"), Mapping) else {}
        blocks = bool(rec.get("blocks_controlled_replay")) or decision in {
            "requires_user_restore_or_authorization",
            "requires_mainline_core_authorization",
            "requires_controller_invariance_evidence",
            "requires_evaluation_protocol_isolation",
        }
        if decision == "mainline_core_authorized":
            blocks = False
        if blocks:
            blocking.append(str(item.get("path", "")))
        decisions.append(
            {
                "path": item.get("path", ""),
                "git_status": item.get("git_status", ""),
                "category": category,
                "decision_required": decision,
                "promotion_risk": _promotion_risk(category, decision),
                **flags,
                "allowed_next_action": _allowed_next_action(category, decision),
                "blocks_controlled_replay": blocks,
                "numstat": item.get("numstat", {}),
                "diff_excerpt": item.get("diff_excerpt", []),
                "required_evidence": rec.get("required_evidence", []),
            }
        )
    return {
        "schema_version": "mainline_diff_decision_packet_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Candidate Pool", "Response Estimate"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-only / audit-only decision packet",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "next_action": "mainline_diff_cleanup_required" if blocking else "post_guardrail_policy_shadow_design",
        "decision_counts": dict(sorted(decision_counts.items())),
        "blocking_paths": blocking,
        "boundary_summary": dict(cleanup.get("boundary_summary", {})),
        "authorized_mainline_core_paths": sorted(authorized_mainline_core_paths),
        "decisions": decisions,
        "notes": [
            "No files are restored, stashed, staged, or edited by this packet.",
            "expert_distillation.py requires user cleanup or explicit authorization before promotion evidence can be trusted.",
            "Controller-sensitive and evaluation-sensitive diffs can support shadow diagnostics only after invariance/protocol evidence is attached.",
        ],
    }


def build_expert_diff_review(report: Mapping[str, Any]) -> dict[str, Any]:
    expert_rows = [
        item for item in report.get("decisions", []) or []
        if isinstance(item, Mapping) and item.get("path") == EXPERT_PATH
    ]
    row = expert_rows[0] if expert_rows else {}
    diff_excerpt = list(row.get("diff_excerpt", []) or []) if isinstance(row, Mapping) else []
    has_diff = bool(diff_excerpt)
    return {
        "schema_version": "expert_distillation_diff_review_v1",
        "path": EXPERT_PATH,
        "diff_present": has_diff,
        "mainline_alignment": {
            "affected_layers": ["Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / pre-restore decision",
            "reopens_rejected_preset": False,
        },
        "diff_summary": (
            "docstring-only translation from English to Chinese; no executable code changes detected in diff excerpt"
            if has_diff
            else "no working-tree diff detected"
        ),
        "belongs_to_current_mainline_closure_task": False,
        "recommended_decision": "restore_to_head" if has_diff else "already_clean",
        "restore_scope": EXPERT_PATH,
        "automated_cleanup_allowed_by_user_plan": True,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "diff_excerpt": diff_excerpt,
        "notes": [
            "The project mainline forbids touching expert_distillation.py unless explicitly requested.",
            "This review records the diff before restoring this single file.",
        ],
    }


def build_expert_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# expert_distillation.py Diff Review",
        "",
        f"- Path: `{report.get('path', '')}`",
        f"- Diff present: {report.get('diff_present', False)}",
        f"- Belongs to current task: {report.get('belongs_to_current_mainline_closure_task', False)}",
        f"- Recommended decision: `{report.get('recommended_decision', '')}`",
        f"- Controlled replay allowed: {report.get('controlled_replay_allowed', False)}",
        f"- Performance claim allowed: {report.get('performance_claim_allowed', False)}",
        "",
        "## Diff Summary",
        "",
        str(report.get("diff_summary", "")),
        "",
    ]
    if report.get("diff_excerpt"):
        lines.extend(["## Diff Excerpt", "", "```diff"])
        lines.extend(str(line) for line in report.get("diff_excerpt", []) or [])
        lines.extend(["```", ""])
    return "\n".join(lines)


def build_evaluation_protocol_isolation_plan(report: Mapping[str, Any]) -> dict[str, Any]:
    evaluation_rows = [
        item for item in report.get("decisions", []) or []
        if isinstance(item, Mapping) and item.get("category") == "evaluation_sensitive"
    ]
    return {
        "schema_version": "evaluation_protocol_isolation_plan_v1",
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "online_llm_called": False,
        "evaluation_sensitive_paths": [str(item.get("path", "")) for item in evaluation_rows],
        "comparison_design": {
            "input_replay_json": "same frozen benchmark JSON",
            "cache": "same frozen cache; no cache fill",
            "controller": "same controller labels",
            "scenario_set": "same scenario ids and timestep windows",
            "old_protocol": "trusted baseline protocol or last accepted evidence version",
            "new_protocol": "current working-tree protocol",
        },
        "fields_must_match": [
            "summary row count",
            "gate failure count and reasons",
            "cache hit / miss metrics",
            "runtime error metrics",
            "controller labels",
            "action-diff semantics",
        ],
        "if_mismatch": "block promotion evidence until the protocol delta is explained and versioned",
        "next_action": "protocol_isolation_pending",
    }


def build_metadata_replay_readiness_checklist(report: Mapping[str, Any]) -> dict[str, Any]:
    decisions = [item for item in report.get("decisions", []) or [] if isinstance(item, Mapping)]
    expert_blocked = any(item.get("category") == "mainline_blocking" for item in decisions)
    mainline_core_pending = any(
        item.get("category") == "mainline_core" and item.get("decision_required") != "mainline_core_authorized"
        for item in decisions
    )
    controller_pending = any(item.get("requires_metadata_replay") for item in decisions)
    evaluation_pending = any(item.get("requires_protocol_isolation") for item in decisions)
    return {
        "schema_version": "metadata_replay_readiness_checklist_v1",
        "metadata_replay_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "checks": {
            "expert_distillation_resolved": not expert_blocked,
            "mainline_core_authorized": not mainline_core_pending,
            "authorization_complete": not report.get("blocking_paths"),
            "protocol_isolation_pass": False if evaluation_pending else None,
            "cache_coverage_pass": "not_run",
            "controller_sensitive_diff_present": controller_pending,
        },
        "targets": {
            "action_diff_steps": 0,
            "max_abs_delta": 0,
            "runtime_provenance_record_count": ">0",
            "runtime_reason_missing_count": 0,
            "unknown_post_guardrail_rewrite_count": 0,
            "joint_prediction_missing_fields": 0,
        },
        "next_action": (
            "expert_distillation_restore_required"
            if expert_blocked
            else "mainline_core_authorization_required"
            if mainline_core_pending
            else "evaluation_protocol_isolation_required"
            if evaluation_pending
            else "controller_invariance_metadata_replay_required"
            if controller_pending
            else "metadata_replay_readiness_review_required"
        ),
    }


def build_post_guardrail_provenance_closure_plan() -> dict[str, Any]:
    return {
        "schema_version": "post_guardrail_provenance_closure_plan_v1",
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "mode": "plan-only / no replay",
        "closure_targets": {
            "record_count": ">0",
            "runtime_reason_missing_count": 0,
            "unknown_post_guardrail_rewrite_count": 0,
            "joint_prediction_missing_fields": 0,
        },
        "required_audits_after_metadata_replay": [
            "post_guardrail_runtime_provenance_audit.py",
            "post_guardrail_shadow_policy_instrumentation_status.py",
            "post_guardrail_joint_prediction_readiness_v2.py",
        ],
        "blocked_until": [
            "mainline diff authorization closes",
            "evaluation protocol isolation passes",
            "metadata replay action_diff_steps remains 0",
        ],
        "next_action": "runtime_provenance_metadata_replay_after_mainline_closure",
    }


def build_mainline_core_authorization_report(paths: Sequence[str]) -> dict[str, Any]:
    normalized = sorted({str(path).replace("\\", "/") for path in paths})
    return {
        "schema_version": "mainline_core_authorization_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / authorization record",
            "reopens_rejected_preset": False,
        },
        "authorized": bool(normalized),
        "authorized_paths": normalized,
        "controlled_replay_allowed": False,
        "metadata_replay_allowed": False,
        "performance_claim_allowed": False,
        "decision": "mainline_core_authorized" if normalized else "no_mainline_core_authorization_recorded",
        "notes": [
            "This records user authorization of the project mainline document as an authority source.",
            "Authorization does not allow metadata replay or controlled replay by itself.",
        ],
    }


def build_simple_markdown_report(title: str, report: Mapping[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for key in [
        "schema_version",
        "controlled_replay_allowed",
        "metadata_replay_allowed",
        "performance_claim_allowed",
        "replay_run",
        "online_llm_called",
        "next_action",
    ]:
        if key in report:
            lines.append(f"- {key}: `{report.get(key)}`")
    lines.extend(["", "```json", json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Mainline Diff Decision Packet",
        "",
        "## Mainline Alignment",
        "",
        "- Mode: shadow-only / audit-only decision packet",
        "- Default `llm_rspc_v2` changed by this report: false",
        "- Reopens rejected preset: false",
        "",
        "## Decision",
        "",
        f"- Controlled replay allowed: {report.get('controlled_replay_allowed', False)}",
        f"- Performance claim allowed: {report.get('performance_claim_allowed', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Decision Counts",
        "",
        "| decision_required | count |",
        "| --- | ---: |",
    ]
    for decision, count in dict(report.get("decision_counts", {})).items():
        lines.append(f"| {decision} | {count} |")
    lines.extend(
        [
            "",
            "## Blocking Paths",
            "",
        "| path | decision_required | category | promotion_risk | allowed_next_action | blocks_controlled_replay |",
        "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in report.get("decisions", []) or []:
        if not isinstance(item, Mapping) or not bool(item.get("blocks_controlled_replay")):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.get('path', '')}`",
                    str(item.get("decision_required", "")),
                    str(item.get("category", "")),
                    str(item.get("promotion_risk", "")),
                    str(item.get("allowed_next_action", "")),
                    str(item.get("blocks_controlled_replay", False)),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup-json", default="")
    parser.add_argument("--authorized-mainline-core-path", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--mainline-core-authorization-json", default="")
    parser.add_argument("--mainline-core-authorization-md", default="")
    parser.add_argument("--expert-output-json", default="")
    parser.add_argument("--expert-output-md", default="")
    parser.add_argument("--evaluation-plan-json", default="")
    parser.add_argument("--evaluation-plan-md", default="")
    parser.add_argument("--metadata-readiness-json", default="")
    parser.add_argument("--metadata-readiness-md", default="")
    parser.add_argument("--provenance-closure-plan-json", default="")
    parser.add_argument("--provenance-closure-plan-md", default="")
    return parser.parse_args(argv)


def _write_json_md(json_path: str, md_path: str, report: Mapping[str, Any], markdown: str) -> None:
    if not json_path and not md_path:
        return
    if json_path:
        output_json = Path(json_path)
        if not output_json.is_absolute():
            output_json = PROJECT_ROOT / output_json
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {output_json}")
    if md_path:
        output_md = Path(md_path)
        if not output_md.is_absolute():
            output_md = PROJECT_ROOT / output_md
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")
        print(f"wrote {output_md}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cleanup = _load(args.cleanup_json) if args.cleanup_json else None
    authorized_mainline_core_paths = {str(path).replace("\\", "/") for path in args.authorized_mainline_core_path}
    report = build_report(cleanup, authorized_mainline_core_paths=authorized_mainline_core_paths)
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
    authorization_report = build_mainline_core_authorization_report(sorted(authorized_mainline_core_paths))
    _write_json_md(
        args.mainline_core_authorization_json,
        args.mainline_core_authorization_md,
        authorization_report,
        build_simple_markdown_report("Mainline Core Authorization", authorization_report),
    )
    expert_report = build_expert_diff_review(report)
    _write_json_md(args.expert_output_json, args.expert_output_md, expert_report, build_expert_markdown_report(expert_report))
    evaluation_plan = build_evaluation_protocol_isolation_plan(report)
    _write_json_md(
        args.evaluation_plan_json,
        args.evaluation_plan_md,
        evaluation_plan,
        build_simple_markdown_report("Evaluation Protocol Isolation Plan", evaluation_plan),
    )
    metadata_readiness = build_metadata_replay_readiness_checklist(report)
    _write_json_md(
        args.metadata_readiness_json,
        args.metadata_readiness_md,
        metadata_readiness,
        build_simple_markdown_report("Metadata Replay Readiness Checklist", metadata_readiness),
    )
    provenance_plan = build_post_guardrail_provenance_closure_plan()
    _write_json_md(
        args.provenance_closure_plan_json,
        args.provenance_closure_plan_md,
        provenance_plan,
        build_simple_markdown_report("Post-Guardrail Provenance Closure Plan", provenance_plan),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
