"""Create a read-only cleanup plan for mainline-sensitive diffs.

This script does not revert, stash, or edit files. It turns the boundary audit
into an explicit review package so the user can decide which changes to keep,
isolate, or clean before any controlled replay discussion.
"""

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

from gl_gym.experiments.mainline_diff_boundary_audit import build_report as build_boundary_report


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _recommendation(item: Mapping[str, Any]) -> dict[str, Any]:
    path = str(item.get("path", ""))
    category = str(item.get("category", ""))
    if category == "mainline_blocking":
        return {
            "cleanup_action": "restore_or_explicitly_authorize",
            "blocks_controlled_replay": True,
            "required_evidence": [
                "user explicitly authorizes keeping this diff, or file is restored before promotion evidence is used",
            ],
        }
    if category == "mainline_core":
        return {
            "cleanup_action": "authorize_mainline_core_before_promotion_evidence",
            "blocks_controlled_replay": True,
            "required_evidence": [
                "user confirms this mainline document belongs to the project authority set",
                "mainline revision log records evidence-backed changes only",
            ],
        }
    if category == "controller_sensitive":
        return {
            "cleanup_action": "isolate_or_prove_default_controller_invariance",
            "blocks_controlled_replay": True,
            "required_evidence": [
                "default llm_rspc_v2 action-diff remains zero against trusted prior trace",
                "no rejected preset is activated by default",
                "no controlled replay or alias path is introduced",
            ],
        }
    if category == "evaluation_sensitive":
        return {
            "cleanup_action": "isolate_protocol_change_or_confirm_metadata_only",
            "blocks_controlled_replay": True,
            "required_evidence": [
                "frozen replay protocol changes are documented",
                "cache hit/runtime/action-diff semantics are unchanged or explicitly versioned",
                "reports generated with modified protocol are not used as promotion evidence until revalidated",
            ],
        }
    if category == "shadow_audit_or_test":
        return {
            "cleanup_action": "keep_as_shadow_audit_candidate",
            "blocks_controlled_replay": False,
            "required_evidence": ["py_compile and targeted tests pass"],
        }
    if category == "shadow_audit":
        return {
            "cleanup_action": "keep_as_shadow_audit_candidate",
            "blocks_controlled_replay": False,
            "required_evidence": ["py_compile and targeted tests pass; do not use as promotion evidence"],
        }
    if category == "test_only":
        return {
            "cleanup_action": "keep_as_test_only_candidate",
            "blocks_controlled_replay": False,
            "required_evidence": ["targeted tests pass; test changes do not redefine promotion gate semantics"],
        }
    if category == "report_only":
        return {
            "cleanup_action": "keep_as_report_only_artifact",
            "blocks_controlled_replay": False,
            "required_evidence": ["confirm artifact is evidence/reporting only"],
        }
    if category == "codex_skill":
        return {
            "cleanup_action": "keep_as_codex_skill_tooling",
            "blocks_controlled_replay": False,
            "required_evidence": ["greenhouse skill sync check passes"],
        }
    if category == "deprecated_or_hold":
        return {
            "cleanup_action": "hold_as_shadow_or_historical_entrypoint",
            "blocks_controlled_replay": False,
            "required_evidence": ["do not run for promotion unless re-authorized by mainline plan"],
        }
    return {
        "cleanup_action": "review_scope_before_commit",
        "blocks_controlled_replay": False,
        "required_evidence": ["confirm file belongs to this audit/reporting change set"],
    }


def build_report(boundary_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    boundary = dict(boundary_report or build_boundary_report())
    items: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    blockers: list[str] = []
    for raw in boundary.get("changed_files", []) or []:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        rec = _recommendation(item)
        item["cleanup_recommendation"] = rec
        items.append(item)
        counts[str(rec.get("cleanup_action", ""))] += 1
        if bool(rec.get("blocks_controlled_replay")):
            blockers.append(str(item.get("path", "")))
    next_action = (
        "mainline_diff_cleanup_required"
        if blockers
        else "shadow_audit_can_continue_without_promotion_evidence"
    )
    return {
        "schema_version": "mainline_diff_cleanup_plan_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only cleanup plan",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "next_action": next_action,
        "blocking_paths": blockers,
        "cleanup_action_counts": dict(sorted(counts.items())),
        "boundary_summary": {
            "controlled_replay_blocked": boundary.get("controlled_replay_blocked", False),
            "controlled_replay_block_reasons": list(boundary.get("controlled_replay_block_reasons", [])),
            "expert_distillation_status": boundary.get("expert_distillation_status", ""),
            "controller_diff_status": boundary.get("controller_diff_status", ""),
            "evaluation_diff_status": boundary.get("evaluation_diff_status", ""),
        },
        "cleanup_items": items,
        "notes": [
            "This is a plan/report only; no revert or stash is performed.",
            "expert_distillation.py cleanup requires explicit user approval.",
            "controller/evaluation diffs can support shadow diagnostics but block controlled replay evidence until isolated or revalidated.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Mainline Diff Cleanup Plan",
        "",
        "## Mainline Alignment",
        "",
        "- Mode: read-only cleanup plan",
        "- Default `llm_rspc_v2` changed by this report: false",
        "- Reopens rejected preset: false",
        "",
        "## Decision",
        "",
        f"- Controlled replay allowed: {report.get('controlled_replay_allowed', False)}",
        f"- Performance claim allowed: {report.get('performance_claim_allowed', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        f"- Blocking paths: {json.dumps(report.get('blocking_paths', []), ensure_ascii=False)}",
        "",
        "## Cleanup Action Counts",
        "",
        "| cleanup_action | count |",
        "| --- | ---: |",
    ]
    for action, count in dict(report.get("cleanup_action_counts", {})).items():
        lines.append(f"| {action} | {count} |")
    lines.extend(
        [
            "",
            "## Cleanup Items",
            "",
            "| path | category | cleanup_action | blocks_controlled_replay |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in report.get("cleanup_items", []) or []:
        if not isinstance(item, Mapping):
            continue
        rec = item.get("cleanup_recommendation", {}) if isinstance(item.get("cleanup_recommendation"), Mapping) else {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.get('path', '')}`",
                    str(item.get("category", "")),
                    str(rec.get("cleanup_action", "")),
                    str(rec.get("blocks_controlled_replay", False)),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    boundary = _load(args.boundary_json) if args.boundary_json else None
    report = build_report(boundary)
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
