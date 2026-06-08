"""Summarize mainline-sensitive diff authorization status.

This report is read-only. It does not restore, stash, stage, or edit files.
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


def _status_for_row(row: Mapping[str, Any]) -> str:
    path = str(row.get("path", ""))
    auth = str(row.get("authorization_required", ""))
    if path == "gl_gym/agent/expert_distillation.py":
        return "not_authorized_for_promotion_evidence"
    if auth == "restore_or_user_authorization_required":
        return "restore_or_user_authorization_required"
    if auth == "mainline_core_authorized":
        return "mainline_core_authorized"
    if auth == "mainline_core_authorization_required":
        return "mainline_core_authorization_required"
    if auth == "controller_invariance_required":
        return "controller_invariance_required"
    if auth == "evaluation_protocol_isolation_required":
        return "evaluation_protocol_isolation_required"
    return "shadow_audit_review_required"


def build_report(authorization_packet: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    blocking_paths: list[str] = []
    status_counts: dict[str, int] = {}
    for item in authorization_packet.get("rows", []) or []:
        if not isinstance(item, Mapping):
            continue
        status = _status_for_row(item)
        blocks = bool(item.get("blocks_controlled_replay"))
        if blocks:
            blocking_paths.append(str(item.get("path", "")))
        status_counts[status] = status_counts.get(status, 0) + 1
        rows.append(
            {
                "path": item.get("path", ""),
                "category": item.get("category", ""),
                "git_status": item.get("git_status", ""),
                "authorization_status": status,
                "blocks_controlled_replay": blocks,
                "automated_cleanup_performed": False,
                "promotion_risk": item.get("promotion_risk", ""),
                "requires_restore": bool(item.get("requires_restore", False)),
                "requires_user_authorization": bool(item.get("requires_user_authorization", False)),
                "requires_metadata_replay": bool(item.get("requires_metadata_replay", False)),
                "requires_protocol_isolation": bool(item.get("requires_protocol_isolation", False)),
                "allowed_next_action": item.get("allowed_next_action", ""),
                "required_evidence": item.get("required_evidence", []),
            }
        )
    return {
        "schema_version": "mainline_diff_authorization_status_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only authorization status",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "authorization_complete": not bool(blocking_paths),
        "automated_cleanup_performed": False,
        "blocking_paths": blocking_paths,
        "status_counts": dict(sorted(status_counts.items())),
        "next_action": (
            "mainline_diff_authorization_required"
            if blocking_paths
            else "runtime_provenance_metadata_replay_required"
        ),
        "rows": rows,
        "notes": [
            "expert_distillation.py remains not authorized for promotion evidence until restored or explicitly approved.",
            "controller-sensitive diffs require action-diff and invariance evidence before promotion evidence can be trusted.",
            "evaluation-sensitive diffs require protocol isolation before controlled replay evidence can be trusted.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Mainline Diff Authorization Status",
        "",
        "- Mode: read-only authorization status",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Authorization complete: {report.get('authorization_complete', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Status Counts",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status, count in dict(report.get("status_counts", {})).items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Blocking Paths", ""])
    for path in report.get("blocking_paths", []) or []:
        lines.append(f"- `{path}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.authorization_json))
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
