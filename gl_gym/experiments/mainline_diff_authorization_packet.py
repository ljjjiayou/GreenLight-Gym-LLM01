"""Build an authorization packet for mainline-sensitive diffs.

The packet is read-only: it never restores, stashes, stages, or edits files.
It turns the existing decision packet into user-facing authorization decisions.
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

from gl_gym.experiments.mainline_diff_decision_packet import build_report as build_decision_report


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _authorization_for(decision: str) -> str:
    if decision == "requires_user_restore_or_authorization":
        return "restore_or_user_authorization_required"
    if decision == "mainline_core_authorized":
        return "mainline_core_authorized"
    if decision == "requires_mainline_core_authorization":
        return "mainline_core_authorization_required"
    if decision == "requires_controller_invariance_evidence":
        return "controller_invariance_required"
    if decision == "requires_evaluation_protocol_isolation":
        return "evaluation_protocol_isolation_required"
    return "review_as_shadow_audit_change"


def build_report(decision_packet: Mapping[str, Any] | None = None) -> dict[str, Any]:
    decision = dict(decision_packet or build_decision_report())
    rows: list[dict[str, Any]] = []
    auth_counts: dict[str, int] = {}
    blocking_paths: list[str] = []
    for item in decision.get("decisions", []) or []:
        if not isinstance(item, Mapping):
            continue
        auth = _authorization_for(str(item.get("decision_required", "")))
        auth_counts[auth] = auth_counts.get(auth, 0) + 1
        blocks = bool(item.get("blocks_controlled_replay"))
        if blocks:
            blocking_paths.append(str(item.get("path", "")))
        rows.append(
            {
                "path": item.get("path", ""),
                "category": item.get("category", ""),
                "git_status": item.get("git_status", ""),
                "authorization_required": auth,
                "blocks_controlled_replay": blocks,
                "risk_level": "blocking" if blocks else "review",
                "promotion_risk": item.get("promotion_risk", ""),
                "requires_restore": bool(item.get("requires_restore", False)),
                "requires_user_authorization": bool(item.get("requires_user_authorization", False)),
                "requires_metadata_replay": bool(item.get("requires_metadata_replay", False)),
                "requires_protocol_isolation": bool(item.get("requires_protocol_isolation", False)),
                "allowed_next_action": item.get("allowed_next_action", ""),
                "required_evidence": item.get("required_evidence", []),
                "diff_excerpt": item.get("diff_excerpt", []),
                "automated_cleanup_performed": False,
            }
        )
    return {
        "schema_version": "mainline_diff_authorization_packet_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only authorization packet",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "automated_cleanup_performed": False,
        "authorization_counts": dict(sorted(auth_counts.items())),
        "blocking_paths": blocking_paths,
        "next_action": "mainline_diff_authorization_required" if blocking_paths else "runtime_provenance_instrumentation_pending",
        "rows": rows,
        "notes": [
            "No restore, stash, stage, or edit is performed by this packet.",
            "expert_distillation.py remains blocking until restored or explicitly authorized by the user.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Mainline Diff Authorization Packet",
        "",
        "- Mode: read-only authorization packet",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Automated cleanup performed: {report.get('automated_cleanup_performed', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Authorization Counts",
        "",
        "| authorization | count |",
        "| --- | ---: |",
    ]
    for key, count in dict(report.get("authorization_counts", {})).items():
        lines.append(f"| {key} | {count} |")
    lines.extend(["", "## Blocking Rows", "", "| path | authorization | category |", "| --- | --- | --- |"])
    for row in report.get("rows", []) or []:
        if not isinstance(row, Mapping) or not bool(row.get("blocks_controlled_replay")):
            continue
        lines.append(
            f"| `{row.get('path', '')}` | {row.get('authorization_required', '')} | {row.get('category', '')} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.decision_json) if args.decision_json else None)
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
