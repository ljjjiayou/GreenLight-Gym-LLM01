"""Close or block mainline diff authorization decisions.

This audit is deliberately read-only. It does not restore, stash, stage, edit,
or replay anything. Its purpose is to say whether the current diff state is
clean enough to allow the next metadata replay step.
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

EXPERT_PATH = "gl_gym/agent/expert_distillation.py"
LLM_AGENT_PATH = "gl_gym/agent/llm_agent.py"
INTENT_PROFILE_PATHS = {
    "gl_gym/agent/intent_contract.py",
    "gl_gym/agent/profile_generator.py",
}
EVALUATION_SENSITIVE_PATHS = {
    "gl_gym/experiments/frozen_benchmark_protocol.py",
    "gl_gym/experiments/run_frozen_benchmark.py",
    "tests/test_frozen_benchmark.py",
    "tests/test_frozen_benchmark_protocol.py",
}


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


def _runtime_metadata_replay_required(status: Mapping[str, Any]) -> bool:
    runtime = status.get("runtime_provenance", {})
    if not isinstance(runtime, Mapping):
        return True
    return bool(runtime.get("metadata_replay_required", True))


def _closure_for_row(
    row: Mapping[str, Any],
    *,
    instrumentation_status: Mapping[str, Any],
    action_diff: Mapping[str, Any],
) -> dict[str, Any]:
    path = str(row.get("path", ""))
    auth_status = str(row.get("authorization_status", ""))
    blocks = bool(row.get("blocks_controlled_replay"))
    action_diff_zero = int(_num(action_diff.get("action_diff_steps"))) == 0
    runtime_replay_required = _runtime_metadata_replay_required(instrumentation_status)

    if path == EXPERT_PATH:
        closure = "not_closed_requires_restore_or_explicit_user_authorization"
        evidence = [
            "user must explicitly authorize keeping this diff, or the file must be restored before promotion evidence",
            "no automatic restore/stash/cleanup is performed by this audit",
        ]
    elif path == LLM_AGENT_PATH:
        closure = (
            "controller_invariance_pending_metadata_replay"
            if runtime_replay_required
            else (
                "closed"
                if action_diff_zero
                else "invariance_pending"
            )
        )
        evidence = [
            "metadata-only runtime provenance instrumentation must be followed by strict metadata replay",
            "default llm_rspc_v2 action_diff_steps must remain 0 on the current instrumented code path",
        ]
    elif auth_status == "mainline_core_authorization_required":
        closure = "mainline_core_authorization_pending"
        evidence = [
            "mainline authority documents must be explicitly accepted before promotion evidence can depend on them",
            "revision log must remain evidence-backed and not silently overwrite prior conclusions",
        ]
    elif auth_status == "mainline_core_authorized":
        closure = "closed"
        evidence = [
            "mainline authority document has an explicit authorization record",
            "authorization alone does not allow metadata replay or controlled replay",
        ]
    elif path in EVALUATION_SENSITIVE_PATHS:
        closure = "evaluation_protocol_isolation_pending"
        evidence = [
            "modified frozen benchmark protocol or tests cannot be used as promotion evidence until isolated or revalidated",
            "historical PASS summaries remain historical references only",
        ]
    elif path in INTENT_PROFILE_PATHS:
        closure = "shadow_only_path_pending_default_path_evidence"
        evidence = [
            "Intent/Profile files must be proven shadow-only and outside the default control path",
            "default llm_rspc_v2 action_diff evidence is required before using them in promotion evidence",
        ]
    elif blocks:
        closure = f"not_closed_{auth_status or 'blocking_diff'}"
        evidence = list(row.get("required_evidence", [])) or ["blocking diff requires explicit closure evidence"]
    else:
        closure = "closed"
        evidence = list(row.get("required_evidence", [])) or ["non-blocking audit/reporting change"]

    return {
        "path": path,
        "category": row.get("category", ""),
        "git_status": row.get("git_status", ""),
        "authorization_status": auth_status,
        "closure_status": closure,
        "closed": closure == "closed",
        "blocks_metadata_replay": closure != "closed",
        "blocks_controlled_replay": True if closure != "closed" and blocks else bool(row.get("blocks_controlled_replay")),
        "required_evidence": evidence,
        "promotion_risk": row.get("promotion_risk", ""),
        "allowed_next_action": row.get("allowed_next_action", ""),
        "automated_cleanup_performed": False,
    }


def _next_action(rows: Sequence[Mapping[str, Any]]) -> str:
    statuses = [str(row.get("closure_status", "")) for row in rows]
    if "not_closed_requires_restore_or_explicit_user_authorization" in statuses:
        return "expert_distillation_user_decision_required"
    if "mainline_core_authorization_pending" in statuses:
        return "mainline_core_authorization_required"
    if "evaluation_protocol_isolation_pending" in statuses:
        return "evaluation_protocol_isolation_required"
    if "controller_invariance_pending_metadata_replay" in statuses or "invariance_pending" in statuses:
        return "controller_invariance_metadata_replay_required"
    if "shadow_only_path_pending_default_path_evidence" in statuses:
        return "shadow_default_path_evidence_required"
    if any(status.startswith("not_closed_") for status in statuses):
        return "mainline_diff_authorization_required"
    return "runtime_provenance_metadata_replay_allowed"


def build_report(
    *,
    authorization_status: Mapping[str, Any],
    instrumentation_status: Mapping[str, Any],
    action_diff: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        _closure_for_row(
            row,
            instrumentation_status=instrumentation_status,
            action_diff=action_diff,
        )
        for row in authorization_status.get("rows", []) or []
        if isinstance(row, Mapping)
    ]
    closure_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("closure_status", ""))
        closure_counts[status] = closure_counts.get(status, 0) + 1
    blocking_rows = [row for row in rows if not bool(row.get("closed")) and bool(row.get("blocks_metadata_replay"))]
    mainline_complete = not bool(blocking_rows)
    next_action = _next_action(rows)
    expert_blocking = any(
        row.get("path") == EXPERT_PATH and row.get("closure_status") != "closed"
        for row in rows
    )
    notes = [
        "This closure report does not restore, stash, stage, edit, or replay anything.",
        "metadata replay is not allowed while mainline_diff_authorization_complete is false.",
    ]
    if expert_blocking:
        notes.insert(
            1,
            "expert_distillation.py remains a hard promotion blocker without user authorization or restoration.",
        )
    else:
        notes.insert(
            1,
            "expert_distillation.py is not present as a current closure blocker in this authorization status.",
        )
    return {
        "schema_version": "mainline_diff_authorization_closure_v2",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / decision-closure-only",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "controlled_replay_allowed": False,
        "metadata_replay_allowed": bool(mainline_complete),
        "performance_claim_allowed": False,
        "rejected_presets_remain_rejected": True,
        "mainline_diff_authorization_complete": bool(mainline_complete),
        "automated_cleanup_performed": False,
        "next_action": next_action,
        "closure_counts": dict(sorted(closure_counts.items())),
        "blocking_paths": [str(row.get("path", "")) for row in blocking_rows],
        "action_diff_reference": {
            "action_diff_steps": int(_num(action_diff.get("action_diff_steps"))),
            "max_abs_delta": float(_num(action_diff.get("max_abs_delta"))),
            "note": "This is closure context only; current instrumented metadata replay is still required for llm_agent.py.",
        },
        "runtime_provenance_reference": {
            "audit_status": (
                instrumentation_status.get("runtime_provenance", {}).get("audit_status", "")
                if isinstance(instrumentation_status.get("runtime_provenance", {}), Mapping)
                else ""
            ),
            "metadata_replay_required": _runtime_metadata_replay_required(instrumentation_status),
            "record_count": int(
                _num(
                    instrumentation_status.get("runtime_provenance", {}).get("record_count", 0)
                    if isinstance(instrumentation_status.get("runtime_provenance", {}), Mapping)
                    else 0
                )
            ),
        },
        "rows": rows,
        "notes": notes,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Mainline Diff Authorization Closure",
        "",
        "- Mode: audit-only / decision-closure-only",
        "- Controlled replay allowed: false",
        f"- Metadata replay allowed: {report.get('metadata_replay_allowed', False)}",
        "- Performance claim allowed: false",
        f"- Mainline diff authorization complete: {report.get('mainline_diff_authorization_complete', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Closure Counts",
        "",
        "| closure_status | count |",
        "| --- | ---: |",
    ]
    for status, count in dict(report.get("closure_counts", {})).items():
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Blocking Paths", ""])
    for path in report.get("blocking_paths", []) or []:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Blocking Row Summary", "", "| path | closure_status |", "| --- | --- |"])
    for row in report.get("rows", []) or []:
        if isinstance(row, Mapping) and not bool(row.get("closed")):
            lines.append(f"| `{row.get('path', '')}` | {row.get('closure_status', '')} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-status-json", required=True)
    parser.add_argument("--instrumentation-status-json", required=True)
    parser.add_argument("--action-diff-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        authorization_status=_load(args.authorization_status_json),
        instrumentation_status=_load(args.instrumentation_status_json),
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
    print(f"metadata_replay_allowed={report['metadata_replay_allowed']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
