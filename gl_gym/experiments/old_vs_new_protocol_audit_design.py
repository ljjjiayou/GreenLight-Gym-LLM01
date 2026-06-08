"""Design an old-vs-new evaluation protocol audit without running replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PROTOCOL_V1_PATHS = [
    "gl_gym/experiments/frozen_benchmark_protocol.py",
    "gl_gym/experiments/run_frozen_benchmark.py",
    "tests/test_frozen_benchmark.py",
    "tests/test_frozen_benchmark_protocol.py",
    "tests/test_planning_extensions.py",
]


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _git_capture(args: Sequence[str], *, text: bool = True) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=text,
        encoding="utf-8" if text else None,
        errors="replace" if text else None,
    )


def build_protocol_v1_snapshot_manifest(
    *,
    source_git_ref: str = "HEAD",
    tracked_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Describe the accepted protocol v1 source snapshot without writing source files."""

    paths = list(tracked_paths or DEFAULT_PROTOCOL_V1_PATHS)
    rev_parse = _git_capture(["rev-parse", source_git_ref])
    source_commit = rev_parse.stdout.strip() if rev_parse.returncode == 0 else ""
    files: list[dict[str, Any]] = []
    for path in paths:
        blob = _git_capture(["show", f"{source_git_ref}:{path}"], text=False)
        status = _git_capture(["status", "--short", "--", path])
        if blob.returncode == 0:
            raw = blob.stdout if isinstance(blob.stdout, bytes) else bytes(blob.stdout)
            files.append(
                {
                    "path": path,
                    "object_available": True,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "byte_count": len(raw),
                    "line_count": raw.count(b"\n") + (1 if raw and not raw.endswith(b"\n") else 0),
                    "working_tree_status": status.stdout.strip(),
                }
            )
        else:
            files.append(
                {
                    "path": path,
                    "object_available": False,
                    "sha256": None,
                    "byte_count": 0,
                    "line_count": 0,
                    "working_tree_status": status.stdout.strip(),
                    "error": blob.stderr.decode("utf-8", errors="replace") if isinstance(blob.stderr, bytes) else str(blob.stderr),
                }
            )
    snapshot_available = bool(files) and all(bool(row.get("object_available")) for row in files)
    return {
        "schema_version": "protocol_v1_snapshot_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "snapshot-manifest-only / no replay",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "source_git_ref": source_git_ref,
        "source_git_commit": source_commit,
        "snapshot_available": snapshot_available,
        "tracked_baseline_file_count": len(files),
        "files": files,
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "metadata_replay_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "notes": [
            "This manifest hashes accepted protocol v1 files from git and does not write a source snapshot.",
            "It is an input boundary for a future old-vs-new audit, not replay evidence.",
        ],
    }


def build_snapshot_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Protocol v1 Snapshot Manifest",
        "",
        "- Mode: snapshot-manifest-only / no replay",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Source git ref: `{report.get('source_git_ref', '')}`",
        f"- Source git commit: `{report.get('source_git_commit', '')}`",
        f"- Snapshot available: {report.get('snapshot_available', False)}",
        "",
        "| path | available | sha256 | working tree status |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.get("files", []) or []:
        if isinstance(row, Mapping):
            lines.append(
                f"| `{row.get('path', '')}` | `{row.get('object_available', False)}` | `{row.get('sha256', '')}` | `{row.get('working_tree_status', '')}` |"
            )
    return "\n".join(lines) + "\n"


def build_report(
    authorization_packet: Mapping[str, Any],
    *,
    plan_cache_path: str,
    controller: str,
    scenario_list: Sequence[str],
    max_steps: int,
    plan_cache_key_policy: str,
) -> dict[str, Any]:
    return {
        "schema_version": "old_vs_new_protocol_audit_design_v2",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "design-only / no replay",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "controlled_replay_allowed": False,
        "metadata_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "protocol_baseline_authorized": bool(authorization_packet.get("protocol_baseline_authorized", False)),
        "recommended_decision": authorization_packet.get("recommended_decision", ""),
        "design_ready": True,
        "safety_boundary_test_oracle_hunk_count": int(
            authorization_packet.get("safety_boundary_test_oracle_hunk_count", 0) or 0
        ),
        "safety_boundary_test_oracle_note": (
            "tests/test_planning_extensions.py changes are safety-boundary test oracle scope and are not proof of old-vs-new replay equivalence"
        ),
        "fixed_inputs": {
            "same_cache": True,
            "plan_cache_path": plan_cache_path,
            "same_controller": True,
            "controller": controller,
            "same_scenario_list": True,
            "scenario_list": list(scenario_list),
            "same_max_steps": True,
            "max_steps": int(max_steps),
            "same_plan_cache_key_policy": True,
            "plan_cache_key_policy": plan_cache_key_policy,
            "no_online_llm": True,
            "no_cache_fill": True,
        },
        "compare_fields": [
            "summary row count",
            "scenario ids and preset labels",
            "cache hit/miss metrics",
            "runtime error metrics",
            "strict miss metrics",
            "gate failure count and reasons",
            "reward/profit/safety metric extraction",
            "action-diff field definitions and values",
            "safety-boundary test oracle changes kept separate from replay protocol equivalence",
        ],
        "allowed_outcomes": [
            "old PASS and new PASS with identical metrics means protocol-equivalence candidate, still not promotion evidence by itself",
            "old PASS and new FAIL is acceptable only if the failing stricter safety gate has been explicitly authorized as protocol v2 baseline",
            "any runner control surface, cache behavior, summary aggregation, or test oracle mismatch blocks metadata replay until explained",
        ],
        "blocked_until": [
            "protocol v2 authorization decision is explicit",
            "old-vs-new audit can be run with same cache, controller, scenario list, max steps, and key policy",
            "cache coverage check passes before any strict metadata replay",
        ],
        "next_action": "protocol_v2_authorization_required_before_equivalence_audit",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    inputs = dict(report.get("fixed_inputs", {}))
    lines = [
        "# Old-vs-New Protocol Audit Design",
        "",
        "- Mode: design-only / no replay",
        "- Replay run: false",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Protocol baseline authorized: {report.get('protocol_baseline_authorized', False)}",
        f"- Design ready: {report.get('design_ready', False)}",
        f"- Safety-boundary test oracle hunks: {report.get('safety_boundary_test_oracle_hunk_count', 0)}",
        "",
        "## Fixed Inputs",
        "",
        f"- Controller: `{inputs.get('controller', '')}`",
        f"- Plan cache path: `{inputs.get('plan_cache_path', '')}`",
        f"- Max steps: `{inputs.get('max_steps', '')}`",
        f"- Key policy: `{inputs.get('plan_cache_key_policy', '')}`",
        "",
        "## Compare Fields",
        "",
    ]
    for item in report.get("compare_fields", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Allowed Outcomes", ""])
    for item in report.get("allowed_outcomes", []) or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def build_readiness_report(
    design: Mapping[str, Any],
    *,
    old_protocol_snapshot_path: str = "",
    snapshot_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    old_snapshot_available = False
    resolved_snapshot = ""
    if old_protocol_snapshot_path:
        path = Path(old_protocol_snapshot_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        old_snapshot_available = path.exists()
        resolved_snapshot = str(path)
    if isinstance(snapshot_manifest, Mapping):
        old_snapshot_available = bool(snapshot_manifest.get("snapshot_available", False))
        resolved_snapshot = str(snapshot_manifest.get("source_git_ref", "")) or resolved_snapshot
    protocol_user_decision_complete = bool(design.get("protocol_user_decision_complete", False))
    return {
        "schema_version": "old_vs_new_protocol_audit_readiness_v2" if isinstance(snapshot_manifest, Mapping) else "old_vs_new_protocol_audit_readiness_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "readiness-only / no replay",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "metadata_replay_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "design_ready": bool(design.get("design_ready", False)),
        "protocol_user_decision_complete": protocol_user_decision_complete,
        "protocol_baseline_authorized": bool(design.get("protocol_baseline_authorized", False)),
        "old_protocol_snapshot_path": resolved_snapshot or None,
        "old_protocol_snapshot_available": old_snapshot_available,
        "protocol_v1_snapshot_manifest_available": bool(isinstance(snapshot_manifest, Mapping)),
        "protocol_v1_snapshot_source_git_ref": (
            snapshot_manifest.get("source_git_ref") if isinstance(snapshot_manifest, Mapping) else None
        ),
        "protocol_v1_snapshot_source_git_commit": (
            snapshot_manifest.get("source_git_commit") if isinstance(snapshot_manifest, Mapping) else None
        ),
        "old_vs_new_protocol_audit_ready": bool(
            design.get("design_ready", False)
            and protocol_user_decision_complete
            and old_snapshot_available
        ),
        "blocked_until": [
            item
            for item, ok in (
                ("old accepted protocol snapshot available", old_snapshot_available),
                ("protocol user decision complete", protocol_user_decision_complete),
            )
            if not ok
        ],
        "next_action": (
            "old_vs_new_protocol_audit_can_be_planned"
            if protocol_user_decision_complete and old_snapshot_available
            else "old_protocol_snapshot_required_before_audit"
        ),
        "notes": [
            "This readiness report does not run old-vs-new replay.",
            "If the old accepted protocol snapshot is unavailable, audit execution remains blocked.",
        ],
    }


def build_readiness_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Old-vs-New Protocol Audit Readiness",
        "",
        "- Mode: readiness-only / no replay",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Design ready: {report.get('design_ready', False)}",
        f"- Protocol user decision complete: {report.get('protocol_user_decision_complete', False)}",
        f"- Old snapshot available: {report.get('old_protocol_snapshot_available', False)}",
        f"- Audit ready: {report.get('old_vs_new_protocol_audit_ready', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
    ]
    if report.get("blocked_until"):
        lines.extend(["", "## Blocked Until", ""])
        lines.extend(f"- {item}" for item in report.get("blocked_until", []) or [])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-json", required=True)
    parser.add_argument("--plan-cache-path", default="gl_gym/result/plan_cache/llm_plan_cache.json")
    parser.add_argument("--controller", default="llm_rspc_v2")
    parser.add_argument("--scenario", action="append", default=["y2020_d120_s44_n240"])
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--plan-cache-key-policy", default="scenario_timestep")
    parser.add_argument("--old-protocol-snapshot", default="")
    parser.add_argument("--snapshot-git-ref", default="HEAD")
    parser.add_argument("--snapshot-path", action="append", default=[])
    parser.add_argument("--snapshot-manifest-json", default="")
    parser.add_argument("--snapshot-output-json", default="")
    parser.add_argument("--snapshot-output-md", default="")
    parser.add_argument("--readiness-output-json", default="")
    parser.add_argument("--readiness-output-md", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        _load(args.authorization_json),
        plan_cache_path=args.plan_cache_path,
        controller=args.controller,
        scenario_list=args.scenario,
        max_steps=args.max_steps,
        plan_cache_key_policy=args.plan_cache_key_policy,
    )
    report["protocol_user_decision_complete"] = bool(
        _load(args.authorization_json).get("protocol_user_decision_complete", False)
    )
    snapshot_manifest = None
    if args.snapshot_manifest_json:
        snapshot_manifest = _load(args.snapshot_manifest_json)
    elif args.snapshot_output_json or args.snapshot_output_md:
        snapshot_manifest = build_protocol_v1_snapshot_manifest(
            source_git_ref=args.snapshot_git_ref,
            tracked_paths=args.snapshot_path or None,
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
    print(f"replay_run={report['replay_run']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    if snapshot_manifest is not None:
        if args.snapshot_output_json:
            snapshot_json = Path(args.snapshot_output_json)
            if not snapshot_json.is_absolute():
                snapshot_json = PROJECT_ROOT / snapshot_json
            snapshot_json.parent.mkdir(parents=True, exist_ok=True)
            snapshot_json.write_text(
                json.dumps(snapshot_manifest, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"wrote {snapshot_json}")
        if args.snapshot_output_md:
            snapshot_md = Path(args.snapshot_output_md)
            if not snapshot_md.is_absolute():
                snapshot_md = PROJECT_ROOT / snapshot_md
            snapshot_md.parent.mkdir(parents=True, exist_ok=True)
            snapshot_md.write_text(build_snapshot_markdown_report(snapshot_manifest), encoding="utf-8")
            print(f"wrote {snapshot_md}")
    if args.readiness_output_json or args.readiness_output_md:
        readiness = build_readiness_report(
            report,
            old_protocol_snapshot_path=args.old_protocol_snapshot,
            snapshot_manifest=snapshot_manifest,
        )
        if args.readiness_output_json:
            readiness_json = Path(args.readiness_output_json)
            if not readiness_json.is_absolute():
                readiness_json = PROJECT_ROOT / readiness_json
            readiness_json.parent.mkdir(parents=True, exist_ok=True)
            readiness_json.write_text(
                json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(f"wrote {readiness_json}")
        if args.readiness_output_md:
            readiness_md = Path(args.readiness_output_md)
            if not readiness_md.is_absolute():
                readiness_md = PROJECT_ROOT / readiness_md
            readiness_md.parent.mkdir(parents=True, exist_ok=True)
            readiness_md.write_text(build_readiness_markdown(readiness), encoding="utf-8")
            print(f"wrote {readiness_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
