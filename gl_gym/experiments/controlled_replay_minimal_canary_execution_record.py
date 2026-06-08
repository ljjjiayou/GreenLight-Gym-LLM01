"""Record authorization and planned commands for a minimal controlled canary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _command_for_group(
    *,
    runner_path: str,
    group: Mapping[str, Any],
    controllers: Sequence[str],
    selected_cache_path: str,
    output_root: str,
) -> list[str]:
    group_id = str(group.get("group_id", "canary_group"))
    output_json = str(Path(output_root) / f"minimal_controlled_canary_{group_id}_20260525.json")
    output_trace_dir = str(Path(output_root) / "traces")
    return [
        "python",
        runner_path,
        "--years",
        ",".join(str(item) for item in group.get("years", []) or []),
        "--days",
        ",".join(str(item) for item in group.get("days", []) or []),
        "--seeds",
        ",".join(str(item) for item in group.get("seeds", []) or []),
        "--controllers",
        ",".join(str(item) for item in controllers),
        "--max-steps",
        "240",
        "--plan-cache-mode",
        "replay",
        "--plan-cache-path",
        selected_cache_path,
        "--plan-cache-strict",
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--output-json",
        output_json,
        "--output-trace-dir",
        output_trace_dir,
    ]


def build_report(
    *,
    admission_review: Mapping[str, Any],
    canary_manifest: Mapping[str, Any],
    authorization_source: str,
    output_root: str,
) -> dict[str, Any]:
    authorized = bool(admission_review.get("minimal_controlled_canary_allowed", False))
    controllers = list(canary_manifest.get("controllers", []) or [])
    runner_path = str(admission_review.get("overlay_runner_path", ""))
    selected_cache_path = str(canary_manifest.get("selected_cache_path", ""))
    commands = [
        _command_for_group(
            runner_path=runner_path,
            group=group,
            controllers=controllers,
            selected_cache_path=selected_cache_path,
            output_root=output_root,
        )
        for group in canary_manifest.get("command_groups", []) or []
        if isinstance(group, Mapping)
    ]
    return {
        "schema_version": "controlled_replay_minimal_canary_execution_record_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC"],
            "default_llm_rspc_v2_changed": False,
            "mode": "minimal controlled canary only",
            "reopens_rejected_preset": False,
        },
        "authorization_source": authorization_source,
        "minimal_controlled_canary_authorized": authorized,
        "controlled_replay_execution_allowed": authorized,
        "controlled_replay_allowed": authorized,
        "controlled_replay_scope": "minimal_controlled_canary_only" if authorized else "blocked",
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "default_llm_rspc_v2_changed": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "baseline_controller": str(canary_manifest.get("baseline_controller", "")),
        "candidate_controller": str(canary_manifest.get("candidate_controller", "")),
        "scenario_ids": list(canary_manifest.get("scenario_ids", []) or []),
        "controllers": controllers,
        "selected_cache_path": selected_cache_path,
        "actual_protocol_implementation_for_execution": str(
            admission_review.get("actual_protocol_implementation_for_execution", "protocol_v1_snapshot_ephemeral_overlay")
        ),
        "protocol_v1_overlay_root": str(admission_review.get("protocol_v1_overlay_root", "")),
        "runner_path": runner_path,
        "planned_environment": {
            "PYTHONPATH": f"{admission_review.get('protocol_v1_overlay_root', '')};{PROJECT_ROOT}",
        },
        "output_root": output_root,
        "planned_commands": commands,
        "post_run_audits": [
            "audit_frozen_benchmark_summary",
            "hot_dry_controlled_replay_trace_audit",
            "hot_dry_strict_effect_audit",
            "post_guardrail_runtime_provenance_audit",
            "post_guardrail_joint_prediction_readiness_v2",
            "metadata_replay_readiness_checklist_v21",
        ],
        "next_action": "execute_minimal_controlled_canary" if authorized else "controlled_canary_not_authorized",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Minimal Controlled Canary Execution Record",
        "",
        f"- Authorized: {report.get('minimal_controlled_canary_authorized', False)}",
        f"- Scope: `{report.get('controlled_replay_scope', '')}`",
        "- Default `llm_rspc_v2` changed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Planned Commands",
        "",
    ]
    for command in report.get("planned_commands", []) or []:
        lines.append("```powershell")
        lines.append(" ".join(str(part) for part in command))
        lines.append("```")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-review-json", required=True)
    parser.add_argument("--canary-manifest-json", required=True)
    parser.add_argument("--authorization-source", required=True)
    parser.add_argument("--canary-output-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        admission_review=_load(args.admission_review_json),
        canary_manifest=_load(args.canary_manifest_json),
        authorization_source=args.authorization_source,
        output_root=args.canary_output_root,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"minimal_controlled_canary_authorized={report['minimal_controlled_canary_authorized']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["minimal_controlled_canary_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
