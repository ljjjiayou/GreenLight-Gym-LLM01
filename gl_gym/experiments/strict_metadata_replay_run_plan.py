"""Draft a strict metadata replay run plan without executing replay."""

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


def _canonical_scenarios(cache_scenario_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in cache_scenario_plan.get("scenario_list", []) or []
        if isinstance(row, Mapping) and row.get("scenario_family") == "canonical_failure"
    ]


def build_report(
    *,
    protocol_v1_snapshot: Mapping[str, Any],
    blocked_categories: Mapping[str, Any],
    cache_scenario_plan: Mapping[str, Any],
    cache_coverage: Mapping[str, Any],
    default_path_plan: Mapping[str, Any],
    expanded_cache_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    canonical = _canonical_scenarios(cache_scenario_plan)
    v1_snapshot_ready = bool(protocol_v1_snapshot.get("snapshot_available", False))
    v1_isolation_ready = bool(blocked_categories.get("blocked_categories_resolved_for_v1_preflight", False))
    cache_plan_ready = bool(cache_scenario_plan.get("cache_scenario_plan_ready", False))
    cache_coverage_pass = bool(cache_coverage.get("cache_coverage_pass", False))
    default_path_plan_ready = bool(default_path_plan.get("plan_ready", False))
    run_plan_ready = bool(v1_snapshot_ready and v1_isolation_ready and cache_plan_ready and cache_coverage_pass and default_path_plan_ready and canonical)
    expanded_rows = list(expanded_cache_plan.get("rows", []) or []) if isinstance(expanded_cache_plan, Mapping) else []
    expanded_blockers = [
        {
            "family": row.get("family", ""),
            "coverage_status": row.get("coverage_status", ""),
            "scenario_count": row.get("scenario_count", 0),
        }
        for row in expanded_rows
        if isinstance(row, Mapping) and not bool(row.get("can_enter_initial_strict_metadata_replay_plan", False))
    ]
    return {
        "schema_version": "strict_metadata_replay_run_plan_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "run-plan-only / no replay",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "strict_metadata_replay_run_plan_ready": run_plan_ready,
        "replay_run": False,
        "metadata_replay_allowed": False,
        "metadata_replay_execution_allowed": False,
        "metadata_replay_execution_requires_separate_user_authorization": True,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "initial_run_scope": "canonical_failure_only",
        "allowed_initial_scenarios": canonical if run_plan_ready else [],
        "expanded_family_blockers": expanded_blockers,
        "fixed_inputs": {
            "controller": cache_scenario_plan.get("controller", "llm_rspc_v2"),
            "selected_cache_path": cache_scenario_plan.get("selected_cache_path", ""),
            "plan_cache_key_policy": cache_scenario_plan.get("plan_cache_key_policy", "scenario_timestep"),
            "max_steps": cache_scenario_plan.get("max_steps", 240),
            "protocol_v1_source_git_ref": protocol_v1_snapshot.get("source_git_ref", "HEAD"),
            "protocol_v1_source_git_commit": protocol_v1_snapshot.get("source_git_commit", ""),
        },
        "preflight_checks": {
            "protocol_v1_snapshot_available": v1_snapshot_ready,
            "blocked_categories_resolved_for_v1_preflight": v1_isolation_ready,
            "blocked_categories_resolved_for_protocol_v2": bool(blocked_categories.get("blocked_categories_resolved_for_protocol_v2", False)),
            "cache_scenario_plan_ready": cache_plan_ready,
            "cache_coverage_pass": cache_coverage_pass,
            "default_path_action_invariance_plan_ready": default_path_plan_ready,
            "canonical_cache_scope_only": cache_scenario_plan.get("coverage_claim_scope") == "canonical_failure_only",
        },
        "targets": {
            "action_diff_steps": 0,
            "max_abs_delta": 0,
            "selected_control": "unchanged",
            "post_guardrail_final_action": "unchanged",
            "runtime_provenance_record_count": ">0",
            "runtime_reason_missing_count": 0,
            "unknown_post_guardrail_rewrite_count": 0,
            "joint_prediction_missing_fields": 0,
            "cache_hit_rate": 1.0,
            "runtime_error_steps": 0,
            "strict_cache_miss_runtime_error_steps": 0,
        },
        "planned_audits_after_execution": [
            "action_diff_audit_requires_zero_steps",
            "post_guardrail_runtime_provenance_audit",
            "post_guardrail_joint_prediction_readiness",
            "metadata_replay_readiness_refresh",
        ],
        "execution_blockers": [
            "metadata replay execution requires separate explicit user authorization",
            "protocol v2 baseline remains unauthorized",
            "protocol v2 blocked categories remain unresolved for promotion evidence",
            "default-path action invariance evidence is not yet proven",
            "mainline diff authorization remains incomplete",
        ],
        "next_action": "request_explicit_metadata_replay_execution_authorization" if run_plan_ready else "repair_strict_metadata_replay_run_plan_inputs",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Strict Metadata Replay Run Plan",
        "",
        "- Mode: run-plan-only / no replay",
        "- Metadata replay allowed: false",
        f"- Run plan ready: {report.get('strict_metadata_replay_run_plan_ready', False)}",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Initial scope: `{report.get('initial_run_scope', '')}`",
        "",
        "## Initial Scenarios",
        "",
    ]
    for row in report.get("allowed_initial_scenarios", []) or []:
        if isinstance(row, Mapping):
            lines.append(f"- `{row.get('scenario_id', '')}` / `{row.get('env_id', '')}`")
    lines.extend(["", "## Targets", ""])
    for key, value in dict(report.get("targets", {})).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Blockers", ""])
    for item in report.get("execution_blockers", []) or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-v1-snapshot-json", required=True)
    parser.add_argument("--blocked-categories-json", required=True)
    parser.add_argument("--cache-scenario-plan-json", required=True)
    parser.add_argument("--cache-coverage-json", required=True)
    parser.add_argument("--default-path-plan-json", required=True)
    parser.add_argument("--expanded-cache-plan-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        protocol_v1_snapshot=_load(args.protocol_v1_snapshot_json),
        blocked_categories=_load(args.blocked_categories_json),
        cache_scenario_plan=_load(args.cache_scenario_plan_json),
        cache_coverage=_load(args.cache_coverage_json),
        default_path_plan=_load(args.default_path_plan_json),
        expanded_cache_plan=_load(args.expanded_cache_plan_json) if args.expanded_cache_plan_json else None,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"strict_metadata_replay_run_plan_ready={report['strict_metadata_replay_run_plan_ready']}")
    print(f"metadata_replay_execution_allowed={report['metadata_replay_execution_allowed']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
