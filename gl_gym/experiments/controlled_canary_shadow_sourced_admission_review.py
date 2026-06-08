"""Review v26 shadow-source evidence before a shadow-sourced controlled canary."""

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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _aggregate(report: Mapping[str, Any]) -> Mapping[str, Any]:
    value = report.get("aggregate", {})
    return value if isinstance(value, Mapping) else {}


def _joint_missing_count(joint_readiness: Mapping[str, Any]) -> int:
    counts = joint_readiness.get("missing_field_counts", {})
    if not isinstance(counts, Mapping):
        return 0
    return sum(int(_num(value)) for value in counts.values())


def _overlay_valid(overlay_preflight: Mapping[str, Any]) -> tuple[bool, bool, str]:
    overlay_ok = bool(
        overlay_preflight.get("controlled_canary_overlay_validated", False)
        and overlay_preflight.get("controlled_canary_runner_surface_validated", False)
        and int(_num(overlay_preflight.get("protocol_hash_mismatch_count"))) == 0
    )
    runner_path = str(overlay_preflight.get("overlay_runner_path", "") or "")
    runner_exists = bool(runner_path and _resolve(runner_path).exists())
    return overlay_ok, runner_exists, runner_path


def build_report(
    *,
    readiness_v26: Mapping[str, Any],
    shadow_replay: Mapping[str, Any],
    controlled_shadow: Mapping[str, Any],
    summary_audit: Mapping[str, Any],
    runtime_provenance_audit: Mapping[str, Any],
    joint_prediction_readiness: Mapping[str, Any],
    overlay_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    shadow_agg = _aggregate(shadow_replay)
    controlled_agg = _aggregate(controlled_shadow)
    summary_agg = _aggregate(summary_audit)

    v26_pass = bool(readiness_v26.get("shadow_trigger_source_sweep_pass", False))
    window_count = int(_num(shadow_agg.get("replay_window_count")))
    signal_steps = int(_num(shadow_agg.get("signal_steps")))
    unsafe_preferred = int(_num(controlled_agg.get("unsafe_preferred_steps"), _num(shadow_agg.get("unsafe_preferred_steps"))))
    unsafe_applied = int(_num(controlled_agg.get("unsafe_applied_steps"), _num(readiness_v26.get("post_run_metrics", {}).get("unsafe_applied_steps", 0))))
    runtime_complete = bool(
        runtime_provenance_audit.get("audit_status") == "runtime_provenance_complete"
        and int(_num(runtime_provenance_audit.get("record_count"))) > 0
        and int(_num(runtime_provenance_audit.get("runtime_reason_missing_count"))) == 0
        and int(_num(runtime_provenance_audit.get("unknown_post_guardrail_rewrite_count"))) == 0
    )
    joint_complete = bool(
        joint_prediction_readiness.get("ready_for_policy_judgment", False)
        and _joint_missing_count(joint_prediction_readiness) == 0
    )
    summary_pass = bool(summary_agg.get("decision") == "pass")
    overlay_valid, runner_exists, runner_path = _overlay_valid(overlay_preflight)

    checks = {
        "v26_shadow_source_sweep_pass": v26_pass,
        "shadow_replay_window_count_positive": window_count > 0,
        "shadow_signal_steps_positive": signal_steps > 0,
        "unsafe_preferred_steps_zero": unsafe_preferred == 0,
        "unsafe_applied_steps_zero": unsafe_applied == 0,
        "runtime_provenance_complete": runtime_complete,
        "joint_prediction_complete": joint_complete,
        "summary_pass": summary_pass,
        "controlled_canary_overlay_valid": overlay_valid,
        "overlay_runner_path_exists": runner_exists,
    }
    blockers = [key for key, value in checks.items() if not value]
    admission_pass = not blockers
    return {
        "schema_version": "controlled_canary_shadow_sourced_admission_review_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-sourced controlled canary admission",
            "reopens_rejected_preset": False,
        },
        "v26_shadow_source_sweep_pass": v26_pass,
        "shadow_sourced_admission_review_ready": True,
        "shadow_sourced_admission_pass": admission_pass,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "admission_checks": checks,
        "blocker_taxonomy": blockers,
        "metrics": {
            "shadow_replay_window_count": window_count,
            "shadow_signal_steps": signal_steps,
            "unsafe_preferred_steps": unsafe_preferred,
            "unsafe_applied_steps": unsafe_applied,
            "runtime_provenance_record_count": int(_num(runtime_provenance_audit.get("record_count"))),
            "runtime_reason_missing_count": int(_num(runtime_provenance_audit.get("runtime_reason_missing_count"))),
            "unknown_post_guardrail_rewrite_count": int(_num(runtime_provenance_audit.get("unknown_post_guardrail_rewrite_count"))),
            "joint_prediction_row_count": int(_num(joint_prediction_readiness.get("row_count"))),
            "joint_prediction_missing_fields": _joint_missing_count(joint_prediction_readiness),
        },
        "actual_protocol_implementation_for_execution": "protocol_v1_controlled_canary_overlay_20260525",
        "protocol_v1_overlay_root": str(overlay_preflight.get("protocol_v1_overlay_root", "") or ""),
        "overlay_runner_path": runner_path,
        "next_action": (
            "build_shadow_sourced_strict_projection_manifest"
            if admission_pass
            else "shadow_sourced_admission_blocker_diagnosis"
        ),
    }


def build_v31_report(
    *,
    readiness_v30: Mapping[str, Any],
    strict_sourcing: Mapping[str, Any],
    summary_audit: Mapping[str, Any],
    runtime_provenance_audit: Mapping[str, Any],
    joint_prediction_readiness: Mapping[str, Any],
    overlay_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    summary_agg = _aggregate(summary_audit)
    overlay_valid, runner_exists, runner_path = _overlay_valid(overlay_preflight)
    v30_pass = bool(
        readiness_v30.get("v30_shadow_sweep_executed", False)
        and readiness_v30.get("post_run_audits_pass", False)
        and readiness_v30.get("runtime_provenance_complete", False)
        and readiness_v30.get("joint_prediction_complete", False)
    )
    strict_source_found = bool(strict_sourcing.get("strict_targeted_source_found", False))
    expected_steps = int(_num(strict_sourcing.get("expected_strict_eligible_applied_steps")))
    strict_rows = [
        row
        for row in strict_sourcing.get("sample_rows", []) or []
        if isinstance(row, Mapping) and str(row.get("strict_targeted_source", "")).strip().lower() in {"1", "true", "yes", "y"}
    ]
    unsafe_preferred = int(
        sum(1 for row in strict_rows if str(row.get("unsafe_preferred", "")).strip().lower() in {"1", "true", "yes", "y"})
    )
    unsafe_conflict = int(
        sum(1 for row in strict_rows if str(row.get("unsafe_conflict", "")).strip().lower() in {"1", "true", "yes", "y"})
    )
    runtime_complete = bool(
        runtime_provenance_audit.get("audit_status") == "runtime_provenance_complete"
        and int(_num(runtime_provenance_audit.get("record_count"))) > 0
        and int(_num(runtime_provenance_audit.get("runtime_reason_missing_count"))) == 0
        and int(_num(runtime_provenance_audit.get("unknown_post_guardrail_rewrite_count"))) == 0
    )
    joint_complete = bool(
        joint_prediction_readiness.get("ready_for_policy_judgment", False)
        and _joint_missing_count(joint_prediction_readiness) == 0
    )
    checks = {
        "v30_shadow_sweep_pass": v30_pass,
        "strict_targeted_source_found": strict_source_found,
        "expected_strict_eligible_applied_steps_positive": expected_steps > 0,
        "strict_source_rows_positive": len(strict_rows) > 0,
        "unsafe_preferred_steps_zero": unsafe_preferred == 0,
        "unsafe_conflict_steps_zero": unsafe_conflict == 0,
        "summary_pass": summary_agg.get("decision") == "pass",
        "runtime_provenance_complete": runtime_complete,
        "joint_prediction_complete": joint_complete,
        "controlled_canary_overlay_valid": overlay_valid,
        "overlay_runner_path_exists": runner_exists,
    }
    blockers = [key for key, value in checks.items() if not value]
    admission_pass = not blockers
    return {
        "schema_version": "controlled_canary_shadow_sourced_admission_review_v31",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "v31 strict-source controlled canary admission",
            "reopens_rejected_preset": False,
        },
        "v30_shadow_sweep_pass": v30_pass,
        "shadow_sourced_admission_review_ready": True,
        "shadow_sourced_admission_pass": admission_pass,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "admission_checks": checks,
        "blocker_taxonomy": blockers,
        "metrics": {
            "expected_strict_eligible_applied_steps": expected_steps,
            "strict_source_row_count": len(strict_rows),
            "would_apply_steps": int(_num(strict_sourcing.get("would_apply_steps"))),
            "strict_filtered_steps": int(_num(strict_sourcing.get("strict_filtered_steps"))),
            "unsafe_preferred_steps": unsafe_preferred,
            "unsafe_conflict_steps": unsafe_conflict,
            "runtime_provenance_record_count": int(_num(runtime_provenance_audit.get("record_count"))),
            "runtime_reason_missing_count": int(_num(runtime_provenance_audit.get("runtime_reason_missing_count"))),
            "unknown_post_guardrail_rewrite_count": int(_num(runtime_provenance_audit.get("unknown_post_guardrail_rewrite_count"))),
            "joint_prediction_row_count": int(_num(joint_prediction_readiness.get("row_count"))),
            "joint_prediction_missing_fields": _joint_missing_count(joint_prediction_readiness),
        },
        "actual_protocol_implementation_for_execution": "protocol_v1_controlled_canary_overlay_20260525",
        "protocol_v1_overlay_root": str(overlay_preflight.get("protocol_v1_overlay_root", "") or ""),
        "overlay_runner_path": runner_path,
        "next_action": (
            "build_v31_strict_source_manifest"
            if admission_pass
            else "v31_strict_source_admission_blocker_diagnosis"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    title = (
        "# Shadow-Sourced Controlled Canary Admission Review v31"
        if report.get("schema_version") == "controlled_canary_shadow_sourced_admission_review_v31"
        else "# Shadow-Sourced Controlled Canary Admission Review v27"
    )
    lines = [
        title,
        "",
        f"- Admission pass: {report.get('shadow_sourced_admission_pass', False)}",
        "- Controlled replay allowed: false",
        "- Controlled replay execution allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Checks",
        "",
        "| gate | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("admission_checks", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Metrics", "", "| metric | value |", "| --- | --- |"])
    for key, value in dict(report.get("metrics", {})).items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Blockers", ""])
    blockers = report.get("blocker_taxonomy", []) or []
    lines.extend(f"- `{item}`" for item in blockers) if blockers else lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-v26-json", default="")
    parser.add_argument("--shadow-replay-json", default="")
    parser.add_argument("--controlled-shadow-json", default="")
    parser.add_argument("--readiness-v30-json", default="")
    parser.add_argument("--strict-sourcing-json", default="")
    parser.add_argument("--summary-audit-json", required=True)
    parser.add_argument("--runtime-provenance-json", required=True)
    parser.add_argument("--joint-readiness-json", required=True)
    parser.add_argument("--overlay-preflight-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.readiness_v30_json or args.strict_sourcing_json:
        if not args.readiness_v30_json or not args.strict_sourcing_json:
            raise SystemExit("--readiness-v30-json and --strict-sourcing-json must be provided together")
        report = build_v31_report(
            readiness_v30=_load(args.readiness_v30_json),
            strict_sourcing=_load(args.strict_sourcing_json),
            summary_audit=_load(args.summary_audit_json),
            runtime_provenance_audit=_load(args.runtime_provenance_json),
            joint_prediction_readiness=_load(args.joint_readiness_json),
            overlay_preflight=_load(args.overlay_preflight_json),
        )
    else:
        if not args.readiness_v26_json or not args.shadow_replay_json or not args.controlled_shadow_json:
            raise SystemExit(
                "--readiness-v26-json, --shadow-replay-json, and --controlled-shadow-json are required unless v31 inputs are provided"
            )
        report = build_report(
            readiness_v26=_load(args.readiness_v26_json),
            shadow_replay=_load(args.shadow_replay_json),
            controlled_shadow=_load(args.controlled_shadow_json),
            summary_audit=_load(args.summary_audit_json),
            runtime_provenance_audit=_load(args.runtime_provenance_json),
            joint_prediction_readiness=_load(args.joint_readiness_json),
            overlay_preflight=_load(args.overlay_preflight_json),
        )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"shadow_sourced_admission_pass={report['shadow_sourced_admission_pass']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["shadow_sourced_admission_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
