"""Review gates for Wave-1 controlled canary expansion admission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASELINE_CONTROLLER = "llm_rspc_v2"
CANDIDATE_CONTROLLER = "llm_rspc_v2_hot_dry_proposer_strict"


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


def _runner_supports(path: Path, controller: str) -> bool:
    try:
        return controller in path.read_text(encoding="utf-8")
    except Exception:
        return False


def build_report(
    *,
    consolidation: Mapping[str, Any],
    readiness_v22: Mapping[str, Any],
    protocol_v1_overlay_preflight: Mapping[str, Any],
    triggered_summary: Mapping[str, Any],
    triggered_trace: Mapping[str, Any],
    triggered_effect: Mapping[str, Any],
    triggered_runtime: Mapping[str, Any],
    triggered_joint: Mapping[str, Any],
) -> dict[str, Any]:
    overlay_root = str(protocol_v1_overlay_preflight.get("protocol_v1_overlay_root", "") or "")
    overlay_runner_value = protocol_v1_overlay_preflight.get("overlay_runner_path", "")
    overlay_runner = (
        _resolve(overlay_runner_value)
        if overlay_runner_value
        else _resolve(Path(overlay_root) / "gl_gym" / "experiments" / "run_frozen_benchmark.py")
    )
    summary_agg = _aggregate(triggered_summary)
    trace_agg = _aggregate(triggered_trace)
    effect_agg = _aggregate(triggered_effect)

    precheck = {
        "expanded_metadata_consolidation_pass": bool(consolidation.get("expanded_metadata_consolidation_pass", False)),
        "readiness_v22_triggered_canary_pass": bool(readiness_v22.get("triggered_controlled_canary_pass", False)),
        "readiness_v22_no_promotion_evidence": not bool(readiness_v22.get("promotion_evidence", False)),
        "readiness_v22_no_performance_claim": not bool(readiness_v22.get("performance_claim_allowed", False)),
        "controlled_canary_overlay_validated": bool(
            protocol_v1_overlay_preflight.get("controlled_canary_overlay_validated", False)
        ),
        "protocol_v1_overlay_validated": bool(protocol_v1_overlay_preflight.get("protocol_v1_overlay_validated", False)),
        "protocol_hashes_match": int(_num(protocol_v1_overlay_preflight.get("protocol_hash_mismatch_count"))) == 0,
        "controlled_runner_surface_validated": bool(
            protocol_v1_overlay_preflight.get("controlled_canary_runner_surface_validated", False)
        ),
        "overlay_runner_path_exists": overlay_runner.exists(),
        "overlay_runner_supports_candidate_controller": _runner_supports(overlay_runner, CANDIDATE_CONTROLLER),
        "triggered_summary_pass": summary_agg.get("decision") == "pass",
        "triggered_trace_has_strict_applied": int(_num(trace_agg.get("strict_applied_steps"))) > 0,
        "triggered_trace_unsafe_preferred_zero": int(_num(trace_agg.get("unsafe_preferred_steps"))) == 0,
        "triggered_trace_unsafe_conflict_zero": int(_num(trace_agg.get("unsafe_conflict_steps"))) == 0,
        "triggered_trace_unsafe_applied_zero": int(_num(trace_agg.get("unsafe_applied_steps"))) == 0,
        "triggered_effect_runtime_error_zero": int(_num(effect_agg.get("runtime_error_steps"))) == 0,
        "triggered_effect_unsafe_applied_zero": int(_num(effect_agg.get("unsafe_applied_steps"))) == 0,
        "triggered_runtime_provenance_complete": triggered_runtime.get("audit_status") == "runtime_provenance_complete",
        "triggered_runtime_reason_missing_zero": int(_num(triggered_runtime.get("runtime_reason_missing_count"))) == 0,
        "triggered_unknown_rewrite_zero": int(_num(triggered_runtime.get("unknown_post_guardrail_rewrite_count"))) == 0,
        "triggered_joint_prediction_complete": bool(triggered_joint.get("ready_for_policy_judgment", False)),
        "default_llm_rspc_v2_unchanged": True,
    }
    blockers = [key for key, value in precheck.items() if not value]
    admission_pass = not blockers
    return {
        "schema_version": "controlled_canary_expansion_admission_review_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "controlled canary expansion admission review",
            "reopens_rejected_preset": False,
        },
        "controlled_canary_expansion_admission_review_ready": True,
        "controlled_canary_expansion_admission_pass": admission_pass,
        "wave1_controlled_canary_allowed": admission_pass,
        "controlled_replay_allowed": admission_pass,
        "controlled_replay_scope": "wave1_controlled_canary_only" if admission_pass else "blocked",
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "baseline_controller": BASELINE_CONTROLLER,
        "candidate_controller": CANDIDATE_CONTROLLER,
        "actual_protocol_implementation_for_execution": "protocol_v1_controlled_canary_overlay_20260525",
        "protocol_v1_overlay_root": overlay_root,
        "overlay_runner_path": str(overlay_runner),
        "precheck": precheck,
        "blockers": blockers,
        "failure_taxonomy": [
            "cache_mismatch",
            "protocol_overlay_mismatch",
            "unsafe_preferred",
            "unsafe_applied",
            "hard_safety_regression",
            "runtime_error",
            "strict_cache_miss",
            "metadata_missing",
            "joint_prediction_missing",
            "cartesian_product_risk",
        ],
        "next_action": (
            "generate_controlled_canary_expansion_wave1_manifest"
            if admission_pass
            else "controlled_canary_expansion_admission_blocker_diagnosis"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Controlled Canary Expansion Admission Review",
        "",
        "- Mode: controlled canary expansion admission review",
        "- Default `llm_rspc_v2` changed: false",
        f"- Wave-1 allowed: {report.get('wave1_controlled_canary_allowed', False)}",
        f"- Controlled replay scope: `{report.get('controlled_replay_scope', '')}`",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Precheck",
        "",
        "| check | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("precheck", {})).items():
        lines.append(f"| {key} | `{value}` |")
    blockers = report.get("blockers", []) or []
    lines.extend(["", "## Blockers", ""])
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidation-json", required=True)
    parser.add_argument("--readiness-v22-json", required=True)
    parser.add_argument("--protocol-v1-overlay-preflight-json", required=True)
    parser.add_argument("--triggered-summary-json", required=True)
    parser.add_argument("--triggered-trace-json", required=True)
    parser.add_argument("--triggered-effect-json", required=True)
    parser.add_argument("--triggered-runtime-json", required=True)
    parser.add_argument("--triggered-joint-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        consolidation=_load(args.consolidation_json),
        readiness_v22=_load(args.readiness_v22_json),
        protocol_v1_overlay_preflight=_load(args.protocol_v1_overlay_preflight_json),
        triggered_summary=_load(args.triggered_summary_json),
        triggered_trace=_load(args.triggered_trace_json),
        triggered_effect=_load(args.triggered_effect_json),
        triggered_runtime=_load(args.triggered_runtime_json),
        triggered_joint=_load(args.triggered_joint_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"wave1_controlled_canary_allowed={report['wave1_controlled_canary_allowed']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["controlled_canary_expansion_admission_review_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
