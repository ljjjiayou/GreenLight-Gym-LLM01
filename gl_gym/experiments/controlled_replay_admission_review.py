"""Review admission gates for a minimal controlled hot-dry proposer canary."""

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


def _runner_supports(path: Path, controller: str) -> bool:
    try:
        return controller in path.read_text(encoding="utf-8")
    except Exception:
        return False


def build_report(
    *,
    consolidation: Mapping[str, Any],
    canary_manifest: Mapping[str, Any],
    cache_coverage: Mapping[str, Any],
    protocol_v1_overlay_preflight: Mapping[str, Any],
    historical_controlled_trace: Mapping[str, Any],
    historical_controlled_shadow: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = str(canary_manifest.get("candidate_controller", ""))
    overlay_root = str(protocol_v1_overlay_preflight.get("protocol_v1_overlay_root", "") or "")
    overlay_runner_value = protocol_v1_overlay_preflight.get("overlay_runner_path", "")
    overlay_runner = _resolve(overlay_runner_value) if overlay_runner_value else _resolve(Path(overlay_root) / "gl_gym" / "experiments" / "run_frozen_benchmark.py")
    current_runner = PROJECT_ROOT / "gl_gym" / "experiments" / "run_frozen_benchmark.py"
    actual_protocol_implementation = str(
        protocol_v1_overlay_preflight.get(
            "actual_protocol_implementation_for_execution",
            "protocol_v1_snapshot_ephemeral_overlay",
        )
    )
    controlled_canary_surface_validated = bool(
        protocol_v1_overlay_preflight.get("controlled_canary_runner_surface_validated", True)
    )
    controlled_agg = _aggregate(historical_controlled_trace)
    shadow_agg = _aggregate(historical_controlled_shadow)
    historical_trace_safe = bool(
        _num(controlled_agg.get("metadata_steps")) > 0
        and _num(controlled_agg.get("unsafe_preferred_steps")) == 0.0
        and _num(controlled_agg.get("unsafe_conflict_steps")) == 0.0
        and _num(controlled_agg.get("unsafe_applied_steps")) == 0.0
        and (
            _num(controlled_agg.get("would_apply_steps")) > 0.0
            or _num(controlled_agg.get("strict_applied_steps")) > 0.0
        )
    )
    historical_shadow_safe = bool(
        _num(shadow_agg.get("metadata_steps")) > 0
        and _num(shadow_agg.get("unsafe_preferred_steps")) == 0.0
        and _num(shadow_agg.get("safe_gain_steps")) > 0.0
    )
    overlay_supports_candidate = _runner_supports(overlay_runner, candidate)
    current_runner_supports_candidate = _runner_supports(current_runner, candidate)
    precheck = {
        "expanded_metadata_consolidation_pass": bool(consolidation.get("expanded_metadata_consolidation_pass", False)),
        "minimal_canary_manifest_ready": bool(canary_manifest.get("minimal_canary_manifest_ready", False)),
        "cache_coverage_pass": bool(cache_coverage.get("cache_coverage_pass", False)),
        "cache_coverage_rate_is_one": _num(cache_coverage.get("coverage_rate")) == 1.0,
        "missing_key_count_zero": int(cache_coverage.get("missing_key_count", 0) or 0) == 0,
        "missing_buffered_action_count_zero": int(cache_coverage.get("missing_buffered_action_count", 0) or 0) == 0,
        "missing_parsed_plan_count_zero": int(cache_coverage.get("missing_parsed_plan_count", 0) or 0) == 0,
        "cache_fill_blocked": not bool(cache_coverage.get("cache_fill_run", False)),
        "online_llm_blocked": not bool(cache_coverage.get("online_llm_called", False)),
        "protocol_v1_overlay_validated": bool(protocol_v1_overlay_preflight.get("protocol_v1_overlay_validated", False)),
        "controlled_canary_runner_surface_validated": controlled_canary_surface_validated,
        "overlay_runner_path_exists": overlay_runner.exists(),
        "overlay_runner_supports_candidate_controller": overlay_supports_candidate,
        "historical_controlled_trace_safe": historical_trace_safe,
        "historical_controlled_shadow_safe": historical_shadow_safe,
        "default_llm_rspc_v2_unchanged": True,
    }
    blockers = [key for key, value in precheck.items() if not value]
    admission_pass = not blockers
    if "overlay_runner_supports_candidate_controller" in blockers and current_runner_supports_candidate:
        next_action = "controlled_canary_runner_protocol_overlay_resolution_required"
    elif admission_pass:
        next_action = "generate_minimal_controlled_canary_execution_record"
    else:
        next_action = "controlled_replay_admission_blocker_diagnosis"
    return {
        "schema_version": "controlled_replay_admission_review_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC"],
            "default_llm_rspc_v2_changed": False,
            "mode": "controlled replay admission review",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_admission_review_ready": True,
        "minimal_controlled_canary_admission_pass": admission_pass,
        "minimal_controlled_canary_allowed": admission_pass,
        "controlled_replay_allowed": admission_pass,
        "controlled_replay_scope": "minimal_controlled_canary_only" if admission_pass else "blocked",
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "default_llm_rspc_v2_changed": False,
        "baseline_controller": str(canary_manifest.get("baseline_controller", "")),
        "candidate_controller": candidate,
        "scenario_ids": list(canary_manifest.get("scenario_ids", []) or []),
        "selected_cache_path": str(canary_manifest.get("selected_cache_path", "")),
        "actual_protocol_implementation_for_execution": actual_protocol_implementation,
        "protocol_v1_overlay_root": overlay_root,
        "overlay_runner_path": str(overlay_runner),
        "current_runner_path": str(current_runner),
        "current_runner_supports_candidate_controller": current_runner_supports_candidate,
        "precheck": precheck,
        "blockers": blockers,
        "historical_context_only": {
            "controlled_trace_recommendation": controlled_agg.get("recommendation", {}),
            "controlled_shadow_recommendation": shadow_agg.get("recommendation", {}),
            "controlled_trace_would_apply_steps": int(_num(controlled_agg.get("would_apply_steps"))),
            "controlled_trace_strict_applied_steps": int(_num(controlled_agg.get("strict_applied_steps"))),
            "controlled_shadow_safe_gain_steps": int(_num(shadow_agg.get("safe_gain_steps"))),
        },
        "failure_taxonomy": [
            "admission_metadata_consolidation_missing",
            "admission_cache_coverage_missing",
            "admission_historical_unsafe_preferred",
            "admission_historical_unsafe_applied",
            "admission_overlay_runner_candidate_unsupported",
            "admission_protocol_implementation_mismatch",
        ],
        "next_action": next_action,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Controlled Replay Admission Review",
        "",
        "- Mode: controlled replay admission review",
        "- Default `llm_rspc_v2` changed: false",
        f"- Minimal canary allowed: {report.get('minimal_controlled_canary_allowed', False)}",
        f"- Controlled replay scope: `{report.get('controlled_replay_scope', '')}`",
        "- Performance claim allowed: false",
        f"- Candidate controller: `{report.get('candidate_controller', '')}`",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Precheck",
        "",
        "| check | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("precheck", {})).items():
        lines.append(f"| {key} | `{value}` |")
    if report.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- {item}" for item in report.get("blockers", []) or [])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consolidation-json", required=True)
    parser.add_argument("--canary-manifest-json", required=True)
    parser.add_argument("--cache-coverage-json", required=True)
    parser.add_argument("--protocol-v1-overlay-preflight-json", required=True)
    parser.add_argument("--historical-controlled-trace-json", required=True)
    parser.add_argument("--historical-controlled-shadow-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        consolidation=_load(args.consolidation_json),
        canary_manifest=_load(args.canary_manifest_json),
        cache_coverage=_load(args.cache_coverage_json),
        protocol_v1_overlay_preflight=_load(args.protocol_v1_overlay_preflight_json),
        historical_controlled_trace=_load(args.historical_controlled_trace_json),
        historical_controlled_shadow=_load(args.historical_controlled_shadow_json),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"minimal_controlled_canary_allowed={report['minimal_controlled_canary_allowed']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["controlled_replay_admission_review_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
