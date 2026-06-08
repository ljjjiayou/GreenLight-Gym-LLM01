"""Discover existing trace rows that can trigger strict controlled canaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.profile_scorer_decision_audit import discover_traces, read_trace, trace_identity  # noqa: E402

STRICT_FILTER_REASONS = {
    "candidate_not_allowed",
    "margin_below_strict_min",
    "not_severe_dry",
    "temp_headroom_low",
    "canopy_reserve_low",
}

BASELINE_CONTROLLER = "llm_rspc_v2"
STRICT_TARGET_CANDIDATE = "shadow_hot_dry_humidity_retention"
STRICT_MIN_MARGIN = 0.20
STRICT_RH_MAX = 45.0
STRICT_VPD_MIN = 2.25
STRICT_TEMP_MAX = 30.5
STRICT_CANOPY_MARGIN_MIN = 3.0


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _truthy(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _first_num(row: Mapping[str, Any], keys: Sequence[str], default: float = 0.0) -> float:
    for key in keys:
        value = row.get(key, "")
        if value is not None and value != "":
            return _num(row, key, default)
    return float(default)


def _first_text(row: Mapping[str, Any], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return default


def _counts(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _scenario_parts(scenario_id: str) -> dict[str, int]:
    parts = {"year": 0, "day": 0, "seed": 0, "max_steps": 240}
    for token in str(scenario_id).split("_"):
        try:
            if token.startswith("y"):
                parts["year"] = int(token[1:])
            elif token.startswith("d"):
                parts["day"] = int(token[1:])
            elif token.startswith("s"):
                parts["seed"] = int(token[1:])
            elif token.startswith("n"):
                parts["max_steps"] = int(token[1:])
        except ValueError:
            continue
    return parts


def _row_record(path: Path, identity: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    control_reason = str(row.get("rspc_hot_dry_proposer_control_reason", "") or "")
    safety_gate = str(row.get("rspc_hot_dry_proposer_control_safety_gate_reason", "none") or "none")
    control_applied = _truthy(row, "rspc_hot_dry_proposer_control_applied")
    control_strict = _truthy(row, "rspc_hot_dry_proposer_control_strict_enabled")
    control_safe = _truthy(row, "rspc_hot_dry_proposer_control_safe_hot_dry")
    strict_eligible_applied = bool(control_applied and control_strict and control_safe and safety_gate in {"", "none"})
    strict_filtered = bool(control_strict and control_reason in STRICT_FILTER_REASONS)
    return {
        "trace_path": str(path),
        "trace_id": identity.get("trace_id", ""),
        "scenario_id": identity.get("scenario_id", ""),
        "year": identity.get("year", 0),
        "day": identity.get("day", 0),
        "seed": identity.get("seed", 0),
        "max_steps": identity.get("max_steps", 240),
        "controller": identity.get("controller", ""),
        "step": int(_num(row, "step", 0.0)),
        "would_apply": _truthy(row, "rspc_hot_dry_replay_would_apply"),
        "strict_filtered": strict_filtered,
        "strict_eligible_applied": strict_eligible_applied,
        "control_applied": control_applied,
        "control_strict_enabled": control_strict,
        "control_safe_hot_dry": control_safe,
        "candidate": str(
            row.get("rspc_hot_dry_proposer_control_candidate")
            or row.get("rspc_hot_dry_replay_best_candidate_name")
            or row.get("rspc_hot_dry_proposer_control_strict_candidate")
            or ""
        ),
        "strict_candidate": str(row.get("rspc_hot_dry_proposer_control_strict_candidate", "") or ""),
        "variant": str(row.get("rspc_hot_dry_proposer_control_variant") or row.get("rspc_hot_dry_replay_best_variant") or ""),
        "reason": control_reason or str(row.get("rspc_hot_dry_replay_reason", "") or ""),
        "safety_gate_reason": safety_gate,
        "margin": round(_num(row, "rspc_hot_dry_proposer_control_margin", _num(row, "rspc_hot_dry_replay_best_margin")), 6),
        "min_margin": round(_num(row, "rspc_hot_dry_proposer_control_min_margin", 0.0), 6),
        "temp_air": round(_num(row, "rspc_hot_dry_proposer_control_temp_air", _num(row, "temp_air")), 6),
        "rh_air": round(_num(row, "rspc_hot_dry_proposer_control_rh_air", _num(row, "rh_air")), 6),
        "vpd_air": round(_num(row, "rspc_hot_dry_proposer_control_vpd_air", _num(row, "vpd_air")), 6),
        "canopy_dew_margin": round(
            _num(row, "rspc_hot_dry_proposer_control_canopy_dew_margin", _num(row, "canopy_dew_margin")), 6
        ),
    }


def _strict_targeted_failure_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    replay_safe = _truthy(row, "rspc_hot_dry_replay_safe_hot_dry") or _truthy(
        row, "rspc_hot_dry_proposer_control_safe_hot_dry"
    )
    safety_gate = _first_text(
        row,
        ["rspc_hot_dry_replay_safety_gate_reason", "rspc_hot_dry_proposer_control_safety_gate_reason"],
        "none",
    ).lower()
    if not replay_safe:
        reasons.append("not_safe_hot_dry")
    if safety_gate not in {"", "none"}:
        reasons.append("safety_gate_blocked")
    if not _truthy(row, "rspc_hot_dry_replay_would_apply"):
        reasons.append("shadow_not_eligible")
    if _first_text(row, ["rspc_hot_dry_replay_best_alignment"], "") != "dry_benefit":
        reasons.append("non_dry_benefit_alignment")
    if _truthy(row, "rspc_hot_dry_replay_unsafe_preferred"):
        reasons.append("unsafe_preferred")
    if _truthy(row, "rspc_hot_dry_replay_unsafe_conflict"):
        reasons.append("unsafe_conflict")
    margin = _first_num(row, ["rspc_hot_dry_replay_best_margin", "rspc_hot_dry_proposer_control_margin"])
    if margin < STRICT_MIN_MARGIN:
        reasons.append("margin_below_strict_min")
    candidate = _first_text(
        row,
        ["rspc_hot_dry_replay_best_candidate_name", "rspc_hot_dry_proposer_control_candidate"],
    )
    if candidate != STRICT_TARGET_CANDIDATE:
        reasons.append("strict_candidate_mismatch")
    rh_air = _first_num(row, ["rh_air", "rspc_hot_dry_proposer_control_rh_air"])
    vpd_air = _first_num(row, ["vpd_air", "rspc_hot_dry_proposer_control_vpd_air"])
    if rh_air > STRICT_RH_MAX or vpd_air < STRICT_VPD_MIN:
        reasons.append("not_severe_dry")
    temp_air = _first_num(row, ["temp_air", "rspc_hot_dry_proposer_control_temp_air"])
    if temp_air > STRICT_TEMP_MAX:
        reasons.append("temp_headroom_low")
    canopy_margin = _first_num(row, ["canopy_dew_margin", "rspc_hot_dry_proposer_control_canopy_dew_margin"])
    if canopy_margin < STRICT_CANOPY_MARGIN_MIN:
        reasons.append("canopy_reserve_low")
    return reasons


def _strict_targeted_row_record(path: Path, identity: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    reasons = _strict_targeted_failure_reasons(row)
    strict_source = not reasons
    candidate = _first_text(
        row,
        ["rspc_hot_dry_replay_best_candidate_name", "rspc_hot_dry_proposer_control_candidate"],
    )
    safety_gate = _first_text(
        row,
        ["rspc_hot_dry_replay_safety_gate_reason", "rspc_hot_dry_proposer_control_safety_gate_reason"],
        "none",
    )
    return {
        "trace_path": str(path),
        "trace_id": identity.get("trace_id", ""),
        "scenario_id": identity.get("scenario_id", ""),
        "year": identity.get("year", 0),
        "day": identity.get("day", 0),
        "seed": identity.get("seed", 0),
        "max_steps": identity.get("max_steps", 240),
        "controller": identity.get("controller", ""),
        "step": int(_num(row, "step", 0.0)),
        "would_apply": _truthy(row, "rspc_hot_dry_replay_would_apply"),
        "strict_filtered": bool(reasons and _truthy(row, "rspc_hot_dry_replay_would_apply")),
        "strict_eligible_applied": strict_source,
        "strict_targeted_source": strict_source,
        "control_applied": False,
        "control_strict_enabled": False,
        "control_safe_hot_dry": _truthy(row, "rspc_hot_dry_replay_safe_hot_dry"),
        "candidate": candidate,
        "strict_candidate": STRICT_TARGET_CANDIDATE,
        "variant": _first_text(row, ["rspc_hot_dry_replay_best_variant", "rspc_hot_dry_proposer_control_variant"]),
        "reason": "strict_targeted_shadow_source" if strict_source else reasons[0],
        "failure_reasons": reasons,
        "safety_gate_reason": safety_gate,
        "margin": round(_first_num(row, ["rspc_hot_dry_replay_best_margin", "rspc_hot_dry_proposer_control_margin"]), 6),
        "min_margin": STRICT_MIN_MARGIN,
        "temp_air": round(_first_num(row, ["temp_air", "rspc_hot_dry_proposer_control_temp_air"]), 6),
        "rh_air": round(_first_num(row, ["rh_air", "rspc_hot_dry_proposer_control_rh_air"]), 6),
        "vpd_air": round(_first_num(row, ["vpd_air", "rspc_hot_dry_proposer_control_vpd_air"]), 6),
        "canopy_dew_margin": round(
            _first_num(row, ["canopy_dew_margin", "rspc_hot_dry_proposer_control_canopy_dew_margin"]), 6
        ),
    }


def discover_strict_triggers(
    inputs: Sequence[str | Path],
    *,
    max_records: int = 200,
    strict_targeted_shadow_only: bool = False,
    excluded_scenario_ids: Iterable[str] = (),
) -> dict[str, Any]:
    relevant_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    trigger_scenario_ids: set[str] = set()
    excluded = {str(item) for item in excluded_scenario_ids if str(item or "")}
    excluded_trace_count = 0
    for path in discover_traces(inputs):
        identity = trace_identity(path)
        scenario_id = str(identity.get("scenario_id", "") or "")
        controller = str(identity.get("controller", "") or "")
        if scenario_id in excluded:
            excluded_trace_count += 1
            continue
        if strict_targeted_shadow_only and controller != BASELINE_CONTROLLER:
            continue
        rows = read_trace(path)
        trace_rows = []
        for row in rows:
            if "rspc_hot_dry_replay_enabled" not in row and "rspc_hot_dry_proposer_control_enabled" not in row:
                continue
            record = (
                _strict_targeted_row_record(path, identity, row)
                if strict_targeted_shadow_only
                else _row_record(path, identity, row)
            )
            if record["strict_eligible_applied"]:
                trigger_scenario_ids.add(str(record.get("scenario_id", "")))
            if record["would_apply"] or record["strict_filtered"] or record["strict_eligible_applied"]:
                trace_rows.append(record)
                if len(relevant_rows) < max_records:
                    relevant_rows.append(record)
        traces.append(
            {
                **identity,
                "rows": len(rows),
                "relevant_rows": len(trace_rows),
                "would_apply_rows": sum(1 for item in trace_rows if item["would_apply"]),
                "strict_filtered_rows": sum(1 for item in trace_rows if item["strict_filtered"]),
                "expected_strict_eligible_applied_rows": sum(1 for item in trace_rows if item["strict_eligible_applied"]),
                "reason_counts": _counts(str(item.get("reason", "")) for item in trace_rows),
                "candidate_counts": _counts(str(item.get("candidate", "")) for item in trace_rows),
                "strict_targeted_source_rows": sum(1 for item in trace_rows if item.get("strict_targeted_source", False)),
            }
        )

    expected_count = sum(int(trace.get("expected_strict_eligible_applied_rows", 0) or 0) for trace in traces)
    would_apply_count = sum(int(trace.get("would_apply_rows", 0) or 0) for trace in traces)
    strict_filtered_count = sum(int(trace.get("strict_filtered_rows", 0) or 0) for trace in traces)
    qualified_ids = sorted(scenario_id for scenario_id in trigger_scenario_ids if scenario_id)
    blockers: list[str] = []
    if expected_count <= 0:
        blockers.append(
            "strict_targeted_source_not_found"
            if strict_targeted_shadow_only
            else "strict_not_triggered_under_current_rules"
        )
        if strict_filtered_count > 0:
            blockers.append("strict_candidates_filtered")
        if would_apply_count > 0:
            blockers.append("only_shadow_would_apply_or_non_strict_signal")
        if would_apply_count <= 0 and strict_filtered_count <= 0:
            blockers.append("no_hot_dry_replay_signal")

    decision = (
        "strict_targeted_source_found"
        if strict_targeted_shadow_only and expected_count > 0
        else "trigger_candidates_found"
        if expected_count > 0
        else "blocked_no_strict_targeted_source"
        if strict_targeted_shadow_only
        else "blocked_no_strict_trigger_candidate"
    )
    scenarios = [
        {
            "scenario_id": scenario_id,
            **_scenario_parts(scenario_id),
            "discovery_role": "strict_targeted_shadow_source" if strict_targeted_shadow_only else "strict_trigger_candidate",
        }
        for scenario_id in qualified_ids
    ]
    return {
        "schema_version": (
            "controlled_canary_strict_targeted_shadow_sourcing_v1"
            if strict_targeted_shadow_only
            else "controlled_canary_strict_trigger_discovery_v1"
        ),
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": (
                "strict-targeted shadow-only sourcing"
                if strict_targeted_shadow_only
                else "read-only strict trigger discovery"
            ),
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "strict_targeted_shadow_only": strict_targeted_shadow_only,
        "strict_target_gates": {
            "candidate": STRICT_TARGET_CANDIDATE,
            "min_margin": STRICT_MIN_MARGIN,
            "rh_max": STRICT_RH_MAX,
            "vpd_min": STRICT_VPD_MIN,
            "temp_max": STRICT_TEMP_MAX,
            "canopy_dew_margin_min": STRICT_CANOPY_MARGIN_MIN,
        },
        "excluded_scenario_ids": sorted(excluded),
        "excluded_trace_count": int(excluded_trace_count),
        "discovery_decision": decision,
        "trigger_candidates_found": expected_count > 0,
        "strict_targeted_source_found": bool(strict_targeted_shadow_only and expected_count > 0),
        "expected_strict_eligible_applied_steps": int(expected_count),
        "would_apply_steps": int(would_apply_count),
        "strict_filtered_steps": int(strict_filtered_count),
        "qualified_trigger_scenarios": scenarios,
        "strict_targeted_source_scenarios": scenarios if strict_targeted_shadow_only else [],
        "blocker_taxonomy": blockers,
        "trace_count": len(traces),
        "traces": traces,
        "sample_rows": relevant_rows,
        "next_action": (
            "strict_targeted_source_manifest_and_cache_precheck"
            if strict_targeted_shadow_only and expected_count > 0
            else "strict_targeted_source_not_found"
            if strict_targeted_shadow_only
            else "triggered_scenario_manifest_and_cache_coverage"
            if expected_count > 0
            else "strict_trigger_blocker_diagnosis"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Controlled Canary Strict Trigger Discovery",
        "",
        f"- Decision: `{report.get('discovery_decision', '')}`",
        f"- Trigger candidates found: {report.get('trigger_candidates_found', False)}",
        f"- Expected strict-eligible applied steps: {report.get('expected_strict_eligible_applied_steps', 0)}",
        f"- Would-apply steps: {report.get('would_apply_steps', 0)}",
        f"- Strict-filtered steps: {report.get('strict_filtered_steps', 0)}",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = report.get("blocker_taxonomy", []) or []
    lines.extend(f"- `{item}`" for item in blockers)
    if not blockers:
        lines.append("- none")
    lines.extend(["", "## Qualified Scenarios", "", "| scenario | role |", "| --- | --- |"])
    for scenario in report.get("qualified_trigger_scenarios", []) or []:
        if isinstance(scenario, Mapping):
            lines.append(f"| {scenario.get('scenario_id', '')} | {scenario.get('discovery_role', '')} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-trace", nargs="+", required=True)
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--strict-targeted-shadow-only", action="store_true")
    parser.add_argument("--exclude-scenario-id", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = discover_strict_triggers(
        args.input_trace,
        max_records=int(args.max_records),
        strict_targeted_shadow_only=bool(args.strict_targeted_shadow_only),
        excluded_scenario_ids=[str(item) for item in args.exclude_scenario_id],
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"discovery_decision={report['discovery_decision']}")
    print(f"expected_strict_eligible_applied_steps={report['expected_strict_eligible_applied_steps']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
