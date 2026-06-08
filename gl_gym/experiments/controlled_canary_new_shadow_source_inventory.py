"""Scan existing traces for new shadow-only strict-source cache acquisition candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.controlled_canary_strict_trigger_discovery import (  # noqa: E402
    _counts,
    _strict_targeted_row_record,
    _truthy,
)
from gl_gym.experiments.profile_scorer_decision_audit import discover_traces, read_trace, trace_identity  # noqa: E402

BASELINE_CONTROLLER = "llm_rspc_v2"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _load_optional(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = _resolve(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _scenario_parts(scenario_id: str) -> dict[str, int]:
    out = {"year": 0, "day": 0, "seed": 0, "max_steps": 240}
    for token in str(scenario_id).split("_"):
        try:
            if token.startswith("y"):
                out["year"] = int(token[1:])
            elif token.startswith("d"):
                out["day"] = int(token[1:])
            elif token.startswith("s"):
                out["seed"] = int(token[1:])
            elif token.startswith("n"):
                out["max_steps"] = int(token[1:])
        except ValueError:
            continue
    return out


def _value_num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _stress_manifest_scenarios(stress_manifest: Mapping[str, Any]) -> set[str]:
    scenarios: set[str] = set()
    for item in stress_manifest.get("scenarios", []) or []:
        if isinstance(item, Mapping) and item.get("scenario_id"):
            scenarios.add(str(item.get("scenario_id")))
    return scenarios


def _row_gap(row: Mapping[str, Any]) -> float:
    return max(0.0, _value_num(row.get("min_margin"), 0.2) - _value_num(row.get("margin"), 0.0))


def _candidate_summary(scenario_id: str, rows: Sequence[Mapping[str, Any]], *, stress_manifest_member: bool) -> dict[str, Any]:
    parts = _scenario_parts(scenario_id)
    strict_rows = [row for row in rows if bool(row.get("strict_targeted_source", False))]
    near_rows = [row for row in rows if row not in strict_rows]
    return {
        "scenario_id": scenario_id,
        **parts,
        "stress_manifest_member": bool(stress_manifest_member),
        "expected_strict_eligible_applied_steps": len(strict_rows),
        "near_miss_rows": len(near_rows),
        "source_rows": strict_rows,
        "near_miss_sample_rows": near_rows[:12],
        "candidate_counts": _counts(str(row.get("candidate", "")) for row in rows),
        "variant_counts": _counts(str(row.get("variant", "")) for row in rows),
        "reason_counts": _counts(str(row.get("reason", "")) for row in rows),
        "min_margin": min((_value_num(row.get("margin")) for row in rows), default=0.0),
        "max_margin": max((_value_num(row.get("margin")) for row in rows), default=0.0),
        "min_rh_air": min((_value_num(row.get("rh_air")) for row in rows), default=0.0),
        "max_vpd_air": max((_value_num(row.get("vpd_air")) for row in rows), default=0.0),
        "max_temp_air": max((_value_num(row.get("temp_air")) for row in rows), default=0.0),
        "min_canopy_dew_margin": min((_value_num(row.get("canopy_dew_margin")) for row in rows), default=0.0),
        "min_margin_gap": min((_row_gap(row) for row in near_rows), default=0.0),
        "max_margin_gap": max((_row_gap(row) for row in near_rows), default=0.0),
        "source_hint_only": True,
        "cache_acquisition_candidate": bool(strict_rows),
    }


def build_report(
    *,
    target_spec: Mapping[str, Any],
    trace_inputs: Sequence[str | Path],
    stress_manifest: Mapping[str, Any] | None = None,
    max_records: int = 500,
) -> dict[str, Any]:
    excluded = {str(item) for item in target_spec.get("excluded_scenario_ids", []) or [] if str(item or "")}
    stress_scenarios = _stress_manifest_scenarios(stress_manifest or {})
    trace_count = 0
    skipped_excluded_trace_count = 0
    relevant_rows: list[dict[str, Any]] = []
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for path in discover_traces([_resolve(item) for item in trace_inputs]):
        identity = trace_identity(path)
        scenario_id = str(identity.get("scenario_id", "") or "")
        controller = str(identity.get("controller", "") or "")
        if controller != BASELINE_CONTROLLER:
            continue
        if scenario_id in excluded:
            skipped_excluded_trace_count += 1
            continue
        trace_count += 1
        for row in read_trace(path):
            if "rspc_hot_dry_replay_enabled" not in row and "rspc_hot_dry_proposer_control_enabled" not in row:
                continue
            record = _strict_targeted_row_record(path, identity, row)
            unsafe = bool(
                _truthy(row, "rspc_hot_dry_replay_unsafe_preferred")
                or _truthy(row, "rspc_hot_dry_replay_unsafe_conflict")
            )
            record["unsafe_source_row"] = unsafe
            if record["strict_targeted_source"] and not unsafe:
                by_scenario.setdefault(scenario_id, []).append(record)
                if len(relevant_rows) < max_records:
                    relevant_rows.append(record)
            elif record["would_apply"] and record["strict_filtered"] and not unsafe:
                # Retain a small source-hint trail; near-miss rows never authorize cache by themselves.
                if len(relevant_rows) < max_records:
                    relevant_rows.append(record)

    candidates = [
        _candidate_summary(scenario_id, rows, stress_manifest_member=scenario_id in stress_scenarios)
        for scenario_id, rows in sorted(by_scenario.items())
    ]
    candidates = [item for item in candidates if int(item.get("expected_strict_eligible_applied_steps", 0)) > 0]
    return {
        "schema_version": "controlled_canary_new_shadow_source_inventory_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only new shadow source inventory",
            "reopens_rejected_preset": False,
        },
        "new_shadow_source_inventory_ready": bool(target_spec.get("target_spec_ready", False)),
        "read_only_trace_inventory": True,
        "cache_fill_run": False,
        "online_llm_called": False,
        "cache_fill_authorized": False,
        "online_llm_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "trace_inputs": [str(_resolve(item)) for item in trace_inputs],
        "trace_count": trace_count,
        "skipped_excluded_trace_count": skipped_excluded_trace_count,
        "excluded_scenario_count": len(excluded),
        "stress_manifest_scenario_count": len(stress_scenarios),
        "candidate_scenario_count": len(candidates),
        "new_shadow_source_candidates_found": bool(candidates),
        "candidate_scenarios": candidates,
        "candidate_scenario_ids": [item["scenario_id"] for item in candidates],
        "sample_rows": relevant_rows,
        "next_action": (
            "shadow_cache_acquisition_request"
            if candidates
            else "external_weather_or_scenario_space_expansion_design"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# New Shadow Source Inventory v34",
        "",
        f"- Inventory ready: {report.get('new_shadow_source_inventory_ready', False)}",
        "- Read-only trace inventory: true",
        "- Cache fill authorized: false",
        "- Online LLM allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Trace count: {report.get('trace_count', 0)}",
        f"- Candidate scenarios: {report.get('candidate_scenario_count', 0)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Candidates",
        "",
        "| scenario | expected strict steps | stress manifest | margin range | RH min | VPD max | canopy min |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for item in report.get("candidate_scenarios", []) or []:
        if isinstance(item, Mapping):
            lines.append(
                "| {scenario} | {steps} | {stress} | {min_margin:.3f}..{max_margin:.3f} | {rh:.3f} | {vpd:.3f} | {canopy:.3f} |".format(
                    scenario=item.get("scenario_id", ""),
                    steps=item.get("expected_strict_eligible_applied_steps", 0),
                    stress=item.get("stress_manifest_member", False),
                    min_margin=_value_num(item.get("min_margin")),
                    max_margin=_value_num(item.get("max_margin")),
                    rh=_value_num(item.get("min_rh_air")),
                    vpd=_value_num(item.get("max_vpd_air")),
                    canopy=_value_num(item.get("min_canopy_dew_margin")),
                )
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-spec-json", required=True)
    parser.add_argument("--input-trace", nargs="+", required=True)
    parser.add_argument("--stress-manifest-json", default="")
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        target_spec=_load(args.target_spec_json),
        trace_inputs=args.input_trace,
        stress_manifest=_load_optional(args.stress_manifest_json),
        max_records=int(args.max_records),
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"new_shadow_source_candidates_found={report['new_shadow_source_candidates_found']}")
    print(f"candidate_scenario_count={report['candidate_scenario_count']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
