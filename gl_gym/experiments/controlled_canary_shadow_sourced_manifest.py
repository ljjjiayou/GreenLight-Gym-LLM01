"""Project v26 shadow windows into current strict controlled-canary eligibility."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASELINE_CONTROLLER = "llm_rspc_v2"
CANDIDATE_CONTROLLER = "llm_rspc_v2_hot_dry_proposer_strict"
STRICT_CANDIDATE = "shadow_hot_dry_humidity_retention"
STRICT_MIN_MARGIN = 0.20
STRICT_RH_MAX = 45.0
STRICT_VPD_MIN = 2.25
STRICT_TEMP_MAX = 30.5
STRICT_CANOPY_MARGIN_MIN = 3.0

PRIORITY_WINDOWS = [
    ("y2015_d240_s42_n240", 57, 62),
    ("y2015_d240_s44_n240", 57, 62),
    ("y2015_d180_s42_n240", 37, 41),
    ("y2015_d180_s43_n240", 37, 41),
    ("y2015_d180_s43_n240", 129, 131),
]

V31_EXPECTED_SOURCE_STEPS = {
    "y2015_d120_s42_n240": [147, 149],
    "y2015_d120_s43_n240": [147, 149],
    "y2015_d120_s44_n240": [141, 143, 145, 147, 149],
}

SCENARIO_RE = re.compile(r"^y(?P<year>\d{4})_d(?P<day>\d+)_s(?P<seed>\d+)_n(?P<max_steps>\d+)$")


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


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _scenario_parts(scenario_id: str) -> dict[str, int]:
    match = SCENARIO_RE.match(scenario_id)
    if not match:
        return {"year": 0, "day": 0, "seed": 0, "max_steps": 240}
    return {key: int(value) for key, value in match.groupdict().items()}


def _trace_csv_path(trace_dir: str | Path, scenario_id: str) -> Path:
    return _resolve(trace_dir) / f"{scenario_id}_{BASELINE_CONTROLLER}.csv"


def _action_present(row: Mapping[str, Any]) -> bool:
    fields = [
        "rspc_hot_dry_replay_best_heat",
        "rspc_hot_dry_replay_best_co2",
        "rspc_hot_dry_replay_best_screen",
        "rspc_hot_dry_replay_best_vent",
        "rspc_hot_dry_replay_best_lamp",
        "rspc_hot_dry_replay_best_shade",
    ]
    return any(str(row.get(field, "")).strip() for field in fields)


def _row_step(row: Mapping[str, Any]) -> int:
    return int(_num(row.get("step", row.get("timestep", 0))))


def _strict_failure_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not _bool(row.get("rspc_hot_dry_replay_safe_hot_dry")):
        reasons.append("not_safe_hot_dry")
    if str(row.get("rspc_hot_dry_replay_safety_gate_reason", "") or "").strip().lower() not in {"", "none"}:
        reasons.append("safety_gate_blocked")
    if not _bool(row.get("rspc_hot_dry_replay_would_apply")):
        reasons.append("shadow_not_eligible")
    if str(row.get("rspc_hot_dry_replay_best_alignment", "") or "").strip() != "dry_benefit":
        reasons.append("non_dry_benefit_alignment")
    if _bool(row.get("rspc_hot_dry_replay_unsafe_preferred")):
        reasons.append("unsafe_preferred")
    if _bool(row.get("rspc_hot_dry_replay_unsafe_conflict")):
        reasons.append("unsafe_conflict")
    if _num(row.get("rspc_hot_dry_replay_best_margin")) < STRICT_MIN_MARGIN:
        reasons.append("margin_below_strict_min")
    candidate = str(row.get("rspc_hot_dry_replay_best_candidate_name", "") or "").strip()
    strict_candidate = str(row.get("rspc_hot_dry_proposer_control_strict_candidate", "") or STRICT_CANDIDATE).strip()
    if candidate != (strict_candidate or STRICT_CANDIDATE):
        reasons.append("strict_candidate_mismatch")
    rh_air = _num(row.get("rh_air", row.get("rspc_hot_dry_proposer_control_rh_air")))
    vpd_air = _num(row.get("vpd_air", row.get("rspc_hot_dry_proposer_control_vpd_air")))
    if rh_air > STRICT_RH_MAX or vpd_air < STRICT_VPD_MIN:
        reasons.append("not_severe_dry")
    temp_air = _num(row.get("temp_air", row.get("rspc_hot_dry_proposer_control_temp_air")))
    if temp_air > STRICT_TEMP_MAX:
        reasons.append("temp_headroom_low")
    canopy_margin = _num(row.get("canopy_dew_margin", row.get("rspc_hot_dry_proposer_control_canopy_dew_margin")))
    if canopy_margin < STRICT_CANOPY_MARGIN_MIN:
        reasons.append("canopy_reserve_low")
    if not _action_present(row):
        reasons.append("missing_best_action")
    return reasons


def _first_reason(reasons: Sequence[str]) -> str:
    order = [
        "not_safe_hot_dry",
        "safety_gate_blocked",
        "shadow_not_eligible",
        "non_dry_benefit_alignment",
        "unsafe_preferred",
        "unsafe_conflict",
        "margin_below_strict_min",
        "strict_candidate_mismatch",
        "not_severe_dry",
        "temp_headroom_low",
        "canopy_reserve_low",
        "missing_best_action",
    ]
    for item in order:
        if item in reasons:
            return item
    return "strict_eligible"


def _window_rows(trace_dir: str | Path, scenario_id: str, start_step: int, end_step: int) -> list[dict[str, Any]]:
    path = _trace_csv_path(trace_dir, scenario_id)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = _row_step(row)
            if start_step <= step <= end_step:
                rows.append(dict(row))
    return rows


def _project_window(trace_dir: str | Path, scenario_id: str, start_step: int, end_step: int) -> dict[str, Any]:
    rows = _window_rows(trace_dir, scenario_id, start_step, end_step)
    projected_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    all_reason_counts: dict[str, int] = {}
    strict_steps = 0
    for row in rows:
        reasons = _strict_failure_reasons(row)
        strict_eligible = not reasons
        if strict_eligible:
            strict_steps += 1
        primary_reason = _first_reason(reasons)
        reason_counts[primary_reason] = reason_counts.get(primary_reason, 0) + 1
        for reason in reasons:
            all_reason_counts[reason] = all_reason_counts.get(reason, 0) + 1
        projected_rows.append(
            {
                "step": _row_step(row),
                "strict_eligible": strict_eligible,
                "primary_reason": primary_reason,
                "failure_reasons": reasons,
                "would_apply": _bool(row.get("rspc_hot_dry_replay_would_apply")),
                "candidate": row.get("rspc_hot_dry_replay_best_candidate_name", ""),
                "variant": row.get("rspc_hot_dry_replay_best_variant", ""),
                "margin": round(_num(row.get("rspc_hot_dry_replay_best_margin")), 6),
                "rh_air": round(_num(row.get("rh_air")), 6),
                "vpd_air": round(_num(row.get("vpd_air")), 6),
                "temp_air": round(_num(row.get("temp_air")), 6),
                "canopy_dew_margin": round(_num(row.get("canopy_dew_margin")), 6),
            }
        )
    return {
        "scenario_id": scenario_id,
        "start_step": start_step,
        "end_step": end_step,
        "window_row_count": len(rows),
        "expected_strict_eligible_applied_steps": strict_steps,
        "strict_eligible_projected": strict_steps > 0,
        "primary_reason_counts": dict(sorted(reason_counts.items())),
        "all_reason_counts": dict(sorted(all_reason_counts.items())),
        "projected_rows": projected_rows,
    }


def _window_is_in_v26(shadow_replay: Mapping[str, Any], scenario_id: str, start_step: int, end_step: int) -> bool:
    aggregate = shadow_replay.get("aggregate", {})
    if not isinstance(aggregate, Mapping):
        return False
    for item in aggregate.get("top_replay_windows", []) or []:
        if not isinstance(item, Mapping):
            continue
        if (
            str(item.get("scenario_id", "")) == scenario_id
            and int(_num(item.get("start_step"))) == start_step
            and int(_num(item.get("end_step"))) == end_step
        ):
            return True
    return False


def _strict_source_rows_from_v30(strict_sourcing: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in strict_sourcing.get("sample_rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        if not _bool(row.get("strict_targeted_source")):
            continue
        scenario_id = str(row.get("scenario_id", "") or "")
        step = _row_step(row)
        rows.append(
            {
                "scenario_id": scenario_id,
                "step": step,
                "candidate": str(row.get("candidate", "") or ""),
                "variant": str(row.get("variant", "") or ""),
                "margin": round(_num(row.get("margin")), 6),
                "rh_air": round(_num(row.get("rh_air")), 6),
                "vpd_air": round(_num(row.get("vpd_air")), 6),
                "temp_air": round(_num(row.get("temp_air")), 6),
                "canopy_dew_margin": round(_num(row.get("canopy_dew_margin")), 6),
                "control_safe_hot_dry": _bool(row.get("control_safe_hot_dry")),
                "safety_gate_reason": str(row.get("safety_gate_reason", "none") or "none"),
                "strict_targeted_source": True,
            }
        )
    return sorted(rows, key=lambda item: (item["scenario_id"], item["step"]))


def _v31_source_step_mismatches(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    by_scenario: dict[str, list[int]] = {scenario: [] for scenario in V31_EXPECTED_SOURCE_STEPS}
    unexpected: list[str] = []
    for row in rows:
        scenario_id = str(row.get("scenario_id", "") or "")
        step = int(_num(row.get("step"), -1))
        if scenario_id not in V31_EXPECTED_SOURCE_STEPS:
            unexpected.append(f"unexpected_scenario:{scenario_id}:{step}")
            continue
        if step not in V31_EXPECTED_SOURCE_STEPS[scenario_id]:
            unexpected.append(f"unexpected_step:{scenario_id}:{step}")
            continue
        by_scenario[scenario_id].append(step)
    missing = [
        f"missing_step:{scenario_id}:{step}"
        for scenario_id, expected_steps in V31_EXPECTED_SOURCE_STEPS.items()
        for step in expected_steps
        if step not in by_scenario.get(scenario_id, [])
    ]
    duplicate = [
        f"duplicate_step:{scenario_id}:{step}"
        for scenario_id, steps in by_scenario.items()
        for step in sorted(set(steps))
        if steps.count(step) > 1
    ]
    return sorted(unexpected + missing + duplicate)


def build_v31_report(
    *,
    admission_review: Mapping[str, Any],
    strict_sourcing: Mapping[str, Any],
    selected_cache_path: str,
    output_root: str,
) -> dict[str, Any]:
    admission_pass = bool(admission_review.get("shadow_sourced_admission_pass", False))
    strict_found = bool(strict_sourcing.get("strict_targeted_source_found", False))
    source_rows = _strict_source_rows_from_v30(strict_sourcing)
    row_mismatches = _v31_source_step_mismatches(source_rows)
    selected_scenarios: list[dict[str, Any]] = []
    for scenario_id, steps in sorted(V31_EXPECTED_SOURCE_STEPS.items()):
        parts = _scenario_parts(scenario_id)
        selected_scenarios.append(
            {
                "scenario_id": scenario_id,
                "year": parts["year"],
                "day": parts["day"],
                "seed": parts["seed"],
                "max_steps": parts["max_steps"],
                "role": "v31_strict_source_controlled_canary_candidate",
                "expected_strict_eligible_applied_steps": len(steps),
                "strict_source_steps": list(steps),
            }
        )
    expected_steps = sum(len(steps) for steps in V31_EXPECTED_SOURCE_STEPS.values())
    observed_steps = int(_num(strict_sourcing.get("expected_strict_eligible_applied_steps")))
    strict_rows_exact = bool(not row_mismatches and len(source_rows) == expected_steps and observed_steps == expected_steps)
    ready = bool(admission_pass and strict_found and strict_rows_exact)
    blockers: list[str] = []
    if not admission_pass:
        blockers.append("admission_failed")
    if not strict_found:
        blockers.append("strict_targeted_source_not_found")
    if not strict_rows_exact:
        blockers.append("v30_strict_source_rows_not_exact")
        blockers.extend(row_mismatches)
    command_groups = [
        {
            "group_id": "v31_y2015_d120_s42_s43_s44",
            "years": [2015],
            "days": [120],
            "seeds": [42, 43, 44],
            "expected_scenario_ids": list(V31_EXPECTED_SOURCE_STEPS.keys()),
            "output_json_name": "strict_source_controlled_canary_v31_y2015_d120_s42_s43_s44_20260528.json",
            "cartesian_product_safe": True,
        }
    ]
    return {
        "schema_version": "controlled_canary_shadow_sourced_manifest_v31",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "v31 strict-source controlled canary manifest",
            "reopens_rejected_preset": False,
        },
        "canary_scope": "strict_source_controlled_canary_v31_only",
        "readiness_schema_version": "metadata_replay_readiness_checklist_v31",
        "shadow_sourced_manifest_ready": ready,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "baseline_controller": BASELINE_CONTROLLER,
        "candidate_controller": CANDIDATE_CONTROLLER,
        "controllers": [BASELINE_CONTROLLER, CANDIDATE_CONTROLLER],
        "selected_cache_path": selected_cache_path,
        "output_root": output_root,
        "strict_projection": {
            "strict_candidate": STRICT_CANDIDATE,
            "strict_min_margin": STRICT_MIN_MARGIN,
            "strict_rh_max": STRICT_RH_MAX,
            "strict_vpd_min": STRICT_VPD_MIN,
            "strict_temp_max": STRICT_TEMP_MAX,
            "strict_canopy_margin_min": STRICT_CANOPY_MARGIN_MIN,
            "source": "controlled_canary_strict_targeted_shadow_sourcing_v30",
            "strict_source_row_count": len(source_rows),
            "expected_strict_eligible_applied_steps": observed_steps,
            "required_strict_source_steps": V31_EXPECTED_SOURCE_STEPS,
            "strict_rows_exact": strict_rows_exact,
        },
        "strict_source_rows": source_rows,
        "selected_scenarios": selected_scenarios,
        "scenario_ids": [scenario["scenario_id"] for scenario in selected_scenarios],
        "command_groups": command_groups,
        "blocker_taxonomy": blockers if not ready else [],
        "next_action": "run_v31_cache_coverage_check" if ready else "v31_strict_source_manifest_blocked",
    }


def build_report(
    *,
    admission_review: Mapping[str, Any],
    shadow_replay: Mapping[str, Any],
    trace_dir: str,
    selected_cache_path: str,
    output_root: str,
) -> dict[str, Any]:
    admission_pass = bool(admission_review.get("shadow_sourced_admission_pass", False))
    windows: list[dict[str, Any]] = []
    selected_scenarios: dict[str, dict[str, Any]] = {}
    aggregate_reason_counts: dict[str, int] = {}
    missing_priority_windows: list[str] = []
    for scenario_id, start_step, end_step in PRIORITY_WINDOWS:
        if not _window_is_in_v26(shadow_replay, scenario_id, start_step, end_step):
            missing_priority_windows.append(f"{scenario_id}:{start_step}-{end_step}")
        window = _project_window(trace_dir, scenario_id, start_step, end_step)
        windows.append(window)
        for reason, count in window["all_reason_counts"].items():
            aggregate_reason_counts[reason] = aggregate_reason_counts.get(reason, 0) + int(count)
        if window["strict_eligible_projected"]:
            parts = _scenario_parts(scenario_id)
            selected_scenarios[scenario_id] = {
                "scenario_id": scenario_id,
                "year": parts["year"],
                "day": parts["day"],
                "seed": parts["seed"],
                "max_steps": parts["max_steps"],
                "role": "shadow_sourced_strict_trigger_candidate",
                "expected_strict_eligible_applied_steps": window["expected_strict_eligible_applied_steps"],
                "selected_windows": [
                    {
                        "start_step": start_step,
                        "end_step": end_step,
                        "expected_strict_eligible_applied_steps": window["expected_strict_eligible_applied_steps"],
                    }
                ],
            }
    scenarios = list(selected_scenarios.values())
    command_groups = [
        {
            "group_id": scenario["scenario_id"],
            "years": [scenario["year"]],
            "days": [scenario["day"]],
            "seeds": [scenario["seed"]],
            "expected_scenario_ids": [scenario["scenario_id"]],
            "cartesian_product_safe": True,
        }
        for scenario in scenarios
    ]
    ready = bool(admission_pass and scenarios and not missing_priority_windows)
    blocker_taxonomy: list[str] = []
    if not admission_pass:
        blocker_taxonomy.append("admission_failed")
    if missing_priority_windows:
        blocker_taxonomy.append("priority_window_missing_from_v26")
    if not scenarios:
        blocker_taxonomy.append("no_shadow_sourced_strict_trigger_candidate")
    if aggregate_reason_counts:
        blocker_taxonomy.extend(
            sorted(
                {
                    reason
                    for reason in aggregate_reason_counts
                    if reason
                    not in {
                        "strict_eligible",
                    }
                }
            )
        )
    return {
        "schema_version": "controlled_canary_shadow_sourced_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "shadow-sourced strict trigger manifest",
            "reopens_rejected_preset": False,
        },
        "shadow_sourced_manifest_ready": ready,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "baseline_controller": BASELINE_CONTROLLER,
        "candidate_controller": CANDIDATE_CONTROLLER,
        "controllers": [BASELINE_CONTROLLER, CANDIDATE_CONTROLLER],
        "selected_cache_path": selected_cache_path,
        "output_root": output_root,
        "strict_projection": {
            "strict_candidate": STRICT_CANDIDATE,
            "strict_min_margin": STRICT_MIN_MARGIN,
            "strict_rh_max": STRICT_RH_MAX,
            "strict_vpd_min": STRICT_VPD_MIN,
            "strict_temp_max": STRICT_TEMP_MAX,
            "strict_canopy_margin_min": STRICT_CANOPY_MARGIN_MIN,
            "priority_window_count": len(PRIORITY_WINDOWS),
            "projected_strict_eligible_window_count": sum(1 for window in windows if window["strict_eligible_projected"]),
            "expected_strict_eligible_applied_steps": sum(
                int(window["expected_strict_eligible_applied_steps"]) for window in windows
            ),
            "reason_counts": dict(sorted(aggregate_reason_counts.items())),
        },
        "priority_windows": windows,
        "missing_priority_windows": missing_priority_windows,
        "selected_scenarios": scenarios,
        "scenario_ids": sorted(selected_scenarios),
        "command_groups": command_groups,
        "blocker_taxonomy": blocker_taxonomy if not ready else [],
        "next_action": "run_shadow_sourced_cache_coverage_check" if ready else "shadow_sourced_strict_projection_blocked",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    projection = report.get("strict_projection", {})
    title = (
        "# Shadow-Sourced Strict Trigger Manifest v31"
        if report.get("schema_version") == "controlled_canary_shadow_sourced_manifest_v31"
        else "# Shadow-Sourced Strict Trigger Manifest v27"
    )
    lines = [
        title,
        "",
        f"- Manifest ready: {report.get('shadow_sourced_manifest_ready', False)}",
        "- Controlled replay allowed: false",
        "- Controlled replay execution allowed: false",
        "- Performance claim allowed: false",
        f"- Expected strict-applied steps: `{projection.get('expected_strict_eligible_applied_steps', 0)}`",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Strict Projection Reasons",
        "",
        "| reason | count |",
        "| --- | ---: |",
    ]
    for reason, count in dict(projection.get("reason_counts", {})).items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(["", "## Windows", "", "| scenario | window | expected strict steps | primary reasons |", "| --- | --- | ---: | --- |"])
    for window in report.get("priority_windows", []) or []:
        if isinstance(window, Mapping):
            lines.append(
                "| {scenario} | {start}-{end} | {steps} | `{reasons}` |".format(
                    scenario=window.get("scenario_id", ""),
                    start=window.get("start_step", ""),
                    end=window.get("end_step", ""),
                    steps=window.get("expected_strict_eligible_applied_steps", 0),
                    reasons=", ".join(dict(window.get("primary_reason_counts", {})).keys()),
                )
            )
    lines.extend(["", "## Blockers", ""])
    blockers = report.get("blocker_taxonomy", []) or []
    lines.extend(f"- `{item}`" for item in blockers) if blockers else lines.append("- none")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-review-json", required=True)
    parser.add_argument("--shadow-replay-json", default="")
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--strict-sourcing-json", default="")
    parser.add_argument("--selected-cache-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.strict_sourcing_json:
        report = build_v31_report(
            admission_review=_load(args.admission_review_json),
            strict_sourcing=_load(args.strict_sourcing_json),
            selected_cache_path=args.selected_cache_path,
            output_root=args.output_root,
        )
    else:
        if not args.shadow_replay_json or not args.trace_dir:
            raise SystemExit("--shadow-replay-json and --trace-dir are required unless --strict-sourcing-json is provided")
        report = build_report(
            admission_review=_load(args.admission_review_json),
            shadow_replay=_load(args.shadow_replay_json),
            trace_dir=args.trace_dir,
            selected_cache_path=args.selected_cache_path,
            output_root=args.output_root,
        )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"shadow_sourced_manifest_ready={report['shadow_sourced_manifest_ready']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["shadow_sourced_manifest_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
