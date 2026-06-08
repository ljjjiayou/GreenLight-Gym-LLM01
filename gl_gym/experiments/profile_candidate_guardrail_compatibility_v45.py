"""Build v45 profile/candidate/guardrail compatibility artifacts.

This audit is offline-only. It backfills compatibility labels from existing
v39/v43 traces and v44 causal attribution. It does not run rollout, call online
LLMs, mutate the default controller, authorize controlled replay, or make
performance claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.diagnose_ppo_vs_llm import derive_candidate_guardrail_compatibility


FAILURE_SCENARIOS = (
    "y2010_d180_s43_n720",
    "y2018_d181_s42_n720",
    "y2018_d181_s43_n720",
)

AUDIT_DIR = PROJECT_ROOT / "gl_gym" / "result" / "audits"
ORIGINAL_TRACE_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "stable_long_horizon_evaluation_pool_v38_20260529"
    / "traces_h720"
)
V43_TRACE_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "transition_gate_shadow_rollout_v43_20260530"
    / "traces"
)
V44_CAUSAL_JSON = AUDIT_DIR / "cvodes_failure_causal_attribution_20260531_v44.json"

AUDIT_JSON = AUDIT_DIR / "profile_candidate_guardrail_compatibility_audit_20260531_v45.json"
AUDIT_MD = AUDIT_DIR / "profile_candidate_guardrail_compatibility_audit_20260531_v45.md"
DESIGN_JSON = AUDIT_DIR / "candidate_guardrail_scoring_patch_design_20260531_v45.json"
DESIGN_MD = AUDIT_DIR / "candidate_guardrail_scoring_patch_design_20260531_v45.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260531_v45.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260531_v45.md"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _rel(path: str | Path) -> str:
    p = _resolve(path)
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _parts_from_scenario_id(value: str) -> dict[str, int]:
    match = re.search(r"y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)(?:_n(?P<steps>\d+))?", str(value))
    if not match:
        return {"year": 0, "day": 0, "seed": 0, "max_steps": 720}
    return {
        "year": int(match.group("year")),
        "day": int(match.group("day")),
        "seed": int(match.group("seed")),
        "max_steps": int(match.group("steps") or 720),
    }


def _trace_path(trace_dir: str | Path, scenario_id: str, controller: str) -> Path:
    root = _resolve(trace_dir)
    expected = root / f"{scenario_id}_{controller}.csv"
    if expected.exists():
        return expected
    parts = _parts_from_scenario_id(scenario_id)
    prefix = f"y{parts['year']}_d{parts['day']}_s{parts['seed']}_"
    matches = sorted(root.glob(f"{prefix}*_{controller}.csv"))
    return matches[0] if matches else expected


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    p = _resolve(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _runtime_failure_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    for row in rows:
        text = " ".join(str(row.get(key, "") or "") for key in ("runtime_error", "runtime_error_type", "replan_reason"))
        if text.strip() and (
            "CV_CONV_FAILURE" in text
            or "CVODES" in text
            or "CvodesInterface" in text
            or "runtime_error" in text
            or str(row.get("runtime_error", "") or "").strip()
            or str(row.get("runtime_error_type", "") or "").strip()
        ):
            return int(_num(row.get("step"), -1))
    return None


def _window(rows: Sequence[Mapping[str, Any]], *, center_step: int | None, window_steps: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    if center_step is None:
        center_step = int(_num(rows[-1].get("step"), len(rows) - 1))
    start = max(0, int(center_step) - int(window_steps) + 1)
    return [
        dict(row)
        for row in rows
        if start <= int(_num(row.get("step"), -1)) <= int(center_step)
    ]


def enrich_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out.update(derive_candidate_guardrail_compatibility(out))
        enriched.append(out)
    return enriched


def _split_values(text: Any) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip() and part.strip() != "none"]


def _reason_values(text: Any) -> list[str]:
    return [part.strip() for part in str(text or "").split(";") if part.strip()]


def summarize_compatibility_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    enriched = enrich_rows(rows)
    label_counts: Counter[str] = Counter()
    conflict_counts: Counter[str] = Counter()
    rewrite_field_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    filter_reason_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for row in enriched:
        label = str(row.get("candidate_guardrail_compatibility_label") or "compatible")
        label_counts[label] += 1
        source_counts[str(row.get("candidate_selection_source") or "unknown")] += 1
        for conflict in _split_values(row.get("profile_action_conflict_label")):
            conflict_counts[conflict] += 1
        for field in _split_values(row.get("candidate_guardrail_rewrite_fields")):
            rewrite_field_counts[field] += 1
        for reason in _reason_values(row.get("candidate_filter_reason_summary")):
            filter_reason_counts[reason] += 1
        pressure = (
            _num(row.get("candidate_guardrail_delta_abs_sum")) * 10.0
            + (4.0 if label == "hard_safety_rewrite" else 2.0 if label == "major_rewrite" else 0.5 if label == "minor_rewrite" else 0.0)
            + len(_split_values(row.get("profile_action_conflict_label")))
            + (1.0 if str(row.get("candidate_filter_reason_summary") or "").strip() else 0.0)
        )
        if pressure > 0:
            examples.append(
                {
                    "step": int(_num(row.get("step"), -1)),
                    "pressure_score": float(pressure),
                    "compatibility_label": label,
                    "rewrite_fields": _split_values(row.get("candidate_guardrail_rewrite_fields")),
                    "delta_abs_sum": float(_num(row.get("candidate_guardrail_delta_abs_sum"))),
                    "profile_action_conflict_label": str(row.get("profile_action_conflict_label") or "none"),
                    "candidate_selection_source": str(row.get("candidate_selection_source") or "unknown"),
                    "candidate_filter_reason_summary": str(row.get("candidate_filter_reason_summary") or ""),
                    "tomato_safety_v2_reasons": str(row.get("tomato_safety_v2_reasons") or ""),
                    "target_temp": float(_num(row.get("target_temp"))),
                    "target_rh": float(_num(row.get("target_rh"))),
                    "target_co2": float(_num(row.get("target_co2"))),
                    "u_heating": float(_num(row.get("u_heating"))),
                    "u_ventilation": float(_num(row.get("u_ventilation"))),
                    "u_screen": float(_num(row.get("u_screen"))),
                    "u_shading": float(_num(row.get("u_shading"))),
                }
            )
    examples = sorted(examples, key=lambda item: item["pressure_score"], reverse=True)[:20]
    major_or_hard = int(label_counts.get("major_rewrite", 0) + label_counts.get("hard_safety_rewrite", 0))
    conflict_steps = int(sum(1 for row in enriched if str(row.get("profile_action_conflict_label") or "none") != "none"))
    filter_steps = int(sum(1 for row in enriched if str(row.get("candidate_filter_reason_summary") or "").strip()))
    return {
        "row_count": len(enriched),
        "compatibility_label_counts": dict(sorted(label_counts.items())),
        "profile_action_conflict_counts": dict(sorted(conflict_counts.items())),
        "rewrite_field_counts": dict(sorted(rewrite_field_counts.items())),
        "candidate_selection_source_counts": dict(sorted(source_counts.items())),
        "candidate_filter_reason_counts": dict(sorted(filter_reason_counts.items())),
        "major_or_hard_rewrite_steps": major_or_hard,
        "profile_action_conflict_steps": conflict_steps,
        "candidate_filter_reason_steps": filter_steps,
        "candidate_guardrail_issue_steps": int(major_or_hard + conflict_steps + filter_steps),
        "top_pressure_examples": examples,
    }


def _scenario_report(
    scenario_id: str,
    *,
    original_trace_dir: str | Path,
    v43_trace_dir: str | Path,
    window_steps: int,
) -> dict[str, Any]:
    original_path = _trace_path(original_trace_dir, scenario_id, "llm_rspc_v2")
    v43_path = _trace_path(v43_trace_dir, scenario_id, "llm_rspc_v2")
    original_rows = _read_rows(original_path)
    v43_rows = _read_rows(v43_path)
    original_failure = _runtime_failure_step(original_rows)
    v43_failure = _runtime_failure_step(v43_rows)
    original_window = _window(original_rows, center_step=original_failure, window_steps=window_steps)
    v43_window = _window(v43_rows, center_step=v43_failure, window_steps=window_steps)
    status = "evaluated" if v43_rows else "pending_trace_not_evaluated"
    return {
        "scenario_id": scenario_id,
        "status": status,
        "original_trace": _rel(original_path),
        "v43_trace": _rel(v43_path) if v43_rows else "",
        "original_failure_step": original_failure,
        "v43_failure_step": v43_failure,
        "original_pre_failure": summarize_compatibility_rows(original_window),
        "v43_pre_failure": summarize_compatibility_rows(v43_window) if v43_rows else {
            "row_count": 0,
            "compatibility_label_counts": {},
            "profile_action_conflict_counts": {},
            "rewrite_field_counts": {},
            "candidate_selection_source_counts": {},
            "candidate_filter_reason_counts": {},
            "major_or_hard_rewrite_steps": 0,
            "profile_action_conflict_steps": 0,
            "candidate_filter_reason_steps": 0,
            "candidate_guardrail_issue_steps": 0,
            "top_pressure_examples": [],
        },
    }


def _aggregate_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    conflicts: Counter[str] = Counter()
    rewrites: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    total_rows = 0
    major_or_hard = 0
    conflict_steps = 0
    filter_steps = 0
    for summary in summaries:
        total_rows += int(summary.get("row_count", 0) or 0)
        major_or_hard += int(summary.get("major_or_hard_rewrite_steps", 0) or 0)
        conflict_steps += int(summary.get("profile_action_conflict_steps", 0) or 0)
        filter_steps += int(summary.get("candidate_filter_reason_steps", 0) or 0)
        labels.update(dict(summary.get("compatibility_label_counts", {}) or {}))
        conflicts.update(dict(summary.get("profile_action_conflict_counts", {}) or {}))
        rewrites.update(dict(summary.get("rewrite_field_counts", {}) or {}))
        sources.update(dict(summary.get("candidate_selection_source_counts", {}) or {}))
        reasons.update(dict(summary.get("candidate_filter_reason_counts", {}) or {}))
    issue_steps = major_or_hard + conflict_steps + filter_steps
    top_issue_type = "none"
    issue_scores = {
        "guardrail_rewrite_pressure": major_or_hard,
        "profile_action_conflict": conflict_steps,
        "candidate_filter_pressure": filter_steps,
    }
    if issue_scores:
        top_issue_type = max(issue_scores, key=issue_scores.get)
    return {
        "row_count": int(total_rows),
        "compatibility_label_counts": dict(sorted(labels.items())),
        "profile_action_conflict_counts": dict(sorted(conflicts.items())),
        "rewrite_field_counts": dict(sorted(rewrites.items())),
        "candidate_selection_source_counts": dict(sorted(sources.items())),
        "candidate_filter_reason_counts": dict(sorted(reasons.items())),
        "major_or_hard_rewrite_steps": int(major_or_hard),
        "profile_action_conflict_steps": int(conflict_steps),
        "candidate_filter_reason_steps": int(filter_steps),
        "candidate_guardrail_issue_steps": int(issue_steps),
        "top_issue_type": top_issue_type,
    }


def build_compatibility_audit(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v43_trace_dir: str | Path = V43_TRACE_DIR,
    v44_causal_json: str | Path = V44_CAUSAL_JSON,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    window_steps: int = 120,
) -> dict[str, Any]:
    v44 = _load_json(v44_causal_json)
    reports = [
        _scenario_report(
            scenario_id,
            original_trace_dir=original_trace_dir,
            v43_trace_dir=v43_trace_dir,
            window_steps=window_steps,
        )
        for scenario_id in failure_scenarios
    ]
    aggregate_original = _aggregate_summaries([report["original_pre_failure"] for report in reports])
    aggregate_v43 = _aggregate_summaries([report["v43_pre_failure"] for report in reports if report["status"] == "evaluated"])
    issue_detected = bool(
        aggregate_original["candidate_guardrail_issue_steps"] > 0
        or aggregate_v43["candidate_guardrail_issue_steps"] > 0
    )
    next_action = "shadow_candidate_guardrail_scoring_patch_plan" if issue_detected else "instrumented_three_failure_scenario_shadow_record_plan"
    return {
        "schema_version": "profile_candidate_guardrail_compatibility_audit_20260531_v45",
        "mainline_alignment": {
            "impact_layer": "Evaluation / Safety Boundary / Metadata Trace",
            "default_llm_rspc_v2_changed": False,
            "mode": "offline_profile_candidate_guardrail_compatibility_audit",
        },
        "online_llm_called": False,
        "new_rollout_run": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "source_v44_schema": v44.get("schema_version", ""),
        "source_v44_dominant_attribution": v44.get("dominant_attribution", ""),
        "source_v44_dominant_attribution_clear": bool(v44.get("dominant_attribution_clear", False)),
        "failure_scenarios": list(failure_scenarios),
        "pending_trace_scenarios": [report["scenario_id"] for report in reports if report["status"] == "pending_trace_not_evaluated"],
        "scenario_reports": reports,
        "aggregate_original_pre_failure": aggregate_original,
        "aggregate_v43_pre_failure": aggregate_v43,
        "compatibility_issue_detected": issue_detected,
        "top_issue_type": aggregate_v43["top_issue_type"] if aggregate_v43["candidate_guardrail_issue_steps"] else aggregate_original["top_issue_type"],
        "next_action": next_action,
    }


def build_scoring_patch_design(audit: Mapping[str, Any]) -> dict[str, Any]:
    top_issue = str(audit.get("top_issue_type") or "none")
    return {
        "schema_version": "candidate_guardrail_scoring_patch_design_20260531_v45",
        "design_status": "design_only_not_authorized",
        "default_llm_rspc_v2_changed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "patch_execution_authorized": False,
        "recommended_next_action": "shadow_candidate_guardrail_scoring_patch_plan",
        "primary_issue_type": top_issue,
        "scoring_terms": {
            "guardrail_expected_delta_cost": "penalize candidates whose predicted post-guardrail delta is major/hard in the current regime",
            "profile_action_conflict_penalty": "penalize candidates that conflict with target RH/CO2/temp direction before hard safety",
            "candidate_filter_reason_penalty": "penalize candidates repeatedly filtered by the same safety gate reason",
            "hard_safety_preservation": "hard Tomato Safety/dew/canopy rewrites remain non-negotiable and are not weakened",
            "anchor_hold_bonus": "prefer low-delta anchor/previous-compatible candidate when all candidates are likely to be rewritten",
        },
        "required_shadow_validation": [
            "candidate_guardrail_issue_steps_shadow < original",
            "major_or_hard_rewrite_steps_shadow <= original",
            "profile_action_conflict_steps_shadow < original",
            "no new canopy/dew hard violation",
            "runtime_error_steps does not increase",
        ],
        "blocked_actions": [
            "do not tune controller thresholds in v45",
            "do not run controlled replay",
            "do not claim performance improvement",
        ],
    }


def build_readiness(audit: Mapping[str, Any], design: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "metadata_replay_readiness_checklist_20260531_v45",
        "stage": "v45_profile_candidate_guardrail_compatibility",
        "compatibility_audit_complete": True,
        "compatibility_issue_detected": bool(audit.get("compatibility_issue_detected", False)),
        "top_issue_type": audit.get("top_issue_type", "none"),
        "pending_trace_scenarios": list(audit.get("pending_trace_scenarios", [])),
        "default_llm_rspc_v2_changed": False,
        "online_llm_called": False,
        "new_rollout_run": False,
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "patch_execution_authorized": bool(design.get("patch_execution_authorized", False)),
        "next_action": audit.get("next_action", "instrumented_three_failure_scenario_shadow_record_plan"),
    }


def _write_json_md(path_json: Path, path_md: Path, data: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    lines = [f"# {title}", ""]
    for key in (
        "schema_version",
        "stage",
        "design_status",
        "compatibility_issue_detected",
        "top_issue_type",
        "source_v44_dominant_attribution",
        "next_action",
        "recommended_next_action",
    ):
        if key in data:
            lines.append(f"- `{key}`: `{data[key]}`")
    for key in (
        "online_llm_called",
        "new_rollout_run",
        "controlled_replay_allowed",
        "controlled_replay_execution_allowed",
        "performance_claim_allowed",
        "promotion_evidence",
        "patch_execution_authorized",
    ):
        if key in data:
            lines.append(f"- `{key}`: `{str(data[key]).lower()}`")
    if "pending_trace_scenarios" in data:
        lines.append(f"- `pending_trace_scenarios`: `{data.get('pending_trace_scenarios')}`")
    if "aggregate_original_pre_failure" in data:
        lines.extend(
            [
                "",
                "## Aggregate Original Pre-Failure",
                "```json",
                json.dumps(data["aggregate_original_pre_failure"], indent=2, sort_keys=True),
                "```",
            ]
        )
    if "aggregate_v43_pre_failure" in data:
        lines.extend(
            [
                "",
                "## Aggregate V43 Pre-Failure",
                "```json",
                json.dumps(data["aggregate_v43_pre_failure"], indent=2, sort_keys=True),
                "```",
            ]
        )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(
    *,
    original_trace_dir: str | Path = ORIGINAL_TRACE_DIR,
    v43_trace_dir: str | Path = V43_TRACE_DIR,
    v44_causal_json: str | Path = V44_CAUSAL_JSON,
    failure_scenarios: Sequence[str] = FAILURE_SCENARIOS,
    window_steps: int = 120,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit = build_compatibility_audit(
        original_trace_dir=original_trace_dir,
        v43_trace_dir=v43_trace_dir,
        v44_causal_json=v44_causal_json,
        failure_scenarios=failure_scenarios,
        window_steps=window_steps,
    )
    design = build_scoring_patch_design(audit)
    readiness = build_readiness(audit, design)
    _write_json_md(AUDIT_JSON, AUDIT_MD, audit, "v45 Profile-Candidate-Guardrail Compatibility Audit")
    _write_json_md(DESIGN_JSON, DESIGN_MD, design, "v45 Candidate-Guardrail Scoring Patch Design")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v45 Metadata Replay Readiness")
    return audit, design, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-trace-dir", type=str, default=str(ORIGINAL_TRACE_DIR))
    parser.add_argument("--v43-trace-dir", type=str, default=str(V43_TRACE_DIR))
    parser.add_argument("--v44-causal-json", type=str, default=str(V44_CAUSAL_JSON))
    parser.add_argument("--failure-scenario", action="append", default=[])
    parser.add_argument("--window-steps", type=int, default=120)
    parser.add_argument("--write-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios = args.failure_scenario or list(FAILURE_SCENARIOS)
    audit, design, readiness = write_all(
        original_trace_dir=args.original_trace_dir,
        v43_trace_dir=args.v43_trace_dir,
        v44_causal_json=args.v44_causal_json,
        failure_scenarios=scenarios,
        window_steps=int(args.window_steps),
    )
    if not args.write_all:
        print(json.dumps({"audit": audit, "design": design, "readiness": readiness}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
