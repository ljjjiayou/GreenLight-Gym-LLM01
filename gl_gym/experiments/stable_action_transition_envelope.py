"""Build v41 stable action transition envelope and shadow gate design.

The audit is read-only. It consumes existing H720 PPO/LLM traces and v40
attribution artifacts to learn which action transitions are usually stable.
It does not run rollouts, call online LLMs, authorize controlled replay, or
mutate the default controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ACTION_FIELDS = ("u_heating", "u_ventilation", "u_co2", "u_lighting", "u_screen", "u_shading")
DEFAULT_FAILURE_SCENARIOS = (
    "y2010_d180_s43_n720",
    "y2018_d181_s42_n720",
    "y2018_d181_s43_n720",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "applied"}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _trace_id(path: Path) -> tuple[str, str] | None:
    match = re.match(r"(?P<scenario>.+)_(?P<controller>ppo|llm_rspc_v2)\.csv$", path.name)
    if not match:
        return None
    return match.group("scenario"), match.group("controller")


def discover_trace_pairs(trace_dir: str | Path) -> dict[str, dict[str, Path]]:
    root = _resolve(trace_dir)
    pairs: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(root.glob("*.csv")):
        parsed = _trace_id(path)
        if parsed is None:
            continue
        scenario_id, controller = parsed
        pairs[scenario_id][controller] = path
    return dict(sorted(pairs.items()))


def _runtime_failure_step(rows: Sequence[Mapping[str, Any]]) -> int | None:
    for row in rows:
        error_text = " ".join(
            str(row.get(key, "") or "") for key in ("runtime_error", "runtime_error_type", "replan_reason")
        )
        if "CV_CONV_FAILURE" in error_text or "CVODES" in error_text or "CvodesInterface" in error_text:
            return int(_num(row.get("step"), -1))
    return None


def _guardrail_changed(row: Mapping[str, Any]) -> bool:
    if _truthy(row.get("tomato_safety_v2_applied")):
        return True
    for name in ("heat", "vent", "screen", "shade"):
        before = _num(row.get(f"tomato_safety_v2_{name}_before"))
        after = _num(row.get(f"tomato_safety_v2_{name}_after"))
        if abs(after - before) > 1e-6:
            return True
    return False


def _regime(row: Mapping[str, Any]) -> str:
    if _truthy(row.get("dew_risk")) or _num(row.get("canopy_dew_margin"), 99.0) < 3.0:
        return "dew_or_canopy_risk"
    if _truthy(row.get("dry_risk")) or (_num(row.get("vpd_air")) >= 2.25 and _num(row.get("rh_air"), 100.0) <= 45.0):
        return "hot_dry_pressure"
    if _num(row.get("rh_air")) >= 85.0:
        return "high_humidity"
    if _num(row.get("temp_air")) >= 28.0:
        return "warm_or_hot"
    return "neutral_or_mild"


def _is_screen_vent_conflict(row: Mapping[str, Any]) -> bool:
    return _num(row.get("u_ventilation")) >= 0.85 and _num(row.get("u_screen")) >= 0.30


def _is_shade_screen_conflict(row: Mapping[str, Any]) -> bool:
    return _num(row.get("u_shading")) >= 0.80 and _num(row.get("u_screen")) >= 0.30


def _failure_window_steps(rows: Sequence[Mapping[str, Any]], window_steps: int) -> set[int]:
    failure_step = _runtime_failure_step(rows)
    if failure_step is None:
        return set()
    start = max(0, failure_step - window_steps + 1)
    return set(range(start, failure_step + 1))


def build_transition_rows(
    *,
    trace_dir: str | Path,
    failure_scenarios: Sequence[str] = DEFAULT_FAILURE_SCENARIOS,
    failure_window_steps: int = 60,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pairs = discover_trace_pairs(trace_dir)
    stable_rows: list[dict[str, Any]] = []
    unstable_rows: list[dict[str, Any]] = []
    missing_pairs: list[str] = []
    failure_set = set(failure_scenarios)
    for scenario_id, controllers in pairs.items():
        if "ppo" not in controllers or "llm_rspc_v2" not in controllers:
            missing_pairs.append(scenario_id)
            continue
        for controller in ("ppo", "llm_rspc_v2"):
            rows = _read_rows(controllers[controller])
            if len(rows) < 2:
                continue
            negative_steps = (
                _failure_window_steps(rows, failure_window_steps)
                if controller == "llm_rspc_v2" and scenario_id in failure_set
                else set()
            )
            previous_values: dict[str, float] = {}
            previous_signs: dict[str, int] = {}
            for index, row in enumerate(rows):
                if index == 0:
                    previous_values = {field: _num(row.get(field)) for field in ACTION_FIELDS}
                    continue
                step = int(_num(row.get("step"), index))
                current_values = {field: _num(row.get(field)) for field in ACTION_FIELDS}
                deltas = {field: current_values[field] - previous_values[field] for field in ACTION_FIELDS}
                abs_deltas = {field: abs(value) for field, value in deltas.items()}
                reversal_fields: list[str] = []
                for field, delta in deltas.items():
                    sign = 1 if delta > 1e-6 else -1 if delta < -1e-6 else 0
                    if sign and previous_signs.get(field, 0) and sign != previous_signs[field]:
                        reversal_fields.append(field)
                    if sign:
                        previous_signs[field] = sign
                guardrail_changed = _guardrail_changed(row)
                if controller == "llm_rspc_v2" and scenario_id in failure_set and step not in negative_steps:
                    previous_values = current_values
                    continue
                row_label = "failure_negative" if step in negative_steps else "stable_positive"
                record = {
                    "scenario_id": scenario_id,
                    "controller": controller,
                    "step": step,
                    "label": row_label,
                    "sample_source": (
                        "llm_cvodes_failure_window"
                        if row_label == "failure_negative"
                        else "ppo_stable_trace"
                        if controller == "ppo"
                        else "llm_stable_trace"
                    ),
                    "regime": _regime(row),
                    "previous_action": previous_values,
                    "current_action": current_values,
                    "delta_action": deltas,
                    "max_abs_delta": max(abs_deltas.values()) if abs_deltas else 0.0,
                    "direction_reversal_count": len(reversal_fields),
                    "direction_reversal_fields": reversal_fields,
                    "screen_vent_conflict": _is_screen_vent_conflict(row),
                    "shade_screen_conflict": _is_shade_screen_conflict(row),
                    "guardrail_rewrite": guardrail_changed,
                    "guardrail_reasons": str(row.get("tomato_safety_v2_reasons", "") or ""),
                    "runtime_error_present": bool(_runtime_failure_step([row]) is not None),
                    "runtime_outcome": "cvodes_failure_window" if row_label == "failure_negative" else "stable_window",
                    "state_summary": {
                        "temp_air": _num(row.get("temp_air")),
                        "rh_air": _num(row.get("rh_air")),
                        "vpd_air": _num(row.get("vpd_air")),
                        "canopy_dew_margin": _num(row.get("canopy_dew_margin"), 99.0),
                        "dew_margin_air": _num(row.get("dew_margin_air"), 99.0),
                    },
                }
                if row_label == "failure_negative":
                    unstable_rows.append(record)
                else:
                    stable_rows.append(record)
                previous_values = current_values
    meta = {
        "trace_pair_count": len(pairs),
        "missing_pairs": missing_pairs,
        "failure_scenarios": list(failure_scenarios),
        "failure_window_steps": failure_window_steps,
    }
    return stable_rows, unstable_rows, meta


def _percentile(values: Sequence[float], pct: float) -> float:
    clean = sorted(float(value) for value in values)
    if not clean:
        return 0.0
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * pct
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    fraction = position - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def _stats(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values]
    return {
        "count": len(clean),
        "mean": mean(clean) if clean else 0.0,
        "p50": _percentile(clean, 0.50),
        "p90": _percentile(clean, 0.90),
        "p95": _percentile(clean, 0.95),
        "p99": _percentile(clean, 0.99),
        "max": max(clean) if clean else 0.0,
    }


def _group_key(row: Mapping[str, Any], *, include_controller: bool) -> tuple[str, ...]:
    if include_controller:
        return (str(row["label"]), str(row["controller"]), str(row["regime"]))
    return (str(row["label"]), str(row["regime"]))


def _transition_stats(rows: Sequence[Mapping[str, Any]], *, include_controller: bool = True) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[_group_key(row, include_controller=include_controller)].append(row)
    output: list[dict[str, Any]] = []
    for key, bucket in sorted(buckets.items()):
        label, *rest = key
        if include_controller:
            controller, regime = rest
        else:
            controller, regime = "all", rest[0]
        action_stats = {}
        for field in ACTION_FIELDS:
            action_stats[field] = _stats([abs(_num(row["delta_action"].get(field))) for row in bucket])
        output.append(
            {
                "label": label,
                "controller": controller,
                "regime": regime,
                "transition_count": len(bucket),
                "max_abs_delta": _stats([_num(row.get("max_abs_delta")) for row in bucket]),
                "direction_reversal_count": sum(int(row.get("direction_reversal_count", 0)) for row in bucket),
                "screen_vent_conflict_count": sum(1 for row in bucket if row.get("screen_vent_conflict")),
                "shade_screen_conflict_count": sum(1 for row in bucket if row.get("shade_screen_conflict")),
                "guardrail_rewrite_count": sum(1 for row in bucket if row.get("guardrail_rewrite")),
                "action_delta_stats": action_stats,
            }
        )
    return output


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_controller = Counter(str(row["controller"]) for row in rows)
    by_regime = Counter(str(row["regime"]) for row in rows)
    return {
        "transition_count": len(rows),
        "controller_counts": dict(sorted(by_controller.items())),
        "regime_counts": dict(sorted(by_regime.items())),
        "large_action_delta_count": sum(1 for row in rows if _num(row.get("max_abs_delta")) > 0.20),
        "direction_reversal_count": sum(int(row.get("direction_reversal_count", 0)) for row in rows),
        "screen_vent_conflict_count": sum(1 for row in rows if row.get("screen_vent_conflict")),
        "guardrail_rewrite_count": sum(1 for row in rows if row.get("guardrail_rewrite")),
    }


def build_envelope(
    *,
    stable_rows: Sequence[Mapping[str, Any]],
    unstable_rows: Sequence[Mapping[str, Any]],
    v40_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stable_summary = _summary(stable_rows)
    unstable_summary = _summary(unstable_rows)
    ppo_stable = [row for row in stable_rows if row.get("controller") == "ppo"]
    llm_stable = [row for row in stable_rows if row.get("controller") == "llm_rspc_v2"]
    stable_p95 = _stats([_num(row.get("max_abs_delta")) for row in stable_rows])["p95"]
    unstable_p95 = _stats([_num(row.get("max_abs_delta")) for row in unstable_rows])["p95"]
    envelope_exists = bool(stable_rows and unstable_rows)
    transition_gap_detected = bool(envelope_exists and unstable_p95 > max(0.20, stable_p95 * 1.5))
    stable_count = max(1, int(stable_summary["transition_count"]))
    unstable_count = max(1, int(unstable_summary["transition_count"]))
    stable_rates = {
        "large_action_delta_rate": stable_summary["large_action_delta_count"] / stable_count,
        "direction_reversal_rate": stable_summary["direction_reversal_count"] / stable_count,
        "guardrail_rewrite_rate": stable_summary["guardrail_rewrite_count"] / stable_count,
        "screen_vent_conflict_rate": stable_summary["screen_vent_conflict_count"] / stable_count,
    }
    unstable_rates = {
        "large_action_delta_rate": unstable_summary["large_action_delta_count"] / unstable_count,
        "direction_reversal_rate": unstable_summary["direction_reversal_count"] / unstable_count,
        "guardrail_rewrite_rate": unstable_summary["guardrail_rewrite_count"] / unstable_count,
        "screen_vent_conflict_rate": unstable_summary["screen_vent_conflict_count"] / unstable_count,
    }
    transition_pressure_detected = bool(
        envelope_exists
        and (
            unstable_rates["large_action_delta_rate"] > stable_rates["large_action_delta_rate"] * 1.5
            or unstable_rates["direction_reversal_rate"] > stable_rates["direction_reversal_rate"] * 1.25
            or unstable_rates["guardrail_rewrite_rate"] > stable_rates["guardrail_rewrite_rate"] * 1.5
        )
    )
    next_action = "shadow_transition_gate_patch_plan" if transition_gap_detected else "candidate_pool_or_profile_audit_required"
    if (
        v40_report
        and str(v40_report.get("main_attribution")) == "action_instability"
        and envelope_exists
        and (transition_gap_detected or transition_pressure_detected)
    ):
        next_action = "shadow_transition_gate_patch_plan"
    return {
        "schema_version": "stable_action_transition_envelope_20260530_v41",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "stable action transition envelope discovery",
        },
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "stable_transition_summary": stable_summary,
        "unstable_transition_summary": unstable_summary,
        "ppo_stable_summary": _summary(ppo_stable),
        "llm_stable_summary": _summary(llm_stable),
        "transition_stats_by_controller_regime": _transition_stats([*stable_rows, *unstable_rows], include_controller=True),
        "transition_stats_by_regime": _transition_stats([*stable_rows, *unstable_rows], include_controller=False),
        "stable_envelope_exists": envelope_exists,
        "transition_gap_detected": transition_gap_detected,
        "transition_pressure_detected": transition_pressure_detected,
        "stable_transition_rates": stable_rates,
        "unstable_transition_rates": unstable_rates,
        "stable_max_abs_delta_p95": stable_p95,
        "unstable_max_abs_delta_p95": unstable_p95,
        "source_v40_main_attribution": (v40_report or {}).get("main_attribution", ""),
        "next_action": next_action,
    }


def build_shadow_gate_design(envelope: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "transition_aware_candidate_scoring_design_20260530_v41",
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "default_llm_rspc_v2_changed": False,
        "shadow_only_design": True,
        "transition_gate_recommended": bool(envelope.get("next_action") == "shadow_transition_gate_patch_plan"),
        "scoring_terms": {
            "delta_cost": "penalize candidate action distance from previous applied action using stable envelope percentiles",
            "reversal_cost": "penalize short-cycle direction reversals per actuator",
            "hold_bonus": "reward keeping a stable action when no hard-safety pressure is present",
            "guardrail_risk_cost": "penalize candidates historically likely to be rewritten by Tomato Safety",
            "profile_consistency_score": "reward candidates that move toward the active profile trajectory",
            "emergency_bypass": "allow hard-safety-driven transitions to bypass soft smoothness penalties with provenance",
        },
        "initial_policy": {
            "do_not_clip_hard_safety_actions": True,
            "do_not_modify_candidate_generation": True,
            "do_not_change_default_llm_rspc_v2": True,
            "target_actuators": list(ACTION_FIELDS),
        },
        "evidence": {
            "stable_envelope_exists": envelope.get("stable_envelope_exists", False),
            "transition_gap_detected": envelope.get("transition_gap_detected", False),
            "transition_pressure_detected": envelope.get("transition_pressure_detected", False),
            "stable_max_abs_delta_p95": envelope.get("stable_max_abs_delta_p95", 0.0),
            "unstable_max_abs_delta_p95": envelope.get("unstable_max_abs_delta_p95", 0.0),
        },
        "next_action": envelope.get("next_action", "candidate_pool_or_profile_audit_required"),
    }


def build_readiness(envelope: Mapping[str, Any], gate_design: Mapping[str, Any]) -> dict[str, Any]:
    next_action = str(gate_design.get("next_action") or envelope.get("next_action"))
    if next_action == "candidate_pool_or_profile_audit_required":
        next_action = "profile_trajectory_fix_plan_or_candidate_pool_expansion_plan"
    return {
        "schema_version": "metadata_replay_readiness_checklist_20260530_v41",
        "mainline_alignment": envelope.get("mainline_alignment", {}),
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "default_llm_rspc_v2_changed": False,
        "stable_action_transition_dataset_ready": True,
        "unstable_failure_transition_dataset_ready": True,
        "stable_envelope_exists": bool(envelope.get("stable_envelope_exists", False)),
        "transition_gap_detected": bool(envelope.get("transition_gap_detected", False)),
        "transition_pressure_detected": bool(envelope.get("transition_pressure_detected", False)),
        "transition_gate_design_ready": bool(gate_design.get("shadow_only_design", False)),
        "stable_transition_summary": envelope.get("stable_transition_summary", {}),
        "unstable_transition_summary": envelope.get("unstable_transition_summary", {}),
        "next_action": next_action,
    }


def _dataset_report(rows: Sequence[Mapping[str, Any]], meta: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    return {
        "schema_version": f"{label}_action_transition_dataset_20260530_v41",
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "metadata": dict(meta),
        "summary": _summary(rows),
        "transition_rows": list(rows),
    }


def _markdown_table(metrics: Mapping[str, Any]) -> list[str]:
    lines = ["| metric | value |", "| --- | ---: |"]
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            rendered = "`complex`"
        else:
            rendered = f"`{value}`"
        lines.append(f"| {key} | {rendered} |")
    return lines


def build_markdown_report(report: Mapping[str, Any], *, title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Summary",
        "",
    ]
    summary = report.get("summary") or report.get("stable_transition_summary") or {}
    lines.extend(_markdown_table(summary))
    if "unstable_transition_summary" in report:
        lines.extend(["", "## Unstable Summary", ""])
        lines.extend(_markdown_table(report.get("unstable_transition_summary", {}) or {}))
    if "stable_envelope_exists" in report:
        lines.extend(["", "## Envelope Decision", ""])
        lines.append(f"- Stable envelope exists: `{report.get('stable_envelope_exists')}`")
        lines.append(f"- Transition gap detected: `{report.get('transition_gap_detected')}`")
        lines.append(f"- Transition pressure detected: `{report.get('transition_pressure_detected')}`")
        lines.append(f"- Stable p95 max delta: `{report.get('stable_max_abs_delta_p95')}`")
        lines.append(f"- Unstable p95 max delta: `{report.get('unstable_max_abs_delta_p95')}`")
    if "scoring_terms" in report:
        lines.extend(["", "## Shadow Scoring Terms", ""])
        for key, value in (report.get("scoring_terms", {}) or {}).items():
            lines.append(f"- `{key}`: {value}")
    return "\n".join(lines) + "\n"


def _write_pair(path_json: str | Path, path_md: str | Path, report: Mapping[str, Any], *, title: str) -> None:
    json_path = _resolve(path_json)
    md_path = _resolve(path_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(build_markdown_report(report, title=title), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--v40-attribution-json", required=True)
    parser.add_argument("--failure-window-steps", type=int, default=60)
    parser.add_argument("--failure-scenario", action="append", default=[])
    parser.add_argument("--stable-dataset-json", required=True)
    parser.add_argument("--stable-dataset-md", required=True)
    parser.add_argument("--unstable-dataset-json", required=True)
    parser.add_argument("--unstable-dataset-md", required=True)
    parser.add_argument("--envelope-json", required=True)
    parser.add_argument("--envelope-md", required=True)
    parser.add_argument("--gate-design-json", required=True)
    parser.add_argument("--gate-design-md", required=True)
    parser.add_argument("--readiness-json", required=True)
    parser.add_argument("--readiness-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    failure_scenarios = args.failure_scenario or list(DEFAULT_FAILURE_SCENARIOS)
    stable_rows, unstable_rows, meta = build_transition_rows(
        trace_dir=args.trace_dir,
        failure_scenarios=failure_scenarios,
        failure_window_steps=args.failure_window_steps,
    )
    v40_report = _load_json(args.v40_attribution_json)
    stable_report = _dataset_report(stable_rows, meta, label="stable")
    unstable_report = _dataset_report(unstable_rows, meta, label="unstable_failure")
    envelope = build_envelope(stable_rows=stable_rows, unstable_rows=unstable_rows, v40_report=v40_report)
    gate_design = build_shadow_gate_design(envelope)
    readiness = build_readiness(envelope, gate_design)
    _write_pair(args.stable_dataset_json, args.stable_dataset_md, stable_report, title="Stable Action Transition Dataset v41")
    _write_pair(
        args.unstable_dataset_json,
        args.unstable_dataset_md,
        unstable_report,
        title="Unstable Failure Transition Dataset v41",
    )
    _write_pair(args.envelope_json, args.envelope_md, envelope, title="Stable Action Transition Envelope v41")
    _write_pair(args.gate_design_json, args.gate_design_md, gate_design, title="Transition-Aware Candidate Scoring Design v41")
    _write_pair(args.readiness_json, args.readiness_md, readiness, title="Metadata Replay Readiness Checklist v41")
    print(f"stable_transition_count={stable_report['summary']['transition_count']}")
    print(f"unstable_transition_count={unstable_report['summary']['transition_count']}")
    print(f"next_action={readiness['next_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
