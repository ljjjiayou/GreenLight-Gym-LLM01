"""Mechanism-level audit for hot-dry LLM-RSPC regressions.

This script is intentionally diagnostic-only. It reads existing frozen replay
traces, aligns baseline/candidate/shadow controllers by step, and reports where
hot-dry performance changes likely come from without changing the controller.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


DEFAULT_SCENARIOS = (
    "y2015_d120_s44_n240",
    "y2015_d180_s44_n240",
    "y2015_d240_s42_n240",
    "y2015_d180_s42_n240",
)

ACTION_KEYS = (
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
)

TARGET_KEYS = ("target_temp", "target_co2", "target_rh")


def _coerce(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    if text == "":
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _num(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _truthy(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    data = [float(value) for value in values]
    return float(mean(data)) if data else float(default)


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _reason_values(row: Mapping[str, Any]) -> List[str]:
    return [
        reason.strip()
        for reason in str(row.get("tomato_safety_v2_reasons") or "").split(",")
        if reason.strip()
    ]


def read_trace(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{key: _coerce(value) for key, value in row.items()} for row in reader]


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _segments(flags: Sequence[bool], *, merge_gap: int = 1) -> List[tuple[int, int]]:
    raw: List[tuple[int, int]] = []
    start: Optional[int] = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            raw.append((start, index - 1))
            start = None
    if start is not None:
        raw.append((start, len(flags) - 1))
    if not raw:
        return []
    merged = [raw[0]]
    for start, end in raw[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end - 1 <= merge_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def find_trace(trace_dirs: Sequence[str | Path], scenario_id: str, controller: str) -> Optional[Path]:
    suffixes = (".csv", ".jsonl")
    names = [f"{scenario_id}_{controller}{suffix}" for suffix in suffixes]
    for trace_dir in trace_dirs:
        root = Path(trace_dir)
        if root.is_file() and root.name in names:
            return root
        if not root.exists():
            continue
        for name in names:
            direct = root / name
            if direct.exists():
                return direct
        matches = sorted(path for name in names for path in root.rglob(name))
        if matches:
            return matches[0]
    return None


def _index_by_step(rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {int(_num(row, "step", _num(row, "timestep", 0.0))): dict(row) for row in rows}


def _is_hot_dry_context(row: Mapping[str, Any]) -> bool:
    temp = _num(row, "temp_air", 20.0)
    rh = _num(row, "rh_air", 70.0)
    vpd = _num(row, "vpd_air", _num(row, "vpd_kpa", 0.0))
    rad = max(
        _num(row, "glob_rad", 0.0),
        _num(row, "forecast_rad_mean_1h", 0.0),
        _num(row, "forecast_rad_peak_2h", 0.0),
    )
    return (temp >= 28.0 and rad >= 300.0 and (rh <= 60.0 or vpd >= 1.4)) or (
        rad >= 600.0 and (rh <= 65.0 or vpd >= 1.2)
    )


def _risk_terms(row: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "reward": _num(row, "reward", 0.0),
        "profit": _num(row, "profit", 0.0),
        "rh_low": _num(row, "rh_low_violation", 0.0),
        "vpd_high": _num(row, "vpd_high_excess", 0.0),
        "temp": _num(row, "temp_violation", 0.0),
        "rh_high": _num(row, "rh_high_violation", 0.0),
        "dew_lt1": 1.0 if _num(row, "dew_margin_air", _num(row, "dew_margin_min", 99.0)) < 1.0 else 0.0,
        "canopy_lt1": 1.0 if _num(row, "canopy_dew_margin", 99.0) < 1.0 else 0.0,
        "canopy_lt0": 1.0 if _num(row, "canopy_dew_margin", 99.0) < 0.0 else 0.0,
    }


def _delta_terms(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, float]:
    base_terms = _risk_terms(base)
    cand_terms = _risk_terms(candidate)
    return {key: float(cand_terms[key] - base_terms[key]) for key in cand_terms}


def _aligned_pairs(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    shadow_rows: Sequence[Mapping[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    baseline = _index_by_step(baseline_rows)
    candidate = _index_by_step(candidate_rows)
    shadow = _index_by_step(shadow_rows or [])
    steps = sorted(set(baseline) & set(candidate))
    pairs = []
    for step in steps:
        base = baseline[step]
        cand = candidate[step]
        shadow_row = shadow.get(step, {})
        pairs.append(
            {
                "step": int(step),
                "baseline": base,
                "candidate": cand,
                "shadow": shadow_row,
                "delta": _delta_terms(base, cand),
                "hot_dry_context": bool(_is_hot_dry_context(base) or _is_hot_dry_context(cand)),
            }
        )
    return pairs


def _degradation_kind(pair: Mapping[str, Any], *, epsilon: float = 1e-9) -> str:
    delta = pair["delta"]
    dry = float(delta.get("rh_low", 0.0)) > epsilon or float(delta.get("vpd_high", 0.0)) > epsilon
    temp = float(delta.get("temp", 0.0)) > epsilon
    dew = float(delta.get("rh_high", 0.0)) > epsilon or float(delta.get("dew_lt1", 0.0)) > epsilon
    canopy = float(delta.get("canopy_lt0", 0.0)) > epsilon
    if canopy:
        return "canopy_safety_degradation"
    if dry and temp:
        return "dry_temp_degradation"
    if dry:
        return "dry_performance_degradation"
    if temp:
        return "temp_performance_degradation"
    if dew:
        return "dew_humidity_degradation"
    return ""


def _action_delta(selected: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out = {}
    for key in ACTION_KEYS:
        out[key] = float(
            _safe_mean(_num(pair["candidate"], key, 0.0) for pair in selected)
            - _safe_mean(_num(pair["baseline"], key, 0.0) for pair in selected)
        )
    return out


def _target_delta(selected: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out = {}
    for key in TARGET_KEYS:
        out[key] = float(
            _safe_mean(_num(pair["candidate"], key, 0.0) for pair in selected)
            - _safe_mean(_num(pair["baseline"], key, 0.0) for pair in selected)
        )
    return out


def _horizon_proxy(pairs: Sequence[Mapping[str, Any]], start_step: int, *, horizon_steps: int) -> Dict[str, Any]:
    selected = [pair for pair in pairs if start_step <= int(pair["step"]) < start_step + horizon_steps]
    if not selected:
        return {"horizon_steps": int(horizon_steps), "available_steps": 0, "delta": {}}
    sums: Dict[str, float] = {}
    for pair in selected:
        for key, value in pair["delta"].items():
            sums[key] = sums.get(key, 0.0) + float(value)
    return {
        "horizon_steps": int(horizon_steps),
        "available_steps": len(selected),
        "delta": {key: round(value, 6) for key, value in sorted(sums.items())},
    }


def _classify_window(summary: Mapping[str, Any]) -> Dict[str, Any]:
    action_delta = summary.get("action_delta", {}) or {}
    target_delta = summary.get("target_delta", {}) or {}
    delta = summary.get("delta", {}) or {}
    reason_counts = summary.get("tomato_safety_v2_reason_counts", {}) or {}
    shadow = summary.get("shadow", {}) or {}
    climate = summary.get("climate", {}) or {}

    scores = {
        "llm_setpoint_or_contract": 0.0,
        "rspc_fallback_action_selection": 0.0,
        "tomato_safety_v2_post_guard": 0.0,
        "rspc_shadow_scoring_disagreement": 0.0,
        "physical_tradeoff": 0.0,
    }
    notes: List[str] = []

    if abs(float(target_delta.get("target_rh", 0.0))) >= 2.0 or abs(float(target_delta.get("target_temp", 0.0))) >= 1.5:
        scores["llm_setpoint_or_contract"] += 2.0
        notes.append("candidate and baseline track meaningfully different target setpoints")

    if abs(float(action_delta.get("u_ventilation", 0.0))) >= 0.12:
        scores["rspc_fallback_action_selection"] += 1.2
        notes.append("ventilation differs enough to explain RH/VPD movement")
    if abs(float(action_delta.get("u_screen", 0.0))) >= 0.20 or abs(float(action_delta.get("u_shading", 0.0))) >= 0.20:
        scores["rspc_fallback_action_selection"] += 0.8
        notes.append("screen/shade differs enough to change heat and humidity balance")

    v2_applied_rate = float(summary.get("tomato_safety_v2_applied_rate", 0.0))
    hot_dry_reasons = [
        key
        for key in reason_counts
        if any(token in key for token in ("hot_dry", "cooldown", "canopy", "severe_dry", "pre_hot"))
    ]
    if v2_applied_rate >= 0.40 and hot_dry_reasons:
        scores["tomato_safety_v2_post_guard"] += 1.4 + min(1.0, v2_applied_rate)
        notes.append("Tomato Safety v2 actively reshaped the window")

    if float(shadow.get("would_select_rate", 0.0)) >= 0.25:
        scores["rspc_shadow_scoring_disagreement"] += 1.5
        notes.append("MC-SERO shadow often prefers a different short-horizon candidate")
    if float(shadow.get("mean_margin", 0.0)) >= 0.10:
        scores["rspc_shadow_scoring_disagreement"] += 0.7

    dry_worse = float(delta.get("rh_low", 0.0)) > 0.0 or float(delta.get("vpd_high", 0.0)) > 0.0
    safety_better = (
        float(delta.get("canopy_lt1", 0.0)) < 0.0
        or float(delta.get("canopy_lt0", 0.0)) < 0.0
        or float(delta.get("rh_high", 0.0)) < 0.0
    )
    if dry_worse and safety_better and float(climate.get("max_rad", 0.0)) >= 500.0:
        scores["physical_tradeoff"] += 1.4
        notes.append("dry loss coincides with dew/canopy safety improvement under high radiation")
    if dry_worse and float(climate.get("max_temp", 0.0)) >= 32.0 and float(climate.get("max_rad", 0.0)) >= 600.0:
        scores["physical_tradeoff"] += 0.8
        notes.append("high heat load makes humidity retention and cooling partly conflicting")

    if max(scores.values()) <= 0.0:
        scores["rspc_fallback_action_selection"] = 0.5
        notes.append("no setpoint or shadow signal dominates; inspect action selection first")

    primary = max(sorted(scores), key=lambda key: scores[key])
    next_layer = {
        "llm_setpoint_or_contract": "inspect intent/setpoint contract before low-level tuning",
        "rspc_fallback_action_selection": "inspect RSPC/fallback candidates and score terms",
        "tomato_safety_v2_post_guard": "inspect post-guard reshaping; do not widen guard before evidence",
        "rspc_shadow_scoring_disagreement": "compare MC-SERO preferred candidates against selected controls",
        "physical_tradeoff": "treat as constrained tradeoff; improve profile/scoring, not a narrow guard",
    }[primary]
    return {
        "primary_layer": primary,
        "layer_scores": {key: round(value, 3) for key, value in sorted(scores.items())},
        "evidence": notes,
        "recommended_next_layer": next_layer,
    }


def _summarize_window(
    scenario_id: str,
    pairs: Sequence[Mapping[str, Any]],
    start_index: int,
    end_index: int,
    *,
    horizon_steps: int,
) -> Dict[str, Any]:
    selected = list(pairs[start_index : end_index + 1])
    start_step = int(selected[0]["step"])
    end_step = int(selected[-1]["step"])
    delta: Dict[str, float] = {}
    for pair in selected:
        for key, value in pair["delta"].items():
            delta[key] = delta.get(key, 0.0) + float(value)

    candidate_rows = [pair["candidate"] for pair in selected]
    baseline_rows = [pair["baseline"] for pair in selected]
    shadow_rows = [pair["shadow"] for pair in selected if pair.get("shadow")]
    reason_counts = _counts(reason for row in candidate_rows for reason in _reason_values(row))
    shadow_candidate_counts = _counts(row.get("mc_sero_best_candidate") for row in shadow_rows)
    shadow_reject_counts = _counts(row.get("mc_sero_reject_reason") for row in shadow_rows)
    v2_applied_rate = _safe_mean(1.0 if _truthy(row, "tomato_safety_v2_applied") else 0.0 for row in candidate_rows)
    shadow_would_select_rate = _safe_mean(1.0 if _truthy(row, "mc_sero_would_select") else 0.0 for row in shadow_rows)

    summary: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "kind": _counts(_degradation_kind(pair) for pair in selected),
        "start_step": start_step,
        "end_step": end_step,
        "duration_steps": len(selected),
        "delta": {key: round(value, 6) for key, value in sorted(delta.items())},
        "action_delta": {key: round(value, 6) for key, value in sorted(_action_delta(selected).items())},
        "target_delta": {key: round(value, 6) for key, value in sorted(_target_delta(selected).items())},
        "climate": {
            "max_temp": round(max(_num(row, "temp_air", 0.0) for row in candidate_rows), 6),
            "min_rh": round(min(_num(row, "rh_air", 100.0) for row in candidate_rows), 6),
            "max_vpd": round(max(_num(row, "vpd_air", _num(row, "vpd_kpa", 0.0)) for row in candidate_rows), 6),
            "max_rad": round(
                max(
                    max(
                        _num(row, "glob_rad", 0.0),
                        _num(row, "forecast_rad_mean_1h", 0.0),
                        _num(row, "forecast_rad_peak_2h", 0.0),
                    )
                    for row in candidate_rows
                ),
                6,
            ),
            "min_canopy_margin": round(min(_num(row, "canopy_dew_margin", 99.0) for row in candidate_rows), 6),
        },
        "baseline_source_counts": _counts(row.get("source") for row in baseline_rows),
        "candidate_source_counts": _counts(row.get("source") for row in candidate_rows),
        "tomato_safety_v2_applied_rate": round(v2_applied_rate, 6),
        "tomato_safety_v2_reason_counts": reason_counts,
        "shadow": {
            "available_steps": len(shadow_rows),
            "would_select_rate": round(shadow_would_select_rate, 6),
            "mean_margin": round(_safe_mean(_num(row, "mc_sero_margin", 0.0) for row in shadow_rows), 6),
            "best_candidate_counts": shadow_candidate_counts,
            "reject_reason_counts": shadow_reject_counts,
        },
        "realized_horizon_proxy": _horizon_proxy(pairs, start_step, horizon_steps=horizon_steps),
    }
    summary["diagnosis"] = _classify_window(summary)
    return summary


def analyze_scenario(
    scenario_id: str,
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    shadow_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    merge_gap: int = 1,
    horizon_steps: int = 6,
    hot_dry_only: bool = False,
) -> Dict[str, Any]:
    pairs = _aligned_pairs(baseline_rows, candidate_rows, shadow_rows)
    if hot_dry_only:
        pairs = [pair for pair in pairs if pair["hot_dry_context"]]
    flags = [bool(_degradation_kind(pair)) for pair in pairs]
    windows = [
        _summarize_window(scenario_id, pairs, start, end, horizon_steps=horizon_steps)
        for start, end in _segments(flags, merge_gap=merge_gap)
    ]
    aggregate: Dict[str, float] = {}
    for pair in pairs:
        for key, value in pair["delta"].items():
            aggregate[key] = aggregate.get(key, 0.0) + float(value)
    aggregate = {key: round(value, 6) for key, value in sorted(aggregate.items())}

    if windows:
        primary = _counts(window["diagnosis"]["primary_layer"] for window in windows)
        main_layer = max(sorted(primary), key=primary.get)
    else:
        primary = {}
        main_layer = "no_degradation_window"

    return {
        "scenario_id": scenario_id,
        "aligned_steps": len(pairs),
        "aggregate_delta": aggregate,
        "window_count": len(windows),
        "primary_layer_counts": primary,
        "main_layer": main_layer,
        "windows": windows,
    }


def audit_from_trace_dirs(
    trace_dirs: Sequence[str | Path],
    scenario_ids: Sequence[str],
    *,
    baseline: str = "llm",
    candidate: str = "llm_rspc_v2",
    shadow: str = "llm_sero_shadow",
    merge_gap: int = 1,
    horizon_steps: int = 6,
    hot_dry_only: bool = False,
) -> Dict[str, Any]:
    scenarios: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for scenario_id in scenario_ids:
        baseline_path = find_trace(trace_dirs, scenario_id, baseline)
        candidate_path = find_trace(trace_dirs, scenario_id, candidate)
        shadow_path = find_trace(trace_dirs, scenario_id, shadow) if shadow else None
        if baseline_path is None or candidate_path is None:
            warnings.append(f"{scenario_id}:missing_trace baseline={baseline_path} candidate={candidate_path}")
            continue
        result = analyze_scenario(
            scenario_id,
            read_trace(baseline_path),
            read_trace(candidate_path),
            read_trace(shadow_path) if shadow_path else [],
            merge_gap=merge_gap,
            horizon_steps=horizon_steps,
            hot_dry_only=hot_dry_only,
        )
        result["trace_paths"] = {
            "baseline": str(baseline_path),
            "candidate": str(candidate_path),
            "shadow": str(shadow_path) if shadow_path else "",
        }
        scenarios.append(result)
    return {
        "schema_version": "hot_dry_mechanism_audit_v1",
        "baseline": baseline,
        "candidate": candidate,
        "shadow": shadow,
        "horizon_steps": int(horizon_steps),
        "hot_dry_only": bool(hot_dry_only),
        "scenario_count": len(scenarios),
        "window_count": int(sum(int(item.get("window_count", 0)) for item in scenarios)),
        "warnings": warnings,
        "scenarios": scenarios,
    }


def _fmt_delta(delta: Mapping[str, Any], *keys: str) -> str:
    return ", ".join(f"{key}={float(delta.get(key, 0.0)):+.3f}" for key in keys)


def _top_counts(counts: Mapping[str, Any], limit: int = 3) -> str:
    items = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items) if items else "-"


def build_report(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Hot-Dry Mechanism Audit v1",
        "",
        f"- Baseline: `{audit.get('baseline')}`",
        f"- Candidate: `{audit.get('candidate')}`",
        f"- Shadow: `{audit.get('shadow')}`",
        f"- Scenarios: {audit.get('scenario_count', 0)}",
        f"- Degradation windows: {audit.get('window_count', 0)}",
        "",
        "## Scenario Summary",
        "",
        "| scenario | aligned | d_reward | d_RHlow | d_VPDhi | d_temp | d_canopy<1 | d_canopy<0 | main layer |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for scenario in audit.get("scenarios", []) or []:
        delta = scenario.get("aggregate_delta", {}) or {}
        lines.append(
            "| {sid} | {steps} | {reward:.3f} | {rhlow:.3f} | {vpd:.3f} | {temp:.3f} | {can1:.0f} | {can0:.0f} | {layer} |".format(
                sid=scenario.get("scenario_id"),
                steps=int(scenario.get("aligned_steps", 0)),
                reward=float(delta.get("reward", 0.0)),
                rhlow=float(delta.get("rh_low", 0.0)),
                vpd=float(delta.get("vpd_high", 0.0)),
                temp=float(delta.get("temp", 0.0)),
                can1=float(delta.get("canopy_lt1", 0.0)),
                can0=float(delta.get("canopy_lt0", 0.0)),
                layer=scenario.get("main_layer"),
            )
        )

    lines.extend(
        [
            "",
            "## Degradation Windows",
            "",
            "| scenario | steps | kind | primary layer | action delta | target delta | v2 reasons | shadow preference | horizon proxy |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for scenario in audit.get("scenarios", []) or []:
        for window in scenario.get("windows", []) or []:
            diagnosis = window.get("diagnosis", {}) or {}
            shadow = window.get("shadow", {}) or {}
            lines.append(
                "| {sid} | {start}-{end} | {kind} | {layer} | {action} | {target} | {reasons} | {shadow} | {proxy} |".format(
                    sid=scenario.get("scenario_id"),
                    start=window.get("start_step"),
                    end=window.get("end_step"),
                    kind=_top_counts(window.get("kind", {}) or {}, limit=2),
                    layer=diagnosis.get("primary_layer", ""),
                    action=_fmt_delta(window.get("action_delta", {}) or {}, "u_ventilation", "u_screen", "u_shading"),
                    target=_fmt_delta(window.get("target_delta", {}) or {}, "target_rh", "target_temp"),
                    reasons=_top_counts(window.get("tomato_safety_v2_reason_counts", {}) or {}, limit=3),
                    shadow=_top_counts(shadow.get("best_candidate_counts", {}) or {}, limit=3),
                    proxy=_fmt_delta(
                        (window.get("realized_horizon_proxy", {}) or {}).get("delta", {}) or {},
                        "rh_low",
                        "vpd_high",
                        "temp",
                        "canopy_lt0",
                    ),
                )
            )

    lines.extend(["", "## Mechanism Interpretation", ""])
    for scenario in audit.get("scenarios", []) or []:
        sid = str(scenario.get("scenario_id"))
        delta = scenario.get("aggregate_delta", {}) or {}
        if not scenario.get("windows"):
            lines.append(f"- `{sid}`: no paired degradation window detected; keep as regression guard.")
            continue
        first = scenario["windows"][0]
        diagnosis = first.get("diagnosis", {}) or {}
        evidence = "; ".join(diagnosis.get("evidence", []) or ["no dominant signal"])
        lines.append(
            f"- `{sid}`: aggregate {_fmt_delta(delta, 'rh_low', 'vpd_high', 'temp', 'canopy_lt0')}; "
            f"main layer `{scenario.get('main_layer')}`. Evidence: {evidence}. "
            f"Next: {diagnosis.get('recommended_next_layer', 'inspect paired traces')}."
        )

    if audit.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in audit.get("warnings", []) or []:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Next Optimization Rule",
            "",
            "Do not add Tomato Safety v2 thresholds from this report alone. If most windows point to action selection or shadow disagreement, change RSPC/profile scoring first. If setpoint deltas dominate, change Intent/Setpoint Contract. If physical tradeoff dominates, move the problem into constrained profile generation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _split_csv(text: str) -> List[str]:
    if str(text).strip().lower() in {"", "all", "*"}:
        return list(DEFAULT_SCENARIOS)
    return [part.strip() for part in str(text).split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit hot-dry LLM-RSPC mechanism regressions.")
    parser.add_argument("--trace-dir", action="append", required=True, help="Trace directory; can be repeated.")
    parser.add_argument("--scenario-id", type=str, default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--baseline", type=str, default="llm")
    parser.add_argument("--candidate", type=str, default="llm_rspc_v2")
    parser.add_argument("--shadow", type=str, default="llm_sero_shadow")
    parser.add_argument("--merge-gap", type=int, default=1)
    parser.add_argument("--horizon-steps", type=int, default=6)
    parser.add_argument("--hot-dry-only", action="store_true")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-report", type=str, required=True)
    args = parser.parse_args()

    audit = audit_from_trace_dirs(
        args.trace_dir,
        _split_csv(args.scenario_id),
        baseline=args.baseline,
        candidate=args.candidate,
        shadow=args.shadow,
        merge_gap=int(args.merge_gap),
        horizon_steps=int(args.horizon_steps),
        hot_dry_only=bool(args.hot_dry_only),
    )
    write_json(args.output_json, audit)
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(audit), encoding="utf-8")
    print(f"Saved hot-dry mechanism audit JSON to {args.output_json}")
    print(f"Saved hot-dry mechanism audit report to {args.output_report}")


if __name__ == "__main__":
    main()
