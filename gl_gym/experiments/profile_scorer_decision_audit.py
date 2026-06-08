"""Audit whether shadow profile-scorer decisions align with climate risks."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


TRACE_NAME_RE = re.compile(
    r"^y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)_n(?P<max_steps>\d+)_(?P<controller>.+)$"
)

RISK_NAMES = (
    "rh_low",
    "vpd_high",
    "temp_high",
    "rh_high",
    "dew_air",
    "canopy_dew",
    "dry_performance",
    "safety",
)


def _coerce(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    if not text:
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
    try:
        value = row.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _truthy(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        key = str(value or "").strip()
        if not key:
            continue
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _top_counts(values: Iterable[Any], limit: int = 4) -> Dict[str, int]:
    counts = _counts(values)
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return dict(items)


def _fmt_counts(counts: Mapping[str, Any], limit: int = 4) -> str:
    items = sorted(counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items) if items else "-"


def read_trace(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", newline="", encoding="utf-8") as f:
        return [{key: _coerce(value) for key, value in row.items()} for row in csv.DictReader(f)]


def discover_traces(inputs: Sequence[str | Path]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file() and path.suffix.lower() in {".csv", ".jsonl"}:
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.csv") if TRACE_NAME_RE.match(p.stem)))
            files.extend(sorted(p for p in path.rglob("*.jsonl") if TRACE_NAME_RE.match(p.stem)))
    by_stem: Dict[str, Path] = {}
    priority = {".csv": 0, ".jsonl": 1}
    for file in sorted(set(files)):
        current = by_stem.get(file.stem)
        if current is None or priority.get(file.suffix.lower(), 9) < priority.get(current.suffix.lower(), 9):
            by_stem[file.stem] = file
    return sorted(by_stem.values())


def trace_identity(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    match = TRACE_NAME_RE.match(path.stem)
    if not match:
        return {"trace_id": path.stem, "scenario_id": path.stem, "controller": ""}
    data = match.groupdict()
    return {
        "trace_id": path.stem,
        "scenario_id": f"y{data['year']}_d{data['day']}_s{data['seed']}_n{data['max_steps']}",
        "year": int(data["year"]),
        "day": int(data["day"]),
        "seed": int(data["seed"]),
        "max_steps": int(data["max_steps"]),
        "controller": data["controller"],
    }


def risk_flags(
    row: Mapping[str, Any],
    *,
    rh_low_threshold: float = 62.0,
    vpd_high_threshold: float = 1.60,
    temp_high_threshold: float = 32.0,
    rh_high_threshold: float = 90.0,
    dew_threshold: float = 1.0,
    canopy_threshold: float = 1.0,
) -> Dict[str, bool]:
    rh_air = _num(row, "rh_air", 70.0)
    vpd = _num(row, "vpd_air", _num(row, "vpd_kpa", _num(row, "strategy_vpd_kpa", 0.0)))
    temp_air = _num(row, "temp_air", 20.0)
    dew_air = _num(row, "dew_margin_air", _num(row, "dew_margin_min", 99.0))
    canopy_margin = _num(row, "canopy_dew_margin", 99.0)
    rh_low = _num(row, "rh_low_violation", 0.0) > 0.0 or rh_air < rh_low_threshold
    vpd_high = _num(row, "vpd_high_excess", 0.0) > 0.0 or vpd > vpd_high_threshold
    temp_high = _num(row, "temp_violation", 0.0) > 0.0 or temp_air >= temp_high_threshold
    rh_high = _num(row, "rh_high_violation", 0.0) > 0.0 or rh_air >= rh_high_threshold
    dew_air_risk = dew_air < dew_threshold
    canopy_risk = canopy_margin < canopy_threshold
    dry_performance = rh_low or vpd_high
    safety = temp_high or rh_high or dew_air_risk or canopy_risk
    return {
        "rh_low": bool(rh_low),
        "vpd_high": bool(vpd_high),
        "temp_high": bool(temp_high),
        "rh_high": bool(rh_high),
        "dew_air": bool(dew_air_risk),
        "canopy_dew": bool(canopy_risk),
        "dry_performance": bool(dry_performance),
        "safety": bool(safety),
    }


def _risk_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = {key: 0 for key in RISK_NAMES}
    for row in rows:
        flags = risk_flags(row)
        for key, active in flags.items():
            if active:
                counts[key] += 1
    return {key: value for key, value in counts.items() if value}


def _scorer_selected(row: Mapping[str, Any]) -> str:
    return str(row.get("profile_scorer_selected_shadow_profile") or "").strip()


def _gate_reason(row: Mapping[str, Any]) -> str:
    return str(row.get("profile_scorer_safety_gate_reason") or "none").strip() or "none"


def _selector_selected(row: Mapping[str, Any]) -> str:
    return str(row.get("profile_generator_selected_shadow_profile") or "").strip()


def _has_scorer(row: Mapping[str, Any]) -> bool:
    return bool(_scorer_selected(row))


def _is_disagreement(row: Mapping[str, Any]) -> bool:
    selector = _selector_selected(row)
    scorer = _scorer_selected(row)
    return bool(selector and scorer and selector != scorer)


def _segments(rows: Sequence[Mapping[str, Any]], predicate) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    active: List[Mapping[str, Any]] = []
    for row in rows:
        if predicate(row):
            active.append(row)
            continue
        if active:
            windows.append(_summarize_window(active))
            active = []
    if active:
        windows.append(_summarize_window(active))
    return windows


def _summarize_window(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "start_step": int(_num(rows[0], "step", 0.0)),
        "end_step": int(_num(rows[-1], "step", 0.0)),
        "duration_steps": int(len(rows)),
        "intent_regime_counts": _top_counts(row.get("intent_regime") for row in rows),
        "selector_selected_counts": _top_counts(_selector_selected(row) for row in rows),
        "scorer_selected_counts": _top_counts(_scorer_selected(row) for row in rows),
        "safety_gate_counts": _top_counts(_gate_reason(row) for row in rows),
        "risk_counts": _risk_counts(rows),
        "mean_score": round(mean(_num(row, "profile_scorer_selected_score", 0.0) for row in rows), 6),
        "mean_margin": round(mean(_num(row, "profile_scorer_margin_to_second", 0.0) for row in rows), 6),
        "mean_delta_target_temp": round(mean(_num(row, "profile_scorer_delta_target_temp", 0.0) for row in rows), 6),
        "mean_delta_target_co2": round(mean(_num(row, "profile_scorer_delta_target_co2", 0.0) for row in rows), 6),
        "mean_delta_target_rh": round(mean(_num(row, "profile_scorer_delta_target_rh", 0.0) for row in rows), 6),
        "mean_dew_risk_score": round(mean(_num(row, "profile_scorer_dew_risk", 0.0) for row in rows), 6),
        "mean_hot_dry_gap_score": round(
            mean(_num(row, "profile_scorer_hot_dry_retention_gap", 0.0) for row in rows),
            6,
        ),
        "mean_humid_gap_score": round(mean(_num(row, "profile_scorer_humid_relief_gap", 0.0) for row in rows), 6),
    }


def _matrix_by(rows: Sequence[Mapping[str, Any]], key_fn) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        key = str(key_fn(row) or "unknown")
        groups.setdefault(key, []).append(row)
    matrix: Dict[str, Dict[str, Any]] = {}
    for key, group in sorted(groups.items()):
        matrix[key] = {
            "steps": len(group),
            "selector_selected_counts": _counts(_selector_selected(row) for row in group),
            "scorer_selected_counts": _counts(_scorer_selected(row) for row in group),
            "risk_counts": _risk_counts(group),
            "safety_gate_counts": _counts(_gate_reason(row) for row in group),
            "agreement_steps": sum(1 for row in group if not _is_disagreement(row)),
            "mean_score": round(mean(_num(row, "profile_scorer_selected_score", 0.0) for row in group), 6),
            "mean_margin": round(mean(_num(row, "profile_scorer_margin_to_second", 0.0) for row in group), 6),
        }
    return matrix


def _risk_matrix(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    matrix: Dict[str, Dict[str, Any]] = {}
    for risk in RISK_NAMES:
        group = [row for row in rows if risk_flags(row).get(risk, False)]
        if not group:
            continue
        matrix[risk] = {
            "steps": len(group),
            "intent_regime_counts": _counts(row.get("intent_regime") for row in group),
            "selector_selected_counts": _counts(_selector_selected(row) for row in group),
            "scorer_selected_counts": _counts(_scorer_selected(row) for row in group),
            "safety_gate_counts": _counts(_gate_reason(row) for row in group),
            "agreement_steps": sum(1 for row in group if not _is_disagreement(row)),
            "hot_dry_protect_steps": sum(1 for row in group if _scorer_selected(row) == "hot_dry_protect"),
            "strict_dehumidify_steps": sum(
                1 for row in group if _scorer_selected(row) == "strict_dehumidify_then_relax"
            ),
            "constant_hold_steps": sum(1 for row in group if _scorer_selected(row) == "constant_hold"),
        }
    return matrix


def _gate_matrix(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_gate_reason(row), []).append(row)
    matrix: Dict[str, Dict[str, Any]] = {}
    for gate, group in sorted(groups.items()):
        matrix[gate] = {
            "steps": len(group),
            "intent_regime_counts": _counts(row.get("intent_regime") for row in group),
            "selector_selected_counts": _counts(_selector_selected(row) for row in group),
            "scorer_selected_counts": _counts(_scorer_selected(row) for row in group),
            "risk_counts": _risk_counts(group),
            "agreement_steps": sum(1 for row in group if not _is_disagreement(row)),
            "hot_dry_protect_steps": sum(1 for row in group if _scorer_selected(row) == "hot_dry_protect"),
        }
    return matrix


def _recommendation(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "decision": "needs_trace_metadata",
            "reasons": ["no profile scorer metadata was found"],
        }
    scorer_steps = len(rows)
    hot_dry_selected = [row for row in rows if _scorer_selected(row) == "hot_dry_protect"]
    hot_dry_safe_dry = [
        row
        for row in rows
        if str(row.get("intent_regime") or "") == "hot_dry_relief"
        and risk_flags(row)["dry_performance"]
        and not risk_flags(row)["safety"]
    ]
    hot_dry_selected_safety = [row for row in hot_dry_selected if risk_flags(row)["safety"]]
    hot_dry_selected_dry = [row for row in hot_dry_selected if risk_flags(row)["dry_performance"]]
    agreement_rate = sum(1 for row in rows if not _is_disagreement(row)) / scorer_steps
    safety_rate = len(hot_dry_selected_safety) / max(len(hot_dry_selected), 1)
    dry_precision = len(hot_dry_selected_dry) / max(len(hot_dry_selected), 1)
    coverage = (
        sum(1 for row in hot_dry_safe_dry if _scorer_selected(row) == "hot_dry_protect")
        / max(len(hot_dry_safe_dry), 1)
    )
    reasons: List[str] = []
    if safety_rate > 0.02:
        reasons.append("hot_dry_protect is still selected in safety-risk rows")
        decision = "unsafe_to_connect"
    elif len(hot_dry_safe_dry) >= 10 and coverage < 0.50:
        reasons.append("hot_dry safe-dry coverage is low; scorer may be too conservative")
        decision = "needs_scorer_calibration"
    elif len(hot_dry_selected) >= 5 and dry_precision < 0.90:
        reasons.append("hot_dry_protect selections are not concentrated in dry-risk rows")
        decision = "needs_intent_or_scorer_review"
    elif agreement_rate < 0.35:
        reasons.append("selector/scorer agreement is low; inspect disagreement windows before connecting")
        decision = "needs_scorer_calibration"
    else:
        reasons.append("scorer decisions look coherent enough for RSPC scoring shadow")
        decision = "ready_for_rspc_shadow_scoring"
    return {
        "decision": decision,
        "reasons": reasons,
        "agreement_rate": round(agreement_rate, 6),
        "hot_dry_selected_steps": len(hot_dry_selected),
        "hot_dry_selected_safety_rate": round(safety_rate, 6),
        "hot_dry_selected_dry_precision": round(dry_precision, 6),
        "hot_dry_safe_dry_steps": len(hot_dry_safe_dry),
        "hot_dry_safe_dry_coverage": round(coverage, 6),
    }


def audit_trace(path: str | Path) -> Dict[str, Any]:
    rows = sorted(read_trace(path), key=lambda row: _num(row, "step", 0.0))
    scorer_rows = [row for row in rows if _has_scorer(row)]
    warnings: List[str] = []
    if rows and not scorer_rows:
        warnings.append("missing_profile_scorer_metadata")
    if not rows:
        warnings.append("empty_trace")
    disagreements = [row for row in scorer_rows if _is_disagreement(row)]
    hot_dry_selected = [row for row in scorer_rows if _scorer_selected(row) == "hot_dry_protect"]
    return {
        **trace_identity(path),
        "path": str(Path(path)),
        "steps": len(rows),
        "scorer_steps": len(scorer_rows),
        "scorer_coverage": round(len(scorer_rows) / max(len(rows), 1), 6),
        "selector_scorer_agreement_steps": len(scorer_rows) - len(disagreements),
        "selector_scorer_disagreement_steps": len(disagreements),
        "intent_regime_matrix": _matrix_by(scorer_rows, lambda row: row.get("intent_regime")),
        "risk_matrix": _risk_matrix(scorer_rows),
        "risk_counts": _risk_counts(scorer_rows),
        "scorer_selected_counts": _counts(_scorer_selected(row) for row in scorer_rows),
        "selector_selected_counts": _counts(_selector_selected(row) for row in scorer_rows),
        "safety_gate_counts": _counts(_gate_reason(row) for row in scorer_rows),
        "safety_gate_matrix": _gate_matrix(scorer_rows),
        "selector_scorer_pair_counts": _counts(
            f"{_selector_selected(row)}->{_scorer_selected(row)}" for row in scorer_rows if _is_disagreement(row)
        ),
        "hot_dry_protect_selected": {
            "steps": len(hot_dry_selected),
            "risk_counts": _risk_counts(hot_dry_selected),
            "mean_score": round(mean(_num(row, "profile_scorer_selected_score", 0.0) for row in hot_dry_selected), 6)
            if hot_dry_selected
            else 0.0,
            "mean_margin": round(mean(_num(row, "profile_scorer_margin_to_second", 0.0) for row in hot_dry_selected), 6)
            if hot_dry_selected
            else 0.0,
        },
        "disagreement_windows": _segments(scorer_rows, _is_disagreement),
        "hot_dry_protect_windows": _segments(scorer_rows, lambda row: _scorer_selected(row) == "hot_dry_protect"),
        "recommendation": _recommendation(scorer_rows),
        "warnings": warnings,
    }


def audit_traces(inputs: Sequence[str | Path]) -> Dict[str, Any]:
    files = discover_traces(inputs)
    traces = [audit_trace(path) for path in files]
    all_rows: List[Mapping[str, Any]] = []
    for path in files:
        all_rows.extend(row for row in read_trace(path) if _has_scorer(row))
    warnings: List[str] = []
    if not files:
        warnings.append("no_trace_files_discovered")
    return {
        "schema_version": "profile_scorer_decision_audit_v1",
        "trace_count": len(traces),
        "scorer_trace_count": sum(1 for trace in traces if int(trace.get("scorer_steps", 0)) > 0),
        "warnings": warnings,
        "aggregate": {
            "steps": len(all_rows),
            "risk_counts": _risk_counts(all_rows),
            "intent_regime_matrix": _matrix_by(all_rows, lambda row: row.get("intent_regime")),
            "risk_matrix": _risk_matrix(all_rows),
            "safety_gate_counts": _counts(_gate_reason(row) for row in all_rows),
            "safety_gate_matrix": _gate_matrix(all_rows),
            "scorer_selected_counts": _counts(_scorer_selected(row) for row in all_rows),
            "selector_selected_counts": _counts(_selector_selected(row) for row in all_rows),
            "selector_scorer_pair_counts": _counts(
                f"{_selector_selected(row)}->{_scorer_selected(row)}" for row in all_rows if _is_disagreement(row)
            ),
            "recommendation": _recommendation(all_rows),
        },
        "traces": traces,
    }


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_report(audit: Mapping[str, Any]) -> str:
    aggregate = audit.get("aggregate", {}) or {}
    recommendation = aggregate.get("recommendation", {}) or {}
    lines = [
        "# Profile Scorer Decision Audit v1",
        "",
        f"- Trace files: {audit.get('trace_count', 0)}",
        f"- Traces with scorer metadata: {audit.get('scorer_trace_count', 0)}",
        f"- Aggregate decision: {recommendation.get('decision', 'unknown')}",
        f"- Aggregate reasons: {'; '.join(recommendation.get('reasons', []) or ['-'])}",
        "",
        "## Trace Decisions",
        "",
        "| trace | controller | scorer steps | agreement | dry risk | safety risk | gate counts | hot-dry selected | hot-dry safety rate | hot-dry dry coverage | decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for trace in audit.get("traces", []) or []:
        rec = trace.get("recommendation", {}) or {}
        scorer_steps = int(trace.get("scorer_steps", 0))
        agreement = int(trace.get("selector_scorer_agreement_steps", 0)) / max(scorer_steps, 1)
        risk_counts = trace.get("risk_counts", {}) or {}
        lines.append(
            "| {trace} | {controller} | {steps} | {agreement:.2f} | {dry} | {safety} | {gates} | {hot} | {hot_safety:.2f} | {coverage:.2f} | {decision} |".format(
                trace=trace.get("trace_id", ""),
                controller=trace.get("controller", ""),
                steps=scorer_steps,
                agreement=agreement,
                dry=int(risk_counts.get("dry_performance", 0)),
                safety=int(risk_counts.get("safety", 0)),
                gates=_fmt_counts(trace.get("safety_gate_counts", {}) or {}, limit=5),
                hot=int(rec.get("hot_dry_selected_steps", 0)),
                hot_safety=float(rec.get("hot_dry_selected_safety_rate", 0.0)),
                coverage=float(rec.get("hot_dry_safe_dry_coverage", 0.0)),
                decision=rec.get("decision", "unknown"),
            )
        )

    lines.extend(["", "## Aggregate Safety Gate Matrix", ""])
    lines.extend(
        [
            "| gate | steps | agreement | risks | regimes | scorer selected | hot-dry selected |",
            "| --- | ---: | ---: | --- | --- | --- | ---: |",
        ]
    )
    for gate, item in sorted((aggregate.get("safety_gate_matrix", {}) or {}).items()):
        steps = int(item.get("steps", 0))
        agreement = int(item.get("agreement_steps", 0)) / max(steps, 1)
        lines.append(
            "| {gate} | {steps} | {agreement:.2f} | {risks} | {regimes} | {scorer} | {hot} |".format(
                gate=gate,
                steps=steps,
                agreement=agreement,
                risks=_fmt_counts(item.get("risk_counts", {}) or {}, limit=6),
                regimes=_fmt_counts(item.get("intent_regime_counts", {}) or {}),
                scorer=_fmt_counts(item.get("scorer_selected_counts", {}) or {}),
                hot=int(item.get("hot_dry_protect_steps", 0)),
            )
        )

    lines.extend(["", "## Aggregate Regime Matrix", ""])
    lines.extend(
        [
            "| regime | steps | agreement | risk counts | selector selected | scorer selected |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for regime, item in sorted((aggregate.get("intent_regime_matrix", {}) or {}).items()):
        steps = int(item.get("steps", 0))
        agreement = int(item.get("agreement_steps", 0)) / max(steps, 1)
        lines.append(
            "| {regime} | {steps} | {agreement:.2f} | {risks} | {selector} | {scorer} |".format(
                regime=regime,
                steps=steps,
                agreement=agreement,
                risks=_fmt_counts(item.get("risk_counts", {}) or {}, limit=6),
                selector=_fmt_counts(item.get("selector_selected_counts", {}) or {}),
                scorer=_fmt_counts(item.get("scorer_selected_counts", {}) or {}),
            )
        )

    lines.extend(["", "## Aggregate Risk Matrix", ""])
    lines.extend(
        [
            "| risk | steps | regimes | selector selected | scorer selected | agreement |",
            "| --- | ---: | --- | --- | --- | ---: |",
        ]
    )
    for risk, item in sorted((aggregate.get("risk_matrix", {}) or {}).items()):
        steps = int(item.get("steps", 0))
        agreement = int(item.get("agreement_steps", 0)) / max(steps, 1)
        lines.append(
            "| {risk} | {steps} | {regimes} | {selector} | {scorer} | {agreement:.2f} |".format(
                risk=risk,
                steps=steps,
                regimes=_fmt_counts(item.get("intent_regime_counts", {}) or {}),
                selector=_fmt_counts(item.get("selector_selected_counts", {}) or {}),
                scorer=_fmt_counts(item.get("scorer_selected_counts", {}) or {}),
                agreement=agreement,
            )
        )

    lines.extend(["", "## Disagreement Windows", ""])
    lines.extend(
        [
            "| trace | steps | duration | regimes | gate | selector | scorer | risks | score | margin | dT | dCO2 | dRH | dew score | hot-dry score | humid score |",
            "| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trace in audit.get("traces", []) or []:
        for window in trace.get("disagreement_windows", []) or []:
            lines.append(
                "| {trace} | {start}-{end} | {duration} | {regimes} | {gate} | {selector} | {scorer} | {risks} | {score:.3f} | {margin:.3f} | {dt:+.3f} | {dc:+.3f} | {drh:+.3f} | {dew:.3f} | {hotdry:.3f} | {humid:.3f} |".format(
                    trace=trace.get("trace_id", ""),
                    start=window.get("start_step"),
                    end=window.get("end_step"),
                    duration=int(window.get("duration_steps", 0)),
                    regimes=_fmt_counts(window.get("intent_regime_counts", {}) or {}),
                    gate=_fmt_counts(window.get("safety_gate_counts", {}) or {}, limit=5),
                    selector=_fmt_counts(window.get("selector_selected_counts", {}) or {}),
                    scorer=_fmt_counts(window.get("scorer_selected_counts", {}) or {}),
                    risks=_fmt_counts(window.get("risk_counts", {}) or {}, limit=6),
                    score=float(window.get("mean_score", 0.0)),
                    margin=float(window.get("mean_margin", 0.0)),
                    dt=float(window.get("mean_delta_target_temp", 0.0)),
                    dc=float(window.get("mean_delta_target_co2", 0.0)),
                    drh=float(window.get("mean_delta_target_rh", 0.0)),
                    dew=float(window.get("mean_dew_risk_score", 0.0)),
                    hotdry=float(window.get("mean_hot_dry_gap_score", 0.0)),
                    humid=float(window.get("mean_humid_gap_score", 0.0)),
                )
            )

    all_warnings = list(audit.get("warnings", []) or [])
    all_warnings.extend(
        f"{trace.get('trace_id')}: {', '.join(trace.get('warnings', []))}"
        for trace in audit.get("traces", []) or []
        if trace.get("warnings")
    )
    if all_warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in all_warnings:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit profile-scorer decisions against climate-risk windows.")
    parser.add_argument("--input-trace", nargs="+", required=True, help="Trace CSV/JSONL files or directories.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    audit = audit_traces(args.input_trace)
    write_json(args.output_json, audit)
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(audit), encoding="utf-8")
    print(f"Saved profile-scorer decision audit JSON to {args.output_json}")
    print(f"Saved profile-scorer decision audit report to {args.output_report}")


if __name__ == "__main__":
    main()
