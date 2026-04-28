"""Mine humidity-economic memory cases from matched PPO and LLM-RSPC traces."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gl_gym.agent.humidity_experience_memory import (
    HumidityExperienceConfig,
    HumidityExperienceMemory,
    _float,
    assess_free_air_exchange_pair,
    build_experience_from_pair,
)


def _coerce_value(value: Any) -> Any:
    if value in ("", None):
        return value
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    try:
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        number = float(text)
        if number.is_integer() and not any(ch in text for ch in ".eE"):
            return int(number)
        return number
    except Exception:
        return value


def load_rows(path: str | Path) -> List[Dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("rows"), list):
            return [dict(row) for row in data["rows"]]
        if isinstance(data, list):
            return [dict(row) for row in data]
        raise ValueError(f"Unsupported JSON trace format: {source}")
    if source.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(dict(json.loads(line)))
        return rows
    if source.suffix.lower() == ".csv":
        with source.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [{key: _coerce_value(value) for key, value in row.items()} for row in reader]
    raise ValueError(f"Unsupported trace file: {source}")


def pair_rows(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[int, int, int, int], Dict[str, Dict[str, Any]]]:
    paired: Dict[Tuple[int, int, int, int], Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        algo = str(row.get("algo", "unknown"))
        if algo not in {"ppo", "llm_director"}:
            continue
        key = (
            int(_float(row, "year", 0.0)),
            int(_float(row, "day", 0.0)),
            int(_float(row, "seed", 0.0)),
            int(_float(row, "step", 0.0)),
        )
        paired.setdefault(key, {})[algo] = row
    return paired


def future_window_metrics(
    paired: Dict[Tuple[int, int, int, int], Dict[str, Dict[str, Any]]],
    key: Tuple[int, int, int, int],
    window: int,
) -> Dict[str, float]:
    if window <= 0:
        return {}
    year, day, seed, step = key
    ppo_profit = llm_profit = 0.0
    ppo_rh = llm_rh = 0.0
    ppo_temp = llm_temp = 0.0
    count = 0
    for offset in range(1, int(window) + 1):
        item = paired.get((year, day, seed, step + offset), {})
        ppo = item.get("ppo")
        llm = item.get("llm_director")
        if ppo is None or llm is None:
            continue
        count += 1
        ppo_profit += _float(ppo, "profit", 0.0)
        llm_profit += _float(llm, "profit", 0.0)
        ppo_rh += _float(ppo, "rh_violation", 0.0)
        llm_rh += _float(llm, "rh_violation", 0.0)
        ppo_temp += _float(ppo, "temp_violation", 0.0)
        llm_temp += _float(llm, "temp_violation", 0.0)
    return {
        "future_window_count": float(count),
        "future_profit_advantage_ppo_minus_llm": float(ppo_profit - llm_profit),
        "future_rh_violation_extra_ppo_minus_llm": float(ppo_rh - llm_rh),
        "future_temp_violation_extra_ppo_minus_llm": float(ppo_temp - llm_temp),
    }


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Humidity Experience Mining Report",
        "",
        f"- Source trace: `{summary['source_trace']}`",
        f"- Paired steps: {summary['paired_steps']}",
        f"- Accepted raw cases: {summary['accepted_raw_cases']}",
        f"- Stored memory cases after merge: {summary['stored_memory_cases']}",
        f"- Accepted risk-intent override cases: {summary['accepted_risk_override_cases']}",
        f"- Rejected cases: {summary['rejected_cases']}",
        "",
        "## Reject Reasons",
    ]
    for reason, count in sorted(summary["reject_reason_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {reason}: {count}")
    lines.extend(["", "## Accepted Evidence Means"])
    for key, value in sorted(summary["accepted_metric_means"].items()):
        lines.append(f"- {key}: {value:.6f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine humidity-economic memory cases from matched traces.")
    parser.add_argument("--trace", type=str, default="gl_gym/result/diagnostics/ppo_vs_llm_labelaudit_v2_s24_day240.json")
    parser.add_argument("--output-memory", type=str, default="gl_gym/result/experience_memory/humidity_memory.jsonl")
    parser.add_argument("--output-rejected", type=str, default="gl_gym/result/experience_memory/humidity_memory_rejected.jsonl")
    parser.add_argument("--output-summary", type=str, default="gl_gym/result/experience_memory/humidity_memory_summary.json")
    parser.add_argument("--output-report", type=str, default="gl_gym/result/experience_memory/humidity_memory_report.md")
    parser.add_argument("--future-window", type=int, default=6)
    parser.add_argument("--max-future-rh-extra", type=float, default=0.05)
    parser.add_argument("--max-future-temp-extra", type=float, default=0.05)
    parser.add_argument("--min-future-profit-advantage", type=float, default=-0.0002)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--min-strategy-confidence", type=float, default=0.55)
    parser.add_argument("--min-intent-confidence", type=float, default=0.60)
    args = parser.parse_args()

    config = HumidityExperienceConfig(
        min_strategy_confidence=float(args.min_strategy_confidence),
        min_intent_confidence=float(args.min_intent_confidence),
    )
    rows = load_rows(args.trace)
    paired = pair_rows(rows)
    memory = HumidityExperienceMemory(config=config)
    rejected: List[Dict[str, Any]] = []
    accepted_metrics: List[Dict[str, float]] = []

    for key in sorted(paired):
        item = paired[key]
        ppo = item.get("ppo")
        llm = item.get("llm_director")
        if ppo is None or llm is None:
            continue
        experience, assessment = build_experience_from_pair(ppo, llm, config=config)
        future = future_window_metrics(paired, key, int(args.future_window))
        future_reasons: List[str] = []
        if experience is not None and future:
            experience.evidence.update(future)
            if future["future_window_count"] > 0:
                if future["future_rh_violation_extra_ppo_minus_llm"] > float(args.max_future_rh_extra):
                    future_reasons.append("future_rh_violation_worse_than_llm")
                if future["future_temp_violation_extra_ppo_minus_llm"] > float(args.max_future_temp_extra):
                    future_reasons.append("future_temp_violation_worse_than_llm")
                if future["future_profit_advantage_ppo_minus_llm"] < float(args.min_future_profit_advantage):
                    future_reasons.append("future_profit_advantage_too_low")
        if experience is not None and not future_reasons:
            accepted_metrics.append(dict(experience.evidence))
            memory.add(experience, merge=not bool(args.no_merge))
        else:
            reasons = tuple(assessment.reasons) + tuple(future_reasons)
            rejected.append(
                {
                    "scenario": {"year": key[0], "day": key[1], "seed": key[2], "step": key[3]},
                    "accepted_before_future_check": bool(assessment.accepted),
                    "reasons": list(reasons),
                    "metrics": {**assessment.metrics, **future},
                    "intent_label": assessment.intent_label,
                    "intent_confidence": assessment.intent_confidence,
                    "strategy_label": assessment.strategy_label,
                    "strategy_confidence": assessment.strategy_confidence,
                }
            )

    memory.save_jsonl(args.output_memory)
    write_jsonl(args.output_rejected, rejected)
    reason_counts: Dict[str, int] = {}
    for row in rejected:
        for reason in row["reasons"]:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    metric_means: Dict[str, float] = {}
    if accepted_metrics:
        keys = sorted({key for row in accepted_metrics for key in row})
        for key in keys:
            values = [float(row[key]) for row in accepted_metrics if key in row]
            metric_means[key] = float(sum(values) / len(values))
    summary = {
        "source_trace": str(args.trace),
        "paired_steps": sum(1 for item in paired.values() if "ppo" in item and "llm_director" in item),
        "accepted_raw_cases": len(accepted_metrics),
        "stored_memory_cases": len(memory.experiences),
        "accepted_risk_override_cases": sum(
            1 for metrics in accepted_metrics if float(metrics.get("plan_intent_overridden", 0.0)) > 0.5
        ),
        "rejected_cases": len(rejected),
        "reject_reason_counts": reason_counts,
        "accepted_metric_means": metric_means,
        "config": config.__dict__,
    }
    out_summary = Path(args.output_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_report).write_text(build_report(summary), encoding="utf-8")
    print(build_report(summary))


if __name__ == "__main__":
    main()
