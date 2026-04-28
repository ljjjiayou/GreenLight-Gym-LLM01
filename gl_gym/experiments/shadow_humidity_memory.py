"""Shadow-evaluate humidity experience memory without changing control."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gl_gym.agent.humidity_experience_memory import (
    HumidityExperienceConfig,
    HumidityExperienceMemory,
    _float,
    action_from_row,
    action_proxy_cost,
    heat_vent_conflict,
    risk_based_economic_intent_override,
    screen_free_air_exchange_context,
)
from gl_gym.experiments.mine_humidity_experiences import load_rows


def control_to_record(action: Iterable[float], prefix: str) -> Dict[str, float]:
    names = ("heating", "co2", "screen", "ventilation", "lighting", "shading")
    values = list(action)
    return {f"{prefix}_{name}": float(values[i]) if i < len(values) else 0.0 for i, name in enumerate(names)}


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Humidity Memory Shadow Report",
        "",
        f"- Trace: `{summary['trace']}`",
        f"- Memory: `{summary['memory']}`",
        f"- Evaluated LLM steps: {summary['evaluated_steps']}",
        f"- Memory hit steps: {summary['memory_hit_steps']}",
        f"- Risk-intent override context steps: {summary['risk_override_context_steps']}",
        f"- Hit rate: {summary['hit_rate']:.3f}",
        f"- Mean proxy cost delta (memory - LLM): {summary['mean_proxy_cost_delta']:.6f}",
        f"- Mean heat+vent delta (memory - LLM): {summary['mean_heat_vent_delta']:.6f}",
        "",
        "## Context Reject Reasons",
    ]
    for reason, count in sorted(summary["context_reject_reason_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {reason}: {count}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Shadow-evaluate humidity memory candidates on an existing trace.")
    parser.add_argument("--trace", type=str, default="gl_gym/result/diagnostics/ppo_vs_llm_labelaudit_v2_s24_day240.json")
    parser.add_argument("--memory-jsonl", type=str, default="gl_gym/result/experience_memory/humidity_memory.jsonl")
    parser.add_argument("--output-json", type=str, default="gl_gym/result/experience_memory/humidity_memory_shadow.json")
    parser.add_argument("--output-csv", type=str, default="gl_gym/result/experience_memory/humidity_memory_shadow.csv")
    parser.add_argument("--output-report", type=str, default="gl_gym/result/experience_memory/humidity_memory_shadow.md")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-distance", type=float, default=1.15)
    parser.add_argument("--min-trust", type=float, default=0.50)
    args = parser.parse_args()

    config = HumidityExperienceConfig(retrieval_max_distance=float(args.max_distance), min_trust=float(args.min_trust))
    memory = HumidityExperienceMemory.load_jsonl(args.memory_jsonl, config=config)
    rows = [row for row in load_rows(args.trace) if str(row.get("algo", "")) == "llm_director"]
    shadow_rows: List[Dict[str, Any]] = []
    context_reasons: Dict[str, int] = {}
    cost_deltas: List[float] = []
    conflict_deltas: List[float] = []

    for row in rows:
        reasons, context_metrics = screen_free_air_exchange_context(row, target_row=row, config=config)
        risk_override = risk_based_economic_intent_override(row, context_metrics, reasons, config)
        if risk_override:
            reasons = tuple(reason for reason in reasons if reason != "target_rh_gap_outside_band")
            context_metrics["risk_based_intent_override"] = 1.0
        else:
            context_metrics["risk_based_intent_override"] = 0.0
        if reasons:
            for reason in reasons:
                context_reasons[reason] = context_reasons.get(reason, 0) + 1
        matches = memory.retrieve(row, target_row=row, top_k=int(args.top_k))
        base_action = action_from_row(row)
        record: Dict[str, Any] = {
            "year": int(_float(row, "year", 0.0)),
            "day": int(_float(row, "day", 0.0)),
            "seed": int(_float(row, "seed", 0.0)),
            "step": int(_float(row, "step", 0.0)),
            "rh_air": _float(row, "rh_air", 0.0),
            "temp_air": _float(row, "temp_air", 0.0),
            "target_rh": _float(row, "target_rh", 0.0),
            "context_reasons": ";".join(reasons),
            "memory_hit": bool(matches),
            "context_rh_gap": context_metrics.get("rh_gap", 0.0),
            "context_vpd_kpa": context_metrics.get("vpd_kpa", 0.0),
            "context_risk_override": bool(risk_override),
            **control_to_record(base_action, "llm"),
        }
        if matches:
            match = matches[0]
            candidate = match["candidate_action"]
            proxy_delta = action_proxy_cost(candidate) - action_proxy_cost(base_action)
            conflict_delta = heat_vent_conflict(candidate) - heat_vent_conflict(base_action)
            cost_deltas.append(float(proxy_delta))
            conflict_deltas.append(float(conflict_delta))
            record.update(
                {
                    "case_id": match["case_id"],
                    "distance": float(match["distance"]),
                    "trust": float(match["trust"]),
                    "support_count": int(match["support_count"]),
                    "proxy_cost_delta_mem_minus_llm": float(proxy_delta),
                    "heat_vent_delta_mem_minus_llm": float(conflict_delta),
                    **control_to_record(candidate, "memory"),
                }
            )
        shadow_rows.append(record)

    summary = {
        "trace": str(args.trace),
        "memory": str(args.memory_jsonl),
        "memory_cases": len(memory.experiences),
        "evaluated_steps": len(rows),
        "memory_hit_steps": sum(1 for row in shadow_rows if row.get("memory_hit")),
        "risk_override_context_steps": sum(1 for row in shadow_rows if row.get("context_risk_override")),
        "hit_rate": float(sum(1 for row in shadow_rows if row.get("memory_hit")) / max(1, len(rows))),
        "mean_proxy_cost_delta": float(sum(cost_deltas) / len(cost_deltas)) if cost_deltas else 0.0,
        "mean_heat_vent_delta": float(sum(conflict_deltas) / len(conflict_deltas)) if conflict_deltas else 0.0,
        "context_reject_reason_counts": context_reasons,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "rows": shadow_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv, shadow_rows)
    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_report).write_text(build_report(summary), encoding="utf-8")
    print(build_report(summary))


if __name__ == "__main__":
    main()
