"""Summarise and gate frozen benchmark outputs.

This audit is intentionally read-only: it consumes benchmark JSON files and
emits a compact Markdown/JSON summary so controller changes can be compared
without mixing in LLM plan drift or silent safety regressions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SUMMARY_FIELDS = [
    "steps",
    "total_reward",
    "total_profit",
    "total_temp_violation",
    "total_rh_low_violation",
    "total_rh_high_violation",
    "total_vpd_high_excess",
    "dry_risk_steps",
    "dew_risk_steps",
    "dew_margin_air_lt1_steps",
    "dew_margin_air_lt0_steps",
    "canopy_dew_margin_lt1_steps",
    "canopy_dew_margin_lt0_steps",
    "runtime_error_steps",
    "strict_cache_miss_runtime_error_steps",
    "plan_cache_enabled_steps",
    "plan_cache_hit_steps",
    "profile_generator_shadow_steps",
    "profile_scorer_shadow_steps",
    "profile_rspc_shadow_would_improve_steps",
    "tomato_safety_v2_applied_steps",
    "heat_vent_conflict_steps",
    "co2_leak_steps",
    "lamp_risk_steps",
    "dry_vent_risk_steps",
]

DELTA_FIELDS = [
    "total_reward",
    "total_profit",
    "total_rh_low_violation",
    "total_rh_high_violation",
    "total_vpd_high_excess",
    "total_temp_violation",
    "dew_margin_air_lt0_steps",
    "canopy_dew_margin_lt0_steps",
]

SUM_FIELDS = [
    "steps",
    "runtime_error_steps",
    "strict_cache_miss_runtime_error_steps",
    "plan_cache_enabled_steps",
    "plan_cache_hit_steps",
    "dry_risk_steps",
    "dew_risk_steps",
    "dew_margin_air_lt0_steps",
    "canopy_dew_margin_lt0_steps",
    "tomato_safety_v2_applied_steps",
    "dry_vent_risk_steps",
]


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}"
    return str(value)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_summaries(payload: Any, source: str) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        summaries = payload.get("summaries")
        if isinstance(summaries, list):
            for summary in summaries:
                if isinstance(summary, dict):
                    row = dict(summary)
                    row["_source"] = source
                    yield row
            return
        # Some scripts emit a bare list-like summary under other names.
        for key in ("results", "rows"):
            values = payload.get(key)
            if isinstance(values, list):
                for summary in values:
                    if isinstance(summary, dict):
                        row = dict(summary)
                        row["_source"] = source
                        yield row
                return
    elif isinstance(payload, list):
        for summary in payload:
            if isinstance(summary, dict):
                row = dict(summary)
                row["_source"] = source
                yield row


def load_summary_rows(paths: Sequence[str | Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            payload = _load_json(path)
        except FileNotFoundError:
            warnings.append(f"missing_input:{path}")
            continue
        except json.JSONDecodeError as exc:
            warnings.append(f"invalid_json:{path}:{exc}")
            continue

        before = len(rows)
        rows.extend(_iter_summaries(payload, str(path)))
        if len(rows) == before:
            warnings.append(f"no_summaries:{path}")
    return rows, warnings


def _normalise_summary(summary: dict[str, Any]) -> dict[str, Any]:
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, dict):
        aggregate = summary
    job = summary.get("job") if isinstance(summary.get("job"), dict) else {}
    scenario_id = (
        summary.get("scenario_id")
        or aggregate.get("scenario_id")
        or job.get("scenario_id")
        or _scenario_id_from_job(job)
        or "unknown"
    )
    controller = summary.get("controller") or aggregate.get("controller") or job.get("controller") or "unknown"

    row: dict[str, Any] = {
        "source": summary.get("_source", ""),
        "scenario_id": str(scenario_id),
        "controller": str(controller),
    }
    for field in SUMMARY_FIELDS:
        row[field] = _num(aggregate.get(field, summary.get(field, 0.0)))
    return row


def _scenario_id_from_job(job: dict[str, Any]) -> str | None:
    year = job.get("year")
    day = job.get("day")
    seed = job.get("seed")
    max_steps = job.get("max_steps")
    if year is None or day is None or seed is None:
        return None
    scenario = f"y{year}_d{day}_s{seed}"
    if max_steps is not None:
        scenario += f"_n{max_steps}"
    return scenario


def _row_gate_failures(row: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    scenario = row["scenario_id"]
    controller = row["controller"]
    if row["runtime_error_steps"] > 0:
        failures.append(
            {
                "type": "runtime_error",
                "scenario_id": scenario,
                "controller": controller,
                "value": row["runtime_error_steps"],
                "message": "runtime_error_steps > 0",
            }
        )
    if row["strict_cache_miss_runtime_error_steps"] > 0:
        failures.append(
            {
                "type": "strict_cache_miss",
                "scenario_id": scenario,
                "controller": controller,
                "value": row["strict_cache_miss_runtime_error_steps"],
                "message": "strict_cache_miss_runtime_error_steps > 0",
            }
        )
    if row["plan_cache_enabled_steps"] > 0 and row["plan_cache_hit_steps"] < row["plan_cache_enabled_steps"]:
        failures.append(
            {
                "type": "incomplete_cache_hit",
                "scenario_id": scenario,
                "controller": controller,
                "value": row["plan_cache_hit_steps"],
                "expected": row["plan_cache_enabled_steps"],
                "message": "plan_cache_hit_steps < plan_cache_enabled_steps",
            }
        )
    return failures


def _build_pairs(
    rows: Sequence[dict[str, Any]],
    baseline_controller: str,
    compare_controller: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    warnings: list[str] = []
    for row in rows:
        key = (row["scenario_id"], row["controller"])
        if key in by_key:
            warnings.append(f"duplicate_summary:{row['scenario_id']}:{row['controller']}")
        by_key[key] = row

    scenario_ids = sorted({row["scenario_id"] for row in rows})
    paired: list[dict[str, Any]] = []
    unpaired: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        baseline = by_key.get((scenario_id, baseline_controller))
        compare = by_key.get((scenario_id, compare_controller))
        if baseline is None or compare is None:
            controllers = sorted(row["controller"] for row in rows if row["scenario_id"] == scenario_id)
            unpaired.append(
                {
                    "scenario_id": scenario_id,
                    "available_controllers": controllers,
                    "missing_baseline": baseline is None,
                    "missing_compare": compare is None,
                }
            )
            continue

        pair: dict[str, Any] = {
            "scenario_id": scenario_id,
            "baseline_controller": baseline_controller,
            "compare_controller": compare_controller,
        }
        for field in DELTA_FIELDS:
            pair[f"d_{field}"] = compare[field] - baseline[field]
        pair["baseline"] = {field: baseline[field] for field in DELTA_FIELDS}
        pair["compare"] = {field: compare[field] for field in DELTA_FIELDS}
        paired.append(pair)
    return paired, unpaired, warnings


def _pair_gate_failures(pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for pair in pairs:
        scenario = pair["scenario_id"]
        if pair["d_canopy_dew_margin_lt0_steps"] > 0:
            failures.append(
                {
                    "type": "canopy_lt0_regression",
                    "scenario_id": scenario,
                    "controller": pair["compare_controller"],
                    "baseline_controller": pair["baseline_controller"],
                    "value": pair["d_canopy_dew_margin_lt0_steps"],
                    "message": "canopy_dew_margin_lt0_steps increased against baseline",
                }
            )
        if pair["d_dew_margin_air_lt0_steps"] > 0:
            failures.append(
                {
                    "type": "dew_air_lt0_regression",
                    "scenario_id": scenario,
                    "controller": pair["compare_controller"],
                    "baseline_controller": pair["baseline_controller"],
                    "value": pair["d_dew_margin_air_lt0_steps"],
                    "message": "dew_margin_air_lt0_steps increased against baseline",
                }
            )
    return failures


def _controller_summary(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"scenario_count": 0})
    for row in rows:
        controller = row["controller"]
        grouped[controller]["scenario_count"] += 1
        for field in SUM_FIELDS:
            grouped[controller][field] = grouped[controller].get(field, 0.0) + row[field]
    return {controller: dict(values) for controller, values in sorted(grouped.items())}


def audit_benchmark_summary(
    inputs: Sequence[str | Path],
    baseline_controller: str = "llm_rspc_v2",
    compare_controller: str = "llm_rspc_v2_hot_dry_proposer_strict",
) -> dict[str, Any]:
    raw_rows, warnings = load_summary_rows(inputs)
    rows = [_normalise_summary(summary) for summary in raw_rows]
    row_failures: list[dict[str, Any]] = []
    for row in rows:
        row_failures.extend(_row_gate_failures(row))

    paired_deltas, unpaired, pair_warnings = _build_pairs(rows, baseline_controller, compare_controller)
    warnings.extend(pair_warnings)
    pair_failures = _pair_gate_failures(paired_deltas)
    gate_failures = row_failures + pair_failures

    decision = "pass" if not gate_failures else "fail"
    return {
        "schema_version": "frozen_benchmark_summary_audit_v1",
        "inputs": [str(Path(path)) for path in inputs],
        "baseline_controller": baseline_controller,
        "compare_controller": compare_controller,
        "aggregate": {
            "decision": decision,
            "summary_count": len(rows),
            "scenario_count": len({row["scenario_id"] for row in rows}),
            "controller_count": len({row["controller"] for row in rows}),
            "paired_count": len(paired_deltas),
            "unpaired_count": len(unpaired),
            "failure_count": len(gate_failures),
            "warning_count": len(warnings) + len(unpaired),
        },
        "controller_summary": _controller_summary(rows),
        "rows": rows,
        "paired_deltas": paired_deltas,
        "unpaired": unpaired,
        "gate_failures": gate_failures,
        "warnings": warnings,
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not rows:
        return "_None._\n"
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header, separator, *body]) + "\n"


def build_markdown_report(audit: dict[str, Any]) -> str:
    aggregate = audit["aggregate"]
    lines = [
        "# Frozen Benchmark Summary Audit",
        "",
        f"- Decision: **{aggregate['decision'].upper()}**",
        f"- Inputs: {', '.join(audit['inputs'])}",
        f"- Baseline controller: `{audit['baseline_controller']}`",
        f"- Compare controller: `{audit['compare_controller']}`",
        f"- Summaries: {aggregate['summary_count']}",
        f"- Paired scenarios: {aggregate['paired_count']}",
        f"- Unpaired scenarios: {aggregate['unpaired_count']}",
        f"- Gate failures: {aggregate['failure_count']}",
        "",
        "## Gate Failures",
        "",
    ]
    failure_rows = [
        [
            failure.get("scenario_id", ""),
            failure.get("controller", ""),
            failure.get("type", ""),
            failure.get("value", ""),
            failure.get("expected", ""),
            failure.get("message", ""),
        ]
        for failure in audit["gate_failures"]
    ]
    lines.append(_markdown_table(["scenario", "controller", "type", "value", "expected", "message"], failure_rows))

    lines.extend(["", "## Controller Summary", ""])
    controller_rows = []
    for controller, values in audit["controller_summary"].items():
        cache_enabled = values.get("plan_cache_enabled_steps", 0.0)
        cache_hits = values.get("plan_cache_hit_steps", 0.0)
        hit_rate = cache_hits / cache_enabled if cache_enabled else 0.0
        controller_rows.append(
            [
                controller,
                values.get("scenario_count", 0),
                values.get("steps", 0),
                values.get("runtime_error_steps", 0),
                values.get("strict_cache_miss_runtime_error_steps", 0),
                f"{hit_rate:.3f}",
                values.get("dry_risk_steps", 0),
                values.get("dew_risk_steps", 0),
                values.get("canopy_dew_margin_lt0_steps", 0),
            ]
        )
    lines.append(
        _markdown_table(
            [
                "controller",
                "scenarios",
                "steps",
                "runtime",
                "cache_miss_runtime",
                "cache_hit_rate",
                "dry_risk",
                "dew_risk",
                "canopy_lt0",
            ],
            controller_rows,
        )
    )

    lines.extend(["", "## Paired Deltas", ""])
    delta_rows = [
        [
            pair["scenario_id"],
            pair["d_total_reward"],
            pair["d_total_profit"],
            pair["d_total_rh_low_violation"],
            pair["d_total_vpd_high_excess"],
            pair["d_total_temp_violation"],
            pair["d_dew_margin_air_lt0_steps"],
            pair["d_canopy_dew_margin_lt0_steps"],
        ]
        for pair in audit["paired_deltas"]
    ]
    lines.append(
        _markdown_table(
            [
                "scenario",
                "d_reward",
                "d_profit",
                "d_RHlow",
                "d_VPDhi",
                "d_temp",
                "d_dew_lt0",
                "d_canopy_lt0",
            ],
            delta_rows,
        )
    )

    lines.extend(["", "## Scenario Rows", ""])
    row_rows = [
        [
            row["scenario_id"],
            row["controller"],
            row["steps"],
            row["total_reward"],
            row["total_profit"],
            row["total_rh_low_violation"],
            row["total_vpd_high_excess"],
            row["total_temp_violation"],
            row["plan_cache_hit_steps"],
            row["plan_cache_enabled_steps"],
            row["runtime_error_steps"],
        ]
        for row in sorted(audit["rows"], key=lambda item: (item["scenario_id"], item["controller"]))
    ]
    lines.append(
        _markdown_table(
            [
                "scenario",
                "controller",
                "steps",
                "reward",
                "profit",
                "RHlow",
                "VPDhi",
                "temp",
                "cache_hit",
                "cache_enabled",
                "runtime",
            ],
            row_rows,
        )
    )

    if audit["unpaired"]:
        lines.extend(["", "## Unpaired Scenarios", ""])
        unpaired_rows = [
            [
                item["scenario_id"],
                ", ".join(item["available_controllers"]),
                item["missing_baseline"],
                item["missing_compare"],
            ]
            for item in audit["unpaired"]
        ]
        lines.append(_markdown_table(["scenario", "controllers", "missing_baseline", "missing_compare"], unpaired_rows))

    if audit["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in audit["warnings"])
        lines.append("")

    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Frozen benchmark JSON file(s).")
    parser.add_argument("--output-md", required=True, help="Markdown report output path.")
    parser.add_argument("--output-json", required=True, help="JSON audit output path.")
    parser.add_argument("--baseline-controller", default="llm_rspc_v2")
    parser.add_argument("--compare-controller", default="llm_rspc_v2_hot_dry_proposer_strict")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = audit_benchmark_summary(
        args.input,
        baseline_controller=args.baseline_controller,
        compare_controller=args.compare_controller,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(audit), encoding="utf-8")
    print(f"decision={audit['aggregate']['decision']} failures={audit['aggregate']['failure_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 1 if audit["aggregate"]["decision"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
