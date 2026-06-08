"""Plan expanded cache-coverage checks before any metadata replay execution.

This is a preflight artifact only. It reads existing manifests and cache
metadata, but it does not run replay, fill cache, or call an online LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.metadata_replay_cache_scenario_plan import _load_cache_env_ids, _parse_scenario_id


FAMILY_ORDER = [
    "canonical_failure",
    "neutral_no_risk",
    "pure_hot_dry",
    "mixed_dry_dew",
    "dawn_dew_or_high_humidity",
    "radiation_spike",
    "wind_stress",
]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _scenario_ids_from_counterexample_suite(suite: Mapping[str, Any], family: str) -> list[str]:
    regime_names = {
        "dawn_dew_or_high_humidity": {"dawn_dew", "cold_humid"},
    }.get(family, {family})
    out: list[str] = []
    for row in suite.get("windows", []) or []:
        if not isinstance(row, Mapping) or str(row.get("regime", "")) not in regime_names:
            continue
        scenario_id = str(row.get("scenario_id", ""))
        if scenario_id and scenario_id not in out:
            out.append(scenario_id)
    return out


def _scenario_ids_from_default_path_plan(plan: Mapping[str, Any], family: str) -> list[str]:
    out: list[str] = []
    family_tokens = {
        "canonical_failure": ("canonical",),
        "pure_hot_dry": ("pure hot-dry", "pure_hot_dry"),
        "mixed_dry_dew": ("mixed dry-dew", "mixed_dry_dew"),
        "dawn_dew_or_high_humidity": ("dawn-dew", "dawn_dew", "high humidity"),
    }.get(family, (family,))
    for item in plan.get("scenario_families", []) or []:
        text = str(item)
        lower = text.lower()
        if not any(token in lower for token in family_tokens):
            continue
        for token in text.replace(":", " ").split():
            if token.startswith("y") and "_d" in token and "_s" in token:
                if token not in out:
                    out.append(token)
    return out


def _parse_many(scenario_ids: Sequence[str], max_steps: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        try:
            rows.append(_parse_scenario_id(scenario_id, max_steps=max_steps))
        except ValueError:
            rows.append({"scenario_id": scenario_id, "env_id": "", "parse_error": True, "max_steps": max_steps})
    return rows


def build_report(
    *,
    cache_scenario_plan: Mapping[str, Any],
    counterexample_suite: Mapping[str, Any] | None = None,
    default_path_plan: Mapping[str, Any] | None = None,
    cache_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    selected_cache_path = str(cache_scenario_plan.get("selected_cache_path", ""))
    max_steps = int(cache_scenario_plan.get("max_steps", 240) or 240)
    selected_cache_env_ids = _load_cache_env_ids(_resolve(selected_cache_path)) if selected_cache_path else set()
    canonical_ids = [
        str(row.get("scenario_id", ""))
        for row in cache_scenario_plan.get("scenario_list", []) or []
        if isinstance(row, Mapping) and row.get("scenario_family") == "canonical_failure"
    ]
    rows: list[dict[str, Any]] = []
    for family in FAMILY_ORDER:
        scenario_ids: list[str] = []
        sources: list[str] = []
        if family == "canonical_failure":
            scenario_ids.extend([sid for sid in canonical_ids if sid])
            if scenario_ids:
                sources.append("metadata_replay_cache_scenario_plan")
        if counterexample_suite:
            suite_ids = _scenario_ids_from_counterexample_suite(counterexample_suite, family)
            scenario_ids.extend([sid for sid in suite_ids if sid not in scenario_ids])
            if suite_ids:
                sources.append("regime_counterexample_suite_v1_20260520")
        if default_path_plan:
            plan_ids = _scenario_ids_from_default_path_plan(default_path_plan, family)
            scenario_ids.extend([sid for sid in plan_ids if sid not in scenario_ids])
            if plan_ids or family in {"pure_hot_dry", "mixed_dry_dew"}:
                sources.append("default_path_action_invariance_plan_20260522")
        scenario_rows = _parse_many(scenario_ids, max_steps=max_steps)
        env_presence = {
            row.get("scenario_id", ""): {
                "env_id": row.get("env_id", ""),
                "present_in_selected_cache": bool(row.get("env_id") and row.get("env_id") in selected_cache_env_ids),
            }
            for row in scenario_rows
        }
        if family == "canonical_failure" and bool(cache_coverage and cache_coverage.get("cache_coverage_pass", False)):
            status = "canonical_coverage_passed"
        elif not scenario_rows:
            status = "blocked_missing_cache_or_scenario"
        elif all(item["present_in_selected_cache"] for item in env_presence.values()):
            status = "coverage_available_pending_no_fill_check"
        else:
            status = "blocked_missing_cache_or_scenario"
        rows.append(
            {
                "family": family,
                "source_artifacts": sources,
                "scenario_ids": scenario_ids,
                "scenario_count": len(scenario_rows),
                "env_presence": env_presence,
                "coverage_status": status,
                "can_enter_initial_strict_metadata_replay_plan": family == "canonical_failure" and status == "canonical_coverage_passed",
            }
        )
    all_expanded_ready = all(row["coverage_status"] != "blocked_missing_cache_or_scenario" for row in rows)
    return {
        "schema_version": "metadata_replay_expanded_cache_coverage_plan_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / expanded cache coverage plan",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "metadata_replay_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "selected_cache_path": selected_cache_path,
        "expanded_cache_coverage_plan_ready": True,
        "expanded_cache_coverage_ready": all_expanded_ready,
        "initial_strict_metadata_replay_scope": "canonical_failure_only",
        "rows": rows,
        "next_action": "draft_canonical_strict_metadata_replay_run_plan",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Metadata Replay Expanded Cache Coverage Plan",
        "",
        "- Mode: audit-only / expanded cache coverage plan",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Expanded cache coverage ready: {report.get('expanded_cache_coverage_ready', False)}",
        f"- Initial strict metadata replay scope: `{report.get('initial_strict_metadata_replay_scope', '')}`",
        "",
        "| family | scenarios | status | initial plan |",
        "| --- | ---: | --- | --- |",
    ]
    for row in report.get("rows", []) or []:
        if isinstance(row, Mapping):
            lines.append(
                f"| `{row.get('family', '')}` | {row.get('scenario_count', 0)} | `{row.get('coverage_status', '')}` | {row.get('can_enter_initial_strict_metadata_replay_plan', False)} |"
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-scenario-plan-json", required=True)
    parser.add_argument("--counterexample-suite-json", default="")
    parser.add_argument("--default-path-plan-json", default="")
    parser.add_argument("--cache-coverage-json", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        cache_scenario_plan=_load(args.cache_scenario_plan_json),
        counterexample_suite=_load(args.counterexample_suite_json) if args.counterexample_suite_json else None,
        default_path_plan=_load(args.default_path_plan_json) if args.default_path_plan_json else None,
        cache_coverage=_load(args.cache_coverage_json) if args.cache_coverage_json else None,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"expanded_cache_coverage_ready={report['expanded_cache_coverage_ready']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
