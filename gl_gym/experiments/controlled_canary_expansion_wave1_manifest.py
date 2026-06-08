"""Build a fixed Wave-1 controlled canary expansion scenario manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BASELINE_CONTROLLER = "llm_rspc_v2"
CANDIDATE_CONTROLLER = "llm_rspc_v2_hot_dry_proposer_strict"
SCENARIOS = (
    {
        "scenario_id": "y2020_d120_s44_n240",
        "year": 2020,
        "day": 120,
        "seed": 44,
        "max_steps": 240,
        "role": "canonical_failure_safety_probe",
    },
    {
        "scenario_id": "y2010_d120_s44_n240",
        "year": 2010,
        "day": 120,
        "seed": 44,
        "max_steps": 240,
        "role": "neutral_radiation_wind_high_humidity_probe",
    },
    {
        "scenario_id": "y2010_d180_s44_n240",
        "year": 2010,
        "day": 180,
        "seed": 44,
        "max_steps": 240,
        "role": "dawn_dew_high_humidity_probe",
    },
    {
        "scenario_id": "y2015_d180_s44_n240",
        "year": 2015,
        "day": 180,
        "seed": 44,
        "max_steps": 240,
        "role": "pure_hot_dry_mixed_dry_dew_probe",
    },
    {
        "scenario_id": "y2020_d180_s44_n240",
        "year": 2020,
        "day": 180,
        "seed": 44,
        "max_steps": 240,
        "role": "pure_hot_dry_mixed_dry_dew_probe",
    },
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def build_report(
    *,
    admission_review: Mapping[str, Any],
    selected_cache_path: str,
    output_root: str,
) -> dict[str, Any]:
    scenario_ids = {item["scenario_id"] for item in SCENARIOS}
    command_groups = [
        {
            "group_id": item["scenario_id"],
            "years": [item["year"]],
            "days": [item["day"]],
            "seeds": [item["seed"]],
            "expected_scenario_ids": [item["scenario_id"]],
            "cartesian_product_safe": True,
        }
        for item in SCENARIOS
    ]
    admission_pass = bool(admission_review.get("controlled_canary_expansion_admission_pass", False))
    exact_fixed_scope = scenario_ids == {
        "y2020_d120_s44_n240",
        "y2010_d120_s44_n240",
        "y2010_d180_s44_n240",
        "y2015_d180_s44_n240",
        "y2020_d180_s44_n240",
    }
    ready = bool(admission_pass and exact_fixed_scope and all(g["cartesian_product_safe"] for g in command_groups))
    return {
        "schema_version": "controlled_canary_expansion_wave1_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "controlled canary expansion wave1 manifest",
            "reopens_rejected_preset": False,
        },
        "wave1_manifest_ready": ready,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "scope": "controlled_canary_expansion_wave1_only" if ready else "blocked",
        "baseline_controller": BASELINE_CONTROLLER,
        "candidate_controller": CANDIDATE_CONTROLLER,
        "controllers": [BASELINE_CONTROLLER, CANDIDATE_CONTROLLER],
        "selected_cache_path": selected_cache_path,
        "output_root": output_root,
        "selected_scenarios": list(SCENARIOS),
        "scenario_ids": [item["scenario_id"] for item in SCENARIOS],
        "command_groups": command_groups,
        "next_action": "run_wave1_cache_coverage_check" if ready else "controlled_canary_expansion_admission_required",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Controlled Canary Expansion Wave-1 Manifest",
        "",
        f"- Ready: {report.get('wave1_manifest_ready', False)}",
        f"- Scope: `{report.get('scope', '')}`",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Scenarios",
        "",
        "| scenario | role |",
        "| --- | --- |",
    ]
    for scenario in report.get("selected_scenarios", []) or []:
        if isinstance(scenario, Mapping):
            lines.append(f"| {scenario.get('scenario_id', '')} | {scenario.get('role', '')} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-review-json", required=True)
    parser.add_argument("--selected-cache-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        admission_review=_load(args.admission_review_json),
        selected_cache_path=args.selected_cache_path,
        output_root=args.output_root,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"wave1_manifest_ready={report['wave1_manifest_ready']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["wave1_manifest_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
