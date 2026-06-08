"""Design runtime provenance instrumentation for post-guardrail rewrites.

This is a design artifact only. It does not modify the controller.
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


REQUIRED_FIELDS = [
    "rule_id",
    "rule_family",
    "rule_priority",
    "rule_reason",
    "source_function",
    "pre_rule_action",
    "post_rule_action",
    "delta_action",
    "rule_input_features",
    "expected_safety_benefit",
]

HOOKS = [
    {
        "hook_id": "apply_safety_guardrails",
        "source_function": "apply_safety_guardrails",
        "expected_rule_families": ["dry_vpd_vent_cap", "heat_vent_conflict", "wind_cap"],
        "required": True,
    },
    {
        "hook_id": "humidity_memory_final_shape",
        "source_function": "_apply_humidity_memory_final_shape",
        "expected_rule_families": ["humidity_memory_free_air_exchange", "night_vent_cap"],
        "required": True,
    },
    {
        "hook_id": "tomato_safety_v2_wrapper",
        "source_function": "_apply_tomato_safety_v2",
        "expected_rule_families": ["tomato_safety_v2"],
        "required": True,
    },
    {
        "hook_id": "rspc_post_score_shaping",
        "source_function": "_apply_rspc_post_score_shadow_shape",
        "expected_rule_families": ["rspc_dry_recovery", "rh_risk_shape", "target_tracking"],
        "required": True,
    },
]


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def build_report(static_provenance: Mapping[str, Any]) -> dict[str, Any]:
    missing = int(_num(static_provenance.get("runtime_reason_missing_count")))
    return {
        "schema_version": "post_guardrail_runtime_provenance_design_v1",
        "mainline_alignment": {
            "affected_layers": ["Safety Boundary", "Evaluation"],
            "default_llm_rspc_v2_changed": False,
            "mode": "design-only / no controller edit",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "controller_files_modified_by_this_design": False,
        "runtime_provenance_instrumentation_pending": missing > 0,
        "target_runtime_reason_missing_count": 0,
        "current_runtime_reason_missing_count": missing,
        "required_fields": REQUIRED_FIELDS,
        "hook_map": HOOKS,
        "acceptance_criteria": {
            "unknown_post_guardrail_rewrite_count": 0,
            "runtime_reason_missing_count": 0,
            "all_rewrites_have_rule_id_and_reason": True,
            "action_diff_must_remain_zero_for_metadata_only_run": True,
        },
        "notes": [
            "This report does not add runtime instrumentation.",
            "Actual instrumentation in llm_agent.py requires a separate explicit implementation step.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Post-Guardrail Runtime Provenance Instrumentation Design",
        "",
        "- Mode: design-only / no controller edit",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Runtime provenance instrumentation pending: {report.get('runtime_provenance_instrumentation_pending', False)}",
        "",
        "## Required Fields",
        "",
    ]
    lines.extend(f"- `{field}`" for field in report.get("required_fields", []))
    lines.extend(["", "## Hook Map", "", "| hook_id | source_function | required |", "| --- | --- | --- |"])
    for hook in report.get("hook_map", []) or []:
        if isinstance(hook, Mapping):
            lines.append(f"| {hook.get('hook_id', '')} | `{hook.get('source_function', '')}` | {hook.get('required', False)} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-provenance-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.static_provenance_json))
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json
    if not output_md.is_absolute():
        output_md = PROJECT_ROOT / output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"runtime_provenance_instrumentation_pending={report['runtime_provenance_instrumentation_pending']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
