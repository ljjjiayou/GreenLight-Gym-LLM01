"""Plan-only action invariance requirements for default-path shadow modules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def build_report(default_path_audit: Mapping[str, Any]) -> dict[str, Any]:
    calls = list(default_path_audit.get("calls", []) or [])
    target_paths = [row.get("path", "") for row in default_path_audit.get("target_paths", []) or [] if isinstance(row, Mapping)]
    return {
        "schema_version": "default_path_action_invariance_plan_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "plan-only / no replay",
            "reopens_rejected_preset": False,
        },
        "metadata_replay_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "plan_ready": bool(calls),
        "default_path_action_invariance_proven": False,
        "target_modules": target_paths,
        "observed_default_path_call_count": len(calls),
        "observed_calls": calls,
        "required_evidence": [
            "action_diff_steps = 0",
            "max_abs_delta = 0",
            "selected_control sequence unchanged",
            "post_guardrail final action unchanged",
            "current_plan changes limited to auditable metadata or fully explained",
            "trace schema complete and backwards-compatible",
        ],
        "scenario_families": [
            "canonical failure: y2020_d120_s44_n240",
            "neutral no-risk",
            "pure hot-dry",
            "mixed dry-dew",
        ],
        "next_action": "evaluation_protocol_isolation_required_before_metadata_replay",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Default-Path Action Invariance Plan",
        "",
        "- Mode: plan-only / no replay",
        "- Metadata replay allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Plan ready: {report.get('plan_ready', False)}",
        f"- Action invariance proven: {report.get('default_path_action_invariance_proven', False)}",
        "",
        "## Required Evidence",
        "",
    ]
    for item in report.get("required_evidence", []) or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Scenario Families", ""])
    for item in report.get("scenario_families", []) or []:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-path-audit-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(_load(args.default_path_audit_json))
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
    print(f"plan_ready={report['plan_ready']}")
    print(f"action_invariance_proven={report['default_path_action_invariance_proven']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
