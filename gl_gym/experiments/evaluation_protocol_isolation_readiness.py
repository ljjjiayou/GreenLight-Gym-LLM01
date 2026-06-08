"""Design-only readiness report for frozen benchmark protocol isolation.

This audit is deliberately read-only. It records how modified evaluation
protocol files must be isolated before any replay result can be treated as
promotion evidence. It does not run replay, fill cache, or call an online LLM.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EVALUATION_SENSITIVE_PATHS = [
    "gl_gym/experiments/frozen_benchmark_protocol.py",
    "gl_gym/experiments/run_frozen_benchmark.py",
    "tests/test_frozen_benchmark.py",
    "tests/test_frozen_benchmark_protocol.py",
]


def _normalize(path: str) -> str:
    return path.strip().replace("\\", "/")


def _git_status_map() -> dict[str, str]:
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return {}
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        path = _normalize(line[3:].strip())
        if " -> " in path:
            path = _normalize(path.split(" -> ", 1)[1])
        out[path] = status
    return out


def build_report(status_map: Mapping[str, str] | None = None) -> dict[str, Any]:
    statuses = dict(_git_status_map() if status_map is None else status_map)
    rows = []
    for path in EVALUATION_SENSITIVE_PATHS:
        git_status = statuses.get(path, "clean_or_untracked_not_reported")
        modified = git_status.strip() != "" and git_status != "clean_or_untracked_not_reported"
        rows.append(
            {
                "path": path,
                "git_status": git_status,
                "evaluation_protocol_impact": True,
                "protocol_isolation_status": "pending" if modified else "not_needed_for_clean_path",
                "can_support_promotion_evidence": False if modified else "not_by_itself",
                "required_evidence": [
                    "compare old and new protocol with the same controller, cache, scenario list, and max steps",
                    "summary rows, cache hit metrics, runtime metrics, gate failures, and action-diff semantics must match or be fully explained",
                ],
            }
        )
    pending_paths = [row["path"] for row in rows if row["protocol_isolation_status"] == "pending"]
    return {
        "schema_version": "evaluation_protocol_isolation_readiness_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / protocol isolation design",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "controlled_replay_allowed": False,
        "metadata_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "protocol_isolation_pass": False,
        "promotion_evidence_allowed": False,
        "pending_paths": pending_paths,
        "comparison_design": {
            "controller": "same llm_rspc_v2 controller label and same controller configuration",
            "cache": "same frozen merged cache; no cache fill",
            "scenario_list": "same scenario ids, presets, seeds, timestep windows, and max steps",
            "old_protocol": "last accepted or HEAD-equivalent protocol snapshot",
            "new_protocol": "current working-tree protocol snapshot",
        },
        "fields_must_match": [
            "summary row count",
            "scenario ids and preset labels",
            "cache hit / miss metrics",
            "runtime error metrics",
            "strict miss metrics",
            "gate failure count and reasons",
            "action-diff field definitions and values",
        ],
        "allowed_differences": [
            "report-only path names or timestamps",
            "new metadata columns only when controller actions and gate semantics are unchanged",
        ],
        "if_mismatch": "block promotion evidence until every protocol delta is explained, versioned, and re-audited",
        "next_action": "evaluation_protocol_isolation_required" if pending_paths else "controller_invariance_metadata_replay_required",
        "rows": rows,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Evaluation Protocol Isolation Readiness",
        "",
        "- Mode: audit-only / protocol isolation design",
        "- Replay run: false",
        "- Cache fill run: false",
        "- Controlled replay allowed: false",
        "- Metadata replay allowed: false",
        "- Performance claim allowed: false",
        f"- Protocol isolation pass: {report.get('protocol_isolation_pass', False)}",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Pending Paths",
        "",
    ]
    for path in report.get("pending_paths", []) or []:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Fields That Must Match", ""])
    for field in report.get("fields_must_match", []) or []:
        lines.append(f"- {field}")
    lines.extend(["", "## Rows", "", "| path | git_status | isolation_status |", "| --- | --- | --- |"])
    for row in report.get("rows", []) or []:
        if isinstance(row, Mapping):
            lines.append(f"| `{row.get('path', '')}` | {row.get('git_status', '')} | {row.get('protocol_isolation_status', '')} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report()
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
    print(f"protocol_isolation_pass={report['protocol_isolation_pass']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
