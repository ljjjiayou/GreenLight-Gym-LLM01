"""Find a qualified baseline trace for canonical action-diff audit.

This precheck only inventories historical traces. It does not execute replay,
fill cache, call an online LLM, or claim action invariance.
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

REJECTED_PRESET_PARTS = {"hot_dry_relief", "conservative_dry"}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _count_jsonl_rows(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


def _trace_identity(path: Path, *, expected_scenario: str) -> dict[str, str]:
    suffix = ".jsonl"
    name = path.name
    if not name.endswith(suffix):
        return {"scenario_id": "", "controller": ""}
    stem = name[: -len(suffix)]
    prefix = f"{expected_scenario}_"
    if not stem.startswith(prefix):
        return {"scenario_id": "", "controller": ""}
    return {
        "scenario_id": expected_scenario,
        "controller": stem[len(prefix) :],
    }


def _rejected_preset_parts(path: Path) -> list[str]:
    return sorted({part for part in path.parts if part in REJECTED_PRESET_PARTS})


def _candidate_rows(
    *,
    roots: Sequence[str],
    scenario_id: str,
    controller: str,
    max_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root_text in roots:
        root = _resolve(root_text)
        if not root.exists():
            rows.append(
                {
                    "candidate_root": str(root),
                    "jsonl_path": None,
                    "csv_path": None,
                    "qualified": False,
                    "reject_reason": "candidate_root_missing",
                    "row_count": None,
                    "controller": "",
                    "scenario_id": "",
                    "rejected_preset_parts": [],
                }
            )
            continue
        matches = [root] if root.is_file() else sorted(root.rglob(f"{scenario_id}_{controller}.jsonl"))
        for jsonl_path in matches:
            if jsonl_path in seen or not jsonl_path.is_file():
                continue
            seen.add(jsonl_path)
            ident = _trace_identity(jsonl_path, expected_scenario=scenario_id)
            csv_path = jsonl_path.with_suffix(".csv")
            row_count = _count_jsonl_rows(jsonl_path)
            rejected_parts = _rejected_preset_parts(jsonl_path)
            reject_reason = ""
            if ident["scenario_id"] != scenario_id:
                reject_reason = "scenario_mismatch"
            elif ident["controller"] != controller:
                reject_reason = "controller_mismatch"
            elif row_count != max_steps:
                reject_reason = "max_steps_or_row_count_mismatch"
            elif not csv_path.exists():
                reject_reason = "matching_csv_trace_missing"
            elif rejected_parts:
                reject_reason = "rejected_preset_trace_path"
            qualified = not reject_reason
            rows.append(
                {
                    "candidate_root": str(root),
                    "jsonl_path": str(jsonl_path),
                    "csv_path": str(csv_path) if csv_path.exists() else None,
                    "trace_dir": str(jsonl_path.parent),
                    "qualified": qualified,
                    "reject_reason": reject_reason,
                    "row_count": row_count,
                    "controller": ident["controller"],
                    "scenario_id": ident["scenario_id"],
                    "rejected_preset_parts": rejected_parts,
                }
            )
    return rows


def _selection_key(row: Mapping[str, Any], preferred_dirs: Sequence[str]) -> tuple[int, int, str]:
    trace_dir = str(row.get("trace_dir", "") or "")
    preferred = 1
    for index, preferred_dir in enumerate(preferred_dirs):
        resolved = str(_resolve(preferred_dir))
        if trace_dir == resolved or trace_dir.startswith(resolved):
            preferred = 0
            return (preferred, index, trace_dir)
    return (preferred, len(preferred_dirs), trace_dir)


def build_report(
    *,
    candidate_trace_roots: Sequence[str],
    preferred_trace_dirs: Sequence[str] | None = None,
    scenario_id: str = "y2020_d120_s44_n240",
    controller: str = "llm_rspc_v2",
    max_steps: int = 240,
) -> dict[str, Any]:
    preferred_dirs = list(preferred_trace_dirs or [])
    roots = list(candidate_trace_roots or ["gl_gym/result/benchmarks"])
    candidates = _candidate_rows(
        roots=roots,
        scenario_id=scenario_id,
        controller=controller,
        max_steps=max_steps,
    )
    qualified = [row for row in candidates if row.get("qualified")]
    selected = min(qualified, key=lambda row: _selection_key(row, preferred_dirs)) if qualified else None
    baseline_trace_qualified = selected is not None
    action_diff_status = (
        "ready_with_qualified_baseline_trace"
        if baseline_trace_qualified
        else "blocked_missing_baseline_trace"
    )
    return {
        "schema_version": "canonical_action_diff_baseline_trace_manifest_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "baseline-trace-precheck-only / no replay",
            "reopens_rejected_preset": False,
            "canonical_case": scenario_id,
        },
        "baseline_trace_manifest_ready": True,
        "baseline_trace_found": bool(candidates),
        "baseline_trace_qualified": baseline_trace_qualified,
        "action_diff_audit_allowed": baseline_trace_qualified,
        "action_diff_audit_status": action_diff_status,
        "metadata_replay_allowed": False,
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "scenario_id": scenario_id,
        "controller": controller,
        "max_steps": int(max_steps),
        "candidate_trace_roots": roots,
        "preferred_trace_dirs": preferred_dirs,
        "candidate_count": len(candidates),
        "qualified_candidate_count": len(qualified),
        "selected_baseline_trace_dir": selected.get("trace_dir") if selected else None,
        "selected_baseline_jsonl": selected.get("jsonl_path") if selected else None,
        "selected_baseline_csv": selected.get("csv_path") if selected else None,
        "selected_baseline_row_count": selected.get("row_count") if selected else None,
        "historical_context_only": True,
        "not_current_working_tree_evidence": True,
        "not_action_invariance_evidence_until_compared": True,
        "candidates": candidates,
        "next_action": (
            "wire_baseline_trace_into_canonical_execution_request"
            if baseline_trace_qualified
            else "canonical_request_can_continue_but_action_diff_audit_blocked_missing_baseline_trace"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Canonical Action-Diff Baseline Trace Manifest",
        "",
        "- Mode: baseline-trace-precheck-only / no replay",
        f"- Scenario: `{report.get('scenario_id', '')}`",
        f"- Controller: `{report.get('controller', '')}`",
        f"- Max steps: `{report.get('max_steps', '')}`",
        f"- Baseline trace qualified: {report.get('baseline_trace_qualified', False)}",
        f"- Action-diff audit status: `{report.get('action_diff_audit_status', '')}`",
        f"- Selected baseline trace dir: `{report.get('selected_baseline_trace_dir', None)}`",
        "",
        "## Candidate Summary",
        "",
        f"- Candidate count: {report.get('candidate_count', 0)}",
        f"- Qualified candidate count: {report.get('qualified_candidate_count', 0)}",
        "",
        "| qualified | row_count | trace_dir | reject_reason |",
        "| --- | ---: | --- | --- |",
    ]
    for row in report.get("candidates", []) or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("qualified", False)),
                    str(row.get("row_count", "")),
                    f"`{row.get('trace_dir', row.get('candidate_root', ''))}`",
                    str(row.get("reject_reason", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-trace-root", action="append", default=[])
    parser.add_argument("--preferred-trace-dir", action="append", default=[])
    parser.add_argument("--scenario-id", default="y2020_d120_s44_n240")
    parser.add_argument("--controller", default="llm_rspc_v2")
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        candidate_trace_roots=args.candidate_trace_root or ["gl_gym/result/benchmarks"],
        preferred_trace_dirs=args.preferred_trace_dir,
        scenario_id=args.scenario_id,
        controller=args.controller,
        max_steps=args.max_steps,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"baseline_trace_qualified={report['baseline_trace_qualified']}")
    print(f"action_diff_audit_status={report['action_diff_audit_status']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
