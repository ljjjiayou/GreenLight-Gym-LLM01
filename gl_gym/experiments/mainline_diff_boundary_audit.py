"""Read-only audit for mainline-sensitive working tree diffs.

The audit does not revert, stash, or edit files.  It only records whether
current working-tree changes touch files that the greenhouse mainline treats as
controller, evaluation, or explicitly blocked scope.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAINLINE_BLOCKING = {
    "gl_gym/agent/expert_distillation.py",
}
MAINLINE_CORE = {
    "docs/project_mainline_guidance_20260519.md",
}
CONTROLLER_SENSITIVE = {
    "gl_gym/agent/llm_agent.py",
    "gl_gym/agent/intent_contract.py",
    "gl_gym/agent/profile_generator.py",
}
EVALUATION_SENSITIVE = {
    "gl_gym/experiments/run_frozen_benchmark.py",
    "gl_gym/experiments/frozen_benchmark_protocol.py",
    "tests/test_frozen_benchmark.py",
    "tests/test_frozen_benchmark_protocol.py",
}
SHADOW_AUDIT_MARKERS = (
    "audit",
    "status",
    "readiness",
    "closure",
    "packet",
    "plan",
    "manifest",
    "inventory",
    "calibration",
)
DEPRECATED_OR_HOLD_MARKERS = ("controlled", "replay", "proposer", "policy", "override")
SKILL_CHECK_TOOLING = {
    "scripts/check_greenhouse_skills.py",
    "tests/test_check_greenhouse_skills.py",
}


def _normalize(path: str) -> str:
    return path.strip().replace("\\", "/")


def classify_path(path: str) -> str:
    normalized = _normalize(path)
    if normalized in MAINLINE_BLOCKING:
        return "mainline_blocking"
    if normalized in MAINLINE_CORE:
        return "mainline_core"
    if normalized in EVALUATION_SENSITIVE:
        return "evaluation_sensitive"
    if normalized in CONTROLLER_SENSITIVE:
        return "controller_sensitive"
    if normalized.startswith("codex_skills/") or normalized in SKILL_CHECK_TOOLING:
        return "codex_skill"
    if normalized.startswith("tests/"):
        return "test_only"
    if normalized.startswith("gl_gym/result/") or normalized.startswith("reports/"):
        return "report_only"
    if normalized.startswith("docs/"):
        return "report_only"
    if normalized.startswith("gl_gym/configs/stress/"):
        return "shadow_audit"
    if normalized.startswith("scripts/"):
        return "shadow_audit"
    if normalized.startswith("gl_gym/experiments/"):
        name = Path(normalized).name.lower()
        if any(marker in name for marker in SHADOW_AUDIT_MARKERS):
            return "shadow_audit"
        if any(marker in name for marker in DEPRECATED_OR_HOLD_MARKERS):
            return "deprecated_or_hold"
        return "shadow_audit"
    return "other"


def _run_git(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _changed_paths_from_git() -> list[dict[str, Any]]:
    status = _run_git(["status", "--short", "--untracked-files=all"])
    if status.returncode != 0:
        raise RuntimeError(status.stderr.strip() or "git status failed")
    entries: list[dict[str, Any]] = []
    for line in status.stdout.splitlines():
        if not line.strip():
            continue
        status_code = line[:2]
        path_text = line[3:].strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1].strip()
        path = _normalize(path_text)
        entries.append(
            {
                "path": path,
                "git_status": status_code,
                "category": classify_path(path),
            }
        )
    return entries


def _numstat() -> dict[str, dict[str, Any]]:
    result = _run_git(["diff", "--numstat"])
    stats: dict[str, dict[str, Any]] = {}
    if result.returncode != 0:
        return stats
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added, deleted, path_text = parts[0], parts[1], _normalize(parts[2])
        stats[path_text] = {
            "added": None if added == "-" else int(added),
            "deleted": None if deleted == "-" else int(deleted),
        }
    return stats


def _diff_excerpt(path: str, *, max_lines: int = 60) -> list[str]:
    result = _run_git(["diff", "--", path])
    if result.returncode != 0:
        return [result.stderr.strip()]
    lines = result.stdout.splitlines()
    return lines[:max_lines]


def build_report(paths: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    entries = [dict(item) for item in (paths if paths is not None else _changed_paths_from_git())]
    stats = _numstat() if paths is None else {}
    counts: Counter[str] = Counter()
    for entry in entries:
        path = _normalize(str(entry.get("path", "")))
        category = str(entry.get("category") or classify_path(path))
        entry["path"] = path
        entry["category"] = category
        if path in stats:
            entry["numstat"] = stats[path]
        if category == "mainline_blocking":
            entry["mainline_note"] = "requires_user_cleanup_or_explicit_authorization"
            entry["diff_excerpt"] = _diff_excerpt(path)
        elif category == "mainline_core":
            entry["mainline_note"] = "requires_user_authorization_before_promotion_evidence"
        elif category == "evaluation_sensitive":
            entry["mainline_note"] = "requires_protocol_isolation_or_confirmation"
        elif category == "controller_sensitive":
            entry["mainline_note"] = "requires_default_controller_invariance_evidence"
        elif category == "deprecated_or_hold":
            entry["mainline_note"] = "hold_as_history_or_shadow_only_not_promotion_entrypoint"
        counts[category] += 1

    mainline_blocking = [item for item in entries if item.get("category") == "mainline_blocking"]
    mainline_core = [item for item in entries if item.get("category") == "mainline_core"]
    evaluation_sensitive = [item for item in entries if item.get("category") == "evaluation_sensitive"]
    controller_sensitive = [item for item in entries if item.get("category") == "controller_sensitive"]
    controlled_replay_blocked = bool(mainline_blocking or mainline_core or evaluation_sensitive or controller_sensitive)
    return {
        "schema_version": "mainline_diff_boundary_audit_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary"],
            "default_llm_rspc_v2_changed": False,
            "mode": "read-only boundary audit",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_blocked": controlled_replay_blocked,
        "controlled_replay_block_reasons": [
            reason
            for reason, active in (
                ("mainline_blocking_diff_present", bool(mainline_blocking)),
                ("mainline_core_diff_present", bool(mainline_core)),
                ("evaluation_sensitive_diff_present", bool(evaluation_sensitive)),
                ("controller_sensitive_diff_present", bool(controller_sensitive)),
            )
            if active
        ],
        "category_counts": dict(sorted(counts.items())),
        "expert_distillation_status": (
            "unrelated_diff_present" if any(item["path"] == "gl_gym/agent/expert_distillation.py" for item in entries) else "clean"
        ),
        "evaluation_diff_status": "requires_isolation_or_confirmation" if evaluation_sensitive else "clean",
        "controller_diff_status": "requires_invariance_evidence" if controller_sensitive else "clean",
        "changed_files": entries,
        "notes": [
            "This audit is read-only and does not revert or stash user changes.",
            "Any cleanup of mainline-blocking files requires explicit user approval.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Mainline Diff Boundary Audit",
        "",
        "## Mainline Alignment",
        "",
        "- Mode: read-only boundary audit",
        "- Default `llm_rspc_v2` changed by this audit: false",
        "- Reopens rejected preset: false",
        "",
        "## Decision",
        "",
        f"- Controlled replay blocked: {report.get('controlled_replay_blocked', False)}",
        f"- Block reasons: {json.dumps(report.get('controlled_replay_block_reasons', []), ensure_ascii=False)}",
        f"- expert_distillation.py status: `{report.get('expert_distillation_status', '')}`",
        f"- evaluation diff status: `{report.get('evaluation_diff_status', '')}`",
        f"- controller diff status: `{report.get('controller_diff_status', '')}`",
        "",
        "## Category Counts",
        "",
        "| category | count |",
        "| --- | ---: |",
    ]
    for category, count in dict(report.get("category_counts", {})).items():
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Changed Files", "", "| status | path | category | note |", "| --- | --- | --- | --- |"])
    for item in report.get("changed_files", []) or []:
        if not isinstance(item, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("git_status", "")),
                    f"`{item.get('path', '')}`",
                    str(item.get("category", "")),
                    str(item.get("mainline_note", "")),
                ]
            )
            + " |"
        )
    excerpts = [
        item
        for item in report.get("changed_files", []) or []
        if isinstance(item, Mapping) and item.get("diff_excerpt")
    ]
    if excerpts:
        lines.extend(["", "## Mainline Blocking Diff Excerpts", ""])
        for item in excerpts:
            lines.append(f"### `{item.get('path', '')}`")
            lines.append("")
            lines.append("```diff")
            lines.extend(str(line) for line in item.get("diff_excerpt", []))
            lines.append("```")
            lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default="gl_gym/result/audits/mainline_diff_boundary_audit_20260520.json")
    parser.add_argument("--output-md", default="gl_gym/result/audits/mainline_diff_boundary_audit_20260520.md")
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
    print(f"controlled_replay_blocked={report['controlled_replay_blocked']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
