"""Inventory untracked files for mainline pollution control.

This audit is read-only. It does not stage, delete, archive, ignore, restore,
or replay anything. Its goal is to prevent shadow audit/report/test artifacts
from being mistaken for promotion evidence or default controller components.
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

from gl_gym.experiments.mainline_diff_boundary_audit import classify_path


def _normalize(path: str) -> str:
    return path.strip().replace("\\", "/")


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


def _untracked_paths_from_git() -> list[str]:
    result = _run_git(["status", "--short", "--untracked-files=all"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed")
    paths: list[str] = []
    for line in result.stdout.splitlines():
        if not line.startswith("?? "):
            continue
        paths.append(_normalize(line[3:].strip()))
    return sorted(paths)


def _recommendation(category: str, path: str) -> str:
    if category == "mainline_core":
        return "authorized_mainline_core_keep_under_user_decision"
    if category == "controller_sensitive":
        return "hold_until_default_path_and_action_invariance_evidence"
    if category == "evaluation_sensitive":
        return "hold_until_evaluation_protocol_isolation"
    if category == "codex_skill":
        return "keep_if_skill_sync_check_passes"
    if category == "test_only":
        return "keep_if_tests_do_not_redefine_promotion_gate"
    if category == "report_only":
        return "archive_or_keep_as_report_only_not_promotion_evidence"
    if category == "deprecated_or_hold":
        return "hold_as_history_or_shadow_only_not_promotion_entrypoint"
    if category == "shadow_audit":
        return "keep_as_shadow_audit_after_py_compile_and_tests"
    return "needs_manual_review_before_staging"


def _row_for_path(path: str) -> dict[str, Any]:
    category = classify_path(path)
    is_test_only = category == "test_only"
    is_report_only = category == "report_only"
    is_skill_only = category == "codex_skill"
    is_doc_only = path.startswith("docs/") and category != "mainline_core"
    may_enter_default_import_path = category == "controller_sensitive" or path.startswith("gl_gym/agent/")
    return {
        "path": path,
        "git_status": "??",
        "category": category,
        "may_enter_default_import_path": may_enter_default_import_path,
        "imports_mainline_modules": "unknown_static_check_not_run",
        "is_imported_by_mainline": "unknown_static_check_not_run" if may_enter_default_import_path else False,
        "is_test_only": is_test_only,
        "is_report_only": is_report_only,
        "is_doc_only": is_doc_only,
        "is_skill_only": is_skill_only,
        "can_be_promotion_evidence": False,
        "needs_gitignore_decision": is_report_only,
        "needs_staging_decision": category
        in {"mainline_core", "controller_sensitive", "evaluation_sensitive", "shadow_audit", "test_only", "codex_skill"},
        "needs_archive_decision": category in {"report_only", "deprecated_or_hold"},
        "needs_user_decision": category
        in {"mainline_core", "controller_sensitive", "evaluation_sensitive", "deprecated_or_hold", "report_only"},
        "allowed_next_action": _recommendation(category, path),
    }


def build_report(paths: Sequence[str] | None = None) -> dict[str, Any]:
    untracked = [_normalize(path) for path in (paths if paths is not None else _untracked_paths_from_git())]
    rows = [_row_for_path(path) for path in sorted(untracked)]
    counts = Counter(str(row["category"]) for row in rows)
    needs_user_decision = [str(row["path"]) for row in rows if bool(row.get("needs_user_decision"))]
    default_path_risk = [str(row["path"]) for row in rows if bool(row.get("may_enter_default_import_path"))]
    return {
        "schema_version": "untracked_file_inventory_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / untracked inventory",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "metadata_replay_allowed": False,
        "performance_claim_allowed": False,
        "untracked_count": len(rows),
        "category_counts": dict(sorted(counts.items())),
        "default_path_risk_paths": default_path_risk,
        "needs_user_decision_paths": needs_user_decision,
        "rows": rows,
        "notes": [
            "This inventory is read-only and does not stage, delete, archive, ignore, restore, or replay files.",
            "Untracked shadow audit/test/report/skill files are development artifacts, not promotion evidence.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Untracked File Inventory",
        "",
        "- Mode: audit-only / untracked inventory",
        "- Controlled replay allowed: false",
        "- Metadata replay allowed: false",
        "- Performance claim allowed: false",
        f"- Untracked count: {report.get('untracked_count', 0)}",
        "",
        "## Category Counts",
        "",
        "| category | count |",
        "| --- | ---: |",
    ]
    for category, count in dict(report.get("category_counts", {})).items():
        lines.append(f"| {category} | {count} |")
    lines.extend(["", "## Default Path Risk Paths", ""])
    for path in report.get("default_path_risk_paths", []) or []:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| path | category | default_path_risk | promotion_evidence | allowed_next_action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.get('path', '')}`",
                    str(row.get("category", "")),
                    str(row.get("may_enter_default_import_path", False)),
                    str(row.get("can_be_promotion_evidence", False)),
                    str(row.get("allowed_next_action", "")),
                ]
            )
            + " |"
        )
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
    print(f"untracked_count={report['untracked_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
