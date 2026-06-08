"""Read-only status check for expert_distillation.py mainline blocker."""

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

EXPERT_PATH = "gl_gym/agent/expert_distillation.py"


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


def build_report() -> dict[str, Any]:
    status = _run_git(["status", "--short", "--", EXPERT_PATH])
    diff = _run_git(["diff", "--", EXPERT_PATH])
    git_status = status.stdout.strip()
    diff_present = bool(diff.stdout.strip())
    clean = not git_status and not diff_present
    return {
        "schema_version": "expert_distillation_status_v1",
        "mainline_alignment": {
            "affected_layers": ["Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "audit-only / status confirmation",
            "reopens_rejected_preset": False,
        },
        "path": EXPERT_PATH,
        "git_status": git_status or "clean",
        "whether_modified": not clean,
        "whether_user_authorized": False,
        "whether_part_of_current_evidence": False,
        "hard_blocker": not clean,
        "metadata_replay_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "decision": "clean_not_current_blocker" if clean else "hard_blocker_restore_or_authorization_required",
        "notes": [
            "The project mainline forbids touching expert_distillation.py unless explicitly requested.",
            "A clean status record does not unlock metadata replay by itself.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# expert_distillation.py Status",
            "",
            f"- Path: `{report.get('path', '')}`",
            f"- Git status: `{report.get('git_status', '')}`",
            f"- Modified: {report.get('whether_modified', False)}",
            f"- Hard blocker: {report.get('hard_blocker', False)}",
            f"- Decision: `{report.get('decision', '')}`",
            "- Metadata replay allowed: false",
            "- Controlled replay allowed: false",
            "- Performance claim allowed: false",
            "",
        ]
    )


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
    print(f"decision={report['decision']}")
    print(f"hard_blocker={report['hard_blocker']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
