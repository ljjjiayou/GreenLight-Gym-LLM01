"""Validate greenhouse project skills and repo/local synchronization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_SKILLS = PROJECT_ROOT / "codex_skills"
DEFAULT_LOCAL_SKILLS = Path.home() / ".codex" / "skills"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _canonical_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _parse_frontmatter(text: str) -> dict[str, str]:
    text = _canonical_text(text)
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---", 4)
    if end < 0:
        return {}
    raw = text[4:end].strip().splitlines()
    data: dict[str, str] = {}
    for line in raw:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _skill_dirs(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    return {
        path.name: path
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name.startswith("greenhouse-") and (path / "SKILL.md").exists()
    }


def _validate_skill(name: str, path: Path) -> dict[str, Any]:
    skill_path = path / "SKILL.md"
    exists = skill_path.exists()
    text = _read(skill_path) if exists else ""
    frontmatter = _parse_frontmatter(text) if exists else {}
    errors: list[str] = []
    if not exists:
        errors.append("missing_skill_md")
    if frontmatter.get("name") != name:
        errors.append("frontmatter_name_mismatch")
    if not frontmatter.get("description"):
        errors.append("missing_description")
    if any(ch for ch in name if not (ch.islower() or ch.isdigit() or ch == "-")):
        errors.append("invalid_skill_folder_name")
    return {
        "name": name,
        "path": str(skill_path),
        "exists": exists,
        "frontmatter": frontmatter,
        "line_count": len(text.splitlines()) if exists else 0,
        "errors": errors,
        "ok": not errors,
    }


def build_report(repo_root: Path = DEFAULT_REPO_SKILLS, local_root: Path = DEFAULT_LOCAL_SKILLS) -> dict[str, Any]:
    repo_skills = _skill_dirs(repo_root)
    local_skills = _skill_dirs(local_root)
    names = sorted(set(repo_skills) | set(local_skills))
    rows: list[dict[str, Any]] = []
    out_of_sync: list[str] = []
    missing_local: list[str] = []
    missing_repo: list[str] = []
    for name in names:
        repo_path = repo_skills.get(name)
        local_path = local_skills.get(name)
        repo_validation = _validate_skill(name, repo_path) if repo_path else {"ok": False, "errors": ["missing_repo_skill"]}
        local_validation = _validate_skill(name, local_path) if local_path else {"ok": False, "errors": ["missing_local_skill"]}
        repo_text = _canonical_text(_read(repo_path / "SKILL.md")) if repo_path else ""
        local_text = _canonical_text(_read(local_path / "SKILL.md")) if local_path else ""
        synced = bool(repo_path and local_path and repo_text == local_text)
        if repo_path and not local_path:
            missing_local.append(name)
        if local_path and not repo_path:
            missing_repo.append(name)
        if repo_path and local_path and not synced:
            out_of_sync.append(name)
        rows.append(
            {
                "name": name,
                "repo": repo_validation,
                "local": local_validation,
                "repo_local_synced": synced,
            }
        )
    ok = (
        not out_of_sync
        and not missing_local
        and not missing_repo
        and all(bool(row["repo"].get("ok", False)) for row in rows if "repo" in row)
        and all(bool(row["local"].get("ok", False)) for row in rows if "local" in row)
    )
    return {
        "schema_version": "greenhouse_skills_check_v1",
        "repo_root": str(repo_root),
        "local_root": str(local_root),
        "skill_count": len(names),
        "ok": ok,
        "out_of_sync": out_of_sync,
        "missing_local": missing_local,
        "missing_repo": missing_repo,
        "rows": rows,
        "notes": [
            "codex_skills is the authoritative project copy.",
            "local .codex skills must match the repo copy for predictable triggering.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Greenhouse Skills Check",
        "",
        f"- OK: {report.get('ok', False)}",
        f"- Skill count: {report.get('skill_count', 0)}",
        "",
        "## Drift",
        "",
        f"- Out of sync: {', '.join(report.get('out_of_sync', []) or []) or 'none'}",
        f"- Missing local: {', '.join(report.get('missing_local', []) or []) or 'none'}",
        f"- Missing repo: {', '.join(report.get('missing_repo', []) or []) or 'none'}",
        "",
        "## Skills",
        "",
        "| skill | repo ok | local ok | synced |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.get("rows", []) or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"| {row.get('name', '')} | {row.get('repo', {}).get('ok', False)} | "
            f"{row.get('local', {}).get('ok', False)} | {row.get('repo_local_synced', False)} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_SKILLS))
    parser.add_argument("--local-root", default=str(DEFAULT_LOCAL_SKILLS))
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(Path(args.repo_root), Path(args.local_root))
    if args.output_json:
        output_json = Path(args.output_json)
        if not output_json.is_absolute():
            output_json = PROJECT_ROOT / output_json
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {output_json}")
    if args.output_md:
        output_md = Path(args.output_md)
        if not output_md.is_absolute():
            output_md = PROJECT_ROOT / output_md
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_markdown_report(report), encoding="utf-8")
        print(f"wrote {output_md}")
    print(f"ok={report['ok']} skill_count={report['skill_count']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
