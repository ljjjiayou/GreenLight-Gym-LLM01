"""Build and validate a protocol v1 ephemeral overlay.

This preflight exports the accepted protocol v1 files from git objects into an
audit overlay directory. It does not modify the working tree, does not execute
replay, does not fill cache, and does not call an online LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_IMPORT_MODULES = (
    "gl_gym.experiments.frozen_benchmark_protocol",
    "gl_gym.experiments.run_frozen_benchmark",
)
PACKAGE_SHIM = (
    "from pkgutil import extend_path\n"
    "__path__ = extend_path(__path__, __name__)\n"
)


class GitObjectError(RuntimeError):
    """Raised when a git object cannot be read."""


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_show_object(source_git_ref: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{source_git_ref}:{path}"],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitObjectError(detail or f"git show failed for {source_git_ref}:{path}")
    return result.stdout


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _target_path(overlay_root: Path, rel_path: str) -> Path:
    target = overlay_root / Path(rel_path)
    if not _inside(overlay_root, target):
        raise ValueError(f"overlay path escapes root: {rel_path}")
    return target


def _write_package_shims(overlay_root: Path) -> list[str]:
    shim_paths = [
        overlay_root / "gl_gym" / "__init__.py",
        overlay_root / "gl_gym" / "experiments" / "__init__.py",
    ]
    written: list[str] = []
    for path in shim_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(PACKAGE_SHIM, encoding="utf-8")
        written.append(str(path.relative_to(overlay_root)).replace("\\", "/"))
    return written


def _compile_files(paths: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            cwd=str(PROJECT_ROOT),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        rows.append(
            {
                "path": str(path),
                "compile_pass": result.returncode == 0,
                "stderr": result.stderr.strip(),
            }
        )
    return {
        "compile_pass": all(row["compile_pass"] for row in rows),
        "compile_results": rows,
    }


def _validate_imports(overlay_root: Path, modules: Sequence[str]) -> dict[str, Any]:
    code = r"""
import importlib
import json
import pathlib
import sys

overlay_root = pathlib.Path(sys.argv[1]).resolve()
modules = sys.argv[2:]
rows = []
ok = True
for module_name in modules:
    try:
        module = importlib.import_module(module_name)
        module_file = pathlib.Path(getattr(module, "__file__", "")).resolve()
        from_overlay = overlay_root in (module_file, *module_file.parents)
        rows.append({
            "module": module_name,
            "module_file": str(module_file),
            "from_overlay": from_overlay,
            "import_pass": True,
        })
        ok = ok and from_overlay
    except Exception as exc:  # pragma: no cover - exercised through subprocess
        rows.append({
            "module": module_name,
            "module_file": "",
            "from_overlay": False,
            "import_pass": False,
            "error": repr(exc),
        })
        ok = False
print(json.dumps({"import_pass": ok, "import_results": rows}, ensure_ascii=False))
raise SystemExit(0 if ok else 1)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(overlay_root), str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(overlay_root), *modules],
        cwd=str(overlay_root),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    try:
        parsed = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        parsed = {
            "import_pass": False,
            "import_results": [],
            "stdout": result.stdout.strip(),
        }
    parsed["import_pass"] = bool(parsed.get("import_pass", False)) and result.returncode == 0
    parsed["stderr"] = result.stderr.strip()
    return parsed


def build_report(
    *,
    protocol_v1_snapshot: Mapping[str, Any],
    overlay_root: str | Path,
    object_reader: Callable[[str, str], bytes] | None = None,
    validate_compile: bool = True,
    validate_import: bool = True,
    import_modules: Sequence[str] = DEFAULT_IMPORT_MODULES,
) -> dict[str, Any]:
    overlay = _resolve(overlay_root)
    reader = object_reader or _git_show_object
    source_ref = str(protocol_v1_snapshot.get("source_git_ref", "HEAD") or "HEAD")
    source_commit = str(protocol_v1_snapshot.get("source_git_commit", "") or "")
    overlay.mkdir(parents=True, exist_ok=True)
    generated_shims = _write_package_shims(overlay)

    rows: list[dict[str, Any]] = []
    exported_paths: list[Path] = []
    for item in protocol_v1_snapshot.get("files", []) or []:
        if not isinstance(item, Mapping):
            continue
        rel_path = str(item.get("path", "") or "")
        expected_sha = str(item.get("sha256", "") or "")
        row: dict[str, Any] = {
            "path": rel_path,
            "expected_sha256": expected_sha,
            "object_available_in_manifest": bool(item.get("object_available", False)),
            "exported": False,
            "hash_matches": False,
        }
        if not row["object_available_in_manifest"] or not rel_path:
            row["error"] = "manifest_object_unavailable_or_missing_path"
            rows.append(row)
            continue
        try:
            data = reader(source_ref, rel_path)
            actual_sha = _sha256(data)
            target = _target_path(overlay, rel_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            row.update(
                {
                    "exported": True,
                    "overlay_path": str(target),
                    "actual_sha256": actual_sha,
                    "hash_matches": actual_sha == expected_sha,
                    "byte_count": len(data),
                }
            )
            exported_paths.append(target)
        except Exception as exc:  # pragma: no cover - defensive for CLI/git failures
            row["error"] = str(exc)
        rows.append(row)

    py_paths = [path for path in exported_paths if path.suffix == ".py"]
    compile_report = (
        _compile_files(py_paths)
        if validate_compile
        else {"compile_pass": True, "compile_results": [], "compile_skipped": True}
    )
    import_report = (
        _validate_imports(overlay, import_modules)
        if validate_import
        else {"import_pass": True, "import_results": [], "import_skipped": True}
    )

    object_missing_count = sum(1 for row in rows if not row.get("exported", False))
    hash_mismatch_count = sum(1 for row in rows if row.get("exported") and not row.get("hash_matches"))
    all_exported = bool(rows) and object_missing_count == 0
    all_hashes_match = all_exported and hash_mismatch_count == 0
    overlay_validated = bool(all_hashes_match and compile_report["compile_pass"] and import_report["import_pass"])

    return {
        "schema_version": "protocol_v1_ephemeral_overlay_preflight_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "overlay-preflight-only / no replay",
            "reopens_rejected_preset": False,
            "canonical_case": "y2020_d120_s44_n240",
        },
        "protocol_v1_overlay_available": overlay_validated,
        "protocol_v1_overlay_validated": overlay_validated,
        "protocol_v1_overlay_root": str(overlay),
        "actual_protocol_implementation_for_execution": (
            "protocol_v1_snapshot_ephemeral_overlay" if overlay_validated else "protocol_v1_snapshot_ephemeral_overlay_required"
        ),
        "controller_agent_source_policy": "current_working_tree_controller_agent_modules",
        "source_git_ref": source_ref,
        "source_git_commit": source_commit,
        "tracked_baseline_file_count": int(protocol_v1_snapshot.get("tracked_baseline_file_count", len(rows)) or len(rows)),
        "exported_file_count": sum(1 for row in rows if row.get("exported")),
        "object_missing_count": object_missing_count,
        "hash_mismatch_count": hash_mismatch_count,
        "compile_pass": bool(compile_report["compile_pass"]),
        "import_pass": bool(import_report["import_pass"]),
        "generated_overlay_shim_files": generated_shims,
        "files": rows,
        "compile_results": compile_report.get("compile_results", []),
        "import_results": import_report.get("import_results", []),
        "import_stderr": import_report.get("stderr", ""),
        "metadata_replay_execution_allowed": False,
        "metadata_replay_allowed": False,
        "controlled_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "notes": [
            "Overlay files are exported from git objects into an audit directory and do not overwrite the working tree.",
            "The overlay carries protocol v1 evaluation files only; controller and agent modules remain current working tree inputs for future action-invariance checks.",
            "This preflight validates implementation feasibility only and does not authorize or execute metadata replay.",
        ],
        "next_action": (
            "refresh_strict_metadata_replay_execution_authorization_packet"
            if overlay_validated
            else "repair_protocol_v1_ephemeral_overlay_preflight"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Protocol v1 Ephemeral Overlay Preflight",
        "",
        "- Mode: overlay-preflight-only / no replay",
        f"- Protocol v1 overlay available: {report.get('protocol_v1_overlay_available', False)}",
        f"- Protocol v1 overlay validated: {report.get('protocol_v1_overlay_validated', False)}",
        f"- Overlay root: `{report.get('protocol_v1_overlay_root', '')}`",
        f"- Source git ref: `{report.get('source_git_ref', '')}`",
        f"- Source git commit: `{report.get('source_git_commit', '')}`",
        f"- Exported files: {report.get('exported_file_count', 0)}",
        f"- Object missing count: {report.get('object_missing_count', 0)}",
        f"- Hash mismatch count: {report.get('hash_mismatch_count', 0)}",
        f"- Compile pass: {report.get('compile_pass', False)}",
        f"- Import pass: {report.get('import_pass', False)}",
        "- Metadata replay execution allowed: false",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Exported Files",
        "",
        "| path | exported | hash matches |",
        "| --- | --- | --- |",
    ]
    for row in report.get("files", []) or []:
        lines.append(
            f"| {row.get('path', '')} | `{row.get('exported', False)}` | `{row.get('hash_matches', False)}` |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-v1-snapshot-json", required=True)
    parser.add_argument("--overlay-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--skip-compile-validation", action="store_true")
    parser.add_argument("--skip-import-validation", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        protocol_v1_snapshot=_load(args.protocol_v1_snapshot_json),
        overlay_root=args.overlay_root,
        validate_compile=not args.skip_compile_validation,
        validate_import=not args.skip_import_validation,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"protocol_v1_overlay_validated={report['protocol_v1_overlay_validated']}")
    print(f"protocol_v1_overlay_root={report['protocol_v1_overlay_root']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
