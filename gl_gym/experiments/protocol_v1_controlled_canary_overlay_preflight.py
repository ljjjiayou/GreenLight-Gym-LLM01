"""Build a protocol v1 overlay with a minimal controlled-canary runner shim.

The base protocol files are copied from the validated protocol v1 overlay. The
only intended behavioral delta is the runner control surface needed to admit the
strict hot-dry proposer canary. This script does not run replay, fill cache, or
call an online LLM.
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OVERLAY_ROOT = PROJECT_ROOT / "gl_gym" / "result" / "audits" / "protocol_v1_controlled_canary_overlay_20260525"
DEFAULT_CANDIDATE_CONTROLLER = "llm_rspc_v2_hot_dry_proposer_strict"
ALLOWED_RUNNER_DELTA_MARKERS = (
    "llm_rspc_v2_hot_dry_proposer",
    "llm_rspc_v2_hot_dry_proposer_strict",
    "parse_agent_config_overrides",
    "agent_config_overrides",
    "rspc_hot_dry_proposer_control_enabled",
    "rspc_hot_dry_proposer_control_strict_enabled",
    "fields",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _safe_rmtree(path: Path) -> None:
    audits_root = PROJECT_ROOT / "gl_gym" / "result" / "audits"
    resolved = path.resolve()
    if not _inside(audits_root, resolved) or not resolved.name.startswith("protocol_v1_controlled_canary_overlay_"):
        raise ValueError(f"refusing to remove non-controlled-canary overlay path: {path}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _copy_overlay(source_root: Path, overlay_root: Path) -> None:
    _safe_rmtree(overlay_root)
    shutil.copytree(source_root, overlay_root)


def _relative_py_files(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*.py"))


def _copy_current_runner(overlay_root: Path, current_runner: Path) -> Path:
    runner = overlay_root / "gl_gym" / "experiments" / "run_frozen_benchmark.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current_runner, runner)
    return runner


def _runner_supports(path: Path, controller: str) -> bool:
    try:
        return controller in path.read_text(encoding="utf-8")
    except Exception:
        return False


def _runner_delta_valid(*, old_runner: Path, new_runner: Path, current_runner: Path, candidate_controller: str) -> dict[str, Any]:
    old_text = old_runner.read_text(encoding="utf-8")
    new_text = new_runner.read_text(encoding="utf-8")
    current_text = current_runner.read_text(encoding="utf-8")
    added_markers = [marker for marker in ALLOWED_RUNNER_DELTA_MARKERS if marker in new_text and marker not in old_text]
    removed_disallowed = candidate_controller not in old_text and candidate_controller in new_text
    new_matches_current_runner = new_text == current_text
    return {
        "runner_delta_classification": "controlled_canary_runner_surface_only",
        "source_runner_hash": _sha256(old_runner),
        "controlled_overlay_runner_hash": _sha256(new_runner),
        "current_working_tree_runner_hash": _sha256(current_runner),
        "new_runner_matches_current_working_tree_runner": new_matches_current_runner,
        "old_runner_supports_candidate_controller": _runner_supports(old_runner, candidate_controller),
        "new_runner_supports_candidate_controller": _runner_supports(new_runner, candidate_controller),
        "current_runner_supports_candidate_controller": _runner_supports(current_runner, candidate_controller),
        "allowed_delta_markers_present": added_markers,
        "runner_delta_valid": bool(
            removed_disallowed
            and _runner_supports(new_runner, candidate_controller)
            and new_matches_current_runner
            and all(marker in new_text for marker in ALLOWED_RUNNER_DELTA_MARKERS)
        ),
    }


def _protocol_hash_rows(*, source_root: Path, overlay_root: Path) -> list[dict[str, Any]]:
    runner_rel = "gl_gym/experiments/run_frozen_benchmark.py"
    rows: list[dict[str, Any]] = []
    rel_paths = sorted(set(_relative_py_files(source_root)) | set(_relative_py_files(overlay_root)))
    for rel in rel_paths:
        source = source_root / rel
        target = overlay_root / rel
        if rel == runner_rel:
            continue
        source_exists = source.exists()
        target_exists = target.exists()
        source_hash = _sha256(source) if source_exists else ""
        target_hash = _sha256(target) if target_exists else ""
        rows.append(
            {
                "path": rel,
                "source_exists": source_exists,
                "overlay_exists": target_exists,
                "source_sha256": source_hash,
                "overlay_sha256": target_hash,
                "hash_matches": source_exists and target_exists and source_hash == target_hash,
            }
        )
    return rows


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
        rows.append({"path": str(path), "compile_pass": result.returncode == 0, "stderr": result.stderr.strip()})
    return {"compile_pass": all(row["compile_pass"] for row in rows), "compile_results": rows}


def _validate_imports(overlay_root: Path) -> dict[str, Any]:
    code = r"""
import importlib
import json
import pathlib
import sys

overlay_root = pathlib.Path(sys.argv[1]).resolve()
modules = [
    "gl_gym.experiments.frozen_benchmark_protocol",
    "gl_gym.experiments.run_frozen_benchmark",
]
rows = []
ok = True
for module_name in modules:
    try:
        module = importlib.import_module(module_name)
        module_file = pathlib.Path(getattr(module, "__file__", "")).resolve()
        from_overlay = overlay_root in (module_file, *module_file.parents)
        rows.append({"module": module_name, "module_file": str(module_file), "from_overlay": from_overlay, "import_pass": True})
        ok = ok and from_overlay
    except Exception as exc:
        rows.append({"module": module_name, "module_file": "", "from_overlay": False, "import_pass": False, "error": repr(exc)})
        ok = False
print(json.dumps({"import_pass": ok, "import_results": rows}, ensure_ascii=False))
raise SystemExit(0 if ok else 1)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(overlay_root), str(PROJECT_ROOT), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, "-c", code, str(overlay_root)],
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
        parsed = {"import_pass": False, "import_results": [], "stdout": result.stdout.strip()}
    parsed["import_pass"] = bool(parsed.get("import_pass", False)) and result.returncode == 0
    parsed["stderr"] = result.stderr.strip()
    return parsed


def build_report(
    *,
    protocol_v1_overlay_preflight: Mapping[str, Any],
    overlay_root: str | Path = DEFAULT_OVERLAY_ROOT,
    current_runner_path: str | Path = PROJECT_ROOT / "gl_gym" / "experiments" / "run_frozen_benchmark.py",
    candidate_controller: str = DEFAULT_CANDIDATE_CONTROLLER,
    validate_compile: bool = True,
    validate_import: bool = True,
) -> dict[str, Any]:
    source_root = _resolve(str(protocol_v1_overlay_preflight.get("protocol_v1_overlay_root", "")))
    overlay = _resolve(overlay_root)
    current_runner = _resolve(current_runner_path)
    source_validated = bool(protocol_v1_overlay_preflight.get("protocol_v1_overlay_validated", False))
    source_runner = source_root / "gl_gym" / "experiments" / "run_frozen_benchmark.py"

    copy_performed = False
    errors: list[str] = []
    if not source_validated:
        errors.append("source_protocol_v1_overlay_not_validated")
    if not source_root.exists():
        errors.append("source_protocol_v1_overlay_root_missing")
    if not source_runner.exists():
        errors.append("source_overlay_runner_missing")
    if not current_runner.exists():
        errors.append("current_working_tree_runner_missing")

    if not errors:
        _copy_overlay(source_root, overlay)
        copy_performed = True
        controlled_runner = _copy_current_runner(overlay, current_runner)
    else:
        controlled_runner = overlay / "gl_gym" / "experiments" / "run_frozen_benchmark.py"

    protocol_rows = _protocol_hash_rows(source_root=source_root, overlay_root=overlay) if copy_performed else []
    protocol_hash_mismatch_count = sum(1 for row in protocol_rows if not row.get("hash_matches", False))
    runner_delta = (
        _runner_delta_valid(
            old_runner=source_runner,
            new_runner=controlled_runner,
            current_runner=current_runner,
            candidate_controller=candidate_controller,
        )
        if copy_performed
        else {
            "runner_delta_classification": "controlled_canary_runner_surface_only",
            "runner_delta_valid": False,
            "new_runner_supports_candidate_controller": False,
            "old_runner_supports_candidate_controller": False,
            "current_runner_supports_candidate_controller": _runner_supports(current_runner, candidate_controller) if current_runner.exists() else False,
        }
    )
    compile_report = (
        _compile_files([controlled_runner])
        if validate_compile and controlled_runner.exists()
        else {"compile_pass": True, "compile_results": [], "compile_skipped": True}
    )
    import_report = (
        _validate_imports(overlay)
        if validate_import and copy_performed
        else {"import_pass": True, "import_results": [], "import_skipped": True}
    )
    overlay_validated = bool(
        source_validated
        and copy_performed
        and protocol_hash_mismatch_count == 0
        and runner_delta.get("runner_delta_valid", False)
        and compile_report.get("compile_pass", False)
        and import_report.get("import_pass", False)
    )
    if copy_performed:
        comparison = filecmp.dircmp(str(source_root), str(overlay))
        changed_top_level_files = sorted(comparison.diff_files)
    else:
        changed_top_level_files = []

    return {
        "schema_version": "protocol_v1_controlled_canary_overlay_preflight_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "controlled-canary-overlay-preflight-only / no replay",
            "reopens_rejected_preset": False,
        },
        "protocol_v1_overlay_available": overlay_validated,
        "protocol_v1_overlay_validated": overlay_validated,
        "controlled_canary_overlay_available": overlay_validated,
        "controlled_canary_overlay_validated": overlay_validated,
        "controlled_canary_runner_surface_validated": bool(runner_delta.get("runner_delta_valid", False)),
        "actual_protocol_implementation_for_execution": (
            "protocol_v1_controlled_canary_overlay"
            if overlay_validated
            else "protocol_v1_controlled_canary_overlay_required"
        ),
        "protocol_v1_overlay_root": str(overlay),
        "source_protocol_v1_overlay_root": str(source_root),
        "candidate_controller": candidate_controller,
        "runner_delta_classification": runner_delta.get("runner_delta_classification", ""),
        "runner_delta": runner_delta,
        "overlay_runner_path": str(controlled_runner),
        "current_runner_path": str(current_runner),
        "copy_performed": copy_performed,
        "protocol_hash_mismatch_count": protocol_hash_mismatch_count,
        "protocol_hash_rows": protocol_rows,
        "changed_top_level_files": changed_top_level_files,
        "compile_pass": bool(compile_report.get("compile_pass", False)),
        "compile_results": compile_report.get("compile_results", []),
        "import_pass": bool(import_report.get("import_pass", False)),
        "import_results": import_report.get("import_results", []),
        "import_stderr": import_report.get("stderr", ""),
        "errors": errors,
        "metadata_replay_execution_allowed": False,
        "controlled_replay_allowed": False,
        "minimal_controlled_canary_allowed": False,
        "performance_claim_allowed": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "replay_run": False,
        "notes": [
            "The controlled canary overlay is copied from the validated protocol v1 overlay.",
            "Protocol files must remain hash-identical to the source overlay.",
            "The only intended delta is the runner control surface needed to admit the strict hot-dry proposer canary.",
        ],
        "next_action": (
            "refresh_controlled_replay_admission_review_with_controlled_canary_overlay"
            if overlay_validated
            else "repair_protocol_v1_controlled_canary_overlay_preflight"
        ),
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Protocol v1 Controlled Canary Overlay Preflight",
        "",
        "- Mode: controlled-canary-overlay-preflight-only / no replay",
        f"- Controlled canary overlay validated: {report.get('controlled_canary_overlay_validated', False)}",
        f"- Actual protocol implementation: `{report.get('actual_protocol_implementation_for_execution', '')}`",
        f"- Overlay root: `{report.get('protocol_v1_overlay_root', '')}`",
        f"- Source overlay root: `{report.get('source_protocol_v1_overlay_root', '')}`",
        f"- Runner delta classification: `{report.get('runner_delta_classification', '')}`",
        f"- Controlled runner surface validated: {report.get('controlled_canary_runner_surface_validated', False)}",
        f"- Protocol hash mismatch count: {report.get('protocol_hash_mismatch_count', 0)}",
        f"- Compile pass: {report.get('compile_pass', False)}",
        f"- Import pass: {report.get('import_pass', False)}",
        "- Controlled replay allowed: false",
        "- Performance claim allowed: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Runner Delta",
        "",
        "| check | value |",
        "| --- | --- |",
    ]
    for key, value in dict(report.get("runner_delta", {})).items():
        lines.append(f"| {key} | `{value}` |")
    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {item}" for item in report.get("errors", []) or [])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-v1-overlay-preflight-json", required=True)
    parser.add_argument("--overlay-root", default=str(DEFAULT_OVERLAY_ROOT))
    parser.add_argument("--current-runner-path", default=str(PROJECT_ROOT / "gl_gym" / "experiments" / "run_frozen_benchmark.py"))
    parser.add_argument("--candidate-controller", default=DEFAULT_CANDIDATE_CONTROLLER)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--skip-compile-validation", action="store_true")
    parser.add_argument("--skip-import-validation", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        protocol_v1_overlay_preflight=_load(args.protocol_v1_overlay_preflight_json),
        overlay_root=args.overlay_root,
        current_runner_path=args.current_runner_path,
        candidate_controller=args.candidate_controller,
        validate_compile=not args.skip_compile_validation,
        validate_import=not args.skip_import_validation,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"controlled_canary_overlay_validated={report['controlled_canary_overlay_validated']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["controlled_canary_overlay_validated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
