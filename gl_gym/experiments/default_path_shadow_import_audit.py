"""Static default-path evidence audit for Intent/Profile shadow modules.

The audit checks whether new Intent/Profile modules are imported or called by
controller-sensitive files. It does not run the controller or claim action
invariance; any default-path usage remains blocked until metadata replay proves
action_diff_steps=0.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET_MODULES = {
    "gl_gym.agent.intent_contract": "gl_gym/agent/intent_contract.py",
    "gl_gym.agent.profile_generator": "gl_gym/agent/profile_generator.py",
}
DEFAULT_PATH_FILES = [
    "gl_gym/agent/llm_agent.py",
]


def _read_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class _ImportAndCallVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.imported_names: dict[str, str] = {}
        self.import_rows: list[dict[str, Any]] = []
        self.call_rows: list[dict[str, Any]] = []
        self._function_stack: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.visit_FunctionDef(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = node.module or ""
        if module in TARGET_MODULES:
            for alias in node.names:
                local_name = alias.asname or alias.name
                self.imported_names[local_name] = f"{module}.{alias.name}"
            self.import_rows.append(
                {
                    "file": self.file_path,
                    "line": node.lineno,
                    "module": module,
                    "names": [alias.asname or alias.name for alias in node.names],
                }
            )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            if alias.name in TARGET_MODULES:
                local_name = alias.asname or alias.name.split(".")[-1]
                self.imported_names[local_name] = alias.name
                self.import_rows.append(
                    {
                        "file": self.file_path,
                        "line": node.lineno,
                        "module": alias.name,
                        "names": [local_name],
                    }
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        callee = ""
        target_module = ""
        if isinstance(node.func, ast.Name):
            local = node.func.id
            target = self.imported_names.get(local)
            if target:
                callee = target
                target_module = ".".join(target.split(".")[:-1])
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            local = node.func.value.id
            target = self.imported_names.get(local)
            if target in TARGET_MODULES:
                callee = f"{target}.{node.func.attr}"
                target_module = target
        if target_module:
            context = self._function_stack[-1] if self._function_stack else "<module>"
            self.call_rows.append(
                {
                    "file": self.file_path,
                    "line": node.lineno,
                    "callee": callee,
                    "target_module": target_module,
                    "function_context": context,
                    "shadow_named_context": "shadow" in context.lower() or "diagnostic" in context.lower(),
                }
            )
        self.generic_visit(node)


def _scan_file(path_text: str) -> dict[str, Any]:
    path = PROJECT_ROOT / path_text
    visitor = _ImportAndCallVisitor(path_text)
    if not path.exists():
        return {
            "file": path_text,
            "exists": False,
            "imports": [],
            "calls": [],
        }
    visitor.visit(_read_ast(path))
    return {
        "file": path_text,
        "exists": True,
        "imports": visitor.import_rows,
        "calls": visitor.call_rows,
    }


def build_report(files: Sequence[str] | None = None) -> dict[str, Any]:
    scanned = [_scan_file(path) for path in (files or DEFAULT_PATH_FILES)]
    imports = [row for item in scanned for row in item.get("imports", [])]
    calls = [row for item in scanned for row in item.get("calls", [])]
    target_path_rows = [
        {
            "path": path,
            "target_module": module,
            "default_path_import_present": any(row.get("module") == module for row in imports),
            "default_path_call_present": any(row.get("target_module") == module for row in calls),
            "closure_status": "shadow_only_path_pending_default_path_evidence",
            "required_evidence": [
                "strict metadata replay on current llm_agent.py",
                "default llm_rspc_v2 action_diff_steps=0",
                "selected_control and post-guardrail final action unchanged",
            ],
        }
        for module, path in TARGET_MODULES.items()
    ]
    default_path_risk = bool(imports or calls)
    return {
        "schema_version": "default_path_shadow_import_audit_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Response Estimate", "Tooling"],
            "default_llm_rspc_v2_changed": False,
            "mode": "static audit-only / no replay",
            "reopens_rejected_preset": False,
        },
        "controlled_replay_allowed": False,
        "metadata_replay_allowed": False,
        "performance_claim_allowed": False,
        "replay_run": False,
        "online_llm_called": False,
        "default_path_evidence_pass": False if default_path_risk else True,
        "shadow_only_path_pending_default_path_evidence": default_path_risk,
        "needs_action_diff_invariance": default_path_risk,
        "default_path_import_count": len(imports),
        "default_path_call_count": len(calls),
        "target_paths": target_path_rows,
        "imports": imports,
        "calls": calls,
        "notes": [
            "Static import/call evidence cannot prove action invariance.",
            "Default-path imports or calls remain blocked until metadata replay proves action_diff_steps=0.",
        ],
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Default-Path Shadow Import Audit",
        "",
        "- Mode: static audit-only / no replay",
        "- Controlled replay allowed: false",
        "- Metadata replay allowed: false",
        "- Performance claim allowed: false",
        f"- Default path evidence pass: {report.get('default_path_evidence_pass', False)}",
        f"- Needs action-diff invariance: {report.get('needs_action_diff_invariance', False)}",
        "",
        "## Target Paths",
        "",
        "| path | import_present | call_present | closure_status |",
        "| --- | --- | --- | --- |",
    ]
    for row in report.get("target_paths", []) or []:
        if isinstance(row, Mapping):
            lines.append(
                f"| `{row.get('path', '')}` | {row.get('default_path_import_present', False)} | "
                f"{row.get('default_path_call_present', False)} | {row.get('closure_status', '')} |"
            )
    lines.extend(["", "## Calls", "", "| file | line | callee | context |", "| --- | ---: | --- | --- |"])
    for row in report.get("calls", []) or []:
        if isinstance(row, Mapping):
            lines.append(
                f"| `{row.get('file', '')}` | {row.get('line', '')} | `{row.get('callee', '')}` | `{row.get('function_context', '')}` |"
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
    print(f"default_path_evidence_pass={report['default_path_evidence_pass']}")
    print(f"default_path_call_count={report['default_path_call_count']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
