"""Record authorization for the fixed Wave-1 controlled canary expansion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _coverage_pass(coverage: Mapping[str, Any]) -> bool:
    return bool(
        coverage.get("cache_coverage_pass", False)
        and float(coverage.get("coverage_rate", 0.0) or 0.0) == 1.0
        and int(coverage.get("missing_key_count", 0) or 0) == 0
        and int(coverage.get("missing_buffered_action_count", 0) or 0) == 0
        and int(coverage.get("missing_parsed_plan_count", 0) or 0) == 0
    )


def _command_for_group(
    *,
    runner_path: str,
    group: Mapping[str, Any],
    controllers: Sequence[str],
    selected_cache_path: str,
    output_root: str,
) -> list[str]:
    group_id = str(group.get("group_id", "wave1"))
    output_json = str(Path(output_root) / f"controlled_canary_expansion_wave1_{group_id}_20260525.json")
    output_trace_dir = str(Path(output_root) / "traces")
    return [
        "python",
        runner_path,
        "--years",
        ",".join(str(item) for item in group.get("years", []) or []),
        "--days",
        ",".join(str(item) for item in group.get("days", []) or []),
        "--seeds",
        ",".join(str(item) for item in group.get("seeds", []) or []),
        "--controllers",
        ",".join(str(item) for item in controllers),
        "--max-steps",
        "240",
        "--plan-cache-mode",
        "replay",
        "--plan-cache-path",
        selected_cache_path,
        "--plan-cache-strict",
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--output-json",
        output_json,
        "--output-trace-dir",
        output_trace_dir,
    ]


def build_report(
    *,
    admission_review: Mapping[str, Any],
    manifest: Mapping[str, Any],
    coverage: Mapping[str, Any],
    authorization_source: str,
    output_root: str,
) -> dict[str, Any]:
    manifest_ready = bool(manifest.get("wave1_manifest_ready", False))
    admission_pass = bool(admission_review.get("controlled_canary_expansion_admission_pass", False))
    cache_ok = _coverage_pass(coverage)
    commands = []
    controllers = list(manifest.get("controllers", []) or [])
    selected_cache_path = str(manifest.get("selected_cache_path", ""))
    runner_path = str(admission_review.get("overlay_runner_path", ""))
    cartesian_ok = all(bool(group.get("cartesian_product_safe", False)) for group in manifest.get("command_groups", []) or [])
    authorized = bool(manifest_ready and admission_pass and cache_ok and cartesian_ok)
    if authorized:
        commands = [
            _command_for_group(
                runner_path=runner_path,
                group=group,
                controllers=controllers,
                selected_cache_path=selected_cache_path,
                output_root=output_root,
            )
            for group in manifest.get("command_groups", []) or []
            if isinstance(group, Mapping)
        ]
    return {
        "schema_version": "controlled_canary_expansion_wave1_execution_record_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC", "Metadata Trace"],
            "default_llm_rspc_v2_changed": False,
            "mode": "controlled canary expansion wave1 execution",
            "reopens_rejected_preset": False,
        },
        "authorization_source": authorization_source,
        "wave1_controlled_canary_authorized": authorized,
        "controlled_replay_execution_allowed": authorized,
        "controlled_replay_allowed": authorized,
        "controlled_replay_scope": "controlled_canary_expansion_wave1_only" if authorized else "blocked",
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "manifest_ready": manifest_ready,
        "admission_pass": admission_pass,
        "cache_coverage_pass": cache_ok,
        "cartesian_product_safe": cartesian_ok,
        "scenario_ids": list(manifest.get("scenario_ids", []) or []),
        "controllers": controllers,
        "selected_cache_path": selected_cache_path,
        "actual_protocol_implementation_for_execution": "protocol_v1_controlled_canary_overlay_20260525",
        "protocol_v1_overlay_root": str(admission_review.get("protocol_v1_overlay_root", "")),
        "runner_path": runner_path,
        "planned_environment": {
            "PYTHONPATH": f"{admission_review.get('protocol_v1_overlay_root', '')};{PROJECT_ROOT}",
        },
        "output_root": output_root,
        "planned_commands": commands,
        "next_action": "execute_controlled_canary_expansion_wave1" if authorized else "wave1_not_authorized",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Controlled Canary Expansion Wave-1 Execution Record",
        "",
        f"- Authorized: {report.get('wave1_controlled_canary_authorized', False)}",
        f"- Scope: `{report.get('controlled_replay_scope', '')}`",
        "- Default `llm_rspc_v2` changed: false",
        "- Cache fill: false",
        "- Online LLM: false",
        "- Performance claim allowed: false",
        "- Promotion evidence: false",
        f"- Next action: `{report.get('next_action', '')}`",
        "",
        "## Planned Commands",
        "",
    ]
    for command in report.get("planned_commands", []) or []:
        lines.append("```powershell")
        lines.append(" ".join(str(part) for part in command))
        lines.append("```")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-review-json", required=True)
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--authorization-source", required=True)
    parser.add_argument("--canary-output-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        admission_review=_load(args.admission_review_json),
        manifest=_load(args.manifest_json),
        coverage=_load(args.coverage_json),
        authorization_source=args.authorization_source,
        output_root=args.canary_output_root,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"wave1_controlled_canary_authorized={report['wave1_controlled_canary_authorized']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["wave1_controlled_canary_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
