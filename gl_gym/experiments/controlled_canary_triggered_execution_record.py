"""Record authorization for a conditional triggered controlled canary."""

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


def _command_for_group(
    *,
    runner_path: str,
    group: Mapping[str, Any],
    controllers: Sequence[str],
    selected_cache_path: str,
    output_root: str,
) -> list[str]:
    group_id = str(group.get("group_id", "triggered_canary"))
    output_json = str(Path(output_root) / f"triggered_controlled_canary_{group_id}_20260525.json")
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
    manifest: Mapping[str, Any],
    coverage: Mapping[str, Any],
    admission_review: Mapping[str, Any],
    authorization_source: str,
    output_root: str,
) -> dict[str, Any]:
    coverage_pass = bool(
        coverage.get("cache_coverage_pass", False)
        and float(coverage.get("coverage_rate", 0.0) or 0.0) == 1.0
        and int(coverage.get("missing_key_count", 0) or 0) == 0
        and int(coverage.get("missing_buffered_action_count", 0) or 0) == 0
        and int(coverage.get("missing_parsed_plan_count", 0) or 0) == 0
    )
    manifest_ready = bool(manifest.get("triggered_manifest_ready", False))
    overlay_ok = bool(admission_review.get("minimal_controlled_canary_admission_pass", False))
    authorized = bool(manifest_ready and coverage_pass and overlay_ok)
    runner_path = str(admission_review.get("overlay_runner_path", ""))
    selected_cache_path = str(manifest.get("selected_cache_path", ""))
    controllers = list(manifest.get("controllers", []) or [])
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
        "schema_version": "controlled_canary_triggered_execution_record_v1",
        "mainline_alignment": {
            "affected_layers": ["Evaluation", "Safety Boundary", "RSPC"],
            "default_llm_rspc_v2_changed": False,
            "mode": "conditional triggered controlled canary",
            "reopens_rejected_preset": False,
        },
        "authorization_source": authorization_source,
        "triggered_controlled_canary_authorized": authorized,
        "controlled_replay_execution_allowed": authorized,
        "controlled_replay_scope": "triggered_controlled_canary_only" if authorized else "blocked",
        "controlled_replay_allowed": authorized,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "cache_fill_run": False,
        "online_llm_called": False,
        "manifest_ready": manifest_ready,
        "cache_coverage_pass": coverage_pass,
        "overlay_admission_pass": overlay_ok,
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
        "next_action": "execute_triggered_controlled_canary" if authorized else "triggered_canary_not_authorized",
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Triggered Controlled Canary Execution Record",
        "",
        f"- Authorized: {report.get('triggered_controlled_canary_authorized', False)}",
        f"- Scope: `{report.get('controlled_replay_scope', '')}`",
        "- Default `llm_rspc_v2` changed: false",
        "- Performance claim allowed: false",
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
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--admission-review-json", required=True)
    parser.add_argument("--authorization-source", required=True)
    parser.add_argument("--canary-output-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        manifest=_load(args.manifest_json),
        coverage=_load(args.coverage_json),
        admission_review=_load(args.admission_review_json),
        authorization_source=args.authorization_source,
        output_root=args.canary_output_root,
    )
    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"triggered_controlled_canary_authorized={report['triggered_controlled_canary_authorized']}")
    print(f"next_action={report['next_action']}")
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report["triggered_controlled_canary_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
