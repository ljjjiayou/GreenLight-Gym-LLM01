"""Build v75 opt-in profile-action envelope shadow acquisition authorization.

This stage authorizes and scopes the next evidence step for v74 envelope
provenance. It generates commands for a future opt-in shadow trace acquisition,
but it does not execute online LLM calls, run rollout, run replay, change the
default controller, alter final actions, or make performance claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (  # noqa: E402
    AUDIT_DIR,
    FAILURE_SCENARIOS,
)
from gl_gym.experiments.qwen37plus_profile_action_envelope_shadow_patch_v74 import (  # noqa: E402
    READINESS_JSON as V74_READINESS_JSON,
)


MODEL_NAME = "qwen3.7-plus"
ARTIFACT_DATE = "20260604"
VERSION = "v75"

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "v75_profile_action_envelope_opt_in_shadow_traces"
)
TRACE_DIR = BENCHMARK_DIR / "traces"
CACHE_PATH = PROJECT_ROOT / "gl_gym" / "result" / "plan_cache" / "qwen37plus_v75_profile_action_envelope_shadow_cache.json"

AUTHORIZATION_JSON = AUDIT_DIR / "qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_authorization_20260604_v75.json"
AUTHORIZATION_MD = AUDIT_DIR / "qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_authorization_20260604_v75.md"
MANIFEST_JSON = AUDIT_DIR / "qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_manifest_20260604_v75.json"
MANIFEST_MD = AUDIT_DIR / "qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_manifest_20260604_v75.md"
READINESS_JSON = AUDIT_DIR / "profile_action_envelope_opt_in_shadow_acquisition_readiness_20260604_v75.json"
READINESS_MD = AUDIT_DIR / "profile_action_envelope_opt_in_shadow_acquisition_readiness_20260604_v75.md"

SCENARIO_GROUPS = (
    {"name": "group_a", "years": "2010", "days": "180", "seeds": "43"},
    {"name": "group_b", "years": "2018", "days": "181", "seeds": "42,43"},
)
AGENT_CONFIG_OVERRIDES: dict[str, Any] = {
    "profile_action_envelope_shadow_enabled": True,
    "profile_action_envelope_shadow_record_provenance": True,
    "profile_action_envelope_shadow_max_candidates": 5,
    "profile_action_candidate_shadow_enabled": True,
    "profile_action_candidate_shadow_record_provenance": True,
    "profile_action_candidate_shadow_max_candidates": 5,
    "transition_gate_enabled": False,
    "profile_feasibility_gate_enabled": False,
    "profile_template_patch_enabled": False,
    "fallback_post_selection_veto_enabled": False,
    "recovery_anchor_enabled": False,
}
REQUIRED_TRACE_FIELDS = (
    "profile_action_envelope_shadow_enabled",
    "profile_action_envelope_shadow_candidate_count",
    "profile_action_envelope_shadow_eligible_candidate_count",
    "profile_action_envelope_shadow_best_name",
    "profile_action_envelope_shadow_best_profile",
    "profile_action_envelope_shadow_best_score",
    "profile_action_envelope_shadow_best_eligible",
    "profile_action_envelope_shadow_best_rejection_reason",
    "profile_action_envelope_shadow_final_action_changed",
    "profile_action_envelope_shadow_candidates_json",
)
BOUNDARY_FALSE_FIELDS = (
    "online_llm_called",
    "new_rollout_run",
    "default_llm_rspc_v2_changed",
    "fallback_enhanced",
    "controlled_replay_allowed",
    "controlled_replay_execution_allowed",
    "metadata_replay_execution_allowed",
    "performance_claim_allowed",
    "promotion_evidence",
    "final_action_changed",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _rel(path: str | Path) -> str:
    try:
        return str(_resolve(path).relative_to(PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(path)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dotenv_key_present(key: str) -> bool:
    if os.getenv(key):
        return True
    dotenv = PROJECT_ROOT / ".env"
    if not dotenv.exists():
        return False
    try:
        for raw in dotenv.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key and value.strip():
                return True
    except Exception:
        return False
    return False


def _with_boundaries(payload: dict[str, Any]) -> dict[str, Any]:
    for field in BOUNDARY_FALSE_FIELDS:
        payload[field] = False
    return payload


def _command_for_group(group: Mapping[str, str], *, output_dir: str | Path = BENCHMARK_DIR) -> list[str]:
    output_json = _resolve(output_dir) / f"qwen37plus_profile_action_envelope_opt_in_shadow_v75_{group['name']}.json"
    return [
        "python",
        "gl_gym\\experiments\\run_frozen_benchmark.py",
        "--years",
        str(group["years"]),
        "--days",
        str(group["days"]),
        "--seeds",
        str(group["seeds"]),
        "--controllers",
        "llm_rspc_v2",
        "--max-steps",
        "720",
        "--llm-model",
        MODEL_NAME,
        "--llm-max-iterations",
        "1",
        "--llm-max-tokens",
        "260",
        "--plan-cache-mode",
        "record",
        "--plan-cache-path",
        _rel(CACHE_PATH),
        "--plan-cache-key-policy",
        "scenario_timestep",
        "--agent-config-overrides",
        json.dumps(AGENT_CONFIG_OVERRIDES, sort_keys=True, separators=(",", ":")),
        "--output-json",
        _rel(output_json),
        "--output-trace-dir",
        _rel(TRACE_DIR),
    ]


def _v74_ready_for_acquisition(readiness: Mapping[str, Any]) -> bool:
    return bool(
        readiness.get("version") == "v74"
        and readiness.get("next_action") == "needs_profile_action_envelope_opt_in_shadow_trace_acquisition"
        and readiness.get("final_action_invariant", False)
        and not readiness.get("online_llm_called", False)
        and not readiness.get("promotion_evidence", False)
    )


def build_authorization(
    *,
    v74_readiness: Mapping[str, Any] | None = None,
    user_authorized: bool = True,
    credentials_present: bool | None = None,
) -> dict[str, Any]:
    readiness = dict(v74_readiness or _load_json(V74_READINESS_JSON))
    if credentials_present is None:
        credentials_present = _dotenv_key_present("BAILIAN_API_KEY")
    authorization = {
        "artifact": "qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_authorization_20260604_v75",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "authorization_type": "user_delegated_opt_in_profile_action_envelope_shadow_trace_acquisition",
        "authorization_source": "user_delegated_codex_authorization_20260604",
        "user_authorized_online_llm_acquisition": bool(user_authorized),
        "codex_authorized_to_issue_acquisition_scope": bool(user_authorized),
        "why_existing_cache_or_trace_is_insufficient": str(
            readiness.get("reason") or "No v74 envelope shadow metadata exists in current traces."
        ),
        "model": MODEL_NAME,
        "api_key_source": ".env:BAILIAN_API_KEY",
        "online_llm_credentials_present": bool(credentials_present),
        "scenario_windows": list(FAILURE_SCENARIOS),
        "scenario_groups": list(SCENARIO_GROUPS),
        "max_calls_or_budget": "three_failure_windows_only_max_steps_720_interval_12_max_iterations_1",
        "cache_write_policy": "record_to_independent_v75_profile_action_envelope_shadow_cache_only",
        "cache_path": _rel(CACHE_PATH),
        "trace_dir": _rel(TRACE_DIR),
        "default_controller_changed": False,
        "final_action_changed": False,
        "shadow_only": True,
        "online_llm_allowed_for_acquisition": bool(user_authorized),
        "online_llm_allowed_for_audit_or_tests": False,
        "required_provenance_fields": list(REQUIRED_TRACE_FIELDS),
        "agent_config_overrides": dict(AGENT_CONFIG_OVERRIDES),
        "success_condition": (
            "v74 audit finds profile_action_envelope_shadow candidates with complete contract, "
            "normal_path_profile_action_envelope source, Tomato Safety projection, and final action invariant."
        ),
        "stop_condition": (
            "action diff, hard-safety regression, runtime collapse, missing envelope metadata, "
            "fallback candidate source, missing Tomato Safety projection, or provider/cache error."
        ),
    }
    authorization["v74_readiness_ready_for_acquisition"] = _v74_ready_for_acquisition(readiness)
    return _with_boundaries(authorization)


def build_manifest(authorization: Mapping[str, Any] | None = None) -> dict[str, Any]:
    auth = dict(authorization or build_authorization())
    commands = [
        {
            "group": group["name"],
            "years": group["years"],
            "days": group["days"],
            "seeds": group["seeds"],
            "argv": _command_for_group(group),
            "command": " ".join(_command_for_group(group)),
            "execution_authorized": bool(auth.get("user_authorized_online_llm_acquisition", False)),
        }
        for group in SCENARIO_GROUPS
    ]
    manifest = {
        "artifact": "qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_manifest_20260604_v75",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "scope": "minimal_profile_action_envelope_opt_in_shadow_trace_acquisition",
        "authorization_source": str(auth.get("authorization_source", "")),
        "model": MODEL_NAME,
        "controller": "llm_rspc_v2",
        "controlled_controller_included": False,
        "scenarios": list(FAILURE_SCENARIOS),
        "scenario_groups": list(SCENARIO_GROUPS),
        "max_steps": 720,
        "llm_interval": 12,
        "llm_max_iterations": 1,
        "llm_max_tokens": 260,
        "plan_cache_mode": "record",
        "plan_cache_key_policy": "scenario_timestep",
        "cache_path": _rel(CACHE_PATH),
        "output_dir": _rel(BENCHMARK_DIR),
        "trace_dir": _rel(TRACE_DIR),
        "agent_config_overrides": dict(AGENT_CONFIG_OVERRIDES),
        "required_trace_fields": list(REQUIRED_TRACE_FIELDS),
        "post_acquisition_audit_command": [
            "python",
            "-m",
            "gl_gym.experiments.qwen37plus_profile_action_envelope_shadow_patch_v74",
            _rel(TRACE_DIR),
        ],
        "online_llm_allowed_for_acquisition": bool(auth.get("online_llm_allowed_for_acquisition", False)),
        "commands": commands,
        "rollout_command_generated": True,
    }
    return _with_boundaries(manifest)


def build_readiness(
    authorization: Mapping[str, Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    auth = dict(authorization or build_authorization())
    man = dict(manifest or build_manifest(auth))
    if not auth.get("v74_readiness_ready_for_acquisition", False):
        next_action = "repair_v74_readiness_before_acquisition_authorization"
        blocked_reason = "v74_readiness_does_not_request_opt_in_trace_acquisition"
        executable = False
    elif not auth.get("user_authorized_online_llm_acquisition", False):
        next_action = "await_user_authorization_for_v75_opt_in_shadow_acquisition"
        blocked_reason = "user_authorization_missing"
        executable = False
    elif not auth.get("online_llm_credentials_present", False):
        next_action = "online_llm_credentials_precheck_blocker"
        blocked_reason = "BAILIAN_API_KEY_not_detected"
        executable = False
    else:
        next_action = "execute_v75_opt_in_shadow_trace_acquisition_commands"
        blocked_reason = ""
        executable = True
    readiness = {
        "artifact": "profile_action_envelope_opt_in_shadow_acquisition_readiness_20260604_v75",
        "version": VERSION,
        "artifact_date": ARTIFACT_DATE,
        "model": MODEL_NAME,
        "authorization_ready": bool(auth.get("user_authorized_online_llm_acquisition", False)),
        "v74_readiness_ready_for_acquisition": bool(auth.get("v74_readiness_ready_for_acquisition", False)),
        "online_llm_credentials_present": bool(auth.get("online_llm_credentials_present", False)),
        "executable": bool(executable),
        "blocked_reason": blocked_reason,
        "scenario_count": len(FAILURE_SCENARIOS),
        "command_count": len(man.get("commands", []) or []),
        "online_llm_allowed_for_acquisition": bool(auth.get("online_llm_allowed_for_acquisition", False)),
        "online_llm_allowed_for_audit_or_tests": False,
        "shadow_only": True,
        "default_controller_changed": False,
        "controlled_controller_included": False,
        "cache_path": str(auth.get("cache_path", "")),
        "trace_dir": str(auth.get("trace_dir", "")),
        "next_action": next_action,
    }
    readiness = _with_boundaries(readiness)
    readiness["rollout_command_generated"] = bool(man.get("rollout_command_generated", False))
    return readiness


def build_authorization_markdown(authorization: Mapping[str, Any]) -> str:
    lines = [
        "# qwen3.7-plus v75 Profile Action Envelope Opt-In Authorization",
        "",
        f"- authorization_type={authorization.get('authorization_type', '')}",
        f"- user_authorized_online_llm_acquisition={bool(authorization.get('user_authorized_online_llm_acquisition', False))}",
        f"- model={authorization.get('model', '')}",
        f"- scenario_windows={','.join(str(item) for item in authorization.get('scenario_windows', []) or [])}",
        f"- online_llm_allowed_for_acquisition={bool(authorization.get('online_llm_allowed_for_acquisition', False))}",
        f"- online_llm_called={bool(authorization.get('online_llm_called', False))}",
        f"- default_controller_changed={bool(authorization.get('default_controller_changed', False))}",
        f"- final_action_changed={bool(authorization.get('final_action_changed', False))}",
        f"- shadow_only={bool(authorization.get('shadow_only', False))}",
        f"- cache_write_policy={authorization.get('cache_write_policy', '')}",
        f"- success_condition={authorization.get('success_condition', '')}",
        f"- stop_condition={authorization.get('stop_condition', '')}",
    ]
    return "\n".join(lines) + "\n"


def build_manifest_markdown(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# qwen3.7-plus v75 Profile Action Envelope Opt-In Acquisition Manifest",
        "",
        f"- controller={manifest.get('controller', '')}",
        f"- controlled_controller_included={bool(manifest.get('controlled_controller_included', False))}",
        f"- plan_cache_mode={manifest.get('plan_cache_mode', '')}",
        f"- cache_path={manifest.get('cache_path', '')}",
        f"- trace_dir={manifest.get('trace_dir', '')}",
        f"- online_llm_allowed_for_acquisition={bool(manifest.get('online_llm_allowed_for_acquisition', False))}",
        f"- online_llm_called={bool(manifest.get('online_llm_called', False))}",
        f"- rollout_command_generated={bool(manifest.get('rollout_command_generated', False))}",
        "",
        "## Commands",
        "",
    ]
    for item in manifest.get("commands", []) or []:
        if isinstance(item, Mapping):
            lines.append(f"- {item.get('group', '')}: `{item.get('command', '')}`")
    lines.extend(["", "## Post Acquisition Audit", ""])
    lines.append("`" + " ".join(str(x) for x in manifest.get("post_acquisition_audit_command", []) or []) + "`")
    return "\n".join(lines) + "\n"


def build_readiness_markdown(readiness: Mapping[str, Any]) -> str:
    lines = [
        "# qwen3.7-plus v75 Profile Action Envelope Opt-In Readiness",
        "",
        f"- authorization_ready={bool(readiness.get('authorization_ready', False))}",
        f"- v74_readiness_ready_for_acquisition={bool(readiness.get('v74_readiness_ready_for_acquisition', False))}",
        f"- online_llm_credentials_present={bool(readiness.get('online_llm_credentials_present', False))}",
        f"- executable={bool(readiness.get('executable', False))}",
        f"- blocked_reason={readiness.get('blocked_reason', '')}",
        f"- online_llm_allowed_for_acquisition={bool(readiness.get('online_llm_allowed_for_acquisition', False))}",
        f"- online_llm_called={bool(readiness.get('online_llm_called', False))}",
        f"- default_controller_changed={bool(readiness.get('default_controller_changed', False))}",
        f"- final_action_changed={bool(readiness.get('final_action_changed', False))}",
        f"- performance_claim_allowed={bool(readiness.get('performance_claim_allowed', False))}",
        f"- promotion_evidence={bool(readiness.get('promotion_evidence', False))}",
        f"- rollout_command_generated={bool(readiness.get('rollout_command_generated', False))}",
        f"- next_action={readiness.get('next_action', '')}",
    ]
    return "\n".join(lines) + "\n"


def write_all(
    *,
    user_authorized: bool = True,
    v74_readiness_json: str | Path = V74_READINESS_JSON,
    credentials_present: bool | None = None,
) -> dict[str, str]:
    authorization = build_authorization(
        v74_readiness=_load_json(v74_readiness_json),
        user_authorized=user_authorized,
        credentials_present=credentials_present,
    )
    manifest = build_manifest(authorization)
    readiness = build_readiness(authorization, manifest)
    outputs = {
        AUTHORIZATION_JSON: (authorization, build_authorization_markdown(authorization)),
        MANIFEST_JSON: (manifest, build_manifest_markdown(manifest)),
        READINESS_JSON: (readiness, build_readiness_markdown(readiness)),
    }
    written: dict[str, str] = {}
    for json_path, (payload, markdown) in outputs.items():
        md_path = json_path.with_suffix(".md")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
        written[json_path.stem + "_json"] = str(json_path)
        written[json_path.stem + "_md"] = str(md_path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v74-readiness-json", default=str(V74_READINESS_JSON))
    parser.add_argument("--not-authorized", action="store_true", help="Build a blocked packet without delegated authorization.")
    parser.add_argument("--credentials-present", action="store_true", help="Force credential-present status for tests/dry planning.")
    parser.add_argument("--credentials-missing", action="store_true", help="Force credential-missing status for tests/dry planning.")
    args = parser.parse_args(argv)

    credentials_present: bool | None = None
    if args.credentials_present:
        credentials_present = True
    if args.credentials_missing:
        credentials_present = False
    written = write_all(
        user_authorized=not bool(args.not_authorized),
        v74_readiness_json=args.v74_readiness_json,
        credentials_present=credentials_present,
    )
    print(json.dumps(written, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
