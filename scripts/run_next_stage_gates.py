"""Run the next-stage engineering gates without calling an online LLM.

The gate is deliberately conservative: compile core modules, run unit tests,
and, when a frozen replay JSON is available, audit it for runtime/cache/safety
regressions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPLAY_JSON = (
    PROJECT_ROOT
    / "gl_gym"
    / "result"
    / "benchmarks"
    / "hot_dry_proposer_strict_holdout_cachefilled_v1_20260517"
    / "replay_all_mergedcache.json"
)

CORE_COMPILE_FILES = [
    "gl_gym/agent/llm_agent.py",
    "gl_gym/agent/intent_contract.py",
    "gl_gym/agent/profile_generator.py",
    "gl_gym/experiments/audit_frozen_benchmark_summary.py",
    "gl_gym/experiments/audit_rspc_scoring_sweep.py",
    "gl_gym/experiments/candidate_pool_audit.py",
    "gl_gym/experiments/candidate_pool_repair_shadow.py",
    "gl_gym/experiments/candidate_response_calibration.py",
    "gl_gym/experiments/canopy_boundary_shadow_audit.py",
    "gl_gym/experiments/check_plan_cache_coverage.py",
    "gl_gym/experiments/diagnose_ppo_vs_llm.py",
    "gl_gym/experiments/extract_rspc_scoring_delta_windows.py",
    "gl_gym/experiments/extract_rspc_scoring_delta_segments.py",
    "gl_gym/experiments/run_frozen_benchmark.py",
    "gl_gym/experiments/frozen_benchmark_protocol.py",
    "gl_gym/experiments/failure_window_extractor.py",
    "gl_gym/experiments/build_regime_stress_suite.py",
    "gl_gym/experiments/run_regime_stress_suite.py",
    "gl_gym/experiments/select_hot_dry_stress_scenarios.py",
    "gl_gym/experiments/rspc_action_scoring_audit.py",
    "gl_gym/experiments/sweep_rspc_scoring_weights.py",
    "gl_gym/experiments/hot_dry_strict_effect_audit.py",
    "gl_gym/experiments/hot_dry_controlled_replay_trace_audit.py",
    "gl_gym/experiments/hot_dry_failure_case_audit.py",
    "gl_gym/experiments/mainline_diff_authorization_closure.py",
    "gl_gym/experiments/mainline_diff_authorization_packet.py",
    "gl_gym/experiments/mainline_diff_authorization_status.py",
    "gl_gym/experiments/mainline_diff_decision_packet.py",
    "gl_gym/experiments/metadata_replay_expanded_cache_coverage_plan.py",
    "gl_gym/experiments/expanded_metadata_first_wave_scenario_manifest.py",
    "gl_gym/experiments/expanded_metadata_second_wave_scenario_manifest.py",
    "gl_gym/experiments/expanded_metadata_coverage_consolidation.py",
    "gl_gym/experiments/controlled_replay_minimal_canary_manifest.py",
    "gl_gym/experiments/controlled_replay_admission_review.py",
    "gl_gym/experiments/controlled_replay_minimal_canary_execution_record.py",
    "gl_gym/experiments/strict_targeted_shadow_sweep_v29_manifest.py",
    "gl_gym/experiments/strict_targeted_shadow_sweep_v29_execution_record.py",
    "gl_gym/experiments/strict_targeted_shadow_sweep_v29_readiness.py",
    "gl_gym/experiments/strict_targeted_shadow_sweep_v30_manifest.py",
    "gl_gym/experiments/strict_targeted_shadow_sweep_v30_execution_record.py",
    "gl_gym/experiments/strict_targeted_shadow_sweep_v30_readiness.py",
    "gl_gym/experiments/controlled_canary_near_miss_source_catalog.py",
    "gl_gym/experiments/controlled_canary_existing_cache_source_inventory.py",
    "gl_gym/experiments/strict_targeted_shadow_sweep_v33_manifest.py",
    "gl_gym/experiments/strict_targeted_shadow_sweep_v33_cache_coverage.py",
    "gl_gym/experiments/strict_targeted_shadow_sweep_v33_execution_record.py",
    "gl_gym/experiments/strict_targeted_shadow_sweep_v33_readiness.py",
    "gl_gym/experiments/controlled_canary_shadow_cache_acquisition_target_spec.py",
    "gl_gym/experiments/controlled_canary_new_shadow_source_inventory.py",
    "gl_gym/experiments/controlled_canary_shadow_cache_acquisition_request.py",
    "gl_gym/experiments/controlled_canary_shadow_cache_acquisition_readiness.py",
    "gl_gym/experiments/controlled_canary_shadow_sourced_admission_review.py",
    "gl_gym/experiments/controlled_canary_shadow_sourced_manifest.py",
    "gl_gym/experiments/controlled_canary_shadow_sourced_execution_record.py",
    "gl_gym/experiments/controlled_canary_shadow_sourced_readiness.py",
    "gl_gym/experiments/controlled_canary_shadow_sourced_result_review.py",
    "gl_gym/experiments/controlled_canary_independent_trigger_blocker_diagnosis.py",
    "gl_gym/experiments/controlled_canary_independent_trigger_discovery.py",
    "gl_gym/experiments/controlled_canary_independent_wave2_manifest.py",
    "gl_gym/experiments/controlled_canary_independent_wave2_execution_record.py",
    "gl_gym/experiments/controlled_canary_independent_wave2_readiness.py",
    "gl_gym/experiments/protocol_v1_controlled_canary_overlay_preflight.py",
    "gl_gym/experiments/expanded_strict_metadata_replay_first_wave_execution_record.py",
    "gl_gym/experiments/expanded_strict_metadata_replay_second_wave_execution_record.py",
    "gl_gym/experiments/canonical_action_diff_baseline_trace_manifest.py",
    "gl_gym/experiments/canonical_strict_metadata_replay_execution_request.py",
    "gl_gym/experiments/canonical_strict_metadata_replay_stage_b_authorization_request.py",
    "gl_gym/experiments/canonical_strict_metadata_replay_stage_b_rerun_execution_record.py",
    "gl_gym/experiments/protocol_v1_ephemeral_overlay_preflight.py",
    "gl_gym/experiments/strict_metadata_replay_two_stage_authorization_packet.py",
    "gl_gym/experiments/strict_metadata_replay_run_plan.py",
    "gl_gym/experiments/strict_metadata_replay_execution_authorization_packet.py",
    "gl_gym/experiments/untracked_file_inventory.py",
    "gl_gym/experiments/post_guardrail_runtime_provenance_audit.py",
    "gl_gym/experiments/post_guardrail_runtime_provenance_design.py",
    "gl_gym/experiments/post_guardrail_vent_rewrite_counterfactual_audit.py",
    "gl_gym/experiments/post_guardrail_rule_provenance_audit.py",
    "gl_gym/experiments/post_guardrail_static_provenance_audit.py",
    "gl_gym/experiments/post_guardrail_vent_floor_shadow_policy_audit.py",
    "gl_gym/experiments/post_guardrail_vent_floor_shadow_policy_v2_design.py",
    "gl_gym/experiments/proxy_warning_not_actionable_audit.py",
    "gl_gym/experiments/canopy_warning_to_action_blocker_design.py",
    "gl_gym/experiments/post_guardrail_joint_prediction_readiness.py",
    "gl_gym/experiments/post_guardrail_joint_prediction_readiness_v2.py",
    "gl_gym/experiments/candidate_family_shadow_ablation.py",
    "gl_gym/experiments/regime_counterexample_suite_manifest.py",
    "gl_gym/experiments/regime_counterexample_shadow_audit.py",
    "gl_gym/experiments/regime_counterexample_replay_readiness.py",
    "gl_gym/experiments/post_guardrail_policy_shadow_status.py",
    "gl_gym/experiments/post_guardrail_policy_next_status.py",
    "gl_gym/experiments/post_guardrail_provenance_and_blocker_status.py",
    "gl_gym/experiments/post_guardrail_shadow_policy_instrumentation_status.py",
    "gl_gym/experiments/post_guardrail_override_audit.py",
    "scripts/check_greenhouse_skills.py",
    "scripts/run_next_stage_gates.py",
]


def _default_pytest_targets() -> list[str]:
    targets = sorted(str(path.relative_to(PROJECT_ROOT)) for path in (PROJECT_ROOT / "tests").glob("test_*.py"))
    return targets or ["tests"]


def _run(command: Sequence[str], *, cwd: Path) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"\n[gate] {printable}", flush=True)
    result = subprocess.run(command, cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _existing_compile_files() -> list[str]:
    files: list[str] = []
    for rel in CORE_COMPILE_FILES:
        if (PROJECT_ROOT / rel).exists():
            files.append(rel)
        else:
            print(f"[gate] skip missing compile target: {rel}")
    return files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Only run py_compile and optional replay audit.",
    )
    parser.add_argument(
        "--full-pytest",
        action="store_true",
        help="Run full pytest discovery, including simulator smoke tests that may be slower or stochastic.",
    )
    parser.add_argument(
        "--replay-json",
        default=str(DEFAULT_REPLAY_JSON),
        help="Frozen benchmark JSON to audit if it exists.",
    )
    parser.add_argument(
        "--audit-output-dir",
        default=str(PROJECT_ROOT / "gl_gym" / "result" / "audits" / "next_stage_gates"),
    )
    parser.add_argument("--baseline-controller", default="llm_rspc_v2")
    parser.add_argument("--compare-controller", default="llm_rspc_v2_hot_dry_proposer_strict")
    parser.add_argument(
        "--include-hot-dry-failure-fixture",
        action="store_true",
        help="Run the canonical rejected hot-dry scoring preset failure fixture audit.",
    )
    return parser.parse_args(argv)


def _run_hot_dry_failure_fixture(output_dir: Path) -> None:
    baseline_dir = (
        PROJECT_ROOT
        / "gl_gym"
        / "result"
        / "benchmarks"
        / "selected_hot_dry_scoring_sweep_v1_20260518"
        / "traces"
        / "balanced"
    )
    compare_dir = (
        PROJECT_ROOT
        / "gl_gym"
        / "result"
        / "benchmarks"
        / "selected_hot_dry_scoring_sweep_v1_20260518"
        / "traces"
        / "hot_dry_relief"
    )
    if not baseline_dir.exists() or not compare_dir.exists():
        print("[gate] skip hot-dry failure fixture; selected-suite traces not found")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "hot_dry_failure_fixture.json"
    output_md = output_dir / "hot_dry_failure_fixture.md"
    _run(
        [
            sys.executable,
            "gl_gym/experiments/hot_dry_failure_case_audit.py",
            "--baseline-trace-dir",
            str(baseline_dir),
            "--compare-trace-dir",
            str(compare_dir),
            "--scenario-id",
            "y2020_d120_s44_n240_llm_rspc_v2",
            "--start-step",
            "220",
            "--end-step",
            "239",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        cwd=PROJECT_ROOT,
    )
    report = json.loads(output_json.read_text(encoding="utf-8"))
    if not bool(report.get("hard_safety_regression", False)):
        raise SystemExit("hot-dry failure fixture did not report hard_safety_regression")
    if int(report.get("first_extra_canopy_lt0_step", -1) or -1) != 234:
        raise SystemExit("hot-dry failure fixture first_extra_canopy_lt0_step was not 234")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    compile_files = _existing_compile_files()
    if not compile_files:
        print("[gate] no compile files found")
        return 1

    _run([sys.executable, "-m", "py_compile", *compile_files], cwd=PROJECT_ROOT)

    if not args.skip_pytest:
        pytest_targets = [] if args.full_pytest else _default_pytest_targets()
        _run([sys.executable, "-m", "pytest", "-q", *pytest_targets], cwd=PROJECT_ROOT)

    replay_json = Path(args.replay_json)
    if not replay_json.is_absolute():
        replay_json = PROJECT_ROOT / replay_json
    if replay_json.exists():
        output_dir = Path(args.audit_output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = replay_json.stem
        _run(
            [
                sys.executable,
                "gl_gym/experiments/audit_frozen_benchmark_summary.py",
                "--input",
                str(replay_json),
                "--output-json",
                str(output_dir / f"{stem}_summary.json"),
                "--output-md",
                str(output_dir / f"{stem}_summary.md"),
                "--baseline-controller",
                str(args.baseline_controller),
                "--compare-controller",
                str(args.compare_controller),
            ],
            cwd=PROJECT_ROOT,
        )
    else:
        print(f"[gate] skip replay audit; file not found: {replay_json}")

    if args.include_hot_dry_failure_fixture:
        fixture_dir = Path(args.audit_output_dir)
        if not fixture_dir.is_absolute():
            fixture_dir = PROJECT_ROOT / fixture_dir
        _run_hot_dry_failure_fixture(fixture_dir)

    print("\n[gate] all requested gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
