import json
from pathlib import Path

from gl_gym.experiments.audit_frozen_benchmark_summary import (
    audit_benchmark_summary,
    build_markdown_report,
    main,
)


def _summary(scenario, controller, **aggregate):
    defaults = {
        "steps": 240,
        "total_reward": 10.0,
        "total_profit": 2.0,
        "total_temp_violation": 0.0,
        "total_rh_low_violation": 1.0,
        "total_rh_high_violation": 0.0,
        "total_vpd_high_excess": 1.0,
        "dew_margin_air_lt0_steps": 0,
        "canopy_dew_margin_lt0_steps": 0,
        "runtime_error_steps": 0,
        "strict_cache_miss_runtime_error_steps": 0,
        "plan_cache_enabled_steps": 240,
        "plan_cache_hit_steps": 240,
    }
    defaults.update(aggregate)
    return {
        "scenario_id": scenario,
        "controller": controller,
        "aggregate": defaults,
    }


def _write_payload(tmp_path: Path, summaries):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"summaries": summaries}), encoding="utf-8")
    return path


def test_audit_passes_paired_safe_run(tmp_path):
    path = _write_payload(
        tmp_path,
        [
            _summary("s1", "llm_rspc_v2", total_rh_low_violation=4.0),
            _summary(
                "s1",
                "llm_rspc_v2_hot_dry_proposer_strict",
                total_reward=11.0,
                total_rh_low_violation=2.0,
            ),
        ],
    )

    audit = audit_benchmark_summary([path])

    assert audit["aggregate"]["decision"] == "pass"
    assert audit["aggregate"]["paired_count"] == 1
    assert audit["paired_deltas"][0]["d_total_reward"] == 1.0
    assert audit["paired_deltas"][0]["d_total_rh_low_violation"] == -2.0
    report = build_markdown_report(audit)
    assert "Frozen Benchmark Summary Audit" in report
    assert "PASS" in report


def test_audit_fails_on_runtime_and_cache_gap(tmp_path):
    path = _write_payload(
        tmp_path,
        [
            _summary(
                "s1",
                "llm_rspc_v2",
                runtime_error_steps=1,
                plan_cache_hit_steps=239,
            )
        ],
    )

    audit = audit_benchmark_summary([path])

    assert audit["aggregate"]["decision"] == "fail"
    failure_types = {failure["type"] for failure in audit["gate_failures"]}
    assert "runtime_error" in failure_types
    assert "incomplete_cache_hit" in failure_types
    assert audit["aggregate"]["unpaired_count"] == 1


def test_audit_fails_on_paired_dew_and_canopy_regression(tmp_path):
    path = _write_payload(
        tmp_path,
        [
            _summary("s1", "llm_rspc_v2", dew_margin_air_lt0_steps=0, canopy_dew_margin_lt0_steps=1),
            _summary(
                "s1",
                "llm_rspc_v2_hot_dry_proposer_strict",
                dew_margin_air_lt0_steps=1,
                canopy_dew_margin_lt0_steps=2,
            ),
        ],
    )

    audit = audit_benchmark_summary([path])

    assert audit["aggregate"]["decision"] == "fail"
    failure_types = {failure["type"] for failure in audit["gate_failures"]}
    assert "dew_air_lt0_regression" in failure_types
    assert "canopy_lt0_regression" in failure_types


def test_cli_writes_outputs_and_returns_pass(tmp_path):
    input_path = _write_payload(
        tmp_path,
        [
            _summary("s1", "llm_rspc_v2"),
            _summary("s1", "llm_rspc_v2_hot_dry_proposer_strict"),
        ],
    )
    output_json = tmp_path / "audit.json"
    output_md = tmp_path / "audit.md"

    code = main(
        [
            "--input",
            str(input_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert code == 0
    assert output_json.exists()
    assert output_md.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["aggregate"]["decision"] == "pass"
