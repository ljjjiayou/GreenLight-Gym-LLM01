import csv
import json
from pathlib import Path

from gl_gym.experiments.hot_dry_strict_effect_audit import audit_effects, build_report


def _summary(scenario_id, controller, **aggregate):
    defaults = {
        "total_reward": 100.0,
        "total_rh_low_violation": 10.0,
        "total_vpd_high_excess": 5.0,
        "total_temp_violation": 0.0,
        "total_rh_high_violation": 0.0,
        "canopy_dew_margin_lt0_steps": 0,
        "runtime_error_steps": 0,
        "plan_cache_hit_steps": 240,
    }
    defaults.update(aggregate)
    return {
        "scenario_id": scenario_id,
        "controller": controller,
        "aggregate": defaults,
    }


def _write_benchmark(tmp_path: Path, summaries):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"summaries": summaries}), encoding="utf-8")
    return path


def _write_trace(tmp_path: Path, scenario_id: str, controller: str, rows):
    path = tmp_path / f"{scenario_id}_{controller}.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _trace_row(**updates):
    row = {
        "step": 0,
        "temp_air": 29.0,
        "rh_air": 40.0,
        "vpd_air": 2.4,
        "canopy_dew_margin": 5.0,
        "rspc_hot_dry_proposer_control_applied": False,
        "rspc_hot_dry_proposer_control_safe_hot_dry": True,
        "rspc_hot_dry_proposer_control_safety_gate_reason": "none",
        "rspc_hot_dry_proposer_control_reason": "disabled",
        "rspc_hot_dry_proposer_control_candidate": "",
        "rspc_hot_dry_proposer_control_margin": 0.0,
        "rspc_hot_dry_proposer_control_delta_screen": 0.0,
        "rspc_hot_dry_proposer_control_delta_vent": 0.0,
        "rspc_hot_dry_proposer_control_delta_shade": 0.0,
    }
    row.update(updates)
    return row


def test_audit_reports_holdout_positive_signal(tmp_path):
    d120 = "y2015_d120_s42_n240"
    d180 = "y2015_d180_s42_n240"
    benchmark = _write_benchmark(
        tmp_path,
        [
            _summary(d120, "llm_rspc_v2"),
            _summary(
                d120,
                "llm_rspc_v2_hot_dry_proposer_strict",
                total_reward=100.2,
                total_rh_low_violation=9.5,
                total_vpd_high_excess=5.02,
            ),
            _summary(d180, "llm_rspc_v2"),
            _summary(d180, "llm_rspc_v2_hot_dry_proposer_strict"),
        ],
    )
    _write_trace(
        tmp_path,
        d120,
        "llm_rspc_v2_hot_dry_proposer_strict",
        [
            _trace_row(
                step=7,
                rspc_hot_dry_proposer_control_applied=True,
                rspc_hot_dry_proposer_control_reason="strict_eligible_applied",
                rspc_hot_dry_proposer_control_candidate="shadow_hot_dry_humidity_retention",
                rspc_hot_dry_proposer_control_margin=0.3,
                rspc_hot_dry_proposer_control_delta_screen=0.4,
                rspc_hot_dry_proposer_control_delta_vent=-0.2,
                rspc_hot_dry_proposer_control_delta_shade=0.3,
            )
        ],
    )
    _write_trace(tmp_path, d180, "llm_rspc_v2_hot_dry_proposer_strict", [_trace_row()])

    audit = audit_effects([benchmark], [tmp_path])
    report = build_report(audit)

    assert audit["aggregate"]["recommendation"]["decision"] == "holdout_positive_signal"
    assert audit["aggregate"]["strict_applied_steps"] == 1
    assert audit["aggregate"]["unsafe_applied_steps"] == 0
    assert "y2015_d120_s42_n240" in report
    assert "Strict Applied Rows" in report


def test_audit_fails_on_d180_d240_regression(tmp_path):
    d180 = "y2015_d180_s42_n240"
    benchmark = _write_benchmark(
        tmp_path,
        [
            _summary(d180, "llm_rspc_v2"),
            _summary(d180, "llm_rspc_v2_hot_dry_proposer_strict", total_rh_low_violation=11.0),
        ],
    )
    _write_trace(tmp_path, d180, "llm_rspc_v2_hot_dry_proposer_strict", [_trace_row()])

    audit = audit_effects([benchmark], [tmp_path])

    assert audit["aggregate"]["recommendation"]["decision"] == "holdout_failed_regression"
    assert audit["aggregate"]["d180_d240_regression_count"] == 1


def test_audit_fails_on_unsafe_applied(tmp_path):
    d120 = "y2015_d120_s42_n240"
    benchmark = _write_benchmark(
        tmp_path,
        [
            _summary(d120, "llm_rspc_v2"),
            _summary(d120, "llm_rspc_v2_hot_dry_proposer_strict", total_reward=101.0),
        ],
    )
    _write_trace(
        tmp_path,
        d120,
        "llm_rspc_v2_hot_dry_proposer_strict",
        [
            _trace_row(
                rspc_hot_dry_proposer_control_applied=True,
                rspc_hot_dry_proposer_control_safe_hot_dry=False,
                rspc_hot_dry_proposer_control_reason="strict_eligible_applied",
            )
        ],
    )

    audit = audit_effects([benchmark], [tmp_path])

    assert audit["aggregate"]["recommendation"]["decision"] == "holdout_failed_runtime_or_safety"
    assert audit["aggregate"]["unsafe_applied_steps"] == 1


def test_audit_detects_no_stable_benefit(tmp_path):
    d120 = "y2015_d120_s42_n240"
    benchmark = _write_benchmark(
        tmp_path,
        [
            _summary(d120, "llm_rspc_v2"),
            _summary(d120, "llm_rspc_v2_hot_dry_proposer_strict", total_reward=99.9),
        ],
    )
    _write_trace(
        tmp_path,
        d120,
        "llm_rspc_v2_hot_dry_proposer_strict",
        [
            _trace_row(
                rspc_hot_dry_proposer_control_applied=True,
                rspc_hot_dry_proposer_control_reason="strict_eligible_applied",
            )
        ],
    )

    audit = audit_effects([benchmark], [tmp_path])

    assert audit["aggregate"]["recommendation"]["decision"] == "no_stable_benefit"
