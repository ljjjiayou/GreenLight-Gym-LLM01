import csv
from pathlib import Path

from gl_gym.experiments.hot_dry_controlled_replay_trace_audit import (
    audit_trace,
    audit_traces,
    build_report,
    classify_row,
)


def _row(**updates):
    row = {
        "step": 0,
        "rspc_hot_dry_replay_enabled": True,
        "rspc_hot_dry_replay_reason": "controlled_shadow_replay_candidate",
        "rspc_hot_dry_replay_safe_hot_dry": True,
        "rspc_hot_dry_replay_would_apply": True,
        "rspc_hot_dry_replay_unsafe_preferred": False,
        "rspc_hot_dry_replay_unsafe_conflict": False,
        "rspc_hot_dry_replay_unsafe_filtered_count": 0,
        "rspc_hot_dry_replay_best_candidate_name": "shadow_hot_dry_balanced_relief",
        "rspc_hot_dry_replay_best_variant": "dry_vpd_x2",
        "rspc_hot_dry_replay_best_margin": 0.5,
        "rspc_hot_dry_proposer_control_enabled": False,
        "rspc_hot_dry_proposer_control_strict_enabled": False,
        "rspc_hot_dry_proposer_control_applied": False,
        "rspc_hot_dry_proposer_control_safe_hot_dry": True,
        "rspc_hot_dry_proposer_control_safety_gate_reason": "none",
        "rspc_hot_dry_proposer_control_reason": "disabled",
    }
    row.update(updates)
    return row


def _write_trace(tmp_path: Path, rows):
    path = tmp_path / "y2015_d180_s44_n240_llm_rspc_v2.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_classify_runtime_candidate():
    item = classify_row(_row())

    assert item["classification"] == "controlled_replay_candidate"
    assert item["would_apply"] is True
    assert item["best_candidate"] == "shadow_hot_dry_balanced_relief"


def test_classify_runtime_applied_candidate():
    item = classify_row(
        _row(
            rspc_hot_dry_proposer_control_enabled=True,
            rspc_hot_dry_proposer_control_applied=True,
            rspc_hot_dry_proposer_control_reason="applied",
        )
    )

    assert item["classification"] == "controlled_access_applied"
    assert item["control_applied"] is True
    assert item["control_unsafe"] is False


def test_classify_strict_runtime_applied_candidate():
    item = classify_row(
        _row(
            rspc_hot_dry_proposer_control_enabled=True,
            rspc_hot_dry_proposer_control_strict_enabled=True,
            rspc_hot_dry_proposer_control_applied=True,
            rspc_hot_dry_proposer_control_reason="strict_eligible_applied",
        )
    )

    assert item["classification"] == "strict_eligible_applied"
    assert item["control_applied"] is True
    assert item["control_strict"] is True


def test_classify_strict_filtered_reason():
    item = classify_row(
        _row(
            rspc_hot_dry_proposer_control_enabled=True,
            rspc_hot_dry_proposer_control_strict_enabled=True,
            rspc_hot_dry_proposer_control_reason="candidate_not_allowed",
        )
    )

    assert item["classification"] == "candidate_not_allowed"
    assert item["control_applied"] is False


def test_classify_runtime_unsafe_apply():
    item = classify_row(
        _row(
            rspc_hot_dry_proposer_control_enabled=True,
            rspc_hot_dry_proposer_control_applied=True,
            rspc_hot_dry_proposer_control_safe_hot_dry=False,
            rspc_hot_dry_proposer_control_reason="applied",
        )
    )

    assert item["classification"] == "controlled_access_unsafe_apply"
    assert item["control_unsafe"] is True


def test_classify_unsafe_preferred_blocks_candidate():
    item = classify_row(
        _row(
            rspc_hot_dry_replay_would_apply=False,
            rspc_hot_dry_replay_unsafe_preferred=True,
            rspc_hot_dry_replay_reason="unsafe_preferred_proposer",
        )
    )

    assert item["classification"] == "unsafe_conflict"
    assert item["unsafe_preferred"] is True


def test_audit_trace_reports_signal_and_filtered_rows(tmp_path):
    path = _write_trace(
        tmp_path,
        [
            _row(),
            _row(
                step=1,
                rspc_hot_dry_replay_would_apply=False,
                rspc_hot_dry_replay_reason="no_eligible_proposer",
                rspc_hot_dry_replay_unsafe_filtered_count=2,
                rspc_hot_dry_replay_best_candidate_name="",
                rspc_hot_dry_replay_best_variant="",
            ),
        ],
    )

    audit = audit_trace(path)

    assert audit["recommendation"]["decision"] == "controlled_replay_signal_detected"
    assert audit["would_apply_steps"] == 1
    assert audit["applied_steps"] == 0
    assert audit["unsafe_filtered_rows"] == 1
    assert audit["unsafe_preferred_steps"] == 0


def test_audit_trace_prefers_runtime_applied_decision(tmp_path):
    path = _write_trace(
        tmp_path,
        [
            _row(
                rspc_hot_dry_proposer_control_enabled=True,
                rspc_hot_dry_proposer_control_applied=True,
                rspc_hot_dry_proposer_control_reason="applied",
            )
        ],
    )

    audit = audit_trace(path)

    assert audit["recommendation"]["decision"] == "controlled_access_applied"
    assert audit["applied_steps"] == 1
    assert audit["unsafe_applied_steps"] == 0


def test_audit_trace_reports_strict_applied_decision(tmp_path):
    path = _write_trace(
        tmp_path,
        [
            _row(
                rspc_hot_dry_proposer_control_enabled=True,
                rspc_hot_dry_proposer_control_strict_enabled=True,
                rspc_hot_dry_proposer_control_applied=True,
                rspc_hot_dry_proposer_control_reason="strict_eligible_applied",
            ),
            _row(
                step=1,
                rspc_hot_dry_proposer_control_enabled=True,
                rspc_hot_dry_proposer_control_strict_enabled=True,
                rspc_hot_dry_proposer_control_applied=False,
                rspc_hot_dry_proposer_control_reason="not_severe_dry",
            ),
        ],
    )

    audit = audit_trace(path)

    assert audit["recommendation"]["decision"] == "strict_eligible_applied"
    assert audit["strict_applied_steps"] == 1
    assert audit["strict_filtered_steps"] == 1


def test_audit_traces_and_report_are_stable(tmp_path):
    _write_trace(tmp_path, [_row()])

    audit = audit_traces([tmp_path])
    report = build_report(audit)

    assert audit["aggregate"]["trace_count"] == 1
    assert audit["aggregate"]["would_apply_steps"] == 1
    assert "Hot-Dry Controlled Replay Trace Audit v1" in report


def test_missing_metadata_does_not_crash(tmp_path):
    path = _write_trace(tmp_path, [{"step": 0, "other": "value"}])

    audit = audit_trace(path)

    assert audit["recommendation"]["decision"] == "needs_trace_metadata"
    assert "missing_runtime_controlled_replay_metadata" in audit["warnings"]
