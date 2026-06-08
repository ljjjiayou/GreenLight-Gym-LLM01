import json
from pathlib import Path

from gl_gym.experiments.hot_dry_action_proposer_controlled_shadow_audit import classify_row
from gl_gym.experiments.hot_dry_action_proposer_shadow_replay import (
    audit_trace,
    audit_traces,
    build_report,
    build_replay_windows,
)


def _candidate(name, *, selected=False, proposer=False, action=None, terms=None):
    return {
        "name": name,
        "selected": selected,
        "shadow_proposer": proposer,
        "action": action
        or {
            "heat": 0.0,
            "co2": 0.0,
            "screen": 0.30,
            "vent": 0.60,
            "lamp": 0.0,
            "shade": 0.10,
        },
        "score_terms": terms
        or {
            "temp_penalty": 0.0,
            "rh_penalty": 0.0,
            "dry_penalty": 3.0,
            "vpd_penalty": 1.0,
            "energy_penalty": 0.0,
            "smooth_penalty": 0.0,
            "conflict_penalty": 0.0,
            "dew_penalty": 0.0,
            "hot_dry_penalty": 1.0,
            "mitigation_bonus": 0.0,
            "temp_next": 29.0,
            "rh_next": 48.0,
            "vpd_next": 1.90,
        },
        "score": 10.0,
    }


def _safe_proposer(name="shadow_hot_dry_balanced_relief"):
    return _candidate(
        name,
        proposer=True,
        action={
            "heat": 0.0,
            "co2": 0.0,
            "screen": 0.56,
            "vent": 0.36,
            "lamp": 0.0,
            "shade": 0.72,
        },
        terms={
            "temp_penalty": 0.20,
            "rh_penalty": 0.0,
            "dry_penalty": 1.0,
            "vpd_penalty": 0.30,
            "energy_penalty": 0.0,
            "smooth_penalty": 0.0,
            "conflict_penalty": 0.0,
            "dew_penalty": 0.0,
            "hot_dry_penalty": 0.20,
            "mitigation_bonus": 0.10,
            "temp_next": 29.5,
            "rh_next": 51.0,
            "vpd_next": 1.74,
        },
    )


def _row(step, *, candidates=None, enabled=True):
    if candidates is None:
        candidates = [_candidate("selected", selected=True), _safe_proposer()]
    return {
        "step": step,
        "temp_air": 29.0,
        "rh_air": 50.0,
        "vpd_air": 1.85,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "intent_regime": "hot_dry_relief",
        "rspc_action_audit_enabled": enabled,
        "rspc_action_hot_dry_proposer_active": True,
        "rspc_action_selected_name": "selected",
        "rspc_action_candidates_json": json.dumps(candidates),
    }


def _write_trace(tmp_path: Path, rows):
    path = tmp_path / "y2015_d120_s44_n240_llm_rspc_v2.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_build_replay_windows_merges_one_step_gap():
    rows = [
        _row(0),
        _row(1),
        _row(2, candidates=[_candidate("selected", selected=True)]),
        _row(3),
    ]
    classified = [classify_row(row) for row in rows]

    windows = build_replay_windows(rows, classified, min_signal_steps=2, max_gap_steps=1)

    assert len(windows) == 1
    assert windows[0]["start_step"] == 0
    assert windows[0]["end_step"] == 3
    assert windows[0]["signal_steps"] == 3
    assert windows[0]["gap_steps"] == 1
    assert windows[0]["mean_delta_vent"] < 0.0
    assert windows[0]["mean_delta_shade"] > 0.0


def test_audit_trace_ready_for_controlled_shadow_replay(tmp_path):
    path = _write_trace(tmp_path, [_row(0), _row(1), _row(2)])

    audit = audit_trace(path, coverage_threshold=0.50)

    assert audit["recommendation"]["decision"] == "ready_for_controlled_shadow_replay"
    assert audit["signal_steps"] == 3
    assert audit["replay_window_count"] == 1
    assert audit["unsafe_preferred_steps"] == 0


def test_unsafe_preferred_blocks_replay(tmp_path):
    unsafe_proposer = _safe_proposer()
    unsafe_proposer["name"] = "shadow_hot_dry_unsafe_heat"
    unsafe_proposer["action"]["heat"] = 0.20
    path = _write_trace(tmp_path, [_row(0, candidates=[_candidate("selected", selected=True), unsafe_proposer])])

    audit = audit_trace(path, min_signal_steps=1)

    assert audit["recommendation"]["decision"] == "unsafe_to_replay"
    assert audit["unsafe_preferred_steps"] == 1


def test_replay_signal_can_be_weak_below_coverage_threshold(tmp_path):
    rows = [
        _row(0),
        _row(1, candidates=[_candidate("selected", selected=True)]),
        _row(2, candidates=[_candidate("selected", selected=True)]),
        _row(3, candidates=[_candidate("selected", selected=True)]),
    ]
    path = _write_trace(tmp_path, rows)

    audit = audit_trace(path, min_signal_steps=1, coverage_threshold=0.75)

    assert audit["recommendation"]["decision"] == "weak_controlled_replay_signal"
    assert audit["signal_steps"] == 1
    assert audit["safe_hot_dry_steps"] == 4


def test_audit_traces_and_report_are_stable(tmp_path):
    path = _write_trace(tmp_path, [_row(0), _row(1)])

    audit = audit_traces([tmp_path], coverage_threshold=0.50)
    report = build_report(audit)

    assert audit["aggregate"]["trace_count"] == 1
    assert audit["aggregate"]["recommendation"]["decision"] == "ready_for_controlled_shadow_replay"
    assert "Hot-Dry Action Proposer Shadow Replay v1" in report
    assert "shadow_hot_dry_balanced_relief" in report


def test_missing_metadata_does_not_crash(tmp_path):
    path = _write_trace(tmp_path, [_row(0, enabled=False)])

    audit = audit_trace(path)

    assert audit["recommendation"]["decision"] == "needs_trace_metadata"
    assert "missing_rspc_action_scoring_metadata" in audit["warnings"]
