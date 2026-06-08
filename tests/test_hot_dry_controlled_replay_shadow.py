from dataclasses import dataclass

import numpy as np

from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector


@dataclass
class DummyState:
    temp_air: float = 29.0
    rh_air: float = 50.0
    glob_rad: float = 700.0
    dew_margin_air: float = 3.0
    canopy_dew_margin: float = 3.0
    temp_violation: float = 0.0
    rh_high_violation: float = 0.0
    rh_low_violation: float = 1.0
    vpd_high_excess: float = 0.2


def _director():
    director = object.__new__(RuleBasedLLMDirector)
    director.config = AgentConfig()
    director.last_rollout_selection = {}
    return director


def _control_director(**cfg):
    director = object.__new__(RuleBasedLLMDirector)
    director.config = AgentConfig(**cfg)
    director.last_rollout_selection = {}
    return director


def _details(*, dry=4.0, vpd=2.0, hot_dry=2.0, temp=0.0, dew=0.0):
    return {
        "temp_penalty": temp,
        "rh_penalty": 0.0,
        "dry_penalty": dry,
        "vpd_penalty": vpd,
        "dew_penalty": dew,
        "energy_penalty": 0.0,
        "smooth_penalty": 0.0,
        "conflict_penalty": 0.0,
        "mitigation_bonus": 0.0,
        "hot_dry_penalty": hot_dry,
        "temp_next": 29.0,
        "rh_next": 50.0,
        "vpd_next": 1.8,
    }


def _row(director, name, control, selected_control, selected_details, candidate_details):
    control = np.asarray(control, dtype=np.float32)
    selected_control = np.asarray(selected_control, dtype=np.float32)
    return {
        "name": name,
        "control": control,
        "details": candidate_details,
        "shadow_proposer": True,
        "action_delta_terms": director._profile_action_delta_terms(selected_control, control),
        "score_delta_terms": director._profile_score_delta_terms(selected_details, candidate_details),
    }


def _replay(**updates):
    replay = {
        "safe_hot_dry": True,
        "safety_gate_reason": "none",
        "would_apply_shadow": True,
        "best_alignment": "dry_benefit",
        "best_margin": 0.5,
        "unsafe_preferred": False,
        "unsafe_conflict": False,
        "best_candidate_name": "shadow_hot_dry_balanced_relief",
        "best_variant": "dry_vpd_x2",
        "best_action": {
            "heat": 0.0,
            "co2": 0.0,
            "screen": 0.58,
            "vent": 0.40,
            "lamp": 0.0,
            "shade": 0.78,
        },
        "best_score_delta": {"dry": -1.0, "vpd": -0.5, "temp": 0.0, "dew": 0.0},
    }
    replay.update(updates)
    return {"hot_dry_action_proposer_controlled_replay": replay}


def test_controlled_replay_selects_safe_hot_dry_proposer():
    director = _director()
    selected_control = np.array([0.0, 0.0, 0.20, 0.70, 0.0, 0.10], dtype=np.float32)
    selected_details = _details()
    proposer_details = _details(dry=1.0, vpd=0.3, hot_dry=0.2)
    proposer = _row(
        director,
        "shadow_hot_dry_balanced_relief",
        [0.0, 0.0, 0.58, 0.40, 0.0, 0.78],
        selected_control,
        selected_details,
        proposer_details,
    )

    result = director._evaluate_hot_dry_action_proposer_controlled_replay_shadow(
        DummyState(),
        selected_name="selected",
        selected_post_shape_score=10.0,
        selected_post_shape_control=selected_control,
        selected_post_shape_details=selected_details,
        post_shape_rows=[proposer],
        gate_reason="none",
    )

    assert result["would_apply_shadow"] is True
    assert result["reason"] == "controlled_shadow_replay_candidate"
    assert result["best_candidate_name"] == "shadow_hot_dry_balanced_relief"
    assert result["best_alignment"] == "dry_benefit"
    assert result["best_margin"] > 0.05
    assert result["unsafe_preferred"] is False


def test_controlled_replay_does_not_apply_under_safety_gate():
    director = _director()
    selected_control = np.array([0.0, 0.0, 0.20, 0.70, 0.0, 0.10], dtype=np.float32)
    selected_details = _details()
    proposer = _row(
        director,
        "shadow_hot_dry_shade_preempt",
        [0.0, 0.0, 0.58, 0.50, 0.0, 0.82],
        selected_control,
        selected_details,
        _details(dry=1.0, vpd=0.3, hot_dry=0.2),
    )

    result = director._evaluate_hot_dry_action_proposer_controlled_replay_shadow(
        DummyState(temp_air=32.5, temp_violation=0.5),
        selected_name="selected",
        selected_post_shape_score=10.0,
        selected_post_shape_control=selected_control,
        selected_post_shape_details=selected_details,
        post_shape_rows=[proposer],
        gate_reason="temp_high_gate",
    )

    assert result["would_apply_shadow"] is False
    assert result["reason"] == "safety_gate_active"
    assert result["safety_gate_reason"] == "temp_high_gate"


def test_controlled_replay_filters_unsafe_preferred_proposer():
    director = _director()
    selected_control = np.array([0.0, 0.0, 0.20, 0.70, 0.0, 0.10], dtype=np.float32)
    selected_details = _details()
    unsafe = _row(
        director,
        "shadow_hot_dry_bad_heat",
        [0.20, 0.0, 0.58, 0.40, 0.0, 0.78],
        selected_control,
        selected_details,
        _details(dry=0.5, vpd=0.1, hot_dry=0.1),
    )

    result = director._evaluate_hot_dry_action_proposer_controlled_replay_shadow(
        DummyState(),
        selected_name="selected",
        selected_post_shape_score=10.0,
        selected_post_shape_control=selected_control,
        selected_post_shape_details=selected_details,
        post_shape_rows=[unsafe],
        gate_reason="none",
    )

    assert result["would_apply_shadow"] is False
    assert result["unsafe_preferred"] is False
    assert result["unsafe_filtered_count"] > 0
    assert result["reason"] == "no_eligible_proposer"
    assert result["variants"][0]["raw_safety_penalty"] is True


def test_hot_dry_proposer_control_flag_off_keeps_control_unchanged():
    director = _control_director(rspc_hot_dry_proposer_control_enabled=False)
    before = np.array([0.0, 0.0, 0.20, 0.70, 0.0, 0.10], dtype=np.float32)

    after = director._apply_hot_dry_proposer_control_access(DummyState(), before, _replay())

    np.testing.assert_allclose(after, before)
    info = director.last_rollout_selection["hot_dry_proposer_control"]
    assert info["enabled"] is False
    assert info["applied"] is False
    assert info["reason"] == "disabled"


def test_hot_dry_proposer_control_applies_safe_eligible_best_action():
    director = _control_director(rspc_hot_dry_proposer_control_enabled=True)
    before = np.array([0.0, 0.0, 0.20, 0.70, 0.0, 0.10], dtype=np.float32)

    after = director._apply_hot_dry_proposer_control_access(DummyState(), before, _replay())

    np.testing.assert_allclose(after, np.array([0.0, 0.0, 0.58, 0.40, 0.0, 0.78], dtype=np.float32))
    info = director.last_rollout_selection["hot_dry_proposer_control"]
    assert info["applied"] is True
    assert info["reason"] == "applied"
    assert info["candidate"] == "shadow_hot_dry_balanced_relief"
    assert info["variant"] == "dry_vpd_x2"
    assert info["delta_action"]["vent"] < 0.0
    assert info["delta_action"]["shade"] > 0.0


def test_hot_dry_proposer_control_blocks_safety_gate():
    director = _control_director(rspc_hot_dry_proposer_control_enabled=True)
    before = np.array([0.0, 0.0, 0.20, 0.70, 0.0, 0.10], dtype=np.float32)

    after = director._apply_hot_dry_proposer_control_access(
        DummyState(temp_air=32.5, temp_violation=0.5),
        before,
        _replay(safety_gate_reason="temp_high_gate"),
    )

    np.testing.assert_allclose(after, before)
    info = director.last_rollout_selection["hot_dry_proposer_control"]
    assert info["applied"] is False
    assert info["reason"] == "safety_gate_active"


def test_hot_dry_proposer_control_blocks_unsafe_or_weak_rows():
    director = _control_director(rspc_hot_dry_proposer_control_enabled=True)
    before = np.array([0.0, 0.0, 0.20, 0.70, 0.0, 0.10], dtype=np.float32)

    for replay in (
        _replay(unsafe_preferred=True),
        _replay(unsafe_conflict=True),
        _replay(best_margin=0.01),
        _replay(best_alignment="neutral_hold"),
        _replay(best_action={}),
    ):
        after = director._apply_hot_dry_proposer_control_access(DummyState(), before, replay)
        np.testing.assert_allclose(after, before)
        assert director.last_rollout_selection["hot_dry_proposer_control"]["applied"] is False


def test_strict_hot_dry_proposer_applies_only_severe_humidity_retention():
    director = _control_director(
        rspc_hot_dry_proposer_control_enabled=True,
        rspc_hot_dry_proposer_control_strict_enabled=True,
    )
    before = np.array([0.0, 0.0, 0.20, 0.70, 0.0, 0.10], dtype=np.float32)

    after = director._apply_hot_dry_proposer_control_access(
        DummyState(temp_air=29.0, rh_air=40.0, canopy_dew_margin=5.0),
        before,
        _replay(best_candidate_name="shadow_hot_dry_humidity_retention", best_margin=0.25),
    )

    np.testing.assert_allclose(after, np.array([0.0, 0.0, 0.58, 0.40, 0.0, 0.78], dtype=np.float32))
    info = director.last_rollout_selection["hot_dry_proposer_control"]
    assert info["applied"] is True
    assert info["strict_enabled"] is True
    assert info["reason"] == "strict_eligible_applied"
    assert info["min_margin"] == 0.20


def test_strict_hot_dry_proposer_blocks_boundary_rows():
    director = _control_director(
        rspc_hot_dry_proposer_control_enabled=True,
        rspc_hot_dry_proposer_control_strict_enabled=True,
    )
    before = np.array([0.0, 0.0, 0.20, 0.70, 0.0, 0.10], dtype=np.float32)

    cases = [
        (
            DummyState(temp_air=29.0, rh_air=40.0, canopy_dew_margin=5.0),
            _replay(best_candidate_name="shadow_hot_dry_shade_preempt", best_margin=0.25),
            "candidate_not_allowed",
        ),
        (
            DummyState(temp_air=29.0, rh_air=40.0, canopy_dew_margin=5.0),
            _replay(best_candidate_name="shadow_hot_dry_humidity_retention", best_margin=0.19),
            "margin_below_strict_min",
        ),
        (
            DummyState(temp_air=29.0, rh_air=50.0, canopy_dew_margin=5.0),
            _replay(best_candidate_name="shadow_hot_dry_humidity_retention", best_margin=0.25),
            "not_severe_dry",
        ),
        (
            DummyState(temp_air=30.8, rh_air=40.0, canopy_dew_margin=5.0),
            _replay(best_candidate_name="shadow_hot_dry_humidity_retention", best_margin=0.25),
            "temp_headroom_low",
        ),
        (
            DummyState(temp_air=29.0, rh_air=40.0, canopy_dew_margin=2.5),
            _replay(best_candidate_name="shadow_hot_dry_humidity_retention", best_margin=0.25),
            "canopy_reserve_low",
        ),
    ]

    for state, replay, reason in cases:
        after = director._apply_hot_dry_proposer_control_access(state, before, replay)
        np.testing.assert_allclose(after, before)
        info = director.last_rollout_selection["hot_dry_proposer_control"]
        assert info["applied"] is False
        assert info["reason"] == reason
