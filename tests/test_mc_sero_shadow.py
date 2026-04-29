from types import SimpleNamespace

import numpy as np

from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector


def _director(mode: str = "shadow") -> RuleBasedLLMDirector:
    director = RuleBasedLLMDirector.__new__(RuleBasedLLMDirector)
    director.config = AgentConfig(mc_sero_mode=mode)
    return director


def _state(**overrides):
    data = {
        "temp_air": 20.0,
        "rh_air": 70.0,
        "co2_air": 430.0,
        "glob_rad": 100.0,
        "temp_out": 12.0,
        "rh_out": 70.0,
        "hour_of_day": 12.0,
        "fruit_weight": 2.0,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "u_boil": 0.0,
        "u_co2": 0.0,
        "u_th_scr": 0.50,
        "u_vent": 0.20,
        "u_lamp": 0.0,
        "u_bl_scr": 0.0,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_mc_sero_off_returns_disabled_diagnostic():
    baseline = np.asarray([0.0, 0.0, 0.5, 0.2, 0.0, 0.0], dtype=np.float32)

    info = _director("off")._evaluate_mc_sero_shadow(
        _state(),
        rollout_candidates=[("anchor", baseline)],
        baseline_source="anchor",
        baseline_control=baseline,
    )

    assert info == {"enabled": False, "mode": "off"}


def test_mc_sero_high_humidity_prefers_dehumidification_candidate():
    baseline = np.asarray([0.0, 0.0, 0.7, 0.12, 0.0, 0.0], dtype=np.float32)
    state = _state(rh_air=95.0, rh_out=70.0, temp_air=18.0, dew_margin_air=0.3, canopy_dew_margin=0.3)

    info = _director()._evaluate_mc_sero_shadow(
        state,
        rollout_candidates=[("anchor", baseline)],
        baseline_source="anchor",
        baseline_control=baseline,
    )

    assert info["available"] is True
    assert info["would_select"] is True
    assert info["best_candidate"] in {"safe_dehumidify", "emergency_dehumidify"}
    assert info["score_terms"]["rh_high"] > 0.0 or info["score_terms"]["dew"] > 0.0


def test_mc_sero_dry_state_avoids_high_ventilation():
    baseline = np.asarray([0.0, 0.0, 0.7, 0.50, 0.0, 0.0], dtype=np.float32)
    state = _state(temp_air=25.0, rh_air=43.0, rh_out=50.0, u_vent=0.40)

    info = _director()._evaluate_mc_sero_shadow(
        state,
        rollout_candidates=[("anchor", baseline)],
        baseline_source="anchor",
        baseline_control=baseline,
    )

    assert info["best_candidate"] in {"economy_hold", "dry_recovery", "economic_dehumidify"}
    assert info["best_control"][3] <= 0.1801
    assert info["reject_reason"] != "dry_vent_risk"


def test_mc_sero_cold_night_prefers_heat_buffer_over_aggressive_venting():
    baseline = np.asarray([0.0, 0.0, 0.7, 0.50, 0.0, 0.0], dtype=np.float32)
    state = _state(
        temp_air=11.0,
        rh_air=70.0,
        temp_out=2.0,
        rh_out=80.0,
        hour_of_day=2.0,
        glob_rad=0.0,
        u_vent=0.40,
    )

    info = _director()._evaluate_mc_sero_shadow(
        state,
        rollout_candidates=[("anchor", baseline)],
        baseline_source="anchor",
        baseline_control=baseline,
    )

    assert info["best_candidate"] == "heat_buffer"
    assert info["best_control"][0] >= 0.20
    assert info["best_control"][3] <= 0.1001


def test_mc_sero_shadow_does_not_mutate_baseline_control():
    baseline = np.asarray([0.1, 0.0, 0.5, 0.2, 0.0, 0.0], dtype=np.float32)
    before = baseline.copy()

    _director()._evaluate_mc_sero_shadow(
        _state(),
        rollout_candidates=[("anchor", baseline)],
        baseline_source="anchor",
        baseline_control=baseline,
    )

    np.testing.assert_allclose(baseline, before)
