import unittest
from dataclasses import dataclass

from gl_gym.agent.intent_contract import DEFAULT_CONSTRAINTS, IntentContract
from gl_gym.agent.profile_generator import (
    PROFILE_KEYS,
    build_profile_candidates,
    build_profile_generator_shadow_payload,
    score_profile_candidates,
)


@dataclass
class DummyState:
    timestep: int = 12
    hour_of_day: float = 12.0
    temp_air: float = 22.0
    rh_air: float = 70.0
    co2_air: float = 430.0
    glob_rad: float = 300.0
    dew_margin_air: float = 3.0
    canopy_dew_margin: float = 3.0
    forecast_rad_mean_1h: float = 300.0
    forecast_rad_peak_2h: float = 300.0
    temp_air_delta_1h: float = 0.0
    temp_violation: float = 0.0


def _intent(regime="economy_hold", **overrides):
    data = {
        "regime": regime,
        "target_range": {
            "temp": (17.0, 19.0),
            "co2": (430.0, 650.0),
            "rh": (72.0, 78.0),
        },
        "priority": ("safety", "energy", "growth"),
        "constraints": dict(DEFAULT_CONSTRAINTS),
        "profile_shape": {"temp": "constant_hold", "co2": "daylight_gate", "rh": "constant_hold"},
        "confidence": 0.75,
    }
    data.update(overrides)
    return IntentContract(**data)


class TestProfileGenerator(unittest.TestCase):
    def assert_profiles_are_well_formed(self, candidates, horizon):
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertEqual(set(candidate.target_profile), set(PROFILE_KEYS))
            for key, values in candidate.target_profile.items():
                self.assertEqual(len(values), horizon)
                if key == "target_temp":
                    self.assertTrue(all(11.5 <= value <= 24.0 for value in values))
                elif key == "target_co2":
                    self.assertTrue(all(380.0 <= value <= 850.0 for value in values))
                elif key == "target_rh":
                    self.assertTrue(all(62.0 <= value <= 88.0 for value in values))

    def test_all_candidates_have_fixed_horizon_and_clipped_bounds(self):
        intent = _intent(
            "hot_dry_relief",
            target_range={"temp": (5.0, 30.0), "co2": (300.0, 1000.0), "rh": (40.0, 95.0)},
            priority=("safety", "humidity", "temperature"),
            profile_shape={"temp": "shade_cooling", "rh": "hot_dry_protect", "co2": "off_when_venting"},
        )

        candidates = build_profile_candidates(intent, DummyState(), horizon_steps=5)

        self.assert_profiles_are_well_formed(candidates, 5)

    def test_hot_dry_generates_protect_and_shade_candidates(self):
        intent = _intent(
            "hot_dry_relief",
            priority=("safety", "humidity", "temperature", "energy"),
            profile_shape={"temp": "shade_cooling", "rh": "hot_dry_protect", "co2": "off_when_venting"},
        )

        payload, diagnostics = build_profile_generator_shadow_payload(intent, DummyState(temp_air=30.0), horizon_steps=6)
        names = {candidate["name"] for candidate in payload}

        self.assertIn("hot_dry_protect", names)
        self.assertIn("shade_cooling", names)
        self.assertEqual(diagnostics["selected_shadow_profile_name"], "hot_dry_protect")
        self.assertTrue(diagnostics["shadow_only"])

    def test_humidity_regimes_generate_dehumidify_profiles(self):
        for regime in ("high_humidity_recovery", "dawn_predehumidify", "cold_humid_recovery"):
            intent = _intent(
                regime,
                priority=("safety", "humidity", "temperature"),
                profile_shape={"rh": "strict_dehumidify_then_relax"},
            )

            candidates = build_profile_candidates(intent, DummyState(rh_air=90.0), horizon_steps=6)
            names = {candidate.name for candidate in candidates}

            self.assertIn("strict_dehumidify_then_relax", names)
            strict = next(candidate for candidate in candidates if candidate.name == "strict_dehumidify_then_relax")
            self.assertLessEqual(strict.target_profile["target_rh"][0], strict.target_profile["target_rh"][-1])

    def test_cold_dry_does_not_generate_aggressive_dehumidify(self):
        intent = _intent(
            "cold_dry_protect",
            priority=("safety", "temperature", "humidity"),
            profile_shape={"temp": "night_heat_hold", "rh": "cold_dry_protect"},
        )

        candidates = build_profile_candidates(intent, DummyState(temp_air=15.5, rh_air=50.0), horizon_steps=4)
        names = {candidate.name for candidate in candidates}

        self.assertIn("cold_dry_protect", names)
        self.assertNotIn("strict_dehumidify_then_relax", names)

    def test_co2_day_boost_ramps_up_without_exceeding_bounds(self):
        intent = _intent(
            "co2_day_boost",
            target_range={"temp": (19.0, 21.0), "co2": (500.0, 760.0), "rh": (70.0, 76.0)},
            priority=("safety", "growth", "energy"),
            profile_shape={"co2": "co2_day_boost"},
        )

        candidates = build_profile_candidates(intent, DummyState(glob_rad=300.0), horizon_steps=5)
        boost = next(candidate for candidate in candidates if candidate.name == "co2_day_boost")
        co2 = boost.target_profile["target_co2"]

        self.assertGreater(co2[-1], co2[0])
        self.assertTrue(all(a <= b for a, b in zip(co2, co2[1:])))
        self.assertLessEqual(max(co2), 850.0)

    def test_missing_or_invalid_intent_falls_back_to_constant_hold(self):
        candidates = build_profile_candidates(None, DummyState(), horizon_steps=3)
        payload, diagnostics = build_profile_generator_shadow_payload({"regime": "bad"}, DummyState(), horizon_steps=3)

        self.assertEqual([candidate.name for candidate in candidates], ["constant_hold"])
        self.assertEqual([candidate["name"] for candidate in payload], ["constant_hold"])
        self.assertEqual(diagnostics["selected_shadow_profile_name"], "constant_hold")
        self.assertTrue(diagnostics["intent_diagnostics"]["fallback"])

    def test_scorer_prefers_hot_dry_profile_when_not_in_hot_danger(self):
        intent = _intent(
            "hot_dry_relief",
            priority=("safety", "humidity", "temperature"),
            profile_shape={"temp": "shade_cooling", "rh": "hot_dry_protect"},
        )
        state = DummyState(
            temp_air=30.0,
            rh_air=48.0,
            glob_rad=720.0,
            forecast_rad_mean_1h=720.0,
            forecast_rad_peak_2h=760.0,
            canopy_dew_margin=4.0,
            dew_margin_air=4.0,
        )

        candidates = build_profile_candidates(intent, state, horizon_steps=6)
        scores, diagnostics = score_profile_candidates(candidates, intent, state, horizon_steps=6)

        self.assertEqual(scores[0].name, "hot_dry_protect")
        self.assertEqual(diagnostics["score_selected_shadow_profile_name"], "hot_dry_protect")
        self.assertEqual(diagnostics["score_safety_gate_reason"], "none")
        self.assertGreater(diagnostics["score_margin_to_second"], 0.0)

    def test_scorer_prefers_shade_cooling_under_hot_danger(self):
        intent = _intent(
            "hot_dry_relief",
            priority=("safety", "humidity", "temperature"),
            profile_shape={"temp": "shade_cooling", "rh": "hot_dry_protect"},
        )
        state = DummyState(
            temp_air=33.0,
            rh_air=52.0,
            glob_rad=760.0,
            forecast_rad_mean_1h=760.0,
            forecast_rad_peak_2h=820.0,
            temp_air_delta_1h=0.8,
            canopy_dew_margin=3.5,
            dew_margin_air=3.5,
        )

        candidates = build_profile_candidates(intent, state, horizon_steps=6)
        scores, diagnostics = score_profile_candidates(candidates, intent, state, horizon_steps=6)

        self.assertEqual(scores[0].name, "shade_cooling")
        self.assertEqual(diagnostics["score_safety_gate_reason"], "temp_high_gate")
        self.assertNotEqual(diagnostics["score_selected_shadow_profile_name"], "hot_dry_protect")

    def test_scorer_blocks_hot_dry_profile_under_canopy_gate(self):
        intent = _intent(
            "hot_dry_relief",
            priority=("safety", "humidity", "temperature"),
            profile_shape={"temp": "shade_cooling", "rh": "hot_dry_protect"},
        )
        state = DummyState(
            temp_air=29.0,
            rh_air=58.0,
            glob_rad=720.0,
            forecast_rad_mean_1h=720.0,
            forecast_rad_peak_2h=760.0,
            canopy_dew_margin=0.7,
            dew_margin_air=2.5,
        )

        candidates = build_profile_candidates(intent, state, horizon_steps=6)
        scores, diagnostics = score_profile_candidates(candidates, intent, state, horizon_steps=6)

        self.assertEqual(diagnostics["score_safety_gate_reason"], "canopy_gate")
        self.assertNotEqual(scores[0].name, "hot_dry_protect")

    def test_scorer_keeps_dew_risk_above_hot_dry_retention(self):
        intent = _intent(
            "hot_humid_relief",
            priority=("safety", "humidity", "temperature"),
            profile_shape={"temp": "shade_cooling", "rh": "strict_dehumidify_then_relax"},
        )
        state = DummyState(
            temp_air=29.0,
            rh_air=91.0,
            glob_rad=650.0,
            forecast_rad_mean_1h=690.0,
            forecast_rad_peak_2h=730.0,
            canopy_dew_margin=0.6,
            dew_margin_air=0.8,
        )

        candidates = build_profile_candidates(intent, state, horizon_steps=6)
        scores, diagnostics = score_profile_candidates(candidates, intent, state, horizon_steps=6)

        self.assertEqual(scores[0].name, "strict_dehumidify_then_relax")
        self.assertEqual(diagnostics["score_safety_gate_reason"], "canopy_gate")
        self.assertGreaterEqual(diagnostics["score_selected_breakdown"]["dew_risk"], 0.0)


if __name__ == "__main__":
    unittest.main()
