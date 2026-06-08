from copy import deepcopy
import unittest

from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector


def _director(**config_overrides):
    director = object.__new__(RuleBasedLLMDirector)
    director.config = AgentConfig(**config_overrides)
    return director


def _row(**overrides):
    data = {
        "name": "hot_dry_protect",
        "eligible": True,
        "ineligible_reason": "",
        "raw_score": 2.0,
        "safety_score": 1.5,
        "selection_score": 1.5,
        "target_temp": 28.0,
        "target_co2": 430.0,
        "target_rh": 82.0,
        "raw_control": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        "safety_control": [0.10, 0.20, 0.35, 0.45, 0.50, 0.60],
        "action_delta_terms": {"heat": 0.0, "co2": 0.0, "screen": 0.05, "vent": 0.05, "lamp": 0.0, "shade": 0.0},
        "score_delta_terms": {"temp": -0.1, "rh": -0.2},
        "safety_details": {"temp_penalty": 0.1, "rh_penalty": 0.2},
        "tomato_safety_v2_applied": True,
        "tomato_safety_v2_reasons": ["canopy_dew_buffer"],
    }
    data.update(overrides)
    return data


def _profile_shadow(*rows):
    return {
        "enabled": True,
        "shadow_only": True,
        "candidate_count": len(rows),
        "eligible_candidate_count": sum(bool(row.get("eligible", False)) for row in rows),
        "top_candidates": list(rows),
        "raw_top_candidates": list(rows),
    }


class TestProfileActionCandidateShadowV69(unittest.TestCase):
    def test_config_defaults_keep_shadow_opt_in_disabled(self):
        config = AgentConfig()

        self.assertFalse(config.profile_action_candidate_shadow_enabled)
        self.assertTrue(config.profile_action_candidate_shadow_record_provenance)
        self.assertEqual(config.profile_action_candidate_shadow_max_candidates, 5)

    def test_converts_profile_rspc_rows_to_v68_contract_shape(self):
        director = _director(profile_action_candidate_shadow_enabled=True)
        profile_shadow = _profile_shadow(_row())
        before = deepcopy(profile_shadow)

        payload = director._compose_profile_action_candidates_shadow(profile_shadow)

        self.assertEqual(profile_shadow, before)
        self.assertTrue(payload["enabled"])
        self.assertTrue(payload["shadow_only"])
        self.assertFalse(payload["final_action_changed"])
        self.assertEqual(payload["candidate_count"], 1)
        candidate = payload["candidates"][0]
        self.assertEqual(candidate["name"], "profile_action:hot_dry_protect")
        self.assertEqual(candidate["profile_name"], "hot_dry_protect")
        self.assertEqual(candidate["candidate_source"], "normal_path_profile_candidate")
        self.assertEqual(candidate["raw_action"]["heat"], 0.10000000149011612)
        self.assertEqual(candidate["raw_action"]["co2"], 0.20000000298023224)
        self.assertEqual(candidate["raw_action"]["screen"], 0.30000001192092896)
        self.assertEqual(candidate["raw_action"]["vent"], 0.4000000059604645)
        self.assertEqual(candidate["raw_action"]["lamp"], 0.5)
        self.assertEqual(candidate["raw_action"]["shade"], 0.6000000238418579)
        self.assertEqual(candidate["post_tomato_action"]["screen"], 0.3499999940395355)
        self.assertTrue(candidate["tomato_safety_v2_applied"])
        self.assertTrue(candidate["eligible"])
        self.assertEqual(candidate["rejection_reason"], "")
        for key in (
            "raw_score",
            "post_tomato_score",
            "selection_score",
            "profile_target_error",
            "action_delta_penalty",
            "tomato_safety_penalty",
            "compatibility_penalty",
            "hard_safety_rewrite_predicted",
        ):
            self.assertIn(key, candidate["score_terms"])

    def test_ineligible_candidate_preserves_rejection_reason(self):
        director = _director(profile_action_candidate_shadow_enabled=True)

        payload = director._compose_profile_action_candidates_shadow(
            _profile_shadow(_row(eligible=False, ineligible_reason="temp_high_gate:profile_forbidden"))
        )

        candidate = payload["candidates"][0]
        self.assertFalse(candidate["eligible"])
        self.assertEqual(candidate["rejection_reason"], "temp_high_gate:profile_forbidden")

    def test_missing_action_provenance_marks_candidate_ineligible(self):
        director = _director(profile_action_candidate_shadow_enabled=True)

        payload = director._compose_profile_action_candidates_shadow(
            _profile_shadow(_row(raw_control=None, safety_control=None))
        )

        candidate = payload["candidates"][0]
        self.assertFalse(candidate["eligible"])
        self.assertEqual(candidate["rejection_reason"], "missing_action_provenance")
        self.assertEqual(candidate["raw_action"], {})
        self.assertEqual(candidate["post_tomato_action"], {})


if __name__ == "__main__":
    unittest.main()
