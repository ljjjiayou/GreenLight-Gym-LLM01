from copy import deepcopy
import json
import unittest

from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector
from gl_gym.experiments.diagnose_ppo_vs_llm import profile_action_envelope_shadow_to_record


def _director(**config_overrides):
    director = object.__new__(RuleBasedLLMDirector)
    director.config = AgentConfig(**config_overrides)
    return director


def _row(**overrides):
    data = {
        "name": "hot_dry_protect",
        "intent": "hot_dry_protection",
        "priority": {"temp": 0.7, "rh": 0.3},
        "constraints": {"vent_floor": 0.30},
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
        "baseline_control": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        "baseline_safety_control": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        "baseline_target_temp": 27.0,
        "baseline_target_co2": 430.0,
        "baseline_target_rh": 84.0,
        "top_candidates": list(rows),
        "raw_top_candidates": list(rows),
    }


class TestProfileActionEnvelopeShadowV74(unittest.TestCase):
    def test_config_defaults_keep_envelope_shadow_opt_in_disabled(self):
        config = AgentConfig()

        self.assertFalse(config.profile_action_envelope_shadow_enabled)
        self.assertTrue(config.profile_action_envelope_shadow_record_provenance)
        self.assertEqual(config.profile_action_envelope_shadow_max_candidates, 5)

    def test_converts_profile_rspc_rows_to_v73_envelope_contract_shape_without_mutation(self):
        director = _director(profile_action_envelope_shadow_enabled=True)
        profile_shadow = _profile_shadow(_row())
        before = deepcopy(profile_shadow)

        payload = director._compose_profile_action_envelopes_shadow(profile_shadow)

        self.assertEqual(profile_shadow, before)
        self.assertTrue(payload["enabled"])
        self.assertTrue(payload["shadow_only"])
        self.assertFalse(payload["final_action_changed"])
        self.assertEqual(payload["candidate_source"], "normal_path_profile_action_envelope")
        self.assertEqual(payload["candidate_count"], 1)
        candidate = payload["candidates"][0]
        for field in (
            "name",
            "profile_name",
            "candidate_source",
            "intent",
            "target_direction",
            "action_bounds",
            "preferred_direction",
            "priority_terms",
            "continuity_constraints",
            "tomato_safety_projection",
            "projected_action",
            "score_terms",
            "eligible",
            "rejection_reason",
            "compatibility_category",
        ):
            self.assertIn(field, candidate)
        self.assertEqual(candidate["name"], "profile_action_envelope:hot_dry_protect")
        self.assertEqual(candidate["candidate_source"], "normal_path_profile_action_envelope")
        self.assertEqual(candidate["intent"], "hot_dry_protection")
        self.assertEqual(candidate["target_direction"]["target_temp"], "increase")
        self.assertEqual(candidate["target_direction"]["target_co2"], "hold")
        self.assertEqual(candidate["target_direction"]["target_rh"], "decrease")
        self.assertAlmostEqual(candidate["projected_action"]["screen"], 0.3499999940395355)
        self.assertAlmostEqual(candidate["action_bounds"]["screen"]["min"], 0.30000001192092896)
        self.assertAlmostEqual(candidate["action_bounds"]["screen"]["max"], 0.3499999940395355)
        self.assertEqual(candidate["preferred_direction"]["screen"], "increase")
        projection = candidate["tomato_safety_projection"]
        self.assertTrue(projection["projection_required"])
        self.assertTrue(projection["projection_applied"])
        self.assertEqual(projection["projection_reasons"], ["canopy_dew_buffer"])
        self.assertAlmostEqual(projection["rewrite_delta_by_field"]["vent"], 0.04999998211860657)
        self.assertEqual(candidate["compatibility_category"], "compatibility_shadow_ready")
        for key in (
            "profile_target_alignment",
            "tomato_projection_delta",
            "continuity_penalty",
            "envelope_width_penalty",
            "compatibility_penalty",
            "selection_score",
        ):
            self.assertIn(key, candidate["score_terms"])

    def test_missing_raw_or_safety_action_marks_envelope_ineligible(self):
        director = _director(profile_action_envelope_shadow_enabled=True)

        payload = director._compose_profile_action_envelopes_shadow(
            _profile_shadow(_row(raw_control=None, safety_control=None))
        )

        candidate = payload["candidates"][0]
        self.assertFalse(candidate["eligible"])
        self.assertEqual(candidate["rejection_reason"], "missing_action_provenance")
        self.assertEqual(candidate["compatibility_category"], "contract_missing")
        self.assertEqual(candidate["projected_action"], {})
        self.assertEqual(candidate["tomato_safety_projection"]["rewrite_delta_by_field"], {})

    def test_diagnose_flattens_envelope_shadow_fields_and_compact_json(self):
        director = _director(profile_action_envelope_shadow_enabled=True)
        shadow = director._compose_profile_action_envelopes_shadow(_profile_shadow(_row()))
        plan = {"rollout_selection": {"profile_action_envelope_shadow": shadow}}

        record = profile_action_envelope_shadow_to_record(plan)

        self.assertTrue(record["profile_action_envelope_shadow_enabled"])
        self.assertEqual(record["profile_action_envelope_shadow_candidate_count"], 1)
        self.assertEqual(record["profile_action_envelope_shadow_eligible_candidate_count"], 1)
        self.assertEqual(record["profile_action_envelope_shadow_best_name"], "profile_action_envelope:hot_dry_protect")
        self.assertFalse(record["profile_action_envelope_shadow_final_action_changed"])
        decoded = json.loads(record["profile_action_envelope_shadow_candidates_json"])
        self.assertEqual(decoded[0]["candidate_source"], "normal_path_profile_action_envelope")
        self.assertIn("tomato_safety_projection", decoded[0])


if __name__ == "__main__":
    unittest.main()
