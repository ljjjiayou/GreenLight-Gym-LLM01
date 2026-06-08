import json
import unittest

from gl_gym.experiments.candidate_pool_audit import classify_step


def _candidate(name, score, *, selected=False, screen=0.4, vent=0.4, rh_next=55, vpd_next=1.2, dry=1.0, vpd_penalty=0.5, canopy=0.0):
    return {
        "name": name,
        "score": score,
        "selected": selected,
        "action": {"heat": 0, "co2": 0, "screen": screen, "vent": vent, "lamp": 0, "shade": 0},
        "post_shape_action": {"heat": 0, "co2": 0, "screen": screen, "vent": vent, "lamp": 0, "shade": 0},
        "score_terms": {
            "temp_next": 24,
            "rh_next": rh_next,
            "vpd_next": vpd_next,
            "dry_penalty": dry,
            "vpd_high_penalty": vpd_penalty,
            "temp_penalty": 0,
            "dew_penalty": 0,
            "canopy_dew_penalty": canopy,
        },
    }


def _row(candidates, **extra):
    data = {
        "step": 10,
        "rspc_action_selected_name": "selected",
        "rspc_action_candidates_json": json.dumps(candidates),
        "rh_air": 58,
        "vpd_air": 1.7,
        "temp_air": 24,
        "dew_margin_air": 2,
        "canopy_dew_margin": 2,
        "rh_low_violation": 0,
        "vpd_high_excess": 0.1,
        "u_screen": 0.4,
        "u_ventilation": 0.4,
        "u_heating": 0,
        "u_co2": 0,
        "u_lighting": 0,
        "u_shading": 0,
        "tomato_safety_v2_applied": False,
    }
    data.update(extra)
    return data


class TestCandidatePoolAudit(unittest.TestCase):
    def test_good_candidate_not_selected_when_safe_dry_candidate_loses(self):
        selected = _candidate("selected", 1.0, selected=True, rh_next=50, vpd_next=1.8, dry=5.0)
        safe = _candidate("safe_dry", 1.5, rh_next=53, vpd_next=1.6, dry=3.0)

        result = classify_step(_row([selected, safe]), {"canopy_dew_margin": 2})

        self.assertEqual(result["label"], "good_candidate_not_selected")
        self.assertEqual(result["safe_dry_candidate_count"], 1)

    def test_guardrail_introduced_risk_when_screen_up_and_next_canopy_negative(self):
        selected = _candidate("selected", 1.0, selected=True, screen=0.3, vent=0.5)

        result = classify_step(
            _row([selected], u_screen=0.7, u_ventilation=0.2, canopy_dew_margin=0.5),
            {"canopy_dew_margin": -0.1},
        )

        self.assertEqual(result["label"], "guardrail_introduced_risk")
        self.assertTrue(result["guardrail_screen_up"])
        self.assertTrue(result["guardrail_vent_down"])

    def test_missing_candidate_metadata(self):
        result = classify_step(_row([], rspc_action_candidates_json=""), None)

        self.assertEqual(result["label"], "missing_candidate_metadata")


if __name__ == "__main__":
    unittest.main()
