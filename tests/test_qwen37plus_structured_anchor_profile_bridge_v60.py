import unittest
from types import SimpleNamespace

from gl_gym.experiments import qwen37plus_structured_anchor_profile_bridge_v60 as v60


class TestQwen37PlusStructuredAnchorProfileBridgeV60(unittest.TestCase):
    def _state(self, **overrides):
        data = {
            "timestep": 12,
            "hour_of_day": 12.0,
            "temp_air": 24.0,
            "rh_air": 70.0,
            "co2_air": 430.0,
            "glob_rad": 350.0,
            "dew_margin_air": 3.0,
            "canopy_dew_margin": 3.0,
            "temp_violation": 0.0,
        }
        data.update(overrides)
        return SimpleNamespace(**data)

    def _anchor(self, **overrides):
        data = {
            "profile_intent": "hot_dry_humidity_retention",
            "target_temp": 22.0,
            "target_co2": 520.0,
            "target_rh": 76.0,
            "risk_flags": ["dry_risk", "high_vpd"],
            "forbidden_intents": [],
            "planning_horizon_steps": 8,
            "confidence": 0.8,
        }
        data.update(overrides)
        return data

    def test_model_constants_and_artifact_names_use_qwen37plus(self):
        self.assertEqual(v60.MODEL_NAME, "qwen3.7-plus")
        self.assertEqual(v60.SOURCE_MODEL_NAME, "qwen3.7-max")
        paths = [
            str(v60.ONLINE_PRECHECK_JSON),
            str(v60.MODEL_SWITCH_READINESS_JSON),
            str(v60.BRIDGE_AUDIT_JSON),
            str(v60.BRIDGE_COMPARISON_JSON),
            str(v60.FUTURE_QWEN37PLUS_CACHE_PATH),
            str(v60.FUTURE_QWEN37PLUS_OUTPUT_DIR),
        ]
        self.assertTrue(all("qwen37plus" in path for path in paths))
        self.assertFalse(any("qwen37_structured_anchor_opt_in_shadow_v59" in path for path in paths))

    def test_target_range_horizon_cap_and_clip(self):
        contract, diagnostics = v60.structured_anchor_to_intent_contract(
            self._anchor(target_temp=100.0, target_co2=1000.0, target_rh=95.0, planning_horizon_steps=99),
            self._state(),
        )

        self.assertEqual(diagnostics["planning_horizon_steps"], 24)
        self.assertTrue(diagnostics["horizon"]["was_clipped"])
        self.assertEqual(contract.target_range["temp"][1], 24.0)
        self.assertEqual(contract.target_range["co2"][1], 850.0)
        self.assertEqual(contract.target_range["rh"][1], 88.0)
        self.assertTrue(any(item["was_clipped"] for item in diagnostics["target_range_diagnostics"]))

    def test_regime_mapping_dew_dry_co2_night_and_unknown(self):
        dew_contract, _ = v60.structured_anchor_to_intent_contract(
            self._anchor(profile_intent="dawn dew relief", risk_flags=["dew", "high_rh"]),
            self._state(hour_of_day=5.0),
        )
        dry_contract, _ = v60.structured_anchor_to_intent_contract(
            self._anchor(profile_intent="radiation spike dry relief", risk_flags=["radiation", "high_vpd"]),
            self._state(hour_of_day=13.0),
        )
        co2_contract, co2_diag = v60.structured_anchor_to_intent_contract(
            self._anchor(profile_intent="co2 opportunity", risk_flags=["co2"], forbidden_intents=[]),
            self._state(hour_of_day=13.0),
        )
        blocked_co2_contract, blocked_diag = v60.structured_anchor_to_intent_contract(
            self._anchor(
                profile_intent="co2 opportunity",
                risk_flags=["co2_vent_conflict"],
                forbidden_intents=["co2_enrichment_high_vent"],
            ),
            self._state(hour_of_day=13.0),
        )
        night_contract, _ = v60.structured_anchor_to_intent_contract(
            self._anchor(profile_intent="night stabilization", risk_flags=[]),
            self._state(hour_of_day=23.0),
        )
        unknown_contract, unknown_diag = v60.structured_anchor_to_intent_contract(
            self._anchor(profile_intent="unrecognized planning phrase", risk_flags=[]),
            self._state(hour_of_day=13.0),
        )

        self.assertEqual(dew_contract.regime, "dawn_predehumidify")
        self.assertEqual(dry_contract.regime, "radiation_spike_relief")
        self.assertEqual(co2_contract.regime, "co2_day_boost")
        self.assertFalse(co2_diag["unknown_profile_intent_mapped"])
        self.assertEqual(blocked_co2_contract.regime, "economy_hold")
        self.assertIn("co2_day_boost_blocked_by_co2_vent_conflict", blocked_diag["regime_mapping_diagnostics"])
        self.assertEqual(night_contract.regime, "night_heat_hold")
        self.assertEqual(unknown_contract.regime, "economy_hold")
        self.assertTrue(unknown_diag["unknown_profile_intent_mapped"])

    def test_bridge_one_entry_generates_profile_candidates_without_control_action(self):
        entry = {
            "env_id": "TomatoEnv_y2010_d180_s43",
            "timestep": 12,
            "model_name": "qwen3.7-max",
            "structured_anchor": {
                "valid": True,
                "shadow_plan": self._anchor(),
            },
            "parsed_plan": {
                "target_temp": 21.0,
                "target_co2": 500.0,
                "target_rh": 74.0,
            },
        }
        row, comparison = v60._bridge_one_entry(
            cache_key="sample",
            entry=entry,
            row=self._state().__dict__,
        )

        self.assertTrue(row["bridgeable"])
        self.assertGreater(row["profile_candidate_count"], 0)
        self.assertFalse(row["final_control_change"])
        self.assertFalse(row["current_plan_modified"])
        self.assertFalse(row["low_level_action_generated"])
        self.assertEqual(comparison["scenario_id"], "y2010_d180_s43_n720")
        self.assertGreater(comparison["profile_candidate_count"], 0)

    def test_model_switch_and_readiness_never_allow_replay_or_claims(self):
        precheck = {
            "online_llm_credentials_present": True,
            "online_llm_accessible": True,
            "provider_error_detected": False,
        }
        bridge = {
            "bridge_pass": True,
            "bridge_input_coverage_rate": 1.0,
            "unknown_profile_intent_mapped_rate": 0.0,
            "hard_safety_profile_violation_count": 0,
        }
        model_switch = v60.build_model_switch_readiness(precheck, bridge)
        readiness = v60.build_readiness(precheck, bridge)

        self.assertEqual(model_switch["model_name"], "qwen3.7-plus")
        self.assertEqual(model_switch["future_online_evidence_model_name"], "qwen3.7-plus")
        self.assertEqual(readiness["next_action"], "qwen37plus_structured_anchor_opt_in_shadow_rollout_plan")
        for payload in (model_switch, readiness):
            self.assertFalse(payload["controlled_replay_allowed"])
            self.assertFalse(payload["controlled_replay_execution_allowed"])
            self.assertFalse(payload["metadata_replay_execution_allowed"])
            self.assertFalse(payload["performance_claim_allowed"])
            self.assertFalse(payload["promotion_evidence"])

    def test_precheck_failure_blocks_future_rollout_authorization(self):
        readiness = v60.build_readiness(
            {
                "online_llm_credentials_present": True,
                "online_llm_accessible": False,
                "blocked_reason": "online_llm_accessibility_blocked",
            },
            {
                "bridge_pass": True,
                "bridge_input_coverage_rate": 1.0,
                "unknown_profile_intent_mapped_rate": 0.0,
                "hard_safety_profile_violation_count": 0,
            },
        )

        self.assertEqual(readiness["next_action"], "online_llm_accessibility_blocked")
        self.assertIn("online_llm_accessibility_blocked", readiness["stop_taxonomy"])


if __name__ == "__main__":
    unittest.main()
