import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.agent.llm_agent import AgentConfig
from gl_gym.experiments.qwen37_structured_anchor_parser_v58 import (
    agent_config_structured_anchor_defaults,
    audit_existing_v56_empty_anchors,
    build_contract_fixtures,
    build_readiness,
    build_shadow_parser_audit,
    parse_structured_anchor,
)


def _valid_anchor(**overrides):
    anchor = {
        "profile_intent": "hot_dry_relief",
        "target_temp": 19.0,
        "target_co2": 430.0,
        "target_rh": 70.0,
        "risk_flags": ["dry_side"],
        "forbidden_intents": ["co2_enrichment_high_vent"],
        "planning_horizon_steps": 12,
        "confidence": 0.8,
    }
    anchor.update(overrides)
    return anchor


class TestStructuredAnchorParserV58(unittest.TestCase):
    def test_complete_structured_anchor_parses_to_shadow_plan(self):
        parsed = parse_structured_anchor(_valid_anchor())

        self.assertTrue(parsed["valid"])
        self.assertTrue(parsed["clean_planning_evidence"])
        self.assertEqual(parsed["shadow_plan"]["source"], "structured_anchor_shadow_parser")
        self.assertEqual(parsed["shadow_plan"]["target_rh"], 70.0)
        self.assertFalse(parsed["shadow_plan"]["final_control_generation_allowed"])

    def test_missing_required_field_is_invalid(self):
        anchor = _valid_anchor()
        anchor.pop("target_rh")

        parsed = parse_structured_anchor(anchor)

        self.assertFalse(parsed["valid"])
        self.assertIn("missing_required_fields", parsed["errors"])
        self.assertIn("target_rh", parsed["missing_fields"])

    def test_low_level_action_field_is_invalid(self):
        parsed = parse_structured_anchor(_valid_anchor(ventilation=0.7))

        self.assertFalse(parsed["valid"])
        self.assertIn("final_control_fields_present", parsed["errors"])
        self.assertIn("unexpected_fields", parsed["errors"])
        self.assertIn("ventilation", parsed["final_control_field_hits"])

    def test_empty_anchor_is_not_clean_planning_evidence(self):
        parsed = parse_structured_anchor({})

        self.assertFalse(parsed["valid"])
        self.assertFalse(parsed["clean_planning_evidence"])
        self.assertIn("empty_anchor", parsed["errors"])

    def test_confidence_and_horizon_are_bounded(self):
        bad_confidence = parse_structured_anchor(_valid_anchor(confidence=1.2))
        bad_horizon = parse_structured_anchor(_valid_anchor(planning_horizon_steps=0))

        self.assertFalse(bad_confidence["valid"])
        self.assertIn("confidence_invalid", bad_confidence["errors"])
        self.assertFalse(bad_horizon["valid"])
        self.assertIn("planning_horizon_steps_invalid", bad_horizon["errors"])

    def test_wrapped_structured_anchor_is_supported(self):
        parsed = parse_structured_anchor({"structured_planning_anchor": _valid_anchor()})

        self.assertTrue(parsed["valid"])
        self.assertTrue(parsed["wrapper_used"])


class TestStructuredAnchorAuditV58(unittest.TestCase):
    def test_agent_config_defaults_are_closed_and_shadow_only(self):
        cfg = AgentConfig()
        defaults = agent_config_structured_anchor_defaults()

        self.assertFalse(cfg.structured_anchor_parser_enabled)
        self.assertTrue(cfg.structured_anchor_shadow_only)
        self.assertTrue(defaults["default_off"])
        self.assertTrue(defaults["shadow_only_default"])

    def test_contract_fixtures_pass_expected_validity(self):
        fixtures = build_contract_fixtures()

        self.assertTrue(fixtures["fixture_pass"])
        self.assertGreaterEqual(fixtures["fixture_count"], 6)

    def test_existing_empty_cache_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            cache.write_text(
                json.dumps(
                    {
                        "entries": {
                            "empty": {
                                "buffered_action": {"action_set": False},
                                "buffered_setpoints": {},
                                "anchor_source": "recent_anchor",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_existing_v56_empty_anchors(cache)

        self.assertEqual(audit["empty_anchor_entries_checked"], 1)
        self.assertEqual(audit["empty_anchor_clean_planning_evidence_count"], 0)
        self.assertTrue(audit["empty_anchors_rejected"])

    def test_readiness_never_allows_replay_or_claims(self):
        readiness = build_readiness(
            {"parser_ready_for_opt_in_shadow_rollout_plan": True},
            {"fixture_pass": True},
        )

        self.assertEqual(readiness["next_action"], "structured_anchor_opt_in_qwen37_shadow_rollout_plan")
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])

    def test_shadow_audit_is_ready_with_fixture_cache_and_v57_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            cache.write_text(
                json.dumps(
                    {
                        "entries": {
                            "empty": {
                                "buffered_action": {"action_set": False},
                                "buffered_setpoints": {},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            v57 = Path(tmp) / "v57.json"
            v57.write_text(json.dumps({"next_action": "structured_anchor_opt_in_shadow_parser_plan"}), encoding="utf-8")

            audit = build_shadow_parser_audit(cache_path=cache, v57_readiness_json=v57)

        self.assertTrue(audit["parser_ready_for_opt_in_shadow_rollout_plan"])
        self.assertTrue(audit["empty_anchors_rejected"])
        self.assertTrue(audit["fixture_pass"])
        self.assertFalse(audit["metadata_replay_execution_allowed"])


if __name__ == "__main__":
    unittest.main()
