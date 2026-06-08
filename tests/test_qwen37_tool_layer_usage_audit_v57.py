import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.qwen37_tool_layer_usage_audit_v57 import (
    FINAL_CONTROL_FIELDS,
    build_readiness,
    build_structured_anchor_contract_design,
    build_tool_layer_deprecation_matrix,
    classify_plan_cache_entry,
    inventory_registered_tools,
    audit_plan_cache,
)


class TestToolLayerInventoryV57(unittest.TestCase):
    def test_tool_inventory_detects_registered_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools_py = Path(tmp) / "tools.py"
            tools_py.write_text(
                """
from langchain_core.tools import StructuredTool
def create_langchain_tools(interface):
    return [
        StructuredTool.from_function(func=object(), name="get_status"),
        StructuredTool.from_function(func=object(), name="set_all_controls"),
        StructuredTool.from_function(func=object(), name="set_heating"),
        StructuredTool.from_function(func=object(), name="set_co2"),
        StructuredTool.from_function(func=object(), name="set_screen"),
        StructuredTool.from_function(func=object(), name="set_ventilation"),
        StructuredTool.from_function(func=object(), name="set_lamps"),
        StructuredTool.from_function(func=object(), name="set_blindscreen"),
    ]
""",
                encoding="utf-8",
            )

            inventory = inventory_registered_tools(tools_py)

        self.assertTrue(inventory["get_status_registered"])
        self.assertTrue(inventory["set_all_controls_registered"])
        self.assertEqual(inventory["missing_expected_tools"], [])
        self.assertIn("set_ventilation", inventory["low_level_action_tools_registered"])


class TestPlanCacheClassificationV57(unittest.TestCase):
    def test_empty_anchor_is_not_clean_planning_evidence(self):
        report = classify_plan_cache_entry(
            {
                "buffered_action": {"action_set": False},
                "buffered_setpoints": {},
                "llm_action_found": False,
                "raw_response": "",
                "anchor_source": "recent_anchor",
            }
        )

        self.assertEqual(report["classification"], "empty_planning_anchor")
        self.assertFalse(report["clean_planning_evidence"])
        self.assertFalse(report["raw_response_mentions_set_all_controls"])

    def test_set_all_controls_anchor_success_requires_action_and_setpoints(self):
        report = classify_plan_cache_entry(
            {
                "buffered_action": {"action_set": True},
                "buffered_setpoints": {"target_temp": 19.0, "target_co2": 450.0, "target_rh": 70.0},
                "llm_action_found": True,
                "raw_response": "set_all_controls(...)",
            }
        )

        self.assertEqual(report["classification"], "set_all_controls_anchor_success")
        self.assertTrue(report["clean_planning_evidence"])
        self.assertTrue(report["raw_response_mentions_set_all_controls"])

    def test_plan_cache_audit_counts_empty_and_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "entries": {
                            "empty": {
                                "buffered_action": {"action_set": False},
                                "buffered_setpoints": {},
                            },
                            "success": {
                                "buffered_action": {"action_set": True},
                                "buffered_setpoints": {
                                    "target_temp": 20.0,
                                    "target_co2": 500.0,
                                    "target_rh": 65.0,
                                },
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_plan_cache(cache_path)

        self.assertEqual(audit["entry_count"], 2)
        self.assertEqual(audit["empty_planning_anchor_count"], 1)
        self.assertEqual(audit["set_all_controls_anchor_success_count"], 1)


class TestDeprecationAndStructuredAnchorV57(unittest.TestCase):
    def test_deprecation_matrix_does_not_authorize_deletion_or_default_change(self):
        matrix = build_tool_layer_deprecation_matrix(
            {
                "registered_tools": [
                    "get_status",
                    "set_all_controls",
                    "set_heating",
                    "set_co2",
                    "set_screen",
                    "set_ventilation",
                    "set_lamps",
                    "set_blindscreen",
                ]
            }
        )

        self.assertFalse(matrix["deletion_authorized"])
        self.assertFalse(matrix["default_controller_change_authorized"])
        for entry in matrix["tool_entries"]:
            self.assertFalse(entry["deletion_authorized"])
            self.assertFalse(entry["default_controller_change_authorized"])

    def test_structured_anchor_contract_excludes_final_control(self):
        design = build_structured_anchor_contract_design()

        self.assertEqual(design["contract_name"], "structured_planning_anchor")
        self.assertFalse(design["final_control_generation_allowed"])
        self.assertFalse(design["default_llm_rspc_v2_path_changed"])
        for field in FINAL_CONTROL_FIELDS:
            self.assertIn(field, design["explicitly_excluded_fields"])
            self.assertNotIn(field, design["required_fields"])

    def test_readiness_never_allows_replay_or_claims(self):
        readiness = build_readiness(
            {
                "legacy_tool_action_contract_unstable": True,
                "profile_candidate_guardrail_conflict_persists": True,
                "dominant_tool_layer_failure_mode": "legacy_tool_action_contract_unstable",
            },
            {
                "deletion_authorized": False,
                "default_controller_change_authorized": False,
            },
            {
                "default_controller_change_authorized": False,
            },
        )

        self.assertEqual(readiness["next_action"], "structured_anchor_opt_in_shadow_parser_plan")
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
