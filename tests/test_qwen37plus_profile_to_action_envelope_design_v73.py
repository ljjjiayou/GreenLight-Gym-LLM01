import unittest

from gl_gym.experiments import qwen37plus_profile_to_action_envelope_design_v73 as v73


def _v70_audit(**aggregate_overrides):
    aggregate = {
        "rows": 912,
        "profile_action_candidate_available_steps": 912,
        "compatibility_shadow_ready_steps": 374,
        "contract_missing_steps": 0,
        "tomato_projection_missing_steps": 0,
        "action_invariance_violation_steps": 0,
        "fallback_candidate_source_violation_steps": 0,
        "tomato_safety_incompatible_steps": 467,
        "profile_target_incompatible_steps": 353,
        "action_continuity_incompatible_steps": 4,
    }
    aggregate.update(aggregate_overrides)
    return {
        "artifact": "qwen37plus_profile_action_candidate_compatibility_shadow_audit_20260604_v70",
        "version": "v70",
        "aggregate": aggregate,
        "hypothesis_check": _v70_hypothesis(),
    }


def _v70_hypothesis(**overrides):
    data = {
        "artifact": "profile_action_candidate_hypothesis_check_20260604_v70",
        "hypothesis_status": "profile_to_action_direct_mapping_hypothesis_needs_revision",
        "alternative_hypothesis": "action_envelope_or_safety_projected_mapping",
        "observed_gap": "Profile actions are often substantially rewritten or rejected by safety logic.",
        "next_action": "profile_to_action_mapping_repair_or_action_envelope_design",
    }
    data.update(overrides)
    return data


def _design_and_readiness(contract=None, audit=None, hypothesis=None):
    contract = contract or v73.build_envelope_contract()
    audit = audit or _v70_audit()
    hypothesis = hypothesis or _v70_hypothesis()
    design = v73.build_design(
        v70_audit=audit,
        v70_hypothesis=hypothesis,
        v72_execution_text="- rows=912\n- profile_action_candidate_available_steps=912\n",
        contract=contract,
    )
    readiness = v73.build_readiness(
        design=design,
        contract=contract,
        v70_audit=audit,
        v70_hypothesis=hypothesis,
    )
    return design, readiness


class TestQwen37PlusProfileToActionEnvelopeDesignV73(unittest.TestCase):
    def test_contract_schema_covers_required_profile_action_envelope_fields(self):
        contract = v73.build_envelope_contract()
        output = contract["output_contract"]
        required = set(output["required_fields"])
        schema = output["field_schema"]

        self.assertTrue(set(v73.ENVELOPE_REQUIRED_OUTPUT_FIELDS) <= required)
        self.assertEqual(output["candidate_name_pattern"], "profile_action_envelope:<profile_name>")
        self.assertEqual(output["candidate_source_value"], "normal_path_profile_action_envelope")
        self.assertEqual(output["action_trace_field_map"]["heat"], "u_heating")
        self.assertEqual(output["action_trace_field_map"]["vent"], "u_ventilation")
        self.assertEqual(set(schema["action_bounds"]["fields"]), set(v73.ACTION_FIELDS))
        self.assertEqual(schema["action_bounds"]["per_field_schema"], {"min": "float", "max": "float"})
        self.assertEqual(set(schema["preferred_direction"]["allowed_values"]), set(v73.DIRECTION_VALUES))
        self.assertTrue(schema["tomato_safety_projection"]["not_final_action_control"])

    def test_v70_like_evidence_routes_to_minimal_shadow_instrumentation_plan(self):
        design, readiness = _design_and_readiness()

        self.assertEqual(design["design_decision"], "define_profile_action_envelope_contract_not_runtime_shadow_patch")
        self.assertEqual(
            readiness["next_action"],
            "minimal_profile_action_envelope_shadow_instrumentation_plan",
        )
        self.assertTrue(readiness["direct_mapping_gap_confirmed"])
        self.assertTrue(readiness["design_only_ready"])

    def test_missing_tomato_projection_schema_routes_to_projection_repair(self):
        contract = v73.build_envelope_contract(include_tomato_projection=False)
        _design, readiness = _design_and_readiness(contract=contract)

        self.assertEqual(readiness["next_action"], "tomato_safety_projection_provenance_repair_plan")
        self.assertFalse(readiness["tomato_safety_projection_contract_present"])

    def test_missing_target_direction_schema_routes_to_target_direction_repair(self):
        contract = v73.build_envelope_contract(include_target_direction=False)
        _design, readiness = _design_and_readiness(contract=contract)

        self.assertEqual(readiness["next_action"], "profile_target_direction_schema_repair_plan")
        self.assertFalse(readiness["target_direction_schema_present"])

    def test_source_action_diff_or_projection_failure_stops_with_diagnostic_value(self):
        audit = _v70_audit(action_invariance_violation_steps=1)
        _design, readiness = _design_and_readiness(audit=audit)

        self.assertEqual(readiness["next_action"], "stop_source_action_diff_or_safety_provenance_diagnostic")
        self.assertFalse(readiness["source_action_invariant"])

    def test_artifacts_never_authorize_runtime_replay_promotion_or_rollout(self):
        contract = v73.build_envelope_contract()
        design, readiness = _design_and_readiness(contract=contract)

        for payload in (contract, design, readiness):
            for field in v73.BOUNDARY_FALSE_FIELDS:
                self.assertFalse(payload[field])
            self.assertFalse(payload["rollout_command_generated"])
            self.assertFalse(payload.get("runtime_control_change", False))
            self.assertNotIn("commands", payload)
            self.assertNotIn("authorization_plan", payload)

    def test_generated_markdown_includes_hypothesis_revision(self):
        _design, readiness = _design_and_readiness()
        report = v73.build_report(readiness, "v73 readiness")

        self.assertIn("profile_to_action_direct_mapping_hypothesis_needs_revision", report)
        self.assertIn("action_envelope_or_safety_projected_mapping", report)


if __name__ == "__main__":
    unittest.main()
