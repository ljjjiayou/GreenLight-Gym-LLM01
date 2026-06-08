import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments import qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_execution_v76 as v76
from gl_gym.experiments import qwen37plus_profile_action_envelope_shadow_patch_v74 as v74


ACTION = {"heat": 0.1, "co2": 0.0, "screen": 0.35, "vent": 0.45, "lamp": 0.0, "shade": 0.5}


def _write_csv(path, rows):
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _candidate(**overrides):
    data = {
        "name": "profile_action_envelope:hot_dry_protect",
        "profile_name": "hot_dry_protect",
        "candidate_source": "normal_path_profile_action_envelope",
        "intent": "hot_dry_protection",
        "target_direction": {"target_temp": "increase", "target_co2": "hold", "target_rh": "decrease"},
        "action_bounds": {
            "heat": {"min": 0.1, "max": 0.1},
            "co2": {"min": 0.0, "max": 0.0},
            "screen": {"min": 0.3, "max": 0.35},
            "vent": {"min": 0.4, "max": 0.45},
            "lamp": {"min": 0.0, "max": 0.0},
            "shade": {"min": 0.5, "max": 0.5},
        },
        "preferred_direction": {
            "heat": "hold",
            "co2": "hold",
            "screen": "increase",
            "vent": "increase",
            "lamp": "hold",
            "shade": "hold",
        },
        "priority_terms": {
            "profile_target_alignment": -0.2,
            "tomato_safety_compatibility": 0.0,
            "action_continuity": -0.1,
            "hard_safety_precedence": True,
        },
        "continuity_constraints": {
            "previous_action_source": "baseline_safety_action",
            "max_delta_from_previous_action": {
                "heat": 0.0,
                "co2": 0.0,
                "screen": 0.05,
                "vent": 0.05,
                "lamp": 0.0,
                "shade": 0.0,
            },
        },
        "tomato_safety_projection": {
            "projection_required": True,
            "projected_action": dict(ACTION),
            "projection_applied": True,
            "projection_reasons": ["canopy_dew_buffer"],
            "rewrite_delta_by_field": {
                "heat": 0.0,
                "co2": 0.0,
                "screen": 0.05,
                "vent": 0.05,
                "lamp": 0.0,
                "shade": 0.0,
            },
        },
        "projected_action": dict(ACTION),
        "score_terms": {
            "profile_target_alignment": -0.2,
            "tomato_projection_delta": 0.1,
            "continuity_penalty": 0.1,
            "envelope_width_penalty": 0.1,
            "compatibility_penalty": 0.0,
            "selection_score": 1.5,
        },
        "eligible": True,
        "rejection_reason": "",
        "compatibility_category": "compatibility_shadow_ready",
    }
    data.update(overrides)
    return data


def _row(step=0, **overrides):
    candidates = overrides.pop("candidates", [_candidate()])
    data = {
        "step": step,
        "profile_action_envelope_shadow_enabled": True,
        "profile_action_envelope_shadow_candidate_count": len(candidates),
        "profile_action_envelope_shadow_eligible_candidate_count": 1 if candidates else 0,
        "profile_action_envelope_shadow_best_name": "profile_action_envelope:hot_dry_protect" if candidates else "",
        "profile_action_envelope_shadow_best_profile": "hot_dry_protect" if candidates else "",
        "profile_action_envelope_shadow_best_score": 1.5 if candidates else "",
        "profile_action_envelope_shadow_best_eligible": bool(candidates),
        "profile_action_envelope_shadow_best_rejection_reason": "",
        "profile_action_envelope_shadow_final_action_changed": False,
        "profile_action_envelope_shadow_candidates_json": json.dumps(candidates, separators=(",", ":")),
    }
    data.update(overrides)
    return data


def _manifest(tmp):
    return {
        "trace_dir": str(tmp),
        "cache_path": str(Path(tmp) / "cache.json"),
        "max_steps": 720,
        "plan_cache_mode": "record",
        "plan_cache_key_policy": "scenario_timestep",
        "agent_config_overrides": {"profile_action_envelope_shadow_enabled": True},
        "commands": [{"group": "synthetic"}],
    }


def _record_for_rows(rows):
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "y2010_d180_s43_n720_llm_rspc_v2.csv"
    _write_csv(path, rows)
    cache = Path(tmp.name) / "cache.json"
    cache.write_text(json.dumps({"entries": {"k": {"model_name": "qwen3.7-plus"}}}), encoding="utf-8")
    audit = v74.audit_traces([tmp.name])
    record = v76.build_execution_record(
        authorization={
            "authorization_source": "unit_test",
            "user_authorized_online_llm_acquisition": True,
            "online_llm_allowed_for_acquisition": True,
            "scenario_windows": ["y2010_d180_s43_n720"],
        },
        manifest=_manifest(tmp.name),
        v75_readiness={"next_action": "execute_v75_opt_in_shadow_trace_acquisition_commands"},
        v74_audit=audit,
        trace_dir=tmp.name,
        cache_path=cache,
    )
    return tmp, record


class TestQwen37PlusProfileActionEnvelopeOptInShadowAcquisitionExecutionV76(unittest.TestCase):
    def test_successful_synthetic_trace_inherits_v74_pass_next_action(self):
        tmp, record = _record_for_rows([_row(step) for step in range(3)])
        self.addCleanup(tmp.cleanup)

        readiness = v76.build_result_readiness(record)

        self.assertEqual(record["acquisition_summary"]["trace_count"], 1)
        self.assertEqual(record["v74_audit_result"]["normal_path_profile_action_envelope_steps"], 3)
        self.assertEqual(readiness["next_action"], "profile_action_envelope_compatibility_shadow_audit_plan")
        self.assertEqual(readiness["recommended_followup"], "profile_action_envelope_scorer_shadow_design_plan")
        self.assertTrue(readiness["contract_complete"])
        self.assertTrue(readiness["tomato_safety_projection_present"])
        self.assertTrue(readiness["final_action_invariant"])

    def test_action_diff_routes_to_stop(self):
        tmp, record = _record_for_rows([_row(profile_action_envelope_shadow_final_action_changed=True)])
        self.addCleanup(tmp.cleanup)

        readiness = v76.build_result_readiness(record)

        self.assertEqual(readiness["next_action"], "stop_profile_action_envelope_shadow_action_invariance_violation")
        self.assertEqual(readiness["blocked_reason"], "v74_audit_detected_action_diff")

    def test_missing_tomato_projection_routes_to_contract_repair(self):
        broken = _candidate()
        broken.pop("tomato_safety_projection")
        tmp, record = _record_for_rows([_row(candidates=[broken])])
        self.addCleanup(tmp.cleanup)

        readiness = v76.build_result_readiness(record)

        self.assertEqual(readiness["next_action"], "profile_action_envelope_contract_or_projection_repair_plan")
        self.assertEqual(readiness["blocked_reason"], "v74_audit_detected_contract_or_projection_gap")

    def test_missing_envelope_candidate_metadata_routes_to_instrumentation_repair(self):
        tmp, record = _record_for_rows([_row(candidates=[])])
        self.addCleanup(tmp.cleanup)

        readiness = v76.build_result_readiness(record)

        self.assertEqual(readiness["next_action"], "profile_action_envelope_shadow_instrumentation_repair_plan")
        self.assertEqual(readiness["blocked_reason"], "v74_audit_missing_envelope_candidates")

    def test_boundary_flags_do_not_claim_replay_promotion_or_performance(self):
        tmp, record = _record_for_rows([_row()])
        self.addCleanup(tmp.cleanup)
        readiness = v76.build_result_readiness(record)

        for payload in (record, readiness):
            self.assertFalse(payload["controlled_replay_allowed"])
            self.assertFalse(payload["controlled_replay_execution_allowed"])
            self.assertFalse(payload["metadata_replay_execution_allowed"])
            self.assertFalse(payload["strict_replay_allowed"])
            self.assertFalse(payload["performance_claim_allowed"])
            self.assertFalse(payload["promotion_evidence"])
            self.assertFalse(payload["final_action_changed"])
        self.assertTrue(record["boundary"]["online_llm_called_for_acquisition"])
        self.assertFalse(record["boundary"]["online_llm_called_for_audit_or_tests"])

    def test_reports_include_execution_and_readiness_status(self):
        tmp, record = _record_for_rows([_row()])
        self.addCleanup(tmp.cleanup)
        readiness = v76.build_result_readiness(record)

        self.assertIn("v76 Profile Action Envelope Acquisition Execution Record", v76.build_execution_report(record))
        self.assertIn("next_action=profile_action_envelope_compatibility_shadow_audit_plan", v76.build_readiness_report(readiness))


if __name__ == "__main__":
    unittest.main()
