import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.qwen37_recovery_anchor_candidate_design_v55 import (
    FAILURE_SCENARIOS,
    _action_conflicts,
    build_readiness,
    build_recovery_anchor_candidate_design,
    build_recovery_anchor_shadow_scoring_audit,
    build_recovery_candidates,
)


FIELDS = [
    "step",
    "source",
    "anchor_source",
    "candidate_selection_source",
    "selected_fallback_candidate",
    "model_name",
    "target_temp",
    "target_rh",
    "target_co2",
    "profile_template_patch_enabled",
    "profile_template_patch_applied",
    "profile_template_patch_mode",
    "profile_template_patch_corrections",
    "profile_template_patch_forbidden_combinations",
    "profile_template_patch_vent_required_by_safety",
    "profile_template_patch_humidity_retention_allowed",
    "profile_template_patch_co2_enrichment_allowed",
    "profile_template_patch_repaired_target_temp",
    "profile_template_patch_repaired_target_rh",
    "profile_template_patch_repaired_target_co2",
    "temp_air",
    "rh_air",
    "co2_air",
    "glob_rad",
    "hour_of_day",
    "dew_margin_air",
    "canopy_dew_margin",
    "forecast_rad_mean_1h",
    "forecast_rad_peak_2h",
    "forecast_temp_out_delta_1h",
    "tomato_safety_v2_before",
    "tomato_safety_v2_after",
    "tomato_safety_v2_reasons",
    "rspc_action_candidates_json",
]


def _candidate_json() -> str:
    return json.dumps(
        [
            {
                "name": "rule",
                "score": -1.0,
                "selected": True,
                "action": {"heat": 1.0, "co2": 0.0, "screen": 0.0, "vent": 0.0, "lamp": 0.0, "shade": 0.0},
            }
        ]
    )


def _row(*, reasons: str = "hot_dry_cooling_guard,canopy_dew_hot_dry_conflict_buffer") -> dict[str, object]:
    return {
        "step": 5,
        "source": "rule",
        "anchor_source": "recent_anchor",
        "candidate_selection_source": "fallback_candidate",
        "selected_fallback_candidate": "recent_anchor",
        "model_name": "qwen3.7-max",
        "target_temp": 18.5,
        "target_rh": 72.0,
        "target_co2": 430.0,
        "profile_template_patch_enabled": "true",
        "profile_template_patch_applied": "true",
        "profile_template_patch_mode": "safety_vent",
        "profile_template_patch_corrections": "target_rh:safety_cap",
        "profile_template_patch_forbidden_combinations": "rh_dew_canopy_pressure_blocks_humidity_retention",
        "profile_template_patch_vent_required_by_safety": "true",
        "profile_template_patch_humidity_retention_allowed": "false",
        "profile_template_patch_co2_enrichment_allowed": "false",
        "profile_template_patch_repaired_target_temp": 18.5,
        "profile_template_patch_repaired_target_rh": 72.0,
        "profile_template_patch_repaired_target_co2": 430.0,
        "temp_air": 26.7,
        "rh_air": 63.5,
        "co2_air": 369.0,
        "glob_rad": 405.9,
        "hour_of_day": 7.25,
        "dew_margin_air": 7.5,
        "canopy_dew_margin": 1.9,
        "forecast_rad_mean_1h": 501.0,
        "forecast_rad_peak_2h": 618.0,
        "forecast_temp_out_delta_1h": 0.5,
        "tomato_safety_v2_before": json.dumps([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "tomato_safety_v2_after": json.dumps([0.0, 0.0, 0.6, 0.6, 0.0, 0.5]),
        "tomato_safety_v2_reasons": reasons,
        "rspc_action_candidates_json": _candidate_json(),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {field: "" for field in FIELDS}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


def _write_inputs(root: Path) -> tuple[Path, Path, str]:
    trace_dir = root / "traces"
    trace_dir.mkdir()
    scenario = "y2010_d180_s43_n720"
    _write_csv(trace_dir / f"{scenario}_llm_rspc_v2.csv", [_row()])
    contract_path = root / "v54_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "no_compatible_anchor_available_count": 1,
                "contract_rows": [
                    {
                        "scenario_id": scenario,
                        "step": 5,
                        "classification": "fallback_scored_no_compatible_alternative",
                        "selected_candidate": "rule",
                        "source": "rule",
                        "selected_prediction": {
                            "hard_safety_rewrite": True,
                            "safety_reasons": ["hot_dry_cooling_guard"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return trace_dir, contract_path, scenario


class TestQwen37RecoveryAnchorCandidateDesignV55(unittest.TestCase):
    def test_projected_anchor_covers_hard_safety_leak_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir, contract_path, scenario = _write_inputs(root)
            audit = build_recovery_anchor_shadow_scoring_audit(
                v54_contract_audit_json=contract_path,
                v52_trace_dir=trace_dir,
                scenarios=[scenario],
            )

        self.assertEqual(audit["evaluated_leak_count"], 1)
        self.assertEqual(audit["recovery_anchor_covered_count"], 1)
        self.assertEqual(audit["no_recovery_anchor_available_count"], 0)
        best = audit["row_reports"][0]["best_recovery_candidate"]
        self.assertIsNotNone(best)
        self.assertFalse(best["tomato_safety_prediction"]["hard_safety_rewrite"])

    def test_hot_dry_canopy_anchor_only_enabled_for_matching_reasons(self):
        hot = build_recovery_candidates(_row())
        neutral = build_recovery_candidates(_row(reasons=""))
        self.assertIn("hot_dry_canopy_relief_anchor", {item["name"] for item in hot})
        self.assertNotIn("hot_dry_canopy_relief_anchor", {item["name"] for item in neutral})

    def test_safe_rule_blend_suppresses_obvious_actuator_conflicts(self):
        candidates = build_recovery_candidates(_row())
        blend = next(item for item in candidates if item["name"] == "safe_rule_blend_anchor")
        self.assertNotIn("heat_vent_conflict", _action_conflicts(blend["action"]))
        self.assertNotIn("co2_vent_leak", _action_conflicts(blend["action"]))

    def test_readiness_never_allows_replay_or_claims(self):
        audit = {
            "candidate_metadata_or_control_missing_count": 0,
            "evaluated_leak_count": 43,
            "recovery_anchor_covered_count": 43,
            "no_recovery_anchor_available_count": 0,
            "recovery_anchor_coverage_rate": 1.0,
            "expected_v54_no_compatible_anchor_count": 43,
        }
        design = build_recovery_anchor_candidate_design(audit)
        readiness = build_readiness(audit, design)

        self.assertEqual(
            set(FAILURE_SCENARIOS),
            {"y2010_d180_s43_n720", "y2018_d181_s42_n720", "y2018_d181_s43_n720"},
        )
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["controlled_replay_execution_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])
        self.assertEqual(readiness["next_action"], "minimal_qwen37_recovery_anchor_opt_in_shadow_rollout_plan")


if __name__ == "__main__":
    unittest.main()
